"""One explicitly enabled, <=$0.05 company-key inference/ledger smoke.

Uses a disposable PostgreSQL database on AOS_TEST_DATABASE_URL. The report never
includes credentials, prompts, completion text, or raw provider responses.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg

from service import db, inference, repo
from service.auth import Principal


async def smoke(with_tools: bool = False, *, model: str = 'gpt-5.4-mini',
                optimizer_contract: bool = False, output: Path | None = None) -> bool:
    if with_tools and optimizer_contract:
        raise ValueError('Choose one diagnostic contract')
    limit = .02 if optimizer_contract else (.01 if with_tools else .05)
    report = {'status':'failed','timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'model':model,'credential_source':'NEOSIGMA_OPEN_API_KEY',
              'endpoint':'https://api.openai.com/v1/chat/completions',
              'maximum_budget_usd':limit,'automatic_retries':0,
              'scope':'one live completion plus PostgreSQL reservation and settlement; no benchmark claim'}
    name='aos_model_smoke_'+uuid4().hex
    admin=pool=job=None
    created=False
    try:
        # Both fail closed before any external dispatch. Never inspect fallback keys.
        inference.company_credential()
        inference.upstream_url()
        dsn=os.environ['AOS_TEST_DATABASE_URL']
        admin=await asyncpg.connect(dsn)
        await admin.execute(f'CREATE DATABASE "{name}"')
        created=True
        parts=urlsplit(dsn)
        isolated_dsn=urlunsplit((parts.scheme,parts.netloc,'/'+name,parts.query,parts.fragment))
        await db.migrate(isolated_dsn)
        pool=await asyncpg.create_pool(isolated_dsn,min_size=1,max_size=2,init=db._init_conn)
        async with pool.acquire() as c:
            x=await repo.create_org(c,'model-smoke','model-smoke@example.test',1)
            p=Principal(x['org_id'],'model-smoke',x['user_id'],x['email'],'admin')
            await repo.create_job(c,p,{'tasks':['fix-git'],'max_iterations':0,
                'budget':{'max_llm_usd':limit,'max_wall_clock_min':2}},'live-smoke')
            job=await repo.claim_job(c,'model-smoke',120)
        lease=repo.Lease(job['id'],job['lease_owner'],job['lease_epoch'])
        body={'messages':[{'role':'user','content':'Return exactly the two uppercase letters OK.'}]}
        if optimizer_contract:
            body={'messages':[{'role':'user','content':'Return exactly this JSON object: {"ok": true}'}],
                  'response_format':{'type':'json_object'}}
            report['scope']='one tool-free optimizer-format JSON completion; no source proposal or benchmark claim'
        if with_tools:
            tree=ast.parse(Path('agent/templates/terminal_bench.py').read_text())
            constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                       and node.targets[0].id in {'TOOLS','AGENT_INSTRUCTION'}}
            body={'messages':[{'role':'system','content':constants['AGENT_INSTRUCTION']},
                              {'role':'user','content':'Task:\nUse bash to print the two uppercase letters OK.'}],
                  'tools':constants['TOOLS'],'tool_choice':'auto'}
            report['scope']='one live completion with unchanged template tool schema and system/user messages; no tool execution'
        normalized=inference.normalize_request(body,report['model'],128)
        report['reasoning_effort']=normalized.get('reasoning_effort')
        bound=inference.reservation_cost(normalized)
        if bound > limit:
            raise RuntimeError('configured request exceeds smoke cap')
        report['reserved_upper_estimate_usd']=bound
        started=time.monotonic()
        result,usage=await inference.paid_completion(pool,lease,body=body,model=report['model'],max_output=128,
            kind='live_validation',reservation_key='only-call',timeout=60)
        report['duration_s']=round(time.monotonic()-started,3)
        choices=result.get('choices') or []
        content=(choices[0].get('message') or {}).get('content') if choices else None
        calls=(choices[0].get('message') or {}).get('tool_calls') if choices else None
        if optimizer_contract:
            try:
                report['expected_output_matched']=json.loads(content)=={'ok':True}
            except (TypeError,ValueError):
                report['expected_output_matched']=False
        else:
            report['expected_output_matched']=bool(calls) if with_tools else isinstance(content,str) and content.strip()=='OK'
        report['usage']={**usage,'pricing_basis':'standard uncached text-token estimate; not provider invoice'}
        async with pool.acquire() as c:
            ledger=await repo.spend_summary(c,job['id'])
            report['ledger']=ledger
            report['reservation_count']=await c.fetchval('SELECT count(*) FROM spend_reservations WHERE job_id=$1',job['id'])
            report['settled_count']=await c.fetchval('SELECT count(*) FROM spend_reservations WHERE job_id=$1 AND settled_at IS NOT NULL',job['id'])
            assert report['reservation_count']==report['settled_count']==1
            assert ledger['reserved_usd']==0 and ledger['spent_usd']==usage['cost_usd'] <= limit
            assert usage['cost_known'] and report['expected_output_matched']
            await repo.finish_job(c,lease,'SUCCEEDED','live_smoke')
        report['status']='passed'
    except Exception as exc:
        # Exception text can include request/provider data, so emit only its type.
        report['error_type']=type(exc).__name__
        if isinstance(exc,inference.InferenceError) and exc.provider_error:
            report['provider_error']=exc.provider_error
        if pool is not None and job is not None:
            async with pool.acquire() as c:
                report['ledger']=await repo.spend_summary(c,job['id'])
    finally:
        if pool is not None:
            await pool.close()
        if admin is not None:
            if created:
                await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
            await admin.close()
        output=output or Path('docs/live_model_tool_smoke.json' if with_tools else 'docs/live_model_smoke.json')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'report':str(output)}))
    return report['status']=='passed'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-paid',action='store_true',help='explicitly authorize one call capped at $0.05')
    parser.add_argument('--with-tools',action='store_true',help='use the baseline template tool schema, capped at $0.01')
    parser.add_argument('--optimizer-contract',action='store_true',help='check tool-free optimizer JSON, capped at $0.02')
    parser.add_argument('--model',choices=['gpt-5.4-mini','gpt-5.4'],default='gpt-5.4-mini')
    parser.add_argument('--output',type=Path,help='separate report path, to preserve prior diagnostics')
    args=parser.parse_args()
    if not args.allow_paid:
        parser.error('--allow-paid is required')
    if args.with_tools and args.optimizer_contract:
        parser.error('choose --with-tools or --optimizer-contract')
    raise SystemExit(0 if asyncio.run(smoke(args.with_tools, model=args.model,
        optimizer_contract=args.optimizer_contract, output=args.output)) else 1)
