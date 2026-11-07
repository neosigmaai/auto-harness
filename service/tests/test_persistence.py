"""Real PostgreSQL correctness tests, isolated in a disposable database.

Set AOS_TEST_DATABASE_URL to a test-server DSN whose user can CREATE DATABASE.
No model/provider calls are made. Existing databases/tables are never truncated.
"""
from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import pytest_asyncio

from service import db, repo
from service.auth import Principal


@pytest_asyncio.fixture
async def pool():
    dsn = os.getenv('AOS_TEST_DATABASE_URL')
    if not dsn:
        pytest.skip('set AOS_TEST_DATABASE_URL to run isolated PostgreSQL integration tests')
    name = 'aos_persistence_' + uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(dsn)
    isolated_dsn = urlunsplit((parts.scheme, parts.netloc, '/' + name, parts.query, parts.fragment))
    p = None
    try:
        await db.migrate(isolated_dsn)
        p = await asyncpg.create_pool(isolated_dsn, min_size=1, max_size=12, init=db._init_conn)
        yield p
    finally:
        if p is not None:
            await p.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


async def principal(pool, *, quota=1):
    async with pool.acquire() as c:
        x = await repo.create_org(c, 'org-' + uuid4().hex, 'admin@example.test', quota)
    return Principal(x['org_id'], 'org', x['user_id'], x['email'], x['role']), x['api_key']


def config(**overrides):
    return {'tasks': ['fix-git'], 'max_iterations': 0,
            'budget': {'max_wall_clock_min': 10, 'max_llm_usd': 1.0}, **overrides}


async def submit(pool, p, key=None, cfg=None):
    async with pool.acquire() as c:
        return await repo.create_job(c, p, cfg or config(), key)


async def claim(pool, worker='w1'):
    async with pool.acquire() as c:
        return await repo.claim_job(c, worker, 60)


def lease(job):
    return repo.Lease(job['id'], job['lease_owner'], job['lease_epoch'])


async def iteration(pool, job):
    async with pool.acquire() as c:
        av = await repo.upsert_agent_version(c, job['org_id'], '# test source', None)
        return await repo.create_iteration(c, lease(job), 0, av['id'], None, None)


async def record(c, l, it, *, status='passed', attempt=1, phase='main', repetition=1, trace=None, cost_known=True):
    return await repo.record_task_result(c, l, it['id'], 'fix-git', status=status,
        reward=None if status == 'error' else float(status == 'passed'),
        failure_class='infra_error' if status == 'error' else None, duration_s=1.0,
        n_steps=2, exit_reason='done', test_stdout='verifier output', trace=trace or [{'step': attempt}],
        digest={'summary': status}, evaluation_phase=phase, repetition=repetition, attempt=attempt,
        input_tokens=10,output_tokens=5,cost_usd=.01 if cost_known else None,
        model_cost_usd=.01 if cost_known else None,sandbox_cost_usd=0 if cost_known else None,cost_known=cost_known)


@pytest.mark.asyncio
async def test_concurrent_idempotency_payload_and_member_scope(pool):
    p, _ = await principal(pool)
    results = await asyncio.gather(*(submit(pool,p,'same-key') for _ in range(8)))
    assert len({x[0]['id'] for x in results}) == 1
    assert sum(x[1] for x in results) == 1
    async with pool.acquire() as c:
        with pytest.raises(repo.IdempotencyConflict):
            await repo.create_job(c,p,config(max_iterations=2),'same-key')
        user = await repo.create_user(c,p.org_id,p.org_name,'member@example.test','member')
        member = Principal(p.org_id,p.org_name,user['user_id'],user['email'],'member')
        other, created = await repo.create_job(c,member,config(),'same-key')
        assert created and other['id'] != results[0][0]['id']
        assert await repo.get_job(c,member,results[0][0]['id']) is None


@pytest.mark.asyncio
async def test_parallel_claims_cannot_overbook_organisation(pool):
    p, _ = await principal(pool,quota=1)
    for _ in range(8):
        await submit(pool,p)
    results = await asyncio.gather(*(claim(pool,'w'+str(i)) for i in range(8)))
    claimed = [x for x in results if x]
    assert len(claimed) == 1
    async with pool.acquire() as c:
        assert await c.fetchval("SELECT count(*) FROM jobs WHERE status='RUNNING'") == 1


@pytest.mark.asyncio
async def test_stale_writes_rejected_and_original_deadline_preserved(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    first = await claim(pool)
    it = await iteration(pool,first)
    async with pool.acquire() as c:
        await c.execute("UPDATE jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",first['id'])
    second = await claim(pool,'w2')
    assert second['lease_epoch'] == first['lease_epoch'] + 1
    assert second['deadline_at'] == first['deadline_at']
    async with pool.acquire() as c:
        await record(c,lease(second),it)
        with pytest.raises(repo.LeaseLost):
            await record(c,lease(first),it,status='failed',attempt=2)
        assert len(await repo.list_attempts(c,it['id'])) == 1


@pytest.mark.asyncio
async def test_lease_lock_serializes_takeover_with_protected_write(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    first = await claim(pool)
    it = await iteration(pool,first)
    async with pool.acquire() as c1, pool.acquire() as c2:
        async with repo.lease_transaction(c1,lease(first)):
            await record(c1,lease(first),it)
            await c1.execute("UPDATE jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",first['id'])
            # A normal claim skips the row until the protected transaction commits.
            assert await repo.claim_job(c2,'w2',60) is None
        second = await repo.claim_job(c2,'w2',60)
        assert second and second['lease_epoch'] == first['lease_epoch'] + 1


@pytest.mark.asyncio
async def test_lease_expiring_while_waiting_for_row_lock_cannot_write(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    async with pool.acquire() as c1, pool.acquire() as c2:
        await c1.execute("UPDATE jobs SET lease_expires_at=clock_timestamp()+interval '150 milliseconds' WHERE id=$1",job['id'])
        async with c1.transaction():
            await c1.fetchrow('SELECT * FROM jobs WHERE id=$1 FOR UPDATE',job['id'])
            waiting = asyncio.create_task(repo.update_job_progress(c2,lease(job),current_iteration=99))
            await asyncio.sleep(.3)
            assert not waiting.done()
        with pytest.raises(repo.LeaseLost):
            await waiting
        assert await c1.fetchval('SELECT current_iteration FROM jobs WHERE id=$1',job['id']) is None


@pytest.mark.asyncio
async def test_cancellation_prevents_promotion_and_success_terminal_state(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    it = await iteration(pool,job)
    async with pool.acquire() as c:
        await repo.request_cancel(c,p,job['id'])
        with pytest.raises(repo.JobCancelled):
            await repo.complete_iteration(c,lease(job),it['id'],score=1,n_pass=1,n_fail=0,n_error=0,
                                          decision='baseline',decision_reason='baseline')
        assert await c.fetchval('SELECT best_iteration_id FROM jobs WHERE id=$1',job['id']) is None
        await repo.finish_job(c,lease(job),'SUCCEEDED','done')
        assert (await repo.get_job(c,p,job['id']))['status'] == 'CANCELLED'


@pytest.mark.asyncio
async def test_deadline_blocks_new_work_but_records_honest_final_status(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    async with pool.acquire() as c:
        await c.execute("UPDATE jobs SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=$1",job['id'])
        with pytest.raises(repo.BudgetExceeded):
            await repo.reserve_spend(c,lease(job),'too-late',.01,'optimizer')
        with pytest.raises(repo.BudgetExceeded):
            await repo.update_job_progress(c,lease(job),current_iteration=1)
        await repo.finish_job(c,lease(job),'FAILED','budget_exhausted',
                              final_evaluation={'status':'incomplete','reason':'budget_exhausted'})
        final = await repo.get_job(c,p,job['id'])
        assert final['final_evaluation']['status'] == 'incomplete'
        assert final['stop_reason'] == 'budget_exhausted'


@pytest.mark.asyncio
async def test_promotion_updates_iteration_and_best_pointer_in_one_transaction(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    it = await iteration(pool,job)
    async with pool.acquire() as c:
        await repo.complete_iteration(c,lease(job),it['id'],score=.5,n_pass=1,n_fail=1,n_error=0,
                                      decision='baseline',decision_reason='complete',metrics={'complete':True})
        saved = await repo.get_job(c,p,job['id'])
        assert saved['best_iteration_id'] == it['id'] and saved['best_score'] == .5
        version = await repo.upsert_agent_version(c,p.org_id,'# candidate source',it['agent_version_id'])
        candidate = await repo.create_iteration(c,lease(job),1,version['id'],it['id'],{})
        await repo.complete_iteration(c,lease(job),candidate['id'],score=1,n_pass=2,n_fail=0,n_error=0,
                                      decision='accepted',decision_reason='paired improvement',metrics={'complete':True})
        saved = await repo.get_job(c,p,job['id'])
        candidate = await repo.get_iteration(c,job['id'],1)
        assert saved['best_iteration_id'] == candidate['id']
        assert saved['best_score'] == candidate['score'] == 1 and saved['baseline_score'] == .5
        assert candidate['decision'] == 'accepted' and candidate['metrics']['complete']


@pytest.mark.asyncio
async def test_attempts_are_immutable_and_confirmations_do_not_replace_main(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    it = await iteration(pool,job)
    async with pool.acquire() as c:
        await record(c,lease(job),it,status='error',attempt=1)
        await record(c,lease(job),it,status='passed',attempt=2)
        await record(c,lease(job),it,status='failed',phase='confirm_1',attempt=1)
        assert (await repo.task_results(c,it['id']))['fix-git']['status'] == 'passed'
        assert (await repo.attempt_results(c,it['id'],'confirm_1'))['fix-git']['status'] == 'failed'
        with pytest.raises(ValueError,match='immutable'):
            await record(c,lease(job),it,status='failed',attempt=2)
        assert len(await repo.list_attempts(c,it['id'])) == 3
        assert await repo.next_attempt_number(c,it['id']) == 3


@pytest.mark.asyncio
async def test_spend_reservations_are_atomic_and_ambiguous_calls_remain_reserved(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    async def reserve(key):
        async with pool.acquire() as c:
            return await repo.reserve_spend(c,lease(job),key,.75,'agent')
    results = await asyncio.gather(reserve('a'),reserve('b'),return_exceptions=True)
    assert sum(isinstance(x,repo.BudgetExceeded) for x in results) == 1
    reservation = next(x for x in results if not isinstance(x,Exception))
    async with pool.acquire() as c:
        summary = await repo.spend_summary(c,job['id'])
        assert summary['reserved_usd'] == .75
        await repo.settle_spend(c,reservation,.2,input_tokens=10,output_tokens=5)
        await repo.settle_spend(c,reservation,.2,input_tokens=10,output_tokens=5)
        await repo.reserve_spend(c,lease(job),'c',.7,'optimizer')
        with pytest.raises(repo.ReservationConflict):
            await repo.reserve_spend(c,lease(job),'c',.7,'optimizer')
        summary = await repo.spend_summary(c,job['id'])
        assert summary['spent_usd'] == .2 and summary['reserved_usd'] == .7


@pytest.mark.asyncio
async def test_fractional_pricing_remains_exact_and_idempotent(pool):
    p, _ = await principal(pool)
    await submit(pool,p)
    job = await claim(pool)
    async with pool.acquire() as c:
        reservation = await repo.reserve_spend(c,lease(job),'tiny',.0000000075,'agent')
        await repo.settle_spend(c,reservation,.0000000023,input_tokens=1)
        await repo.settle_spend(c,reservation,.0000000023,input_tokens=1)
        assert (await repo.spend_summary(c,job['id']))['spent_usd'] == .0000000023


@pytest.mark.asyncio
async def test_unknown_attempt_costs_remain_null_in_attempts_and_projection(pool):
    p,_=await principal(pool)
    await submit(pool,p)
    job=await claim(pool)
    it=await iteration(pool,job)
    async with pool.acquire() as c:
        await record(c,lease(job),it,cost_known=False)
        attempt=(await repo.list_attempts(c,it['id']))[0]
        projection=(await repo.task_results(c,it['id']))['fix-git']
        for row in (attempt,projection):
            assert row['cost_usd'] is row['model_cost_usd'] is row['sandbox_cost_usd'] is None
            assert not row['cost_known']


@pytest.mark.asyncio
async def test_global_sandbox_slots_bound_different_organisations(pool):
    p1,_ = await principal(pool)
    p2,_ = await principal(pool)
    await submit(pool,p1)
    await submit(pool,p2)
    j1,j2 = await claim(pool),await claim(pool,'w2')
    async with pool.acquire() as c:
        assert await repo.acquire_sandbox_slot(c,lease(j1),'s1',1,30)
        assert not await repo.acquire_sandbox_slot(c,lease(j2),'s2',1,30)
        await repo.release_sandbox_slot(c,lease(j1),'s1')
        assert await repo.acquire_sandbox_slot(c,lease(j2),'s2',1,30)


@pytest.mark.asyncio
async def test_slot_reconciliation_is_provider_scoped_and_legacy_slots_fail_closed(pool):
    p,_=await principal(pool,quota=3)
    for _ in range(3):
        await submit(pool,p)
    jobs=[await claim(pool,'w'+str(i)) for i in range(3)]
    async with pool.acquire() as c:
        for job,key,provider in zip(jobs,('local','foreign','legacy'),('docker:local','docker:other',None)):
            assert await repo.acquire_sandbox_slot(c,lease(job),key,3,30,provider_id=provider)
            await repo.finish_job(c,lease(job),'CANCELLED','cancelled')
        await c.execute("UPDATE sandbox_slots SET expires_at=clock_timestamp()-interval '1 second'")
        slots=await repo.list_reconcilable_sandbox_slots(c,'docker:local')
        assert [s['slot_key'] for s in slots]==['local']
        observed=slots[0]
        assert not await repo.reconcile_sandbox_slot(c,lease(jobs[0]),'local','docker:other',observed['expires_at'])
        assert await repo.reconcile_sandbox_slot(c,lease(jobs[0]),'local','docker:local',observed['expires_at'])
        assert await c.fetchval('SELECT count(*) FROM sandbox_slots')==2
        with pytest.raises(ValueError):
            await repo.list_reconcilable_sandbox_slots(c,None)
        await repo.create_job(c,p,config(),None)
        fresh=await repo.claim_job(c,'new-worker',60)
        # Expired foreign/legacy slots still count until their provider is proven empty.
        assert not await repo.acquire_sandbox_slot(c,lease(fresh),'new',2,30,provider_id='docker:local')
        event=await c.fetchrow("SELECT payload FROM job_events WHERE job_id=$1 AND kind='sandbox.slot_reconciled'",jobs[0]['id'])
        assert event['payload']['provider_absence_confirmed']


@pytest.mark.asyncio
async def test_reconciliation_requires_expiry_and_stale_ownership(pool):
    p,_=await principal(pool,quota=2)
    await submit(pool,p)
    await submit(pool,p)
    active,terminal=await claim(pool),await claim(pool,'w2')
    async with pool.acquire() as c:
        assert await repo.acquire_sandbox_slot(c,lease(active),'active',2,60,provider_id='docker:local')
        assert await repo.acquire_sandbox_slot(c,lease(terminal),'terminal',2,60,provider_id='docker:local')
        assert not await repo.acquire_sandbox_slot(c,lease(active),'active',2,60,provider_id='docker:different')
        active_expiry=await c.fetchval("UPDATE sandbox_slots SET expires_at=clock_timestamp()-interval '1 second' WHERE slot_key='active' RETURNING expires_at")
        await repo.finish_job(c,lease(terminal),'CANCELLED','cancelled')
        terminal_expiry=await c.fetchval("SELECT expires_at FROM sandbox_slots WHERE slot_key='terminal'")
        assert await repo.list_reconcilable_sandbox_slots(c,'docker:local')==[]
        assert not await repo.reconcile_sandbox_slot(c,lease(active),'active','docker:local',active_expiry)
        assert not await repo.reconcile_sandbox_slot(c,lease(terminal),'terminal','docker:local',terminal_expiry)


@pytest.mark.asyncio
async def test_delayed_reconciliation_cannot_release_replaced_slot(pool):
    from datetime import timedelta
    p,_=await principal(pool)
    await submit(pool,p)
    old=await claim(pool)
    async with pool.acquire() as c:
        assert await repo.acquire_sandbox_slot(c,lease(old),'attempt',1,60,provider_id='docker:local')
        expiry=await c.fetchval("UPDATE sandbox_slots SET expires_at=clock_timestamp()-interval '1 second' WHERE slot_key='attempt' RETURNING expires_at")
        await c.execute("UPDATE jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",old['id'])
        fresh=await repo.claim_job(c,'replacement',60)
        assert fresh['lease_epoch']>old['lease_epoch']
        assert not await repo.reconcile_sandbox_slot(c,lease(old),'attempt','docker:local',expiry+timedelta(seconds=1))
        assert await repo.reconcile_sandbox_slot(c,lease(old),'attempt','docker:local',expiry)
        assert await repo.acquire_sandbox_slot(c,lease(fresh),'attempt',1,60,provider_id='docker:local')
        assert not await repo.reconcile_sandbox_slot(c,lease(old),'attempt','docker:local',expiry)
        await repo.release_sandbox_slot(c,lease(old),'attempt')
        assert await c.fetchval("SELECT lease_epoch FROM sandbox_slots WHERE slot_key='attempt'")==fresh['lease_epoch']


@pytest.mark.asyncio
async def test_cancelled_reconciliation_rolls_back_slot_delete_and_audit(pool,monkeypatch):
    p,_=await principal(pool)
    await submit(pool,p)
    job=await claim(pool)
    async with pool.acquire() as c:
        assert await repo.acquire_sandbox_slot(c,lease(job),'attempt',1,60,provider_id='docker:local')
        await repo.finish_job(c,lease(job),'CANCELLED','cancelled')
        expiry=await c.fetchval("UPDATE sandbox_slots SET expires_at=clock_timestamp()-interval '1 second' WHERE slot_key='attempt' RETURNING expires_at")
        entered=asyncio.Event()
        original=repo.add_event
        async def blocked_event(*args,**kwargs):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(repo,'add_event',blocked_event)
        task=asyncio.create_task(repo.reconcile_sandbox_slot(c,lease(job),'attempt','docker:local',expiry))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await c.fetchval('SELECT count(*) FROM sandbox_slots')==1
        assert await c.fetchval("SELECT count(*) FROM job_events WHERE kind='sandbox.slot_reconciled'")==0
        monkeypatch.setattr(repo,'add_event',original)
        assert await repo.reconcile_sandbox_slot(c,lease(job),'attempt','docker:local',expiry)
        assert await c.fetchval('SELECT count(*) FROM sandbox_slots')==0


@pytest.mark.asyncio
async def test_http_tenancy_and_idempotency_conflicts(pool,monkeypatch):
    from service.api.app import create_app
    p1,key1 = await principal(pool)
    p2,key2 = await principal(pool)
    monkeypatch.setattr(db,'_pool',pool)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()),base_url='http://test') as client:
        headers={'Authorization':'Bearer '+key1,'Idempotency-Key':'http-key'}
        body={'tasks':['fix-git'],'max_iterations':0}
        r=await client.post('/v1/jobs',json=body,headers=headers)
        assert r.status_code == 202,r.text
        jid=r.json()['job_id']
        assert (await client.post('/v1/jobs',json=body,headers=headers)).status_code == 200
        changed={**body,'patience':2}
        assert (await client.post('/v1/jobs',json=changed,headers=headers)).status_code == 409
        for field in ('agent_model','optimizer_model'):
            unsupported={**body,field:'unpriced-model'}
            assert (await client.post('/v1/jobs',json=unsupported,headers=headers)).status_code == 422
        for suffix in ('','/iterations','/events','/agent'):
            r=await client.get(f'/v1/jobs/{jid}{suffix}',headers={'Authorization':'Bearer '+key2})
            assert r.status_code == 404,r.text
        assert (await client.post(f'/v1/jobs/{jid}/cancel',headers={'Authorization':'Bearer '+key2})).status_code == 404
        own=await client.get(f'/v1/jobs/{jid}',headers=headers)
        assert own.status_code == 200 and own.json()['spent_usd'] == 0


async def _prior_experiment_fixture(pool, client, key):
    body={'tasks':['fix-git','regex-log','nginx-request-logging'],'max_iterations':2,
          'agent_source':'# frozen original baseline',
          'evaluation_split':{'feedback':['fix-git'],'selection':['regex-log'],'holdout':['nginx-request-logging']}}
    headers={'Authorization':'Bearer '+key}
    response=await client.post('/v1/jobs',json=body,headers=headers)
    assert response.status_code==202,response.text
    job_id=UUID(response.json()['job_id'])
    async with pool.acquire() as c:
        job=await repo.claim_job(c,'prior-fixture',60)
        assert job['id']==job_id
        l=lease(job)
        baseline_version=await repo.upsert_agent_version(c,job['org_id'],body['agent_source'],None)
        baseline=await repo.create_iteration(c,l,0,baseline_version['id'],None,None)
        await repo.complete_iteration(c,l,baseline['id'],score=.5,n_pass=1,n_fail=1,n_error=0,
                                      decision='baseline',decision_reason='baseline')
        proposal={'objective':'quality','hypothesis':'verify command output','targets':['fix-git'],
                  'rationale':'feedback evidence','change_summary':'add validation',
                  'unexpected_raw_trace':'DO_NOT_COPY_PROPOSAL_EXTRA'}
        rejected_version=await repo.upsert_agent_version(c,job['org_id'],'# rejected candidate',baseline_version['id'])
        rejected=await repo.create_iteration(c,l,1,rejected_version['id'],baseline['id'],proposal)
        for task in body['tasks']:
            await c.execute('''INSERT INTO task_runs(iteration_id,task_name,status,reward,trace,test_stdout,digest)
                VALUES($1,$2,'failed',0,$3,$4,$5)''',rejected['id'],task,
                [{'secret':'DO_NOT_COPY_RAW_TRACE'}],'DO_NOT_COPY_STDOUT',{'secret':'DO_NOT_COPY_DIGEST'})
        await repo.complete_iteration(c,l,rejected['id'],score=0,n_pass=0,n_fail=1,n_error=0,
                                      decision='rejected',decision_reason='screened_out')
        interrupted_version=await repo.upsert_agent_version(c,job['org_id'],'# interrupted candidate',baseline_version['id'])
        interrupted=await repo.create_iteration(c,l,2,interrupted_version['id'],baseline['id'],proposal)
        await repo.complete_iteration(c,l,interrupted['id'],score=None,n_pass=0,n_fail=0,n_error=0,
                                      decision='rejected',decision_reason='cancelled',status='errored')
        await repo.finish_job(c,l,'CANCELLED','cancelled',final_evaluation={'status':'incomplete'})
    return body,job_id


@pytest.mark.asyncio
async def test_related_job_freezes_only_completed_rejections_and_unconfirmed_source(pool,monkeypatch):
    from service.api.app import create_app
    import json
    p,key=await principal(pool)
    monkeypatch.setattr(db,'_pool',pool)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()),base_url='http://test') as client:
        body,prior_id=await _prior_experiment_fixture(pool,client,key)
        request={**body,'prior_job_id':str(prior_id),'resume_candidate_index':2}
        headers={'Authorization':'Bearer '+key,'Idempotency-Key':'continuation'}
        response=await client.post('/v1/jobs',json=request,headers=headers)
        assert response.status_code==202,response.text
        assert (await client.post('/v1/jobs',json=request,headers=headers)).status_code==200
        new_id=UUID(response.json()['job_id'])
        async with pool.acquire() as c:
            saved=await repo.get_job(c,p,new_id)
            cfg=saved['config']
            assert cfg['agent_source']==body['agent_source']
            seed=cfg['optimizer_history_seed']
            assert len(seed)==1 and seed[0]['index']==1 and seed[0]['decision']=='rejected'
            assert seed[0]['agent_sha256']==repo.sha256_text('# rejected candidate')
            assert seed[0]['agent_sha256']!=repo.sha256_text('# interrupted candidate')
            assert seed[0]['score'] is None
            assert [r['task_name'] for r in seed[0]['feedback_summary']['tasks']]==['fix-git']
            snapshot=cfg['resume_candidate']
            assert snapshot['source']=='# interrupted candidate' and snapshot['source_iteration_index']==2
            assert snapshot['source_job_id']==str(prior_id)
            assert set(snapshot)=={'source','proposal','source_job_id','source_iteration_index'}
            serialized=json.dumps({'seed':seed,'resume':snapshot})
            assert 'DO_NOT_COPY' not in serialized
            assert 'regex-log' not in serialized and 'nginx-request-logging' not in serialized
            assert saved['best_score'] is None and saved['best_iteration_id'] is None
            assert await c.fetchval('SELECT count(*) FROM iterations WHERE job_id=$1',new_id)==0
            # New jobs own their bytes; subsequent prior-row edits cannot mutate the snapshot.
            await c.execute("UPDATE agent_versions SET content='# external replacement' WHERE id=(SELECT agent_version_id FROM iterations WHERE job_id=$1 AND index=2)",prior_id)
            assert (await repo.get_job(c,p,new_id))['config']['resume_candidate']['source']=='# interrupted candidate'


@pytest.mark.asyncio
async def test_related_job_rejects_unauthorized_and_incompatible_experiments(pool,monkeypatch):
    from service.api.app import create_app
    p,key=await principal(pool)
    _,other_key=await principal(pool)
    monkeypatch.setattr(db,'_pool',pool)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()),base_url='http://test') as client:
        body,prior_id=await _prior_experiment_fixture(pool,client,key)
        request={**body,'prior_job_id':str(prior_id),'resume_candidate_index':2}
        headers={'Authorization':'Bearer '+key}
        denied=await client.post('/v1/jobs',json=request,headers={'Authorization':'Bearer '+other_key})
        assert denied.status_code==404,denied.text
        async with pool.acquire() as c:
            user=await repo.create_user(c,p.org_id,p.org_name,'related-member@example.test','member')
        denied=await client.post('/v1/jobs',json=request,headers={'Authorization':'Bearer '+user['api_key']})
        assert denied.status_code==404
        mismatches=[{'agent_source':'# changed original'}, {'agent_model':'gpt-5.4'},
                    {'tasks':list(reversed(body['tasks']))},
                    {'evaluation_split':{'feedback':['regex-log'],'selection':['fix-git'],'holdout':['nginx-request-logging']}}]
        for change in mismatches:
            response=await client.post('/v1/jobs',json={**request,**change},headers=headers)
            assert response.status_code==422,response.text
        for index in (0,1):
            response=await client.post('/v1/jobs',json={**request,'resume_candidate_index':index},headers=headers)
            assert response.status_code==422,response.text
        missing=await client.post('/v1/jobs',json={**request,'resume_candidate_index':9},headers=headers)
        assert missing.status_code==404
        async with pool.acquire() as c:
            await c.execute("UPDATE jobs SET config=jsonb_set(config,'{benchmark_revision}','\"different-revision\"') WHERE id=$1",prior_id)
        response=await client.post('/v1/jobs',json=request,headers=headers)
        assert response.status_code==422,response.text


@pytest.mark.asyncio
async def test_migration_preserves_existing_results_and_idempotency():
    dsn = os.getenv('AOS_TEST_DATABASE_URL')
    if not dsn:
        pytest.skip('set AOS_TEST_DATABASE_URL to test legacy migration')
    name = 'aos_legacy_' + uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(dsn)
    legacy_dsn = urlunsplit((parts.scheme,parts.netloc,'/'+name,parts.query,parts.fragment))
    c = None
    try:
        c = await asyncpg.connect(legacy_dsn)
        await db._init_conn(c)
        await c.execute((db.MIGRATIONS_DIR/'0001_init.sql').read_text())
        await c.execute('CREATE TABLE schema_migrations(name text PRIMARY KEY,applied_at timestamptz DEFAULT now())')
        await c.execute("INSERT INTO schema_migrations(name) VALUES('0001_init.sql')")
        x = await repo.create_org(c,'legacy','legacy@example.test',1)
        p = Principal(x['org_id'],'legacy',x['user_id'],x['email'],'admin')
        job_id = await c.fetchval("""INSERT INTO jobs(org_id,created_by,config,idempotency_key,status)
            VALUES($1,$2,$3,'legacy-key','QUEUED') RETURNING id""",p.org_id,p.user_id,config())
        av = await repo.upsert_agent_version(c,p.org_id,'# legacy agent',None)
        it_id = await c.fetchval("""INSERT INTO iterations(job_id,index,agent_version_id,phase,status)
            VALUES($1,0,$2,'DONE','completed') RETURNING id""",job_id,av['id'])
        await c.execute("""INSERT INTO task_runs(iteration_id,task_name,status,reward,trace,test_stdout)
            VALUES($1,'fix-git','passed',1,$2,'original verifier')""",it_id,[{'original':True}])
        assert await db.migrate(legacy_dsn) == ['0002_durable_execution.sql','0003_nullable_costs.sql']
        assert await db.migrate(legacy_dsn) == []
        preserved = await repo.list_attempts(c,it_id)
        assert len(preserved) == 1
        assert preserved[0]['evaluation_phase'] == 'main'
        assert preserved[0]['trace'] == [{'original':True}]
        assert preserved[0]['test_stdout'] == 'original verifier'
        same,created = await repo.create_job(c,p,config(),'legacy-key')
        assert same['id'] == job_id and not created
    finally:
        if c is not None:
            await c.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()
