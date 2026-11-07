import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from service import inference
from service.config import Settings, settings


def test_personal_credentials_are_never_a_fallback(monkeypatch):
    monkeypatch.delenv("NEOSIGMA_OPEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "personal-do-not-use")
    monkeypatch.setenv("OPEN_API_KEY", "also-personal")
    cfg = Settings(_env_file=None)
    assert cfg.neosigma_open_api_key is None
    monkeypatch.setattr(settings, "neosigma_open_api_key", None)
    with pytest.raises(inference.InferenceError, match="personal-key fallback is disabled"):
        inference.company_credential()


def test_candidate_cannot_choose_model_spend_or_multimodal_input():
    p = inference.normalize_request({"model": "expensive", "n": 100,
        "max_completion_tokens": 1000000, "stream": True,
        "messages": [{"role": "user", "content": "hello"}]}, "gpt-5.4-mini", 200)
    assert (p["model"], p["n"], p["max_completion_tokens"], p["stream"]) == ("gpt-5.4-mini", 1, 200, False)
    assert inference.reservation_cost(p) > 0
    with pytest.raises(inference.InferenceError, match="Only text"):
        inference.normalize_request({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "gpt-5.4-mini", 200)


def test_credential_destination_is_fixed(monkeypatch):
    for url in ["https://other.example/v1", "http://api.openai.com/v1", "https://api.openai.com/v1?x=1", "https://user@api.openai.com/v1",
                "https://api.openai.com:8080/v1", "https://:password@api.openai.com/v1"]:
        monkeypatch.setattr(settings, "neosigma_base_url", url)
        with pytest.raises(inference.InferenceError):
            inference.upstream_url()


class Pool:
    @asynccontextmanager
    async def acquire(self):
        yield object()


@pytest.mark.asyncio
async def test_reserve_before_network_and_settle_actual_usage(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "neosigma_open_api_key", SecretStr("company-test"))

    async def reserve(*args, **kwargs):
        seen.append("reserve")
        return "r1"

    async def settle(conn, reservation, amount, **kwargs):
        seen.append(("settle", amount, kwargs["input_tokens"], kwargs["output_tokens"]))

    async def handler(request):
        assert seen == ["reserve"]
        assert request.headers["authorization"] == "Bearer company-test"
        seen.append("network")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    original = httpx.AsyncClient
    monkeypatch.setattr(inference.repo, "reserve_spend", reserve)
    monkeypatch.setattr(inference.repo, "settle_spend", settle)
    monkeypatch.setattr(inference.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    _, usage = await inference.paid_completion(Pool(), object(), body={"messages": [{"role": "user", "content": "hi"}]},
        model="gpt-5.4-mini", max_output=200, kind="test", reservation_key="unique")
    assert seen[0:2] == ["reserve", "network"]
    assert usage["cost_usd"] == pytest.approx(0.000165)
    assert seen[2] == ("settle", usage["cost_usd"], 100, 20)


@pytest.mark.asyncio
async def test_ambiguous_network_timeout_retains_reservation(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "neosigma_open_api_key", SecretStr("company-test"))

    async def reserve(*args):
        seen.append("reserve")
        return "r1"

    async def settle(*args, **kw):
        seen.append("settle")

    async def handler(request):
        raise httpx.ReadTimeout("ambiguous")

    original = httpx.AsyncClient
    monkeypatch.setattr(inference.repo, "reserve_spend", reserve)
    monkeypatch.setattr(inference.repo, "settle_spend", settle)
    monkeypatch.setattr(inference.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(httpx.ReadTimeout):
        await inference.paid_completion(Pool(), object(), body={"messages": [{"role": "user", "content": "hi"}]},
            model="gpt-5.4-mini", max_output=200, kind="test", reservation_key="unique")
    assert seen == ["reserve"]


@pytest.mark.asyncio
async def test_gateway_requires_capability_and_fixes_model(monkeypatch):
    seen = []
    async def paid(*args, **kw):
        seen.append(kw)
        return {"choices": []}, {"input_tokens": 2, "output_tokens": 1, "cost_usd": 0.01, "cost_known": True}
    monkeypatch.setattr(inference, "paid_completion", paid)
    session = inference.GatewaySession("http://unused/v1", "scoped-token")
    ctx = SimpleNamespace(job_id=uuid4(), iteration_id=uuid4(), lease_owner="w", lease_epoch=1,
                          agent_model="gpt-5.4-mini", deadline_at=None, extra={'max_tokens_per_call':128})
    app = inference.create_gateway_app(session, ctx, "task", Pool())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/v1/chat/completions", json={})).status_code == 403
        r = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer scoped-token"},
                              json={"model": "expensive", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert seen[0]["model"] == ctx.agent_model
        assert seen[0]['max_output'] == 128
        session.token = "revoked"
        assert (await client.post("/v1/chat/completions", headers={"Authorization": "Bearer scoped-token"}, json={})).status_code == 403
    assert session.usage["cost_usd"] == 0.01


def test_text_estimator_rejects_hidden_audio_and_invalid_output_limits():
    with pytest.raises(inference.InferenceError,match='Unsupported message fields'):
        inference.normalize_request({'messages':[{'role':'assistant','content':None,'audio':{'id':'audio-id'}}]},
                                    'gpt-5.4-mini',200)
    for limit in (0,-1,True):
        with pytest.raises(inference.InferenceError,match='output-token'):
            inference.normalize_request({'messages':[{'role':'user','content':'hello'}]},'gpt-5.4-mini',limit)


def test_tool_requests_fix_compatible_reasoning_mode_and_error_diagnostics_are_structural():
    body={'messages':[{'role':'user','content':'hello'}],
          'tools':[{'type':'function','function':{'name':'bash','parameters':{'type':'object'}}}]}
    assert inference.normalize_request(body,'gpt-5.4-mini',200)['reasoning_effort']=='none'
    response=httpx.Response(400,json={'error':{'type':'invalid_request_error','param':'reasoning_effort',
        'code':None,'message':'Do not persist this free text','secret':'sk-private'}})
    assert inference.provider_error_fields(response)=={'type':'invalid_request_error','param':'reasoning_effort'}


@pytest.mark.asyncio
@pytest.mark.parametrize('provider_usage',[None, [], {'prompt_tokens':None,'completion_tokens':1},
    {'prompt_tokens':True,'completion_tokens':1},{'prompt_tokens':-1,'completion_tokens':1}])
async def test_invalid_usage_is_charged_conservatively(monkeypatch,provider_usage):
    charges=[]
    monkeypatch.setattr(settings,'neosigma_open_api_key',SecretStr('company-test'))
    async def reserve(*args):
        return 'r1'
    async def settle(conn,reservation,amount,**kwargs):
        charges.append((amount,kwargs))
    async def handler(request):
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':provider_usage})
    original=httpx.AsyncClient
    monkeypatch.setattr(inference.repo,'reserve_spend',reserve)
    monkeypatch.setattr(inference.repo,'settle_spend',settle)
    monkeypatch.setattr(inference.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    body={'messages':[{'role':'user','content':'hi'}]}
    _,usage=await inference.paid_completion(Pool(),object(),body=body,model='gpt-5.4-mini',max_output=200,
                                           kind='test',reservation_key='unique')
    assert not usage['cost_known']
    assert charges[0][0] == inference.reservation_cost(inference.normalize_request(body,'gpt-5.4-mini',200))
    assert charges[0][1]['input_tokens'] == charges[0][1]['output_tokens'] == 0


@pytest.mark.asyncio
async def test_gateway_revokes_calls_queued_behind_active_completion(monkeypatch):
    entered,release=asyncio.Event(),asyncio.Event()
    calls=[]
    async def paid(*args,**kw):
        calls.append(kw)
        entered.set()
        await release.wait()
        return {'choices':[]},{'input_tokens':1,'output_tokens':1,'cost_usd':.01,'cost_known':True}
    monkeypatch.setattr(inference,'paid_completion',paid)
    session=inference.GatewaySession('http://unused/v1','trial-token')
    ctx=SimpleNamespace(job_id=uuid4(),iteration_id=uuid4(),lease_owner='w',lease_epoch=1,
                        agent_model='gpt-5.4-mini',deadline_at=None)
    app=inference.create_gateway_app(session,ctx,'task',Pool())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        async def post():
            return await client.post('/v1/chat/completions',headers={'Authorization':'Bearer trial-token'},
                                     json={'messages':[{'role':'user','content':'hi'}]})
        first=asyncio.create_task(post())
        await entered.wait()
        second=asyncio.create_task(post())
        await asyncio.sleep(.01)
        session.token='revoked'
        release.set()
        assert (await first).status_code == 200
        assert (await second).status_code == 403
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('error,reason',[
    (inference.repo.JobCancelled,inference.GatewayStopReason.CANCELLED),
    (inference.repo.BudgetExceeded,inference.GatewayStopReason.BUDGET_EXHAUSTED),
    (inference.repo.LeaseLost,inference.GatewayStopReason.LEASE_LOST),
])
async def test_gateway_keeps_trusted_stop_reason_and_blocks_further_calls(monkeypatch,error,reason):
    calls=[]
    async def paid(*args,**kw):
        calls.append(kw)
        raise error()
    monkeypatch.setattr(inference,'paid_completion',paid)
    session=inference.GatewaySession('http://unused/v1','token')
    ctx=SimpleNamespace(job_id=uuid4(),iteration_id=uuid4(),lease_owner='w',lease_epoch=1,
                        agent_model='gpt-5.4-mini',deadline_at=None)
    app=inference.create_gateway_app(session,ctx,'task',Pool())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.post('/v1/chat/completions',headers={'Authorization':'Bearer token'},
                                  json={'messages':[{'role':'user','content':'hi'}]})
        assert response.status_code == 409 and response.json()['detail'] == error.__name__
        assert session.stop_reason == reason
        again=await client.post('/v1/chat/completions',headers={'Authorization':'Bearer token'},
                                 json={'messages':[{'role':'user','content':'try again'}]})
        assert again.status_code == 409 and len(calls) == 1


@pytest.mark.asyncio
async def test_gateway_marks_ambiguous_usage_unknown(monkeypatch):
    async def paid(*args,**kw):
        raise httpx.ReadTimeout('provider response lost')
    monkeypatch.setattr(inference,'paid_completion',paid)
    session=inference.GatewaySession('http://unused/v1','token')
    ctx=SimpleNamespace(job_id=uuid4(),iteration_id=uuid4(),lease_owner='w',lease_epoch=1,
                        agent_model='gpt-5.4-mini',deadline_at=None)
    app=inference.create_gateway_app(session,ctx,'task',Pool())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.post('/v1/chat/completions',headers={'Authorization':'Bearer token'},
                                  json={'messages':[{'role':'user','content':'hi'}]})
        assert response.status_code == 502 and not session.usage['cost_known']
        assert session.stop_reason == inference.GatewayStopReason.INFERENCE_ERROR


@pytest.mark.asyncio
async def test_optimizer_handles_empty_provider_choices(monkeypatch):
    from service.worker import llm_optimizer
    async def paid(*args,**kwargs):
        return {'choices':[]},{}
    monkeypatch.setattr(llm_optimizer,'paid_completion',paid)
    optimizer=llm_optimizer.LLMOptimizer(Pool(),SimpleNamespace(job_id=uuid4()),{'config':{}})
    inp=SimpleNamespace(history=[],agent_source='# source',scoreboard={},failure_dossiers=[],
                        learnings='',constraints={},iteration_index=1)
    with pytest.raises(inference.InferenceError,match='no completion choice'):
        await optimizer.propose(inp)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_stage', ['acquire', 'reserve', 'settle'])
async def test_gateway_database_failures_stop_trial_and_keep_errors_redacted(monkeypatch, failure_stage):
    import asyncpg
    calls = {'provider': 0, 'reserved': 0}
    monkeypatch.setattr(settings, 'neosigma_open_api_key', SecretStr('company-test'))

    class DatabasePool:
        @asynccontextmanager
        async def acquire(self):
            if failure_stage == 'acquire':
                raise asyncpg.PostgresConnectionError('private database connection details')
            yield object()

    async def reserve(*args, **kwargs):
        if failure_stage == 'reserve':
            raise asyncpg.PostgresConnectionError('private database connection details')
        calls['reserved'] += 1
        return 'reservation'

    async def settle(*args, **kwargs):
        if failure_stage == 'settle':
            raise asyncpg.PostgresConnectionError('private database connection details')

    async def provider(request):
        calls['provider'] += 1
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(inference.repo, 'reserve_spend', reserve)
    monkeypatch.setattr(inference.repo, 'settle_spend', settle)
    monkeypatch.setattr(inference.httpx, 'AsyncClient',
        lambda **kw: original_client(transport=httpx.MockTransport(provider), **kw))
    session = inference.GatewaySession('http://unused/v1', 'trial-token')
    ctx = SimpleNamespace(job_id=uuid4(), iteration_id=uuid4(), lease_owner='w', lease_epoch=1,
                          agent_model='gpt-5.4-mini', deadline_at=None)
    app = inference.create_gateway_app(session, ctx, 'task', DatabasePool())
    async with original_client(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                               base_url='http://test') as client:
        arguments = {'headers': {'Authorization': 'Bearer trial-token'},
                     'json': {'messages': [{'role': 'user', 'content': 'hello'}]}}
        response = await client.post('/v1/chat/completions', **arguments)
        assert response.status_code == 502
        assert 'private database' not in response.text
        assert session.stop_reason == inference.GatewayStopReason.INFERENCE_ERROR
        assert not session.usage['cost_known']
        assert session.error_details == {'error_type': 'PostgresConnectionError'}
        again = await client.post('/v1/chat/completions', **arguments)
        assert again.status_code == 409
    expected = 1 if failure_stage == 'settle' else 0
    assert calls == {'provider': expected, 'reserved': expected}
