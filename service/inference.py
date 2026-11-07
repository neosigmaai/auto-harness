"""Company-key-only inference, metered before dispatch; scoped trial HTTP gateway.

Candidate code receives a revocable token, never the upstream credential. Prices
are conservative text-token estimates, not provider invoices. Ambiguous requests
keep their maximum reservation rather than silently releasing possible spend.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import secrets
import socket
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from . import repo
from .config import settings
from .db import get_pool


class InferenceError(RuntimeError):
    def __init__(self, message: str, *, provider_error: dict | None = None):
        super().__init__(message)
        self.provider_error = provider_error


def provider_error_fields(response: httpx.Response) -> dict:
    """Keep only bounded structural diagnostics; never provider free text."""
    try:
        value = response.json()
    except ValueError:
        return {}
    error = value.get('error') if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return {}
    return {key: value for key in ('type', 'code', 'param')
            if isinstance(value := error.get(key), str)
            and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.\[\]-]{0,79}', value)
            and not value.startswith('sk-')}


def company_credential() -> str:
    value = settings.neosigma_open_api_key
    if value is None or not value.get_secret_value():
        raise InferenceError("NEOSIGMA_OPEN_API_KEY is required; personal-key fallback is disabled")
    return value.get_secret_value()


def upstream_url() -> str:
    base = settings.neosigma_base_url.rstrip("/")
    parsed = urlsplit(base)
    try:
        allowed_port = parsed.port in (None, 443)
    except ValueError:
        allowed_port = False
    if (parsed.scheme != "https" or parsed.hostname != "api.openai.com" or parsed.path != "/v1"
            or parsed.query or parsed.fragment or parsed.username is not None or not allowed_port):
        raise InferenceError("Only the explicitly approved https://api.openai.com/v1 endpoint is enabled")
    return base + "/chat/completions"


def normalize_request(body: dict, model: str, max_output: int) -> dict:
    """Allow text chat/tool calls only. All expensive/privileged options are fixed."""
    if model not in settings.model_prices:
        raise InferenceError(f"No verified price configuration for model {model!r}")
    prices = settings.model_prices[model]
    if len(prices) != 2 or any(not math.isfinite(p) or p < 0 for p in prices):
        raise InferenceError("Model prices must be finite nonnegative input/output rates")
    if type(max_output) is not int or max_output <= 0:
        raise InferenceError("A positive trusted output-token limit is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 240:
        raise InferenceError("messages must contain between 1 and 240 entries")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise InferenceError("Invalid message")
        if m.get("content") is not None and not isinstance(m["content"], str):
            raise InferenceError("Only text content is supported by the budget estimator")
        if set(m) - {"role", "content", "name", "tool_call_id", "tool_calls"}:
            raise InferenceError("Unsupported message fields; only text and function calls are allowed")
        calls = m.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list) or any(
            not isinstance(call, dict) or call.get("type") != "function"
            or not isinstance(call.get("function"), dict)
            or not isinstance(call["function"].get("name"), str)
            or not isinstance(call["function"].get("arguments"), str)
            for call in calls
        ):
            raise InferenceError("Only function call messages are supported")
    payload = {"model": model, "messages": messages, "max_completion_tokens": max_output, "stream": False, "n": 1}
    if model.startswith("gpt-5"):
        # GPT-5.4 Chat Completions rejected tools + 'low' in the live smoke.
        # Keep this fixed for both baseline/candidate. Reasoning with tools needs
        # a Responses adapter rather than a silent per-request retry/downgrade.
        payload["reasoning_effort"] = "none" if body.get('tools') else "low"
    for name in ("tools", "tool_choice", "response_format"):
        if name in body:
            payload[name] = body[name]
    tools = payload.get("tools", [])
    if not isinstance(tools, list) or len(tools) > 16 or any(not isinstance(t, dict) or t.get("type") != "function" for t in tools):
        raise InferenceError("Only bounded function tools are supported")
    if len(json.dumps(payload, ensure_ascii=False).encode()) > settings.max_inference_body_bytes:
        raise InferenceError("Inference input exceeds the configured size limit")
    return payload


def reservation_cost(payload: dict) -> float:
    # UTF-8 bytes are a conservative bound on BPE text tokens. Extra headroom
    # covers chat/tool framing; reserve twice the serialized size plus framing.
    tokens_bound = 2 * len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
    pin, pout = settings.model_prices[payload["model"]]
    return (tokens_bound * pin + payload["max_completion_tokens"] * pout) / 1_000_000


async def paid_completion(pool, lease: repo.Lease, *, body: dict, model: str,
                          max_output: int, kind: str, reservation_key: str,
                          timeout: float | None = None) -> tuple[dict, dict]:
    credential = company_credential()
    endpoint = upstream_url()
    payload = normalize_request(body, model, max_output)
    bound = reservation_cost(payload)
    async with pool.acquire() as conn:
        reservation = await repo.reserve_spend(conn, lease, reservation_key, bound, kind)
    started = time.monotonic()
    try:
        # No SDK/environment defaults or automatic retries: every external call
        # has its own durable reservation and uses only the company credential.
        async with httpx.AsyncClient(timeout=timeout or settings.inference_timeout_sec,
                                     trust_env=False, follow_redirects=False) as client:
            response = await client.post(endpoint, headers={"Authorization": "Bearer " + credential}, json=payload)
        if response.status_code != 200:
            # A rejected HTTP request is not a successful completion. Do not
            # expose provider bodies, request prompts, headers, or secrets.
            diagnostics = provider_error_fields(response)
            if response.status_code in {400, 401, 403, 404, 422, 429}:
                async with pool.acquire() as conn:
                    await repo.settle_spend(conn, reservation, 0.0,
                        metadata={"http_status": response.status_code, 'provider_error': diagnostics})
            raise InferenceError(f"Model provider returned HTTP {response.status_code}", provider_error=diagnostics)
        data = response.json()
        if not isinstance(data, dict):
            raise InferenceError("Model provider returned an invalid response object")
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        known = all(type(usage.get(name)) is int and usage[name] >= 0
                    for name in ("prompt_tokens", "completion_tokens"))
        ni, no = (usage["prompt_tokens"], usage["completion_tokens"]) if known else (0, 0)
        pin, pout = settings.model_prices[model]
        cost = (ni * pin + no * pout) / 1_000_000 if known else bound
        async with pool.acquire() as conn:
            await repo.settle_spend(conn, reservation, cost, input_tokens=ni, output_tokens=no,
                                   metadata={"model": model, "usage_present": known, "pricing_basis": "standard_text_uncached_estimate",
                                             "duration_s": round(time.monotonic() - started, 3)})
        return data, {"input_tokens": ni, "output_tokens": no, "cost_usd": cost, "cost_known": known}
    except (asyncio.CancelledError, httpx.HTTPError, ValueError):
        # The request may have reached the provider. Its reservation remains
        # charged in available-budget calculations until reconciled explicitly.
        raise


class GatewayStopReason(str, Enum):
    BUDGET_EXHAUSTED = 'budget_exhausted'
    CANCELLED = 'cancelled'
    LEASE_LOST = 'lease_lost'
    INFERENCE_ERROR = 'inference_error'


@dataclass
class GatewaySession:
    base_url: str
    token: str
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_known": True})
    stop_reason: GatewayStopReason | None = None
    error_details: dict | None = None


def create_gateway_app(session: GatewaySession, ctx, task: str, pool) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    calls = 0
    lock = asyncio.Lock()
    lease = repo.Lease(ctx.job_id, ctx.lease_owner, ctx.lease_epoch)

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        nonlocal calls
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied, "Bearer " + session.token):
            raise HTTPException(403, "Invalid or expired trial capability")
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
            if len(buf) > settings.max_inference_body_bytes:
                raise HTTPException(413, "Request too large")
        try:
            body = json.loads(buf)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "Invalid JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "Object required")
        async with lock:
            # A request may have queued while a previous completion was active.
            # Revoke queued work too when its trial context has already closed.
            if not secrets.compare_digest(supplied, "Bearer " + session.token):
                raise HTTPException(403, "Invalid or expired trial capability")
            if session.stop_reason is not None:
                raise HTTPException(409, session.stop_reason.value)
            if calls >= settings.max_model_calls_per_trial:
                raise HTTPException(429, "Trial model-call limit reached")
            deadline = getattr(ctx, "deadline_at", None)
            remaining = settings.inference_timeout_sec
            if deadline:
                if isinstance(deadline, str):
                    deadline = datetime.fromisoformat(deadline)
                remaining = min(remaining, (deadline - datetime.now(timezone.utc)).total_seconds())
                if remaining <= 0:
                    session.stop_reason = GatewayStopReason.BUDGET_EXHAUSTED
                    raise HTTPException(410, "Job deadline reached")
            calls += 1
            try:
                requested_output = (getattr(ctx, 'extra', None) or {}).get(
                    'max_tokens_per_call', settings.agent_max_output_tokens)
                if type(requested_output) is not int or requested_output <= 0:
                    raise InferenceError('Invalid trial output-token limit')
                data, usage = await paid_completion(pool, lease, body=body, model=ctx.agent_model,
                    max_output=min(requested_output, settings.agent_max_output_tokens), kind="agent",
                    reservation_key=f"agent:{ctx.iteration_id}:{task}:{getattr(ctx, 'evaluation_phase', 'main')}:{getattr(ctx, 'repetition', 1)}:{uuid.uuid4().hex}",
                    timeout=remaining)
            except (repo.LeaseLost, repo.BudgetExceeded, repo.JobCancelled) as exc:
                session.stop_reason = (
                    GatewayStopReason.LEASE_LOST if isinstance(exc, repo.LeaseLost)
                    else GatewayStopReason.BUDGET_EXHAUSTED if isinstance(exc, repo.BudgetExceeded)
                    else GatewayStopReason.CANCELLED)
                raise HTTPException(409, type(exc).__name__) from None
            except asyncio.CancelledError:
                session.usage['cost_known'] = False
                raise
            except Exception as exc:
                # Trusted DB/client failures must stop the trial too. A candidate
                # can catch HTTP errors, so an HTTP 500 alone is not a fence.
                # Failed requests can retain a charge in the durable ledger.
                # The sum of successful calls is then only partial task usage.
                session.usage['cost_known'] = False
                session.stop_reason = GatewayStopReason.INFERENCE_ERROR
                session.error_details = {'error_type': type(exc).__name__}
                if isinstance(exc, InferenceError) and exc.provider_error:
                    session.error_details['provider_error'] = exc.provider_error
                raise HTTPException(502, "Inference failed; see trusted usage ledger") from None
            for name in ("input_tokens", "output_tokens", "cost_usd"):
                session.usage[name] += usage[name]
            session.usage["cost_known"] = session.usage["cost_known"] and usage["cost_known"]
            return data

    return app


@asynccontextmanager
async def gateway_session(ctx, task: str):
    """One ephemeral capability/server per task attempt; closed on cancellation."""
    pool = await get_pool()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((settings.inference_bind_host, 0))
    port = sock.getsockname()[1]
    session = GatewaySession(f"http://{settings.inference_advertise_host}:{port}/v1", secrets.token_urlsafe(32))
    server = uvicorn.Server(uvicorn.Config(create_gateway_app(session, ctx, task, pool),
        log_level="error", access_log=False, lifespan="off", limit_concurrency=8, timeout_graceful_shutdown=2))
    server.install_signal_handlers = lambda: None
    server.capture_signals = lambda: nullcontext()
    running = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if running.done():
                    await running
                    raise InferenceError("Gateway failed to start")
                await asyncio.sleep(0.01)
        yield session
    finally:
        session.token = "revoked-" + secrets.token_urlsafe(32)
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(running), 4)
        except (TimeoutError, asyncio.CancelledError):
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        sock.close()
