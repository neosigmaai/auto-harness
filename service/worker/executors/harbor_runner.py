"""Pinned Harbor runner with a full candidate process inside each Docker task.

Each task has its own subprocess, container, revocable model session, admission
slot and deadline. No candidate module is imported by this trusted worker.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import time
from uuid import uuid4

from ... import repo
from ...config import settings
from ...db import get_pool
from ...tasks_catalog import BY_NAME
from .base import OnResult, RunContext, TaskResult

ROOT = Path(__file__).resolve().parents[3]
SANDBOX = ROOT / "service" / "sandbox"
BENCHMARK_URL = "https://github.com/harbor-framework/terminal-bench-2.git"
HARBOR_VERSION = "0.20.0"
RELAY_IMAGE = "python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54"
FINALIZE_TIMEOUT_SEC = 45
log = logging.getLogger("worker.sandbox")
VERIFIER_CONTRACTS = json.loads((SANDBOX / "verifier_contracts.json").read_text())


def runner_environment() -> dict[str, str]:
    """Allowlist only OS/Docker configuration; never inherit model/DB credentials."""
    allowed = {"PATH", "HOME", "USER", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", NO_COLOR="1", GIT_TERMINAL_PROMPT="0")
    return result


def write_private_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2))
    path.chmod(0o600)


def raise_gateway_stop(session):
    """Trusted control-plane stops must never become candidate task outcomes."""
    reason = getattr(session, "stop_reason", None)
    if reason == "budget_exhausted":
        raise repo.BudgetExceeded("Inference budget exhausted")
    if reason == "cancelled":
        raise repo.JobCancelled("Job cancellation requested")
    if reason == "lease_lost":
        raise repo.LeaseLost("Inference lease is no longer current")
    if reason == "inference_error":
        raise RuntimeError("Trusted inference failed; this trial cannot be scored")


def sandbox_overlay(path: Path, *, gateway_url: str, attempt_id: str, job_id: str,
                    epoch: int, expires_at: float) -> dict:
    """Only the trusted relay sees a host mount; the candidate sees none."""
    labels = {"aos.managed": "true", "aos.job": job_id, "aos.epoch": str(epoch),
              "aos.attempt": attempt_id, "aos.expires_at": str(expires_at)}
    proxy = "http://aos-relay:8080"
    main = {
        "networks": ["aos-internal"], "labels": labels,
        "security_opt": ["no-new-privileges:true"], "cap_drop": ["NET_RAW", "NET_ADMIN"],
        "pids_limit": 256, "stop_grace_period": "1s",
        "tmpfs": ["/logs:rw,nosuid,noexec,size=134217728,mode=1777"],
        "environment": {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "http_proxy": proxy,
                        "https_proxy": proxy, "NO_PROXY": "localhost,127.0.0.1,::1,aos-relay",
                        "no_proxy": "localhost,127.0.0.1,::1,aos-relay"},
        "depends_on": {"aos-relay": {"condition": "service_healthy"}},
        # Compute the absolute deadline inside the container *at startup*.
        # Image download/build time must not extend an abandoned trial's life.
        # Docker Compose consumes one dollar-escaping layer before the shell.
        "command": ["sh", "-c", f'aos_remaining=$$(({int(expires_at)} - $$(date +%s))); '
                    'if [ "$$aos_remaining" -gt 0 ]; then exec sleep "$$aos_remaining"; fi'],
    }
    relay = {
        "image": RELAY_IMAGE, "command": ["python3", "-B", "/relay.py"],
        "networks": ["aos-internal", "aos-egress"], "labels": labels,
        "volumes": [{"type": "bind", "source": str(SANDBOX / "relay.py"), "target": "/relay.py", "read_only": True}],
        "environment": {"AOS_GATEWAY_URL": gateway_url, "AOS_EXPIRES_AT": str(expires_at)},
        "extra_hosts": ["host.docker.internal:host-gateway"],
        "security_opt": ["no-new-privileges:true"], "cap_drop": ["ALL"],
        "read_only": True, "user": "65534:65534", "pids_limit": 64,
        "mem_limit": "128m", "cpus": 0.5, "stop_grace_period": "1s",
        "healthcheck": {"test": ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"],
                        "interval": "1s", "timeout": "3s", "retries": 15},
    }
    value = {"services": {"main": main, "aos-relay": relay},
             "networks": {"aos-internal": {"internal": True, "labels": labels},
                          "aos-egress": {"labels": labels}}}
    path.write_text(json.dumps(value, indent=2))
    return value


def harbor_config(task: str, directory: Path, adapter_config: Path, overlay: Path,
                  agent_timeout: float, verifier_timeout: float, *,
                  benchmark_revision: str | None = None, prepared_image: dict | None = None) -> dict:
    return {
        "job_name": "trial", "jobs_dir": str(directory / "harbor"), "n_attempts": 1,
        "n_concurrent_trials": 1, "quiet": True, "retry": {"max_retries": 0},
        "environment": {"import_path": "service.sandbox.docker_environment:IsolatedDockerEnvironment",
                        "delete": True, "force_build": False,
                        "override_cpus": 1, "override_memory_mb": 2048,
                        "cpu_enforcement_policy": "limit", "memory_enforcement_policy": "limit",
                        "kwargs": {"provenance_path": str(directory / "sandbox-provenance.json"),
                                   "prepared_image": prepared_image},
                        "extra_docker_compose": [str(overlay)]},
        "agents": [{"import_path": "service.sandbox.installed_agent:ServiceInstalledAgent",
                    "override_timeout_sec": agent_timeout + 10, "max_timeout_sec": agent_timeout + 10,
                    "override_setup_timeout_sec": 420,
                    "kwargs": {"host_config_path": str(adapter_config)}}],
        "verifier": {"override_timeout_sec": verifier_timeout, "max_timeout_sec": verifier_timeout,
                     "import_path": "service.sandbox.verifier:PinnedPytestVerifier",
                     "kwargs": {"expected_script_sha256": VERIFIER_CONTRACTS["tasks"].get(task)}},
        "tasks": [{"path": task, "git_url": BENCHMARK_URL,
                   "git_commit_id": benchmark_revision or settings.benchmark_revision}],
    }


async def docker_command(*arguments: str, timeout=30) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec("docker", *arguments,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=runner_environment())
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    return process.returncode, (stdout if process.returncode == 0 else stderr).decode(errors="replace")


async def remove_attempt(attempt_id: str):
    code, output = await docker_command("ps", "-aq", "--filter", f"label=aos.attempt={attempt_id}")
    if code:
        raise RuntimeError("Cannot verify sandbox cleanup: Docker is unavailable")
    ids = output.split()
    if ids:
        code, _ = await docker_command("rm", "-f", *ids)
        if code:
            raise RuntimeError("Docker container removal failed; keeping admission slot reserved")
    code, remaining = await docker_command("ps", "-aq", "--filter", f"label=aos.attempt={attempt_id}")
    if code or remaining.strip():
        raise RuntimeError("Cannot confirm sandbox container removal; keeping admission slot reserved")
    code, output = await docker_command("network", "ls", "-q", "--filter", f"label=aos.attempt={attempt_id}")
    if code:
        raise RuntimeError("Cannot inspect sandbox networks; keeping admission slot reserved")
    if output.split():
        code, _ = await docker_command("network", "rm", *output.split())
        if code:
            raise RuntimeError("Docker network removal failed; keeping admission slot reserved")
    code, remaining = await docker_command("network", "ls", "-q", "--filter", f"label=aos.attempt={attempt_id}")
    if code or remaining.strip():
        raise RuntimeError("Cannot confirm sandbox network removal; keeping admission slot reserved")


async def docker_provider_id() -> str:
    code, identity = await docker_command("info", "--format", "{{.ID}}")
    identity = identity.strip()
    if code or not identity or len(identity) > 200:
        raise RuntimeError("Cannot establish Docker daemon identity")
    return "docker:" + identity


async def await_finalizer(task):
    """Delay repeated cancellation until a bounded cleanup task has finished."""
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    result = task.result()
    if interrupted:
        raise asyncio.CancelledError()
    return result


async def reap_expired_sandboxes(*, pool=None, reconcile_slots=True):
    """Run periodically on each Docker worker; only service-labelled containers."""
    code, output = await docker_command("ps", "-aq", "--filter", "label=aos.managed=true")
    if code:
        raise RuntimeError("Cannot inspect managed Docker sandboxes")
    attempts = set()
    if output.split():
        code, inspected = await docker_command("inspect", *output.split())
        if code:
            raise RuntimeError("Cannot inspect managed Docker sandbox deadlines")
        for item in json.loads(inspected):
            labels = item.get("Config", {}).get("Labels", {}) or {}
            try:
                if float(labels.get("aos.expires_at", "inf")) < time.time():
                    attempts.add(labels["aos.attempt"])
            except (ValueError, KeyError):
                continue
    for attempt in attempts:
        await remove_attempt(attempt)
    if reconcile_slots:
        provider = await docker_provider_id()
        pool = pool or await get_pool()
        async with pool.acquire() as connection:
            rows = await repo.list_reconcilable_sandbox_slots(connection, provider)
        for row in rows:
            # Only expired, stale, exact-daemon reservations are eligible. An
            # early negative listing could otherwise race an orphan image pull.
            await remove_attempt(row["slot_key"])
            if await docker_provider_id() != provider:
                raise RuntimeError("Docker daemon changed during reconciliation")
            lease = repo.Lease(row["job_id"], row["lease_owner"], row["lease_epoch"])
            async with pool.acquire() as connection:
                await repo.reconcile_sandbox_slot(connection, lease, row["slot_key"], provider, row["expires_at"])


def _read_text(path: Path, limit=32000):
    if path.is_file() and not path.is_symlink():
        with path.open(errors="replace") as handle:
            return handle.read(limit)
    return None


def parse_trial_result(task: str, path: Path, duration: float, *,
                       benchmark_revision: str | None = None) -> TaskResult:
    data = json.loads(path.read_text())
    if data.get("task_name") != task:
        raise ValueError("Harbor returned a result for a different task")
    actual_revision = ((data.get("config") or {}).get("task") or {}).get("git_commit_id")
    if actual_revision and benchmark_revision and actual_revision != benchmark_revision:
        raise ValueError("Harbor result does not match the job's frozen benchmark revision")
    revision = actual_revision or benchmark_revision or settings.benchmark_revision
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    raw_reward = rewards.get("reward")
    reward_valid = type(raw_reward) in (int, float) and math.isfinite(raw_reward) and raw_reward in (0, 1)
    reward = float(raw_reward) if reward_valid else None
    error = data.get("exception_info") or {}
    error_type = error.get("exception_type", "")
    if error:
        status = "timeout" if "Timeout" in error_type else "error"
        failure = "timeout" if status == "timeout" else (
            "agent_error" if "Candidate runtime exited" in error.get("exception_message", "") else "infra_error")
    else:
        status = "passed" if reward is not None and reward >= 1.0 else "failed"
        failure = None if status == "passed" else "wrong_output"
    evidence_text = _read_text(path.parent / "verifier-execution.json")
    try:
        evidence = json.loads(evidence_text) if evidence_text else {}
    except ValueError:
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}
    reason = None
    if not reward_valid:
        reason = "missing_or_invalid_binary_reward"
    if evidence.get("valid") is not True or evidence.get("contract") != VERIFIER_CONTRACTS["contract"]:
        reason = evidence.get("reason") or "missing_valid_verifier_execution_evidence"
    if reason:
        # Shell scripts can echo reward=0 and exit successfully even when pytest
        # never starts. Such results must not enter quality/efficiency decisions.
        reward = None
        status = "error"
        failure = failure if failure in {"agent_error", "timeout"} else "infra_error"
        error_type = error_type or "VerifierProtocolError"
    trace_text = _read_text(path.parent / "agent" / "trace.json", 2_000_000)
    trace = None
    if trace_text:
        try:
            trace = json.loads(trace_text)
        except ValueError:
            pass
    events_text = _read_text(path.parent / "agent" / "events.jsonl", 4_000_000) or ""
    events = []
    for line in events_text.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    # A crash can occur before the candidate saves its final trace.
    if trace is None:
        messages = [e.get("messages") for e in events if e.get("event") == "model_request"]
        trace = messages[-1] if messages else None
    stdout = _read_text(path.parent / "verifier" / "test-stdout.txt")
    if stdout is None:
        stdout = _read_text(path.parent / "verifier" / "test_stdout.txt")
    return TaskResult(task, status, reward, failure, duration,
                      sum(e.get("event") == "model_request" for e in events),
                      error_type or "agent_declared_done", stdout, trace,
                      data.get("trial_name"),
                      metadata={"harbor_version": HARBOR_VERSION, "benchmark_revision": revision,
                                "verifier_contract_version": VERIFIER_CONTRACTS["contract"],
                                "verifier_validity": evidence,
                                "verification_invalid_reason": reason,
                                "raw_reward": raw_reward if reward_valid else repr(raw_reward),
                                "exception_type": error_type or None,
                                "error_detail": error.get("exception_message", "")[:2000]})


class HarborExecutor:
    name = "harbor"

    def __init__(self, *, session_factory=None, pool=None):
        self._cancelled = asyncio.Event()
        self._processes = set()
        self._attempts = set()
        self._session_factory = session_factory
        self._pool = pool
        self._slot_records = {}
        self._attempt_processes = {}
        self._finalizers = {}

    async def _finalize(self, key, record):
        async with asyncio.timeout(FINALIZE_TIMEOUT_SEC):
            # Admission/spawn may have reached an external system when their
            # caller was cancelled. Wait for their registered tasks first.
            if record:
                for name in ("admission", "spawn"):
                    pending = record.get(name)
                    if pending is not None:
                        try:
                            await asyncio.shield(pending)
                        except Exception:
                            pass  # Cleanup still has to verify the provider.
                if await docker_provider_id() != record["provider_id"]:
                    raise RuntimeError("Docker daemon changed; reservation retained")
            process = self._attempt_processes.get(key)
            if process:
                await self._terminate(process)
            await remove_attempt(key)
            if record:
                async with record["pool"].acquire() as connection:
                    await repo.release_sandbox_slot(connection, record["lease"], key)
            self._attempts.discard(key)
            self._slot_records.pop(key, None)
            self._attempt_processes.pop(key, None)
            if process:
                self._processes.discard(process)

    async def _finish_attempt(self, key):
        task = self._finalizers.get(key)
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.create_task(self._finalize(key, self._slot_records.get(key)))
            self._finalizers[key] = task
        await await_finalizer(task)

    @asynccontextmanager
    async def _slot(self, ctx, key, ttl):
        pool = self._pool or await get_pool()
        lease = ctx.lease or repo.Lease(ctx.job_id, ctx.lease_owner, ctx.lease_epoch)
        provider = await docker_provider_id()
        record = {"pool": pool, "lease": lease, "provider_id": provider}
        self._slot_records[key] = record
        acquired = False

        async def admit():
            async with pool.acquire() as connection:
                return await repo.acquire_sandbox_slot(connection, lease, key, settings.global_trial_slots, ttl,
                                                       provider_id=provider)

        try:
            while not self._cancelled.is_set():
                record["admission"] = asyncio.create_task(admit())
                acquired = await asyncio.shield(record["admission"])
                if acquired:
                    break
                await asyncio.sleep(0.5)
            if not acquired or self._cancelled.is_set():
                raise asyncio.CancelledError()
            yield
        finally:
            # Even an interrupted admission may have committed. Exact-key
            # release is safe after provider proof, whether admission won or not.
            await self._finish_attempt(key)

    async def run_benchmark(self, ctx: RunContext, agent_source: str, tasks: list[str],
                            on_result: OnResult, on_start=None) -> dict[str, TaskResult]:
        if settings.sandbox_provider != "docker":
            raise ValueError("Only the Docker sandbox adapter is implemented")
        semaphore = asyncio.Semaphore(settings.max_concurrency_per_iteration)
        results = {}

        async def one(task):
            async with semaphore:
                if self._cancelled.is_set():
                    return
                if task not in BY_NAME:
                    raise ValueError(f"Unknown pinned benchmark task: {task}")
                if on_start:
                    await on_start(task)
                result = await self._run_task(ctx, agent_source, task)
                if self._cancelled.is_set():
                    return
                results[task] = result
                await on_result(result)

        workers = [asyncio.create_task(one(task)) for task in tasks]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            await self.cancel()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return results

    async def _run_task(self, ctx, source, task):
        from ...inference import gateway_session
        start = time.monotonic()
        attempt_id = uuid4().hex
        directory = (ctx.work_dir or ROOT / settings.runs_dir / str(ctx.job_id)) / f"{ctx.iteration_index}-{ctx.evaluation_phase}-r{ctx.repetition}-a{ctx.attempt}-{task}-{attempt_id[:8]}"
        directory = directory.resolve()
        directory.mkdir(parents=True, mode=0o700)
        source_path = directory / "candidate.py"
        source_path.write_text(source)
        revision = ctx.extra.get("benchmark_revision") or settings.benchmark_revision
        harness = ctx.extra.get("sandbox_harness") or {"mode": "cold-v1"}
        prepared = None
        if harness.get("mode") == "prepared-tools-v1":
            from ...sandbox.prepared_images import validate_local_entry
            if harness.get("benchmark_revision") != revision:
                raise ValueError("Prepared harness benchmark revision changed")
            prepared = (harness.get("entries") or {}).get(task)
            await validate_local_entry(task, prepared, harness.get("provider_id", ""))
        elif harness.get("mode") != "cold-v1":
            raise ValueError("Unsupported frozen sandbox harness mode")
        agent_timeout = float(ctx.extra.get("agent_timeout_sec", BY_NAME[task]["agent_timeout_sec"]))
        verifier_timeout = float(ctx.extra.get("verifier_timeout_sec", BY_NAME[task]["verifier_timeout_sec"]))
        # The job deadline also bounds setup and verification, not just agent calls.
        lifetime = agent_timeout + verifier_timeout + 500
        if ctx.deadline_at:
            lifetime = min(lifetime, (ctx.deadline_at - datetime.now(timezone.utc)).total_seconds())
        if lifetime <= 0:
            raise repo.BudgetExceeded("Job wall-clock deadline exceeded")
        expires_at = time.time() + lifetime
        self._attempts.add(attempt_id)
        result = None
        session_factory = self._session_factory or gateway_session
        try:
            async with self._slot(ctx, attempt_id, lifetime + 60):
                async with session_factory(ctx, task) as session:
                    if self._cancelled.is_set():
                        raise asyncio.CancelledError()
                    adapter = directory / "adapter.json"
                    write_private_json(adapter, {
                        "candidate_host_path": str(source_path), "base_url": "http://aos-relay:8080/v1",
                        "benchmark_task": task,
                        "prepared_image": prepared,
                        "token": session.token, "model": ctx.agent_model,
                        "timeout_sec": min(agent_timeout, lifetime),
                        "max_steps": int(ctx.extra.get("max_agent_steps", 80)),
                        "max_tokens": int(ctx.extra.get("max_tokens_per_call", settings.agent_max_output_tokens)),
                        "model_timeout_sec": settings.inference_timeout_sec + 10,
                    })
                    overlay = directory / "sandbox-compose.json"
                    sandbox_overlay(overlay, gateway_url=session.base_url, attempt_id=attempt_id,
                                    job_id=str(ctx.job_id), epoch=ctx.lease_epoch, expires_at=expires_at)
                    config_path = directory / "harbor-config.json"
                    write_private_json(config_path, harbor_config(task, directory, adapter, overlay,
                        agent_timeout, verifier_timeout, benchmark_revision=revision, prepared_image=prepared))
                    try:
                        with (directory / "runner.log").open("wb") as log:
                            if self._cancelled.is_set():
                                raise asyncio.CancelledError()

                            async def spawn():
                                child = await asyncio.create_subprocess_exec(settings.harbor_executable, "run", "--config", str(config_path),
                                    cwd=ROOT, env=runner_environment(), stdout=log, stderr=asyncio.subprocess.STDOUT,
                                    start_new_session=True)
                                self._attempt_processes[attempt_id] = child
                                self._processes.add(child)
                                return child

                            spawn_task = asyncio.create_task(spawn())
                            self._slot_records[attempt_id]["spawn"] = spawn_task
                            process = await asyncio.shield(spawn_task)
                            if self._cancelled.is_set():
                                raise asyncio.CancelledError()
                            try:
                                while process.returncode is None:
                                    raise_gateway_stop(session)
                                    remaining = expires_at - time.time()
                                    if remaining <= 0:
                                        raise asyncio.TimeoutError()
                                    try:
                                        await asyncio.wait_for(process.wait(), min(0.25, remaining))
                                    except asyncio.TimeoutError:
                                        continue
                                raise_gateway_stop(session)
                            except asyncio.TimeoutError:
                                await self._terminate(process)
                                result = TaskResult(task, "timeout", None, "timeout", time.monotonic() - start,
                                                    exit_reason="trial_wall_clock_deadline", sandbox_id=attempt_id)
                            except BaseException:
                                await self._terminate(process)
                                raise
                            finally:
                                self._processes.discard(process)
                        if result is None:
                            paths = list((directory / "harbor" / "trial").glob("*/result.json"))
                            if len(paths) != 1:
                                result = TaskResult(task, "error", None, "infra_error", time.monotonic() - start,
                                    exit_reason="harbor_missing_result", sandbox_id=attempt_id,
                                    metadata={"runner_exit_code": process.returncode, "runner_log": str(directory / "runner.log")})
                            else:
                                result = parse_trial_result(task, paths[0], time.monotonic() - start,
                                                            benchmark_revision=revision)
                        result.input_tokens = int(session.usage.get("input_tokens", 0))
                        result.output_tokens = int(session.usage.get("output_tokens", 0))
                        result.model_cost_usd = session.usage.get("cost_usd")
                        # Local Docker resource cost has no configured monetary rate.
                        result.cost_usd = None
                        result.cost_known = bool(session.usage.get("cost_known", False))
                        result.instruction = _read_text(directory / "instruction.txt")
                        provenance_text = _read_text(directory / "sandbox-provenance.json")
                        provenance = json.loads(provenance_text) if provenance_text else None
                        if provenance is not None:
                            provenance["benchmark_revision"] = revision
                            write_private_json(directory / "sandbox-provenance.json", provenance)
                        result.metadata.update(model_cost_known=bool(session.usage.get("cost_known", False)),
                            benchmark_revision=revision,
                            sandbox_harness_mode=harness["mode"],
                            agent_sha256=hashlib.sha256(source.encode()).hexdigest(),
                            attempt_id=attempt_id, artifact_dir=str(directory), task_attempt_wall_seconds=result.duration_s,
                            duration_basis="end_to_end_attempt_including_admission_wait_setup_verification_and_cleanup",
                            sandbox_provenance=provenance)
                        preflight_text = _read_text(directory / "verifier-preflight.json", 32000)
                        result.metadata["verifier_preflight"] = json.loads(preflight_text) if preflight_text else None
                    finally:
                        adapter.unlink(missing_ok=True)
        finally:
            if attempt_id in self._attempts:
                await self._finish_attempt(attempt_id)
        if result is not None:
            result.duration_s = time.monotonic() - start
            result.metadata["task_attempt_wall_seconds"] = result.duration_s
        return result

    async def _terminate(self, process):
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        except ProcessLookupError:
            pass

    async def cancel(self):
        self._cancelled.set()
        async def cleanup():
            async with asyncio.timeout(FINALIZE_TIMEOUT_SEC):
                await asyncio.gather(*(self._terminate(process) for process in list(self._processes)), return_exceptions=True)
                errors = await asyncio.gather(*(self._finish_attempt(key) for key in list(self._attempts)), return_exceptions=True)
                for error in errors:
                    if isinstance(error, Exception):
                        log.warning("Sandbox finalization failed; reservation retained: %s", error)
        await await_finalizer(asyncio.create_task(cleanup()))
