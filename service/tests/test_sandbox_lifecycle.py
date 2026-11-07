"""Exercise admission and finalization races without external model calls."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from service import repo
from service.worker.executors import harbor_runner as runner


class Pool:
    @asynccontextmanager
    async def acquire(self):
        yield object()


def context():
    lease = repo.Lease(uuid4(), "worker", 1)
    return SimpleNamespace(lease=lease, job_id=lease.job_id, lease_owner=lease.owner, lease_epoch=lease.epoch)


def setup(monkeypatch):
    events = []
    slots = set()

    async def acquire(_conn, _lease, key, _limit, _ttl, **kwargs):
        assert kwargs["provider_id"] == "docker:fixture"
        slots.add(key)
        events.append("admitted")
        return True

    async def remove(_key):
        events.append("provider_empty")

    async def release(_conn, _lease, key):
        assert events[-1] == "provider_empty"
        slots.discard(key)
        events.append("released")

    monkeypatch.setattr(runner, "docker_provider_id", AsyncMock(return_value="docker:fixture"))
    monkeypatch.setattr(runner, "remove_attempt", remove)
    monkeypatch.setattr(repo, "acquire_sandbox_slot", acquire)
    monkeypatch.setattr(repo, "release_sandbox_slot", release)
    return events, slots


def test_repeated_cancel_during_database_release_does_not_leak_slot(monkeypatch):
    async def check():
        events, slots = setup(monkeypatch)
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = repo.release_sandbox_slot

        async def delayed_release(*args):
            entered.set()
            await proceed.wait()
            await original(*args)

        monkeypatch.setattr(repo, "release_sandbox_slot", delayed_release)
        executor = runner.HarborExecutor(pool=Pool())
        executor._attempts.add("attempt")

        async def use():
            async with executor._slot(context(), "attempt", 60):
                pass

        task = asyncio.create_task(use())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and "attempt" in slots
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not slots
        assert events == ["admitted", "provider_empty", "released"]
    asyncio.run(check())


def test_cancel_during_admission_never_yields_and_releases_committed_slot(monkeypatch):
    async def check():
        events, slots = setup(monkeypatch)
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = repo.acquire_sandbox_slot

        async def delayed_acquire(*args, **kwargs):
            entered.set()
            await proceed.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(repo, "acquire_sandbox_slot", delayed_acquire)
        executor = runner.HarborExecutor(pool=Pool())
        executor._attempts.add("attempt")

        async def use():
            async with executor._slot(context(), "attempt", 60):
                pytest.fail("cancelled admission must not dispatch a runner")

        task = asyncio.create_task(use())
        await entered.wait()
        cancelling = asyncio.create_task(executor.cancel())
        await asyncio.sleep(0)
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await cancelling
        assert not slots and events[-1] == "released"
    asyncio.run(check())


def test_cleanup_failure_retains_admission(monkeypatch):
    async def check():
        _, slots = setup(monkeypatch)
        monkeypatch.setattr(runner, "remove_attempt", AsyncMock(side_effect=RuntimeError("Docker unavailable")))
        executor = runner.HarborExecutor(pool=Pool())
        executor._attempts.add("attempt")
        with pytest.raises(RuntimeError, match="Docker unavailable"):
            async with executor._slot(context(), "attempt", 60):
                pass
        assert slots == {"attempt"}
    asyncio.run(check())


def test_cleanup_waits_for_inflight_spawn_then_stops_runner_before_release(monkeypatch):
    async def check():
        events, slots = setup(monkeypatch)
        ready, proceed = asyncio.Event(), asyncio.Event()
        executor = runner.HarborExecutor(pool=Pool())
        executor._attempts.add("attempt")
        slots.add("attempt")
        class Process:
            returncode = None
        process = Process()

        async def spawn():
            ready.set()
            await proceed.wait()
            executor._attempt_processes["attempt"] = process
            events.append("spawned")
            return process

        async def terminate(child):
            assert child is process
            child.returncode = -15
            events.append("runner_stopped")

        monkeypatch.setattr(executor, "_terminate", terminate)
        executor._slot_records["attempt"] = {"pool": Pool(), "lease": context().lease,
            "provider_id": "docker:fixture", "spawn": asyncio.create_task(spawn())}
        await ready.wait()
        finalizing = asyncio.create_task(executor._finish_attempt("attempt"))
        await asyncio.sleep(0)
        assert events == [] and slots == {"attempt"}
        proceed.set()
        await finalizing
        assert events == ["spawned", "runner_stopped", "provider_empty", "released"]
        assert not slots
    asyncio.run(check())


def test_daemon_change_cannot_release_another_providers_slot(monkeypatch):
    async def check():
        _, slots = setup(monkeypatch)
        monkeypatch.setattr(runner, "docker_provider_id", AsyncMock(side_effect=["docker:fixture", "docker:other"]))
        remove = AsyncMock()
        monkeypatch.setattr(runner, "remove_attempt", remove)
        executor = runner.HarborExecutor(pool=Pool())
        with pytest.raises(RuntimeError, match="daemon changed"):
            async with executor._slot(context(), "attempt", 60):
                pass
        assert slots == {"attempt"}
        remove.assert_not_awaited()
    asyncio.run(check())


@pytest.mark.parametrize("responses", [
    [(1, "unavailable")],
    [(0, "container"), (1, "remove failed")],
    [(0, "container"), (0, ""), (0, "still-present")],
    [(0, ""), (0, ""), (1, "network unavailable")],
    [(0, ""), (0, ""), (0, "network"), (1, "remove failed")],
    [(0, ""), (0, ""), (0, ""), (0, "still-present")],
])
def test_provider_cleanup_fails_closed_on_errors_or_remaining_resources(monkeypatch, responses):
    monkeypatch.setattr(runner, "docker_command", AsyncMock(side_effect=responses))
    with pytest.raises(RuntimeError):
        asyncio.run(runner.remove_attempt("attempt"))


def test_finalizer_timeout_retains_slot_and_is_bounded(monkeypatch):
    async def check():
        _, slots = setup(monkeypatch)
        monkeypatch.setattr(runner, "FINALIZE_TIMEOUT_SEC", 0.05)

        async def stuck(_key):
            await asyncio.Event().wait()

        monkeypatch.setattr(runner, "remove_attempt", stuck)
        executor = runner.HarborExecutor(pool=Pool())
        with pytest.raises(TimeoutError):
            async with executor._slot(context(), "attempt", 60):
                pass
        assert slots == {"attempt"}
    asyncio.run(check())


def test_task_duration_includes_final_provider_cleanup(monkeypatch, tmp_path):
    async def check():
        @asynccontextmanager
        async def session(*_):
            yield SimpleNamespace(token="fixture", base_url="http://fixture/v1", usage={}, stop_reason=None)

        executor = runner.HarborExecutor(session_factory=session)

        @asynccontextmanager
        async def slot(_ctx, key, _ttl):
            executor._slot_records[key] = {}
            yield

        async def finish(key):
            await asyncio.sleep(0.04)
            executor._attempts.discard(key)

        class Process:
            returncode = 0

        monkeypatch.setattr(executor, "_slot", slot)
        monkeypatch.setattr(executor, "_finish_attempt", finish)
        monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", AsyncMock(return_value=Process()))
        ctx = runner.RunContext(job_id=uuid4(), iteration_index=0, iteration_id=uuid4(),
                                org_id=uuid4(), agent_model="fixture", seed=1, work_dir=tmp_path)
        result = await executor._run_task(ctx, "pass", "fix-git")
        assert result.duration_s >= 0.04
        assert result.metadata["task_attempt_wall_seconds"] == result.duration_s
        assert result.metadata["duration_basis"].endswith("verification_and_cleanup")
    asyncio.run(check())


def test_ungraded_outer_deadline_cannot_complete_a_comparison(monkeypatch, tmp_path):
    # Exercise the actual executor timeout branch, retaining task cost/cleanup,
    # with a fake runner/session/clock; no Docker, PostgreSQL or model calls.
    from dataclasses import asdict
    import time
    from service.worker.job_runner import evaluation_metrics, promotion_decision

    async def check():
        events = []
        usage = {"input_tokens": 10, "output_tokens": 5, "cost_usd": .02, "cost_known": True}
        @asynccontextmanager
        async def session(*_):
            yield SimpleNamespace(token="fixture", base_url="http://fixture/v1", usage=usage, stop_reason=None)

        executor = runner.HarborExecutor(session_factory=session)
        @asynccontextmanager
        async def slot(_ctx, key, _ttl):
            executor._slot_records[key] = {}
            yield

        class Process:
            returncode = None
        process = Process()
        async def terminate(child):
            assert child is process
            child.returncode = -15
            events.append("runner_stopped")
        async def finish(key):
            assert process.returncode is not None
            executor._attempts.discard(key)
            events.append("cleaned")

        # Default attempt lifetime = 900 + 900 + 500 seconds. Advance wall time
        # past that limit immediately after the fake runner has been registered.
        wall_times = iter([1000.0, 3301.0])
        monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: next(wall_times), monotonic=time.monotonic))
        monkeypatch.setattr(executor, "_slot", slot)
        monkeypatch.setattr(executor, "_finish_attempt", finish)
        monkeypatch.setattr(executor, "_terminate", terminate)
        monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        ctx = runner.RunContext(job_id=uuid4(), iteration_index=0, iteration_id=uuid4(),
            org_id=uuid4(), agent_model="fixture", seed=1, work_dir=tmp_path)
        result = await executor._run_task(ctx, "pass", "fix-git")
        assert result.status == "timeout" and result.reward is None
        assert result.exit_reason == "trial_wall_clock_deadline"
        assert result.model_cost_usd == .02 and result.input_tokens == 10
        assert events == ["runner_stopped", "cleaned"]
        measured = evaluation_metrics([asdict(result)], 1)
        assert measured["complete"] is False
        assert measured["mean_reward"] is None
        valid = evaluation_metrics([{**asdict(result), "status": "failed", "reward": 0.0}], 1)
        assert valid["complete"] is True
        accepted, reason = promotion_decision(measured, valid, {})
        assert not accepted and reason.startswith("incomplete_comparison")
    asyncio.run(check())
