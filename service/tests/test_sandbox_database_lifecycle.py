"""Opt-in real Docker + disposable PostgreSQL slot lifecycle proof.

Set AOS_RUN_DOCKER_TESTS=1 and AOS_TEST_DATABASE_URL. No provider calls occur.
"""
import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from service import repo
from service.tests.test_persistence import pool, principal, submit, claim, lease
from service.worker.executors import harbor_runner as runner


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("AOS_RUN_DOCKER_TESTS") != "1", reason="opt-in Docker lifecycle proof")
async def test_real_cancel_during_db_release_and_scoped_orphan_recovery(pool, monkeypatch):
    actor, _ = await principal(pool)
    await submit(pool, actor)
    job = await claim(pool, "sandbox-lifecycle-fixture")
    ownership = lease(job)
    ctx = SimpleNamespace(lease=ownership, job_id=job["id"], lease_owner=ownership.owner, lease_epoch=ownership.epoch)
    provider = await runner.docker_provider_id()
    executor = runner.HarborExecutor(pool=pool)
    key = "db-cancel-" + uuid4().hex
    executor._attempts.add(key)
    resources_ready, release_entered, allow_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_release = repo.release_sandbox_slot
    process = None
    created = {key}
    facts = {"real_provider_calls": 0, "provider_id": provider}

    async def delayed_release(connection, current_lease, slot_key):
        if slot_key == key:
            release_entered.set()
            await allow_release.wait()
        await original_release(connection, current_lease, slot_key)

    monkeypatch.setattr(repo, "release_sandbox_slot", delayed_release)

    async def container(attempt, expires):
        code, output = await runner.docker_command("run", "-d", "--network", "none",
            "--label", "aos.managed=true", "--label", "aos.attempt=" + attempt,
            "--label", "aos.expires_at=" + str(expires), "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--read-only", "--memory", "64m",
            "--cpus", "0.25", "--pids-limit", "16", runner.RELAY_IMAGE, "sleep", "120")
        assert code == 0, output
        created.add(attempt)

    async def run_attempt():
        nonlocal process
        async with executor._slot(ctx, key, 120):
            await container(key, time.time() + 120)
            process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(120)",
                                                          start_new_session=True)
            executor._attempt_processes[key] = process
            executor._processes.add(process)
            resources_ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run_attempt())
    try:
        await asyncio.wait_for(resources_ready.wait(), 30)
        task.cancel()
        await asyncio.wait_for(release_entered.wait(), 30)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT count(*) FROM sandbox_slots WHERE slot_key=$1", key) == 1
        assert process.returncode is not None
        code, containers = await runner.docker_command("ps", "-aq", "--filter", "label=aos.attempt=" + key)
        assert code == 0 and not containers.strip()
        allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT count(*) FROM sandbox_slots WHERE slot_key=$1", key) == 0
        facts["repeated_cancel_during_real_db_release_completed"] = True
        facts["runner_and_container_stopped_before_slot_release"] = True

        local, foreign, legacy, current = [name + "-" + uuid4().hex for name in ("local", "foreign", "legacy", "current")]
        async with pool.acquire() as connection:
            for slot, scope in [(local, provider), (foreign, "docker:other-daemon"), (legacy, None), (current, provider)]:
                assert await repo.acquire_sandbox_slot(connection, ownership, slot, 8, 120, provider_id=scope)
            await connection.execute("UPDATE sandbox_slots SET expires_at=clock_timestamp()-interval '1 second' WHERE slot_key=ANY($1::text[])", [local, foreign, legacy])
            await repo.finish_job(connection, ownership, "CANCELLED", "lifecycle_fixture")
        await container(local, time.time() - 5)
        await container(current, time.time() + 120)
        await runner.reap_expired_sandboxes(pool=pool)
        async with pool.acquire() as connection:
            remaining = set(await connection.fetch("SELECT slot_key FROM sandbox_slots"))
            remaining = {row["slot_key"] for row in remaining}
        assert remaining == {foreign, legacy, current}
        code, alive = await runner.docker_command("ps", "-q", "--filter", "label=aos.attempt=" + current)
        assert code == 0 and alive.strip()
        facts["expired_same_daemon_slot_reconciled"] = True
        facts["foreign_legacy_and_unexpired_slots_retained"] = True
        facts["unexpired_container_preserved"] = True
        destination = runner.ROOT / "runs" / ("db-slot-lifecycle-proof-" + uuid4().hex + ".json")
        destination.write_text(json.dumps(facts, indent=2))
        print("Lifecycle proof:", destination)
    finally:
        allow_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await executor.cancel()
        for attempt in created:
            await runner.remove_attempt(attempt)
