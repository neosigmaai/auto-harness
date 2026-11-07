"""Full PostgreSQL/API/worker/test_client smoke, with no model or sandbox calls.

AOS_TEST_DATABASE_URL must point to an explicitly chosen disposable-test server
whose role can CREATE DATABASE. Every run creates/drops its own database.
Run: python -m service.tests.smoke_simulated
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import asyncpg
import httpx

from service.db import migrate
from service.tasks_catalog import DEFAULT_TASKS, CALIBRATED_SPLIT

ROOT = Path(__file__).resolve().parents[2]


async def main():
    dsn = os.environ["AOS_TEST_DATABASE_URL"]
    name = "aos_workflow_e2e_" + uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(dsn)
    isolated = urlunsplit((parts.scheme, parts.netloc, "/" + name, parts.query, parts.fragment))
    processes, logs = [], []
    bootstrap = "smoke-" + uuid4().hex
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "DATABASE_URL": isolated, "EXECUTOR": "simulated", "OPTIMIZER": "simulated",
           "BOOTSTRAP_ADMIN_TOKEN": bootstrap, "SIMULATED_TASK_DELAY_SEC": ".2",
           "HEARTBEAT_SEC": "1", "LEASE_TTL_SEC": "5", "CLAIM_POLL_SEC": ".05",
           "WORKER_CONCURRENCY": "1"}
    # Do not pass any personal or company provider credentials to this test.
    for key in ("OPENAI_API_KEY", "OPEN_API_KEY", "NEOSIGMA_OPEN_API_KEY", "E2B_API_KEY"):
        env.pop(key, None)
    async def wait_job(client, headers, job_id, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
            response.raise_for_status()
            job = response.json()
            if predicate(job):
                return job
            await asyncio.sleep(.05)
        raise AssertionError(f"Job did not reach requested state: {job['status']}")
    def start(command, label):
        log = tempfile.TemporaryFile(mode="w+")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        logs.append((label, log))
        return process
    try:
        await migrate(isolated)
        start([sys.executable, "-m", "uvicorn", "service.api.app:app", "--host", "127.0.0.1", "--port", str(port)], "api")
        worker = start([sys.executable, "-m", "service.worker.main"], "worker")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            for _ in range(100):
                try:
                    ready = await client.get("/v1/readyz")
                    if ready.status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(.05)
            else:
                raise AssertionError("API failed to start")
            response = await client.post("/v1/orgs", headers={"X-Bootstrap-Token": bootstrap},
                json={"name": "simulated smoke", "admin_email": "smoke@example.test"})
            response.raise_for_status()
            key = response.json()["api_key"]
            headers = {"Authorization": "Bearer " + key}
            command = [sys.executable, "test_client.py", "--base-url", f"http://127.0.0.1:{port}",
                       "--max-iterations", "2", "--patience", "2", "--seed", "7", "--quiet", "--json", "--timeout", "40"]
            cli = await asyncio.to_thread(subprocess.run, command, cwd=ROOT,
                env={**env, "AOS_API_KEY": key}, capture_output=True, text=True, timeout=45)
            assert cli.returncode == 0, cli.stderr + cli.stdout[-2000:]
            payload = json.loads(cli.stdout[cli.stdout.index('\n{\n  "job"'):])
            job, iterations = payload["job"], payload["iterations"]
            assert job["config"]["profile"] == "calibrated12"
            assert job["config"]["evaluation_split"] == CALIBRATED_SPLIT
            assert job["status"] == "SUCCEEDED" and job["final_evaluation"]["status"] == "complete"
            assert len(iterations) == 3 and iterations[0]["decision"] == "baseline"
            assert all(i["decision"] in ("accepted", "rejected") for i in iterations[1:])
            assert iterations[0]["metrics"]["evidence_kind"] == "simulated"
            baseline_attempts = await client.get(f"/v1/jobs/{job['job_id']}/iterations/0/attempts", headers=headers)
            baseline_attempts.raise_for_status()
            assert any(a["evaluation_phase"] == "holdout_final" for a in baseline_attempts.json())
            assert len((await client.get(f"/v1/jobs/{job['job_id']}/agent", headers=headers)).json()["source"]) > 0
            # A single-task benchmark remains available without a split.
            baseline_response = await client.post("/v1/jobs", headers=headers,
                json={"tasks": DEFAULT_TASKS[:1], "max_iterations": 0})
            baseline_response.raise_for_status()
            baseline = await wait_job(client, headers, baseline_response.json()["job_id"],
                                      lambda j: j["status"] in {"SUCCEEDED", "FAILED"})
            assert baseline["status"] == "SUCCEEDED"
            assert baseline["final_evaluation"]["status"] == "not_applicable"
            # Cancellation must stop currently executing work, not a future iteration.
            cancel_response = await client.post("/v1/jobs", headers=headers,
                json={"profile": "calibrated12", "max_iterations": 20})
            cancel_response.raise_for_status()
            cancel_id = cancel_response.json()["job_id"]
            cancelling = await wait_job(client, headers, cancel_id,
                lambda j: j["status"] == "RUNNING" and (j.get("current_iteration") or 0) >= 1)
            interrupted_index = cancelling["current_iteration"]
            start_cancel = time.monotonic()
            cancelled_response = await client.post(f"/v1/jobs/{cancel_id}/cancel", headers=headers)
            cancelled_response.raise_for_status()
            cancelled = await wait_job(client, headers, cancel_id, lambda j: j["status"] == "CANCELLED", timeout=5)
            cancel_elapsed = time.monotonic() - start_cancel
            assert cancelled["final_evaluation"]["status"] == "incomplete"
            # A related job can re-evaluate an interrupted source without
            # inheriting any verifier result or calling the proposer again.
            continuation_response = await client.post("/v1/jobs", headers=headers,
                json={"profile": "calibrated12", "max_iterations": 1, "prior_job_id": cancel_id,
                      "resume_candidate_index": interrupted_index})
            continuation_response.raise_for_status()
            continuation_id = continuation_response.json()["job_id"]
            continuation = await wait_job(client, headers, continuation_id,
                lambda j: j["status"] in {"SUCCEEDED", "FAILED"}, timeout=20)
            assert continuation["status"] == "SUCCEEDED"
            resumed_detail = await client.get(f"/v1/jobs/{continuation_id}/iterations/1", headers=headers)
            resumed_detail.raise_for_status()
            assert resumed_detail.json()["proposal"]["resumed_from"] == {"job_id": cancel_id, "iteration": interrupted_index}
            resumed_attempts = await client.get(f"/v1/jobs/{continuation_id}/iterations/1/attempts", headers=headers)
            resumed_attempts.raise_for_status()
            assert len([a for a in resumed_attempts.json() if a["evaluation_phase"] == "main"]) == 8
            continued_events = (await client.get(f"/v1/jobs/{continuation_id}/events", headers=headers)).json()
            assert not any(e["kind"] == "optimizer.proposal_started" for e in continued_events)
            prior_baseline = await client.get(f"/v1/jobs/{cancel_id}/iterations/0", headers=headers)
            fresh_baseline = await client.get(f"/v1/jobs/{continuation_id}/iterations/0", headers=headers)
            prior_baseline.raise_for_status()
            fresh_baseline.raise_for_status()
            assert fresh_baseline.json()["agent_source"] == prior_baseline.json()["agent_source"]
            # Graceful worker shutdown leaves an owned job recoverable. The frozen
            # deadline and completed attempt identities survive the next epoch.
            resume_response = await client.post("/v1/jobs", headers=headers,
                json={"tasks": DEFAULT_TASKS, "max_iterations": 0})
            resume_response.raise_for_status()
            resume_id = resume_response.json()["job_id"]
            started = await wait_job(client, headers, resume_id, lambda j: j["status"] == "RUNNING")
            worker.terminate()
            await asyncio.to_thread(worker.wait, timeout=10)
            # Let the existing five-second lease expire naturally.
            start([sys.executable, "-m", "service.worker.main"], "recovery-worker")
            resumed = await wait_job(client, headers, resume_id, lambda j: j["status"] in {"SUCCEEDED", "FAILED"}, timeout=15)
            assert resumed["status"] == "SUCCEEDED"
            assert resumed["deadline_at"] == started["deadline_at"]
            conn = await asyncpg.connect(isolated)
            try:
                epoch = await conn.fetchval("SELECT lease_epoch FROM jobs WHERE id=$1", UUID(resume_id))
                assert epoch >= 2
            finally:
                await conn.close()
            print(json.dumps({"status": "passed", "mode": "simulated_only", "cli_iterations": len(iterations),
                "profile": job["config"]["profile"], "split_sizes": [len(CALIBRATED_SPLIT[k]) for k in ("feedback","selection","holdout")],
                "holdout": job["final_evaluation"]["status"], "cancel_latency_s": round(cancel_elapsed, 3),
                "prior_candidate_resumed_with_fresh_trials": True,
                "recovered_lease_epoch": epoch, "durable_deadline_preserved": True}, indent=2))
    except BaseException:
        for label, log in logs:
            log.seek(0)
            print(f"{label} test log:\n{log.read()[-5000:]}", file=sys.stderr)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                await asyncio.to_thread(process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        for _, log in logs:
            log.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


if __name__ == "__main__":
    asyncio.run(main())
