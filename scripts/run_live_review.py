"""Paid end-to-end review, with an explicit gate and a persistent evidence database.

Usage: AOS_TEST_DATABASE_URL=... python scripts/run_live_review.py --allow-paid
Only NEOSIGMA_OPEN_API_KEY is used. Default spending ceiling is $14.90, below the
session's $15 authorization after the separately recorded credential smoke.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncpg
import httpx

from service.config import settings
from service.db import migrate
from service.models import JobCreate
from service.tasks_catalog import STRESS_TASKS
from service.worker.job_runner import TEMPLATE_PATH
from service.repo import sha256_text

REPORT = ROOT / "docs" / "live_benchmark_review.json"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def implementation_hashes():
    paths = [path for path in (ROOT / "service").rglob("*.py") if "tests" not in path.parts]
    paths.extend((ROOT / "service" / "migrations").glob("*.sql"))
    paths.extend((ROOT / "service" / "sandbox").rglob("*.json"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def write_private(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))
    path.chmod(0o600)


async def run(args):
    if not args.allow_paid:
        raise SystemExit("No requests sent. Pass --allow-paid only after authorizing the $14.90 maximum.")
    credential = settings.neosigma_open_api_key
    if credential is None or not credential.get_secret_value():
        raise SystemExit("NEOSIGMA_OPEN_API_KEY is missing; personal-key fallback is forbidden.")
    if settings.neosigma_base_url != "https://api.openai.com/v1":
        raise SystemExit("Only the approved OpenAI endpoint is enabled for this review.")
    dsn = os.environ.get("AOS_TEST_DATABASE_URL")
    if not dsn:
        raise SystemExit("AOS_TEST_DATABASE_URL is required; the user's default database is never used.")
    prior_control = json.loads(args.prior_control.read_text()) if args.prior_control else None
    cap = 12.0 if prior_control else 14.90
    body = {"tasks": STRESS_TASKS, "max_iterations": 3 if prior_control else 2, "patience": 3 if prior_control else 2, "seed": 1234,
            "agent_model": "gpt-5.4-mini", "optimizer_model": "gpt-5.4-mini",
            "budget": {"max_wall_clock_min": 90, "max_llm_usd": cap},
            "options": {"confirmation_repetitions": 2, "max_infrastructure_attempts": 1}}
    if prior_control:
        body.update(prior_job_id=prior_control["job_id"], resume_candidate_index=2)
    frozen = JobCreate(**body)
    split = frozen.evaluation_split
    assert len(STRESS_TASKS) == 16 and (len(split.feedback), len(split.selection), len(split.holdout)) == (8, 4, 4)
    body["evaluation_split"] = split.model_dump()
    name = "aos_live_review_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + uuid4().hex[:8]
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, mode=0o700)
    previous_spend, previous_reserved = 0., 0.
    if prior_control:
        isolated = prior_control["database_url"]
        admin = await asyncpg.connect(isolated)
        try:
            if await admin.fetchval("SELECT count(*) FROM jobs WHERE status IN ('RUNNING','QUEUED')"):
                raise RuntimeError("Prior evidence database still has active work; do not overbook its budget")
            ledger = await admin.fetchrow("""SELECT coalesce(sum(actual_usd),0) AS spent,
                coalesce(sum(reserved_usd) FILTER(WHERE settled_at IS NULL),0) AS reserved FROM spend_reservations""")
            previous_spend, previous_reserved = float(ledger["spent"]), float(ledger["reserved"])
            if previous_spend + previous_reserved + .00093575 + cap > 15:
                raise RuntimeError("Requested continuation cap would exceed the $15 cumulative authorization")
        finally:
            await admin.close()
    else:
        admin = await asyncpg.connect(dsn)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        parts = urlsplit(dsn)
        isolated = urlunsplit((parts.scheme, parts.netloc, "/" + name, parts.query, parts.fragment))
    database_name = urlsplit(isolated).path.lstrip("/")
    await migrate(isolated)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    bootstrap = "review-" + uuid4().hex
    env = {**os.environ, "DATABASE_URL": isolated, "EXECUTOR": "harbor", "OPTIMIZER": "llm",
           "SANDBOX_PROVIDER": "docker", "HARBOR_EXECUTABLE": str(Path.home() / ".local/bin/harbor"),
           "BOOTSTRAP_ADMIN_TOKEN": bootstrap, "MAX_CONCURRENCY_PER_ITERATION": "2", "GLOBAL_TRIAL_SLOTS": "2",
           "WORKER_CONCURRENCY": "1", "CLAIM_POLL_SEC": ".5", "HEARTBEAT_SEC": "5", "LEASE_TTL_SEC": "60",
           "RUNS_DIR": str(directory / "tasks"), "NEOSIGMA_OPEN_API_KEY": credential.get_secret_value(),
           "AGENT_MODEL": "gpt-5.4-mini", "OPTIMIZER_MODEL": "gpt-5.4-mini"}
    for key in ("OPENAI_API_KEY", "OPEN_API_KEY", "E2B_API_KEY"):
        env.pop(key, None)
    processes, handles = [], []
    def start(command, label):
        log = (directory / (label + ".log")).open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        handles.append(log)
        return process
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "evidence_database": database_name, "run_directory": str(directory.relative_to(ROOT)),
              "mode": "real_harbor_docker_and_company_model", "credential_source": "NEOSIGMA_OPEN_API_KEY",
              "baseline_sha256": sha256_text(TEMPLATE_PATH.read_text()), "request": body,
              "implementation_sha256": implementation_hashes(), "benchmark_revision": settings.benchmark_revision,
              "fixed_reasoning_policy": {"tool_bearing_agent_requests": "none", "optimizer_requests": "low"},
              "previous_diagnostic_spend_usd": 0.00093575,
              "previous_experiment_spent_usd": previous_spend, "previous_experiment_reserved_usd": previous_reserved,
              "prior_experiment_report": "docs/live_benchmark_review_diversity.json" if prior_control else None,
              "hardware": {"docker_architecture": "aarch64", "cpus": 8, "memory_bytes": 8218316800,
                  "global_trial_slots": 2, "task_cpu_limit": 1, "task_memory_mib": 2048,
                  "benchmark_images": "amd64; local timings include CPU emulation"},
              "prior_aborted_run": "docs/live_benchmark_review_http400.json" if (ROOT / "docs/live_benchmark_review_http400.json").exists() else None,
              "limitations": ["Small fixed subset, not a full Terminal-Bench result.",
                  "Model USD estimates use uncached token rates; local Docker dollars are unpriced.",
                  "Selection repetitions are screening evidence, not statistical significance.",
                  "Only fresh matched selection confirmation can support observed efficiency claims; initial timings may include cold setup.",
                  "Holdout runs baseline first; order/provider/setup effects confound latency comparisons even with images pre-pulled.",
                  "Local amd64-on-aarch64 emulation limits transfer of timing measurements to native amd64 production workers."]}
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    api = start([sys.executable, "-m", "uvicorn", "service.api.app:app", "--host", "127.0.0.1", "--port", str(port)], "api")
    worker = start([sys.executable, "-m", "service.worker.main"], "worker")
    job_id = None
    headers = None
    async def snapshot(client):
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        job = response.json()
        iterations_response = await client.get(f"/v1/jobs/{job_id}/iterations", headers=headers)
        iterations_response.raise_for_status()
        iterations = iterations_response.json()
        attempts = []
        for iteration in iterations:
            offset = 0
            while True:
                response = await client.get(f"/v1/jobs/{job_id}/iterations/{iteration['index']}/attempts",
                    headers=headers, params={"include_trace": "false", "limit": 100, "offset": offset})
                response.raise_for_status()
                page = response.json()
                # Redacted, bounded evidence: no raw traces, task instructions,
                # candidate source, provider bodies, or ephemeral capabilities.
                fields = ("task_name", "evaluation_phase", "repetition", "attempt", "lease_epoch", "status", "reward",
                          "failure_class", "duration_s", "n_steps", "exit_reason", "input_tokens", "output_tokens",
                          "model_cost_usd", "sandbox_cost_usd", "cost_usd", "cost_known", "created_at")
                for item in page:
                    attempts.append({"iteration_index": iteration["index"], **{field: item.get(field) for field in fields}})
                if len(page) < 100:
                    break
                offset += len(page)
        report.update(status=job["status"], updated_at=datetime.now(timezone.utc).isoformat(),
            implementation_changed_during_run=report["implementation_sha256"] != implementation_hashes(),
            cumulative_spent_usd=previous_spend + job["spent_usd"] + .00093575,
            cumulative_reserved_usd=previous_reserved + job["reserved_usd"],
            job={key: job.get(key) for key in ("job_id", "status", "current_iteration", "baseline_score", "best_score",
                "best_iteration_index", "stop_reason", "error", "cancel_requested", "progress", "started_at", "finished_at",
                "deadline_at", "final_evaluation", "spent_usd", "reserved_usd")},
            iterations=iterations, attempts=attempts)
        REPORT.write_text(json.dumps(report, indent=2, default=str))
        return job, attempts
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=20) as client:
            for _ in range(100):
                try:
                    if (await client.get("/v1/readyz")).status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(.1)
            else:
                raise RuntimeError("API failed readiness before paid execution")
            if prior_control:
                key = prior_control["api_key"]
            else:
                response = await client.post("/v1/orgs", headers={"X-Bootstrap-Token": bootstrap},
                    json={"name": "review evidence", "admin_email": "review@example.test"})
                response.raise_for_status()
                key = response.json()["api_key"]
            headers = {"Authorization": "Bearer " + key}
            response = await client.post("/v1/jobs", headers={**headers, "Idempotency-Key": name}, json=body)
            response.raise_for_status()
            job_id = response.json()["job_id"]
            write_private(directory / "control.json", {"database_url": isolated, "api_url": f"http://127.0.0.1:{port}",
                "api_key": key, "job_id": job_id, "api_pid": api.pid, "worker_pid": worker.pid})
            print(json.dumps({"event": "submitted", "job_id": job_id, "report": str(REPORT),
                              "control_file": str(directory / "control.json"), "cap_usd": cap}), flush=True)
            last = None
            deadline = time.monotonic() + 95 * 60
            seen_errors = set()
            while time.monotonic() < deadline:
                job, attempts = await snapshot(client)
                state = {"status": job["status"], "progress": job.get("progress"), "spent_usd": job["spent_usd"],
                         "reserved_usd": job["reserved_usd"], "attempts_finished": len(attempts)}
                if state != last:
                    print(json.dumps({"event": "progress", **state}), flush=True)
                    last = state
                for attempt in attempts:
                    identity = (attempt["iteration_index"], attempt["task_name"], attempt["evaluation_phase"], attempt["repetition"], attempt["attempt"])
                    if attempt["status"] == "error" and identity not in seen_errors:
                        print(json.dumps({"event": "trial_error", **attempt}), flush=True)
                        seen_errors.add(identity)
                if job["status"] in TERMINAL:
                    print(json.dumps({"event": "terminal", "status": job["status"], "reason": job["stop_reason"],
                        "spent_usd": job["spent_usd"], "reserved_usd": job["reserved_usd"], "report": str(REPORT)}), flush=True)
                    return
                if worker.poll() is not None:
                    raise RuntimeError("Worker process exited; preserve evidence and investigate before any retry")
                await asyncio.sleep(10)
            await client.post(f"/v1/jobs/{job_id}/cancel", headers=headers)
            raise TimeoutError("Review monitoring deadline reached; cancellation requested")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                await asyncio.to_thread(process.wait, timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        for handle in handles:
            handle.close()
        print(json.dumps({"event": "processes_stopped", "evidence_database_preserved": database_name}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--prior-control", type=Path, help="Private control.json of the authorized interrupted experiment; resume its candidate2 under a fresh $12 job cap")
    asyncio.run(run(parser.parse_args()))
