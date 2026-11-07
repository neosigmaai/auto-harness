"""A bounded calibrated12 optimization with explicit prior-exposure handling.

The original baseline is fetched through authenticated API access and pinned by
hash. No candidate or evaluation score is inherited. Optional prior-job linkage
imports only compatible rejected-proposal diagnostics through the existing API.
The core workflow performs its own frozen confirmation and final evaluation.
An explicit development-repeat mode records reused tasks; it never claims a new
blind holdout. Every run remains capped at $6 despite the lifted user session cap.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncpg
import httpx

from scripts.run_final_holdout_review import CONTROLLER_LOCK, redact_attempt
from scripts.run_paired_selection_review import (
    DIAGNOSTIC_USD, SESSION_AUTHORIZATION_USD, TERMINAL, cancel_owned_submissions,
    ledger, remaining_budget, stop_process, verify_source_proof,
)
from service.config import settings
from service.db import _init_conn
from service.models import JobCreate
from service.repo import sha256_text
from service.tasks_catalog import CALIBRATED_SPLIT, CALIBRATION_EXCLUSIONS

BASELINE_SHA256 = "575887cfda6d4e2bb1beee371223f1e27cd3b03080daa2eb40115412fd5808e7"
JOB_FIELDS = ("job_id", "status", "current_iteration", "baseline_score", "best_score",
              "best_iteration_index", "stop_reason", "cancel_requested", "started_at",
              "finished_at", "deadline_at", "spent_usd", "reserved_usd")
ITERATION_FIELDS = ("index", "phase", "status", "decision", "score", "agent_sha256",
                    "started_at", "finished_at")
METRIC_FIELDS = {
    "complete", "planned_trials", "observed_trials", "passed", "failed", "errors", "mean_reward",
    "mean_task_duration_s", "p95_task_duration_s", "mean_model_cost_usd", "total_model_cost_usd",
    "total_sandbox_cost_usd", "input_tokens", "output_tokens", "physical_attempts",
    "confirmation_repetitions", "interrupted", "evaluation_skipped", "observed_tasks",
    "planned_tasks", "coverage_complete", "valid_observations", "mean_reward_observed",
    "mean_model_cost_usd_observed", "p95_task_duration_s_observed",
}


def validate_arguments(args):
    if not args.allow_paid:
        raise SystemExit("No model calls sent. --allow-paid requires the final verified implementation and approval.")
    if not math.isfinite(args.max_total_usd) or not 0 < args.max_total_usd <= 6:
        raise SystemExit("The authorized calibrated-run ceiling must be greater than $0 and at most $6.")
    if not 1 <= args.max_wall_clock_min <= 90:
        raise SystemExit("The calibrated-run wall-clock limit must be between 1 and 90 minutes.")
    if getattr(args, 'prior_job_id', None) is not None and not getattr(args, 'development_repeat', False):
        raise SystemExit("Prior rejection history requires explicit --development-repeat exposure disclosure.")


def build_request(source, cap, wall_minutes=90, *, prior_job_id=None):
    if sha256_text(source) != BASELINE_SHA256:
        raise ValueError("Baseline source differs from the original pinned template")
    if not math.isfinite(float(cap)) or not 0 < Decimal(str(cap)) <= Decimal("6"):
        raise ValueError("Fresh job budget exceeds the calibrated controller allowance")
    body = {"profile": "calibrated12", "agent_source": source,
        "agent_model": "gpt-5.4-mini", "optimizer_model": "gpt-5.4",
        "max_iterations": 3, "patience": 3, "seed": 1234,
        "budget": {"max_llm_usd": float(cap), "max_wall_clock_min": wall_minutes},
        "options": {"confirmation_repetitions": 2, "max_infrastructure_attempts": 2}}
    if prior_job_id is not None:
        # No resume_candidate_index or candidate source: the API authorizes and
        # freezes only compatible rejected-proposal feedback diagnostics.
        body['prior_job_id'] = str(UUID(str(prior_job_id)))
    frozen = JobCreate(**body)
    if frozen.evaluation_split.model_dump() != CALIBRATED_SPLIT:
        raise ValueError("Calibrated profile roles changed before dispatch")
    body["options"] = frozen.options.model_dump()
    body["evaluation_split"] = frozen.evaluation_split.model_dump()
    return body


def summarize_metrics(metrics):
    if not isinstance(metrics, dict):
        return {}
    result = {key: value for key, value in metrics.items()
              if key in METRIC_FIELDS and (value is None or type(value) in (int, float, bool))}
    for key in ("incumbent", "feedback_summary", "preliminary"):
        if key in metrics:
            result[key] = summarize_metrics(metrics[key])
    eligibility = metrics.get("confirmation_eligibility")
    if isinstance(eligibility, dict) and type(eligibility.get("eligible")) is bool:
        result["confirmation_eligibility"] = {"eligible": eligibility["eligible"]}
    return result


def summarize_job(job):
    result = {key: job.get(key) for key in JOB_FIELDS}
    for key, value in result.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
    final = job.get("final_evaluation")
    if isinstance(final, dict):
        result["final_evaluation"] = {
            key: final[key] for key in ("status", "reason", "baseline_iteration", "final_iteration",
                                       "used_for_selection") if key in final}
        for key in ("baseline", "final"):
            if key in final:
                result["final_evaluation"][key] = summarize_metrics(final[key])
    else:
        result["final_evaluation"] = None
    return result


def summarize_iteration(iteration):
    result = {key: iteration.get(key) for key in ITERATION_FIELDS}
    result["metrics"] = summarize_metrics(iteration.get("metrics"))
    summary = iteration.get("summary")
    if isinstance(summary, dict):
        result["summary"] = {key: summary.get(key) for key in ("n_tasks", "passed", "failed", "errored", "score")}
    return result


async def acquire_controller(connection, *, development_repeat=False):
    # Shared with the final-holdout runner, plus an explicit no-active-job check.
    if not await connection.fetchval("SELECT pg_try_advisory_lock($1)", CONTROLLER_LOCK):
        raise RuntimeError("Another review controller owns this evidence database")
    if await connection.fetchval("SELECT count(*) FROM jobs WHERE status IN ('RUNNING','QUEUED')"):
        raise RuntimeError("Active jobs exist; the fresh experiment cannot overlap their spending")
    if await connection.fetchval("SELECT count(*) FROM sandbox_slots"):
        raise RuntimeError("Sandbox reservations remain; reconcile before dispatch")
    prior_jobs = await connection.fetchval("SELECT count(*) FROM jobs WHERE left(idempotency_key,length($1::text))=$1",
                                          "aos_calibrated_")
    if prior_jobs and not development_repeat:
        raise RuntimeError("A calibrated review was already submitted; do not silently repeat this experiment")
    prior_holdout = await connection.fetchval("SELECT count(*) FROM task_attempts WHERE task_name=ANY($1::text[])",
                                             CALIBRATED_SPLIT["holdout"])
    if prior_holdout and not development_repeat:
        raise RuntimeError("The preserved holdout already has observations in this evidence database")
    return {"development_repeat": development_repeat, "prior_calibrated_jobs": prior_jobs,
            "prior_holdout_observations": prior_holdout,
            "previously_unobserved_holdout": prior_holdout == 0}


async def api_snapshot(client, headers, job_id):
    async def get(path, **kwargs):
        response = await client.get(path, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    job = await get(f"/v1/jobs/{job_id}")
    iterations = await get(f"/v1/jobs/{job_id}/iterations")
    attempts = []
    for iteration in iterations:
        offset = 0
        while True:
            rows = await get(f"/v1/jobs/{job_id}/iterations/{iteration['index']}/attempts",
                             params={"include_trace": "false", "limit": 100, "offset": offset})
            attempts.extend({"iteration_index": iteration["index"], **redact_attempt(row)} for row in rows)
            if len(rows) < 100:
                break
            offset += len(rows)
    return summarize_job(job), [summarize_iteration(item) for item in iterations], attempts


async def database_snapshot(connection, authorized_job_id):
    """Only called with an ID returned by authenticated exact-key cleanup."""
    job_id = UUID(str(authorized_job_id))
    job = dict(await connection.fetchrow("""SELECT j.*,
        (SELECT coalesce(sum(actual_usd),0) FROM spend_reservations s WHERE s.job_id=j.id) AS spent_usd,
        (SELECT coalesce(sum(reserved_usd),0) FROM spend_reservations s
         WHERE s.job_id=j.id AND s.settled_at IS NULL) AS reserved_usd
        FROM jobs j WHERE j.id=$1""", job_id))
    job["job_id"] = job["id"]
    job["best_iteration_index"] = await connection.fetchval(
        "SELECT index FROM iterations WHERE id=$1", job.get("best_iteration_id"))
    iterations = await connection.fetch("""SELECT i.*,a.sha256 AS agent_sha256 FROM iterations i
        JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.job_id=$1 ORDER BY i.index""", job_id)
    attempts = await connection.fetch("""SELECT a.task_name,a.evaluation_phase,a.repetition,a.attempt,
        a.lease_epoch,a.status,a.reward,a.failure_class,a.duration_s,a.n_steps,a.exit_reason,a.input_tokens,
        a.output_tokens,a.model_cost_usd,a.sandbox_cost_usd,a.cost_usd,a.cost_known,a.metadata,i.index AS iteration_index
        FROM task_attempts a JOIN iterations i ON i.id=a.iteration_id
        WHERE i.job_id=$1 ORDER BY i.index,a.finished_at,a.id""", job_id)
    return summarize_job(job), [summarize_iteration(dict(row)) for row in iterations], [
        {"iteration_index": row["iteration_index"], **redact_attempt(dict(row))} for row in attempts]


async def run(args):
    validate_arguments(args)
    proof = verify_source_proof(args.source_proof)
    credential = settings.neosigma_open_api_key
    if credential is None or not credential.get_secret_value():
        raise SystemExit("NEOSIGMA_OPEN_API_KEY is required; personal-key fallback is forbidden.")
    if settings.neosigma_base_url != "https://api.openai.com/v1":
        raise SystemExit("Only the approved inference endpoint is permitted")
    for model in ("gpt-5.4-mini", "gpt-5.4"):
        if model not in settings.model_prices:
            raise RuntimeError("Both explicitly selected models need configured token prices")
    control = json.loads(args.baseline_control.read_text())
    dsn = control["database_url"]
    development_repeat = bool(getattr(args, "development_repeat", False))
    prior_job_id = getattr(args, 'prior_job_id', None)
    default_report = ROOT / "docs" / "live_calibrated_benchmark_review.json"
    report_path = getattr(args, "report_path", None) or default_report
    if development_repeat and report_path.resolve() == default_report.resolve():
        raise ValueError("A repeated development experiment requires a separate --report-path")
    name = ("aos_development_" if development_repeat else "aos_calibrated_") + datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + uuid4().hex[:8]
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, mode=0o700)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        (directory / "previous_report.json").write_bytes(report_path.read_bytes())
    helper_paths = ("scripts/run_paired_selection_review.py", "scripts/run_final_holdout_review.py")
    helper_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in helper_paths}
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "repeated_calibrated12_development" if development_repeat else "fresh_calibrated12_real_optimization",
        "run_directory": str(directory.relative_to(ROOT)),
        "baseline_source_job_id": control["job_id"], "baseline_sha256": BASELINE_SHA256,
        "profile": "calibrated12", "evaluation_split": CALIBRATED_SPLIT, "exclusions": CALIBRATION_EXCLUSIONS,
        "inherited_candidates": False, "inherited_scores": False, "inherited_rejection_memory": False,
        "rejection_memory_source_job_id": str(prior_job_id) if prior_job_id is not None else None,
        "source_fixture_proof": str(args.source_proof), "implementation_sha256": proof,
        "audit_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_helpers_sha256": helper_hashes, "credential_source": "NEOSIGMA_OPEN_API_KEY",
        "max_total_usd": args.max_total_usd, "diagnostic_spent_usd": float(DIAGNOSTIC_USD),
        "benchmark_revision": settings.benchmark_revision, "iterations": [], "attempts": [],
        "fixed_reasoning_policy": {"tool_bearing_agent_requests": "none", "optimizer_requests": "low"},
        "limitations": [
            "The calibrated12 cohort differs from stress16; scores cannot be compared directly across those experiments.",
            "Using gpt-5.4 for proposals is a new optimizer hypothesis, not demonstrated improvement or an isolated model ablation.",
            "The original baseline is re-evaluated; no old evaluation score or candidate source is reused as new evidence.",
            "Small fixed task sets and paired confirmation do not establish broad benchmark generalization.",
            "The workflow freezes the final source before its own holdout; holdout results are not optimizer feedback.",
            "Holdout runs baseline first; identical final source reuses its attempts, so this is not an independent repetition.",
            "Model cost uses uncached token-price estimates; local Docker dollars are unpriced.",
            "Measured runtime includes setup, verification, cleanup and amd64 emulation on aarch64.",
            "Pinned verifier completion is operational evidence, not tamper-proof adversarial grading.",
            "One paid controller is required; advisory exclusion covers only cooperating review scripts."]}
    if development_repeat:
        report["limitations"].append("These task roles have been evaluated before. Held-back tasks remain excluded from optimizer prompts, but this is repeated development validation, not a previously unseen generalization test.")
    if prior_job_id is not None:
        report['limitations'].append('Compatible rejected-proposal feedback diagnostics are inherited as optimizer memory; all baseline, candidate, confirmation and final evaluation scores require new trials.')
    def write_report():
        for path in (directory / "report.json", report_path):
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, default=str))
            temporary.replace(path)
    write_report()
    processes, handles = [], []
    controller = api = worker = None
    active_job = None
    initial_spent = None
    headers = {"Authorization": "Bearer " + control["api_key"]}
    private = {**control, "baseline_job_id": control["job_id"], "job_id": None, "submissions": []}
    def save_private():
        path = directory / "control.json"
        path.write_text(json.dumps(private, indent=2))
        path.chmod(0o600)
    save_private()
    async def save():
        spent, reserved = await ledger(dsn)
        report.update(updated_at=datetime.now(timezone.utc).isoformat(),
            cumulative_spent_usd=float(spent + DIAGNOSTIC_USD), cumulative_reserved_usd=float(reserved))
        if initial_spent is not None:
            report.update(experiment_spent_usd=float(spent - initial_spent),
                remaining_experiment_budget_usd=float(remaining_budget(initial_spent, spent, reserved, args.max_total_usd)))
        write_report()
    def check_frozen():
        if verify_source_proof(args.source_proof) != proof:
            raise RuntimeError("Verified production implementation changed during the experiment")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report["audit_runner_sha256"]:
            raise RuntimeError("Experiment controller changed during execution")
        if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest for path, digest in helper_hashes.items()):
            raise RuntimeError("Shared review helpers changed during execution")
    try:
        controller = await asyncpg.connect(dsn, timeout=5, command_timeout=10)
        report["evaluation_exposure"] = await acquire_controller(controller, development_repeat=development_repeat)
        initial_spent, initial_reserved = await ledger(dsn)
        if initial_reserved:
            raise RuntimeError("Unresolved reservations must be reconciled before a new experiment")
        if initial_spent + DIAGNOSTIC_USD + Decimal(str(args.max_total_usd)) > SESSION_AUTHORIZATION_USD:
            raise RuntimeError("The requested experiment cap exceeds cumulative session authorization")
        report["previous_spent_usd"] = float(initial_spent + DIAGNOSTIC_USD)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "DATABASE_URL": dsn, "EXECUTOR": "harbor", "OPTIMIZER": "llm",
            "SANDBOX_PROVIDER": "docker", "HARBOR_EXECUTABLE": str(Path.home() / ".local/bin/harbor"),
            "MAX_CONCURRENCY_PER_ITERATION": "2", "GLOBAL_TRIAL_SLOTS": "2", "WORKER_CONCURRENCY": "1",
            "CLAIM_POLL_SEC": ".5", "HEARTBEAT_SEC": "5", "LEASE_TTL_SEC": "60",
            "RUNS_DIR": str(directory / "tasks"), "NEOSIGMA_OPEN_API_KEY": credential.get_secret_value(),
            "AGENT_MODEL": "gpt-5.4-mini", "OPTIMIZER_MODEL": "gpt-5.4"}
        for key in ("OPENAI_API_KEY", "OPEN_API_KEY", "E2B_API_KEY"):
            env.pop(key, None)
        def start(command, label):
            handle = (directory / (label + ".log")).open("w")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            return process
        api = start([sys.executable, "-m", "uvicorn", "service.api.app:app", "--host", "127.0.0.1", "--port", str(port)], "api")
        private.update(api_url=base_url, api_pid=api.pid)
        save_private()
        deadline = time.monotonic() + args.max_wall_clock_min * 60
        async with httpx.AsyncClient(base_url=base_url, timeout=20) as client:
            for _ in range(100):
                try:
                    if (await client.get("/v1/readyz")).status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(.1)
            else:
                raise RuntimeError("API readiness failed")
            prior_response = await client.get(f"/v1/jobs/{control['job_id']}", headers=headers)
            prior_response.raise_for_status()
            prior = prior_response.json()
            if prior["status"] not in TERMINAL or prior["config"]["benchmark_revision"] != settings.benchmark_revision:
                raise RuntimeError("Original baseline experiment must be terminal at the same pinned revision")
            baseline_response = await client.get(f"/v1/jobs/{control['job_id']}/iterations/0", headers=headers)
            baseline_response.raise_for_status()
            baseline = baseline_response.json()
            if baseline["agent_sha256"] != BASELINE_SHA256:
                raise ValueError("Authorized source is not the original baseline")
            spent, reserved = await ledger(dsn)
            allowance = remaining_budget(initial_spent, spent, reserved, args.max_total_usd)
            body = build_request(baseline["agent_source"], allowance, args.max_wall_clock_min,
                                 prior_job_id=prior_job_id)
            report["request"] = {key: value for key, value in body.items() if key != "agent_source"}
            report["effective_inference_limits"] = {
                "max_output_tokens": min(body["options"]["max_tokens_per_call"], settings.agent_max_output_tokens),
                "max_request_bytes": settings.max_inference_body_bytes,
                "max_model_calls_per_trial": settings.max_model_calls_per_trial}
            check_frozen()
            # All source and budget checks precede the only new job.
            private["submissions"].append({"key": name})
            save_private()
            report.update(status="SUBMITTING", idempotency_key=name)
            await save()
            response = await client.post("/v1/jobs", headers={**headers, "Idempotency-Key": name}, json=body)
            response.raise_for_status()
            active_job = response.json()["job_id"]
            private["job_id"] = active_job
            save_private()
            admitted = await client.get(f"/v1/jobs/{active_job}", headers=headers)
            admitted.raise_for_status()
            admitted_config = admitted.json()['config']
            report["sandbox_harness"] = admitted_config.get("sandbox_harness", {"mode": "cold-v1"})
            report['inherited_rejection_memory_entries'] = len(admitted_config.get('optimizer_history_seed') or [])
            report['inherited_rejection_memory'] = report['inherited_rejection_memory_entries'] > 0
            worker = start([sys.executable, "-m", "service.worker.main"], "worker")
            private["worker_pid"] = worker.pid
            save_private()
            print(json.dumps({"event": "calibrated_submitted", "job_id": active_job,
                "cap_usd": float(allowance), "report": str(report_path)}), flush=True)
            last = None
            while True:
                check_frozen()
                job, iterations, attempts = await api_snapshot(client, headers, active_job)
                report.update(status=job["status"], job=job, iterations=iterations, attempts=attempts)
                await save()
                state = (job["status"], job["current_iteration"], len(attempts), job["spent_usd"], job["reserved_usd"])
                if state != last:
                    print(json.dumps({"event": "calibrated_progress", "status": state[0], "iteration": state[1],
                        "attempts_finished": state[2], "spent_usd": state[3], "reserved_usd": state[4]}), flush=True)
                    last = state
                if job["status"] in TERMINAL:
                    report["terminal_observed"] = True
                    await save()
                    return
                if worker.poll() is not None:
                    raise RuntimeError("Worker exited before the experiment completed")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Experiment wall-clock limit reached")
                await asyncio.sleep(10)
    except BaseException as exc:
        report.update(status="INCOMPLETE", error_type=type(exc).__name__)
        if api is not None:
            if active_job:
                try:
                    async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
                        response = await client.post(f"/v1/jobs/{active_job}/cancel", headers=headers)
                        response.raise_for_status()
                except Exception as cleanup_error:
                    report["api_cancel_error_type"] = type(cleanup_error).__name__
            # No "not found" cleanup can race a still-committing POST.
            await stop_process(api, timeout=10)
        report["durable_cleanup_confirmed"] = not private["submissions"]
        if private["submissions"]:
            for cleanup_attempt in range(3):
                connection = None
                try:
                    connection = await asyncpg.connect(dsn, timeout=5, command_timeout=5)
                    await _init_conn(connection)
                    for _ in range(40):
                        rows = await cancel_owned_submissions(connection, control["api_key"],
                            [item["key"] for item in private["submissions"]])
                        report["durable_cleanup_confirmed"] = all(
                            row["status"] in TERMINAL or row["cancel_requested"] for row in rows)
                        for row in rows:
                            private["job_id"] = row["job_id"]
                            save_private()
                            job, iterations, attempts = await database_snapshot(connection, row["job_id"])
                            report.update(job=job, iterations=iterations, attempts=attempts)
                        if all(row["status"] in TERMINAL for row in rows):
                            break
                        await asyncio.sleep(1)
                    break
                except Exception as cleanup_error:
                    report["database_cleanup_error_type"] = type(cleanup_error).__name__
                    if cleanup_attempt < 2:
                        await asyncio.sleep(1)
                finally:
                    if connection is not None:
                        await connection.close()
        try:
            await save()
        except Exception as accounting_error:
            report["final_accounting_error_type"] = type(accounting_error).__name__
        write_report()
        raise
    finally:
        for process in processes:
            await stop_process(process)
        for handle in handles:
            handle.close()
        if controller is not None:
            await controller.close()
        report["owned_processes_stopped"] = all(process.poll() is not None for process in processes)
        write_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--baseline-control", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path, required=True)
    parser.add_argument("--max-total-usd", type=float, default=6)
    parser.add_argument("--max-wall-clock-min", type=int, default=90)
    parser.add_argument("--development-repeat", action="store_true",
                        help="Explicitly reuse previously evaluated tasks; never claim an untouched holdout")
    parser.add_argument("--prior-job-id", type=UUID,
                        help="Import compatible rejected-proposal feedback memory only; never scores or a prior candidate")
    parser.add_argument("--report-path", type=Path,
                        help="Separate report required for a repeated development experiment")
    asyncio.run(run(parser.parse_args()))
