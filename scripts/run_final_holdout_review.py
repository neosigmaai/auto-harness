"""Run one sealed final holdout after an explicitly frozen selection decision.

No optimizer is invoked, old decisions are not rewritten, and no holdout evidence
is fed back into selection. This script is inert without --allow-paid. Sources
are read as data and executed only by the service's sandbox.
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
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncpg
import httpx

from scripts.run_paired_selection_review import (
    ATTEMPT_FIELDS, DIAGNOSTIC_USD, SESSION_AUTHORIZATION_USD, TERMINAL,
    audit_options, cancel_owned_submissions, ledger, remaining_budget,
    stop_process, verify_source_proof, redact_verifier, verifier_valid,
)
from service.config import settings
from service.db import _init_conn
from service.models import JobCreate
from service.repo import sha256_text
from service.worker.job_runner import evaluation_metrics

HOLDOUT_TASKS = ("regex-log", "sparql-university", "tune-mjcf", "polyglot-c-py")
CONTROLLER_LOCK = 0x414F53484F4C44


def validate_arguments(args):
    # Check permission before reading settings, source reports, or any database.
    if not args.allow_paid:
        raise SystemExit("No model calls sent. --allow-paid is required after the final source is frozen.")
    if not math.isfinite(args.max_total_usd) or not 0 < args.max_total_usd <= 3:
        raise SystemExit("The authorized final-holdout ceiling is greater than $0 and at most $3.")
    if not 1 <= args.max_wall_clock_min <= 90:
        raise SystemExit("The final-holdout wall-clock limit must be between 1 and 90 minutes.")
    if args.selected_source not in ("baseline", "candidate"):
        raise ValueError("An explicit baseline or candidate source must be frozen")
    if (len(args.selected_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in args.selected_sha256)):
        raise ValueError("An explicit lowercase SHA-256 of the final source is required")


def freeze_plan(selection, cfg, source_job_id, sources, proof, selected_label, selected_sha256):
    """Validate the final choice without opening any holdout instructions/traces."""
    if selection.get("status") != "COMPLETE" or selection.get("holdout_executed") is not False:
        raise ValueError("Selection must be complete and must not have used holdout evidence")
    if selection.get("source_job_id") != str(source_job_id):
        raise ValueError("Selection/source experiment mismatch")
    if selection.get("implementation_sha256") != proof:
        raise ValueError("Holdout must use the exact verified selection implementation")
    if (selection.get("benchmark_revision") != cfg.get("benchmark_revision") or
            selection.get("agent_model") != cfg.get("agent_model") or
            selection.get("tasks") != cfg.get("evaluation_split", {}).get("selection")):
        raise ValueError("Frozen selection benchmark/model/tasks mismatch")
    options = audit_options(cfg["options"])
    if selection.get("options") != options:
        raise ValueError("Holdout must use the same frozen selection options")
    limits = {"max_output_tokens": min(options["max_tokens_per_call"], settings.agent_max_output_tokens),
              "max_request_bytes": settings.max_inference_body_bytes,
              "max_model_calls_per_trial": settings.max_model_calls_per_trial}
    if selection.get("effective_inference_limits") != limits:
        raise ValueError("Effective inference limits differ from selection")
    if selection.get("fixed_reasoning_policy") != {"tool_bearing_agent_requests": "none", "optimizer_calls": 0}:
        raise ValueError("Reasoning policy differs from the frozen selection contract")
    tasks = cfg.get("evaluation_split", {}).get("holdout", [])
    if len(tasks) != 4 or set(tasks) != set(HOLDOUT_TASKS):
        raise ValueError("Holdout must contain exactly the original four sealed tasks")
    hashes = {label: sha256_text(source) for label, source in sources.items()}
    if set(hashes) != {"baseline", "candidate"} or hashes != selection.get("source_hashes"):
        raise ValueError("Stored sources differ from the completed selection audit")
    if selected_label not in hashes or hashes[selected_label] != selected_sha256:
        raise ValueError("Explicit final source hash does not match the selected source")
    metrics = selection.get("metrics") or {}
    if not all((metrics.get(label) or {}).get("complete") is True for label in hashes):
        raise ValueError("Selection lacks complete baseline and candidate metrics")
    decision = (selection.get("observed_policy_decision") or {}).get("candidate_qualifies")
    if type(decision) is not bool:
        raise ValueError("Selection lacks a declared final policy decision")
    # Final choice is explicit, but cannot quietly override the policy used to
    # justify the experiment. A new policy requires a separately reviewed plan.
    expected = "candidate" if decision else "baseline"
    if selected_label != expected:
        raise ValueError("Explicit final source contradicts the completed selection policy")
    same_source = hashes["baseline"] == selected_sha256
    return {"tasks": list(tasks), "options": options, "source_hashes": hashes,
            "selected_source": selected_label, "selected_sha256": selected_sha256,
            "seed": cfg.get("seed") or 1234,
            "same_source": same_source,
            "planned_order": ["baseline"] if same_source else ["baseline", "selected"],
            "comparison_status": "same_source_no_improvement" if same_source else "not_yet_measured"}


def redact_attempt(row):
    """Keep numeric verifier evidence, never test bodies, output, or model traces."""
    result = {key: row.get(key) for key in ATTEMPT_FIELDS}
    metadata = row.get("metadata") or {}
    validity = metadata.get("verifier_validity") or {}
    result["metadata"] = {
        "verifier_validity": redact_verifier(validity),
        "duration_basis": metadata.get("duration_basis"),
        "model_cost_known": metadata.get("model_cost_known", row.get("cost_known", False)),
        "benchmark_revision": metadata.get("benchmark_revision"),
        "agent_sha256": metadata.get("agent_sha256"),
    }
    return result


def aggregate_holdout(attempts, task_names):
    """Each fixed task counts once; infrastructure retries still cost time/money."""
    if any(a.get("task_name") not in task_names or a.get("evaluation_phase") != "main" or
           a.get("repetition") != 1 for a in attempts):
        raise ValueError("Unexpected task/phase/repetition in the sealed holdout")
    samples, verifiers = [], {}
    for task in task_names:
        rows = sorted((a for a in attempts if a["task_name"] == task), key=lambda a: a["attempt"])
        if not rows:
            continue
        if len({a["attempt"] for a in rows}) != len(rows):
            raise ValueError("Duplicate immutable trial attempt")
        chosen = next((a for a in rows if a["status"] != "error"), rows[-1])
        sample = {**chosen, "physical_attempts": len(rows)}
        for key in ("duration_s", "model_cost_usd", "sandbox_cost_usd", "cost_usd"):
            values = [a.get(key) for a in rows]
            sample[key] = sum(values) if all(v is not None for v in values) else None
        for key in ("input_tokens", "output_tokens"):
            sample[key] = sum(a.get(key) or 0 for a in rows)
        sample["model_cost_known"] = all((a.get("metadata") or {}).get(
            "model_cost_known", a.get("cost_known", False)) for a in rows)
        sample["cost_known"] = all(a.get("cost_known", False) for a in rows)
        validity = (chosen.get("metadata") or {}).get("verifier_validity") or {}
        verifiers[task] = {"valid": verifier_valid(chosen),
                           "chosen_attempt": chosen["attempt"], "counts": validity.get("counts"),
                           "pytest_exit_code": validity.get("pytest_exit_code"),
                           "contract": validity.get("contract"), "case_counts": validity.get("case_counts"),
                           "tests_collected": validity.get("tests_collected")}
        samples.append(sample)
    metrics = evaluation_metrics(samples, len(task_names))
    verifier_complete = len(verifiers) == len(task_names) and all(v["valid"] for v in verifiers.values())
    return {"metrics": metrics, "verifier_complete": verifier_complete, "verifiers": verifiers,
            "comparison_eligible": metrics["complete"] and verifier_complete,
            "infrastructure_retries": len(attempts) - len(samples)}


def compare_holdout(baseline, selected, same_source):
    """Describe the final observation; never reselect/promote using holdout data."""
    if not baseline["comparison_eligible"] or not selected["comparison_eligible"]:
        return {"status": "incomplete_verifier_evidence", "quality_delta": None,
                "selection_changed": False, "improvement_demonstrated": False}
    before, after = baseline["metrics"], selected["metrics"]
    return {"status": "same_source_no_improvement" if same_source else "complete_observation",
            "quality_delta": after["mean_reward"] - before["mean_reward"],
            "mean_model_cost_delta_usd": (after["mean_model_cost_usd"] - before["mean_model_cost_usd"]
                if after["mean_model_cost_usd"] is not None and before["mean_model_cost_usd"] is not None else None),
            "p95_task_duration_delta_s": (after["p95_task_duration_s"] - before["p95_task_duration_s"]
                if after["p95_task_duration_s"] is not None and before["p95_task_duration_s"] is not None else None),
            "selection_changed": False, "same_source_result_reused": same_source,
            "improvement_demonstrated": False,
            "interpretation": "One observation per task is descriptive; it does not establish a general performance gain."}


async def acquire_controller(connection):
    if not await connection.fetchval("SELECT pg_try_advisory_lock($1)", CONTROLLER_LOCK):
        raise RuntimeError("Another final-holdout controller owns this database")
    if await connection.fetchval("SELECT count(*) FROM jobs WHERE status IN ('RUNNING','QUEUED')"):
        raise RuntimeError("Active evidence jobs exist; holdout cannot overlap their spending")
    if await connection.fetchval("SELECT count(*) FROM sandbox_slots"):
        raise RuntimeError("Sandbox reservations remain; reconcile them before holdout")
    if await connection.fetchval("SELECT count(*) FROM jobs WHERE left(idempotency_key,length($1::text))=$1",
                                 "aos_final_holdout_"):
        raise RuntimeError("A final holdout was already submitted; repeated inspection requires a new reviewed protocol")
    if await connection.fetchval("SELECT count(*) FROM task_attempts WHERE task_name=ANY($1::text[])",
                                 list(HOLDOUT_TASKS)):
        raise RuntimeError("Original holdout tasks were already observed in this evidence database")


async def run(args):
    validate_arguments(args)
    proof = verify_source_proof(args.source_proof)
    selection_bytes = args.selection_report.read_bytes()
    selection = json.loads(selection_bytes)
    control = json.loads(args.prior_control.read_text())
    credential = settings.neosigma_open_api_key
    if credential is None or not credential.get_secret_value():
        raise SystemExit("NEOSIGMA_OPEN_API_KEY is required; personal-key fallback is forbidden.")
    if settings.neosigma_base_url != "https://api.openai.com/v1":
        raise SystemExit("Unexpected inference endpoint")
    dsn = control["database_url"]
    controller = await asyncpg.connect(dsn, timeout=5, command_timeout=10)
    processes, handles = [], []
    api = worker = None
    report = None
    report_path = ROOT / "docs" / "live_final_holdout_review.json"
    try:
        await acquire_controller(controller)
        initial_spent, initial_reserved = await ledger(dsn)
        if initial_reserved:
            raise RuntimeError("Unsettled prior requests exist; reconcile before holdout")
        if initial_spent + DIAGNOSTIC_USD + Decimal(str(args.max_total_usd)) > SESSION_AUTHORIZATION_USD:
            raise RuntimeError("Holdout ceiling exceeds cumulative session authorization")
        name = "aos_final_holdout_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + uuid4().hex[:8]
        directory = ROOT / "runs" / name
        directory.mkdir(mode=0o700, parents=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "DATABASE_URL": dsn, "EXECUTOR": "harbor", "OPTIMIZER": "llm",
            "SANDBOX_PROVIDER": "docker", "HARBOR_EXECUTABLE": str(Path.home() / ".local/bin/harbor"),
            "MAX_CONCURRENCY_PER_ITERATION": "2", "GLOBAL_TRIAL_SLOTS": "2", "WORKER_CONCURRENCY": "1",
            "CLAIM_POLL_SEC": ".5", "HEARTBEAT_SEC": "5", "LEASE_TTL_SEC": "60",
            "RUNS_DIR": str(directory / "tasks"), "NEOSIGMA_OPEN_API_KEY": credential.get_secret_value()}
        for key in ("OPENAI_API_KEY", "OPEN_API_KEY", "E2B_API_KEY"):
            env.pop(key, None)
        def start(command, label):
            handle = (directory / (label + ".log")).open("w")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            return process
        api = start([sys.executable, "-m", "uvicorn", "service.api.app:app", "--host", "127.0.0.1", "--port", str(port)], "api")
        base_url = f"http://127.0.0.1:{port}"
        headers = {"Authorization": "Bearer " + control["api_key"]}
        private = {**control, "api_url": base_url, "api_pid": api.pid, "jobs": [], "submissions": []}
        def save_private():
            path = directory / "control.json"
            path.write_text(json.dumps(private, indent=2))
            path.chmod(0o600)
        save_private()
        report = {"status": "starting", "mode": "sealed_final_holdout_via_benchmark_only_api_jobs",
            "started_at": datetime.now(timezone.utc).isoformat(), "run_directory": str(directory.relative_to(ROOT)),
            "source_job_id": control["job_id"], "selection_report_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "implementation_sha256": proof, "source_fixture_proof": str(args.source_proof),
            "audit_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_helpers_sha256": hashlib.sha256((ROOT / "scripts/run_paired_selection_review.py").read_bytes()).hexdigest(),
            "credential_source": "NEOSIGMA_OPEN_API_KEY", "provider_seed_enforced": False,
            "max_total_usd": args.max_total_usd, "previous_spent_usd": float(initial_spent + DIAGNOSTIC_USD),
            "jobs": [], "optimizer_calls": 0, "holdout_used_for_selection": False,
            "holdout_attempts_observed": 0,
            "old_experiment_decisions_modified": False,
            "limitations": ["Four predeclared holdout tasks with one observation each do not establish statistical significance or generalization.",
                "Baseline runs first; cache/order effects and provider variance can influence the second source.",
                "Same-source results are reused from one four-task run, not represented as independent repetitions.",
                "Model dollars are uncached token-price estimates; local Docker dollars remain unpriced.",
                "Warm amd64 task images run on an aarch64 daemon; measured time includes emulation, admission, setup, verification and cleanup.",
                "This protocol assumes one paid controller; its advisory lock excludes other holdout runners only.",
                "Pinned verifier completion is operational evidence, not tamper-proof adversarial grading."]}
        active_job = None
        deadline = time.monotonic() + args.max_wall_clock_min * 60
        async def save():
            spent, reserved = await ledger(dsn)
            report.update(updated_at=datetime.now(timezone.utc).isoformat(),
                holdout_attempts_observed=sum(len(job["attempts"]) for job in report["jobs"]),
                holdout_spent_usd=float(spent - initial_spent), outstanding_reserved_usd=float(reserved),
                cumulative_spent_usd=float(spent + DIAGNOSTIC_USD),
                remaining_holdout_budget_usd=float(remaining_budget(initial_spent, spent, reserved, args.max_total_usd)))
            report_path.write_text(json.dumps(report, indent=2, default=str))
        async def request_json(client, path):
            response = await client.get(path, headers=headers)
            response.raise_for_status()
            return response.json()
        def check_frozen():
            if verify_source_proof(args.source_proof) != proof:
                raise RuntimeError("Verified implementation changed during holdout")
            if args.selection_report.read_bytes() != selection_bytes:
                raise RuntimeError("Completed selection evidence changed during holdout")
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report["audit_runner_sha256"]:
                raise RuntimeError("Holdout runner changed during execution")
            if hashlib.sha256((ROOT / "scripts/run_paired_selection_review.py").read_bytes()).hexdigest() != report["shared_helpers_sha256"]:
                raise RuntimeError("Shared audit helpers changed during holdout")
        try:
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
                prior = await request_json(client, f"/v1/jobs/{control['job_id']}")
                if prior["status"] not in TERMINAL:
                    raise RuntimeError("Source experiment must be terminal")
                cfg = prior["config"]
                if cfg["benchmark_revision"] != settings.benchmark_revision:
                    raise RuntimeError("Benchmark revision differs from the frozen experiment")
                sources = {}
                candidate_index = selection.get("source_candidate_index")
                if type(candidate_index) is not int or candidate_index < 1:
                    raise ValueError("Selection must identify an existing candidate iteration")
                for label, index in (("baseline", 0), ("candidate", candidate_index)):
                    detail = await request_json(client, f"/v1/jobs/{control['job_id']}/iterations/{index}")
                    sources[label] = detail["agent_source"]
                    if sha256_text(sources[label]) != detail["agent_sha256"]:
                        raise RuntimeError("Stored source hash mismatch")
                plan = freeze_plan(selection, cfg, control["job_id"], sources, proof,
                                   args.selected_source, args.selected_sha256)
                report.update(status="RUNNING", frozen_plan=plan, benchmark_revision=cfg["benchmark_revision"],
                    agent_model=cfg["agent_model"], options=plan["options"],
                    fixed_reasoning_policy=selection["fixed_reasoning_policy"],
                    effective_inference_limits=selection["effective_inference_limits"])
                # All source/model/options checks happen before a worker can dispatch.
                worker = start([sys.executable, "-m", "service.worker.main"], "worker")
                private["worker_pid"] = worker.pid
                save_private()
                await save()
                for label in plan["planned_order"]:
                    check_frozen()
                    spent, reserved = await ledger(dsn)
                    remaining = remaining_budget(initial_spent, spent, reserved, args.max_total_usd)
                    seconds_left = deadline - time.monotonic()
                    if remaining <= 0 or seconds_left <= 0:
                        raise RuntimeError("Holdout budget/deadline exhausted before the next source")
                    source = sources["baseline"] if label == "baseline" else sources[args.selected_source]
                    body = {"benchmark": cfg["benchmark"], "tasks": plan["tasks"], "max_iterations": 0,
                        "agent_source": source, "agent_model": cfg["agent_model"], "optimizer_model": cfg["optimizer_model"],
                        "seed": plan["seed"], "options": plan["options"], "budget": {"max_llm_usd": float(remaining),
                            "max_wall_clock_min": max(1, math.ceil(seconds_left / 60))}}
                    JobCreate(**body)
                    key = f"{name}-{label}"
                    private["submissions"].append({"key": key, "source": label})
                    save_private()
                    entry = {"job_id": None, "idempotency_key": key, "source": label,
                             "agent_sha256": sha256_text(source), "status": "SUBMITTING", "attempts": []}
                    report["jobs"].append(entry)
                    await save()
                    response = await client.post("/v1/jobs", headers={**headers, "Idempotency-Key": key}, json=body)
                    response.raise_for_status()
                    active_job = response.json()["job_id"]
                    entry.update(job_id=active_job, status="QUEUED")
                    private["jobs"].append({"source": label, "job_id": active_job})
                    save_private()
                    print(json.dumps({"event": "holdout_submitted", "source": label, "job_id": active_job,
                                      "remaining_cap_usd": float(remaining)}), flush=True)
                    last = None
                    while True:
                        check_frozen()
                        job = await request_json(client, f"/v1/jobs/{active_job}")
                        iterations = await request_json(client, f"/v1/jobs/{active_job}/iterations")
                        attempts = []
                        if iterations:
                            raw = await request_json(client, f"/v1/jobs/{active_job}/iterations/0/attempts?include_trace=false&limit=100")
                            attempts = [redact_attempt(row) for row in raw]
                        summary = aggregate_holdout(attempts, plan["tasks"])
                        entry.update(status=job["status"], stop_reason=job["stop_reason"],
                            spent_usd=job["spent_usd"], reserved_usd=job["reserved_usd"], attempts=attempts,
                            evaluation=summary, workflow_metrics=iterations[0]["metrics"] if iterations else None)
                        await save()
                        state = (job["status"], len(attempts), job["spent_usd"], job["reserved_usd"])
                        if state != last:
                            print(json.dumps({"event": "holdout_progress", "source": label, "status": state[0],
                                "attempts_done": state[1], "spent_usd": state[2], "reserved_usd": state[3]}), flush=True)
                            last = state
                        if job["status"] in TERMINAL:
                            active_job = None
                            if job["status"] != "SUCCEEDED" or not summary["comparison_eligible"]:
                                raise RuntimeError("Incomplete verifier evidence; stop final holdout without a comparison")
                            break
                        if worker.poll() is not None:
                            raise RuntimeError("Worker exited during holdout")
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Overall holdout deadline reached")
                        await asyncio.sleep(10)
                baseline = report["jobs"][0]["evaluation"]
                selected = baseline if plan["same_source"] else report["jobs"][1]["evaluation"]
                report.update(status="COMPLETE", baseline=baseline, selected=selected,
                    selected_result_job_id=report["jobs"][0 if plan["same_source"] else 1]["job_id"],
                    comparison=compare_holdout(baseline, selected, plan["same_source"]))
                await save()
                print(json.dumps({"event": "holdout_complete", "comparison": report["comparison"],
                                  "report": str(report_path)}), flush=True)
        except BaseException as exc:
            report.update(status="INCOMPLETE", error_type=type(exc).__name__)
            if active_job:
                try:
                    async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
                        response = await client.post(f"/v1/jobs/{active_job}/cancel", headers=headers)
                        response.raise_for_status()
                except Exception as cleanup_error:
                    report["api_cancel_error_type"] = type(cleanup_error).__name__
            # Drain the API before looking up keys, including a POST whose
            # response was lost after its database transaction committed.
            await stop_process(api, timeout=10)
            report["durable_cleanup_confirmed"] = not private["submissions"]
            for cleanup_attempt in range(3):
                connection = None
                try:
                    connection = await asyncpg.connect(dsn, timeout=5, command_timeout=5)
                    await _init_conn(connection)
                    for _ in range(40):
                        rows = await cancel_owned_submissions(connection, control["api_key"],
                            [item["key"] for item in private["submissions"]])
                        for row in rows:
                            entry = next(j for j in report["jobs"] if j["idempotency_key"] == row["idempotency_key"])
                            # Shared helper intentionally omits metadata. Keep any
                            # already-read verifier evidence; missing remains unknown.
                            previous = {(a["task_name"], a["attempt"]): a for a in entry["attempts"]}
                            for attempt in row["attempts"]:
                                old = previous.get((attempt["task_name"], attempt["attempt"]), {})
                                if "metadata" in old:
                                    attempt["metadata"] = old["metadata"]
                            entry.update(row)
                        report["durable_cleanup_confirmed"] = all(
                            row["status"] in TERMINAL or row["cancel_requested"] for row in rows)
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
            report_path.write_text(json.dumps(report, indent=2, default=str))
            raise
    finally:
        for process in processes:
            await stop_process(process)
        for handle in handles:
            handle.close()
        await controller.close()
        if report is not None:
            print(json.dumps({"event": "holdout_processes_stopped", "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--prior-control", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path, required=True)
    parser.add_argument("--selected-source", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--selected-sha256", required=True)
    parser.add_argument("--max-total-usd", type=float, default=3)
    parser.add_argument("--max-wall-clock-min", type=int, default=90)
    asyncio.run(run(parser.parse_args()))
