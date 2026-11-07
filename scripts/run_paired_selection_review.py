"""External paired audit through four ordinary benchmark-only API jobs.

This never changes an old experiment's decisions. It freezes two existing sources,
runs the same selection tasks twice in alternating order, and reports the declared
policy against fresh evidence. No optimizer or holdout is invoked.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
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

from service.config import settings
from service.auth import Principal, hash_api_key
from service.db import migrate, _init_conn
from service.models import JobCreate
from service import repo
from service.repo import sha256_text
from service.worker.job_runner import evaluation_metrics, promotion_decision

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
DIAGNOSTIC_USD = Decimal("0.00093575")
SESSION_AUTHORIZATION_USD = Decimal("15")
ATTEMPT_FIELDS = ("task_name", "evaluation_phase", "repetition", "attempt", "lease_epoch",
    "status", "reward", "failure_class", "duration_s", "n_steps", "exit_reason", "input_tokens",
    "output_tokens", "model_cost_usd", "sandbox_cost_usd", "cost_usd", "cost_known")


def remaining_budget(start_spent, current_spent, reserved, cap):
    """Round down so API float serialization never intentionally raises the cap."""
    start, current, outstanding, limit = map(lambda x: Decimal(str(x)),
                                             (start_spent, current_spent, reserved, cap))
    if min(start, current, outstanding, limit) < 0 or current < start:
        raise ValueError("Invalid or regressed spending ledger")
    remaining = min(limit - (current - start) - outstanding,
                    SESSION_AUTHORIZATION_USD - DIAGNOSTIC_USD - current - outstanding)
    return max(Decimal(0), remaining.quantize(Decimal("0.000001"), rounding=ROUND_DOWN))


def verify_source_proof(path):
    proof = json.loads(path.read_text())
    if not isinstance(proof.get("files"), dict) or not proof["files"]:
        raise ValueError("Fixture source proof has no files")
    actual_paths = {str(p.relative_to(ROOT)) for p in (ROOT / "service").rglob("*.py")
                    if "tests" not in p.parts}
    actual_paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "service" / "migrations").glob("*.sql"))
    actual_paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "service" / "sandbox").rglob("*.json"))
    if actual_paths != set(proof["files"]):
        raise ValueError("Fixture proof does not cover the complete current service source")
    for relative, expected in proof["files"].items():
        current = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if current != expected:
            raise ValueError(f"Source differs from verified fixture: {relative}")
    return proof


def selection_task_names(config):
    """Keep the exact source experiment's task order and denominator."""
    split = config.get("evaluation_split") or {}
    names = split.get("selection")
    if (not isinstance(names, list) or len(names) not in (3, 4) or
            any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names)):
        raise ValueError("Expected exactly three or four distinct frozen selection tasks")
    if not set(names).issubset(config.get("tasks", [])):
        raise ValueError("Selection tasks do not belong to the source experiment")
    if set(names).intersection(split.get("feedback", []) + split.get("holdout", [])):
        raise ValueError("Selection tasks overlap another frozen evaluation role")
    return list(names)


def pinned_source_hashes(sources, baseline_hash, candidate_hash):
    expected = {"baseline": baseline_hash, "candidate": candidate_hash}
    actual = {label: sha256_text(source) for label, source in sources.items()}
    if any(not isinstance(value, str) or len(value) != 64 or
           any(c not in "0123456789abcdef" for c in value) for value in expected.values()):
        raise ValueError("Explicit lowercase SHA-256 pins are required for both sources")
    if actual != expected:
        raise ValueError("Source bytes differ from the explicitly authorized hashes")
    if baseline_hash == candidate_hash:
        raise ValueError("Candidate equals baseline")
    return actual


def archive_report(report_path, directory):
    """Preserve prior observations even when a caller reuses the default path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        return None
    archived = directory / "previous_report.json"
    archived.write_bytes(report_path.read_bytes())
    return str(archived)


VERIFIER_FIELDS = ("contract", "valid", "counts", "pytest_exit_code", "fresh_session_marker",
                   "orchestration_sha256", "reason", "error_type", "case_counts", "tests_collected",
                   "selected_cases", "executed_cases", "deselected_cases", "unexpected_cases", "duplicate_phase")


def redact_verifier(validity):
    result = {key: validity[key] for key in VERIFIER_FIELDS if key in validity}
    for field, keys in (("counts", ("tests", "passed", "failed", "skipped", "pending", "other")),
                        ("case_counts", ("passed", "failed", "skipped", "incomplete"))):
        if isinstance(result.get(field), dict):
            result[field] = {key: result[field][key] for key in keys if key in result[field]}
    return result


def redact_attempt(row):
    result = {key: row.get(key) for key in ATTEMPT_FIELDS}
    metadata = row.get("metadata") or {}
    validity = metadata.get("verifier_validity") or {}
    result["metadata"] = {
        "verifier_validity": redact_verifier(validity),
        "model_cost_known": metadata.get("model_cost_known", row.get("cost_known", False)),
        "duration_basis": metadata.get("duration_basis"),
    }
    return result


def verifier_valid(row):
    """A status/reward alone cannot stand in for executed-test evidence."""
    reward = row.get("reward")
    if (row.get("status") not in ("passed", "failed", "timeout") or
            type(reward) not in (int, float) or not math.isfinite(reward) or reward not in (0, 1)):
        return False
    validity = (row.get("metadata") or {}).get("verifier_validity") or {}
    if (validity.get("valid") is not True or
            validity.get("contract") not in {"pinned-pytest-ctrf-v1", "pinned-pytest-cases-v2"} or
            validity.get("fresh_session_marker") is not True):
        return False
    counts = validity.get("counts") or {}
    names = ("tests", "passed", "failed", "skipped", "pending", "other")
    if any(type(counts.get(name)) is not int or not 0 <= counts[name] <= 100000 for name in names):
        return False
    if (counts["tests"] <= 0 or counts["tests"] != sum(counts[name] for name in names[1:]) or
            counts["pending"] or counts["other"]):
        return False
    code = validity.get("pytest_exit_code")
    if type(code) is not int:
        return False
    if validity["contract"] == "pinned-pytest-ctrf-v1":
        # Historical evidence keeps its original contract; never manufacture
        # exact-case coverage for an old grouped-CTRF observation.
        return (counts["passed"] + counts["failed"] > 0 and
                ((code == 0 and reward == 1 and counts["failed"] == 0) or
                 (code == 1 and reward == 0 and counts["failed"] > 0)))
    collected = validity.get("tests_collected")
    if (type(collected) is not int or not 0 < collected <= 100000 or collected < counts["tests"] or
            type(validity.get("selected_cases")) is not int or validity["selected_cases"] != collected or
            type(validity.get("executed_cases")) is not int or not 0 < validity["executed_cases"] <= collected or
            any(type(validity.get(key)) is not int or validity[key] != 0
                for key in ("deselected_cases", "unexpected_cases")) or validity.get("duplicate_phase") is not False):
        return False
    cases = validity.get("case_counts") or {}
    keys = ("passed", "failed", "skipped", "incomplete")
    if (any(type(cases.get(key)) is not int or not 0 <= cases[key] <= 100000 for key in keys) or
            sum(cases[key] for key in keys) != collected or cases["incomplete"] or
            cases["passed"] + cases["failed"] <= 0 or validity["executed_cases"] < cases["passed"]):
        return False
    return ((code == 0 and reward == 1 and cases["failed"] == counts["failed"] == 0) or
            (code == 1 and reward == 0 and cases["failed"] > 0))


def validate_audit_limits(args):
    if not args.allow_paid:
        raise SystemExit("No model calls sent. --allow-paid is required for this bounded audit.")
    if not math.isfinite(args.max_total_usd) or not 0 < args.max_total_usd <= 2:
        raise SystemExit("The authorized paired-audit ceiling is greater than $0 and at most $2.")
    if not 1 <= args.max_wall_clock_min <= 45:
        raise SystemExit("The paired-audit wall-clock limit must be between 1 and 45 minutes.")


def aggregate_source(jobs, task_names, *, repetitions=2):
    """Each planned trial counts once; every physical attempt's cost is retained."""
    samples = []
    validity = []
    for job in jobs:
        if any(a.get("task_name") not in task_names or a.get("evaluation_phase") != "main" or
               a.get("repetition") != 1 for a in job["attempts"]):
            raise ValueError("Unexpected task or phase in the fixed paired audit")
        for task in task_names:
            attempts = [a for a in job["attempts"] if a["task_name"] == task]
            if not attempts:
                continue
            ordered = sorted(attempts, key=lambda a: a["attempt"])
            if len({a["attempt"] for a in ordered}) != len(ordered):
                raise ValueError("Duplicate immutable attempt identity")
            chosen = next((a for a in ordered if a["status"] != "error"), ordered[-1])
            row = {**chosen, "physical_attempts": len(attempts)}
            valid = verifier_valid(chosen)
            validity.append(valid)
            if not valid:
                # Preserve the raw attempt, but make its aggregate unscorable.
                # Never skip an unverified first outcome to select a later pass.
                row.update(status="error", reward=None)
            for key in ("duration_s", "model_cost_usd", "sandbox_cost_usd", "cost_usd"):
                values = [a.get(key) for a in attempts]
                row[key] = sum(values) if all(v is not None for v in values) else None
            for key in ("input_tokens", "output_tokens"):
                row[key] = sum(a.get(key) or 0 for a in attempts)
            row["cost_known"] = all(a.get("cost_known", False) for a in attempts)
            row["model_cost_known"] = all((a.get("metadata") or {}).get(
                "model_cost_known", a.get("cost_known", False)) for a in attempts)
            samples.append(row)
    result = evaluation_metrics(samples, len(task_names) * repetitions)
    result["verifier_complete"] = (len(jobs) == repetitions and len(validity) == len(task_names) * repetitions
                                   and all(validity))
    if not result["verifier_complete"]:
        result.update(complete=False, mean_reward=None)
    return result


def audit_options(original):
    # Declared once before dispatch, identical for both frozen sources. Recovery
    # attempts retain their full time/cost in both workflow and audit metrics.
    return {**original, "max_infrastructure_attempts": 2}


async def ledger(dsn):
    connection = await asyncpg.connect(dsn)
    try:
        row = await connection.fetchrow("""SELECT coalesce(sum(actual_usd),0) AS spent,
            coalesce(sum(reserved_usd) FILTER(WHERE settled_at IS NULL),0) AS reserved
            FROM spend_reservations""")
        return row["spent"], row["reserved"]
    finally:
        await connection.close()


async def cancel_owned_submissions(connection, api_key, keys):
    """Resolve a possibly committed POST by its persisted, caller-scoped key."""
    caller = await connection.fetchrow("""SELECT k.org_id, o.name AS org_name,
        u.id AS user_id, u.email, u.role FROM api_keys k
        JOIN users u ON u.id=k.user_id JOIN organizations o ON o.id=k.org_id
        WHERE k.key_hash=$1 AND k.revoked_at IS NULL""", hash_api_key(api_key))
    if caller is None:
        raise RuntimeError("Cannot authenticate database cleanup fallback")
    principal = Principal(**dict(caller))
    rows = await connection.fetch("""SELECT id,idempotency_key FROM jobs
        WHERE org_id=$1 AND created_by=$2 AND idempotency_key=ANY($3::text[])""",
        principal.org_id, principal.user_id, keys)
    results = []
    for row in rows:
        await repo.request_cancel(connection, principal, row["id"])
        job = await repo.get_job(connection, principal, row["id"])
        iterations = await repo.list_iterations(connection, row["id"])
        attempts = await repo.list_attempts(connection, iterations[0]["id"]) if iterations else []
        results.append({"job_id":str(row["id"]), "idempotency_key":row["idempotency_key"],
            "status":job["status"], "cancel_requested":job["cancel_requested"],
            "stop_reason":job["stop_reason"], "spent_usd":float(job["spent_usd"]),
            "reserved_usd":float(job["reserved_usd"]),
            "metrics":iterations[0].get("metrics") if iterations else None,
            "attempts":[redact_attempt(dict(a)) for a in attempts]})
    return results


async def stop_process(process, timeout=30):
    if process.poll() is None:
        process.terminate()
    try:
        await asyncio.to_thread(process.wait, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        await asyncio.to_thread(process.wait)


async def run(args):
    validate_audit_limits(args)
    proof = verify_source_proof(args.source_proof)
    credential = settings.neosigma_open_api_key
    if credential is None or not credential.get_secret_value():
        raise SystemExit("NEOSIGMA_OPEN_API_KEY is required; personal-key fallback is forbidden.")
    if settings.neosigma_base_url != "https://api.openai.com/v1":
        raise SystemExit("Unexpected inference endpoint")
    control = json.loads(args.prior_control.read_text())
    dsn = control["database_url"]
    await migrate(dsn)
    connection = await asyncpg.connect(dsn)
    try:
        if await connection.fetchval("SELECT count(*) FROM jobs WHERE status IN ('RUNNING','QUEUED')"):
            raise RuntimeError("Active evidence jobs exist; audit cannot overlap their spending")
        if await connection.fetchval("SELECT count(*) FROM sandbox_slots"):
            raise RuntimeError("Sandbox reservations remain; reconcile them before starting")
    finally:
        await connection.close()
    initial_spent, initial_reserved = await ledger(dsn)
    if initial_reserved:
        raise RuntimeError("Unsettled prior requests exist; reconcile before a new audit")
    if initial_spent + DIAGNOSTIC_USD + Decimal(str(args.max_total_usd)) > SESSION_AUTHORIZATION_USD:
        raise RuntimeError("Audit ceiling exceeds cumulative session authorization")

    name = "aos_paired_selection_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + uuid4().hex[:8]
    directory = ROOT / "runs" / name
    directory.mkdir(mode=0o700, parents=True)
    report_path = args.report_path
    previous_report_archive = archive_report(report_path, directory)
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
    processes, handles = [], []
    def start(command, label):
        handle = (directory / (label + ".log")).open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append(process)
        handles.append(handle)
        return process
    api = start([sys.executable, "-m", "uvicorn", "service.api.app:app", "--host", "127.0.0.1", "--port", str(port)], "api")
    worker = start([sys.executable, "-m", "service.worker.main"], "worker")
    base_url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer " + control["api_key"]}
    private = {**control, "api_url": base_url, "api_pid": api.pid, "worker_pid": worker.pid,
               "jobs": [], "submissions": []}
    def save_private():
        path = directory / "control.json"
        path.write_text(json.dumps(private, indent=2))
        path.chmod(0o600)
    save_private()
    report = {"status": "starting", "mode": "external_confirmatory_audit_via_benchmark_only_api_jobs",
        "started_at": datetime.now(timezone.utc).isoformat(), "run_directory": str(directory.relative_to(ROOT)),
        "source_job_id": control["job_id"], "source_candidate_index": args.candidate_index,
        "previous_report_archive": previous_report_archive,
        "expected_source_hashes": {"baseline": args.expected_baseline_sha256, "candidate": args.expected_candidate_sha256},
        "source_fixture_proof": str(args.source_proof), "implementation_sha256": proof,
        "audit_runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "credential_source": "NEOSIGMA_OPEN_API_KEY", "provider_seed_enforced": False,
        "max_total_usd": args.max_total_usd, "previous_spent_usd": float(initial_spent + DIAGNOSTIC_USD),
        "jobs": [], "holdout_executed": False, "old_experiment_decisions_modified": False,
        "limitations": ["Two repetitions of the source job's fixed selection tasks do not establish statistical significance or generalization.",
            "Model dollars are uncached token-price estimates; local Docker dollars remain unpriced.",
            "Warm amd64 images run on an aarch64 Docker daemon; timings include emulation.",
            "Alternating source order reduces simple order effects but does not eliminate provider variance.",
            "This independent audit does not retroactively promote a candidate or change the source experiment's recorded decisions."]}
    active_job = None
    deadline = time.monotonic() + args.max_wall_clock_min * 60
    async def save():
        spent, reserved = await ledger(dsn)
        report.update(updated_at=datetime.now(timezone.utc).isoformat(),
            audit_spent_usd=float(spent - initial_spent), outstanding_reserved_usd=float(reserved),
            cumulative_spent_usd=float(spent + DIAGNOSTIC_USD),
            remaining_audit_budget_usd=float(remaining_budget(initial_spent, spent, reserved, args.max_total_usd)))
        report_path.write_text(json.dumps(report, indent=2, default=str))
    async def request_json(client, path):
        response = await client.get(path, headers=headers)
        response.raise_for_status()
        return response.json()
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
            report.update(source_job_status=prior["status"], source_job_stop_reason=prior.get("stop_reason"),
                source_job_final_evaluation_status=(prior.get("final_evaluation") or {}).get("status"))
            if prior["status"] not in TERMINAL:
                raise RuntimeError("Source experiment must be terminal")
            cfg = prior["config"]
            options = audit_options(cfg["options"])
            if cfg["benchmark_revision"] != settings.benchmark_revision:
                raise RuntimeError("Benchmark revision differs from the source experiment")
            tasks = selection_task_names(cfg)
            sources = {}
            for label, index in (("baseline", 0), ("candidate", args.candidate_index)):
                detail = await request_json(client, f"/v1/jobs/{control['job_id']}/iterations/{index}")
                sources[label] = detail["agent_source"]
                if sha256_text(sources[label]) != detail["agent_sha256"]:
                    raise RuntimeError("Stored source hash mismatch")
                if label == "candidate":
                    report.update(source_candidate_status=detail["status"], source_candidate_decision=detail.get("decision"),
                                  source_candidate_original_score=detail.get("score"))
            source_hashes = pinned_source_hashes(sources, args.expected_baseline_sha256, args.expected_candidate_sha256)
            report.update(status="RUNNING", tasks=tasks, selection_task_count=len(tasks),
                planned_trials_per_source=len(tasks) * 2, benchmark_revision=cfg["benchmark_revision"],
                agent_model=cfg["agent_model"], options=options,
                harness_overrides={"max_infrastructure_attempts":{"prior":cfg["options"].get("max_infrastructure_attempts"),
                    "audit":2,"applies_to":"both sources","accounting":"all retry time and model cost retained"}},
                effective_inference_limits={"max_output_tokens":min(options["max_tokens_per_call"], settings.agent_max_output_tokens),
                    "max_request_bytes":settings.max_inference_body_bytes,
                    "max_model_calls_per_trial":settings.max_model_calls_per_trial},
                source_hashes=source_hashes,
                fixed_reasoning_policy={"tool_bearing_agent_requests": "none", "optimizer_calls": 0},
                planned_order=[{"repetition": rep, "source": label} for rep, label in
                               ((1,"baseline"),(1,"candidate"),(2,"candidate"),(2,"baseline"))])
            await save()
            for rep, label in ((1,"baseline"),(1,"candidate"),(2,"candidate"),(2,"baseline")):
                if verify_source_proof(args.source_proof) != proof:
                    raise RuntimeError("Fixture source proof changed during the audit")
                spent, reserved = await ledger(dsn)
                remaining = remaining_budget(initial_spent, spent, reserved, args.max_total_usd)
                remaining_seconds = deadline - time.monotonic()
                if remaining <= 0 or remaining_seconds <= 0:
                    raise RuntimeError("Audit budget/deadline exhausted before the next paired job")
                body = {"benchmark": cfg["benchmark"], "tasks": tasks, "max_iterations": 0,
                    "agent_source": sources[label], "agent_model": cfg["agent_model"],
                    "optimizer_model": cfg["optimizer_model"], "seed": (cfg.get("seed") or 1234) + rep - 1,
                    "options": options, "budget": {"max_llm_usd": float(remaining),
                        "max_wall_clock_min": max(1, math.ceil(remaining_seconds / 60))}}
                JobCreate(**body)
                submission_key = f"{name}-{rep}-{label}"
                # Persist the intended key BEFORE sending: a POST can commit and
                # then time out without ever returning its job identifier.
                private["submissions"].append({"key":submission_key,"source":label,"repetition":rep})
                save_private()
                entry = {"job_id":None, "idempotency_key":submission_key, "repetition":rep,
                    "source":label,"agent_sha256":sha256_text(sources[label]),"status":"SUBMITTING","attempts":[]}
                report["jobs"].append(entry)
                await save()
                response = await client.post("/v1/jobs", headers={**headers,
                    "Idempotency-Key": submission_key}, json=body)
                response.raise_for_status()
                active_job = response.json()["job_id"]
                private["jobs"].append({"repetition": rep, "source": label, "job_id": active_job})
                save_private()
                entry.update(job_id=active_job,status="QUEUED")
                print(json.dumps({"event": "submitted", **{k:v for k,v in entry.items() if k != "attempts"},
                                  "remaining_cap_usd": float(remaining)}), flush=True)
                last = None
                while True:
                    if verify_source_proof(args.source_proof) != proof:
                        raise RuntimeError("Fixture source proof changed during the audit")
                    job = await request_json(client, f"/v1/jobs/{active_job}")
                    iterations = await request_json(client, f"/v1/jobs/{active_job}/iterations")
                    attempts = []
                    if iterations:
                        raw = await request_json(client, f"/v1/jobs/{active_job}/iterations/0/attempts?include_trace=false&limit=100")
                        attempts = [redact_attempt(item) for item in raw]
                    entry.update(status=job["status"], stop_reason=job["stop_reason"], spent_usd=job["spent_usd"],
                                 reserved_usd=job["reserved_usd"], attempts=attempts,
                                 metrics=aggregate_source([{"attempts": attempts}], tasks, repetitions=1),
                                 workflow_metrics=iterations[0]["metrics"] if iterations else None)
                    await save()
                    state = (job["status"], len(attempts), job["spent_usd"], job["reserved_usd"])
                    if state != last:
                        print(json.dumps({"event": "progress", "source": label,"repetition":rep,
                            "status":state[0],"tasks_done":state[1],"spent_usd":state[2],"reserved_usd":state[3]}),flush=True)
                        last = state
                    if job["status"] in TERMINAL:
                        active_job = None
                        if (job["status"] != "SUCCEEDED" or {a["task_name"] for a in attempts} != set(tasks)
                                or not entry["metrics"]["complete"] or not entry["metrics"]["verifier_complete"]):
                            raise RuntimeError("Incomplete paired job; preserve costs and stop before new dispatch")
                        break
                    if worker.poll() is not None:
                        raise RuntimeError("Worker exited during audit")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Overall audit deadline reached")
                    await asyncio.sleep(10)
            metrics = {label: aggregate_source([j for j in report["jobs"] if j["source"] == label], tasks)
                       for label in ("baseline", "candidate")}
            accepted, reason = promotion_decision(metrics["candidate"], metrics["baseline"], options)
            report.update(status="COMPLETE", metrics=metrics,
                observed_policy_decision={"candidate_qualifies":accepted,"reason":reason,
                    "scope":"external fresh paired selection audit; not an old-job database promotion"})
            await save()
            print(json.dumps({"event":"audit_complete","decision":report["observed_policy_decision"],"report":str(report_path)}),flush=True)
    except BaseException as exc:
        report.update(status="INCOMPLETE", error_type=type(exc).__name__)
        if active_job:
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
                    response = await client.post(f"/v1/jobs/{active_job}/cancel", headers=headers)
                    response.raise_for_status()
            except Exception as cleanup_error:
                report["api_cancel_error_type"] = type(cleanup_error).__name__
        # Drain/stop the owned API before resolving an ambiguous POST; otherwise
        # its transaction could commit just after a premature "no job" query.
        await stop_process(api, timeout=10)
        # Reconcile exact keys even when active_job is unknown or the API failed.
        # An unavailable DB cannot be treated as successful durable cancellation.
        report["durable_cleanup_confirmed"] = not private["submissions"]
        for cleanup_attempt in range(3):
            connection = None
            try:
                connection = await asyncpg.connect(dsn, timeout=5, command_timeout=5)
                await _init_conn(connection)
                for _ in range(40):
                    reconciled = await cancel_owned_submissions(connection, control["api_key"],
                        [item["key"] for item in private["submissions"]])
                    for row in reconciled:
                        entry = next(j for j in report["jobs"] if j["idempotency_key"] == row["idempotency_key"])
                        entry.update(row)
                    report["durable_cleanup_confirmed"] = all(
                        row["status"] in TERMINAL or row["cancel_requested"] for row in reconciled)
                    if all(row["status"] in TERMINAL for row in reconciled):
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
        # Always leave a durable local explanation, even if accounting is down.
        report_path.write_text(json.dumps(report, indent=2, default=str))
        raise
    finally:
        for process in processes:
            await stop_process(process)
        for handle in handles:
            handle.close()
        print(json.dumps({"event":"processes_stopped","report":str(report_path)}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--prior-control", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--report-path", type=Path, default=ROOT / "docs" / "live_paired_selection_review.json")
    parser.add_argument("--max-total-usd", type=float, default=2)
    parser.add_argument("--max-wall-clock-min", type=int, default=45)
    asyncio.run(run(parser.parse_args()))
