#!/usr/bin/env python3
"""
Exercise the auto-harness benchmark API end-to-end.

Two modes:

  job  (default) — submit an OPTIMIZATION JOB (Milestone 4): the service evaluates
                   the agent, asks an LLM to improve its prompt/config, re-evaluates,
                   and repeats until the score plateaus or a cap is hit. Prints the
                   full iteration history and the winning agent spec.
  run            — submit a single benchmark RUN (Milestone 2/3), no optimization.

Usage:
  # Terminal 1: start the API
  uvicorn api.main:app --port 8000

  # Terminal 2: start a worker (executes queued work)
  python -m worker.main -v

  # Terminal 3: this client
  python test_client.py                                   # optimization job, configured tasks
  python test_client.py --task-ids fix-git regex-log --max-iterations 3
  python test_client.py --mode run --task-ids fix-git     # single run, no loop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

TERMINAL = {"completed", "failed", "cancelled"}


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return exc.code, {"error": {"code": "non_json_response", "message": raw}}
    except urllib.error.URLError as exc:
        print(f"Cannot reach {url}: {exc.reason}")
        print("Is the API running?  uvicorn api.main:app --port 8000")
        sys.exit(1)


def _fail(what: str, payload: dict[str, Any]) -> None:
    print(f"Failed to {what}:", json.dumps(payload, indent=2))
    sys.exit(1)


# ── single run (Milestone 2/3) ─────────────────────────────────────────────


def submit_run(
    base_url: str,
    *,
    task_ids: list[str] | None,
    agent_model: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if task_ids is not None:
        body["task_ids"] = task_ids
    if agent_model is not None:
        body["agent_model"] = agent_model

    status, payload = _request("POST", f"{base_url}/v1/runs", body=body)
    if status != 202:
        _fail("submit run", payload)
    return payload


def poll_run(base_url: str, run_id: str, *, interval: float, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, payload = _request("GET", f"{base_url}/v1/runs/{run_id}")
        if status != 200:
            _fail("poll run", payload)
        last = payload
        print(f"  status={payload.get('status')} summary={payload.get('summary')}")
        if payload.get("status") in TERMINAL:
            return payload
        time.sleep(interval)

    print("Timed out waiting for run to finish. Last payload:")
    print(json.dumps(last, indent=2))
    sys.exit(1)


def print_run_summary(run: dict[str, Any]) -> None:
    print("\n=== Run summary ===")
    print(f"run_id:  {run.get('run_id')}")
    print(f"status:  {run.get('status')}")
    print(f"request: {json.dumps(run.get('request'), indent=2)}")
    print(f"summary: {json.dumps(run.get('summary'), indent=2)}")
    print("\nTasks:")
    for task in run.get("tasks") or []:
        line = f"  - {task.get('task_id')}: {task.get('status')} (reward={task.get('reward')})"
        if task.get("remarks"):
            line += f" — {task['remarks']}"
        print(line)

    failures = run.get("failure_summary") or []
    print("\nFailure summary:")
    if not failures:
        print("  (none)")
    for item in failures:
        print(f"  - {item.get('task_id')} [{item.get('category')}]: {item.get('message')}")

    if run.get("error"):
        print("\nRun error:")
        print(json.dumps(run["error"], indent=2))


# ── optimization job (Milestone 4) ────────────────────────────────────────


def submit_job(
    base_url: str,
    *,
    task_ids: list[str] | None,
    agent_model: str | None,
    improver_model: str | None,
    max_iterations: int | None,
    patience: int | None,
    min_iterations: int | None,
    min_delta: float | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in (
        ("task_ids", task_ids),
        ("agent_model", agent_model),
        ("improver_model", improver_model),
        ("max_iterations", max_iterations),
        ("patience", patience),
        ("min_iterations", min_iterations),
        ("min_delta", min_delta),
    ):
        if value is not None:
            body[key] = value

    status, payload = _request("POST", f"{base_url}/v1/jobs", body=body)
    if status != 202:
        _fail("submit job", payload)
    return payload


def poll_job(base_url: str, job_id: str, *, interval: float, timeout: float) -> dict[str, Any]:
    """Poll until terminal, reporting each iteration's score as it lands."""
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    seen: set[tuple[int, Any]] = set()

    while time.monotonic() < deadline:
        status, payload = _request("GET", f"{base_url}/v1/jobs/{job_id}")
        if status != 200:
            _fail("poll job", payload)
        last = payload

        for it in payload.get("iterations") or []:
            key = (it.get("iteration"), it.get("score"))
            if key in seen:
                continue
            seen.add(key)
            score = it.get("score")
            state = "running" if score is None else f"score={score}"
            print(f"  iteration {it.get('iteration')} (v{it.get('version')}): {state}")

        if payload.get("status") in TERMINAL:
            return payload
        time.sleep(interval)

    print("Timed out waiting for job to finish. Last payload:")
    print(json.dumps(last, indent=2))
    sys.exit(1)


def fetch_best(base_url: str, job_id: str) -> dict[str, Any] | None:
    status, payload = _request("GET", f"{base_url}/v1/jobs/{job_id}/best")
    if status == 200:
        return payload
    if status == 409:  # no_evaluation_yet — nothing scored, nothing to show
        return None
    _fail("fetch best agent", payload)
    return None


def print_job_summary(job: dict[str, Any], best: dict[str, Any] | None) -> None:
    print("\n=== Job summary ===")
    print(f"job_id:      {job.get('job_id')}")
    print(f"status:      {job.get('status')}")
    print(f"stop_reason: {job.get('stop_reason')}")
    print(f"iterations:  {job.get('current_iteration')}")
    print(f"config:      {json.dumps(job.get('config'), indent=2)}")

    best_ref = job.get("best")
    if best_ref:
        print(f"\nBest agent:  version {best_ref.get('version')} "
              f"(score={best_ref.get('score')}, id={best_ref.get('agent_version_id')})")

    print("\n=== Iteration history ===")
    for it in job.get("iterations") or []:
        summary = it.get("summary") or {}
        counts = (
            f"{summary.get('passed', '?')}/{summary.get('total', '?')} passed"
            if summary else "not evaluated yet"
        )
        print(f"\niteration {it.get('iteration')}  (agent version {it.get('version')})")
        print(f"  score:     {it.get('score')}   improved: {it.get('improved')}")
        print(f"  tasks:     {counts}")
        print(f"  run_id:    {it.get('run_id')}")
        if it.get("fixed_tasks") or it.get("regressed_tasks"):
            print(f"  fixed:     {it.get('fixed_tasks')}")
            print(f"  regressed: {it.get('regressed_tasks')}")
        proposal = it.get("proposal")
        if proposal:
            print(f"  proposed change (based on version {proposal.get('based_on_version')}):")
            print(f"    changed: {proposal.get('changed_fields')}")
            print(f"    why:     {proposal.get('rationale')}")
        else:
            print("  proposal:  (baseline — no change proposed)")

    if best:
        print("\n=== Winning agent spec ===")
        print(f"version {best.get('version')} (score={best.get('score')})")
        if best.get("rationale"):
            print(f"rationale: {best['rationale']}")
        print(json.dumps(best.get("spec"), indent=2))

    if job.get("error"):
        print("\nJob error:")
        print(json.dumps(job["error"], indent=2))


# ── entrypoint ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="auto-harness API test client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="API base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--mode", choices=("job", "run"), default="job",
                        help="job = optimization loop (Milestone 4); run = single benchmark run")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Task IDs (default: the server's configured subset)")
    parser.add_argument("--agent-model", default=None, help="Agent model override")
    parser.add_argument("--improver-model", default=None, help="Improver model override (job mode)")
    parser.add_argument("--max-iterations", type=int, default=None, help="Job iteration cap")
    parser.add_argument("--patience", type=int, default=None,
                        help="Non-improving iterations tolerated before stopping")
    parser.add_argument("--min-iterations", type=int, default=None,
                        help="Floor before no_improvement may stop the loop")
    parser.add_argument("--min-delta", type=float, default=None,
                        help="Score gain required to count as an improvement")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    parser.add_argument("--timeout", type=float, default=7200.0,
                        help="Max seconds to wait (jobs run benchmark containers; default 2h)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    if args.mode == "run":
        print("1) Submitting benchmark run…")
        created = submit_run(base, task_ids=args.task_ids, agent_model=args.agent_model)
        run_id = created["run_id"]
        print(f"   accepted run_id={run_id} status={created.get('status')}")

        print("2) Polling for result…")
        run = poll_run(base, run_id, interval=args.poll_interval, timeout=args.timeout)

        print("3) Structured summary")
        print_run_summary(run)
        return

    print("1) Submitting optimization job…")
    created = submit_job(
        base,
        task_ids=args.task_ids,
        agent_model=args.agent_model,
        improver_model=args.improver_model,
        max_iterations=args.max_iterations,
        patience=args.patience,
        min_iterations=args.min_iterations,
        min_delta=args.min_delta,
    )
    job_id = created["job_id"]
    print(f"   accepted job_id={job_id} status={created.get('status')}")

    print("2) Polling for result (each iteration is a full benchmark run)…")
    job = poll_job(base, job_id, interval=args.poll_interval, timeout=args.timeout)

    print("3) Structured summary with full iteration history")
    print_job_summary(job, fetch_best(base, job_id))


if __name__ == "__main__":
    main()
