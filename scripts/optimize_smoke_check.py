from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TASK_ID = "break-filter-js-from-html"
TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
GUIDED_ORDER = (
    ("1", "Submit", "POST /runs returns a run_id immediately"),
    ("2", "Baseline", "Harbor/Daytona runs the requested Terminal-Bench tasks"),
    ("3", "Collect", "service persists task results, traces, logs, and artifacts"),
    ("4", "Optimize", "LLM proposes one AGENT_INSTRUCTION patch from failures"),
    ("5", "Rerun", "service applies the patch and reruns the same task ids"),
    ("6", "Decide", "service accepts improved patches or reverts rejected patches"),
    ("7", "FinalSummary", "client fetches final results and iteration history"),
)


def _headers(role: str) -> dict[str, str]:
    return {
        "X-Org-Id": "demo-org",
        "X-User-Id": "demo-user",
        "X-Role": role,
    }


def submit_run(
    client: httpx.Client,
    *,
    task_ids: list[str],
    requested_concurrency: int,
) -> str:
    response = client.post(
        "/runs",
        json={
            "task_ids": task_ids,
            "mode": "real",
            "sandbox_provider": "daytona",
            "requested_concurrency": requested_concurrency,
            "max_iterations": 1,
        },
        headers=_headers("runner"),
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


def poll_until_done(
    client: httpx.Client,
    run_id: str,
    *,
    poll_interval_sec: float,
    timeout_sec: int,
    quiet_poll: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        response = client.get(f"/runs/{run_id}", headers=_headers("viewer"))
        response.raise_for_status()
        status = response.json()
        iterations = fetch_iterations_for_poll(client, run_id)
        if not quiet_poll:
            print(
                format_step_status_line(
                    status,
                    iterations,
                    elapsed_sec=time.monotonic() - started,
                ),
                flush=True,
            )
        if status["status"] in TERMINAL_STATUSES:
            return status
        if time.monotonic() - started >= timeout_sec:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_sec}s")
        time.sleep(poll_interval_sec)


def format_guided_order(task_ids: list[str]) -> str:
    order = " -> ".join(
        f"{number} {name}" for number, name, _description in GUIDED_ORDER
    )
    details = "\n".join(
        f"[guided-order]   {number}. {name}: {description}"
        for number, name, description in GUIDED_ORDER
    )
    return (
        f"[guided-order] tasks={','.join(task_ids)}\n"
        f"[guided-order] order={order}\n"
        f"{details}"
    )


def format_step_status_line(
    status: dict[str, Any],
    iterations: dict[str, Any],
    *,
    elapsed_sec: float,
) -> str:
    progress = status.get("progress") or {}
    task_results = status.get("task_results") or []
    task_statuses = ",".join(
        f"{item.get('task_id')}={item.get('status')}" for item in task_results
    )
    iteration_statuses = _format_iteration_statuses(iterations)
    optimize = _format_optimization_status(iterations)
    return (
        f"[{elapsed_sec:6.1f}s] phase={derive_phase(status, iterations)} "
        f"run={status.get('status')} "
        f"progress={progress.get('completed')}/{progress.get('total')} "
        f"running={progress.get('running')} queued={progress.get('queued')} "
        f"score={status.get('score')} tasks=[{task_statuses}] "
        f"iterations=[{iteration_statuses}] {optimize}"
    )


def fetch_json(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(path, headers=_headers("viewer"))
    response.raise_for_status()
    return dict(response.json())


def fetch_iterations_for_poll(client: httpx.Client, run_id: str) -> dict[str, Any]:
    try:
        return fetch_json(client, f"/runs/{run_id}/iterations")
    except httpx.HTTPError as exc:
        return {"iterations": [], "error": str(exc)}


def derive_phase(status: dict[str, Any], iterations: dict[str, Any]) -> str:
    run_status = status.get("status")
    if run_status == "queued":
        return "1_submit_queued"

    iteration_by_index = {
        item.get("iteration"): item
        for item in iterations.get("iterations", []) or []
        if isinstance(item, dict)
    }
    initial = iteration_by_index.get(0) or {}
    optimized = iteration_by_index.get(1) or {}
    optimized_status = optimized.get("status")

    if run_status in TERMINAL_STATUSES:
        return "7_final_summary"
    if optimized_status in {"patch_applied", "rerun_running"}:
        return "5_rerun_after_patch"
    if optimized_status in {"patch_rejected", "completed", "proposal_failed"}:
        return "6_decide"
    if optimized_status:
        return "4_optimize"
    if initial.get("status") == "completed":
        return "3_collect_results"
    if run_status == "running":
        return "2_baseline_running"
    return "unknown"


def _format_iteration_statuses(iterations: dict[str, Any]) -> str:
    items = iterations.get("iterations", []) or []
    if not items:
        error = iterations.get("error")
        return f"unavailable:{error}" if error else "none"
    return ",".join(
        f"{item.get('iteration')}:{item.get('status')}"
        for item in items
        if isinstance(item, dict)
    )


def _format_optimization_status(iterations: dict[str, Any]) -> str:
    items = iterations.get("iterations", []) or []
    optimized = next(
        (
            item
            for item in items
            if isinstance(item, dict) and item.get("iteration") == 1
        ),
        None,
    )
    if not optimized:
        return "optimize=not_started accepted=None"
    return (
        f"optimize={optimized.get('status')} "
        f"accepted={optimized.get('accepted')} "
        f"opt_score={optimized.get('score')}"
    )


def parse_proposal(proposal_text: str | None) -> dict[str, Any]:
    if not proposal_text:
        return {}
    try:
        parsed = json.loads(proposal_text)
    except json.JSONDecodeError:
        return {"raw": proposal_text[:500]}
    return parsed if isinstance(parsed, dict) else {"raw": proposal_text[:500]}


def build_attempt_timelines(iterations: dict[str, Any]) -> list[dict[str, Any]]:
    optimized = next(
        (
            item
            for item in iterations.get("iterations", []) or []
            if isinstance(item, dict) and item.get("iteration") == 1
        ),
        None,
    )
    if not optimized:
        return []

    proposal = parse_proposal(optimized.get("proposal"))
    timelines: list[dict[str, Any]] = []
    for attempt_name, task_key in (
        ("baseline", "baseline_tasks"),
        ("proposal-1", "rerun_tasks"),
    ):
        for task in proposal.get(task_key, []) or []:
            if not isinstance(task, dict):
                continue
            timeline = _build_task_attempt_timeline(attempt_name, task)
            if timeline:
                timelines.append(timeline)
    return timelines


def format_attempt_timeline(timelines: list[dict[str, Any]]) -> str:
    if not timelines:
        return "[attempt-timeline] no Harbor/Daytona attempt artifacts found"
    lines = []
    for item in timelines:
        lines.append(
            "[attempt-timeline] "
            f"{item.get('attempt')} task={item.get('task_id')} "
            f"reward={item.get('reward')} status={item.get('status')} "
            f"harbor_job_started_at={item.get('harbor_job_started_at')} "
            f"harbor_job_finished_at={item.get('harbor_job_finished_at')} "
            f"daytona_strategy={item.get('daytona_strategy')} "
            "daytona_sandbox_create_signal="
            f"{item.get('daytona_sandbox_create_signal')} "
            f"daytona_sandbox_finished_at={item.get('daytona_sandbox_finished_at')} "
            f"job_log={item.get('job_log_path')}"
        )
    return "\n".join(lines)


def _build_task_attempt_timeline(
    attempt_name: str,
    task: dict[str, Any],
) -> dict[str, Any] | None:
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    result_path = _first_path(
        task.get("result_path"),
        artifacts.get("trial_result"),
    )
    if result_path is None:
        return None

    job_dir = result_path.parent.parent
    job_log_path = _first_path(artifacts.get("job_log"), job_dir / "job.log")
    job_result_path = _first_path(artifacts.get("job_result"), job_dir / "result.json")
    trial_result = _read_json_file(result_path)
    job_result = _read_json_file(job_result_path) if job_result_path else {}
    job_log_text = _read_text_file(job_log_path) if job_log_path else ""

    return {
        "attempt": attempt_name,
        "task_id": task.get("task_id") or trial_result.get("task_name"),
        "status": task.get("status"),
        "reward": task.get("reward"),
        "harbor_job_started_at": job_result.get("started_at")
        or trial_result.get("started_at"),
        "harbor_job_finished_at": job_result.get("finished_at")
        or trial_result.get("finished_at"),
        "daytona_strategy": _log_value(job_log_text, "Selected strategy:"),
        "daytona_sandbox_create_signal": _log_line(
            job_log_text,
            "Creating new AsyncDaytona client",
        ),
        "daytona_prebuilt_image": _log_value(job_log_text, "Using prebuilt image:"),
        "daytona_sandbox_started_at": trial_result.get("started_at"),
        "daytona_sandbox_finished_at": trial_result.get("finished_at")
        or job_result.get("finished_at"),
        "result_path": str(result_path),
        "job_log_path": str(job_log_path) if job_log_path else None,
    }


def _first_path(*values: Any) -> Path | None:
    for value in values:
        if value is None:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _log_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return None


def _log_value(text: str, prefix: str) -> str | None:
    line = _log_line(text, prefix)
    if line is None:
        return None
    return line.removeprefix(prefix).strip()


def summarize(
    *,
    run_id: str,
    elapsed_sec: float,
    status: dict[str, Any],
    results: dict[str, Any],
    iterations: dict[str, Any],
) -> dict[str, Any]:
    iteration_items = iterations.get("iterations", []) or []
    optimized = next(
        (item for item in iteration_items if item.get("iteration") == 1),
        None,
    )
    optimized_proposal = parse_proposal(
        optimized.get("proposal") if isinstance(optimized, dict) else None
    )
    attempt_timelines = build_attempt_timelines(iterations)
    failed_tasks = [
        {
            "task_id": item.get("task_id"),
            "status": item.get("status"),
            "reward": item.get("reward"),
            "failure_type": item.get("failure_type"),
            "error_summary": item.get("error_summary"),
        }
        for item in results.get("task_results", []) or []
        if item.get("status") != "passed"
    ]
    return {
        "run_id": run_id,
        "elapsed_sec": round(elapsed_sec, 1),
        "run_status": status.get("status"),
        "score": results.get("score"),
        "tasks": {
            "total": results.get("tasks_total"),
            "passed": results.get("tasks_passed"),
            "failed": results.get("tasks_failed"),
            "infra_failed": results.get("tasks_infra_failed"),
        },
        "failed_tasks": failed_tasks,
        "iterations": [
            {
                "iteration": item.get("iteration"),
                "status": item.get("status"),
                "score": item.get("score"),
                "accepted": item.get("accepted"),
            }
            for item in iteration_items
        ],
        "optimization": (
            {
                "status": optimized.get("status"),
                "accepted": optimized.get("accepted"),
                "baseline_score": optimized_proposal.get("baseline_score"),
                "rerun_score": optimized_proposal.get("rerun_score"),
                "decision_reason": optimized_proposal.get("decision_reason"),
                "changed_section": optimized_proposal.get("changed_section"),
                "reverted": optimized_proposal.get("reverted"),
                "discarded_snapshot_paths": optimized_proposal.get(
                    "discarded_snapshot_paths"
                ),
            }
            if isinstance(optimized, dict)
            else None
        ),
        "attempt_timelines": attempt_timelines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real optimize smoke check and print a concise summary."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--requested-concurrency", type=int, default=1)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--quiet-poll",
        action="store_true",
        help="Suppress compact polling status lines and only print the final summary.",
    )
    args = parser.parse_args()

    task_ids = args.task_ids or [DEFAULT_TASK_ID]
    started = time.monotonic()
    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(format_guided_order(task_ids), flush=True)
        run_id = submit_run(
            client,
            task_ids=task_ids,
            requested_concurrency=args.requested_concurrency,
        )
        print(
            f"[step 1/7 Submit] submitted run_id={run_id} "
            f"task_ids={','.join(task_ids)} mode=real provider=daytona",
            flush=True,
        )
        status = poll_until_done(
            client,
            run_id,
            poll_interval_sec=args.poll_interval_sec,
            timeout_sec=args.timeout_sec,
            quiet_poll=args.quiet_poll,
        )
        results = fetch_json(client, f"/runs/{run_id}/results")
        iterations = fetch_json(client, f"/runs/{run_id}/iterations")
        print(
            format_attempt_timeline(build_attempt_timelines(iterations)),
            flush=True,
        )
        print("[step 7/7 FinalSummary] fetched results and iterations", flush=True)

    summary = summarize(
        run_id=run_id,
        elapsed_sec=time.monotonic() - started,
        status=status,
        results=results,
        iterations=iterations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
