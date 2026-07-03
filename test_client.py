from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any, Mapping

import httpx

DEMO_BATCH_TASKS = [
    "break-filter-js-from-html",
    "multi-source-data-merger",
]
DEFAULT_MODE = "real"
DEMO_HEADERS = {
    "X-Org-Id": "demo-org",
    "X-User-Id": "demo-user",
}
TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}


def build_headers(role: str) -> dict[str, str]:
    return {**DEMO_HEADERS, "X-Role": role}


def build_run_request(
    *,
    task_ids: list[str],
    mode: str,
    max_iterations: int,
    requested_concurrency: int,
) -> dict[str, Any]:
    if mode != DEFAULT_MODE:
        raise ValueError("manual test client only supports real mode")
    return {
        "task_ids": task_ids,
        "mode": mode,
        "max_iterations": max_iterations,
        "requested_concurrency": requested_concurrency,
        "sandbox_provider": "daytona",
    }


def build_summary(
    *,
    status: Mapping[str, Any],
    results: Mapping[str, Any],
    iterations: Mapping[str, Any],
) -> dict[str, Any]:
    task_results = results.get("task_results", []) or []
    iteration_items = list(iterations.get("iterations", []) or [])
    return {
        "run_id": status["run_id"],
        "status": status["status"],
        "score": results.get("score"),
        "tasks_passed": results.get("tasks_passed"),
        "tasks_failed": results.get("tasks_failed"),
        "tasks_infra_failed": results.get("tasks_infra_failed"),
        "failure_summary": results.get("failure_summary"),
        "failed_task_ids": [
            item["task_id"] for item in task_results if item.get("status") != "passed"
        ],
        "iteration_statuses": _iteration_statuses(iteration_items),
        "optimization_status": _optimization_status(iteration_items),
        "iterations": iteration_items,
    }


def _iteration_statuses(iterations: list[Mapping[str, Any]]) -> list[str]:
    return [
        f"{item.get('iteration')}:{item.get('status')}"
        for item in sorted(iterations, key=lambda item: item.get("iteration", -1))
    ]


def _optimization_status(iterations: list[Mapping[str, Any]]) -> str | None:
    for item in iterations:
        if item.get("iteration") != 1:
            continue
        if item.get("accepted") is True:
            return "accepted"
        if item.get("accepted") is False:
            return "rejected"
        return str(item.get("status"))
    return None


def submit_run(
    client: httpx.Client,
    *,
    task_ids: list[str],
    mode: str,
    max_iterations: int,
    requested_concurrency: int,
) -> str:
    response = client.post(
        "/runs",
        json=build_run_request(
            task_ids=task_ids,
            mode=mode,
            max_iterations=max_iterations,
            requested_concurrency=requested_concurrency,
        ),
        headers=build_headers("runner"),
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


def poll_run(
    client: httpx.Client,
    run_id: str,
    *,
    poll_interval_sec: float,
    timeout_sec: int,
    on_status: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        response = client.get(f"/runs/{run_id}", headers=build_headers("viewer"))
        response.raise_for_status()
        payload = response.json()
        if on_status is not None:
            on_status(payload)
        if payload["status"] in TERMINAL_STATUSES:
            return payload
        if time.monotonic() - started >= timeout_sec:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_sec}s")
        time.sleep(poll_interval_sec)


def fetch_run_artifacts(
    client: httpx.Client, run_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    headers = build_headers("viewer")
    results = client.get(f"/runs/{run_id}/results", headers=headers)
    results.raise_for_status()
    iterations = client.get(f"/runs/{run_id}/iterations", headers=headers)
    iterations.raise_for_status()
    return results.json(), iterations.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit and inspect an AutoHarness run"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument(
        "--demo-batch",
        action="store_true",
        help="Use the real demo Terminal-Bench tasks exposed by the service /tasks endpoint.",
    )
    parser.add_argument("--mode", choices=[DEFAULT_MODE], default=DEFAULT_MODE)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--requested-concurrency", type=int, default=1)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--print-status-updates",
        action="store_true",
        help="Print every GET /runs/{run_id} status payload while polling.",
    )
    args = parser.parse_args()

    task_ids = DEMO_BATCH_TASKS if args.demo_batch else args.task_ids
    if not task_ids:
        parser.error("provide at least one --task-id or pass --demo-batch")

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        run_id = submit_run(
            client,
            task_ids=task_ids,
            mode=args.mode,
            max_iterations=args.max_iterations,
            requested_concurrency=args.requested_concurrency,
        )
        print(f"submitted run_id={run_id}")
        status = poll_run(
            client,
            run_id,
            poll_interval_sec=args.poll_interval_sec,
            timeout_sec=args.timeout_sec,
            on_status=print_status_update if args.print_status_updates else None,
        )
        results, iterations = fetch_run_artifacts(client, run_id)
        summary = build_summary(status=status, results=results, iterations=iterations)
        print(json.dumps(summary, indent=2, sort_keys=True))


def print_status_update(status: Mapping[str, Any]) -> None:
    print("\n=== GET /runs/{run_id} ===".format(run_id=status["run_id"]))
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
