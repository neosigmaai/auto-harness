#!/usr/bin/env python3
"""
Exercise the auto-harness benchmark API end-to-end.

Usage:
  # Terminal 1: start the API
  uvicorn api.main:app --reload --port 8000

  # Terminal 2: run this client
  python test_client.py
  python test_client.py --base-url http://127.0.0.1:8000 --task-ids fix-git regex-log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


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
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"error": {"code": "http_error", "message": raw or str(exc)}}
        return exc.code, payload


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
        print("Failed to submit run:", json.dumps(payload, indent=2))
        sys.exit(1)
    return payload


def poll_run(
    base_url: str,
    run_id: str,
    *,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    terminal = {"completed", "failed", "cancelled"}
    last: dict[str, Any] = {}

    while time.monotonic() < deadline:
        status, payload = _request("GET", f"{base_url}/v1/runs/{run_id}")
        if status != 200:
            print("Failed to poll run:", json.dumps(payload, indent=2))
            sys.exit(1)
        last = payload
        run_status = payload.get("status")
        print(f"  status={run_status} summary={payload.get('summary')}")
        if run_status in terminal:
            return payload
        time.sleep(interval)

    print("Timed out waiting for run to finish. Last payload:")
    print(json.dumps(last, indent=2))
    sys.exit(1)


def print_summary(run: dict[str, Any]) -> None:
    print("\n=== Run summary ===")
    print(f"run_id:  {run.get('run_id')}")
    print(f"status:  {run.get('status')}")
    print(f"request: {json.dumps(run.get('request'), indent=2)}")
    print(f"summary: {json.dumps(run.get('summary'), indent=2)}")
    print("\nTasks:")
    for task in run.get("tasks") or []:
        reward = task.get("reward")
        remarks = task.get("remarks")
        line = f"  - {task.get('task_id')}: {task.get('status')} (reward={reward})"
        if remarks:
            line += f" — {remarks}"
        print(line)

    failures = run.get("failure_summary") or []
    print("\nFailure summary:")
    if not failures:
        print("  (none)")
    else:
        for item in failures:
            print(
                f"  - {item.get('task_id')} [{item.get('category')}]: {item.get('message')}"
            )

    if run.get("error"):
        print("\nRun error:")
        print(json.dumps(run["error"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="auto-harness API test client")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Optional task IDs (default: server configured subset)",
    )
    parser.add_argument("--agent-model", default=None, help="Optional agent model override")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="Seconds between polls")
    parser.add_argument("--timeout", type=float, default=60.0, help="Max seconds to wait")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print("1) Submitting benchmark run…")
    created = submit_run(base, task_ids=args.task_ids, agent_model=args.agent_model)
    run_id = created["run_id"]
    print(f"   accepted run_id={run_id} status={created.get('status')}")

    print("2) Polling for result…")
    run = poll_run(
        base,
        run_id,
        interval=args.poll_interval,
        timeout=args.timeout,
    )

    print("3) Structured summary")
    print_summary(run)


if __name__ == "__main__":
    main()
