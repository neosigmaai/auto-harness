from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from test_client import (  # noqa: E402
    DEFAULT_MODE,
    DEMO_BATCH_TASKS,
    build_headers,
    build_run_request,
    build_summary,
    fetch_run_artifacts,
    poll_run,
    print_status_update,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a run survives backend restart and resumes from the durable queue."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--mode", choices=[DEFAULT_MODE], default=DEFAULT_MODE)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--requested-concurrency", type=int, default=1)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    task_ids = args.task_ids or DEMO_BATCH_TASKS
    base_url = f"http://{args.host}:{args.port}"

    first_backend = start_backend(
        host=args.host,
        port=args.port,
        background_enabled=False,
    )
    try:
        wait_for_health(base_url)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            run_id = submit_run(
                client,
                task_ids=task_ids,
                mode=args.mode,
                max_iterations=args.max_iterations,
                requested_concurrency=args.requested_concurrency,
            )
            print(f"\nsubmitted run_id={run_id}")
            print("\nfirst backend is running with AUTOHARNESS_START_BACKGROUND=0")
            print_status_update(fetch_status(client, run_id))
    finally:
        stop_backend(first_backend)

    print("\nfirst backend stopped; restarting with background worker enabled")
    second_backend = start_backend(
        host=args.host,
        port=args.port,
        background_enabled=True,
    )
    try:
        wait_for_health(base_url)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            print("\nrun is still readable after restart")
            print_status_update(fetch_status(client, run_id))

            final_status = poll_run(
                client,
                run_id,
                poll_interval_sec=args.poll_interval_sec,
                timeout_sec=args.timeout_sec,
                on_status=print_status_update,
            )
            results, iterations = fetch_run_artifacts(client, run_id)
            summary = build_summary(
                status=final_status,
                results=results,
                iterations=iterations,
            )

            print("\n=== FINAL SUMMARY ===")
            print(json.dumps(summary, indent=2, sort_keys=True))
            print_artifact_locations(results)
    finally:
        stop_backend(second_backend)


def start_backend(
    *,
    host: str,
    port: int,
    background_enabled: bool,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["AUTOHARNESS_START_BACKGROUND"] = "1" if background_enabled else "0"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "autoharness_service.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_health(base_url: str, timeout_sec: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(base_url=base_url, timeout=2.0) as client:
                response = client.get("/health")
                if response.status_code == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    raise TimeoutError(f"backend did not become healthy at {base_url}: {last_error}")


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


def fetch_status(client: httpx.Client, run_id: str) -> dict[str, Any]:
    response = client.get(f"/runs/{run_id}", headers=build_headers("viewer"))
    response.raise_for_status()
    return response.json()


def print_artifact_locations(results: dict[str, Any]) -> None:
    print("\n=== ARTIFACT LOCATIONS ===")
    for task in results.get("task_results", []):
        metadata = task.get("metadata", {}) or {}
        print(
            json.dumps(
                {
                    "task_id": task.get("task_id"),
                    "status": task.get("status"),
                    "trace_path": task.get("trace_path"),
                    "result_path": task.get("result_path"),
                    "artifacts": metadata.get("artifacts", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
