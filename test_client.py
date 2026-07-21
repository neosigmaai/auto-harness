#!/usr/bin/env python3
"""End-to-end client for the Agent Optimization Service.

Run this from the auto-harness checkout against a running optimization service.
When a job completes with an improved agent, this script overwrites local
``agent/agent.py`` only (no git commit or push).

Requires: ``pip install httpx`` (or ``uv pip install httpx``).

Example::

    export AOS_BASE_URL=http://localhost:8000
    python test_client.py --executor harbor --tasks-file tasks.txt --max-iterations 5
    # or: python test_client.py --executor harbor --task-ids regex-log extract-elf
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = os.environ.get("AOS_BASE_URL", "http://localhost:8000")
DEFAULT_ORG_NAME = os.environ.get("AOS_ORG", "default")
DEFAULT_USER_EMAIL = os.environ.get("AOS_EMAIL", "admin@example.com")
DEFAULT_USER_PASSWORD = os.environ.get("AOS_PASSWORD", "assignment-password")
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}

REPO_ROOT = Path(__file__).resolve().parent
AGENT_PATH = REPO_ROOT / "agent" / "agent.py"


def load_task_ids(path: Path) -> list[str]:
    """Load task IDs from a text file (one per line) or a JSON string/array file.

    Blank lines and ``#`` comments are ignored for the line-oriented format.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"Tasks file is empty: {path}")

    if raw.startswith("[") or raw.startswith("{"):
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("task_ids") or data.get("tasks") or data.get("train")
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError(
                f"JSON tasks file must be a string array or an object with "
                f"task_ids/tasks/train: {path}"
            )
        task_ids = [tid.strip() for tid in data if tid.strip()]
    else:
        task_ids = []
        for line in raw.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                task_ids.append(line)

    if not task_ids:
        raise ValueError(f"No task IDs found in: {path}")
    return task_ids


@dataclass
class JobCleanup:
    base_url: str
    job_id: str
    headers: dict[str, str]
    armed: bool = True

    def disarm(self) -> None:
        self.armed = False

    def cancel(self) -> None:
        """Best-effort cleanup used when the client exits before the job does."""
        if not self.armed:
            return
        self.armed = False
        try:
            with httpx.Client(base_url=self.base_url, timeout=15.0) as client:
                response = client.post(f"/jobs/{self.job_id}/cancel", headers=self.headers)
                if response.status_code == 409:
                    print(f"\nJob {self.job_id} already reached a terminal state.")
                    return
                response.raise_for_status()
                print(f"\nCancelled job {self.job_id} because the client stopped tracking it.")
        except Exception as exc:
            print(f"\nWarning: could not cancel job {self.job_id}: {exc}")


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def login(client: httpx.Client, org_name: str, email: str, password: str) -> str:
    resp = client.post(
        "/auth/login",
        json={"org_name": org_name, "email": email, "password": password},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def _tasks_summary(it: dict[str, Any]) -> dict[str, list[str]]:
    summary = it.get("tasks_summary") or {}
    return {
        "pending": list(summary.get("pending") or []),
        "running": list(summary.get("running") or []),
        "completed": list(summary.get("completed") or []),
        "passed": list(summary.get("passed") or []),
        "failed": list(summary.get("failed") or []),
        "infra_error": list(summary.get("infra_error") or []),
    }


def _benchmark_snapshot(it: dict[str, Any]) -> tuple[Any, ...]:
    ts = _tasks_summary(it)
    return (
        it.get("val_score"),
        tuple(ts["passed"]),
        tuple(ts["failed"]),
        tuple(ts["infra_error"]),
        it.get("accepted"),
    )


def _proposal_snapshot(it: dict[str, Any]) -> tuple[Any, ...]:
    return (
        it.get("proposed_agent_version_no"),
        it.get("improvement_rationale"),
    )


def _progress_snapshot(it: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ts = _tasks_summary(it)
    return tuple(ts["pending"]), tuple(ts["running"]), tuple(ts["completed"])


def _format_task_counts(it: dict[str, Any]) -> str:
    ts = _tasks_summary(it)
    passed = len(ts["passed"])
    failed = len(ts["failed"])
    infra = len(ts["infra_error"])
    total = passed + failed + infra
    return f"{passed}/{total} passed, {failed} failed, {infra} infra_error"


def _format_progress(pending: int, running: int, completed: int) -> str:
    total = pending + running + completed
    return f"tasks: {pending} pending, {running} running, {completed} completed ({completed}/{total})"


def _print_benchmark_progress(it: dict[str, Any], *, first: bool) -> None:
    iteration_no = it["iteration_no"]
    agent_v = it.get("agent_version_no")
    pending, running, completed = _progress_snapshot(it)
    progress = _format_progress(len(pending), len(running), len(completed))
    if first:
        print(f"  [iter {iteration_no}] running benchmark with agent_v={agent_v} | {progress}")
    else:
        print(f"  [iter {iteration_no}] {progress}")


def _fmt_score(score: Any) -> str:
    return f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"


def _print_benchmark_done(it: dict[str, Any], *, best_val_score: float | None) -> None:
    iteration_no = it["iteration_no"]
    agent_v = it.get("agent_version_no")
    val_score = it.get("val_score")
    counts = _format_task_counts(it)
    failed_ids = _tasks_summary(it)["failed"] + _tasks_summary(it)["infra_error"]
    failed_note = f" ({', '.join(failed_ids)})" if failed_ids else ""
    print(
        f"  [iter {iteration_no}] benchmark done: agent_v={agent_v} "
        f"val_score={_fmt_score(val_score)} | {counts}{failed_note}"
    )

    if it.get("accepted") is True and iteration_no > 0:
        print(
            f"  [iter {iteration_no}] accepted — "
            f"new best_val_score={_fmt_score(best_val_score)} (agent_v={agent_v})"
        )
    elif it.get("accepted") is False:
        print(
            f"  [iter {iteration_no}] rejected — val_score={_fmt_score(val_score)} "
            f"(best remains {_fmt_score(best_val_score)})"
        )


def write_best_agent_locally(
    client: httpx.Client,
    job: dict[str, Any],
    headers: dict[str, str],
) -> None:
    """Overwrite local agent/agent.py with the job's best improved agent (no git)."""
    job_id = job["id"]
    version_no = job.get("best_agent_version_no")
    if version_no is None:
        print("\nNo best_agent_version_no on job — skipping agent.py update.")
        return
    if int(version_no) <= 0:
        print(
            f"\nBest agent is still baseline (agent_v={version_no}) — "
            "leaving local agent/agent.py unchanged."
        )
        return

    resp = client.get(f"/jobs/{job_id}/agent-versions/{version_no}", headers=headers)
    resp.raise_for_status()
    content = resp.json().get("content")
    if not isinstance(content, str) or not content.strip():
        print(f"\nAgent version {version_no} has empty/invalid content — skipping update.")
        return

    AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = AGENT_PATH.read_text(encoding="utf-8") if AGENT_PATH.exists() else None
    if previous == content:
        print(f"\nagent/agent.py already matches best agent_v={version_no}.")
        return

    tmp_path = AGENT_PATH.with_suffix(AGENT_PATH.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(AGENT_PATH)
    print(
        f"\nWrote best agent_v={version_no} "
        f"(best_val_score={_fmt_score(job.get('best_val_score'))}) → "
        f"{AGENT_PATH.relative_to(REPO_ROOT)} (local only; not committed)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Optimization Service test client")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--org", default=DEFAULT_ORG_NAME)
    parser.add_argument("--email", default=DEFAULT_USER_EMAIL)
    parser.add_argument("--password", default=DEFAULT_USER_PASSWORD)
    parser.add_argument("--executor", choices=["simulated", "harbor"], default="harbor")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help=(
            "File of task IDs to submit (one per line, or a JSON string array). "
            "Required unless --task-ids is provided."
        ),
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Task IDs to submit (overrides --tasks-file when both are set)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not overwrite local agent/agent.py after the job finishes",
    )
    args = parser.parse_args()

    if args.task_ids:
        task_ids = [tid.strip() for tid in args.task_ids if tid.strip()]
        if not task_ids:
            parser.error("--task-ids was provided but contained no task IDs")
    elif args.tasks_file is not None:
        try:
            task_ids = load_task_ids(args.tasks_file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    else:
        parser.error("Provide task IDs via --tasks-file PATH or --task-ids ID [ID ...]")

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("Health:", health.json())

        access_token = login(client, args.org, args.email, args.password)
        headers = {"Authorization": f"Bearer {access_token}"}
        print(f"Logged in as {args.email} @ {args.org}")
        print(f"Tasks ({len(task_ids)}): {', '.join(task_ids)}")

        payload: dict[str, Any] = {
            "max_iterations": args.max_iterations,
            "patience": args.patience,
            "executor": args.executor,
            "task_ids": task_ids,
            "config": {},
        }

        created = client.post("/jobs", json=payload, headers=headers)
        created.raise_for_status()
        job = created.json()
        job_id = job["id"]
        print(f"Submitted job {job_id} (status={job['status']})")
        cleanup = JobCleanup(args.base_url, job_id, headers)
        atexit.register(cleanup.cancel)

        seen_benchmarks: dict[int, tuple[Any, ...]] = {}
        seen_proposals: dict[int, tuple[Any, ...]] = {}
        seen_progress: dict[
            int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
        ] = {}
        seen_running: set[int] = set()
        last_status: str | None = None

        while True:
            resp = client.get(f"/jobs/{job_id}", headers=headers)
            resp.raise_for_status()
            job = resp.json()
            status = job["status"]
            best_val_score = job.get("best_val_score")

            if status != last_status:
                score_note = (
                    f"best_val_score={best_val_score:.3f}" if best_val_score is not None else "best_val_score=None"
                )
                print(f"  job status={status} {score_note}")
                last_status = status

            for it in job.get("iterations", []):
                iteration_no = it["iteration_no"]
                phase = it.get("phase")

                if phase in {"running_benchmark", "pending"}:
                    progress = _progress_snapshot(it)
                    first = iteration_no not in seen_running
                    if first or seen_progress.get(iteration_no) != progress:
                        _print_benchmark_progress(it, first=first)
                        seen_running.add(iteration_no)
                        seen_progress[iteration_no] = progress

                bench_key = _benchmark_snapshot(it)
                if it.get("val_score") is not None and seen_benchmarks.get(iteration_no) != bench_key:
                    _print_benchmark_done(it, best_val_score=best_val_score)
                    seen_benchmarks[iteration_no] = bench_key
                    seen_progress[iteration_no] = _progress_snapshot(it)

                proposal_key = _proposal_snapshot(it)
                if (
                    it.get("proposed_agent_version_no") is not None
                    and it.get("improvement_rationale")
                    and seen_proposals.get(iteration_no) != proposal_key
                ):
                    rationale = it["improvement_rationale"].strip()
                    if len(rationale) > 160:
                        rationale = rationale[:157] + "..."
                    print(
                        f"  [iter {iteration_no}] proposed agent_v="
                        f"{it['proposed_agent_version_no']} for next run: {rationale}"
                    )
                    seen_proposals[iteration_no] = proposal_key

            if status in _TERMINAL_JOB_STATUSES:
                cleanup.disarm()
                break
            time.sleep(args.poll_interval)

        print("\n=== Job summary ===")
        print(
            json.dumps(
                {
                    "id": job["id"],
                    "status": job["status"],
                    "stop_reason": job.get("stop_reason"),
                    "best_val_score": job.get("best_val_score"),
                    "best_agent_version_no": job.get("best_agent_version_no"),
                    "task_ids": job.get("task_ids"),
                },
                indent=2,
            )
        )

        print("\n=== Latest task results ===")
        for tr in job.get("latest_task_results", []):
            print(f"  {tr['task_id']}: {tr['status']} reward={tr.get('reward')}")

        iterations = job.get("iterations", [])
        print(f"\n=== Iteration history ({len(iterations)} iterations) ===")
        for it in iterations:
            accepted = it.get("accepted")
            mark = "✓" if accepted else ("✗" if accepted is False else "-")
            counts = _format_task_counts(it)
            print(
                f"  [{mark}] iter={it['iteration_no']} "
                f"phase={it['phase']} val_score={it.get('val_score')} "
                f"agent_v={it['agent_version_no']} | {counts}"
            )
            if _tasks_summary(it)["failed"] or _tasks_summary(it)["infra_error"]:
                failed = _tasks_summary(it)["failed"] + _tasks_summary(it)["infra_error"]
                print(f"       failed: {', '.join(failed)}")
            if it.get("improvement_rationale"):
                print(f"       changes: {it['improvement_rationale'][:160]}")

        if job.get("status") == "completed" and not args.no_promote:
            write_best_agent_locally(client, job, headers)
        elif args.no_promote:
            print("\nSkipping agent.py update (--no-promote).")
        else:
            print(f"\nJob status={job.get('status')} — skipping agent.py update.")

    return 0


def cli() -> int:
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        return main()
    except KeyboardInterrupt:
        print("\nInterrupted; cancelling the active job before exit.")
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(cli())
