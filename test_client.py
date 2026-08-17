#!/usr/bin/env python3
"""End-to-end client for the Auto-Harness optimization service.

Submits a benchmark job, polls until it finishes, and prints a structured
summary — including the full iteration history when the job runs in optimize
mode (Milestone 4).

Usage:
    python test_client.py                        # simulated, single run, "core" subset
    python test_client.py --mode optimize --max-iterations 5
    python test_client.py --executor harbor --subset smoke
    python test_client.py --base-url http://localhost:8000 --api-key dev-key

Only depends on `requests`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "dev-key"
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class HarnessClient:
    """Thin, stateless HTTP wrapper — one method per endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": api_key})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict[str, Any]:
        r = self._session.get(self._url("/health"), timeout=10)
        r.raise_for_status()
        return r.json()

    def submit_job(self, **body: Any) -> dict[str, Any]:
        r = self._session.post(self._url("/v1/jobs"), json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        r = self._session.get(self._url(f"/v1/jobs/{job_id}"), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_iterations(self, job_id: str) -> list[dict[str, Any]]:
        r = self._session.get(self._url(f"/v1/jobs/{job_id}/iterations"), timeout=30)
        r.raise_for_status()
        return r.json()


@dataclass
class JobRun:
    """Stateful view of ONE submitted job."""

    client: HarnessClient
    job_id: str | None = None
    status: str = "queued"
    job: dict[str, Any] = field(default_factory=dict)
    iterations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_val_score(self) -> float | None:
        return self.job.get("best_val_score")

    def submit(self, **body: Any) -> "JobRun":
        created = self.client.submit_job(**body)
        self.job_id = created["id"]
        self.status = created["status"]
        self.job = created
        print(f"→ submitted job {self.job_id} (status={self.status}, "
              f"mode={created['mode']}, executor={created['executor']}, "
              f"tasks={len(created['subset'])})")
        return self

    def poll_until_done(self, interval: float = 3.0, timeout: float = 1800.0) -> "JobRun":
        assert self.job_id, "submit() first"
        deadline = time.time() + timeout
        while True:
            self.job = self.client.get_job(self.job_id)
            self.status = self.job["status"]
            print(f"  … status={self.status} best_val={self.job.get('best_val_score')}")
            if self.status in TERMINAL_STATES:
                break
            if time.time() > deadline:
                raise TimeoutError(f"job {self.job_id} not done after {timeout}s")
            time.sleep(interval)
        self.iterations = self.client.get_iterations(self.job_id)
        return self

    # ── presentation ──
    def print_summary(self) -> None:
        print("\n" + "=" * 68)
        print(f"JOB {self.job_id}  —  {self.status.upper()}")
        print("=" * 68)
        if self.status == "failed":
            print(f"error: {self.job.get('error')}")
            return

        j = self.job
        print(f"mode={j['mode']}  executor={j['executor']}  "
              f"iterations={j['n_iterations']}  best_val={self.best_val_score}")

        # Optimize headline: baseline → best (train) and held-out test.
        if j["mode"] == "optimize":
            base, best = j.get("baseline_val_score"), j.get("best_val_score")
            imp = j.get("improvement")
            print(f"\nOPTIMIZATION  train: baseline={base} → best={best}"
                  + (f"  (Δ={imp:+.3f})" if imp is not None else ""))
            print(f"  train tasks: {len(j.get('train_subset') or [])}   "
                  f"held-out test: {len(j.get('test_subset') or [])}  "
                  f"test_val_score={j.get('test_val_score')}")

        summary = j.get("summary") or {}
        if summary:
            scope = "train" if j["mode"] == "optimize" else "subset"
            print(f"\nBest iteration ({scope}): val_score={summary['val_score']:.3f}  "
                  f"passed={summary['n_passed']}  failed={summary['n_failed']}")
            if summary.get("failures"):
                print("Remaining failures:")
                for f in summary["failures"]:
                    print(f"  ✗ {f['task_id']}: {(f['failure_reason'] or '')[:120]}")

        # Full iteration history.
        print("\nIteration history:")
        for it in self.iterations:
            total = it["n_passed"] + it["n_failed"]
            print(f"  [{it['idx']}] {it['decision']:<9} val={it['val_score']:.3f}  "
                  f"pass={it['n_passed']}/{total}  ({it.get('decision_reason','')[:60]})")
            if it.get("proposal_rationale"):
                print(f"        └ proposal ({it.get('proposer')}): {it['proposal_rationale'][:150]}")
        print("=" * 68)


def main() -> int:
    p = argparse.ArgumentParser(description="Auto-Harness service test client")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--mode", choices=["single_run", "optimize"], default="optimize")
    p.add_argument("--executor", choices=["simulated", "harbor"], default=None)
    p.add_argument("--subset", default="core", help="subset name (core|smoke) or comma-separated task ids")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=1800.0)
    args = p.parse_args()

    client = HarnessClient(args.base_url, args.api_key)
    try:
        health = client.health()
        print(f"service healthy: {health}")
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach service at {args.base_url}: {exc}", file=sys.stderr)
        return 2

    subset: str | list[str] = (
        [s.strip() for s in args.subset.split(",")] if "," in args.subset else args.subset
    )
    body: dict[str, Any] = {"mode": args.mode, "subset": subset}
    if args.executor:
        body["executor"] = args.executor
    if args.max_iterations is not None:
        body["max_iterations"] = args.max_iterations
    if args.patience is not None:
        body["patience"] = args.patience

    run = JobRun(client).submit(**body)
    run.poll_until_done(interval=args.poll_interval, timeout=args.timeout)
    run.print_summary()
    return 0 if run.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
