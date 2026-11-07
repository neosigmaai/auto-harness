#!/usr/bin/env python3
"""End-to-end client for the Agent Optimization Service.

Submits a benchmark/optimisation job, polls until it finishes, and prints a structured summary
including the full iteration history (Milestone 4) and the diff of the best agent vs baseline.

Usage:
    python test_client.py --api-key aos_... [--base-url http://localhost:8000] [--max-iterations 5]
    AOS_API_KEY=aos_... python test_client.py --tasks fix-git regex-log --max-iterations 2

Only the standard library + httpx are required (httpx is in service/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import httpx

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def submit(client: httpx.Client, args) -> dict:
    body = {
        "max_iterations": args.max_iterations,
        "patience": args.patience,
        "options": {"two_phase_eval": not args.no_two_phase, "confirm_flips": not args.no_confirm_flips},
    }
    if args.tasks:
        body["tasks"] = args.tasks
    if args.seed is not None:
        body["seed"] = args.seed
    headers = _hdr(args.api_key) | {"Idempotency-Key": args.idempotency_key or str(uuid.uuid4())}
    r = client.post("/v1/jobs", json=body, headers=headers)
    if r.status_code not in (200, 202):
        print(f"submit failed: HTTP {r.status_code}\n{r.text}", file=sys.stderr)
        sys.exit(1)
    return r.json()


def poll(client: httpx.Client, key: str, job_id: str, timeout_s: float, quiet: bool) -> dict:
    t0, delay, last_line = time.time(), 1.0, None
    while True:
        r = client.get(f"/v1/jobs/{job_id}", headers=_hdr(key))
        r.raise_for_status()
        job = r.json()
        p = job.get("progress") or {}
        line = (f"[{int(time.time()-t0):4d}s] {job['status']:9s} iter={p.get('iteration','-')} "
                f"phase={p.get('phase','-'):9s} tasks={p.get('tasks_done','-')}/{p.get('tasks_total','-')} "
                f"best={job.get('best_score')}")
        if not quiet and line != last_line:
            print(line); last_line = line
        if job["status"] in TERMINAL:
            return job
        if time.time() - t0 > timeout_s:
            print(f"timed out after {timeout_s}s waiting for job {job_id}", file=sys.stderr)
            return job
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)


def fmt_score(s) -> str:
    return "-" if s is None else f"{s:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("AOS_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("AOS_API_KEY"))
    ap.add_argument("--tasks", nargs="*", help="task names (default: curated subset)")
    ap.add_argument("--max-iterations", type=int, default=int(os.environ.get("AOS_MAX_ITERATIONS", "5")))
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-two-phase", action="store_true")
    ap.add_argument("--no-confirm-flips", action="store_true")
    ap.add_argument("--idempotency-key", default=None)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("AOS_TIMEOUT", "7200")))
    ap.add_argument("--job-id", help="attach to an existing job instead of submitting")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the final summary as JSON too")
    args = ap.parse_args()
    if not args.api_key:
        ap.error("--api-key or AOS_API_KEY is required")

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        me = client.get("/v1/me", headers=_hdr(args.api_key)); me.raise_for_status(); me = me.json()
        print(f"authenticated as {me['email']} ({me['role']}) in org '{me['org_name']}' @ {args.base_url}")

        if args.job_id:
            job_id = args.job_id
        else:
            acc = submit(client, args)
            job_id = acc["job_id"]
            print(f"submitted job {job_id} -> status {acc['status']} ({acc['status_url']})")

        job = poll(client, args.api_key, job_id, args.timeout, args.quiet)

        its = client.get(f"/v1/jobs/{job_id}/iterations", headers=_hdr(args.api_key)); its.raise_for_status(); its = its.json()
        agent = client.get(f"/v1/jobs/{job_id}/agent", headers=_hdr(args.api_key))
        agent = agent.json() if agent.status_code == 200 else None

        print("\n" + "=" * 78)
        print(f"job {job_id}: {job['status']}  stop_reason={job.get('stop_reason')}  error={job.get('error')}")
        print(f"tasks: {len(job['config']['tasks'])}   baseline={fmt_score(job.get('baseline_score'))}   "
              f"best={fmt_score(job.get('best_score'))} (iteration {job.get('best_iteration_index')})")
        print("-" * 78)
        print(f"{'it':>3} {'decision':9} {'score':>6} {'pass':>4} {'fail':>4} {'err':>3}  hypothesis / reason")
        for it in its:
            s = it.get("summary") or {}
            print(f"{it['index']:>3} {str(it.get('decision')):9} {fmt_score(it.get('score')):>6} "
                  f"{s.get('passed','-'):>4} {s.get('failed','-'):>4} {s.get('errored','-'):>3}  "
                  f"{it.get('hypothesis') or '(baseline)'}")
            if it.get("decision_reason"):
                print(f"{'':30}{it['decision_reason']}")

        # Per-task matrix across iterations
        if its:
            print("-" * 78)
            tasks = job["config"]["tasks"]
            detail = {}
            for it in its:
                d = client.get(f"/v1/jobs/{job_id}/iterations/{it['index']}", headers=_hdr(args.api_key))
                if d.status_code == 200:
                    detail[it["index"]] = {t["task_name"]: t for t in d.json()["tasks"]}
            idxs = sorted(detail)
            print(f"{'task':32} " + " ".join(f"{i:>3}" for i in idxs))
            sym = {"passed": "  ✓", "failed": "  ✗", "timeout": "  T", "error": "  E", "pending": "  .", "running": "  ~"}
            for t in tasks:
                print(f"{t:32} " + " ".join(sym.get(detail[i].get(t, {}).get("status"), "  -") for i in idxs))
            last = its[-1]
            d = client.get(f"/v1/jobs/{job_id}/iterations/{job.get('best_iteration_index', last['index'])}", headers=_hdr(args.api_key))
            if d.status_code == 200 and d.json()["failure_summary"]:
                print("-" * 78)
                print("remaining failures in best iteration:")
                for f in d.json()["failure_summary"]:
                    print(f"  - {f['task_name']:30} [{f.get('failure_class')}] {f.get('summary') or ''}")

        if agent and agent.get("diff_vs_baseline"):
            print("-" * 78)
            print("diff baseline -> best agent.py:")
            print(agent["diff_vs_baseline"])
        print("=" * 78)

        if args.json:
            print(json.dumps({"job": job, "iterations": its}, indent=2, default=str))
        sys.exit(0 if job["status"] == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
