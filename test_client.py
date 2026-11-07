#!/usr/bin/env python3
"""End-to-end client for the Agent Optimization Service.

Submits a benchmark/optimisation job, polls until it finishes, and prints a structured summary
including the full iteration history (Milestone 4) and the diff of the best agent vs baseline.

Usage:
    python test_client.py --api-key aos_... [--base-url http://localhost:8000] [--max-iterations 5]
    AOS_API_KEY=aos_... python test_client.py --profile stress16 --max-iterations 2
    AOS_API_KEY=aos_... python test_client.py --tasks fix-git regex-log --max-iterations 0

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
        "budget": {"max_llm_usd": args.max_llm_usd, "max_wall_clock_min": args.max_wall_clock_min},
    }
    if args.tasks:
        body["tasks"] = args.tasks
    else:
        body["profile"] = args.profile or "calibrated12"
    if args.seed is not None:
        body["seed"] = args.seed
    if args.agent_model is not None:
        body["agent_model"] = args.agent_model
    if args.optimizer_model is not None:
        body["optimizer_model"] = args.optimizer_model
    if args.prior_job_id:
        body["prior_job_id"] = args.prior_job_id
    if args.resume_candidate_index is not None:
        body["resume_candidate_index"] = args.resume_candidate_index
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


def fmt_measure(value, decimals=4) -> str:
    return "unknown" if value is None else f"{value:.{decimals}f}"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("AOS_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("AOS_API_KEY"))
    task_choice = ap.add_mutually_exclusive_group()
    task_choice.add_argument("--profile", choices=("calibrated12", "stress16"),
                             help="frozen task/split profile (default: calibrated12)")
    task_choice.add_argument("--tasks", nargs="+", help="custom task names, with a deterministic custom split")
    ap.add_argument("--max-iterations", type=int, default=int(os.environ.get("AOS_MAX_ITERATIONS", "5")))
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agent-model", help="execution model override; must have configured server-side token prices")
    ap.add_argument("--optimizer-model", help="proposal model override; must have configured server-side token prices")
    ap.add_argument("--max-llm-usd", type=float, default=2.0, help="per-job model spending ceiling (reservations enforced)")
    ap.add_argument("--max-wall-clock-min", type=int, default=60)
    ap.add_argument("--no-two-phase", action="store_true")
    ap.add_argument("--no-confirm-flips", action="store_true",
                    help="deprecated compatibility flag; nonoperative because candidate confirmation is mandatory")
    ap.add_argument("--idempotency-key", default=None)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("AOS_TIMEOUT", "7200")))
    ap.add_argument("--job-id", help="attach to an existing job instead of submitting")
    ap.add_argument("--prior-job-id", help="new compatible experiment with an authorized prior job's rejection memory")
    ap.add_argument("--resume-candidate-index", type=int, help="re-evaluate an interrupted prior candidate; requires --prior-job-id")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the final summary as JSON too")
    return ap


def main():
    ap = build_parser()
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
            metrics = it.get("metrics") or {}
            incumbent = metrics.get("incumbent")
            if incumbent:
                print(f"{'':6}fresh matched trials: candidate {metrics.get('passed')}/{metrics.get('planned_trials')} "
                      f"vs incumbent {incumbent.get('passed')}/{incumbent.get('planned_trials')}")
                print(f"{'':6}mean model cost/task: ${fmt_measure(metrics.get('mean_model_cost_usd'), 6)} "
                      f"vs ${fmt_measure(incumbent.get('mean_model_cost_usd'), 6)}; "
                      f"p95 attempt seconds: {fmt_measure(metrics.get('p95_task_duration_s'), 2)} "
                      f"vs {fmt_measure(incumbent.get('p95_task_duration_s'), 2)}")

        # Per-task matrix across iterations
        full_details = []
        if its:
            print("-" * 78)
            tasks = job["config"]["tasks"]
            detail = {}
            for it in its:
                d = client.get(f"/v1/jobs/{job_id}/iterations/{it['index']}", headers=_hdr(args.api_key))
                if d.status_code == 200:
                    payload = d.json()
                    if args.json:
                        attempts, offset = [], 0
                        while True:
                            a = client.get(f"/v1/jobs/{job_id}/iterations/{it['index']}/attempts",
                                           params={"include_trace": True, "limit": 100, "offset": offset}, headers=_hdr(args.api_key))
                            a.raise_for_status()
                            batch = a.json()
                            attempts.extend(batch)
                            if len(batch) < 100:
                                break
                            offset += len(batch)
                        payload["attempts"] = attempts
                    full_details.append(payload)
                    detail[it["index"]] = {t["task_name"]: t for t in payload["tasks"]}
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
        print(f"model usage estimate: ${job.get('spent_usd', 0):.6f}; "
              f"unresolved reservations: ${job.get('reserved_usd', 0):.6f}")
        if job.get("final_evaluation"):
            print("final evaluation (separate from selection scores):")
            print(json.dumps(job["final_evaluation"], indent=2, default=str))

        if args.json:
            print(json.dumps({"job": job, "iterations": full_details or its}, indent=2, default=str))
        sys.exit(0 if job["status"] == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
