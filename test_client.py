#!/usr/bin/env python3
"""Exercise the Agent Optimization Service end to end: submit, poll, print the history.

    python test_client.py --key ao_...            # mock mode, seconds, no spend
    python test_client.py --key ao_... --real     # real benchmark run

The key comes from `python service/seed.py`, which prints it once.

Stdlib only, deliberately: this is the script a reviewer runs first, and it should not
need a virtualenv or a pip install to work.

Progress goes through logging (a real run takes 20+ minutes, so knowing *when* each poll
happened matters); the final report stays on stdout as plain text, because timestamping
every line of a formatted table makes it harder to read, not easier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

TASK_SETS = {
    "demo": ["fix-git", "prove-plus-comm", "crack-7z-hash", "raman-fitting"],

    "headroom": ["git-leak-recovery", "kv-store-grpc", "largest-eigenval",
                 "pytorch-model-recovery", "db-wal-recovery", "polyglot-c-py",
                 "raman-fitting", "fix-git", "regex-log"],

    "fragile": ["nginx-request-logging", "cobol-modernization", "openssl-selfsigned-cert",
                "extract-elf", "fix-git", "regex-log"],
    "all": [
        "fix-git", "prove-plus-comm", "cobol-modernization", "overfull-hbox",
        "crack-7z-hash", "raman-fitting", "kv-store-grpc", "pytorch-model-recovery",
        "nginx-request-logging", "polyglot-c-py", "openssl-selfsigned-cert",
        "hf-model-inference", "multi-source-data-merger", "extract-elf",
        "git-leak-recovery", "sanitize-git-repo", "chess-best-move", "regex-log",
        "db-wal-recovery", "largest-eigenval", "configure-git-webserver",
    ],
}

logging.basicConfig(                      
    format="%(levelname)-5s | %(asctime)s | %(name)-9s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
log = logging.getLogger("client")


def call(url: str, key: str, path: str, body: dict | None = None,
         headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(
        url + path, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        log.error("%s -> HTTP %s: %s", path, e.code, e.read().decode()[:300])
        sys.exit(1)
    except urllib.error.URLError as e:
        log.error("cannot reach %s: %s — is the API running?", url, e.reason)
        sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.getenv("AOS_URL", "http://localhost:8000"))
    p.add_argument("--key", default=os.getenv("AOS_API_KEY"))
    p.add_argument("--real", action="store_true", help="run the real benchmark, not mock")
    p.add_argument("--task-set", choices=sorted(TASK_SETS),
                   help="named group of tasks; omit to run the server's whole subset")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    if not args.key:
        log.error("need --key or AOS_API_KEY (from: python service/seed.py)")
        return 1

    url, key = args.url.rstrip("/"), args.key

    body = {"mode": "real" if args.real else "mock",
            "max_iterations": args.max_iterations}
    if args.task_set:
        body["task_ids"] = TASK_SETS[args.task_set]
    r = call(url, key, "/jobs", body,
             {"Idempotency-Key": f"test-client-{time.time()}"})
    job_id = r["job_id"]
    log.info("submitted %s  status=%s set=%s mode=%s", job_id, r["status"],
             args.task_set or "default", body["mode"])

    deadline = time.time() + args.timeout
    seen = -1
    while time.time() < deadline:
        job = call(url, key, f"/jobs/{job_id}")
        if job["iterations_completed"] != seen:
            seen = job["iterations_completed"]
            p = job.get("progress") or {}
            log.info("%-10s done=%s in_flight=%s elapsed=%ss%s", job["status"], seen,
                     p.get("current_iteration"), p.get("elapsed_s"),
                     f"  {job['queue_reason']}" if job["queue_reason"] else "")
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(2)
    else:
        log.error("timed out after %ss waiting for the job", args.timeout)
        return 1

    print(f"\n=== job {job_id} ===")
    print(f"status            {job['status']}")
    if job["failure_reason"]:
        print(f"failure           {job['failure_reason']}: {job['error_detail']}")
    fo = job["final_outcome"]
    print(f"stopped_because   {fo['stopped_because']}")
    print(f"baseline          {fmt(fo['baseline_score'])}")
    print(f"best visible      {fmt(fo['best_visible_score'])}   ")
    print(f"holdout           {fmt(fo['holdout_score'])}   ")
    print(f"improved          {fo['improved']}")
    u = job["usage"]
    print(f"usage             {u['llm_calls']} llm calls, "
          f"{u['input_tokens']}+{u['output_tokens']} tokens, "
          f"{u['sandboxes_used']} sandboxes, {u['sandbox_seconds']}s")
    print(f"held out          {job['holdout_task_ids']}")

    print("\n=== iteration history ===")
    for it in call(url, key, f"/jobs/{job_id}/iterations"):
        passed = sorted(t for t, r in it["results"].items() if r and r >= 1.0)
        print(f"\n[{it['n']}] {it['outcome']:<12} accepted={str(it['accepted']):<5} "
              f"visible={fmt(it['visible_score'])} holdout={fmt(it['holdout_score'])}")
        if it["proposal"]:
            print(f"     proposal: {it['proposal']}")
        if it["error_detail"]:
            print(f"     error:    {it['error_detail']}")
        print(f"     passed:   {passed or '-'}")
        for f in it["failures"][:2]:
            print(f"     failed:   {f['task_id']}: {f['verifier_output'][:60]}")
        print(f"     agent:    {len(it['agent_source'])} chars persisted")
    return 0


def fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.2f}"


if __name__ == "__main__":
    sys.exit(main())
