"""
Append one iteration result to workspace/results.tsv.

Called by the coding agent after a change passes the gate and is committed.

Usage:
    python record.py --val-score 0.82 --evals-passed 8 --evals-total 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone

RESULTS_FILE = "workspace/results.tsv"


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def next_iteration() -> int:
    if not os.path.exists(RESULTS_FILE):
        return 1
    with open(RESULTS_FILE) as f:
        data_rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    return len(data_rows)


def record(val_score: float, evals_passed: int, evals_total: int) -> None:
    iteration = next_iteration()
    commit = current_commit()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = f"{iteration}\t{val_score:.4f}\t{commit}\t{evals_passed}\t{evals_total}\t{ts}\n"

    with open(RESULTS_FILE, "a") as f:
        f.write(row)

    print(
        f"[record] iteration {iteration}: val_score={val_score:.4f}, "
        f"evals={evals_passed}/{evals_total}, commit={commit}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record iteration result")
    parser.add_argument("--val-score", type=float, required=True, help="Mean reward on full test set")
    parser.add_argument("--evals-passed", type=int, required=True, help="Eval suite tasks that passed")
    parser.add_argument("--evals-total", type=int, required=True, help="Total eval suite tasks")
    args = parser.parse_args()
    record(args.val_score, args.evals_passed, args.evals_total)
