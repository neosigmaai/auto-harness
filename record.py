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
import sys
from datetime import datetime, timezone

from gating import file_guard_enabled, file_guard_violations, report_file_guard_failure

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


def record(val_score: float, evals_passed: int, evals_total: int) -> int:
    """Append the iteration row. Returns 0 on success, 1 if the file guard rejects."""
    # File guard: also inspect HEAD vs HEAD~1 so an agent that committed
    # forbidden files before invoking record cannot slip past.
    if file_guard_enabled():
        violations = file_guard_violations(check_last_commit=True)
        if violations:
            report_file_guard_failure(violations, prefix="[record]")
            return 1

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
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record iteration result")
    parser.add_argument("--val-score", type=float, required=True, help="Mean reward on full test set")
    parser.add_argument("--evals-passed", type=int, required=True, help="Eval suite tasks that passed")
    parser.add_argument("--evals-total", type=int, required=True, help="Total eval suite tasks")
    args = parser.parse_args()
    sys.exit(record(args.val_score, args.evals_passed, args.evals_total))
