#!/usr/bin/env python3
"""
Seed ``workspace/suite.json`` tasks from ``workspace/train_results.json``.

Use this to skip an empty eval suite and start gating Step 1 with IDs you already
ran on the train split.

Modes:
  failing   — instance ids with score < 0.5 (default). Good “don’t regress these fixes”
             once they start passing; until then Step 1 may fail until the suite passes.
  passing   — ids with score >= 0.5. Smaller smoke: “these must stay green.”
  all       — every id in train_results.

Usage:
  python scripts/seed_suite_from_train_results.py
  python scripts/seed_suite_from_train_results.py --mode passing --replace
  python scripts/seed_suite_from_train_results.py --train-results workspace/train_results.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = _REPO / "workspace" / "train_results.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Seed suite.json from train_results.json")
    p.add_argument(
        "--train-results",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to train_results.json",
    )
    p.add_argument(
        "--mode",
        choices=("failing", "passing", "all"),
        default="failing",
        help="Which instance ids to include (default: failing)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override suite threshold (default: keep existing or 0.8)",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace suite tasks entirely; default merges with existing task ids",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tasks that would be written; do not write suite.json",
    )
    args = p.parse_args()

    tr_path = args.train_results.resolve()
    if not tr_path.is_file():
        print(f"ERROR: not found: {tr_path}", file=sys.stderr)
        return 1

    data = json.loads(tr_path.read_text(encoding="utf-8"))
    results: dict[str, float] = {}
    raw = data.get("results") or {}
    for k, v in raw.items():
        try:
            results[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    if args.mode == "all":
        chosen = sorted(results.keys())
    elif args.mode == "passing":
        chosen = sorted(iid for iid, s in results.items() if s >= 0.5)
    else:
        chosen = sorted(iid for iid, s in results.items() if s < 0.5)

    if not chosen:
        print(f"[seed_suite] no ids selected (mode={args.mode!r}, n={len(results)})")
        return 0

    suite_path = _REPO / "workspace" / "suite.json"
    existing: dict = {"tasks": [], "threshold": 0.8, "last_results": {}}
    if suite_path.is_file():
        existing = json.loads(suite_path.read_text(encoding="utf-8"))

    if args.replace:
        tasks = list(chosen)
    else:
        tasks = sorted(set(existing.get("tasks") or []) | set(chosen))

    threshold = args.threshold if args.threshold is not None else float(existing.get("threshold", 0.8))

    out = {
        "tasks": tasks,
        "threshold": threshold,
        "last_results": existing.get("last_results") or {},
    }

    print(
        f"[seed_suite] train_results={tr_path} mode={args.mode!r} "
        f"-> {len(chosen)} selected, {len(tasks)} tasks in suite (replace={args.replace})"
    )
    for i in tasks[:20]:
        print(f"  - {i}")
    if len(tasks) > 20:
        print(f"  ... and {len(tasks) - 20} more")

    if args.dry_run:
        print("[seed_suite] dry-run — not writing")
        return 0

    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[seed_suite] wrote {suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
