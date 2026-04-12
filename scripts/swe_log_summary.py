#!/usr/bin/env python3
"""
Make a SWE-Bench ``run_instance.log`` readable.

The harness often repeats the same multi-line traceback hundreds of times; the
useful content is usually the **first** section (container start, patch apply) plus
``patch.diff`` next to the log.

Usage:
  python scripts/swe_log_summary.py path/to/run_instance.log
  python scripts/swe_log_summary.py --head-bytes 12000 path/to/run_instance.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.swe.log_collapse import collapse_run_instance_log_text  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collapse repeated timestamp-blocks in SWE harness run_instance.log"
    )
    p.add_argument("log_path", type=Path, help="Path to run_instance.log")
    p.add_argument(
        "--head-bytes",
        type=int,
        default=0,
        help="After collapse, print only the first N bytes (0 = full output)",
    )
    args = p.parse_args()

    text = args.log_path.read_text(encoding="utf-8", errors="replace")
    collapsed = collapse_run_instance_log_text(text)
    full_len = len(collapsed)
    if args.head_bytes > 0:
        collapsed = collapsed[: args.head_bytes]
        if args.head_bytes < full_len:
            collapsed += "\n[... truncated by --head-bytes ...]\n"
    print(collapsed, end="" if collapsed.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
