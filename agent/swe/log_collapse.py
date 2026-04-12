"""
Collapse repeated timestamp-blocks in SWE-Bench ``run_instance.log`` text.

Shared by ``scripts/swe_log_summary.py`` (CLI) and ``agent.swe.diag`` (benchmark loop).
"""

from __future__ import annotations

import re

_TS_START = re.compile(r"^(?=\d{4}-\d{2}-\d{2})", re.MULTILINE)


def split_timestamp_blocks(text: str) -> list[str]:
    """Split log into chunks that start with a ``YYYY-mm-dd`` line (typical swebench format)."""
    parts = _TS_START.split(text)
    return [p for p in parts if p.strip()]


def collapse_duplicate_blocks(blocks: list[str]) -> list[str]:
    if not blocks:
        return []
    out: list[str] = []
    i = 0
    while i < len(blocks):
        j = i + 1
        while j < len(blocks) and blocks[j] == blocks[i]:
            j += 1
        run_len = j - i
        out.append(blocks[i])
        if run_len > 1:
            out.append(
                f"\n[... omitted {run_len - 1} duplicate block(s) identical to the block above ...]\n"
            )
        i = j
    return out


def collapse_run_instance_log_text(text: str) -> str:
    """Return log text with duplicate timestamp-blocks collapsed (same as CLI default)."""
    blocks = split_timestamp_blocks(text)
    return "".join(collapse_duplicate_blocks(blocks))
