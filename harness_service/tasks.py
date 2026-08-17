"""Named TerminalBench task subsets.

For the SIMULATED executor these IDs are opaque labels (any string works). For the
real HARBOR executor (M3) they must match the live ``terminal-bench@2.0`` dataset;
the exact IDs are reconciled against ``harbor tasks -d terminal-bench@2.0`` then.
The selection below is a representative, fast-to-run spread across the benchmark's
categories (coding / sysadmin / data / security), per the brief.
"""

from __future__ import annotations

# 12 tasks — representative + fast. Rationale documented in the README.
CORE_SUBSET: list[str] = [
    # coding
    "fix-git-merge-conflict",
    "implement-lru-cache",
    "cobol-modernization",
    # sysadmin
    "configure-nginx-reverse-proxy",
    "cron-log-rotation",
    "recover-deleted-file",
    # data
    "csv-join-aggregate",
    "sqlite-schema-migration",
    "regex-log-parse",
    # security
    "crack-weak-hash",
    "detect-sandbox-escape",
    "tls-cert-setup",
]

SMOKE_SUBSET: list[str] = ["implement-lru-cache", "regex-log-parse", "tls-cert-setup"]

SUBSETS: dict[str, list[str]] = {
    "core": CORE_SUBSET,
    "smoke": SMOKE_SUBSET,
}


def resolve_subset(ref: str | list[str]) -> list[str]:
    """Resolve a subset name or an explicit task-id list to a list of task ids."""
    if isinstance(ref, list):
        if not ref:
            raise ValueError("subset task list must be non-empty")
        return ref
    if ref not in SUBSETS:
        raise ValueError(f"Unknown subset '{ref}'. Known: {sorted(SUBSETS)}")
    return list(SUBSETS[ref])
