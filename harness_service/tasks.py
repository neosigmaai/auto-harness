"""Named TerminalBench 2.0 task subsets.

These are REAL ``terminal-bench@2.0`` task IDs (enumerated via
``harbor download terminal-bench@2.0``), so the same subset drives both the
SIMULATED executor (dev) and the HARBOR executor (real E2B sandbox, M3).

Selection method (evidence-based, from each task's ``task.toml`` metadata):
  * **Fast bucket** — agent timeout ≤ 900s, build 600s, 2G RAM / 1 CPU. These are
    the cheapest tasks to run in parallel on E2B and keep the full subset finishing
    in a reasonable time (the brief's requirement).
  * **Category spread** — 8 of the benchmark's categories, so an improvement that
    helps has to generalize rather than overfit one skill.
  * **Difficulty mix** — mostly ``medium`` with a few ``easy`` and one ``hard``, so
    the untuned baseline agent fails a meaningful fraction (real optimization signal)
    without the subset being dominated by 40-minute monsters.

Each entry: (category, difficulty, expert_time_min, why).
"""

from __future__ import annotations

CORE_TASKS: dict[str, tuple[str, str, float, str]] = {
    "fix-git": ("software-engineering", "easy", 5,
                "common git-state recovery; fast, exercises VCS reasoning"),
    "cobol-modernization": ("software-engineering", "easy", 20,
                "read + port legacy code; quick but needs careful comprehension"),
    "cancel-async-tasks": ("software-engineering", "hard", 120,
                "concurrency correctness; the deliberate 'hard' case for failure signal"),
    "nginx-request-logging": ("system-administration", "medium", 20,
                "service config + verification; classic sysadmin multi-step"),
    "sqlite-with-gcov": ("system-administration", "medium", 30,
                "build/instrument tooling; environment manipulation"),
    "openssl-selfsigned-cert": ("security", "medium", 20,
                "precise CLI incantation; correctness-sensitive"),
    "crack-7z-hash": ("security", "medium", 5,
                "short, well-defined security task; cheap to run"),
    "regex-log": ("data-processing", "medium", 45,
                "text transformation with exact-output verification"),
    "hf-model-inference": ("data-science", "medium", 20,
                "run a model + produce output; light data-science path"),
    "largest-eigenval": ("mathematics", "medium", 60,
                "numeric compute with a checkable answer"),
    "extract-elf": ("file-operations", "medium", 30,
                "binary/file inspection; different skill axis"),
    "overfull-hbox": ("debugging", "easy", 60,
                "the only 'easy' debugging task; fast (750s agent budget)"),
}

# Smallest, cheapest real runs (expert_time ≈ 5 min) — for a quick sandbox smoke test.
SMOKE_TASKS = ["fix-git", "crack-7z-hash", "raman-fitting"]

CORE_SUBSET: list[str] = list(CORE_TASKS)
SMOKE_SUBSET: list[str] = SMOKE_TASKS

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
