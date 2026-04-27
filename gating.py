"""
Multi-step gate for agent changes.

Step 0 — File guard:        rejects iterations where tracked files outside the
                             agent's allowlist (`agent/agent.py`) were touched.
Step 1 — Regression suite:  re-runs tasks in workspace/suite.json, checks pass rate >= threshold.
Step 2 — Full test:         always runs the full benchmark, checks val_score >= best seen in results.tsv.
Step 3 — Suite promotion:   only if Steps 1+2 pass — re-runs previously-failing train tasks,
                             promotes newly-passing ones into suite.json.

Exit 0 only after all steps complete successfully. Any failing step returns 1
from `run_gate`, which the script entry surfaces as exit 1 — the standard
"revert and try a different approach" signal documented in PROGRAM.md.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import yaml

# `benchmark` is heavyweight (pulls in tau-bench, terminal-bench, bird-interact
# stacks). `record.py` re-uses the file-guard helpers below, and we don't want
# `python record.py` to pay for that import when all it does is append a row
# to results.tsv. Concrete runner classes are imported lazily inside
# `_create_runners()` (the only consumer); type-only references stay as
# strings thanks to `from __future__ import annotations` above.
if TYPE_CHECKING:
    from benchmark import BenchmarkRunner

CONFIG_FILE = "experiment_config.yaml"

# Files the agent is allowed to modify across iterations.
#   - `agent/agent.py`: the agent's own scaffold, edited on every iteration.
#   - `PROGRAM.md`:     rewritten by `prepare.py` from `program_templates/`
#                       on a fresh checkout, so it always shows up as dirty
#                       in `git diff HEAD` until committed; whitelisting
#                       avoids forcing every user to commit the generated
#                       file before the first gate run.
# Everything under `workspace/` is gitignored and therefore invisible to git,
# so it is not listed here — the agent edits `workspace/learnings.md` freely.
ALLOWED_AGENT_FILES = frozenset({"agent/agent.py", "PROGRAM.md"})


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


# Module-level latch so we warn at most once per process when git is missing
# or the cwd is not a repo (otherwise every gate step would re-print the same
# message). Kept private; reset only by re-importing the module.
_GIT_WARNED = False


def _warn_once(reason: str) -> None:
    """Print a one-shot stderr warning when the file guard can't run.

    No caller-side prefix (``[gate]`` / ``[record]``) on this line: the same
    helper fires from both ``gating.py`` and ``record.py``, and a wrong prefix
    is more confusing in logs than no prefix at all. The body is unambiguous.
    """
    global _GIT_WARNED
    if _GIT_WARNED:
        return
    print(
        f"WARNING: file guard skipped — {reason}. "
        "Set `file_guard: false` in experiment_config.yaml to silence.",
        file=sys.stderr,
    )
    _GIT_WARNED = True


def _git_unavailable_reason() -> str | None:
    """Return None if `git` works in this directory, else a human-readable reason.

    Distinguishes "git binary missing" from "not in a git repo" so the warning
    actually tells the user what to fix. Anything else (transient permission
    errors, etc.) surfaces as the generic CalledProcessError branch.
    """
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return None
    except FileNotFoundError:
        return "`git` is not installed or not on PATH"
    except subprocess.CalledProcessError:
        return "current directory is not a git repository"


def _has_parent_commit() -> bool:
    """True iff `HEAD~1` exists. False on the very first commit in a repo."""
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD~1"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _git_lines(*args: str) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line for line in out.strip().splitlines() if line]


def file_guard_violations(*, check_last_commit: bool = False) -> list[str]:
    """Return tracked paths the agent has touched outside ``ALLOWED_AGENT_FILES``.

    Always inspects:
      - ``git diff-index --name-only HEAD`` — files differing from HEAD in
        either the working tree OR the index (catches a `git add` that was
        followed by a working-tree restore, which `git diff HEAD` misses).
      - ``git ls-files --others --exclude-standard`` — new files that aren't
        gitignored. ``--exclude-standard`` honours `.gitignore`,
        `.git/info/exclude`, and the user's global gitignore, so editor
        droppings (`.idea/`, `*.swp`, ...) only leak through if they aren't
        ignored anywhere; the repo `.gitignore` covers the common cases.

    With ``check_last_commit=True`` the diff of HEAD vs HEAD~1 is also
    inspected. Used from ``record.py`` so an agent that committed forbidden
    files before invoking record cannot slip past the gate. Skipped silently
    when there is no parent commit (first commit only — the working-tree
    check above has already run by the time we get here, so anything bad
    would have been caught upstream in `gating.py`).

    If git is unavailable or we're not in a git repo, prints a one-time
    warning to stderr and returns ``[]`` (treated as no violations) so the
    rest of the gate can still run in degraded mode rather than failing
    confusingly mid-pipeline.
    """
    reason = _git_unavailable_reason()
    if reason is not None:
        _warn_once(reason)
        return []

    paths: set[str] = set()
    paths.update(_git_lines("diff-index", "--name-only", "HEAD"))
    paths.update(_git_lines("ls-files", "--others", "--exclude-standard"))
    if check_last_commit and _has_parent_commit():
        paths.update(_git_lines("diff", "--name-only", "HEAD~1", "HEAD"))
    return sorted(paths - ALLOWED_AGENT_FILES)


def file_guard_enabled() -> bool:
    """File guard is on by default.

    Disabled by any of these in ``experiment_config.yaml``:
      - boolean ``false`` / ``no`` / ``off``  (PyYAML parses these as Python ``False``)
      - integer ``0``
      - string ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` / ``""`` (case-insensitive)

    Anything else — including the missing key, ``file_guard:`` (empty value),
    ``file_guard: null``, ``file_guard: ~``, and unknown strings like
    ``file_guard: maybe`` — leaves the guard on. Conservative by design:
    a typo in the config shouldn't silently disable the safety guard.
    """
    val = load_config().get("file_guard", True)
    # Treat YAML null / empty value as "no opinion expressed" → leave guard on.
    # Otherwise `bool(None) is False` would silently disable the guard, which
    # contradicts the conservative-default contract above.
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() not in {"false", "no", "off", "0", ""}
    return bool(val)


def report_file_guard_failure(violations: list[str], *, prefix: str) -> None:
    """Print a uniform file-guard failure message to stdout.

    ``prefix`` is the caller-side label (``"[gate]"`` or ``"[record]"``) so the
    message slots into the existing log format the agent already parses for
    Step 1/2 failures.
    """
    allow = ", ".join(sorted(ALLOWED_AGENT_FILES))
    print(f"{prefix} FAILED — file guard: {len(violations)} file(s) outside the allowlist")
    print(f"{prefix}          allowed: {allow}  (workspace/ is gitignored — edit there freely)")
    for path in violations:
        print(f"{prefix}            - {path}")
    print(f"{prefix}          revert with `git checkout -- <file>` (or `git rm <file>` if untracked) and re-run.")
    print(f"{prefix}          bypass: set `file_guard: false` in experiment_config.yaml.")


SUITE_FILE = "workspace/suite.json"
RESULTS_FILE = "workspace/results.tsv"
TRAIN_RESULTS_FILE = "workspace/train_results.json"


def load_suite() -> dict:
    if not os.path.exists(SUITE_FILE):
        return {"tasks": [], "threshold": 0.8, "last_results": {}}
    with open(SUITE_FILE) as f:
        return json.load(f)


def save_suite(suite: dict) -> None:
    with open(SUITE_FILE, "w") as f:
        json.dump(suite, f, indent=2)


def load_train_results() -> dict[str, float | None]:
    if not os.path.exists(TRAIN_RESULTS_FILE):
        return {}
    with open(TRAIN_RESULTS_FILE) as f:
        data = json.load(f)
    return data.get("results", {})


def best_val_score() -> float | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    scores = [float(r["val_score"]) for r in rows if r.get("val_score")]
    return max(scores) if scores else None


def run_gate(train_runner: BenchmarkRunner, gate_runner: BenchmarkRunner) -> int:
    # ── Step 0: File-edit guard ───────────────────────────────────────────────
    # Cheap deterministic check: did the agent touch tracked files outside its
    # allowlist? If so, fail the gate the same way Step 1 / Step 2 fail —
    # return 1, let PROGRAM.md drive the revert-and-retry loop. No abort.
    if file_guard_enabled():
        violations = file_guard_violations()
        if violations:
            print("\n[gate] Step 0: file guard")
            report_file_guard_failure(violations, prefix="[gate]")
            return 1

    suite = load_suite()
    task_ids: list[str] = suite.get("tasks", [])
    threshold: float = suite.get("threshold", 0.8)

    # ── Step 1: Eval suite gate (train split) ─────────────────────────────────
    if task_ids:
        print(f"\n[gate] Step 1: eval suite ({len(task_ids)} tasks, threshold={threshold:.0%})")
        results = train_runner.run(task_ids=task_ids)
        valid = {k: v for k, v in results.items() if v is not None}
        passed = sum(1 for r in valid.values() if r >= 0.5)
        pass_rate = passed / len(valid) if valid else 0

        suite["last_results"] = valid
        save_suite(suite)

        print(f"       {passed}/{len(valid)} passed ({pass_rate:.0%})", end="  ")
        suite_passed = pass_rate >= threshold
        if suite_passed:
            print("PASS ✓")
        else:
            print("FAIL ✗")
    else:
        print("\n[gate] Step 1: eval suite is empty — skipping")
        passed, results, suite_passed = 0, {}, True

    # ── Step 2: Full benchmark gate (test split) — always runs ───────────────
    print("\n[gate] Step 2: full benchmark (test split)")
    all_results = gate_runner.run()
    val = gate_runner.val_score(all_results)
    best = best_val_score()

    print(f"       val_score={val:.4f}", end="  ")
    test_passed = best is None or val >= best
    if test_passed:
        suffix = f"(prev best: {best:.4f})" if best is not None else "(first run)"
        print(f"PASS ✓  {suffix}")
    else:
        print(f"FAIL ✗  (best so far: {best:.4f})")

    if not suite_passed:
        print(f"\n[gate] FAILED — eval suite pass rate {pass_rate:.0%} < threshold {threshold:.0%}")
        return 1
    if not test_passed:
        print(f"\n[gate] FAILED — val_score {val:.4f} < best {best:.4f}")
        return 1

    # ── Step 3: Promote newly-fixed train tasks into suite ────────────────────
    print("\n[gate] Step 3: suite promotion")
    train_results = load_train_results()
    if not train_results:
        print(f"       {TRAIN_RESULTS_FILE} not found — run benchmark.py first to populate it")
    else:
        suite_set = set(suite["tasks"])
        failing_non_suite = [tid for tid, r in train_results.items()
                             if r is not None and r < 0.5 and tid not in suite_set]
        if failing_non_suite:
            print(f"       re-running {len(failing_non_suite)} previously-failing train tasks")
            recheck = train_runner.run(task_ids=failing_non_suite)
            newly_fixed = sorted(tid for tid, r in recheck.items() if r is not None and r >= 0.5)
            if newly_fixed:
                suite["tasks"] = sorted(suite_set | set(newly_fixed))
                suite["last_results"].update({k: v for k, v in recheck.items() if v is not None})
                save_suite(suite)
                print(f"       promoted {len(newly_fixed)} task(s) into regression suite: {newly_fixed}")
            else:
                print("       no new tasks promoted")
        else:
            print("       all failing train tasks already in suite — nothing to promote")

    print(f"\n[gate] PASSED ✓  All steps clear. (val_score={val:.4f})")
    return 0


def _create_runners(cfg: dict) -> tuple[BenchmarkRunner, BenchmarkRunner]:
    """Create train and gate runners based on benchmark config."""
    # Deferred import — see top-of-file note. Importing here means
    # `record.py` (which only needs the file-guard helpers above) doesn't
    # pull in the entire benchmark dependency tree at import time.
    from benchmark import BirdInteractRunner, TauBenchRunner, TerminalBenchRunner

    benchmark = cfg.get("benchmark", "tau-bench")

    if benchmark == "terminal-bench":
        train_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("split", "train"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/train",
            reasoning_effort=cfg.get("reasoning_effort"),
        )
        gate_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/test",
            reasoning_effort=cfg.get("reasoning_effort"),
        )
    elif benchmark == "bird-interact":
        train_runner = BirdInteractRunner(
            bird_repo=cfg.get("bird_repo"),
            bird_python_bin=cfg.get("bird_python_bin"),
            split=cfg.get("split", "train"),
            mode=cfg.get("mode", "a-interact"),
            dataset=cfg.get("dataset", "lite"),
            data_path=cfg.get("bird_data_path"),
            agent_model=cfg.get("agent_model"),
            user_model=cfg.get("user_model"),
            patience=cfg.get("patience", 3),
            n_concurrent=cfg.get("max_concurrency", 3),
            per_task_timeout=cfg.get("per_task_timeout", 1800),
            jobs_dir="workspace/bird_runs/train",
            system_agent_port=cfg.get("system_agent_port", 6100),
            user_sim_port=cfg.get("user_sim_port", 6101),
            db_env_port=cfg.get("db_env_port", 6102),
            pg_host=cfg.get("pg_host"),
            pg_port=cfg.get("pg_port"),
            pg_user=cfg.get("pg_user"),
            pg_password=cfg.get("pg_password"),
        )
        gate_runner = BirdInteractRunner(
            bird_repo=cfg.get("bird_repo"),
            bird_python_bin=cfg.get("bird_python_bin"),
            split=cfg.get("gate_split", "test"),
            mode=cfg.get("mode", "a-interact"),
            dataset=cfg.get("dataset", "lite"),
            data_path=cfg.get("bird_data_path"),
            agent_model=cfg.get("agent_model"),
            user_model=cfg.get("user_model"),
            patience=cfg.get("patience", 3),
            n_concurrent=cfg.get("max_concurrency", 3),
            per_task_timeout=cfg.get("per_task_timeout", 1800),
            jobs_dir="workspace/bird_runs/test",
            system_agent_port=cfg.get("system_agent_port", 6100),
            user_sim_port=cfg.get("user_sim_port", 6101),
            db_env_port=cfg.get("db_env_port", 6102),
            pg_host=cfg.get("pg_host"),
            pg_port=cfg.get("pg_port"),
            pg_user=cfg.get("pg_user"),
            pg_password=cfg.get("pg_password"),
        )
    elif benchmark == "tau-bench":
        if "domain" not in cfg:
            print("ERROR: 'domain' not set in experiment_config.yaml")
            sys.exit(1)
        train_runner = TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=cfg.get("split", "train"),
            max_concurrency=cfg.get("max_concurrency", 3),
            reasoning_effort=cfg.get("reasoning_effort"),
            user_model=cfg.get("user_model"),
        )
        gate_runner = TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            max_concurrency=cfg.get("max_concurrency", 3),
            reasoning_effort=cfg.get("reasoning_effort"),
            user_model=cfg.get("user_model"),
        )
    else:
        print(f"ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    return train_runner, gate_runner


if __name__ == "__main__":
    cfg = load_config()
    train_runner, gate_runner = _create_runners(cfg)
    sys.exit(run_gate(train_runner, gate_runner))
