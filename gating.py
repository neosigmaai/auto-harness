"""
Three-step gate for agent changes.

Step 1 — Regression suite:  re-runs tasks in workspace/suite.json, checks pass rate >= threshold.
Step 2 — Gate benchmark:    runs the full ``gate_split`` (or optional SWE ``swe_gate_task_ids``),
                              checks val_score >= best seen in results.tsv.
Step 3 — Suite promotion:   only if Steps 1+2 pass — re-runs previously-failing train tasks,
                             promotes newly-passing ones into suite.json.

Exit 0 only after all three steps complete successfully. Steps 1 or 2 failing exits 1 immediately.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import yaml

from benchmark import BenchmarkRunner, make_runner_from_config, resolve_gate_task_ids

CONFIG_FILE = "experiment_config.yaml"


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}

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


def load_train_results() -> dict[str, float]:
    if not os.path.exists(TRAIN_RESULTS_FILE):
        return {}
    with open(TRAIN_RESULTS_FILE) as f:
        data = json.load(f)
    return data.get("results", {})


def best_val_score() -> float | None:
    """
    Best val_score from recorded iterations, excluding ``commit=baseline`` rows.

    Baseline from ``prepare.py`` may use the train split (e.g. Lite dev) while Step 2
    uses ``gate_split`` (e.g. test); mixing those numbers would distort the gate.
    """
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    scores = [
        float(r["val_score"])
        for r in rows
        if r.get("val_score") and (r.get("commit") or "").strip() != "baseline"
    ]
    return max(scores) if scores else None


def run_gate(
    train_runner: BenchmarkRunner,
    gate_runner: BenchmarkRunner,
    *,
    gate_task_ids: list[str] | None = None,
    gate_split: str = "test",
    mini: bool = False,
) -> int:
    suite = load_suite()
    task_ids: list[str] = suite.get("tasks", [])
    threshold: float = suite.get("threshold", 0.8)

    # ── Step 1: Eval suite gate (train split) ─────────────────────────────────
    if task_ids:
        print(f"\n[gate] Step 1: eval suite ({len(task_ids)} tasks, threshold={threshold:.0%})")
        results = train_runner.run(task_ids=task_ids)
        passed = sum(1 for r in results.values() if r >= 0.5)
        pass_rate = passed / len(results)

        suite["last_results"] = results
        save_suite(suite)

        print(f"       {passed}/{len(results)} passed ({pass_rate:.0%})", end="  ")
        suite_passed = pass_rate >= threshold
        if suite_passed:
            print("PASS ✓")
        else:
            print("FAIL ✗")
    else:
        print("\n[gate] Step 1: eval suite is empty — skipping")
        passed, results, suite_passed = 0, {}, True

    # ── Step 2: Full benchmark gate (gate_split, optional SWE task-id subset) ─
    scope = f"{gate_split} split"
    if gate_task_ids:
        scope = f"{gate_split} split, {len(gate_task_ids)} task(s)"
    if mini:
        scope = f"mini — {scope}"
    print(f"\n[gate] Step 2: benchmark gate ({scope})")
    all_results = gate_runner.run(task_ids=gate_task_ids)
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
                             if r < 0.5 and tid not in suite_set]
        if failing_non_suite:
            print(f"       re-running {len(failing_non_suite)} previously-failing train tasks")
            recheck = train_runner.run(task_ids=failing_non_suite)
            newly_fixed = sorted(tid for tid, r in recheck.items() if r >= 0.5)
            if newly_fixed:
                suite["tasks"] = sorted(suite_set | set(newly_fixed))
                suite["last_results"].update(recheck)
                save_suite(suite)
                print(f"       promoted {len(newly_fixed)} task(s) into regression suite: {newly_fixed}")
            else:
                print("       no new tasks promoted")
        else:
            print("       all failing train tasks already in suite — nothing to promote")

    print("\n[gate] PASSED ✓  All steps clear.")
    return 0


if __name__ == "__main__":
    cfg = load_config()
    benchmark = cfg.get("benchmark", "tau")
    if benchmark == "tau" and "domain" not in cfg:
        print("ERROR: 'domain' not set in experiment_config.yaml (required when benchmark: tau)")
        sys.exit(1)
    # Step 1 uses train split (same tasks as benchmark.py); step 2 uses gate_split (test)
    train_runner = make_runner_from_config(cfg, role="train")
    gate_runner = make_runner_from_config(cfg, role="gate")
    gate_split = str(cfg.get("gate_split", "test"))
    try:
        gate_task_ids = resolve_gate_task_ids(cfg)
    except ValueError as e:
        print(f"[gate] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(
        run_gate(
            train_runner,
            gate_runner,
            gate_task_ids=gate_task_ids,
            gate_split=gate_split,
            mini=bool(cfg.get("mini")),
        )
    )
