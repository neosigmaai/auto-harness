"""
Three-step gate for agent changes.

Step 1 — Regression suite:  re-runs tasks in workspace/suite.json, checks pass rate >= threshold.
Step 2 — Full test:         always runs the full benchmark, checks val_score >= best seen in results.tsv.
Step 3 — Suite promotion:   only if Steps 1+2 pass — re-runs previously-failing train tasks,
                             promotes newly-passing ones into suite.json.

Exit 0 only after all three steps complete successfully. Steps 1 or 2 failing exits 1 immediately.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future

import yaml

from benchmark import BenchmarkRunner, TauBenchRunner, TerminalBenchRunner

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
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    scores = [float(r["val_score"]) for r in rows if r.get("val_score")]
    return max(scores) if scores else None


def _run_suite_gate(train_runner: BenchmarkRunner, suite: dict) -> tuple[bool, int, dict]:
    """Step 1: regression suite. Returns (passed, n_passed, results)."""
    task_ids = suite.get("tasks", [])
    threshold = suite.get("threshold", 0.8)

    if not task_ids:
        print("\n[gate] Step 1: eval suite is empty — skipping")
        return True, 0, {}

    print(f"\n[gate] Step 1: eval suite ({len(task_ids)} tasks, threshold={threshold:.0%})")
    results = train_runner.run(task_ids=task_ids)
    passed = sum(1 for r in results.values() if r >= 0.5)
    pass_rate = passed / len(results) if results else 0

    suite["last_results"] = results
    save_suite(suite)

    suite_passed = pass_rate >= threshold
    print(f"       {passed}/{len(results)} passed ({pass_rate:.0%})  "
          f"{'PASS ✓' if suite_passed else 'FAIL ✗'}")
    return suite_passed, passed, results


def _run_test_gate(gate_runner: BenchmarkRunner) -> tuple[bool, float, float | None]:
    """Step 2: full test benchmark. Returns (passed, val_score, best)."""
    print("\n[gate] Step 2: full benchmark (test split)")
    all_results = gate_runner.run()
    val = gate_runner.val_score(all_results)
    best = best_val_score()

    test_passed = best is None or val >= best
    if test_passed:
        suffix = f"(prev best: {best:.4f})" if best is not None else "(first run)"
        print(f"       val_score={val:.4f}  PASS ✓  {suffix}")
    else:
        print(f"       val_score={val:.4f}  FAIL ✗  (best so far: {best:.4f})")
    return test_passed, val, best


def run_gate(
    train_runner: BenchmarkRunner,
    gate_runner: BenchmarkRunner,
    parallel: bool = True,
) -> int:
    suite = load_suite()

    if parallel and suite.get("tasks"):
        # Run Steps 1 and 2 in parallel
        print("\n[gate] Running Steps 1 and 2 in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            suite_future: Future = executor.submit(_run_suite_gate, train_runner, suite)
            test_future: Future = executor.submit(_run_test_gate, gate_runner)
            suite_passed, passed, suite_results = suite_future.result()
            test_passed, val, best = test_future.result()
    else:
        # Sequential: skip Step 2 if Step 1 fails
        suite_passed, passed, suite_results = _run_suite_gate(train_runner, suite)
        if not suite_passed:
            threshold = suite.get("threshold", 0.8)
            pass_rate = passed / len(suite_results) if suite_results else 0
            print(f"\n[gate] FAILED — eval suite pass rate {pass_rate:.0%} < threshold {threshold:.0%}")
            return 1
        test_passed, val, best = _run_test_gate(gate_runner)

    if not suite_passed:
        threshold = suite.get("threshold", 0.8)
        pass_rate = passed / len(suite_results) if suite_results else 0
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

    print(f"\n[gate] PASSED ✓  All steps clear. (val_score={val:.4f})")
    return 0


def _create_runners(cfg: dict) -> tuple[BenchmarkRunner, BenchmarkRunner]:
    """Create train and gate runners based on benchmark config."""
    benchmark = cfg.get("benchmark", "tau-bench")

    if benchmark == "terminal-bench":
        train_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("split", "train"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/train",
        )
        gate_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/test",
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
        )
        gate_runner = TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            max_concurrency=cfg.get("max_concurrency", 3),
        )
    else:
        print(f"ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    return train_runner, gate_runner


if __name__ == "__main__":
    cfg = load_config()
    parallel = cfg.get("parallel_gate", True)
    train_runner, gate_runner = _create_runners(cfg)
    sys.exit(run_gate(train_runner, gate_runner, parallel=parallel))
