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

import yaml

from benchmark import BenchmarkRunner, BirdInteractRunner, TauBenchRunner, TerminalBenchRunner

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
    suite = load_suite()
    task_ids: list[str] = suite.get("tasks", [])
    threshold: float = suite.get("threshold", 0.8)

    # ── Step 1: Eval suite gate (train split) ─────────────────────────────────
    if task_ids:
        print(f"\n[gate] Step 1: eval suite ({len(task_ids)} tasks, threshold={threshold:.0%})")
        results = train_runner.run(task_ids=task_ids)

        # Mirror BenchmarkRunner.val_score semantics so Step 1's pass_rate is
        # consistent with Step 2's val_score. Under "agent", a None reward
        # (typically a per-task timeout) is the agent's fault and counts
        # against the denominator. Under "infra", None is excluded.
        if train_runner.timeout_policy == "agent":
            denominator = len(results)
            passed = sum(1 for r in results.values() if r is not None and r >= 0.5)
            suite["last_results"] = results
        else:
            valid = {k: v for k, v in results.items() if v is not None}
            denominator = len(valid)
            passed = sum(1 for r in valid.values() if r >= 0.5)
            suite["last_results"] = valid
        pass_rate = passed / denominator if denominator else 0
        save_suite(suite)

        print(f"       {passed}/{denominator} passed ({pass_rate:.0%})", end="  ")
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
        # If the run has any None rewards under the "agent" policy, the failure
        # may be a one-time scoring-semantics change rather than a real regression.
        # Historical val_scores in results.tsv were written under whatever policy
        # was active at the time; the first gate after switching policies (or
        # after this change first lands) can fail spuriously. Operators who
        # confirm the agent itself hasn't regressed can drop or rewrite the
        # offending row in workspace/results.tsv to re-baseline.
        if (
            gate_runner.timeout_policy == "agent"
            and any(v is None for v in all_results.values())
        ):
            n_none = sum(1 for v in all_results.values() if v is None)
            print(
                f"\n[gate] note: {n_none} task(s) returned None and were charged as 0.0 "
                f"under timeout_policy='agent'. If the prior best was recorded under "
                f"'infra' (None excluded), re-baseline by editing workspace/results.tsv "
                f"or set timeout_policy='infra' in experiment_config.yaml."
            )
        print(f"\n[gate] FAILED — val_score {val:.4f} < best {best:.4f}")
        return 1

    # ── Step 3: Promote newly-fixed train tasks into suite ────────────────────
    print("\n[gate] Step 3: suite promotion")
    train_results = load_train_results()
    if not train_results:
        print(f"       {TRAIN_RESULTS_FILE} not found — run benchmark.py first to populate it")
    else:
        suite_set = set(suite["tasks"])
        # Symmetric with Step 1/Step 2: under "agent" policy, a timed-out task
        # (None reward) is a failure and is eligible for re-run; under "infra"
        # we ignore it because the failure was deemed non-agent.
        if train_runner.timeout_policy == "agent":
            failing_non_suite = [tid for tid, r in train_results.items()
                                 if (r is None or r < 0.5) and tid not in suite_set]
        else:
            failing_non_suite = [tid for tid, r in train_results.items()
                                 if r is not None and r < 0.5 and tid not in suite_set]
        if failing_non_suite:
            print(f"       re-running {len(failing_non_suite)} previously-failing train tasks")
            recheck = train_runner.run(task_ids=failing_non_suite)
            # Promotion still requires a real verifier pass — None can't be promoted.
            newly_fixed = sorted(tid for tid, r in recheck.items() if r is not None and r >= 0.5)
            if newly_fixed:
                suite["tasks"] = sorted(suite_set | set(newly_fixed))
                if train_runner.timeout_policy == "agent":
                    suite["last_results"].update(recheck)
                else:
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
    benchmark = cfg.get("benchmark", "tau-bench")
    timeout_policy = cfg.get("timeout_policy", "agent")

    if benchmark == "terminal-bench":
        train_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("split", "train"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/train",
            reasoning_effort=cfg.get("reasoning_effort"),
            timeout_policy=timeout_policy,
        )
        gate_runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=cfg.get("max_concurrency", 50),
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            jobs_dir="workspace/tbench_jobs/test",
            reasoning_effort=cfg.get("reasoning_effort"),
            timeout_policy=timeout_policy,
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
            timeout_policy=timeout_policy,
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
            timeout_policy=timeout_policy,
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
            timeout_policy=timeout_policy,
        )
        gate_runner = TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            max_concurrency=cfg.get("max_concurrency", 3),
            reasoning_effort=cfg.get("reasoning_effort"),
            user_model=cfg.get("user_model"),
            timeout_policy=timeout_policy,
        )
    else:
        print(f"ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    return train_runner, gate_runner


if __name__ == "__main__":
    cfg = load_config()
    train_runner, gate_runner = _create_runners(cfg)
    sys.exit(run_gate(train_runner, gate_runner))
