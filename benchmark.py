"""
Benchmark execution layer.

BenchmarkRunner: abstract base class — subclass to plug in your own benchmark.
TauBenchRunner:  implementation for tau-bench (https://github.com/sierra-research/tau2-bench).
TerminalBenchRunner: implementation for terminal-bench (https://github.com/harbor-framework/terminal-bench).

Both gating.py and the coding agent call this directly.
All agent logic (prompts, tools, state) lives in agent/, not here.
This file is deterministic infrastructure — it never changes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path


# ============================================================================
# BENCHMARK RUNNER BASE CLASS
# ============================================================================

class BenchmarkRunner(ABC):
    """Abstract benchmark runner. Subclass and implement `run` to plug in your own benchmark."""

    @abstractmethod
    def run(self, task_ids: list[str] | None = None) -> dict[str, float]:
        """
        Run the benchmark on the given tasks.

        Args:
            task_ids: specific task IDs to run. None runs the full benchmark.

        Returns:
            Mapping of task_id -> reward (float in [0.0, 1.0]).
        """

    def val_score(self, results: dict[str, float]) -> float:
        """Mean reward across all results."""
        if not results:
            return 0.0
        return sum(results.values()) / len(results)


# ============================================================================
# TAU-BENCH
# ============================================================================

class TauBenchRunner(BenchmarkRunner):
    """
    Runner for tau-bench (https://github.com/sierra-research/tau2-bench).
    Uses the tau2 Python API directly (no subprocess).

    Usage:
        runner = TauBenchRunner(domain="retail", split="test")
        results = runner.run()                            # full benchmark
        results = runner.run(task_ids=["0", "1", "42"])  # specific tasks
    """

    def __init__(
        self,
        domain: str,
        agent_model: str | None = None,
        split: str = "test",
        max_concurrency: int = 3,
        seed: int = 300,
    ):
        self.domain = domain
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.max_concurrency = max_concurrency
        self.seed = seed

    def run(self, task_ids: list[str] | None = None) -> dict[str, float]:
        from tau2.data_model.simulation import TextRunConfig
        from tau2 import registry
        from tau2.run import run_domain
        from agent.agent_tau import TauHarnessAgent

        def _create_harness_agent(tools, domain_policy, **kwargs):
            return TauHarnessAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args"),
            )

        if registry.get_agent_factory("custom_agent") is None:
            registry.register_agent_factory(_create_harness_agent, "custom_agent")

        config = TextRunConfig(
            domain=self.domain,
            agent="custom_agent",
            llm_agent=self.agent_model,
            task_split_name=self.split,
            task_ids=task_ids,
            max_concurrency=self.max_concurrency,
            seed=self.seed,
        )

        results = run_domain(config)

        return {
            str(sim.task_id): float(sim.reward_info.reward) if sim.reward_info else 0.0
            for sim in results.simulations
        }


# ============================================================================
# TERMINAL-BENCH
# ============================================================================

class TerminalBenchRunner(BenchmarkRunner):
    """
    Runner for TerminalBench (https://github.com/harbor-framework/terminal-bench).
    Invokes the tb CLI via subprocess and parses per-task reward outputs.

    Usage:
        runner = TerminalBenchRunner()
        results = runner.run()                                # full benchmark
        results = runner.run(task_ids=["build-linux-kernel"]) # specific tasks
    """

    def __init__(
        self,
        agent_model: str | None = None,
        dataset_name: str = "terminal-bench-core",
        dataset_version: str = "0.1.1",
        n_concurrent: int = 8,
        jobs_dir: str = "jobs",
    ):
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "openai/gpt-4o")
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.n_concurrent = n_concurrent
        self.jobs_dir = jobs_dir

    def run(self, task_ids: list[str] | None = None) -> dict[str, float]:
        cmd = [
            "tb", "run",
            "--agent-import-path", "agent.agent_harbor:HarborAgent",
            "--model", self.agent_model,
            "--dataset-name", self.dataset_name,
            "--dataset-version", self.dataset_version,
            "--n-concurrent", str(self.n_concurrent),
            "--output-path", self.jobs_dir,
        ]

        if task_ids:
            for task_id in task_ids:
                cmd += ["--task-name", task_id]

        subprocess.run(cmd, check=True)

        return self._parse_results()

    def _parse_results(self) -> dict[str, float]:
        """Parse per-task reward files from jobs output directory."""
        results = {}
        for task_dir in Path(self.jobs_dir).iterdir():
            reward_file = task_dir / "logs" / "reward.txt"
            if reward_file.exists():
                try:
                    reward = float(reward_file.read_text().strip())
                    results[task_dir.name] = reward
                except ValueError:
                    results[task_dir.name] = 0.0
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml
    import datetime
    import json as _json

    def _load_config() -> dict:
        if os.path.exists("experiment_config.yaml"):
            with open("experiment_config.yaml") as f:
                return yaml.safe_load(f) or {}
        return {}

    cfg = _load_config()
    benchmark = cfg.get("benchmark", "tau_bench")

    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("--task-ids", nargs="*", help="Task IDs to run (default: all)")
    parser.add_argument("--benchmark", default=benchmark)
    parser.add_argument("--domain", default=cfg.get("domain"))
    parser.add_argument("--split", default=cfg.get("split", "test"))
    parser.add_argument("--concurrency", type=int, default=cfg.get("max_concurrency", 3))
    args = parser.parse_args()

    if args.benchmark == "tau_bench":
        if not args.domain:
            print("ERROR: 'domain' not set in experiment_config.yaml")
            sys.exit(1)
        runner = TauBenchRunner(
            domain=args.domain,
            agent_model=cfg.get("agent_model"),
            split=args.split,
            max_concurrency=args.concurrency,
        )
    elif args.benchmark == "terminal_bench":
        runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
        )
    else:
        print(f"ERROR: unknown benchmark '{args.benchmark}'")
        sys.exit(1)

    results = runner.run(task_ids=args.task_ids)
    val = runner.val_score(results)

    print(f"\nval_score: {val:.4f}  ({sum(v >= 0.5 for v in results.values())}/{len(results)} passed)")
    for task_id, reward in sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        status = "PASS" if reward >= 0.5 else "FAIL"
        print(f"  {status}  {task_id}: {reward:.2f}")

    train_results_path = "workspace/train_results.json"
    os.makedirs("workspace", exist_ok=True)
    with open(train_results_path, "w") as f:
        _json.dump({
            "benchmark": args.benchmark,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }, f, indent=2)
    print(f"[benchmark] results saved to {train_results_path}")