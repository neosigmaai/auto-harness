"""
Benchmark execution layer.

BenchmarkRunner: abstract base class — subclass to plug in your own benchmark.
TauBenchRunner:  example implementation for tau-bench (https://github.com/sierra-research/tau2-bench).

Both gating.py and the coding agent call this directly.

To use a different benchmark (e.g. Harbor), subclass BenchmarkRunner and swap
the runner in gating.py — the rest of the loop is unchanged. Example:

    class HarborBenchmarkRunner(BenchmarkRunner):
        def run(self, task_ids=None):
            # invoke: uv run harbor run -p tasks/ -n 100 ...
            # parse /logs/reward.txt outputs per task
            # return {task_id: reward}
            ...
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod


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


class TauBenchRunner(BenchmarkRunner):
    """
    Runner for tau-bench (https://github.com/sierra-research/tau2-bench).

    Uses the tau2 Python API directly (no subprocess).

    Usage:
        runner = TauBenchRunner(domain="retail", split="test")
        results = runner.run()                          # full benchmark
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

        from agent.agent import HarnessAgent

        def _create_harness_agent(tools, domain_policy, **kwargs):
            return HarnessAgent(
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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import yaml

    def _load_config() -> dict:
        if os.path.exists("experiment_config.yaml"):
            with open("experiment_config.yaml") as f:
                return yaml.safe_load(f) or {}
        return {}

    cfg = _load_config()
    if "domain" not in cfg:
        print("ERROR: 'domain' not set in experiment_config.yaml")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("--task-ids", nargs="*", help="Task IDs to run (default: all)")
    parser.add_argument("--domain", default=cfg["domain"])
    parser.add_argument("--split", default=cfg.get("split", "test"))
    parser.add_argument("--concurrency", type=int, default=cfg.get("max_concurrency", 3))
    args = parser.parse_args()

    runner = TauBenchRunner(
        domain=args.domain,
        agent_model=cfg.get("agent_model"),
        split=args.split,
        max_concurrency=args.concurrency,
    )
    results = runner.run(task_ids=args.task_ids)
    val = runner.val_score(results)

    print(f"\nval_score: {val:.4f}  ({sum(v >= 0.5 for v in results.values())}/{len(results)} passed)")
    for task_id, reward in sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        status = "PASS" if reward >= 0.5 else "FAIL"
        print(f"  {status}  {task_id}: {reward:.2f}")

    import datetime
    train_results_path = "workspace/train_results.json"
    os.makedirs("workspace", exist_ok=True)
    with open(train_results_path, "w") as f:
        import json as _json
        _json.dump({
            "split": args.split,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }, f, indent=2)
    print(f"[benchmark] results saved to {train_results_path}")
