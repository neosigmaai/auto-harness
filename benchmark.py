"""
Benchmark execution layer.

BenchmarkRunner: abstract base class — subclass to plug in your own benchmark.
TauBenchRunner:  implementation for tau-bench (https://github.com/sierra-research/tau2-bench).
TerminalBenchRunner: implementation for Terminal-Bench 2.0 via Harbor framework.

Both gating.py and the coding agent call this directly.
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


class TerminalBenchRunner(BenchmarkRunner):
    """
    Runner for Terminal-Bench 2.0 via Harbor framework.

    Invokes `harbor run` as a subprocess and parses per-task results from the
    output directory.

    Usage:
        runner = TerminalBenchRunner(split="train")
        results = runner.run()                                    # full split
        results = runner.run(task_ids=["cobol-modernization"])    # specific tasks
    """

    SPLIT_FILE = "tbench_data/task_split.json"

    def __init__(
        self,
        agent_model: str | None = None,
        split: str = "train",
        env_provider: str = "e2b",
        n_concurrent: int = 50,
        dataset: str = "terminal-bench@2.0",
        agent_import_path: str = "agent.agent:HarnessAgent",
        per_task_timeout: int = 1200,
    ):
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.env_provider = env_provider
        self.n_concurrent = n_concurrent
        self.dataset = dataset
        self.agent_import_path = agent_import_path
        self.per_task_timeout = per_task_timeout

    def _load_split_tasks(self) -> list[str]:
        """Load task names for the configured split from the split file."""
        import json

        if not os.path.exists(self.SPLIT_FILE):
            raise FileNotFoundError(
                f"{self.SPLIT_FILE} not found. Run prepare.py first."
            )
        with open(self.SPLIT_FILE) as f:
            splits = json.load(f)
        tasks = splits.get(self.split)
        if tasks is None:
            raise ValueError(
                f"Split '{self.split}' not found in {self.SPLIT_FILE}. "
                f"Available: {list(splits.keys())}"
            )
        return tasks

    def run(self, task_ids: list[str] | None = None) -> dict[str, float]:
        import json
        import subprocess
        import tempfile

        if task_ids is None:
            task_ids = self._load_split_tasks()

        # Create a unique output directory for this run
        jobs_dir = os.path.join("workspace", "tbench_jobs")
        os.makedirs(jobs_dir, exist_ok=True)

        # Build harbor run command
        n = min(self.n_concurrent, len(task_ids))
        agent_timeout_mult = self.per_task_timeout / 180  # Harbor default is 180s
        cmd = [
            "harbor", "run",
            "-d", self.dataset,
            "--agent-import-path", self.agent_import_path,
            "--model", self.agent_model,
            "--env", self.env_provider,
            "-n", str(n),
            "--jobs-dir", jobs_dir,
            "--agent-timeout-multiplier", f"{agent_timeout_mult:.2f}",
            "-y",
        ]
        for tid in task_ids:
            cmd.extend(["-i", tid])

        # Set PYTHONPATH so Harbor can import the agent module
        env = os.environ.copy()
        repo_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        # Disable trace saving for test splits (prevent coding agent from reading test traces)
        if self.split == "test":
            env["HARNESS_SAVE_TRACE"] = "0"

        # Subprocess timeout: per_task_timeout × number of batches + 5 min buffer
        import math
        n_batches = math.ceil(len(task_ids) / max(n, 1))
        timeout_sec = self.per_task_timeout * n_batches + 300
        print(f"[benchmark] running {len(task_ids)} terminal-bench tasks "
              f"(model={self.agent_model}, env={self.env_provider}, "
              f"n={n}, per_task_timeout={self.per_task_timeout}s, "
              f"subprocess_timeout={timeout_sec}s)")

        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[benchmark] WARNING: harbor run timed out after {timeout_sec}s")

        # Find the job directory (most recent in jobs_dir)
        job_dirs = sorted(
            [d for d in os.listdir(jobs_dir)
             if os.path.isdir(os.path.join(jobs_dir, d))],
            reverse=True,
        )
        if not job_dirs:
            print("[benchmark] ERROR: no job output found")
            return {}

        job_dir = os.path.join(jobs_dir, job_dirs[0])

        # Parse per-trial result.json files
        results = {}
        for trial_name in os.listdir(job_dir):
            trial_result = os.path.join(job_dir, trial_name, "result.json")
            if not os.path.exists(trial_result):
                continue
            try:
                with open(trial_result) as f:
                    data = json.load(f)
                task_name = data.get("task_name", trial_name)
                vr = data.get("verifier_result")
                if vr and isinstance(vr, dict):
                    reward = float(vr.get("rewards", {}).get("reward", 0.0))
                else:
                    reward = 0.0
                results[task_name] = reward
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[benchmark] WARNING: failed to parse {trial_result}: {e}")
                continue

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import datetime
    import json as _json

    import yaml

    def _load_config() -> dict:
        if os.path.exists("experiment_config.yaml"):
            with open("experiment_config.yaml") as f:
                return yaml.safe_load(f) or {}
        return {}

    cfg = _load_config()
    benchmark = cfg.get("benchmark", "tau-bench")

    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("--task-ids", nargs="*", help="Task IDs to run (default: all)")
    parser.add_argument("--split", default=cfg.get("split", "train"))
    parser.add_argument("--concurrency", type=int, default=cfg.get("max_concurrency", 3))
    args = parser.parse_args()

    if benchmark == "terminal-bench":
        runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=args.split,
            env_provider=cfg.get("env_provider", "daytona"),
            n_concurrent=args.concurrency,
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
        )
    else:
        if "domain" not in cfg:
            print("ERROR: 'domain' not set in experiment_config.yaml")
            sys.exit(1)
        runner = TauBenchRunner(
            domain=cfg["domain"],
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

    train_results_path = "workspace/train_results.json"
    os.makedirs("workspace", exist_ok=True)
    with open(train_results_path, "w") as f:
        _json.dump({
            "split": args.split,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }, f, indent=2)
    print(f"[benchmark] results saved to {train_results_path}")
