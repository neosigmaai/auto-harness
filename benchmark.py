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
import threading
from abc import ABC, abstractmethod

_registry_lock = threading.Lock()


class BenchmarkRunner(ABC):
    """Abstract benchmark runner. Subclass and implement `run` to plug in your own benchmark."""

    @abstractmethod
    def run(self, task_ids: list[str] | None = None) -> dict[str, float | None]:
        """
        Run the benchmark on the given tasks.

        Args:
            task_ids: specific task IDs to run. None runs the full benchmark.

        Returns:
            Mapping of task_id -> reward (float in [0.0, 1.0]), or None if the
            task could not be evaluated due to an infrastructure error.
        """

    def val_score(self, results: dict[str, float | None]) -> float:
        """Mean reward across all results, excluding infra errors (None values)."""
        valid = [v for v in results.values() if v is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)


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
        reasoning_effort: str | None = None,
    ):
        self.domain = domain
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.max_concurrency = max_concurrency
        self.seed = seed
        self.reasoning_effort = reasoning_effort

    def run(self, task_ids: list[str] | None = None) -> dict[str, float | None]:
        # tau2 reads TAU2_DATA_DIR at import time — set it before the first import
        if "TAU2_DATA_DIR" not in os.environ:
            os.environ["TAU2_DATA_DIR"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tau2_data"
            )
        if self.reasoning_effort:
            os.environ["AGENT_REASONING_EFFORT"] = self.reasoning_effort

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

        with _registry_lock:
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
        split: str | None = "train",
        env_provider: str = "e2b",
        n_concurrent: int = 50,
        dataset: str = "terminal-bench@2.0",
        agent_import_path: str = "agent.agent:HarnessAgent",
        per_task_timeout: int = 1200,
        jobs_dir: str = "workspace/tbench_jobs",
        reasoning_effort: str | None = None,
    ):
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.env_provider = env_provider
        self.n_concurrent = n_concurrent
        self.dataset = dataset
        self.agent_import_path = agent_import_path
        self.per_task_timeout = per_task_timeout
        self.jobs_dir = jobs_dir
        self.reasoning_effort = reasoning_effort

    def _load_split_tasks(self) -> list[str] | None:
        """Load task names for the configured split. Returns None to run all tasks."""
        import json

        if self.split is None:
            return None  # run all tasks in the dataset

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

    def run(self, task_ids: list[str] | None = None) -> dict[str, float | None]:
        import json
        import subprocess

        if task_ids is None:
            task_ids = self._load_split_tasks()

        # Output directory for harbor job results (harbor creates one subdirectory per job)
        jobs_dir = self.jobs_dir
        os.makedirs(jobs_dir, exist_ok=True)

        # Build harbor run command
        agent_timeout_mult = self.per_task_timeout / 180  # Harbor default is 180s
        cmd = [
            "harbor", "run",
            "-d", self.dataset,
            "--agent-import-path", self.agent_import_path,
            "--model", self.agent_model,
            "--env", self.env_provider,
            "--agent-timeout-multiplier", f"{agent_timeout_mult:.2f}",
            "--jobs-dir", jobs_dir,
            "-y",
        ]
        if task_ids is not None:
            n = min(self.n_concurrent, len(task_ids))
            cmd.extend(["-n", str(n)])
            for tid in task_ids:
                cmd.extend(["-i", tid])
        else:
            n = self.n_concurrent
            cmd.extend(["-n", str(n)])

        # Set PYTHONPATH so Harbor can import the agent module
        env = os.environ.copy()
        repo_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        env["AGENT_MODEL"] = self.agent_model  # explicit — don't rely on parent env
        if self.reasoning_effort:
            env["AGENT_REASONING_EFFORT"] = self.reasoning_effort
        # Disable trace saving for test/baseline runs (prevent coding agent from reading test traces).
        # split=None means the baseline all-tasks run; the train/test split doesn't exist yet so
        # we can't know which tasks are test tasks — safest to save nothing.
        if self.split != "train":
            env["HARNESS_SAVE_TRACE"] = "0"

        # Subprocess timeout: generous for full dataset, computed for splits
        import math
        n_tasks = len(task_ids) if task_ids else 150  # conservative upper bound for full dataset
        n_batches = math.ceil(n_tasks / max(n, 1))
        timeout_sec = self.per_task_timeout * n_batches + 300
        print(f"[benchmark] running {n_tasks} terminal-bench tasks "
              f"(model={self.agent_model}, env={self.env_provider}, "
              f"n={n}, per_task_timeout={self.per_task_timeout}s, "
              f"subprocess_timeout={timeout_sec}s)")

        import time
        run_start = time.time()

        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[benchmark] WARNING: harbor run timed out after {timeout_sec}s")

        # Find the job directory created by THIS run. Filter out stale dirs from previous
        # runs — if harbor fails before creating a new directory, we must not silently
        # return results from a prior run.
        all_dirs = [
            d for d in os.listdir(jobs_dir)
            if os.path.isdir(os.path.join(jobs_dir, d))
            and os.path.getmtime(os.path.join(jobs_dir, d)) >= run_start - 1
        ]
        if not all_dirs:
            print("[benchmark] ERROR: no job output found for this run (harbor may have failed before creating output)")
            return {}
        job_dirs = sorted(
            all_dirs,
            key=lambda d: os.path.getmtime(os.path.join(jobs_dir, d)),
            reverse=True,
        )

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
                    rewards = vr.get("rewards", {})
                    reward: float | None = float(rewards.get("reward", 0.0)) if isinstance(rewards, dict) else 0.0
                else:
                    reward = None  # verifier did not run — infra error
                results[task_name] = reward
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                print(f"[benchmark] WARNING: failed to parse {trial_result}: {e}")
                continue

        # Copy train traces for the coding agent
        # workspace/traces/baseline/ — immutable first-run traces (never overwritten)
        # workspace/traces/latest/   — most recent run (overwritten each iteration)
        if self.split == "train":
            import shutil
            latest_dir = os.path.join("workspace", "traces", "latest")
            baseline_dir = os.path.join("workspace", "traces", "baseline")
            os.makedirs(latest_dir, exist_ok=True)
            for trial_name in os.listdir(job_dir):
                trial_dir = os.path.join(job_dir, trial_name)
                trace_file = os.path.join(trial_dir, "agent", "trace.json")
                result_file = os.path.join(trial_dir, "result.json")
                if not os.path.isdir(trial_dir):
                    continue
                task_name = trial_name.rsplit("__", 1)[0]
                # Always update latest
                dest = os.path.join(latest_dir, task_name)
                os.makedirs(dest, exist_ok=True)
                if os.path.exists(trace_file):
                    shutil.copy2(trace_file, os.path.join(dest, "trace.json"))
                if os.path.exists(result_file):
                    shutil.copy2(result_file, os.path.join(dest, "result.json"))
                # Only write baseline if it doesn't exist yet
                base_dest = os.path.join(baseline_dir, task_name)
                if not os.path.exists(base_dest):
                    os.makedirs(base_dest, exist_ok=True)
                    if os.path.exists(trace_file):
                        shutil.copy2(trace_file, os.path.join(base_dest, "trace.json"))
                    if os.path.exists(result_file):
                        shutil.copy2(result_file, os.path.join(base_dest, "result.json"))
            print(f"[benchmark] traces: latest/ updated, baseline/ preserved")

        # Prune old job directories to prevent unbounded disk growth.
        # Train traces are already copied to workspace/traces/; raw harbor output is no longer needed.
        for old in os.listdir(jobs_dir):
            old_path = os.path.join(jobs_dir, old)
            if os.path.isdir(old_path) and old_path != job_dir:
                import shutil as _shutil
                _shutil.rmtree(old_path, ignore_errors=True)

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
    if benchmark == "tau-bench":
        parser.add_argument("--domain", default=cfg.get("domain"), help="tau-bench domain (overrides experiment_config.yaml)")
    parser.add_argument("--split", default=cfg.get("split", "train"))
    _concurrency_default = cfg.get("max_concurrency", 50 if benchmark == "terminal-bench" else 3)
    parser.add_argument("--concurrency", type=int, default=_concurrency_default)
    args = parser.parse_args()

    if benchmark == "terminal-bench":
        runner = TerminalBenchRunner(
            agent_model=cfg.get("agent_model"),
            split=args.split,
            env_provider=cfg.get("env_provider", "e2b"),
            n_concurrent=args.concurrency,
            dataset=cfg.get("dataset", "terminal-bench@2.0"),
            reasoning_effort=cfg.get("reasoning_effort"),
        )
    elif benchmark == "tau-bench":
        if not args.domain:
            print("ERROR: 'domain' not set in experiment_config.yaml (or pass --domain)")
            sys.exit(1)
        runner = TauBenchRunner(
            domain=args.domain,
            agent_model=cfg.get("agent_model"),
            split=args.split,
            max_concurrency=args.concurrency,
            reasoning_effort=cfg.get("reasoning_effort"),
        )
    else:
        print(f"ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    results = runner.run(task_ids=args.task_ids)
    val = runner.val_score(results)

    valid_results = [v for v in results.values() if v is not None]
    print(f"\nval_score: {val:.4f}  ({sum(v >= 0.5 for v in valid_results)}/{len(valid_results)} passed)")
    for task_id, reward in sorted(results.items(), key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0])):
        status = "PASS" if reward is not None and reward >= 0.5 else ("INFRA_ERR" if reward is None else "FAIL")
        print(f"  {status}  {task_id}: {f'{reward:.2f}' if reward is not None else 'N/A'}")

    train_results_path = "workspace/train_results.json"
    os.makedirs("workspace", exist_ok=True)
    with open(train_results_path, "w") as f:
        _json.dump({
            "split": args.split,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }, f, indent=2)
    print(f"[benchmark] results saved to {train_results_path}")
