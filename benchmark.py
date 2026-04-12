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
from typing import Literal


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


def make_runner_from_config(
    cfg: dict,
    *,
    split: str | None = None,
    role: Literal["train", "gate"] | None = None,
) -> BenchmarkRunner:
    """
    Build the appropriate runner from ``experiment_config.yaml``.

    If ``split`` is set, it is used as the data split (train / gate splits are usually
    set via ``split`` vs ``gate_split`` in config — pass the resolved split here).

    If ``role`` is set and ``split`` is None, uses ``cfg['split']`` for ``train`` and
    ``cfg['gate_split']`` for ``gate``.
    """
    benchmark = cfg.get("benchmark", "tau")
    if role is not None and split is None:
        if role == "train":
            split = cfg.get("split", "train")
        else:
            split = cfg.get("gate_split", "test")
    if split is None:
        split = "test"

    if benchmark == "tau":
        if "domain" not in cfg:
            raise ValueError("'domain' is required when benchmark: tau")
        return TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=split,
            max_concurrency=cfg.get("max_concurrency", 3),
        )

    if benchmark == "swe":
        from agent.swe.runner import SWEBenchRunner

        return SWEBenchRunner(
            dataset_name=cfg.get("swe_dataset", "SWE-bench/SWE-bench_Lite"),
            split=split,
            agent_model=cfg.get("agent_model"),
            skip_harness=cfg.get("swe_skip_harness", True),
            use_llm=cfg.get("swe_use_llm", True),
            max_workers=int(cfg.get("swe_max_workers", cfg.get("max_concurrency", 4))),
            timeout=int(cfg.get("swe_timeout", 1800)),
            namespace=cfg.get("swe_namespace", "swebench"),
            predictions_dir=cfg.get("swe_predictions_dir"),
        )

    raise ValueError(f"Unknown benchmark: {benchmark!r} (expected 'tau' or 'swe')")


MINI_TASK_COUNT = 3


def resolve_mini_task_ids(cfg: dict) -> list[str]:
    """
    When ``mini: true`` in config, run exactly ``MINI_TASK_COUNT`` tasks everywhere
    (default ``benchmark.py``, gating Step 2, ``prepare`` baseline).

    Sources (in order): ``mini_task_ids`` (first 3), else for SWE first 3 of
    ``swe_default_task_ids``.
    """
    raw: list = list(cfg.get("mini_task_ids") or [])
    if not raw and cfg.get("benchmark") == "swe" and cfg.get("swe_default_task_ids"):
        raw = list(cfg["swe_default_task_ids"])
    out = [str(x) for x in raw[:MINI_TASK_COUNT]]
    if len(out) < MINI_TASK_COUNT:
        raise ValueError(
            f"mini: true requires at least {MINI_TASK_COUNT} ids: set mini_task_ids or, "
            f"for SWE, at least {MINI_TASK_COUNT} entries in swe_default_task_ids."
        )
    return out[:MINI_TASK_COUNT]


def resolve_swe_run_task_ids(cfg: dict, cli_task_ids: list[str] | None) -> list[str] | None:
    """
    Task ids for ``SWEBenchRunner.run``.

    If the CLI did not pass ``--task-ids`` (``None``) and ``mini: true``, use
    :func:`resolve_mini_task_ids`. Else if ``swe_default_task_ids`` is set, use that list.
    """
    if cfg.get("benchmark") != "swe":
        return cli_task_ids
    if cli_task_ids is not None:
        return cli_task_ids if cli_task_ids else None
    if cfg.get("mini"):
        return resolve_mini_task_ids(cfg)
    raw = cfg.get("swe_default_task_ids")
    if not raw:
        return None
    return [str(x) for x in raw]


def resolve_swe_gate_task_ids(cfg: dict) -> list[str] | None:
    """
    Optional task ids for ``gating.py`` Step 2 (SWE only).

    When ``None``, Step 2 runs the full ``gate_split`` (e.g. all of Lite ``dev`` or ``test``).
    When set, Step 2 runs only these instance ids on the configured ``gate_split``'s dataset
    (still valid HF ids for that split).

    Config:
        ``swe_gate_task_ids``: explicit list, or
        ``swe_gate_match_default_task_ids: true`` with ``swe_default_task_ids`` set — reuse
        the same list as default ``benchmark.py`` for a aligned but smaller gate.
    """
    if cfg.get("benchmark") != "swe":
        return None
    raw = cfg.get("swe_gate_task_ids")
    if raw:
        return [str(x) for x in raw]
    if cfg.get("swe_gate_match_default_task_ids") and cfg.get("swe_default_task_ids"):
        return [str(x) for x in cfg["swe_default_task_ids"]]
    return None


def resolve_gate_task_ids(cfg: dict) -> list[str] | None:
    """
    Task ids for ``gating.py`` Step 2 (tau or SWE).

    When ``mini: true``, always the same :func:`resolve_mini_task_ids` list (3 tasks).
    Otherwise SWE uses :func:`resolve_swe_gate_task_ids`; tau uses full split (``None``).
    """
    if cfg.get("mini"):
        return resolve_mini_task_ids(cfg)
    if cfg.get("benchmark") == "swe":
        return resolve_swe_gate_task_ids(cfg)
    return None


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
    benchmark = cfg.get("benchmark", "tau")
    if benchmark == "tau" and "domain" not in cfg:
        print("ERROR: 'domain' not set in experiment_config.yaml (required when benchmark: tau)")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Task IDs to run (default: full split, or swe_default_task_ids when set for SWE)",
    )
    parser.add_argument("--domain", default=cfg.get("domain"))
    parser.add_argument("--split", default=cfg.get("split", "test"))
    parser.add_argument("--concurrency", type=int, default=cfg.get("max_concurrency", 3))
    args = parser.parse_args()

    if benchmark == "tau":
        runner = TauBenchRunner(
            domain=args.domain or cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=args.split,
            max_concurrency=args.concurrency,
        )
    else:
        cfg_run = {**cfg, "split": args.split, "max_concurrency": args.concurrency}
        runner = make_runner_from_config(cfg_run, split=args.split)

    if benchmark == "swe":
        try:
            task_ids = resolve_swe_run_task_ids(cfg, args.task_ids)
        except ValueError as e:
            print(f"[benchmark] ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.task_ids is not None:
            task_ids = args.task_ids if args.task_ids else None
        elif cfg.get("mini"):
            try:
                task_ids = resolve_mini_task_ids(cfg)
            except ValueError as e:
                print(f"[benchmark] ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            task_ids = None

    if cfg.get("mini"):
        print(f"[benchmark] mini mode: {MINI_TASK_COUNT} tasks — {task_ids}")
    elif benchmark == "swe" and task_ids:
        print(f"[benchmark] SWE task_ids ({len(task_ids)}): {', '.join(task_ids)}")

    results = runner.run(task_ids=task_ids)
    val = runner.val_score(results)

    print(f"\nval_score: {val:.4f}  ({sum(v >= 0.5 for v in results.values())}/{len(results)} passed)")

    def _sort_key(item: tuple[str, float]):
        tid, _ = item
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    for task_id, reward in sorted(results.items(), key=_sort_key):
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

    if benchmark == "swe":
        from agent.swe.diag import finalize_swe_benchmark_artifacts

        finalize_swe_benchmark_artifacts(runner, results, train_results_path)
