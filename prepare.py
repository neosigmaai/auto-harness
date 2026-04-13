"""
Run once before starting an experiment.

Checks required environment variables, validates data/tools for the
configured benchmark, initializes workspace/ files, copies the correct
agent template into agent/agent.py, and runs a baseline benchmark.

Supports both tau-bench and terminal-bench.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

WORKSPACE = "workspace"
SUITE_FILE = os.path.join(WORKSPACE, "suite.json")
LEARNINGS_FILE = os.path.join(WORKSPACE, "learnings.md")
RESULTS_FILE = os.path.join(WORKSPACE, "results.tsv")
TRAIN_RESULTS_FILE = os.path.join(WORKSPACE, "train_results.json")
CONFIG_FILE = "experiment_config.yaml"

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        print(f"[prepare] ERROR: {CONFIG_FILE} not found.")
        print(f"          Copy experiment_config.yaml.template to {CONFIG_FILE} and configure it.")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


# ── Environment checks ───────────────────────────────────────────────────────


def check_env_tau_bench(cfg: dict) -> bool:
    """Check environment for tau-bench."""
    model = cfg.get("agent_model", "")
    if model.startswith("gemini"):
        required = ["GEMINI_API_KEY"]
    elif model.startswith("claude"):
        required = ["ANTHROPIC_API_KEY"]
    else:
        required = ["OPENAI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[prepare] ERROR: missing env vars for tau-bench: {', '.join(missing)}")
        return False
    return True


def check_env_terminal_bench(cfg: dict) -> bool:
    """Check environment for terminal-bench."""
    env_provider = cfg.get("env_provider", "e2b")
    required = []

    # Need at least one LLM API key
    model = cfg.get("agent_model", "gpt-5.4")
    if model.startswith("gemini"):
        required.append("GEMINI_API_KEY")
    elif model.startswith("claude"):
        required.append("ANTHROPIC_API_KEY")
    else:
        required.append("OPENAI_API_KEY")

    # Need sandbox provider key
    if env_provider == "e2b":
        required.append("E2B_API_KEY")
    elif env_provider == "daytona":
        required.append("DAYTONA_API_KEY")
    # docker needs no key

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[prepare] ERROR: missing env vars for terminal-bench: {', '.join(missing)}")
        return False

    # Check harbor CLI
    if shutil.which("harbor") is None:
        print("[prepare] ERROR: harbor CLI not found. Install with: uv tool install harbor")
        return False
    print(f"[prepare] harbor CLI found: {shutil.which('harbor')}")

    # Task split will be created after baseline run if needed
    return True


# ── Tau-bench data ────────────────────────────────────────────────────────────

TAU2_DATA_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_DATA_SUBDIR = "tau2"


def fetch_tau2_data(tau2_data_dir: str) -> bool:
    """Clone tau2-bench and copy its data/ into tau2_data_dir if not already present."""
    sentinel = os.path.join(tau2_data_dir, TAU2_DATA_SUBDIR)
    if os.path.isdir(sentinel):
        return True

    print(f"[prepare] tau2 data not found at {tau2_data_dir} — cloning from {TAU2_DATA_REPO} ...")
    os.makedirs(tau2_data_dir, exist_ok=True)
    tmp = os.path.join(tau2_data_dir, "_tau2-bench-tmp")
    try:
        # Remove stale tmp left by a previously interrupted clone.
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        subprocess.run(
            ["git", "clone", "--depth", "1", TAU2_DATA_REPO, tmp],
            check=True,
        )
        src = os.path.join(tmp, "data", "tau2")
        if not os.path.isdir(src):
            print("[prepare] ERROR: expected data/tau2 inside cloned repo but not found.")
            return False
        os.rename(src, sentinel)
        print(f"[prepare] tau2 data ready at {tau2_data_dir}")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[prepare] ERROR: failed to fetch tau2 data: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


DEFAULT_TAU2_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tau2_data")


def check_tau2_data(cfg: dict) -> bool:
    """Ensure tau2 data dir has the configured domain's task file, cloning if needed."""
    tau2_data_dir = os.getenv("TAU2_DATA_DIR") or DEFAULT_TAU2_DATA_DIR

    if not fetch_tau2_data(tau2_data_dir):
        return False

    if not os.path.isdir(tau2_data_dir):
        print(f"[prepare] ERROR: tau2 data dir {tau2_data_dir!r} is not a directory.")
        return False
    if "domain" not in cfg:
        print(f"[prepare] ERROR: 'domain' not set in {CONFIG_FILE}.")
        return False
    domain = cfg["domain"]
    full_path = os.path.join(tau2_data_dir, f"tau2/domains/{domain}/tasks.json")
    if not os.path.exists(full_path):
        print(f"[prepare] ERROR: tau2 data missing expected file:")
        print(f"          {full_path}")
        print(f"          Check that domain={domain!r} is correct in {CONFIG_FILE}.")
        return False

    print(f"[prepare] tau2 data OK: {tau2_data_dir} (domain={domain})")
    return True


# ── Workspace init ────────────────────────────────────────────────────────────


def init_workspace(cfg: dict) -> None:
    """Create workspace directory and initialize files if they don't exist."""
    os.makedirs(WORKSPACE, exist_ok=True)

    if not os.path.exists(SUITE_FILE):
        with open(SUITE_FILE, "w") as f:
            json.dump({"tasks": [], "threshold": cfg.get("threshold", 0.8), "last_results": {}}, f, indent=2)
        print(f"[prepare] created {SUITE_FILE}")
    else:
        print(f"[prepare] {SUITE_FILE} already exists — skipping")

    if not os.path.exists(LEARNINGS_FILE):
        with open(LEARNINGS_FILE, "w") as f:
            f.write("# Learnings\n\n")
        print(f"[prepare] created {LEARNINGS_FILE}")
    else:
        print(f"[prepare] {LEARNINGS_FILE} already exists — skipping")

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("iteration\tval_score\tcommit\tevals_passed\tevals_total\ttimestamp\n")
        print(f"[prepare] created {RESULTS_FILE}")
    else:
        print(f"[prepare] {RESULTS_FILE} already exists — skipping")

    if not os.path.exists(TRAIN_RESULTS_FILE):
        with open(TRAIN_RESULTS_FILE, "w") as f:
            json.dump({"split": None, "timestamp": None, "results": {}}, f, indent=2)
        print(f"[prepare] created {TRAIN_RESULTS_FILE}")
    else:
        print(f"[prepare] {TRAIN_RESULTS_FILE} already exists — skipping")

    print("[prepare] workspace ready.")


# ── Agent template ────────────────────────────────────────────────────────────


def copy_agent_template(benchmark: str) -> None:
    """Copy the correct agent template into agent/agent.py."""
    templates = {
        "tau-bench": "agent/templates/tau_bench.py",
        "terminal-bench": "agent/templates/terminal_bench.py",
    }
    template = templates.get(benchmark)
    if not template or not os.path.exists(template):
        print(f"[prepare] ERROR: no agent template for benchmark '{benchmark}'")
        sys.exit(1)

    shutil.copy2(template, "agent/agent.py")
    print(f"[prepare] copied {template} → agent/agent.py")


def copy_program_template(benchmark: str) -> None:
    """Compose PROGRAM.md from the shared base and the benchmark-specific section."""
    templates = {
        "tau-bench": "program_templates/tau_bench.md",
        "terminal-bench": "program_templates/terminal_bench.md",
    }
    template = templates.get(benchmark)
    if not template or not os.path.exists(template):
        print(f"[prepare] ERROR: no PROGRAM.md template for benchmark '{benchmark}'")
        sys.exit(1)

    with open("program_templates/base.md") as f:
        base = f.read()
    with open(template) as f:
        benchmark_content = f.read()

    with open("PROGRAM.md", "w") as f:
        f.write(base.rstrip("\n") + "\n\n" + benchmark_content)
    print(f"[prepare] composed PROGRAM.md from program_templates/base.md + {template}")


# ── Baseline run ──────────────────────────────────────────────────────────────


SPLIT_FILE = "tbench_data/task_split.json"


def generate_terminal_bench_split(results: dict[str, float], seed: int = 42) -> None:
    """Generate train/test split from baseline results. 70/30 stratified by pass/fail."""
    import random

    passed = sorted(k for k, v in results.items() if v >= 0.5)
    failed = sorted(k for k, v in results.items() if v < 0.5)

    random.seed(seed)
    random.shuffle(passed)
    random.shuffle(failed)

    train_pass_n = int(len(passed) * 0.7)
    train_fail_n = int(len(failed) * 0.7)
    train = sorted(passed[:train_pass_n] + failed[:train_fail_n])
    test = sorted(passed[train_pass_n:] + failed[train_fail_n:])

    os.makedirs("tbench_data", exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump({
            "train": train,
            "test": test,
            "metadata": {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "total_tasks": len(results),
                "seed": seed,
            },
        }, f, indent=2)
    print(f"[prepare] task split created: {len(train)} train, {len(test)} test")


def run_baseline(cfg: dict) -> None:
    """Run baseline benchmark, generate split if needed, record iteration 0."""
    with open(RESULTS_FILE) as f:
        rows = [line for line in f if line.strip() and not line.startswith("iteration")]
    if rows:
        print("[prepare] baseline already recorded — skipping")
        return

    benchmark = cfg.get("benchmark", "tau-bench")

    if benchmark == "terminal-bench":
        from benchmark import TerminalBenchRunner

        # First run: all tasks (no split yet) to generate the split
        if not os.path.exists(SPLIT_FILE):
            print("[prepare] running ALL terminal-bench tasks to generate train/test split...")
            all_runner = TerminalBenchRunner(
                agent_model=cfg.get("agent_model"),
                split=None,  # run all tasks
                env_provider=cfg.get("env_provider", "e2b"),
                n_concurrent=cfg.get("max_concurrency", 50),
                reasoning_effort=cfg.get("reasoning_effort"),
            )
            all_results = all_runner.run()

            # Filter out infra errors (reward stays 0 but no verifier ran)
            actual_results = {k: v for k, v in all_results.items() if v is not None}
            infra_errors = [k for k, v in all_results.items() if v is None]
            if infra_errors:
                print(f"[prepare] WARNING: {len(infra_errors)} task(s) had infra errors and are "
                      f"permanently excluded from the train/test split: {infra_errors}")
                print(f"          To include them, delete {SPLIT_FILE} and re-run prepare.py.")
            generate_terminal_bench_split(actual_results)

            # Record baseline using the test split score
            with open(SPLIT_FILE) as f:
                split = json.load(f)
            test_results = {k: actual_results.get(k, 0.0) for k in split["test"]}
            val = sum(test_results.values()) / len(test_results) if test_results else 0.0
        else:
            # Split exists — just run the test split for baseline
            runner = TerminalBenchRunner(
                agent_model=cfg.get("agent_model"),
                split=cfg.get("gate_split", "test"),
                env_provider=cfg.get("env_provider", "e2b"),
                n_concurrent=cfg.get("max_concurrency", 50),
                reasoning_effort=cfg.get("reasoning_effort"),
            )
            test_results = runner.run()
            val = runner.val_score(test_results)
    elif benchmark == "tau-bench":
        from benchmark import TauBenchRunner
        runner = TauBenchRunner(
            domain=cfg["domain"],
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            max_concurrency=cfg.get("max_concurrency", 3),
            reasoning_effort=cfg.get("reasoning_effort"),
        )
        test_results = runner.run()
        val = runner.val_score(test_results)
    else:
        print(f"[prepare] ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"0\t{val:.4f}\tbaseline\t0\t0\t{ts}\n")

    passed = sum(v >= 0.5 for v in test_results.values() if v is not None)
    print(f"[prepare] baseline val_score={val:.4f} ({passed}/{len(test_results)} passed) — recorded as iteration 0")


# ── Main ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    cfg = load_config()
    benchmark = cfg.get("benchmark", "tau-bench")
    print(f"[prepare] benchmark: {benchmark}")

    # Check environment
    if benchmark == "terminal-bench":
        if not check_env_terminal_bench(cfg):
            sys.exit(1)
    elif benchmark == "tau-bench":
        if not check_env_tau_bench(cfg):
            sys.exit(1)
        if not check_tau2_data(cfg):
            sys.exit(1)
    else:
        print(f"[prepare] ERROR: unknown benchmark '{benchmark}'")
        sys.exit(1)

    # Initialize workspace
    init_workspace(cfg)

    # Copy templates
    copy_agent_template(benchmark)
    copy_program_template(benchmark)

    # Run baseline
    run_baseline(cfg)

    print(f"\n[prepare] done. Ready to start the optimization loop.")
    print(f"          Read PROGRAM.md and run: python benchmark.py")
