"""
Run once before starting an experiment.

Checks required environment variables, validates TAU2_DATA_DIR, and
initializes workspace/ files (suite.json, learnings.md, results.tsv).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import yaml

WORKSPACE = "workspace"
SUITE_FILE = os.path.join(WORKSPACE, "suite.json")
LEARNINGS_FILE = os.path.join(WORKSPACE, "learnings.md")
RESULTS_FILE = os.path.join(WORKSPACE, "results.tsv")
TRAIN_RESULTS_FILE = os.path.join(WORKSPACE, "train_results.json")
CONFIG_FILE = "experiment_config.yaml"

REQUIRED_ENV = ["OPENAI_API_KEY"]


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def check_env() -> bool:
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[prepare] ERROR: missing env vars: {', '.join(missing)}")
        print("          Copy .env.example to .env and fill in the values.")
        return False
    return True


TAU2_DATA_REPO = "https://github.com/sierra-research/tau2-bench.git"
# In the tau2-bench repo, data lives under data/tau2/domains/...
# TAU2_DATA_DIR should point at that data/ directory.
TAU2_DATA_SUBDIR = "tau2"  # sentinel: data is present when tau2/ exists under TAU2_DATA_DIR


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
            print(f"[prepare] ERROR: expected data/tau2 inside cloned repo but not found.")
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


def check_tau2_data() -> bool:
    """Ensure tau2 data dir has the configured domain's task file, cloning if needed."""
    tau2_data_dir = os.getenv("TAU2_DATA_DIR") or DEFAULT_TAU2_DATA_DIR

    if not fetch_tau2_data(tau2_data_dir):
        return False

    if not os.path.isdir(tau2_data_dir):
        print(f"[prepare] ERROR: tau2 data dir {tau2_data_dir!r} is not a directory.")
        return False

    cfg = load_config()
    if "domain" not in cfg:
        print(f"[prepare] ERROR: 'domain' not set in {CONFIG_FILE}.")
        print(f"          Add 'domain: <your-domain>' to {CONFIG_FILE}.")
        return False
    domain = cfg["domain"]
    required_path = f"tau2/domains/{domain}/tasks.json"
    full_path = os.path.join(tau2_data_dir, required_path)

    if not os.path.exists(full_path):
        print(f"[prepare] ERROR: tau2 data missing expected file:")
        print(f"          {full_path}")
        print(f"          Check that domain={domain!r} is correct in {CONFIG_FILE}.")
        return False

    print(f"[prepare] tau2 data OK: {tau2_data_dir} (domain={domain})")
    return True


def init_workspace() -> None:
    os.makedirs(WORKSPACE, exist_ok=True)

    if not os.path.exists(SUITE_FILE):
        with open(SUITE_FILE, "w") as f:
            json.dump({"tasks": [], "threshold": 0.8, "last_results": {}}, f, indent=2)
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


def run_baseline(cfg: dict) -> None:
    """Run test benchmark once to establish baseline val_score in results.tsv."""
    # Check whether results.tsv already has data rows (baseline already recorded).
    with open(RESULTS_FILE) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    if rows:
        print("[prepare] baseline already recorded — skipping test run")
        return

    from datetime import datetime, timezone
    from benchmark import TauBenchRunner

    print("[prepare] running baseline test benchmark (this may take a few minutes)...")
    runner = TauBenchRunner(
        domain=cfg["domain"],
        agent_model=cfg.get("agent_model"),
        split=cfg.get("gate_split", "test"),
        max_concurrency=cfg.get("max_concurrency", 3),
    )
    results = runner.run()
    val = runner.val_score(results)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(RESULTS_FILE, "a") as f:
        f.write(f"0\t{val:.4f}\tbaseline\t0\t0\t{ts}\n")

    passed = sum(v >= 0.5 for v in results.values())
    print(f"[prepare] baseline val_score={val:.4f} ({passed}/{len(results)} passed) — recorded as iteration 0")


if __name__ == "__main__":
    if not check_env():
        sys.exit(1)
    if not check_tau2_data():
        sys.exit(1)
    cfg = load_config()
    init_workspace()
    run_baseline(cfg)
