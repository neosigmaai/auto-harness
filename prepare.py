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
from pathlib import Path

import yaml

WORKSPACE = "workspace"
SUITE_FILE = os.path.join(WORKSPACE, "suite.json")
LEARNINGS_FILE = os.path.join(WORKSPACE, "learnings.md")
RESULTS_FILE = os.path.join(WORKSPACE, "results.tsv")
TRAIN_RESULTS_FILE = os.path.join(WORKSPACE, "train_results.json")
CONFIG_FILE = "experiment_config.yaml"

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def check_env(cfg: dict) -> bool:
    """OPENAI always; TAU2_DATA_DIR only when ``benchmark: tau``."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[prepare] ERROR: missing env var: OPENAI_API_KEY")
        print("          Copy .env.example to .env and fill in the values.")
        return False
    if cfg.get("benchmark", "tau") == "tau" and not os.getenv("TAU2_DATA_DIR"):
        print("[prepare] ERROR: missing env var: TAU2_DATA_DIR (required when benchmark: tau)")
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
        subprocess.run(
            ["git", "clone", "--depth", "1", TAU2_DATA_REPO, tmp],
            check=True,
        )
        # copy data/tau2 -> TAU2_DATA_DIR/tau2
        src = os.path.join(tmp, "data", "tau2")
        if not os.path.isdir(src):
            print(f"[prepare] ERROR: expected data/tau2 inside cloned repo but not found.")
            return False
        os.rename(src, sentinel)
        subprocess.run(["rm", "-rf", tmp], check=True)
        print(f"[prepare] tau2 data ready at {tau2_data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR: failed to clone tau2 data: {e}")
        return False
    return True


def check_tau2_data() -> bool:
    """Ensure TAU2_DATA_DIR has the configured domain's task file, cloning if needed."""
    tau2_data_dir = os.getenv("TAU2_DATA_DIR", "")

    if not tau2_data_dir:
        print("[prepare] ERROR: TAU2_DATA_DIR is not set.")
        print("          Set TAU2_DATA_DIR to the path where tau2 data should live.")
        return False

    if not fetch_tau2_data(tau2_data_dir):
        return False

    if not os.path.isdir(tau2_data_dir):
        print(f"[prepare] ERROR: TAU2_DATA_DIR={tau2_data_dir!r} is not a directory.")
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
        print(f"[prepare] ERROR: TAU2_DATA_DIR is set but missing expected file:")
        print(f"          {full_path}")
        print(f"          Ensure TAU2_DATA_DIR points to a valid tau2 data directory")
        print(f"          and that domain={domain!r} is correct in {CONFIG_FILE}.")
        return False

    print(f"[prepare] TAU2_DATA_DIR OK: {tau2_data_dir} (domain={domain})")
    return True


def check_swe_env(cfg: dict) -> bool:
    """Optional ``swebench`` install; Docker when ``swe_skip_harness`` is false."""
    try:
        import swebench  # noqa: F401
    except ImportError:
        print("[prepare] ERROR: SWE-Bench requires the 'swe' extra.")
        print("          pip install -e '.[swe]'")
        return False

    if not cfg.get("swe_skip_harness", True):
        # Host: docker CLI. Compose app image: often no CLI, only mounted socket.
        docker_ok = False
        if shutil.which("docker"):
            r = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
            docker_ok = r.returncode == 0
        else:
            sock = Path("/var/run/docker.sock")
            try:
                docker_ok = sock.exists() and sock.is_socket()
            except OSError:
                docker_ok = False
        if not docker_ok:
            print("[prepare] ERROR: Docker is not available (required when swe_skip_harness is false).")
            return False
        print("[prepare] Docker OK (swe_skip_harness=false)")
    else:
        print("[prepare] swe_skip_harness=true — official harness/Docker not required for stub scoring")

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
    """
    Run one benchmark to establish iteration 0 in results.tsv.

    ``prepare_baseline_role`` (default ``gate``):

    - ``gate`` — same split as gating Step 2 (``gate_split``). For SWE-Bench Lite this
      is often the full **test** set (~300 instances): slow/expensive.
    - ``train`` — same split and task selection as default ``python benchmark.py``
      (``split``; uses ``swe_default_task_ids`` for SWE when set, else full split).

    Baseline rows use ``commit=baseline``; ``gating.py`` ignores them when computing
    the best val_score to compare against (avoids mixing train vs test means).
    """
    # Check whether results.tsv already has data rows (baseline already recorded).
    with open(RESULTS_FILE) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    if rows:
        print("[prepare] baseline already recorded — skipping test run")
        return

    from datetime import datetime, timezone
    from benchmark import make_runner_from_config, resolve_mini_task_ids, resolve_swe_run_task_ids

    role = cfg.get("prepare_baseline_role", "gate")
    if role not in ("train", "gate"):
        print(f"[prepare] WARNING: invalid prepare_baseline_role={role!r} — using 'gate'")
        role = "gate"

    split_name = (
        cfg.get("gate_split", "test") if role == "gate" else cfg.get("split", "train")
    )
    print(
        f"[prepare] running baseline benchmark (prepare_baseline_role={role!r} → "
        f"split {split_name!r}; this may take a while)..."
    )
    runner = make_runner_from_config(cfg, role="train" if role == "train" else "gate")
    try:
        if cfg.get("benchmark") == "swe":
            task_ids = resolve_swe_run_task_ids(cfg, None)
        elif cfg.get("mini"):
            task_ids = resolve_mini_task_ids(cfg)
        else:
            task_ids = None
    except ValueError as e:
        print(f"[prepare] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if cfg.get("mini") and task_ids:
        print(f"[prepare] mini baseline: {len(task_ids)} task(s) — {', '.join(task_ids)}")
    elif cfg.get("benchmark") == "swe" and task_ids:
        print(f"[prepare] SWE baseline task_ids ({len(task_ids)}): {', '.join(task_ids)}")
    results = runner.run(task_ids=task_ids)
    val = runner.val_score(results)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(RESULTS_FILE, "a") as f:
        f.write(f"0\t{val:.4f}\tbaseline\t0\t0\t{ts}\n")

    passed = sum(v >= 0.5 for v in results.values())
    print(f"[prepare] baseline val_score={val:.4f} ({passed}/{len(results)} passed) — recorded as iteration 0")


if __name__ == "__main__":
    cfg = load_config()
    if not check_env(cfg):
        sys.exit(1)
    if cfg.get("benchmark", "tau") == "tau":
        if not check_tau2_data():
            sys.exit(1)
    elif cfg.get("benchmark") == "swe":
        if not check_swe_env(cfg):
            sys.exit(1)
    else:
        print(f"[prepare] ERROR: unknown benchmark: {cfg.get('benchmark')!r}")
        sys.exit(1)

    init_workspace()
    run_baseline(cfg)
