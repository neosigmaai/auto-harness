"""
Run once before starting an experiment.

Checks required environment variables, validates data/tools for the
configured benchmark, initializes workspace/ files, copies the correct
agent template into agent/agent.py, and runs a baseline benchmark.

Supports tau-bench, terminal-bench, and BIRD-Interact.
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


def _required_key_for_model(model: str | None) -> str | None:
    if not model:
        return None
    if model.startswith("anthropic/") or model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if model.startswith("openai/") or model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "OPENAI_API_KEY"
    if model.startswith("gemini"):
        return "GEMINI_API_KEY"
    return None


# ── BIRD-Interact auto-provisioning ───────────────────────────────────────────

BIRD_INTERACT_REPO = "https://github.com/bird-bench/BIRD-Interact.git"
BIRD_DATASET_REPO = {
    "lite": "https://huggingface.co/datasets/birdsql/bird-interact-lite",
    "full": "https://huggingface.co/datasets/birdsql/bird-interact-full",
}

DEFAULT_BIRD_ADK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bird_interact_adk")
DEFAULT_BIRD_ADK_DIR = os.path.join(DEFAULT_BIRD_ADK_ROOT, "BIRD-Interact-ADK")


def fetch_bird_interact_adk(target_root: str = DEFAULT_BIRD_ADK_ROOT) -> bool:
    """Clone BIRD-Interact to target_root/BIRD-Interact-ADK if not already present."""
    adk_dir = os.path.join(target_root, "BIRD-Interact-ADK")
    if os.path.exists(os.path.join(adk_dir, "orchestrator", "runner.py")):
        return True

    print(f"[prepare] BIRD-Interact-ADK not found — cloning from {BIRD_INTERACT_REPO} ...")
    os.makedirs(target_root, exist_ok=True)
    tmp = os.path.join(target_root, "_bird-interact-tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    try:
        subprocess.run(["git", "clone", "--depth", "1", BIRD_INTERACT_REPO, tmp], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR cloning BIRD-Interact: {e}")
        return False

    src = os.path.join(tmp, "BIRD-Interact-ADK")
    if not os.path.isdir(src):
        print("[prepare] ERROR: expected BIRD-Interact-ADK subdir in cloned repo but not found.")
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    shutil.move(src, adk_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[prepare] BIRD-Interact-ADK ready at {adk_dir}")
    return True


def fetch_bird_dataset(adk_dir: str, dataset: str = "lite") -> bool:
    """Clone the HF bird-interact-<dataset> repo into adk_dir/bird-interact-<dataset>/."""
    target = os.path.join(adk_dir, f"bird-interact-{dataset}")
    if os.path.exists(os.path.join(target, "bird_interact_data.jsonl")):
        return True

    repo = BIRD_DATASET_REPO.get(dataset)
    if not repo:
        print(f"[prepare] ERROR: unknown BIRD dataset '{dataset}' (expected 'lite' or 'full').")
        return False

    print(f"[prepare] BIRD dataset ({dataset}) not found — cloning from {repo} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo, target], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR cloning BIRD dataset: {e}")
        print("          Make sure git-lfs is installed (brew install git-lfs && git lfs install).")
        return False
    print(f"[prepare] BIRD dataset ({dataset}) ready at {target}")
    return True


def ensure_bird_venv(adk_dir: str) -> str | None:
    """Create .venv-adk in adk_dir and install requirements. Returns python bin path."""
    venv_dir = os.path.join(adk_dir, ".venv-adk")
    python_bin = os.path.join(venv_dir, "bin", "python")
    if os.path.exists(python_bin):
        return python_bin

    print(f"[prepare] creating BIRD-Interact-ADK venv at {venv_dir} (may take 1–3 min) ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR creating venv: {e}")
        return None

    requirements = os.path.join(adk_dir, "requirements.txt")
    if not os.path.exists(requirements):
        print(f"[prepare] WARNING: {requirements} not found; skipping dep install.")
        return python_bin

    pip_bin = os.path.join(venv_dir, "bin", "pip")
    try:
        subprocess.run([pip_bin, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_bin, "install", "-r", requirements], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR installing BIRD deps: {e}")
        return None

    print(f"[prepare] BIRD venv ready at {venv_dir}")
    return python_bin


def ensure_bird_postgres(adk_dir: str) -> bool:
    """Start the BIRD Postgres container via docker compose if it isn't already running."""
    if shutil.which("docker") is None:
        print("[prepare] ERROR: docker CLI not found.")
        return False

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=bird_interact_postgresql", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True,
        )
        if "bird_interact_postgresql" in result.stdout:
            return True
    except subprocess.CalledProcessError:
        pass

    compose_file = os.path.join(adk_dir, "docker-compose.yml")
    if not os.path.exists(compose_file):
        print(f"[prepare] ERROR: {compose_file} not found; cannot start Postgres.")
        return False

    print(f"[prepare] starting BIRD Postgres via docker compose from {adk_dir} ...")
    try:
        subprocess.run(["docker", "compose", "up", "-d"], cwd=adk_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[prepare] ERROR starting Postgres: {e}")
        return False
    return True


def _check_bird_gt_merged(data_path: str) -> bool:
    """Return True if any task in the dataset has a non-empty sol_sql (GT merged)."""
    import json as _json
    try:
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = _json.loads(line)
                if entry.get("sol_sql"):
                    return True
    except Exception:
        return False
    return False


def check_env_bird_interact(cfg: dict) -> bool:
    """Auto-provision BIRD-Interact-ADK if needed, then validate the environment."""
    from benchmark import resolve_bird_adk_dir, resolve_bird_data_path

    # 1. Resolve or auto-provision the ADK repo
    bird_repo_override = cfg.get("bird_repo")
    if bird_repo_override:
        try:
            adk_dir = resolve_bird_adk_dir(bird_repo_override)
        except FileNotFoundError as e:
            print(f"[prepare] ERROR: {e}")
            return False
    else:
        if not fetch_bird_interact_adk():
            return False
        adk_dir = DEFAULT_BIRD_ADK_DIR

    # 2. Auto-provision dataset if missing
    dataset = cfg.get("dataset", "lite")
    if not cfg.get("bird_data_path") and not fetch_bird_dataset(adk_dir, dataset):
        return False

    # 3. Resolve Python interpreter (auto-create venv unless overridden)
    python_override = cfg.get("bird_python_bin")
    if python_override:
        python_bin = os.path.abspath(os.path.expanduser(python_override))
        if not os.path.exists(python_bin):
            print(f"[prepare] ERROR: bird_python_bin not found: {python_bin}")
            return False
    else:
        python_bin = ensure_bird_venv(adk_dir)
        if not python_bin:
            return False

    print(f"[prepare] BIRD-Interact-ADK: {adk_dir}")
    print(f"[prepare] BIRD Python: {python_bin}")

    # 4. Verify dataset file exists + has merged GT
    data_path = resolve_bird_data_path(adk_dir, dataset, cfg.get("bird_data_path"))
    if not os.path.exists(data_path):
        print(f"[prepare] ERROR: BIRD dataset jsonl missing at {data_path}")
        return False
    if not _check_bird_gt_merged(data_path):
        print("[prepare] ERROR: BIRD ground truth (sol_sql) is not merged into the dataset.")
        print("          The public dataset ships without gold SQL to prevent data leakage.")
        print("          To obtain GT:")
        print("            1. Email bird.bench25@gmail.com with subject:")
        print(f"               [bird-interact-{dataset} GT&Test Cases]")
        print("            2. Once you receive the GT jsonl, merge it:")
        combine = os.path.join(adk_dir, "scripts", "combine_public_with_gt.py")
        print(f"               python {combine} \\")
        print(f"                 {data_path} \\")
        print(f"                 <path/to/received_gt.jsonl> \\")
        print("                 /tmp/merged.jsonl")
        print(f"               mv /tmp/merged.jsonl {data_path}")
        print("            3. Re-run: python prepare.py")
        return False
    print(f"[prepare] BIRD data OK (GT merged): {data_path}")

    # 5. Ensure Postgres is running
    if not ensure_bird_postgres(adk_dir):
        return False

    # 6. Check Python deps in the chosen interpreter
    dep_check = subprocess.run(
        [python_bin, "-c", "import fastapi, uvicorn, httpx, pydantic_settings; import google.adk"],
        capture_output=True, text=True,
    )
    if dep_check.returncode != 0:
        stderr = dep_check.stderr.strip() or dep_check.stdout.strip()
        print("[prepare] ERROR: BIRD Python environment is missing required dependencies.")
        if stderr:
            print(f"          {stderr}")
        return False

    # 7. Check required LLM API keys based on configured models
    required = set()
    for model_key in ("agent_model", "user_model"):
        key = _required_key_for_model(cfg.get(model_key))
        if key:
            required.add(key)
    missing = [k for k in sorted(required) if not os.getenv(k)]
    if missing:
        print(f"[prepare] ERROR: missing env vars for bird-interact: {', '.join(missing)}")
        return False

    # 8. Expose resolved paths to downstream callers (benchmark.py, gating.py) via env
    os.environ.setdefault("BIRD_REPO", adk_dir)
    os.environ.setdefault("BIRD_PYTHON_BIN", python_bin)

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
        "bird-interact": "agent/templates/bird_interact.py",
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
        "bird-interact": "program_templates/bird_interact.md",
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
BIRD_SPLIT_FILE = "bird_data/task_split.json"


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


def generate_bird_interact_split(results: dict[str, float], seed: int = 42) -> None:
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

    os.makedirs("bird_data", exist_ok=True)
    with open(BIRD_SPLIT_FILE, "w") as f:
        json.dump({
            "train": train,
            "test": test,
            "metadata": {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "total_tasks": len(results),
                "seed": seed,
            },
        }, f, indent=2)
    print(f"[prepare] BIRD task split created: {len(train)} train, {len(test)} test")


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
    elif benchmark == "bird-interact":
        from benchmark import BirdInteractRunner

        if not os.path.exists(BIRD_SPLIT_FILE):
            print("[prepare] running ALL bird-interact tasks to generate train/test split...")
            all_runner = BirdInteractRunner(
                bird_repo=cfg.get("bird_repo"),
                bird_python_bin=cfg.get("bird_python_bin"),
                split=None,
                mode=cfg.get("mode", "a-interact"),
                dataset=cfg.get("dataset", "lite"),
                data_path=cfg.get("bird_data_path"),
                agent_model=cfg.get("agent_model"),
                user_model=cfg.get("user_model"),
                patience=cfg.get("patience", 3),
                n_concurrent=cfg.get("max_concurrency", 3),
                per_task_timeout=cfg.get("per_task_timeout", 1800),
                jobs_dir="workspace/bird_runs/all",
                system_agent_port=cfg.get("system_agent_port", 6100),
                user_sim_port=cfg.get("user_sim_port", 6101),
                db_env_port=cfg.get("db_env_port", 6102),
                pg_host=cfg.get("pg_host"),
                pg_port=cfg.get("pg_port"),
                pg_user=cfg.get("pg_user"),
                pg_password=cfg.get("pg_password"),
            )
            all_results = all_runner.run()

            actual_results = {k: v for k, v in all_results.items() if v is not None}
            infra_errors = [k for k, v in all_results.items() if v is None]
            if infra_errors:
                print(f"[prepare] WARNING: {len(infra_errors)} BIRD task(s) had infra errors and are "
                      f"permanently excluded from the train/test split: {infra_errors}")
                print(f"          To include them, delete {BIRD_SPLIT_FILE} and re-run prepare.py.")

            generate_bird_interact_split(actual_results)

            with open(BIRD_SPLIT_FILE) as f:
                split = json.load(f)
            test_results = {k: actual_results.get(k, 0.0) for k in split["test"]}
            val = sum(test_results.values()) / len(test_results) if test_results else 0.0
        else:
            runner = BirdInteractRunner(
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
            user_model=cfg.get("user_model"),
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
    elif benchmark == "bird-interact":
        if not check_env_bird_interact(cfg):
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
