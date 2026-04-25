"""Setup helpers for the auto-harness BIRD-Interact integration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

BIRD_INTERACT_REPO = "https://github.com/bird-bench/BIRD-Interact.git"
BIRD_DATASET_REPO = {
    "lite": "https://huggingface.co/datasets/birdsql/bird-interact-lite",
    "full": "https://huggingface.co/datasets/birdsql/bird-interact-full",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_BIRD_ADK_ROOT = os.path.join(REPO_ROOT, "bird_interact_adk")
DEFAULT_BIRD_ADK_DIR = os.path.join(DEFAULT_BIRD_ADK_ROOT, "BIRD-Interact-ADK")


def required_key_for_model(model: str | None) -> str | None:
    if not model:
        return None
    if model.startswith("anthropic/") or model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if (
        model.startswith("openai/")
        or model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("o3")
    ):
        return "OPENAI_API_KEY"
    if model.startswith("gemini"):
        return "GEMINI_API_KEY"
    return None


def fetch_bird_interact_adk(target_root: str = DEFAULT_BIRD_ADK_ROOT) -> bool:
    """Clone BIRD-Interact to target_root/BIRD-Interact-ADK if needed."""
    adk_dir = os.path.join(target_root, "BIRD-Interact-ADK")
    if os.path.exists(os.path.join(adk_dir, "orchestrator", "runner.py")):
        return True

    print(f"[prepare] BIRD-Interact-ADK not found; cloning from {BIRD_INTERACT_REPO} ...")
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
    """Clone the HF bird-interact-<dataset> repo into the ADK directory."""
    target = os.path.join(adk_dir, f"bird-interact-{dataset}")
    if os.path.exists(os.path.join(target, "bird_interact_data.jsonl")):
        return True

    repo = BIRD_DATASET_REPO.get(dataset)
    if not repo:
        print(f"[prepare] ERROR: unknown BIRD dataset '{dataset}' (expected 'lite' or 'full').")
        return False

    print(f"[prepare] BIRD dataset ({dataset}) not found; cloning from {repo} ...")
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

    print(f"[prepare] creating BIRD-Interact-ADK venv at {venv_dir} (may take 1-3 min) ...")
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
    """Start the BIRD Postgres container via docker compose if needed."""
    if shutil.which("docker") is None:
        print("[prepare] ERROR: docker CLI not found.")
        return False

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=bird_interact_postgresql", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
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


def check_bird_gt_merged(data_path: str) -> bool:
    """Return True if any task in the dataset has a non-empty sol_sql."""
    try:
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("sol_sql"):
                    return True
    except Exception:
        return False
    return False


def check_env_bird_interact(cfg: dict) -> bool:
    """Auto-provision BIRD-Interact-ADK if needed, then validate the environment."""
    from benchmark import resolve_bird_adk_dir, resolve_bird_data_path

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

    dataset = cfg.get("dataset", "lite")
    if not cfg.get("bird_data_path") and not fetch_bird_dataset(adk_dir, dataset):
        return False

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

    data_path = resolve_bird_data_path(adk_dir, dataset, cfg.get("bird_data_path"))
    if not os.path.exists(data_path):
        print(f"[prepare] ERROR: BIRD dataset jsonl missing at {data_path}")
        return False
    if not check_bird_gt_merged(data_path):
        print("[prepare] ERROR: BIRD ground truth (sol_sql) is not merged into the dataset.")
        print("          The public dataset ships without gold SQL to prevent data leakage.")
        print("          To obtain GT:")
        print("            1. Email bird.bench25@gmail.com with subject:")
        print(f"               [bird-interact-{dataset} GT&Test Cases]")
        print("            2. Once you receive the GT jsonl, merge it:")
        combine = os.path.join(adk_dir, "scripts", "combine_public_with_gt.py")
        print(f"               python {combine} \\")
        print(f"                 {data_path} \\")
        print("                 <path/to/received_gt.jsonl> \\")
        print("                 /tmp/merged.jsonl")
        print(f"               mv /tmp/merged.jsonl {data_path}")
        print("            3. Re-run: python prepare.py")
        return False
    print(f"[prepare] BIRD data OK (GT merged): {data_path}")

    if not ensure_bird_postgres(adk_dir):
        return False

    dep_check = subprocess.run(
        [
            python_bin,
            "-c",
            "import fastapi, uvicorn, httpx, pydantic_settings; "
            "import google.adk; import shared.config; import shared.llm",
        ],
        cwd=adk_dir,
        capture_output=True,
        text=True,
    )
    if dep_check.returncode != 0:
        stderr = dep_check.stderr.strip() or dep_check.stdout.strip()
        print("[prepare] ERROR: BIRD Python environment is missing required dependencies.")
        if stderr:
            print(f"          {stderr}")
        return False

    required = set()
    for model_key in ("agent_model", "user_model"):
        key = required_key_for_model(cfg.get(model_key))
        if key:
            required.add(key)
    missing = [k for k in sorted(required) if not os.getenv(k)]
    if missing:
        print(f"[prepare] ERROR: missing env vars for bird-interact: {', '.join(missing)}")
        return False

    os.environ.setdefault("BIRD_REPO", adk_dir)
    os.environ.setdefault("BIRD_PYTHON_BIN", python_bin)

    return True
