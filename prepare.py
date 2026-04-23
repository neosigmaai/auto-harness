"""
Run once before starting an experiment.

Checks required environment variables, validates data/tools for the
configured benchmark, initializes workspace/ files, copies the correct
agent template into agent/agent.py, and runs a baseline benchmark.

Supports tau-bench, terminal-bench, and webarena (full + smoke modes).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import _env  # noqa: F401  — auto-loads .env into os.environ on import
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


# ── WebArena environment ──────────────────────────────────────────────────────

WEBARENA_REPO_URL = "https://github.com/web-arena-x/webarena.git"
WEBARENA_LOCAL = "webarena_repo"
WEBARENA_SPLIT_FILE = "webarena_data/task_split.json"
WEBARENA_CANDIDATES_FILE = "webarena_data/reddit_candidate_tasks.json"

# Smoke mode: 5 hand-crafted tasks against the local Flask fixture in
# fixtures/webarena_smoke/app.py. Keep the IDs in sync with the JSONs under
# webarena_data/smoke_tasks/.
WEBARENA_SMOKE_TASKS_DIR = "webarena_data/smoke_tasks"
WEBARENA_SMOKE_FIXTURE_URL = "http://localhost:9999"
SMOKE_SPLIT = {
    # Mix of passing and intentionally-failing tasks so the optimization loop
    # has room to improve the agent. 10006/10008 (train) and 10007 (test)
    # require clicking into the post detail page — a coding-agent-tractable
    # failure mode fixable via AGENT_INSTRUCTION edits in agent/agent.py.
    "train": ["10001", "10003", "10005", "10006", "10008"],
    "test": ["10002", "10004", "10007"],
}
# Pin to a known-good commit so prepare.py is reproducible. Main has historically
# been unstable; this is the head of main at integration time. Set the env var
# WEBARENA_COMMIT to override for debugging.
WEBARENA_PIN = os.environ.get("WEBARENA_COMMIT", "")  # empty → default branch

# Env var per supported site. Extend here when enabling more sites.
WEBARENA_SITE_ENV_VARS = {
    "reddit": "REDDIT",
    "shopping": "SHOPPING",
    "shopping_admin": "SHOPPING_ADMIN",
    "gitlab": "GITLAB",
    "wikipedia": "WIKIPEDIA",
    "map": "MAP",
}


def _smoke_remediation() -> str:
    return (
        f"  1. In a separate terminal, start the fixture:\n"
        f"     python fixtures/webarena_smoke/app.py\n"
        f"  2. Confirm it is serving:\n"
        f"     curl -s {WEBARENA_SMOKE_FIXTURE_URL}/ | head\n"
        f"  3. Re-run: python prepare.py"
    )


def _reddit_remediation(url_env: str) -> str:
    # WebArena's Reddit/Postmill image is NOT published on Docker Hub. It ships
    # as a ~4 GB tarball that you docker-load locally. See:
    #   https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md
    return (
        f"  1. Download the Reddit/Postmill image tarball (~4 GB):\n"
        f"     curl -LO http://metis.lti.cs.cmu.edu/webarena-images/"
        f"postmill-populated-exposed-withimg.tar\n"
        f"     (mirror: https://archive.org/download/webarena-env-forum-image)\n"
        f"  2. Load it into Docker:\n"
        f"     docker load --input postmill-populated-exposed-withimg.tar\n"
        f"  3. Start the container:\n"
        f"     docker run --name forum -p 9999:80 -d "
        f"postmill-populated-exposed-withimg\n"
        f"  4. Export the URL for the harness:\n"
        f"     export {url_env}=http://localhost:9999\n"
        f"  5. Re-run: python prepare.py"
    )


def _probe_url(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """HEAD/GET the URL to verify reachability. Returns (ok, message)."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # Reddit/Postmill returns 200 on / — any HTTP response proves the
        # container is up. 4xx/5xx still counts as "reachable".
        return True, f"HTTP {e.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"


def check_env_webarena(cfg: dict) -> bool:
    """Check environment for webarena. Returns True iff every preflight passes."""
    site = cfg.get("site", "reddit")
    url_env = WEBARENA_SITE_ENV_VARS.get(site)
    if url_env is None:
        print(f"[prepare] ERROR: unsupported webarena site '{site}'. "
              f"Supported: {', '.join(sorted(WEBARENA_SITE_ENV_VARS))}")
        return False

    smoke = cfg.get("smoke", False)
    ok = True

    # 1. LLM key (required in all modes — agent still makes real LLM calls).
    model = cfg.get("agent_model", "gpt-4o-mini")
    if model.startswith("gemini"):
        llm_key = "GEMINI_API_KEY"
    elif model.startswith("claude"):
        llm_key = "ANTHROPIC_API_KEY"
    else:
        llm_key = "OPENAI_API_KEY"
    if not os.getenv(llm_key):
        print(f"[prepare] ERROR: missing env var {llm_key} for model={model!r}")
        ok = False

    # 2. Site URL — smoke mode pins it to the local fixture; full mode reads env.
    if smoke:
        # Ignore whatever the user put in $REDDIT — smoke tasks hardcode
        # http://localhost:9999 in webarena_data/smoke_tasks/*.json.
        reachable, msg = _probe_url(WEBARENA_SMOKE_FIXTURE_URL)
        if reachable:
            print(f"[prepare] webarena smoke fixture OK: {WEBARENA_SMOKE_FIXTURE_URL} ({msg})")
        else:
            print(f"[prepare] ERROR: webarena smoke fixture not reachable at "
                  f"{WEBARENA_SMOKE_FIXTURE_URL}: {msg}")
            print(_smoke_remediation())
            ok = False
    else:
        site_url = os.getenv(url_env)
        if not site_url:
            print(f"[prepare] ERROR: {url_env} is not set. WebArena requires a "
                  f"reachable {site} instance.")
            if site == "reddit":
                print(_reddit_remediation(url_env))
            ok = False
        else:
            reachable, msg = _probe_url(site_url)
            if reachable:
                print(f"[prepare] webarena site OK: {url_env}={site_url} ({msg})")
            else:
                print(f"[prepare] ERROR: {url_env}={site_url} is not reachable: {msg}")
                if site == "reddit":
                    print(_reddit_remediation(url_env))
                ok = False

    # 3. Playwright importable + browsers installed.
    try:
        import importlib

        importlib.import_module("playwright")
    except ImportError:
        print("[prepare] ERROR: `playwright` is not installed. "
              "Install it with: pip install playwright && playwright install chromium")
        ok = False
    else:
        # Playwright browsers live under ~/.cache/ms-playwright/ on Linux and
        # ~/Library/Caches/ms-playwright/ on macOS.
        candidates = [
            os.path.expanduser("~/.cache/ms-playwright"),
            os.path.expanduser("~/Library/Caches/ms-playwright"),
        ]
        has_chromium = any(
            os.path.isdir(c) and any(name.startswith("chromium") for name in os.listdir(c))
            for c in candidates
        )
        if not has_chromium:
            # Smoke mode may run before deps install; fail loudly either way —
            # the agent can't drive a headless browser without Chromium.
            msg = ("Playwright Chromium not installed. "
                   "Run: playwright install chromium")
            if smoke:
                print(f"[prepare] WARNING: {msg}")
            else:
                print(f"[prepare] ERROR: {msg}")
                ok = False

    # 4. webarena_repo/ exists or can be cloned.
    if not os.path.isdir(WEBARENA_LOCAL):
        print(f"[prepare] webarena repo not found at {WEBARENA_LOCAL}/ — will clone.")
        if shutil.which("git") is None:
            print("[prepare] ERROR: `git` CLI not found; required to clone WebArena.")
            ok = False

    # 5. Task-source files present. Smoke mode uses webarena_data/smoke_tasks/;
    #    full mode uses the curated candidate list we ship.
    if smoke:
        missing = [
            tid for tid in (SMOKE_SPLIT["train"] + SMOKE_SPLIT["test"])
            if not os.path.exists(os.path.join(WEBARENA_SMOKE_TASKS_DIR, f"{tid}.json"))
        ]
        if missing:
            print(f"[prepare] ERROR: smoke task configs missing under "
                  f"{WEBARENA_SMOKE_TASKS_DIR}/: {missing}")
            ok = False
    else:
        if not os.path.exists(WEBARENA_CANDIDATES_FILE):
            print(f"[prepare] ERROR: {WEBARENA_CANDIDATES_FILE} missing from repo.")
            ok = False

    return ok


def fetch_webarena_repo() -> bool:
    """Shallow-clone WebArena into `webarena_repo/` if missing. Pin to a commit."""
    if os.path.isdir(WEBARENA_LOCAL):
        return True
    print(f"[prepare] cloning WebArena into {WEBARENA_LOCAL}/ ...")
    try:
        if WEBARENA_PIN:
            subprocess.run(
                ["git", "clone", WEBARENA_REPO_URL, WEBARENA_LOCAL], check=True
            )
            subprocess.run(
                ["git", "-C", WEBARENA_LOCAL, "checkout", WEBARENA_PIN],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 WEBARENA_REPO_URL, WEBARENA_LOCAL],
                check=True,
            )
        print(f"[prepare] webarena repo ready at {WEBARENA_LOCAL}/")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[prepare] ERROR: failed to clone WebArena: {e}")
        print(f"          Manual fix: git clone {WEBARENA_REPO_URL} {WEBARENA_LOCAL}")
        return False


def install_webarena_requirements() -> bool:
    """pip install WebArena's requirements.txt into the current environment."""
    req = os.path.join(WEBARENA_LOCAL, "requirements.txt")
    if not os.path.exists(req):
        print(f"[prepare] WARNING: {req} not found; skipping webarena deps install.")
        return True
    print(f"[prepare] installing webarena deps from {req} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req],
            check=True,
        )
        # Ensure playwright browsers are present.
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[prepare] ERROR: failed to install webarena deps: {e}")
        print(f"          Manual fix: pip install -r {req} && playwright install chromium")
        return False


def materialize_webarena_configs(cfg: dict) -> bool:
    """
    Expand `config_files/test.raw.json` into per-task `<id>.json` files.

    WebArena ships only the raw template (with `__REDDIT__`, `__SHOPPING__`, ...
    placeholders) and leaves generation to `scripts/generate_test_data.py`.
    That script asserts every site URL is set, but our starter subset only
    needs one real URL — fill unused sites with a harmless placeholder so the
    assertion passes. We only run pure-site tasks in the split, so placeholder
    URLs are never contacted.
    """
    site = cfg.get("site", "reddit")
    per_task_dir_has_files = any(
        name.endswith(".json") and name != "test.raw.json" and name != "test.json"
        and not os.path.isdir(os.path.join(WEBARENA_LOCAL, "config_files", name))
        for name in os.listdir(os.path.join(WEBARENA_LOCAL, "config_files"))
    )
    if per_task_dir_has_files:
        print("[prepare] webarena per-task config_files already expanded — skipping")
        return True

    raw_path = os.path.join(WEBARENA_LOCAL, "config_files", "test.raw.json")
    if not os.path.exists(raw_path):
        print(f"[prepare] ERROR: {raw_path} missing in webarena clone")
        return False

    # Inline equivalent of webarena's scripts/generate_test_data.py. Doing it in
    # pure Python avoids pulling in beartype/playwright/... just to expand JSON.
    placeholder = "http://unused.local"
    subs = {
        f"__{var}__": os.environ.get(var) or placeholder
        for var in list(WEBARENA_SITE_ENV_VARS.values()) + ["HOMEPAGE"]
    }
    print(f"[prepare] expanding webarena config_files (site={site}) ...")
    try:
        with open(raw_path) as f:
            raw = f.read()
        for token, url in subs.items():
            raw = raw.replace(token, url)
        tasks = json.loads(raw)
        out_dir = os.path.join(WEBARENA_LOCAL, "config_files")
        for idx, item in enumerate(tasks):
            out_path = os.path.join(out_dir, f"{idx}.json")
            # Preserve task_id already set inside the item if present, else use idx.
            if "task_id" not in item:
                item["task_id"] = idx
            with open(out_path, "w") as f:
                json.dump(item, f, indent=2)
        print(f"[prepare] wrote {len(tasks)} per-task config files to {out_dir}/")
    except (OSError, json.JSONDecodeError) as e:
        print(f"[prepare] ERROR: failed to expand test.raw.json: {e}")
        return False
    return True


def _is_site_only_task(config_path: str, site: str) -> bool:
    """A task belongs to the starter subset iff its `sites` is exactly [site]."""
    try:
        with open(config_path) as f:
            c = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    sites = c.get("sites") or []
    return len(sites) == 1 and sites[0] == site


def write_smoke_webarena_split() -> bool:
    """Overwrite webarena_data/task_split.json with the hardcoded smoke split.

    Smoke is a fixed, tiny, reproducible set; no shuffling, no validation against
    webarena_repo (those tasks live in webarena_data/smoke_tasks/).
    """
    os.makedirs(os.path.dirname(WEBARENA_SPLIT_FILE), exist_ok=True)
    payload = {
        "train": list(SMOKE_SPLIT["train"]),
        "test": list(SMOKE_SPLIT["test"]),
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "smoke",
            "total_tasks": len(SMOKE_SPLIT["train"]) + len(SMOKE_SPLIT["test"]),
            "source": WEBARENA_SMOKE_TASKS_DIR,
        },
    }
    with open(WEBARENA_SPLIT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[prepare] webarena smoke split: "
          f"{len(payload['train'])} train, {len(payload['test'])} test "
          f"→ {WEBARENA_SPLIT_FILE}")
    return True


def generate_webarena_split(cfg: dict, seed: int = 42) -> bool:
    """Pick 10 single-site tasks (7 train / 3 test) from the curated candidate list.
    Validates every ID against webarena_repo/config_files/ and filters out
    multi-site / missing tasks."""
    import random

    if os.path.exists(WEBARENA_SPLIT_FILE):
        print(f"[prepare] {WEBARENA_SPLIT_FILE} already exists — skipping split generation")
        return True

    with open(WEBARENA_CANDIDATES_FILE) as f:
        candidates = json.load(f)
    site = cfg.get("site", candidates.get("site", "reddit"))
    candidate_ids = candidates.get("candidate_task_ids", [])

    config_dir = os.path.join(WEBARENA_LOCAL, "config_files")
    if not os.path.isdir(config_dir):
        print(f"[prepare] ERROR: {config_dir} missing — cannot generate split")
        return False

    valid: list[str] = []
    skipped: list[tuple[str, str]] = []
    for tid in candidate_ids:
        path = os.path.join(config_dir, f"{tid}.json")
        if not os.path.exists(path):
            skipped.append((tid, "config_files/<id>.json missing"))
            continue
        if not _is_site_only_task(path, site):
            skipped.append((tid, f"not a pure {site} task"))
            continue
        valid.append(tid)

    if skipped:
        print(f"[prepare] filtered {len(skipped)} candidate task(s): "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}")

    if len(valid) < 4:
        print(f"[prepare] ERROR: only {len(valid)} valid {site} candidate tasks "
              f"(need >= 4). Extend {WEBARENA_CANDIDATES_FILE} or change site.")
        return False

    target_total = min(10, len(valid))
    train_n = max(2, int(round(target_total * 0.7)))
    test_n = max(1, target_total - train_n)

    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:train_n + test_n]
    train = sorted(selected[:train_n], key=lambda x: int(x))
    test = sorted(selected[train_n:train_n + test_n], key=lambda x: int(x))

    os.makedirs(os.path.dirname(WEBARENA_SPLIT_FILE), exist_ok=True)
    with open(WEBARENA_SPLIT_FILE, "w") as f:
        json.dump({
            "train": train,
            "test": test,
            "metadata": {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "site": site,
                "total_tasks": len(train) + len(test),
                "seed": seed,
                "source": WEBARENA_CANDIDATES_FILE,
            },
        }, f, indent=2)
    print(f"[prepare] webarena split: {len(train)} train, {len(test)} test "
          f"(site={site}) → {WEBARENA_SPLIT_FILE}")
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
        "webarena": "agent/templates/webarena.py",
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
        "webarena": "program_templates/webarena.md",
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
    elif benchmark == "webarena":
        from benchmark import WebArenaRunner
        runner = WebArenaRunner(
            agent_model=cfg.get("agent_model"),
            split=cfg.get("gate_split", "test"),
            site=cfg.get("site", "reddit"),
            n_concurrent=cfg.get("max_concurrency", 1),
            max_steps=cfg.get("max_steps", 30),
            per_task_timeout=cfg.get("per_task_timeout", 300),
            webarena_repo=cfg.get("webarena_repo", WEBARENA_LOCAL),
            reasoning_effort=cfg.get("reasoning_effort"),
            smoke=cfg.get("smoke", False),
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
    elif benchmark == "webarena":
        if not check_env_webarena(cfg):
            sys.exit(1)
        # Driver still imports from webarena_repo/ even in smoke mode — clone
        # if the user hasn't already.
        if not fetch_webarena_repo():
            sys.exit(1)
        # WebArena's string_match evaluator calls nltk.word_tokenize which needs
        # 'punkt_tab'. Download silently — cheap and avoids first-run infra errors.
        try:
            import nltk  # type: ignore
            for pkg in ("punkt_tab", "punkt"):
                try:
                    nltk.data.find(f"tokenizers/{pkg}")
                except LookupError:
                    nltk.download(pkg, quiet=True)
        except ImportError:
            pass  # Will be installed via install_webarena_requirements().
        if cfg.get("smoke", False):
            # Smoke mode: skip heavy dep install + raw-config expansion. Tasks
            # are hand-authored in webarena_data/smoke_tasks/ and the fixture
            # replaces Postmill.
            print("[prepare] smoke mode: skipping webarena deps + config expansion")
            if not write_smoke_webarena_split():
                sys.exit(1)
        else:
            if cfg.get("install_webarena_deps", True):
                if not install_webarena_requirements():
                    sys.exit(1)
            if not materialize_webarena_configs(cfg):
                sys.exit(1)
            if not generate_webarena_split(cfg):
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
