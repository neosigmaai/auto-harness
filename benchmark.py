"""
Benchmark execution layer.

BenchmarkRunner: abstract base class — subclass to plug in your own benchmark.
TauBenchRunner:  implementation for tau-bench (https://github.com/sierra-research/tau2-bench).
TerminalBenchRunner: implementation for Terminal-Bench 2.0 via Harbor framework.
BirdInteractRunner: implementation for BIRD-Interact via external BIRD-Interact-ADK.
BFCLRunner: implementation for BFCL (Berkeley Function-Calling Leaderboard) via
            the pinned `bfcl-eval` package and a subprocess shim.

Both gating.py and the coding agent call this directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
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
        user_model: str | None = None,
    ):
        self.domain = domain
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.max_concurrency = max_concurrency
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.user_model = user_model or self.agent_model

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
            llm_user=self.user_model,
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


def resolve_bird_adk_dir(configured_path: str | None = None) -> str:
    """Resolve the BIRD-Interact-ADK directory from a repo root or direct path."""
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.getenv("BIRD_REPO", ""),
        # auto-provisioned location (prepare.py clones here)
        os.path.join(here, "bird_interact_adk", "BIRD-Interact-ADK"),
        os.path.join(here, "bird_interact_adk"),
        # sibling-repo fallback (advanced users)
        os.path.join(here, "..", "BIRD-Interact"),
        os.path.join(here, "..", "BIRD-Interact", "BIRD-Interact-ADK"),
        os.path.join(here, "BIRD-Interact-ADK"),
    ])

    for raw in candidates:
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        direct = os.path.join(path, "orchestrator", "runner.py")
        nested = os.path.join(path, "BIRD-Interact-ADK", "orchestrator", "runner.py")
        if os.path.exists(direct):
            return path
        if os.path.exists(nested):
            return os.path.join(path, "BIRD-Interact-ADK")

    raise FileNotFoundError(
        "Could not locate BIRD-Interact-ADK. "
        "Run `python prepare.py` to auto-provision it into ./bird_interact_adk/, "
        "or set bird_repo in experiment_config.yaml to point at an existing install."
    )


def resolve_bird_python_bin(adk_dir: str, configured_python: str | None = None) -> str | None:
    """Pick a Python interpreter that has the BIRD-Interact-ADK dependencies installed."""
    candidates = []
    if configured_python:
        candidates.append(configured_python)
    candidates.extend([
        os.getenv("BIRD_PYTHON_BIN", ""),
        os.path.join(adk_dir, ".venv-adk", "bin", "python"),
        os.path.join(adk_dir, ".venv", "bin", "python"),
        os.path.join(adk_dir, ".conda-py310", "bin", "python"),
        shutil.which("python3") or "",
        shutil.which("python") or "",
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_bird_data_path(
    adk_dir: str,
    dataset: str = "lite",
    configured_data_path: str | None = None,
) -> str:
    """Resolve the bird_interact_data.jsonl path."""
    if configured_data_path:
        return os.path.abspath(os.path.expanduser(configured_data_path))
    return os.path.join(adk_dir, f"bird-interact-{dataset}", "bird_interact_data.jsonl")


class BirdInteractRunner(BenchmarkRunner):
    """Runner for BIRD-Interact via the external BIRD-Interact-ADK repo."""

    SPLIT_FILE = "bird_data/task_split.json"

    def __init__(
        self,
        bird_repo: str | None = None,
        bird_python_bin: str | None = None,
        split: str | None = "train",
        mode: str = "a-interact",
        dataset: str = "lite",
        data_path: str | None = None,
        agent_model: str | None = None,
        user_model: str | None = None,
        patience: int = 3,
        n_concurrent: int = 3,
        per_task_timeout: int = 1800,
        jobs_dir: str = "workspace/bird_runs",
        system_agent_port: int = 6100,
        user_sim_port: int = 6101,
        db_env_port: int = 6102,
        pg_host: str | None = None,
        pg_port: int | None = None,
        pg_user: str | None = None,
        pg_password: str | None = None,
    ):
        self.adk_dir = resolve_bird_adk_dir(bird_repo)
        self.python_bin = resolve_bird_python_bin(self.adk_dir, bird_python_bin)
        if not self.python_bin:
            raise FileNotFoundError(
                "Could not find a Python interpreter for BIRD-Interact-ADK. "
                "Set bird_python_bin in experiment_config.yaml."
            )

        self.split = split
        self.mode = mode
        self.dataset = dataset
        self.data_path = resolve_bird_data_path(self.adk_dir, dataset, data_path)
        self.agent_model = agent_model
        self.user_model = user_model
        self.patience = patience
        self.n_concurrent = n_concurrent
        self.per_task_timeout = per_task_timeout
        self.jobs_dir = jobs_dir
        self.system_agent_port = system_agent_port
        self.user_sim_port = user_sim_port
        self.db_env_port = db_env_port
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password

    def _load_split_tasks(self) -> list[str] | None:
        if self.split is None:
            return None

        if not os.path.exists(self.SPLIT_FILE):
            raise FileNotFoundError(f"{self.SPLIT_FILE} not found. Run prepare.py first.")

        with open(self.SPLIT_FILE) as f:
            splits = json.load(f)
        tasks = splits.get(self.split)
        if tasks is None:
            raise ValueError(
                f"Split '{self.split}' not found in {self.SPLIT_FILE}. "
                f"Available: {list(splits.keys())}"
            )
        return tasks

    def _load_tasks(self) -> list[dict]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"BIRD-Interact dataset not found at {self.data_path}. "
                "Download the dataset and set bird_data_path or bird_repo in experiment_config.yaml."
            )

        tasks = []
        with open(self.data_path) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks

    def _select_tasks(self, task_ids: list[str]) -> list[dict]:
        task_set = {str(tid) for tid in task_ids}
        all_tasks = self._load_tasks()
        selected = [task for task in all_tasks if str(task.get("instance_id")) in task_set]
        found = {str(task.get("instance_id")) for task in selected}
        missing = [tid for tid in task_ids if str(tid) not in found]
        if missing:
            raise KeyError(f"BIRD-Interact task(s) not found: {missing}")

        order = {str(tid): i for i, tid in enumerate(task_ids)}
        selected.sort(key=lambda task: order[str(task.get("instance_id"))])
        return selected

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        auto_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = auto_root + os.pathsep + self.adk_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["NO_PROXY"] = env.get("NO_PROXY", "127.0.0.1,localhost")
        env["no_proxy"] = env.get("no_proxy", env["NO_PROXY"])
        env["SYSTEM_AGENT_PORT"] = str(self.system_agent_port)
        env["USER_SIM_PORT"] = str(self.user_sim_port)
        env["DB_ENV_PORT"] = str(self.db_env_port)
        env["DATASET"] = self.dataset
        env["PATIENCE"] = str(self.patience)
        env["PYTHONUNBUFFERED"] = "1"

        if self.agent_model:
            env["SYSTEM_AGENT_MODEL"] = self.agent_model
        if self.user_model:
            env["USER_SIM_MODEL"] = self.user_model
        if env.get("OPENAI_API_KEY") and not env.get("LITELLM_API_KEY"):
            env["LITELLM_API_KEY"] = env["OPENAI_API_KEY"]
        if env.get("OPENAI_API_BASE") and not env.get("LITELLM_API_BASE"):
            env["LITELLM_API_BASE"] = env["OPENAI_API_BASE"]
        # Homebrew installs PostgreSQL client tools in a keg-only prefix.
        for libpq_bin in (
            "/opt/homebrew/opt/libpq/bin",
            "/usr/local/opt/libpq/bin",
        ):
            if os.path.isdir(libpq_bin):
                env["PATH"] = libpq_bin + os.pathsep + env.get("PATH", "")
                break
        if self.pg_host:
            env["PG_HOST"] = self.pg_host
        if self.pg_port is not None:
            env["PG_PORT"] = str(self.pg_port)
        if self.pg_user:
            env["PG_USER"] = self.pg_user
        if self.pg_password:
            env["PG_PASSWORD"] = self.pg_password

        return env

    def _start_service(
        self,
        module: str,
        port: int,
        log_name: str,
        env: dict[str, str],
    ) -> tuple[subprocess.Popen, object]:
        os.makedirs(self.jobs_dir, exist_ok=True)
        log_dir = os.path.join(self.jobs_dir, "service_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name)
        log_file = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                [
                    self.python_bin,
                    "-m",
                    "uvicorn",
                    f"{module}:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                cwd=self.adk_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise
        return proc, log_file

    def _wait_for_health(self, port: int, timeout_sec: int = 30) -> None:
        deadline = time.time() + timeout_sec
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                time.sleep(1)
        raise RuntimeError(f"Timed out waiting for health endpoint: {url}")

    def _start_services(self) -> list[tuple[subprocess.Popen, object]]:
        env = self._base_env()
        services = [
            ("agent.helpers.bird_interact.bird_service", self.system_agent_port, "system_agent.log"),
            ("user_simulator.server", self.user_sim_port, "user_simulator.log"),
            ("db_environment.server", self.db_env_port, "db_environment.log"),
        ]

        started: list[tuple[subprocess.Popen, object]] = []
        try:
            for module, port, log_name in services:
                proc, log_file = self._start_service(module, port, log_name, env)
                started.append((proc, log_file))
                self._wait_for_health(port)
            return started
        except Exception:
            self._stop_services(started)
            raise

    def _stop_services(self, services: list[tuple[subprocess.Popen, object]]) -> None:
        for proc, log_file in reversed(services):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            finally:
                try:
                    log_file.close()
                except Exception:
                    pass

    def _copy_train_traces(self, results: list[dict]) -> None:
        latest_dir = os.path.join("workspace", "traces", "latest")
        baseline_dir = os.path.join("workspace", "traces", "baseline")
        os.makedirs(latest_dir, exist_ok=True)

        for item in results:
            task_name = str(item.get("instance_id") or item.get("task_id") or "")
            if not task_name:
                continue

            trace_payload = {
                "dialogue_history": item.get("dialogue_history", []),
                "tool_trajectory": item.get("tool_trajectory", []),
                "adk_events": item.get("adk_events", []),
                "final_response": item.get("final_response", ""),
            }

            dest = os.path.join(latest_dir, task_name)
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "trace.json"), "w") as f:
                json.dump(trace_payload, f, indent=2)
            with open(os.path.join(dest, "result.json"), "w") as f:
                json.dump(item, f, indent=2, default=str)

            base_dest = os.path.join(baseline_dir, task_name)
            if not os.path.exists(base_dest):
                os.makedirs(base_dest, exist_ok=True)
                with open(os.path.join(base_dest, "trace.json"), "w") as f:
                    json.dump(trace_payload, f, indent=2)
                with open(os.path.join(base_dest, "result.json"), "w") as f:
                    json.dump(item, f, indent=2, default=str)

        print("[benchmark] traces: latest/ updated, baseline/ preserved")

    def run(self, task_ids: list[str] | None = None) -> dict[str, float | None]:
        selected_ids = task_ids if task_ids is not None else self._load_split_tasks()
        selected_tasks = None if selected_ids is None else self._select_tasks(selected_ids)

        os.makedirs(self.jobs_dir, exist_ok=True)
        input_path = None
        if selected_tasks is not None:
            tmp_input = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                prefix="bird_input_",
                dir=self.jobs_dir,
                delete=False,
            )
            with tmp_input as f:
                for task in selected_tasks:
                    f.write(json.dumps(task) + "\n")
            input_path = tmp_input.name
            n_tasks = len(selected_tasks)
        else:
            input_path = self.data_path
            n_tasks = 600 if self.dataset == "full" else 300

        tmp_output = tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="bird_output_",
            dir=self.jobs_dir,
            delete=False,
        )
        tmp_output.close()
        output_path = tmp_output.name

        def _cleanup_temp_files() -> None:
            if selected_tasks is not None and input_path:
                try:
                    os.remove(input_path)
                except OSError:
                    pass
            try:
                os.remove(output_path)
            except OSError:
                pass

        env = self._base_env()
        concurrency = max(1, self.n_concurrent)
        timeout_sec = max(
            600,
            int((n_tasks / concurrency) * self.per_task_timeout) + 300,
        )

        print(
            f"[benchmark] running {n_tasks} bird-interact tasks "
            f"(mode={self.mode}, dataset={self.dataset}, n={concurrency}, "
            f"subprocess_timeout={timeout_sec}s)"
        )

        services = self._start_services()
        try:
            cmd = [
                self.python_bin,
                "-m",
                "orchestrator.runner",
                "--mode",
                self.mode,
                "--data",
                input_path,
                "--output",
                output_path,
                "--concurrency",
                str(concurrency),
            ]
            result = subprocess.run(
                cmd,
                cwd=self.adk_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[benchmark] WARNING: BIRD-Interact run timed out after {timeout_sec}s")
        finally:
            self._stop_services(services)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("[benchmark] ERROR: no BIRD-Interact output file produced")
            _cleanup_temp_files()
            return {}

        try:
            with open(output_path) as f:
                output = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[benchmark] ERROR: failed to parse BIRD-Interact output: {e}")
            _cleanup_temp_files()
            return {}

        raw_results = output.get("results", [])
        results: dict[str, float | None] = {}
        for item in raw_results:
            task_name = str(item.get("instance_id") or item.get("task_id") or "")
            if not task_name:
                continue
            if item.get("error"):
                results[task_name] = None
            else:
                results[task_name] = float(item.get("total_reward", 0.0))

        if selected_ids is not None:
            for tid in selected_ids:
                results.setdefault(str(tid), None)

        if self.split == "train":
            self._copy_train_traces(raw_results)

        # Prune stale temp outputs from prior runs.
        for old in os.listdir(self.jobs_dir):
            if old.startswith("bird_input_") or old.startswith("bird_output_"):
                old_path = os.path.join(self.jobs_dir, old)
                if old_path not in {input_path, output_path}:
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

        _cleanup_temp_files()

        return results


class BFCLRunner(BenchmarkRunner):
    """
    Runner for BFCL (Berkeley Function-Calling Leaderboard) via `bfcl-eval`.

    Each `run()` invocation creates an isolated `BFCL_PROJECT_ROOT` under
    `workspace/bfcl_runs/<split>/<timestamp>/`, writes a fresh
    `test_case_ids_to_generate.json`, and shells out to the
    `agent.helpers.bfcl.run` shim for `generate` and `evaluate`. Per-task
    rewards are reconstructed from the result JSONL (generated IDs) minus the
    score JSONL (failure IDs).

    Anti-cheating:
    - For `split == "train"`, per-task result/score entries are copied to
      `workspace/traces/latest/<task_id>/` for the coding agent to inspect.
    - For other splits, the entire raw run directory is removed after parsing
      so the coding agent cannot read test failure details.

    Usage:
        runner = BFCLRunner(category="multi_turn_base", split="train")
        results = runner.run()                                          # full split
        results = runner.run(task_ids=["multi_turn_base_0"])            # specific tasks
    """

    SPLIT_FILE = "bfcl_data/task_split.json"

    def __init__(
        self,
        category: str = "multi_turn_base",
        agent_model: str | None = None,
        split: str | None = "train",
        n_concurrent: int = 10,
        runs_dir: str = "workspace/bfcl_runs",
        include_input_log_train: bool = True,
        per_task_timeout: int = 300,
    ):
        self.category = category
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.n_concurrent = max(1, n_concurrent)
        self.runs_dir = runs_dir
        self.include_input_log_train = include_input_log_train
        self.per_task_timeout = per_task_timeout

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_split_tasks(self) -> list[str] | None:
        if self.split is None:
            return None  # caller will fall back to all packaged tasks
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

    def _load_all_packaged_tasks(self) -> list[str]:
        """Load every packaged task ID for the configured category."""
        from pathlib import Path

        import bfcl_eval
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX

        data_file = (
            Path(bfcl_eval.__file__).parent
            / "data"
            / f"{VERSION_PREFIX}_{self.category}.json"
        )
        if not data_file.exists():
            raise FileNotFoundError(f"BFCL data file not found: {data_file}")
        ids: list[str] = []
        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("id"):
                    ids.append(entry["id"])
        return ids

    def _find_artifact(self, base_dir: str, filename: str) -> str | None:
        """Locate `filename` anywhere under `base_dir`. BFCL nests files under
        `<registry_dir_name>/<group_dir>/`, which depends on the category, so
        a walk is more robust than computing the exact path."""
        if not os.path.isdir(base_dir):
            return None
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def _read_jsonl(self, path: str) -> list[dict]:
        """Read a BFCL JSONL file (note: `.json` extension, JSONL on the wire)."""
        entries: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _parse_rewards(
        self, run_root: str, requested_ids: list[str]
    ) -> dict[str, float | None]:
        """Reconstruct per-task rewards from result + score JSONL.

        Algorithm (per BFCL_INTEGRATION_PLAN.md, Reward Parsing):
        1. All requested IDs default to None (infrastructure error).
        2. IDs that appear in the result file move to 1.0 (generated successfully).
        3. IDs that appear in the score file (after the header row) drop to 0.0
           (BFCL judged the response incorrect).
        """
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX

        result_filename = f"{VERSION_PREFIX}_{self.category}_result.json"
        score_filename = f"{VERSION_PREFIX}_{self.category}_score.json"

        rewards: dict[str, float | None] = {tid: None for tid in requested_ids}

        result_path = self._find_artifact(
            os.path.join(run_root, "result"), result_filename
        )
        if not result_path:
            return rewards  # generation never produced output

        generated_ids = {
            entry["id"] for entry in self._read_jsonl(result_path) if entry.get("id")
        }
        for tid in requested_ids:
            if tid in generated_ids:
                rewards[tid] = 1.0

        score_path = self._find_artifact(
            os.path.join(run_root, "score"), score_filename
        )
        if not score_path:
            # Generation succeeded but evaluation never wrote a score file —
            # treat as infra error rather than silently calling everything correct.
            return {tid: None for tid in requested_ids}

        for i, entry in enumerate(self._read_jsonl(score_path)):
            if i == 0 and "accuracy" in entry:
                continue  # header row inserted by save_eval_results
            tid = entry.get("id")
            if tid in rewards:
                rewards[tid] = 0.0

        return rewards

    def _copy_train_traces(self, run_root: str, requested_ids: list[str]) -> None:
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX

        result_filename = f"{VERSION_PREFIX}_{self.category}_result.json"
        score_filename = f"{VERSION_PREFIX}_{self.category}_score.json"

        requested = set(requested_ids)

        result_path = self._find_artifact(
            os.path.join(run_root, "result"), result_filename
        )
        result_by_id: dict[str, dict] = {}
        if result_path:
            for entry in self._read_jsonl(result_path):
                tid = entry.get("id")
                if tid and tid in requested:
                    result_by_id[tid] = entry

        score_path = self._find_artifact(
            os.path.join(run_root, "score"), score_filename
        )
        score_by_id: dict[str, dict] = {}
        if score_path:
            for i, entry in enumerate(self._read_jsonl(score_path)):
                if i == 0 and "accuracy" in entry:
                    continue
                tid = entry.get("id")
                if tid and tid in requested:
                    score_by_id[tid] = entry

        latest_dir = os.path.join("workspace", "traces", "latest")
        baseline_dir = os.path.join("workspace", "traces", "baseline")
        os.makedirs(latest_dir, exist_ok=True)

        for tid, entry in result_by_id.items():
            dest = os.path.join(latest_dir, tid)
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "trace.json"), "w") as f:
                json.dump(entry, f, indent=2, default=str)
            with open(os.path.join(dest, "result.json"), "w") as f:
                json.dump(entry, f, indent=2, default=str)
            if tid in score_by_id:
                with open(os.path.join(dest, "score.json"), "w") as f:
                    json.dump(score_by_id[tid], f, indent=2, default=str)

            base_dest = os.path.join(baseline_dir, tid)
            if not os.path.exists(base_dest):
                os.makedirs(base_dest, exist_ok=True)
                with open(os.path.join(base_dest, "trace.json"), "w") as f:
                    json.dump(entry, f, indent=2, default=str)
                with open(os.path.join(base_dest, "result.json"), "w") as f:
                    json.dump(entry, f, indent=2, default=str)
                if tid in score_by_id:
                    with open(os.path.join(base_dest, "score.json"), "w") as f:
                        json.dump(score_by_id[tid], f, indent=2, default=str)

        print("[benchmark] traces: latest/ updated, baseline/ preserved")

    # ── main entry ─────────────────────────────────────────────────────────

    def run(self, task_ids: list[str] | None = None) -> dict[str, float | None]:
        import datetime

        if task_ids is None:
            task_ids = self._load_split_tasks()
        if task_ids is None:
            task_ids = self._load_all_packaged_tasks()
        if not task_ids:
            return {}

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        split_label = self.split or "all"
        run_root = os.path.abspath(
            os.path.join(self.runs_dir, split_label, timestamp)
        )
        os.makedirs(run_root, exist_ok=True)

        # Fresh test_case_ids_to_generate.json for THIS invocation. The
        # gating-step subset call passes a different `task_ids` list so the
        # file must be rewritten every run.
        with open(
            os.path.join(run_root, "test_case_ids_to_generate.json"), "w"
        ) as f:
            json.dump({self.category: list(task_ids)}, f, indent=2)

        env = os.environ.copy()
        repo_root = os.path.dirname(os.path.abspath(__file__))
        env["BFCL_PROJECT_ROOT"] = run_root
        env["AGENT_MODEL"] = self.agent_model
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        timeout_sec = max(
            600,
            int((len(task_ids) / self.n_concurrent) * self.per_task_timeout) + 300,
        )
        print(
            f"[benchmark] running {len(task_ids)} bfcl tasks "
            f"(category={self.category}, model={self.agent_model}, "
            f"n={self.n_concurrent}, subprocess_timeout={timeout_sec}s)"
        )

        # ── Generate ──────────────────────────────────────────────────────
        gen_cmd = [
            sys.executable, "-m", "agent.helpers.bfcl.run", "generate",
            "--model", "harness-agent",
            "--test-category", self.category,
            "--run-ids",
            "--allow-overwrite",
            "--num-threads", str(self.n_concurrent),
        ]
        if self.split == "train" and self.include_input_log_train:
            gen_cmd.append("--include-input-log")

        try:
            gen = subprocess.run(
                gen_cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
            )
            if gen.stdout:
                print(gen.stdout)
            if gen.stderr:
                print(gen.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[benchmark] WARNING: bfcl generate timed out after {timeout_sec}s")
            # Fall through to parse partial results

        # ── Evaluate ──────────────────────────────────────────────────────
        eval_cmd = [
            sys.executable, "-m", "agent.helpers.bfcl.run", "evaluate",
            "--model", "harness-agent",
            "--test-category", self.category,
            "--partial-eval",
        ]
        try:
            ev = subprocess.run(
                eval_cmd, env=env, capture_output=True, text=True, timeout=600,
            )
            if ev.stdout:
                print(ev.stdout)
            if ev.stderr:
                print(ev.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("[benchmark] WARNING: bfcl evaluate timed out after 600s")

        # ── Parse rewards ─────────────────────────────────────────────────
        rewards = self._parse_rewards(run_root, task_ids)

        # ── Trace handling ────────────────────────────────────────────────
        if self.split == "train":
            self._copy_train_traces(run_root, task_ids)

        # ── Cleanup ───────────────────────────────────────────────────────
        # Test/baseline runs: delete raw artifacts immediately so the coding
        # agent cannot read failure details. Train runs: also delete the raw
        # run dir — useful traces are already copied to workspace/traces/.
        shutil.rmtree(run_root, ignore_errors=True)

        # Prune any stale run directories under the same split.
        split_dir = os.path.join(self.runs_dir, split_label)
        if os.path.isdir(split_dir):
            for old in os.listdir(split_dir):
                old_path = os.path.join(split_dir, old)
                if os.path.isdir(old_path):
                    shutil.rmtree(old_path, ignore_errors=True)

        return rewards


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
    _concurrency_defaults = {
        "terminal-bench": 50,
        "bfcl": 10,
    }
    _concurrency_default = cfg.get(
        "max_concurrency", _concurrency_defaults.get(benchmark, 3)
    )
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
    elif benchmark == "bird-interact":
        runner = BirdInteractRunner(
            bird_repo=cfg.get("bird_repo"),
            bird_python_bin=cfg.get("bird_python_bin"),
            split=args.split,
            mode=cfg.get("mode", "a-interact"),
            dataset=cfg.get("dataset", "lite"),
            data_path=cfg.get("bird_data_path"),
            agent_model=cfg.get("agent_model"),
            user_model=cfg.get("user_model"),
            patience=cfg.get("patience", 3),
            n_concurrent=args.concurrency,
            per_task_timeout=cfg.get("per_task_timeout", 1800),
            jobs_dir="workspace/bird_runs/cli",
            system_agent_port=cfg.get("system_agent_port", 6100),
            user_sim_port=cfg.get("user_sim_port", 6101),
            db_env_port=cfg.get("db_env_port", 6102),
            pg_host=cfg.get("pg_host"),
            pg_port=cfg.get("pg_port"),
            pg_user=cfg.get("pg_user"),
            pg_password=cfg.get("pg_password"),
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
            user_model=cfg.get("user_model"),
        )
    elif benchmark == "bfcl":
        runner = BFCLRunner(
            category=cfg.get("category", "multi_turn_base"),
            agent_model=cfg.get("agent_model"),
            split=args.split,
            n_concurrent=args.concurrency,
            per_task_timeout=cfg.get("per_task_timeout", 300),
            runs_dir="workspace/bfcl_runs",
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
