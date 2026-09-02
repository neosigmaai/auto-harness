"""Benchmark execution services: mock (tests) and Harbor (Milestone 3)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

from api.config import BenchmarkConfig, REPO_ROOT, load_config
from api.schemas import RunError, RunStatus, TaskStatus
from api.store import PostgresRunStore, _utcnow, store as default_store

logger = logging.getLogger(__name__)

DEFAULT_AGENT_IMPORT_PATH = "agent.agent:HarnessAgent"


class ExecutionUnavailableError(Exception):
    """Raised when the execution environment cannot start a run."""


def reward_to_task_status(reward: float | None) -> tuple[TaskStatus, float | None, str | None]:
    """
    Map a Harbor/Terminal-Bench reward to API task fields.

    Returns (status, reward, remarks).
    """
    if reward is None:
        return (
            TaskStatus.error,
            None,
            "No verifier result (timeout or infra)",
        )
    if reward >= 1.0:
        return TaskStatus.passed, 1.0, None
    if reward <= 0.0:
        return TaskStatus.failed, 0.0, "Verifier failed"
    # Partial rewards (rare): treat as failed with score preserved.
    return TaskStatus.failed, float(reward), f"Partial reward {reward}"


class MockBenchmarkRunner:
    """
    Simulated Terminal-Bench runner.

    Deterministically marks tasks as passed / failed / error based on task_id
    so clients can exercise the full response shape without Harbor/E2B.
    """

    def __init__(
        self,
        store: PostgresRunStore,
        *,
        step_delay_sec: float = 0.05,
        execution_available: bool = True,
    ) -> None:
        self.store = store
        self.step_delay_sec = step_delay_sec
        self.execution_available = execution_available

    def check_available(self) -> None:
        """Raise if the execution environment cannot accept new runs."""
        if not self.execution_available:
            raise ExecutionUnavailableError(
                "Execution environment is unavailable (mock flag)"
            )

    async def execute(self, run_id: str) -> None:
        record = self.store.get(run_id)
        if record is None:
            return

        if record.status != RunStatus.running:
            self.store.update(run_id, status=RunStatus.running)

        try:
            self.check_available()
            for task in list(record.tasks):
                self.store.set_task(
                    run_id,
                    task.task_id,
                    status=TaskStatus.running,
                )
                if self.step_delay_sec > 0:
                    await asyncio.sleep(self.step_delay_sec)

                outcome = self._outcome_for(task.task_id)
                self.store.set_task(
                    run_id,
                    task.task_id,
                    status=outcome["status"],
                    reward=outcome["reward"],
                    remarks=outcome["remarks"],
                )

            self.store.update(
                run_id,
                status=RunStatus.completed,
                finished_at=_utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            self.store.update(
                run_id,
                status=RunStatus.failed,
                finished_at=_utcnow(),
                error=RunError(
                    code="internal_error",
                    message=str(exc),
                ),
            )

    def execute_sync(self, run_id: str) -> None:
        """Synchronous wrapper for the worker process."""
        asyncio.run(self.execute(run_id))

    @staticmethod
    def _outcome_for(task_id: str) -> dict:
        digest = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
        bucket = digest % 5
        if bucket == 0:
            return {
                "status": TaskStatus.error,
                "reward": None,
                "remarks": "Mock sandbox timeout while running task",
            }
        if bucket == 1:
            return {
                "status": TaskStatus.failed,
                "reward": 0.0,
                "remarks": f"Verifier failed: mock assertion did not pass for {task_id}",
            }
        return {
            "status": TaskStatus.passed,
            "reward": 1.0,
            "remarks": None,
        }


class HarborBenchmarkRunner:
    """
    Runs Terminal-Bench tasks via Harbor (Docker / E2B / etc.).

    Wraps ``TerminalBenchRunner`` and writes results into PostgresRunStore.
    """

    def __init__(
        self,
        store: PostgresRunStore,
        *,
        config: BenchmarkConfig | None = None,
        agent_import_path: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.config = config or load_config()
        # Layer B (jobs) passes "agent.spec_agent:HarnessAgent"; /v1/runs keeps agent/agent.py.
        self.agent_import_path = agent_import_path or DEFAULT_AGENT_IMPORT_PATH
        self.extra_env = dict(extra_env or {})

    def check_available(self) -> None:
        if shutil.which("harbor") is None:
            raise ExecutionUnavailableError(
                "harbor CLI not found on PATH (install with: uv tool install harbor)"
            )

        self._check_agent_import()
        self._check_env_provider()

    def _agent_module_relpath(self) -> str:
        """Turn e.g. 'agent.spec_agent:HarnessAgent' into 'agent/spec_agent.py'."""
        module = self.agent_import_path.split(":", 1)[0]
        return "/".join(module.split(".")) + ".py"

    def _check_agent_import(self) -> None:
        rel = self._agent_module_relpath()
        agent_path = REPO_ROOT / rel
        if rel == "agent/agent.py":
            hint = "Copy agent/templates/terminal_bench.py to agent/agent.py."
        else:
            hint = f"{rel} ships with the repo — restore it from git."

        if not agent_path.is_file():
            raise ExecutionUnavailableError(f"{rel} is missing. {hint}")
        source = agent_path.read_text(encoding="utf-8")
        if "Placeholder — do not edit" in source:
            raise ExecutionUnavailableError(f"{rel} is still the placeholder. {hint}")
        if "class HarnessAgent" not in source:
            raise ExecutionUnavailableError(f"{rel} has no HarnessAgent class. {hint}")

    def _check_env_provider(self) -> None:
        provider = self.config.env_provider
        if provider == "docker":
            try:
                subprocess.run(
                    ["docker", "info"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise ExecutionUnavailableError(
                    "Docker is not available (required for env_provider=docker). "
                    "Start Colima/Docker Desktop and retry."
                ) from exc
        elif provider == "e2b":
            if not os.environ.get("E2B_API_KEY"):
                raise ExecutionUnavailableError("E2B_API_KEY is required for env_provider=e2b")
        elif provider == "daytona":
            if not os.environ.get("DAYTONA_API_KEY"):
                raise ExecutionUnavailableError(
                    "DAYTONA_API_KEY is required for env_provider=daytona"
                )
        elif provider == "modal":
            has_pair = os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")
            has_toml = (Path.home() / ".modal.toml").exists()
            if not has_pair and not has_toml:
                raise ExecutionUnavailableError(
                    "Modal credentials required (MODAL_TOKEN_ID/SECRET or ~/.modal.toml)"
                )

    def execute_sync(self, run_id: str) -> None:
        record = self.store.get(run_id)
        if record is None:
            return

        if record.status != RunStatus.running:
            self.store.update(run_id, status=RunStatus.running)

        try:
            self.check_available()
        except ExecutionUnavailableError as exc:
            self.store.update(
                run_id,
                status=RunStatus.failed,
                finished_at=_utcnow(),
                error=RunError(code="execution_unavailable", message=str(exc)),
            )
            return

        task_ids = list(record.request.task_ids)
        for tid in task_ids:
            self.store.set_task(run_id, tid, status=TaskStatus.running)

        jobs_dir = str(REPO_ROOT / "workspace" / "runs" / run_id)
        Path(jobs_dir).mkdir(parents=True, exist_ok=True)

        try:
            from benchmark import TerminalBenchRunner

            tbr = TerminalBenchRunner(
                agent_model=record.request.agent_model,
                split=None,  # do not require tbench_data/task_split.json
                env_provider=self.config.env_provider,
                n_concurrent=self.config.max_concurrency,
                dataset=self.config.dataset,
                per_task_timeout=self.config.per_task_timeout,
                jobs_dir=jobs_dir,
                agent_import_path=self.agent_import_path,
                extra_env=self.extra_env,
            )
            logger.info(
                "starting harbor run run_id=%s tasks=%s env=%s model=%s",
                run_id,
                task_ids,
                self.config.env_provider,
                record.request.agent_model,
            )
            results = tbr.run(task_ids=task_ids)
        except Exception as exc:  # noqa: BLE001
            logger.exception("harbor execution failed run_id=%s", run_id)
            self.store.update(
                run_id,
                status=RunStatus.failed,
                finished_at=_utcnow(),
                error=RunError(code="internal_error", message=str(exc)),
            )
            for tid in task_ids:
                current = self.store.get(run_id)
                if current is None:
                    break
                for t in current.tasks:
                    if t.task_id == tid and t.status == TaskStatus.running:
                        self.store.set_task(
                            run_id,
                            tid,
                            status=TaskStatus.error,
                            reward=None,
                            remarks=f"Harbor execution failed: {exc}",
                        )
            return

        if not results:
            self.store.update(
                run_id,
                status=RunStatus.failed,
                finished_at=_utcnow(),
                error=RunError(
                    code="internal_error",
                    message="Harbor returned no results (check jobs_dir / harbor logs)",
                ),
            )
            for tid in task_ids:
                self.store.set_task(
                    run_id,
                    tid,
                    status=TaskStatus.error,
                    reward=None,
                    remarks="No verifier result (timeout or infra)",
                )
            return

        for tid in task_ids:
            reward = results.get(tid)
            status, mapped_reward, remarks = reward_to_task_status(reward)
            self.store.set_task(
                run_id,
                tid,
                status=status,
                reward=mapped_reward,
                remarks=remarks,
            )

        self.store.update(
            run_id,
            status=RunStatus.completed,
            finished_at=_utcnow(),
        )
        logger.info("harbor run completed run_id=%s jobs_dir=%s", run_id, jobs_dir)


def create_runner(
    store: PostgresRunStore,
    *,
    config: BenchmarkConfig | None = None,
    step_delay_sec: float = 0.05,
) -> MockBenchmarkRunner | HarborBenchmarkRunner:
    """Factory: mock or harbor based on config / EXECUTION_BACKEND."""
    cfg = config or load_config()
    if cfg.execution_backend == "mock":
        return MockBenchmarkRunner(store=store, step_delay_sec=step_delay_sec)
    return HarborBenchmarkRunner(store=store, config=cfg)


default_runner = MockBenchmarkRunner(store=default_store)
# Backwards-compatible alias (avoid shadowing the submodule name on api.services).
runner = default_runner
