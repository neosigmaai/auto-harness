"""Benchmark execution services (mock for Milestone 2)."""

from __future__ import annotations

import asyncio
import hashlib

from api.schemas import RunError, RunStatus, TaskStatus
from api.store import PostgresRunStore, store as default_store


class ExecutionUnavailableError(Exception):
    """Raised when the execution environment cannot start a run."""


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

        # Status should already be running after claim; ensure timestamps exist.
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

            from api.store import _utcnow

            self.store.update(
                run_id,
                status=RunStatus.completed,
                finished_at=_utcnow(),
            )
        except Exception as exc:  # noqa: BLE001 — surface as run-level failure
            from api.store import _utcnow

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
        """
        Deterministic mock outcomes:
        - hash % 5 == 0 -> error (infra/timeout)
        - hash % 5 == 1 -> failed (verifier)
        - else -> passed
        """
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


runner = MockBenchmarkRunner(store=default_store)
