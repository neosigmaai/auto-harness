from __future__ import annotations

import threading

from autoharness_service.models import TaskResultRecord
from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_missing_result,
    normalize_reward_result,
)
from autoharness_service.optimizer import Optimizer
from autoharness_service.runner import (
    SimulatedBenchmarkRunner,
    TerminalBenchRunnerAdapter,
)
from autoharness_service.schemas import (
    FailureSummaryResponse,
    RunCreateRequest,
    RunProgress,
    RunResultsResponse,
    RunStatusResponse,
    TaskResultResponse,
)

MAX_ERROR_SUMMARY_LENGTH = 4000


class RunService:
    def __init__(
        self,
        store,
        simulated_runner: SimulatedBenchmarkRunner | None = None,
        terminal_runner: TerminalBenchRunnerAdapter | None = None,
        optimizer: Optimizer | None = None,
        max_local_concurrency: int = 4,
    ):
        self.store = store
        self.simulated_runner = simulated_runner or SimulatedBenchmarkRunner()
        self.terminal_runner = terminal_runner or TerminalBenchRunnerAdapter()
        self.optimizer = optimizer or Optimizer()
        self.max_local_concurrency = max_local_concurrency
        self._real_run_lock = threading.BoundedSemaphore(value=1)

    def submit_run(
        self,
        request: RunCreateRequest,
        org_id: str,
        created_by: str,
        start_background: bool = True,
    ):
        self.store.init_schema()
        run = self.store.create_run(request, org_id=org_id, created_by=created_by)
        self.store.create_iteration(
            run.run_id,
            org_id=org_id,
            iteration_index=0,
            status="queued",
            agent_version="initial",
        )
        if start_background:
            thread = threading.Thread(
                target=self.execute_run,
                kwargs={"run_id": run.run_id, "org_id": org_id},
                daemon=True,
            )
            thread.start()
        return run

    def execute_run(self, run_id: str, org_id: str) -> None:
        run = self.store.get_run(run_id, org_id)
        if run is None:
            return
        try:
            self.store.mark_run_running(run_id, org_id)
            raw_results = self._run_benchmark(run)
            task_results = self._normalize_results(
                run.run_id,
                run.task_ids,
                raw_results,
                source="simulated" if run.mode == "simulated" else "harbor",
            )
            self.store.replace_task_results(run_id, org_id, task_results)
            score = _score(task_results)
            self.store.create_iteration(
                run_id,
                org_id,
                iteration_index=0,
                status=(
                    "failed" if run.mode == "real" and not raw_results else "completed"
                ),
                agent_version="initial",
                score=score,
            )
            if run.mode == "real" and not raw_results:
                self.store.mark_run_failed(
                    run_id,
                    org_id,
                    status="failed",
                    error="runner produced no task results",
                )
                return
            if run.max_iterations > 0:
                summary = build_failure_summary(task_results)
                try:
                    proposal = self.optimizer.propose(
                        task_results,
                        summary,
                        model=run.model,
                    )
                    proposal_status = "proposal_created"
                except Exception as exc:
                    proposal = f"LLM proposal failed: {exc}"
                    proposal_status = "proposal_failed"
                self.store.create_iteration(
                    run_id,
                    org_id,
                    iteration_index=1,
                    status=proposal_status,
                    agent_version="proposal-1",
                    score=score,
                    proposal=proposal,
                    accepted=None,
                )
            self.store.mark_run_succeeded(run_id, org_id, score=score)
        except TimeoutError as exc:
            self.store.replace_task_results(
                run_id,
                org_id,
                self._runner_failed_results(run_id, run.task_ids, exc),
            )
            self.store.mark_run_failed(
                run_id, org_id, status="timed_out", error=str(exc)
            )
        except Exception as exc:
            self.store.replace_task_results(
                run_id,
                org_id,
                self._runner_failed_results(run_id, run.task_ids, exc),
            )
            self.store.mark_run_failed(run_id, org_id, status="failed", error=str(exc))

    def _run_benchmark(self, run):
        if run.mode == "simulated":
            return self.simulated_runner.run(run.task_ids)
        self._real_run_lock.acquire(blocking=True)
        try:
            return self.terminal_runner.run(
                run.task_ids,
                model=run.model,
                sandbox_provider=run.sandbox_provider,
                requested_concurrency=min(
                    run.requested_concurrency,
                    self.max_local_concurrency,
                ),
                run_id=run.run_id,
            )
        finally:
            self._real_run_lock.release()

    def _runner_failed_results(
        self, run_id: str, task_ids: list[str], exc: Exception
    ) -> list[TaskResultRecord]:
        return [
            TaskResultRecord(
                task_id=task_id,
                status="infra_failed",
                reward=None,
                failure_type="runner_failed",
                error_summary=_truncate_error_summary(str(exc)),
                metadata={
                    "source": "runner_failed",
                    "run_id": run_id,
                    "artifact_scope": "omitted_shared_latest",
                    "trace_exists": False,
                    "result_exists": False,
                },
            )
            for task_id in task_ids
        ]

    def _normalize_results(
        self,
        run_id: str,
        task_ids: list[str],
        raw_results: dict[str, float | None],
        *,
        source: str,
    ) -> list[TaskResultRecord]:
        normalized: list[TaskResultRecord] = []
        for task_id in task_ids:
            metadata = {
                "source": source if raw_results else "missing",
                "run_id": run_id,
                "artifact_scope": "omitted_shared_latest",
                "trace_exists": False,
                "result_exists": False,
            }
            if task_id not in raw_results:
                normalized.append(
                    normalize_missing_result(
                        task_id,
                        "Task result missing from runner output",
                        trace_path=None,
                        result_path=None,
                        metadata=metadata,
                    )
                )
                continue
            normalized.append(
                normalize_reward_result(
                    task_id,
                    raw_results.get(task_id),
                    trace_path=None,
                    result_path=None,
                    metadata=metadata,
                )
            )
        return normalized

    def get_run_status(self, run_id: str, org_id: str) -> RunStatusResponse | None:
        run = self.store.get_run(run_id, org_id)
        if run is None:
            return None
        task_results = self.store.list_task_results(run_id, org_id)
        completed = len(task_results)
        total = len(run.task_ids)
        running = 1 if run.status == "running" and completed < total else 0
        queued = max(total - completed - running, 0)
        return RunStatusResponse(
            run_id=run.run_id,
            status=run.status,
            progress=RunProgress(
                total=total,
                queued=queued,
                running=running,
                completed=completed,
            ),
            score=run.score,
            error=run.error,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    def get_run_results(self, run_id: str, org_id: str) -> RunResultsResponse | None:
        run = self.store.get_run(run_id, org_id)
        if run is None:
            return None
        task_results = self.store.list_task_results(run_id, org_id)
        summary = build_failure_summary(task_results)
        return RunResultsResponse(
            run_id=run.run_id,
            status=run.status,
            score=run.score,
            tasks_total=summary.tasks_total,
            tasks_passed=summary.tasks_passed,
            tasks_failed=summary.tasks_failed,
            tasks_infra_failed=summary.tasks_infra_failed,
            task_results=[_task_response(result) for result in task_results],
            failure_summary=FailureSummaryResponse(**summary.__dict__),
        )


def _score(task_results: list[TaskResultRecord]) -> float:
    if not task_results:
        return 0.0
    return sum(result.reward or 0.0 for result in task_results) / len(task_results)


def _truncate_error_summary(error: str) -> str:
    return error[:MAX_ERROR_SUMMARY_LENGTH]


def _task_response(result: TaskResultRecord) -> TaskResultResponse:
    return TaskResultResponse(
        task_id=result.task_id,
        status=result.status,
        reward=result.reward,
        failure_type=result.failure_type,
        error_summary=result.error_summary,
        trace_path=result.trace_path,
        result_path=result.result_path,
        metadata=result.metadata,
    )
