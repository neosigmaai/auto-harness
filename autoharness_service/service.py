from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from autoharness_service.agent_patch import AgentPatchService
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
MAX_PROPOSAL_TEXT_FIELD_LENGTH = 1000
MAX_TASK_TEXT_FIELD_LENGTH = 500
TERMINAL_TASK_STATUSES = {"passed", "failed", "infra_failed", "timed_out"}
INFRA_ONLY_FAILURE_TYPES = {"missing_result", "runner_failed", "runner_timeout"}
SENSITIVE_TEXT_MARKERS = (
    "OPENAI_API_KEY",
    "DAYTONA_API_KEY",
    "DATABASE_URL",
    "API_KEY",
    "SECRET",
    ".env",
)
API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


class RunService:
    def __init__(
        self,
        store,
        simulated_runner: SimulatedBenchmarkRunner | None = None,
        terminal_runner: TerminalBenchRunnerAdapter | None = None,
        optimizer: Optimizer | None = None,
        agent_patcher: AgentPatchService | None = None,
        service_run_root: Path | str = "workspace/service_runs",
        max_local_concurrency: int = 4,
    ):
        self.store = store
        self.simulated_runner = simulated_runner or SimulatedBenchmarkRunner()
        self.terminal_runner = terminal_runner or TerminalBenchRunnerAdapter()
        self.optimizer = optimizer or Optimizer()
        self.agent_patcher = agent_patcher or AgentPatchService()
        self.service_run_root = Path(service_run_root)
        self.max_local_concurrency = max_local_concurrency
        self._real_run_lock = threading.BoundedSemaphore(value=1)
        self._poller_lock = threading.Lock()
        self._poller_stop = threading.Event()
        self._poller_thread: threading.Thread | None = None

    def submit_run(
        self,
        request: RunCreateRequest,
        org_id: str,
        created_by: str,
        start_background: bool = True,
    ):
        self.store.init_schema()
        run = self.store.create_run(request, org_id=org_id, created_by=created_by)
        self.store.create_task_queue(run.run_id, org_id=org_id, task_ids=run.task_ids)
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

    def start_polling(self, interval_sec: float = 2.0, limit: int = 10) -> None:
        with self._poller_lock:
            if self._poller_thread is not None and self._poller_thread.is_alive():
                return
            self._poller_stop.clear()
            self._poller_thread = threading.Thread(
                target=self._poll_loop,
                kwargs={"interval_sec": interval_sec, "limit": limit},
                daemon=True,
            )
            self._poller_thread.start()

    def stop_polling(self, timeout: float = 1.0) -> None:
        self._poller_stop.set()
        thread = self._poller_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _poll_loop(self, interval_sec: float, limit: int) -> None:
        requeue_running = True
        while not self._poller_stop.is_set():
            try:
                self.resume_incomplete_runs(
                    limit=limit,
                    requeue_running=requeue_running,
                )
            except Exception:
                # Keep the worker alive; individual run failures are persisted by
                # execute_run when a run can be loaded.
                pass
            requeue_running = False
            self._poller_stop.wait(interval_sec)

    def resume_incomplete_runs(
        self,
        limit: int = 10,
        *,
        requeue_running: bool = True,
    ) -> int:
        self.store.init_schema()
        resumed = 0
        for run in self.store.list_incomplete_runs(limit=limit):
            if requeue_running:
                self.store.requeue_running_tasks(run.run_id, run.org_id)
            self.execute_run(run.run_id, org_id=run.org_id)
            resumed += 1
        return resumed

    def execute_run(self, run_id: str, org_id: str) -> None:
        run = self.store.get_run(run_id, org_id)
        if run is None:
            return
        try:
            self.store.mark_run_running(run_id, org_id)
            task_results = self._execute_task_rows(run, org_id, attempt="baseline")
            if task_results is None:
                return
            score = _score(task_results)
            runner_produced_result = _has_numeric_reward(task_results)
            failed_run_status = _failed_run_status(task_results)
            self.store.create_iteration(
                run_id,
                org_id,
                iteration_index=0,
                status=(
                    failed_run_status
                    if run.mode == "real" and not runner_produced_result
                    else "completed"
                ),
                agent_version="initial",
                score=score,
            )
            final_score = score
            if run.mode == "real" and not runner_produced_result:
                if run.max_iterations == 1:
                    self._run_optimization_iteration(
                        run,
                        org_id,
                        baseline_results=task_results,
                        baseline_score=score,
                    )
                self.store.mark_run_failed(
                    run_id,
                    org_id,
                    status=failed_run_status,
                    error=_runner_error(task_results),
                )
                return
            if run.max_iterations == 1:
                final_score, _ = self._run_optimization_iteration(
                    run,
                    org_id,
                    baseline_results=task_results,
                    baseline_score=score,
                )
            self.store.mark_run_succeeded(run_id, org_id, score=final_score)
        except TimeoutError as exc:
            self._record_runner_exception(
                run_id,
                org_id,
                run.task_ids,
                status="timed_out",
                exc=exc,
            )
        except Exception as exc:
            self._record_runner_exception(
                run_id,
                org_id,
                run.task_ids,
                status="failed",
                exc=exc,
            )

    def _execute_task_rows(
        self,
        run,
        org_id: str,
        *,
        attempt: str,
    ) -> list[TaskResultRecord] | None:
        for task_id in run.task_ids:
            existing = self._task_by_id(run.run_id, org_id, task_id)
            if existing is not None and existing.status in TERMINAL_TASK_STATUSES:
                continue

            claimed = self.store.mark_task_running(run.run_id, org_id, task_id)
            if not claimed:
                continue
            try:
                raw_results = self._run_benchmark(run, [task_id], attempt=attempt)
            except TimeoutError as exc:
                self.store.upsert_task_result(
                    run.run_id,
                    org_id,
                    _with_attempt_metadata(
                        self._runner_timeout_result(run.run_id, task_id, exc),
                        attempt,
                    ),
                )
                continue
            except Exception as exc:
                self.store.upsert_task_result(
                    run.run_id,
                    org_id,
                    _with_attempt_metadata(
                        self._runner_failed_results(run.run_id, [task_id], exc)[0],
                        attempt,
                    ),
                )
                continue
            artifacts_by_task = (
                getattr(self.terminal_runner, "last_artifacts", {})
                if run.mode == "real"
                else {}
            )
            task_result = self._normalize_results(
                run.run_id,
                [task_id],
                raw_results,
                source="simulated" if run.mode == "simulated" else "harbor",
                artifacts_by_task=artifacts_by_task,
            )[0]
            task_result = _with_attempt_metadata(task_result, attempt)
            self.store.upsert_task_result(run.run_id, org_id, task_result)

        task_results = self.store.list_task_results(run.run_id, org_id)
        if any(result.status not in TERMINAL_TASK_STATUSES for result in task_results):
            return None
        return task_results

    def _run_optimization_iteration(
        self,
        run,
        org_id: str,
        baseline_results: list[TaskResultRecord],
        baseline_score: float,
    ) -> tuple[float, list[TaskResultRecord]]:
        if baseline_score == 1.0:
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="skipped_no_failures",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=False,
                    decision_reason="baseline score is already 1.0",
                    baseline_results=baseline_results,
                    rerun_results=None,
                ),
                accepted=False,
            )
            return baseline_score, baseline_results

        if _infra_only_without_reward(baseline_results):
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="proposal_failed",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=False,
                    decision_reason="baseline failures are infrastructure-only",
                    baseline_results=baseline_results,
                    rerun_results=None,
                ),
                accepted=False,
            )
            return baseline_score, baseline_results

        summary = build_failure_summary(baseline_results)
        current_instruction: str | None = None
        try:
            current_instruction = self.agent_patcher.read_instruction()
            proposal = self.optimizer.propose_instruction_patch(
                baseline_results,
                summary,
                model=run.model,
                current_instruction=current_instruction,
            )
        except Exception as exc:
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="proposal_failed",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=False,
                    decision_reason=str(exc),
                    baseline_results=baseline_results,
                    rerun_results=None,
                    redaction_terms=(
                        [current_instruction] if current_instruction is not None else []
                    ),
                ),
                accepted=False,
            )
            return baseline_score, baseline_results

        redaction_terms = [current_instruction, proposal.new_agent_instruction]
        proposal_json = _iteration_proposal_json(
            proposal=proposal,
            baseline_score=baseline_score,
            rerun_score=None,
            accepted=None,
            decision_reason="proposal created",
            baseline_results=baseline_results,
            rerun_results=None,
            redaction_terms=redaction_terms,
        )
        self.store.create_iteration(
            run.run_id,
            org_id,
            iteration_index=1,
            status="proposal_created",
            agent_version="proposal-1",
            score=baseline_score,
            proposal=proposal_json,
            accepted=None,
        )

        patch_result = None
        try:
            patch_result = self.agent_patcher.apply_instruction_patch(
                proposal.new_agent_instruction,
                snapshot_dir=self.service_run_root / run.run_id / "agent_versions",
            )
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="patch_applied",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    proposal=proposal,
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=None,
                    decision_reason="patch applied",
                    snapshot_paths=patch_result.snapshot_paths,
                    baseline_results=baseline_results,
                    rerun_results=None,
                    redaction_terms=redaction_terms,
                ),
                accepted=None,
            )
            self.store.reset_task_queue(
                run.run_id,
                org_id,
                run.task_ids,
                metadata={"source": "queued", "attempt": "proposal-1"},
            )
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="rerun_running",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    proposal=proposal,
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=None,
                    decision_reason="rerun started",
                    snapshot_paths=patch_result.snapshot_paths,
                    baseline_results=baseline_results,
                    rerun_results=None,
                    redaction_terms=redaction_terms,
                ),
                accepted=None,
            )
            rerun_results = self._execute_task_rows(
                run,
                org_id,
                attempt="proposal-1",
            )
            if rerun_results is None:
                raise RuntimeError("rerun did not reach terminal task results")
        except Exception as exc:
            discarded_snapshot_paths: dict[str, str] = {}
            if patch_result is not None:
                self.agent_patcher.restore(patch_result.original_source)
                discarded_snapshot_paths = self.agent_patcher.discard_proposal_snapshot(
                    patch_result.snapshot_paths
                )
            self.store.replace_task_results(run.run_id, org_id, baseline_results)
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="patch_rejected",
                agent_version="proposal-1",
                score=baseline_score,
                proposal=_iteration_proposal_json(
                    proposal=proposal,
                    baseline_score=baseline_score,
                    rerun_score=None,
                    accepted=False,
                    decision_reason=str(exc),
                    snapshot_paths=(
                        patch_result.snapshot_paths
                        if patch_result is not None
                        else None
                    ),
                    discarded_snapshot_paths=discarded_snapshot_paths,
                    reverted=patch_result is not None,
                    baseline_results=baseline_results,
                    rerun_results=None,
                    redaction_terms=redaction_terms,
                ),
                accepted=False,
            )
            return baseline_score, baseline_results

        rerun_score = _score(rerun_results)
        accepted = rerun_score > baseline_score
        if accepted:
            self.store.create_iteration(
                run.run_id,
                org_id,
                iteration_index=1,
                status="completed",
                agent_version="proposal-1",
                score=rerun_score,
                proposal=_iteration_proposal_json(
                    proposal=proposal,
                    baseline_score=baseline_score,
                    rerun_score=rerun_score,
                    accepted=True,
                    decision_reason="rerun score improved baseline score",
                    snapshot_paths=patch_result.snapshot_paths,
                    baseline_results=baseline_results,
                    rerun_results=rerun_results,
                    redaction_terms=redaction_terms,
                ),
                accepted=True,
            )
            return rerun_score, rerun_results

        self.agent_patcher.restore(patch_result.original_source)
        discarded_snapshot_paths = self.agent_patcher.discard_proposal_snapshot(
            patch_result.snapshot_paths
        )
        self.store.replace_task_results(run.run_id, org_id, baseline_results)
        self.store.create_iteration(
            run.run_id,
            org_id,
            iteration_index=1,
            status="patch_rejected",
            agent_version="proposal-1",
            score=rerun_score,
            proposal=_iteration_proposal_json(
                proposal=proposal,
                baseline_score=baseline_score,
                rerun_score=rerun_score,
                accepted=False,
                decision_reason="rerun score did not improve baseline score",
                snapshot_paths=patch_result.snapshot_paths,
                discarded_snapshot_paths=discarded_snapshot_paths,
                reverted=True,
                baseline_results=baseline_results,
                rerun_results=rerun_results,
                redaction_terms=redaction_terms,
            ),
            accepted=False,
        )
        return baseline_score, baseline_results

    def _record_runner_exception(
        self,
        run_id: str,
        org_id: str,
        task_ids: list[str],
        *,
        status: str,
        exc: Exception,
    ) -> None:
        task_results = self._runner_failed_results(run_id, task_ids, exc)
        score = _score(task_results)
        self.store.replace_task_results(run_id, org_id, task_results)
        self.store.create_iteration(
            run_id,
            org_id,
            iteration_index=0,
            status=status,
            agent_version="initial",
            score=score,
        )
        self.store.mark_run_failed(run_id, org_id, status=status, error=str(exc))

    def _run_benchmark(
        self,
        run,
        task_ids: list[str] | None = None,
        *,
        attempt: str,
    ):
        selected_task_ids = task_ids or run.task_ids
        if run.mode == "simulated":
            return self.simulated_runner.run(selected_task_ids)
        self._real_run_lock.acquire(blocking=True)
        try:
            runner_kwargs = {
                "model": run.model,
                "sandbox_provider": run.sandbox_provider,
                "requested_concurrency": min(
                    run.requested_concurrency,
                    self.max_local_concurrency,
                ),
                "run_id": run.run_id,
                "attempt": attempt,
            }
            try:
                return self.terminal_runner.run(selected_task_ids, **runner_kwargs)
            except TypeError as exc:
                if "unexpected keyword argument 'attempt'" not in str(exc):
                    raise
                runner_kwargs.pop("attempt")
                return self.terminal_runner.run(selected_task_ids, **runner_kwargs)
        finally:
            self._real_run_lock.release()

    def _task_by_id(
        self, run_id: str, org_id: str, task_id: str
    ) -> TaskResultRecord | None:
        for result in self.store.list_task_results(run_id, org_id):
            if result.task_id == task_id:
                return result
        return None

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

    def _runner_timeout_result(
        self, run_id: str, task_id: str, exc: Exception
    ) -> TaskResultRecord:
        return TaskResultRecord(
            task_id=task_id,
            status="timed_out",
            reward=None,
            failure_type="runner_timeout",
            error_summary=_truncate_error_summary(str(exc)),
            metadata={
                "source": "runner_timeout",
                "run_id": run_id,
                "artifact_scope": "omitted_shared_latest",
                "trace_exists": False,
                "result_exists": False,
            },
        )

    def _normalize_results(
        self,
        run_id: str,
        task_ids: list[str],
        raw_results: dict[str, float | None],
        *,
        source: str,
        artifacts_by_task: dict[str, dict[str, Any]] | None = None,
    ) -> list[TaskResultRecord]:
        normalized: list[TaskResultRecord] = []
        artifacts_by_task = artifacts_by_task or {}
        for task_id in task_ids:
            artifacts = dict(artifacts_by_task.get(task_id, {}))
            trace_path = artifacts.get("trace")
            result_path = artifacts.get("trial_result")
            if not isinstance(trace_path, str):
                trace_path = None
            if not isinstance(result_path, str):
                result_path = None
            metadata = {
                "source": source if raw_results else "missing",
                "run_id": run_id,
                "artifact_scope": (
                    "harbor_job" if artifacts else "omitted_shared_latest"
                ),
                "trace_exists": bool(trace_path),
                "result_exists": bool(result_path),
            }
            if artifacts:
                metadata["artifacts"] = artifacts
            if task_id not in raw_results:
                normalized.append(
                    normalize_missing_result(
                        task_id,
                        "Task result missing from runner output",
                        trace_path=trace_path,
                        result_path=result_path,
                        metadata=metadata,
                    )
                )
                continue
            normalized.append(
                normalize_reward_result(
                    task_id,
                    raw_results.get(task_id),
                    trace_path=trace_path,
                    result_path=result_path,
                    metadata=metadata,
                )
            )
        return normalized

    def get_run_status(self, run_id: str, org_id: str) -> RunStatusResponse | None:
        run = self.store.get_run(run_id, org_id)
        if run is None:
            return None
        task_results = self.store.list_task_results(run_id, org_id)
        completed = sum(
            1 for result in task_results if result.status in TERMINAL_TASK_STATUSES
        )
        total = len(run.task_ids)
        running = sum(1 for result in task_results if result.status == "running")
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
            task_results=[_task_response(result) for result in task_results],
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


def _has_numeric_reward(task_results: list[TaskResultRecord]) -> bool:
    return any(result.reward is not None for result in task_results)


def _failed_run_status(task_results: list[TaskResultRecord]) -> str:
    if any(result.failure_type == "runner_timeout" for result in task_results):
        return "timed_out"
    return "failed"


def _runner_error(task_results: list[TaskResultRecord]) -> str:
    if all(result.failure_type == "missing_result" for result in task_results):
        return "runner produced no task results"
    for result in task_results:
        if result.error_summary:
            return result.error_summary
    return "runner produced no task results"


def _infra_only_without_reward(task_results: list[TaskResultRecord]) -> bool:
    return (
        bool(task_results)
        and not _has_numeric_reward(task_results)
        and all(
            result.status in {"infra_failed", "timed_out"}
            and (result.failure_type in INFRA_ONLY_FAILURE_TYPES)
            for result in task_results
        )
    )


def _with_attempt_metadata(
    result: TaskResultRecord,
    attempt: str,
) -> TaskResultRecord:
    if attempt == "baseline":
        return result
    return replace(result, metadata={**result.metadata, "attempt": attempt})


def _iteration_proposal_json(
    *,
    baseline_score: float,
    rerun_score: float | None,
    accepted: bool | None,
    decision_reason: str,
    baseline_results: list[TaskResultRecord],
    rerun_results: list[TaskResultRecord] | None,
    proposal: Any | None = None,
    snapshot_paths: dict[str, str] | None = None,
    discarded_snapshot_paths: dict[str, str] | None = None,
    reverted: bool = False,
    redaction_terms: list[str] | None = None,
) -> str:
    terms = redaction_terms or []
    payload: dict[str, Any] = {
        "baseline_score": baseline_score,
        "rerun_score": rerun_score,
        "accepted": accepted,
        "decision_reason": _sanitize_text(
            decision_reason,
            terms,
            max_chars=MAX_PROPOSAL_TEXT_FIELD_LENGTH,
        ),
        "changed_section": "AGENT_INSTRUCTION",
        "snapshot_paths": _sanitize_mapping(snapshot_paths or {}, terms),
        "discarded_snapshot_paths": _sanitize_mapping(
            discarded_snapshot_paths or {},
            terms,
        ),
        "reverted": reverted,
        "baseline_tasks": _compact_task_summaries(baseline_results, terms),
        "rerun_tasks": _compact_task_summaries(rerun_results or [], terms),
    }
    if proposal is not None:
        payload.update(
            {
                "hypothesis": _sanitize_text(
                    proposal.hypothesis,
                    terms,
                    max_chars=MAX_PROPOSAL_TEXT_FIELD_LENGTH,
                ),
                "expected_effect": _sanitize_text(
                    proposal.expected_effect,
                    terms,
                    max_chars=MAX_PROPOSAL_TEXT_FIELD_LENGTH,
                ),
                "risks": _sanitize_text(
                    proposal.risks,
                    terms,
                    max_chars=MAX_PROPOSAL_TEXT_FIELD_LENGTH,
                ),
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compact_task_summaries(
    task_results: list[TaskResultRecord],
    redaction_terms: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": _sanitize_text(
                result.task_id,
                redaction_terms,
                max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
            ),
            "status": result.status,
            "reward": result.reward,
            "failure_type": result.failure_type,
            "error_summary": _sanitize_text(
                result.error_summary,
                redaction_terms,
                max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
            ),
            "trace_path": _sanitize_text(
                result.trace_path,
                redaction_terms,
                max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
            ),
            "result_path": _sanitize_text(
                result.result_path,
                redaction_terms,
                max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
            ),
            "attempt": _sanitize_text(
                result.metadata.get("attempt"),
                redaction_terms,
                max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
            ),
            "artifacts": _sanitize_mapping(
                _compact_artifact_paths(result.metadata),
                redaction_terms,
            ),
        }
        for result in task_results
    ]


def _compact_artifact_paths(metadata: dict[str, Any]) -> dict[str, str]:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}

    selected_keys = (
        "job_log",
        "job_result",
        "trial_log",
        "trial_result",
        "trace",
        "verifier_reward",
        "verifier_stdout",
    )
    return {
        key: value
        for key in selected_keys
        if isinstance((value := artifacts.get(key)), str)
    }


def _sanitize_mapping(
    values: dict[str, str],
    redaction_terms: list[str],
) -> dict[str, str]:
    return {
        _sanitize_text(
            key,
            redaction_terms,
            max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
        )
        or "redacted": _sanitize_text(
            value,
            redaction_terms,
            max_chars=MAX_TASK_TEXT_FIELD_LENGTH,
        )
        or ""
        for key, value in values.items()
    }


def _sanitize_text(
    value: str | None,
    redaction_terms: list[str],
    *,
    max_chars: int,
) -> str | None:
    if value is None:
        return None
    text = str(value)
    for term in redaction_terms:
        if term:
            text = text.replace(term, "[REDACTED]")
    text = API_KEY_PATTERN.sub("[REDACTED]", text)
    for marker in SENSITIVE_TEXT_MARKERS:
        text = text.replace(marker, "[REDACTED]")
    if len(text) > max_chars:
        text = f"{text[:max_chars]}...[truncated]"
    return text


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
