"""Postgres-backed store for benchmark runs (Milestone 2)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import selectinload

from api.db import get_session_factory
from api.models import RunRow, RunTaskRow
from api.schemas import (
    FailureCategory,
    FailureSummaryItem,
    RunError,
    RunRequestEcho,
    RunResponse,
    RunStatus,
    RunSummary,
    TaskResult,
    TaskStatus,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def compute_summary(tasks: list[TaskResult]) -> RunSummary:
    counts = {
        TaskStatus.passed: 0,
        TaskStatus.failed: 0,
        TaskStatus.error: 0,
        TaskStatus.running: 0,
        TaskStatus.pending: 0,
    }
    for task in tasks:
        counts[task.status] = counts.get(task.status, 0) + 1

    total = len(tasks)
    finished = counts[TaskStatus.passed] + counts[TaskStatus.failed] + counts[TaskStatus.error]
    pass_rate = (counts[TaskStatus.passed] / finished) if finished else None

    return RunSummary(
        total=total,
        passed=counts[TaskStatus.passed],
        failed=counts[TaskStatus.failed],
        error=counts[TaskStatus.error],
        running=counts[TaskStatus.running],
        pending=counts[TaskStatus.pending],
        pass_rate=pass_rate,
    )


def build_failure_summary(tasks: list[TaskResult]) -> list[FailureSummaryItem]:
    items: list[FailureSummaryItem] = []
    for task in tasks:
        if task.status == TaskStatus.failed:
            items.append(
                FailureSummaryItem(
                    task_id=task.task_id,
                    category=FailureCategory.verifier_failed,
                    message=task.remarks or "Verifier failed",
                )
            )
        elif task.status == TaskStatus.error:
            category = FailureCategory.unknown
            remarks = task.remarks or "Task error"
            lower = remarks.lower()
            if "timeout" in lower:
                category = FailureCategory.timeout
            elif "sandbox" in lower:
                category = FailureCategory.sandbox_error
            elif "infra" in lower:
                category = FailureCategory.infrastructure
            items.append(
                FailureSummaryItem(
                    task_id=task.task_id,
                    category=category,
                    message=remarks,
                )
            )
    return items


class RunRecord:
    """Snapshot of a run used by the mock runner and API responses."""

    def __init__(
        self,
        *,
        run_id: str,
        status: RunStatus,
        created_at: datetime,
        started_at: datetime | None,
        finished_at: datetime | None,
        task_ids: list[str],
        agent_model: str,
        tasks: list[TaskResult],
        error: RunError | None = None,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.created_at = created_at
        self.started_at = started_at
        self.finished_at = finished_at
        self.request = RunRequestEcho(task_ids=list(task_ids), agent_model=agent_model)
        self.tasks = list(tasks)
        self.error = error

    def to_response(self) -> RunResponse:
        return RunResponse(
            run_id=self.run_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            request=self.request,
            summary=compute_summary(self.tasks),
            tasks=list(self.tasks),
            failure_summary=build_failure_summary(self.tasks),
            error=self.error,
        )


def _row_to_record(row: RunRow) -> RunRecord:
    tasks = [
        TaskResult(
            task_id=t.task_id,
            status=TaskStatus(t.status),
            reward=t.reward,
            remarks=t.remarks,
        )
        for t in sorted(row.tasks, key=lambda x: x.position)
    ]
    error = None
    if row.error_code:
        error = RunError(code=row.error_code, message=row.error_message or "")
    return RunRecord(
        run_id=str(row.id),
        status=RunStatus(row.status),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        task_ids=[t.task_id for t in tasks],
        agent_model=row.agent_model,
        tasks=tasks,
        error=error,
    )


class PostgresRunStore:
    """Run store persisted in PostgreSQL; also acts as the job queue."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _factory(self):
        return self._session_factory or get_session_factory()

    def create(
        self,
        *,
        task_ids: list[str],
        agent_model: str,
        claimed_by: str | None = None,
        job_id: str | None = None,
    ) -> RunRecord:
        """
        Insert a new run row.

        By default the row is inserted ``queued``/unclaimed, exactly as before -
        this is what ``POST /v1/runs`` relies on. Pass ``claimed_by`` to insert
        the row already owned (``status=running``, ``worker_id=claimed_by``,
        ``started_at`` set) *in the same INSERT*, so the row can never be
        visible to ``claim_next``'s ``status='queued'`` predicate. This is how
        a job-driven evaluate run stays owned by the step that created it
        instead of being claimable by the legacy standalone-run queue - see C1
        in the Milestone 4 final review. A follow-up UPDATE after ``create()``
        would leave exactly that window open, so the ownership must be set at
        insert time.

        ``job_id`` marks job-owned evaluate runs (M16) so ops queries can
        distinguish them from ``POST /v1/runs`` rows.

        Deliberately does NOT set ``claimed_at`` for a ``claimed_by`` row (this
        is the one place this method departs from the final review's literal
        suggested shape, which also set ``claimed_at=now()``). ``claim_next``'s
        stale-running sweep below matches ANY row with
        ``status=running AND claimed_at IS NOT NULL AND claimed_at < stale_before``,
        with no notion of job ownership (see M16) - setting ``claimed_at`` here
        would make a still-legitimately-running evaluate run (harbor runs can
        take hours) eligible for that sweep once it merely runs longer than
        ``stale_after_sec`` (default 1800s), which would silently flip it back
        to ``queued`` and hand it straight to the legacy queue anyway - the
        exact theft this fix exists to prevent, just delayed instead of
        immediate. Leaving ``claimed_at`` unset keeps this row exactly as
        invisible to that sweep as it was before this fix (a pre-existing gap
        - see I3, deliberately out of scope here) while still being fully
        invisible to claim_next's normal claim, which only checks ``status``.
        """
        run_id = uuid.uuid4()
        now = _utcnow()
        status = RunStatus.queued.value
        started_at: datetime | None = None
        worker_id: str | None = None
        if claimed_by is not None:
            status = RunStatus.running.value
            started_at = now
            worker_id = claimed_by
        job_uid = None
        if job_id:
            try:
                job_uid = UUID(job_id)
            except ValueError:
                job_uid = None
        row = RunRow(
            id=run_id,
            status=status,
            agent_model=agent_model,
            created_at=now,
            started_at=started_at,
            worker_id=worker_id,
            job_id=job_uid,
            tasks=[
                RunTaskRow(
                    task_id=tid,
                    position=i,
                    status=TaskStatus.pending.value,
                )
                for i, tid in enumerate(task_ids)
            ],
        )
        session = self._factory()()
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            # reload with tasks
            row = session.scalar(
                select(RunRow)
                .where(RunRow.id == run_id)
                .options(selectinload(RunRow.tasks))
            )
            assert row is not None
            return _row_to_record(row)
        finally:
            session.close()

    def get(self, run_id: str) -> RunRecord | None:
        try:
            uid = UUID(run_id)
        except ValueError:
            return None
        session = self._factory()()
        try:
            row = session.scalar(
                select(RunRow)
                .where(RunRow.id == uid)
                .options(selectinload(RunRow.tasks))
            )
            if row is None:
                return None
            return _row_to_record(row)
        finally:
            session.close()

    def update(self, run_id: str, **fields: Any) -> RunRecord | None:
        try:
            uid = UUID(run_id)
        except ValueError:
            return None

        session = self._factory()()
        try:
            row = session.scalar(
                select(RunRow)
                .where(RunRow.id == uid)
                .options(selectinload(RunRow.tasks))
            )
            if row is None:
                return None

            if "status" in fields:
                status = fields["status"]
                row.status = status.value if isinstance(status, RunStatus) else status
            if "started_at" in fields:
                row.started_at = fields["started_at"]
            if "finished_at" in fields:
                row.finished_at = fields["finished_at"]
            if "error" in fields:
                err = fields["error"]
                if err is None:
                    row.error_code = None
                    row.error_message = None
                elif isinstance(err, RunError):
                    row.error_code = err.code
                    row.error_message = err.message
                else:
                    row.error_code = getattr(err, "code", "internal_error")
                    row.error_message = str(getattr(err, "message", err))
            if "worker_id" in fields:
                row.worker_id = fields["worker_id"]
            if "claimed_at" in fields:
                row.claimed_at = fields["claimed_at"]

            session.commit()
            session.refresh(row)
            return _row_to_record(row)
        finally:
            session.close()

    def set_task(
        self,
        run_id: str,
        task_id: str,
        *,
        status: TaskStatus,
        reward: float | None = None,
        remarks: str | None = None,
    ) -> None:
        try:
            uid = UUID(run_id)
        except ValueError:
            return

        session = self._factory()()
        try:
            task = session.scalar(
                select(RunTaskRow).where(
                    RunTaskRow.run_id == uid,
                    RunTaskRow.task_id == task_id,
                )
            )
            if task is None:
                return
            task.status = status.value if isinstance(status, TaskStatus) else status
            task.reward = reward
            task.remarks = remarks
            session.commit()
        finally:
            session.close()

    def claim_next(
        self,
        worker_id: str,
        *,
        stale_after_sec: int = 1800,
    ) -> str | None:
        """
        Atomically claim the next queued run (or a stale running run).

        Uses SELECT ... FOR UPDATE SKIP LOCKED so concurrent workers do not
        claim the same job.
        """
        session = self._factory()()
        try:
            now = _utcnow()
            stale_before = now - timedelta(seconds=stale_after_sec)

            # Requeue stale running jobs first (best-effort).
            session.execute(
                update(RunRow)
                .where(
                    RunRow.status == RunStatus.running.value,
                    RunRow.claimed_at.is_not(None),
                    RunRow.claimed_at < stale_before,
                )
                .values(
                    status=RunStatus.queued.value,
                    worker_id=None,
                    claimed_at=None,
                    started_at=None,
                )
            )

            row = session.scalar(
                select(RunRow)
                .where(RunRow.status == RunStatus.queued.value)
                .order_by(RunRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                session.commit()
                return None

            row.status = RunStatus.running.value
            row.worker_id = worker_id
            row.claimed_at = now
            row.started_at = now
            session.commit()
            return str(row.id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def clear(self) -> None:
        session = self._factory()()
        try:
            session.execute(delete(RunTaskRow))
            session.execute(delete(RunRow))
            session.commit()
        finally:
            session.close()


# Default process-wide store (API and worker construct their own as needed).
store = PostgresRunStore()

# Backwards-compatible alias used by older imports/tests.
RunStore = PostgresRunStore
