from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

RunStatus = Literal["queued", "running", "succeeded", "failed", "timed_out", "cancelled"]
TaskStatus = Literal["queued", "running", "passed", "failed", "infra_failed", "timed_out"]

RUN_STATUSES = {"queued", "running", "succeeded", "failed", "timed_out", "cancelled"}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
TASK_STATUSES = {"queued", "running", "passed", "failed", "infra_failed", "timed_out"}


@dataclass(frozen=True)
class TaskResultRecord:
    task_id: str
    status: TaskStatus
    reward: float | None
    failure_type: str | None = None
    error_summary: str | None = None
    trace_path: str | None = None
    result_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureSummary:
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    agent_failures: int
    infra_failures: int
    top_failure_modes: list[str]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: RunStatus
    task_ids: list[str]
    mode: str
    model: str
    sandbox_provider: str
    requested_concurrency: int
    max_iterations: int
    org_id: str
    created_by: str
    score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class IterationRecord:
    run_id: str
    iteration_index: int
    status: str
    agent_version: str
    score: float | None = None
    proposal: str | None = None
    accepted: bool | None = None
