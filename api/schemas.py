"""Pydantic request/response models for the benchmark run API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    passed = "passed"
    failed = "failed"
    error = "error"


class FailureCategory(str, Enum):
    verifier_failed = "verifier_failed"
    timeout = "timeout"
    sandbox_error = "sandbox_error"
    infrastructure = "infrastructure"
    unknown = "unknown"


# ── Requests ───────────────────────────────────────────────────────────────


class CreateRunRequest(BaseModel):
    """Body for POST /v1/runs."""

    task_ids: list[str] | None = Field(
        default=None,
        description="Tasks to run. Omit or null to use the configured default subset.",
    )
    agent_model: str | None = Field(
        default=None,
        description="Optional LLM model override for the harness agent.",
    )

    @field_validator("task_ids")
    @classmethod
    def task_ids_not_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("task_ids must be non-empty when provided")
        return value


# ── Nested response pieces ─────────────────────────────────────────────────


class RunRequestEcho(BaseModel):
    task_ids: list[str]
    agent_model: str


class RunSummary(BaseModel):
    total: int
    passed: int
    failed: int
    error: int
    running: int
    pending: int
    pass_rate: float | None = None


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    reward: float | None = None
    remarks: str | None = None


class FailureSummaryItem(BaseModel):
    task_id: str
    category: FailureCategory
    message: str


class RunError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


# ── Top-level responses ────────────────────────────────────────────────────


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: datetime


class RunResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    request: RunRequestEcho
    summary: RunSummary
    tasks: list[TaskResult]
    failure_summary: list[FailureSummaryItem] = Field(default_factory=list)
    error: RunError | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


class TaskListResponse(BaseModel):
    default_task_ids: list[str]
    default_agent_model: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
