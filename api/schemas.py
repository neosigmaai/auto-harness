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


# ── Jobs: iterative optimization loop (Milestone 4) ────────────────────────


class CreateJobRequest(BaseModel):
    """Body for POST /v1/jobs. Every field is optional; omitted fields fall
    back to config defaults at the route layer, not here."""

    task_ids: list[str] | None = Field(
        default=None,
        description="Tasks to evaluate each iteration. Omit or null to use the configured default subset.",
    )
    agent_model: str | None = Field(
        default=None,
        description="Optional LLM model override for the harness agent (spec v0).",
    )
    improver_model: str | None = Field(
        default=None,
        description="Optional LLM model override for the improver.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum evaluate iterations before stopping.",
    )
    patience: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Consecutive non-improving evaluations tolerated before stopping.",
    )
    min_iterations: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Noise floor: no_improvement cannot fire before this many iterations.",
    )
    min_delta: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Score increase required to count an iteration as an improvement.",
    )

    @field_validator("task_ids")
    @classmethod
    def task_ids_not_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("task_ids must be non-empty when provided")
        return value


class CreateJobResponse(BaseModel):
    job_id: str
    status: RunStatus
    created_at: datetime
    warnings: list[str] = Field(default_factory=list)


class JobConfigEcho(BaseModel):
    task_ids: list[str]
    agent_model: str
    improver_model: str
    max_iterations: int
    patience: int
    min_iterations: int
    min_delta: float


class JobBest(BaseModel):
    agent_version_id: str
    version: int
    score: float | None = None


class AgentSpecView(BaseModel):
    """Read-only mirror of api.agent_spec.AgentSpec for API responses."""

    system_prompt: str
    agent_model: str
    max_steps: int
    max_output_chars: int
    exec_timeout_sec: int


class ProposalView(BaseModel):
    rationale: str
    changed_fields: list[str] = Field(default_factory=list)
    based_on_version: int | None = None


class IterationView(BaseModel):
    iteration: int
    agent_version_id: str
    version: int
    run_id: str | None = None
    score: float | None = None
    improved: bool | None = None
    summary: RunSummary | None = None
    proposal: ProposalView | None = None
    fixed_tasks: list[str] = Field(default_factory=list)
    regressed_tasks: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    config: JobConfigEcho
    current_iteration: int = 0
    best: JobBest | None = None
    stop_reason: str | None = None
    iterations: list[IterationView] = Field(default_factory=list)
    error: RunError | None = None


class BestAgentResponse(BaseModel):
    job_id: str
    agent_version_id: str
    version: int
    score: float | None = None
    rationale: str = ""
    spec: AgentSpecView


class AgentVersionResponse(BaseModel):
    agent_version_id: str
    job_id: str
    version: int
    parent_version_id: str | None = None
    rationale: str = ""
    created_by: str
    created_at: datetime
    spec: AgentSpecView
