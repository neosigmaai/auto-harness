"""Pydantic wire schemas (server side) — serialize the domain model to/from JSON."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from harness_service import __version__
from harness_service.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_SUBSET,
    ExecutorKind,
    IterationDecision,
    JobMode,
    JobStatus,
    ProposerKind,
)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    db: str  # "ok" | "error: <msg>"
    worker_enabled: bool


# ── requests ──
class JobCreate(BaseModel):
    mode: JobMode = JobMode.SINGLE_RUN
    executor: ExecutorKind | None = None  # None → server default
    subset: str | list[str] = DEFAULT_SUBSET  # subset name or explicit task ids
    max_iterations: int | None = Field(default=None, ge=1, le=50)
    patience: int | None = Field(default=None, ge=1, le=20)
    config: dict = Field(default_factory=dict)  # e.g. agent_model, reasoning_effort


# ── responses ──
class TaskResultRead(BaseModel):
    task_id: str
    reward: float | None
    passed: bool
    duration_s: float | None = None
    failure_reason: str | None = None
    trace_excerpt: str | None = None


class IterationRead(BaseModel):
    idx: int
    decision: IterationDecision
    decision_reason: str
    val_score: float
    n_passed: int
    n_failed: int
    agent_hash: str
    agent_params: dict
    proposer: ProposerKind | None = None
    proposal_rationale: str | None = None
    proposal_diff: str | None = None
    created_at: datetime
    task_results: list[TaskResultRead] = []
    # full source is available but bulky; included so history is truly lossless.
    agent_source: str | None = None
    # Full audit trail of the proposal LLM call (what it was shown / what it returned).
    llm_request: dict | None = None
    llm_response: dict | None = None


class JobSummary(BaseModel):
    """Structured result of the best iteration (M1 = the baseline iteration)."""

    val_score: float
    n_passed: int
    n_failed: int
    passed_tasks: list[str]
    failed_tasks: list[str]
    failures: list[dict]  # [{task_id, failure_reason}]


class JobRead(BaseModel):
    id: UUID
    org_id: UUID
    user_id: UUID
    mode: JobMode
    executor: ExecutorKind
    status: JobStatus
    subset: list[str]
    max_iterations: int
    patience: int
    n_iterations: int
    best_val_score: float | None = None       # best TRAIN score (optimize) / only score (single_run)
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    summary: JobSummary | None = None
    # ── optimize (M4) ──
    baseline_val_score: float | None = None    # iter 0 (train)
    train_subset: list[str] | None = None
    test_subset: list[str] | None = None
    test_val_score: float | None = None        # best agent on held-out test
    improvement: float | None = None           # best_train - baseline_train


class JobListItem(BaseModel):
    id: UUID
    mode: JobMode
    executor: ExecutorKind
    status: JobStatus
    best_val_score: float | None = None
    n_iterations: int
    created_at: datetime


# echo the defaults so clients can discover them
class Defaults(BaseModel):
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    patience: int = DEFAULT_PATIENCE
    subset: str = DEFAULT_SUBSET
