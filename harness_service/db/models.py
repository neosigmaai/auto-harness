"""SQLAlchemy ORM — 1:1 with the domain model (PLAN.md §3a).

Nothing is lost: every candidate agent source (accepted *or* rejected), every
LLM request/response, and every per-task trace excerpt is persisted and reachable
via the API.

Multi-tenancy columns (org_id, user_id, role, api_key) exist from day one;
enforcement is added in M5.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from harness_service.constants import (
    ExecutorKind,
    IterationDecision,
    JobMode,
    JobStatus,
    ProposerKind,
    Role,
)
from harness_service.db.base import Base


def _uuid_col(**kw) -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), default=uuid.uuid4, **kw)


def _enum(py_enum, **kw):
    # native_enum=False → stored as VARCHAR, no Postgres enum type to migrate.
    return mapped_column(SAEnum(py_enum, native_enum=False, length=32), **kw)


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = _uuid_col(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship(back_populates="org")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_col(primary_key=True)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    role = _enum(Role, default=Role.MEMBER)
    api_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    org: Mapped["Organization"] = relationship(back_populates="users")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_col(primary_key=True)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)

    mode = _enum(JobMode, default=JobMode.SINGLE_RUN)
    executor = _enum(ExecutorKind, default=ExecutorKind.SIMULATED)
    status = _enum(JobStatus, default=JobStatus.QUEUED, index=True)

    subset: Mapped[list] = mapped_column(JSONB, default=list)   # resolved task_ids
    config: Mapped[dict] = mapped_column(JSONB, default=dict)   # model, effort, provider…
    max_iterations: Mapped[int] = mapped_column(Integer, default=1)
    patience: Mapped[int] = mapped_column(Integer, default=2)

    best_val_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # best TRAIN score
    best_iteration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("iterations.id", use_alter=True), nullable=True
    )
    accumulated_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Optimize mode (M4): train/test split + held-out generalization ──
    train_subset: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    test_subset: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    baseline_val_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # iter0 TRAIN
    test_val_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # best agent on TEST
    test_results: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # per-task on TEST

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    iterations: Mapped[list["Iteration"]] = relationship(
        back_populates="job",
        order_by="Iteration.idx",
        cascade="all, delete-orphan",
        foreign_keys="Iteration.job_id",
    )


class Iteration(Base):
    __tablename__ = "iterations"
    __table_args__ = (UniqueConstraint("job_id", "idx", name="uq_iteration_job_idx"),)

    id: Mapped[uuid.UUID] = _uuid_col(primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)

    # ── AgentState (snapshot, inlined per iteration) ──
    agent_source: Mapped[str] = mapped_column(Text)
    agent_params: Mapped[dict] = mapped_column(JSONB, default=dict)
    agent_hash: Mapped[str] = mapped_column(String(64), index=True)

    # ── Improvement that PRODUCED this state (nullable on baseline) ──
    proposer = _enum(ProposerKind, nullable=True)
    proposal_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposal_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_request: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    llm_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Result rollup + decision ──
    val_score: Mapped[float] = mapped_column(Float, default=0.0)
    n_passed: Mapped[int] = mapped_column(Integer, default=0)
    n_failed: Mapped[int] = mapped_column(Integer, default=0)
    decision = _enum(IterationDecision, default=IterationDecision.BASELINE)
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    context_snapshot: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="iterations", foreign_keys=[job_id])
    task_results: Mapped[list["TaskResult"]] = relationship(
        back_populates="iteration", cascade="all, delete-orphan"
    )


class TaskResult(Base):
    __tablename__ = "task_results"

    id: Mapped[uuid.UUID] = _uuid_col(primary_key=True)
    iteration_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("iterations.id"), index=True)

    task_id: Mapped[str] = mapped_column(String(255))
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = infra error
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    iteration: Mapped["Iteration"] = relationship(back_populates="task_results")
