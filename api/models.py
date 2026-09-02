"""SQLAlchemy ORM models for runs and tasks."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    agent_model: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(128))

    tasks: Mapped[list[RunTaskRow]] = relationship(
        "RunTaskRow",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunTaskRow.position",
    )


class RunTaskRow(Base):
    __tablename__ = "run_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[str] = mapped_column(String(256), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reward: Mapped[float | None] = mapped_column(Float)
    remarks: Mapped[str | None] = mapped_column(Text)

    run: Mapped[RunRow] = relationship("RunRow", back_populates="tasks")


class JobRow(Base):
    """An iterative-improvement job (Milestone 4)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    task_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    agent_model: Mapped[str] = mapped_column(String(256), nullable=False)
    improver_model: Mapped[str] = mapped_column(String(256), nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    patience: Mapped[int] = mapped_column(Integer, nullable=False)
    min_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    min_delta: Mapped[float] = mapped_column(Float, nullable=False)
    max_job_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluate_stale_after_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    current_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    non_improving_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Deliberately NOT a ForeignKey: jobs -> agent_versions -> jobs would be a cycle
    # and would break ON DELETE CASCADE for the job.
    best_agent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    best_score: Mapped[float | None] = mapped_column(Float)
    stop_reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class AgentVersionRow(Base):
    """Immutable snapshot of an AgentSpec for one job iteration."""

    __tablename__ = "agent_versions"
    __table_args__ = (
        UniqueConstraint("job_id", "version", name="uq_agent_versions_job_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    spec: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_by: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StepRow(Base):
    """A unit of queued work for a job: an evaluate step or an improve step."""

    __tablename__ = "steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    score: Mapped[float | None] = mapped_column(Float)
    # Per-task reward snapshot for an evaluate step: {task_id: reward | None}.
    # Stored here so per-task movement (fixed / regressed) is derivable without
    # joining back to run_tasks, and survives even if a run row is later pruned.
    task_rewards: Mapped[dict | None] = mapped_column(JSONB)
    stale_after_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
