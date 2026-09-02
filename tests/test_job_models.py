"""ORM-level tests for the Milestone 4 jobs / agent_versions / steps tables."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, func, inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError

from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.models import AgentVersionRow, JobRow, StepRow

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)

SPEC = {
    "system_prompt": "You are a terminal agent.",
    "agent_model": "gpt-4.1-mini",
    "max_steps": 80,
    "max_output_chars": 8000,
    "exec_timeout_sec": 120,
}


def _truncate(factory) -> None:
    session = factory()
    try:
        session.execute(delete(StepRow))
        session.execute(delete(AgentVersionRow))
        session.execute(delete(JobRow))
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def factory():
    # DATABASE_URL must be in the environment BEFORE get_session_factory(): db.py only
    # caches the global engine when called with url=None, which reads the env var.
    os.environ["DATABASE_URL"] = DATABASE_URL
    reset_engine()
    init_db(url=DATABASE_URL)
    session_factory = get_session_factory()
    _truncate(session_factory)
    yield session_factory
    _truncate(session_factory)
    reset_engine()


def _new_job_row(job_id: uuid.UUID, now: datetime) -> JobRow:
    return JobRow(
        id=job_id,
        status="queued",
        task_ids=["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=5,
        patience=2,
        min_delta=0.01,
        max_job_duration_sec=21600,
        evaluate_stale_after_sec=3600,
        current_iteration=0,
        non_improving_streak=0,
        created_at=now,
    )


def test_init_db_creates_the_three_job_tables(factory) -> None:
    inspector = inspect(get_engine())
    names = set(inspector.get_table_names())
    assert {"jobs", "agent_versions", "steps"} <= names

    job_columns = {c["name"] for c in inspector.get_columns("jobs")}
    assert "evaluate_stale_after_sec" in job_columns
    assert "max_job_duration_sec" in job_columns
    assert "non_improving_streak" in job_columns
    assert "best_agent_version_id" in job_columns

    step_columns = {c["name"] for c in inspector.get_columns("steps")}
    assert {"type", "iteration", "agent_version_id", "stale_after_sec"} <= step_columns


def test_deleting_a_job_cascades_versions_and_steps(factory) -> None:
    job_id = uuid.uuid4()
    version_id = uuid.uuid4()
    step_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    session = factory()
    try:
        session.add(_new_job_row(job_id, now))
        session.flush()
        session.add(
            AgentVersionRow(
                id=version_id,
                job_id=job_id,
                version=0,
                parent_version_id=None,
                spec=SPEC,
                rationale="baseline",
                created_by="baseline",
                created_at=now,
            )
        )
        session.flush()
        session.add(
            StepRow(
                id=step_id,
                job_id=job_id,
                type="evaluate",
                status="queued",
                iteration=0,
                agent_version_id=version_id,
                stale_after_sec=3600,
                created_at=now,
            )
        )
        session.commit()

        assert session.scalar(select(func.count()).select_from(AgentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(StepRow)) == 1

        # Bulk DELETE so the database (not the ORM) performs the cascade.
        session.execute(delete(JobRow).where(JobRow.id == job_id))
        session.commit()

        assert session.scalar(select(func.count()).select_from(JobRow)) == 0
        assert session.scalar(select(func.count()).select_from(AgentVersionRow)) == 0
        assert session.scalar(select(func.count()).select_from(StepRow)) == 0
    finally:
        session.close()


def test_agent_version_number_is_unique_per_job(factory) -> None:
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    session = factory()
    try:
        session.add(_new_job_row(job_id, now))
        session.flush()
        session.add(
            AgentVersionRow(
                id=uuid.uuid4(),
                job_id=job_id,
                version=0,
                spec=SPEC,
                rationale="baseline",
                created_by="baseline",
                created_at=now,
            )
        )
        session.commit()

        session.add(
            AgentVersionRow(
                id=uuid.uuid4(),
                job_id=job_id,
                version=0,
                spec=SPEC,
                rationale="duplicate",
                created_by="improver",
                created_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    finally:
        session.close()
