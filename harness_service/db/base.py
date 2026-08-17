"""Async SQLAlchemy engine, session factory, and schema bootstrap.

MVP uses ``create_all`` on startup rather than Alembic migrations (see PLAN.md
§2 — Alembic is the noted next step).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from harness_service.config import get_settings

_settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    _settings.database_url,
    echo=_settings.db_echo,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create tables if they don't exist. Imports models to register metadata."""
    from harness_service.db import models  # noqa: F401  (register mappers)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for worker / service code: commit on success, roll back on error."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — one session per request."""
    async with SessionLocal() as session:
        yield session
