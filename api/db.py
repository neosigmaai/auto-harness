"""Database engine and session helpers."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
    )


def get_engine(*, url: str | None = None, force_new: bool = False) -> Engine:
    global _engine, _SessionLocal
    if _engine is not None and not force_new and url is None:
        return _engine

    db_url = url or get_database_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    if url is None or force_new:
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return engine


def get_session_factory(*, url: str | None = None) -> sessionmaker[Session]:
    global _SessionLocal
    get_engine(url=url)
    assert _SessionLocal is not None
    return _SessionLocal


def init_db(*, url: str | None = None) -> None:
    """Create tables if they do not exist."""
    engine = get_engine(url=url)
    Base.metadata.create_all(bind=engine)


def reset_engine() -> None:
    """Dispose the global engine (for tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_db() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
