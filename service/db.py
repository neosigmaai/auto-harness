"""asyncpg pool + tiny SQL migration runner (numbered files in service/migrations)."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys

import asyncpg

from .config import settings

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def _init_conn(conn: asyncpg.Connection) -> None:
    # Make jsonb transparent: Python dict <-> jsonb.
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def get_pool() -> asyncpg.Pool:
    global _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                init=_init_conn,
            )
    return _pool


async def close_pool() -> None:
    global _pool
    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None


async def migrate(dsn: str | None = None) -> list[str]:
    """Apply any migration files not yet recorded in schema_migrations. Idempotent."""
    conn = await asyncpg.connect(dsn or settings.database_url)
    applied: list[str] = []
    try:
        # Multiple API/worker deployments may start together. Serialize schema
        # changes without imposing a lock on normal application transactions.
        await conn.execute('SELECT pg_advisory_lock(781204991)')
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute("INSERT INTO schema_migrations(name) VALUES ($1)", path.name)
            applied.append(path.name)
    finally:
        try:
            await conn.execute('SELECT pg_advisory_unlock(781204991)')
        finally:
            await conn.close()
    return applied


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        names = asyncio.run(migrate(os.environ.get("DATABASE_URL")))
        print(f"[db] applied {len(names)} migration(s): {names}")
    else:
        print("usage: python -m service.db migrate")
