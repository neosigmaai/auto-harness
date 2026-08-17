#!/usr/bin/env python3
"""Embedded Postgres for local development — no Docker, no system install.

Uses `pgserver` (bundled Postgres binaries via pip) to run a real Postgres
server out of a local data directory. Everything lives inside the venv +
`.pgdata/`; nothing is installed system-wide, and deleting `.pgdata/` fully
removes it.

Usage:
    uv pip install -e ".[dev]"     # one-time: installs pgserver
    python scripts/dev_postgres.py

Leave it running in its own terminal, then in another terminal:
    uvicorn harness_service.api.app:app --reload
    python test_client.py ...

The script prints the DATABASE_URL to put in .env (or export directly — see below).
"""

from __future__ import annotations

import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / ".pgdata"


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # visible immediately even when piped/logged
    try:
        import pgserver
    except ImportError:
        print("pgserver not installed. Run: uv pip install -e '.[dev]'", file=sys.stderr)
        raise SystemExit(1)

    DATA_DIR.mkdir(exist_ok=True)
    server = pgserver.get_server(DATA_DIR)
    async_url = server.get_uri().replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"\nPostgres is running (data dir: {DATA_DIR})")
    print(f"DATABASE_URL={async_url}")
    print("\nPut that line in your .env, then in another terminal:")
    print("  uvicorn harness_service.api.app:app --reload")
    print("\nCtrl+C to stop this server.\n", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping postgres...")
        server.cleanup()


if __name__ == "__main__":
    main()
