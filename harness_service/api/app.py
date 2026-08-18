"""FastAPI application factory + lifespan wiring.

Lifespan: create DB schema, (optionally) seed a dev org/user, start the worker.
Shutdown: stop the worker cleanly.

Run locally:
    uvicorn harness_service.api.app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from harness_service import __version__
from harness_service.api.routes import health, jobs
from harness_service.config import get_settings
from harness_service.db import init_db
from harness_service.services.seed import ensure_dev_principal
from harness_service.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("harness.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    logger.info("schema ready (db=%s)", settings.database_url.split("@")[-1])

    if settings.seed_dev_principal:
        await ensure_dev_principal(settings)

    worker = Worker(settings)
    if settings.worker_enabled:
        worker.start()
    app.state.worker = worker

    try:
        yield
    finally:
        await worker.stop()


_DESCRIPTION = """\
Backend service that runs the TerminalBench agent against a task subset, observes failures,
and (optimize mode) uses an LLM to iteratively improve the agent — proposing a new `agent.py`,
running it in an isolated **E2B sandbox**, and keeping the change only if it beats the best
score seen so far.

**Auth:** all `/v1` endpoints require an `X-API-Key` header (seeded dev key: `dev-key`).

**Job lifecycle:** `POST /v1/jobs` returns immediately with a `queued` job; a background worker
claims it, runs it, and persists the full, lossless iteration history (agent source, per-task
rewards + traces, the LLM's proposal request/response). Poll `GET /v1/jobs/{id}` until the
status is terminal, then read `GET /v1/jobs/{id}/iterations` for the complete trajectory.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Auto-Harness Optimization Service",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    return app


app = create_app()
