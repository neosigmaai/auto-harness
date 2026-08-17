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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Auto-Harness Optimization Service",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    return app


app = create_app()
