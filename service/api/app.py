"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import settings
from ..db import close_pool, get_pool
from ..errors import install_error_handlers
from . import routes_admin, routes_jobs, routes_tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    yield
    await close_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agent Optimization Service",
        version="0.1.0",
        description=(
            "Runs a Terminal-Bench agent against a curated task subset inside sandboxes, observes failures, "
            "asks an LLM for one improvement per iteration, and re-runs. Submit a job, poll for the result."
        ),
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.include_router(routes_jobs.router)
    app.include_router(routes_tasks.router)
    app.include_router(routes_admin.router)

    @app.get("/v1/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok", "executor": settings.executor, "optimizer": settings.optimizer}

    @app.get("/v1/readyz", tags=["ops"])
    async def readyz():
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ready"}

    return app


app = create_app()
