"""FastAPI application entrypoint for the auto-harness benchmark API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import load_config
from api.db import init_db, reset_engine
from api.env import load_repo_dotenv
from api.job_store import PostgresJobStore, job_store as default_job_store
from api.routes import agent_versions, jobs, runs, tasks
from api.schemas import ErrorDetail, ErrorResponse, HealthResponse
from api.store import PostgresRunStore, store as default_store


def create_app(
    *,
    store: PostgresRunStore | None = None,
    job_store: PostgresJobStore | None = None,
    database_url: str | None = None,
    init_database: bool = True,
) -> FastAPI:
    # Load .env before config / DB URL resolution so E2B_API_KEY etc. are visible.
    load_repo_dotenv()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if init_database:
            if database_url:
                reset_engine()
                init_db(url=database_url)
            else:
                init_db()
        yield

    app = FastAPI(
        title="auto-harness Benchmark API",
        version="0.2.0",
        description=(
            "Submit and poll Terminal-Bench agent runs. "
            "API enqueues work; a separate worker process executes it."
        ),
        lifespan=lifespan,
    )

    run_store = store or default_store
    app.state.store = run_store
    app.state.job_store = job_store or default_job_store

    # Eager-load config so misconfiguration fails at startup.
    load_config()

    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(agent_versions.router)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        code = "validation_error"
        message = "Request validation failed"
        details: dict | None = {"errors": exc.errors()}

        for err in exc.errors():
            loc = err.get("loc", ())
            if "task_ids" in loc:
                err_msg = str(err.get("msg", ""))
                if "non-empty" in err_msg.lower() or "at least" in err_msg.lower():
                    code = "empty_task_ids"
                    message = "task_ids must be non-empty when provided"
                    details = None
                    break

        body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
        return JSONResponse(status_code=422, content=body.model_dump())

    return app


app = create_app()
