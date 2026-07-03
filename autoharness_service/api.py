from __future__ import annotations

from contextlib import asynccontextmanager

from autoharness_service.config import load_settings
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.schemas import (
    IterationResponse,
    IterationsResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunResultsResponse,
    RunStatusResponse,
    TaskListResponse,
)
from autoharness_service.service import RunService
from autoharness_service.store import PostgresStore
from fastapi import FastAPI, Header, HTTPException, Response, status

DEFAULT_TASKS = [
    "break-filter-js-from-html",
    "multi-source-data-merger",
]
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
READ_ROLES = {"viewer", "runner", "admin"}
WRITE_ROLES = {"runner", "admin"}


def create_app(
    service: RunService | None = None,
    *,
    start_background: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_background:
            app.state.service.start_polling(interval_sec=2.0, limit=10)
        try:
            yield
        finally:
            app.state.service.stop_polling()

    app = FastAPI(
        title="Agent Optimization Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service or _build_service()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks", response_model=TaskListResponse)
    def list_tasks() -> TaskListResponse:
        return TaskListResponse(tasks=DEFAULT_TASKS)

    @app.post(
        "/runs",
        response_model=RunCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        request: RunCreateRequest,
        response: Response,
        x_org_id: str = Header(default="default-org", alias="X-Org-Id"),
        x_user_id: str = Header(default="local-user", alias="X-User-Id"),
        x_role: str = Header(alias="X-Role"),
    ) -> RunCreateResponse:
        if x_role not in WRITE_ROLES:
            raise HTTPException(status_code=403, detail="viewer cannot create runs")
        run = app.state.service.submit_run(
            request,
            org_id=x_org_id,
            created_by=x_user_id,
            start_background=start_background,
        )
        response.headers["Location"] = f"/runs/{run.run_id}"
        return RunCreateResponse(
            run_id=run.run_id,
            status=run.status,
            created_at=run.created_at,
            status_url=f"/runs/{run.run_id}",
            result_url=f"/runs/{run.run_id}/results",
        )

    @app.get("/runs/{run_id}", response_model=RunStatusResponse)
    def get_run(
        run_id: str,
        x_org_id: str = Header(default="default-org", alias="X-Org-Id"),
        x_user_id: str = Header(default="local-user", alias="X-User-Id"),
        x_role: str = Header(alias="X-Role"),
    ) -> RunStatusResponse:
        _authorize_run_read(app.state.service, run_id, x_org_id, x_user_id, x_role)
        run_status = app.state.service.get_run_status(run_id, org_id=x_org_id)
        if run_status is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run_status

    @app.get("/runs/{run_id}/results", response_model=RunResultsResponse)
    def get_results(
        run_id: str,
        x_org_id: str = Header(default="default-org", alias="X-Org-Id"),
        x_user_id: str = Header(default="local-user", alias="X-User-Id"),
        x_role: str = Header(alias="X-Role"),
    ) -> RunResultsResponse:
        _authorize_run_read(app.state.service, run_id, x_org_id, x_user_id, x_role)
        run_results = app.state.service.get_run_results(run_id, org_id=x_org_id)
        if run_results is None:
            raise HTTPException(status_code=404, detail="run not found")
        if run_results.status not in TERMINAL_RUN_STATUSES:
            raise HTTPException(status_code=409, detail="run is not finished")
        return run_results

    @app.get("/runs/{run_id}/iterations", response_model=IterationsResponse)
    def get_iterations(
        run_id: str,
        x_org_id: str = Header(default="default-org", alias="X-Org-Id"),
        x_user_id: str = Header(default="local-user", alias="X-User-Id"),
        x_role: str = Header(alias="X-Role"),
    ) -> IterationsResponse:
        _authorize_run_read(app.state.service, run_id, x_org_id, x_user_id, x_role)
        run_status = app.state.service.get_run_status(run_id, org_id=x_org_id)
        if run_status is None:
            raise HTTPException(status_code=404, detail="run not found")
        iterations = app.state.service.store.list_iterations(run_id, org_id=x_org_id)
        return IterationsResponse(
            run_id=run_id,
            iterations=[
                IterationResponse(
                    iteration=_iteration_field(item, "iteration_index"),
                    agent_version=_iteration_field(item, "agent_version"),
                    status=_iteration_field(item, "status"),
                    score=_iteration_field(item, "score"),
                    proposal=_iteration_field(item, "proposal"),
                    accepted=_iteration_field(item, "accepted"),
                )
                for item in iterations
            ],
        )

    return app


def _build_service() -> RunService:
    settings = load_settings()
    store = PostgresStore(settings.database_url)
    return RunService(
        store=store,
        simulated_runner=SimulatedBenchmarkRunner(),
        max_local_concurrency=settings.max_local_concurrency,
    )


def _authorize_run_read(
    service: RunService,
    run_id: str,
    org_id: str,
    user_id: str,
    role: str,
) -> None:
    if role not in READ_ROLES:
        raise HTTPException(status_code=403, detail="invalid role")

    run = service.store.get_run(run_id, org_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if role == "admin":
        return
    if run.created_by != user_id:
        raise HTTPException(status_code=403, detail="user cannot access this run")


def _iteration_field(item: object, name: str):
    if hasattr(item, name):
        return getattr(item, name)
    return item[name]  # type: ignore[index]
