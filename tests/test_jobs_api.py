"""API tests for the Milestone 4 iterative-improvement job endpoints."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api.config import clear_config_cache, load_config
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import EvaluateOutcome, PostgresJobStore
from api.main import create_app
from api.services.improver import FakeImprover, create_improver
from api.services.runner import MockBenchmarkRunner
from api.store import PostgresRunStore
from worker.main import process_one
from worker.steps import StepExecutor

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


@pytest.fixture()
def db_store() -> PostgresRunStore:
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    store = PostgresRunStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


@pytest.fixture()
def job_store(db_store: PostgresRunStore) -> PostgresJobStore:
    store = PostgresJobStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()


@pytest.fixture()
def client(db_store: PostgresRunStore, job_store: PostgresJobStore) -> TestClient:
    clear_config_cache()
    app = create_app(
        store=db_store,
        job_store=job_store,
        database_url=DATABASE_URL,
        init_database=True,
    )
    with TestClient(app) as test_client:
        yield test_client
    clear_config_cache()


def test_post_job_with_empty_body_uses_config_defaults(client: TestClient) -> None:
    cfg = load_config()
    resp = client.post("/v1/jobs", json={})
    assert resp.status_code == 202
    created = resp.json()
    assert created["job_id"]
    assert created["status"] == "queued"
    assert created["created_at"]

    got = client.get(f"/v1/jobs/{created['job_id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["job_id"] == created["job_id"]
    assert body["status"] == "queued"
    assert body["config"]["task_ids"] == cfg.default_task_ids
    assert body["config"]["agent_model"] == cfg.default_agent_model
    assert body["config"]["improver_model"] == cfg.improver_model
    assert body["config"]["max_iterations"] == cfg.max_iterations
    assert body["config"]["patience"] == cfg.patience
    assert body["config"]["min_iterations"] == cfg.min_iterations
    assert body["config"]["min_delta"] == cfg.min_delta


def test_post_job_honours_explicit_overrides(client: TestClient) -> None:
    resp = client.post(
        "/v1/jobs",
        json={
            "task_ids": ["fix-git", "regex-log"],
            "agent_model": "override-agent",
            "improver_model": "override-improver",
            "max_iterations": 2,
            "patience": 1,
            "min_iterations": 1,
            "min_delta": 0.0,
        },
    )
    assert resp.status_code == 202
    body = client.get(f"/v1/jobs/{resp.json()['job_id']}").json()
    assert body["config"] == {
        "task_ids": ["fix-git", "regex-log"],
        "agent_model": "override-agent",
        "improver_model": "override-improver",
        "max_iterations": 2,
        "patience": 1,
        "min_iterations": 1,
        "min_delta": 0.0,
    }


def test_post_job_unknown_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/jobs", json={"task_ids": ["not-a-real-task"]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unknown_task_ids"
    assert "not-a-real-task" in body["error"]["details"]["unknown"]


def test_post_job_empty_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/jobs", json={"task_ids": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_task_ids"


def test_get_job_not_found(client: TestClient) -> None:
    resp = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"


def test_fresh_job_shows_queued_iteration_zero_and_no_best(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"task_ids": ["fix-git"]}).json()["job_id"]
    body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["current_iteration"] == 0
    assert body["best"] is None
    assert body["stop_reason"] is None
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["error"] is None
    # create_job enqueues an evaluate step at iteration 0, and get_job returns one
    # IterationRecord per evaluate step regardless of status — so a fresh job reports
    # exactly one iteration, still queued, with no score and no proposal yet.
    assert len(body["iterations"]) == 1
    it = body["iterations"][0]
    assert it["iteration"] == 0
    assert it["version"] == 0
    assert it["score"] is None
    assert it["run_id"] is None
    assert it["proposal"] is None
    assert it["summary"] is None


def test_evaluate_stale_after_sec_uses_config_formula(client: TestClient) -> None:
    cfg = load_config()
    from api.routes.jobs import evaluate_stale_after_sec

    expected = int(math.ceil(4 / cfg.max_concurrency) * cfg.per_task_timeout * 1.2)
    assert evaluate_stale_after_sec(4, cfg) == expected


def test_best_is_409_before_any_evaluation(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"task_ids": ["fix-git"]}).json()["job_id"]
    resp = client.get(f"/v1/jobs/{job_id}/best")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "no_evaluation_yet"


def test_best_not_found_for_unknown_job(client: TestClient) -> None:
    resp = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000/best")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"


def test_best_returns_winning_spec_inline(
    client: TestClient,
    db_store: PostgresRunStore,
    job_store: PostgresJobStore,
) -> None:
    created = client.post(
        "/v1/jobs",
        json={"task_ids": ["fix-git", "regex-log"], "agent_model": "winning-model"},
    ).json()
    job_id = created["job_id"]

    # Complete iteration 0's evaluate step by hand (no worker involved).
    step = job_store.claim_next_step("manual-worker")
    assert step is not None
    assert step.type == "evaluate"
    assert step.iteration == 0
    assert step.version == 0
    run = db_store.create(task_ids=step.task_ids, agent_model=step.spec.agent_model)
    job_store.complete_step_and_advance(
        step.step_id,
        EvaluateOutcome(run_id=run.run_id, score=0.5),
    )

    resp = client.get(f"/v1/jobs/{job_id}/best")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["agent_version_id"] == step.agent_version_id
    assert body["version"] == 0
    assert body["score"] == pytest.approx(0.5)
    assert body["rationale"] == "baseline"
    assert body["spec"]["agent_model"] == "winning-model"
    assert body["spec"]["system_prompt"]
    assert body["spec"]["max_steps"] >= 1

    # The job view now shows the scored iteration and its run summary.
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["best"] == {
        "agent_version_id": step.agent_version_id,
        "version": 0,
        "score": pytest.approx(0.5),
    }
    assert len(job["iterations"]) == 1
    iteration = job["iterations"][0]
    assert iteration["iteration"] == 0
    assert iteration["run_id"] == run.run_id
    assert iteration["score"] == pytest.approx(0.5)
    assert iteration["improved"] is True
    assert iteration["proposal"] is None
    assert iteration["summary"]["total"] == 2
    assert iteration["summary"]["pending"] == 2


def test_agent_version_not_found_for_random_uuid(client: TestClient) -> None:
    resp = client.get("/v1/agent-versions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_version_not_found"


def test_agent_version_not_found_for_malformed_id(client: TestClient) -> None:
    resp = client.get("/v1/agent-versions/not-a-uuid")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_version_not_found"


def test_agent_version_returns_baseline_v0(
    client: TestClient, job_store: PostgresJobStore
) -> None:
    created = client.post(
        "/v1/jobs",
        json={"task_ids": ["fix-git"], "agent_model": "baseline-model"},
    ).json()

    step = job_store.claim_next_step("manual-worker")
    assert step is not None
    version_id = step.agent_version_id

    resp = client.get(f"/v1/agent-versions/{version_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_version_id"] == version_id
    assert body["job_id"] == created["job_id"]
    assert body["version"] == 0
    assert body["parent_version_id"] is None
    assert body["created_by"] == "baseline"
    assert body["rationale"] == "baseline"
    assert body["created_at"]
    assert body["spec"]["agent_model"] == "baseline-model"
    assert body["spec"]["system_prompt"]
    assert body["spec"]["max_steps"] >= 1
    assert body["spec"]["max_output_chars"] >= 500
    assert body["spec"]["exec_timeout_sec"] >= 10


def _drive_job_to_completion(
    job_store: PostgresJobStore,
    run_store: PostgresRunStore,
    artifacts_root: Path,
    *,
    max_steps: int = 24,
) -> None:
    """Run the worker step loop synchronously until nothing is claimable."""
    cfg = load_config()
    assert cfg.execution_backend == "mock"

    improver = create_improver(cfg)
    assert isinstance(improver, FakeImprover)

    from api.services.artifacts import LocalArtifactStore

    executor = StepExecutor(
        job_store,
        run_store,
        config=cfg,
        improver=improver,
        artifacts=LocalArtifactStore(artifacts_root),
        step_delay_sec=0.0,
    )
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    for _ in range(max_steps):
        did_work = process_one(
            run_store,
            runner,
            worker_id="e2e-worker",
            stale_after_sec=1800,
            job_store=job_store,
            step_executor=executor,
        )
        if not did_work:
            return
    raise AssertionError(f"job did not settle within {max_steps} worker steps")


def test_job_end_to_end_through_worker(
    client: TestClient,
    db_store: PostgresRunStore,
    job_store: PostgresJobStore,
    tmp_path,
) -> None:
    created = client.post(
        "/v1/jobs",
        json={
            "task_ids": ["fix-git", "regex-log"],
            "agent_model": "e2e-model",
            "max_iterations": 3,
            "patience": 2,
            "min_delta": 0.0,
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    _drive_job_to_completion(job_store, db_store, tmp_path / "artifacts")

    body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["status"] == "completed"
    # Fully determined: MockBenchmarkRunner scores fix-git 1.0 and regex-log 0.0 every
    # time, so score is 0.5 at every iteration. With min_delta=0.0 iteration 0 improves
    # (best is None) and 1-2 do not; iteration 2 satisfies BOTH max_iterations and
    # patience, and stop precedence puts max_iterations first.
    assert body["stop_reason"] == "max_iterations"
    assert body["current_iteration"] == 2
    assert len(body["iterations"]) == 3
    assert [it["improved"] for it in body["iterations"]] == [True, False, False]
    assert body["best"]["version"] == 0
    assert body["best"]["score"] == pytest.approx(0.5)
    assert body["finished_at"] is not None
    assert body["error"] is None
    assert body["iterations"], "expected at least the baseline iteration"
    assert body["config"]["min_iterations"] == 3

    for index, iteration in enumerate(body["iterations"]):
        assert iteration["iteration"] == index
        assert iteration["run_id"]
        assert iteration["score"] is not None
        assert iteration["summary"] is not None
        assert iteration["summary"]["total"] == 2
        assert iteration["summary"]["pending"] == 0
        # The mock runner's rewards are a pure function of task_id, so no task ever
        # moves between iterations.
        assert iteration["fixed_tasks"] == []
        assert iteration["regressed_tasks"] == []
    assert body["iterations"][0]["proposal"] is None
    assert body["iterations"][0]["improved"] is True

    best = client.get(f"/v1/jobs/{job_id}/best")
    assert best.status_code == 200
    best_body = best.json()
    assert best_body["spec"]["agent_model"] == "e2e-model"
    assert best_body["score"] == pytest.approx(body["best"]["score"])
    assert best_body["version"] == body["best"]["version"]

    version = client.get(f"/v1/agent-versions/{best_body['agent_version_id']}")
    assert version.status_code == 200
    assert version.json()["spec"] == best_body["spec"]
