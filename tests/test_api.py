"""API + worker tests for Milestone 2 (Postgres queue)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api.config import clear_config_cache, load_config
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.main import create_app
from api.services.runner import MockBenchmarkRunner
from api.store import PostgresRunStore
from worker.main import process_one

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
def client(db_store: PostgresRunStore) -> TestClient:
    clear_config_cache()
    app = create_app(store=db_store, database_url=DATABASE_URL, init_database=True)
    with TestClient(app) as test_client:
        yield test_client
    clear_config_cache()


def _run_worker_once(store: PostgresRunStore) -> bool:
    runner = MockBenchmarkRunner(store=store, step_delay_sec=0.0)
    return process_one(store, runner, worker_id="test-worker", stale_after_sec=1800)


def _wait_terminal(client: TestClient, run_id: str, store: PostgresRunStore, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        _run_worker_once(store)
        resp = client.get(f"/v1/runs/{run_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {last}")


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_tasks(client: TestClient) -> None:
    cfg = load_config()
    resp = client.get("/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_task_ids"] == cfg.default_task_ids
    assert body["default_agent_model"] == cfg.default_agent_model


def test_post_leaves_run_queued_without_worker(
    client: TestClient, db_store: PostgresRunStore
) -> None:
    resp = client.post("/v1/runs", json={"task_ids": ["fix-git", "regex-log"]})
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    # No worker invoked — must stay queued.
    time.sleep(0.1)
    got = client.get(f"/v1/runs/{run_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "queued"
    assert body["summary"]["pending"] == 2
    assert all(t["status"] == "pending" for t in body["tasks"])


def test_worker_completes_run(client: TestClient, db_store: PostgresRunStore) -> None:
    cfg = load_config()
    resp = client.post("/v1/runs", json={})
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    run = _wait_terminal(client, run_id, db_store)
    assert run["status"] == "completed"
    assert run["request"]["task_ids"] == cfg.default_task_ids
    assert run["summary"]["total"] == len(cfg.default_task_ids)
    for task in run["tasks"]:
        assert task["status"] in {"passed", "failed", "error"}


def test_create_run_with_explicit_task_ids(
    client: TestClient, db_store: PostgresRunStore
) -> None:
    resp = client.post(
        "/v1/runs",
        json={"task_ids": ["fix-git", "regex-log"], "agent_model": "test-model"},
    )
    assert resp.status_code == 202
    run = _wait_terminal(client, resp.json()["run_id"], db_store)
    assert run["request"]["task_ids"] == ["fix-git", "regex-log"]
    assert run["request"]["agent_model"] == "test-model"
    assert run["summary"]["total"] == 2


def test_unknown_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/runs", json={"task_ids": ["not-a-real-task"]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unknown_task_ids"
    assert "not-a-real-task" in body["error"]["details"]["unknown"]


def test_empty_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/runs", json={"task_ids": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "empty_task_ids"


def test_run_not_found(client: TestClient) -> None:
    resp = client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "run_not_found"


def test_two_workers_do_not_double_claim(db_store: PostgresRunStore) -> None:
    r1 = db_store.create(task_ids=["fix-git"], agent_model="m")
    r2 = db_store.create(task_ids=["regex-log"], agent_model="m")

    claimed: list[str | None] = []

    def claim(worker_id: str) -> str | None:
        return db_store.claim_next(worker_id, stale_after_sec=1800)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(claim, "w1")
        f2 = pool.submit(claim, "w2")
        claimed = [f1.result(), f2.result()]

    assert None not in claimed
    assert set(claimed) == {r1.run_id, r2.run_id}

    # Third claim should find nothing queued.
    assert db_store.claim_next("w3", stale_after_sec=1800) is None

    # Both claimed runs should be running with distinct workers.
    for run_id in claimed:
        rec = db_store.get(run_id)
        assert rec is not None
        assert rec.status.value == "running"


def test_default_create_is_still_queued_and_claimable(db_store: PostgresRunStore) -> None:
    """POST /v1/runs's path (no claimed_by) must be unaffected by the C1 fix."""
    run = db_store.create(task_ids=["fix-git"], agent_model="m")
    assert run.status.value == "queued"

    claimed = db_store.claim_next("w1", stale_after_sec=1800)
    assert claimed == run.run_id

    rec = db_store.get(run.run_id)
    assert rec is not None
    assert rec.status.value == "running"


def test_job_owned_run_is_not_stealable_by_legacy_queue(db_store: PostgresRunStore) -> None:
    """C1: a run created with claimed_by= must never be visible to claim_next.

    This is the exact theft the final review reproduced live against
    Postgres: worker W1 creates a job-owned evaluate run, and a second
    worker's fallback to the legacy /v1/runs queue (claim_next) must not be
    able to claim and re-execute it with the wrong agent.
    """
    owned = db_store.create(
        task_ids=["fix-git"], agent_model="m", claimed_by="w1-evaluate-step"
    )
    assert owned.status.value == "running"

    stolen = db_store.claim_next("w2", stale_after_sec=1800)
    assert stolen is None
    assert stolen != owned.run_id

    # A legacy-queue run created alongside it is unaffected and still claimable.
    legacy = db_store.create(task_ids=["regex-log"], agent_model="m")
    stolen_legacy = db_store.claim_next("w2", stale_after_sec=1800)
    assert stolen_legacy == legacy.run_id
    assert stolen_legacy != owned.run_id
