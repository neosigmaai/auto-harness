import os
import uuid

import pytest

from autoharness_service.models import IterationRecord, TaskResultRecord
from autoharness_service.normalizer import normalize_reward_result
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.store import PostgresStore


pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set for live Postgres store tests",
)


def test_store_init_schema_is_idempotent_and_persists_records():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4()}"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
    )

    run = store.create_run(request, org_id="org-test", created_by="user-test")
    store.mark_run_running(run.run_id, org_id="org-test")

    result = normalize_reward_result(request.task_ids[0], 1.0)
    store.replace_task_results(
        run.run_id,
        org_id="org-test",
        task_results=[result],
    )

    iteration = store.create_iteration(
        run.run_id,
        org_id="org-test",
        iteration_index=0,
        status="completed",
        agent_version="agent-v1",
        score=1.0,
        proposal="proposal text",
        accepted=True,
    )

    store.mark_run_succeeded(run.run_id, org_id="org-test", score=1.0)

    loaded = store.get_run(run.run_id, org_id="org-test")
    results = store.list_task_results(run.run_id, org_id="org-test")
    iterations = store.list_iterations(run.run_id, org_id="org-test")

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.score == 1.0
    assert results == [result]
    assert iterations == [iteration]


def test_store_enforces_org_boundaries_for_reads_and_writes():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4()}"],
        mode="simulated",
        sandbox_provider="simulated",
    )

    run = store.create_run(request, org_id="org-a", created_by="user-a")
    original_result = normalize_reward_result(request.task_ids[0], 1.0)
    store.replace_task_results(
        run.run_id,
        org_id="org-a",
        task_results=[original_result],
    )
    original_iteration = store.create_iteration(
        run.run_id,
        org_id="org-a",
        iteration_index=0,
        status="completed",
        agent_version="agent-v1",
    )

    assert store.get_run(run.run_id, org_id="org-a") is not None
    assert store.get_run(run.run_id, org_id="org-b") is None
    assert store.list_task_results(run.run_id, org_id="org-b") == []
    assert store.list_iterations(run.run_id, org_id="org-b") == []

    store.replace_task_results(
        run.run_id,
        org_id="org-b",
        task_results=[
            TaskResultRecord(
                task_id=original_result.task_id,
                status="failed",
                reward=0.0,
                failure_type="agent_failed",
                error_summary="wrong org",
            )
        ],
    )

    with pytest.raises(KeyError):
        store.create_iteration(
            run.run_id,
            org_id="org-b",
            iteration_index=1,
            status="completed",
            agent_version="agent-v2",
        )

    assert store.list_task_results(run.run_id, org_id="org-a") == [original_result]
    assert store.list_iterations(run.run_id, org_id="org-a") == [original_iteration]
