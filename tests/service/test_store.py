import json
import os
import uuid

import pytest
from autoharness_service.models import TaskResultRecord
from autoharness_service.normalizer import normalize_reward_result
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.store import (
    PROPOSAL_CHAR_LIMIT,
    TASK_RESULTS_TABLE,
    PostgresStore,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set for live Postgres store tests",
)


def test_store_init_schema_is_idempotent_and_persists_records():
    store = PostgresStore(os.environ["DATABASE_URL"])
    # The store uses aos_* tables so a legacy public runs table does not collide.
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

    before = store.get_run(run.run_id, org_id="org-a")
    assert before is not None

    store.mark_run_running(run.run_id, org_id="org-b")
    store.mark_run_succeeded(run.run_id, org_id="org-b", score=2.0)
    store.mark_run_failed(
        run.run_id,
        org_id="org-b",
        status="failed",
        error="wrong org",
    )

    after = store.get_run(run.run_id, org_id="org-a")
    assert after is not None
    assert after.status == before.status
    assert after.score == before.score
    assert after.error == before.error

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


def test_store_marks_run_failed_for_matching_org():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4()}"],
        mode="simulated",
        sandbox_provider="simulated",
    )

    run = store.create_run(request, org_id="org-a", created_by="user-a")
    store.mark_run_running(run.run_id, org_id="org-a")
    store.mark_run_failed(
        run.run_id,
        org_id="org-a",
        status="failed",
        error="boom",
    )

    loaded = store.get_run(run.run_id, org_id="org-a")
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error == "boom"
    assert loaded.completed_at is not None


def test_store_claims_queued_task_without_ambiguous_timestamp_column():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-pass-{uuid.uuid4().hex[:8]}"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
    )

    run = store.create_run(request, org_id="org-claim", created_by="user-claim")
    store.create_task_queue(run.run_id, org_id="org-claim", task_ids=request.task_ids)

    claimed = store.mark_task_running(
        run.run_id,
        org_id="org-claim",
        task_id=request.task_ids[0],
    )

    results = store.list_task_results(run.run_id, org_id="org-claim")
    assert claimed is True
    assert len(results) == 1
    assert results[0].status == "running"


def test_reset_task_queue_requeues_terminal_rows_for_same_org_only():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    task_ids = [
        f"task-pass-{uuid.uuid4().hex[:8]}",
        f"task-fail-{uuid.uuid4().hex[:8]}",
    ]
    request = RunCreateRequest(
        task_ids=task_ids,
        mode="simulated",
        sandbox_provider="simulated",
    )

    run = store.create_run(request, org_id="org-reset", created_by="user-reset")
    store.upsert_task_result(
        run.run_id,
        org_id="org-reset",
        result=normalize_reward_result(
            task_ids[0],
            1.0,
            trace_path="/tmp/pass.trace",
            result_path="/tmp/pass.result",
        ),
    )
    store.upsert_task_result(
        run.run_id,
        org_id="org-reset",
        result=normalize_reward_result(
            task_ids[1],
            0.0,
            trace_path="/tmp/fail.trace",
            result_path="/tmp/fail.result",
        ),
    )
    with store._connect() as conn:
        timestamp_rows = conn.execute(
            f"""
            SELECT started_at, completed_at
            FROM {TASK_RESULTS_TABLE}
            WHERE run_id = %s AND task_id = ANY(%s)
            """,
            (run.run_id, task_ids),
        ).fetchall()
    assert len(timestamp_rows) == 2
    assert all(row["started_at"] is not None for row in timestamp_rows)
    assert all(row["completed_at"] is not None for row in timestamp_rows)

    store.reset_task_queue(
        run.run_id,
        org_id="org-reset",
        task_ids=task_ids,
        metadata={"attempt": "proposal-1"},
    )

    results = store.list_task_results(run.run_id, org_id="org-reset")
    results_by_task_id = {result.task_id: result for result in results}
    assert set(results_by_task_id) == set(task_ids)
    for result in results:
        assert result.status == "queued"
        assert result.reward is None
        assert result.failure_type is None
        assert result.error_summary is None
        assert result.trace_path is None
        assert result.result_path is None
        assert result.metadata == {"attempt": "proposal-1"}
    with store._connect() as conn:
        reset_timestamp_rows = conn.execute(
            f"""
            SELECT started_at, completed_at
            FROM {TASK_RESULTS_TABLE}
            WHERE run_id = %s AND task_id = ANY(%s)
            """,
            (run.run_id, task_ids),
        ).fetchall()
    assert all(row["started_at"] is None for row in reset_timestamp_rows)
    assert all(row["completed_at"] is None for row in reset_timestamp_rows)

    store.replace_task_results(
        run.run_id,
        org_id="org-reset",
        task_results=[
            normalize_reward_result(task_ids[0], 1.0),
            normalize_reward_result(task_ids[1], 0.0),
        ],
    )

    store.reset_task_queue(
        run.run_id,
        org_id="org-other",
        task_ids=task_ids,
        metadata={"attempt": "wrong-org"},
    )

    unchanged = store.list_task_results(run.run_id, org_id="org-reset")
    unchanged_by_task_id = {result.task_id: result for result in unchanged}
    assert unchanged_by_task_id[task_ids[0]].status == "passed"
    assert unchanged_by_task_id[task_ids[1]].status == "failed"
    assert unchanged_by_task_id[task_ids[0]].metadata == {}
    assert unchanged_by_task_id[task_ids[1]].metadata == {}


def test_create_iteration_preserves_structured_proposal_json():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4().hex[:8]}"],
        mode="simulated",
        sandbox_provider="simulated",
    )
    run = store.create_run(request, org_id="org-iteration", created_by="user-iteration")

    proposal = json.dumps(
        {
            "summary": "x" * 5000,
            "steps": [f"step-{index}-{'y' * 300}" for index in range(25)],
            "metadata": {"kind": "structured"},
        }
    )
    assert 4000 < len(proposal) < 20000

    iteration = store.create_iteration(
        run.run_id,
        org_id="org-iteration",
        iteration_index=0,
        status="completed",
        agent_version="agent-v1",
        proposal=proposal,
    )

    iterations = store.list_iterations(run.run_id, org_id="org-iteration")
    assert iteration.proposal == proposal
    assert iterations[0].proposal == proposal


def test_create_iteration_rejects_proposal_over_explicit_cap():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()

    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4().hex[:8]}"],
        mode="simulated",
        sandbox_provider="simulated",
    )
    run = store.create_run(request, org_id="org-iteration-cap", created_by="user-cap")

    with pytest.raises(ValueError, match=f"exceeds {PROPOSAL_CHAR_LIMIT}"):
        store.create_iteration(
            run.run_id,
            org_id="org-iteration-cap",
            iteration_index=0,
            status="completed",
            agent_version="agent-v1",
            proposal="x" * (PROPOSAL_CHAR_LIMIT + 1),
        )
