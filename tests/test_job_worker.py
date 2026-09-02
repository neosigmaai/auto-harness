"""End-to-end job loop tests: Postgres + mock backend + FakeImprover."""

from __future__ import annotations

import json
import os
import shutil

import pytest
from sqlalchemy.exc import OperationalError

from api.config import REPO_ROOT, clear_config_cache, load_config
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import PostgresJobStore
from api.schemas import RunStatus, TaskStatus
from api.services.artifacts import LocalArtifactStore
from api.services.improver import FakeImprover, ImproverError
from api.services.runner import MockBenchmarkRunner
from api.store import PostgresRunStore
from worker.main import process_one
from worker.steps import StepExecutor

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)

# MockBenchmarkRunner buckets task_id by sha256 % 5: "fix-git" -> passed (1.0),
# "regex-log" -> failed (0.0). The mean reward is therefore exactly 0.5 on every
# iteration, so a mock job can never improve and always plateaus.
TASK_IDS = ["fix-git", "regex-log"]
PLATEAU_SCORE = 0.5


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
def stores(tmp_path):
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    factory = get_session_factory()
    run_store = PostgresRunStore(session_factory=factory)
    job_store = PostgresJobStore(session_factory=factory)
    job_store.clear()
    run_store.clear()

    yield run_store, job_store, LocalArtifactStore(tmp_path)

    job_store.clear()
    run_store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


class _RaisingImprover:
    """Improver that always fails, to exercise the failed_improve path."""

    def propose(self, *, spec, evaluation, history):  # noqa: ANN001, ANN201
        raise ImproverError("no proposal today")


def _executor(run_store, job_store, artifacts, improver):  # noqa: ANN001, ANN201
    return StepExecutor(
        job_store,
        run_store,
        config=load_config(),
        improver=improver,
        artifacts=artifacts,
        step_delay_sec=0.0,
    )


def _make_job(job_store, *, max_iterations: int, patience: int, min_iterations: int = 1):  # noqa: ANN001, ANN201
    # min_iterations defaults to 1 (disabling the noise floor) so these tests can
    # drive no_improvement / max_iterations stops directly off patience/max_iterations
    # without also having to satisfy the min_iterations threshold covered elsewhere
    # (tests/test_scoring.py, tests/test_job_store.py).
    return job_store.create_job(
        task_ids=list(TASK_IDS),
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=max_iterations,
        patience=patience,
        min_iterations=min_iterations,
        min_delta=0.01,
        max_job_duration_sec=3600,
        evaluate_stale_after_sec=1800,
    )


def _drain(run_store, job_store, executor, runner, *, limit: int = 20) -> int:
    """Call process_one until it reports no work; returns the number of units done."""
    done = 0
    for _ in range(limit):
        did_work = process_one(
            run_store,
            runner,
            worker_id="worker-test",
            stale_after_sec=1800,
            job_store=job_store,
            step_executor=executor,
        )
        if not did_work:
            break
        done += 1
    return done


def _cleanup_run_dirs(job) -> None:  # noqa: ANN001
    for iteration in job.iterations:
        if iteration.run_id:
            shutil.rmtree(REPO_ROOT / "workspace" / "runs" / iteration.run_id, ignore_errors=True)


def test_mock_job_plateaus_and_stops_with_no_improvement(stores) -> None:
    run_store, job_store, artifacts = stores
    improver = FakeImprover()
    executor = _executor(run_store, job_store, artifacts, improver)
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    job = _make_job(job_store, max_iterations=3, patience=1)

    # evaluate(0) -> improve(0) -> evaluate(1) -> stop. Exactly three units.
    assert _drain(run_store, job_store, executor, runner) == 3

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "no_improvement"
    assert final.best_agent_version_id is not None
    assert final.best_version == 0
    assert final.best_score == pytest.approx(PLATEAU_SCORE)

    assert [it.iteration for it in final.iterations] == [0, 1]
    assert final.iterations[0].score == pytest.approx(PLATEAU_SCORE)
    assert final.iterations[0].improved is True
    assert final.iterations[0].changed_fields == []
    assert final.iterations[1].score == pytest.approx(PLATEAU_SCORE)
    assert final.iterations[1].improved is False
    assert final.iterations[1].changed_fields == ["system_prompt"]
    assert final.iterations[1].rationale == "fake improver deterministic revision 1"

    assert improver.calls == 1
    _cleanup_run_dirs(final)


def test_evaluate_step_materializes_agent_spec(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, FakeImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # max_iterations=1 stops right after the baseline evaluation.
    job = _make_job(job_store, max_iterations=1, patience=2)
    assert _drain(run_store, job_store, executor, runner) == 1

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "max_iterations"

    iteration = final.iterations[0]
    assert iteration.run_id is not None
    spec_path = REPO_ROOT / "workspace" / "runs" / iteration.run_id / "agent_spec.json"
    assert spec_path.is_file()

    written = json.loads(spec_path.read_text(encoding="utf-8"))
    version = job_store.get_agent_version(iteration.agent_version_id)
    assert written["system_prompt"] == version.spec.system_prompt
    assert written["agent_model"] == "gpt-4.1-mini"
    assert written["max_steps"] == version.spec.max_steps

    _cleanup_run_dirs(final)


def test_process_one_falls_back_to_standalone_run(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, FakeImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # No job exists, so no step is claimable: the plain /v1/runs path must run.
    record = run_store.create(task_ids=["fix-git"], agent_model="gpt-4.1-mini")

    assert process_one(
        run_store,
        runner,
        worker_id="worker-test",
        stale_after_sec=1800,
        job_store=job_store,
        step_executor=executor,
    ) is True

    final = run_store.get(record.run_id)
    assert final.status == RunStatus.completed
    assert final.tasks[0].task_id == "fix-git"
    assert final.tasks[0].status == TaskStatus.passed
    assert final.tasks[0].reward == pytest.approx(1.0)


def test_store_traces_copies_harbor_trial_layout(stores) -> None:
    """
    The mock backend never produces traces (see PLATEAU_SCORE comment above), so
    this drives StepExecutor._store_traces directly against a hand-built harbor
    trial layout to prove the walk-and-copy logic works without harbor installed.
    """
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, FakeImprover())
    _make_job(job_store, max_iterations=1, patience=1)

    run_id = "cccccccc-0000-0000-0000-000000000000"
    run_dir = REPO_ROOT / "workspace" / "runs" / run_id
    # harbor layout: <jobs_dir>/<harbor_job>/<task_id>__<trial>/agent/trace.json
    fix_git_trace = run_dir / "harbor-job" / "fix-git__trial0" / "agent" / "trace.json"
    regex_log_trace = run_dir / "harbor-job" / "regex-log__trial0" / "agent" / "trace.json"
    unknown_task_trace = run_dir / "harbor-job" / "unknown-task__trial0" / "agent" / "trace.json"
    for path, content in (
        (fix_git_trace, '[{"role": "user", "content": "fix it"}]'),
        (regex_log_trace, '[{"role": "user", "content": "regex it"}]'),
        (unknown_task_trace, '[{"role": "user", "content": "not requested"}]'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    step = job_store.claim_next_step("w")  # iteration-0 evaluate step for job_id/iteration

    try:
        copied = executor._store_traces(step, run_id, list(TASK_IDS))
        assert copied == 2  # only the two requested task_ids, not "unknown-task"

        assert artifacts.get(
            "jobs/%s/iterations/0/tasks/fix-git/trace.json" % step.job_id
        ).decode() == '[{"role": "user", "content": "fix it"}]'
        assert artifacts.get(
            "jobs/%s/iterations/0/tasks/regex-log/trace.json" % step.job_id
        ).decode() == '[{"role": "user", "content": "regex it"}]'
        assert not artifacts.exists(
            "jobs/%s/iterations/0/tasks/unknown-task/trace.json" % step.job_id
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_improver_error_completes_job_with_failed_improve(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, _RaisingImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # patience=2 keeps the loop alive past the baseline evaluation, so the
    # improve step actually gets to run and fail.
    job = _make_job(job_store, max_iterations=3, patience=2)

    # evaluate(0) -> improve(0) fails -> job closes. Exactly two units.
    assert _drain(run_store, job_store, executor, runner) == 2

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "failed_improve"
    # The best-so-far agent is still a valid answer.
    assert final.best_agent_version_id is not None
    assert final.best_score == pytest.approx(PLATEAU_SCORE)
    assert [it.iteration for it in final.iterations] == [0]

    # The improver failure is recorded as an artifact for auditability.
    response = json.loads(artifacts.get("jobs/%s/iterations/0/improver/response.json" % job.job_id))
    assert response["error"] == "no proposal today"

    _cleanup_run_dirs(final)
