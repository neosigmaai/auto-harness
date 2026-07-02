import builtins
import importlib
import sys
import threading
from datetime import datetime, timezone

from autoharness_service.models import RunRecord
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.service import RunService


class FakeStore:
    def __init__(self):
        self.runs = {}
        self.results = {}
        self.iterations = {}
        self.calls = []
        self._next_run_id = 1

    def init_schema(self):
        self.calls.append(("init_schema",))

    def create_run(self, request, org_id, created_by):
        run_id = f"run-{self._next_run_id}"
        self._next_run_id += 1
        run = RunRecord(
            run_id=run_id,
            status="queued",
            task_ids=request.task_ids,
            mode=request.mode,
            model=request.model,
            sandbox_provider=request.sandbox_provider,
            requested_concurrency=request.requested_concurrency,
            max_iterations=request.max_iterations,
            org_id=org_id,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
        )
        self.runs[run.run_id] = run
        self.calls.append(("create_run", request.task_ids, org_id, created_by))
        return run

    def get_run(self, run_id, org_id):
        self.calls.append(("get_run", run_id, org_id))
        run = self.runs.get(run_id)
        if run is None or run.org_id != org_id:
            return None
        return run

    def mark_run_running(self, run_id, org_id):
        self.calls.append(("mark_run_running", run_id, org_id))
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(**{**run.__dict__, "status": "running"})

    def mark_run_succeeded(self, run_id, org_id, score):
        self.calls.append(("mark_run_succeeded", run_id, org_id, score))
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(
            **{**run.__dict__, "status": "succeeded", "score": score}
        )

    def mark_run_failed(self, run_id, org_id, status, error):
        self.calls.append(("mark_run_failed", run_id, org_id, status, error))
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(
            **{**run.__dict__, "status": status, "error": error}
        )

    def replace_task_results(self, run_id, org_id, task_results):
        self.calls.append(("replace_task_results", run_id, org_id, list(task_results)))
        self.results[run_id] = list(task_results)

    def list_task_results(self, run_id, org_id):
        self.calls.append(("list_task_results", run_id, org_id))
        return self.results.get(run_id, [])

    def create_iteration(
        self,
        run_id,
        org_id,
        iteration_index,
        status,
        agent_version,
        score=None,
        proposal=None,
        accepted=None,
    ):
        record = {
            "run_id": run_id,
            "org_id": org_id,
            "iteration_index": iteration_index,
            "status": status,
            "agent_version": agent_version,
            "score": score,
            "proposal": proposal,
            "accepted": accepted,
        }
        existing = self.iterations.setdefault(run_id, [])
        for index, item in enumerate(existing):
            if item["iteration_index"] == iteration_index:
                existing[index] = record
                break
        else:
            existing.append(record)
        self.calls.append(("create_iteration", record))
        return record

    def list_iterations(self, run_id, org_id):
        self.calls.append(("list_iterations", run_id, org_id))
        return self.iterations.get(run_id, [])


def test_run_service_executes_simulated_run():
    store = FakeStore()
    service = RunService(store=store, simulated_runner=SimulatedBenchmarkRunner())
    request = RunCreateRequest(
        task_ids=["task-pass", "task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 0.5
    assert results is not None
    assert results.tasks_passed == 1
    assert results.tasks_failed == 1
    assert results.tasks_infra_failed == 0
    assert all(result.trace_path is None for result in results.task_results)
    assert all(result.result_path is None for result in results.task_results)
    assert all(
        result.metadata["artifact_scope"] == "omitted_shared_latest"
        for result in results.task_results
    )
    assert all(
        result.metadata["trace_exists"] is False for result in results.task_results
    )
    assert all(
        result.metadata["result_exists"] is False for result in results.task_results
    )


def test_run_service_marks_real_run_failed_when_runner_returns_no_results():
    class EmptyRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            return {}

    store = FakeStore()
    service = RunService(
        store=store, terminal_runner=EmptyRunner(), max_local_concurrency=2
    )
    request = RunCreateRequest(
        task_ids=["task-pass", "task-missing"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=8,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "failed"
    assert status.error == "runner produced no task results"
    assert results is not None
    assert results.tasks_infra_failed == 2
    assert all(result.status == "infra_failed" for result in results.task_results)
    assert any(
        result.error_summary == "Task result missing from runner output"
        for result in results.task_results
    )


def test_run_service_fails_fast_when_a_real_run_is_already_active():
    class BlockingRunner:
        def __init__(self):
            self.started_runs = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.second_started = threading.Event()

        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            self.started_runs.append(run_id)
            if len(self.started_runs) == 1:
                self.first_started.set()
                self.release_first.wait(timeout=1)
            else:
                self.second_started.set()
            return {task_ids[0]: 1.0}

    store = FakeStore()
    runner = BlockingRunner()
    service = RunService(store=store, terminal_runner=runner)
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    run_2 = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )

    assert run.run_id != run_2.run_id

    first = threading.Thread(
        target=service.execute_run, kwargs={"run_id": run.run_id, "org_id": "org-1"}
    )
    second = threading.Thread(
        target=service.execute_run, kwargs={"run_id": run_2.run_id, "org_id": "org-1"}
    )

    first.start()
    assert runner.first_started.wait(timeout=1)
    second.start()

    second.join(timeout=0.2)
    assert not second.is_alive()
    assert not runner.second_started.is_set()

    runner.release_first.set()

    first.join(timeout=1)

    first_status = service.get_run_status(run.run_id, org_id="org-1")
    second_status = service.get_run_status(run_2.run_id, org_id="org-1")
    assert first_status is not None and first_status.status == "succeeded"
    assert second_status is not None and second_status.status == "failed"
    assert second_status.error == "another real Harbor/Daytona run is already active"


def test_service_module_import_does_not_import_benchmark(monkeypatch):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "benchmark":
            imported.append(name)
            raise AssertionError("benchmark was imported during service import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "autoharness_service.service", raising=False)
    monkeypatch.delitem(sys.modules, "autoharness_service.runner", raising=False)

    module = importlib.import_module("autoharness_service.service")

    assert imported == []
    assert hasattr(module, "RunService")
