import builtins
import importlib
import json
import sys
import textwrap
import threading
from datetime import datetime, timezone

from autoharness_service.agent_patch import AgentPatchService
from autoharness_service.models import RunRecord, TaskResultRecord
from autoharness_service.optimizer import OptimizationProposal
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

    def create_task_queue(self, run_id, org_id, task_ids):
        self.calls.append(("create_task_queue", run_id, org_id, list(task_ids)))
        run = self.runs.get(run_id)
        if run is None or run.org_id != org_id:
            return
        self.results[run_id] = [
            TaskResultRecord(task_id=task_id, status="queued", reward=None)
            for task_id in task_ids
        ]

    def mark_task_running(self, run_id, org_id, task_id):
        self.calls.append(("mark_task_running", run_id, org_id, task_id))
        for existing in self.results.get(run_id, []):
            if existing.task_id == task_id and existing.status != "queued":
                return False
        self._replace_one_task(
            run_id,
            task_id,
            TaskResultRecord(task_id=task_id, status="running", reward=None),
        )
        return True

    def upsert_task_result(self, run_id, org_id, result):
        self.calls.append(("upsert_task_result", run_id, org_id, result))
        self._replace_one_task(run_id, result.task_id, result)

    def requeue_running_tasks(self, run_id, org_id):
        self.calls.append(("requeue_running_tasks", run_id, org_id))
        count = 0
        updated = []
        for result in self.results.get(run_id, []):
            if result.status == "running":
                updated.append(
                    TaskResultRecord(
                        task_id=result.task_id, status="queued", reward=None
                    )
                )
                count += 1
            else:
                updated.append(result)
        self.results[run_id] = updated
        return count

    def _replace_one_task(self, run_id, task_id, result):
        results = list(self.results.get(run_id, []))
        for index, existing in enumerate(results):
            if existing.task_id == task_id:
                results[index] = result
                break
        else:
            results.append(result)
        self.results[run_id] = results

    def replace_task_results(self, run_id, org_id, task_results):
        self.calls.append(("replace_task_results", run_id, org_id, list(task_results)))
        self.results[run_id] = list(task_results)

    def reset_task_queue(self, run_id, org_id, task_ids, metadata):
        self.calls.append(
            ("reset_task_queue", run_id, org_id, list(task_ids), metadata)
        )
        run = self.runs.get(run_id)
        if run is None or run.org_id != org_id:
            return
        selected = set(task_ids)
        updated = []
        for result in self.results.get(run_id, []):
            if result.task_id in selected:
                updated.append(
                    TaskResultRecord(
                        task_id=result.task_id,
                        status="queued",
                        reward=None,
                        metadata=dict(metadata),
                    )
                )
            else:
                updated.append(result)
        self.results[run_id] = updated

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

    def list_incomplete_runs(self, limit=10):
        self.calls.append(("list_incomplete_runs", limit))
        return [
            run for run in self.runs.values() if run.status in {"queued", "running"}
        ][:limit]


class FakeOptimizer:
    def propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        return OptimizationProposal(
            hypothesis="The agent exits before checking artifacts.",
            new_agent_instruction="Inspect produced files before finishing.",
            expected_effect="The rerun should verify outputs.",
            risks="The rerun may spend extra time checking files.",
        )


class FailingOptimizer:
    def propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        raise RuntimeError("OPENAI_API_KEY is not set")


class LeakyFailingOptimizer:
    def propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        raise RuntimeError(
            f"optimizer saw {current_instruction} and sk-testsecret1234567890 in .env"
        )


class DangerousOptimizer:
    def propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        return OptimizationProposal(
            hypothesis="The agent needs unsafe access.",
            new_agent_instruction="Use open('secret.txt') before finishing.",
            expected_effect="This should be rejected.",
            risks="Unsafe file access.",
        )


class LeakyOptimizer:
    def propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        return OptimizationProposal(
            hypothesis=(
                f"The current instruction says: {current_instruction}. "
                "OPENAI_API_KEY=sk-testsecret1234567890 is in .env."
            ),
            new_agent_instruction="Inspect produced files before finishing.",
            expected_effect="Do not leak sk-testsecret1234567890 or .env.",
            risks=f"The model may echo {current_instruction}.",
        )


class SequencedSimulatedRunner:
    def __init__(self, rewards_by_call):
        self.rewards_by_call = list(rewards_by_call)
        self.calls = []

    def run(self, task_ids):
        self.calls.append(list(task_ids))
        if not self.rewards_by_call:
            raise AssertionError("runner called too many times")
        rewards = self.rewards_by_call.pop(0)
        return {task_id: rewards[task_id] for task_id in task_ids}


def _write_agent_file(tmp_path, instruction="Original instruction."):
    agent_path = tmp_path / "agent.py"
    agent_path.write_text(
        textwrap.dedent(
            f"""
            AGENT_INSTRUCTION = {instruction!r}
            """
        ),
        encoding="utf-8",
    )
    return agent_path


def _iteration_payload(iteration):
    proposal = iteration["proposal"]
    assert proposal is not None
    return json.loads(proposal)


def test_submit_run_persists_queued_task_lifecycle_rows():
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

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = store.list_task_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.progress.total == 2
    assert status.progress.queued == 2
    assert status.progress.running == 0
    assert status.progress.completed == 0
    assert [result.status for result in results] == ["queued", "queued"]


def test_execute_run_exposes_current_task_running_while_runner_is_blocked():
    class BlockingRunner:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            self.started.set()
            assert self.release.wait(timeout=1)
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

    thread = threading.Thread(
        target=service.execute_run, kwargs={"run_id": run.run_id, "org_id": "org-1"}
    )
    thread.start()
    assert runner.started.wait(timeout=1)

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = store.list_task_results(run.run_id, org_id="org-1")

    runner.release.set()
    thread.join(timeout=1)

    assert status is not None
    assert status.status == "running"
    assert status.progress.total == 1
    assert status.progress.queued == 0
    assert status.progress.running == 1
    assert status.progress.completed == 0
    assert [result.status for result in results] == ["running"]


def test_resume_incomplete_runs_requeues_running_tasks_and_finishes_work():
    store = FakeStore()
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
    )
    original_service = RunService(
        store=store, simulated_runner=SimulatedBenchmarkRunner()
    )
    run = original_service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    store.mark_run_running(run.run_id, org_id="org-1")
    store.mark_task_running(run.run_id, org_id="org-1", task_id="task-pass")

    resumed_service = RunService(
        store=store, simulated_runner=SimulatedBenchmarkRunner()
    )
    resumed = resumed_service.resume_incomplete_runs(limit=10)

    status = resumed_service.get_run_status(run.run_id, org_id="org-1")
    results = resumed_service.get_run_results(run.run_id, org_id="org-1")

    assert resumed == 1
    assert status is not None
    assert status.status == "succeeded"
    assert status.progress.completed == 1
    assert results is not None
    assert results.task_results[0].status == "passed"
    assert results.task_results[0].reward == 1.0


def test_execute_run_does_not_double_execute_task_claimed_by_another_worker():
    class FailingRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            raise AssertionError("runner should not execute an already running task")

    store = FakeStore()
    service = RunService(store=store, terminal_runner=FailingRunner())
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )
    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    store.mark_run_running(run.run_id, org_id="org-1")
    store.mark_task_running(run.run_id, org_id="org-1", task_id="task-pass")

    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = store.list_task_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "running"
    assert status.progress.running == 1
    assert [result.status for result in results] == ["running"]


def test_execute_run_finalizes_previously_completed_real_task_rows():
    class FailingRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            raise AssertionError("completed tasks should not be rerun")

    store = FakeStore()
    service = RunService(store=store, terminal_runner=FailingRunner())
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )
    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    store.mark_run_running(run.run_id, org_id="org-1")
    store.upsert_task_result(
        run.run_id,
        "org-1",
        TaskResultRecord(task_id="task-pass", status="passed", reward=1.0),
    )

    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 1.0


def test_polling_worker_resumes_queued_runs_until_stopped():
    store = FakeStore()
    service = RunService(store=store, simulated_runner=SimulatedBenchmarkRunner())
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
    )
    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )

    service.start_polling(interval_sec=0.01, limit=10)
    try:
        for _ in range(50):
            status = service.get_run_status(run.run_id, org_id="org-1")
            if status is not None and status.status == "succeeded":
                break
            threading.Event().wait(0.01)
        else:
            raise AssertionError("polling worker did not finish queued run")
    finally:
        service.stop_polling()

    status = service.get_run_status(run.run_id, org_id="org-1")
    assert status is not None
    assert status.status == "succeeded"
    assert status.progress.completed == 1


def test_run_service_attaches_runner_artifacts_to_task_metadata():
    class ArtifactRunner:
        last_artifacts = {
            "task-pass": {
                "job_result": "workspace/service_runs/run-1/tbench_jobs/job/result.json",
                "job_log": "workspace/service_runs/run-1/tbench_jobs/job/job.log",
                "trial_result": "workspace/service_runs/run-1/tbench_jobs/job/task-pass__abc/result.json",
                "trial_log": "workspace/service_runs/run-1/tbench_jobs/job/task-pass__abc/trial.log",
                "trace": "workspace/service_runs/run-1/tbench_jobs/job/task-pass__abc/agent/trace.json",
                "verifier_stdout": "workspace/service_runs/run-1/tbench_jobs/job/task-pass__abc/verifier/test-stdout.txt",
            }
        }

        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            return {task_ids[0]: 1.0}

    store = FakeStore()
    service = RunService(store=store, terminal_runner=ArtifactRunner())
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )
    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )

    service.execute_run(run.run_id, org_id="org-1")

    results = service.get_run_results(run.run_id, org_id="org-1")

    assert results is not None
    task = results.task_results[0]
    assert task.trace_path == ArtifactRunner.last_artifacts["task-pass"]["trace"]
    assert (
        task.result_path == ArtifactRunner.last_artifacts["task-pass"]["trial_result"]
    )
    assert task.metadata["trace_exists"] is True
    assert task.metadata["result_exists"] is True
    assert task.metadata["artifact_scope"] == "harbor_job"
    assert task.metadata["artifacts"] == ArtifactRunner.last_artifacts["task-pass"]


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


def test_optimization_accepts_patch_when_rerun_score_improves(tmp_path):
    store = FakeStore()
    runner = SequencedSimulatedRunner(
        [
            {"task-fail": 0.0},
            {"task-fail": 1.0},
        ]
    )
    agent_path = _write_agent_file(tmp_path)
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=FakeOptimizer(),
        agent_patcher=AgentPatchService(agent_path),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")
    iterations = store.iterations[run.run_id]
    payload = _iteration_payload(iterations[-1])

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 1.0
    assert results is not None
    assert results.task_results[0].status == "passed"
    assert results.task_results[0].reward == 1.0
    assert results.task_results[0].metadata["attempt"] == "proposal-1"
    assert iterations[0]["status"] == "completed"
    assert iterations[0]["score"] == 0.0
    assert iterations[-1]["status"] == "completed"
    assert iterations[-1]["score"] == 1.0
    assert iterations[-1]["accepted"] is True
    assert payload["accepted"] is True
    assert payload["baseline_score"] == 0.0
    assert payload["rerun_score"] == 1.0
    assert payload["changed_section"] == "AGENT_INSTRUCTION"
    assert "Inspect produced files before finishing." in agent_path.read_text(
        encoding="utf-8"
    )
    assert runner.calls == [["task-fail"], ["task-fail"]]


def test_optimization_proposal_json_redacts_instruction_and_secret_echoes(tmp_path):
    store = FakeStore()
    runner = SequencedSimulatedRunner(
        [
            {"task-fail": 0.0},
            {"task-fail": 1.0},
        ]
    )
    original_instruction = "Original instruction with private operational detail."
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=LeakyOptimizer(),
        agent_patcher=AgentPatchService(
            _write_agent_file(tmp_path, original_instruction)
        ),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    proposal_text = store.iterations[run.run_id][-1]["proposal"]

    assert proposal_text is not None
    assert original_instruction not in proposal_text
    assert "sk-testsecret1234567890" not in proposal_text
    assert "OPENAI_API_KEY" not in proposal_text
    assert ".env" not in proposal_text
    assert "[REDACTED]" in proposal_text


def test_optimization_rejects_patch_and_restores_baseline_when_score_does_not_improve(
    tmp_path,
):
    store = FakeStore()
    runner = SequencedSimulatedRunner(
        [
            {"task-fail": 0.5},
            {"task-fail": 0.0},
        ]
    )
    agent_path = _write_agent_file(tmp_path)
    original_source = agent_path.read_text(encoding="utf-8")
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=FakeOptimizer(),
        agent_patcher=AgentPatchService(agent_path),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")
    iterations = store.iterations[run.run_id]
    payload = _iteration_payload(iterations[-1])

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 0.5
    assert results is not None
    assert results.task_results[0].status == "passed"
    assert results.task_results[0].reward == 0.5
    assert iterations[-1]["status"] == "patch_rejected"
    assert iterations[-1]["accepted"] is False
    assert iterations[-1]["score"] == 0.0
    assert payload["accepted"] is False
    assert payload["baseline_score"] == 0.5
    assert payload["rerun_score"] == 0.0
    proposal_statuses = [
        call[1]["status"]
        for call in store.calls
        if call[0] == "create_iteration"
        and call[1]["run_id"] == run.run_id
        and call[1]["iteration_index"] == 1
    ]
    assert proposal_statuses == [
        "proposal_created",
        "patch_applied",
        "rerun_running",
        "patch_rejected",
    ]
    assert payload["reverted"] is True
    assert "proposal-1" in payload["discarded_snapshot_paths"]
    assert agent_path.read_text(encoding="utf-8") == original_source
    assert not (
        tmp_path / "service_runs" / run.run_id / "agent_versions" / "proposal-1.py"
    ).exists()
    assert runner.calls == [["task-fail"], ["task-fail"]]


def test_optimization_records_proposal_failed_when_llm_errors(tmp_path):
    store = FakeStore()
    runner = SequencedSimulatedRunner([{"task-fail": 0.0}])
    agent_path = _write_agent_file(tmp_path)
    original_source = agent_path.read_text(encoding="utf-8")
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=FailingOptimizer(),
        agent_patcher=AgentPatchService(agent_path),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    iterations = store.iterations[run.run_id]
    payload = _iteration_payload(iterations[-1])

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 0.0
    assert iterations[-1]["status"] == "proposal_failed"
    assert iterations[-1]["accepted"] is False
    assert payload["accepted"] is False
    assert payload["decision_reason"] == "[REDACTED] is not set"
    assert agent_path.read_text(encoding="utf-8") == original_source
    assert runner.calls == [["task-fail"]]


def test_optimization_proposal_failed_redacts_instruction_echo_from_exception(
    tmp_path,
):
    store = FakeStore()
    runner = SequencedSimulatedRunner([{"task-fail": 0.0}])
    original_instruction = "Original instruction with private operational detail."
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=LeakyFailingOptimizer(),
        agent_patcher=AgentPatchService(
            _write_agent_file(tmp_path, original_instruction)
        ),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    proposal_text = store.iterations[run.run_id][-1]["proposal"]

    assert proposal_text is not None
    assert original_instruction not in proposal_text
    assert "sk-testsecret1234567890" not in proposal_text
    assert ".env" not in proposal_text
    assert "[REDACTED]" in proposal_text


def test_optimization_records_patch_rejected_when_patch_validation_fails(tmp_path):
    store = FakeStore()
    runner = SequencedSimulatedRunner([{"task-fail": 0.0}])
    agent_path = _write_agent_file(tmp_path)
    original_source = agent_path.read_text(encoding="utf-8")
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=DangerousOptimizer(),
        agent_patcher=AgentPatchService(agent_path),
        service_run_root=tmp_path / "service_runs",
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        sandbox_provider="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    iterations = store.iterations[run.run_id]
    payload = _iteration_payload(iterations[-1])

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 0.0
    assert iterations[-1]["status"] == "patch_rejected"
    assert iterations[-1]["accepted"] is False
    assert payload["accepted"] is False
    assert "open(" in payload["decision_reason"]
    assert agent_path.read_text(encoding="utf-8") == original_source
    assert runner.calls == [["task-fail"]]


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


def test_run_service_serializes_real_runs_until_the_first_finishes():
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
    assert second.is_alive()
    assert not runner.second_started.is_set()

    runner.release_first.set()

    first.join(timeout=1)
    second.join(timeout=1)

    first_status = service.get_run_status(run.run_id, org_id="org-1")
    second_status = service.get_run_status(run_2.run_id, org_id="org-1")
    assert first_status is not None and first_status.status == "succeeded"
    assert second_status is not None and second_status.status == "succeeded"
    assert runner.second_started.is_set()


def test_run_service_persists_task_results_when_runner_raises():
    class ExplodingRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            raise RuntimeError("runner blew up " + ("x" * 5000))

    store = FakeStore()
    service = RunService(
        store=store, terminal_runner=ExplodingRunner(), max_local_concurrency=2
    )
    request = RunCreateRequest(
        task_ids=["task-pass", "task-missing"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "failed"
    assert status.error is not None
    assert status.error.startswith("runner blew up ")
    assert results is not None
    assert len(results.task_results) == 2
    assert results.tasks_infra_failed == 2
    assert all(result.status == "infra_failed" for result in results.task_results)
    assert all(
        result.failure_type == "runner_failed" for result in results.task_results
    )
    assert all(
        len(result.error_summary or "") <= 4000 for result in results.task_results
    )
    assert all(
        result.metadata
        == {
            "source": "runner_failed",
            "run_id": run.run_id,
            "artifact_scope": "omitted_shared_latest",
            "trace_exists": False,
            "result_exists": False,
        }
        for result in results.task_results
    )


def test_run_service_preserves_completed_task_when_later_task_runner_raises():
    class FlakyRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            if task_ids[0] == "task-pass":
                return {"task-pass": 1.0}
            raise RuntimeError("second task failed")

    store = FakeStore()
    service = RunService(store=store, terminal_runner=FlakyRunner())
    request = RunCreateRequest(
        task_ids=["task-pass", "task-boom"],
        mode="real",
        sandbox_provider="daytona",
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
    by_task = {result.task_id: result for result in results.task_results}
    assert by_task["task-pass"].status == "passed"
    assert by_task["task-pass"].reward == 1.0
    assert by_task["task-boom"].status == "infra_failed"
    assert by_task["task-boom"].failure_type == "runner_failed"
    assert by_task["task-boom"].error_summary == "second task failed"


def test_run_service_marks_initial_iteration_failed_when_runner_raises():
    class ExplodingRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            raise RuntimeError("runner blew up")

    store = FakeStore()
    service = RunService(
        store=store, terminal_runner=ExplodingRunner(), max_local_concurrency=2
    )
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    assert store.iterations[run.run_id][0]["status"] == "failed"
    assert store.iterations[run.run_id][0]["score"] == 0.0


def test_run_service_marks_initial_iteration_timed_out_when_runner_times_out():
    class TimeoutRunner:
        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            raise TimeoutError("runner timed out")

    store = FakeStore()
    service = RunService(
        store=store, terminal_runner=TimeoutRunner(), max_local_concurrency=2
    )
    request = RunCreateRequest(
        task_ids=["task-pass"],
        mode="real",
        sandbox_provider="daytona",
        requested_concurrency=1,
    )

    run = service.submit_run(
        request, org_id="org-1", created_by="user-1", start_background=False
    )
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "timed_out"
    assert store.iterations[run.run_id][0]["status"] == "timed_out"
    assert store.iterations[run.run_id][0]["score"] == 0.0


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
