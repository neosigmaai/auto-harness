import builtins
import importlib
import sys
import types
from pathlib import Path

from autoharness_service.runner import (
    SimulatedBenchmarkRunner,
    TerminalBenchRunnerAdapter,
)


def test_simulated_runner_returns_deterministic_rewards():
    runner = SimulatedBenchmarkRunner()

    results = runner.run(["task-pass", "task-fail", "task-infra"])

    assert results["task-pass"] == 1.0
    assert results["task-fail"] == 0.0
    assert results["task-infra"] is None


def test_terminal_runner_adapter_uses_per_run_jobs_dir_and_caps_concurrency(
    monkeypatch, tmp_path
):
    created = {}

    class FakeTerminalBenchRunner:
        def __init__(
            self,
            *,
            agent_model,
            split,
            env_provider,
            n_concurrent,
            jobs_dir,
            dataset="terminal-bench@2.0",
            agent_import_path="agent.agent:HarnessAgent",
            per_task_timeout=1200,
            reasoning_effort=None,
        ):
            created["agent_model"] = agent_model
            created["split"] = split
            created["env_provider"] = env_provider
            created["n_concurrent"] = n_concurrent
            created["jobs_dir"] = jobs_dir

        def run(self, task_ids):
            created["task_ids"] = list(task_ids)
            return {task_id: 1.0 for task_id in task_ids}

    monkeypatch.chdir(tmp_path)
    template_path = tmp_path / "agent" / "templates" / "terminal_bench.py"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("class HarnessAgent:\n    pass\n")

    benchmark_module = types.ModuleType("benchmark")
    benchmark_module.TerminalBenchRunner = FakeTerminalBenchRunner
    monkeypatch.setitem(sys.modules, "benchmark", benchmark_module)

    runner = TerminalBenchRunnerAdapter(split="train")
    results = runner.run(
        ["task-a", "task-b"],
        model="gpt-5.4",
        sandbox_provider="daytona",
        requested_concurrency=8,
        run_id="run-123",
        attempt="baseline",
    )

    assert results == {"task-a": 1.0, "task-b": 1.0}
    assert created["agent_model"] == "gpt-5.4"
    assert created["split"] == "train"
    assert created["env_provider"] == "daytona"
    assert created["n_concurrent"] == 2
    assert Path(created["jobs_dir"]) == Path(
        "workspace/service_runs/run-123/tbench_jobs/baseline"
    )
    assert created["task_ids"] == ["task-a", "task-b"]


def test_terminal_runner_adapter_uses_distinct_jobs_dir_per_attempt(
    monkeypatch, tmp_path
):
    created_dirs = []

    class FakeTerminalBenchRunner:
        def __init__(self, **kwargs):
            created_dirs.append(kwargs["jobs_dir"])

        def run(self, task_ids):
            return {task_id: 0.0 for task_id in task_ids}

    monkeypatch.chdir(tmp_path)
    template_path = tmp_path / "agent" / "templates" / "terminal_bench.py"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("class HarnessAgent:\n    pass\n")
    benchmark_module = types.ModuleType("benchmark")
    benchmark_module.TerminalBenchRunner = FakeTerminalBenchRunner
    monkeypatch.setitem(sys.modules, "benchmark", benchmark_module)

    runner = TerminalBenchRunnerAdapter(split="train")
    for attempt in ("baseline", "proposal-1"):
        runner.run(
            ["task-a"],
            model="gpt-5.4",
            sandbox_provider="daytona",
            requested_concurrency=1,
            run_id="run-123",
            attempt=attempt,
        )

    assert [Path(path) for path in created_dirs] == [
        Path("workspace/service_runs/run-123/tbench_jobs/baseline"),
        Path("workspace/service_runs/run-123/tbench_jobs/proposal-1"),
    ]


def test_terminal_runner_adapter_installs_terminal_agent_template_for_real_run(
    monkeypatch, tmp_path
):
    created = {}
    agent_path = tmp_path / "agent" / "agent.py"
    template_path = tmp_path / "agent" / "templates" / "terminal_bench.py"
    template_path.parent.mkdir(parents=True)
    agent_path.write_text("# Placeholder\n")
    template_path.write_text("class HarnessAgent:\n    pass\n")

    class FakeTerminalBenchRunner:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def run(self, task_ids):
            created["task_ids"] = list(task_ids)
            return {task_id: 1.0 for task_id in task_ids}

    monkeypatch.chdir(tmp_path)
    benchmark_module = types.ModuleType("benchmark")
    benchmark_module.TerminalBenchRunner = FakeTerminalBenchRunner
    monkeypatch.setitem(sys.modules, "benchmark", benchmark_module)

    runner = TerminalBenchRunnerAdapter(split="train")
    results = runner.run(
        ["task-a"],
        model="gpt-5.4",
        sandbox_provider="daytona",
        requested_concurrency=1,
        run_id="run-123",
    )

    assert results == {"task-a": 1.0}
    assert agent_path.read_text() == template_path.read_text()
    assert created["agent_import_path"] == "agent.agent:HarnessAgent"


def test_terminal_runner_adapter_exposes_last_artifacts(monkeypatch, tmp_path):
    artifacts = {
        "task-a": {
            "trace": "workspace/service_runs/run-123/tbench_jobs/job/task-a/trace.json",
            "trial_result": "workspace/service_runs/run-123/tbench_jobs/job/task-a/result.json",
        }
    }

    class FakeTerminalBenchRunner:
        def __init__(self, **kwargs):
            self.last_artifacts = artifacts

        def run(self, task_ids):
            return {task_id: 1.0 for task_id in task_ids}

    monkeypatch.chdir(tmp_path)
    template_path = tmp_path / "agent" / "templates" / "terminal_bench.py"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("class HarnessAgent:\n    pass\n")
    benchmark_module = types.ModuleType("benchmark")
    benchmark_module.TerminalBenchRunner = FakeTerminalBenchRunner
    monkeypatch.setitem(sys.modules, "benchmark", benchmark_module)

    runner = TerminalBenchRunnerAdapter(split="train")
    runner.run(
        ["task-a"],
        model="gpt-5.4",
        sandbox_provider="daytona",
        requested_concurrency=1,
        run_id="run-123",
    )

    assert runner.last_artifacts == artifacts


def test_runner_module_import_does_not_import_benchmark(monkeypatch):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "benchmark":
            imported.append(name)
            raise AssertionError("benchmark was imported during module import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "autoharness_service.runner", raising=False)

    module = importlib.import_module("autoharness_service.runner")

    assert imported == []
    assert hasattr(module, "SimulatedBenchmarkRunner")
