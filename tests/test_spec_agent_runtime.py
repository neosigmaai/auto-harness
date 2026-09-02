# tests/test_spec_agent_runtime.py
"""Unit tests for the spec-driven agent runtime and its harbor plumbing.

Nothing here imports agent.spec_agent: it imports harbor and litellm at module
scope and neither is installed in the test environment. The spec-loading logic
lives in agent.spec_loader (stdlib only) precisely so it can be tested here.
"""

from __future__ import annotations

import json
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.spec_loader import (
    DEFAULT_EXEC_TIMEOUT_SEC,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYSTEM_PROMPT,
    default_spec,
    load_spec,
    load_spec_from_env,
)

from api.agent_spec import AgentSpec

SPEC_KEYS = set(AgentSpec.model_fields)


# --------------------------------------------------------------------------
# agent/spec_loader.py
# --------------------------------------------------------------------------


def test_default_prompt_matches_api_agent_spec_baseline() -> None:
    """Drift guard: the agent-side copy and the API-side copy must stay identical."""
    from api.agent_spec import BASELINE_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT == BASELINE_SYSTEM_PROMPT


def test_default_limits_match_api_agent_spec_defaults() -> None:
    from api.agent_spec import baseline_spec

    spec = baseline_spec("gpt-4.1-mini")
    assert DEFAULT_MAX_STEPS == spec.max_steps
    assert DEFAULT_MAX_OUTPUT_CHARS == spec.max_output_chars
    assert DEFAULT_EXEC_TIMEOUT_SEC == spec.exec_timeout_sec


def test_default_spec_has_exactly_the_spec_keys() -> None:
    assert set(default_spec()) == SPEC_KEYS


def test_load_spec_reads_json_from_path(tmp_path: Path) -> None:
    path = tmp_path / "agent_spec.json"
    path.write_text(
        json.dumps(
            {
                "system_prompt": "Be terse. Verify everything.",
                "agent_model": "claude-sonnet-4",
                "max_steps": 30,
                "max_output_chars": 1500,
                "exec_timeout_sec": 45,
            }
        ),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert spec == {
        "system_prompt": "Be terse. Verify everything.",
        "agent_model": "claude-sonnet-4",
        "max_steps": 30,
        "max_output_chars": 1500,
        "exec_timeout_sec": 45,
    }


def test_load_spec_overlays_only_the_provided_fields(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"max_steps": 7}), encoding="utf-8")

    spec = load_spec(str(path))

    assert spec["max_steps"] == 7
    assert spec["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert spec["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert spec["exec_timeout_sec"] == DEFAULT_EXEC_TIMEOUT_SEC


def test_load_spec_ignores_unknown_fields(tmp_path: Path) -> None:
    """A spec written by a newer AgentSpec must not break an older runtime."""
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"max_steps": 11, "temperature": 0.4, "tools": ["python"]}),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert set(spec) == SPEC_KEYS
    assert spec["max_steps"] == 11


def test_load_spec_coerces_numeric_strings(tmp_path: Path) -> None:
    path = tmp_path / "stringy.json"
    path.write_text(
        json.dumps({"max_steps": "25", "max_output_chars": "900", "exec_timeout_sec": "60"}),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert spec["max_steps"] == 25
    assert spec["max_output_chars"] == 900
    assert spec["exec_timeout_sec"] == 60


@pytest.mark.parametrize("path", [None, ""])
def test_load_spec_without_a_path_returns_defaults(path) -> None:
    assert load_spec(path) == default_spec()


def test_load_spec_falls_back_when_file_is_missing(tmp_path: Path) -> None:
    assert load_spec(str(tmp_path / "nope.json")) == default_spec()


def test_load_spec_falls_back_when_file_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_spec(str(path)) == default_spec()


def test_load_spec_falls_back_when_json_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_spec(str(path)) == default_spec()


@pytest.mark.parametrize("bad_prompt", ["", "   ", None, 42])
def test_load_spec_rejects_an_unusable_prompt(tmp_path: Path, bad_prompt) -> None:
    path = tmp_path / "bad_prompt.json"
    path.write_text(json.dumps({"system_prompt": bad_prompt}), encoding="utf-8")
    assert load_spec(str(path))["system_prompt"] == DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("bad_number", ["abc", None, [1]])
def test_load_spec_rejects_an_unusable_number(tmp_path: Path, bad_number) -> None:
    path = tmp_path / "bad_number.json"
    path.write_text(json.dumps({"max_steps": bad_number}), encoding="utf-8")
    assert load_spec(str(path))["max_steps"] == DEFAULT_MAX_STEPS


def test_load_spec_from_env_reads_harness_agent_spec(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "agent_spec.json"
    path.write_text(json.dumps({"max_steps": 3}), encoding="utf-8")
    monkeypatch.setenv("HARNESS_AGENT_SPEC", str(path))

    assert load_spec_from_env()["max_steps"] == 3


def test_load_spec_from_env_without_the_var_returns_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_AGENT_SPEC", raising=False)
    assert load_spec_from_env() == default_spec()


def test_spec_agent_module_exists_and_is_dependency_clean() -> None:
    """Static check: spec_agent.py must not import api.* (it runs under harbor)."""
    from api.config import REPO_ROOT

    source = (REPO_ROOT / "agent" / "spec_agent.py").read_text(encoding="utf-8")
    assert "class HarnessAgent" in source
    assert "from agent.spec_loader import" in source
    assert "HARNESS_SAVE_TRACE" in source
    assert "import api" not in source
    assert "from api" not in source


# --------------------------------------------------------------------------
# benchmark.py: extra_env
# --------------------------------------------------------------------------


def _fake_subprocess_run(captured: dict):
    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    return fake_run


def test_extra_env_overrides_harness_save_trace(tmp_path: Path, monkeypatch) -> None:
    """split=None forces HARNESS_SAVE_TRACE=0; extra_env is applied last and wins."""
    from benchmark import TerminalBenchRunner

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(captured))

    spec_path = str(tmp_path / "agent_spec.json")
    runner = TerminalBenchRunner(
        agent_model="gpt-4.1-mini",
        split=None,
        env_provider="docker",
        n_concurrent=2,
        jobs_dir=str(tmp_path / "jobs"),
        agent_import_path="agent.spec_agent:HarnessAgent",
        extra_env={"HARNESS_AGENT_SPEC": spec_path, "HARNESS_SAVE_TRACE": "1"},
    )
    runner.run(task_ids=["fix-git"])

    assert captured["env"]["HARNESS_SAVE_TRACE"] == "1"
    assert captured["env"]["HARNESS_AGENT_SPEC"] == spec_path
    assert captured["env"]["AGENT_MODEL"] == "gpt-4.1-mini"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent.spec_agent:HarnessAgent"


def test_without_extra_env_behaviour_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    from benchmark import TerminalBenchRunner

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(captured))

    runner = TerminalBenchRunner(
        agent_model="gpt-4.1-mini",
        split=None,
        env_provider="docker",
        n_concurrent=2,
        jobs_dir=str(tmp_path / "jobs"),
    )
    runner.run(task_ids=["fix-git"])

    assert captured["env"]["HARNESS_SAVE_TRACE"] == "0"
    assert "HARNESS_AGENT_SPEC" not in captured["env"]
    cmd = captured["cmd"]
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent.agent:HarnessAgent"


# --------------------------------------------------------------------------
# api/services/runner.py: HarborBenchmarkRunner passthrough
# --------------------------------------------------------------------------


class _FakeRunStore:
    """Minimal duck-typed stand-in for PostgresRunStore (keeps these tests DB-free)."""

    def __init__(self, record) -> None:
        self.record = record
        self.updates: list[dict] = []
        self.task_updates: list[tuple[str, dict]] = []

    def get(self, run_id: str):
        return self.record

    def update(self, run_id: str, **kwargs) -> None:
        self.updates.append(kwargs)

    def set_task(self, run_id: str, task_id: str, **kwargs) -> None:
        self.task_updates.append((task_id, kwargs))


def _run_record(task_ids: list[str]):
    from api.schemas import RunStatus, TaskStatus
    from api.store import RunRecord
    from api.schemas import TaskResult

    return RunRecord(
        run_id="00000000-0000-0000-0000-000000000001",
        status=RunStatus.running,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=None,
        task_ids=task_ids,
        agent_model="gpt-4.1-mini",
        tasks=[TaskResult(task_id=t, status=TaskStatus.pending) for t in task_ids],
    )


def _harbor_runner(store, tmp_path: Path, monkeypatch, **kwargs):
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    hb = runner_mod.HarborBenchmarkRunner(
        store=store,
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        **kwargs,
    )
    monkeypatch.setattr(hb, "check_available", lambda: None)
    return hb


def _capture_terminal_bench_runner(monkeypatch, captured: dict, results: dict):
    import benchmark as benchmark_mod

    class _Recorder:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def run(self, task_ids=None):
            captured["task_ids"] = task_ids
            return results

    monkeypatch.setattr(benchmark_mod, "TerminalBenchRunner", _Recorder)


def test_harbor_runner_passes_agent_import_path_and_extra_env(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}
    _capture_terminal_bench_runner(monkeypatch, captured, {"fix-git": 1.0})
    store = _FakeRunStore(_run_record(["fix-git"]))
    spec_path = str(tmp_path / "workspace" / "runs" / "r1" / "agent_spec.json")

    hb = _harbor_runner(
        store,
        tmp_path,
        monkeypatch,
        agent_import_path="agent.spec_agent:HarnessAgent",
        extra_env={"HARNESS_AGENT_SPEC": spec_path, "HARNESS_SAVE_TRACE": "1"},
    )
    hb.execute_sync("00000000-0000-0000-0000-000000000001")

    assert captured["init"]["agent_import_path"] == "agent.spec_agent:HarnessAgent"
    assert captured["init"]["extra_env"] == {
        "HARNESS_AGENT_SPEC": spec_path,
        "HARNESS_SAVE_TRACE": "1",
    }
    assert captured["task_ids"] == ["fix-git"]


def test_harbor_runner_defaults_keep_the_legacy_agent(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    _capture_terminal_bench_runner(monkeypatch, captured, {"fix-git": 1.0})
    store = _FakeRunStore(_run_record(["fix-git"]))

    hb = _harbor_runner(store, tmp_path, monkeypatch)
    hb.execute_sync("00000000-0000-0000-0000-000000000001")

    assert captured["init"]["agent_import_path"] == "agent.agent:HarnessAgent"
    assert captured["init"]["extra_env"] == {}


def test_check_agent_import_validates_the_spec_agent_file(tmp_path: Path, monkeypatch) -> None:
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    # agent/agent.py is the placeholder — it must NOT be consulted for a job-driven run.
    (agent_dir / "agent.py").write_text(
        "# Placeholder — do not edit this file directly.\n", encoding="utf-8"
    )
    (agent_dir / "spec_agent.py").write_text(
        "class HarnessAgent:\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)

    hb = runner_mod.HarborBenchmarkRunner(
        store=_FakeRunStore(None),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        agent_import_path="agent.spec_agent:HarnessAgent",
    )
    hb._check_agent_import()  # must not raise


def test_check_agent_import_reports_a_missing_spec_agent(tmp_path: Path, monkeypatch) -> None:
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    (tmp_path / "agent").mkdir()
    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)

    hb = runner_mod.HarborBenchmarkRunner(
        store=_FakeRunStore(None),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        agent_import_path="agent.spec_agent:HarnessAgent",
    )
    with pytest.raises(runner_mod.ExecutionUnavailableError, match="agent/spec_agent.py"):
        hb._check_agent_import()
