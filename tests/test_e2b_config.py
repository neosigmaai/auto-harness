"""Tests for ENV_PROVIDER override and Harbor E2B availability checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import BenchmarkConfig, clear_config_cache, load_config
from api.env import load_repo_dotenv
from api.services.runner import ExecutionUnavailableError, HarborBenchmarkRunner
from api.store import PostgresRunStore

MINIMAL_YAML = """\
default_agent_model: gpt-4.1-mini
default_task_ids:
  - fix-git
"""


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    clear_config_cache()
    yield
    clear_config_cache()


def _write_yaml(tmp_path: Path, extra: str) -> str:
    path = tmp_path / "benchmark.yaml"
    path.write_text(MINIMAL_YAML + extra, encoding="utf-8")
    return str(path)


def test_repo_config_defaults_to_e2b() -> None:
    cfg = load_config()
    assert cfg.env_provider == "e2b"
    assert cfg.max_concurrency == 8


def test_env_provider_env_var_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_PROVIDER", "docker")
    cfg = load_config(_write_yaml(tmp_path, "env_provider: e2b\n"))
    assert cfg.env_provider == "docker"


def test_harbor_env_provider_alias_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ENV_PROVIDER", raising=False)
    monkeypatch.setenv("HARBOR_ENV_PROVIDER", "modal")
    cfg = load_config(_write_yaml(tmp_path, "env_provider: e2b\n"))
    assert cfg.env_provider == "modal"


def test_e2b_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setattr(
        "api.services.runner.shutil.which",
        lambda name: "/usr/bin/harbor" if name == "harbor" else None,
    )
    hb = HarborBenchmarkRunner(
        store=PostgresRunStore(),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="test",
            execution_backend="harbor",
            env_provider="e2b",
        ),
    )
    # Skip the agent-file check so we only exercise the E2B credential gate.
    monkeypatch.setattr(hb, "_check_agent_import", lambda: None)
    with pytest.raises(ExecutionUnavailableError, match="E2B_API_KEY"):
        hb.check_available()


def test_e2b_provider_passes_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2B_API_KEY", "e2b_test_key_not_real")
    monkeypatch.setattr(
        "api.services.runner.shutil.which",
        lambda name: "/usr/bin/harbor" if name == "harbor" else None,
    )
    hb = HarborBenchmarkRunner(
        store=PostgresRunStore(),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="test",
            execution_backend="harbor",
            env_provider="e2b",
        ),
    )
    monkeypatch.setattr(hb, "_check_agent_import", lambda: None)
    hb.check_available()  # must not raise


def test_load_repo_dotenv_does_not_override_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("E2B_API_KEY=from_file\nOTHER_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("E2B_API_KEY", "from_process")
    monkeypatch.delenv("OTHER_KEY", raising=False)

    assert load_repo_dotenv(path=env_file) is True
    assert __import__("os").environ["E2B_API_KEY"] == "from_process"
    assert __import__("os").environ["OTHER_KEY"] == "from_file"


def test_load_repo_dotenv_missing_file(tmp_path: Path) -> None:
    assert load_repo_dotenv(path=tmp_path / "nope.env") is False
