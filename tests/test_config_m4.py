"""Unit tests for the Milestone 4 BenchmarkConfig additions (no DB, no harbor)."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import BenchmarkConfig, clear_config_cache, load_config

# A minimal valid config; each test appends the field under test.
MINIMAL_YAML = """\
default_agent_model: gpt-4.1-mini
default_task_ids:
  - fix-git
"""


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    """load_config is lru_cached with maxsize=1 — never let one test's config leak."""
    clear_config_cache()
    yield
    clear_config_cache()


def _write_yaml(tmp_path: Path, extra: str) -> str:
    path = tmp_path / "benchmark.yaml"
    path.write_text(MINIMAL_YAML + extra, encoding="utf-8")
    return str(path)


def test_repo_config_provides_m4_defaults() -> None:
    """The checked-in config/benchmark.yaml carries the Milestone 4 defaults."""
    cfg = load_config()
    assert cfg.improver_model == "gpt-5.4"
    assert cfg.max_iterations == 5
    assert cfg.patience == 2
    assert cfg.min_delta == pytest.approx(0.01)
    assert cfg.max_job_duration_sec == 21600
    assert cfg.improver_context_budget == 60000
    assert cfg.artifacts_dir == "workspace/artifacts"


def test_dataclass_defaults_match_contract() -> None:
    """Constructing BenchmarkConfig directly (as tests and workers do) needs no YAML."""
    cfg = BenchmarkConfig(default_task_ids=["fix-git"], default_agent_model="gpt-4.1-mini")
    assert cfg.improver_model == "gpt-5.4"
    assert cfg.max_iterations == 5
    assert cfg.patience == 2
    assert cfg.min_delta == pytest.approx(0.01)
    assert cfg.max_job_duration_sec == 21600
    assert cfg.improver_context_budget == 60000
    assert cfg.artifacts_dir == "workspace/artifacts"


def test_yaml_values_override_defaults(tmp_path: Path) -> None:
    cfg = load_config(
        _write_yaml(
            tmp_path,
            "improver_model: claude-opus-4\n"
            "max_iterations: 9\n"
            "patience: 3\n"
            "min_delta: 0.25\n"
            "max_job_duration_sec: 600\n"
            "improver_context_budget: 1234\n"
            "artifacts_dir: /var/tmp/artifacts\n",
        )
    )
    assert cfg.improver_model == "claude-opus-4"
    assert cfg.max_iterations == 9
    assert cfg.patience == 3
    assert cfg.min_delta == pytest.approx(0.25)
    assert cfg.max_job_duration_sec == 600
    assert cfg.improver_context_budget == 1234
    assert cfg.artifacts_dir == "/var/tmp/artifacts"


def test_min_delta_zero_is_accepted(tmp_path: Path) -> None:
    """0.0 is inside [0, 1) — it must survive parsing, not be defaulted to 0.01."""
    cfg = load_config(_write_yaml(tmp_path, "min_delta: 0.0\n"))
    assert cfg.min_delta == 0.0


@pytest.mark.parametrize(
    "extra,expected_message",
    [
        ("max_iterations: 0\n", "max_iterations"),
        ("max_iterations: -3\n", "max_iterations"),
        ("patience: 0\n", "patience"),
        ("patience: -1\n", "patience"),
        ("max_job_duration_sec: 0\n", "max_job_duration_sec"),
        ("improver_context_budget: 0\n", "improver_context_budget"),
        ("min_delta: 1.0\n", "min_delta"),
        ("min_delta: 1.5\n", "min_delta"),
        ("min_delta: -0.5\n", "min_delta"),
    ],
)
def test_invalid_m4_values_raise(tmp_path: Path, extra: str, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        load_config(_write_yaml(tmp_path, extra))
