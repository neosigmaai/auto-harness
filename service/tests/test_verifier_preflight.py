"""Audited dependency preparation must finish before candidate/model execution."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from service.sandbox import preflight


def test_every_supported_verifier_has_a_version_only_preflight():
    manifest = preflight.MANIFEST
    assert len(manifest["tasks"]) == 22
    assert set(manifest["tasks"]) == set(manifest["bootstrap"])
    for recipe in manifest["bootstrap"].values():
        command = preflight.preflight_command(recipe)
        assert recipe["warm_argv"][-1] == "--version"
        assert "/tests" not in command and "test_outputs" not in command
        assert "--ctrf" not in command
        if recipe["mode"] == "uvx":
            assert manifest["uv_archive"]["sha256"] in command
            assert "sha256sum -c -" in command
            assert "uv python find 3.13" in command


def test_test_execution_is_rejected_as_a_preflight_recipe():
    with pytest.raises(ValueError, match="tool version"):
        preflight.preflight_command({"mode": "pip", "warm_argv": ["pytest", "/tests"]})


def test_preflight_success_records_versions_and_pinned_dependencies(tmp_path):
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=0, stdout="uv 0.9.5\npytest 8.4.1\nPython 3.13.9\n", stderr="")))
    path = tmp_path / "setup.json"
    result = asyncio.run(preflight.run_preflight(environment, "adaptive-rejection-sampler", path,
                                                deadline=time.monotonic() + 400))
    assert result["status"] == "passed"
    assert result["direct_packages"] == ["pytest==8.4.1", "numpy==2.3.3", "scipy==1.16.2", "pytest-json-ctrf==0.3.5"]
    assert result["observed_versions"] == ["uv 0.9.5", "pytest 8.4.1", "Python 3.13.9"]
    assert result["hidden_assertions_copied"] is False
    assert json.loads(path.read_text())["duration_s"] >= 0
    assert environment.exec.await_args.kwargs["timeout_sec"] == 168
    assert environment.exec.await_args.args[0].startswith("timeout --kill-after=5s 160s")


def test_failure_is_bounded_and_prevents_candidate_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.asyncio, "sleep", AsyncMock())
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=124, stdout="", stderr="dependency unavailable")))
    path = tmp_path / "setup.json"
    with pytest.raises(RuntimeError, match="before agent execution"):
        asyncio.run(preflight.run_preflight(environment, "fix-git", path,
                                           deadline=time.monotonic() + 400))
    assert environment.exec.await_count == 2
    metadata = json.loads(path.read_text())
    assert metadata["status"] == "failed" and len(metadata["attempts"]) == 2


def test_expired_setup_deadline_does_not_start_an_install(tmp_path):
    environment = SimpleNamespace(exec=AsyncMock())
    path = tmp_path / "setup.json"
    with pytest.raises(RuntimeError, match="before agent execution"):
        asyncio.run(preflight.run_preflight(environment, "fix-git", path,
                                           deadline=time.monotonic() - 1))
    environment.exec.assert_not_awaited()
    assert json.loads(path.read_text())["attempts"] == []


def test_unknown_task_cannot_supply_an_unaudited_recipe(tmp_path):
    environment = SimpleNamespace(exec=AsyncMock())
    with pytest.raises(RuntimeError, match="No audited"):
        asyncio.run(preflight.run_preflight(environment, "unknown-task", tmp_path / "setup.json",
                    deadline=time.monotonic() + 400,
                    fixture_override={"mode": "pip", "warm_argv": ["pytest", "--version"]}))
    environment.exec.assert_not_awaited()
