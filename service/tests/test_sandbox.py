"""Sandbox contract tests; provider calls are replaced with local fixtures."""
import asyncio
import json
import os
from pathlib import Path
import socket
import signal
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from service.sandbox import relay
from service.sandbox.runtime import LocalEnvironment, read_bounded
from service.worker.executors.harbor_runner import (
    RELAY_IMAGE, harbor_config, parse_trial_result, raise_gateway_stop, runner_environment, sandbox_overlay,
)
from service import repo


def trusted_verifier_evidence(directory):
    (directory / "verifier-execution.json").write_text(json.dumps({
        "contract": "pinned-pytest-cases-v2", "valid": True,
        "counts": {"tests": 1, "passed": 1, "failed": 0, "skipped": 0, "pending": 0, "other": 0}}))


def test_runner_never_inherits_provider_or_database_credentials(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPEN_API_KEY", "NEOSIGMA_OPEN_API_KEY", "DATABASE_URL", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "secret-test-value")
    environment = runner_environment()
    assert not any("secret-test-value" in value for value in environment.values())


@pytest.mark.parametrize("reason,exception", [
    ("budget_exhausted", repo.BudgetExceeded), ("cancelled", repo.JobCancelled), ("lease_lost", repo.LeaseLost),
    ("inference_error", RuntimeError),
])
def test_trusted_gateway_stops_cannot_be_scored_as_agent_outcomes(reason, exception):
    with pytest.raises(exception):
        raise_gateway_stop(SimpleNamespace(stop_reason=reason))


@pytest.mark.parametrize("host", ["127.0.0.1", "10.3.4.5", "172.17.0.1", "192.168.65.254", "169.254.169.254", "::1", "::ffff:127.0.0.1", "0.0.0.0"])
def test_relay_rejects_control_plane_and_metadata(host):
    with pytest.raises(ValueError):
        relay.public_address(host, 443)


def test_relay_rejects_mixed_public_private_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
    ])
    with pytest.raises(ValueError):
        relay.public_address("mixed.invalid", 443)


def test_relay_pins_public_address_and_denies_other_ports(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
    ])
    assert relay.public_address("public.invalid", 443) == "1.1.1.1"
    with pytest.raises(ValueError):
        relay.public_address("public.invalid", 5432)


def test_candidate_has_only_internal_network_and_no_host_mount(tmp_path):
    overlay = sandbox_overlay(tmp_path / "compose.json", gateway_url="http://host.docker.internal:9999/v1",
        attempt_id="attempt", job_id="job", epoch=3, expires_at=time.time() + 60)
    main = overlay["services"]["main"]
    assert main["networks"] == ["aos-internal"]
    assert overlay["networks"]["aos-internal"]["internal"] is True
    assert "volumes" not in main and "ports" not in main
    assert main["pids_limit"] == 256
    assert "NET_ADMIN" in main["cap_drop"] and "NET_RAW" in main["cap_drop"]
    sidecar = overlay["services"]["aos-relay"]
    assert "ports" not in sidecar
    assert len(sidecar["volumes"]) == 1 and sidecar["volumes"][0]["read_only"]
    assert "TOKEN" not in json.dumps(sidecar)


@pytest.mark.parametrize("container_clock,expected_sleep", [(1090, "10"), (1101, None)])
def test_cold_start_cannot_extend_absolute_sandbox_lifetime(tmp_path, container_clock, expected_sleep):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    slept = tmp_path / "sleep-argument"
    (bin_dir / "date").write_text(f"#!/bin/sh\necho {container_clock}\n")
    (bin_dir / "sleep").write_text(f'#!/bin/sh\nprintf "%s" "$1" > "{slept}"\n')
    for executable in bin_dir.iterdir():
        executable.chmod(0o700)
    overlay = sandbox_overlay(tmp_path / "compose.json", gateway_url="http://gateway/v1",
        attempt_id="attempt", job_id="job", epoch=1, expires_at=1100)
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose CLI required; this check does not use the daemon")
    overlay["services"]["main"]["image"] = RELAY_IMAGE
    config_path = tmp_path / "compose.json"
    config_path.write_text(json.dumps(overlay))
    environment = {key: value for key, value in os.environ.items() if key != "aos_remaining"}
    translated = subprocess.run([docker, "compose", "-f", str(config_path), "config", "--format", "json"],
                               env=environment, capture_output=True, text=True, check=True, timeout=10)
    assert "aos_remaining" not in translated.stderr
    # Compose config retains $$ for serialization; container creation consumes
    # that layer. The real Docker smoke verifies the final execution boundary.
    command = json.loads(translated.stdout)["services"]["main"]["command"][2]
    assert '"$$aos_remaining"' in command
    subprocess.run(["/bin/sh", "-c", command.replace("$$", "$" )],
                   env={"PATH": str(bin_dir)}, check=True, timeout=2)
    assert (slept.read_text() if slept.exists() else None) == expected_sleep


def test_harbor_config_uses_only_fixed_adapter_and_pinned_tasks(tmp_path):
    config = harbor_config("fix-git", tmp_path, tmp_path / "adapter.json", tmp_path / "overlay.json", 90, 50)
    assert config["agents"][0]["import_path"] == "service.sandbox.installed_agent:ServiceInstalledAgent"
    assert config["environment"]["import_path"] == "service.sandbox.docker_environment:IsolatedDockerEnvironment"
    assert len(config["tasks"][0]["git_commit_id"]) == 40
    assert config["retry"]["max_retries"] == 0
    assert config["agents"][0]["override_timeout_sec"] == 100
    assert "timeout_multiplier" not in config


def test_frozen_job_revision_overrides_worker_default(tmp_path, monkeypatch):
    from service.config import settings
    frozen = "a" * 40
    monkeypatch.setattr(settings, "benchmark_revision", "b" * 40)
    config = harbor_config("fix-git", tmp_path, tmp_path / "adapter.json", tmp_path / "overlay.json", 90, 50,
                           benchmark_revision=frozen)
    assert config["tasks"][0]["git_commit_id"] == frozen
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"task_name": "fix-git", "config": {"task": {"git_commit_id": frozen}},
                               "verifier_result": {"rewards": {"reward": 1}}}))
    trusted_verifier_evidence(tmp_path)
    result = parse_trial_result("fix-git", path, 1, benchmark_revision=frozen)
    assert result.metadata["benchmark_revision"] == frozen
    with pytest.raises(ValueError, match="frozen benchmark revision"):
        parse_trial_result("fix-git", path, 1, benchmark_revision="c" * 40)


def test_command_timeout_kills_process_group_and_drains_output(tmp_path):
    async def check():
        env = LocalEnvironment(tmp_path / "events.jsonl", 0.15)
        started = time.monotonic()
        result = await env.exec("printf start; sleep 20")
        assert time.monotonic() - started < 3
        assert result.return_code == 124
        assert result.stdout == "start"
        assert "process group was killed" in result.stderr
    asyncio.run(check())


def test_command_output_bounded_without_deadlock(tmp_path):
    async def check():
        env = LocalEnvironment(tmp_path / "events.jsonl", 5)
        result = await env.exec("python3 -c 'print(\"x\" * 500000)'")
        assert result.return_code == 0
        assert len(result.stdout) < 132000
        assert "bytes omitted" in result.stdout
    asyncio.run(check())


@pytest.mark.parametrize("redirect", [" >/dev/null 2>&1", ""])
def test_successful_command_preserves_background_service(tmp_path, redirect):
    async def check():
        env = LocalEnvironment(tmp_path / "events.jsonl", 2)
        started = time.monotonic()
        result = await env.exec(f"sleep 10{redirect} & echo $!")
        pid = int(result.stdout.strip())
        try:
            assert result.return_code == 0
            assert time.monotonic() - started < 1
            os.kill(pid, 0)  # A following task command can still use its service.
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    asyncio.run(check())


def test_trial_parser_uses_exact_task_and_retains_incomplete_trace(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "events.jsonl").write_text(json.dumps({"event": "model_request", "messages": [{"role": "user", "content": "task"}]}) + "\ntruncated")
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"task_name": "fix-git", "trial_name": "fixed", "verifier_result": {"rewards": {"reward": 1}}, "exception_info": None}))
    trusted_verifier_evidence(tmp_path)
    result = parse_trial_result("fix-git", path, 3.1)
    assert result.passed and result.n_steps == 1 and result.duration_s == 3.1
    assert result.trace[0]["content"] == "task"
    with pytest.raises(ValueError):
        parse_trial_result("other-task", path, 1)


@pytest.mark.parametrize("reward", [None, float("nan"), float("inf"), -1, 2, True, "1"])
def test_unusable_rewards_are_unscorable_even_with_completed_verifier(tmp_path, reward):
    trusted_verifier_evidence(tmp_path)
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"task_name": "fix-git", "verifier_result": {"rewards": {"reward": reward}}}))
    result = parse_trial_result("fix-git", path, 1)
    assert result.status == "error" and result.reward is None and result.failure_class == "infra_error"


def test_reward_without_trusted_verifier_completion_is_not_quality_evidence(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"task_name": "fix-git", "verifier_result": {"rewards": {"reward": 0}}}))
    result = parse_trial_result("fix-git", path, 1)
    assert result.status == "error" and result.reward is None
    assert result.metadata["raw_reward"] == 0


def test_runtime_source_never_loaded_by_installed_adapter():
    adapter = Path("service/sandbox/installed_agent.py").read_text()
    assert "importlib" not in adapter and "exec(" not in adapter.replace("environment.exec(", "")
    assert "upload_file(self.config[\"candidate_host_path\"]" in adapter



def test_executed_verifier_timeout_keeps_explicit_zero_reward(tmp_path):
    from dataclasses import asdict
    from service.worker.job_runner import evaluation_metrics
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"task_name": "fix-git",
        "exception_info": {"exception_type": "AgentTimeoutError", "exception_message": "agent timed out"},
        "verifier_result": {"rewards": {"reward": 0}}}))
    (tmp_path / "verifier-execution.json").write_text(json.dumps({
        "contract": "pinned-pytest-cases-v2", "valid": True, "fresh_session_marker": True,
        "pytest_exit_code": 1,
        "counts": {"tests": 1, "passed": 0, "failed": 1, "skipped": 0, "pending": 0, "other": 0}}))
    result = parse_trial_result("fix-git", path, 1)
    assert result.status == "timeout" and result.reward == 0.0
    measured = evaluation_metrics([asdict(result)], 1)
    assert measured["complete"] is True and measured["mean_reward"] == 0.0
