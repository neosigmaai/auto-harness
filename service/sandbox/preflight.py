"""Warm audited verifier tools before any mutable candidate/model execution.

Only public tool dependencies are installed. Task assertions are neither copied
nor executed. Caches stay inside this attempt; no shared writable host cache.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import time

MANIFEST = json.loads(Path(__file__).with_name("verifier_contracts.json").read_text())
PROTOCOL = "verifier-tool-preflight-v1"
PYTHON_PREREQUISITE = "command -v python3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3)"


def offline_probe_command(task: str) -> str:
    """Inspect prepared tools without network access or test collection."""
    spec = MANIFEST["bootstrap"][task]
    commands = ["set -eu", "export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never PIP_NO_INDEX=1",
                "command -v python3 >/dev/null", "python3 --version"]
    if spec["mode"] == "uvx":
        commands += ['. "$HOME/.local/bin/env"', 'uv --version']
    commands.append(shlex.join(spec["warm_argv"]))
    return "\n".join(commands)


async def probe_prepared(environment, task: str, metadata_path: Path, entry: dict) -> dict:
    started = time.monotonic()
    metadata = {"protocol": "prepared-tools-v1", "task": task, "status": "failed",
                "cache_mode": "immutable_image_fresh_container", "image_id": entry["image_id"],
                "cache_key": entry["cache_key"], "candidate_visible_tooling": True,
                "hidden_assertions_copied": False, "downloads_permitted_by_probe": False}
    try:
        command = "timeout --kill-after=3s 20s bash -c " + shlex.quote(offline_probe_command(task))
        result = await environment.exec(command, timeout_sec=25)
        metadata["return_code"] = result.return_code
        metadata["observed_versions"] = [line for line in ((result.stdout or "") + "\n" + (result.stderr or "")).splitlines()
                                         if line.startswith(("uv ", "Python ", "pytest "))]
        if result.return_code:
            raise RuntimeError("Prepared verifier tools failed their offline readiness probe")
        metadata["status"] = "passed"
        return metadata
    finally:
        metadata["duration_s"] = time.monotonic() - started
        metadata_path.write_text(json.dumps(metadata, indent=2))


def preflight_command(spec: dict) -> str:
    warm = spec["warm_argv"]
    if warm[-1:] != ["--version"] or "pytest" not in warm:
        raise ValueError("Preflight may only inspect the pytest tool version")
    commands = ["set -eu", "export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=1"]
    if spec["mode"] == "uvx":
        archive = MANIFEST["uv_archive"]
        commands += [
            'test "$(uname -m)" = x86_64',
            'if ! command -v curl >/dev/null; then apt-get update -qq && apt-get install -y -qq curl ca-certificates; fi',
            'aos_uv_dir=$(mktemp -d)',
            'trap \'rm -rf "$aos_uv_dir"\' EXIT',
            'curl --fail --location --silent --show-error --connect-timeout 10 --max-time 75 '
            + shlex.quote(archive["url"]) + ' -o "$aos_uv_dir/uv.tar.gz"',
            'printf \'%s  %s\\n\' ' + shlex.quote(archive["sha256"]) + ' "$aos_uv_dir/uv.tar.gz" | sha256sum -c -',
            'tar -xzf "$aos_uv_dir/uv.tar.gz" -C "$aos_uv_dir"',
            'mkdir -p "$HOME/.local/bin"',
            'install -m 755 "$aos_uv_dir/uv-x86_64-unknown-linux-gnu/uv" "$HOME/.local/bin/uv"',
            'install -m 755 "$aos_uv_dir/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/uvx"',
            'printf \'%s\\n\' \'export PATH="$HOME/.local/bin:$PATH"\' > "$HOME/.local/bin/env"',
            '. "$HOME/.local/bin/env"', 'uv --version', shlex.join(warm),
            'aos_python=$(uv python find 3.13)', '"$aos_python" --version',
        ]
    elif spec["mode"] == "pip":
        commands += [shlex.join(spec["install_argv"]), shlex.join(warm), "python3 --version"]
    else:
        raise ValueError("Unsupported verifier preflight mode")
    return "\n".join(commands)


async def run_preflight(environment, task: str, metadata_path: Path, *, deadline: float,
                        fixture_override: dict | None = None) -> dict:
    spec = fixture_override if task == "sandbox-fixture" else MANIFEST["bootstrap"].get(task)
    if not spec:
        raise RuntimeError("No audited verifier preflight contract for this task")
    command = preflight_command(spec)
    started = time.monotonic()
    packages = [arg for arg in spec.get("install_argv", spec["warm_argv"]) if "==" in arg]
    metadata = {"protocol": PROTOCOL, "task": task, "mode": spec["mode"],
                "direct_packages": packages, "cache_mode": "isolated_container_cache_warmed_before_agent",
                "candidate_visible_tooling": True, "hidden_assertions_copied": False,
                "max_attempts": 2, "attempts": [], "status": "failed"}
    if spec["mode"] == "uvx":
        metadata["uv_archive"] = MANIFEST["uv_archive"]
    try:
        for index in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 15:
                break
            limit = min(160, int(remaining) - 10)
            attempt_start = time.monotonic()
            entry = {"attempt": index + 1, "timeout_sec": limit}
            try:
                # The container's timeout kills the command group even if a
                # Docker exec client is interrupted. The host has a second cap.
                wrapped = f"timeout --kill-after=5s {limit}s bash -c " + shlex.quote(command)
                result = await environment.exec(wrapped, timeout_sec=limit + 8)
                entry.update(return_code=result.return_code,
                             stdout_tail=(result.stdout or "")[-4000:], stderr_tail=(result.stderr or "")[-4000:])
                if result.return_code == 0:
                    observed = (result.stdout or "") + "\n" + (result.stderr or "")
                    metadata["observed_versions"] = [line for line in observed.splitlines()
                        if line.startswith(("uv ", "Python ", "pytest "))]
                    metadata["status"] = "passed"
            except Exception as exc:
                entry["error_type"] = type(exc).__name__
            finally:
                entry["duration_s"] = time.monotonic() - attempt_start
                metadata["attempts"].append(entry)
            if metadata["status"] == "passed":
                return metadata
            if index == 0:
                await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
        raise RuntimeError("Verifier dependency preflight failed before agent execution")
    finally:
        metadata["duration_s"] = time.monotonic() - started
        metadata_path.write_text(json.dumps(metadata, indent=2))
