"""Opt-in, zero-provider-spend end-to-end Docker/Harbor sandbox check.

Run with the service virtualenv: python -m service.sandbox.smoke.
A local fake OpenAI-compatible gateway drives the unchanged starting agent;
the fixture tests process boundaries, network restrictions and verification.
This is an infrastructure test, not evidence of benchmark quality improvement.
"""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import http.server
import json
import shlex
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4

from service.config import settings
from service.worker.executors.harbor_runner import (
    ROOT, RELAY_IMAGE, harbor_config, parse_trial_result, remove_attempt,
    runner_environment, sandbox_overlay, write_private_json,
)


def production_source_proof():
    paths = [p for p in (ROOT / "service").rglob("*.py") if "tests" not in p.parts]
    paths += list((ROOT / "service" / "migrations").glob("*.sql"))
    paths += list((ROOT / "service" / "sandbox").rglob("*.json"))
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths)}
    return {"files": files, "aggregate_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def main(*, expect_bootstrap_failure=False, expect_assertion_failure=False, tool_task=None, prepared_manifest=None):
    from service.sandbox.preflight import MANIFEST
    from service.sandbox.prepared_images import freeze_harness, validate_local_entry
    from service.tasks_catalog import BY_NAME
    if prepared_manifest and not tool_task:
        raise ValueError("A prepared fixture requires --tool-task")
    prepared = None
    if prepared_manifest:
        harness = freeze_harness([tool_task], MANIFEST["benchmark_revision"], prepared_manifest)
        prepared = harness["entries"][tool_task]
        asyncio.run(validate_local_entry(tool_task, prepared, harness["provider_id"]))
    attempt = "smoke-" + uuid4().hex
    directory = ROOT / "runs" / attempt
    directory.mkdir(parents=True)
    source_proof = production_source_proof()
    (directory / "source-proof.json").write_text(json.dumps(source_proof, indent=2))
    sentinel = directory / "host-only-marker"
    sentinel.write_text("must never be visible in candidate container")
    fixture = directory / "sandbox-fixture"
    (fixture / "environment").mkdir(parents=True)
    (fixture / "tests").mkdir()
    (fixture / "task.toml").write_text(f'''version = "1.0"
[agent]
timeout_sec = 60
[verifier]
timeout_sec = 120
[environment]
docker_image = "{BY_NAME[tool_task]['docker_image'] if tool_task else RELAY_IMAGE}"
cpus = 1
memory_mb = 512
''')
    (fixture / "instruction.md").write_text("Complete the local sandbox infrastructure checks and write /tmp/sandbox-ok.")
    (fixture / "tests" / "test_outputs.py").write_text('''from pathlib import Path
import pytest

@pytest.mark.parametrize('case', range(27))
def test_sandbox_result(case):
    assert Path('/tmp/sandbox-ok').read_text() == 'sandbox-ok'
''')
    if expect_assertion_failure:
        test_file = fixture / "tests" / "test_outputs.py"
        test_file.write_text(test_file.read_text() + '    assert case != 13\n')
    verifier_script = fixture / "tests" / "test.sh"
    verifier_script.write_text('''#!/bin/bash
mkdir -p /logs/verifier
python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
''')
    if expect_bootstrap_failure:
        verifier_script.write_text('#!/bin/bash\naos_missing_verifier_dependency\necho 0 > /logs/verifier/reward.txt\n')
    elif tool_task:
        recipe = MANIFEST["bootstrap"][tool_task]
        invoke = shlex.join(recipe["warm_argv"][:-1] + ["--ctrf", "/logs/verifier/ctrf.json", "/tests/test_outputs.py", "-rA"])
        setup = '. "$HOME/.local/bin/env"\n' if recipe["mode"] == "uvx" else ""
        verifier_script.write_text('#!/bin/bash\n' + setup + invoke + '\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n')
    facts = {"model_calls": 0, "real_provider_calls": 0,
             "tool_task": tool_task, "harness_mode": "prepared-tools-v1" if prepared else "cold-v1",
             "production_source_sha256": source_proof["aggregate_sha256"],
             "expected_outcome": "unscorable" if expect_bootstrap_failure else ("failed" if expect_assertion_failure else "passed")}

    class FakeGateway(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers.get("Authorization") == "Bearer fake-local-capability"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            facts["model_calls"] += 1
            if facts["model_calls"] == 1:
                inspection = subprocess.run(["docker", "ps", "-q", "--filter", f"label=aos.attempt={attempt}",
                    "--filter", "label=com.docker.compose.service=main"], capture_output=True, text=True, check=True)
                record = json.loads(subprocess.check_output(["docker", "inspect", inspection.stdout.strip()]))[0]
                assert not record["HostConfig"]["Privileged"]
                assert not any(m["Type"] == "bind" for m in record["Mounts"])
                assert len(record["NetworkSettings"]["Networks"]) == 1
                facts["no_host_bind_mount"] = True
                facts["single_internal_network"] = True
                checks = f'''
import json, os, pathlib, socket, urllib.request, urllib.error
assert not pathlib.Path({str(sentinel)!r}).exists()
assert not any(k in os.environ for k in ["NEOSIGMA_OPEN_API_KEY", "OPENAI_API_KEY", "OPEN_API_KEY", "DATABASE_URL"])
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
proxy = urllib.request.build_opener(urllib.request.ProxyHandler({{"http": "http://aos-relay:8080", "https": "http://aos-relay:8080"}}))
for url in ["http://127.0.0.1:5432/", "http://169.254.169.254/latest/meta-data/", "http://host.docker.internal:{self.server.server_port}/private"]:
    try:
        proxy.open(url, timeout=4)
        raise AssertionError("private request was allowed: " + url)
    except urllib.error.HTTPError as e:
        assert e.code == 403
# Direct host access must not bypass the relay's fixed model route.
try:
    s = socket.create_connection(("host.docker.internal", {self.server.server_port}), timeout=2)
except OSError:
    pass
else:
    s.close()
    raise AssertionError("direct host access was allowed")
with proxy.open("https://example.com", timeout=20) as response:
    assert response.status == 200
pathlib.Path("/tmp/sandbox-ok").write_text("sandbox-ok")
print("host marker absent; credential env absent; private/metadata proxy denied; direct host denied; public HTTPS works")
'''
                command = "python3 - <<'AOS_SMOKE'\n" + checks + "\nAOS_SMOKE"
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "check", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]}
            else:
                output = payload["messages"][-1]["content"]
                facts["checks_output"] = output
                assert "public HTTPS works" in output, output
                message = {"role": "assistant", "content": "Infrastructure checks completed.", "tool_calls": None}
            data = json.dumps({"choices": [{"message": message}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), FakeGateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source = ROOT / "agent" / "templates" / "terminal_bench.py"
    adapter = directory / "adapter.json"
    write_private_json(adapter, {"candidate_host_path": str(source), "base_url": "http://aos-relay:8080/v1",
        "token": "fake-local-capability", "model": "fake-model", "timeout_sec": 60, "max_steps": 3, "max_tokens": 100,
        "benchmark_task": tool_task or "sandbox-fixture", "prepared_image": prepared,
        "preflight_override": {"mode": "pip", "install_argv": ["python3", "-m", "pip", "install", "--no-input",
             "pytest==8.4.1", "pytest-json-ctrf==0.3.5"], "warm_argv": ["python3", "-m", "pytest", "--version"]}})
    overlay = directory / "sandbox-compose.json"
    sandbox_overlay(overlay, gateway_url=f"http://host.docker.internal:{server.server_port}/v1", attempt_id=attempt,
                    job_id="fixture", epoch=1, expires_at=time.time() + 480)
    config = harbor_config("sandbox-fixture", directory, adapter, overlay, 60, 120, prepared_image=prepared)
    config["tasks"] = [{"path": str(fixture)}]
    config["verifier"]["kwargs"]["expected_script_sha256"] = hashlib.sha256(verifier_script.read_bytes()).hexdigest()
    config_path = directory / "harbor-config.json"
    write_private_json(config_path, config)
    started = time.monotonic()
    try:
        with (directory / "runner.log").open("w") as output:
            completed = subprocess.run([settings.harbor_executable, "run", "--config", str(config_path)],
                cwd=ROOT, env=runner_environment(), stdout=output, stderr=subprocess.STDOUT, timeout=480)
        paths = list((directory / "harbor" / "trial").glob("*/result.json"))
        if len(paths) != 1:
            raise AssertionError(f"No trial result: {directory / 'runner.log'}")
        result = parse_trial_result("sandbox-fixture", paths[0], time.monotonic() - started)
        preflight_path = directory / "verifier-preflight.json"
        facts["preflight"] = json.loads(preflight_path.read_text()) if preflight_path.exists() else None
        facts.update(status=result.status, reward=result.reward, duration_s=result.duration_s, runner_exit_code=completed.returncode,
                     verifier_validity=result.metadata.get("verifier_validity"))
        if expect_bootstrap_failure:
            assert result.status == "error" and result.reward is None and result.failure_class == "infra_error", facts
        elif expect_assertion_failure:
            assert result.status == "failed" and result.reward == 0 and result.metadata['verifier_validity']['valid'], facts
        else:
            assert result.passed, facts
        assert facts["model_calls"] == 2, facts
    finally:
        asyncio.run(remove_attempt(attempt))
        server.shutdown()
        adapter.unlink(missing_ok=True)
        sentinel.unlink(missing_ok=True)
        facts["production_source_unchanged"] = production_source_proof() == source_proof
        (directory / "proof.json").write_text(json.dumps(facts, indent=2))
    assert facts["production_source_unchanged"], "Production source changed during fixture"
    print(json.dumps({**facts, "artifact_dir": str(directory)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-bootstrap-failure", action="store_true")
    parser.add_argument("--expect-assertion-failure", action="store_true")
    parser.add_argument("--tool-task", help="Use this task's base image and audited tool recipe; fixture assertions remain synthetic")
    parser.add_argument("--prepared-manifest", help="Use an already prepared immutable image for --tool-task")
    main(**vars(parser.parse_args()))
