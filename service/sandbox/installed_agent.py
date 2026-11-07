"""Fixed Harbor adapter. Candidate bytes are uploaded, never imported here."""
from __future__ import annotations

import json
from pathlib import Path
import time

from harbor.agents.installed.base import BaseInstalledAgent
from service.sandbox.preflight import PYTHON_PREREQUISITE, probe_prepared, run_preflight


class ServiceInstalledAgent(BaseInstalledAgent):
    def __init__(self, *args, host_config_path: str, **kwargs):
        super().__init__(*args, version="1.2.0", **kwargs)
        self.config_path = Path(host_config_path)
        self.config = json.loads(self.config_path.read_text())

    @staticmethod
    def name():
        return "aos-installed-candidate"

    async def install(self, environment):
        setup_deadline = time.monotonic() + 400  # Harbor grants 420s including copies.
        prepared = self.config.get("prepared_image")
        if prepared:
            await probe_prepared(environment, self.config["benchmark_task"],
                                 self.config_path.with_name("verifier-preflight.json"), prepared)
        else:
            result = await environment.exec(PYTHON_PREREQUISITE, user="root", timeout_sec=180)
            if result.return_code:
                raise RuntimeError("Task image needs Python 3; installation failed")
            await run_preflight(environment, self.config["benchmark_task"],
                                self.config_path.with_name("verifier-preflight.json"), deadline=setup_deadline,
                                fixture_override=self.config.get("preflight_override"))
        await environment.exec("mkdir -p /installed-agent/aos /logs/agent /logs/verifier /logs/artifacts", user="root")
        await environment.upload_file(Path(__file__).with_name("runtime.py"), "/installed-agent/aos/runtime.py")
        await environment.upload_file(self.config["candidate_host_path"], "/installed-agent/aos/candidate.py")

    async def run(self, instruction, environment, context):
        # Only a revocable per-attempt token crosses this boundary. The provider
        # credential, DB credentials and control-plane environment never do.
        # Internal setup recipes and benchmark identifiers stay in the trusted
        # adapter. The candidate receives only its runtime protocol fields.
        runtime_fields = {"base_url", "token", "model", "timeout_sec", "max_steps", "max_tokens", "model_timeout_sec"}
        config = {key: value for key, value in self.config.items() if key in runtime_fields}
        config.update(instruction=instruction, candidate_path="/installed-agent/aos/candidate.py")
        self.config_path.with_name("instruction.txt").write_text(instruction)
        target = self.config_path.with_name("container-config.json")
        target.write_text(json.dumps(config))
        target.chmod(0o600)
        try:
            await environment.upload_file(target, "/installed-agent/aos/config.json")
            result = await environment.exec(
                "python3 /installed-agent/aos/runtime.py --config /installed-agent/aos/config.json"
                " > /logs/agent/runtime.stdout 2> /logs/agent/runtime.stderr",
                timeout_sec=float(config["timeout_sec"]) + 5,
            )
        finally:
            target.unlink(missing_ok=True)
        if result.return_code:
            raise RuntimeError(f"Candidate runtime exited with code {result.return_code}")
        context.metadata = {"execution_boundary": "task-container", "adapter_version": "1.2.0"}
