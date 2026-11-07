"""Probe verifier download endpoints through the actual isolated relay; no models.

Downloads bounded prefixes only; never executes downloaded content. Creates and
removes its own Docker Compose project. Run on the Docker worker host.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from service.worker.executors.harbor_runner import RELAY_IMAGE, runner_environment, sandbox_overlay


PROBE = r'''
import concurrent.futures, json, time, urllib.error, urllib.request
urls = {
    "pypi_ctrf": "https://pypi.org/simple/pytest-json-ctrf/",
    "uv_installer": "https://astral.sh/uv/0.9.5/install.sh",
    "uv_binary": "https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz",
}
def one(item):
    name, url, repetition = item
    start = time.monotonic()
    user_agent = {1: "Python-urllib/3.12", 2: "curl/8.7.1", 3: "uv/0.9.5"}[repetition]
    result = {"endpoint": name, "repetition": repetition, "user_agent": user_agent}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=45) as response:
            result.update(status=response.status, bytes_read=len(response.read(65536)))
    except urllib.error.HTTPError as error:
        result.update(error_type=type(error).__name__, status=error.code,
                      server=error.headers.get("Server"),
                      error_body_prefix=error.read(256).decode(errors="replace"))
    except Exception as error:
        result.update(error_type=type(error).__name__, detail=str(error)[:300])
    result["elapsed_s"] = round(time.monotonic()-start, 3)
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(one, [(name, url, repeat) for repeat in range(1, 4) for name, url in urls.items()]))
print(json.dumps(results))
'''


async def main():
    ident = "aos-egress-" + uuid4().hex[:12]
    directory = (Path("runs") / ident).resolve()
    directory.mkdir(parents=True)
    overlay = sandbox_overlay(directory / "compose.json", gateway_url="http://host.docker.internal:9/v1",
        attempt_id=ident, job_id=ident, epoch=1, expires_at=time.time()+240)
    # Snapshot the relay so diagnostics describe one fixed implementation.
    original = Path(overlay["services"]["aos-relay"]["volumes"][0]["source"])
    shutil.copyfile(original, directory / "relay.py")
    overlay["services"]["aos-relay"]["volumes"][0]["source"] = str(directory / "relay.py")
    overlay["services"]["main"].update(image=RELAY_IMAGE, mem_limit="256m", cpus=1)
    config = directory / "compose.json"
    config.write_text(json.dumps(overlay))
    command = ["docker", "compose", "-p", ident, "-f", str(config)]
    environment = runner_environment()

    async def run(*arguments, timeout=180):
        process = await asyncio.create_subprocess_exec(*command, *arguments,
            env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace")[-1500:])
        return stdout.decode()

    try:
        await run("up", "-d", "--wait")
        results = json.loads(await run("exec", "-T", "main", "python3", "-c", PROBE))
        report = {"mode": "actual_docker_relay_no_model_calls", "model_calls": 0,
                  "relay_sha256": hashlib.sha256((directory / "relay.py").read_bytes()).hexdigest(),
                  "requests": results, "successful_requests": sum(r.get("status") == 200 for r in results)}
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"report": str(directory / "report.json"), **report}, indent=2))
    finally:
        await run("down", "--remove-orphans", timeout=45)


if __name__ == "__main__":
    asyncio.run(main())
