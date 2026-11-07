"""Opt-in immutable tool images, built only from clean benchmark base images.

No trial container is ever committed. Preparation receives no candidate, task
instruction, verifier assertions, gateway token, or service/provider credentials.
The original verifier script is unchanged and may still access the network.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
from types import SimpleNamespace
from uuid import uuid4

from service.sandbox.preflight import MANIFEST, PYTHON_PREREQUISITE, offline_probe_command, run_preflight
from service.tasks_catalog import BY_NAME

PROTOCOL = "prepared-tools-v1"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recipe_digest(task: str) -> str:
    return digest({"protocol": PROTOCOL, "recipe": MANIFEST["bootstrap"][task],
                   "uv_archive": MANIFEST["uv_archive"], "python_prerequisite": PYTHON_PREREQUISITE,
                   "verifier_script": MANIFEST["tasks"][task],
                   "implementation": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("preflight.py", "prepared_images.py")}})


def cache_key(task: str, base_image_id: str) -> str:
    return digest({"base_image_id": base_image_id, "platform": "linux/amd64", "recipe_digest": recipe_digest(task)})


def validate_entry(task: str, entry: dict) -> None:
    if not isinstance(entry, dict):
        raise ValueError("Invalid prepared-image entry")
    if (entry.get("base_reference") != BY_NAME[task]["docker_image"] or
        entry.get("recipe_digest") != recipe_digest(task) or entry.get("platform") != "linux/amd64" or
        not SHA.fullmatch(str(entry.get("base_image_id", ""))) or not SHA.fullmatch(str(entry.get("image_id", ""))) or
        entry.get("cache_key") != cache_key(task, entry.get("base_image_id", ""))):
        raise ValueError("Prepared-image provenance does not match the audited task recipe")


def freeze_harness(tasks: list[str], benchmark_revision: str, manifest_path: str | None = None) -> dict:
    if not manifest_path:
        return {"mode": "cold-v1"}
    path = Path(manifest_path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Prepared-image manifest is oversized")
    manifest = json.loads(path.read_text())
    if (manifest.get("protocol") != PROTOCOL or manifest.get("benchmark_revision") != benchmark_revision or
        not isinstance(manifest.get("provider_id"), str) or not manifest["provider_id"].startswith("docker:")):
        raise ValueError("Prepared-image manifest protocol, revision or provider is invalid")
    entries = {}
    for task in tasks:
        if task not in manifest.get("entries", {}):
            raise ValueError("Prepared-image manifest does not cover all selected tasks")
        validate_entry(task, manifest["entries"][task])
        entries[task] = copy.deepcopy(manifest["entries"][task])
    return {"mode": PROTOCOL, "benchmark_revision": benchmark_revision,
            "provider_id": manifest["provider_id"], "manifest_sha256": digest(manifest), "entries": entries}


async def docker(*args: str, timeout=45):
    allowed = {"PATH", "HOME", "USER", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR"}
    env = {name: value for name, value in os.environ.items() if name in allowed}
    process = await asyncio.create_subprocess_exec("docker", *args, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError("Docker preparation operation failed: " + err.decode(errors="replace")[-1000:])
    return out.decode(errors="replace")


async def inspect_image(image: str) -> dict:
    return json.loads(await docker("image", "inspect", image))[0]


async def validate_local_entry(task: str, entry: dict, provider_id: str):
    validate_entry(task, entry)
    if "docker:" + (await docker("info", "--format", "{{.ID}}")).strip() != provider_id:
        raise RuntimeError("Prepared image belongs to another Docker daemon")
    base, prepared = await asyncio.gather(inspect_image(entry["base_reference"]), inspect_image(entry["image_id"]))
    if base["Id"] != entry["base_image_id"] or prepared["Id"] != entry["image_id"]:
        raise RuntimeError("Prepared or base image identity changed")
    if prepared["Architecture"] != "amd64" or prepared["Os"] != "linux":
        raise RuntimeError("Prepared image platform changed")
    if (prepared["Config"].get("Labels") or {}).get("aos.prepared.key") != entry["cache_key"]:
        raise RuntimeError("Prepared image provenance label is missing")
    labels = prepared["Config"].get("Labels") or {}
    if labels.get("aos.managed") not in (None, "false") or any(labels.get(name) not in (None, "")
            for name in ("aos.attempt", "aos.job", "aos.epoch", "aos.expires_at")):
        raise RuntimeError("Prepared image retained active builder lifecycle labels")
    base_labels = base["Config"].get("Labels") or {}
    if any(value != base_labels.get(name, "") for name, value in labels.items()
           if name.startswith("com.docker.compose.")):
        raise RuntimeError("Prepared image retained transient Compose labels")
    layers = base["RootFS"]["Layers"]
    if prepared["RootFS"]["Layers"][:len(layers)] != layers:
        raise RuntimeError("Prepared image does not extend the expected base layers")
    for field in ("Env", "User", "WorkingDir", "Volumes", "ExposedPorts", "Healthcheck"):
        if prepared["Config"].get(field) != base["Config"].get(field):
            raise RuntimeError("Prepared image altered base runtime configuration: " + field)
    for field in ("Cmd", "Entrypoint"):
        if (prepared["Config"].get(field) or []) != (base["Config"].get(field) or []):
            raise RuntimeError("Prepared image retained its builder command")


async def prepare(task: str, directory: Path, provider_id: str) -> dict:
    # Import only in the explicit preparation CLI, never at API manifest load.
    from service.worker.executors.harbor_runner import sandbox_overlay
    base = await inspect_image(BY_NAME[task]["docker_image"])
    if base["Architecture"] != "amd64" or base["Os"] != "linux":
        raise ValueError("Only audited linux/amd64 benchmark images are supported")
    key = cache_key(task, base["Id"])
    attempt = "prepare-" + uuid4().hex
    folder = directory / (task + "-" + attempt[-8:]); folder.mkdir(parents=True)
    compose = folder / "compose.json"
    config = sandbox_overlay(compose, gateway_url="http://host.docker.internal:9/v1", attempt_id=attempt,
                             job_id="trusted-tool-preparation", epoch=1, expires_at=time.time() + 480)
    main = config["services"]["main"]
    main.update(image=base["Id"], platform="linux/amd64", mem_limit="2048m", cpus=1,
                entrypoint=[])
    # Proxy variables are supplied only to individual trusted exec commands;
    # therefore the committed image retains exactly its base ENV.
    main.pop("environment", None)
    compose.write_text(json.dumps(config, indent=2))
    prefix = ("compose", "-p", attempt, "-f", str(compose))
    call = 0
    class Environment:
        async def exec(self, command, timeout_sec):
            nonlocal call
            call += 1
            # Keep installer output even if its client times out.
            log = folder / ("step-" + str(call) + ".log")
            arguments = [*prefix, "exec", "-T", "--user", "0"]
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                arguments += ["-e", name + "=http://aos-relay:8080"]
            process = await asyncio.create_subprocess_exec("docker", *arguments, "main", "bash", "-c", command,
                env={k:v for k,v in os.environ.items() if k in {"PATH","HOME","USER","TMPDIR","DOCKER_HOST","DOCKER_CONTEXT","DOCKER_CONFIG","XDG_RUNTIME_DIR"}},
                stdout=log.open("wb"), stderr=asyncio.subprocess.STDOUT)
            try:
                await asyncio.wait_for(process.wait(), timeout_sec)
            except BaseException:
                process.kill(); await process.wait(); raise
            output = log.read_text(errors="replace")
            return SimpleNamespace(return_code=process.returncode, stdout=output, stderr="")
    started = time.monotonic()
    image_id = None
    try:
        await docker(*prefix, "up", "-d", timeout=60)
        setup_started = time.monotonic()
        environment = Environment()
        initial = await environment.exec(PYTHON_PREREQUISITE, 180)
        if initial.return_code:
            raise RuntimeError("Prepared image Python prerequisite failed")
        preflight = await run_preflight(environment, task, folder / "preflight.json", deadline=setup_started + 400)
        cold_setup_s = time.monotonic() - setup_started
        container = (await docker(*prefix, "ps", "-q", "main")).strip()
        container_record = json.loads(await docker("inspect", container))[0]
        base_labels = base["Config"].get("Labels") or {}
        label_changes = []
        for name in (container_record["Config"].get("Labels") or {}):
            if name.startswith("com.docker.compose."):
                label_changes += ["--change", "LABEL " + name + "=" + json.dumps(base_labels.get(name, ""))]
        # Only this clean builder can be committed: no upload/candidate/gateway
        # capability APIs exist anywhere on this construction path.
        image_id = (await docker("commit", "--change", "CMD " + json.dumps(base["Config"].get("Cmd") or []),
            "--change", "ENTRYPOINT " + json.dumps(base["Config"].get("Entrypoint") or []),
            "--change", 'LABEL aos.managed="false" aos.attempt="" aos.job="" aos.epoch="" aos.expires_at=""',
            "--change", "LABEL aos.prepared.key=" + key, *label_changes, container, timeout=90)).strip()
        entry = {"base_reference": BY_NAME[task]["docker_image"], "base_image_id": base["Id"],
                 "image_id": image_id, "platform": "linux/amd64", "recipe_digest": recipe_digest(task),
                 "cache_key": key, "cold_setup_s": cold_setup_s, "preflight": preflight,
                 "candidate_ever_uploaded": False, "instructions_or_assertions_uploaded": False,
                 "credential_ever_supplied": False}
        await validate_local_entry(task, entry, provider_id)
        # Demonstrate readiness in a separate fresh network-disabled container.
        probe_started = time.monotonic()
        output = await docker("run", "--rm", "--network", "none", "--cpus", "1", "--memory", "2g",
            "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--entrypoint", "bash", image_id, "-c", "timeout --kill-after=3s 20s bash -c " + shlex.quote(offline_probe_command(task)), timeout=30)
        entry.update(offline_probe_s=time.monotonic() - probe_started, offline_probe_versions=output.splitlines(),
                     build_and_probe_s=time.monotonic() - started)
        (folder / "entry.json").write_text(json.dumps(entry, indent=2))
        return entry
    finally:
        await docker(*prefix, "down", "--volumes", "--remove-orphans", timeout=45)
        # A prepared image is a retained artifact, never a retained trial.
        if (await docker("ps", "-aq", "--filter", "label=com.docker.compose.project=" + attempt)).strip():
            raise RuntimeError("Preparation container cleanup could not be verified")


async def main(args):
    directory = Path(args.output).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    provider = "docker:" + (await docker("info", "--format", "{{.ID}}")).strip()
    manifest = {"protocol": PROTOCOL, "provider_id": provider, "benchmark_revision": MANIFEST["benchmark_revision"], "entries": {}}
    if Path(args.output).exists():
        existing = json.loads(Path(args.output).read_text())
        if (existing.get("protocol"), existing.get("provider_id"), existing.get("benchmark_revision")) != (
                PROTOCOL, provider, MANIFEST["benchmark_revision"]):
            raise ValueError("Use a new manifest path for a different provider/protocol/revision")
        for task, entry in existing.get("entries", {}).items():
            validate_entry(task, entry)
        manifest = existing
    sem = asyncio.Semaphore(2)
    async def build(task):
        async with sem:
            if task in manifest["entries"]:
                await validate_local_entry(task, manifest["entries"][task], provider)
                print(json.dumps({"task": task, "status": "existing_immutable_image"}), flush=True)
                return
            entry = await prepare(task, directory, provider)
            manifest["entries"][task] = entry
            temporary = Path(args.output).with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(args.output)
            print(json.dumps({"task": task, "status": "prepared", "cold_setup_s": entry["cold_setup_s"],
                              "offline_probe_s": entry["offline_probe_s"], "image_id": entry["image_id"]}), flush=True)
    await asyncio.gather(*(build(task) for task in args.tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(MANIFEST["bootstrap"]))
    parser.add_argument("--output", required=True, help="Versioned local manifest path, outside tracked source")
    asyncio.run(main(parser.parse_args()))
