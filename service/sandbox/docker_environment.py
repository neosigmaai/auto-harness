"""Harbor Docker environment without writable host log mounts.

Task-authored Compose files are deliberately unsupported in this first adapter:
otherwise a benchmark manifest could silently add host mounts or host networking.
The pinned selected tasks use ordinary single-container environments.
"""
import asyncio
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from harbor.environments.docker.docker import DockerEnvironment


class IsolatedDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, **kwargs):
        self._provenance_path = kwargs.pop("provenance_path", None)
        self._prepared_image = kwargs.pop("prepared_image", None)
        if self._prepared_image:
            config = kwargs.get("task_env_config")
            if config is None or config.docker_image != self._prepared_image["base_reference"]:
                raise ValueError("Prepared image does not match the pinned task's base image")
            # Harbor caches docker_image in its compose variables in __init__.
            kwargs["task_env_config"] = config.model_copy(update={"docker_image": self._prepared_image["image_id"]})
        kwargs["mounts"] = []
        super().__init__(*args, **kwargs)
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Task-authored Docker Compose requires a separately audited sandbox adapter")

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"mounted": False})

    async def start(self, force_build=False):
        await super().start(force_build=force_build)
        result = await self._run_docker_compose_command(["ps", "-q", "main"])
        container_id = (result.stdout or "").strip()
        process = await asyncio.create_subprocess_exec("docker", "inspect", container_id,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        stdout, _ = await process.communicate()
        if process.returncode:
            raise RuntimeError("Cannot verify task container isolation")
        record = json.loads(stdout)[0]
        if self._prepared_image and record["Image"] != self._prepared_image["image_id"]:
            raise RuntimeError("Docker launched an image different from the frozen prepared image")
        host = record["HostConfig"]
        networks = record["NetworkSettings"]["Networks"]
        if host["Privileged"] or any(m["Type"] == "bind" for m in record["Mounts"]) or len(networks) != 1:
            raise RuntimeError("Task container isolation does not match configured boundary")
        if self._provenance_path:
            Path(self._provenance_path).write_text(json.dumps({
                "container_id": container_id, "image_id": record["Image"],
                "image_reference": record["Config"]["Image"],
                "memory_limit_bytes": host["Memory"], "nano_cpus": host["NanoCpus"],
                "pids_limit": host["PidsLimit"], "cap_drop": host.get("CapDrop", []),
                "network_names": list(networks), "host_bind_mounts": False,
                "prepared_image": self._prepared_image,
            }, indent=2))

    async def _copy_out(self, source: str, target, directory: bool, service="main"):
        """Docker cp cannot read tmpfs; stream tar with bounded, safe extraction.

        Symlinks, special files and traversal entries from the candidate are
        rejected rather than interpreted on the host. Nothing is executed here.
        """
        result = await self._run_docker_compose_command(["ps", "-q", service])
        container_id = (result.stdout or "").strip()
        if not container_id:
            raise RuntimeError("Task container is no longer running")
        path = PurePosixPath(source)
        parent, name = (str(path), ".") if directory else (str(path.parent), path.name)
        process = await asyncio.create_subprocess_exec("docker", "exec", "-u", "0", container_id,
            "tar", "-C", parent, "-cf", "-", "--", name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        payload = bytearray()
        try:
            async with asyncio.timeout(45):
                while chunk := await process.stdout.read(65536):
                    payload.extend(chunk)
                    if len(payload) > 64 * 1024 * 1024:
                        raise RuntimeError("Candidate artifact exceeds 64 MiB collection limit")
                if await process.wait():
                    raise RuntimeError(f"Container artifact unavailable: {source}")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        destination = Path(target)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            extracted_size = 0
            for index, member in enumerate(archive):
                extracted_size += member.size
                if index > 10000 or extracted_size > 64 * 1024 * 1024:
                    raise RuntimeError("Candidate artifact exceeds extraction limit")
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError("Unsafe candidate artifact path")
                if not member.isdir() and not member.isfile():
                    continue  # Never follow or create candidate-controlled links.
                parts = member_path.parts
                relative = Path(*parts) if directory else Path(*parts[1:])
                output = destination / relative
                # Existing paths may themselves originate from prior artifacts.
                for ancestor in [output, *output.parents]:
                    if ancestor.is_symlink():
                        raise RuntimeError("Refusing artifact symlink destination")
                    if ancestor == destination.parent:
                        break
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as incoming, output.open("wb") as outgoing:
                        while chunk := incoming.read(65536):
                            outgoing.write(chunk)

    async def download_dir(self, source_dir, target_dir):
        await self._copy_out(source_dir, target_dir, True)

    async def download_file(self, source_path, target_path):
        await self._copy_out(source_path, target_path, False)
