"""Run with Harbor's Python to test hostile artifacts without launching Docker.

The normal service-only environment skips these; the pinned Harbor environment
can run them with ``python -m unittest service.tests.test_sandbox_artifacts``.
"""
import asyncio
import io
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

try:
    from service.sandbox.docker_environment import IsolatedDockerEnvironment
    from service.sandbox.installed_agent import ServiceInstalledAgent
except ModuleNotFoundError:
    IsolatedDockerEnvironment = None


def archive_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, value, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
            elif kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = value
                archive.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
    return buffer.getvalue()


@unittest.skipIf(IsolatedDockerEnvironment is None, "requires pinned Harbor environment")
class InstalledAdapterBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_install_checks_offline_readiness_before_candidate_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = object.__new__(ServiceInstalledAgent)
            adapter.config_path = Path(directory) / "adapter.json"
            adapter.config = {"benchmark_task": "fix-git", "prepared_image": {"image_id": "frozen"},
                              "candidate_host_path": "/trusted/candidate.py"}
            events = []
            async def probe(*args):
                events.append("offline_readiness")
            async def upload(source, destination):
                self.assertEqual(events[0], "offline_readiness")
                events.append("upload")
            environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
                                          upload_file=upload)
            with patch("service.sandbox.installed_agent.probe_prepared", probe), patch(
                    "service.sandbox.installed_agent.run_preflight", AsyncMock()) as cold:
                await adapter.install(environment)
            cold.assert_not_awaited()
            self.assertEqual(events, ["offline_readiness", "upload", "upload"])
            self.assertNotIn("apt-get", environment.exec.await_args.args[0])

    async def test_internal_preflight_configuration_never_reaches_candidate(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            adapter = object.__new__(ServiceInstalledAgent)
            adapter.config_path = Path(directory) / "adapter.json"
            adapter.config = {"timeout_sec": 10, "token": "scoped", "model": "fixture",
                              "base_url": "http://relay/v1", "benchmark_task": "private-task",
                              "preflight_override": {"secret_fixture_key": True},
                              "candidate_host_path": "/host/private/source.py"}
            observed = {}

            async def upload(source, destination):
                observed.update(json.loads(Path(source).read_text()))

            environment = SimpleNamespace(upload_file=upload,
                exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
            await adapter.run("instruction", environment, SimpleNamespace())
            self.assertEqual(set(observed), {"timeout_sec", "token", "model", "base_url", "instruction", "candidate_path"})
            self.assertFalse((Path(directory) / "container-config.json").exists())

    async def test_dependency_failure_stops_before_candidate_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = object.__new__(ServiceInstalledAgent)
            adapter.config_path = Path(directory) / "adapter.json"
            adapter.config = {"benchmark_task": "fix-git"}
            environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
                                          upload_file=AsyncMock())
            with patch("service.sandbox.installed_agent.run_preflight", AsyncMock(side_effect=RuntimeError("dependency failure"))):
                with self.assertRaisesRegex(RuntimeError, "dependency failure"):
                    await adapter.install(environment)
            environment.upload_file.assert_not_awaited()


@unittest.skipIf(IsolatedDockerEnvironment is None, "requires pinned Harbor environment")
class ArtifactBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def copy(self, payload, destination, directory=True):
        environment = object.__new__(IsolatedDockerEnvironment)
        environment._run_docker_compose_command = AsyncMock(return_value=SimpleNamespace(stdout="fixture-container"))
        stream = asyncio.StreamReader()
        stream.feed_data(payload)
        stream.feed_eof()
        process = SimpleNamespace(stdout=stream, returncode=0, wait=AsyncMock(return_value=0))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await environment._copy_out("/logs/agent" if directory else "/logs/reward.txt", destination, directory)

    async def test_regular_files_are_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "agent"
            await self.copy(archive_bytes([("./trace.json", b"[]", "file")]), destination)
            self.assertEqual((destination / "trace.json").read_bytes(), b"[]")

    async def test_single_file_maps_to_exact_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "reward.txt"
            await self.copy(archive_bytes([("reward.txt", b"1", "file")]), destination, False)
            self.assertEqual(destination.read_bytes(), b"1")

    async def test_traversal_and_absolute_paths_rejected(self):
        for name in ["../outside", "/tmp/outside", "./a/../../outside"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "Unsafe candidate artifact"):
                    await self.copy(archive_bytes([(name, b"bad", "file")]), Path(directory) / "agent")
                self.assertFalse((Path(directory) / "outside").exists())

    async def test_symlinks_and_special_files_never_created(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "agent"
            await self.copy(archive_bytes([
                ("escape", "/tmp/elsewhere", "link"), ("pipe", b"", "fifo"),
                ("trace.json", b"[]", "file"),
            ]), destination)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ["trace.json"])

    async def test_existing_destination_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            destination = root / "agent"
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink destination"):
                await self.copy(archive_bytes([("trace.json", b"bad", "file")]), destination)
            self.assertFalse((outside / "trace.json").exists())

    async def test_sparse_or_declared_huge_member_rejected_before_extraction(self):
        info = tarfile.TarInfo("huge")
        info.size = 65 * 1024 * 1024
        payload = info.tobuf() + bytes(1024)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "extraction limit"):
                await self.copy(payload, Path(directory) / "agent")


if __name__ == "__main__":
    unittest.main()
