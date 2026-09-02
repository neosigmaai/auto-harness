"""Artifact storage for job traces and improver prompts/responses.

Artifacts are addressed by convention, not by a database table:

    jobs/<job_id>/iterations/<n>/tasks/<task_id>/trace.json
    jobs/<job_id>/iterations/<n>/tasks/<task_id>/result.json
    jobs/<job_id>/iterations/<n>/improver/{prompt.txt,response.json}

``LocalArtifactStore`` is the only implementation today; an S3/GCS backend is a
drop-in behind ``ArtifactStore`` (same factory pattern as ``create_runner``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from api.config import REPO_ROOT, BenchmarkConfig, load_config


class ArtifactStore(Protocol):
    """Content-addressed-by-convention blob store keyed by relative POSIX paths."""

    def put(self, key: str, data: bytes | str | Path) -> None:
        """Store ``data``; a ``Path`` is copied, a ``str`` is utf-8 encoded."""
        ...

    def get(self, key: str) -> bytes:
        """Return the stored bytes. Raises ``FileNotFoundError`` for unknown keys."""
        ...

    def list(self, prefix: str) -> list[str]:
        """Sorted keys of every stored object whose key starts with ``prefix``."""
        ...

    def exists(self, key: str) -> bool:
        ...


def _validate_prefix(prefix: str) -> str:
    """Reject anything that could escape the store root. Empty means 'everything'."""
    if ".." in prefix or prefix.startswith("/") or "\\" in prefix:
        raise ValueError(f"unsafe artifact key prefix: {prefix!r}")
    return prefix


def _validate_key(key: str) -> str:
    """Validate a key that must name a single object."""
    _validate_prefix(key)
    if not key or key.endswith("/"):
        raise ValueError(f"artifact key must name a file: {key!r}")
    return key


class LocalArtifactStore:
    """Filesystem-backed artifact store rooted at a single directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _validate_key(key)

    def put(self, key: str, data: bytes | str | Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Path):
            shutil.copyfile(data, dest)
        elif isinstance(data, str):
            dest.write_bytes(data.encode("utf-8"))
        else:
            dest.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list(self, prefix: str) -> list[str]:
        _validate_prefix(prefix)
        if not self.root.is_dir():
            return []
        keys = [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        ]
        return sorted(key for key in keys if key.startswith(prefix))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


def create_artifact_store(config: BenchmarkConfig | None = None) -> ArtifactStore:
    """Factory mirroring ``create_runner``: local disk under ``config.artifacts_dir``."""
    cfg = config or load_config()
    root = Path(cfg.artifacts_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return LocalArtifactStore(root)


def trace_key(job_id: str, iteration: int, task_id: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/tasks/{task_id}/trace.json"


def result_key(job_id: str, iteration: int, task_id: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/tasks/{task_id}/result.json"


def improver_key(job_id: str, iteration: int, name: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/improver/{name}"
