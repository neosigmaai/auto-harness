"""Package a container's /workspace into submission.tar.gz on the host."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from programbench.constants import DOCKER_EXECUTABLE, WORKSPACE_DIR
from programbench.container import ContainerEnvironment


def package_submission(env: ContainerEnvironment, dest: Path) -> Path:
    """Tar /workspace inside the container and copy the archive to ``dest``.

    The eval pipeline (programbench.eval.eval) wipes /workspace then streams
    this tar back in, so the contents must be everything the build needs —
    sources plus a working compile.sh at the root.

    Cheat-prevention: strip ``/workspace/executable`` at tar time and remove
    any exact copy of the stashed cleanroom binary before archiving.
    ``compile.sh`` is then forced to actually build the executable; if it
    doesn't, eval reports ``copy_executable_failed``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    container_tar = "/tmp/submission.tar.gz"
    workspace = shlex.quote(WORKSPACE_DIR)
    container_tar_q = shlex.quote(container_tar)
    tar_result = env.execute(
        f"""
        set -eu
        rm -f {workspace}/executable
        if [ -s /opt/orig/executable.sha256 ]; then
          orig_hash="$(head -n1 /opt/orig/executable.sha256 | awk '{{print $1}}')"
          while IFS= read -r -d '' path; do
            file_hash="$(sha256sum "$path" 2>/dev/null | awk '{{print $1}}')" || continue
            if [ "$file_hash" = "$orig_hash" ]; then
              rm -f -- "$path"
            fi
          done < <(find {workspace} -type f -print0)
        fi
        tar -C {workspace} -czf {container_tar_q} .
        """,
        timeout=600,
    )
    if tar_result.get("returncode") != 0:
        raise RuntimeError(f"tar inside container failed: {tar_result.get('output', '').strip()[-2000:]}")
    cp = subprocess.run(
        [DOCKER_EXECUTABLE, "cp", f"{env.container_id}:{container_tar}", str(dest)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"docker cp failed: {cp.stderr.strip()}")
    return dest


def write_manifest(env: ContainerEnvironment, dest: Path) -> Path:
    """Write a `find /workspace -type f` listing for trace introspection."""
    listing = env.execute(f"find {WORKSPACE_DIR} -maxdepth 6 -type f | head -2000", timeout=60)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(listing.get("output", ""))
    return dest
