"""Run inference for a batch of ProgramBench tasks, producing submissions.

One container per task, lifecycle managed in try/finally so we never leak.
The agent (loaded from agent.agent.HarnessAgent) drives the LLM and bash
execution; this module owns container start, timeout, packaging, cleanup.

Cheat-prevention: the cleanroom image ships ``/workspace/executable`` (the
original binary). If we tar /workspace as-is, eval restores that binary and
``compile.sh`` doesn't actually need to build anything. ProgramBench's
``eval_clean_hashes`` filter only removes files matching upstream hashes, but
the cleanroom binary is rebuilt with different flags so its hash differs and
the filter doesn't catch it. We pre-move the original to ``/opt/orig/executable``
before the agent starts; the agent can execute it for behavioural checks but
cannot read or copy it, and the submission tar contains only what the agent
built.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from programbench.constants import DOCKER_EXECUTABLE, WORKSPACE_DIR
from programbench.container import ContainerEnvironment

from agent.helpers.program_bench.submission import package_submission, write_manifest

ORIG_BINARY_PATH = "/opt/orig/executable"
ORIG_BINARY_HASH_PATH = "/opt/orig/executable.sha256"
AGENT_EXEC_USER = "1000:1000"
AGENT_HOME = "/tmp/pbagent-home"


def _prepare_cleanroom(env: ContainerEnvironment) -> bool:
    """Move the reference binary out of /workspace and demote agent execution.

    The agent gets execute-only access to the original binary as uid/gid 1000.
    It can compare behaviour by running it, but it cannot read, chmod, delete,
    or copy it into the submission workspace.

    Returns True if a binary was found and stashed, False otherwise.
    """
    out = env.execute(
        f"""
        set -eu
        mkdir -p /opt/orig {AGENT_HOME}
        stashed=missing
        if [ -e /workspace/executable ]; then
          mv /workspace/executable {ORIG_BINARY_PATH}
          chown 0:0 {ORIG_BINARY_PATH} 2>/dev/null || true
          chmod 0111 {ORIG_BINARY_PATH}
          sha256sum {ORIG_BINARY_PATH} | awk '{{print $1}}' > {ORIG_BINARY_HASH_PATH}
          chown 0:0 {ORIG_BINARY_HASH_PATH} 2>/dev/null || true
          chmod 0444 {ORIG_BINARY_HASH_PATH}
          stashed=stashed
        fi
        chown -R 1000:1000 /workspace {AGENT_HOME}
        chmod 0755 /opt
        chmod 0555 /opt/orig
        printf '%s\\n' "$stashed"
        """,
        timeout=15,
    )
    if out.get("returncode") != 0:
        raise RuntimeError(f"cleanroom setup failed: {out.get('output', '').strip()[-2000:]}")
    setattr(env, "agent_exec_user", AGENT_EXEC_USER)
    return "stashed" in out.get("output", "")


def _run_agent_with_timeout(
    agent_cls: Callable,
    env: ContainerEnvironment,
    instance: dict,
    model_name: str,
    reasoning_effort: str | None,
    per_task_timeout: int,
) -> tuple[dict, str | None]:
    """Run the agent under a hard wall-clock budget. Returns (trace, error)."""
    async def _go() -> dict:
        setattr(env, "agent_deadline", time.monotonic() + per_task_timeout)
        agent = agent_cls(
            env=env,
            instance=instance,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        return await asyncio.wait_for(agent.run(), timeout=per_task_timeout)

    try:
        return asyncio.run(_go()), None
    except asyncio.TimeoutError:
        return {"timeout": True, "per_task_timeout": per_task_timeout}, f"timeout after {per_task_timeout}s"
    except Exception as e:
        return {"crash": True, "error": f"{type(e).__name__}: {e}"}, f"agent crash: {type(e).__name__}: {e}"


def _orchestrate_one(
    *,
    agent_cls: Callable,
    iid: str,
    instance: dict,
    run_dir: Path,
    model_name: str,
    reasoning_effort: str | None,
    per_task_timeout: int,
    docker_cpus: int,
    cleanroom_tag: str,
) -> dict[str, Any]:
    """Drive one task end-to-end: container → agent → submission → cleanup."""
    started = time.time()
    iid_dir = run_dir / iid
    iid_dir.mkdir(parents=True, exist_ok=True)
    tar_path = iid_dir / "submission.tar.gz"
    image_ref = f"{instance['image_name']}:{cleanroom_tag}"
    env: ContainerEnvironment | None = None
    trace: dict = {}
    error: str | None = None

    try:
        env = ContainerEnvironment(
            image=image_ref,
            cwd=WORKSPACE_DIR,
            executable=DOCKER_EXECUTABLE,
            cpus=docker_cpus,
            run_args=["--network", "none", "--security-opt", "no-new-privileges:true", "--user", "0:0"],
        )
    except Exception as e:
        error = f"container_start_failed: {type(e).__name__}: {e}"

    if env is not None:
        try:
            stashed = _prepare_cleanroom(env)
            if not stashed:
                print(f"[inference] WARNING: {iid} cleanroom missing /workspace/executable")
            trace, error = _run_agent_with_timeout(
                agent_cls=agent_cls,
                env=env,
                instance=instance,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                per_task_timeout=per_task_timeout,
            )
            # Always attempt to package whatever the agent produced, even on
            # timeout/crash — partial /workspace contents may still score.
            try:
                package_submission(env, tar_path)
                write_manifest(env, iid_dir / "manifest.txt")
            except Exception as e:
                pkg_err = f"package_failed: {type(e).__name__}: {e}"
                error = f"{error}; {pkg_err}" if error else pkg_err
        finally:
            env.cleanup()

    (iid_dir / "trace.json").write_text(
        json.dumps({"trace": trace, "error": error, "wall_time": time.time() - started}, indent=2, default=str)
    )

    return {
        "iid": iid,
        "status": "error" if error and not tar_path.exists() else ("ok" if not error else "partial"),
        "error": error,
        "tar_path": str(tar_path) if tar_path.exists() else None,
        "wall_time": time.time() - started,
    }


def run_inference(
    *,
    agent_cls: Callable,
    task_ids: list[str],
    instances: dict[str, dict],
    run_dir: Path,
    model_name: str,
    reasoning_effort: str | None = None,
    max_concurrency: int = 4,
    per_task_timeout: int = 1800,
    docker_cpus: int = 4,
    cleanroom_tag: str = "task_cleanroom",
) -> dict[str, dict[str, Any]]:
    """Run agent inference over ``task_ids`` in parallel, returning per-iid summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}

    print(
        f"[inference] {len(task_ids)} task(s), max_concurrency={max_concurrency}, "
        f"per_task_timeout={per_task_timeout}s, docker_cpus={docker_cpus}"
    )

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        futures = {
            pool.submit(
                _orchestrate_one,
                agent_cls=agent_cls,
                iid=iid,
                instance=instances[iid],
                run_dir=run_dir,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                per_task_timeout=per_task_timeout,
                docker_cpus=docker_cpus,
                cleanroom_tag=cleanroom_tag,
            ): iid
            for iid in task_ids
            if iid in instances
        }
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:
                summary = {
                    "iid": iid,
                    "status": "error",
                    "error": f"orchestrator_crash: {type(e).__name__}: {e}",
                    "tar_path": None,
                    "wall_time": 0.0,
                }
            summaries[iid] = summary
            print(
                f"[inference] {iid}: {summary['status']} "
                f"({summary['wall_time']:.0f}s){' err=' + summary['error'] if summary.get('error') else ''}"
            )

    return summaries
