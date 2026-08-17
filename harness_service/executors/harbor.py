"""HarborExecutor — runs a candidate agent in a real E2B sandbox (M3).

Boundary: the agent runs inside harbor's sandbox subprocess (E2B cloud container),
never in the API or worker process. We shell out to the `harbor` CLI, point it at a
per-job candidate agent module, and parse per-task rewards + trace excerpts from the
job output directory.

Isolation (PLAN.md §4b): each candidate agent source is written to a UNIQUE module
under ``agent/_candidates/`` and referenced via ``-a agent._candidates.<mod>:HarnessAgent``
— so concurrent jobs never clobber each other or the tracked ``agent/agent.py``, and no
change to ``benchmark.py`` is needed (we read the per-job jobs-dir ourselves).

harbor 0.21.0 note: the agent import flag is ``-a/--agent`` (the repo's benchmark.py uses
the older ``--agent-import-path``, which this version renamed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path

from harness_service.agent_source import REPO_ROOT
from harness_service.config import Settings
from harness_service.constants import TRACE_EXCERPT_CHARS, ExecutorKind
from harness_service.domain import AgentState, BenchmarkResult, TaskOutcome

logger = logging.getLogger("harness.harbor")

CANDIDATES_DIR = REPO_ROOT / "agent" / "_candidates"


class HarborExecutor:
    kind = ExecutorKind.HARBOR

    def __init__(self, settings: Settings):
        self._s = settings

    async def run(self, agent: AgentState, task_ids: list[str]) -> BenchmarkResult:
        # Blocking harbor subprocess → off the event loop.
        return await asyncio.to_thread(self._run_sync, agent, task_ids)

    # ── internals ──
    def _write_candidate(self, agent: AgentState) -> tuple[str, Path]:
        CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        (CANDIDATES_DIR / "__init__.py").touch(exist_ok=True)
        mod = f"job_{agent.content_hash[:12]}_{int(time.time() * 1000)}"
        path = CANDIDATES_DIR / f"{mod}.py"
        path.write_text(agent.source, encoding="utf-8")
        import_path = f"agent._candidates.{mod}:HarnessAgent"
        return import_path, path

    def _build_env(self, agent: AgentState) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["AGENT_MODEL"] = agent.model
        if agent.reasoning_effort:
            env["AGENT_REASONING_EFFORT"] = agent.reasoning_effort
        # Ensure the sandbox + LLM credentials are present for the subprocess.
        if self._s.e2b_api_key and not env.get("E2B_API_KEY"):
            env["E2B_API_KEY"] = self._s.e2b_api_key
        if self._s.openai_api_key and not env.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = self._s.openai_api_key
        return env

    def _run_sync(self, agent: AgentState, task_ids: list[str]) -> BenchmarkResult:
        import_path, cand_path = self._write_candidate(agent)
        jobs_dir = REPO_ROOT / "workspace" / "harbor_jobs" / cand_path.stem
        jobs_dir.mkdir(parents=True, exist_ok=True)

        n = min(self._s.harbor_concurrency, len(task_ids))
        timeout_mult = self._s.harbor_per_task_timeout_s / 180.0  # harbor default 180s
        cmd = [
            self._s.harbor_bin, "run",
            "-d", self._s.harbor_dataset,
            "-a", import_path,
            "-m", agent.model,
            "-e", self._s.env_provider,
            "-o", str(jobs_dir),
            "--agent-timeout-multiplier", f"{timeout_mult:.2f}",
            "-n", str(n),
            "-y",
        ]
        for tid in task_ids:
            cmd += ["-i", tid]

        n_batches = math.ceil(len(task_ids) / max(n, 1))
        subprocess_timeout = self._s.harbor_per_task_timeout_s * n_batches + 600
        logger.info("harbor run: %d tasks, n=%d, timeout=%ds, agent=%s",
                    len(task_ids), n, subprocess_timeout, import_path)

        run_start = time.time()
        try:
            proc = subprocess.run(
                cmd, env=self._build_env(agent), capture_output=True, text=True,
                timeout=subprocess_timeout, cwd=str(REPO_ROOT),
            )
            if proc.returncode != 0:
                logger.warning("harbor exited %d; stderr tail: %s",
                               proc.returncode, (proc.stderr or "")[-500:])
        except subprocess.TimeoutExpired:
            logger.warning("harbor run timed out after %ds", subprocess_timeout)
        finally:
            cand_path.unlink(missing_ok=True)  # candidate module is transient

        outcomes = self._parse_results(jobs_dir, task_ids, run_start)
        return BenchmarkResult(outcomes=tuple(outcomes))

    def _parse_results(
        self, jobs_dir: Path, task_ids: list[str], run_start: float
    ) -> list[TaskOutcome]:
        """Parse per-task result.json + trace from the newest job dir under jobs_dir."""
        rewards: dict[str, float | None] = {}
        traces: dict[str, str] = {}

        # result.json files created by this run (defensive against layout changes).
        for result_file in jobs_dir.rglob("result.json"):
            if result_file.stat().st_mtime < run_start - 1:
                continue
            try:
                data = json.loads(result_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            task_name = data.get("task_name") or result_file.parent.name.rsplit("__", 1)[0]
            vr = data.get("verifier_result")
            if isinstance(vr, dict):
                r = vr.get("rewards", {})
                reward = float(r.get("reward", 0.0)) if isinstance(r, dict) else 0.0
            else:
                reward = None  # verifier didn't run → infra error/timeout
            rewards[task_name] = reward
            trace = self._find_trace(result_file.parent)
            if trace:
                traces[task_name] = trace

        outcomes: list[TaskOutcome] = []
        for tid in task_ids:
            reward = rewards.get(tid)  # missing → None (didn't produce a result)
            passed = reward is not None and reward >= 1.0
            outcomes.append(TaskOutcome(
                task_id=tid,
                reward=reward,
                passed=passed,
                trace_excerpt=traces.get(tid),
                failure_reason=None if passed else self._failure_reason(reward, traces.get(tid)),
            ))
        return outcomes

    @staticmethod
    def _find_trace(trial_dir: Path) -> str | None:
        for candidate in (trial_dir / "agent" / "trace.json", trial_dir / "trace.json"):
            if candidate.exists():
                try:
                    return candidate.read_text()[-TRACE_EXCERPT_CHARS:]
                except OSError:
                    return None
        return None

    @staticmethod
    def _failure_reason(reward: float | None, trace: str | None) -> str:
        if reward is None:
            return "no verifier result (agent timed out or sandbox/infra error)"
        tail = (trace or "").strip()[-300:]
        return f"reward={reward:.2f}; trace tail: {tail}" if tail else f"reward={reward:.2f}"
