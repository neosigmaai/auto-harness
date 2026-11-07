"""Deterministic fake executor. No sandbox, no LLM.

Pass probability per task is derived from the public leaderboard average and improves with every
'# improvement' line the simulated optimizer adds, so the loop visibly accepts and rejects
candidates. Outcomes are deterministic in (agent sha, task, seed, attempt), which makes tests
reproducible while still letting confirmation re-runs disagree occasionally.
"""
from __future__ import annotations

import asyncio
import hashlib
import random

from ...config import settings
from ...tasks_catalog import BY_NAME
from .base import Executor, OnResult, RunContext, TaskResult

FAILURE_CLASSES = ["wrong_output", "gave_up_early", "timeout", "loop_detected", "tool_misuse", "env_setup_failure"]


def _pass_probability(task: str, agent_source: str) -> float:
    meta = BY_NAME.get(task, {})
    lb = (meta.get("leaderboard_solve_rate") or 50.0) / 100.0
    base = 0.25 + 0.55 * lb                       # canaries ~0.78, mid tasks ~0.55, hard ~0.4
    k = sum(1 for line in agent_source.splitlines() if line.startswith("# improvement"))
    return min(0.97, base + 0.06 * k)


class SimulatedExecutor:
    name = "simulated"

    def __init__(self) -> None:
        self._cancelled = False

    async def run_benchmark(self, ctx: RunContext, agent_source: str, tasks: list[str],
                            on_result: OnResult, on_start=None) -> dict[str, TaskResult]:
        sha = hashlib.sha256(agent_source.encode()).hexdigest()
        results: dict[str, TaskResult] = {}
        sem = asyncio.Semaphore(settings.max_concurrency_per_iteration)

        async def one(task: str) -> None:
            async with sem:
                if self._cancelled:
                    return
                if on_start:
                    await on_start(task)
                rng = random.Random(f"{sha}:{task}:{ctx.seed}:{ctx.evaluation_phase}:{ctx.repetition}:{ctx.attempt}")
                await asyncio.sleep(settings.simulated_task_delay_sec * rng.uniform(0.5, 1.5))
                p = _pass_probability(task, agent_source)
                passed = rng.random() < p
                steps = rng.randint(6, 40)
                dur = round(rng.uniform(40, 500), 1)
                if passed:
                    r = TaskResult(task, "passed", 1.0, None, dur, steps, "agent_declared_done",
                                   "============ 4 passed in 1.2s ============",
                                   _trace(task, steps, True), f"sim-{task[:8]}-{ctx.iteration_index}")
                else:
                    fc = rng.choices(FAILURE_CLASSES, weights=[5, 3, 3, 1, 1, 1])[0]
                    status = "timeout" if fc == "timeout" else "failed"
                    r = TaskResult(task, status, 0.0, fc, 900.0 if fc == "timeout" else dur, 80 if fc == "timeout" else steps,
                                   "agent_timeout" if fc == "timeout" else "agent_declared_done",
                                   f"FAILED test_outputs.py::test_result - AssertionError: expected output not found ({fc})",
                                   _trace(task, steps, False), f"sim-{task[:8]}-{ctx.iteration_index}")
                # Synthetic values exercise accounting; they are never evidence
                # of real model spend, runtime or benchmark improvement.
                r.input_tokens = steps * 200
                r.output_tokens = steps * 50
                r.model_cost_usd = r.cost_usd = round(steps * .0004, 6)
                r.cost_known = True
                r.metadata = {"simulated": True, "cost_basis": "synthetic_fixture"}
                results[task] = r
                await on_result(r)

        await asyncio.gather(*(one(t) for t in tasks))
        return results

    async def cancel(self) -> None:
        self._cancelled = True


def _trace(task: str, steps: int, passed: bool) -> list[dict]:
    msgs = [{"role": "system", "content": "You are an autonomous terminal agent."},
            {"role": "user", "content": f"Task:\n(simulated instruction for {task})"}]
    for i in range(min(steps, 6)):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "ls -la /app"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "total 0\n(simulated output)"})
    msgs.append({"role": "assistant", "content": "I have completed the task." if passed else "I believe the task is done."})
    return msgs
