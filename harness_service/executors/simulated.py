"""Deterministic dummy executor (M1 default).

No external dependencies — lets the whole service + test_client.py run anywhere,
instantly. Rewards are seeded by ``hash(agent_hash + task_id)`` so:
  * the same agent always yields the same result (reproducibility), and
  * edits to the agent source can move the score — which lets the M4 loop
    demonstrably improve, without any real sandbox.

A light heuristic rewards agent sources that contain known-good techniques
(a "plan"/TODO discipline, verification, environment exploration) so simulated
optimization mirrors the real levers documented in program_templates/.
"""

from __future__ import annotations

import asyncio
import hashlib

from harness_service.constants import ExecutorKind, TRACE_EXCERPT_CHARS
from harness_service.domain import AgentState, BenchmarkResult, TaskOutcome

# Substrings that (heuristically) make an agent better, and their score weight.
_TECHNIQUE_BONUS = {
    "plan": 0.10,
    "todo": 0.08,
    "verify": 0.07,
    "explore": 0.05,
    "check the output": 0.04,
    "step by step": 0.03,
}
_MAX_BONUS = 0.35


def _unit_hash(*parts: str) -> float:
    """Deterministic float in [0, 1) from the given parts."""
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _technique_bonus(source: str) -> float:
    src = source.lower()
    bonus = sum(w for kw, w in _TECHNIQUE_BONUS.items() if kw in src)
    return min(bonus, _MAX_BONUS)


class SimulatedExecutor:
    kind = ExecutorKind.SIMULATED

    def __init__(self, base_difficulty: float = 0.45):
        # Higher base_difficulty → harder tasks (lower baseline pass rate).
        self._base_difficulty = base_difficulty

    async def run(self, agent: AgentState, task_ids: list[str]) -> BenchmarkResult:
        await asyncio.sleep(0)  # cooperative — behaves like the async I/O path
        bonus = _technique_bonus(agent.source)
        outcomes: list[TaskOutcome] = []
        for task_id in task_ids:
            # Per-task intrinsic difficulty, stable across runs.
            difficulty = self._base_difficulty + 0.5 * (_unit_hash("difficulty", task_id) - 0.5)
            # An agent "passes" when its (heuristic) competence clears the difficulty.
            competence = _unit_hash(agent.content_hash, task_id) * 0.5 + bonus
            passed = competence >= difficulty
            reward = 1.0 if passed else 0.0
            duration = round(2.0 + 8.0 * _unit_hash("dur", agent.content_hash, task_id), 2)
            outcome = TaskOutcome(
                task_id=task_id,
                reward=reward,
                passed=passed,
                duration_s=duration,
                trace_excerpt=self._fake_trace(task_id, passed),
                failure_reason=None if passed else self._fake_failure(task_id),
            )
            outcomes.append(outcome)
        return BenchmarkResult(outcomes=tuple(outcomes))

    @staticmethod
    def _fake_trace(task_id: str, passed: bool) -> str:
        verdict = "solution verified, exit 0" if passed else "tests failed, exit 1"
        text = (
            f"$ echo 'starting {task_id}'\n"
            f"$ ls -la\n"
            f"$ # ... agent explored the environment ...\n"
            f"$ ./run_tests.sh\n"
            f"{verdict}\n"
        )
        return text[:TRACE_EXCERPT_CHARS]

    @staticmethod
    def _fake_failure(task_id: str) -> str:
        reasons = [
            "agent did not verify its solution before finishing",
            "gave up after a single failed command",
            "misread the task and edited the wrong file",
            "did not explore the environment before acting",
            "got stuck retrying the same failing command",
        ]
        idx = int(_unit_hash("reason", task_id) * len(reasons))
        return reasons[min(idx, len(reasons) - 1)]
