
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

# score at each iteration; None means "infra ate the run" — every task returns None
TRAJECTORY: list[float | None] = [0.40, 0.60, 0.53, None, 0.60, 0.60]


@dataclass
class RunResult:
    """The one contract the worker knows. E2BRunner returns the same shape."""
    results: dict[str, float | None]
    failures: list[dict[str, Any]]
    usage: dict[str, int]
    error_detail: str | None = None


@dataclass
class MockRunner:
    """`visible` is mock-only: the trajectory has to land on the number the ratchet reads,
    or the scripted score drop gets averaged away and never tests anything."""
    job_id: str
    visible: list[str]
    calls: int = field(default=0)

    def run(self, task_ids: list[str], agent_source: str) -> RunResult:
        time.sleep(0.4)   # so test_client actually polls at least once
        score = TRAJECTORY[min(self.calls, len(TRAJECTORY) - 1)]
        self.calls += 1

        if score is None:
            return RunResult(
                results={t: None for t in task_ids}, failures=[],
                usage={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                       "sandboxes_used": 1, "sandbox_seconds": 12},
                error_detail="mock: sandbox died before the verifier ran")

        # visible and holdout filled independently, so visible tracks the trajectory
        results: dict[str, float | None] = {}
        for group, rate in ((self.visible, score),
                            ([t for t in task_ids if t not in self.visible], score * 0.8)):
            order = sorted(group)
            random.Random(f"{self.job_id}{len(group)}").shuffle(order)
            n_pass = round(rate * len(order))
            results |= {t: (1.0 if i < n_pass else 0.0) for i, t in enumerate(order)}
        failures = [_failure(t) for t, r in results.items() if r == 0.0]
        return RunResult(
            results=results, failures=failures,
            usage={"llm_calls": 4 * len(task_ids), "input_tokens": 9000 * len(task_ids),
                   "output_tokens": 800 * len(task_ids),
                   "sandboxes_used": 1 + len(task_ids), "sandbox_seconds": 90})


def _failure(task_id: str) -> dict[str, Any]:
    """Same shape the real distiller produces."""
    return {
        "task_id": task_id,
        "reward": 0.0,
        "tool_calls": 3,         
        "failing_commands": [
            {"command": "python solve.py", "exit_code": 1,
             "stderr": "Traceback (most recent call last): ... KeyError: 'config'"}],
        "tail": "assistant: I'll try running the script directly.\n"
                "tool: [exit code: 1]\nassistant: Let me check the config file.",
        "verifier_output": "FAILED tests/test_task.py::test_output - assert 0 == 1",
    }


@dataclass
class MockOptimizer:
    """Deterministic stand-in for the LLM."""
    calls: int = field(default=0)

    def propose(self, agent_source: str, failures: list[dict[str, Any]],
                ledger: list[str]) -> tuple[str, str, dict[str, int]]:
        self.calls += 1
        proposal = (f"mock proposal #{self.calls}: enforce a TODO plan before acting — "
                    f"{len(failures)} failing tasks showed the agent acting before "
                    f"exploring")
        source = f"# mock revision {self.calls}\n{agent_source}"
        return proposal, source, {"llm_calls": 1, "input_tokens": 3200,
                                  "output_tokens": 1400}
