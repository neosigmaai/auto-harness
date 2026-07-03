from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SimulatedBenchmarkRunner:
    def run(self, task_ids: list[str]) -> dict[str, float | None]:
        results: dict[str, float | None] = {}
        for task_id in task_ids:
            lowered = task_id.lower()
            if "infra" in lowered or "timeout" in lowered:
                results[task_id] = None
            elif "fail" in lowered:
                results[task_id] = 0.0
            else:
                results[task_id] = 1.0
        return results


@dataclass(frozen=True)
class TerminalBenchRunnerAdapter:
    split: str = "train"
    last_artifacts: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        compare=False,
    )

    def run(
        self,
        task_ids: list[str],
        *,
        model: str,
        sandbox_provider: str,
        requested_concurrency: int,
        run_id: str,
        attempt: str = "baseline",
    ) -> dict[str, float | None]:
        _ensure_terminal_bench_agent()

        from benchmark import TerminalBenchRunner

        jobs_dir = (
            Path("workspace")
            / "service_runs"
            / run_id
            / "tbench_jobs"
            / _safe_attempt_name(attempt)
        )
        jobs_dir.mkdir(parents=True, exist_ok=True)
        runner = TerminalBenchRunner(
            agent_model=model,
            split=self.split,
            env_provider=sandbox_provider,
            n_concurrent=max(1, min(requested_concurrency, len(task_ids))),
            jobs_dir=str(jobs_dir),
            agent_import_path="agent.agent:HarnessAgent",
        )
        results = runner.run(task_ids=task_ids)
        object.__setattr__(
            self,
            "last_artifacts",
            dict(getattr(runner, "last_artifacts", {})),
        )
        return results


def _ensure_terminal_bench_agent() -> None:
    agent_path = Path("agent") / "agent.py"
    template_path = Path("agent") / "templates" / "terminal_bench.py"

    try:
        agent_text = agent_path.read_text()
    except FileNotFoundError:
        agent_text = ""

    if "class HarnessAgent" in agent_text:
        return

    if not template_path.exists():
        raise FileNotFoundError(
            "Terminal-Bench agent template not found at "
            f"{template_path}. Run from the auto-harness repository root."
        )

    agent_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, agent_path)


def _safe_attempt_name(attempt: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in attempt.strip()
    )
    return safe or "attempt"
