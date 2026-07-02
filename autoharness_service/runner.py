from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark import TerminalBenchRunner


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

    def run(
        self,
        task_ids: list[str],
        *,
        model: str,
        sandbox_provider: str,
        requested_concurrency: int,
        run_id: str,
    ) -> dict[str, float | None]:
        jobs_dir = Path("workspace") / "service_runs" / run_id / "tbench_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        runner = TerminalBenchRunner(
            agent_model=model,
            split=self.split,
            env_provider=sandbox_provider,
            n_concurrent=max(1, min(requested_concurrency, len(task_ids))),
            jobs_dir=str(jobs_dir),
        )
        return runner.run(task_ids=task_ids)
