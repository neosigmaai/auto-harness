from __future__ import annotations

import os

from autoharness_service.models import FailureSummary, TaskResultRecord


def build_optimizer_prompt(
    task_results: list[TaskResultRecord],
    failure_summary: FailureSummary,
) -> str:
    failed_lines: list[str] = []
    for result in task_results:
        if result.status == "passed":
            continue
        failed_lines.append(
            f"- {result.task_id}: status={result.status}, "
            f"reward={result.reward}, failure_type={result.failure_type}, "
            f"error={result.error_summary}"
        )
    failures = "\n".join(failed_lines) if failed_lines else "- none"
    return (
        "You are improving a Terminal-Bench bash agent. "
        "Propose one focused improvement to the agent prompt or behavior. "
        "Do not propose multiple candidates or broad rewrites.\n\n"
        f"Summary: passed={failure_summary.tasks_passed}, "
        f"failed={failure_summary.tasks_failed}, "
        f"infra_failed={failure_summary.tasks_infra_failed}, "
        f"failure_modes={failure_summary.top_failure_modes}\n\n"
        f"Failed tasks:\n{failures}\n\n"
        "Return four short sections: hypothesis, proposed_change, expected_effect, risks."
    )


class Optimizer:
    def propose(
        self,
        task_results: list[TaskResultRecord],
        failure_summary: FailureSummary,
        *,
        model: str,
    ) -> str:
        prompt = build_optimizer_prompt(task_results, failure_summary)
        if not os.getenv("OPENAI_API_KEY"):
            return (
                "LLM proposal skipped because OPENAI_API_KEY is not set. "
                "Benchmark results were still recorded; set OPENAI_API_KEY to enable proposals."
            )

        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You propose concise improvements for coding agents.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text
