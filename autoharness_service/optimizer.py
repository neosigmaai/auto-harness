from __future__ import annotations

import json
import os
from dataclasses import dataclass

from autoharness_service.agent_patch import AgentPatchService
from autoharness_service.models import FailureSummary, TaskResultRecord


@dataclass(frozen=True)
class OptimizationProposal:
    hypothesis: str
    new_agent_instruction: str
    expected_effect: str
    risks: str


def parse_optimizer_json(text: str) -> OptimizationProposal:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Optimizer response must be a JSON object") from exc

    if not isinstance(payload, dict):
        raise ValueError("Optimizer response must be a JSON object")

    required_fields = {
        "hypothesis",
        "new_agent_instruction",
        "expected_effect",
        "risks",
    }
    if set(payload) != required_fields:
        raise ValueError(
            "Optimizer response must contain exactly hypothesis, "
            "new_agent_instruction, expected_effect, and risks"
        )
    values: dict[str, str] = {}
    for field_name in required_fields:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Optimizer response is missing {field_name}")
        values[field_name] = value.strip()

    return OptimizationProposal(**values)


def build_optimizer_prompt(
    task_results: list[TaskResultRecord],
    failure_summary: FailureSummary,
    *,
    current_instruction: str,
) -> str:
    failed_lines: list[str] = []
    for result in task_results:
        if result.status == "passed":
            continue
        artifact_paths = result.metadata.get("artifacts", [])
        artifact_summary = (
            ", ".join(str(path) for path in artifact_paths)
            if artifact_paths
            else "none"
        )
        failed_lines.append(
            f"- {result.task_id}: status={result.status}, "
            f"reward={result.reward}, failure_type={result.failure_type}, "
            f"error={result.error_summary}, "
            f"trace_path={result.trace_path or 'none'}, "
            f"result_path={result.result_path or 'none'}, "
            f"artifacts={artifact_summary}"
        )
    failures = "\n".join(failed_lines) if failed_lines else "- none"
    return (
        "You are improving a Terminal-Bench bash agent. "
        "Propose one focused replacement for the agent instruction only. "
        "Do not propose multiple candidates, broad rewrites, rankings, or merges.\n\n"
        f"Current instruction:\n{current_instruction}\n\n"
        f"Summary: passed={failure_summary.tasks_passed}, "
        f"failed={failure_summary.tasks_failed}, "
        f"infra_failed={failure_summary.tasks_infra_failed}, "
        f"failure_modes={failure_summary.top_failure_modes}\n\n"
        f"Failed tasks:\n{failures}\n\n"
        "Return JSON only as a single object with exactly these string fields: "
        "hypothesis, new_agent_instruction, expected_effect, risks. "
        "The new_agent_instruction must be one full replacement instruction. "
        "Use no Markdown."
    )


class Optimizer:
    def propose_instruction_patch(
        self,
        task_results: list[TaskResultRecord],
        failure_summary: FailureSummary,
        *,
        model: str,
        current_instruction: str,
    ) -> OptimizationProposal:
        prompt = build_optimizer_prompt(
            task_results,
            failure_summary,
            current_instruction=current_instruction,
        )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You propose concise instruction-only improvements for coding "
                        "agents and must answer with one JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return parse_optimizer_json(response.output_text)

    def propose(
        self,
        task_results: list[TaskResultRecord],
        failure_summary: FailureSummary,
        *,
        model: str,
    ) -> str:
        current_instruction = AgentPatchService().read_instruction()
        proposal = self.propose_instruction_patch(
            task_results,
            failure_summary,
            model=model,
            current_instruction=current_instruction,
        )
        return json.dumps(proposal.__dict__)
