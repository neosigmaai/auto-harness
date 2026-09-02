# Spec-driven HarnessAgent for Terminal-Bench 2.0 — the service's agent runtime.
#
# Reads its system prompt and limits from the AgentSpec JSON named by
# $HARNESS_AGENT_SPEC, falling back to the template defaults when that is unset or
# unreadable (so it is runnable standalone). Fixed bash tool, same trace saving,
# same token accounting as agent/templates/terminal_bench.py.
#
# This file is NOT agent/agent.py: that file is what the Layer A coding agent edits
# and what prepare.py overwrites from templates. The service never touches it.
# Dependencies are stdlib + litellm + harbor only — no api.* imports, because harbor
# spawns the agent with just the repo root on PYTHONPATH.
import json
import os

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.spec_loader import load_spec_from_env

MODEL = os.environ.get("AGENT_MODEL", "gpt-5.4")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the container. Returns stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _truncate(text: str, limit: int) -> str:
    """Truncate long output, keeping the beginning and end."""
    if not text or len(text) <= limit:
        return text or ""
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - limit} chars truncated] ...\n\n"
        + text[-half:]
    )


class HarnessAgent(BaseAgent):
    """Agent under optimization, configured entirely by its AgentSpec."""

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        spec = load_spec_from_env()
        system_prompt = spec["system_prompt"]
        max_steps = spec["max_steps"]
        max_output_chars = spec["max_output_chars"]
        exec_timeout_sec = spec["exec_timeout_sec"]

        model = self.model_name or spec["agent_model"] or MODEL
        self.logger.info(
            f"spec: model={model} max_steps={max_steps} "
            f"max_output_chars={max_output_chars} exec_timeout_sec={exec_timeout_sec} "
            f"prompt_chars={len(system_prompt)}"
        )

        total_input_tokens = 0
        total_output_tokens = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task:\n{instruction}"},
        ]

        for step in range(max_steps):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                self.logger.error(f"LLM call failed at step {step}: {e}")
                break

            usage = response.usage
            if usage:
                total_input_tokens += usage.prompt_tokens or 0
                total_output_tokens += usage.completion_tokens or 0

            choice = response.choices[0]
            message = choice.message

            # Build the assistant message for history
            assistant_msg = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # If the model returned text without tool calls → task complete
            if not message.tool_calls:
                self.logger.info(f"Agent declared complete at step {step}")
                break

            # Execute each tool call
            for tc in message.tool_calls:
                if tc.function.name != "bash":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.function.name}",
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Error: invalid JSON arguments",
                    })
                    continue

                command = args.get("command", "")
                self.logger.info(f"Step {step} | bash: {command[:200]}")

                result = await environment.exec(command, timeout_sec=exec_timeout_sec)

                output_parts = []
                if result.stdout:
                    output_parts.append(result.stdout)
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                if result.return_code != 0:
                    output_parts.append(f"[exit code: {result.return_code}]")

                output = "\n".join(output_parts) if output_parts else "(no output)"
                output = _truncate(output, max_output_chars)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        # Save full conversation trace for failure analysis (the improver reads these)
        if os.environ.get("HARNESS_SAVE_TRACE", "1") == "1":
            trace_path = self.logs_dir / "trace.json"
            try:
                with open(trace_path, "w") as f:
                    json.dump(messages, f, indent=2, default=str)
                self.logger.info(f"Trace saved to {trace_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save trace: {e}")

        # Populate context
        context.n_input_tokens = total_input_tokens
        context.n_output_tokens = total_output_tokens
