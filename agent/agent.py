# HarnessAgent for Terminal-Bench 2.0 — starting template.
import json
import os
import re

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

MAX_STEPS = 80
MAX_OUTPUT_CHARS = 8000
MODEL = os.environ.get("AGENT_MODEL", "gpt-5.4")

AGENT_INSTRUCTION = """\
You are an autonomous terminal agent. You are given a task and a Linux container.
You solve tasks by executing bash commands. Work step by step.

Rules:
- Read the task carefully before acting.
- Explore the environment first to understand what you have.
- Check command output for errors before proceeding.
- Install missing dependencies as needed.
- Verify your solution before finishing.
- When you are done, send a final text message (no tool call) summarizing what you did.
"""

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


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text or ""
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - limit} chars truncated] ...\n\n"
        + text[-half:]
    )


_WRITE_FILE_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:
        cat\s+>\s*\S+ |
        cat\s+>>\s*\S+ |
        tee\s+(?:-[a-zA-Z]+\s+)*\S+ |
        printf\b[^\n;&|]*>\s*\S+ |
        echo\b[^\n;&|]*>\s*\S+ |
        install\s+[^\n;&|]*\s+\S+ |
        cp\s+[^\n;&|]*\s+\S+ |
        mv\s+[^\n;&|]*\s+\S+ |
        chmod\s+[^\n;&|]*\s+\S+
    )
    """,
    re.VERBOSE,
)

_PREVIEW_OUTPUT_RE = re.compile(
    r"""
    (?:
        \|\s*head\b |
        \|\s*tail\b |
        ^\s*head\b |
        ^\s*tail\b |
        ;\s*head\b |
        ;\s*tail\b |
        &&\s*head\b |
        &&\s*tail\b
    )
    """,
    re.VERBOSE,
)


def _looks_like_file_write(command: str) -> bool:
    """Best-effort detection of commands that likely changed a task deliverable."""
    return bool(_WRITE_FILE_RE.search(command or ""))


def _looks_like_truncated_preview(command: str) -> bool:
    """Detect commands that intentionally show only a small prefix/suffix of output."""
    return bool(_PREVIEW_OUTPUT_RE.search(command or ""))


class HarnessAgent(BaseAgent):
    """Agent under optimization for Terminal-Bench 2.0."""

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
        model = self.model_name or MODEL
        total_input_tokens = 0
        total_output_tokens = 0
        wrote_file = False
        last_bash_command = ""
        preview_final_guard_used = False

        messages = [
            {"role": "system", "content": AGENT_INSTRUCTION},
            {"role": "user", "content": f"Task:\n{instruction}"},
        ]

        for step in range(MAX_STEPS):
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

            if not message.tool_calls:
                if (
                    wrote_file
                    and not preview_final_guard_used
                    and _looks_like_truncated_preview(last_bash_command)
                ):
                    preview_final_guard_used = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before finalizing: your most recent command only displayed a truncated "
                            "preview using head/tail after you created or modified a file. Run one "
                            "non-preview check that exercises the requested deliverable/output enough "
                            "to catch format, count, range, or value errors, or briefly explain with "
                            "evidence why no further check is applicable."
                        ),
                    })
                    self.logger.info(
                        "Agent attempted to finish after truncated preview; requesting one fuller check"
                    )
                    continue

                self.logger.info(f"Agent declared complete at step {step}")
                break

            for tc in message.tool_calls:
                if tc.function.name != "bash":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.function.name}",
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments or "")
                    if not isinstance(args, dict):
                        raise TypeError("Arguments must be a dictionary")
                    command = args.get("command", "")
                    if not isinstance(command, str):
                        raise TypeError("Command must be a string")
                except (json.JSONDecodeError, TypeError):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Error: invalid JSON arguments",
                    })
                    continue

                last_bash_command = command
                if _looks_like_file_write(command):
                    wrote_file = True
                self.logger.info(f"Step {step} | bash: {command[:200]}")

                result = await environment.exec(command, timeout_sec=120)

                output_parts = []
                if result.stdout:
                    output_parts.append(result.stdout)
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                if result.return_code != 0:
                    output_parts.append(f"[exit code: {result.return_code}]")

                output = "\n".join(output_parts) if output_parts else "(no output)"
                output = _truncate(output)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        if os.environ.get("HARNESS_SAVE_TRACE", "1") == "1":
            trace_path = self.logs_dir / "trace.json"
            try:
                with open(trace_path, "w") as f:
                    json.dump(messages, f, indent=2, default=str)
                self.logger.info(f"Trace saved to {trace_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save trace: {e}")

        context.n_input_tokens = total_input_tokens
        context.n_output_tokens = total_output_tokens
