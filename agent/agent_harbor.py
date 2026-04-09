"""Harbor/TerminalBench wrapper — connects BaseHarnessAgent to Harbor."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agents import Agent, Runner, function_tool, MessageOutputItem, ReasoningItem, ToolCallItem, ToolCallOutputItem, ItemHelpers
from agents.extensions.models.litellm_model import LitellmModel
from agents.usage import Usage
from agent.agent import AGENT_MODEL, MAX_TURNS, SYSTEM_PROMPT
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.terminal.tmux_session import TmuxSession

def create_tools(session: TmuxSession) -> list:
    """TerminalBench tools — shell execution via tmux."""

    @function_tool
    async def run_shell(command: str) -> str:
        """Run a shell command in the terminal. Returns output."""
        try:
            session.send_keys([command, "Enter"], block=True, max_timeout_sec=120)
            output = session.capture_pane()
            return output or "(no output)"
        except Exception as exc:
            return f"ERROR: {exc}"

    return [run_shell]


async def run_task(session: TmuxSession, instruction: str) -> tuple[object, int]:
    """Run the harbor agent on a task using tmux session."""
    tools = create_tools(session)
    agent = Agent(
        name="harness-agent",
        instructions=SYSTEM_PROMPT,
        tools=tools,
        model=LitellmModel(model=AGENT_MODEL),
    )
    t0 = time.time()
    result = await Runner.run(agent, input=instruction, max_turns=MAX_TURNS)
    duration_ms = int((time.time() - t0) * 1000)
    return result, duration_ms


# ── fixed Harbor adapter boundary ────────────────────────────────

class HarborAgent(BaseAgent):
    """TerminalBench entry point."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def version(self) -> str | None:
        return "0.1.0"

    def perform_task(self, instruction: str, session: TmuxSession,logging_dir: Path | None = None,**kwargs,) -> AgentResult:
        import asyncio
        result, duration_ms = asyncio.run(run_task(session, instruction))

        usage = Usage()
        for response in result.raw_responses:
            usage.add(response.usage)
        print(
            f"turns={len(result.raw_responses)} duration_ms={duration_ms} "
            f"input={usage.input_tokens} output={usage.output_tokens}"
        )

        return AgentResult(success=True)


def _to_atif(result: object, model: str, duration_ms: int = 0) -> dict:
    steps: list[dict] = []
    step_id = 0
    now = datetime.now(timezone.utc).isoformat()

    def _step(source: str, message: str, **extra: object) -> dict:
        nonlocal step_id
        step_id += 1
        step = {"step_id": step_id, "timestamp": now, "source": source, "message": message}
        step.update({k: v for k, v in extra.items() if v is not None})
        return step

    pending_tool_call = None
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                steps.append(_step("agent", text, model_name=model))
        elif isinstance(item, ReasoningItem):
            summaries = getattr(item.raw_item, "summary", None)
            reasoning = "\n".join(s.text for s in summaries if hasattr(s, "text")) if summaries else None
            if reasoning:
                steps.append(_step("agent", "(thinking)", reasoning_content=reasoning, model_name=model))
        elif isinstance(item, ToolCallItem):
            raw = item.raw_item
            if hasattr(raw, "name"):
                pending_tool_call = raw
        elif isinstance(item, ToolCallOutputItem) and pending_tool_call:
            arguments = (
                json.loads(pending_tool_call.arguments)
                if isinstance(pending_tool_call.arguments, str)
                else pending_tool_call.arguments
            )
            output_str = str(item.output) if item.output else ""
            steps.append(_step(
                "agent", f"Tool: {pending_tool_call.name}",
                tool_calls=[{"tool_call_id": pending_tool_call.call_id,
                              "function_name": pending_tool_call.name,
                              "arguments": arguments}],
                observation={"results": [{"source_call_id": pending_tool_call.call_id,
                                          "content": output_str}]},
            ))
            pending_tool_call = None

    if pending_tool_call:
        arguments = (
            json.loads(pending_tool_call.arguments)
            if isinstance(pending_tool_call.arguments, str)
            else pending_tool_call.arguments
        )
        steps.append(_step(
            "agent", f"Tool: {pending_tool_call.name}",
            tool_calls=[{"tool_call_id": pending_tool_call.call_id,
                          "function_name": pending_tool_call.name,
                          "arguments": arguments}],
        ))

    if not steps:
        steps.append(_step("user", "(empty)"))

    usage = Usage()
    for response in result.raw_responses:
        usage.add(response.usage)

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": getattr(result, "last_response_id", None) or "unknown",
        "agent": {"name": "harness-agent", "version": "0.1.0", "model_name": model},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": usage.input_tokens,
            "total_completion_tokens": usage.output_tokens,
            "total_cached_tokens": getattr(usage.input_tokens_details, "cached_tokens", 0) or 0,
            "total_cost_usd": None,
            "total_steps": len(steps),
            "extra": {"duration_ms": duration_ms, "num_turns": len(result.raw_responses)},
        },
    }


__all__ = ["HarborAgent"]