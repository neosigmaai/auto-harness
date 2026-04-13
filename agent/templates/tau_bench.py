# HarnessAgent for tau-bench — starting template.
import os
from dataclasses import dataclass, field
from typing import cast

from tau2.agent.base_agent import ValidAgentInputMessage, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.utils.llm_utils import generate

AGENT_MODEL: str = os.environ.get("AGENT_MODEL", "")

AGENT_INSTRUCTION = """
You are a helpful assistant that completes tasks according to the <policy> provided below.
""".strip()


@dataclass
class HarnessState:
    messages: list[Message] = field(default_factory=list)


class HarnessAgent(LLMAgent):
    """Agent under optimization."""

    @property
    def system_prompt(self) -> str:
        if self.domain_policy:
            return (
                "<instructions>\n"
                f"{AGENT_INSTRUCTION}\n"
                "</instructions>\n"
                "<policy>\n"
                f"{self.domain_policy}\n"
                "</policy>"
            )
        return AGENT_INSTRUCTION

    def get_init_state(
        self, message_history: list[Message] | None = None
    ) -> HarnessState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return HarnessState(messages=list(message_history))

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: HarnessState,
    ) -> tuple[AssistantMessage, HarnessState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        system = SystemMessage(role="system", content=self.system_prompt)
        reasoning_effort = os.environ.get("AGENT_REASONING_EFFORT", "")
        generate_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        generate_kwargs.update(self.llm_args)
        response = cast(
            AssistantMessage,
            generate(
                model=self.llm or AGENT_MODEL,
                tools=self.tools,
                messages=[system, *state.messages],
                **generate_kwargs,
            ),
        )
        state.messages.append(response)
        return response, state
