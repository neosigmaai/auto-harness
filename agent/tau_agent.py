"""Tau-bench adapter: LLMAgent + tau2 message loop."""

from __future__ import annotations

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

from agent.core import (
    AGENT_INSTRUCTION,
    AGENT_MODEL,
    build_system_prompt,
    reasoning_effort_kwargs,
)


@dataclass
class HarnessState:
    messages: list[Message] = field(default_factory=list)


class HarnessAgent(LLMAgent):
    """Agent under optimization (tau-bench)."""

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(AGENT_INSTRUCTION, self.domain_policy)

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
        generate_kwargs = reasoning_effort_kwargs()
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
