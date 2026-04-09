"""Tau-bench wrapper — connects BaseHarnessAgent to tau2."""

from typing import cast

from tau2.agent.base_agent import ValidAgentInputMessage, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage, Message, MultiToolMessage, SystemMessage,
)
from tau2.utils.llm_utils import generate

from agent.agent import BaseHarnessAgent, HarnessState, AGENT_MODEL, SYSTEM_PROMPT

AGENT_INSTRUCTION = SYSTEM_PROMPT


class TauHarnessAgent(BaseHarnessAgent, LLMAgent):
    """Tau-bench specific wrapper around BaseHarnessAgent."""

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
        generate_kwargs = self.get_generate_kwargs()
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


__all__ = ["TauHarnessAgent"]