# HarnessHandler for BFCL — starting template.
#
# `prepare.py` copies this file to `agent/agent.py`. The coding agent owns the
# copy and edits it freely. The benchmark runner imports `HarnessHandler` from
# `agent.agent` so any change here is picked up automatically.

from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler

AGENT_INSTRUCTION = """\
You are an autonomous function-calling agent.

Choose the minimal correct function call sequence for each user request. In
multi-turn tasks, preserve state across turns and use tool execution results to
decide whether more calls are needed.
"""


class HarnessHandler(OpenAIResponsesHandler):
    """BFCL handler under optimization."""

    def add_first_turn_message_FC(
        self,
        inference_data: dict,
        first_turn_message: list[dict],
    ) -> dict:
        # OpenAI Responses uses `developer` for harness-level instructions. The
        # injected message is appended directly so it bypasses BFCL's
        # `_substitute_prompt_role` (which only rewrites system->developer for
        # `test_entry["question"]`).
        already_injected = any(
            isinstance(m, dict)
            and m.get("role") == "developer"
            and m.get("content") == AGENT_INSTRUCTION
            for m in inference_data.get("message", [])
        )
        if AGENT_INSTRUCTION.strip() and not already_injected:
            inference_data.setdefault("message", []).append(
                {"role": "developer", "content": AGENT_INSTRUCTION}
            )
        return super().add_first_turn_message_FC(inference_data, first_turn_message)

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """Visible hook for tool schema/description rewriting."""
        return super()._compile_tools(inference_data, test_entry)

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """Visible hook for post-processing tool output before the next step."""
        return super()._add_execution_results_FC(
            inference_data,
            execution_results,
            model_response_data,
        )
