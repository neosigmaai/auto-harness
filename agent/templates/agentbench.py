"""
AgentBench OS HarnessAgent, Phase 1 generalist baseline.

The optimization loop may edit anything in this file.
"""
from __future__ import annotations

import os

from openai import OpenAI

SYSTEM_PROMPT = """You are an expert Linux agent solving OS-level tasks in an Ubuntu container.

RULES:
1. Read the instruction carefully. Identify exactly what value is being requested.
2. Run the minimum commands needed. Do not explore unnecessarily.
3. If a command returns empty output or an error, reassess and try a different approach.
   Do not repeat similar commands. Do not loop.
4. If you already have high confidence in the answer, output ANSWER immediately
   instead of running more commands.
5. Before outputting ANSWER, verify it exactly matches the expected format:
   - Single value only, no extra text, no trailing whitespace
   - If the task asks for a number, output only the number
6. Output your final answer as: ANSWER: <value>

Think step by step. Use standard Unix tools. Never guess."""


class HarnessAgent:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o")
        self.client = OpenAI()

    def step(self, instruction: str, history: list[dict], container: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not history:
            messages.append({"role": "user", "content": instruction})
        else:
            messages.append({"role": "user", "content": instruction})
            messages.extend(history)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return resp.choices[0].message.content
