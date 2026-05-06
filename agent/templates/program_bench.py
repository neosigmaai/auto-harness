# HarnessAgent for ProgramBench — starting template.
#
# At task start the harness moves the cleanroom's /workspace/executable to
# /opt/orig/executable and runs bash commands as an unprivileged user, so the
# original can be executed for behavioral comparison but not read or copied.
# /workspace then contains only upstream metadata (verified cmatrix + zoxide):
#   /workspace/README.md   — original repo's README
#   /workspace/<bin>.<n>   — man pages, when the upstream ships them
#   /workspace/LICENSE, /workspace/.git/  — repo metadata (often present)
#
# /opt/orig/executable     — original binary, execute-only behavioral reference
#
# Submission contract: at done, /workspace must contain everything the eval
# image needs to rebuild — sources plus a working `compile.sh` at /workspace/
# that produces an executable named `executable`. The eval image wipes
# /workspace, restores your tar, runs `compile.sh`, then runs ~248k behavioral
# tests against the resulting binary.
import json
import os

import litellm

from agent.helpers.program_bench.bash_tool import BashTool

MAX_STEPS = 100
MAX_OUTPUT_CHARS = 8000
COMMAND_TIMEOUT = 300
MODEL = os.environ.get("AGENT_MODEL", "gpt-5.4")

AGENT_INSTRUCTION = """\
You are reverse-engineering a binary into a working codebase from scratch.

The container layout (verified across multiple tasks):
- /opt/orig/executable     — the original binary, execute-only
- /workspace/README.md      — the upstream repo's README
- /workspace/<name>.<n>     — man page(s), when shipped (sections 1, 5, etc.)
- /workspace/LICENSE        — license file
- Build tools available:    gcc, make, cmake, go, cargo, rustc, python3, javac

You operate ONLY through the bash tool — no internet, no decompiler.

To finish: ensure /workspace has the source tree AND a working
/workspace/compile.sh that builds an executable at /workspace/executable.
Eval wipes /workspace,
restores your tar, runs compile.sh, runs behavioral tests against your
output.

Process:
1. Explore. Run /opt/orig/executable with --help, --version, common
   subcommands. Read /workspace/README.md and any man pages.
2. Plan an architecture before writing code.
3. Build incrementally — write compile.sh early, run it, fix errors.
4. Verify by running your build's output against /opt/orig/executable on
   the same inputs and diffing.
5. When done, send a final assistant message (no tool call) summarizing
   what you built and what you tested.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container. Returns combined stdout+stderr and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout": {"type": "integer", "description": "Optional timeout in seconds (default 300)."},
                },
                "required": ["command"],
            },
        },
    }
]

BOOTSTRAP_COMMANDS = [
    "uname -a; cat /etc/os-release 2>/dev/null | head -3",
    "ls -la /workspace",
    "find /workspace -maxdepth 3 -type f ! -path '*/\\.git/*' | head -40",
    "/opt/orig/executable --help 2>&1 | head -60; echo '----'; /opt/orig/executable --version 2>&1 | head -3",
    "head -200 /workspace/README.md 2>/dev/null",
    "for f in /workspace/*.[1-9]; do [ -e \"$f\" ] && echo \"=== $f ===\" && head -80 \"$f\"; done 2>/dev/null",
    "command -v gcc make cmake go cargo rustc python3 javac 2>&1",
]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text or ""
    half = limit // 2
    return text[:half] + f"\n\n... [{len(text) - limit} chars truncated] ...\n\n" + text[-half:]


def _bootstrap_context(bash: BashTool) -> str:
    """Run a fixed set of discovery commands; return findings as a string."""
    parts = []
    for cmd in BOOTSTRAP_COMMANDS:
        out, rc = bash(cmd, timeout=60)
        parts.append(f"$ {cmd}\n{_truncate(out, 1500)}\n[exit {rc}]")
    return "\n\n".join(parts)


class HarnessAgent:
    """Agent under optimization for ProgramBench."""

    def __init__(self, env, instance: dict, model_name: str | None = None, reasoning_effort: str | None = None):
        self.env = env
        self.instance = instance
        self.model_name = model_name or MODEL
        self.reasoning_effort = reasoning_effort

    async def run(self) -> dict:
        bash = BashTool(self.env, default_timeout=COMMAND_TIMEOUT)
        bootstrap = _bootstrap_context(bash)

        task_card = (
            f"instance_id: {self.instance.get('instance_id')}\n"
            f"repository:  {self.instance.get('repository')}\n"
            f"language:    {self.instance.get('language')}\n"
            f"difficulty:  {self.instance.get('difficulty')}"
        )

        messages = [
            {"role": "system", "content": AGENT_INSTRUCTION},
            {
                "role": "user",
                "content": f"Task:\n{task_card}\n\nInitial environment scan:\n{bootstrap}",
            },
        ]
        total_input = 0
        total_output = 0

        for step in range(MAX_STEPS):
            kwargs: dict = {"model": self.model_name, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                messages.append({"role": "system", "content": f"[harness] LLM call failed at step {step}: {e}"})
                break

            usage = getattr(response, "usage", None)
            if usage:
                total_input += getattr(usage, "prompt_tokens", 0) or 0
                total_output += getattr(usage, "completion_tokens", 0) or 0

            message = response.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if not message.tool_calls:
                break

            budget_exhausted = False
            for tc in message.tool_calls:
                if tc.function.name != "bash":
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Unknown tool: {tc.function.name}"})
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Error: invalid JSON arguments"})
                    continue
                command = str(args.get("command", ""))
                try:
                    requested_timeout = int(args.get("timeout") or COMMAND_TIMEOUT)
                except (TypeError, ValueError):
                    requested_timeout = COMMAND_TIMEOUT
                cmd_timeout = bash.clamp_timeout(requested_timeout)
                if cmd_timeout <= 0:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Error: task budget exhausted"})
                    budget_exhausted = True
                    continue
                output, rc = bash(command, timeout=cmd_timeout)
                tool_content = _truncate(output) + (f"\n[exit {rc}]" if rc != 0 else "")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content or "(no output)"})
            if budget_exhausted:
                break

        return {
            "messages": messages,
            "steps": step + 1,
            "tokens": {"input": total_input, "output": total_output},
            "instance_id": self.instance.get("instance_id"),
        }
