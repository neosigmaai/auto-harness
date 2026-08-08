from __future__ import annotations

import ast
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import config

log = logging.getLogger("optimizer")

PRIORS = pathlib.Path(__file__).parents[1] / "program_templates/terminal_bench.md"

SYSTEM = """You improve a Terminal-Bench agent by rewriting its source file.

The tasks you see are a SAMPLE from a larger benchmark. You are scored on tasks you cannot
see. Propose general strategy changes — how the agent explores, plans, verifies, recovers.

Never special-case a task: no branching on a task name, no hardcoded path, command or
expected output lifted from a failure report. A change that only helps the tasks shown is
worse than no change, and it will be rejected automatically.

HARD CONSTRAINT on the message list. `messages` is sent verbatim to the chat completions
API and must stay a valid conversation at all times:

  * a {"role": "tool"} entry is legal ONLY immediately after an assistant message whose
    tool_calls contain that exact tool_call_id;
  * never seed, prepend or inject a tool message the model did not ask for -- put setup
    output in the system or user message instead, or run it as a real tool call;
  * never remove an assistant message while keeping its tool responses.

Violating this raises on the next API call, and the agent's loop catches the exception and
stops immediately -- it then scores zero on every task while looking perfectly reasonable
in review. This has already happened twice; it is the single most common way a promising
rewrite is destroyed.

Reply in exactly this format:

PROPOSAL: <one line: the change, and why it generalises beyond these tasks>
```python
<the complete rewritten file>
```"""


class ProposalRejected(Exception):
    """Costs one LLM call instead of a benchmark run."""


@dataclass
class Optimizer:
    """Same contract as MockOptimizer, so the worker cannot tell them apart."""
    model: str = config.OPTIMIZER_MODEL

    def propose(self, agent_source: str, failures: list[dict[str, Any]],
                ledger: list[str]) -> tuple[str, str, dict[str, int]]:
        prompt = build_prompt(agent_source, failures, ledger)
        usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
        note = ""
        for attempt in range(2):        
            text, u = self._complete(prompt + note)
            for k in usage:
                usage[k] += u[k]
            try:
                proposal, source = parse(text)
                # flag rides on the proposal so it lands in the history next to the score
                flag = preflight(agent_source, source, failures)
                if flag:
                    log.warning("proposal%s", flag)
                return proposal + flag, source, usage
            except ProposalRejected as e:
                log.warning(f"proposal rejected ({attempt + 1}/2): {e}")
                note = (f"\n\nYour previous reply was rejected: {e}\n"
                        f"Return the complete file in one ```python block, and lift "
                        f"nothing verbatim from the failure reports.")
        raise ProposalRejected("model failed pre-flight twice")

    def _complete(self, prompt: str) -> tuple[str, dict[str, int]]:
        from openai import OpenAI

        r = OpenAI(api_key=config.PLATFORM_OPENAI_KEY).chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content or "", {
            "llm_calls": 1,
            "input_tokens": r.usage.prompt_tokens if r.usage else 0,
            "output_tokens": r.usage.completion_tokens if r.usage else 0}


def build_prompt(agent_source: str, failures: list[dict[str, Any]],
                 ledger: list[str]) -> str:
    """Current source, this iteration's failures, the ledger, static priors."""
    parts = [f"## Current agent source\n```python\n{agent_source}\n```"]

    if failures:
        shown = "\n\n".join(_render(f) for f in failures[:8])
        parts.append(f"## Tasks that failed this iteration ({len(failures)})\n{shown}")
    else:
        parts.append("## No failures this iteration — look for robustness, not fixes.")

    if ledger:
        parts.append("## Already tried (do not repeat)\n" + "\n".join(ledger[-10:]))

    if PRIORS.exists():
        parts.append("## Techniques that have helped before (unverified priors)\n"
                     + PRIORS.read_text()[:4000])
    return "\n\n".join(parts)


def _render(f: dict[str, Any]) -> str:
    cmds = "\n".join(f"  $ {c.get('command', '')[:200]} -> exit {c.get('exit_code')}"
                     f"\n    {(c.get('stderr') or '')[:300]}"
                     for c in (f.get("failing_commands") or [])[:5])
    return (f"### {f['task_id']} (reward {f.get('reward')})\n"
            f"{cmds}\n  verifier: {(f.get('verifier_output') or '')[:500]}\n"
            f"  tail: {(f.get('tail') or '')[:1500]}")


def parse(text: str) -> tuple[str, str]:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks:
        raise ProposalRejected("no ```python code block in the reply")
    m = re.search(r"^PROPOSAL:\s*(.+)$", text, re.M)
    return (m.group(1).strip() if m else "(no proposal line)"), max(blocks, key=len)


def preflight(old: str, new: str, failures: list[dict[str, Any]]) -> str:
    """Reject what provably cannot run; return a note about what merely looks wrong.
    """
    try:
        tree = ast.parse(new)
    except SyntaxError as e:
        raise ProposalRejected(f"the file does not parse: {e}") from e
    if not any(isinstance(n, ast.ClassDef) and n.name == "HarnessAgent"
               for n in ast.walk(tree)):
        raise ProposalRejected("class HarnessAgent is missing from the file")
    if lifted := lifted_literals(old, new, failures):
        return f" [flagged: possibly lifted from failure reports: {lifted[:5]}]"
    return ""


def lifted_literals(old: str, new: str, failures: list[dict[str, Any]]) -> list[str]:
    """Task ids and absolute paths only — lexical, so it catches a hardcoded branch and
    misses a paraphrase.
    """
    candidates: set[str] = set()
    for f in failures:
        candidates.add(str(f["task_id"]))
        parts = [str(f.get("verifier_output") or ""), str(f.get("tail") or "")]
        for c in f.get("failing_commands") or []:
            parts += [str(c.get("command", "")), str(c.get("stderr") or "")]
        candidates |= set(re.findall(r"/[\w.-]+(?:/[\w.-]+)+", " ".join(parts)))
    return sorted(t for t in candidates if t in new and t not in old)


def demo() -> None:
    old = "class HarnessAgent:\n    PROMPT = 'be careful'\n"
    fails = [{"task_id": "cobol-modernization", "reward": 0.0,
              "failing_commands": [{"command": "python /srv/legacy_payroll.cbl",
                                    "exit_code": 1, "stderr": "boom"}],
              "verifier_output": "assert 0 == 1", "tail": "..."}]

    # Fatal: the run could not possibly score anything.
    for bad, why in [("class HarnessAgent:\n  x = (", "syntax"),
                     ("class Other:\n    pass\n", "missing class")]:
        try:
            preflight(old, bad, fails)
            raise AssertionError(f"pre-flight should have rejected: {why}")
        except ProposalRejected:
            pass

    # Non-fatal: suspected hardcoding is flagged, never rejected — a false positive here
    # would kill a job, while a false negative costs one run the holdout score catches.
    assert "flagged" in preflight(
        old, "class HarnessAgent:\n    P = 'cobol-modernization'\n", fails)
    assert "flagged" in preflight(
        old, "class HarnessAgent:\n    P = '/srv/legacy_payroll.cbl'\n", fails)
    assert preflight(old, "class HarnessAgent:\n    PROMPT = 'plan first'\n", fails) == ""

    # Ordinary vocabulary appearing in a trace must NOT be treated as a lift. Observed for
    # real: a run collected `compiled`, `matching`, `position`, `ensure_ascii` and rejected
    # every proposal, failing the job.
    wordy = [{"task_id": "chess-best-move", "reward": 0.0, "tail": "",
              "failing_commands": [{"command": "python check.py --position 3",
                                    "exit_code": 1, "stderr": "no matching placements"}],
              "verifier_output": "compiled coordinates unavailable; inspection generated"}]
    preflight(old, "class HarnessAgent:\n"
                   "    P = 'ensure the plan is compiled before matching position'\n",
              wordy)

    proposal, src = parse("PROPOSAL: plan first\n```python\nclass HarnessAgent: pass\n```")
    assert proposal == "plan first" and "HarnessAgent" in src
    try:
        parse("I think you should plan more carefully.")
        raise AssertionError("prose-only reply should be rejected")
    except ProposalRejected:
        pass

    assert "Already tried" in build_prompt(old, fails, ["iter 1: x -> 0.40 (rejected)"])
    print("ok — pre-flight guards, parser, prompt assembly")


if __name__ == "__main__":
    demo()
