"""
LLM-as-judge failure classifier for AgentBench OS traces.

Reads workspace/{split}_traces.json, classifies failing traces, and writes
workspace/failure_taxonomy.json.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from openai import OpenAI

TRACES_PATH_TEMPLATE = "workspace/{split}_traces.json"
TAXONOMY_PATH = "workspace/failure_taxonomy.json"

FAILURE_TYPES = [
    "wrong_command",
    "incomplete_execution",
    "wrong_interpretation",
    "format_error",
    "env_interaction_failure",
]

AFFECTED_LEVERS = [
    "system_prompt",
    "self_critique",
    "context_construction",
    "output_formatting",
    "tool_use_strategy",
]

CLASSIFIER_SYSTEM_PROMPT = """You are a precise failure analyst for OS-interaction AI agents.
You will be given a failed task trace: the instruction, the agent's turn-by-turn responses
(including which commands succeeded or failed), and the expected labels.

Respond ONLY with valid JSON. No preamble, markdown, or explanation.
The JSON must have exactly these fields:
{
  "failure_type": one of """ + str(FAILURE_TYPES) + """,
  "affected_lever": one of """ + str(AFFECTED_LEVERS) + """,
  "cluster_id": short snake_case string grouping similar failures,
  "hypothesis": one sentence stating what specifically went wrong and why
}

Use cmd_success=false in the trace to identify where execution broke down."""


def classify_trace(client: OpenAI, trace: dict) -> dict:
    turns_text = "\n".join(
        f"Turn {turn['turn']}: {turn.get('response', '')[:300]}"
        + (f"\n  CMD: {turn['cmd'][:200]}" if "cmd" in turn else "")
        + (f"\n  CMD_SUCCESS: {turn.get('cmd_success', 'unknown')}" if "cmd" in turn else "")
        + (f"\n  ENV: {turn.get('env_output', '')[:200]}" if "env_output" in turn else "")
        for turn in trace.get("turns", [])
    )
    user_message = f"""INSTRUCTION: {trace.get('instruction', '')}

EXPECTED LABELS: {trace.get('labels', [])}

AGENT INTERACTION:
{turns_text}

FINAL OUTPUT: {trace.get('final_output')}

Classify this failure."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    result["task_id"] = trace["task_id"]
    return result


def run_classifier(split: str = "dev") -> None:
    traces_path = TRACES_PATH_TEMPLATE.format(split=split)
    if not os.path.exists(traces_path):
        print(f"[classifier] {traces_path} not found — run benchmark.py first")
        return

    with open(traces_path) as f:
        traces = json.load(f)

    failures = [trace for trace in traces if trace.get("reward", 0.0) < 0.5]
    print(f"[classifier] classifying {len(failures)} failures out of {len(traces)} traces")

    current_task_ids = {trace["task_id"] for trace in traces}
    current_failure_ids = {trace["task_id"] for trace in failures}
    existing: dict[str, dict] = {}
    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH) as f:
            for record in json.load(f):
                task_id = record["task_id"]
                if task_id not in current_task_ids or task_id in current_failure_ids:
                    existing[task_id] = record

    client = OpenAI()
    new_records = []
    for i, trace in enumerate(failures):
        try:
            record = classify_trace(client, trace)
            new_records.append(record)
            print(
                f"  [{i + 1}/{len(failures)}] {record['task_id']}: "
                f"{record['failure_type']} / {record['cluster_id']}"
            )
        except Exception as exc:
            print(f"  [{i + 1}/{len(failures)}] {trace['task_id']}: classifier error — {exc}")

    existing.update({record["task_id"]: record for record in new_records})
    all_records = sorted(existing.values(), key=lambda record: record["task_id"])

    with open(TAXONOMY_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[classifier] wrote {len(all_records)} records to {TAXONOMY_PATH}")

    clusters = Counter(record["cluster_id"] for record in all_records)
    print("\n[classifier] failure clusters:")
    for cluster, count in clusters.most_common():
        print(f"  {count:3d}  {cluster}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    run_classifier(args.split)
