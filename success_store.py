"""
Success trace clustering and few-shot retrieval for AgentBench OS.
"""
from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

TRACES_PATH_TEMPLATE = "workspace/{split}_traces.json"
SUCCESS_STORE_PATH = "workspace/success_store.json"
TAXONOMY_PATH = "workspace/failure_taxonomy.json"

CLUSTER_SYSTEM_PROMPT = """You are classifying OS-interaction tasks into semantic clusters.
Given a task instruction, return ONLY a JSON object:
{
  "cluster_id": short snake_case label for the task type
}
No other text."""


def cluster_instruction(client: OpenAI, instruction: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"INSTRUCTION: {instruction}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content).get("cluster_id", "general")


def format_few_shot(trace: dict) -> str:
    lines = [f"TASK: {trace['instruction']}"]
    for turn in trace.get("turns", []):
        if turn.get("cmd"):
            lines.append(f"  $ {turn['cmd']}")
        if turn.get("env_output"):
            lines.append(f"  > {turn['env_output'][:200]}")
    lines.append(f"ANSWER: {trace.get('final_output', '')}")
    return "\n".join(lines)


def run_success_store(split: str = "dev", max_per_cluster: int = 3) -> None:
    traces_path = TRACES_PATH_TEMPLATE.format(split=split)
    if not os.path.exists(traces_path):
        print(f"[success_store] {traces_path} not found")
        return

    with open(traces_path) as f:
        traces = json.load(f)

    passing = [trace for trace in traces if trace.get("reward", 0.0) >= 0.5]
    print(f"[success_store] processing {len(passing)} passing traces")

    taxonomy_index: dict[str, str] = {}
    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH) as f:
            for record in json.load(f):
                taxonomy_index[record["task_id"]] = record.get("cluster_id", "general")

    store: dict[str, list[dict]] = {}
    if os.path.exists(SUCCESS_STORE_PATH):
        with open(SUCCESS_STORE_PATH) as f:
            store = json.load(f)

    client = OpenAI()
    for trace in passing:
        if trace["task_id"] in taxonomy_index:
            cluster = taxonomy_index[trace["task_id"]]
        else:
            cluster = cluster_instruction(client, trace["instruction"])

        store.setdefault(cluster, [])
        existing_ids = {example["task_id"] for example in store[cluster]}
        if trace["task_id"] not in existing_ids:
            store[cluster].append(
                {
                    "task_id": trace["task_id"],
                    "cluster_id": cluster,
                    "few_shot": format_few_shot(trace),
                }
            )
        store[cluster] = store[cluster][:max_per_cluster]

    with open(SUCCESS_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

    print(
        "[success_store] clusters: "
        + ", ".join(f"{name}({len(values)})" for name, values in sorted(store.items()))
    )


def get_few_shots(cluster_id: str, n: int = 2) -> list[str]:
    if not os.path.exists(SUCCESS_STORE_PATH):
        return []
    with open(SUCCESS_STORE_PATH) as f:
        store = json.load(f)
    return [example["few_shot"] for example in store.get(cluster_id, [])[:n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-per-cluster", type=int, default=3)
    args = parser.parse_args()
    run_success_store(args.split, args.max_per_cluster)
