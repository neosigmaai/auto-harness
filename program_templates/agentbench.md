# AgentBench OS Notes

You are optimizing an AgentBench OS agent. The train split is `dev`; the held-out
gate split is `test`.

## Trace Files

Read only train/dev traces while improving the agent:

| File | Purpose |
|------|---------|
| `workspace/dev_traces.json` | Train/dev interaction traces |
| `workspace/test_traces.json` | Gate/test traces; never use these to guide changes |
| `workspace/failure_taxonomy.json` | Classifier output for train/dev failures |
| `workspace/success_store.json` | Few-shot examples, updated only after stable wins |

## Classify Failures

After reading the train traces, run the classifier:

```bash
python classifier.py --split dev
```

Read `workspace/failure_taxonomy.json`. It tells you:

- `failure_type` — what category of mistake the agent made
- `affected_lever` — which part of `agent/agent.py` to edit
- `cluster_id` — which group of tasks this failure belongs to
- `hypothesis` — what specifically went wrong

Use the taxonomy this way:

1. Count cluster sizes.
2. Pick the largest cluster.
3. Read 3-5 raw traces from that cluster manually before editing.
4. Make one hypothesis-driven change to `agent/agent.py`.
5. Re-run benchmark on 10-15 tasks from that cluster before a full run.

One hypothesis per iteration.

After a gate passes and dev pass rate is high enough to provide useful examples,
run:

```bash
python success_store.py --split dev
```

## Gate

```bash
python gating.py
```

Four benchmark gates run after the file guard:

- Step 1 — regression suite on train/dev tasks
- Step 2 — full held-out test split
- Step 3 — cluster regression check
- Step 4 — suite promotion
