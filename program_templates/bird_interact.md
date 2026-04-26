---

## BIRD-Interact: Benchmark-specific Guidance

### Modes

Set `mode` in `experiment_config.yaml`:

- `a-interact` — autonomous tool-using SQL agent
- `c-interact` — clarification-first conversational SQL agent

The default integration assumes `a-interact`.

### Task IDs

Task IDs are BIRD `instance_id` strings, not integers.

Run a subset with:

```bash
python benchmark.py --task-ids task_001 task_017
```

### Analyzing Failures (Step 2)

Read train traces here:

```text
workspace/traces/latest/<instance_id>/trace.json
workspace/traces/latest/<instance_id>/result.json
```

- `trace.json` contains dialogue history, tool trajectory, ADK events, and final response
- `result.json` contains the raw per-task reward and metadata
- Only train traces are copied into `workspace/traces/`

### Editing agent/agent.py (Step 3)

`agent/agent.py` defines the BIRD system agent that the external BIRD-Interact-ADK services call.

Focus changes here:

- `AINTERACT_INSTRUCTION` — the autonomous agent policy
- `CINTERACT_INSTRUCTION` — the conversational clarification policy
- `build_agent()` — how the model is configured for each mode

Do not edit the external BIRD repo from the optimization loop. The harness treats that repo as read-only benchmark infrastructure.
