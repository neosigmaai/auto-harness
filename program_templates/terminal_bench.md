---

## Terminal-Bench 2.0: Benchmark-specific Guidance

### Current Benchmark

- **Agent:** `agent/agent.py` — single bash tool, minimal system prompt
- **Task split:** `tbench_data/task_split.json`
- **Baseline scores and task counts:** see `workspace/results.tsv` (iteration 0) and `tbench_data/task_split.json`

### Additional Read-only Files

| File | Purpose |
|------|---------|
| `agent/templates/terminal_bench.py` | Starting-point template — diff against `agent.py` to see your changes |
| `tbench_data/task_split.json` | Train/test split |

### Task IDs

Task IDs are string names: `python benchmark.py --task-ids cobol-modernization regex-log`

### Analyzing Failures (Step 2)

Read train task traces to understand root cause:

```
workspace/traces/latest/<task_name>/trace.json    ← full conversation (messages, tool calls, outputs)
workspace/traces/latest/<task_name>/result.json   ← reward, duration, config
```

**Only read traces in `workspace/traces/latest/`.** Do not look in `workspace/tbench_jobs/` — it contains both train and test data.

For each failing task, examine:
- What commands did the agent run?
- Did it understand the task correctly?
- Did it explore the environment before acting?
- Did it check command output for errors?
- Did it verify its solution?
- Did it give up too early or get stuck in a loop?

### Editing agent/agent.py (Step 3)

You own the **entire file**. Everything is fair game:

- **`AGENT_INSTRUCTION`** — the system prompt (primary optimization target)
- **`TOOLS`** — tool definitions (add `analysis`/`plan` fields, change descriptions)
- **`MAX_STEPS`**, **`MAX_OUTPUT_CHARS`** — execution parameters
- **`_truncate()`** — output processing strategy
- **`HarnessAgent.run()`** — the full agent loop
- **`HarnessAgent.setup()`** — pre-execution setup

Diff against the starting template to track your changes:

```bash
diff agent/templates/terminal_bench.py agent/agent.py
```

### Known Techniques That Improve Terminal-Bench Scores

1. **Environment bootstrapping** — gather OS, installed tools, file listing before starting (+5-10%)
2. **Enforced TODO planning** — make the model create and maintain a plan (+10-20%, biggest single win)
3. **Non-interactive mode** — never ask questions, just act (+3-5%)
4. **Double-confirmation** — verify task completion before declaring done (+3-5%)
5. **Progressive reasoning** — high effort for first 10 steps, low after (+2-5%)
6. **Forced reasoning in tool schema** — add `analysis` and `plan` fields to bash tool

### NEVER DO THESE

- **Never modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any file in `agent/templates/`, `program_templates/`, `tbench_data/`
- **Never change** concurrency, timeout, env_provider, or any infrastructure setting
- **Never hardcode** `MODEL` / `AGENT_MODEL` or `AGENT_REASONING_EFFORT` — these are set by the harness from `experiment_config.yaml`
- **Never install packages** or modify the Python environment
- **Never read traces from `workspace/tbench_jobs/`** — only use `workspace/traces/latest/`
- **Never search the web** or fetch any online resources
- **Never create new files** outside of `agent/agent.py` and `workspace/learnings.md`
