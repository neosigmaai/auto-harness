---

## tau-bench: Benchmark-specific Guidance

### Task IDs

Task IDs are integers: `python benchmark.py --task-ids 0 1 42`

### Analyzing Failures (Step 2)

- Read train-split simulation traces for failing tasks
- **Never read test-split traces** — only train traces are available for analysis

### Editing agent/agent.py (Step 3)

- **Instructions** — change `AGENT_INSTRUCTION` or the `system_prompt` property
- **Architecture** — change `generate_next_message()`, state management (`HarnessState`), or how messages are constructed
- **Tools** — tau-bench injects its fixed domain tools; you cannot add new tools for tau-bench runs

**Never modify** `experiment_config.yaml` — in particular, do not change `reasoning_effort`.
