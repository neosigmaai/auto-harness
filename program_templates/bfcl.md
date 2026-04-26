---

## BFCL: Benchmark-specific Guidance

### Current Benchmark

- **Agent:** `agent/agent.py` — `HarnessHandler(OpenAIResponsesHandler)` for the OpenAI Responses path
- **Category:** `multi_turn_base` (200 stateful function-calling tasks; configured in `experiment_config.yaml`)
- **Task split:** `bfcl_data/task_split.json` — deterministic 70/30 random split with seed 42 (NOT stratified by baseline pass/fail)
- **Baseline scores:** see `workspace/results.tsv` (iteration 0)

### Additional Read-only Files

| File | Purpose |
|------|---------|
| `agent/templates/bfcl.py` | Starting-point template — diff against `agent.py` to see your changes |

`bfcl_data/task_split.json` exists on disk but is **off-limits** — reading it would reveal which task IDs are in the test set. The harness uses it internally; you do not.

### Task IDs

Task IDs are strings of the form `<category>_<index>`, e.g. `multi_turn_base_0`, `multi_turn_base_42`.

```bash
python benchmark.py --task-ids multi_turn_base_0 multi_turn_base_42
```

### Analyzing Failures (Step 2)

Read train task traces:

```text
workspace/traces/latest/<task_id>/trace.json    ← model inference log (messages, tool calls, results)
workspace/traces/latest/<task_id>/result.json   ← raw BFCL result entry (one JSON object)
workspace/traces/latest/<task_id>/score.json    ← BFCL score entry if the task failed
```

**Only read traces in `workspace/traces/latest/`.** The directory under `workspace/bfcl_runs/` is internal harness state — it may contain test-split artifacts and must not be inspected.

For each failing task, examine:
- Did the agent pick the right function on each turn?
- Did it pass the right arguments (types, names, order)?
- Did it carry state correctly across turns? Multi-turn rewards are binary: one wrong turn usually fails the whole task.
- Did it call extra unnecessary functions? BFCL checks both correctness AND minimality of the call sequence.
- Did it hallucinate a tool that isn't in the schema?

### Editing agent/agent.py (Step 3)

You own the **entire file**. The class signature must remain
`HarnessHandler(OpenAIResponsesHandler)` because the harness registers it as
the BFCL handler for `harness-agent`. Inside that, the visible levers are:

- **`AGENT_INSTRUCTION`** — the developer-role system prompt injected on turn 1 (primary optimization target)
- **`add_first_turn_message_FC()`** — first-turn message construction; injects `AGENT_INSTRUCTION` into `inference_data["message"]`
- **`_compile_tools()`** — tool schema/description rewriting before the model sees the tool list
- **`_add_execution_results_FC()`** — post-processing of tool output before the next inference step

You can override any non-`@final` method on `OpenAIResponsesHandler`. The four
core inference loops on `BaseHandler` are `@final` and cannot be changed —
this preserves canonical BFCL execution semantics.

Diff against the starting template to track your changes:

```bash
diff agent/templates/bfcl.py agent/agent.py
```

### Important behaviors to preserve

- **Use the `developer` role**, not `system`. OpenAI Responses converts `system` → `developer` only for `test_entry["question"]`; messages you append directly are sent unmodified, so use `developer` explicitly.
- **Idempotent injection.** The starting template guards against duplicate developer messages. Preserve that guard if you change the injection logic — BFCL may call `add_first_turn_message_FC` more than once.
- **Don't hardcode `AGENT_MODEL`.** It is read from the environment by the harness via `experiment_config.yaml`.

### NEVER DO THESE

- **Never modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any file in `agent/templates/`, `agent/helpers/`, `program_templates/`, `bfcl_data/`, `scripts/`
- **Never read** raw test artifacts under `workspace/bfcl_runs/` — they may include test-split score details
- **Never inspect** `bfcl_data/task_split.json` to learn which tasks are in test
- **Never override** the four core inference loops in `BaseHandler` (they are `@final` for a reason)
- **Never hardcode** the BFCL model id `harness-agent` or change the class name `HarnessHandler` — both are wired into the runner
- **Never search the web** or fetch online resources during a run
- **Never create new files** outside of `agent/agent.py` and `workspace/learnings.md`
