---

## WebArena: Benchmark-specific Guidance

### Current Benchmark

- **Agent:** `agent/agent.py` — thin wrapper over WebArena's `PromptAgent` with a CoT prompt constructor
- **Site:** Reddit/Postmill starter subset (configurable via `experiment_config.yaml: site`)
- **Task split:** `webarena_data/task_split.json`
- **Baseline score and task counts:** see `workspace/results.tsv` (iteration 0) and `webarena_data/task_split.json`

### Smoke Mode vs Full Mode

If `experiment_config.yaml` has `smoke: true`, the harness is running against a **local Flask fixture** (see `fixtures/webarena_smoke/app.py`) instead of the full WebArena Postmill container. In smoke mode:

- Task IDs are `10001`–`10008` (5 train / 3 test). Tasks `10006`–`10008` ask about post-body and comment content only present on the detail page, so they typically fail on the baseline — these are your optimization targets.
- Task configs live in `webarena_data/smoke_tasks/` — **still off-limits for reading**; the same anti-leakage rules apply
- The fixture HTML is simpler than real Postmill; a few shots may transfer 1:1 to the full benchmark but some won't
- Otherwise the agent, evaluator, and Playwright pipeline are identical to full mode

If `smoke` is absent or `false`, IDs are the real WebArena task IDs (e.g. 27, 28, 66...) and the harness uses `webarena_repo/config_files/`.

### Additional Read-only Files

| File | Purpose |
|------|---------|
| `agent/templates/webarena.py` | Starting template — diff against `agent.py` to see your changes |
| `webarena_data/task_split.json` | Train/test split (do not edit) |
| `webarena_data/reddit_candidate_tasks.json` | Curated read-only Reddit task IDs the split samples from |
| `webarena_repo/` | WebArena source clone. **Never** browse `webarena_repo/config_files/` — it contains test task definitions |
| `webarena_data/smoke_tasks/` | Smoke-mode task definitions. **Never** browse — contains test task answers |
| `fixtures/webarena_smoke/` | Smoke-mode Flask fixture. Infrastructure — never edit |
| `webarena/driver.py` | Per-task subprocess orchestrator. Infrastructure — never edit |

### Task IDs

Task IDs are integers: `python benchmark.py --task-ids 27 33`

### Analyzing Failures (Step 2)

Read train task traces to understand root cause:

```
workspace/traces/latest/<task_id>/trace.json    ← harness-summarized actions + observation previews
workspace/traces/latest/<task_id>/result.json   ← reward, steps, duration, error
```

**Only read traces in `workspace/traces/latest/`.** Test-split traces are never persisted (the runner sets `HARNESS_SAVE_TRACE=0` for non-train runs).

For each failing task, examine:
- Did the agent parse the accessibility tree correctly?
- Did it click/hover/type on the right element IDs?
- Did it infer when it had the answer, or loop past it?
- Did it hit the max-step early-stop before completing?
- Did it emit the `stop [...]` action with a valid answer payload?

### Editing agent/agent.py (Step 3)

You own the **entire file**. Safe-to-edit surfaces:

- **`AGENT_INSTRUCTION`** — the text prepended to WebArena's default CoT intro (primary optimization target)
- **`DEFAULT_INSTRUCTION_PATH`** — swap the prompt JSON (e.g., use `p_direct_id_actree_2s.json` for direct prediction)
- **`_build_lm_config(...)`** — temperature, top_p, max_tokens, max_obs_length, max_retry
- **`HarnessAgent.__init__`** — inject a different prompt constructor or preprocess observations
- **Override `next_action`** — the whole action-selection loop is yours

Diff against the starting template to track changes:

```bash
diff agent/templates/webarena.py agent/agent.py
```

### Known Techniques That Improve WebArena Scores

1. **Stronger answer-format reminders** — models often forget the `stop [answer]` syntax; reinforce it in the intro (+3-8%)
2. **Explicit "don't click randomly" guardrails** — reduce parse failures/loops (+2-5%)
3. **Shorter `max_obs_length`** with "scroll if you need more" guidance — keeps the model focused on visible elements
4. **Task-specific few-shots** — for read-only info-seeking tasks, add an example showing early `stop [answer]` (+5-10%)
5. **Temperature 0 during evaluation** — cuts variance at the cost of exploration

### NEVER DO THESE

- **Never modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `webarena/driver.py`, `experiment_config.yaml`, or any file in `agent/templates/`, `program_templates/`, `webarena/`, `webarena_data/`, `webarena_repo/`, `fixtures/`
- **Never read** from `webarena_repo/config_files/` or `webarena_data/smoke_tasks/` directly — they contain both train and test tasks
- **Never read** any file under `workspace/webarena_jobs/` — raw runner output, contains test-split results; use `workspace/traces/latest/` instead
- **Never change** `max_steps`, `per_task_timeout`, `max_concurrency`, site, or any infrastructure setting
- **Never hardcode** `AGENT_MODEL` or `AGENT_REASONING_EFFORT` — these are set by the harness from `experiment_config.yaml`
- **Never install packages** or modify the Python environment
- **Never search the web** or fetch any online resources
- **Never create new files** outside of `agent/agent.py` and `workspace/learnings.md`
