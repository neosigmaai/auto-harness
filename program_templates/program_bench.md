---

## ProgramBench: Benchmark-specific Guidance

### Current Benchmark

- **Agent:** `agent/agent.py` — single bash tool, scripted bootstrap, then LLM-driven loop
- **Task split:** `programbench_data/task_split.json` (random 70/30, no stratification)
- **Slice:** the slice in `experiment_config.yaml` defines the universe of tasks; the split is over that universe
- **Per-task scoring:** fraction of behavioral tests passed in `<iid>.eval.json` (continuous in [0, 1])
- **Cheat-prevention:** the harness moves the cleanroom's `/workspace/executable` to `/opt/orig/executable` before your agent starts. You may invoke that binary to fingerprint behavior, but you cannot bundle it back into your submission.

### Additional Read-only Files

| File | Purpose |
|------|---------|
| `agent/templates/program_bench.py` | Starting-point template — diff against `agent.py` to see your changes |
| `agent/helpers/program_bench/` | Container lifecycle, image pulls, submission packaging — DO NOT modify |
| `programbench_data/task_split.json` | Train/test split |

### Task IDs

Task IDs look like `abishekvashok__cmatrix.5c082c6` (repo__name.short_sha). Pass to benchmark.py:

```
python benchmark.py --task-ids abishekvashok__cmatrix.5c082c6 ajeetdsouza__zoxide.67ca1bc
```

### Analyzing Failures (Step 2)

Read train task traces under `workspace/traces/latest/<iid>/`:

```
workspace/traces/latest/<iid>/trace.json    ← full LLM conversation + bash transcript
workspace/traces/latest/<iid>/manifest.txt  ← `find /workspace -type f` snapshot
workspace/traces/latest/<iid>/eval.json     ← per-test pass/fail with error messages
```

**Only read traces in `workspace/traces/latest/`.** Gate/test artifacts are kept outside `workspace/` and removed after scoring.

For each failing task, examine:
- Did the bootstrap discover the binary, docs, build tools?
- Did the agent run the binary's `--help` early enough?
- Did it produce a `compile.sh` at all? Did the eval find it?
- Were the failures in build (compile.sh exit code) or in test (mismatched outputs)?
- Did it run out of steps, run out of time, or declare done prematurely?

### Editing agent/agent.py (Step 3)

You own the **entire file**. Everything is fair game:

- **`AGENT_INSTRUCTION`** — system prompt (primary optimization target)
- **`TOOLS`** — tool schema (consider adding `analysis`/`plan` fields)
- **`BOOTSTRAP_COMMANDS`** — what to discover before the LLM drives
- **`MAX_STEPS`**, **`MAX_OUTPUT_CHARS`**, **`COMMAND_TIMEOUT`**
- **`_truncate()`**, **`_bootstrap_context()`**
- **`HarnessAgent.run()`** — the full agent loop

Diff against the starting template to track your changes:

```bash
diff agent/templates/program_bench.py agent/agent.py
```

### Known Patterns to Try

1. **Aggressive bootstrap** — `man <binary>`, `<binary> --help`, run a few sample inputs to fingerprint behavior before generating any code.
2. **Plan-then-build** — force the model to write `/workspace/PLAN.md` listing subcommands, flags, output formats before opening any source file.
3. **Build early, build often** — write `compile.sh` after the first source file; re-run after every change. Don't accumulate untested code.
4. **Behavioral parity loop** — for each subcommand, run the original binary on N inputs, capture outputs, then assert your build matches.
5. **Tighter step budget per phase** — short bursts of focused work (explore / scaffold / fill / verify) usually beat one long monolithic prompt.

### NEVER DO THESE

- **Never modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any file in `agent/templates/`, `agent/helpers/`, `program_templates/`, `programbench_data/`
- **Never change** concurrency, timeout, docker_cpus, or any infrastructure setting from inside the agent
- **Never hardcode** `MODEL` / `AGENT_MODEL` or `AGENT_REASONING_EFFORT` — these are set by the harness from `experiment_config.yaml`
- **Never assume internet is available inside the container** — it isn't, by ProgramBench's design
- **Never use a decompiler** — execute-only binary perms are enforced by the harness
- **Never try to inspect gate/test traces** — gate-split artifacts are intentionally unavailable
- **Never overfit to specific instance_ids in the slice.** ProgramBench's authors deliberately avoided harness tuning to prevent inflated scores on curated tasks. Prefer general-purpose harness improvements that should generalize beyond the visible slice.
