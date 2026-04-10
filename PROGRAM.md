# auto-harness — Agent Program

## What You Are Doing

You are an autonomous coding agent optimizing `agent/agent.py` to perform better on Terminal-Bench 2.0. You run a tight loop:

```
run benchmark → analyze failures → improve agent → gate → commit → repeat
```

Your edit targets are `agent/agent.py` and `workspace/learnings.md`. Everything else is infrastructure.

---

## Current Benchmark: Terminal-Bench 2.0

- **49 train tasks**, **22 test tasks**
- **Baseline:** train val_score = 0.3878, test val_score = 0.4091
- **Model:** GPT-5.4
- **Agent:** `agent/agent.py` — single bash tool, minimal system prompt
- **Task split:** `tbench_data/task_split.json`

---

## Files You Own

| File | Purpose |
|------|---------|
| `agent/agent.py` | The complete agent you optimize — prompt, loop, tools, strategy |
| `workspace/learnings.md` | Persistent learnings log — append after every iteration |

**Read-only files** (do not edit):

| File | Purpose |
|------|---------|
| `agent/templates/terminal_bench.py` | Starting-point template — diff against agent.py to see changes |
| `tbench_data/task_split.json` | Train/test split (49 train, 22 test) |
| `workspace/suite.json` | Regression suite — auto-managed by gating.py |
| `workspace/train_results.json` | Last train benchmark results |
| `workspace/results.tsv` | Iteration history with val_scores |

---

## How agent/agent.py works

`agent/agent.py` is the **complete Terminal-Bench agent**. It was initially copied from `agent/templates/terminal_bench.py`. You can see what you've changed:

```bash
diff agent/templates/terminal_bench.py agent/agent.py
```

### What you can change

You own the **entire file**. Everything is fair game:

- **`AGENT_INSTRUCTION`** — the system prompt (primary optimization target)
- **`TOOLS`** — tool definitions (add `analysis`/`plan` fields, change descriptions)
- **`MAX_STEPS`**, **`MAX_OUTPUT_CHARS`** — execution parameters
- **`_truncate()`** — output processing strategy
- **`HarnessAgent.run()`** — the full agent loop (add environment bootstrapping, planning enforcement, completion verification, context management, etc.)
- **`HarnessAgent.setup()`** — pre-execution setup

### Known techniques that improve Terminal-Bench scores

(From studying top-performing agents — see `TERMINAL_BENCH_AGENT_STRATEGIES.md` one directory above)

1. **Environment bootstrapping** — gather OS, installed tools, file listing before starting (+5-10%)
2. **Enforced TODO planning** — make the model create and maintain a plan (+10-20%, biggest single win)
3. **Non-interactive mode** — never ask questions, just act (+3-5%)
4. **Double-confirmation** — verify task completion before declaring done (+3-5%)
5. **Progressive reasoning** — high effort for first 10 steps, low after (+2-5%)
6. **Forced reasoning in tool schema** — add `analysis` and `plan` fields to bash tool

---

## Commands

| Command | What it does |
|---------|-------------|
| `python benchmark.py` | Run train split (49 tasks), print per-task pass/fail, save `workspace/train_results.json` |
| `python benchmark.py --task-ids cobol-modernization regex-log` | Run specific tasks ad-hoc |
| `python gating.py` | Three-step gate. Exit 0 = all clear, commit and record |
| `python record.py --val-score X --evals-passed N --evals-total M` | Append iteration result |

---

## The Loop

### 1. Run Benchmark

```bash
python benchmark.py
```

Read the stdout output. Note which tasks failed (task name, reward). Results are saved to `workspace/train_results.json`.

---

### 2. Analyze Failures

Read train task traces to understand root cause:

```
workspace/traces/latest/<task_name>/trace.json      ← most recent run (updated after each benchmark)
workspace/traces/latest/<task_name>/result.json     ← reward, duration, config
workspace/traces/baseline/<task_name>/trace.json    ← original baseline run (never overwritten)
```

Compare `latest/` vs `baseline/` to see if your changes helped or hurt a specific task.

**IMPORTANT: Only read traces in `workspace/traces/`.** Test traces are not available. Do not look in `workspace/tbench_jobs/` directly.

For each failing task, examine:
- What commands did the agent run?
- Did it understand the task correctly?
- Did it explore the environment before acting?
- Did it check command output for errors?
- Did it verify its solution?
- Did it give up too early or get stuck in a loop?

Append findings to `workspace/learnings.md`.

---

### 3. Improve Agent

Edit `agent/agent.py`. Everything is fair game: the system prompt, the agent loop, tool definitions, parameters, and strategy.

Make one focused change per iteration. Smaller changes are easier to gate and easier to revert.

**Do not modify** `benchmark.py`, `gating.py`, `record.py`, template files, `tbench_data/`, or any workspace file other than `learnings.md`.

---

### 4. Gate

```bash
python gating.py
```

Three steps run in sequence:

- **Step 1 — Regression suite**: re-runs tasks in `suite.json` on the train split. Pass rate must be ≥ 80%. Protects previously-fixed tasks from regressing.
- **Step 2 — Full test**: runs the 22 test tasks. val_score must be ≥ best recorded in `results.tsv` (currently 0.4091).
- **Step 3 — Suite promotion**: re-runs previously-failing train tasks; newly-passing ones are added to `suite.json`.

**Exit 0** → proceed to Record.

**Exit 1** → revert and try a different approach:

```bash
git checkout agent/agent.py
```

If the same hypothesis fails 3 times in a row, abandon it and try something different.

---

### 5. Record

After exit 0, commit and record:

```bash
git add agent/agent.py
git commit -m "improve: <what changed and why>"
python record.py --val-score <val_score from Step 2 output> --evals-passed <n> --evals-total <m>
```

---

### 6. Update Learnings

After every iteration — gate passed or failed — append to `workspace/learnings.md`:

```markdown
## Iteration N — val_score: X.XX → Y.YY ✓/✗

**What changed:** <one sentence>

**Pattern confirmed:** <failure mode>

**What worked / didn't work:** <specifics>

**Needs from human:** <or "none">
```

---

### 7. Repeat

Go to step 1.

---

## Rules

1. **Only edit `agent/agent.py` and `workspace/learnings.md`**
2. **Only read traces from `workspace/traces/latest/`** — never access test traces
3. **Never skip the gate** — every committed change must pass all three steps
4. **One hypothesis per iteration** — keep changes small and reversible
5. **Always update `learnings.md`** — even on failure; the log is your memory
6. **Stop when** val_score has not improved for 5 consecutive iterations

## NEVER DO THESE

- **Never modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any file in `agent/templates/`, `program_templates/`, `tbench_data/`
- **Never change** concurrency, timeout, env_provider, or any infrastructure setting
- **Never install packages** or modify the Python environment
- **Never read traces from `workspace/tbench_jobs/`** — only use `workspace/traces/latest/`
- **Never search the web** or fetch any online resources
- **Never create new files** outside of `agent/agent.py` and `workspace/learnings.md`

---

## File Formats

### `tbench_data/task_split.json`

```json
{
  "train": ["cobol-modernization", "regex-log", ...],
  "test": ["code-from-image", "portfolio-optimization", ...],
  "infra_excluded": ["caffe-cifar-10", ...]
}
```

### `workspace/suite.json`

```json
{
  "tasks": ["cobol-modernization", "regex-log"],
  "threshold": 0.8,
  "last_results": {"cobol-modernization": 1.0, "regex-log": 1.0}
}
```

### `workspace/train_results.json`

```json
{
  "split": "train",
  "timestamp": "2026-04-10T21:21:54+00:00",
  "results": {"cobol-modernization": 1.0, "regex-log": 1.0, "chess-best-move": 0.0}
}
```

### `workspace/results.tsv`

```
iteration	val_score	commit	evals_passed	evals_total	timestamp
0	0.4091	baseline	0	0	2026-04-10T21:21:54+00:00
```
