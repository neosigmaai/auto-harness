# auto-harness — Agent Program

## What You Are Doing

You are an autonomous coding agent optimizing `agent/agent.py` to perform better on a benchmark. You run a tight loop:

```
run benchmark → analyze failures → improve agent → gate → commit → repeat
```

Your edit targets are `agent/agent.py` and `workspace/learnings.md`. Everything else is infrastructure.

---

## Files You Own

| File | Purpose |
|------|---------|
| `agent/agent.py` | The agent you optimize, including any internal helper tools it defines |
| `workspace/learnings.md` | Persistent learnings log — patterns, hypotheses, requests to the human — **append after every iteration** |
| `workspace/results.tsv` | Iteration history — written by `record.py` after each successful gate |

**Read-only workspace files** (managed automatically — do not edit):

| File | Purpose |
|------|---------|
| `workspace/suite.json` | Regression suite — tasks promoted here automatically after each successful gate |
| `workspace/train_results.json` | Last train benchmark results — written by `benchmark.py` |

---

## Commands

| Command | What it does |
|---------|-------------|
| `python benchmark.py` | Run the full train benchmark, print per-task pass/fail, save `workspace/train_results.json` |
| `python benchmark.py --task-ids <id> ...` | Run specific tasks ad-hoc |
| `python gating.py` | Gate runner. Exit 0 = all clear, commit and record |
| `python record.py --val-score X --evals-passed N --evals-total M` | Append iteration result |
| `python prepare.py` | Initialize workspace (run once at start) |

---

## The Loop

### 1. Run Benchmark

```bash
python benchmark.py
```

Read the stdout output. Note which tasks failed. The results are also saved to `workspace/train_results.json`.

---

### 2. Analyze Failures

- Read train-split traces for failing tasks to understand root cause
- **Never use test data to guide changes** — only train traces are available for analysis
- Note patterns: what did the agent do wrong? Is this a prompt issue or a tool issue?
- Append findings to `workspace/learnings.md`

---

### 3. Improve Agent

Edit `agent/agent.py` — you own the entire file. The benchmark runner imports `HarnessAgent` directly, so any change here is picked up automatically.

`agent.py` may design and use its own internal tools/helpers when that is the
cleanest way to improve benchmark behavior. Examples include helper functions
for trace-aware prompting, command templates, answer validation, prompt routing,
or small deterministic pre/post-processing utilities inside `agent.py`.

Do not add new tracked helper files for these tools unless the human explicitly
approves it. Keep tool logic inside `agent.py`, and do not modify the benchmark
runner, scoring, gating, or dataset to support the tool.

Make one focused change per iteration. Smaller changes are easier to gate and easier to revert.

**Do not modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any workspace file.

---

### 4. Gate

```bash
python gating.py
```

Steps run in sequence:

- **Step 0 — File guard**: rejects the iteration if any tracked files outside the allowlist (`agent/agent.py`, `PROGRAM.md`) were modified. Fails immediately with exit 1.
- **Step 1 — Regression suite**: re-runs tasks in `suite.json` on the train split. Pass rate must be ≥ threshold. Protects previously-fixed tasks from regressing.
- **Step 2 — Full test**: runs the test split. val_score must be ≥ best recorded in `results.tsv`.
- **Step 3+ — Benchmark-specific checks and suite promotion** *(only if Steps 1+2 pass)*: may run extra regression checks; newly-passing train tasks are automatically added to `suite.json`.

**Exit 0** → proceed to Record.

**Exit 1** (any step failed) → revert and try a different approach:

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

The `evals-passed` and `evals-total` refer to the regression suite results from Step 1.

---

### 6. Update Learnings

After every iteration — gate passed or failed — append to `workspace/learnings.md`:

- **What you tried and what happened**
- **Patterns confirmed** — failure modes that appear repeatedly
- **What worked** — prompt changes that improved the score
- **Needs from human** — things you cannot fix autonomously

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

1. **Only edit `agent/agent.py` and `workspace/learnings.md`** — never touch infrastructure files. `gating.py` and `record.py` enforce this with a `git diff` check; modifying any other tracked file fails the gate immediately.
2. **Never skip the gate** — every committed change must pass every gate step
3. **One hypothesis per iteration** — keep changes small and reversible
   Internal tools are allowed, but each tool change still counts as the single
   hypothesis for that iteration and must be evaluated like any prompt change.
4. **Always update `learnings.md`** — even on failure; the log is your memory
5. **Never use test data to guide changes** — only train failures inform improvements
6. **Per-task timeouts count as failures** — any task that does not produce a verifier result within `per_task_timeout` scores `0.0` in `val_score`. If you see consistent timeouts, treat that as a signal to make the prompt more direct, not to ignore the missing reward.
7. **Stop when** val_score has not improved for 5 consecutive iterations — write a summary in `learnings.md` and surface your top findings to the human

---

## File Formats

### `workspace/suite.json`

Managed automatically by `gating.py`. Do not edit.

```json
{
  "tasks": ["<task-id>", "<task-id>"],
  "threshold": 0.8,
  "last_results": {
    "<task-id>": 1.0,
    "<task-id>": 1.0
  }
}
```

`tasks` grows as iterations fix previously-failing train tasks and both gates pass.

### `workspace/train_results.json`

Written by `benchmark.py`. Do not edit.

```json
{
  "split": "train",
  "timestamp": "<timestamp>",
  "results": {
    "<task-id>": 1.0,
    "<task-id>": 0.0
  }
}
```

### `workspace/results.tsv`

Tab-separated. Written by `record.py`.

```
iteration	val_score	commit	evals_passed	evals_total	timestamp
0	0.XXXX	baseline	0	0	<timestamp>
1	0.XXXX	abc1234	4	5	<timestamp>
```

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
