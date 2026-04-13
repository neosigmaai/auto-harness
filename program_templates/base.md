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
| `agent/agent.py` | The agent you optimize |
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
| `python gating.py` | Three-step gate. Exit 0 = all clear, commit and record |
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

Make one focused change per iteration. Smaller changes are easier to gate and easier to revert.

**Do not modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `experiment_config.yaml`, or any workspace file.

---

### 4. Gate

```bash
python gating.py
```

Three steps run in sequence:

- **Step 1 — Regression suite**: re-runs tasks in `suite.json` on the train split. Pass rate must be ≥ threshold. Protects previously-fixed tasks from regressing.
- **Step 2 — Full test**: runs the test split. val_score must be ≥ best recorded in `results.tsv`.
- **Step 3 — Suite promotion** *(only if Steps 1+2 pass)*: re-runs previously-failing train tasks; newly-passing ones are automatically added to `suite.json`.

**Exit 0** → proceed to Record.

**Exit 1** (Step 1 or 2 failed) → revert and try a different approach:

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

1. **Only edit `agent/agent.py` and `workspace/learnings.md`** — never touch infrastructure files
2. **Never skip the gate** — every committed change must pass all three steps
3. **One hypothesis per iteration** — keep changes small and reversible
4. **Always update `learnings.md`** — even on failure; the log is your memory
5. **Never use test data to guide changes** — only train failures inform improvements
6. **Stop when** val_score has not improved for 5 consecutive iterations — write a summary in `learnings.md` and surface your top findings to the human

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
