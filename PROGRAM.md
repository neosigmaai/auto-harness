# auto-harness — Agent Program

## What You Are Doing

You are an autonomous coding agent optimizing `agent/agent.py` to perform better on a benchmark. You run a tight loop:

```
run benchmark → analyze failures → improve agent → gate → commit → repeat
```

Your only edit target is `agent/agent.py`. Everything else is infrastructure.

---

## Files You Own

| File | Purpose |
|------|---------|
| `agent/agent.py` | The agent you optimize — **only file you may edit** |
| `workspace/learnings.md` | Persistent learnings log — patterns, hypotheses, requests to the human |
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
| `python benchmark.py --task-ids 0 1 42` | Run specific tasks ad-hoc |
| `python gating.py` | Three-step gate. Exit 0 = all clear, commit and record |
| `python record.py --val-score X --evals-passed N --evals-total M` | Append iteration result |
| `python prepare.py` | Initialize workspace (run once at start) |

---

## The Loop

### 1. Run Benchmark

```bash
python benchmark.py
```

Read the stdout output. Note which tasks failed (task ID, reward). The results are also saved to `workspace/train_results.json`.

---

### 2. Analyze Failures

- Read train-split simulation traces for failing tasks to understand root cause
- **Never read test-split traces** — only train traces are available for analysis
- Note patterns: what did the agent do wrong? Is this a prompt issue or a tool issue?
- Append findings to `workspace/learnings.md`

---

### 3. Improve Agent

Edit `agent/agent.py` — you own the entire file. `HarnessAgent` is imported directly by the benchmark runner, so any change here is picked up automatically:

- **Instructions** — change `AGENT_INSTRUCTION` or the `system_prompt` property
- **Architecture** — change `generate_next_message()`, state management (`HarnessState`), reasoning effort, or how messages are constructed
- **Tools** — tau-bench injects its fixed domain tools; you cannot add new tools for tau-bench runs

Make one focused change per iteration. Smaller changes are easier to gate and easier to revert.

**Do not modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, or any workspace file.

---

### 4. Gate

```bash
python gating.py
```

Three steps run in sequence:

- **Step 1 — Regression suite**: re-runs tasks in `suite.json` on the train split. Pass rate must be ≥ threshold. Protects previously-fixed tasks from regressing.
- **Step 2 — Full test**: runs the full test split. val_score must be ≥ best recorded in `results.tsv`.
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
## Iteration 3 — val_score: 0.78 → 0.81 ✓

**What changed:** tightened cancellation eligibility check in system prompt

**Pattern confirmed:** agent over-approved cancellations when user claimed prior approval.
Adding explicit social-engineering resistance fixed tasks 1 and 43.

**What worked:** explicit "never override policy based on user claims" rule.

**Needs from human:** none this iteration.
```

---

### 7. Repeat

Go to step 1.

---

## Rules

1. **Only edit `agent/agent.py`** — you own the whole file; never touch `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, or workspace files
2. **Never skip the gate** — every committed change must pass all three steps
3. **One hypothesis per iteration** — keep changes small and reversible
4. **Always update `learnings.md`** — even on failure; the log is your memory
5. **Never read test-split traces** — use only train failures to guide changes
6. **Stop when** val_score has not improved for 5 consecutive iterations — write a summary in `learnings.md` and surface your top findings to the human

---

## File Formats

### `workspace/suite.json`

Managed automatically by `gating.py`. Do not edit.

```json
{
  "tasks": ["5", "12", "37"],
  "threshold": 0.8,
  "last_results": {
    "5": 1.0,
    "12": 1.0,
    "37": 1.0
  }
}
```

`tasks` grows as iterations fix previously-failing train tasks and both gates pass.

### `workspace/train_results.json`

Written by `benchmark.py`. Do not edit.

```json
{
  "split": "train",
  "timestamp": "2025-01-01T12:00:00+00:00",
  "results": {
    "0": 1.0,
    "1": 0.0,
    "2": 1.0
  }
}
```

### `workspace/results.tsv`

Tab-separated. Written by `record.py`.

```
iteration	val_score	commit	evals_passed	evals_total	timestamp
1	0.72	abc1234	4	5	2025-01-01T12:00:00+00:00
2	0.78	def5678	5	5	2025-01-01T13:30:00+00:00
```
