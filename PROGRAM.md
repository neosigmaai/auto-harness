# auto-harness — Agent Program

Shared instructions for the self-improvement loop. **Mode-specific details** live in:

- **`TAU-PROGRAM.md`** — when `benchmark: tau` (tau-bench)
- **`SWE-PROGRAM.md`** — when `benchmark: swe` (SWE-Bench)

Read the doc that matches `experiment_config.yaml` → `benchmark`.

---

## What You Are Doing

You are an autonomous coding agent optimizing code under `agent/` to perform better on a benchmark. You run a tight loop:

```
run benchmark → analyze failures → improve agent → gate → commit → repeat
```

Set **`benchmark: tau`** or **`benchmark: swe`** in `experiment_config.yaml`, then follow **TAU-PROGRAM.md** or **SWE-PROGRAM.md** for edit targets, what to read after a run, and leakage rules.

Everything outside `agent/` (except `workspace/learnings.md` as noted) is infrastructure — do not modify it unless the human asks.

---

## Files You Own

| File | Purpose |
|------|---------|
| `agent/agent.py` | Stable imports — tau (`HarnessAgent`) + SWE helpers (`generate_patch`, etc.) |
| `agent/core.py` | Shared instructions and `build_*_system_prompt` helpers |
| `agent/tau_agent.py` | tau-bench `LLMAgent` loop — **tau mode** |
| `agent/swe_agent.py` | SWE-Bench patch generation (`generate_patch`) — **SWE mode** |
| `workspace/learnings.md` | Persistent learnings log — **append after every iteration** |

| File | Purpose |
|------|---------|
| `workspace/results.tsv` | Iteration history — written by `record.py` after each successful gate |

**Read-only workspace files** (managed automatically — do not edit):

| File | Purpose |
|------|---------|
| `workspace/suite.json` | Regression suite — tasks promoted after successful gates |
| `workspace/train_results.json` | Last train benchmark results — written by `benchmark.py` |

Mode-specific read-only paths (predictions, logs, traces) are listed in **TAU-PROGRAM.md** and **SWE-PROGRAM.md**.

---

## Commands

| Command | What it does |
|---------|-------------|
| `python benchmark.py` | Run the train benchmark; print per-task pass/fail; save `workspace/train_results.json` |
| `python benchmark.py --task-ids …` | Override task IDs (see mode doc for ID format) |
| `python gating.py` | Three-step gate. Exit 0 = all clear, commit and record |
| `python record.py --val-score X --evals-passed N --evals-total M` | Append iteration result |
| `python prepare.py` | Initialize workspace (run once at start) |

SWE-only helper (mini-swe-agent batch baseline): see **SWE-PROGRAM.md**.

---

## The Loop

### 1. Run Benchmark

```bash
python benchmark.py
```

Read stdout: which tasks failed (task ID, reward). Results are saved to `workspace/train_results.json`.

---

### 2. Analyze Failures

Follow **TAU-PROGRAM.md** (tau) or **SWE-PROGRAM.md** (SWE) — what to read, what not to leak from held-out splits, and how to classify failures.

Append findings to `workspace/learnings.md`.

---

### 3. Improve Agent

Follow **TAU-PROGRAM.md** or **SWE-PROGRAM.md** for which files and knobs to change.

Make one focused change per iteration.

**Do not modify** `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, or workspace JSON/TSV managed by the loop (except `learnings.md`).

---

### 4. Gate

```bash
python gating.py
```

Three steps:

- **Step 1 — Regression suite**: tasks in `suite.json` on the **train** split; pass rate ≥ threshold.
- **Step 2 — Gate benchmark**: **gate** split (`gate_split` in config), optionally a subset for SWE (`swe_gate_task_ids` / `swe_gate_match_default_task_ids` — see **SWE-PROGRAM.md**). `val_score` must be ≥ best in `results.tsv` (excluding `commit=baseline` rows).
- **Step 3 — Suite promotion** *(if Steps 1+2 pass)*: re-run previously failing train tasks; newly passing ones are added to `suite.json`.

**Exit 0** → Record. **Exit 1** → revert, e.g. `git checkout -- agent/`. If the same hypothesis fails three times, try something else.

---

### 5. Record

After exit 0:

```bash
git add agent/
git commit -m "improve: <what changed and why>"
python record.py --val-score <val_score from Step 2> --evals-passed <n> --evals-total <m>
```

`evals-passed` / `evals-total` are from Step 1 (regression suite).

---

### 6. Update Learnings

After every iteration (pass or fail), append to `workspace/learnings.md`:

- What you tried and what happened
- Patterns confirmed
- What worked
- Needs from human

---

### 7. Repeat

Go to step 1.

---

## Rules

1. **Only edit `agent/` and `workspace/learnings.md`** — never `benchmark.py`, `gating.py`, `record.py`, `prepare.py`, `workspace/suite.json`, or `workspace/train_results.json`.
2. **Never skip the gate** — every committed change must pass all gate steps.
3. **One hypothesis per iteration** — small, reversible changes.
4. **Always update `learnings.md`** — even on failure.
5. **No data leakage** — see **TAU-PROGRAM.md** and **SWE-PROGRAM.md** for train vs test / gate split rules.
6. **Stop when** `val_score` has not improved for five consecutive iterations — summarize in `learnings.md` and surface findings to the human.

---

## File Formats

### `workspace/suite.json`

Managed by `gating.py`. Do not edit.

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

### `workspace/train_results.json`

Written by `benchmark.py`. Do not edit.

```json
{
  "split": "train",
  "timestamp": "2025-01-01T12:00:00+00:00",
  "results": {
    "0": 1.0,
    "1": 0.0
  }
}
```

Task keys are **numeric strings** for tau (e.g. `"5"`) and **string instance IDs** for SWE (e.g. `django__django-12345`). See mode doc for your benchmark.

### `workspace/results.tsv`

Tab-separated; written by `record.py`.

```
iteration	val_score	commit	evals_passed	evals_total	timestamp
1	0.72	abc1234	4	5	2025-01-01T12:00:00+00:00
```
