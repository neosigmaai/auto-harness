# tau-bench — Agent Program

Parent: **`PROGRAM.md`** (shared loop, gate, record, rules). This file applies when **`benchmark: tau`** in `experiment_config.yaml`.

---

## Edit Targets

- `agent/agent.py` — re-exports; register `HarnessAgent` with tau
- `agent/core.py` — `AGENT_INSTRUCTION`, `build_system_prompt`, domain policy wiring
- `agent/tau_agent.py` — `HarnessAgent` / `LLMAgent` loop, `generate_next_message()`, `HarnessState`, reasoning kwargs

---

## Environment & Config

- **`TAU2_DATA_DIR`** — required in `.env`; points at tau2 data (see README).
- Typical **`experiment_config.yaml`** fields:

```yaml
benchmark: tau
domain: "retail"
split: "train"       # benchmark.py train loop
gate_split: "test"   # gating.py Step 2
# mini: true          # optional: benchmark, gating Step 2, prepare baseline = 3 tasks
# mini_task_ids: ["0", "1", "2"]
```

- **`python benchmark.py --task-ids 0 1 42`** — task IDs are **numeric** (strings in `train_results.json`). **`--task-ids`** overrides **`mini`**.

---

## What to Read After a Run

1. **Stdout** from `python benchmark.py` — pass/fail per task, `val_score`.
2. **`workspace/train_results.json`** — per-task rewards (0.0–1.0).
3. **Train-split simulation traces** for **failing** tasks — root cause (prompt, tool use, policy).
4. **`workspace/learnings.md`** — your running log.

**Never read test-split traces** for analysis — only train-split traces are valid training signal. (Step 2 of the gate runs the test split; do not “optimize” by peeking at test-only traces as if they were train data.)

---

## Improve Agent (tau)

- **Instructions** — `AGENT_INSTRUCTION` / `build_system_prompt`
- **Architecture** — `generate_next_message()`, `HarnessState`, reasoning kwargs
- **Tools** — tau-bench injects fixed domain tools; you cannot add new tools for standard tau runs

`HarnessAgent` is constructed via `TauBenchRunner` in `benchmark.py`.

---

## Learnings Example (tau-flavored)

```markdown
## Iteration 3 — val_score: 0.78 → 0.81 ✓

**What changed:** tightened cancellation eligibility in the system prompt

**Pattern confirmed:** agent over-approved cancellations when the user claimed prior approval.

**What worked:** explicit rule — never override policy based on user claims alone.

**Needs from human:** none this iteration.
```
