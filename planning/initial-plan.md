Principles (guardrails)
Do not change run_gate()’s logic — only how BenchmarkRunner is built and what task_ids mean (SWE instance IDs as strings).
One scoring contract everywhere: run() → dict[str, float], values in [0,1], resolved = 1.0, else 0.0 (works with your >= 0.5 rule).
Do not reimplement SWE’s test runner — use the official SWE-Bench harness (Docker) for grading after you have a patch per instance (predictions file or API the harness expects).
Split “patch generation” and “grading” so you can debug them separately; the runner orchestrates both.
Phase A — Config + factory (small, mechanical)
Goal: benchmark.py / gating.py stop assuming tau only.

Add benchmark: tau | swe to experiment_config.yaml (+ tau-only vs swe-only keys so validation is clear).
Add make_train_runner(cfg) / make_gate_runner(cfg) (or one make_runner(cfg, role) with split vs gate_split) that returns TauBenchRunner or SWEBenchRunner.
Tau path: behavior unchanged when benchmark: tau.
Done when: switching YAML switches runners without editing gating.py’s step logic.

Phase B — SWEBenchRunner (the only new big piece)
Goal: Same interface as TauBenchRunner.

Responsibilities:

Load tasks

Instances from Hugging Face (e.g. Lite) or a cached JSONL on disk.
Build the list for run(task_ids=None) = full configured set; run(task_ids=[...]) = filter by instance_id (same strings as suite.json / train_results.json).
Produce patches (per instance)

Minimal approach: one module you own (e.g. agent/swe_agent.py + core for prompts) that runs an LLM + tools loop until it emits a final patch; OR a subprocess to SWE-agent with env/model from config.
Pick one for v1; don’t build both until one works.
Write predictions in the format the official harness expects (typically a JSONL of instance_id + patch/diff).

Invoke official evaluation (subprocess: python -m swebench.harness.run_evaluation or the project’s documented CLI) with paths to predictions + dataset/slice.

Parse harness output (report JSON/JSONL) into {instance_id: 1.0|0.0}.

Explicitly out of scope for v1: fancy caching, distributed orchestration, custom test harnesses.

Done when: a dry run on 2–3 Lite instances returns sensible {id: float} and val_score matches mean resolve rate.

Phase C — prepare.py (environment only)
Goal: Fail fast before long runs.

For benchmark: swe: check Docker available, disk space hint, HF cache / dataset access if you load from HF, document AGENT_MODEL / API keys.
Do not duplicate tau checks when in swe mode.
Phase D — Workspace + gating consistency (no new logic)
Goal: Same files, same meaning.

benchmark.py CLI still writes workspace/train_results.json with results = map of instance_id → float (same schema as tau, different key shape).
suite.json tasks = list of instance_id strings for regression.
gating.py: unchanged Steps 1–3; it already uses train_runner / gate_runner and >= 0.5.
Operational rule: changing benchmark or dataset invalidates old suite.json / train_results.json — document that (one paragraph), don’t build automatic migration unless you need it.

Phase E — Docs (PROGRAM.md) — one parallel section
Goal: Autonomous agent knows what to edit and what not to touch.

Editable: agent/core.py, agent/swe_agent.py (or chosen patch-generation module), and keep agent/agent.py as re-exports if you want a stable import story.
Frozen: benchmark.py runner contract, gating.py, record.py (unless you add a column — optional).
Analysis: replace “tau traces only” with “train instance logs only; no peeking at gate/holdout artifacts.”
Ordering (short checklist)
Config + make_runner + SWEBenchRunner stub that returns zeros (validates wiring).
Real load from Lite + real harness with oracle/gold patches if available for a smoke test or skip to LLM once predictions format is verified.
Patch generation (one strategy only).
prepare.py swe branch.
PROGRAM.md swe notes.
Run full gate on a small YAML-defined subset before “all Lite.”
What this deliberately avoids (anti–over-engineering)
No second abstraction over BenchmarkRunner beyond the factory.
No requirement that HarnessAgent and SWE share a single Python class — only shared core prompts if useful.
No fork of SWE-agent unless subprocess isn’t enough.
No custom Dockerfiles until official images fail you.
