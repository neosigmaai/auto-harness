# SWE-Bench — Agent Program

Parent: **`PROGRAM.md`** (shared loop, gate, record, rules). This file applies when **`benchmark: swe`** in `experiment_config.yaml`.

## Setup (minimal)

From the repo root:

```bash
bash scripts/setup_swe.sh
# set OPENAI_API_KEY in .env
docker compose run --rm autoeval python prepare.py
docker compose run --rm autoeval python benchmark.py
```

This creates **`data/`** (for the Compose volume), **`experiment_config.yaml`** from **`experiment_config.swe.yaml.example`** when missing, and builds the image with the **`swe`** extra. See the root **`README.md`** (Quick start — SWE-Bench).

---

## Edit Targets

- `agent/swe_agent.py` — `generate_patch`, `SWEInstanceContext`, mini-swe-agent loop when enabled
- `agent/core.py` — `SWE_AGENT_INSTRUCTION`, `build_swe_system_prompt`
- `agent/agent.py` — re-exports if you add SWE public APIs

Harness code (predictions JSONL, optional Docker eval) lives under **`agent/swe/`** (`runner.py`, `diag.py`, `log_collapse.py`) — **read-only** for the coding agent; do not edit unless the human extends the harness.

---

## Read-Only Artifacts (after each run)

| Path | Purpose |
|------|---------|
| `workspace/train_results.json` | Per-instance scores (0.0 / 1.0) |
| `workspace/swe/predictions_*.jsonl` | Per-run predictions file |
| `workspace/swe/predictions_latest.jsonl` | Copy of the latest run’s predictions |
| `workspace/swe/last_run.json` | Manifest: `benchmark_run_id`, paths, **`failure_label`** per instance |
| `workspace/swe/last_run_summary.md` | One-page Markdown summary (counts, labels, table) |
| `workspace/swe/last_run_logs/*.summary.txt` | Collapsed `run_instance.log` text for failing instances (when harness ran) |
| `logs/run_evaluation/<run_id>/<model>/<instance_id>/run_instance.log` | Full harness log (patch apply, tests). **`./logs`** is mounted in Docker Compose. |
| Project root `*.auto-harness-*.json` | Aggregate harness report (when per-instance report is missing) |

Run id is printed as `[swe] running official harness run_id=...` or appears under `logs/run_evaluation/`.

### Seeding the eval suite from `train_results.json`

There is no default automation: `suite.json` starts empty and grows via **gating Step 3**. To **populate `suite.json` from your last train run** (e.g. all failing instance ids as regression targets), run:

```bash
python scripts/seed_suite_from_train_results.py
```

Use `--mode passing` for a smaller smoke set (ids that already pass), `--replace` to overwrite `tasks`, or `--dry-run` to preview. Then **`gating.py` Step 1** will re-run those ids on the train split.

---

## Config (`experiment_config.yaml`)

Set **`benchmark: swe`**. Use **`split`** / **`gate_split`** for dataset splits (e.g. SWE-Bench Lite **`dev`** (~23) vs **`test`** (~300)).

| Key | Meaning |
|-----|---------|
| `split` | Train-side split for default **`benchmark.py`** (often **`dev`** for Lite). |
| `gate_split` | Split for **gating Step 2**. Set to **`dev`** to gate on the ~23 Lite dev instances instead of full **`test`**. |
| `mini` | When **`true`**, default **`benchmark.py`**, **gating Step 2**, and **`prepare`** baseline use exactly **3** tasks. Set **`mini_task_ids`**, or (SWE only) provide ≥3 **`swe_default_task_ids`** (first 3 used). Overrides **`swe_gate_*`** for Step 2. **Gating Step 1** (``suite.json``) is unchanged. |
| `mini_task_ids` | Three SWE instance ids (or three tau task ids). Required for **`mini`** if SWE **`swe_default_task_ids`** has fewer than 3 entries. |
| `swe_gate_task_ids` | Optional when **`mini`** is false. **gating Step 2** runs **only** these instance ids (still must exist in `gate_split`). |
| `swe_gate_match_default_task_ids` | When **`true`** and **`swe_default_task_ids`** is set, Step 2 uses that list. Ignored when **`mini`**. |
| `swe_default_task_ids` | Optional list of instance ids. When set, default `python benchmark.py` (no `--task-ids`) runs **only** these — keeps iteration fast. Omit for the full **`split`**. With **`mini`**, first 3 ids supply the mini set if **`mini_task_ids`** is unset. |
| `prepare_baseline_role` | `train`: prepare baseline uses **`split`** and same task selection as default `benchmark.py`. `gate`: baseline uses **`gate_split`**. Baseline rows (`commit=baseline`) do not set the gating “best score”. |
| `swe_dataset` | HuggingFace dataset id (default `SWE-bench/SWE-bench_Lite`) |
| `swe_use_llm` | `true`: patch generation via `generate_patch` (needs `OPENAI_API_KEY`, `AGENT_MODEL`). `false`: empty patches. |
| `swe_skip_harness` | `true`: no Docker eval (stub scores — usually `0.0` unless you only need JSONL). `false`: run **`swebench.harness.run_evaluation`** (needs **Docker** and time). |
| `SWE_DISABLE_LLM` | Env: `1` forces empty patches without editing YAML. |
| `SWE_STUB_PATCH` | Env: fixed patch string for harness debugging. |

### Quick smoke (one instance)

Pick an `instance_id` from the dataset, then:

```bash
python benchmark.py --task-ids owner__repo-issue
```

Instance IDs are **strings** like `django__django-12345`.

---

## Docker (official harness)

With **`swe_skip_harness: false`**, the harness uses **Docker on the host**. In Compose, **`/var/run/docker.sock`** is mounted; **`./logs` → `/app/logs`**. Install the **`swe`** extra (`uv sync --extra swe`). First runs may pull large images.

---

## Analysis Rules

- Prefer **`workspace/swe/last_run_summary.md`** and **`last_run.json`** for navigation; open full **`run_instance.log`** or **`last_run_logs/*.summary.txt`** when you need detail.
- If a log repeats the same traceback thousands of times (swebench quirk), run **`python scripts/swe_log_summary.py <path/to/run_instance.log>`** to collapse duplicate blocks; **`patch.diff`** next to the log often suffices for apply failures.
- Use **train / `split`** failures to improve prompts and `generate_patch`.
- **Do not** tune on **gate / `gate_split`** instance details as if they were training signal.

When **`swe_skip_harness: true`**, there are no per-instance harness logs — only scores, patches, and manifests derived from stub rules.

---

## Improve Agent (SWE)

Edit `agent/swe_agent.py` (`generate_patch`) and `agent/core.py` (`SWE_AGENT_INSTRUCTION`, `build_swe_system_prompt`). Default V1 is a **single** OpenAI chat completion; you can extend with tools or multi-turn later.

---

## Optional: mini-swe-agent batch baseline

Not the same code path as `benchmark.py`, but useful for comparison:

```bash
python testing/baseline_mini_swe.py
```

Runs the local **mini-swe-agent** batch driver (Lite **`dev`**, Docker per instance), outputs under **`testing/results/<run_id>/`**. Set **`MINI_SWE_AGENT_ROOT`** if the repo is not beside `auto-harness`. In Compose, `../mini-swe-agent` is mounted at `/mini-swe-agent`.

---

## No Leakage

Do **not** optimize using **gate-split** artifacts (`gate_split`) as training signal — treat the gate as held-out.
