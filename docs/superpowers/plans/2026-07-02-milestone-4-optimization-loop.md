# Milestone 4 Optimization Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the previous proposal-history optimizer into a real single-iteration, single-optimized-version closed loop: run baseline, observe failures, ask an LLM for exactly one restricted agent instruction patch, apply it, rerun the same tasks through Harbor/Daytona, accept the patch only if score improves, otherwise revert it, and persist the full iteration history.

**Architecture:** Keep the take-home scope intentionally narrow. `RunService` remains the orchestration layer, `Optimizer` produces one structured JSON instruction proposal, and a new `AgentPatchService` is the only component allowed to edit `agent/agent.py`. The patch service may replace only the top-level `AGENT_INSTRUCTION` string; it may not modify imports, tool execution, Harbor/Daytona execution, filesystem handling, or network logic.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, psycopg, PostgreSQL, pytest, existing OpenAI Responses client usage, existing Harbor/Daytona real runner.

## Global Constraints

- Work only in `/Users/yinfeiwang/workspace/neosigma/auto-harness/.worktrees/takehome-mvp`.
- `AGENTS.md` must remain ignored and must never be staged or pushed.
- Do not use `git push --force` or `git push --force-with-lease`.
- Do not commit `.env`, `workspace/`, raw benchmark jobs, or API keys.
- Keep `max_iterations <= 1`; do not implement multi-round optimization in this milestone.
- Generate at most one optimized version per run: `proposal-1`.
- Do not create multiple optimized versions, rank optimized versions, merge optimized results, or merge patches.
- Do not implement GateEngine, CandidateGraphManager, beam search, suite promotion, or multi-candidate merge logic.
- Real execution remains Harbor-first: `TerminalBenchRunner -> harbor run --env daytona`.
- Only `agent/agent.py` may be modified by the optimization loop.
- Within `agent/agent.py`, only the `AGENT_INSTRUCTION` assignment may be changed.
- If a patch is rejected or rerun fails, restore the original `agent/agent.py` content.
- If a patch is rejected, restore baseline task results so API `score` and final visible task results match the final agent state.
- Use the same `task_ids`, `mode`, `sandbox_provider`, `model`, and `requested_concurrency` for baseline and rerun.
- Accept rule for this MVP is strict: `accepted = rerun_score > baseline_score`.
- Existing SQL ambiguity in `mark_task_running` is already fixed in this branch with `aos_task_results.started_at`; treat that as a completed prerequisite.

---

## Current State Summary

Existing flow:

```text
POST /runs
-> create run
-> create queued task rows
-> run each task through simulated runner or Harbor/Daytona
-> normalize task results
-> create iteration 0
-> if max_iterations > 0, call Optimizer.propose(...)
-> create iteration 1 as proposal_created/proposal_failed
-> mark run succeeded
```

Target flow:

```text
POST /runs
-> create run
-> create queued task rows
-> baseline task execution
-> persist iteration 0 with baseline score
-> if max_iterations == 1:
     -> build failure summary from baseline results
     -> call LLM for exactly one JSON instruction proposal
     -> validate proposal
     -> snapshot initial agent.py
     -> replace only AGENT_INSTRUCTION
     -> py_compile agent/agent.py
     -> reset same task rows to queued
     -> rerun same task_ids
     -> compare rerun_score against baseline_score
     -> accept patch and keep rerun task rows if improved
     -> reject patch, restore agent.py, and restore baseline task rows otherwise
     -> persist iteration 1 with proposal JSON, accepted flag, score, and reason
-> mark run terminal
```

---

## File Structure

Create:

```text
autoharness_service/agent_patch.py
```

Purpose:

- Parse `agent/agent.py` with `ast`.
- Locate exactly one top-level `AGENT_INSTRUCTION` assignment.
- Validate an LLM-produced replacement instruction.
- Replace only that assignment.
- Write snapshots under `workspace/service_runs/<run_id>/agent_versions/`.
- Run `python -m py_compile agent/agent.py`.
- Restore the previous file content on validation or compile failure.

Modify:

```text
autoharness_service/optimizer.py
```

Purpose:

- Change proposal output from free text to structured JSON.
- Add `OptimizationProposal` model/dataclass.
- Add JSON parsing and schema validation for:
  - `hypothesis`
  - `new_agent_instruction`
  - `expected_effect`
  - `risks`

Modify:

```text
autoharness_service/service.py
```

Purpose:

- Factor task execution into a reusable helper for baseline and rerun.
- Run one optimization iteration when `max_iterations == 1`.
- Preserve baseline task-result snapshots before rerun.
- Reset task rows before rerun.
- Accept or reject the patch by score comparison.
- Persist iteration 1 with final status and proposal metadata.

Modify:

```text
autoharness_service/store.py
```

Purpose:

- Add `reset_task_queue(run_id, org_id, task_ids, metadata) -> None`.
- Increase or remove the current `proposal[:4000]` truncation so compact JSON iteration history does not get silently cut.
- Keep `aos_task_results` as the durable queue; do not add Redis.

Modify:

```text
autoharness_service/models.py
```

Purpose:

- Add any lightweight internal models required for patch results.
- Keep public API backwards-compatible.
- Do not add an iteration `metadata` field for this MVP; store compact structured iteration details as JSON text in the existing `proposal` field.

Modify:

```text
tests/service/test_agent_patch.py
tests/service/test_optimizer.py
tests/service/test_service.py
tests/service/test_store.py
tests/service/test_api.py
```

Purpose:

- Prove patch safety.
- Prove structured LLM parsing.
- Prove accepted/rejected optimization behavior.
- Prove rerun uses same task IDs and final API history exposes iteration state.

Modify:

```text
README.md
docs/takehome_mvp_system_design.md
```

Purpose:

- Replace stale Milestone 4 proposal-history wording with the implemented single-optimized-version closed loop.
- Document current limits: one iteration, one optimized version named `proposal-1`, instruction-only patching, no GateEngine.

---

## Task 1: Add Restricted AgentPatchService

**Files:**
- Create: `autoharness_service/agent_patch.py`
- Test: `tests/service/test_agent_patch.py`

**Interfaces:**
- Produces: `AgentPatchService(agent_path: Path | str = "agent/agent.py")`
- Produces: `AgentPatchService.read_instruction() -> str`
- Produces: `AgentPatchService.apply_instruction_patch(new_instruction: str, snapshot_dir: Path | str) -> AgentPatchResult`
- Produces: `AgentPatchService.restore(source: str) -> None`
- Produces: `AgentPatchResult(original_source: str, patched_source: str, original_instruction: str, new_instruction: str, snapshot_paths: dict[str, str])`

- [ ] Write `test_read_instruction_finds_top_level_agent_instruction`.
  - Arrange a temp `agent.py` containing imports, `MAX_STEPS`, and `AGENT_INSTRUCTION = """old"""`.
  - Assert `read_instruction()` returns `old`.

- [ ] Write `test_apply_instruction_patch_changes_only_agent_instruction`.
  - Apply a new multi-line instruction.
  - Assert imports and `TOOLS` source text are unchanged.
  - Assert the file compiles.
  - Assert snapshot files `initial.py` and `proposal-1.py` exist in the provided snapshot directory.

- [ ] Write `test_apply_instruction_patch_rejects_dangerous_content`.
  - Reject strings containing any of:
    - ``` 
    - `import `
    - `from `
    - `os.environ`
    - `subprocess`
    - `open(`
    - `eval(`
    - `exec(`
    - `__`
  - Assert original file content is unchanged after rejection.

- [ ] Write `test_apply_instruction_patch_rejects_missing_or_duplicate_assignment`.
  - Missing `AGENT_INSTRUCTION` raises `ValueError`.
  - Two top-level `AGENT_INSTRUCTION` assignments raise `ValueError`.

- [ ] Implement `AgentPatchService` using `ast.parse`, not regex-only parsing.
  - Use AST node `lineno` / `end_lineno` to replace the whole assignment.
  - Generate replacement source as `AGENT_INSTRUCTION = <json-escaped python string>`.
  - Run `py_compile.compile(str(agent_path), doraise=True)` after writing.
  - On any validation or compile error, write the original source back before raising.

- [ ] Run:

```bash
python -m pytest tests/service/test_agent_patch.py -q
```

Expected: all tests pass.

---

## Task 2: Make Optimizer Return Structured JSON Proposal

**Files:**
- Modify: `autoharness_service/optimizer.py`
- Test: `tests/service/test_optimizer.py`

**Interfaces:**
- Produces: `OptimizationProposal`
- Produces: `Optimizer.propose_instruction_patch(task_results, failure_summary, *, model: str, current_instruction: str) -> OptimizationProposal`
- Produces: `parse_optimizer_json(text: str) -> OptimizationProposal`

- [ ] Write `test_parse_optimizer_json_accepts_required_fields`.
  - Input JSON:

```json
{
  "hypothesis": "The agent stops before verifying artifacts.",
  "new_agent_instruction": "A valid long replacement instruction...",
  "expected_effect": "The agent verifies output files before finishing.",
  "risks": "The agent may spend extra time verifying."
}
```

  - Assert each field is preserved.

- [ ] Write `test_parse_optimizer_json_rejects_free_text`.
  - Input: `hypothesis: maybe this helps`.
  - Expected: `ValueError`.

- [ ] Write `test_build_optimizer_prompt_includes_current_instruction_and_artifact_paths`.
  - Use one failed task result with `trace_path`, `result_path`, and `metadata["artifacts"]`.
  - Assert the prompt contains task id, failure type, error summary, and artifact paths.
  - Assert the prompt asks for JSON only.

- [ ] Modify `build_optimizer_prompt`.
  - Include current `AGENT_INSTRUCTION`.
  - Include compact per-task failure lines.
  - Include trace/result/artifact paths, not full raw logs.
  - Ask for one focused JSON object and no Markdown.

- [ ] Modify `Optimizer`.
  - If `OPENAI_API_KEY` is missing, raise `RuntimeError("OPENAI_API_KEY is not set")`.
  - Call the existing OpenAI Responses API path.
  - Parse `response.output_text` through `parse_optimizer_json`.
  - Do not let arbitrary free text become a patch.

- [ ] Run:

```bash
python -m pytest tests/service/test_optimizer.py -q
```

Expected: all tests pass.

---

## Task 3: Add Store Support For Rerun Queue Reset And Longer Iteration History

**Files:**
- Modify: `autoharness_service/store.py`
- Test: `tests/service/test_store.py`

**Interfaces:**
- Produces: `PostgresStore.reset_task_queue(run_id: str, org_id: str, task_ids: Iterable[str], metadata: dict[str, Any]) -> None`
- Modifies: `PostgresStore.create_iteration(...)` should preserve compact JSON proposal text without silently truncating at 4000 chars.

- [ ] Write `test_reset_task_queue_requeues_terminal_rows_for_same_org_only`.
  - Create a run with task rows.
  - Mark one task passed and one failed.
  - Call `reset_task_queue` with metadata `{"attempt": "proposal-1"}`.
  - Assert both rows become `queued`, reward/failure fields are null, and metadata contains attempt.
  - Assert a different org cannot reset the rows.

- [ ] Write `test_create_iteration_preserves_structured_proposal_json`.
  - Store a JSON proposal string longer than 4000 characters but smaller than 20000 characters.
  - Read it back through `list_iterations`.
  - Assert it is not truncated.

- [ ] Implement `reset_task_queue`.
  - Set:
    - `status = 'queued'`
    - `reward = NULL`
    - `failure_type = NULL`
    - `error_summary = NULL`
    - `trace_path = NULL`
    - `result_path = NULL`
    - `metadata = metadata`
    - `started_at = NULL`
    - `completed_at = NULL`
  - Join through `aos_runs` so `org_id` is enforced.

- [ ] Replace `proposal[:4000]` with a larger explicit cap such as 20000 characters.
  - Raise `ValueError` if proposal text is longer than the cap.
  - Do not silently truncate.

- [ ] Run:

```bash
python -m pytest tests/service/test_store.py -q
```

Expected: all tests pass.

---

## Task 4: Refactor RunService For Baseline And Rerun Execution

**Files:**
- Modify: `autoharness_service/service.py`
- Test: `tests/service/test_service.py`

**Interfaces:**
- Produces internal helper: `RunService._execute_task_rows(run, org_id: str, *, attempt: str) -> list[TaskResultRecord] | None`
- Produces internal helper: `RunService._run_optimization_iteration(run, org_id: str, baseline_results: list[TaskResultRecord], baseline_score: float) -> tuple[float, list[TaskResultRecord]]`

- [ ] Write `test_optimization_accepts_patch_when_rerun_score_improves`.
  - Use a fake runner returning `{"task-fail": 0.0}` on baseline and `{"task-fail": 1.0}` on rerun.
  - Use a fake optimizer returning valid `OptimizationProposal`.
  - Use a temp agent file through `AgentPatchService`.
  - Assert:
    - iteration 0 is `completed`, score `0.0`
    - iteration 1 is `completed`, score `1.0`, accepted `True`
    - final run score is `1.0`
    - final task result is passed
    - `agent.py` contains the new instruction
    - runner saw the same task id twice

- [ ] Write `test_optimization_rejects_patch_and_restores_baseline_when_score_does_not_improve`.
  - Fake runner returns `0.5` then `0.0`.
  - Assert:
    - iteration 1 status is `patch_rejected`
    - accepted is `False`
    - final run score is baseline `0.5`
    - final task rows are restored to baseline rows
    - `agent.py` is restored to the original source

- [ ] Write `test_optimization_records_proposal_failed_when_llm_errors`.
  - Fake optimizer raises `RuntimeError("OPENAI_API_KEY is not set")`.
  - Assert:
    - iteration 1 status is `proposal_failed`
    - accepted is `False`
    - run still succeeds with baseline score
    - no rerun happens

- [ ] Write `test_optimization_records_patch_rejected_when_patch_validation_fails`.
  - Fake optimizer returns a proposal whose `new_agent_instruction` contains `open(`.
  - Assert:
    - iteration 1 status is `patch_rejected`
    - accepted is `False`
    - run still succeeds with baseline score
    - no rerun happens

- [ ] Refactor `execute_run`.
  - Keep durable task lifecycle behavior from Milestones 1-3.
  - Baseline execution should call `_execute_task_rows(..., attempt="baseline")`.
  - If any task rows are still not terminal, return and let polling resume.
  - Create/update iteration 0 after baseline task rows are terminal.
  - If `max_iterations == 0`, mark run succeeded with baseline score.
  - If `max_iterations == 1`, call `_run_optimization_iteration`.
  - Mark run succeeded with the final accepted score if accepted, otherwise baseline score.

- [ ] Implement `_run_optimization_iteration`.
  - Build `failure_summary` from baseline results.
  - If baseline score is `1.0`, create iteration 1 with status `skipped_no_failures`, accepted `False`, score `1.0`, and do not call the LLM.
  - If failures are hard infrastructure only (`runner_failed`, `runner_timeout`) and no task has a numeric reward, create iteration 1 with status `proposal_failed`, accepted `False`, and do not patch.
  - Read current instruction through `AgentPatchService.read_instruction`.
  - Call `Optimizer.propose_instruction_patch`.
  - Create iteration 1 as `proposal_created` with compact JSON proposal.
  - Apply patch with snapshot directory `workspace/service_runs/<run_id>/agent_versions`.
  - Create iteration 1 as `patch_applied`.
  - Call `store.reset_task_queue(..., metadata={"source": "queued", "attempt": "proposal-1"})`.
  - Rerun tasks through `_execute_task_rows(..., attempt="proposal-1")`.
  - Compare scores.
  - If improved:
    - keep patched agent
    - keep rerun task rows
    - create iteration 1 as `completed`, accepted `True`, score `rerun_score`
  - If not improved:
    - restore original agent source
    - restore baseline task rows through `replace_task_results`
    - create iteration 1 as `patch_rejected`, accepted `False`, score `rerun_score`

- [ ] Store compact proposal metadata as JSON in `proposal`.
  - Include:
    - `hypothesis`
    - `expected_effect`
    - `risks`
    - `baseline_score`
    - `rerun_score`
    - `accepted`
    - `decision_reason`
    - `changed_section: AGENT_INSTRUCTION`
    - `snapshot_paths`
    - compact baseline and rerun task summaries
  - Do not store raw API keys, `.env`, or full trace contents.

- [ ] Run:

```bash
python -m pytest tests/service/test_service.py -q
```

Expected: all tests pass.

---

## Task 5: Expose Iteration History Clearly Through API And Client

**Files:**
- Modify: `test_client.py`
- Test: `tests/service/test_api.py`
- Test: `tests/service/test_test_client.py`

**Interfaces:**
- Preserves: `GET /runs/{run_id}/iterations`
- Preserves: existing iteration response shape; `proposal` contains compact JSON text for iteration 1.

- [ ] Write `test_api_iterations_show_completed_optimization_attempt`.
  - Submit a simulated run with `max_iterations=1` using fake service pieces.
  - Execute it.
  - Assert `/runs/{run_id}/iterations` returns iteration 0 and iteration 1.
  - Assert iteration 1 includes `accepted` and final score.
  - Assert iteration 1 `proposal` parses as JSON and contains `baseline_score`, `rerun_score`, `decision_reason`, and `changed_section`.

- [ ] Update `test_client.py` summary output.
  - Keep current result summary.
  - Print iteration statuses in order.
  - Print accepted/rejected status for iteration 1.
  - Do not print raw full prompt or full instruction unless explicitly requested later.

- [ ] Run:

```bash
python -m pytest tests/service/test_api.py tests/service/test_test_client.py -q
```

Expected: all tests pass.

---

## Task 6: Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `docs/takehome_mvp_system_design.md`
- Optional: create `docs/superpowers/context/2026-07-02-milestone-4-optimization-loop.md`

- [ ] Update README take-home section.
  - State Milestone 4 is now implemented as a single-optimized-version loop.
  - Explain that only `AGENT_INSTRUCTION` can be changed.
  - Explain accept/reject rule.
  - Explain that rejected patches are reverted.
  - Keep "not implemented" list honest:
    - no multi-round loop
    - no multiple optimized versions
    - no merge of optimized versions or optimized results
    - no GateEngine
    - no candidate graph
    - no suite promotion

- [ ] Update setup notes.
  - Real M4 requires:
    - `OPENAI_API_KEY`
    - `DAYTONA_API_KEY`
    - `DATABASE_URL`
    - Harbor CLI
  - `.env` still must be sourced manually unless a later task adds dotenv loading.

- [ ] Update system design doc.
  - Add:

```text
baseline -> failure summary -> structured LLM proposal -> restricted patch -> rerun -> score compare -> accept/revert
```

- [ ] Run:

```bash
rg -n "<stale Milestone 4 proposal-history phrases>" README.md docs
```

Expected: no stale statement claims M4 stops at proposal history after implementation.

---

## Task 7: Full Verification

**Files:**
- Verify: `agent/agent.py`
- Verify: `autoharness_service/*`
- Verify: `tests/service/*`
- Verify: `test_client.py`

- [ ] Run focused tests:

```bash
python -m pytest tests/service/test_agent_patch.py tests/service/test_optimizer.py tests/service/test_service.py -q
```

- [ ] Run full service tests:

```bash
python -m pytest tests/service -q
```

- [ ] Run formatting checks:

```bash
python -m black --check agent/agent.py benchmark.py autoharness_service tests/service test_client.py scripts/durable_queue_restart_check.py
```

- [ ] Run import sorting checks:

```bash
python -m isort --check-only agent/agent.py benchmark.py autoharness_service tests/service test_client.py scripts/durable_queue_restart_check.py
```

- [ ] Run diff whitespace check:

```bash
git diff --check
```

- [ ] Run optional local API smoke with fake or simulated optimizer pieces only if real keys are not available.

- [ ] Run real Harbor/Daytona smoke only after confirming the shell has:

```bash
printenv OPENAI_API_KEY
printenv DAYTONA_API_KEY
printenv DATABASE_URL
```

Expected: each variable is set in the same shell that starts Uvicorn.

---

## Edge Cases To Preserve

- If OpenAI is not configured, the run should not crash after baseline; iteration 1 should record `proposal_failed`.
- If Daytona/Harbor fails before producing task results, do not patch the agent for pure runner failures.
- If the LLM returns non-JSON, do not patch.
- If the LLM returns dangerous text, do not patch.
- If `py_compile` fails, revert the file immediately.
- If rerun score ties baseline score, reject the patch.
- If rerun score is worse than baseline score, reject and restore baseline task rows.
- If rerun raises, reject and restore baseline task rows.
- If baseline score is already `1.0`, skip optimization.
- If baseline has mixed pass/fail rows, still attempt one patch.
- If task results include artifact paths, preserve compact paths in iteration proposal JSON, but do not inline full log files.

---

## Review Checklist

- Does any code path allow the LLM to edit files other than `agent/agent.py`?
- Does any code path allow the LLM to edit anything except `AGENT_INSTRUCTION`?
- Does one run create only one optimized version, `proposal-1`?
- Is there no merge/ranking logic for optimized versions?
- Does rejection restore both the agent file and final visible task rows?
- Does acceptance leave the patched agent file in place?
- Does the final `run.score` match the final visible agent state?
- Does `/runs/{run_id}/iterations` show baseline and optimization attempt?
- Are API keys and `.env` absent from responses, metadata, logs, and docs examples?
- Are real Harbor/Daytona reruns still serialized by the existing process-local semaphore?
- Do tests prove the patch safety boundary rather than only happy-path behavior?
