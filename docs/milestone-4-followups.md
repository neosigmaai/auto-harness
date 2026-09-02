# Milestone 4 — follow-up work

Everything here was found *after* M4 was implemented, reviewed and merged-ready: by the
whole-branch review, or by running the loop for real against Harbor. Nothing here blocks
the feature working — it works — but the P1 items block *trusting its results*.

Each item states the problem with evidence, and the deliverable that closes it.

---

## Evidence base: the first real run

Job `e34478e7-61cc-4721-b548-930f5213ac52` — one task (`polyglot-c-py`), `gpt-4.1-mini`,
`max_iterations=3`, real Docker/Harbor execution, real `LLMImprover`.

| Iteration | Version | Based on | Score | Improved |
|---|---|---|---|---|
| 0 | v0 (baseline) | — | 0.0 | — (sets initial best) |
| 1 | v1 | v0 | 0.0 | no |
| 2 | v2 | **v0** | 0.0 | no |

`stop_reason=max_iterations`, `best=v0 (0.0)`.

**What this proves works:** the full cycle on real infrastructure — spec materialisation,
Harbor execution of the spec-driven agent, trace capture to the artifact store, a real LLM
producing a schema-valid proposal, re-evaluation, and stopping. Note iteration 2 is based on
**v0, not v1** — backtracking behaved exactly as designed once v1 failed to improve.

**What it does not show:** any improvement. Three attempts, no movement on this task. That
is an honest negative result, and the P1 items below are the most likely reason.

---

## P1 — Improver signal quality (fix before trusting any optimization result)

### F1. The improver never sees the task statement

**Problem.** `_render_trace` (`api/services/improver.py:105`) keeps `data[-N:]` — the *tail*
of the trace. The task statement is the *second* message in an agent trace
(`{"role":"user","content":"Task:\n<instruction>"}`), i.e. at the head.

Measured on iteration 0 of the run above: the trace held **101 messages**; the improver
received the last ~12 with `...[89 earlier messages omitted]...`. The omitted block included
the task statement. The improver was asked to diagnose a failure without being told what the
task required.

This is a spec defect, not an implementation slip — design §7 says "the tail of each
`trace.json`" and nothing else supplies the instruction. Grep confirms: no reference to the
task instruction anywhere in `improver.py`.

**Consequence, visible in the run.** Iteration 1's proposal blamed missing utilities
(`file: command not found`) and added `apt-get` instructions — because a shell error was the
only concrete thing in its window. Command errors are loud in a trace; "the model never
understood the polyglot trick" is silent. By iteration 2 it had inferred the real difficulty
from file paths alone, which suggests it would have got there sooner with the statement.

**Deliverable.**
- The improver context includes the task statement for every failing task, in full, in a
  labelled section that cannot be truncated away (place it with the mandatory prefix, before
  the budget-gated failure details).
- Prefer plumbing the instruction through explicitly (Harbor's per-trial metadata, or the
  first user message of the trace) rather than relying on the trace tail happening to contain
  it.
- Test: build a context from a trace of >100 messages and assert the task statement appears in
  the output; assert it survives a budget small enough to drop all failure details.
- Update design §7 to state that the head (task statement) is mandatory and only the trace
  *tail* is budget-gated.

### F2. The improver only ever sees `"Verifier failed"`

**Problem.** `remarks` reaches the improver as the generic string produced by
`reward_to_task_status` (`api/services/runner.py`). Harbor's per-trial `result.json` typically
carries a far more specific verifier explanation, and it is never plumbed through. Related:
M6 — `result.json` is not even copied into the artifact store, so design §9 ("copies each
trial's `trace.json` and harbor's `result.json`") is half met.

**Deliverable.**
- `StepExecutor` copies each trial's `result.json` into the artifact store via the existing
  unused `result_key()` helper.
- Any verifier message/diagnostic available there is surfaced as that task's `remarks`, or as
  its own labelled line in the failure-details block, replacing the generic string.
- Test: a trial whose `result.json` carries a verifier message produces a context containing
  that message rather than `"Verifier failed"`.

---

## P2 — Optimization quality and cost control

### F3. Prompt accretion with no length pressure

**Problem.** Proposals replace the system prompt wholesale, and the improver is told to make
"ONE focused change". In practice that biases toward *adding* rules. Measured: 479 → 828
chars (+73%) in a single iteration; v2 grew again. The only ceiling is `AgentSpec`'s
20,000-char bound.

This compounds twice: the prompt is resent on every one of 80–100 agent steps, so bloat is a
real recurring token cost; and a rule-salad prompt tends to get *less* effective, not more.

**Deliverable.**
- `IMPROVER_SYSTEM_PROMPT` gains explicit length pressure — prefer rewriting or replacing an
  existing rule over appending; keep the prompt within a stated budget.
- Optionally a soft cap enforced in code (reject or re-prompt when a proposal exceeds, say,
  1.5× the baseline length) — decide whether that is worth the retry cost.
- The iteration history shown to the improver includes each version's prompt length, so the
  model can see the trend it is contributing to.

### F4. Nothing penalises making iterations more expensive

**Problem.** The improver may raise `max_steps` to 200 and `exec_timeout_sec` to 1200. It went
80 → 100 on its very first proposal. "Try harder" is the easiest change to reach for and the
most expensive to run — each iteration is a full benchmark sweep.

**Deliverable.**
- Decide and document a policy: either the improver is told that raising step/time budgets has
  a real cost and should be a last resort, or those two keys are removed from
  `_ALLOWED_CONFIG_KEYS` so only the prompt is mutable.
- If they stay mutable, record per-iteration cost (agent tokens, wall-clock) on the step row
  and expose it in the API, so a "score improved but cost tripled" outcome is visible.

### F5. Single-trial binary scoring makes `min_delta` inert

**Problem.** With one task, mean reward ∈ {0.0, 1.0}. `min_delta=0.01` cannot filter anything:
any pass is a +1.0 jump. On a stochastic agent one lucky pass becomes `best` permanently, and
one unlucky failure discards a genuinely better version. The design acknowledges run-to-run
variance but the loop still takes exactly one sample per iteration.

**Deliverable.**
- Document the minimum sensible task-set size for a trustworthy job (the 16-task default is
  fine; 1–2 tasks is a smoke test, not an experiment), and surface a warning when a job is
  created with fewer than N tasks.
- Optionally support `trials_per_iteration` (>1) with the score as the mean across trials —
  this is the real fix for variance, at linear cost. Decide whether it is worth it before
  quoting any score as evidence of improvement.

### F6. One bad LLM response can end a job early

**Problem.** `LLMImprover` retries once, then raises `ImproverError` → `failed_improve` → the
job completes. A transient formatting failure therefore ends a multi-iteration job at
iteration 1. The policy is right (the best-so-far agent is returned) but the trigger is
thin. Related: M13 — on the retry the previous assistant turn is appended as
`{"role":"assistant","content":self.last_response}`, and when attempt 0 failed inside
`_extract_content` that value is still `""`; some providers reject an empty assistant turn,
turning a recoverable rejection into a hard failure.

**Deliverable.**
- Fix M13 first: never append an empty assistant turn on retry.
- Then decide whether a failed improve step should end the job or be tolerated once (e.g. skip
  that iteration and continue). If it should end the job, say so in the design; if not,
  implement a bounded skip.

---

## P3 — Robustness (parked by the final review, not merge blockers)

### F7. Orphaned `running` runs after a worker dies (review finding I3)

**Problem.** Job-owned runs are created `running` with `claimed_at = NULL` — deliberately, so
the legacy stale sweep cannot reset them and re-expose them to theft (the C1 fix). The
consequence is that a SIGKILLed worker leaves its run `running` forever: the sweep requires
`claimed_at IS NOT NULL` (`api/store.py:358`), so nothing reclaims it. Design §6's
`superseded` marking is unimplemented, and `StepRecord.run_id` (`api/job_store.py:61`) is dead
in production (M2).

**Deliverable.**
- A requeued evaluate step marks its previous, orphaned run `failed` with
  `error_code="superseded"` (as design §6 already specifies) and points the step at the fresh
  run — which also gives `StepRecord.run_id` its purpose.
- Test: force a stale requeue of an evaluate step whose run is `running`; assert the old run
  ends `superseded` and exactly one run is live for that step.

### F8. An unusable head-of-queue row can block every job (review finding I5)

**Problem.** `claim_next_step` (`api/job_store.py:577-606`) takes one candidate with no
`jobs.status` filter and no fallback. A queued step belonging to an already-terminal job would
be selected, roll back, and be selected again forever — starving every other job. Latent
today because nothing produces that state; reachable the moment job cancellation (design §12)
lands.

**Deliverable.**
- Join `jobs` in the claim query and require `jobs.status IN ('queued','running')`.
- Test: a queued step under a `cancelled` job is skipped and a healthy job's step is claimed
  instead.
- Do this *before* implementing cancellation.

---

## P4 — Hygiene, correctness guards, and dead code

Grouped; none is urgent, several are five-minute fixes worth doing while nearby.

**Drift guards (highest value in this group).**
- **M7** — AgentSpec's bounds are hardcoded in three places:
  `improver.py:270` (`_ALLOWED_CONFIG_KEYS`), `improver.py:287-289` (the system prompt's
  literal "max_steps (1-200), max_output_chars (500-100000), exec_timeout_sec (10-1200)") and
  `agent_spec.py:41-43`. Change a bound and the improver's prompt starts lying to the LLM, so
  every proposal at the old boundary burns a retry. *Deliverable:* derive the allowlist and
  the prompt text from `AgentSpec.model_fields` metadata, or add a test asserting they agree.
- **M8** — `tests/test_spec_agent_runtime.py:29-34`'s `SPEC_KEYS` is hand-written, so a sixth
  `AgentSpec` field would silently never reach `spec_loader.default_spec()` with the test still
  green. *Deliverable:* derive it from `AgentSpec.model_fields`.

**Invariant hardening.**
- **M10** — `steps` has no `UNIQUE (job_id, type, iteration)` constraint; the "never two
  successors" invariant rests entirely on the behavioural guard at `job_store.py:663-665`.
  That guard is correct and correctly placed under both row locks, but a DB constraint would
  make the invariant unfalsifiable rather than merely tested. *Deliverable:* add the
  constraint (note it must tolerate the legitimate requeue-and-recreate path).
- **M11** — `IMPROVE_STALE_AFTER_SEC = 1800` (`job_store.py:28`) duplicates
  `worker/main.py:165`'s `--stale-after-sec` default; a comment asserts they are coupled,
  nothing enforces it.

**Layering.**
- **M9** — `improver.py:15` imports `IterationRecord` from `api/job_store.py`, so a *service*
  depends on the *store* (and transitively SQLAlchemy + psycopg) purely for a dataclass, while
  `job_store.py:21` imports `services/scoring.py`. No cycle, but it defeats the spirit of
  keeping services ORM-free — and `improver.py` is the module that most wants to be importable
  standalone. *Deliverable:* move `IterationRecord` to a neutral module.

**Dead or half-done code.**
- **M1** — `worker/steps.py:125`'s `except ExecutionUnavailableError` is unreachable; neither
  runner raises it (both record it on the run row, which `_evaluate` already reads).
- **M2** — `StepRecord.run_id` dead in production (closed by F7).
- **M5** — `ArtifactStore.list` / `LocalArtifactStore.list` have no production caller; design
  §5 says "the API can enumerate a prefix" but no endpoint does. Either add the endpoint or
  drop the method.
- **M6** — `result.json` never copied (folded into F2).

**Convention violations.**
- **M3** — `worker/steps.py:65,67` compare `step.type` to bare `"evaluate"`/`"improve"` and
  `:310` compares `it.status == "completed"`, instead of the `STEP_EVALUATE`/`STEP_IMPROVE`
  constants and `RunStatus.completed.value`.
- **M4** — `tests/test_job_worker.py:80` defines its own `_RaisingImprover` instead of
  importing the one at `improver.py:521` put there for exactly this purpose; two copies of the
  same double. Worse, `tests/test_improver.py:288-294` then tests the production copy — a test
  of a test double that asserts nothing about production code.
- **M14** — `PER-TASK MOVEMENT VS BEST` is the only `build_context` section without the `## `
  prefix the others use.

**API edge case.**
- **M12** — `api/routes/jobs.py:78-83` builds `ProposalView` only when `it.rationale is not
  None`, and `job_store.py:201` sets `rationale = version.rationale or None`. An
  improver-created version with an empty rationale would therefore lose its `changed_fields`
  and `based_on_version` in the API response. Unreachable today (`LLMImprover._parse`
  substitutes `"(no rationale provided)"`), but it couples proposal *metadata* presence to
  rationale *text*.

**Provenance.**
- **M16** — job-driven `runs` rows carry no marker distinguishing them from `/v1/runs` rows.
  Now that they are created `running`/owned they are functionally distinct, but nothing in the
  schema says so, which makes operational queries ("show me standalone runs") guesswork.
  *Deliverable:* consider a nullable `job_id` or `origin` column on `runs`.

**Test coverage gap (not a defect).**
- **M15** — `MockBenchmarkRunner._outcome_for` is a pure function of `task_id`, so mock scores
  never change between iterations and **no test covers an iteration that actually improves**.
  Every "improved" path is exercised only at iteration 0, where `best_score is None` makes it
  true by definition. *Deliverable:* a runner double whose reward varies per call, and an
  end-to-end test where iteration 1 genuinely beats iteration 0 and `best_*` moves to v1.
  This is the single most valuable test missing from the suite.

---

## Suggested order

1. **F1** — without the task statement, no optimization result means anything.
2. **F2** + M6 — the other half of the diagnostic signal.
3. **M15** — before tuning anything, get a test that proves the improving path works.
4. **F3**, **F4** — stop the prompt and the cost from growing unchecked.
5. **F5** — decide the trials/task-count policy before quoting any score as evidence.
6. **F8** — cheap, and must precede cancellation.
7. **F7**, **F6** (incl. M13) — robustness.
8. **P4** hygiene, starting with the M7/M8 drift guards.
