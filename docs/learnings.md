# BIRD-Interact iteration learnings

**Setting:** BIRD-Interact lite, a-interact mode, `openai/gpt-5.4`. Stratified 70/30 subsample of the 300-task lite set (70 train tasks, 30 val).

**Baseline (iteration 0):** train 0.1129 (10/70) · val 0.1467 (5/30) — stock BIRD-Interact-ADK system agent, byte-for-byte copy of upstream.

## TL;DR — iteration summary

| iter | val | train | mechanism | outcome |
|---|---|---|---|---|
| 0 | 0.1467 | 0.1129 | stock BIRD agent | baseline |
| 1 | 0.1033 | 0.1214 | taught simulator's `ask_user` protocol | ✗ reverted — ask count dropped 63% but pass count unchanged |
| 2 | **0.1933** | 0.1843 | enforce `execute_sql` verification before `submit_sql` | **✓ committed (+4.7pp val, +5 passes)** |
| 3 | 0.1933 | 0.1600 | literal-default rules for SQL-structure choices | ✓ committed (flat val, but suite task `cybermarket_M_6` promoted) |
| 4 | 0.1933* | 0.1414 | custom `note_plan`/`revise_plan` tools + state callbacks | ✗ reverted — mechanism worked, pass count dropped |
| 5 | 0.1933* | 0.1471 | custom `compare_sql_candidates` tool | ✗ reverted — 68% adoption, no pass lift (suite fail) |
| 6 | 0.1367 | 0.1271 | custom `search_kb` content-search tool | ✗ reverted — real val regression; tool used on 96% of tasks, context bloat |

*val would have passed Step 2 (unchanged from best); gate failed on Step 1 due to stochastic suite flips.

**Net result:** one clean win (iter2, +4.7pp val over baseline). iter3 was a neutral-but-committed refinement (suite grew from 0 to 2 tasks). iter4-6 all failed to clear the gate; each taught a durable lesson about what doesn't work.

## Takeaways from 6 iterations

1. **Prompt changes that teach discipline work, but saturate fast.** iter2's execute-before-submit bullet was +5 passes. iter3's literal defaults was marginal. Subsequent prompt-shaped interventions plateaued.
2. **Ritualizing internal reasoning doesn't pay.** iter4's `note_plan`/`revise_plan` tools forced the agent to externalize planning it was already doing internally. 100% adoption, zero lift, +4 turns/task overhead.
3. **Empirical disambiguation helps per-call but not aggregate.** iter5's `compare_sql_candidates` was used on 68% of tasks, but selection bias means users are hard tasks. Comparisons showed similar plausible distributions; agent still picked wrong.
4. **Tools that solve a real information problem can regress via context bloat.** iter6's `search_kb` DID find gold KB entries in 44% of calls (mechanism worked). But 96% adoption meant the tool fired on tasks that didn't need it, adding ~1500 chars of KB definitions to context — and val regressed 5.7pp.
5. **Suite-by-Step-3-promotion is fragile.** Tasks promoted because they passed ONCE on a stochastic re-run (not consistently) tripped Step 1 on every subsequent iteration. Both `cybermarket_M_6` and `disaster_2` caused gate failures in iter4-6 without reflecting actual agent regression on val.
6. **The failure center of gravity shifted.** Post-iter3, 41% of fails are "had right KB, wrote wrong SQL" (formula fidelity / exact parameter types / CTE structure) — a SQL-generation precision problem, not a discovery problem. Remaining prompt/tool levers don't address it.

## Full iteration detail follows

## Iteration 1 — val_score: 0.1467 → 0.1033 ✗

**What changed:** Replaced two generic `ask_user` bullets in `AINTERACT_INSTRUCTION` with three protocol-specific ones: (1) the user simulator only reveals hidden definitions for registered ambiguous TERMS and replies "out of scope" to bilateral choices / schema / SQL-structure / confirmation questions; (2) prefer `get_knowledge_definition` (0.5 coins) over `ask_user` (2 coins); (3) phrase any `ask_user` as "What does '<term>' mean in this context?" — never propose alternatives.

**Pattern confirmed:** 48/60 failing train tasks made an `ask_user` call that got "Sorry, this question is out of scope" — wasting 2 bird-coins each (96 coins across 60 fails). 85.7% of all `ask_user` calls in failing tasks got OOS'd. Losing tasks asked bilateral-choice questions ("Should I use A or B?"), schema questions, or confirmations; the 3 both-phase winners averaged 0.33 `ask_user` calls, 9 turns, 37s elapsed — short, targeted, KB-driven.

**What worked:**
- `ask_user` call count dropped 63% on train (59 → 22).
- Budget reclaimed: ~64 coins across 70 tasks.
- Mean elapsed −37% on train (89s → 56s) — faster tasks (less waiting on simulator).
- Phase 2 pass count +2 on train (3 → 5).
- Mean train reward +0.86pp (0.1129 → 0.1214).

**What didn't work:**
- Train phase-1 pass count: unchanged (10 → 10).
- Val regressed: 5 → 4 pass (−1 task: `vaccine_M_3` flipped out, zero flipped in). Mean val −4.3pp.
- **OOS rate of remaining asks stayed 86.4%** (19/22). Agent now uses correct "What does X mean?" form, but is asking about terms that aren't in the task's `critical_ambiguity` registry — which the agent cannot know in advance.
- Extra budget saved flowed into more submits (mean submits 2.67 → 3.01), but the evaluator returns only `"SQL failed Phase 1. Your SQL is not correct."` — uninformative. More retries produce similarly-wrong SQL.
- 3 train tasks regressed (`news_10`, `solar_M_4`, `vaccine_M_2` — the last was a both-phase winner). Matched by 3 new passes (`credit_M_2`, `robot_M_1`, `robot_M_4`). Net zero, likely stochastic (GPT-5 runs without `temperature=0`).

**Key insight:** Budget is not the limiting factor on this benchmark. **SQL correctness on first/second attempt is.** More retries with the same agent reproduce similar errors because the evaluator does not reveal which assertion failed. Future iterations should target first-submit SQL quality, not budget efficiency. Specifically:
- Disambiguation defaults for `intent_ambiguity` and `schema_linking_ambiguity` terms (not resolvable by KB or simulator).
- Execute-before-submit discipline for `category: Query` tasks (18/41 Query fails never ran `execute_sql` on the exact SQL they submitted).
- Possibly: `reasoning_effort: "high"` for GPT-5 — one-line model-config change.

**Action:** Reverted `agent/agent.py` via template re-copy (git checkout reverts past BIRD baseline because the stock copy was never committed — confirmed separately). Gate failed at Step 2, no commit, no `results.tsv` append.

**Needs from human:** Consider whether "never call `ask_user`" is an acceptable policy. 86% of asks OOS regardless of form makes the expected value negative. This is a more aggressive version of iter1's hypothesis and would free the agent from wasted clarification loops entirely.

## Iteration 2 — val_score: 0.1467 → 0.1933 ✓  (commit `a35e4dd`)

**What changed:** Replaced the soft hint *"Test SQL with execute_sql before submit_sql when useful"* with explicit execute-before-submit discipline. For SELECT/Query tasks: before `submit_sql` (3 coins), run the exact final SELECT body as `execute_sql` (1 coin) and verify column names, row count, and value ranges match the task; only submit if the execute_sql output is semantically correct. For CREATE/ALTER/INSERT/UPDATE/DELETE/CREATE FUNCTION/CREATE VIEW tasks (where execute_sql is SELECT-only and cannot run the body), manually verify referenced tables/columns against get_schema and formula terms against get_knowledge_definition before each submit. Single-bullet rewrite; no other changes in the file.

**Pattern confirmed:** iter1's key insight — "extra budget flows into more submits but produces no extra passes because evaluator feedback is uninformative" — is the actual bottleneck. The fix is to make each submit verified rather than give the agent more of them.

**What worked:**
- **Train pass count: 10 → 15 (+5)**, phase1 10→15, phase2 3→8 (+5 and +5; phase2 more than doubled).
- **Train mean: 0.1129 → 0.1843 (+7.1pp, +63.3%)** — well outside the ~3pp noise band from iter1.
- **Val pass count: 5 → 7 (+2)** on 30-task held-out split; val mean 0.1467 → 0.1933 (+4.7pp, +31.7%).
- Gate PASSED all three steps. Step 3 promoted `cybermarket_M_6` into the regression suite.
- Flips (train): +6 in (`credit_1`, `credit_4`, `credit_M_2`, `cross_db_M_4`, `robot_M_1`, `robot_M_4`) vs 1 out (`vaccine_M_2`, same task that flipped in iter1 — stochastic on this specific task).

**What didn't work (subtle):**
- execute_sql call count barely moved in aggregate (3.53 → 3.56/task). The agent didn't add more probes — instead, the existing probes became more purposeful. The prompt internalized "verify before submit" as a discipline rather than a tool-count mandate. Good outcome but worth noting: mechanism != what I predicted.
- Budget exhaustion rose slightly (29/70 → 36/70). The agent spends more on careful SELECT verification, which is the right trade for Query tasks but leaves less headroom on Management tasks where verification isn't possible.
- `vaccine_M_2` regressed for the second iteration in a row. Structurally stochastic on this specific task — worth noting, not worth debugging.

**Key insight:** Two iterations in, clear pattern: **SQL correctness on first/second attempt matters more than budget elasticity.** Iter1 (fix ask_user waste) was neutral. Iter2 (enforce submit-time verification) was a real win. The evaluator's uninformative "SQL failed Phase 1" error means the agent can't iterate toward correctness AT runtime — it must be correct out of the gate. Future hypotheses should keep targeting first-submit quality.

**Needs from human:** none. Ready for iter3.

## Iteration 3 — val_score: 0.1933 → 0.1933 ✓ (gate passed; no delta)  (commit `c7121e1`)

**What changed:** Added one bullet to AINTERACT_INSTRUCTION teaching literal-default interpretations for SQL-structure choices the simulator cannot clarify. Defaults: "view" → CREATE VIEW (not MATERIALIZED); "column" → regular column (not GENERATED STORED unless user says "computed"/"derived"); "function" → CREATE FUNCTION LANGUAGE plpgsql; "create some N"/"a few N" → smallest defensible count; "latest"/"current" → DISTINCT ON primary key ORDER BY timestamp DESC; "delete"/"remove" → DELETE (not soft-delete); "update"/"set" → UPDATE in place. Single-bullet addition; stock ask_user bullets untouched (so current ask_user-winners like credit_4, credit_M_2 can keep their winning path).

**Pattern confirmed:** The data from iter2's remaining fails showed 18 `intent_ambiguity` terms across 15 Management fails — e.g., museum_M_3 (generated vs regular column), archeology_M_1 (VIEW vs MATERIALIZED), virtual_M_2 (function form). Simulator OOS's these because they're SQL-structure questions, not term-definition questions. Literal defaults give the agent a predictable fallback.

**What worked:**
- Suite task `cybermarket_M_6` (the whole reason iter2 Step 3 promoted it) passed 0.7 on iter3 Step 1 — mechanism confirmed for target.
- Step 3 promoted `disaster_2` into suite (suite now 2 tasks — harder regression target for iter4+).
- Gate cleared all three steps.

**What didn't work:**
- **Val was unchanged** (0.1933 → 0.1933 — exact same 7/30 phase1, 3/30 phase2 mix on test split). Gate passed on `>= best` rule, not an improvement.
- **Train regressed slightly** (15/70 → 13/70, mean 0.1843 → 0.1600). 2 flips in (cybermarket_M_6, vaccine_M_2), 4 flips out (credit_1, credit_4 — stock Query winners; cross_db_M_4, robot_M_4 — iter2 stochastic flip-ins). Net -2 on train.
- Credit_4 flipping out is notable — it was a deterministic iter2 winner whose winning path used `ask_user`. The iter3 change didn't touch ask_user guidance, so this is most likely stochastic, not mechanism-driven. But worth flagging.
- The literal-defaults bullet, while correct for its target, is contributing marginal value. Most Management fails (`robot_M_2`, `virtual_M_2` formula-derivation cases) are knowledge_linking_ambiguity where the ROOT problem is that the formula isn't in the KB — literal defaults don't help these.

**Key insight:** Two passed iterations in (iter2 + iter3), val went 0.1467 → 0.1933 → 0.1933. The easy wins from single-bullet prompt changes are drying up. Remaining failures fall into tougher categories:
1. **Formula-not-in-KB cases** (robot_M_2, virtual_M_2 — agent needs formula to solve, can't find it, can't ask for it). Structural ceiling without an external tool.
2. **Sophisticated SELECT semantics** (complex joins, window functions, CTEs with correct bounds) — execute_sql validates shape but not correctness.
3. **Intent ambiguities that don't match any literal default** — the user's text is genuinely ambiguous and has no canonical reading.

**Needs from human:** Worth discussing whether to:
(a) pivot to enabling targeted ask_user for category 3 (the deferred H-E1 "bullet A" from iter3 planning);
(b) accept a ceiling effect and focus on robustness (reduce stochastic flips);
(c) consider that the gate threshold with a 2-task suite (cybermarket_M_6, disaster_2) is now tighter — any stochastic flip on either = Step 1 fail at 50% < 80%. Suite promotion has a fragility cost.

## Iteration 4 — val_score: 0.1933 → 0.1933 ✗ (gate failed on Step 1; reverted)

**What changed:** First structural intervention (beyond prompt-only). Added two custom ADK tools (`note_plan`, `revise_plan`) directly in agent.py via `FunctionTool` registration, plus two wrapped callbacks:
- `note_plan(interpretations, sql_operation, key_tables, expected_columns)` — records a committed interpretation as persistent state. Cost: 0 coins.
- `revise_plan(what_changed, why)` — records explicit revision after a submit failure. Cost: 0 coins.
- `before_tool_callback` wrapper: blocks `submit_sql` if no plan exists; blocks `submit_sql` after failure if no `revise_plan` called since.
- `after_tool_callback` wrapper: tracks `submit_sql` failure state to drive the block logic.

Rationale: iter3 failure analysis showed fails over-explore (17.1 vs 12.4 turns, 5.5 vs 3.3 KB lookups), 29/56 fails are 3+ submit retries, and 45% of those retries show genuinely divergent SQL. Hypothesis: forcing explicit hypothesis commitment + revision-before-retry would reduce thrash and pivot behaviors.

**Pattern confirmed:** Observed failure modes accurately characterized. `alien_2` smoke test showed the agent DID correctly:
1. Commit to a plan with `note_plan`
2. Submit, fail
3. Call `revise_plan` with concrete "what/why" ("Changed score-level bucketing from corrscore to anomscore because anomscore has clear distribution across all three buckets")
4. Submit, fail again
5. Call `revise_plan` a second time
6. Hit budget exhaustion on 3rd submit

The mechanism fired exactly as designed on all 70 tasks (100% `note_plan` adoption, 91% `revise_plan` adoption after failure).

**What worked:**
- Tool registration via `FunctionTool` works cleanly alongside stock tools (11 total registered).
- Wrapped callbacks correctly chain to stock ones (budget accounting + trajectory logging both functional).
- Gate enforcement blocks pathological behavior (no plan → submit blocked; unrevised retry → submit blocked). No bugs.
- The agent traces show genuine articulated hypotheses instead of silent thrashing — that's a qualitative win for interpretability.

**What didn't work:**
- **Train: 13/70 → 12/70 (-1 pass), mean 0.1600 → 0.1414.** The commitment discipline added overhead without producing passes.
- **Suite task `cybermarket_M_6` regressed (1.0 → 0.0)** — this was iter2's serendipitous Step-3 promotion. Under iter4's extra planning turns, it tipped back to failure. Directly caused Step 1 gate fail.
- **Turns up 15.7 → 19.9** (+4 turns/task, consistent with 2.0 `note_plan` + 1.8 `revise_plan` calls on average). Turns alone don't cost budget but each represents an extra LLM call with latency and context inflation.
- **Budget-exhausted tasks: 29 → 33** (+4). More turns = more ways for the agent to over-commit before the end.
- **Val unchanged at 0.1933.** Despite gate Step 2 passing, Step 1 failure made the iteration a no-commit.

**Key insight:** Structural intervention worked mechanically but didn't solve the fundamental problem: **the agent was already mentally planning; forcing explicit planning added ritual overhead without better decisions.** GPT-5's internal reasoning already covers hypothesis commitment; the tool-level commit forced it to externalize what it was implicitly doing anyway, at the cost of extra turns. For iter5, the structural lever needs to give the agent something it *can't already do internally* — e.g., persistent state across context-window compression, empirical disambiguation via SQL probes, or a genuinely novel tool (not a ritualization of internal reasoning).

**Fragility of suite-by-Step-3-promotion:** `cybermarket_M_6` was promoted because it passed ONCE in iter2's Step 3 re-run (not in iter2's original train run). That's stochastically fragile. Under any non-trivial agent change, it can flip back. Suite threshold 80% with 2-task suite means ONE flip = gate fail. Worth considering either (a) require a task to pass multiple independent re-runs before promotion, or (b) lower threshold to tolerate one flip on small suites.

**Action:** Reverted `agent/agent.py` via `git checkout` (worked cleanly this time since iter3 was committed). Gate failed at Step 1, no commit, no `results.tsv` append.

**Needs from human:** For iter5, directions that give the agent *new capability* rather than ritualized existing capability:
1. **Empirical disambiguation tool** — a tool that takes N candidate column/formula variants, runs execute_sql on each, returns a comparison table. Gives the agent DATA for choosing between candidates (addresses `alien_2` thrash directly).
2. **KB pre-hydration via HTTP calls in `before_model_callback`** — front-load relevant KB entries into context on turn 0 so the agent doesn't spend turns/coins fetching them reactively. Concrete budget savings.
3. **Accept ceiling and focus on suite robustness** — re-run current agent 3× per suite task, require unanimous pass before promotion. Makes suite less fragile; may enable higher thresholds later.

## Iteration 5 — val_score: 0.1933 → 0.1933 ✗ (gate failed on Step 1; reverted)

**What changed:** Added a single custom tool `compare_sql_candidates(candidates, labels)` — runs 2-5 SELECT variants via internal calls to stock `execute_sql`, returns side-by-side results (SQL preview + first ~10 result lines per candidate). Cost: 2 bird-coins flat, enforced via a wrapped `before_tool_callback` that deducts from budget. One new prompt bullet explains when to use it (multiple interpretations differing in ONE disambiguation choice). Nothing else changed — no `note_plan`/`revise_plan`, no other structural modifications.

**Pattern confirmed:** Agent adoption was strong — 48/70 train tasks (69%) used the tool, averaging 2.88 candidates per call. Smoke test on `alien_2` showed the agent correctly picked out the exact corrscore/techsigprob/anomscore disambiguation I had predicted, and called `compare_sql_candidates` with those three variants. Tool mechanics work end-to-end: cost deduction via wrapped `before_tool_callback`, inner HTTP calls to db_env service via stock `execute_sql`, side-by-side formatting.

**What worked:**
- Tool registration, cost accounting, and internal execute_sql calls all functioned correctly across 70 tasks and 49 total compare calls.
- Mean task elapsed **60s** (vs 89s baseline, 76s iter3) — tool use is faster than submitting guesses.
- Val unchanged at 0.1933 — iter5's agent is NOT worse on held-out test; the gate failure was entirely a suite-robustness issue.

**What didn't work:**
- **Train: 13/70 pass (same as iter3), mean 0.1471 (down from iter3's 0.1600, within noise).** Only 1 flip in (`virtual_M_2`) and 1 flip out (`vaccine_M_2`) — stochastic.
- **Tool usage is anti-correlated with pass outcome at first glance:** 3 pass / 45 fail among users (6%) vs 10 pass / 12 fail among non-users (45%). This is selection bias — easy tasks with obvious mappings don't trigger the tool; hard ambiguous tasks do. The tool is routing to the right cases, but the cases are genuinely hard.
- **The tool's comparison output isn't discriminating enough.** When all candidates produce plausible-looking distributions (similar row counts, similar value ranges), the agent still picks wrong. The evaluator checks specific test-case values that don't show up in summary previews. For `alien_2`, all three columns (corrscore/techsigprob/anomscore) produced superficially similar CASE-bucketed distributions, so the comparison didn't help.
- **Gate Step 1 failed on suite fragility (again).** Fresh re-run of cybermarket_M_6 and disaster_2 both returned 0 — neither deterministically passes. Same root cause as iter4: Step-3-promoted tasks with single-pass-at-promotion are fragile under any agent change.

**Key insight:** Val has been flat at 0.1933 for three iterations in a row (iter3 committed, iter4 reverted, iter5 reverted). The ceiling is real for prompt-style and single-tool interventions at this agent configuration. Further incremental changes to the a-interact agent are unlikely to unlock new val passes without one of:
1. **Richer tool feedback** — the current BIRD evaluator's binary pass/fail is the information bottleneck. We can't fix that from agent.py alone.
2. **Genuinely different agent architecture** — e.g., a planner/executor sub-agent split via ADK `sub_agents`, or an LLM-based critic that evaluates draft SQL against the user query before submit. Both are complex and high-risk.
3. **Different benchmark setting** — c-interact mode instead of a-interact has higher paper baselines; could test as a one-shot comparison run.

**Action:** Reverted `agent/agent.py` via `git checkout`. Gate failed at Step 1 (not Step 2, which passed with val=0.1933). No commit. No `results.tsv` append.

**Needs from human:** 3 iterations without val improvement suggests a real ceiling. Options to consider:
(a) **Address suite fragility first** — re-run current iter3 agent 3× per failing train task, only promote tasks that pass ≥2 of 3 re-runs. Removes stochasticity from suite gating. Doesn't change val_score itself but unblocks future iterations from spurious Step 1 failures.
(b) **Try c-interact mode** as a one-shot comparison — may have higher ceiling given paper numbers (o3-mini 24.4% vs Claude-3.7 17.78% a-Interact). Just a config change + different template copy.
(c) **Accept ceiling and stop** — we're at +46.7% over baseline val (0.1467 → 0.1933 once committed at iter3). That's a respectable single-iteration lift even if subsequent iterations haven't added. Ship iter3 as the final, document the ceiling.

## Iteration 6 — val_score: 0.1933 → 0.1367 ✗ (actual val regression; reverted)

**What changed:** Added a single custom tool `search_kb(query, max_results)` that performs content-based ranking of KB entries (keyword overlap on name + description + definition fields) and returns top-K with full definitions. Cost: 0.5 bird-coins, enforced via wrapped `before_tool_callback`. KB entries cached in state after first call to avoid repeated HTTP. One prompt bullet explains the tool as a replacement for the "browse name list → guess → fetch definitions" loop.

**Pattern confirmed (motivating analysis was correct):** Error analysis on iter3's 58 failing train tasks showed:
- 50 of 58 fails have gold SQL that references specific KB entries
- Only 12/50 (24%) had the agent fetch ALL gold-referenced KB entries
- 13/50 (26%) fetched NONE of the right entries
- Classic lexical-trap: agent sees "signal quality" in query, picks "Signal-to-Noise Quality Indicator" (lexical match), when gold uses "Technological Origin Likelihood Score" (semantic match)

Example KB misses:
- `alien_2`: gold=TOLS, agent=[BFR, CIP Label, High Confidence Signals]
- `credit_1`: gold=Net Worth, agent=[] (nothing)
- `cybermarket_5`: gold=MSI, agent=[Security Posture, Identity Protection, ...]

**What worked (mechanics):**
- Tool adoption immediate and heavy: **96% of train tasks used it**, 192 total calls across 59 failing tasks (mean ~3 calls per user).
- Caching worked (no repeat HTTP fetches).
- **The tool IS effective at its stated job: in 84/192 (44%) of search_kb calls on failing tasks, the gold-referenced KB entry name appeared in the output.** Finding the right KB is no longer the bottleneck.

**What didn't work:**
- **Train: 11/70 pass (down from iter3's 13), mean 0.1271 (down 3.3pp).** 0 flips in, 2 flips out (`cybermarket_M_6`, `news_10`).
- **Val: 0.1933 → 0.1367 (−5.7pp, real regression on held-out set).** First actual val degradation since iter1.
- Gate failed on BOTH Step 1 (suite 0/2) AND Step 2 (val 0.1367 < best 0.1933).

**Key insight — the bottleneck moved upstream but another one took its place:**

Finding the right KB entry is no longer the problem — search_kb surfaces it in ~44% of calls. But agents still fail. Even with gold KB visible in tool output, the agent doesn't consistently:
1. Recognize the correct entry among the top-K returned
2. Translate the KB formula accurately into SQL
3. Apply the formula in the right SQL structure (CASE bins, JOIN condition, WHERE clause)

The failure mode shifted from "didn't find the right KB" to "found it but didn't use it correctly."

**Why the val regression?** Hypothesis — context bloat. 96% adoption means the agent called search_kb on most tasks, including simple ones with unambiguous KB entries. Each call returns 3-5 full KB definitions (~500 chars each). On easy tasks where the agent would otherwise have picked the right entry cheaply, the extra 2-3 KB dumps in context distracted the reasoning. Adding a tool that's almost always used costs context on every task, even tasks that didn't need it.

This is a second, harder-earned lesson: **tool use isn't free even when priced cheap — the cognitive cost of extra context can regress tasks the tool wasn't meant to help.**

**Action:** Reverted `agent/agent.py` via `git checkout`. No commit. Gate failed on both Steps 1 and 2.

**Status — approaching PROGRAM.md stop rule (§6):** val has been at-or-below 0.1933 for 4 consecutive iterations (iter3 committed, iter4/5 reverted with Step-2 val=0.1933 unchanged, iter6 reverted with val REGRESSED to 0.1367). One more non-improvement iteration hits the 5-in-a-row threshold where PROGRAM.md calls for stopping and summarizing.

**Cumulative lessons after 6 iterations (ranked by evidence quality):**

1. **Prompt changes that teach discipline work (iter2 +7pp) but saturate quickly** (iter3 marginal, iter1/3 additional bullets neutral or negative).
2. **Tools that ritualize internal reasoning don't pay** (iter4 plan/revise).
3. **Tools that enable empirical disambiguation help per-call but don't lift aggregate pass rate** (iter5 compare_sql_candidates — 68% adoption, same pass count as iter3).
4. **Tools that solve a real information-discovery problem can REGRESS when used indiscriminately** (iter6 search_kb — 96% adoption, val regressed). Adding capability ≠ adding value.
5. **The suite promotion mechanism has real fragility** — Step-3 serendipitous promotions tip gates on every iteration.
6. **Val has appeared stuck at 0.1933** for a-Interact with gpt-5.4 through this tool/prompt surface. Crossing it may need a different lever: different model, c-interact mode, or something structural we haven't tried.

**Needs from human:** Three concrete options:
1. **Stop per PROGRAM.md §6** — we're one iteration from the stop criterion. Ship iter3 (val 0.1933, +31.7% over baseline). Document the arc. Move on.
2. **One more narrow attempt — selective search_kb** — add the tool but with a restriction: agent may call it at most ONCE per task, must include the user's exact masked term in the query. Tests whether the regression was from indiscriminate use rather than the tool itself.
3. **Change the underlying — try c-interact mode or a different model** (protocol-legal via experiment_config.yaml; resets the baseline but gives new headroom).

---

## Future directions (not yet tried)

Things we think would work but didn't get to in 6 iterations. Each is grounded in a specific observed failure mode from the traces.

### 1. Curated few-shot bank of in-scope `ask_user` examples

**What:** Mine the 8 in-scope answered asks from iter5's trace corpus and embed the 2-3 that led to passes (`credit_4`, `credit_M_2`) as POSITIVE few-shot examples in `AINTERACT_INSTRUCTION`. Add 2-3 OOS'd asks as NEGATIVE examples with the OOS response shown, so the agent learns the contrast. Pair each positive example with its task description so the contextual trigger is visible.

**Why it might work:** The trace data shows the agent has the right instinct (`alien_1`: "signal quality" → asked about "signal quality" — correct term) but the wrong form (bilateral "A or B" question types get OOS'd even on registered terms). Few-shot examples teach form more effectively than abstract rules — iter1's protocol-description bullet didn't change OOS rate; a concrete shape might.

**Implementation:** Pure prompt change. No code. Risk: prompt grows ~300-500 tokens and may dilute attention on other instructions.

### 2. Staged disambiguation pipeline (decompose → resolve → ask-as-last-resort)

**What:** Enforce a strict ordering: (a) parse the user's query into {entities, operations, conditions, outputs}, flagging ambiguous noun phrases; (b) for each ambiguous term, try `get_knowledge_definition` first; (c) if not in KB, try `get_column_meaning` / `get_schema`; (d) if still multiple plausible mappings, try `execute_sql` to probe each; (e) ONLY if all four fail AND the term exactly matches the user's literal phrasing, call `ask_user`. Implement as a hard ordering via either a prompt preamble or — more structurally — a `before_tool_callback` that blocks `ask_user` until evidence of prior resolution attempts is in state.

**Why it might work:** iter5 showed 14% of asks get in-scope answers but 45% of persistent fails are "thrasher" cases where the agent didn't try empirical disambiguation first. A staged pipeline forces empirical-first; asks become the genuinely residual cases (where their in-scope hit rate is highest).

**Implementation:** Prompt-driven version = one bullet. Callback-enforced version ≈ 50 LOC tracking per-task "resolution attempts" in state.

### 3. KB-formula verification callback (deterministic, targets Mode C)

**What:** Before `submit_sql`, a `before_tool_callback` that (a) detects KB entries relevant to the task via the existing search; (b) extracts formulas from the KB `definition` field via regex (BIRD KB formulas are consistently LaTeX-style, e.g., `$\text{BFR} = \frac{\text{BwHz}}{\text{CenterFreqMhz} \times 10^6}$`); (c) checks whether the candidate submit SQL contains the extracted columns and operators; (d) if not, returns an error instructing the agent to copy the KB formula verbatim.

**Why it might work:** Mode C (24/58 = 41% of fails) is "agent had the right KB but wrote a different formula anyway." `cybermarket_M_6`'s gold-vs-pred diff was identical body with different parameter types. `disaster_2` invented a formula despite having OEI in context. A deterministic pre-submit check forces fidelity to the KB-provided formula where one exists.

**Implementation:** ~80 LOC. Needs a modest LaTeX-to-PostgreSQL expression translator (formulas are mostly simple arithmetic). Highest-leverage single change for the largest remaining fail bucket.

### 4. Mine passing-task SQL templates, expose as a tool

**What:** From the passing-train traces across iterations, extract structural SQL patterns by task category (e.g., "Query + knowledge_linking: CTE with KB-formula expression, GROUP BY bucketing"). Store as a JSON library. Add a `get_sql_template(category, ambiguity_types)` tool returning 2-3 pattern snippets. Agent uses them as starting scaffolds.

**Why it might work:** Passing tasks converge on ~4-5 structural patterns (observed in deep-reads of iter3 passes). Failing tasks often pick the wrong pattern (`disaster_2` flattened a multi-CTE into one-level SELECT). Seeing good templates up-front reduces structural mistakes.

**Implementation:** ~50 LOC for the tool + library build-time script. Cost 0.5 coins/call. Risk: template lock-in on tasks that don't match a registered pattern.

### 5. Structured diff feedback on failed submits

**What:** `after_tool_callback` on `submit_sql` where the result contains "failed Phase 1". Before returning the result to the agent, compute and inject a diff between this submit and the previous submit (if any): "Changes vs previous submit: +3 lines, -1 line. Columns added: {X}. Columns removed: {Y}." Makes iteration MORE deliberate.

**Why it might work:** iter1 analysis showed 29/56 persistent fails made 3+ submits. Only 13 had diverse submits (<50% similar); the majority were minor tweaks without clear rationale. A diff summary forces the agent to see what it IS and IS NOT changing — and either commit to a bigger pivot or stop burning budget on near-duplicates.

**Implementation:** ~40 LOC using difflib. No extra budget cost (runs in callback).

---

Ordered roughly by impact × feasibility. **#3 (KB-formula verification)** targets the dominant remaining fail bucket (41%) and is the highest-leverage single change. The rest target overlapping slices of the fail distribution and could plausibly stack.


