"""Pure tests for improver context assembly (no DB, no LLM, no network)."""

from __future__ import annotations

import json

from api.agent_spec import AgentSpec
from api.job_store import IterationRecord
from api.services.improver import (
    EvaluationSummary,
    TaskOutcome,
    build_context,
)


def _spec(system_prompt: str = "BASE PROMPT") -> AgentSpec:
    return AgentSpec(system_prompt=system_prompt, agent_model="gpt-4.1-mini")


def _iteration(
    *,
    iteration: int,
    version: int,
    score: float | None,
    improved: bool | None,
    changed_fields: list[str],
    rationale: str | None,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        agent_version_id=f"00000000-0000-0000-0000-00000000000{version}",
        version=version,
        run_id=f"11111111-1111-1111-1111-11111111111{iteration}",
        score=score,
        improved=improved,
        rationale=rationale,
        changed_fields=changed_fields,
        status="completed",
    )


def _history() -> list[IterationRecord]:
    return [
        _iteration(
            iteration=0,
            version=0,
            score=0.5,
            improved=True,
            changed_fields=[],
            rationale="baseline",
        ),
        _iteration(
            iteration=1,
            version=1,
            score=0.5,
            improved=False,
            changed_fields=["max_steps", "system_prompt"],
            rationale="Told the agent to verify\nits work before finishing",
        ),
    ]


def _trace(text: str) -> str:
    return json.dumps(
        [
            {"role": "system", "content": "you are an agent"},
            {"role": "assistant", "content": "running the build"},
            {"role": "tool", "content": text},
        ]
    )


def test_history_table_has_one_row_per_iteration_and_rationales() -> None:
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=0.5, tasks=[], traces={}),
        history=_history(),
        budget=60_000,
    )

    block = ctx.split("## ITERATION HISTORY (oldest first)\n", 1)[1].split("\n\n", 1)[0]
    lines = block.strip().splitlines()

    assert lines[0] == "iteration | version | score | improved | changed_fields | rationale"
    assert len(lines) == 3, lines
    assert lines[1].startswith("0 | 0 | 0.5000 | yes | - | baseline")
    assert lines[2].startswith("1 | 1 | 0.5000 | no | max_steps,system_prompt | ")
    # Multi-line rationales are flattened onto their single row.
    assert "Told the agent to verify its work before finishing" in lines[2]


def test_failure_details_are_worst_first() -> None:
    tasks = [
        TaskOutcome(task_id="t-pass", status="passed", reward=1.0, remarks=None),
        TaskOutcome(task_id="t-partial", status="failed", reward=0.4, remarks="Partial reward 0.4"),
        TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed"),
        TaskOutcome(task_id="t-err", status="error", reward=None, remarks="sandbox timeout"),
    ]
    traces = {
        "t-partial": _trace("partial trace body"),
        "t-zero": _trace("zero trace body"),
        "t-err": _trace("error trace body"),
    }
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=0.35, tasks=tasks, traces=traces),
        history=_history(),
        budget=60_000,
    )

    assert "## FAILURE DETAILS (worst tasks first)" in ctx
    i_err = ctx.index("### t-err")
    i_zero = ctx.index("### t-zero")
    i_partial = ctx.index("### t-partial")
    assert i_err < i_zero < i_partial
    # Passing tasks appear in the result table but get no failure block.
    assert "### t-pass" not in ctx
    # JSON message traces are rendered as role-tagged lines.
    assert "[tool] error trace body" in ctx


def test_output_never_exceeds_budget_with_huge_trace() -> None:
    # Note: this does not by itself exercise the final `[:budget]` slice -
    # `_render_trace`'s own 4_000-char tail cap binds well before the 5_000
    # budget does, and the mandatory sections here are tiny. It still verifies
    # a huge single trace can't blow the budget through some other path (e.g.
    # a missed truncation inside `_render_trace`). See the test below for a
    # case that genuinely forces the outer slice.
    tasks = [TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed")]
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.0,
            tasks=tasks,
            traces={"t-zero": "X" * 500_000},
        ),
        history=_history(),
        budget=5_000,
    )

    assert len(ctx) <= 5_000
    assert "BASE PROMPT" in ctx


def test_output_never_exceeds_budget_when_mandatory_sections_alone_overflow() -> None:
    """Genuinely forces the final `[:budget]` slice.

    Sections 1-3 (spec, history, per-task result table) are emitted in full
    regardless of budget, and the per-task table grows with every task
    regardless of any single trace's internal cap. With enough failing tasks,
    that mandatory prefix alone exceeds a small budget before a single
    failure block is even considered - so the only thing keeping
    len(ctx) <= budget true is the trailing slice, not the block-selection
    loop (which, unlike here, ordinarily keeps the running total within
    budget by construction and never needs the slice to do real work).
    """
    tasks = [
        TaskOutcome(
            task_id=f"t-{i:04d}",
            status="failed",
            reward=0.0,
            remarks=f"failure remark number {i} " * 5,
        )
        for i in range(500)
    ]
    traces = {task.task_id: _trace(f"trace body for {task.task_id}") for task in tasks}
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=0.0, tasks=tasks, traces=traces),
        history=_history(),
        budget=2_000,
    )

    assert len(ctx) <= 2_000
    assert ctx.startswith("## CURRENT AGENT SPEC (JSON)")
    assert "BASE PROMPT" in ctx


def test_no_failure_section_when_all_tasks_pass() -> None:
    tasks = [
        TaskOutcome(task_id="t-a", status="passed", reward=1.0, remarks=None),
        TaskOutcome(task_id="t-b", status="passed", reward=1.0, remarks=None),
    ]
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=1.0, tasks=tasks, traces={}),
        history=_history(),
        budget=60_000,
    )

    assert "FAILURE DETAILS" not in ctx
    assert "## LATEST EVALUATION (score=1.0000)" in ctx


def test_current_spec_survives_tiny_budget() -> None:
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.0,
            tasks=[TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed")],
            traces={"t-zero": _trace("body")},
        ),
        history=_history(),
        budget=300,
    )

    assert len(ctx) <= 300
    assert ctx.startswith("## CURRENT AGENT SPEC (JSON)")
    assert "BASE PROMPT" in ctx


def test_context_reports_regressions_prominently() -> None:
    # A distinctive task id: asserting plain "a" would pass on the surrounding
    # boilerplate alone and prove nothing about the REGRESSED section's content.
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.5,
            tasks=[TaskOutcome("zzz-task", "failed", 0.0, None)],
            traces={},
            fixed_tasks=["b"],
            regressed_tasks=["zzz-task"],
        ),
        history=[],
        budget=60_000,
    )
    assert "PER-TASK MOVEMENT VS BEST" in ctx
    assert "REGRESSED" in ctx
    # Assert on the REGRESSED line itself, not "everything after the word
    # REGRESSED" - zzz-task also recurs in the sections that follow the
    # movement section (## LATEST EVALUATION, ## FAILURE DETAILS), so
    # `"zzz-task" in ctx.split("REGRESSED")[1]` would still pass even if the
    # movement section named no task at all. See tests/test_improver_context.py
    # I4 in the Milestone 4 final review.
    regressed_line = next(line for line in ctx.splitlines() if line.startswith("REGRESSED"))
    assert "zzz-task" in regressed_line


def test_context_movement_section_present_when_no_movement() -> None:
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.5, tasks=[TaskOutcome("a", "passed", 1.0, None)], traces={}
        ),
        history=[],
        budget=60_000,
    )
    assert "No per-task movement vs the best version." in ctx
