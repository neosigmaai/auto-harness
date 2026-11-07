"""Regression tests for honest evaluation, immutable evidence and interruption.

No paid calls or Docker needed. The executor and persistence fake implement the
public contracts; these tests run the actual state machine and decision logic.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from service import repo
from service.models import JobCreate
from service.tasks_catalog import DEFAULT_TASKS, BY_NAME
from service.worker.executors.base import TaskResult
from service.worker.job_runner import (Cancelled, JobRunner, TEMPLATE_PATH,
                                       confirmation_eligibility, evaluation_metrics, promotion_decision)
from service.worker.optimizer import Proposal


def trial(task="selection", *, passed=True, cost=1.0, duration=10.0, **kw):
    return dict(task_name=task, status="passed" if passed else "failed",
                reward=float(passed), duration_s=duration, model_cost_usd=cost,
                cost_usd=cost, sandbox_cost_usd=None, cost_known=True,
                input_tokens=100, output_tokens=10, n_steps=10, **kw)


def metrics(*rows):
    return evaluation_metrics(list(rows), len(rows))


def test_score_denominator_cannot_shrink_when_a_trial_is_missing_or_infrastructure_failed():
    assert evaluation_metrics([trial()], 2)["mean_reward"] is None
    broken = trial()
    broken.update(status="error", reward=None)
    assert metrics(trial(), broken)["complete"] is False
    partial = trial()
    partial.update(reward=.25, status="failed")
    assert metrics(partial)["mean_reward"] == .25


def test_unknown_cost_is_not_treated_as_free_execution():
    result = trial(cost=0)
    result["cost_known"] = False
    summary = metrics(result)
    assert summary["mean_model_cost_usd"] is None
    assert summary["total_sandbox_cost_usd"] is None
    assert not promotion_decision(summary, metrics(trial()), {})[0]


def test_selection_balances_measured_quality_cost_and_tail_latency():
    incumbent = metrics(trial(), trial(passed=False))
    assert promotion_decision(metrics(trial(), trial()), incumbent, {})[0]
    assert not promotion_decision(metrics(trial(cost=2), trial(cost=2)), incumbent, {})[0]
    assert not promotion_decision(metrics(trial(duration=20), trial(duration=20)), incumbent, {})[0]
    assert not promotion_decision(metrics(trial(passed=False, cost=.1, duration=1)), incumbent, {})[0]


def test_perfect_score_does_not_prevent_efficiency_improvement():
    before = metrics(trial())
    after = metrics(trial(cost=.75, duration=10))
    assert promotion_decision(after, before, {})[0]
    # Step count is not one of the selected objective measurements.
    cheaper_steps = trial()
    cheaper_steps["n_steps"] = 1
    assert not promotion_decision(metrics(cheaper_steps), before, {})[0]
    assert not promotion_decision(metrics(trial(passed=False, cost=.1)), metrics(trial(passed=False)), {})[0]


def test_confirmation_requires_a_measured_minimum_signal_without_preliminary_regression_ceilings():
    incumbent = metrics(trial(), trial(passed=False))
    costly_quality_gain = metrics(trial(cost=2, duration=30), trial(cost=2, duration=30))
    assert confirmation_eligibility(costly_quality_gain, incumbent, {})[0]
    assert not promotion_decision(costly_quality_gain, incumbent, {})[0]
    assert confirmation_eligibility(metrics(trial(cost=.8, duration=30)), metrics(trial()), {})[0]
    assert confirmation_eligibility(metrics(trial(cost=2, duration=8)), metrics(trial()), {})[0]
    assert not confirmation_eligibility(metrics(trial(cost=.95)), metrics(trial()), {})[0]
    assert not confirmation_eligibility(metrics(trial(cost=.8)), metrics(trial()),
                                       {"min_efficiency_improvement": .25})[0]
    assert not confirmation_eligibility(costly_quality_gain, incumbent, {"min_improvement": .5})[0]
    assert not confirmation_eligibility(metrics(trial(passed=False, cost=.1)), metrics(trial()), {})[0]
    assert not confirmation_eligibility(metrics(trial(passed=False, cost=.1)),
                                       metrics(trial(passed=False)), {})[0]
    assert not confirmation_eligibility(metrics(trial()), metrics(trial()), {})[0]


@pytest.mark.parametrize("key", ["mean_reward", "mean_model_cost_usd", "p95_task_duration_s"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True])
def test_confirmation_does_not_spend_on_missing_or_invalid_measurements(key, value):
    good = metrics(trial())
    broken = {**good, key: value}
    for candidate, incumbent in ((broken, good), (good, broken)):
        allowed, reason = confirmation_eligibility(candidate, incumbent, {})
        assert not allowed and reason == f"unmeasured_objective:{key}"
    assert not confirmation_eligibility({**good, "complete": False}, good, {})[0]


def test_split_is_frozen_disjoint_and_does_not_put_all_canaries_in_feedback():
    job = JobCreate()
    split = job.evaluation_split
    flat = split.feedback + split.selection + split.holdout
    assert len(flat) == len(set(flat)) == len(job.resolved_tasks())
    assert set(flat) == set(job.resolved_tasks())
    assert JobCreate().evaluation_split == split
    for part in (split.feedback, split.selection, split.holdout):
        assert any(BY_NAME[t]["role"] == "canary" for t in part)
    assert JobCreate(tasks=DEFAULT_TASKS[:1], max_iterations=0).evaluation_split is None
    with pytest.raises(ValidationError, match="at least three"):
        JobCreate(tasks=DEFAULT_TASKS[:2])
    with pytest.raises(ValidationError, match="disjoint"):
        JobCreate(tasks=DEFAULT_TASKS[:3], evaluation_split={"feedback": DEFAULT_TASKS[:1],
            "selection": DEFAULT_TASKS[:1], "holdout": DEFAULT_TASKS[1:3]})


class Pool:
    @asynccontextmanager
    async def acquire(self):
        yield self


class MemoryRepository:
    """Only test doubles for SQL; actual workflow code executes unchanged."""
    def __init__(self, monkeypatch):
        self.attempts, self.iterations, self.versions, self.events = [], [], {}, []
        self.finished = None
        for name in ("add_event", "list_events", "set_phase", "ensure_task_runs", "mark_task_running",
                     "attempt_results", "list_attempts", "next_attempt_number", "record_task_result",
                     "list_iterations", "upsert_agent_version", "create_iteration", "get_agent_version",
                     "complete_iteration", "update_job_progress", "finish_job", "lease_transaction"):
            monkeypatch.setattr(repo, name, getattr(self, name))

    @asynccontextmanager
    async def lease_transaction(self, conn, lease, **kw):
        yield {}

    async def add_event(self, conn, job, kind, payload=None):
        self.events.append((kind, payload))

    async def list_events(self, conn, job, after_id, limit):
        return [dict(id=i+1, kind=kind, payload=payload) for i, (kind, payload) in
                enumerate(self.events) if i+1 > after_id][:limit]

    async def set_phase(self, conn, lease, iteration, phase):
        pass

    async def ensure_task_runs(self, conn, iteration, tasks, *, lease):
        pass

    async def mark_task_running(self, conn, iteration, task, *, lease):
        pass

    async def list_attempts(self, conn, iteration, task=None):
        return [a for a in self.attempts if a["iteration_id"] == iteration and
                (task is None or a["task_name"] == task)]

    async def attempt_results(self, conn, iteration, evaluation_phase="main", repetition=1):
        rows = [a for a in self.attempts if a["iteration_id"] == iteration and
                a["evaluation_phase"] == evaluation_phase and a["repetition"] == repetition]
        chosen = {}
        for row in sorted(rows, key=lambda a: a["attempt"]):
            old = chosen.get(row["task_name"])
            if old is None or old["status"] == "error":
                chosen[row["task_name"]] = dict(row)
        return chosen

    async def next_attempt_number(self, conn, iteration, phase="main", repetition=1):
        return 1 + max([a["attempt"] for a in self.attempts if a["iteration_id"] == iteration and
                        a["evaluation_phase"] == phase and a["repetition"] == repetition], default=0)

    async def record_task_result(self, conn, lease, iteration, task, **kw):
        identity = (iteration, task, kw["evaluation_phase"], kw["repetition"], kw["attempt"])
        assert not any((a["iteration_id"], a["task_name"], a["evaluation_phase"], a["repetition"], a["attempt"]) == identity
                       for a in self.attempts), "an immutable attempt was overwritten"
        self.attempts.append(dict(iteration_id=iteration, task_name=task, **kw))

    async def list_iterations(self, conn, job):
        return list(self.iterations)

    async def upsert_agent_version(self, conn, org, source, parent=None):
        version = {"id": uuid4(), "content": source}
        self.versions[version["id"]] = version
        return version

    async def get_agent_version(self, conn, version):
        return self.versions[version]

    async def create_iteration(self, conn, lease, index, version, parent, proposal):
        row = dict(id=uuid4(), index=index, agent_version_id=version, parent_iteration_id=parent,
                   proposal=proposal, status="running", agent_sha256=repo.sha256_text(self.versions[version]["content"]))
        self.iterations.append(row)
        return row

    async def complete_iteration(self, conn, lease, iteration, **kw):
        row = next(i for i in self.iterations if i["id"] == iteration)
        row.update({"status": "completed", **kw})

    async def update_job_progress(self, conn, lease, **kw):
        pass

    async def finish_job(self, conn, lease, status, reason, **kw):
        self.finished = dict(status=status, reason=reason, **kw)


class FixtureExecutor:
    name = "fixture"
    def __init__(self, outcome=None):
        self.calls = []
        self.cancelled = False
        self.outcome = outcome or (lambda ctx, task, source: trial(task))

    async def run_benchmark(self, ctx, source, tasks, on_result, on_start=None):
        self.calls.append((ctx, list(tasks), source))
        results = {}
        for task in tasks:
            if on_start:
                await on_start(task)
            data = self.outcome(ctx, task, source)
            result = TaskResult(**data)
            results[task] = result
            await on_result(result)
        return results

    async def cancel(self):
        self.cancelled = True


class FixtureOptimizer:
    name = "fixture"
    def __init__(self, objective="efficiency", duplicate=False):
        self.inputs = []
        self.objective, self.duplicate = objective, duplicate

    async def propose(self, inp):
        self.inputs.append(deepcopy(inp))
        source = inp.agent_source if self.duplicate else inp.agent_source + f"\n# fixture candidate {inp.iteration_index}\n"
        return Proposal("controlled fixture", [], "fixture reason", source, "fixture change", objective=self.objective)


def runner(executor=None, optimizer=None, *, max_iterations=1, deadline=None, options=None):
    job_id = uuid4()
    job = {"id": job_id, "org_id": uuid4(), "deadline_at": deadline or datetime.now(timezone.utc) + timedelta(minutes=1),
           "config": {"tasks": ["feedback", "selection", "holdout"], "max_iterations": max_iterations,
                      "patience": 3, "agent_source": TEMPLATE_PATH.read_text(),
                      "evaluation_split": {"feedback": ["feedback"], "selection": ["selection"], "holdout": ["holdout"]},
                      "options": {"confirmation_repetitions": 2, **(options or {})}}}
    return JobRunner(Pool(), job, repo.Lease(job_id, "fixture", 1), executor or FixtureExecutor(), optimizer or FixtureOptimizer())


@pytest.mark.asyncio
async def test_screened_candidate_cannot_inherit_parent_passes_or_be_promoted(monkeypatch):
    db = MemoryRepository(monkeypatch)
    executor = FixtureExecutor(lambda ctx, task, source: trial(task, passed=task != "feedback"))
    optimizer = FixtureOptimizer(objective="quality")
    r = runner(executor, optimizer)
    await r.run()
    assert db.finished["status"] == "SUCCEEDED"
    candidate = db.iterations[1]
    assert candidate["decision"] == "rejected"
    assert candidate["score"] is None
    assert candidate["metrics"]["complete"] is False
    candidate_calls = [tasks for ctx, tasks, _ in executor.calls if ctx.iteration_index == 1]
    assert candidate_calls == [["feedback"]]
    assert not any(a["task_name"] == "selection" and a["iteration_id"] == candidate["id"] for a in db.attempts)


@pytest.mark.asyncio
async def test_paired_confirmation_preserves_main_evidence_and_rejects_lucky_candidate(monkeypatch):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        if task == "selection":
            if ctx.evaluation_phase == "main":
                return trial(task, passed=ctx.iteration_index == 1)
            return trial(task, passed=ctx.iteration_index == 0)
        return trial(task)
    executor = FixtureExecutor(outcome)
    r = runner(executor)
    await r.run()
    candidate = db.iterations[1]
    assert candidate["decision"] == "rejected"
    assert candidate["decision_reason"] == "quality_regression"
    assert candidate["score"] == 0
    assert candidate["metrics"]["incumbent"]["mean_reward"] == 1
    selection = [a for a in db.attempts if a["task_name"] == "selection"]
    assert len(selection) == 6  # baseline + candidate main, plus two paired repeats
    assert [a["status"] for a in selection if a["iteration_id"] == candidate["id"] and a["evaluation_phase"] == "main"] == ["passed"]
    assert all(tasks == ["selection"] for ctx, tasks, _ in executor.calls if ctx.evaluation_phase.startswith("confirm_"))


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["cost", "duration"])
async def test_preliminary_quality_gain_reaches_fresh_confirmation_despite_resource_spike(monkeypatch, resource):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        row = trial(task, passed=task != "selection" or ctx.iteration_index == 1)
        if ctx.iteration_index == 1 and ctx.evaluation_phase == "main":
            row["model_cost_usd" if resource == "cost" else "duration_s"] *= 3
        return row
    executor = FixtureExecutor(outcome)
    await runner(executor).run()
    candidate = db.iterations[1]
    assert candidate["decision"] == "accepted"
    assert candidate["metrics"]["preliminary"]["confirmation_eligibility"]["eligible"] is True
    assert not promotion_decision(candidate["metrics"]["preliminary"], db.iterations[0]["metrics"], {})[0]
    assert candidate["metrics"]["confirmation_repetitions"] == 2
    confirms = [(ctx.iteration_index, ctx.repetition, tasks) for ctx, tasks, _ in executor.calls
                if ctx.evaluation_phase.startswith("confirm_")]
    assert confirms == [(0, 1, ["selection"]), (1, 1, ["selection"]),
                        (1, 2, ["selection"]), (0, 2, ["selection"])]
    assert len([a for a in db.attempts if a["task_name"] == "selection"]) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("resource,reason", [("cost", "mean_model_cost_usd"), ("duration", "p95_task_duration_s")])
async def test_final_confirmation_still_rejects_resource_regression(monkeypatch, resource, reason):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        row = trial(task, passed=task != "selection" or ctx.iteration_index == 1)
        if ctx.iteration_index == 1:
            row["model_cost_usd" if resource == "cost" else "duration_s"] *= 2
        return row
    executor = FixtureExecutor(outcome)
    await runner(executor).run()
    candidate = db.iterations[1]
    assert candidate["metrics"]["confirmation_repetitions"] == 2
    assert candidate["decision"] == "rejected"
    assert candidate["decision_reason"] == f"efficiency_guardrail:{reason}"
    assert db.finished["best_iteration_id"] == db.iterations[0]["id"]


@pytest.mark.asyncio
async def test_preliminary_efficiency_signal_can_be_confirmed_despite_other_objective_spike(monkeypatch):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        return trial(task, cost=.8 if ctx.iteration_index else 1.,
                     duration=30. if ctx.iteration_index and ctx.evaluation_phase == "main" else 10.)
    await runner(FixtureExecutor(outcome)).run()
    candidate = db.iterations[1]
    assert candidate["decision"] == "accepted"
    assert candidate["metrics"]["preliminary"]["confirmation_eligibility"]["reason"].startswith("preliminary_efficiency")
    assert candidate["decision_reason"] == "verified_efficiency_gain_without_observed_quality_regression"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["quality_regression", "missing_reward", "unknown_cost"])
async def test_invalid_or_regressing_preliminary_evidence_never_dispatches_confirmation(monkeypatch, invalid_kind):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        row = trial(task, cost=.1 if ctx.iteration_index else 1.)
        if ctx.iteration_index and task == "selection":
            if invalid_kind == "quality_regression":
                row.update(status="failed", reward=0.)
            elif invalid_kind == "missing_reward":
                row.update(status="timeout", reward=None)
            else:
                row.update(cost_known=False)
        return row
    executor = FixtureExecutor(outcome)
    await runner(executor).run()
    assert db.iterations[1]["decision"] == "rejected"
    assert not any(ctx.evaluation_phase.startswith("confirm_") for ctx, _, _ in executor.calls)


@pytest.mark.asyncio
async def test_budget_exhaustion_during_newly_eligible_confirmation_never_promotes(monkeypatch):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        if ctx.evaluation_phase.startswith("confirm_"):
            raise repo.BudgetExceeded("bounded fixture budget exhausted")
        return trial(task, passed=task != "selection" or ctx.iteration_index == 1,
                     cost=3. if ctx.iteration_index else 1.)
    executor = FixtureExecutor(outcome)
    await runner(executor).run()
    assert db.iterations[1]["status"] == "errored"
    assert db.iterations[1]["decision"] == "rejected"
    assert db.finished["reason"] == "budget_exhausted"
    assert db.finished["best_iteration_id"] == db.iterations[0]["id"]
    assert db.finished["final_evaluation"]["status"] == "incomplete"
    assert executor.cancelled
    assert not any("holdout" in tasks for _, tasks, _ in executor.calls)


@pytest.mark.asyncio
async def test_perfect_score_efficiency_search_and_holdout_separation_end_to_end(monkeypatch):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        improved = ctx.iteration_index == 1
        return trial(task, passed=task != "holdout" or improved,
                     cost=.5 if improved else 1., duration=5. if improved else 10.,
                     instruction=f"instruction-{task}", test_stdout=f"verifier-{task}")
    executor, optimizer = FixtureExecutor(outcome), FixtureOptimizer()
    r = runner(executor, optimizer)
    await r.run()
    assert len(optimizer.inputs) == 1  # a perfect baseline did not stop efficiency work
    assert db.iterations[1]["decision"] == "accepted"
    final = db.finished["final_evaluation"]
    assert final["status"] == "complete" and final["used_for_selection"] is False
    assert final["baseline"]["mean_reward"] == 0 and final["final"]["mean_reward"] == 1
    inp = optimizer.inputs[0]
    assert set(inp.scoreboard) == {"feedback"}
    assert "holdout" not in repr(inp) and "selection" not in repr(inp.failure_dossiers)
    holdout_positions = [i for i, (ctx, tasks, _) in enumerate(executor.calls) if "holdout" in tasks]
    assert min(holdout_positions) > max(i for i, (ctx, _, _) in enumerate(executor.calls) if ctx.evaluation_phase.startswith("confirm_"))


@pytest.mark.asyncio
async def test_duplicate_source_is_durably_rejected_without_sandbox_execution(monkeypatch):
    db = MemoryRepository(monkeypatch)
    executor, optimizer = FixtureExecutor(), FixtureOptimizer(duplicate=True)
    r = runner(executor, optimizer, max_iterations=2)
    await r.run()
    assert [i["decision"] for i in db.iterations] == ["baseline", "rejected", "rejected"]
    assert all("duplicate_source" in i["decision_reason"] for i in db.iterations[1:])
    assert not any(ctx.iteration_index > 0 for ctx, _, _ in executor.calls)
    assert "duplicate_source" in optimizer.inputs[1].history[-1].decision_reason
    assert optimizer.inputs[1].scoreboard == {"feedback": [True, None]}


@pytest.mark.asyncio
async def test_invalid_source_preserves_feedback_scoreboard_iteration_alignment(monkeypatch):
    db = MemoryRepository(monkeypatch)
    snapshots = []
    contexts = []
    class InvalidThenValid(FixtureOptimizer):
        async def propose(self, inp):
            snapshots.append({task: list(values) for task, values in inp.scoreboard.items()})
            contexts.append(inp.learnings)
            proposal = await super().propose(inp)
            if inp.iteration_index == 1:
                proposal.new_source = "def broken(:"
                proposal.learnings = "A speculative note from an invalid proposal."
            return proposal
    executor = FixtureExecutor()
    await runner(executor, InvalidThenValid(), max_iterations=2).run()
    assert db.iterations[1]["decision_reason"].startswith("invalid_agent:")
    assert snapshots[1] == {"feedback": [True, None]}
    assert json.loads(contexts[1])["evaluation"] == {
        "decision": "rejected", "reason": db.iterations[1]["decision_reason"]}
    assert not any(ctx.iteration_index == 1 for ctx, _, _ in executor.calls)


@pytest.mark.asyncio
async def test_pretest_learning_is_labelled_with_observed_rejection_and_restored_after_lease_loss(monkeypatch):
    db = MemoryRepository(monkeypatch)
    raw_note = "This change always improves correctness."
    before_restart = []
    class InterruptedOptimizer(FixtureOptimizer):
        async def propose(self, inp):
            if inp.iteration_index == 2:
                before_restart.append(inp.learnings)
                raise repo.LeaseLost("fixture handoff after a completed rejection")
            proposal = await super().propose(inp)
            proposal.learnings = raw_note
            return proposal
    first = runner(optimizer=InterruptedOptimizer(), max_iterations=2)
    await first.run()
    rejected = db.iterations[1]
    assert rejected["decision"] == "rejected"
    assert rejected["learnings"] == raw_note  # durable source note remains verbatim
    recovered_optimizer = FixtureOptimizer()
    recovered = JobRunner(first.pool, first.job, first.lease, FixtureExecutor(), recovered_optimizer)
    await recovered.run()
    context = json.loads(recovered_optimizer.inputs[0].learnings)
    assert context == json.loads(before_restart[0])
    assert context["iteration"] == 1
    assert context["source"] == "optimizer_before_evaluation"
    assert context["status"] == "unverified_hypothesis"
    assert context["proposal_note"] == raw_note
    assert context["evaluation"] == {"decision": "rejected", "reason": rejected["decision_reason"]}


@pytest.mark.asyncio
async def test_infrastructure_retry_keeps_all_attempt_costs_and_never_retries_task_failure(monkeypatch):
    db = MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        row = trial(task, passed=task != "ordinary-failure", cost=.1, duration=3.)
        if task == "infra" and ctx.attempt == 1:
            row.update(status="error", reward=None, failure_class="infra_error", exit_reason="infra_boot_failed")
        return row
    executor = FixtureExecutor(outcome)
    r = runner(executor)
    it = uuid4()
    rows = await r._benchmark(it, 0, "unused", ["infra", "ordinary-failure"])
    assert rows["infra"]["status"] == "passed"
    assert rows["infra"]["physical_attempts"] == 2
    assert rows["infra"]["model_cost_usd"] == .2
    assert rows["infra"]["duration_s"] == 6.
    assert [tasks for _, tasks, _ in executor.calls] == [["infra", "ordinary-failure"], ["infra"]]
    assert len(db.attempts) == 3
    await r._benchmark(it, 0, "unused", ["infra", "ordinary-failure"])
    assert len(executor.calls) == 2  # resume reuses immutable completed observations


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen_harness", [None, {"mode": "prepared-v1", "images": {"selection": "sha256:frozen"}}])
async def test_worker_passes_persisted_harness_snapshot_without_reloading_configuration(monkeypatch, frozen_harness):
    MemoryRepository(monkeypatch)
    executor = FixtureExecutor()
    r = runner(executor)
    if frozen_harness is not None:
        r.cfg["sandbox_harness"] = frozen_harness
    await r._benchmark(uuid4(), 0, "source-as-data", ["selection"])
    ctx = executor.calls[0][0]
    assert ctx.extra["sandbox_harness"] == (frozen_harness if frozen_harness is not None else {"mode": "cold-v1"})
    assert ("sandbox_harness" in r.cfg) is (frozen_harness is not None)


@pytest.mark.asyncio
async def test_cancel_and_lease_loss_interrupt_running_external_work():
    for reason, exception in [("cancelled", Cancelled), ("lease_lost", repo.LeaseLost), ("database_unavailable", repo.LeaseLost)]:
        r = runner()
        started, interrupted = asyncio.Event(), asyncio.Event()
        async def operation():
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                interrupted.set()
        task = asyncio.create_task(r._interruptible(operation()))
        await started.wait()
        r.request_stop(reason)
        with pytest.raises(exception):
            await asyncio.wait_for(task, .5)
        assert interrupted.is_set() and r.executor.cancelled


@pytest.mark.asyncio
async def test_deadline_is_durable_and_interrupts_current_batch(monkeypatch):
    r = runner(deadline=datetime.now(timezone.utc) + timedelta(milliseconds=30))
    with pytest.raises(TimeoutError):
        await r._interruptible(asyncio.sleep(60))
    assert r.executor.cancelled
    old_job = {**r.job, "deadline_at": None,
               "started_at": datetime.now(timezone.utc) - timedelta(hours=5)}
    recovered = JobRunner(Pool(), old_job, r.lease, FixtureExecutor(), FixtureOptimizer())
    with pytest.raises(TimeoutError):
        await recovered._check_cancel()


@pytest.mark.asyncio
async def test_missing_callback_leaves_baseline_incomplete_instead_of_improving_score(monkeypatch):
    db = MemoryRepository(monkeypatch)
    executor = FixtureExecutor()
    async def incomplete(ctx, source, tasks, on_result, on_start=None):
        return {}
    executor.run_benchmark = incomplete
    r = runner(executor)
    await r.run()
    assert db.finished["status"] == "FAILED"
    assert "baseline incomplete" in db.finished["error"]
    assert db.finished["final_evaluation"]["status"] == "incomplete"
    assert db.iterations[0]["status"] == "errored"


@pytest.mark.asyncio
async def test_malformed_optimizer_response_has_one_bounded_repair_and_durable_reason(monkeypatch):
    from service.inference import InferenceError
    db = MemoryRepository(monkeypatch)
    optimizer = FixtureOptimizer()
    original = optimizer.propose
    calls = []
    async def propose(inp):
        calls.append(inp)
        if len(calls) == 1:
            raise InferenceError("Optimizer did not return valid JSON")
        return await original(inp)
    optimizer.propose = propose
    r = runner(optimizer=optimizer)
    await r.run()
    assert len(calls) == 2
    assert calls[1].constraints["previous_proposal_rejection"] == "invalid_json"
    rejected = [payload for kind, payload in db.events if kind == "optimizer.rejected"]
    assert rejected == [{"iteration": 1, "proposal_attempt": 1, "reason": "invalid_json", "repair_remaining": 1}]
    assert db.finished["status"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_optimizer_repair_limit_survives_process_recovery(monkeypatch):
    from service.inference import InferenceError
    db = MemoryRepository(monkeypatch)
    optimizer = FixtureOptimizer()
    async def invalid(inp):
        optimizer.inputs.append(inp)
        raise InferenceError("Optimizer output reached its token limit; private provider body must not be persisted")
    optimizer.propose = invalid
    r = runner(optimizer=optimizer)
    await r.run()
    assert len(optimizer.inputs) == 2
    assert db.finished["status"] == "FAILED"
    assert "private provider" not in repr(db.events)
    recovered = runner(optimizer=optimizer)
    await recovered.run()
    assert len(optimizer.inputs) == 2  # restart cannot create more paid retries
    assert "two durable metered attempts" in db.finished["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,reason", [(None, "cancelled"), (repo.LeaseLost("expired"), "lease_lost"),
                                            (ConnectionError("database unavailable"), "database_unavailable")])
async def test_heartbeat_stops_external_work_when_cancelled_or_ownership_uncertain(monkeypatch, failure, reason):
    from service.worker.main import _heartbeat_loop, settings
    monkeypatch.setattr(settings, "heartbeat_sec", .01)
    async def heartbeat(conn, lease, ttl):
        if failure:
            raise failure
        return True
    monkeypatch.setattr(repo, "heartbeat", heartbeat)
    r = runner()
    await asyncio.wait_for(_heartbeat_loop(r.pool, r.lease, r, asyncio.Event()), .5)
    assert r.cancel_flag.is_set() and r.stop_reason == reason
    assert r.executor.cancelled


@pytest.mark.asyncio
async def test_rejection_memory_contains_only_feedback_costs_causes_and_numeric_policy(monkeypatch):
    db = MemoryRepository(monkeypatch)
    executor = FixtureExecutor(lambda ctx, task, source: trial(task, passed=task != "feedback", failure_class="wrong_output" if task == "feedback" else None))
    optimizer = FixtureOptimizer(objective="quality")
    r = runner(executor, optimizer, max_iterations=2, options={"efficiency_tolerance": .2, "min_efficiency_improvement": .15})
    await r.run()
    inp = optimizer.inputs[1]
    policy = inp.constraints["promotion_policy"]
    assert policy["maximum_relative_model_cost_increase"] == .2
    assert policy["minimum_relative_efficiency_gain"] == .15
    assert "job_deadline_at" in inp.constraints["fixed_execution_limits"]
    summary = inp.history[-1].feedback_summary
    assert summary["scope"] == "feedback_only" and summary["failure_classes"] == {"wrong_output": 1}
    assert summary["mean_model_cost_usd_observed"] == 1
    assert "selection" not in repr(summary) and "holdout" not in repr(summary)
    assert set(inp.constraints["feedback_metrics"]) == {"feedback"}
    assert db.iterations[1]["metrics"]["feedback_summary"] == summary


@pytest.mark.asyncio
async def test_structural_observations_reach_optimizer_for_feedback_only(monkeypatch):
    MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        trace = [{"role":"assistant","tool_calls":[{"id":"tool1","function":{
            "name":"bash","arguments":json.dumps({"command":f"private-{task}-command"})}}]},
            {"role":"tool","tool_call_id":"tool1","content":"head "*300+"[999 chars truncated]"+" tail"*300}]
        return trial(task, passed=False, trace=trace, failure_class="wrong_output")
    optimizer = FixtureOptimizer(objective="quality")
    await runner(FixtureExecutor(outcome), optimizer).run()
    dossier = optimizer.inputs[0].failure_dossiers[0]
    assert dossier["task_name"] == "feedback"
    assert dossier["structural_observations"]["truncation_examples"][0]["reported_omitted_chars"] == 999
    assert "private-selection-command" not in repr(optimizer.inputs)
    assert "private-holdout-command" not in repr(optimizer.inputs)


@pytest.mark.asyncio
async def test_recent_rejected_dossier_exposes_actual_command_tail_without_other_splits(monkeypatch):
    MemoryRepository(monkeypatch)
    def outcome(ctx, task, source):
        command = "python3 - <<'PY'\n" + "# body\n" * 200 + f"print('data')\nPY\n> /tmp/{task}.json"
        trace = [{"role": "user", "content": "Task:\n" + f"visible-{task}-instruction"},
            {"role": "assistant", "tool_calls": [{"id": "one", "function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]},
            {"role": "tool", "tool_call_id": "one", "content": "data"}]
        return trial(task, passed=False, trace=trace, test_stdout=f"private-{task}-diagnostic")
    optimizer = FixtureOptimizer(objective="quality")
    await runner(FixtureExecutor(outcome), optimizer, max_iterations=2).run()
    recent = optimizer.inputs[1].constraints["recent_rejected_feedback_dossiers"]
    assert len(recent) == 1 and recent[0]["iteration_index"] == 1
    assert recent[0]["decision"] == "rejected" and "screened_out" in recent[0]["decision_reason"]
    assert "PY\n> /tmp/feedback.json" in recent[0]["dossier"]["last_steps"][0]["command"]
    assert "private-selection" not in repr(optimizer.inputs) and "private-holdout" not in repr(optimizer.inputs)
    capabilities = optimizer.inputs[0].constraints["runtime_capabilities"]
    assert "fresh bash shell" in capabilities["shell"]
    assert "unavailable during execution" in capabilities["verification"]


@pytest.mark.asyncio
async def test_prior_recent_dossiers_are_feedback_filtered_and_bounded(monkeypatch):
    MemoryRepository(monkeypatch)
    optimizer = FixtureOptimizer()
    r = runner(optimizer=optimizer)
    r.cfg["optimizer_recent_feedback_seed"] = [
        {"source_job_id": "prior", "iteration_index": index, "decision": "rejected",
         "decision_reason": "observed_failure", "task_name": task,
         "dossier": {"instruction": f"private-{task}-{index}"}}
        for index in range(5) for task in ("feedback", "selection", "holdout")]
    await r.run()
    recent = optimizer.inputs[0].constraints["recent_rejected_feedback_dossiers"]
    assert [d["iteration_index"] for d in recent] == [3, 4]
    assert all(d["task_name"] == "feedback" for d in recent)
    assert "private-selection" not in repr(optimizer.inputs) and "private-holdout" not in repr(optimizer.inputs)


@pytest.mark.asyncio
async def test_recent_feedback_context_restores_identically_after_lease_loss(monkeypatch):
    MemoryRepository(monkeypatch)
    observed = []
    class Handoff(FixtureOptimizer):
        async def propose(self, inp):
            if inp.iteration_index == 2:
                observed.append(deepcopy(inp.constraints["recent_rejected_feedback_dossiers"]))
                raise repo.LeaseLost("fixture handoff")
            return await super().propose(inp)
    executor = FixtureExecutor(lambda ctx, task, source: trial(task, passed=False,
        test_stdout=f"observed-{task}-failure", instruction=f"specification-{task}"))
    first = runner(executor, Handoff(objective="quality"), max_iterations=2)
    await first.run()
    optimizer = FixtureOptimizer(objective="quality")
    await JobRunner(first.pool, first.job, first.lease, executor, optimizer).run()
    restored = optimizer.inputs[0].constraints["recent_rejected_feedback_dossiers"]
    assert restored == observed[0] and len(restored) == 1
    assert "observed-feedback-failure" in repr(restored)
    assert "observed-selection-failure" not in repr(restored)


@pytest.mark.asyncio
async def test_prior_rejections_do_not_count_against_new_experiment_patience(monkeypatch):
    MemoryRepository(monkeypatch)
    optimizer = FixtureOptimizer()
    r = runner(optimizer=optimizer)
    r.patience = 1
    r.cfg["optimizer_history_seed"] = [{"index": i, "hypothesis": "prior idea", "decision": "rejected", "score": None,
        "decision_reason": "screened_out", "source_job_id": "prior", "feedback_summary": {"scope": "feedback_only"}}
        for i in (1, 2, 3)]
    await r.run()
    assert len(optimizer.inputs) == 1
    seeded = optimizer.inputs[0].history[:3]
    assert all(entry.source_job_id == "prior" and entry.score is None for entry in seeded)


@pytest.mark.asyncio
async def test_resume_candidate_keeps_original_baseline_and_evaluates_fresh_without_optimizer(monkeypatch):
    db = MemoryRepository(monkeypatch)
    source = TEMPLATE_PATH.read_text() + "\n# prior unconfirmed candidate\n"
    def outcome(ctx, task, candidate):
        return trial(task, passed=task != "feedback", cost=.5 if ctx.iteration_index else 1,
                     duration=5 if ctx.iteration_index else 10)
    executor, optimizer = FixtureExecutor(outcome), FixtureOptimizer()
    r = runner(executor, optimizer)
    r.cfg["resume_candidate"] = {"source": source,
        "proposal": {"hypothesis": "prior verification", "objective": "quality"},
        "source_job_id": "prior", "source_iteration_index": 2}
    await r.run()
    assert optimizer.inputs == []
    assert db.versions[db.iterations[0]["agent_version_id"]]["content"] == TEMPLATE_PATH.read_text()
    assert db.versions[db.iterations[1]["agent_version_id"]]["content"] == source
    assert db.iterations[1]["proposal"]["resumed_from"] == {"job_id": "prior", "iteration": 2}
    # The prior failure has not become a fictional pass. This fresh full plan
    # bypasses screening and must produce its own main + paired evidence.
    candidate_calls = [(ctx, tasks) for ctx, tasks, _ in executor.calls if ctx.iteration_index == 1]
    assert candidate_calls[0][1] == ["feedback", "selection"]
    assert db.iterations[1]["decision"] == "accepted"
    assert len([a for a in db.attempts if a["iteration_id"] == db.iterations[1]["id"] and a["task_name"] == "selection"]) == 3


@pytest.mark.asyncio
async def test_exact_rejected_source_hash_is_deduplicated_across_related_jobs(monkeypatch):
    db = MemoryRepository(monkeypatch)
    executor, optimizer = FixtureExecutor(), FixtureOptimizer()
    prior_source = TEMPLATE_PATH.read_text() + "\n# fixture candidate 1\n"
    r = runner(executor, optimizer)
    r.cfg["optimizer_history_seed"] = [{"index": 1, "hypothesis": "old candidate", "decision": "rejected", "score": None,
        "decision_reason": "screened_out", "source_job_id": "prior", "agent_sha256": repo.sha256_text(prior_source)}]
    await r.run()
    assert "duplicate_source" in db.iterations[1]["decision_reason"]
    assert not any(ctx.iteration_index > 0 for ctx, _, _ in executor.calls)


@pytest.mark.parametrize("status", ["passed", "failed", "timeout"])
@pytest.mark.parametrize("reward", [None, True, False])
def test_terminal_status_never_substitutes_for_an_actual_numeric_reward(status, reward):
    observation = trial()
    observation.update(status=status, reward=reward)
    incomplete = metrics(observation)
    assert incomplete["complete"] is False
    assert incomplete["mean_reward"] is None
    accepted, reason = promotion_decision(incomplete, metrics(trial()), {})
    assert accepted is False and reason.startswith("incomplete_comparison")


def test_verified_timeout_with_explicit_zero_remains_a_scored_outcome():
    observation = trial(passed=False)
    observation.update(status="timeout", reward=0.0)
    result = metrics(observation)
    assert result["complete"] is True
    assert result["mean_reward"] == 0.0
