"""Durable candidate evaluation with immutable evidence and a sealed final holdout.

Phases are progress markers. On recovery, persisted attempt identities determine what
can be reused; no unexecuted task is assigned another agent's result.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from .. import repo
from ..config import settings
from ..inference import InferenceError
from ..repo import Lease, LeaseLost
from .digest import build_digest, dossier_from_row
from .executors.base import Executor, RunContext, TaskResult
from .optimizer import HistoryEntry, Optimizer, OptimizerInput

log = logging.getLogger("worker.job")
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "agent" / "templates" / "terminal_bench.py"
TERMINAL_TRIAL_STATUSES = {"passed", "failed", "timeout"}


def rejected_feedback_dossiers(rows, feedback, *, source_job_id, index, reason):
    """Only measured diagnostic failures, bounded separately from incumbent data."""
    return [{"source_job_id": str(source_job_id), "iteration_index": index,
             "decision": "rejected", "decision_reason": reason, "task_name": task,
             "dossier": dossier_from_row(dict(rows[task]))}
            for task in feedback if task in rows and rows[task].get("status") == "failed"
            and type(rows[task].get("reward")) in (int, float)
            and math.isfinite(rows[task]["reward"])][:3]


def bounded_recent_feedback(entries, feedback):
    retained, groups = [], []
    for entry in reversed(entries):
        if entry.get("task_name") not in feedback or entry.get("decision") != "rejected":
            continue
        identity = (entry.get("source_job_id"), entry.get("iteration_index"))
        if identity not in groups:
            if len(groups) == 2:
                continue
            groups.append(identity)
        retained.append(entry)
        if len(retained) == 6:
            break
    return list(reversed(retained))


class Cancelled(Exception):
    pass


@dataclass
class IterOutcome:
    iteration_id: UUID
    index: int
    agent_version_id: UUID
    agent_source: str
    results: dict[str, Any]
    score: float | None
    n_pass: int
    n_fail: int
    n_error: int
    metrics: dict[str, Any] = field(default_factory=dict)
    screened: bool = False


def evaluation_metrics(samples: list[dict], expected_count: int) -> dict:
    """Summarize a fixed plan. Missing/infra-failed observations never improve score.

    Dollar comparisons use attributable model cost. Local Docker resource use is
    reported as task time, not invented as a zero-dollar cloud bill.
    """
    terminal = [r for r in samples if r.get("status") in TERMINAL_TRIAL_STATUSES]
    complete = len(samples) == expected_count and len(terminal) == expected_count and expected_count > 0
    passed = sum(r.get("status") == "passed" for r in samples)
    errors = sum(r.get("status") == "error" for r in samples)
    rewards = []
    for r in terminal:
        reward = r.get("reward")
        # A terminal status is not a verifier reward. In particular, the
        # executor's outer timeout has no grading evidence. Keep such results
        # incomplete, including historical rows, instead of inventing zero/one.
        # Simulated outcomes and verified execution supply explicit rewards.
        if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
            complete = False
        else:
            rewards.append(float(reward))
    durations = [r.get("duration_s") for r in samples]
    runtime_known = complete and all(isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 for d in durations)
    costs = [r.get("model_cost_usd") if r.get("model_cost_usd") is not None else r.get("cost_usd") for r in samples]
    cost_known = complete and all(r.get("model_cost_known", (r.get("metadata") or {}).get(
        "model_cost_known", r.get("cost_known", False))) for r in samples) and all(
        isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in costs)
    sandbox_costs = [r.get("sandbox_cost_usd") for r in samples]
    sandbox_known = complete and all(isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in sandbox_costs)
    return {
        "complete": complete, "planned_trials": expected_count, "observed_trials": len(samples),
        "passed": passed, "failed": len(terminal) - passed, "errors": errors,
        "mean_reward": sum(rewards) / expected_count if complete else None,
        "mean_task_duration_s": sum(durations) / expected_count if runtime_known else None,
        "p95_task_duration_s": sorted(durations)[max(0, math.ceil(.95 * expected_count) - 1)] if runtime_known else None,
        "mean_model_cost_usd": sum(costs) / expected_count if cost_known else None,
        "total_model_cost_usd": sum(costs) if cost_known else None,
        "total_sandbox_cost_usd": sum(sandbox_costs) if sandbox_known else None,
        "input_tokens": sum(r.get("input_tokens") or 0 for r in samples),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in samples),
        "physical_attempts": sum(r.get("physical_attempts", 1) for r in samples),
        "cost_scope": "model_inference; sandbox dollars reported separately when priced",
        "evidence_kind": "simulated" if samples and all((r.get("metadata") or {}).get("simulated") for r in samples) else "execution",
    }


def promotion_decision(candidate: dict, incumbent: dict, options: dict) -> tuple[bool, str]:
    """A declared epsilon-Pareto policy, with no step-count cost proxy."""
    if not candidate.get("complete") or not incumbent.get("complete"):
        return False, "incomplete_comparison: all planned trials must finish with valid verifier outcomes"
    quality = candidate["mean_reward"]
    old_quality = incumbent["mean_reward"]
    if quality < old_quality:
        return False, "quality_regression"
    tolerance = float(options.get("efficiency_tolerance", .10))
    threshold = float(options.get("min_efficiency_improvement", .10))
    efficiency_gain = False
    for key in ("mean_model_cost_usd", "p95_task_duration_s"):
        current, previous = candidate.get(key), incumbent.get(key)
        if current is None or previous is None:
            return False, f"unmeasured_objective:{key}"
        if current > previous * (1 + tolerance) + 1e-12:
            return False, f"efficiency_guardrail:{key}"
        if previous > 0 and current <= previous * (1 - threshold):
            efficiency_gain = True
    if quality > old_quality + float(options.get("min_improvement", 0)):
        return True, "verified_quality_gain_within_declared_efficiency_tolerances"
    if quality > 0 and efficiency_gain:
        return True, "verified_efficiency_gain_without_observed_quality_regression"
    return False, "no_meaningful_improvement"


def confirmation_eligibility(candidate: dict, incumbent: dict, options: dict) -> tuple[bool, str]:
    """Decide whether to spend on fresh paired evidence, not whether to promote.

    A single preliminary run can contain setup/provider noise. Keep the declared
    minimum improvement signal and reject missing evidence or observed quality
    regression, but apply cost/latency regression ceilings only to confirmation.
    The durable job budget and deadline still bound this extra measurement.
    """
    if not candidate.get("complete") or not incumbent.get("complete"):
        return False, "incomplete_comparison: all planned trials must finish with valid verifier outcomes"
    for key in ("mean_reward", "mean_model_cost_usd", "p95_task_duration_s"):
        values = (candidate.get(key), incumbent.get(key))
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               or (key == "mean_reward" and value > 1) for value in values):
            return False, f"unmeasured_objective:{key}"
    quality, old_quality = candidate["mean_reward"], incumbent["mean_reward"]
    if quality < old_quality:
        return False, "quality_regression"
    if quality > old_quality + float(options.get("min_improvement", 0)):
        return True, "preliminary_quality_signal_requires_confirmation"
    threshold = float(options.get("min_efficiency_improvement", .10))
    if quality > 0 and any(
        incumbent[key] > 0 and candidate[key] <= incumbent[key] * (1 - threshold)
        for key in ("mean_model_cost_usd", "p95_task_duration_s")
    ):
        return True, "preliminary_efficiency_signal_requires_confirmation"
    return False, "no_meaningful_preliminary_signal"


def hypothesis_context(index: int, note: str, decision: str | None, reason: str | None) -> str:
    """Separate a pre-test optimizer suggestion from the measured decision.

    Persist the original note unchanged; this labelled representation is only
    for subsequent proposal context, including when rebuilding after a restart.
    """
    return json.dumps({"iteration": index, "source": "optimizer_before_evaluation",
        "status": "unverified_hypothesis", "proposal_note": note,
        "evaluation": {"decision": decision, "reason": reason}}, ensure_ascii=False)


def summarize_feedback(rows: dict, tasks: list[str]) -> dict:
    """Measured diagnostic evidence; partial screens keep an explicit denominator."""
    observed = [dict(rows[task]) for task in tasks if task in rows]
    summary = evaluation_metrics(observed, len(observed))
    failures: dict[str, int] = {}
    for row in observed:
        if row.get("status") != "passed":
            cause = row.get("failure_class") or (row.get("digest") or {}).get("stats", {}).get("failure_class") or "unknown"
            failures[cause] = failures.get(cause, 0) + 1
    return {"scope": "feedback_only", "observed_tasks": len(observed), "planned_tasks": len(tasks),
        "coverage_complete": len(observed) == len(tasks), "valid_observations": summary["complete"],
        "passed": summary["passed"], "failure_classes": failures,
        "mean_reward_observed": summary["mean_reward"], "mean_model_cost_usd_observed": summary["mean_model_cost_usd"],
        "p95_task_duration_s_observed": summary["p95_task_duration_s"], "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"], "physical_attempts": summary["physical_attempts"]}


class JobRunner:
    def __init__(self, pool: asyncpg.Pool, job, lease: Lease, executor: Executor, optimizer: Optimizer):
        self.pool, self.job, self.lease, self.executor, self.optimizer = pool, job, lease, executor, optimizer
        self.cfg = job["config"]
        self.tasks = list(self.cfg["tasks"])
        self.max_iterations = int(self.cfg.get("max_iterations", 5))
        self.patience = int(self.cfg.get("patience", 3))
        self.opts = self.cfg.get("options") or {}
        self.seed = int(self.cfg.get("seed") if self.cfg.get("seed") is not None else 1234)
        split = self.cfg.get("evaluation_split")
        if self.max_iterations > 0:
            if not split:
                raise ValueError("optimization jobs require a persisted feedback/selection/holdout split")
            self.feedback, self.selection, self.holdout = (list(split[k]) for k in ("feedback", "selection", "holdout"))
            if (not self.feedback or not self.selection or not self.holdout or
                len(set(self.feedback + self.selection + self.holdout)) != len(self.tasks) or
                set(self.feedback + self.selection + self.holdout) != set(self.tasks)):
                raise ValueError("invalid persisted evaluation split")
        else:
            self.feedback, self.selection, self.holdout = [], self.tasks, []
        frozen_deadline = job.get("deadline_at")
        if frozen_deadline is None:
            # Compatibility with older rows. Never reset a recovered job's clock.
            anchor = job.get("started_at") or job.get("created_at")
            if anchor is None:
                raise ValueError("job needs a durable deadline or creation timestamp")
            frozen_deadline = anchor + timedelta(minutes=int((self.cfg.get("budget") or {}).get("max_wall_clock_min", 120)))
        self.deadline_at = frozen_deadline if frozen_deadline.tzinfo else frozen_deadline.replace(tzinfo=timezone.utc)
        self.cancel_flag = asyncio.Event()
        self.stop_reason: str | None = None
        self.proposal_attempts: dict[int, int] = {}
        self.proposal_rejections: dict[int, str] = {}
        self.work_root = Path(settings.runs_dir) / str(job["id"])

    def request_stop(self, reason: str = "cancelled") -> None:
        self.stop_reason = reason
        self.cancel_flag.set()

    def _seconds_left(self) -> float:
        return (self.deadline_at - datetime.now(timezone.utc)).total_seconds()

    async def _check_cancel(self):
        if self.cancel_flag.is_set():
            if self.stop_reason in ("lease_lost", "database_unavailable", "shutdown"):
                raise LeaseLost(self.stop_reason)
            raise Cancelled()
        if self._seconds_left() <= 0:
            raise TimeoutError("durable job deadline exhausted")

    async def _interruptible(self, operation):
        """Interrupt running model/provider work instead of waiting for a whole batch."""
        task = asyncio.ensure_future(operation)
        waiter = asyncio.create_task(self.cancel_flag.wait())
        try:
            await self._check_cancel()
            done, _ = await asyncio.wait({task, waiter}, timeout=self._seconds_left(), return_when=asyncio.FIRST_COMPLETED)
            if waiter in done or not done:
                await self.executor.cancel()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(task, timeout=5)
                await self._check_cancel()
                raise TimeoutError("durable job deadline exhausted")
            result = await task
            await self._check_cancel()
            return result
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(task, timeout=5)

    async def _event(self, kind: str, payload: dict | None = None):
        async with self.pool.acquire() as conn:
            async with repo.lease_transaction(conn, self.lease):
                await repo.add_event(conn, self.lease.job_id, kind, payload)

    async def _set_phase(self, iteration_id: UUID, phase: str):
        async with self.pool.acquire() as conn:
            await repo.set_phase(conn, self.lease, iteration_id, phase)

    async def _restore_proposal_attempts(self):
        # Started attempts are durable before dispatch. A process restart cannot
        # reset the two-request cap, including an ambiguous interrupted request.
        cursor = 0
        while True:
            async with self.pool.acquire() as conn:
                events = await repo.list_events(conn, self.lease.job_id, cursor, 500)
            for event in events:
                payload = event["payload"] or {}
                index = payload.get("iteration")
                if type(index) is not int:
                    continue
                if event["kind"] == "optimizer.proposal_started":
                    self.proposal_attempts[index] = max(self.proposal_attempts.get(index, 0), payload.get("proposal_attempt", 0))
                elif event["kind"] == "optimizer.rejected":
                    self.proposal_rejections[index] = payload.get("reason", "invalid_response")
            if len(events) < 500:
                break
            cursor = events[-1]["id"]

    async def _propose(self, inp: OptimizerInput):
        index = inp.iteration_index
        while self.proposal_attempts.get(index, 0) < 2:
            attempt = self.proposal_attempts.get(index, 0) + 1
            if attempt > 1:
                inp.constraints = {**inp.constraints, "repair_attempt": 1,
                    "previous_proposal_rejection": self.proposal_rejections.get(index, "interrupted_before_candidate_persistence")}
            await self._event("optimizer.proposal_started", {"iteration": index, "proposal_attempt": attempt})
            self.proposal_attempts[index] = attempt
            try:
                return await self._interruptible(self.optimizer.propose(inp))
            except InferenceError as exc:
                # Record a stable diagnostic, never an upstream body, prompt or
                # provider credential. Budget/cancel/lease exceptions propagate.
                message = str(exc).lower()
                reason = next((code for phrase, code in (
                    ("valid json", "invalid_json"), ("token limit", "truncated_output"),
                    ("required string", "missing_fields"), ("targets", "invalid_targets"),
                    ("outside its diagnostic", "invalid_targets"), ("80kb", "source_too_large"),
                    ("objective", "invalid_objective"), ("completion choice", "missing_completion"))
                    if phrase in message), "provider_or_validation_error")
                self.proposal_rejections[index] = reason
                await self._event("optimizer.rejected", {"iteration": index, "proposal_attempt": attempt,
                                                         "reason": reason, "repair_remaining": 2 - attempt})
        raise InferenceError("Optimizer failed to produce a valid proposal within two durable metered attempts")

    @staticmethod
    def _score(rows: dict) -> tuple[float | None, int, int, int]:
        metrics = evaluation_metrics([dict(r) for r in rows.values()], len(rows))
        return metrics["mean_reward"], metrics["passed"], metrics["failed"], metrics["errors"]

    async def _phase_results(self, iteration_id, phase="main", repetition=1):
        async with self.pool.acquire() as conn:
            chosen = await repo.attempt_results(conn, iteration_id, phase, repetition)
            attempts = await repo.list_attempts(conn, iteration_id)
        rows = {task: dict(row) for task, row in chosen.items()}
        for task, row in rows.items():
            trials = [a for a in attempts if a["task_name"] == task and
                      a["evaluation_phase"] == phase and a["repetition"] == repetition]
            row["physical_attempts"] = len(trials)
            # A recovered infrastructure retry must not make a candidate appear
            # cheaper or faster by discarding the failed attempt's resource use.
            for field in ("duration_s", "model_cost_usd", "cost_usd", "sandbox_cost_usd"):
                values = [a.get(field) for a in trials]
                row[field] = sum(values) if values and all(v is not None for v in values) else None
            for field in ("input_tokens", "output_tokens"):
                row[field] = sum(a.get(field) or 0 for a in trials)
            row["cost_known"] = bool(trials) and all(a.get("cost_known", False) for a in trials)
            row["model_cost_known"] = bool(trials) and all((a.get("metadata") or {}).get(
                "model_cost_known", a.get("cost_known", False)) for a in trials)
        return rows

    async def _benchmark(self, iteration_id, index, agent_source, tasks, *, phase="main", repetition=1):
        await self._check_cancel()
        previous = await self._phase_results(iteration_id, phase, repetition)
        pending = [task for task in tasks if task not in previous or (
            previous[task]["status"] == "error" and
            (previous[task].get("failure_class") == "infra_error" or
             (previous[task].get("exit_reason") or "").startswith("infra")) and
            previous[task].get("physical_attempts", 1) < self.opts.get("max_infrastructure_attempts", 2))]
        if not pending:
            return previous
        async with self.pool.acquire() as conn:
            if phase == "main" and repetition == 1:
                await repo.ensure_task_runs(conn, iteration_id, tasks, lease=self.lease)
            attempt = await repo.next_attempt_number(conn, iteration_id, phase, repetition)
        ctx = RunContext(job_id=self.lease.job_id, iteration_index=index, iteration_id=iteration_id,
            org_id=self.job["org_id"], agent_model=self.cfg.get("agent_model", settings.agent_model),
            seed=self.seed + repetition - 1, attempt=attempt,
            work_dir=self.work_root / f"iter{index}" / phase / f"rep{repetition}" / f"attempt{attempt}",
            extra={**self.opts, "benchmark_revision": self.cfg.get("benchmark_revision", settings.benchmark_revision),
                   "sandbox_harness": self.cfg.get("sandbox_harness", {"mode": "cold-v1"})},
            lease_owner=self.lease.owner, lease_epoch=self.lease.epoch,
            deadline_at=self.deadline_at, evaluation_phase=phase, repetition=repetition, lease=self.lease)

        async def on_start(task):
            if phase == "main" and repetition == 1:
                async with self.pool.acquire() as conn:
                    await repo.mark_task_running(conn, iteration_id, task, lease=self.lease)

        async def on_result(result: TaskResult):
            if result.task_name not in pending:
                raise ValueError("executor returned a task outside its frozen plan")
            digest = build_digest(result, result.instruction, self.opts.get("max_agent_steps", 80))
            async with self.pool.acquire() as conn:
                await repo.record_task_result(conn, self.lease, iteration_id, result.task_name,
                    status=result.status, reward=result.reward, failure_class=digest["stats"]["failure_class"],
                    duration_s=result.duration_s, n_steps=result.n_steps, exit_reason=result.exit_reason,
                    test_stdout=result.test_stdout, trace=result.trace, digest=digest, sandbox_id=result.sandbox_id,
                    evaluation_phase=phase, repetition=repetition, attempt=attempt,
                    input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                    cost_usd=result.cost_usd, model_cost_usd=result.model_cost_usd,
                    sandbox_cost_usd=result.sandbox_cost_usd, cost_known=result.cost_known, metadata=result.metadata)
            await self._event("task.finished", {"iteration": index,
                "phase": phase, "task": result.task_name, "status": result.status,
                "repetition": repetition, "attempt": attempt})

        await self._interruptible(self.executor.run_benchmark(ctx, agent_source, pending, on_result, on_start))
        results = await self._phase_results(iteration_id, phase, repetition)
        retryable = any(row["status"] == "error" and
                        (row.get("failure_class") == "infra_error" or
                         (row.get("exit_reason") or "").startswith("infra")) and
                        row.get("physical_attempts", 1) < self.opts.get("max_infrastructure_attempts", 2)
                        for task, row in results.items() if task in tasks)
        if retryable:
            await self._event("tasks.infrastructure_retry", {"iteration": index, "phase": phase})
            return await self._benchmark(iteration_id, index, agent_source, tasks, phase=phase, repetition=repetition)
        return results

    def _outcome(self, iteration, source, rows, *, screened=False, metrics=None):
        if metrics is None:
            metrics = evaluation_metrics([dict(rows[t]) for t in self.selection if t in rows], len(self.selection))
        metrics = {**metrics, "scope": "selection" if self.max_iterations else "benchmark",
                   "selection_tasks": self.selection, "screened": screened}
        if screened:
            metrics["complete"] = False
            metrics["mean_reward"] = None
        return IterOutcome(iteration["id"], iteration["index"], iteration["agent_version_id"], source, dict(rows),
            metrics["mean_reward"], metrics["passed"], metrics["failed"], metrics["errors"], metrics, screened)

    async def _run_iteration(self, iteration, agent_source, parent: IterOutcome | None):
        index, iteration_id = iteration["index"], iteration["id"]
        await self._set_phase(iteration_id, "BENCHMARK")
        proposal = iteration.get("proposal") or {}
        if (parent is not None and not proposal.get("resumed_from") and self.opts.get("two_phase_eval", True)
                and proposal.get("objective", "quality") == "quality"):
            failing = [t for t in self.feedback if t in parent.results and parent.results[t]["status"] != "passed"]
            controls = [t for t in self.feedback if t in parent.results and parent.results[t]["status"] == "passed"][:1]
            if failing:
                rows = await self._benchmark(iteration_id, index, agent_source, failing + controls)
                has_gain = any(rows.get(t, {}).get("status") == "passed" for t in failing)
                if not has_gain:
                    await self._event("iteration.screened_out", {"iteration": index, "tasks": failing + controls})
                    return self._outcome(iteration, agent_source, rows, screened=True)
        rows = await self._benchmark(iteration_id, index, agent_source, self.feedback + self.selection)
        out = self._outcome(iteration, agent_source, rows)
        if parent is None:
            return out
        eligible, reason = confirmation_eligibility(out.metrics, parent.metrics, self.opts)
        out.metrics["confirmation_eligibility"] = {"eligible": eligible, "reason": reason}
        if not eligible:
            return out
        await self._event("confirmation.scheduled", {"iteration": index, "reason": reason})
        if not self.opts.get("confirm_flips", True):
            # The legacy switch never disables the correctness gate in live optimization.
            await self._event("confirmation.required", {"iteration": index, "legacy_switch_ignored": True})
        await self._set_phase(iteration_id, "CONFIRM")
        repetitions = int(self.opts.get("confirmation_repetitions", 2))
        phase = f"confirm_{index}"
        parent_samples, candidate_samples = [], []
        for rep in range(1, repetitions + 1):
            # Alternate order across repetitions to reduce simple timing/provider bias.
            evaluations = [(parent, parent_samples), (out, candidate_samples)]
            if rep % 2 == 0:
                evaluations.reverse()
            for version, samples in evaluations:
                result = await self._benchmark(version.iteration_id, version.index, version.agent_source,
                    self.selection, phase=phase, repetition=rep)
                samples.extend(dict(result[t]) for t in self.selection if t in result)
        expected = len(self.selection) * repetitions
        candidate_metrics = evaluation_metrics(candidate_samples, expected)
        parent_metrics = evaluation_metrics(parent_samples, expected)
        candidate_metrics.update({"scope": "selection_confirmation", "selection_tasks": self.selection,
                                  "confirmation_repetitions": repetitions, "incumbent": parent_metrics,
                                  "preliminary": out.metrics})
        out.metrics = candidate_metrics
        out.score = candidate_metrics["mean_reward"]
        out.n_pass, out.n_fail, out.n_error = (candidate_metrics[k] for k in ("passed", "failed", "errors"))
        return out

    async def _load_outcome(self, iteration):
        async with self.pool.acquire() as conn:
            version = await repo.get_agent_version(conn, iteration["agent_version_id"])
        rows = await self._phase_results(iteration["id"])
        metrics = dict(iteration.get("metrics") or {})
        if not metrics:
            metrics = evaluation_metrics([dict(rows[t]) for t in self.selection if t in rows], len(self.selection))
        return self._outcome(iteration, version["content"], rows, metrics=metrics)

    async def _finalize(self, reason, best, baseline):
        final = {"status": "not_applicable", "reason": "benchmark_only"}
        if self.holdout and best is not None and baseline is not None:
            await self._event("holdout.started", {"baseline_iteration": baseline.index, "final_iteration": best.index})
            comparisons = {}
            for label, version in (("baseline", baseline), ("final", best)):
                rows = await self._benchmark(version.iteration_id, version.index, version.agent_source,
                                             self.holdout, phase="holdout_final")
                comparisons[label] = evaluation_metrics([dict(rows[t]) for t in self.holdout if t in rows], len(self.holdout))
            final = {"status": "complete" if all(v["complete"] for v in comparisons.values()) else "incomplete",
                     "tasks": self.holdout, "baseline_iteration": baseline.index, "final_iteration": best.index,
                     **comparisons, "used_for_selection": False,
                     "limitation": "small fixed subset; repeats do not establish broad benchmark generalization"}
        async with self.pool.acquire() as conn:
            await repo.finish_job(conn, self.lease, "SUCCEEDED", reason,
                best_iteration_id=best.iteration_id if best else None,
                best_score=best.score if best else None,
                baseline_score=baseline.score if baseline else None, final_evaluation=final)

    async def _abandon_iteration(self, iteration_id, reason):
        if iteration_id is None:
            return
        async with self.pool.acquire() as conn:
            await repo.complete_iteration(conn, self.lease, iteration_id, score=None, n_pass=0,
                n_fail=0, n_error=0, decision="rejected", decision_reason=reason, status="errored",
                metrics={"complete": False, "interrupted": True, "reason": reason})

    async def run(self):
        best = baseline = None
        active_iteration_id = None
        try:
            await self._restore_proposal_attempts()
            async with self.pool.acquire() as conn:
                existing = await repo.list_iterations(conn, self.lease.job_id)
            history = [HistoryEntry(**entry) for entry in self.cfg.get("optimizer_history_seed", [])]
            recent_feedback = bounded_recent_feedback(self.cfg.get("optimizer_recent_feedback_seed", []), self.feedback)
            learnings = []
            seen_hashes = {entry.agent_sha256 for entry in history if entry.agent_sha256}
            scoreboard = {task: [] for task in self.feedback}
            no_improve = 0
            for iteration in existing:
                seen_hashes.add(iteration["agent_sha256"])
                if iteration["status"] != "completed":
                    continue
                out = await self._load_outcome(iteration)
                prop = iteration.get("proposal") or {}
                history.append(HistoryEntry(iteration["index"], prop.get("hypothesis"), iteration["decision"],
                    iteration["score"], decision_reason=iteration.get("decision_reason"), change_summary=prop.get("change_summary"),
                    feedback_summary=(iteration.get("metrics") or {}).get("feedback_summary") or summarize_feedback(out.results, self.feedback)))
                for task in self.feedback:
                    scoreboard[task].append(out.results[task]["status"] == "passed" if task in out.results else None)
                if iteration.get("learnings"):
                    learnings.append(hypothesis_context(iteration["index"], iteration["learnings"],
                                                       iteration["decision"], iteration.get("decision_reason")))
                if iteration["decision"] == "baseline":
                    baseline = out
                if iteration["decision"] in ("baseline", "accepted"):
                    best, no_improve = out, 0
                else:
                    no_improve += 1
                    recent_feedback = bounded_recent_feedback(recent_feedback + rejected_feedback_dossiers(
                        out.results, self.feedback, source_job_id=self.lease.job_id,
                        index=iteration["index"], reason=iteration.get("decision_reason")), self.feedback)
            resume = existing[-1] if existing and existing[-1]["status"] == "running" else None
            index = resume["index"] if resume else (existing[-1]["index"] + 1 if existing else 0)
            while True:
                await self._check_cancel()
                if best is not None and resume is None:
                    if self.max_iterations == 0 or index > self.max_iterations:
                        return await self._finalize("baseline_only" if self.max_iterations == 0 else "max_iterations", best, baseline)
                    if no_improve >= self.patience:
                        return await self._finalize("patience_exhausted", best, baseline)
                learn_note = None
                if resume is not None:
                    iteration, resume = resume, None
                    active_iteration_id = iteration["id"]
                    async with self.pool.acquire() as conn:
                        version = await repo.get_agent_version(conn, iteration["agent_version_id"])
                    source = version["content"]
                else:
                    if best is None:
                        source = self.cfg.get("agent_source") or TEMPLATE_PATH.read_text()
                        proposal_json, parent_version = None, None
                    elif index == 1 and self.cfg.get("resume_candidate"):
                        candidate = self.cfg["resume_candidate"]
                        source = candidate["source"]
                        proposal_json = {**(candidate.get("proposal") or {}), "resumed_from": {
                            "job_id": candidate["source_job_id"], "iteration": candidate["source_iteration_index"]}}
                        parent_version = best.agent_version_id
                        await self._event("iteration.resuming_candidate", {"iteration": index,
                            "source_job_id": candidate["source_job_id"], "source_iteration_index": candidate["source_iteration_index"]})
                    else:
                        dossiers = []
                        for task in self.feedback:
                            row = best.results.get(task)
                            if row is not None and row["status"] != "passed":
                                digest = dossier_from_row(dict(row))
                                digest["task_name"] = task
                                dossiers.append(digest)
                        inp = OptimizerInput(index, best.agent_source, scoreboard, dossiers, history,
                            "\n".join(learnings[-10:]) or None,
                            {"agent_model": self.cfg.get("agent_model"), "feedback_tasks": self.feedback,
                             "may_change": "agent prompt, tools, loop; preserve HarnessAgent interface",
                             "objective": "balance verifier reward, model cost and task latency",
                             "runtime_capabilities": {
                                 "shell": "Each environment.exec starts a fresh bash shell at the task working directory; cd, export and shell options do not persist between calls. Files persist within the trial.",
                                 "verification": "Hidden benchmark verifier tests and grading results are unavailable during execution and installed only after the agent finishes. Do not probe /tests or benchmark verifier paths. Self-created checks must use task-visible requirements and inputs.",
                                 "candidate_scope": "General prompt/tools/loop changes only; never embed task-specific answers, paths or expected values from feedback.",
                                 "logs": "Untrusted partial observations; distinguish what happened from an untested causal explanation."},
                             "recent_rejected_feedback_dossiers": bounded_recent_feedback(recent_feedback, self.feedback),
                             "promotion_policy": {"maximum_relative_model_cost_increase": self.opts.get("efficiency_tolerance", .10),
                                 "maximum_relative_p95_duration_increase": self.opts.get("efficiency_tolerance", .10),
                                 "minimum_relative_efficiency_gain": self.opts.get("min_efficiency_improvement", .10),
                                 "minimum_absolute_reward_gain": self.opts.get("min_improvement", 0.),
                                 "quality_regression_allowed": False,
                                 "confirmation_repetitions": self.opts.get("confirmation_repetitions", 2)},
                             "fixed_execution_limits": {"agent_model": self.cfg.get("agent_model", settings.agent_model),
                                 "max_tokens_per_call": min(self.opts.get("max_tokens_per_call", settings.agent_max_output_tokens), settings.agent_max_output_tokens),
                                 "max_inference_body_bytes": settings.max_inference_body_bytes,
                                 "max_model_calls_per_trial": settings.max_model_calls_per_trial,
                                 "agent_timeout_sec": self.opts.get("agent_timeout_sec", 900),
                                 "job_deadline_at": self.deadline_at.isoformat(), "seconds_remaining": max(0, self._seconds_left())},
                             "feedback_metrics": {task: {key: row.get(key) for key in (
                                 "status", "duration_s", "model_cost_usd", "input_tokens", "output_tokens", "n_steps")}
                                 for task, row in best.results.items() if task in self.feedback},
                             "max_agent_steps": self.opts.get("max_agent_steps", 80)})
                        await self._event("iteration.proposing", {"iteration": index, "feedback_failures": len(dossiers)})
                        proposal = await self._propose(inp)
                        source, proposal_json, learn_note = proposal.new_source, proposal.to_json(), proposal.learnings
                        parent_version = best.agent_version_id
                    ok, why = validate_agent_source(source)
                    sha = repo.sha256_text(source)
                    rejection = None if ok else f"invalid_agent:{why}"
                    if best is not None and sha in seen_hashes:
                        rejection = "duplicate_source: already proposed under this frozen experiment"
                    async with self.pool.acquire() as conn:
                        version = await repo.upsert_agent_version(conn, self.job["org_id"], source, parent_version)
                        iteration = await repo.create_iteration(conn, self.lease, index, version["id"],
                            best.iteration_id if best else None, proposal_json)
                        active_iteration_id = iteration["id"]
                        await repo.update_job_progress(conn, self.lease, current_iteration=index)
                    seen_hashes.add(sha)
                    if rejection:
                        async with self.pool.acquire() as conn:
                            await repo.complete_iteration(conn, self.lease, iteration["id"], score=None, n_pass=0,
                                n_fail=0, n_error=0, decision="rejected", decision_reason=rejection,
                                learnings=learn_note, metrics={"complete": False, "evaluation_skipped": True})
                        active_iteration_id = None
                        history.append(HistoryEntry(index, (proposal_json or {}).get("hypothesis"), "rejected", None,
                                                    decision_reason=rejection))
                        for task in self.feedback:
                            scoreboard[task].append(None)
                        if learn_note:
                            learnings.append(hypothesis_context(index, learn_note, "rejected", rejection))
                        if best is None:
                            raise ValueError("invalid baseline: " + rejection)
                        no_improve += 1
                        index += 1
                        continue
                await self._event("iteration.started", {"iteration": index, "agent_sha": repo.sha256_text(source)})
                out = await self._run_iteration(iteration, source, best)
                out.metrics["feedback_summary"] = summarize_feedback(out.results, self.feedback)
                if best is None:
                    if not out.metrics.get("complete"):
                        raise RuntimeError("baseline incomplete; no candidate can be compared against missing evidence")
                    decision, reason = "baseline", "complete baseline on the fixed selection set"
                elif out.screened:
                    decision, reason = "rejected", "screened_out: no feedback repair; unexecuted tasks remain unmeasured"
                else:
                    compared_parent = out.metrics.get("incumbent") or best.metrics
                    accepted, reason = promotion_decision(out.metrics, compared_parent, self.opts)
                    # Every acceptance needs fresh paired confirmation, even if a preliminary result improved.
                    if accepted and not out.metrics.get("confirmation_repetitions"):
                        accepted, reason = False, "confirmation_required"
                    decision = "accepted" if accepted else "rejected"
                async with self.pool.acquire() as conn:
                    await repo.complete_iteration(conn, self.lease, out.iteration_id, score=out.score,
                        n_pass=out.n_pass, n_fail=out.n_fail, n_error=out.n_error, decision=decision,
                        decision_reason=reason, learnings=learn_note, metrics=out.metrics)
                active_iteration_id = None
                prop = iteration.get("proposal") or {}
                history.append(HistoryEntry(index, prop.get("hypothesis"), decision, out.score,
                    decision_reason=reason, change_summary=prop.get("change_summary"),
                    feedback_summary=out.metrics["feedback_summary"]))
                for task in self.feedback:
                    scoreboard[task].append(out.results[task]["status"] == "passed" if task in out.results else None)
                if learn_note:
                    learnings.append(hypothesis_context(index, learn_note, decision, reason))
                if decision == "baseline":
                    baseline = out
                if decision in ("baseline", "accepted"):
                    best, no_improve = out, 0
                else:
                    no_improve += 1
                    recent_feedback = bounded_recent_feedback(recent_feedback + rejected_feedback_dossiers(
                        out.results, self.feedback, source_job_id=self.lease.job_id,
                        index=index, reason=reason), self.feedback)
                index += 1
        except (Cancelled, repo.JobCancelled):
            await self.executor.cancel()
            await self._abandon_iteration(active_iteration_id, "cancelled")
            async with self.pool.acquire() as conn:
                await repo.finish_job(conn, self.lease, "CANCELLED", "cancelled",
                    final_evaluation={"status": "incomplete", "reason": "cancelled", "used_for_selection": False})
        except LeaseLost:
            await self.executor.cancel()
            log.warning("job %s lost ownership; no result promotion", self.lease.job_id)
        except (TimeoutError, repo.BudgetExceeded):
            await self.executor.cancel()
            await self._abandon_iteration(active_iteration_id, "budget_exhausted")
            async with self.pool.acquire() as conn:
                # A stopped search is distinct from a completed holdout comparison.
                await repo.finish_job(conn, self.lease, "SUCCEEDED" if best is not None else "FAILED", "budget_exhausted",
                    best_iteration_id=best.iteration_id if best else None, best_score=best.score if best else None,
                    final_evaluation={"status": "incomplete", "reason": "budget_exhausted", "used_for_selection": False})
        except Exception as exc:
            await self.executor.cancel()
            await self._abandon_iteration(active_iteration_id, "execution_error")
            log.exception("job %s failed", self.lease.job_id)
            async with self.pool.acquire() as conn:
                await repo.finish_job(conn, self.lease, "FAILED", "error", error=str(exc)[:1000],
                    final_evaluation={"status": "incomplete", "reason": "error", "used_for_selection": False})


def validate_agent_source(source: str) -> tuple[bool, str]:
    """Structural compatibility check ONLY; execution isolation is the sandbox's job."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > 128_000:
        return False, "agent source must be text no larger than 128 KB"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error at line {exc.lineno}: {exc.msg}"
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HarnessAgent"]
    if len(classes) != 1:
        return False, "exactly one top-level HarnessAgent class is required"
    agent = classes[0]
    if not any(isinstance(node, ast.AsyncFunctionDef) and node.name == "run" for node in agent.body):
        return False, "HarnessAgent.run must be async"
    if not any((isinstance(base, ast.Name) and base.id == "BaseAgent") or
               (isinstance(base, ast.Attribute) and base.attr == "BaseAgent") for base in agent.bases):
        return False, "HarnessAgent must inherit BaseAgent"
    return True, "structural validation passed; candidate remains untrusted"
