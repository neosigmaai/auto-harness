"""Worker-side execution of job steps (evaluate / improve)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from api.agent_spec import AgentSpec
from api.config import REPO_ROOT, BenchmarkConfig
from api.job_store import (
    EvaluateOutcome,
    ImproveOutcome,
    IterationRecord,
    PostgresJobStore,
    STEP_EVALUATE,
    STEP_IMPROVE,
    StepRecord,
)
from api.schemas import RunError, RunStatus
from api.services.artifacts import ArtifactStore, improver_key, result_key, trace_key
from api.services.improver import (
    EvaluationSummary,
    Improver,
    ImproverError,
    Proposal,
    TaskOutcome,
    build_context,
    create_improver,
    extract_verifier_message,
)
from api.services.runner import (
    HarborBenchmarkRunner,
    MockBenchmarkRunner,
)
from api.services.scoring import mean_reward
from api.store import PostgresRunStore, _utcnow

logger = logging.getLogger("worker")

SPEC_AGENT_IMPORT_PATH = "agent.spec_agent:HarnessAgent"


class StepExecutor:
    """Executes one claimed step and advances the job in the same call."""

    def __init__(
        self,
        job_store: PostgresJobStore,
        run_store: PostgresRunStore,
        *,
        config: BenchmarkConfig,
        improver: Improver,
        artifacts: ArtifactStore,
        step_delay_sec: float = 0.05,
        worker_id: str = "step-executor",
    ) -> None:
        self.job_store = job_store
        self.run_store = run_store
        self.config = config
        # Default/fallback improver: used unconditionally on the mock backend
        # (where tests inject FakeImprover/_RaisingImprover doubles that must
        # keep being used regardless of any per-job improver_model override -
        # a mock backend has no notion of "model" anyway), and as the LLM-path
        # fallback when a step's improver_model matches the config default.
        self.improver = improver
        self.artifacts = artifacts
        self.step_delay_sec = step_delay_sec
        # Every job-owned run this executor creates is inserted already
        # claimed (see PostgresRunStore.create's `claimed_by`) so the legacy
        # /v1/runs queue (`store.claim_next`) can never steal and re-execute
        # it with the wrong agent. Defaults to a fixed id rather than None so
        # this safety property holds even for callers that don't wire a real
        # per-process worker id (e.g. tests).
        self.worker_id = worker_id

    # ----------------------------------------------------------------- #
    # Dispatch
    # ----------------------------------------------------------------- #

    def execute(self, step: StepRecord) -> None:
        if step.type == STEP_EVALUATE:
            self._evaluate(step)
        elif step.type == STEP_IMPROVE:
            self._improve(step)
        else:
            self.job_store.fail_step(
                step.step_id,
                error_code="internal_error",
                error_message=f"unknown step type {step.type!r}",
            )

    # ----------------------------------------------------------------- #
    # Evaluate
    # ----------------------------------------------------------------- #

    def _evaluate(self, step: StepRecord) -> None:
        # F7: a stale-requeued evaluate may still point at an orphaned run that
        # was left `running` with claimed_at=NULL (immune to the legacy sweep).
        if step.run_id:
            self._supersede_run(step.run_id)

        record = self.run_store.create(
            task_ids=list(step.task_ids),
            agent_model=step.spec.agent_model,
            claimed_by=self.worker_id,
            job_id=step.job_id,
        )
        run_id = record.run_id
        self.job_store.set_step_run_id(step.step_id, run_id)
        logger.info(
            "evaluate step_id=%s job_id=%s iteration=%s version=%s run_id=%s tasks=%s",
            step.step_id,
            step.job_id,
            step.iteration,
            step.version,
            run_id,
            step.task_ids,
        )

        try:
            spec_path = self._materialize_spec(run_id, step.spec)
            runner = self._build_runner(spec_path)
            runner.execute_sync(run_id)

            finished = self.run_store.get(run_id)
            if finished is None:
                raise RuntimeError(f"run {run_id} disappeared during evaluation")

            if finished.error is not None:
                outcome = EvaluateOutcome(
                    run_id=run_id,
                    score=None,
                    error_code=finished.error.code,
                    error_message=finished.error.message,
                )
            else:
                copied = self._store_trial_artifacts(
                    step, run_id, [t.task_id for t in finished.tasks]
                )
                self._enrich_remarks_from_results(
                    run_id, step.job_id, step.iteration, [t.task_id for t in finished.tasks]
                )
                finished = self.run_store.get(run_id) or finished
                task_rewards = {t.task_id: t.reward for t in finished.tasks}
                score = mean_reward(task_rewards.values())
                logger.info(
                    "evaluate done run_id=%s score=%.4f artifacts=%s",
                    run_id,
                    score,
                    copied,
                )
                outcome = EvaluateOutcome(
                    run_id=run_id, score=score, task_rewards=task_rewards
                )
        except Exception as exc:  # noqa: BLE001
            # ExecutionUnavailableError is recorded on the run row by runners;
            # any unexpected crash here is an internal_error on the step.
            logger.exception("evaluate failed step_id=%s", step.step_id)
            outcome = EvaluateOutcome(
                run_id=run_id,
                score=None,
                error_code="internal_error",
                error_message=str(exc),
            )

        self.job_store.complete_step_and_advance(step.step_id, outcome)

    def _supersede_run(self, run_id: str) -> None:
        existing = self.run_store.get(run_id)
        if existing is None:
            return
        if existing.status in (RunStatus.completed, RunStatus.failed):
            return
        self.run_store.update(
            run_id,
            status=RunStatus.failed,
            finished_at=_utcnow(),
            error=RunError(
                code="superseded",
                message="Evaluate step was requeued; this run was abandoned",
            ),
        )

    def _run_dir(self, run_id: str) -> Path:
        return REPO_ROOT / "workspace" / "runs" / run_id

    def _materialize_spec(self, run_id: str, spec: AgentSpec) -> Path:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "agent_spec.json"
        path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")
        return path

    def _build_runner(self, spec_path: Path) -> MockBenchmarkRunner | HarborBenchmarkRunner:
        if self.config.execution_backend == "mock":
            return MockBenchmarkRunner(store=self.run_store, step_delay_sec=self.step_delay_sec)
        return HarborBenchmarkRunner(
            self.run_store,
            config=self.config,
            agent_import_path=SPEC_AGENT_IMPORT_PATH,
            extra_env={
                "HARNESS_AGENT_SPEC": str(spec_path),
                "HARNESS_SAVE_TRACE": "1",
            },
        )

    def _store_trial_artifacts(self, step: StepRecord, run_id: str, task_ids: list[str]) -> int:
        """
        Copy harbor trial traces and result.json into the artifact store.

        Harbor writes <jobs_dir>/<job>/<task_id>__<trial>/agent/trace.json and
        <jobs_dir>/<job>/<task_id>__<trial>/result.json; the mock backend writes
        nothing, in which case this is a no-op.
        """
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir():
            return 0
        known = set(task_ids)
        copied = 0
        for trace_path in sorted(run_dir.rglob("trace.json")):
            if trace_path.parent.name != "agent":
                continue
            trial_dir = trace_path.parent.parent
            task_id = trial_dir.name.rsplit("__", 1)[0]
            if task_id not in known:
                continue
            try:
                self.artifacts.put(trace_key(step.job_id, step.iteration, task_id), trace_path)
                copied += 1
            except Exception:  # noqa: BLE001
                logger.warning("failed to store trace for task_id=%s run_id=%s", task_id, run_id)
            result_path = trial_dir / "result.json"
            if result_path.is_file():
                try:
                    self.artifacts.put(
                        result_key(step.job_id, step.iteration, task_id), result_path
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "failed to store result.json for task_id=%s run_id=%s", task_id, run_id
                    )
        return copied

    # Back-compat alias used by older tests.
    def _store_traces(self, step: StepRecord, run_id: str, task_ids: list[str]) -> int:
        return self._store_trial_artifacts(step, run_id, task_ids)

    def _enrich_remarks_from_results(
        self,
        run_id: str,
        job_id: str,
        iteration: int,
        task_ids: list[str],
    ) -> None:
        """Replace generic verifier remarks with Harbor result.json diagnostics."""
        record = self.run_store.get(run_id)
        if record is None:
            return
        by_id = {t.task_id: t for t in record.tasks}
        for task_id in task_ids:
            task = by_id.get(task_id)
            if task is None:
                continue
            key = result_key(job_id, iteration, task_id)
            try:
                if not self.artifacts.exists(key):
                    continue
                data = json.loads(self.artifacts.get(key).decode("utf-8"))
            except Exception:  # noqa: BLE001
                logger.warning("could not read result artifact %s", key)
                continue
            message = extract_verifier_message(data)
            if not message:
                continue
            # Prefer Harbor diagnostics over the generic reward_to_task_status string.
            if task.remarks in (None, "Verifier failed") or task.remarks.startswith(
                "Partial reward "
            ):
                self.run_store.set_task(
                    run_id,
                    task_id,
                    status=task.status,
                    reward=task.reward,
                    remarks=message,
                )

    # ----------------------------------------------------------------- #
    # Improve
    # ----------------------------------------------------------------- #

    def _improve(self, step: StepRecord) -> None:
        # Everything below is guarded the same way _evaluate() is guarded: any
        # unexpected failure — before propose() (looking up the job or the
        # evaluation run), after it (persisting improver artifacts, advancing
        # the step) — must be reported through complete_step_and_advance()
        # rather than escape to process_one()'s fail_step fallback, which
        # fails the job unconditionally and would discard a good
        # best_agent_version_id. ImproverError keeps its own error code;
        # everything else maps to "internal_error", matching _evaluate().
        evaluation: EvaluationSummary | None = None
        history: list[IterationRecord] = []
        improver: Improver = self.improver
        try:
            job = self.job_store.get_job(step.job_id)
            if job is None:
                self.job_store.fail_step(
                    step.step_id,
                    error_code="internal_error",
                    error_message=f"job {step.job_id} disappeared",
                )
                return

            latest = self._latest_evaluation(job.iterations)
            if latest is None:
                self.job_store.complete_step_and_advance(
                    step.step_id,
                    ImproveOutcome(
                        spec=None,
                        error_code="improver_failed",
                        error_message="no completed evaluation to improve on",
                    ),
                )
                return

            record = self.run_store.get(latest.run_id or "")
            if record is None:
                self.job_store.complete_step_and_advance(
                    step.step_id,
                    ImproveOutcome(
                        spec=None,
                        error_code="improver_failed",
                        error_message=f"evaluation run {latest.run_id} not found",
                    ),
                )
                return

            evaluation = EvaluationSummary(
                score=float(latest.score or 0.0),
                tasks=[
                    TaskOutcome(
                        task_id=t.task_id,
                        status=t.status.value,
                        reward=t.reward,
                        remarks=t.remarks,
                    )
                    for t in record.tasks
                ],
                traces=self._read_traces(step.job_id, latest.iteration, [t.task_id for t in record.tasks]),
                # Movement is already derived by get_job() from the stored task_rewards
                # snapshots; recomputing it here would duplicate that logic.
                fixed_tasks=latest.fixed_tasks,
                regressed_tasks=latest.regressed_tasks,
            )
            history = list(job.iterations)

            logger.info(
                "improve step_id=%s job_id=%s iteration=%s from_score=%.4f traces=%s",
                step.step_id,
                step.job_id,
                step.iteration,
                evaluation.score,
                len(evaluation.traces),
            )

            improver = self._improver_for_step(step)
            proposal = improver.propose(
                spec=step.spec,
                evaluation=evaluation,
                history=history,
            )

            self._persist_improver_io(step, evaluation, history, improver=improver, proposal=proposal)
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(spec=proposal.spec, rationale=proposal.rationale),
            )
        except ImproverError as exc:
            logger.error("improver failed step_id=%s: %s", step.step_id, exc)
            if evaluation is not None:
                self._safe_persist_improver_io(step, evaluation, history, improver=improver, error=str(exc))
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="improver_failed",
                    error_message=str(exc),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("improve step crashed step_id=%s", step.step_id)
            if evaluation is not None:
                self._safe_persist_improver_io(step, evaluation, history, improver=improver, error=str(exc))
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="internal_error",
                    error_message=str(exc),
                ),
            )

    def _improver_for_step(self, step: StepRecord) -> Improver:
        """
        Resolve the improver to use for one improve step.

        ``self.improver`` is the executor-wide default: on the mock backend
        it (and only it) is used, unconditionally, so tests can inject
        FakeImprover/_RaisingImprover doubles that keep being used regardless
        of any per-job improver_model — a mock backend has no real notion of
        "model" to honour. On a real backend, a step whose job set a
        non-default ``improver_model`` (see CreateJobRequest.improver_model /
        spec §10-11) must actually use that model, so build a fresh improver
        for it via the same seam create_improver already exposes for this
        purpose; a step with no override (or one that matches the config
        default) keeps using the executor-wide default improver.
        """
        if self.config.execution_backend == "mock":
            return self.improver
        if step.improver_model and step.improver_model != self.config.improver_model:
            return create_improver(self.config, improver_model=step.improver_model)
        return self.improver

    @staticmethod
    def _latest_evaluation(iterations: list[IterationRecord]) -> IterationRecord | None:
        completed = [
            it
            for it in iterations
            if it.status == RunStatus.completed.value
            and it.run_id is not None
            and it.score is not None
        ]
        return completed[-1] if completed else None

    def _read_traces(self, job_id: str, iteration: int, task_ids: list[str]) -> dict[str, str]:
        traces: dict[str, str] = {}
        for task_id in task_ids:
            key = trace_key(job_id, iteration, task_id)
            try:
                if self.artifacts.exists(key):
                    traces[task_id] = self.artifacts.get(key).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                logger.warning("could not read trace artifact %s", key)
        return traces

    def _safe_persist_improver_io(
        self,
        step: StepRecord,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
        *,
        improver: Improver,
        error: str,
    ) -> None:
        """Best-effort audit logging for an already-failed improve step.

        Used only from _improve()'s except blocks, where the job outcome has
        already been decided (failure) - a problem persisting the prompt/error
        artifact here (e.g. build_context() or json.dumps() raising, which
        _persist_improver_io()'s own try/except does not cover) must never
        prevent complete_step_and_advance() from reporting that outcome.
        """
        try:
            self._persist_improver_io(step, evaluation, history, improver=improver, error=error)
        except Exception:  # noqa: BLE001
            logger.warning(
                "failed to persist improver failure artifacts job_id=%s iteration=%s",
                step.job_id,
                step.iteration,
            )

    def _persist_improver_io(
        self,
        step: StepRecord,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
        *,
        improver: Improver,
        proposal: Proposal | None = None,
        error: str | None = None,
    ) -> None:
        prompt = getattr(improver, "last_prompt", "") or build_context(
            spec=step.spec,
            evaluation=evaluation,
            history=history,
            budget=self.config.improver_context_budget,
        )
        if proposal is not None:
            body = json.dumps(
                {"rationale": proposal.rationale, "spec": proposal.spec.model_dump()},
                indent=2,
            )
        else:
            body = json.dumps(
                {
                    "error": error or "unknown improver error",
                    "raw_response": getattr(improver, "last_response", ""),
                },
                indent=2,
            )
        try:
            self.artifacts.put(improver_key(step.job_id, step.iteration, "prompt.txt"), prompt)
            self.artifacts.put(improver_key(step.job_id, step.iteration, "response.json"), body)
        except Exception:  # noqa: BLE001
            logger.warning(
                "failed to persist improver artifacts job_id=%s iteration=%s",
                step.job_id,
                step.iteration,
            )
