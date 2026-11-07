"""The per-job state machine. One instance per claimed job. Every phase is checkpointed; a
worker that takes over after a crash calls run() again and resumes where the last one stopped."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import asyncpg

from .. import repo
from ..config import settings
from ..repo import Lease, LeaseLost
from .digest import build_digest
from .executors.base import Executor, RunContext, TaskResult
from .optimizer import HistoryEntry, Optimizer, OptimizerInput

log = logging.getLogger("worker.job")

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "agent" / "templates" / "terminal_bench.py"


class Cancelled(Exception):
    pass


@dataclass
class IterOutcome:
    iteration_id: UUID
    index: int
    agent_version_id: UUID
    agent_source: str
    results: dict[str, repo.asyncpg.Record]   # task -> task_runs row
    score: float | None
    n_pass: int
    n_fail: int
    n_error: int


class JobRunner:
    def __init__(self, pool: asyncpg.Pool, job: asyncpg.Record, lease: Lease, executor: Executor, optimizer: Optimizer):
        self.pool, self.job, self.lease, self.executor, self.optimizer = pool, job, lease, executor, optimizer
        self.cfg = job["config"]
        self.tasks: list[str] = list(self.cfg["tasks"])
        self.max_iterations: int = int(self.cfg.get("max_iterations", 5))
        self.patience: int = int(self.cfg.get("patience", 3))
        self.opts = self.cfg.get("options") or {}
        self.seed = int(self.cfg.get("seed") or 1234)
        self.deadline = time.monotonic() + 60 * int((self.cfg.get("budget") or {}).get("max_wall_clock_min", 120))
        self.cancel_flag = asyncio.Event()
        self.work_root = Path(settings.runs_dir) / str(job["id"])

    # ── helpers ───────────────────────────────────────────────────────────────
    async def _check_cancel(self):
        if self.cancel_flag.is_set():
            raise Cancelled()
        if time.monotonic() > self.deadline:
            raise TimeoutError("wall-clock budget exhausted")

    async def _event(self, kind: str, payload: dict | None = None):
        async with self.pool.acquire() as c:
            await repo.add_event(c, self.lease.job_id, kind, payload)

    async def _set_phase(self, iteration_id: UUID, phase: str):
        async with self.pool.acquire() as c:
            await repo.set_phase(c, self.lease, iteration_id, phase)

    @staticmethod
    def _score(rows: dict) -> tuple[float | None, int, int, int]:
        n_pass = sum(1 for r in rows.values() if r["status"] == "passed")
        n_err = sum(1 for r in rows.values() if r["status"] == "error")
        n_fail = len(rows) - n_pass - n_err
        total = len(rows)
        return (n_pass / total if total else None), n_pass, n_fail, n_err

    # ── benchmark phase (resumable, incremental) ──────────────────────────────
    async def _benchmark(self, iteration_id: UUID, index: int, agent_source: str, tasks: list[str], attempt: int = 1) -> dict[str, TaskResult]:
        async with self.pool.acquire() as c:
            await repo.ensure_task_runs(c, iteration_id, tasks)
            pending = [t for t in await repo.pending_task_names(c, iteration_id) if t in tasks]
        if not pending:
            return {}
        ctx = RunContext(job_id=self.lease.job_id, iteration_index=index, iteration_id=iteration_id,
                         org_id=self.job["org_id"], agent_model=self.cfg.get("agent_model", settings.agent_model),
                         seed=self.seed, attempt=attempt, work_dir=self.work_root / f"iter{index}")
        collected: dict[str, TaskResult] = {}

        async def on_start(task: str):
            async with self.pool.acquire() as c:
                await repo.mark_task_running(c, iteration_id, task)

        async def on_result(r: TaskResult):
            d = build_digest(r, r.instruction)
            collected[r.task_name] = r
            async with self.pool.acquire() as c:
                await repo.record_task_result(
                    c, self.lease, iteration_id, r.task_name, status=r.status, reward=r.reward,
                    failure_class=d["stats"]["failure_class"], duration_s=r.duration_s, n_steps=r.n_steps,
                    exit_reason=r.exit_reason, test_stdout=r.test_stdout, trace=r.trace, digest=d, sandbox_id=r.sandbox_id)
                await repo.add_event(c, self.lease.job_id, "task.finished",
                                     {"iteration": index, "task": r.task_name, "status": r.status, "attempt": attempt})

        await self.executor.run_benchmark(ctx, agent_source, pending, on_result, on_start)
        await self._check_cancel()
        return collected

    async def _confirm_flips(self, iteration_id: UUID, index: int, agent_source: str, flipped: list[str],
                             parent_results: dict) -> None:
        """Re-run flipped tasks once; a flip must reproduce or we revert that task to the parent's outcome."""
        if not flipped:
            return
        await self._event("iteration.confirm_flips", {"iteration": index, "tasks": flipped})
        async with self.pool.acquire() as c:
            first = await repo.task_results(c, iteration_id)
            await c.execute("UPDATE task_runs SET status='pending' WHERE iteration_id=$1 AND task_name = ANY($2::text[])",
                            iteration_id, flipped)
        second = await self._benchmark(iteration_id, index, agent_source, flipped, attempt=2)
        async with self.pool.acquire() as c:
            for t in flipped:
                a, b = first[t], second.get(t)
                if b is None:
                    continue
                agree = (a["status"] == "passed") == b.passed
                note = {"confirmation": {"first": a["status"], "second": b.status, "reproduced": agree}}
                if not agree:
                    # Conservative: a flip that does not reproduce is treated as noise -> revert this task to
                    # the parent's outcome (so neither a lucky pass nor an unlucky fail drives the decision).
                    pr = parent_results[t]
                    note["confirmation"]["resolved_as"] = f"parent:{pr['status']}"
                    await c.execute(
                        """UPDATE task_runs SET status=$3, reward=$4, digest = coalesce(digest,'{}'::jsonb) || $5::jsonb
                           WHERE iteration_id=$1 AND task_name=$2""",
                        iteration_id, t, pr["status"], pr["reward"], note)
                else:
                    await c.execute("UPDATE task_runs SET digest = coalesce(digest,'{}'::jsonb) || $3::jsonb WHERE iteration_id=$1 AND task_name=$2",
                                    iteration_id, t, note)

    # ── one iteration end-to-end (idempotent on resume) ───────────────────────
    async def _run_iteration(self, it: asyncpg.Record, agent_source: str, parent: IterOutcome | None) -> IterOutcome:
        iteration_id, index = it["id"], it["index"]
        phase = it["phase"]
        await self._check_cancel()

        if phase in ("PROVISION",):
            self.work_root.mkdir(parents=True, exist_ok=True)
            (self.work_root / f"iter{index}").mkdir(parents=True, exist_ok=True)
            await self._set_phase(iteration_id, "BENCHMARK"); phase = "BENCHMARK"

        if phase == "BENCHMARK":
            two_phase = bool(self.opts.get("two_phase_eval", True)) and parent is not None
            if two_phase:
                failing = [t for t, r in parent.results.items() if r["status"] != "passed"]
                passing = [t for t in self.tasks if t not in failing]
                first = await self._benchmark(iteration_id, index, agent_source, failing)
                async with self.pool.acquire() as c:
                    rows = await repo.task_results(c, iteration_id)
                flipped_up = [t for t in failing if rows.get(t) and rows[t]["status"] == "passed"]
                if not flipped_up and passing:
                    # Screened out: nothing improved, do not spend on the passing tasks.
                    await self._event("iteration.screened_out", {"iteration": index, "screened": len(failing)})
                    async with self.pool.acquire() as c:
                        await c.execute("DELETE FROM task_runs WHERE iteration_id=$1 AND status='pending'", iteration_id)
                        await repo.set_phase(c, self.lease, iteration_id, "DECIDE")
                    phase = "DECIDE"
                else:
                    await self._benchmark(iteration_id, index, agent_source, passing)
            else:
                await self._benchmark(iteration_id, index, agent_source, self.tasks)
            if phase == "BENCHMARK":
                await self._set_phase(iteration_id, "COLLECT"); phase = "COLLECT"

        if phase == "COLLECT":
            if parent is not None and bool(self.opts.get("confirm_flips", True)):
                async with self.pool.acquire() as c:
                    rows = await repo.task_results(c, iteration_id)
                flipped = [t for t, r in rows.items() if t in parent.results and (r["status"] == "passed") != (parent.results[t]["status"] == "passed")]
                await self._confirm_flips(iteration_id, index, agent_source, flipped, parent.results)
            await self._set_phase(iteration_id, "DECIDE"); phase = "DECIDE"

        async with self.pool.acquire() as c:
            rows = await repo.task_results(c, iteration_id)
        if parent is not None and len(rows) < len(self.tasks):
            # screened iteration: fill in parent's outcomes for tasks we did not run, for scoring only
            merged = dict(rows)
            for t in self.tasks:
                if t not in merged:
                    merged[t] = parent.results[t]
            score, n_pass, n_fail, n_err = self._score(merged)
            results = merged
        else:
            score, n_pass, n_fail, n_err = self._score(rows)
            results = rows
        return IterOutcome(iteration_id, index, it["agent_version_id"], agent_source, results, score, n_pass, n_fail, n_err)

    # ── main loop ─────────────────────────────────────────────────────────────
    async def run(self) -> None:
        job_id = self.lease.job_id
        try:
            async with self.pool.acquire() as c:
                existing = await repo.list_iterations(c, job_id)
            best: IterOutcome | None = None
            history: list[HistoryEntry] = []
            scoreboard: dict[str, list[bool | None]] = {t: [] for t in self.tasks}
            no_improve = 0
            learnings: list[str] = []

            # Replay completed iterations (resume path) to rebuild in-memory state.
            for it in existing:
                if it["status"] != "completed":
                    continue
                async with self.pool.acquire() as c:
                    rows = await repo.task_results(c, it["id"])
                    av = await repo.get_agent_version(c, it["agent_version_id"])
                results = dict(rows)
                if best is not None:
                    for t in self.tasks:
                        results.setdefault(t, best.results[t])
                out = IterOutcome(it["id"], it["index"], it["agent_version_id"], av["content"], results, it["score"],
                                  it["n_pass"] or 0, it["n_fail"] or 0, it["n_error"] or 0)
                for t in self.tasks:
                    scoreboard[t].append(results[t]["status"] == "passed" if t in results else None)
                hyp = (it["proposal"] or {}).get("hypothesis")
                history.append(HistoryEntry(it["index"], hyp, it["decision"], it["score"]))
                if it["learnings"]:
                    learnings.append(it["learnings"])
                if it["decision"] in ("baseline", "accepted"):
                    best = out; no_improve = 0
                else:
                    no_improve += 1

            index = existing[-1]["index"] + 1 if existing else 0
            resume_it = existing[-1] if existing and existing[-1]["status"] == "running" else None
            if resume_it is not None:
                index = resume_it["index"]
                await self._event("job.resumed", {"iteration": index, "phase": resume_it["phase"], "epoch": self.lease.epoch})

            while True:
                await self._check_cancel()
                # ── stop checks (before starting a new iteration) ──
                if best is not None and resume_it is None:
                    if index > self.max_iterations:
                        return await self._finish("max_iterations", best)
                    if best.score is not None and best.score >= 1.0:
                        return await self._finish("perfect_score", best)
                    if no_improve >= self.patience:
                        return await self._finish("patience_exhausted", best)

                # ── provision agent version + iteration row (or pick up the running one) ──
                learn_note = None
                if resume_it is not None:
                    it = resume_it; resume_it = None
                    async with self.pool.acquire() as c:
                        av = await repo.get_agent_version(c, it["agent_version_id"])
                    agent_source = av["content"]
                    proposal_json = it["proposal"]
                else:
                    if best is None:
                        agent_source = self.cfg.get("agent_source") or TEMPLATE_PATH.read_text()
                        proposal_json, learn_note = None, None
                        parent_version_id = None
                    else:
                        dossiers = []
                        for t, r in best.results.items():
                            if r["status"] != "passed":
                                d = dict(r["digest"] or {}); d["task_name"] = t; dossiers.append(d)
                        inp = OptimizerInput(index, best.agent_source, scoreboard, dossiers, history,
                                             "\n".join(learnings[-10:]) or None,
                                             {"agent_model": self.cfg.get("agent_model"), "may_change": "whole file except model/timeouts"})
                        await self._event("iteration.proposing", {"iteration": index, "n_failing": len(dossiers)})
                        proposal = await self.optimizer.propose(inp)
                        agent_source, proposal_json, learn_note = proposal.new_source, proposal.to_json(), proposal.learnings
                        parent_version_id = best.agent_version_id
                        ok, why = validate_agent_source(agent_source)
                        if not ok:
                            async with self.pool.acquire() as c:
                                av = await repo.upsert_agent_version(c, self.job["org_id"], agent_source, parent_version_id)
                                it = await repo.create_iteration(c, self.lease, index, av["id"], best.iteration_id, proposal_json)
                                await repo.complete_iteration(c, self.lease, it["id"], score=None, n_pass=0, n_fail=0, n_error=0,
                                                              decision="rejected", decision_reason=f"invalid_agent: {why}", learnings=learn_note)
                            history.append(HistoryEntry(index, proposal.hypothesis, "rejected", None))
                            no_improve += 1; index += 1
                            continue
                    async with self.pool.acquire() as c:
                        av = await repo.upsert_agent_version(c, self.job["org_id"], agent_source, parent_version_id)
                        it = await repo.create_iteration(c, self.lease, index, av["id"], best.iteration_id if best else None, proposal_json)
                        await repo.update_job_progress(c, self.lease, current_iteration=index)
                await self._event("iteration.started", {"iteration": index, "agent_sha": repo.sha256_text(agent_source)[:12]})

                # ── run it ──
                out = await self._run_iteration(it, agent_source, best)

                # ── decide ──
                min_imp = float(self.opts.get("min_improvement", 0.0))
                if best is None:
                    decision, reason = "baseline", "first run"
                else:
                    prev = best.score or 0.0
                    cur = out.score or 0.0
                    flips_up = [t for t in self.tasks if out.results[t]["status"] == "passed" and best.results[t]["status"] != "passed"]
                    flips_down = [t for t in self.tasks if out.results[t]["status"] != "passed" and best.results[t]["status"] == "passed"]
                    if cur > prev + min_imp:
                        decision, reason = "accepted", f"score {cur:.3f} > best {prev:.3f} (+{flips_up} -{flips_down})"
                    elif cur == prev and len(flips_up) + len(flips_down) == 0 and sum((r['n_steps'] or 0) for r in out.results.values()) < sum((r['n_steps'] or 0) for r in best.results.values()):
                        decision, reason = "accepted", f"same score, fewer total steps"
                    else:
                        decision, reason = "rejected", f"score {cur:.3f} <= best {prev:.3f} (+{flips_up} -{flips_down})"
                async with self.pool.acquire() as c:
                    await repo.complete_iteration(c, self.lease, out.iteration_id, score=out.score, n_pass=out.n_pass,
                                                  n_fail=out.n_fail, n_error=out.n_error, decision=decision, decision_reason=reason,
                                                  learnings=learn_note)
                    fields = {}
                    if decision == "baseline":
                        fields.update(baseline_score=out.score, best_score=out.score, best_iteration_id=out.iteration_id)
                    elif decision == "accepted":
                        fields.update(best_score=out.score, best_iteration_id=out.iteration_id)
                    if fields:
                        await repo.update_job_progress(c, self.lease, **fields)
                    await repo.add_event(c, job_id, "iteration.finished", {"iteration": index, "decision": decision, "score": out.score, "reason": reason})
                for t in self.tasks:
                    scoreboard[t].append(out.results[t]["status"] == "passed" if t in out.results else None)
                history.append(HistoryEntry(index, (it["proposal"] or {}).get("hypothesis"), decision, out.score))
                if learn_note:
                    learnings.append(learn_note)
                if decision in ("baseline", "accepted"):
                    best = out; no_improve = 0 if decision == "accepted" else no_improve
                else:
                    no_improve += 1
                index += 1
                if self.max_iterations == 0:
                    return await self._finish("baseline_only", best)

        except Cancelled:
            await self.executor.cancel()
            await self._mark_running_iteration("errored", "cancelled")
            async with self.pool.acquire() as c:
                await repo.finish_job(c, self.lease, "CANCELLED", "cancelled")
        except LeaseLost as e:
            log.warning("job %s: %s — stopping without writing", job_id, e)
            await self.executor.cancel()
        except TimeoutError as e:
            await self.executor.cancel()
            await self._mark_running_iteration("errored", str(e))
            async with self.pool.acquire() as c:
                await repo.finish_job(c, self.lease, "SUCCEEDED", "budget_exhausted")
        except Exception as e:  # infra failure: record and fail the job (retry policy lives in the worker)
            log.exception("job %s failed", job_id)
            await self.executor.cancel()
            await self._mark_running_iteration("errored", repr(e))
            async with self.pool.acquire() as c:
                await repo.finish_job(c, self.lease, "FAILED", "error", error=repr(e)[:2000])

    async def _mark_running_iteration(self, status: str, error: str):
        async with self.pool.acquire() as c:
            it = await repo.latest_iteration(c, self.lease.job_id)
            if it and it["status"] == "running":
                await c.execute("UPDATE iterations SET status=$2, error=$3, finished_at=now(), decision='errored' WHERE id=$1",
                                it["id"], status, error[:2000])

    async def _finish(self, reason: str, best: IterOutcome | None):
        async with self.pool.acquire() as c:
            await repo.finish_job(c, self.lease, "SUCCEEDED", reason,
                                  best_iteration_id=best.iteration_id if best else None,
                                  best_score=best.score if best else None)


# ── static policy gate for LLM-produced agent code ─────────────────────────────
import ast

ALLOWED_IMPORTS = {"json", "os", "re", "asyncio", "time", "math", "typing", "dataclasses", "collections", "textwrap",
                   "litellm", "harbor", "pathlib", "logging", "functools", "itertools", "string", "random"}
FORBIDDEN_NAMES = {"subprocess", "socket", "shutil", "sys", "ctypes", "importlib", "pickle", "marshal", "eval", "exec",
                   "compile", "__import__", "requests", "httpx", "urllib"}


def validate_agent_source(src: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    has_agent = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HarnessAgent":
            has_agent = True
            if not any(isinstance(b, ast.AsyncFunctionDef) and b.name == "run" for b in node.body):
                return False, "HarnessAgent.run must be an async def"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                root = n.split(".")[0]
                if root in FORBIDDEN_NAMES:
                    return False, f"forbidden import: {n}"
                if root not in ALLOWED_IMPORTS:
                    return False, f"import not in allowlist: {n}"
        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile", "__import__"}:
            return False, f"forbidden call: {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            # writes only allowed under logs_dir; a plain open('w') elsewhere is rejected
            for a in node.args[1:] + [k.value for k in node.keywords if k.arg == "mode"]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and "w" in a.value:
                    first = node.args[0] if node.args else None
                    if not (isinstance(first, ast.Name) and first.id == "trace_path"):
                        return False, "file writes outside logs_dir are not allowed"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            if isinstance(node.slice, ast.Constant) and node.slice.value not in ("AGENT_MODEL", "AGENT_REASONING_EFFORT", "HARNESS_SAVE_TRACE"):
                return False, f"os.environ access to {node.slice.value!r} not allowed"
    if not has_agent:
        return False, "no HarnessAgent class"
    return True, "ok"
