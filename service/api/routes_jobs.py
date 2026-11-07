from __future__ import annotations

import difflib
import math
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response

from ..auth import Principal, get_principal
from ..config import settings
from ..db import get_pool
from ..errors import ApiError
from ..sandbox.prepared_images import freeze_harness
from ..worker.digest import dossier_from_row
from ..models import (AgentOut, EventOut, IterationDetail, IterationSummary, JobAccepted, JobCreate, JobList,
                      JobOut, ResultSummary, TaskRunDetail, TaskRunOut)
from .. import repo

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_out(row, progress=None, best_index=None) -> JobOut:
    return JobOut(
        job_id=row["id"], status=row["status"], created_by=row["created_by"], config=row["config"],
        current_iteration=row["current_iteration"], baseline_score=row["baseline_score"], best_score=row["best_score"],
        best_iteration_index=best_index, stop_reason=row["stop_reason"], error=row["error"],
        cancel_requested=row["cancel_requested"], progress=progress, created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
        deadline_at=row.get('deadline_at'),final_evaluation=row.get('final_evaluation'),
        spent_usd=float(row.get('spent_usd',0)),reserved_usd=float(row.get('reserved_usd',0)),
    )


def _summary(it) -> ResultSummary | None:
    if it["n_pass"] is None:
        return None
    n = (it["n_pass"] or 0) + (it["n_fail"] or 0) + (it["n_error"] or 0)
    return ResultSummary(n_tasks=n, passed=it["n_pass"] or 0, failed=it["n_fail"] or 0, errored=it["n_error"] or 0, score=it["score"])


def _iter_summary(it) -> IterationSummary:
    prop = it["proposal"] or {}
    return IterationSummary(
        index=it["index"], phase=it["phase"], status=it["status"], decision=it["decision"],
        decision_reason=it["decision_reason"], score=it["score"], summary=_summary(it),
        hypothesis=prop.get("hypothesis"), agent_version_id=it["agent_version_id"], agent_sha256=it["agent_sha256"],
        started_at=it["started_at"], finished_at=it["finished_at"],
        metrics=it.get('metrics') or {},
    )


async def _owned_job(conn, p: Principal, job_id: UUID):
    job = await repo.get_job(conn, p, job_id)
    if job is None:
        raise ApiError(404, "Not found", "Job not found.")  # same answer for 'other tenant' and 'does not exist'
    return job


async def _freeze_prior_experiment(conn, p: Principal, config: dict, prior_job_id: UUID,
                                   resume_candidate_index: int | None) -> None:
    """Copy bounded diagnostics/source as data, never old scores or task results."""
    prior = await _owned_job(conn, p, prior_job_id)
    if prior['status'] not in {'SUCCEEDED', 'FAILED', 'CANCELLED'}:
        raise ApiError(409, 'Prior job active', 'The prior experiment must be terminal before taking a snapshot.')
    prior_config = prior['config']
    for field in ('benchmark', 'tasks', 'evaluation_split', 'agent_model'):
        if prior_config.get(field) != config.get(field):
            raise ApiError(422, 'Incompatible experiment', f'Prior job must use the same {field}.')
    if prior_config.get('sandbox_harness', {'mode': 'cold-v1'}) != config.get('sandbox_harness', {'mode': 'cold-v1'}):
        raise ApiError(422, 'Incompatible experiment', 'Prior job must use the same frozen sandbox preparation.')
    revisions = await conn.fetch("""SELECT DISTINCT t.metadata->>'benchmark_revision' AS revision
        FROM task_attempts t JOIN iterations i ON i.id=t.iteration_id
        WHERE i.job_id=$1 AND t.metadata->>'benchmark_revision' IS NOT NULL""", prior_job_id)
    known_revisions = {row['revision'] for row in revisions}
    if prior_config.get('benchmark_revision'):
        known_revisions.add(prior_config['benchmark_revision'])
    if known_revisions != {config['benchmark_revision']}:
        raise ApiError(422, 'Incompatible experiment', 'Prior benchmark revision is unknown or differs from this deployment.')
    baseline = await repo.get_iteration(conn, prior_job_id, 0)
    initial_source = config.get('agent_source') or (
        Path(__file__).resolve().parents[2] / 'agent/templates/terminal_bench.py').read_text()
    if baseline is None or baseline['agent_source'] != initial_source:
        raise ApiError(422, 'Incompatible experiment', 'Prior job must have the same original baseline source.')
    # Pin the exact bytes now; a later template edit cannot change this new job.
    config['agent_source'] = initial_source
    feedback = (config.get('evaluation_split') or {}).get('feedback', [])
    iterations = await repo.list_iterations(conn, prior_job_id)
    rejected = [it for it in iterations if it['status'] == 'completed' and it['decision'] == 'rejected'][-12:]
    seed = []
    for it in rejected:
        proposal = it['proposal'] or {}
        rows = await conn.fetch('''SELECT task_name,status,reward,duration_s,model_cost_usd,
            input_tokens,output_tokens,cost_known FROM task_runs
            WHERE iteration_id=$1 AND task_name=ANY($2::text[]) ORDER BY task_name''', it['id'], feedback)
        seed.append({'index': it['index'], 'source_job_id': str(prior_job_id), 'decision': 'rejected',
            'agent_sha256': it['agent_sha256'],
            'hypothesis': str(proposal.get('hypothesis') or '')[:3000], 'score': None,
            'decision_reason': (it['decision_reason'] or '')[:3000],
            'change_summary': str(proposal.get('change_summary') or '')[:3000],
            'feedback_summary': {'planned_tasks': len(feedback), 'observed_tasks': len(rows),
                                 'tasks': [dict(row) for row in rows]}})
    config['optimizer_history_seed'] = seed
    recent_feedback = []
    for it in rejected[-2:]:
        # Preserve the same first non-error logical observation used by evaluation;
        # never cherry-pick a later retry, confirmation, or held-back attempt.
        attempts = await repo.attempt_results(conn, it['id'], evaluation_phase='main', repetition=1)
        failures = []
        for task in feedback:
            row = attempts.get(task)
            if row is None or row['status'] != 'failed':
                continue
            reward = row['reward']
            if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
                continue
            failures.append(row)
        for row in failures[:3]:
            recent_feedback.append({'source_job_id': str(prior_job_id), 'iteration_index': it['index'],
                'decision': 'rejected', 'decision_reason': (it['decision_reason'] or '')[:3000],
                'task_name': row['task_name'], 'dossier': dossier_from_row(dict(row))})
    config['optimizer_recent_feedback_seed'] = recent_feedback
    if resume_candidate_index is None:
        return
    candidate = await repo.get_iteration(conn, prior_job_id, resume_candidate_index)
    if candidate is None:
        raise ApiError(404, 'Not found', 'Prior candidate not found.')
    if (candidate['index'] < 1 or candidate['decision'] in {'baseline', 'accepted'}
            or candidate['status'] == 'completed' or candidate['agent_source'] == initial_source):
        raise ApiError(422, 'Invalid candidate', 'Only a distinct, interrupted, unconfirmed candidate may be resumed.')
    proposal = candidate['proposal'] or {}
    safe_proposal = {key: proposal[key] for key in ('objective', 'hypothesis', 'targets', 'rationale', 'change_summary')
                     if key in proposal}
    targets = safe_proposal.get('targets', [])
    if not isinstance(targets, list) or any(task not in feedback for task in targets):
        raise ApiError(422, 'Invalid candidate', 'Candidate targets must belong to the frozen feedback set.')
    config['resume_candidate'] = {'source': candidate['agent_source'], 'proposal': safe_proposal,
        'source_job_id': str(prior_job_id), 'source_iteration_index': candidate['index']}


@router.post("", response_model=JobAccepted, status_code=202, summary="Submit an optimisation job (async)")
async def submit_job(body: JobCreate, response: Response, p: Principal = Depends(get_principal),
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    pool = await get_pool()
    async with pool.acquire() as conn:
        org = await repo.get_org(conn, p.org_id)
        if body.max_iterations > org["max_iterations"]:
            raise ApiError(422, "Quota exceeded", f"max_iterations may not exceed {org['max_iterations']} for this organisation.")
        config = body.model_dump(mode='json')
        config["tasks"] = body.resolved_tasks()
        config["agent_model"] = body.agent_model or settings.agent_model
        config["optimizer_model"] = body.optimizer_model or settings.optimizer_model
        config['benchmark_revision'] = settings.benchmark_revision
        try:
            config['sandbox_harness'] = freeze_harness(
                config['tasks'], config['benchmark_revision'], settings.prepared_images_manifest)
        except (ValueError, OSError):
            raise ApiError(422, 'Invalid sandbox preparation',
                           'Configured prepared-image manifest is unavailable or incompatible.') from None
        for field in ('agent_model', 'optimizer_model'):
            if config[field] not in settings.model_prices:
                raise ApiError(422, 'Unsupported model', f'{field} must have configured token prices.')
        if body.prior_job_id is not None:
            await _freeze_prior_experiment(conn, p, config, body.prior_job_id, body.resume_candidate_index)
        try:
            job, created = await repo.create_job(conn, p, config, idempotency_key)
        except repo.IdempotencyConflict as e:
            raise ApiError(409,'Idempotency conflict',str(e)) from e
    response.status_code = 202 if created else 200
    response.headers["Location"] = f"/v1/jobs/{job['id']}"
    return JobAccepted(job_id=job["id"], status=job["status"], status_url=f"/v1/jobs/{job['id']}")


@router.get("", response_model=JobList, summary="List jobs visible to the caller")
async def list_jobs(p: Principal = Depends(get_principal), status: str | None = Query(default=None),
                    limit: int = Query(default=50, ge=1, le=200), before: str | None = Query(default=None)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await repo.list_jobs(conn, p, status, limit, before)
    items = [_job_out(r) for r in rows]
    next_cursor = rows[-1]["created_at"].isoformat() if len(rows) == limit else None
    return JobList(items=items, next_cursor=next_cursor)


@router.get("/{job_id}", response_model=JobOut, summary="Job status (cheap; poll this)")
async def get_job(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await _owned_job(conn, p, job_id)
        progress = await repo.job_progress(conn, job_id)
        best_index = None
        if job["best_iteration_id"]:
            best_index = await conn.fetchval("SELECT index FROM iterations WHERE id=$1", job["best_iteration_id"])
    return _job_out(job, progress, best_index)


@router.post("/{job_id}/cancel", response_model=JobOut, summary="Request cancellation")
async def cancel_job(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await repo.request_cancel(conn, p, job_id)
        if job is None:
            raise ApiError(404, "Not found", "Job not found.")
        job = await repo.get_job(conn,p,job_id)
        progress = await repo.job_progress(conn, job_id)
    return _job_out(job, progress)


@router.get("/{job_id}/iterations", response_model=list[IterationSummary], summary="Iteration history")
async def list_iterations(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        its = await repo.list_iterations(conn, job_id)
    return [_iter_summary(i) for i in its]


@router.get("/{job_id}/iterations/{index}", response_model=IterationDetail, summary="One iteration in full")
async def get_iteration(job_id: UUID, index: int, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        it = await repo.get_iteration(conn, job_id, index)
        if it is None:
            raise ApiError(404, "Not found", "Iteration not found.")
        runs = await repo.list_task_runs(conn, it["id"])
        parent_src = None
        if it["parent_iteration_id"]:
            parent = await conn.fetchrow(
                "SELECT a.content FROM iterations i JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.id=$1",
                it["parent_iteration_id"])
            parent_src = parent["content"] if parent else None
        digests = await conn.fetch(
            "SELECT task_name, failure_class, digest FROM task_runs WHERE iteration_id=$1 AND status IN ('failed','timeout','error') ORDER BY task_name",
            it["id"])
    diff = None
    if parent_src is not None:
        diff = "".join(difflib.unified_diff(parent_src.splitlines(True), it["agent_source"].splitlines(True),
                                            fromfile=f"agent.py@iter{index - 1}", tofile=f"agent.py@iter{index}"))
    base = _iter_summary(it).model_dump()
    return IterationDetail(
        **base, proposal=it["proposal"], learnings=it["learnings"], agent_source=it["agent_source"], diff_vs_parent=diff,
        tasks=[TaskRunOut(**dict(r)) for r in runs],
        failure_summary=[{"task_name": d["task_name"], "failure_class": d["failure_class"],
                          "summary": (d["digest"] or {}).get("summary")} for d in digests],
    )


@router.get("/{job_id}/iterations/{index}/tasks/{task}", response_model=TaskRunDetail, summary="Task run detail (trace, test output)")
async def get_task_run(job_id: UUID, index: int, task: str, p: Principal = Depends(get_principal),
                       include_trace: bool = Query(default=False)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        it = await repo.get_iteration(conn, job_id, index)
        if it is None:
            raise ApiError(404, "Not found", "Iteration not found.")
        tr = await repo.get_task_run(conn, it["id"], task)
        if tr is None:
            raise ApiError(404, "Not found", "Task run not found.")
    d = dict(tr)
    return TaskRunDetail(task_name=d["task_name"], status=d["status"], reward=d["reward"], failure_class=d["failure_class"],
                         duration_s=d["duration_s"], n_steps=d["n_steps"], exit_reason=d["exit_reason"],
                         test_stdout=d["test_stdout"], digest=d["digest"], trace=d["trace"] if include_trace else None,
                         input_tokens=d['input_tokens'],output_tokens=d['output_tokens'],cost_usd=d['cost_usd'],
                         model_cost_usd=d['model_cost_usd'],sandbox_cost_usd=d['sandbox_cost_usd'],cost_known=d['cost_known'])


@router.get('/{job_id}/iterations/{index}/attempts',summary='Immutable trial attempts, including retries and confirmations')
async def get_attempts(job_id: UUID,index: int,p: Principal=Depends(get_principal),
                       include_trace: bool=False,limit: int=Query(default=100,ge=1,le=500),
                       offset: int=Query(default=0,ge=0)):
    pool=await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn,p,job_id)
        it=await repo.get_iteration(conn,job_id,index)
        if it is None:
            raise ApiError(404,'Not found','Iteration not found.')
        columns='a.*' if include_trace else "a.id,a.iteration_id,a.evaluation_phase,a.task_name,a.repetition,a.attempt,a.lease_epoch,a.status,a.reward,a.failure_class,a.duration_s,a.n_steps,a.exit_reason,a.sandbox_id,a.input_tokens,a.output_tokens,a.cost_usd,a.model_cost_usd,a.sandbox_cost_usd,a.cost_known,a.metadata,a.finished_at"
        rows=await conn.fetch(f'''SELECT {columns} FROM task_attempts a WHERE iteration_id=$1
            ORDER BY finished_at,id LIMIT $2 OFFSET $3''',it['id'],limit,offset)
    return [dict(row) for row in rows]


@router.get("/{job_id}/agent", response_model=AgentOut, summary="Best agent source + diff vs baseline")
async def get_agent(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await _owned_job(conn, p, job_id)
        base = await repo.get_iteration(conn, job_id, 0)
        if base is None:
            raise ApiError(409, "Not ready", "Baseline iteration has not started yet.")
        best = None
        if job["best_iteration_id"]:
            best = await conn.fetchrow(
                "SELECT i.index, a.sha256, a.content FROM iterations i JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.id=$1",
                job["best_iteration_id"])
    src = best["content"] if best else base["agent_source"]
    diff = None
    if best and best["sha256"] != base["agent_sha256"]:
        diff = "".join(difflib.unified_diff(base["agent_source"].splitlines(True), src.splitlines(True),
                                            fromfile="agent.py@baseline", tofile=f"agent.py@iter{best['index']}"))
    return AgentOut(job_id=job_id, best_iteration_index=best["index"] if best else None, baseline_sha256=base["agent_sha256"],
                    best_sha256=best["sha256"] if best else None, source=src, diff_vs_baseline=diff)


@router.get("/{job_id}/events", response_model=list[EventOut], summary="Event timeline")
async def list_events(job_id: UUID, p: Principal = Depends(get_principal), after: int = Query(default=0, ge=0),
                      limit: int = Query(default=200, ge=1, le=1000)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        rows = await repo.list_events(conn, job_id, after, limit)
    return [EventOut(**dict(r)) for r in rows]
