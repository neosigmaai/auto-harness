from __future__ import annotations

import logging
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any

import config
import db
import e2b_runner
import mock
import optimizer

INFRA_RETRY_BUDGET = 2
NO_GAIN_LIMIT = 2          
INFRA_NONE_FRACTION = 0.5   

log = logging.getLogger("worker")

AGENT_TEMPLATE = pathlib.Path(__file__).parents[1] / "agent/templates/terminal_bench.py"


def score(results: dict[str, float | None], subset: list[str]) -> float:
    if not subset:
        return 0.0
    return sum((results.get(t) or 0.0) for t in subset) / len(subset)


def infra_failed(results: dict[str, float | None]) -> bool:
    if not results:
        return True
    return sum(r is None for r in results.values()) / len(results) > INFRA_NONE_FRACTION


def regressed(stable: set[str] | None, now: dict[str, float | None],
              visible: list[str]) -> list[str]:
    """Tasks from the stable core (passed in every accepted iteration) that just broke."""
    if stable is None:
        return []
    return sorted(t for t in stable if t in visible and (now.get(t) or 0.0) < 1.0)


def passing(results: dict[str, float | None], visible: list[str]) -> set[str]:
    return {t for t in visible if (results.get(t) or 0.0) >= 1.0}


def canary_failure(run: Any, task_id: str) -> str | None:
    if run.error_detail:
        return None      
    rec = next((f for f in run.failures if f["task_id"] == task_id), None)
    if rec is not None and rec.get("tool_calls") == 0:
        return (f"the agent ran no commands at all on {task_id} and stopped immediately; "
                f"its loop almost certainly raised on the first API call")
    return None


def kill_sandbox(sandbox_id: str | None) -> None:
    if not sandbox_id:
        return
    try:
        from e2b import Sandbox

        Sandbox.kill(sandbox_id, api_key=config.E2B_API_KEY)
    except Exception as e:                                   # noqa: BLE001
        log.warning("could not kill sandbox %s: %s", sandbox_id, e)


def stop_signal(job: dict[str, Any]) -> str | None:
    """Iteration boundaries only — cancel is cooperative."""
    if db.job_status(job["org_id"], job["id"]) == "cancelling":
        return "cancelled"
    if job["deadline_at"] and datetime.now(timezone.utc) >= job["deadline_at"]:
        return "time_limit"
    return None


def run_job(job: dict[str, Any]) -> None:
    """Releases the sandbox on every exit path. SIGKILL is covered by the sandbox's own
    E2B timeout, not by this."""
    runner = opt = None
    try:
        runner, opt = _build(job)
        _loop(job, runner, opt)
    finally:
        if runner is not None and hasattr(runner, "close"):
            runner.close()


def _build(job: dict[str, Any]) -> tuple[Any, Any]:
    org, jid = job["org_id"], job["id"]
    visible = [t for t in job["task_ids"] if t not in job["holdout_task_ids"]]
    if job["mode"] == "mock":
        return mock.MockRunner(job_id=str(jid), visible=visible), mock.MockOptimizer()
    deadline = int((job["deadline_at"] - datetime.now(timezone.utc)).total_seconds())
    return (e2b_runner.E2BRunner(job_id=str(jid), deadline_seconds=max(deadline, 60),
                                 on_sandbox=lambda sid: db.set_sandbox(org, jid, sid)),
            optimizer.Optimizer())


def _loop(job: dict[str, Any], runner: Any, opt: Any) -> None:
    org, jid = job["org_id"], job["id"]
    visible = [t for t in job["task_ids"] if t not in job["holdout_task_ids"]]
    holdout = list(job["holdout_task_ids"])

    source = AGENT_TEMPLATE.read_text()             
    baseline = best = None
    best_results: dict[str, float | None] = {}
    stable: set[str] | None = None      
    ledger: list[str] = []
    proposal: str | None = None
    
    opt_usage: dict[str, int] = {}
    n = infra_used = no_gain = 0

    log.info(f"job {jid} mode={job['mode']} tasks={len(job['task_ids'])} "
          f"visible={len(visible)} holdout={len(holdout)}")

    while True:
        reason = stop_signal(job)
        if reason:
            return _finish(job, reason, baseline, best, best_results, holdout)

        run = runner.run(job["task_ids"], source)
        for k, v in opt_usage.items():         
            run.usage[k] = run.usage.get(k, 0) + v
        opt_usage = {}
        scored = not infra_failed(run.results)

        if not scored:
            db.add_iteration(org, jid, n, agent_source=source, proposal=proposal,
                             results=run.results, failures=[], outcome="infra_failed",
                             error_detail=run.error_detail, accepted=False, **run.usage)
            infra_used += 1
            log.info(f"job {jid} iter {n} infra_failed ({infra_used}/"
                  f"{INFRA_RETRY_BUDGET}): {run.error_detail}")
            if infra_used > INFRA_RETRY_BUDGET:
                db.finish_job(org, jid, status="failed", stopped_because="error",
                              failure_reason="infra",
                              error_detail="benchmark infrastructure failed repeatedly")
                return
        else:
            vis = score(run.results, visible)
            hold = score(run.results, holdout) if holdout else None
            broke = regressed(stable, run.results, visible) if n else []
            gained = sorted(passing(run.results, visible)
                            - passing(best_results, visible)) if n else []
            
            traded = bool(broke) and len(gained) > len(broke)
            n_broke = len(broke)                      
            if traded:
                log.info("job %s iter %s trade: +%s / -%s — veto waived",
                         jid, n, gained, broke)
                broke = []
            accepted = n == 0 or (not broke and (best is None or vis > best))
            outcome = "baseline" if n == 0 else "improved" if accepted else "regressed"

            db.add_iteration(org, jid, n, agent_source=source, proposal=proposal,
                             results=run.results, failures=run.failures,
                             visible_score=vis, holdout_score=hold, outcome=outcome,
                             accepted=accepted, **run.usage)
            why = f" (broke {broke})" if broke else ""
            log.info(f"job {jid} iter {n} visible={vis:.2f} "
                  f"holdout={hold if hold is None else f'{hold:.2f}'} "
                  f"{outcome} accepted={accepted}{why}")
            ledger.append(f"iter {n}: {proposal or 'baseline'} -> {vis:.2f} "
                          f"({'accepted' if accepted else 'rejected'}"
                          f"{f', traded +{len(gained)}/-{n_broke}' if traded else ''})")

            if n == 0:
                baseline = vis
                if vis >= 1.0:
                    log.info("job %s baseline is 1.00 on every visible task — nothing "
                             "to optimise, stopping", jid)
                    return _finish(job, "no_improvement", baseline, vis,
                                   run.results, holdout)
            if accepted:
                best, best_results, no_gain = vis, run.results, 0
                now_passing = passing(run.results, visible)
                stable = now_passing if stable is None else stable & now_passing
            else:
                no_gain += 1
                source = db.best_accepted(org, jid)["agent_source"]  # last good branch

        n += 1
        if no_gain >= NO_GAIN_LIMIT:
            return _finish(job, "no_improvement", baseline, best, best_results, holdout)
        if n > job["max_iterations"]:
            return _finish(job, "max_iterations", baseline, best, best_results, holdout)
        if not scored:
            continue    

        failures = [f for f in run.failures if f["task_id"] in visible]  # holdout out
        base, extra = source, []
        for attempt in range(2):
            try:
                proposal, source, opt_usage = opt.propose(base, failures + extra, ledger)
            except optimizer.ProposalRejected as e:
                db.finish_job(org, jid, status="failed", stopped_because="error",
                              failure_reason="llm", error_detail=str(e))
                return

            probe = runner.run([visible[0]], source)
            for k, v in probe.usage.items():        # the canary is not free; count it
                opt_usage[k] = opt_usage.get(k, 0) + v
            broken = canary_failure(probe, visible[0])
            if not broken:
                break
            log.info(f"job {jid} canary {attempt + 1}/2 rejected the proposal: "
                  f"{broken}")
            
            extra = [{"task_id": visible[0], "reward": 0.0, "tool_calls": 0,
                      "failing_commands": [], "tail": (probe.failures[0]["tail"]
                                                       if probe.failures else ""),
                      "verifier_output": f"YOUR PREVIOUS PROPOSAL WAS REJECTED: {broken}"}]
        else:
            return _finish(job, "no_improvement", baseline, best, best_results, holdout)


def _finish(job: dict[str, Any], reason: str, baseline: float | None, best: float | None,
            best_results: dict[str, float | None], holdout: list[str]) -> None:
    cancelled = reason == "cancelled"
    never_ran = baseline is None and not cancelled
    status = "cancelled" if cancelled else "failed" if never_ran else "succeeded"
    db.finish_job(
        job["org_id"], job["id"], status=status,
        stopped_because="error" if never_ran else reason,
        failure_reason="cancelled" if cancelled else "infra" if never_ran else None,
        error_detail=("every iteration failed before producing a score; see the "
                      "iteration rows for why") if never_ran else None,
        scores={"baseline": baseline, "best_visible": best,
                "holdout": score(best_results, holdout) if holdout and best_results
                else None})
    log.info(f"job {job['id']} done: {status}/{reason} "
          f"baseline={baseline} best={best}")


def main() -> None:
    log.info("claim loop up")
    while True:
        for lost in db.reclaim_expired():
            log.info(f"reclaiming abandoned job {lost['id']}")
            kill_sandbox(lost["sandbox_id"])       
        job = db.claim_job()
        if not job:
            time.sleep(1)
            continue
        try:
            run_job(job)
        except Exception as e:                                   # noqa: BLE001
            log.exception("job %s crashed", job["id"])
            kill_sandbox(job.get("sandbox_id"))
            db.finish_job(job["org_id"], job["id"], status="failed",
                          stopped_because="error", failure_reason="infra",
                          error_detail=f"{type(e).__name__}: {e}")


def demo() -> None:
    """Pure loop logic. No database, no sandbox."""
    r = lambda **kw: type("R", (), {"error_detail": None, "failures": [], **kw})()  # noqa: E731

    # Acted but failed: a legitimate bad score, not a broken agent.
    tried = r(failures=[{"task_id": "t", "tool_calls": 7, "tail": ""}])
    assert canary_failure(tried, "t") is None
    # Never acted: broken.
    assert canary_failure(r(failures=[{"task_id": "t", "tool_calls": 0, "tail": ""}]),
                          "t") is not None
    # Passed the task outright: no failure record at all.
    assert canary_failure(r(), "t") is None
    # Infra died: not the agent's fault, let the full run decide.
    assert canary_failure(type("R", (), {"error_detail": "sandbox died",
                                         "failures": []})(), "t") is None

    assert score({"a": 1.0, "b": 0.0}, ["a", "b"]) == 0.5
    assert score({"a": None, "b": 1.0}, ["a", "b"]) == 0.5      # None counts as 0.0
    assert score({}, []) == 0.0
    assert infra_failed({"a": None, "b": None}) and not infra_failed({"a": None, "b": 1.0})
    assert regressed({"a"}, {"a": 0.0}, ["a"]) == ["a"]         # stable task broke
    assert regressed({"a"}, {"a": 1.0}, ["a"]) == []
    assert regressed(set(), {"a": 0.0}, ["a"]) == []            # nothing stable yet
    assert regressed(None, {"a": 0.0}, ["a"]) == []             # baseline: no history
    assert regressed({"a"}, {"a": 0.0}, ["b"]) == []            # holdout never counts

    # the core only shrinks, so proven-flaky tasks stop vetoing
    core = passing({"a": 1.0, "b": 1.0, "c": 0.0}, ["a", "b", "c"])
    assert core == {"a", "b"}
    core &= passing({"a": 1.0, "b": 0.0, "c": 1.0}, ["a", "b", "c"])
    assert core == {"a"}
    assert regressed(core, {"a": 1.0, "b": 0.0}, ["a", "b"]) == []   # b is free to churn

    # a real run: 0.47 -> 0.60, gained 3, lost nginx-request-logging. Strict guard said no.
    vis_tasks = ["nginx-request-logging", "git-leak-recovery", "kv-store-grpc",
                 "largest-eigenval", "fix-git"]
    before = {"nginx-request-logging": 1.0, "fix-git": 1.0, "git-leak-recovery": 0.0,
              "kv-store-grpc": 0.0, "largest-eigenval": 0.0}
    after = {"nginx-request-logging": 0.0, "fix-git": 1.0, "git-leak-recovery": 1.0,
             "kv-store-grpc": 1.0, "largest-eigenval": 1.0}
    broke = regressed(passing(before, vis_tasks), after, vis_tasks)
    gain = sorted(passing(after, vis_tasks) - passing(before, vis_tasks))
    assert broke == ["nginx-request-logging"] and len(gain) == 3
    assert len(gain) > len(broke), "3 gained vs 1 lost must waive the veto"
    assert score(after, vis_tasks) > score(before, vis_tasks), "and the score must rise"

    # But the hardcode shape -- gain one, lose one -- must still be vetoed.
    swap = {"nginx-request-logging": 0.0, "fix-git": 1.0, "git-leak-recovery": 1.0,
            "kv-store-grpc": 0.0, "largest-eigenval": 0.0}
    assert not (len(sorted(passing(swap, vis_tasks) - passing(before, vis_tasks)))
                > len(regressed(passing(before, vis_tasks), swap, vis_tasks)))
    print("ok — canary, scoring, infra threshold, regression guard")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        main()
