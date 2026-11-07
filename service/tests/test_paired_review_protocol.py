"""Paired protocol checks; no inference, database or Docker required."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.run_paired_selection_review import (
    aggregate_source, archive_report, pinned_source_hashes, redact_attempt,
    selection_task_names, validate_audit_limits, verifier_valid,
)
from service.repo import sha256_text


def config(count):
    tasks = [f"selection-{n}" for n in range(count)]
    return {"tasks": ["feedback", *tasks, "holdout"], "evaluation_split": {
        "feedback": ["feedback"], "selection": tasks, "holdout": ["holdout"]}}


def attempt(task="selection-0", status="passed", reward=1, index=1):
    return {"task_name": task, "evaluation_phase": "main", "repetition": 1, "attempt": index,
        "status": status, "reward": reward, "duration_s": 2, "model_cost_usd": .1,
        "cost_known": True, "input_tokens": 10, "output_tokens": 2,
        "metadata": {"model_cost_known": True, "verifier_validity": {
            "valid": True, "contract": "pinned-pytest-ctrf-v1", "fresh_session_marker": True,
            "pytest_exit_code": 0 if reward == 1 else 1,
            "counts": {"tests": 1, "passed": int(reward == 1), "failed": int(reward != 1),
                       "skipped": 0, "pending": 0, "other": 0}}}}


def v2_attempt(passed=True):
    row = attempt(status="passed" if passed else "failed", reward=int(passed))
    validity = row["metadata"]["verifier_validity"]
    validity.update(contract="pinned-pytest-cases-v2", tests_collected=27, selected_cases=27,
                    executed_cases=27, deselected_cases=0, unexpected_cases=0, duplicate_phase=False,
                    case_counts={"passed": 27 if passed else 26, "failed": 0 if passed else 1,
                                 "skipped": 0, "incomplete": 0})
    validity["counts"].update(tests=3, passed=3 if passed else 2, failed=0 if passed else 1)
    return row


@pytest.mark.parametrize("passed", [True, False])
def test_v2_accepts_complete_exact_cases_with_fewer_ctrf_groups(passed):
    row = v2_attempt(passed)
    assert verifier_valid(row)
    summary = aggregate_source([{"attempts": [row]}], [row["task_name"]], repetitions=1)
    assert summary["verifier_complete"] and summary["mean_reward"] == int(passed)
    assert verifier_valid(attempt())  # explicitly supported historical v1


@pytest.mark.parametrize("mutation", [
    lambda v: v.pop("case_counts"),
    lambda v: v.update(tests_collected=True),
    lambda v: v.update(selected_cases=3),
    lambda v: v.update(executed_cases=0),
    lambda v: v.update(executed_cases=26),
    lambda v: v.update(deselected_cases=1),
    lambda v: v.update(unexpected_cases=1),
    lambda v: v.update(duplicate_phase=True),
    lambda v: v["case_counts"].update(passed=26, incomplete=1),
    lambda v: v["case_counts"].update(passed=0, skipped=27),
    lambda v: v["case_counts"].update(passed=26),
    lambda v: v["case_counts"].update(passed=26, failed=1),
])
def test_v2_never_infers_case_coverage_from_grouped_summary(mutation):
    row = v2_attempt()
    mutation(row["metadata"]["verifier_validity"])
    assert not verifier_valid(row)


def test_v2_exact_failure_can_have_sticky_skipped_grouped_status():
    row = v2_attempt(False)
    v = row["metadata"]["verifier_validity"]
    v["counts"].update(passed=0, failed=0, skipped=3)
    assert verifier_valid(row)
    row["reward"] = 1
    assert not verifier_valid(row)


def test_both_report_redactors_keep_case_coverage_without_test_names():
    from scripts.run_final_holdout_review import redact_attempt as redact_holdout
    row = v2_attempt()
    v = row["metadata"]["verifier_validity"]
    v.update(nodeids=["private assertion name"], raw_report="private test body")
    v["case_counts"]["private_nodes"] = ["private node"]
    for redact in (redact_attempt, redact_holdout):
        result = redact(row)
        assert verifier_valid(result)
        assert result["metadata"]["verifier_validity"]["case_counts"]["passed"] == 27
        assert "private" not in repr(result)


@pytest.mark.parametrize("count", [3, 4])
def test_uses_exact_source_selection_and_full_two_repetition_denominator(count):
    source = config(count)
    tasks = selection_task_names(source)
    assert tasks == source["evaluation_split"]["selection"]
    tasks.reverse()
    assert source["evaluation_split"]["selection"][0] == "selection-0"
    rows = [attempt(task) for task in tasks]
    metrics = aggregate_source([{"attempts": rows}, {"attempts": rows}], tasks)
    assert metrics["planned_trials"] == count * 2
    assert metrics["observed_trials"] == count * 2
    assert metrics["complete"] and metrics["verifier_complete"]
    partial = aggregate_source([{"attempts": rows}], tasks)
    assert not partial["complete"] and partial["mean_reward"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_unsupported_source_selection_count_rejected(count):
    with pytest.raises(ValueError, match="three or four"):
        selection_task_names(config(count))


def test_duplicate_unplanned_or_overlapping_selection_rejected():
    for mutate in (
        lambda c: c["evaluation_split"]["selection"].__setitem__(1, "selection-0"),
        lambda c: c["tasks"].remove("selection-0"),
        lambda c: c["evaluation_split"]["holdout"].append("selection-0"),
    ):
        source = config(3)
        mutate(source)
        with pytest.raises(ValueError):
            selection_task_names(source)


@pytest.mark.parametrize("status,reward", [("timeout", None), ("error", None), ("failed", None),
                                          ("passed", None), ("passed", True)])
def test_ungraded_or_invalid_outcome_never_completes_external_comparison(status, reward):
    row = attempt(status=status, reward=reward)
    assert not verifier_valid(row)
    result = aggregate_source([{"attempts": [row]}, {"attempts": [row]}], [row["task_name"]])
    assert not result["complete"] and not result["verifier_complete"]
    assert result["mean_reward"] is None


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(valid=False),
    lambda v: v.update(contract="unknown"),
    lambda v: v.update(fresh_session_marker=False),
    lambda v: v.update(pytest_exit_code=5),
    lambda v: v["counts"].update(tests=0, passed=0),
    lambda v: v["counts"].update(tests=2),
])
def test_trusted_completion_contract_must_be_consistent(mutation):
    row = attempt()
    mutation(row["metadata"]["verifier_validity"])
    assert not verifier_valid(row)


def test_verified_timeout_remains_valid_but_unverified_first_outcome_cannot_be_replaced_by_later_pass():
    valid_timeout = attempt(status="timeout", reward=0)
    assert verifier_valid(valid_timeout)
    assert aggregate_source([{"attempts": [valid_timeout]}], [valid_timeout["task_name"]], repetitions=1)["complete"]
    unverified = attempt(status="timeout", reward=None)
    later = attempt(index=2)
    result = aggregate_source([{"attempts": [unverified, later]}], [later["task_name"]], repetitions=1)
    assert not result["complete"] and result["mean_reward"] is None
    assert result["physical_attempts"] == 2


def test_first_verified_failure_and_all_retry_resource_costs_are_retained():
    rows = [attempt(status="error", reward=None), attempt(status="failed", reward=0, index=2), attempt(index=3)]
    result = aggregate_source([{"attempts": rows}], [rows[0]["task_name"]], repetitions=1)
    assert result["complete"] and result["mean_reward"] == 0
    assert result["total_model_cost_usd"] == pytest.approx(.3)
    assert result["p95_task_duration_s"] == 6
    assert result["physical_attempts"] == 3


def test_explicit_source_pins_reject_other_candidate_bytes():
    sources = {"baseline": "original", "candidate": "candidate3"}
    expected = {key: sha256_text(value) for key, value in sources.items()}
    assert pinned_source_hashes(sources, expected["baseline"], expected["candidate"]) == expected
    with pytest.raises(ValueError, match="authorized hashes"):
        pinned_source_hashes({**sources, "candidate": "different"}, expected["baseline"], expected["candidate"])


def test_existing_report_is_preserved_before_reuse(tmp_path):
    report = tmp_path / "docs" / "comparison.json"
    report.parent.mkdir()
    old = b'{"status":"INCOMPLETE","old_spend":0.25}'
    report.write_bytes(old)
    run = tmp_path / "new-run"
    run.mkdir()
    archived = archive_report(report, run)
    report.write_text('{"status":"starting"}')
    from pathlib import Path
    assert Path(archived).read_bytes() == old


def test_redacted_evidence_retains_verifier_counts_without_private_payloads():
    row = attempt()
    row.update(trace=["PRIVATE"], test_stdout="PRIVATE")
    row["metadata"]["provider_body"] = "PRIVATE"
    row["metadata"]["verifier_validity"]["raw_report"] = "PRIVATE"
    clean = redact_attempt(row)
    assert "PRIVATE" not in str(clean)
    assert verifier_valid(clean)


@pytest.mark.parametrize("cap,minutes", [(2.01,45), (6,45), (2,46), (2,90), (float("nan"),45)])
def test_new_authorization_is_hard_capped_to_two_dollars_and_45_minutes(cap, minutes):
    with pytest.raises(SystemExit):
        validate_audit_limits(SimpleNamespace(allow_paid=True, max_total_usd=cap, max_wall_clock_min=minutes))


def test_explicit_paid_gate_and_authorized_limits():
    with pytest.raises(SystemExit, match="No model calls"):
        validate_audit_limits(SimpleNamespace(allow_paid=False))
    validate_audit_limits(SimpleNamespace(allow_paid=True, max_total_usd=2, max_wall_clock_min=45))
