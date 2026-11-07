"""Validate the supported pinned pytest/CTRF grading protocol.

This checks operational grading completion, not adversarial reward integrity.
"""
import math


class VerifierProtocolError(RuntimeError):
    pass


CONTRACT = 'pinned-pytest-cases-v2'


def pytest_probe_source(nonce: str) -> str:
    """Count pytest cases independently of CTRF 0.3.5's parameter grouping."""
    return '''import json, time
from pathlib import Path
_started = None
_selected = set()
_deselected = 0
_reports = {}
_duplicate_phase = False

def pytest_sessionstart(session):
    global _started
    _started = time.time()

def pytest_deselected(items):
    global _deselected
    _deselected += len(items)

def pytest_collection_finish(session):
    _selected.update(item.nodeid for item in session.items)

def pytest_runtest_logreport(report):
    global _duplicate_phase
    phases = _reports.setdefault(report.nodeid, {})
    if report.when in phases:
        _duplicate_phase = True
    phases[report.when] = report.outcome

def pytest_sessionfinish(session, exitstatus):
    counts = dict(passed=0, failed=0, skipped=0, incomplete=0)
    for nodeid in _selected:
        phases = _reports.get(nodeid, {})
        complete = ('setup' in phases and 'teardown' in phases
                    and (phases['setup'] != 'passed' or 'call' in phases)
                    and set(phases) <= {'setup', 'call', 'teardown'}
                    and all(value in {'passed', 'failed', 'skipped'} for value in phases.values()))
        if not complete:
            counts['incomplete'] += 1
        elif 'failed' in phases.values():
            counts['failed'] += 1
        elif 'skipped' in phases.values():
            counts['skipped'] += 1
        elif phases.get('call') == 'passed':
            counts['passed'] += 1
        else:
            counts['incomplete'] += 1
    Path('/logs/verifier/aos-session.json').write_text(json.dumps({
        'nonce': NONCE, 'started_at': _started, 'finished_at': time.time(),
        'tests_collected': session.testscollected, 'pytest_exit_code': int(exitstatus),
        'selected_cases': len(_selected), 'deselected_cases': _deselected,
        'executed_cases': sum(phases.get('call') in {'passed', 'failed'} for phases in _reports.values()),
        'unexpected_cases': len(set(_reports) - _selected),
        'duplicate_phase': _duplicate_phase, 'case_counts': counts}))
'''.replace('NONCE', repr(nonce))


def validate_report(report, session, expected_nonce, shell_exit, reward):
    def require(condition, message):
        if not condition:
            raise VerifierProtocolError(message)

    require(type(shell_exit) is int and shell_exit == 0, 'verifier_shell_failed')
    require(isinstance(session, dict) and session.get('nonce') == expected_nonce,
            'missing_or_stale_pytest_session')
    collected = session.get('tests_collected')
    exit_code = session.get('pytest_exit_code')
    require(type(collected) is int and collected > 0, 'no_tests_collected')
    require(type(exit_code) is int and exit_code in (0, 1), 'pytest_did_not_complete_tests')
    require(type(session.get('selected_cases')) is int and session['selected_cases'] == collected,
            'selected_case_count_mismatch')
    require(type(session.get('deselected_cases')) is int and session['deselected_cases'] == 0,
            'deselected_tests_not_supported')
    require(type(session.get('unexpected_cases')) is int and session['unexpected_cases'] == 0
            and session.get('duplicate_phase') is False, 'unexpected_or_repeated_test_cases')
    cases = session.get('case_counts')
    case_names = ('passed', 'failed', 'skipped', 'incomplete')
    require(isinstance(cases, dict) and all(type(cases.get(k)) is int and 0 <= cases[k] <= 100000
                                          for k in case_names), 'invalid_case_counts')
    require(sum(cases[k] for k in case_names) == collected and cases['incomplete'] == 0,
            'incomplete_case_execution')
    require(cases['passed'] + cases['failed'] > 0, 'no_test_cases_executed')
    require(type(session.get('executed_cases')) is int and 0 < session['executed_cases'] <= collected,
            'no_test_case_body_executed')
    require(isinstance(report, dict), 'missing_ctrf_report')
    result = report.get('results')
    require(isinstance(result, dict), 'invalid_ctrf_report')
    require(isinstance(result.get('tool'), dict) and result['tool'].get('name') == 'pytest', 'unsupported_test_tool')
    counts = result.get('summary')
    names = ('tests', 'passed', 'failed', 'skipped', 'pending', 'other')
    require(isinstance(counts, dict), 'missing_test_counts')
    require(all(type(counts.get(k)) is int and 0 <= counts[k] <= 100000 for k in names), 'invalid_test_counts')
    require(counts['tests'] == sum(counts[k] for k in names[1:]), 'inconsistent_test_counts')
    require(counts['pending'] == counts['other'] == 0, 'incomplete_test_report')
    tests = result.get('tests')
    require(isinstance(tests, list) and len(tests) == counts['tests'], 'test_list_count_mismatch')
    require(all(isinstance(t, dict) and t.get('status') in names[1:] for t in tests), 'invalid_test_status')
    for status in names[1:]:
        require(sum(t['status'] == status for t in tests) == counts[status], 'status_count_mismatch')
    require(collected >= counts['tests'], 'collection_count_mismatch')
    times = [session.get('started_at'), session.get('finished_at'), counts.get('start'), counts.get('stop')]
    require(all(type(t) in (int, float) and math.isfinite(t) for t in times), 'invalid_report_timestamps')
    start, stop, report_start, report_stop = times
    require(start <= stop and report_start <= report_stop and report_start >= start - 2 and report_stop <= stop + 2,
            'stale_or_outside_session_report')
    require(type(reward) in (int, float) and reward in (0, 1), 'invalid_reward')
    if exit_code == 0:
        require(cases['failed'] == 0 and counts['failed'] == 0 and reward == 1, 'reward_report_mismatch')
    else:
        # The pinned CTRF plugin groups parameter cases and can retain a
        # skipped status for a group that later fails. Exact nonce-hook cases
        # are authoritative; grouped CTRF statuses are only structural evidence.
        require(cases['failed'] > 0 and reward == 0, 'reward_report_mismatch')
    return {k: counts[k] for k in names}

