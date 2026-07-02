# Task 2 Report

## Summary

Implemented the service domain models, request/response schemas, and result normalization helpers for the AutoHarness take-home MVP.

## Files

- `autoharness_service/models.py`
- `autoharness_service/schemas.py`
- `autoharness_service/normalizer.py`
- `tests/service/test_normalizer.py`

## Behavior Delivered

- `TaskResultRecord` and `FailureSummary` dataclasses.
- `RunStatus` and `TaskStatus` aliases plus status sets.
- `normalize_reward_result(...)` for passed, failed, and non-finite reward cases.
- `normalize_missing_result(...)` for infra failures caused by missing results.
- `build_failure_summary(...)` with stable top-failure ordering.
- `RunCreateRequest` validation for task-id safety and mode/provider pairing.
- `RunCreateResponse` and `RunProgress` Pydantic schemas.

## Verification

- `python -m pytest tests/service/test_normalizer.py -q`
- `python -m pytest tests/service/test_imports.py tests/service/test_normalizer.py -q`
- `git diff --check`

All passed.

## Commit

- `8833623` - `feat: add service schemas and result normalizer`

## Fix

- Tests run: `python -m pytest tests/service/test_normalizer.py -q`
- Tests run: `python -m pytest tests/service/test_imports.py tests/service/test_normalizer.py -q`
- Tests run: `git diff --check`
- Commit: `340804c` - `fix: align service schemas and normalizer`
