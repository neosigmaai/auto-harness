# Task 1 Report: Restricted AgentPatchService

## What I implemented

- Added `autoharness_service/agent_patch.py` with:
  - `AgentPatchService(agent_path: Path | str = "agent/agent.py")`
  - `read_instruction() -> str`
  - `apply_instruction_patch(new_instruction: str, snapshot_dir: Path | str) -> AgentPatchResult`
  - `restore(source: str) -> None`
  - `AgentPatchResult(...)`
- Implementation details:
  - Uses `ast.parse` to find exactly one top-level `AGENT_INSTRUCTION` assignment.
  - Uses AST `lineno` / `end_lineno` to replace the whole assignment statement.
  - Generates replacement source as `AGENT_INSTRUCTION = <json-escaped python string>`.
  - Rejects dangerous LLM-controlled content containing any of:
    - ``````
    - `import `
    - `from `
    - `os.environ`
    - `subprocess`
    - `open(`
    - `eval(`
    - `exec(`
    - `__`
  - Writes snapshots to `initial.py` and `proposal-1.py`.
  - Runs `py_compile.compile(str(agent_path), doraise=True)` after writing.
  - Restores the original source on any validation or compile failure.

## Tests and results

- Added `tests/service/test_agent_patch.py`.
- Coverage in this file:
  - `test_read_instruction_finds_top_level_agent_instruction`
  - `test_apply_instruction_patch_changes_only_agent_instruction`
  - `test_apply_instruction_patch_rejects_dangerous_content`
  - `test_apply_instruction_patch_rejects_missing_or_duplicate_assignment`

## TDD evidence

### RED

Command:

```bash
python -m pytest tests/service/test_agent_patch.py -q
```

Output:

```text
==================================== ERRORS ====================================
__ ERROR collecting .worktrees/takehome-mvp/tests/service/test_agent_patch.py __
ImportError while importing test module '/Users/yinfeiwang/workspace/neosigma/auto-harness/.worktrees/takehome-mvp/tests/service/test_agent_patch.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/anaconda3/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/service/test_agent_patch.py:9: in <module>
    from autoharness_service.agent_patch import AgentPatchService
E   ModuleNotFoundError: No module named 'autoharness_service.agent_patch'
=========================== short test summary info ============================
ERROR tests/service/test_agent_patch.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.06s
```

### GREEN

Command:

```bash
python -m pytest tests/service/test_agent_patch.py -q
```

Output:

```text
............                                                             [100%]
12 passed in 0.02s
```

## Files changed

- `autoharness_service/agent_patch.py`
- `tests/service/test_agent_patch.py`

## Self-review findings

- Scope is contained to the two requested files.
- Only top-level `AGENT_INSTRUCTION` assignments are eligible; nested assignments are ignored.
- The patch path only rewrites the assignment statement and leaves surrounding imports and `TOOLS` source untouched.
- Compile failures and validation failures restore the original `agent/agent.py` contents before raising.
- Snapshot naming stays within the Task 1 contract: `initial.py` and `proposal-1.py`.

## Concerns

- The dangerous-content validation is a substring denylist because that is what the brief requires; it is intentionally conservative but not a full semantic sandbox.
- Snapshots are written before compile verification, so a compile failure would leave a rejected `proposal-1.py` snapshot on disk while the live `agent/agent.py` is restored. That matches the current implementation and does not violate the task brief.

## Review fixes follow-up

### Review finding 1: no workspace `__pycache__`

- Added a RED test asserting `apply_instruction_patch()` does not create `agent/__pycache__`.
- Changed compile verification to write bytecode to a temporary `cfile` outside the workspace via `tempfile.NamedTemporaryFile(...)`.

### Review finding 2: detect all top-level `AGENT_INSTRUCTION` writes

- Added RED tests for:
  - top-level `AGENT_INSTRUCTION: str = "old"` replacement support
  - rejecting `AGENT_INSTRUCTION += "x"`
  - rejecting mixed `Assign` + `AnnAssign` duplicates as ambiguous
- Updated AST scanning to count top-level `Assign`, `AnnAssign`, and `AugAssign` writes to `AGENT_INSTRUCTION`.
- The service now supports exactly one simple `Assign` or `AnnAssign` replacement target and rejects ambiguous write patterns.

### Additional test strengthening

- Strengthened the only-instruction-changed test to assert the exact patched source, not just substring preservation.

### RED/GREEN evidence for review fixes

RED:

```bash
python -m pytest tests/service/test_agent_patch.py -q
```

```text
5 failed, 12 passed in 0.06s
```

GREEN:

```bash
python -m pytest tests/service/test_agent_patch.py -q
```

```text
17 passed in 0.03s
```

## Task 1 re-review follow-up

- Added a regression test for `AGENT_INSTRUCTION: str` without a value and kept the service rejecting it cleanly.
- Narrowed top-level AST handling before `_extract_instruction_assignment()` so mypy no longer sees a raw `ast.stmt` flow into the extractor.
- Guarded `AnnAssign.value is None` before `ast.literal_eval()`.
- Added `encoding="utf-8"` to all `read_text()` / `write_text()` calls in the Task 1 service and test files.
- Normalized imports for isort in the two Task 1 files.

### Re-verify

```bash
python -m pytest tests/service/test_agent_patch.py -q
python -m isort --check-only autoharness_service/agent_patch.py tests/service/test_agent_patch.py
python -m black --check autoharness_service/agent_patch.py tests/service/test_agent_patch.py
python -m mypy autoharness_service/agent_patch.py
```

```text
18 passed in 0.06s
All checks passed
Success: no issues found in 1 source file
```
