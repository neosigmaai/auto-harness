from __future__ import annotations

import py_compile
import textwrap
from pathlib import Path

import pytest

from autoharness_service.agent_patch import AgentPatchService


def _write_agent(tmp_path: Path, source: str) -> Path:
    agent_path = tmp_path / "agent" / "agent.py"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(textwrap.dedent(source))
    return agent_path


def test_read_instruction_finds_top_level_agent_instruction(tmp_path: Path) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        import os

        MAX_STEPS = 80
        AGENT_INSTRUCTION = \"\"\"old\"\"\"
        """,
    )

    service = AgentPatchService(agent_path)

    assert service.read_instruction() == "old"


def test_apply_instruction_patch_changes_only_agent_instruction(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        import os

        MAX_STEPS = 80
        AGENT_INSTRUCTION = \"\"\"old\"\"\"
        TOOLS = [{"name": "bash"}]
        """,
    )
    original_source = agent_path.read_text()
    snapshot_dir = tmp_path / "snapshots"

    service = AgentPatchService(agent_path)
    result = service.apply_instruction_patch(
        "new line 1\nnew line 2",
        snapshot_dir=snapshot_dir,
    )

    patched_source = agent_path.read_text()

    assert result.original_source == original_source
    assert result.patched_source == patched_source
    assert result.original_instruction == "old"
    assert result.new_instruction == "new line 1\nnew line 2"
    assert "import os" in patched_source
    assert 'TOOLS = [{"name": "bash"}]' in patched_source
    assert 'AGENT_INSTRUCTION = "new line 1\\nnew line 2"' in patched_source
    py_compile.compile(str(agent_path), doraise=True)
    assert (snapshot_dir / "initial.py").read_text() == original_source
    assert (snapshot_dir / "proposal-1.py").read_text() == patched_source
    assert result.snapshot_paths == {
        "initial": str(snapshot_dir / "initial.py"),
        "proposal-1": str(snapshot_dir / "proposal-1.py"),
    }


@pytest.mark.parametrize(
    "dangerous_content",
    [
        "```",
        "import os",
        "from os import path",
        "os.environ",
        "subprocess.run(['pwd'])",
        "open('secret.txt')",
        "eval('1+1')",
        "exec('print(1)')",
        "__builtins__",
    ],
)
def test_apply_instruction_patch_rejects_dangerous_content(
    tmp_path: Path,
    dangerous_content: str,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = \"\"\"old\"\"\"
        """,
    )
    original_source = agent_path.read_text()
    service = AgentPatchService(agent_path)

    with pytest.raises(ValueError):
        service.apply_instruction_patch(
            dangerous_content,
            snapshot_dir=tmp_path / "snapshots",
        )

    assert agent_path.read_text() == original_source


def test_apply_instruction_patch_rejects_missing_or_duplicate_assignment(
    tmp_path: Path,
) -> None:
    missing_agent_path = _write_agent(
        tmp_path / "missing",
        """
        MAX_STEPS = 80
        """,
    )
    duplicate_agent_path = _write_agent(
        tmp_path / "duplicate",
        """
        AGENT_INSTRUCTION = "old"
        AGENT_INSTRUCTION = "new"
        """,
    )

    missing_service = AgentPatchService(missing_agent_path)
    duplicate_service = AgentPatchService(duplicate_agent_path)

    with pytest.raises(ValueError):
        missing_service.apply_instruction_patch(
            "updated",
            snapshot_dir=tmp_path / "missing-snapshots",
        )

    with pytest.raises(ValueError):
        duplicate_service.apply_instruction_patch(
            "updated",
            snapshot_dir=tmp_path / "duplicate-snapshots",
        )
