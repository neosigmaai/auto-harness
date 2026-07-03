from __future__ import annotations

import py_compile
import textwrap
from pathlib import Path

import pytest
from autoharness_service.agent_patch import AgentPatchService


def _write_agent(tmp_path: Path, source: str) -> Path:
    agent_path = tmp_path / "agent" / "agent.py"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(textwrap.dedent(source), encoding="utf-8")
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
    original_source = agent_path.read_text(encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"

    service = AgentPatchService(agent_path)
    result = service.apply_instruction_patch(
        "new line 1\nnew line 2",
        snapshot_dir=snapshot_dir,
    )

    patched_source = agent_path.read_text(encoding="utf-8")

    assert result.original_source == original_source
    assert result.patched_source == patched_source
    assert result.original_instruction == "old"
    assert result.new_instruction == "new line 1\nnew line 2"
    assert "import os" in patched_source
    assert 'TOOLS = [{"name": "bash"}]' in patched_source
    assert 'AGENT_INSTRUCTION = "new line 1\\nnew line 2"' in patched_source
    py_compile.compile(str(agent_path), doraise=True)
    assert (snapshot_dir / "initial.py").read_text(encoding="utf-8") == original_source
    assert (snapshot_dir / "proposal-1.py").read_text(
        encoding="utf-8"
    ) == patched_source
    assert result.snapshot_paths == {
        "initial": str(snapshot_dir / "initial.py"),
        "proposal-1": str(snapshot_dir / "proposal-1.py"),
    }
    assert patched_source == textwrap.dedent(
        """
        import os

        MAX_STEPS = 80
        AGENT_INSTRUCTION = "new line 1\\nnew line 2"
        TOOLS = [{"name": "bash"}]
        """
    )


def test_apply_instruction_patch_does_not_create_agent_pycache(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = "old"
        """,
    )

    service = AgentPatchService(agent_path)
    service.apply_instruction_patch(
        "updated",
        snapshot_dir=tmp_path / "snapshots",
    )

    assert not (agent_path.parent / "__pycache__").exists()


def test_apply_instruction_patch_allows_natural_language_from(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = "old"
        """,
    )
    service = AgentPatchService(agent_path)

    result = service.apply_instruction_patch(
        "Learn from previous failures before deciding the final answer.",
        snapshot_dir=tmp_path / "snapshots",
    )

    assert "Learn from previous failures" in result.new_instruction
    assert "Learn from previous failures" in agent_path.read_text(encoding="utf-8")


def test_apply_instruction_patch_can_disable_guard_for_local_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = "old"
        """,
    )
    service = AgentPatchService(agent_path)
    monkeypatch.setenv("AUTOHARNESS_DISABLE_PATCH_GUARD", "1")

    result = service.apply_instruction_patch(
        "For demo only: import os, from pathlib import Path, open('x')",
        snapshot_dir=tmp_path / "snapshots",
    )

    assert "import os" in result.new_instruction
    assert "open('x')" in agent_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "dangerous_import",
    [
        "import os",
        "  import pathlib",
        "- import subprocess",
        "from os import path",
        "  from pathlib import Path",
        "- from subprocess import run",
    ],
)
def test_apply_instruction_patch_rejects_code_like_import_statements(
    tmp_path: Path,
    dangerous_import: str,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = "old"
        """,
    )
    original_source = agent_path.read_text(encoding="utf-8")
    service = AgentPatchService(agent_path)

    with pytest.raises(ValueError):
        service.apply_instruction_patch(
            dangerous_import,
            snapshot_dir=tmp_path / "snapshots",
        )

    assert agent_path.read_text(encoding="utf-8") == original_source


def test_discard_proposal_snapshot_removes_rejected_optimized_version(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION = "old"
        """,
    )
    service = AgentPatchService(agent_path)
    result = service.apply_instruction_patch(
        "updated",
        snapshot_dir=tmp_path / "snapshots",
    )

    discarded = service.discard_proposal_snapshot(result.snapshot_paths)

    assert discarded == {"proposal-1": result.snapshot_paths["proposal-1"]}
    assert not Path(result.snapshot_paths["proposal-1"]).exists()
    assert Path(result.snapshot_paths["initial"]).exists()


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
    original_source = agent_path.read_text(encoding="utf-8")
    service = AgentPatchService(agent_path)

    with pytest.raises(ValueError):
        service.apply_instruction_patch(
            dangerous_content,
            snapshot_dir=tmp_path / "snapshots",
        )

    assert agent_path.read_text(encoding="utf-8") == original_source


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


def test_apply_instruction_patch_supports_annotated_assignment(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION: str = "old"
        """,
    )
    service = AgentPatchService(agent_path)

    result = service.apply_instruction_patch(
        "updated",
        snapshot_dir=tmp_path / "snapshots",
    )

    assert agent_path.read_text() == '\nAGENT_INSTRUCTION = "updated"\n'
    assert result.original_instruction == "old"


def test_apply_instruction_patch_rejects_annotated_assignment_without_value(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(
        tmp_path,
        """
        AGENT_INSTRUCTION: str
        """,
    )
    original_source = agent_path.read_text(encoding="utf-8")
    service = AgentPatchService(agent_path)

    with pytest.raises(ValueError):
        service.apply_instruction_patch(
            "updated",
            snapshot_dir=tmp_path / "snapshots",
        )

    assert agent_path.read_text(encoding="utf-8") == original_source


@pytest.mark.parametrize(
    "source",
    [
        """
        AGENT_INSTRUCTION = "old"
        AGENT_INSTRUCTION += "more"
        """,
        """
        AGENT_INSTRUCTION: str = "old"
        AGENT_INSTRUCTION = "new"
        """,
        """
        AGENT_INSTRUCTION = "old"
        AGENT_INSTRUCTION: str = "new"
        """,
    ],
)
def test_apply_instruction_patch_rejects_ambiguous_instruction_writes(
    tmp_path: Path,
    source: str,
) -> None:
    agent_path = _write_agent(tmp_path, source)
    original_source = agent_path.read_text()
    service = AgentPatchService(agent_path)

    with pytest.raises(ValueError):
        service.apply_instruction_patch(
            "updated",
            snapshot_dir=tmp_path / "snapshots",
        )

    assert agent_path.read_text() == original_source
