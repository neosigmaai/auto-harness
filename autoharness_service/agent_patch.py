from __future__ import annotations

import ast
import json
import py_compile
import tempfile
from dataclasses import dataclass
from pathlib import Path

_DANGEROUS_SUBSTRINGS = (
    "```",
    "import ",
    "from ",
    "os.environ",
    "subprocess",
    "open(",
    "eval(",
    "exec(",
    "__",
)


@dataclass(frozen=True)
class AgentPatchResult:
    original_source: str
    patched_source: str
    original_instruction: str
    new_instruction: str
    snapshot_paths: dict[str, str]


@dataclass(frozen=True)
class _InstructionAssignment:
    value: str
    lineno: int
    end_lineno: int


class AgentPatchService:
    def __init__(self, agent_path: Path | str = "agent/agent.py") -> None:
        self.agent_path = Path(agent_path)

    def read_instruction(self) -> str:
        source = self.agent_path.read_text(encoding="utf-8")
        return self._find_instruction_assignment(source).value

    def apply_instruction_patch(
        self,
        new_instruction: str,
        snapshot_dir: Path | str,
    ) -> AgentPatchResult:
        original_source = self.agent_path.read_text(encoding="utf-8")
        try:
            assignment = self._find_instruction_assignment(original_source)
            self._validate_instruction(new_instruction)
            patched_source = self._build_patched_source(
                original_source=original_source,
                assignment=assignment,
                new_instruction=new_instruction,
            )
            snapshot_paths = self._write_snapshots(
                snapshot_dir=Path(snapshot_dir),
                original_source=original_source,
                patched_source=patched_source,
            )
            self.agent_path.write_text(patched_source, encoding="utf-8")
            self._compile_agent()
        except Exception:
            self.restore(original_source)
            raise

        return AgentPatchResult(
            original_source=original_source,
            patched_source=patched_source,
            original_instruction=assignment.value,
            new_instruction=new_instruction,
            snapshot_paths=snapshot_paths,
        )

    def restore(self, source: str) -> None:
        self.agent_path.write_text(source, encoding="utf-8")

    def _find_instruction_assignment(self, source: str) -> _InstructionAssignment:
        module = ast.parse(source)
        matches: list[_InstructionAssignment] = []
        write_count = 0

        for node in module.body:
            if not self._is_instruction_write(node):
                continue
            write_count += 1

            if isinstance(node, ast.Assign):
                matches.append(self._extract_instruction_assignment(node))
                continue
            if isinstance(node, ast.AnnAssign):
                matches.append(self._extract_instruction_assignment(node))
                continue

        if write_count != 1 or len(matches) != 1:
            raise ValueError(
                "Expected exactly one top-level AGENT_INSTRUCTION assignment"
            )
        return matches[0]

    def _is_instruction_write(self, node: ast.stmt) -> bool:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                return False
            return self._is_instruction_name(node.targets[0])
        if isinstance(node, ast.AnnAssign):
            return self._is_instruction_name(node.target)
        if isinstance(node, ast.AugAssign):
            return self._is_instruction_name(node.target)
        return False

    def _is_instruction_name(self, target: ast.expr) -> bool:
        return isinstance(target, ast.Name) and target.id == "AGENT_INSTRUCTION"

    def _extract_instruction_assignment(
        self,
        node: ast.Assign | ast.AnnAssign,
    ) -> _InstructionAssignment:
        if node.end_lineno is None:
            raise ValueError("AGENT_INSTRUCTION assignment is missing end_lineno")

        value = node.value
        if value is None:
            raise ValueError("AGENT_INSTRUCTION must be a string literal")

        try:
            instruction = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("AGENT_INSTRUCTION must be a string literal") from exc

        if not isinstance(instruction, str):
            raise ValueError("AGENT_INSTRUCTION must be a string literal")

        return _InstructionAssignment(
            value=instruction,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
        )

    def _validate_instruction(self, new_instruction: str) -> None:
        for token in _DANGEROUS_SUBSTRINGS:
            if token in new_instruction:
                raise ValueError(f"Rejected dangerous instruction content: {token}")

    def _build_patched_source(
        self,
        *,
        original_source: str,
        assignment: _InstructionAssignment,
        new_instruction: str,
    ) -> str:
        replacement = f"AGENT_INSTRUCTION = {json.dumps(new_instruction)}\n"
        lines = original_source.splitlines(keepends=True)
        start_index = assignment.lineno - 1
        end_index = assignment.end_lineno
        updated_lines = lines[:start_index] + [replacement] + lines[end_index:]
        return "".join(updated_lines)

    def _write_snapshots(
        self,
        *,
        snapshot_dir: Path,
        original_source: str,
        patched_source: str,
    ) -> dict[str, str]:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        initial_path = snapshot_dir / "initial.py"
        proposal_path = snapshot_dir / "proposal-1.py"
        initial_path.write_text(original_source, encoding="utf-8")
        proposal_path.write_text(patched_source, encoding="utf-8")
        return {
            "initial": str(initial_path),
            "proposal-1": str(proposal_path),
        }

    def _compile_agent(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pyc") as compiled_file:
            py_compile.compile(
                str(self.agent_path),
                cfile=compiled_file.name,
                doraise=True,
            )
