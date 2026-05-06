"""Thin wrapper around Docker exec exposed to the agent.

Why this exists:
- ``ContainerEnvironment.execute`` returns a dict with merged stdout+stderr;
  agents expect a (output, returncode) tuple.
- ProgramBench inference runs the agent as an unprivileged numeric user, while
  the harness keeps setup/packaging operations on the container's default user.
- Tool timeouts are clamped to the orchestrator's wall-clock deadline before a
  synchronous ``docker exec`` starts, so a model-requested timeout cannot pin a
  worker past the task budget.
"""

from __future__ import annotations

import subprocess
import time

from programbench.container import ContainerEnvironment


class BashTool:
    """Callable: ``bash(command, timeout=300) -> (output, returncode)``."""

    def __init__(
        self,
        env: ContainerEnvironment,
        default_timeout: int = 300,
        *,
        exec_user: str | None = None,
        deadline: float | None = None,
    ) -> None:
        self.env = env
        self.default_timeout = default_timeout
        self.exec_user = exec_user if exec_user is not None else getattr(env, "agent_exec_user", None)
        self.deadline = deadline if deadline is not None else getattr(env, "agent_deadline", None)
        # Probe once so traces expose container/user setup problems early.
        probe = self._execute("true", timeout=10)
        self.login_shell_ok = probe.get("returncode") == 0
        self.probe_output = probe.get("output", "")

    def __call__(self, command: str, timeout: int | None = None) -> tuple[str, int]:
        result = self._execute(command, timeout=timeout)
        return result.get("output", ""), int(result.get("returncode", -1))

    def remaining_timeout(self) -> int:
        """Whole seconds left before the orchestrator deadline."""
        if self.deadline is None:
            return self.default_timeout
        return max(0, int(self.deadline - time.monotonic()))

    def clamp_timeout(self, timeout: int | None) -> int:
        try:
            requested = self.default_timeout if timeout is None else int(timeout)
        except (TypeError, ValueError):
            requested = self.default_timeout
        requested = max(1, requested)
        if self.deadline is None:
            return requested
        remaining = self.remaining_timeout()
        if remaining <= 0:
            return 0
        return min(requested, remaining)

    def _execute(self, command: str, *, timeout: int | None = None) -> dict:
        actual_timeout = self.clamp_timeout(timeout)
        if actual_timeout <= 0:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": "Task budget exhausted before command start",
            }

        if not self.exec_user:
            return self.env.execute(command, timeout=actual_timeout)

        cmd = [
            self.env.executable,
            "exec",
            "-w",
            self.env.cwd,
            "-u",
            self.exec_user,
            "-e",
            "HOME=/tmp/pbagent-home",
            self.env.container_id,
            "bash",
            "-lc",
            command,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=actual_timeout,
            )
            return {
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"Command timed out after {actual_timeout}s",
            }
