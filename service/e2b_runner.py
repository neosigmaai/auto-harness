from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

import config
from mock import RunResult

log = logging.getLogger("sandbox")

HOME = "/home/user"          # sandboxes run as `user`; / is not writable
HARNESS = f"{HOME}/harness"
SANDBOX_PATH = f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
REPO = pathlib.Path(__file__).parents[1]


DRIVER = f'''
import glob, json, sys
sys.path.insert(0, "{HARNESS}")
from benchmark import TerminalBenchRunner
cfg = json.load(open("{HARNESS}/_run.json"))
runner = TerminalBenchRunner(
    agent_model=cfg["model"], split="train", env_provider="e2b",
    n_concurrent=cfg["n_concurrent"], per_task_timeout=cfg["per_task_timeout"],
    jobs_dir="{HARNESS}/workspace/tbench_jobs")
json.dump(runner.run(cfg["task_ids"]),
          open("{HARNESS}/workspace/train_results.json", "w"))

tokens = {{"input_tokens": 0, "output_tokens": 0}}   # the runner returns rewards only
for p in glob.glob("{HARNESS}/workspace/tbench_jobs/*/*/result.json"):
    try:
        ar = json.load(open(p)).get("agent_result") or {{}}
    except Exception:
        continue
    tokens["input_tokens"] += ar.get("n_input_tokens") or 0
    tokens["output_tokens"] += ar.get("n_output_tokens") or 0
json.dump(tokens, open("{HARNESS}/workspace/tokens.json", "w"))
'''


class SandboxFailed(Exception):
    pass


class E2BRunner:
    """Same contract as MockRunner."""

    def __init__(self, job_id: str, deadline_seconds: int, on_sandbox: Any = None):
        self.job_id = job_id
        self.deadline_seconds = deadline_seconds
        self.on_sandbox = on_sandbox
        self.sandbox: Any = None
        self.sandbox_id: str | None = None
        self._provisioned = False
        self._outer_counted = False
        self._created_at = 0.0

    def _log(self, msg: str) -> None:
        log.info("job=%s id=%s %s", self.job_id, self.sandbox_id or "-", msg)

    def _ensure(self) -> None:
        if self._provisioned:
            self.sandbox.set_timeout(self.deadline_seconds + config.LEASE_SLACK_SECONDS)
            return
        self.close()      

        from e2b import Sandbox

        t = time.time()
        self.sandbox = Sandbox.create(         
            timeout=self.deadline_seconds + config.LEASE_SLACK_SECONDS,
            api_key=config.E2B_API_KEY)
        self.sandbox_id = self.sandbox.sandbox_id
        self._created_at = time.time()
        self._log(f"created in {time.time() - t:.1f}s")
        if self.on_sandbox:
            self.on_sandbox(self.sandbox_id)

        # base image is py3.11, harbor needs 3.12. [e2b] is not optional — without it
        # --env e2b raises MissingExtraError.
        t = time.time()
        self._sh("curl -LsSf https://astral.sh/uv/install.sh | sh", 300)
        self._sh("uv tool install 'harbor[e2b]' --python 3.12", 900)
        version = self._sh("harbor --version", 120).strip()
        self._log(f"provisioned {version or 'harbor'} in {time.time() - t:.1f}s")

        self._sh(f"mkdir -p {HARNESS}/agent")
        self.sandbox.files.write(f"{HARNESS}/benchmark.py",
                                 (REPO / "benchmark.py").read_text())
        self._provisioned = True

    def _sh(self, cmd: str, timeout: int = 120) -> str:
        r = self.sandbox.commands.run(
            cmd, timeout=timeout, request_timeout=timeout + 60,
            envs={"E2B_API_KEY": config.E2B_API_KEY,
                  "OPENAI_API_KEY": config.PLATFORM_OPENAI_KEY,
                  "AGENT_MODEL": config.AGENT_MODEL, "HOME": HOME,
                  "PATH": SANDBOX_PATH})
        if r.exit_code != 0:
            raise SandboxFailed(f"`{cmd[:60]}` exited {r.exit_code}: "
                                f"{(r.stderr or r.stdout or '')[-500:]}")
        return r.stdout or ""

    def close(self) -> None:
        if self.sandbox is None:
            return
        alive = time.time() - self._created_at
        try:
            self.sandbox.kill()
            self._log(f"killed after {alive:.0f}s")
        except Exception as e:                                       # noqa: BLE001
            self._log(f"kill failed after {alive:.0f}s: {e}")
        finally:
            self.sandbox = None

    def run(self, task_ids: list[str], agent_source: str) -> RunResult:
        started = time.time()
        try:
            self._ensure()

            self._sh(f"rm -rf {HARNESS}/workspace && mkdir -p {HARNESS}/workspace", 60)
            self.sandbox.files.write(f"{HARNESS}/agent/agent.py", agent_source)
            self.sandbox.files.write(f"{HARNESS}/_run.py", DRIVER)
            self.sandbox.files.write(f"{HARNESS}/_run.json", json.dumps({
                "task_ids": task_ids, "model": config.AGENT_MODEL,
                "n_concurrent": config.N_CONCURRENT,
                "per_task_timeout": config.PER_TASK_TIMEOUT}))

            self._log(f"running {len(task_ids)} tasks, "
                      f"{min(config.N_CONCURRENT, len(task_ids))} at a time")
            self._sh(f"cd {HARNESS} && python3 _run.py", self.deadline_seconds)

            results = json.loads(
                self.sandbox.files.read(f"{HARNESS}/workspace/train_results.json"))
            failures = [self._failure_record(t, r)
                        for t, r in sorted(results.items()) if (r or 0.0) < 1.0]
            try:
                tokens = json.loads(
                    self.sandbox.files.read(f"{HARNESS}/workspace/tokens.json"))
            except Exception:                                        # noqa: BLE001
                tokens = {}
        except Exception as e:    
            self._log(f"run failed: {type(e).__name__}: {e}")
            return RunResult(
                results={t: None for t in task_ids}, failures=[],
                usage=self._usage(task_ids, started),
                error_detail=f"{type(e).__name__}: {str(e)[:500]}")

        return RunResult(results=results, failures=failures,
                         usage=self._usage(task_ids, started, tokens))

    def _usage(self, task_ids: list[str], started: float,
               tokens: dict[str, int] | None = None) -> dict[str, int]:
        outer = 0 if self._outer_counted else 1      # created once per job
        self._outer_counted = True
        t = tokens or {}
        return {"sandboxes_used": outer + len(task_ids),
                "sandbox_seconds": int(time.time() - started),
                "input_tokens": t.get("input_tokens", 0),
                "output_tokens": t.get("output_tokens", 0)}

    def _failure_record(self, task_id: str, reward: float | None) -> dict[str, Any]:
        try:
            messages = json.loads(self.sandbox.files.read(
                f"{HARNESS}/workspace/traces/latest/{task_id}/trace.json"))
        except Exception:                                            # noqa: BLE001
            return {"task_id": task_id, "reward": reward, "tool_calls": 0,
                    "failing_commands": [], "tail": "",
                    "verifier_output": "(no trace written)"}
        return distil(task_id, reward, messages)


def distil(task_id: str, reward: float | None,
           messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw traces are tens of MB and never enter Postgres. Lossy on purpose."""
    commands: dict[str, str] = {}
    failing: list[dict[str, Any]] = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            try:
                commands[tc["id"]] = json.loads(
                    tc["function"]["arguments"]).get("command", "")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if m.get("role") == "tool" and "[exit code:" in str(m.get("content", "")):
            content = str(m["content"])
            code = content.rsplit("[exit code:", 1)[-1].split("]")[0].strip()
            
            if code not in ("0", ""):
                failing.append({
                    "command": commands.get(m.get("tool_call_id", ""), "")[:500],
                    "exit_code": code, "stderr": content[-800:]})

    tail = "\n".join(f"{m.get('role')}: {str(m.get('content') or '')[:400]}"
                     for m in messages[-8:] if m.get("role") != "system")
    return {"task_id": task_id, "reward": reward,
            "tool_calls": len(commands),      # 0 => never acted; the canary reads this
            "failing_commands": failing[-8:], "tail": tail[-4000:],
            "verifier_output": "" if reward is None else f"reward {reward}"}


def demo() -> None:
    call = lambda i, cmd: {  # noqa: E731
        "role": "assistant", "content": None,
        "tool_calls": [{"id": i, "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": cmd})}}]}

    d = distil("regex-log", 0.0, [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "fix the thing"},
        call("c1", "python solve.py"),
        {"role": "tool", "tool_call_id": "c1", "content": "KeyError\n[exit code: 1]"},
        call("c2", "ls -la"),
        {"role": "tool", "tool_call_id": "c2", "content": "file.txt"},
        {"role": "assistant", "content": "I give up"},
    ])
    assert d["tool_calls"] == 2
    assert [c["command"] for c in d["failing_commands"]] == ["python solve.py"]
    assert d["failing_commands"][0]["exit_code"] == "1"
    assert "KeyError" in d["failing_commands"][0]["stderr"]
    assert "I give up" in d["tail"] and "you are an agent" not in d["tail"]

    # printing the marker is not failing
    assert distil("t", 0.0, [{"role": "tool", "tool_call_id": "z",
                              "content": "ok\n[exit code: 0]"}])["failing_commands"] == []
    # malformed tool_calls must not take the run down
    assert distil("t", 0.0, [{"role": "assistant",
                              "tool_calls": [{"id": "x", "function": {}}]}]) is not None
    assert distil("t", None, [])["failing_commands"] == []
    assert "per_task_timeout" in DRIVER
    print("ok — trace distillation, malformed traces, driver script")


if __name__ == "__main__":
    demo()
