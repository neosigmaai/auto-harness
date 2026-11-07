"""Run a mutable HarnessAgent inside a task container, using only the stdlib.

This file is copied as bytes by the installed adapter. It is never used to load
candidate code on the worker. The compatibility layer preserves the submitted
baseline's Harbor/litellm interface while keeping dependencies out of task images.
The scoped gateway, rather than this mutable process, accounts for real usage.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import types
import urllib.request


class Object(dict):
    def __getattr__(self, name):
        return self.get(name)


def objects(value):
    if isinstance(value, dict):
        return Object({key: objects(item) for key, item in value.items()})
    if isinstance(value, list):
        return [objects(item) for item in value]
    return value


def event(path: Path, kind: str, **data):
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"event": kind, "time": time.time(), **data}, default=str) + "\n")
        out.flush()


async def read_bounded(stream, limit=131072):
    """Drain all output without letting an infinite command fill host memory."""
    head = bytearray()
    tail = bytearray()
    total = 0
    try:
        while chunk := await stream.read(16384):
            total += len(chunk)
            room = max(0, limit // 2 - len(head))
            head.extend(chunk[:room])
            tail.extend(chunk[room:])
            if len(tail) > limit // 2:
                del tail[:len(tail) - limit // 2]
    except asyncio.CancelledError:
        # A background process may inherit a finished shell's output pipes.
        # Preserve captured output when the drain grace expires.
        pass
    marker = f"\n... [{total - len(head) - len(tail)} bytes omitted] ...\n".encode() if total > len(head) + len(tail) else b""
    return (head + marker + tail).decode("utf-8", errors="replace")


class LocalEnvironment:
    def __init__(self, events: Path, command_timeout: float):
        self.events = events
        self.command_timeout = command_timeout

    @staticmethod
    async def _wait_exit(process):
        # asyncio.Process.wait can wait for inherited output pipes after the
        # shell itself exits. Observe child exit independently of pipe EOF.
        while process.returncode is None:
            await asyncio.sleep(0.01)

    async def exec(self, command: str, timeout_sec=None, cwd=None, env=None, **kwargs):
        timeout = min(float(timeout_sec or self.command_timeout), self.command_timeout)
        child_env = dict(os.environ)
        child_env.update({"DEBIAN_FRONTEND": "noninteractive", "GIT_TERMINAL_PROMPT": "0", "PAGER": "cat"})
        child_env.update(env or {})
        event(self.events, "command_start", command=command[:16000])
        process = await asyncio.create_subprocess_exec(
            "/bin/bash", "--noprofile", "--norc", "-c", command,
            cwd=cwd, env=child_env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout = asyncio.create_task(read_bounded(process.stdout))
        stderr = asyncio.create_task(read_bounded(process.stderr))
        timed_out = False
        try:
            await asyncio.wait_for(self._wait_exit(process), timeout)
        except asyncio.TimeoutError:
            timed_out = True
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._wait_exit(process)
            stdout.cancel()
            stderr.cancel()
            await asyncio.gather(stdout, stderr)
            raise
        finally:
            if timed_out:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self._wait_exit(process)
        # Do not kill deliberately backgrounded services after a successful
        # command. Bound only capture draining when children inherit the pipes;
        # services should redirect their own logs. The trial lifetime still
        # bounds every process in the container.
        _, pending = await asyncio.wait({stdout, stderr}, timeout=0.25)
        if pending:
            for reader in pending:
                reader.cancel()
            for descriptor in (1, 2):
                transport = process._transport.get_pipe_transport(descriptor)
                if transport:
                    transport.close()
        result = Object(stdout=await stdout, stderr=await stderr,
                        return_code=124 if timed_out else process.returncode)
        if timed_out:
            result.stderr += f"\nCommand exceeded {timeout:g}s and its process group was killed."
        event(self.events, "command_end", return_code=result.return_code,
              stdout=result.stdout, stderr=result.stderr)
        return result


def install_compat(config, logs: Path, events: Path):
    class BaseAgent:
        def __init__(self, logs_dir=logs, model_name=None, logger=None, **kwargs):
            self.logs_dir = Path(logs_dir)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.model_name = model_name
            self.logger = logger or logging.getLogger("candidate")

    class AgentContext:
        pass

    calls = 0

    async def acompletion(**payload):
        nonlocal calls
        calls += 1
        if calls > config["max_steps"]:
            raise RuntimeError("Per-trial model call limit reached")
        payload["model"] = config["model"]
        payload["stream"] = False
        payload.pop("api_key", None)
        payload.pop("api_base", None)
        payload.pop("base_url", None)
        payload.setdefault("max_completion_tokens", config["max_tokens"])
        event(events, "model_request", call=calls, messages=payload.get("messages"))

        def request():
            req = urllib.request.Request(
                config["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["token"]},
            )
            # Never route the internal relay request through the public proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=config.get("model_timeout_sec", 100)) as response:
                return json.loads(response.read(2_000_000))

        response = await asyncio.to_thread(request)
        event(events, "model_response", call=calls, response=response)
        return objects(response)

    modules = {
        "harbor": {}, "harbor.agents": {}, "harbor.agents.base": {"BaseAgent": BaseAgent},
        "harbor.environments": {}, "harbor.environments.base": {"BaseEnvironment": LocalEnvironment},
        "harbor.models": {}, "harbor.models.agent": {},
        "harbor.models.agent.context": {"AgentContext": AgentContext},
        "litellm": {"acompletion": acompletion},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    return AgentContext


async def run(config_path: Path):
    config = json.loads(config_path.read_text())
    logs = Path(config.get("logs_dir", "/logs/agent"))
    logs.mkdir(parents=True, exist_ok=True)
    events = logs / "events.jsonl"
    logging.basicConfig(level=logging.INFO)
    context_type = install_compat(config, logs, events)
    os.environ["AGENT_MODEL"] = config["model"]
    spec = importlib.util.spec_from_file_location("candidate", config["candidate_path"])
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)  # Deliberately only executed inside container.
    if hasattr(candidate, "MAX_STEPS"):
        candidate.MAX_STEPS = min(int(candidate.MAX_STEPS), int(config["max_steps"]))
    agent = candidate.HarnessAgent(logs_dir=logs, model_name=config["model"])
    context = context_type()
    environment = LocalEnvironment(events, config.get("command_timeout_sec", 120))
    event(events, "agent_start")
    try:
        if hasattr(agent, "setup"):
            await agent.setup(environment)
        await asyncio.wait_for(agent.run(config["instruction"], environment, context), config["timeout_sec"])
        event(events, "agent_end", context=vars(context))
    except BaseException as exc:
        event(events, "agent_error", error_type=type(exc).__name__, detail=str(exc)[:2000])
        raise
    finally:
        (logs / "context.json").write_text(json.dumps(vars(context), default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config))
