"""SWE-Bench patch generation — wraps mini-swe-agent for a proper agentic loop."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SWEInstanceContext:
    """One SWE-Bench instance as seen by the agent (runner fills this from the dataset row)."""

    instance_id: str
    problem_statement: str
    repo: str | None = None
    workspace_root: str | None = None
    use_llm: bool = True
    fail_to_pass: list[str] | None = None   # test IDs that must go from failing → passing
    hints_text: str | None = None           # issue comment thread (may contain repro/hints)


def _agent_model() -> str:
    """
    Resolve the LLM model name.
    Priority: AGENT_MODEL env var → experiment_config.yaml agent_model → 'gpt-4o'.
    """
    if model := os.environ.get("AGENT_MODEL"):
        return model
    try:
        import yaml

        with open("experiment_config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        if model := cfg.get("agent_model"):
            return str(model)
    except Exception:
        pass
    return "gpt-4o"


def _looks_like_diff(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.startswith("diff --git ") or s.startswith("--- ") or s.startswith("+++ "):
        return True
    return bool(re.search(r"^@@ ", s, re.MULTILINE))


# ---------------------------------------------------------------------------
# mini-swe-agent wrapper
# ---------------------------------------------------------------------------

def _build_task(context: SWEInstanceContext) -> str:
    """
    Build the task string passed to the agent.

    Appends FAIL_TO_PASS test IDs with a concise workflow so the agent
    runs the failing tests first to observe the exact error, then fixes
    only what the test requires, and verifies before submitting.
    """
    parts = [context.problem_statement.strip()]

    if context.hints_text and context.hints_text.strip():
        parts.append(
            "\n<hints>\n"
            "The following comments from the issue thread may help:\n"
            + context.hints_text.strip()
            + "\n</hints>"
        )

    if context.fail_to_pass:
        tests_joined = " ".join(context.fail_to_pass)
        tests_list = "\n".join(f"  {t}" for t in context.fail_to_pass)
        parts.append(
            "\n<tests_to_fix>\n"
            "The following tests are currently FAILING. Make them PASS.\n\n"
            "IMPORTANT: Start by running the test to see the exact error:\n"
            f"  cd /testbed && python -m pytest {tests_joined} -x --tb=long 2>&1 | tail -60\n\n"
            "Read the assertion error output carefully:\n"
            "  - For an AssertionError with string comparison, the 'expected' vs 'actual'\n"
            "    values show EXACTLY what text needs to change in the source code.\n"
            "    Use grep to find that exact string in the source and change it.\n"
            "  - For a logic failure, trace the code path from the test to find the bug.\n\n"
            "After making a fix, verify it passes:\n"
            f"  cd /testbed && python -m pytest {tests_joined} -x --tb=short 2>&1 | tail -20\n\n"
            "Also verify no regressions:\n"
            "  cd /testbed && python -m pytest <test_file_for_modified_source> -x -q 2>&1 | tail -20\n\n"
            "Failing tests:\n"
            + tests_list
            + "\n</tests_to_fix>"
        )

    return "\n".join(parts)


def _swebench_image(instance_id: str) -> str:
    """Derive the official SWE-bench Docker image name from an instance_id."""
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


def _generate_patch_mini_swe(context: SWEInstanceContext) -> str:
    """
    Run mini-swe-agent's DefaultAgent inside the official SWE-bench Docker image
    for this instance (Docker-in-Docker via mounted socket).

    The image has the repo pre-installed at /testbed with all test dependencies,
    matching exactly what the harness evaluator expects.
    Submission is captured from result["submission"].
    """
    try:
        import yaml
        from importlib.resources import files as pkg_files

        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments import get_environment
        from minisweagent.models.litellm_model import LitellmModel
    except ImportError as exc:
        print(f"[swe_agent] minisweagent not installed ({exc}) — returning empty patch")
        return ""

    try:
        # 1. Load swebench.yaml config as-is (/testbed is correct for the Docker image)
        config_text = (
            pkg_files("minisweagent.config")
            .joinpath("benchmarks", "swebench.yaml")
            .read_text()
        )
        cfg = yaml.safe_load(config_text)
        agent_cfg = cfg["agent"]
        model_cfg = cfg.get("model", {})
        env_cfg = dict(cfg.get("environment", {}))

        # 2. Point environment at the instance-specific SWE-bench image
        image = _swebench_image(context.instance_id)
        print(f"[swe_agent] using Docker image: {image}")
        env_cfg["environment_class"] = "docker"
        env_cfg["image"] = image

        env = get_environment(env_cfg)

        # 3. Build model
        model_name = _agent_model()
        model_kwargs = dict(model_cfg.get("model_kwargs", {}))
        model = LitellmModel(
            model_name=model_name,
            model_kwargs=model_kwargs,
            observation_template=model_cfg.get("observation_template", "{{output.output}}"),
            format_error_template=model_cfg.get("format_error_template", "{{error}}"),
            cost_tracking="ignore_errors",
        )

        # 4. Build agent — use full swebench.yaml settings (step_limit=250, cost_limit=3.0)
        #    Allow env var overrides for experimentation
        agent_init = dict(agent_cfg)
        if sl := os.environ.get("SWE_STEP_LIMIT"):
            agent_init["step_limit"] = int(sl)
        if cl := os.environ.get("SWE_COST_LIMIT"):
            agent_init["cost_limit"] = float(cl)
        agent = DefaultAgent(model=model, env=env, **agent_init)

        # 5. Run the agent with enriched task (problem + hints + fail_to_pass tests)
        task = _build_task(context)
        print(f"[swe_agent] running mini-swe-agent on {context.instance_id} ...")
        result = agent.run(task)

        exit_status = result.get("exit_status", "")
        # Strip only leading whitespace; preserve the trailing newline that
        # `git diff` emits — removing it produces "patch unexpectedly ends in
        # middle of line" errors in the SWE-bench harness.
        submission = (result.get("submission") or "").lstrip()
        # Guarantee the patch ends with a newline (required by `git apply`).
        if submission and not submission.endswith("\n"):
            submission += "\n"
        print(f"[swe_agent] exit_status={exit_status}  submission_len={len(submission)}")

        if submission and _looks_like_diff(submission):
            return submission

        print(f"[swe_agent] no valid patch produced for {context.instance_id}")
        return ""

    except Exception as exc:
        print(f"[swe_agent] unexpected error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_patch(context: SWEInstanceContext) -> str:
    """
    Return a unified diff string for `model_patch` in official predictions JSONL.

    Priority:
    1. ``SWE_STUB_PATCH`` — fixed patch (testing).
    2. If ``context.use_llm`` is False — empty string.
    3. If ``SWE_DISABLE_LLM`` is set — empty string.
    4. Otherwise — mini-swe-agent agentic loop on a fresh repo clone.
    """
    if os.environ.get("SWE_STUB_PATCH") is not None:
        return os.environ["SWE_STUB_PATCH"]
    if not context.use_llm:
        return ""
    if os.environ.get("SWE_DISABLE_LLM", "").strip().lower() in ("1", "true", "yes"):
        return ""
    return _generate_patch_mini_swe(context)


def model_name_for_predictions() -> str:
    """Value for ``model_name_or_path`` in predictions JSONL."""
    return os.environ.get("AGENT_MODEL", "auto-harness-stub")
