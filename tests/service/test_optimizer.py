from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_reward_result,
)
from autoharness_service.optimizer import Optimizer, build_optimizer_prompt


def test_build_optimizer_prompt_contains_failed_tasks():
    results = [
        normalize_reward_result("task-pass", 1.0),
        normalize_reward_result("task-fail", 0.0),
    ]
    summary = build_failure_summary(results)

    prompt = build_optimizer_prompt(results, summary)

    assert "task-fail" in prompt
    assert "agent_failed" in prompt
    assert "one focused improvement" in prompt


def test_optimizer_propose_skips_without_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    results = [normalize_reward_result("task-fail", 0.0)]
    summary = build_failure_summary(results)

    proposal = Optimizer().propose(results, summary, model="test-model")

    assert "OPENAI_API_KEY" in proposal
    assert "skipped" in proposal
