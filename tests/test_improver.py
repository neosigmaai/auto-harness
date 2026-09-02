"""Tests for FakeImprover, LLMImprover and create_improver (no network)."""

from __future__ import annotations

import json

import pytest

from api.agent_spec import AgentSpec
from api.config import BenchmarkConfig
from api.job_store import IterationRecord
from api.services import improver as improver_mod
from api.services.improver import (
    EvaluationSummary,
    FakeImprover,
    ImproverError,
    LLMImprover,
    Proposal,
    TaskOutcome,
    _ALLOWED_CONFIG_KEYS,
    _RaisingImprover,
    create_improver,
)


def _spec(system_prompt: str = "BASE PROMPT") -> AgentSpec:
    return AgentSpec(system_prompt=system_prompt, agent_model="gpt-4.1-mini", max_steps=80)


def _evaluation() -> EvaluationSummary:
    return EvaluationSummary(
        score=0.5,
        tasks=[
            TaskOutcome(task_id="t-pass", status="passed", reward=1.0, remarks=None),
            TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed"),
        ],
        traces={"t-zero": json.dumps([{"role": "tool", "content": "boom"}])},
    )


def _history() -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=0,
            agent_version_id="00000000-0000-0000-0000-000000000000",
            version=0,
            run_id="11111111-1111-1111-1111-111111111111",
            score=0.5,
            improved=True,
            rationale="baseline",
            changed_fields=[],
            status="completed",
        )
    ]


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _StubLitellm:
    """Stands in for the litellm module attribute; records every call."""

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def completion(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _Response(self.payloads[index])


def _payload(**overrides) -> str:  # noqa: ANN003
    body = {
        "system_prompt": "IMPROVED PROMPT",
        "config_changes": {"max_steps": 120},
        "rationale": "Added a verification step",
    }
    body.update(overrides)
    return json.dumps(body)


def test_fake_improver_returns_scripted_proposals_in_order() -> None:
    first = Proposal(spec=_spec("FIRST"), rationale="first")
    second = Proposal(spec=_spec("SECOND"), rationale="second")
    fake = FakeImprover([first, second])

    got_first = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())
    got_second = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert got_first is first
    assert got_second is second
    assert fake.calls == 2


def test_fake_improver_cycles_deterministic_revision_when_exhausted() -> None:
    fake = FakeImprover()

    first = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())
    second = fake.propose(spec=first.spec, evaluation=_evaluation(), history=_history())

    assert first.spec.system_prompt == "BASE PROMPT\n\n[fake-improver revision 1]"
    assert first.rationale == "fake improver deterministic revision 1"
    assert second.spec.system_prompt.endswith("[fake-improver revision 2]")
    # Exhaustion is not an error: FakeImprover keeps producing valid proposals.
    assert first.spec.max_steps == 80


def test_fake_improver_never_raises_when_fed_its_own_output_many_times() -> None:
    """Task 13 depends on FakeImprover never raising. Feeding a proposal's own
    spec back in is exactly what the mock end-to-end loop does every
    iteration; prove the invariant holds for far more than a couple of calls,
    not just the first two (system_prompt must not grow toward AgentSpec's
    20_000-char cap)."""
    fake = FakeImprover()
    spec = _spec()

    for _ in range(1_000):
        proposal = fake.propose(spec=spec, evaluation=_evaluation(), history=_history())
        spec = proposal.spec

    assert spec.system_prompt.endswith("[fake-improver revision 1000]")
    assert len(spec.system_prompt) < 200


def test_fake_improver_applies_mutate_callable() -> None:
    fake = FakeImprover(mutate=lambda spec: spec.model_copy(update={"max_steps": 150}))

    proposal = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.max_steps == 150
    assert proposal.spec.system_prompt == "BASE PROMPT"


def test_llm_improver_merges_config_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm([_payload()])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "IMPROVED PROMPT"
    assert proposal.spec.max_steps == 120
    # Untouched fields are carried over from the base spec.
    assert proposal.spec.agent_model == "gpt-4.1-mini"
    assert proposal.spec.exec_timeout_sec == 120
    assert proposal.rationale == "Added a verification step"
    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == "gpt-5.4"
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    assert "BASE PROMPT" in stub.calls[0]["messages"][-1]["content"]
    assert llm.last_response == _payload()


def test_llm_improver_retries_once_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(["this is not json at all", _payload()])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "IMPROVED PROMPT"
    assert len(stub.calls) == 2
    retry_text = stub.calls[1]["messages"][-1]["content"]
    assert "not valid JSON" in retry_text


def test_llm_improver_retries_once_on_empty_first_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/malformed response body gets the same single retry as an
    invalid-JSON or validation failure - _extract_content must route through
    _ProposalRejected rather than raising ImproverError directly, or this
    retry would never happen."""

    class _EmptyResponse:
        choices: list = []

    class _StubSequence:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def completion(self, **kwargs):  # noqa: ANN003, ANN201
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _EmptyResponse()
            return _Response(_payload())

    stub = _StubSequence()
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "IMPROVED PROMPT"
    assert len(stub.calls) == 2
    # M13: empty first response must not inject an empty assistant turn on retry.
    assert all(
        m.get("content")
        for m in stub.calls[1]["messages"]
        if m["role"] == "assistant"
    )


def test_llm_improver_raises_after_two_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(["nope", "still nope"])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    with pytest.raises(ImproverError) as excinfo:
        llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert "invalid proposal twice" in str(excinfo.value)
    assert len(stub.calls) == 2


def test_llm_improver_rejects_out_of_bounds_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(
        [
            _payload(config_changes={"max_steps": 9999}),
            _payload(config_changes={"max_steps": 120}),
        ]
    )
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.max_steps == 120
    assert len(stub.calls) == 2
    retry_text = stub.calls[1]["messages"][-1]["content"]
    assert "AgentSpec validation" in retry_text
    assert "max_steps" in retry_text


def test_llm_improver_rejects_unknown_config_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(
        [
            _payload(config_changes={"agent_model": "gpt-4o", "tools": ["python"]}),
            _payload(),
        ]
    )
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.agent_model == "gpt-4.1-mini"
    assert "unsupported keys" in stub.calls[1]["messages"][-1]["content"]


def test_llm_improver_wraps_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def completion(self, **kwargs):  # noqa: ANN003, ANN201
            raise RuntimeError("connection reset")

    monkeypatch.setattr(improver_mod, "litellm", _Boom())

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    with pytest.raises(ImproverError) as excinfo:
        llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert "connection reset" in str(excinfo.value)


def _config(backend: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        execution_backend=backend,
    )


def test_create_improver_returns_fake_for_mock_backend() -> None:
    assert isinstance(create_improver(_config("mock")), FakeImprover)


def test_create_improver_returns_llm_for_harbor_backend() -> None:
    improver = create_improver(_config("harbor"), improver_model="gpt-5.4-mini")

    assert isinstance(improver, LLMImprover)
    assert improver.model == "gpt-5.4-mini"
    assert improver.budget == 60_000


def test_raising_improver_always_raises_improver_error() -> None:
    """The failure-path stub Task 10/13 use instead of an 'exhausted' FakeImprover."""
    raiser = _RaisingImprover()

    with pytest.raises(ImproverError):
        raiser.propose(spec=_spec(), evaluation=_evaluation(), history=_history())


def test_allowed_config_keys_match_agent_spec_mutable_fields() -> None:
    """M7: improver allowlist must track AgentSpec (minus prompt/model)."""
    expected = set(AgentSpec.model_fields) - {"system_prompt", "agent_model"}
    assert _ALLOWED_CONFIG_KEYS == expected
    from api.services.improver import IMPROVER_SYSTEM_PROMPT, _allowed_config_prompt_text

    assert _allowed_config_prompt_text() in IMPROVER_SYSTEM_PROMPT


def test_llm_improver_rejects_oversized_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.services.improver import _MAX_SYSTEM_PROMPT_CHARS

    too_long = "X" * (_MAX_SYSTEM_PROMPT_CHARS + 10)
    stub = _StubLitellm(
        [
            _payload(system_prompt=too_long),
            _payload(system_prompt="short enough"),
        ]
    )
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "short enough"
    assert "soft cap" in stub.calls[1]["messages"][-1]["content"]


class _StubImprover:
    """Records nothing; only exists so identity/attribute checks are cheap."""

    def __init__(self, model: str) -> None:
        self.model = model

    def propose(self, **kwargs):  # noqa: ANN003, ANN201
        raise AssertionError("propose() should not be called by this test")


def _step_record(*, improver_model: str):
    """A minimal StepRecord for an improve step (no DB - a plain dataclass)."""
    from api.job_store import StepRecord

    return StepRecord(
        step_id="s1",
        job_id="j1",
        type="improve",
        iteration=0,
        agent_version_id="v1",
        version=0,
        spec=_spec(),
        task_ids=["fix-git"],
        agent_model="gpt-4.1-mini",
        improver_model=improver_model,
        run_id=None,
        stale_after_sec=1800,
    )


def test_improve_step_honors_per_job_improver_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """I2 (final review): CreateJobRequest.improver_model is validated, stored
    and echoed by the API, but was never actually used to build the improver
    that runs the improve step - every LLM call used the config default
    regardless of what the client asked for. StepExecutor._improver_for_step
    is the fix: it must call the existing create_improver(..., improver_model=)
    seam with the step's model whenever that step's job requested a
    non-default one, and use the improver that seam returns (not the
    executor's default).
    """
    from worker.steps import StepExecutor

    captured: dict[str, object] = {}

    def _fake_create_improver(config, *, improver_model=None):  # noqa: ANN001, ANN201
        captured["config"] = config
        captured["improver_model"] = improver_model
        return _StubImprover(improver_model)

    monkeypatch.setattr("worker.steps.create_improver", _fake_create_improver)

    cfg = _config("harbor")  # non-mock: the override must take effect here
    default_improver = FakeImprover()
    executor = StepExecutor(
        job_store=None,
        run_store=None,
        config=cfg,
        improver=default_improver,
        artifacts=None,
    )
    step = _step_record(improver_model="gpt-5.4-mini")
    assert step.improver_model != cfg.improver_model  # sanity: a real override

    resolved = executor._improver_for_step(step)

    assert captured["improver_model"] == "gpt-5.4-mini"
    assert captured["config"] is cfg
    assert isinstance(resolved, _StubImprover)
    assert resolved.model == "gpt-5.4-mini"
    assert resolved is not default_improver


def test_improve_step_keeps_default_improver_when_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """No per-job override (improver_model == the config default) must reuse
    the executor's default improver rather than building a fresh one on every
    step."""
    from worker.steps import StepExecutor

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("create_improver must not be called with no override")

    monkeypatch.setattr("worker.steps.create_improver", _boom)

    cfg = _config("harbor")
    default_improver = FakeImprover()
    executor = StepExecutor(
        job_store=None,
        run_store=None,
        config=cfg,
        improver=default_improver,
        artifacts=None,
    )
    step = _step_record(improver_model=cfg.improver_model)

    resolved = executor._improver_for_step(step)
    assert resolved is default_improver


def test_improve_step_always_uses_default_improver_on_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mock backend has no real notion of "model", so it must keep using
    whatever improver double the executor/test injected - even when the
    step's job set an improver_model override - rather than silently
    replacing a test's injected _RaisingImprover/FakeImprover with a fresh
    FakeImprover() from create_improver()."""
    from worker.steps import StepExecutor

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("create_improver must not be called on the mock backend")

    monkeypatch.setattr("worker.steps.create_improver", _boom)

    cfg = _config("mock")
    default_improver = _RaisingImprover()
    executor = StepExecutor(
        job_store=None,
        run_store=None,
        config=cfg,
        improver=default_improver,
        artifacts=None,
    )
    step = _step_record(improver_model="gpt-5.4-mini")  # differs from cfg default

    resolved = executor._improver_for_step(step)
    assert resolved is default_improver
