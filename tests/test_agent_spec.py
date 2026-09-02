"""Unit tests for AgentSpec — the mutable surface the improver edits (no DB, no harbor)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.agent_spec import (
    BASELINE_SYSTEM_PROMPT,
    AgentSpec,
    baseline_spec,
    changed_fields,
)
from api.config import REPO_ROOT


def _template_agent_instruction() -> str:
    """Extract AGENT_INSTRUCTION from the template without importing it (needs harbor)."""
    source = (REPO_ROOT / "agent" / "templates" / "terminal_bench.py").read_text(
        encoding="utf-8"
    )
    marker = 'AGENT_INSTRUCTION = """\\\n'
    start = source.index(marker) + len(marker)
    end = source.index('"""', start)
    return source[start:end]


def test_baseline_prompt_is_verbatim_copy_of_template() -> None:
    assert BASELINE_SYSTEM_PROMPT == _template_agent_instruction()


def test_baseline_spec_uses_given_model_and_template_defaults() -> None:
    spec = baseline_spec("gpt-4.1-mini")
    assert spec.agent_model == "gpt-4.1-mini"
    assert spec.system_prompt == BASELINE_SYSTEM_PROMPT
    assert spec.max_steps == 80
    assert spec.max_output_chars == 8000
    assert spec.exec_timeout_sec == 120


def test_valid_spec_round_trips_through_json() -> None:
    spec = AgentSpec(
        system_prompt="do the thing",
        agent_model="claude-sonnet-4",
        max_steps=12,
        max_output_chars=999,
        exec_timeout_sec=45,
    )
    restored = AgentSpec.model_validate(spec.model_dump())
    assert restored == spec
    assert set(spec.model_dump()) == {
        "system_prompt",
        "agent_model",
        "max_steps",
        "max_output_chars",
        "exec_timeout_sec",
    }


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSpec(
            system_prompt="p",
            agent_model="m",
            temperature=0.7,  # not part of the mutable surface
        )


@pytest.mark.parametrize("bad_steps", [0, -1, 201, 10_000])
def test_max_steps_out_of_bounds_is_rejected(bad_steps: int) -> None:
    with pytest.raises(ValidationError):
        AgentSpec(system_prompt="p", agent_model="m", max_steps=bad_steps)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("system_prompt", ""),
        ("system_prompt", "x" * 20_001),
        ("agent_model", ""),
        ("agent_model", "m" * 257),
        ("max_output_chars", 499),
        ("max_output_chars", 100_001),
        ("exec_timeout_sec", 9),
        ("exec_timeout_sec", 1201),
    ],
)
def test_field_bounds_are_enforced(field: str, bad_value: object) -> None:
    kwargs: dict[str, object] = {"system_prompt": "p", "agent_model": "m"}
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        AgentSpec(**kwargs)


@pytest.mark.parametrize(
    "bounds_ok",
    [
        {"max_steps": 1},
        {"max_steps": 200},
        {"max_output_chars": 500},
        {"max_output_chars": 100_000},
        {"exec_timeout_sec": 10},
        {"exec_timeout_sec": 1200},
    ],
)
def test_field_bounds_are_inclusive(bounds_ok: dict) -> None:
    spec = AgentSpec(system_prompt="p", agent_model="m", **bounds_ok)
    for key, value in bounds_ok.items():
        assert getattr(spec, key) == value


def test_changed_fields_detects_exactly_the_differing_fields() -> None:
    old = baseline_spec("gpt-4.1-mini")
    new = old.model_copy(update={"system_prompt": "new prompt", "max_steps": 120})
    assert changed_fields(old, new) == ["max_steps", "system_prompt"]


def test_changed_fields_is_empty_for_identical_specs() -> None:
    old = baseline_spec("gpt-4.1-mini")
    assert changed_fields(old, old.model_copy()) == []


def test_changed_fields_is_sorted_and_covers_every_field() -> None:
    old = baseline_spec("a")
    new = AgentSpec(
        system_prompt="different",
        agent_model="b",
        max_steps=1,
        max_output_chars=500,
        exec_timeout_sec=10,
    )
    assert changed_fields(old, new) == [
        "agent_model",
        "exec_timeout_sec",
        "max_output_chars",
        "max_steps",
        "system_prompt",
    ]
