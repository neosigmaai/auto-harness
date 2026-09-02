"""Load benchmark API configuration from config/benchmark.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "benchmark.yaml"


@dataclass(frozen=True)
class BenchmarkConfig:
    default_task_ids: list[str]
    default_agent_model: str

    @property
    def known_task_ids(self) -> frozenset[str]:
        """Allowlist of task IDs accepted by POST /v1/runs."""
        return frozenset(self.default_task_ids)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> BenchmarkConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    task_ids = raw.get("default_task_ids") or []
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"{config_path} must define a non-empty default_task_ids list")

    model = raw.get("default_agent_model") or "gpt-4.1-mini"
    return BenchmarkConfig(
        default_task_ids=[str(t) for t in task_ids],
        default_agent_model=str(model),
    )


def clear_config_cache() -> None:
    load_config.cache_clear()
