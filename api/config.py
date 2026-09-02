"""Load benchmark API configuration from config/benchmark.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "benchmark.yaml"

KNOWN_ENV_PROVIDERS = frozenset({"e2b", "daytona", "modal", "docker"})
KNOWN_EXECUTION_BACKENDS = frozenset({"harbor", "mock"})


@dataclass(frozen=True)
class BenchmarkConfig:
    default_task_ids: list[str]
    default_agent_model: str
    env_provider: str = "docker"
    dataset: str = "terminal-bench@2.0"
    max_concurrency: int = 2
    per_task_timeout: int = 1200
    execution_backend: str = "harbor"
    jobs_dir: str = "workspace/tbench_jobs"

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
    env_provider = str(raw.get("env_provider") or "docker").lower()
    if env_provider not in KNOWN_ENV_PROVIDERS:
        raise ValueError(
            f"Unknown env_provider {env_provider!r}; expected one of {sorted(KNOWN_ENV_PROVIDERS)}"
        )

    backend = (
        os.environ.get("EXECUTION_BACKEND")
        or raw.get("execution_backend")
        or "harbor"
    )
    backend = str(backend).lower()
    if backend not in KNOWN_EXECUTION_BACKENDS:
        raise ValueError(
            f"Unknown execution_backend {backend!r}; expected one of {sorted(KNOWN_EXECUTION_BACKENDS)}"
        )

    return BenchmarkConfig(
        default_task_ids=[str(t) for t in task_ids],
        default_agent_model=str(model),
        env_provider=env_provider,
        dataset=str(raw.get("dataset") or "terminal-bench@2.0"),
        max_concurrency=int(raw.get("max_concurrency") or 2),
        per_task_timeout=int(raw.get("per_task_timeout") or 1200),
        execution_backend=backend,
        jobs_dir=str(raw.get("jobs_dir") or "workspace/tbench_jobs"),
    )


def clear_config_cache() -> None:
    load_config.cache_clear()
