from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_DATABASE_URL = "postgresql://autoharness:autoharness@localhost:5432/autoharness"


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str
    default_model: str
    default_sandbox_provider: str
    default_mode: str
    max_local_concurrency: int
    poll_interval_sec: float


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def load_settings() -> ServiceSettings:
    return ServiceSettings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        default_model=os.getenv("AGENT_MODEL", "gpt-5.4"),
        default_sandbox_provider=os.getenv("AUTOHARNESS_SANDBOX_PROVIDER", "daytona"),
        default_mode=os.getenv("AUTOHARNESS_SERVICE_MODE", "simulated"),
        max_local_concurrency=_int_from_env("AUTOHARNESS_MAX_LOCAL_CONCURRENCY", 4),
        poll_interval_sec=_float_from_env("AUTOHARNESS_POLL_INTERVAL_SEC", 2.0),
    )
