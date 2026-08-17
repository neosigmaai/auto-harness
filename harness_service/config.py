"""Runtime configuration, loaded from environment / .env.

Field names map to env vars case-insensitively, so ``database_url`` reads
``DATABASE_URL`` and ``openai_api_key`` reads ``OPENAI_API_KEY`` — matching the
repo's existing .env.example keys with no prefix gymnastics.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from harness_service.constants import ExecutorKind


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ── Persistence ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/harness"
    db_echo: bool = False

    # ── LLM / proposer (M4) ──
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"          # model that PROPOSES improvements

    # ── Agent under optimization ──
    agent_model: str = "gpt-5.4"          # model the agent itself runs on
    agent_reasoning_effort: str | None = None

    # ── Sandbox (M3) ──
    e2b_api_key: str | None = None
    env_provider: str = "e2b"

    # ── Executor selection ──
    default_executor: ExecutorKind = ExecutorKind.SIMULATED

    # ── Worker ──
    worker_enabled: bool = True
    worker_poll_interval_s: float = 1.0
    worker_concurrency: int = 2           # max jobs processed in parallel

    # ── Auth (M1–M4 dev default; M5 enforces) ──
    seed_dev_principal: bool = True       # create a default org/user + api key on boot
    dev_api_key: str = "dev-key"


@lru_cache
def get_settings() -> Settings:
    return Settings()
