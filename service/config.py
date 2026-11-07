"""Service configuration. Everything comes from environment variables (or .env)."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── persistence ───────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://harness:harness@localhost:5433/harness",
        description="asyncpg DSN. Postgres is the queue and the state store.",
    )
    db_pool_min: int = 1
    db_pool_max: int = 10

    # ── executor / optimizer selection ───────────────────────────────────────
    executor: str = Field(default="simulated", description="simulated | harbor")
    optimizer: str = Field(default="simulated", description="simulated | llm")
    sandbox_provider: str = Field(default="e2b", description="e2b | docker | modal (harbor --env)")

    # ── models / keys (only forwarded to the runner subprocess, never to sandboxes) ──
    agent_model: str = "gpt-5.4"
    optimizer_model: str = "gpt-5.4"
    openai_api_key: str | None = None
    e2b_api_key: str | None = None

    # ── worker tuning ─────────────────────────────────────────────────────────
    worker_id: str | None = None
    lease_ttl_sec: int = 60
    heartbeat_sec: int = 15
    claim_poll_sec: float = 1.0
    worker_concurrency: int = 2          # jobs a single worker process runs at once
    runs_dir: str = "runs"               # per-job/iteration working dirs for the runner
    simulated_task_delay_sec: float = 0.15

    # ── defaults for job config ───────────────────────────────────────────────
    default_max_iterations: int = 5
    default_patience: int = 3
    max_concurrency_per_iteration: int = 16

    # ── bootstrap / admin ─────────────────────────────────────────────────────
    bootstrap_admin_token: str | None = Field(
        default=None, description="If set, POST /v1/orgs accepts this token to create organisations."
    )


settings = Settings()
