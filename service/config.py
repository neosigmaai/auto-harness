"""Service configuration. Everything comes from environment variables (or .env)."""
from __future__ import annotations

from pydantic import Field, SecretStr
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
    sandbox_provider: str = Field(default="docker", description="docker (implemented); remote providers require a separate adapter")

    # The company credential is the ONLY accepted model credential. Never fall back
    # to OPENAI_API_KEY / OPEN_API_KEY. It stays in the trusted inference process.
    agent_model: str = "gpt-5.4-mini"
    optimizer_model: str = "gpt-5.4-mini"
    neosigma_open_api_key: SecretStr | None = Field(default=None, exclude=True)
    neosigma_base_url: str = "https://api.openai.com/v1"
    inference_bind_host: str = "0.0.0.0"
    inference_advertise_host: str = "host.docker.internal"
    inference_timeout_sec: float = 90.0
    agent_max_output_tokens: int = 2048
    optimizer_max_output_tokens: int = 8192
    max_inference_body_bytes: int = 512_000
    max_model_calls_per_trial: int = 90
    # Standard text-token prices per million, verified 2026-11-07. Cached input
    # is deliberately charged at the full rate: this is an estimate, not an invoice.
    model_prices: dict[str, list[float]] = {
        "gpt-5.4-mini": [0.75, 4.50],
        "gpt-5.4-mini-2026-03-17": [0.75, 4.50],
        "gpt-5.4": [2.50, 15.00],
        "gpt-5.4-2026-03-05": [2.50, 15.00],
    }

    # ── worker tuning ─────────────────────────────────────────────────────────
    worker_id: str | None = None
    lease_ttl_sec: int = 60
    heartbeat_sec: int = 15
    claim_poll_sec: float = 1.0
    worker_concurrency: int = 1          # bounded further by global trial slots
    runs_dir: str = "runs"               # per-job/iteration working dirs for the runner
    simulated_task_delay_sec: float = 0.15

    # ── defaults for job config ───────────────────────────────────────────────
    default_max_iterations: int = 5
    default_patience: int = 3
    max_concurrency_per_iteration: int = 2
    # Two 2-GiB tasks + relays leave headroom on an 8-GiB Docker VM.
    # Increase only after sizing the worker pool and database explicitly.
    global_trial_slots: int = 2
    harbor_executable: str = "harbor"
    benchmark_revision: str = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
    # Opt-in immutable, locally prepared tool images. API freezes the selected
    # manifest entries per job; existing jobs never reread this mutable path.
    prepared_images_manifest: str | None = None

    # ── bootstrap / admin ─────────────────────────────────────────────────────
    bootstrap_admin_token: str | None = Field(
        default=None, description="If set, POST /v1/orgs accepts this token to create organisations."
    )


settings = Settings()
