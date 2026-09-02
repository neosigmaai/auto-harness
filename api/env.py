"""Load repo-root ``.env`` into ``os.environ`` without clobbering exports."""

from __future__ import annotations

from pathlib import Path

from api.config import REPO_ROOT

_DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_repo_dotenv(*, path: Path | None = None) -> bool:
    """Load key=value pairs from the repo ``.env`` if it exists.

    Existing process environment variables are never overwritten, so an
    explicitly exported ``E2B_API_KEY`` always wins over the file. Returns
    ``True`` when a file was found and parsed (even if empty).
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "python-dotenv is required to load .env (pip install python-dotenv)"
        ) from exc

    env_path = path if path is not None else _DEFAULT_ENV_PATH
    if not env_path.is_file():
        return False
    load_dotenv(env_path, override=False)
    return True
