import os

from autoharness_service.api import create_app


def background_enabled() -> bool:
    value = os.getenv("AUTOHARNESS_START_BACKGROUND", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


app = create_app(start_background=background_enabled())
