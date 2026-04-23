"""
Tiny stdlib .env loader. Imported at the top of prepare.py, benchmark.py, and
gating.py so `python prepare.py` works without a separate `source .env` step.

- Skipped if .env is absent or AUTO_HARNESS_SKIP_DOTENV=1 is set.
- Does NOT overwrite values already in the environment — real shell exports win.
- Runs once per process (guarded by a module-level flag).
- No external dependencies; supports KEY=VALUE, `export KEY=VALUE`, single/double
  quoted values, and `#` line comments. No variable interpolation by design.
"""

from __future__ import annotations

import os

_loaded = False


def load_dotenv_once() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if os.environ.get("AUTO_HARNESS_SKIP_DOTENV") == "1":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if key and key not in os.environ and value != "":
                    os.environ[key] = value
    except OSError:
        pass


load_dotenv_once()
