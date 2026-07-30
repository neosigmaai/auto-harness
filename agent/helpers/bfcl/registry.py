"""Register the auto-harness BFCL handler into BFCL's MODEL_CONFIG_MAPPING.

Must be called before any `bfcl generate` / `bfcl evaluate` invocation that uses
`harness-agent` as the model id. The registration only persists for the
lifetime of the current Python process — subprocesses must call this themselves
(see `agent.helpers.bfcl.run`).
"""

from __future__ import annotations

import os


def register_harness_handler() -> None:
    """Mutate BFCL's MODEL_CONFIG_MAPPING so `harness-agent` resolves to HarnessHandler.

    `BFCL_PROJECT_ROOT` must already be set in the environment because importing
    `bfcl_eval.constants.eval_config` (transitively imported here) runs
    `mkdir(...)` for `RESULT_PATH`, `SCORE_PATH`, and `LOCK_DIR` at module load.
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    from agent.agent import HarnessHandler

    MODEL_CONFIG_MAPPING["harness-agent"] = ModelConfig(
        model_name=os.environ.get("AGENT_MODEL", "gpt-5.4"),
        display_name="auto-harness BFCL agent",
        url="https://github.com/neosigmaai/auto-harness",
        org="auto-harness",
        license="",
        model_handler=HarnessHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )
