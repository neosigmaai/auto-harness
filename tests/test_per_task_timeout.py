import os
import tempfile
import unittest
from unittest import mock

import gating
import prepare
from benchmark import TerminalBenchRunner

TIMEOUT = 42

TERMINAL_BENCH_CFG = {
    "benchmark": "terminal-bench",
    "agent_model": "dummy-model",
    "env_provider": "docker",
    "max_concurrency": 1,
    "per_task_timeout": TIMEOUT,
}


class _Stop(Exception):
    pass


class TerminalBenchPerTaskTimeoutTest(unittest.TestCase):
    def test_gate_runners_use_configured_timeout(self):
        train_runner, gate_runner = gating._create_runners(TERMINAL_BENCH_CFG)

        self.assertEqual(train_runner.per_task_timeout, TIMEOUT)
        self.assertEqual(gate_runner.per_task_timeout, TIMEOUT)

    def test_baseline_runner_uses_configured_timeout(self):
        for split_exists in (False, True):
            with self.subTest(split_exists=split_exists):
                self.assertEqual(
                    self._baseline_runner(split_exists).per_task_timeout, TIMEOUT
                )

    def _baseline_runner(self, split_exists: bool) -> TerminalBenchRunner:
        with tempfile.TemporaryDirectory() as tmp:
            results_file = os.path.join(tmp, "results.tsv")
            split_file = os.path.join(tmp, "task_split.json")
            open(results_file, "w").close()
            if split_exists:
                open(split_file, "w").close()

            with (
                mock.patch.object(prepare, "RESULTS_FILE", results_file),
                mock.patch.object(prepare, "SPLIT_FILE", split_file),
                mock.patch.object(
                    TerminalBenchRunner, "run", autospec=True, side_effect=_Stop
                ) as run,
                self.assertRaises(_Stop),
            ):
                prepare.run_baseline(TERMINAL_BENCH_CFG)

        return run.call_args.args[0]


if __name__ == "__main__":
    unittest.main()
