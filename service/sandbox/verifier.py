"""Fixed Harbor 0.20 verifier for an explicitly pinned pytest suite.

Import path: service.sandbox.verifier:PinnedPytestVerifier.
Shared task filesystem remains susceptible to adversarial tampering; this guard
establishes operational grading completion, not a cryptographic trust boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import tempfile
import time

from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.scripts import build_execution_command
from harbor.verifier.verifier import Verifier

from service.sandbox.verifier_contract import CONTRACT, VerifierProtocolError, pytest_probe_source, validate_report


class PinnedPytestVerifier(Verifier):
    def __init__(self, *args, expected_script_sha256: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_script_sha256 = expected_script_sha256

    async def verify(self):
        _, tests_source_dir, script = self._resolve_tests()
        if hashlib.sha256(script.read_bytes()).hexdigest() != self.expected_script_sha256:
            raise VerifierProtocolError('unsupported_verifier_orchestration_digest')
        nonce = secrets.token_hex(16)
        module = '_aos_pytest_' + nonce
        metadata = {'contract': CONTRACT, 'valid': False,
                    'orchestration_sha256': self.expected_script_sha256}
        evidence_path = self.trial_paths.trial_dir / 'verifier-execution.json'
        paths = EnvironmentPaths.for_os(self.environment.os)
        container_script = str(paths.tests_dir / script.relative_to(tests_source_dir).as_posix())
        container_stdout = str(paths.verifier_dir / self.trial_paths.test_stdout_path.relative_to(self.trial_paths.verifier_dir).as_posix())
        command = build_execution_command(container_script, stdout_path=container_stdout, task_os=self.environment.os)
        original_exec = self.environment.exec
        original_override = self.override_env
        observed_exit = None

        async def capture_exec(*args, **kwargs):
            nonlocal observed_exit
            submitted = kwargs.get('command', args[0] if args else None)
            if submitted == command:
                metadata['shell_started_at'] = time.time()
            result = await original_exec(*args, **kwargs)
            if submitted == command:
                observed_exit = result.return_code
                metadata.update(shell_finished_at=time.time(), shell_exit_code=observed_exit)
            return result

        try:
            # Clear prior evidence before the trusted verifier invocation. This
            # avoids inheriting candidate-written or prior-attempt reward files.
            reset = await original_exec(
                'mkdir -p /tests /logs/verifier && rm -f -- /logs/verifier/ctrf.json '
                '/logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/aos-session.json', user='root')
            if reset.return_code:
                raise VerifierProtocolError('cannot_reset_verifier_evidence')
            for name in ('ctrf.json', 'reward.txt', 'reward.json', 'aos-session.json'):
                (self.trial_paths.verifier_dir / name).unlink(missing_ok=True)
            plugin = pytest_probe_source(nonce)
            with tempfile.TemporaryDirectory(prefix='aos-verifier-') as directory:
                local_plugin = Path(directory) / (module + '.py')
                local_plugin.write_text(plugin)
                await self.environment.upload_file(local_plugin, '/tests/' + module + '.py')
            # This is an explicit supported-suite contract. A future task that
            # depends on conflicting pytest injection/PYTHONPATH needs an audit.
            self.override_env = {**original_override, 'PYTHONPATH': '/tests',
                                 'PYTEST_ADDOPTS': '-p ' + module}
            self.environment.exec = capture_exec
            result = await super().verify()
            raw_reward = result.rewards.get('reward')
            metadata['raw_reward'] = raw_reward if raw_reward in (0, 1) else repr(raw_reward)

            def read_report(name):
                path = self.trial_paths.verifier_dir / name
                if not path.is_file() or path.is_symlink() or path.stat().st_size > 4_000_000:
                    raise VerifierProtocolError('missing_or_oversized_' + name)
                try:
                    return json.loads(path.read_text())
                except (ValueError, UnicodeDecodeError) as exc:
                    raise VerifierProtocolError('invalid_' + name) from exc

            report = read_report('ctrf.json')
            session = read_report('aos-session.json')
            counts = validate_report(report, session, nonce, observed_exit, raw_reward)
            metadata.update(valid=True, counts=counts, pytest_exit_code=session['pytest_exit_code'],
                            fresh_session_marker=True, case_counts=session['case_counts'],
                            tests_collected=session['tests_collected'], deselected_cases=session['deselected_cases'],
                            selected_cases=session['selected_cases'], executed_cases=session['executed_cases'],
                            unexpected_cases=session['unexpected_cases'], duplicate_phase=session['duplicate_phase'])
            return result
        except Exception as exc:
            metadata['error_type'] = type(exc).__name__
            metadata['reason'] = str(exc) if isinstance(exc, VerifierProtocolError) else 'verifier_execution_failed'
            raise VerifierProtocolError('verifier_evidence_invalid:' + type(exc).__name__) from exc
        finally:
            self.environment.exec = original_exec
            self.override_env = original_override
            evidence_path.write_text(json.dumps(metadata, indent=2))

