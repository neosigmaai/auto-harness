"""Opt-in Docker cancellation/reaper fixture; zero provider calls.

Run with ``python -m service.sandbox.cleanup_smoke`` outside timed experiments.
Creates isolated, labelled fixtures and removes them in a finally block.
"""
import asyncio
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from service.worker.executors.harbor_runner import HarborExecutor, ROOT, RELAY_IMAGE, docker_command, reap_expired_sandboxes, remove_attempt


async def create_container(attempt, expiry, with_network=False):
    network = 'none'
    if with_network:
        network = 'aos-cleanup-' + attempt[-10:]
        code, output = await docker_command('network', 'create', '--internal', '--label', 'aos.attempt=' + attempt, network)
        assert code == 0, output
    code, output = await docker_command('run', '-d', '--network', network,
        '--label', 'aos.managed=true', '--label', 'aos.attempt=' + attempt,
        '--label', 'aos.expires_at=' + str(expiry), '--label', 'aos.job=cleanup-fixture',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--read-only',
        '--memory', '64m', '--cpus', '0.25', '--pids-limit', '16',
        '--user', '65534:65534', RELAY_IMAGE, 'sleep', '300')
    assert code == 0, output
    return output.strip()


async def main():
    suffix = uuid4().hex
    cancel_id, expired_id, current_id = ['cleanup-' + role + '-' + suffix for role in ('cancel', 'expired', 'control')]
    facts = {'real_provider_calls': 0}
    try:
        await create_container(cancel_id, time.time() + 120, with_network=True)
        process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(300)', start_new_session=True)
        executor = HarborExecutor()
        executor._processes.add(process)
        executor._attempts.add(cancel_id)
        started = time.monotonic()
        await executor.cancel()
        facts['cancel_duration_s'] = time.monotonic() - started
        assert process.returncode is not None
        assert facts['cancel_duration_s'] < 8
        code, remaining = await docker_command('ps', '-aq', '--filter', 'label=aos.attempt=' + cancel_id)
        assert code == 0 and not remaining.strip()
        code, remaining = await docker_command('network', 'ls', '-q', '--filter', 'label=aos.attempt=' + cancel_id)
        assert code == 0 and not remaining.strip()
        facts['cancel_stopped_runner_and_container_and_network'] = True
        await create_container(expired_id, time.time() - 5)
        control_container = await create_container(current_id, time.time() + 120)
        await reap_expired_sandboxes(reconcile_slots=False)
        code, remaining = await docker_command('ps', '-aq', '--filter', 'label=aos.attempt=' + expired_id)
        assert code == 0 and not remaining.strip()
        code, remaining = await docker_command('ps', '-q', '--no-trunc', '--filter', 'label=aos.attempt=' + current_id)
        assert code == 0 and remaining.strip() == control_container
        facts['reaper_removed_expired_and_preserved_unexpired'] = True
    finally:
        for attempt in (cancel_id, expired_id, current_id):
            await remove_attempt(attempt)
    destination = ROOT / 'runs' / ('cleanup-proof-' + suffix + '.json')
    destination.write_text(json.dumps(facts, indent=2))
    print(json.dumps({**facts, 'artifact_path': str(destination)}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())

