"""Zero-model fresh-container readiness/contamination probe for prepared images.

No benchmark instructions or assertions are accessed. Cold setup observations
come from the clean image build; warm probes time the version-only exec stage
and separately report container lifecycle time. This is not an agent score.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import time
from uuid import uuid4

from service.sandbox.preflight import offline_probe_command
from service.sandbox.prepared_images import docker, freeze_harness, validate_local_entry, MANIFEST


async def main(args):
    frozen = freeze_harness(args.tasks, MANIFEST['benchmark_revision'], args.manifest)
    reports = []
    for task in args.tasks:
        entry = frozen['entries'][task]
        await validate_local_entry(task, entry, frozen['provider_id'])
        repetitions = []
        for repetition in range(args.repetitions):
            name = 'aos-prepared-probe-' + uuid4().hex
            start = time.monotonic()
            record = {'repetition': repetition + 1}
            try:
                await docker('run', '-d', '--pull', 'never', '--name', name, '--network', 'none',
                    '--cpus', '1', '--memory', '2g', '--pids-limit', '128', '--cap-drop', 'ALL',
                    '--security-opt', 'no-new-privileges:true', '--entrypoint', 'sh',
                    entry['image_id'], '-c', 'exec sleep 90')
                info = json.loads(await docker('inspect', name))[0]
                assert info['Image'] == entry['image_id']
                assert not info['HostConfig']['Privileged']
                assert not any(m['Type'] == 'bind' for m in info['Mounts'])
                env_names = {v.split('=',1)[0] for v in info['Config']['Env']}
                assert not env_names.intersection({'NEOSIGMA_OPEN_API_KEY','OPENAI_API_KEY','OPEN_API_KEY','AOS_GATEWAY_TOKEN','DATABASE_URL'})
                # The builder never uploads these candidate/capability files.
                await docker('exec', name, 'sh', '-c',
                    'test ! -e /installed-agent/aos/candidate.py && test ! -e /installed-agent/aos/config.json '
                    '&& test ! -e /tmp/aos-prepared-contamination-proof')
                before = time.monotonic()
                output = await docker('exec', '-u', '0', name, 'bash', '-c',
                    'timeout --kill-after=3s 20s bash -c ' + shlex.quote(offline_probe_command(task)), timeout=25)
                record.update(probe_s=time.monotonic()-before, versions=output.splitlines(),
                              no_candidate_or_capability_files=True, prior_container_marker_absent=True)
                if repetition == 0:
                    # Poison only this disposable writable layer. The next
                    # fresh container must retain intact tools and no marker.
                    poison = 'printf isolated > /tmp/aos-prepared-contamination-proof'
                    if MANIFEST['bootstrap'][task]['mode'] == 'uvx':
                        poison += '; rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"'
                    await docker('exec', '-u', '0', name, 'sh', '-c', poison)
                    record['disposable_layer_poisoned_after_probe'] = True
            finally:
                await docker('rm', '-f', name)
                record['container_lifecycle_s'] = time.monotonic()-start
            repetitions.append(record)
        assert len(repetitions) >= 2
        reports.append({'task':task,'base_image_id':entry['base_image_id'],'prepared_image_id':entry['image_id'],
            'cache_key':entry['cache_key'],'cold_setup_s':entry['cold_setup_s'],
            'cold_sample_count':1,'warm_probe_samples':repetitions,
            'warm_probe_median_s':statistics.median(r['probe_s'] for r in repetitions),
            'one_time_build_and_probe_s':entry['build_and_probe_s'],
            'one_time_cost_basis':'includes provision/setup/commit/offline validation, excludes final builder cleanup',
            'fresh_layer_contamination_isolation_observed':True})
    report={'protocol':'prepared-tool-infrastructure-probe-v1','real_provider_calls':0,'task_assertions_executed':False,
        'candidate_executed':False,'manifest_sha256':frozen['manifest_sha256'],'provider_id':frozen['provider_id'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tasks':reports,
        'limitations':['One cold preparation per task; warm samples are not a randomized end-to-end agent comparison.',
            'Original benchmark verifier scripts remain unchanged and may still download dependencies.',
            'These results measure tool preparation, not benchmark success or model cost improvements.']}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':args.output,'tasks':[{'task':r['task'],'cold_setup_s':r['cold_setup_s'],
         'warm_probe_median_s':r['warm_probe_median_s']} for r in reports]},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--tasks',nargs='+',required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 2 <= args.repetitions <= 10:parser.error('repetitions must be between2and10')
    asyncio.run(main(args))
