import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from service.sandbox import prepared_images as prepared
from service.sandbox.preflight import offline_probe_command, probe_prepared

TASK = 'fix-git'
REVISION = prepared.MANIFEST['benchmark_revision']


def entry():
    base = 'sha256:' + 'a' * 64
    return {'base_reference': prepared.BY_NAME[TASK]['docker_image'], 'base_image_id': base,
            'image_id': 'sha256:' + 'b' * 64, 'platform': 'linux/amd64',
            'recipe_digest': prepared.recipe_digest(TASK), 'cache_key': prepared.cache_key(TASK, base)}


def manifest(path, item=None):
    value = {'protocol': prepared.PROTOCOL, 'provider_id': 'docker:fixture',
             'benchmark_revision': REVISION, 'entries': {TASK: item or entry()}}
    path.write_text(json.dumps(value))
    return value


def test_cold_mode_is_explicit_and_has_no_manifest_dependency():
    assert prepared.freeze_harness([TASK], REVISION) == {'mode': 'cold-v1'}


def test_submission_freezes_exact_entries_independent_of_later_file_edits(tmp_path):
    path = tmp_path / 'manifest.json'
    initial = manifest(path)
    frozen = prepared.freeze_harness([TASK], REVISION, str(path))
    path.write_text('{}')
    assert frozen['entries'] == initial['entries']
    assert frozen['manifest_sha256'] == prepared.digest(initial)


@pytest.mark.parametrize('field,value', [('base_reference','other'), ('image_id','mutable:tag'),
    ('base_image_id','mutable:tag'), ('recipe_digest','stale'), ('cache_key','stale'), ('platform','linux/arm64')])
def test_mismatched_provenance_is_rejected(tmp_path, field, value):
    item=entry(); item[field]=value
    path=tmp_path/'manifest.json'; manifest(path,item)
    with pytest.raises(ValueError): prepared.freeze_harness([TASK],REVISION,str(path))


def test_missing_task_or_wrong_revision_is_rejected(tmp_path):
    path=tmp_path/'manifest.json'; manifest(path)
    with pytest.raises(ValueError):prepared.freeze_harness(['regex-log'],REVISION,str(path))
    with pytest.raises(ValueError):prepared.freeze_harness([TASK],'other-revision',str(path))


def test_offline_probe_runs_versions_only_and_cannot_collect_hidden_tests():
    for task in prepared.MANIFEST['bootstrap']:
        command=offline_probe_command(task)
        assert 'UV_OFFLINE=1' in command and 'UV_PYTHON_DOWNLOADS=never' in command
        assert '--version' in command and '/tests' not in command and '--ctrf' not in command
        assert 'pip install' not in command and 'curl ' not in command and 'apt-get' not in command


def test_probe_failure_prevents_candidate_upload(tmp_path):
    environment=SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1,stdout='',stderr='')))
    path=tmp_path/'probe.json'
    with pytest.raises(RuntimeError):asyncio.run(probe_prepared(environment,TASK,path,entry()))
    assert json.loads(path.read_text())['status']=='failed'
    assert environment.exec.await_args.kwargs['timeout_sec']==25


def test_local_validation_rejects_wrong_daemon_before_image_access(monkeypatch):
    monkeypatch.setattr(prepared,'docker',AsyncMock(return_value='other'))
    inspect=AsyncMock();monkeypatch.setattr(prepared,'inspect_image',inspect)
    with pytest.raises(RuntimeError,match='another Docker daemon'):
        asyncio.run(prepared.validate_local_entry(TASK,entry(),'docker:fixture'))
    inspect.assert_not_awaited()


def test_local_validation_checks_base_layers_and_original_runtime_config(monkeypatch):
    item=entry()
    base={'Id':item['base_image_id'],'Config':{'Env':['PATH=/usr/bin'],'Cmd':['bash'],'Entrypoint':None},
          'RootFS':{'Layers':['base']},'Architecture':'amd64','Os':'linux'}
    cached=copy.deepcopy(base);cached.update(Id=item['image_id']);cached['RootFS']['Layers'].append('trusted-tools')
    cached['Config']['Labels']={'aos.prepared.key':item['cache_key']}
    monkeypatch.setattr(prepared,'docker',AsyncMock(return_value='fixture'))
    monkeypatch.setattr(prepared,'inspect_image',AsyncMock(side_effect=lambda image: copy.deepcopy(base if image==item['base_reference'] else cached)))
    asyncio.run(prepared.validate_local_entry(TASK,item,'docker:fixture'))
    cached['Config']['Env'].append('SENSITIVE=not-allowed')
    with pytest.raises(RuntimeError,match='runtime configuration'):
        asyncio.run(prepared.validate_local_entry(TASK,item,'docker:fixture'))
    cached['Config']['Env']=base['Config']['Env']
    cached['Config']['Cmd']=['sleep','expired']
    with pytest.raises(RuntimeError,match='builder command'):
        asyncio.run(prepared.validate_local_entry(TASK,item,'docker:fixture'))
    cached['Config']['Cmd']=base['Config']['Cmd']
    cached['Config']['Labels']['com.docker.compose.project']='private-build-identifier'
    with pytest.raises(RuntimeError,match='transient Compose labels'):
        asyncio.run(prepared.validate_local_entry(TASK,item,'docker:fixture'))
    cached['Config']['Labels']['com.docker.compose.project']=''
    cached['Config']['Labels']['aos.managed']='true'
    with pytest.raises(RuntimeError,match='active builder lifecycle labels'):
        asyncio.run(prepared.validate_local_entry(TASK,item,'docker:fixture'))
