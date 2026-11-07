"""Run real pytest cases against the nonce hook; no benchmark tests or model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from service.sandbox.verifier_contract import pytest_probe_source, validate_report, VerifierProtocolError


def run_probe(tmp_path, source, *args):
    marker = tmp_path / 'session.json'
    plugin = pytest_probe_source('fixture').replace("Path('/logs/verifier/aos-session.json')", f'Path({str(marker)!r})')
    (tmp_path / 'aos_probe.py').write_text(plugin)
    (tmp_path / 'test_cases.py').write_text(source)
    env = {'PATH':os.environ.get('PATH',''), 'HOME':str(tmp_path),
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1', 'PYTHONPATH':str(tmp_path)}
    result = subprocess.run([sys.executable,'-m','pytest','-p','aos_probe','-q',*args,str(tmp_path/'test_cases.py')],
                            env=env,cwd=tmp_path,capture_output=True,text=True,timeout=20)
    session = json.loads(marker.read_text())
    return session,result.returncode


def grouped_report(session, failed=False):
    # CTRF0.3.5 groups parameterized cases into one entry. The actual pinned
    # plugin is also exercised by the Docker smoke's 27-case synthetic test.
    status = 'failed' if failed else 'passed'
    return {'results':{'tool':{'name':'pytest'}, 'summary':{
        'tests':1,'passed':int(not failed),'failed':int(failed),'skipped':0,'pending':0,'other':0,
        'start':session['started_at'],'stop':session['finished_at']},'tests':[{'status':status}]}}


def test_27_parameter_cases_accept_grouped_report(tmp_path):
    session,code=run_probe(tmp_path,"import pytest\n@pytest.mark.parametrize('n', range(27))\ndef test_case(n): assert n >= 0\n")
    assert code==0 and session['tests_collected']==27
    assert session['case_counts']=={'passed':27,'failed':0,'skipped':0,'incomplete':0}
    assert validate_report(grouped_report(session),session,'fixture',0,1)['passed']==1
    with pytest.raises(VerifierProtocolError,match='reward_report_mismatch'):
        validate_report(grouped_report(session),session,'fixture',0,0)


def test_27_cases_failure_keeps_real_zero(tmp_path):
    session,code=run_probe(tmp_path,"import pytest\n@pytest.mark.parametrize('n', range(27))\ndef test_case(n): assert n != 13\n")
    assert code==1 and session['case_counts']['failed']==1 and session['executed_cases']==27
    validate_report(grouped_report(session,True),session,'fixture',0,0)


@pytest.mark.parametrize('args,reason',[(['-k','not 13'],'deselected'),(['-x'],'incomplete_case_execution')])
def test_deselected_or_early_stop_is_unscorable(tmp_path,args,reason):
    session,_=run_probe(tmp_path,"import pytest\n@pytest.mark.parametrize('n', range(27))\ndef test_case(n): assert n != 0\n",*args)
    with pytest.raises(VerifierProtocolError,match=reason):
        validate_report(grouped_report(session,True),session,'fixture',0,0)


@pytest.mark.parametrize('source',[
    "import pytest\n@pytest.mark.skip(reason='fixture')\ndef test_case(): pass\n",
    "import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError('setup')\ndef test_case(broken): pass\n",
    "raise RuntimeError('collection')\n",
])
def test_no_test_body_or_collection_error_is_unscorable(tmp_path,source):
    session,_=run_probe(tmp_path,source)
    with pytest.raises(VerifierProtocolError):
        validate_report(grouped_report(session,True),session,'fixture',0,0)


def test_teardown_failure_after_executed_body_is_valid_zero(tmp_path):
    session,code=run_probe(tmp_path,"import pytest\n@pytest.fixture\ndef broken():\n yield\n raise RuntimeError('teardown')\ndef test_case(broken): assert True\n")
    assert code==1 and session['executed_cases']==1 and session['case_counts']['failed']==1
    validate_report(grouped_report(session,True),session,'fixture',0,0)


def test_grouped_skipped_status_does_not_hide_real_case_failure(tmp_path):
    session,code=run_probe(tmp_path,"import pytest\n@pytest.mark.parametrize('n', range(3))\ndef test_case(n):\n if n == 0: pytest.skip('fixture')\n assert n != 1\n")
    assert code==1 and session['case_counts']=={'passed':1,'failed':1,'skipped':1,'incomplete':0}
    report=grouped_report(session)
    report['results']['summary'].update(passed=0,skipped=1)
    report['results']['tests'][0]['status']='skipped'
    validate_report(report,session,'fixture',0,0)
