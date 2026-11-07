import unittest
from service.sandbox.verifier_contract import validate_report, VerifierProtocolError


def evidence(passed=1, failed=0, skipped=0):
    statuses=['passed']*passed+['failed']*failed+['skipped']*skipped
    report={'results':{'tool':{'name':'pytest'},'summary':{'tests':len(statuses),'passed':passed,'failed':failed,'skipped':skipped,'pending':0,'other':0,'start':101,'stop':102},'tests':[{'status':s} for s in statuses]}}
    session={'nonce':'fresh','tests_collected':len(statuses),'pytest_exit_code':1 if failed else 0,'started_at':100,'finished_at':103,
             'selected_cases':len(statuses),'executed_cases':passed+failed,'deselected_cases':0,'unexpected_cases':0,'duplicate_phase':False,
             'case_counts':{'passed':passed,'failed':failed,'skipped':skipped,'incomplete':0}}
    return report,session


class ContractTests(unittest.TestCase):
    def test_valid_pass(self):
        report,session=evidence(2,0,1)
        self.assertEqual(validate_report(report,session,'fresh',0,1)['passed'],2)
    def test_real_assertion_failure_is_valid_zero(self):
        report,session=evidence(1,1)
        self.assertEqual(validate_report(report,session,'fresh',0,0)['failed'],1)
    def test_missing_ctrf_and_session_reject_echoed_zero(self):
        with self.assertRaises(VerifierProtocolError): validate_report(None,None,'fresh',0,0)
    def test_stale_nonce(self):
        report,session=evidence()
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'new-invocation',0,1)
    def test_skipped_only_is_not_a_pass(self):
        report,session=evidence(0,0,2)
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_invalid_counts(self):
        for bad in [-1,True,1.5,'1',100001]:
            report,session=evidence()
            report['results']['summary']['passed']=bad
            with self.subTest(bad=bad),self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_inconsistent_totals(self):
        report,session=evidence()
        report['results']['summary']['tests']=2
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_reward_conflicts_with_completed_tests(self):
        report,session=evidence(0,1)
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_collection_or_interrupted_run(self):
        for code in [2,3,4,5]:
            report,session=evidence()
            session['pytest_exit_code']=code
            with self.subTest(code=code),self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_report_does_not_belong_to_fresh_session(self):
        report,session=evidence()
        report['results']['summary'].update(start=10,stop=12)
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)
    def test_missing_reward_or_shell_error(self):
        report,session=evidence()
        for code,reward in [(1,1),(0,None),(0,True)]:
            with self.subTest(code=code,reward=reward),self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',code,reward)
    def test_status_list_must_match_counts(self):
        report,session=evidence()
        report['results']['tests'][0]['status']='failed'
        with self.assertRaises(VerifierProtocolError): validate_report(report,session,'fresh',0,1)

if __name__=='__main__': unittest.main()

