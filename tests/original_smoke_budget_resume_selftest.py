"""Explicit cap changes preserve the original stopped ledger and liabilities."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_original_bc_smoke as smoke

class BudgetResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.profile={'model':'glm-5.3-flash','stopped':True,'max_calls':1200,'admitted_calls':2,
            'max_cny':20,'budget_consumed_cny':18.45,'unsettled_reservation_calls':1,'source_drift':[],
            'stop_reason':{'reason':'estimated_budget_limit'}}
        self.rows=[{'event':'request','model':self.profile['model'],'call_id':x} for x in ('a','b')]
        self.trace=self.root/'trace.jsonl';self.trace.write_text('\n'.join(json.dumps(x) for x in self.rows))
        self.authorization={'policy':'explicit-cumulative-budget-increase/v1','user_authorization':'Increase cumulative cap to 35',
            'new_max_cny':35,'previous_max_cny':20,'model':self.profile['model'],'max_calls':1200,
            'profile_sha256':self.digest(self.profile),'trace_sha256':hashlib.sha256(self.trace.read_bytes()).hexdigest()}
        self.path=self.root/'authorization.json'
    def digest(self,p):return hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    def check(self):
        self.path.write_text(json.dumps(self.authorization));return smoke.budget_cap_resume_evidence(self.profile,self.trace,self.path)
    def test_cap_increase_preserves_calls_cost_and_unknown(self):
        before=deepcopy(self.profile);result=self.check()
        self.assertEqual(self.profile,before);self.assertEqual(result['new_max_cny'],35)
        self.assertEqual(result['prior_budget_consumed_cny'],18.45)
        self.assertEqual(result['prior_admitted_calls'],2);self.assertEqual(result['unknown_reservations_preserved'],1)
    def test_nonbudget_stop_drift_exhausted_calls_and_modified_profile_stay_closed(self):
        before=deepcopy(self.profile)
        for key,value in [('stop_reason',{'reason':'provider_failure','kind':'http_status_401'}),('source_drift',['config.py']),('admitted_calls',1200),('budget_consumed_cny',18.4)]:
            self.profile=deepcopy(before);self.profile[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
    def test_no_authorization_or_increased_model_call_allowance(self):
        before=deepcopy(self.authorization)
        for key,value in [('user_authorization',''),('max_calls',1201),('model','other'),('trace_sha256','changed'),('new_max_cny',20),('new_max_cny',float('nan'))]:
            self.authorization=deepcopy(before);self.authorization[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.check()
    def test_missing_duplicate_or_wrong_model_trace_rejected(self):
        for rows in [self.rows[:1],[self.rows[0]]*2,[self.rows[0],dict(self.rows[1],model='other')]]:
            self.trace.write_text('\n'.join(json.dumps(x) for x in rows))
            self.authorization['trace_sha256']=hashlib.sha256(self.trace.read_bytes()).hexdigest()
            with self.subTest(rows=rows),self.assertRaises(ValueError):self.check()
    def test_terminal_zero_call_resume_does_not_reapply_increase(self):
        self.profile.update(stopped=False,max_cny=35,admitted_calls=100)
        self.assertIsNone(self.check())
        self.authorization['new_max_cny']=40
        with self.assertRaises(ValueError):self.check()

if __name__=='__main__':unittest.main(verbosity=2)
