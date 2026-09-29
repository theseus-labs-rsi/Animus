"""Controller accepts the wrapper's bound budget transition and rejects other cap changes."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.production import observe_budget,digest
class BudgetTransitionTests(unittest.TestCase):
    def setUp(self):
        self.ledger={'model':'glm-5.3-flash','max_calls':1200,'max_cny':35,'admitted_calls':582,
            'budget_consumed_cny':18.451827444,'reported_usage_estimate_cny':18.451827444,'unsettled_reservation_calls':0}
        self.proof={'reason':'explicit_cumulative_budget_resume','model':self.ledger['model'],'max_calls':1200,
            'prior_max_cny':20,'new_max_cny':35,'prior_admitted_calls':582,'prior_budget_consumed_cny':18.451827444,
            'authorization_sha256':'a'*64,'trace_sha256':'b'*64}
        self.state={'round':3,'history':['keep'],'generation_budget':{'bounded':True,
            'cap_identity':digest({'model':self.ledger['model'],'max_calls':1200,'max_cny':20}),
            'admitted_calls':570,'occupied_estimate_cny':18.0}}
    def invoke(self):
        self.ledger['resume_history']=[{'transport_recovery':self.proof}]
        ledger=self.ledger
        class Run:
            def has(self,n):return True
            def read(self,n):return deepcopy(ledger)
        observe_budget(Run(),self.state)
    def test_transition_keeps_controller_history_and_accounting(self):
        self.invoke();self.assertEqual(self.state['round'],3);self.assertEqual(self.state['history'],['keep'])
        self.assertEqual(self.state['generation_budget']['admitted_calls'],582)
        self.assertEqual(self.state['generation_budget']['authorized_cap_transition']['new_max_cny'],35)
    def test_other_model_calls_or_cap_changes_rejected(self):
        original=deepcopy(self.proof)
        for key,value in [('model','other'),('max_calls',1201),('prior_max_cny',12),('new_max_cny',40),('reason','other'),('prior_admitted_calls',583),('authorization_sha256','')]:
            self.proof=deepcopy(original);self.proof[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.invoke()
    def test_reset_calls_or_cost_rejected(self):
        self.ledger['admitted_calls']=569
        with self.assertRaises(ValueError):self.invoke()
        self.ledger['admitted_calls']=582;self.ledger['budget_consumed_cny']=1
        with self.assertRaises(ValueError):self.invoke()
    def test_unchanged_cap_needs_no_transition(self):
        self.ledger['max_cny']=20;self.proof={}
        self.invoke();self.assertNotIn('authorized_cap_transition',self.state['generation_budget'])
if __name__=='__main__':unittest.main(verbosity=2)
