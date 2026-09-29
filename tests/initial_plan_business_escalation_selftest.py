"""Offline controller tests for the original, single business revision slot."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import initial_plan_recovery_selftest as fixtures
from pipeline import initial_plan_recovery as recovery
from pipeline.instance_plan import PlanConflict, finding
from tools import run_original_bc_smoke as smoke
from copy import deepcopy
from unittest.mock import patch
import unittest


class BusinessEscalationTests(unittest.TestCase):
    def setUp(self):
        self.wp,self.candidate,*_=fixtures.setup_plan()
        self.context={'candidate':deepcopy(self.candidate),'compiler_findings':[finding('fixture','bound failure')],
            'failure_history':{'original':'immutable'},'seed_input':{},'seed_pack':{},'delivery_target':{}}
        self.permit={'escalation':self.context,'escalation_sha256':recovery._digest(self.context),
            'source_migration':{'source':'fixture'},'retry_policy':{'version':'fixture'},
            'retry_policy_sha256':'policy','receipt_sha256':'request','original':{'manifest':{'config':{}}}}
        self.record={'business_revision_remaining':1,'logical_plan_attempt':2,'transaction':{'original':'unchanged'}}
        self.snapshots=[];self.calls=[]
        outer=self
        class Trace:
            def chat_json(self,step,messages,*args,**kwargs):
                outer.assertEqual(outer.snapshots[-1]['business_call_history'][-1]['status'],'request_pending')
                outer.calls.append(step)
                return {'returned':step}
        self.trace=Trace()
        self.diagnostic={'decision':'repair','author_allowed':True,'review':{'decision':'repair'},'responsibility_targets':[{'unit_id':'bound'}]}
        self.validator=patch('pipeline.blueprint_feasibility.validate_uncompiled_diagnostic',return_value={},create=True)
        self.validator.start();self.addCleanup(self.validator.stop)

    def checkpoint(self,*args,**kwargs):self.snapshots.append(deepcopy(self.record))

    def diagnose(self,wp,candidate,findings,tracer,**kwargs):
        self.assertEqual(self.record['business_revision_remaining'],1)
        self.assertEqual(self.record['logical_plan_attempt'],2)
        self.assertEqual(candidate,self.context['candidate'])
        self.assertEqual(kwargs['remaining_business_attempts'],1)
        self.assertEqual(kwargs['history'],self.context['failure_history'])
        tracer.chat_json('council.blueprint_feasibility',[{'role':'user','content':'original diagnostic'}])
        return deepcopy(self.diagnostic)

    def revise(self,wp,candidate,feedback,tracer,audit,*,max_attempts,checkpoint,uncompiled_diagnostic=None):
        self.assertEqual(max_attempts,1)
        self.assertEqual(self.record['business_revision_remaining'],0)
        self.assertEqual(self.record['logical_plan_attempt'],3)
        self.assertEqual(uncompiled_diagnostic,self.diagnostic)
        tracer.chat_json('council.instance_plan_business_repair',[{'role':'user','content':'original remaining author'}])
        audit['business_revisions']=[{'attempt':1,'business_review':{'decision':'accept'}}]
        checkpoint(audit)
        return {'business_instance_plan':candidate,'supply_plan':{'verified':'fixture'}}

    def run_escalation(self,revise=None):
        with patch('pipeline.blueprint_feasibility.diagnose_uncompiled_plan',side_effect=self.diagnose,create=True),patch(
                'pipeline.layout_revision.revise',side_effect=revise or self.revise):
            return recovery._run_business_escalation(self.wp,self.permit,self.record,self.trace,self.checkpoint)

    def test_independent_repair_uses_only_original_one_business_slot(self):
        result=self.run_escalation()
        self.assertEqual(self.calls,['council.blueprint_feasibility','council.instance_plan_business_repair'])
        self.assertEqual(self.record['business_revision_remaining'],0)
        self.assertEqual(self.record['transaction'],{'original':'unchanged'})
        self.assertIn('business_instance_plan',result)

    def test_unresolved_never_starts_author_or_consumes_business_slot(self):
        self.diagnostic.update(decision='unresolved',author_allowed=False,review={'decision':'unresolved'})
        with self.assertRaises(PlanConflict):self.run_escalation()
        self.assertEqual(self.calls,['council.blueprint_feasibility'])
        self.assertEqual(self.record['business_revision_remaining'],1)
        self.assertEqual(self.record['status'],'design_unresolved')

    def test_accept_cannot_bypass_original_compiler_failure(self):
        self.diagnostic.update(decision='accept',author_allowed=False,review={'decision':'accept'})
        with patch('pipeline.instance_plan.compile_plan',side_effect=PlanConflict([finding('fixture','still invalid')])):
            with self.assertRaises(PlanConflict):self.run_escalation()
        self.assertEqual(self.calls,['council.blueprint_feasibility'])
        self.assertEqual(self.record['business_revision_remaining'],1)

    def test_accept_still_requires_original_post_compile_review(self):
        self.diagnostic.update(decision='accept',author_allowed=False,review={'decision':'accept'})
        def assess(wp,tracer,**kwargs):
            tracer.chat_json('council.blueprint_feasibility',[{'role':'user','content':'compiled review'}])
            return {'decision':'accept'}
        with patch('pipeline.instance_plan.compile_plan',return_value={'compiled':True}),patch(
                'pipeline.blueprint_feasibility.assess',side_effect=assess):
            self.assertEqual(self.run_escalation(),{'compiled':True})
        self.assertEqual(self.calls,['council.blueprint_feasibility','council.blueprint_feasibility'])
        self.assertEqual(self.record['business_revision_remaining'],1)

    def test_missing_responsibility_permission_never_calls_author(self):
        self.diagnostic['author_allowed']=False
        with self.assertRaises(ValueError):self.run_escalation()
        self.assertEqual(self.calls,['council.blueprint_feasibility'])
        self.assertEqual(self.record['business_revision_remaining'],1)

    def test_validator_failure_does_not_spend_business_slot(self):
        with patch('pipeline.blueprint_feasibility.validate_uncompiled_diagnostic',side_effect=ValueError('binding'),create=True):
            with self.assertRaises(ValueError):self.run_escalation()
        self.assertEqual(self.record['business_revision_remaining'],1)
        self.assertEqual(self.calls,['council.blueprint_feasibility'])

    def test_author_failure_keeps_used_business_slot_and_old_transaction(self):
        def fail(*args,**kwargs):
            self.assertEqual(self.record['business_revision_remaining'],0)
            raise PlanConflict([finding('fixture','author plan invalid')])
        with self.assertRaises(PlanConflict):self.run_escalation(revise=fail)
        self.assertEqual(self.record['business_revision_remaining'],0)
        self.assertEqual(self.record['logical_plan_attempt'],3)
        self.assertEqual(self.record['transaction'],{'original':'unchanged'})

    def test_existing_used_business_slot_cannot_restart(self):
        self.record['business_revision_remaining']=0
        with self.assertRaises(ValueError):self.run_escalation()
        self.assertEqual(self.calls,[])

    def test_global_stop_bubbles_without_author_or_slot_reset(self):
        class GlobalStop(BaseException):pass
        def stop(*args,**kwargs):raise GlobalStop('budget')
        with patch.object(self.trace,'chat_json',side_effect=stop):
            with self.assertRaises(GlobalStop):self.run_escalation()
        self.assertEqual(self.record['business_revision_remaining'],1)
        self.assertEqual(self.record['business_call_history'][-1]['status'],'call_failed')

    def test_escalation_context_cannot_change_after_receipt(self):
        self.context['compiler_findings']=[]
        with self.assertRaises(ValueError):self.run_escalation()
        self.assertEqual(self.calls,[])

    def test_final_author_output_without_independent_accept_rejected(self):
        def missing(*args,**kwargs):return {'fake':'compiled'}
        with self.assertRaises(ValueError):self.run_escalation(revise=missing)
        self.assertEqual(self.record['business_revision_remaining'],0)

    def test_wrapper_flag_cannot_supply_or_change_escalation(self):
        smoke._validate_plan_escalation_option(None,{})
        with self.assertRaises(ValueError):smoke._validate_plan_escalation_option(recovery.ESCALATION,{})
        permit={'plan_escalation':recovery.ESCALATION,'escalation':self.context,
            'escalation_sha256':self.permit['escalation_sha256'],
            'source_migration':{'escalation_sha256':self.permit['escalation_sha256']},
            'first_new_step':'council.blueprint_feasibility'}
        smoke._validate_plan_escalation_option(recovery.ESCALATION,permit)
        with self.assertRaises(ValueError):smoke._validate_plan_escalation_option(None,permit)
        permit['first_new_step']='council.instance_plan_unit_repair'
        with self.assertRaises(ValueError):smoke._validate_plan_escalation_option(recovery.ESCALATION,permit)


if __name__=='__main__':unittest.main(verbosity=2)
