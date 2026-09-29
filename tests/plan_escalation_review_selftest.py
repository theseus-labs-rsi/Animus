"""Original independent review of the real rejected A5 plan; all calls are offline."""
import os
import sys
import socket
import types
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

os.environ.update(PYTHON_DOTENV_DISABLED='1', OPENAI_API_KEY='offline-dummy',
                  OPENAI_BASE_URL='http://offline.invalid', MODEL='offline-dummy')
def forbidden(*args, **kwargs): raise AssertionError('Network/provider prohibited')
socket.socket.connect=forbidden; socket.socket.connect_ex=forbidden
socket.create_connection=forbidden; socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv'); dotenv.load_dotenv=lambda *a, **k:False
dotenv.find_dotenv=lambda *a, **k:''; sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('ESCALATION_TEST_REPO', Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import openai
openai.OpenAI=forbidden; openai.AsyncOpenAI=forbidden
import config
config.chat=forbidden
from pipeline import blueprint_feasibility as review
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.run import Tracer as ProductionTracer
from execution_control import ExecutionStopped
from blueprint_feasibility_selftest import Tracer

EVIDENCE=Path(os.environ.get('ESCALATION_A5_EVIDENCE',
    ROOT/'output/fresh_plan_failure_after_protocol_v4_20260928'))


def a5_fixture():
    # The local archived paid output is read-only. No old run is resumed here.
    wp=json.loads((EVIDENCE/'01_whitepaper.json').read_text(encoding='utf-8'))
    record=json.loads((EVIDENCE/'01_initial_plan_recovery.json').read_text(encoding='utf-8'))
    candidate=deepcopy(record['transaction']['drafts'][-1]['candidate'])
    findings=deepcopy(record['transaction']['drafts'][-1]['findings'])
    history={'cumulative_transaction':record['transaction'], 'terminal_findings':record['findings'],
             'logical_plan_attempt':record['logical_plan_attempt'],
             'business_revision_remaining':record['business_revision_remaining']}
    bindings={'history':history,
        'source_binding':{'source_migration':record['source_migration'],
                          'whitepaper_sha256':hashlib.sha256((EVIDENCE/'01_whitepaper.json').read_bytes()).hexdigest(),
                          'run_id':'showcase_supply_v1_fresh_20260928'},
        'retry_policy_binding':{'policy':record['retry_policy'],'sha256':record['retry_policy_sha256']},
        'remaining_business_attempts':record['business_revision_remaining']}
    return wp,candidate,findings,bindings


def response(wp, decision='repair'):
    return {'decision':decision,'reason':'Offline scripted diagnosis, not a real new reviewer opinion',
        'mechanism_checks':[{'mechanism_id':m['id'], 'status':'feasible' if decision=='accept' else 'blocked',
                             'walkthrough':'Inspect original business allocation and original mechanical evidence'}
                            for m in wp['seed_contract']['mechanisms']],
        'issues':[] if decision=='accept' else [{'finding':'Original carrier has insufficient actual write periods',
             'suggestion':'The original architect must reassess the failed obligation under unchanged seed and limits'}],
        'responsibility_targets':[{'unit_id':'u_tower_case','collection':'obligations','key':'L7-2',
              'finding_indices':[0],'reason':'The existing failed obligation owns this allocation'}] if decision=='repair' else []}


class PlanEscalationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVIDENCE.is_dir():raise RuntimeError('Explicit real A5 evidence directory is required: '+str(EVIDENCE))

    def setUp(self):self.wp,self.candidate,self.findings,self.bindings=a5_fixture()

    def diagnose(self,tracer,**overrides):
        kwargs={**self.bindings,**overrides}
        return review.diagnose_uncompiled_plan(self.wp,self.candidate,self.findings,tracer,**kwargs)

    def test_a5_bad_target_type_then_valid_repair_retains_original_input(self):
        before=deepcopy((self.wp,self.candidate,self.findings,self.bindings))
        bad=response(self.wp);bad['responsibility_targets'][0]['key']={'id':'L7-2'}
        good=response(self.wp);tracer=Tracer([bad,good]);receipt=self.diagnose(tracer)
        self.assertEqual((self.wp,self.candidate,self.findings,self.bindings),before)
        self.assertEqual(len(tracer.calls),2);self.assertEqual(receipt['decision'],'repair')
        self.assertTrue(receipt['author_allowed']);payload=review.validate_uncompiled_diagnostic(self.wp,self.candidate,receipt)
        self.assertEqual(payload['uncompiled_candidate'],self.candidate)
        self.assertEqual(payload['mechanical_status'],'rejected_uncompiled')
        self.assertNotIn('instance_execution',payload);self.assertNotIn('compiled_execution',payload)
        self.assertEqual(payload['remaining_business_attempts'],1)
        self.assertEqual(payload['history'],self.bindings['history'])
        first=deepcopy(tracer.calls[0]['payload']);second=deepcopy(tracer.calls[1]['payload'])
        correction=second.pop('protocol_feedback');self.assertEqual(first,second)
        self.assertEqual(correction['previous_response'],bad)
        self.assertEqual([e['path'] for e in correction['errors']],['responsibility_targets[0].key'])
        self.assertEqual([r['review'] for r in receipt['review']['protocol']['attempts']],[bad,good])
        self.assertTrue(all(c['step']=='council.blueprint_feasibility' and c['params']['retries']==3 for c in tracer.calls))
        self.assertEqual({(r['collection'],r['key']) for r in payload['responsibility_catalog']},
                         {('obligations','L7-2'),('objects','d3')})

    def test_unresolved_is_first_valid_negative_and_never_calls_author(self):
        tracer=Tracer([response(self.wp,'unresolved'),response(self.wp)])
        receipt=self.diagnose(tracer);self.assertEqual(len(tracer.calls),1)
        self.assertEqual(receipt['decision'],'unresolved');self.assertFalse(receipt['author_allowed'])
        review.validate_uncompiled_diagnostic(self.wp,self.candidate,receipt)

    def test_accept_cannot_override_original_mechanical_rejection(self):
        tracer=Tracer([response(self.wp,'accept'),response(self.wp)])
        receipt=self.diagnose(tracer);self.assertEqual(len(tracer.calls),1)
        self.assertEqual(receipt['decision'],'accept');self.assertFalse(receipt['author_allowed'])
        self.assertEqual(receipt['review']['input']['findings'],self.findings)
        self.assertNotIn('business_instance_plan',self.wp)
        review.validate_uncompiled_diagnostic(self.wp,self.candidate,receipt)

    def test_target_forgery_and_unrelated_existing_record_stop_without_retry(self):
        variants=[('unit_id','u_invented'),('collection','schema'),('key','L7-invented'),
                  ('key','L7-3'),('finding_indices',[99])]
        for key,value in variants:
            with self.subTest(key=key,value=value):
                bad=response(self.wp);bad['responsibility_targets'][0][key]=value
                tracer=Tracer([bad,response(self.wp)])
                with self.assertRaises(review.PlanDiagnosticBindingError) as got:self.diagnose(tracer)
                self.assertEqual(len(tracer.calls),1);self.assertEqual(got.exception.review,bad)
                self.assertIn('semantic_rejection',got.exception.protocol['attempts'][0])

    def test_unknown_mechanism_is_semantic_scope_not_format_retry(self):
        bad=response(self.wp);bad['mechanism_checks'][0]['mechanism_id']='invented'
        tracer=Tracer([bad,response(self.wp)])
        with self.assertRaises(review.PlanDiagnosticBindingError):self.diagnose(tracer)
        self.assertEqual(len(tracer.calls),1)

    def test_bad_shapes_have_precise_feedback_and_do_not_hash_untrusted_arrays(self):
        for key,value in [('unit_id',[]),('collection',{}),('key',[]),('finding_indices',[{}]),('reason',None)]:
            with self.subTest(key=key):
                bad=response(self.wp);bad['responsibility_targets'][0][key]=value
                tracer=Tracer([bad,response(self.wp,'unresolved')]);receipt=self.diagnose(tracer)
                self.assertEqual(receipt['decision'],'unresolved');self.assertEqual(len(tracer.calls),2)
                self.assertIn('responsibility_targets[0].'+key,
                    [e['path'] for e in receipt['review']['protocol']['attempts'][0]['errors']])

    def test_three_invalid_replies_stop_with_all_original_evidence(self):
        bad=response(self.wp);bad['responsibility_targets']=[];tracer=Tracer([bad]*4)
        with self.assertRaises(review.BlueprintReviewProtocolError) as got:self.diagnose(tracer)
        self.assertEqual(len(tracer.calls),3);self.assertEqual(got.exception.protocol['max_physical_requests'],9)
        self.assertEqual([r['review'] for r in got.exception.protocol['attempts']],[bad]*3)

    def test_budget_and_provider_execution_errors_propagate_without_protocol_retry(self):
        stopped={'__error__':'budget stopped','__error_metadata__':{'global_stop':True,'kind':'estimated_budget_limit'}}
        tracer=Tracer([{},stopped,response(self.wp)])
        with self.assertRaises(ExecutionStopped):self.diagnose(tracer)
        self.assertEqual(len(tracer.calls),2)
        failed={'__error__':'provider failed','__error_metadata__':{'kind':'http_status_500'}}
        tracer=Tracer([failed,response(self.wp)])
        with self.assertRaises(WorldBlueprintError):self.diagnose(tracer)
        self.assertEqual(len(tracer.calls),1)
        class Stopped:
            def check_global_stop(_):raise ExecutionStopped('already stopped',kind='call_limit')
            chat_json=staticmethod(forbidden)
        with self.assertRaises(ExecutionStopped):self.diagnose(Stopped())

    def test_validator_refuses_tampered_candidate_contract_and_envelope(self):
        receipt=self.diagnose(Tracer([response(self.wp)]))
        for key in ('candidate_sha256','blueprint_sha256','requirements_sha256','limits_sha256','seed_contract_sha256','review_input_hash'):
            tampered=deepcopy(receipt);tampered[key]='0'*64
            with self.subTest(key=key),self.assertRaises(review.PlanDiagnosticBindingError):
                review.validate_uncompiled_diagnostic(self.wp,self.candidate,tampered)
        tampered=deepcopy(receipt);tampered['author_allowed']=False
        with self.assertRaises(review.PlanDiagnosticBindingError):review.validate_uncompiled_diagnostic(self.wp,self.candidate,tampered)
        candidate=deepcopy(self.candidate);candidate['reason']='changed candidate'
        with self.assertRaises(review.PlanDiagnosticBindingError):review.validate_uncompiled_diagnostic(self.wp,candidate,receipt)
        for key in ('seed_contract','world_blueprint','supply_plan','delivery_target'):
            wp=deepcopy(self.wp);wp[key]['tampered']='changed contract'
            with self.subTest(key=key),self.assertRaises(WorldBlueprintError):
                review.validate_uncompiled_diagnostic(wp,self.candidate,receipt)

    def test_validator_checks_protocol_input_hash_and_first_valid_opinion(self):
        bad=response(self.wp);bad['responsibility_targets'][0]['key']=[]
        receipt=self.diagnose(Tracer([bad,response(self.wp)]))
        for number in (0,1):
            tampered=deepcopy(receipt);tampered['review']['protocol']['attempts'][number]['input_hash']='changed'
            with self.subTest(number=number),self.assertRaises(review.PlanDiagnosticBindingError):
                review.validate_uncompiled_diagnostic(self.wp,self.candidate,tampered)
        tampered=deepcopy(receipt)
        entry=tampered['review']['protocol']['attempts'][0]
        entry.update(review=response(self.wp,'unresolved'),errors=[])
        with self.assertRaises(review.PlanDiagnosticBindingError):review.validate_uncompiled_diagnostic(self.wp,self.candidate,tampered)

    def test_outer_exhaustion_wrapper_does_not_grant_all_unit_records(self):
        tracer=Tracer([response(self.wp)])
        receipt=review.diagnose_uncompiled_plan(self.wp,self.candidate,
            self.bindings['history']['terminal_findings'],tracer,**self.bindings)
        catalog=receipt['review']['input']['responsibility_catalog']
        self.assertEqual({(r['collection'],r['key']) for r in catalog},{('obligations','L7-2'),('objects','d3')})

    def test_original_business_allowance_is_exactly_one_and_required_before_call(self):
        for count in (0,2,3,True):
            with self.subTest(count=count),self.assertRaises(review.PlanDiagnosticBindingError):
                self.diagnose(Tracer([]),remaining_business_attempts=count)
        with self.assertRaises(review.PlanDiagnosticBindingError):self.diagnose(Tracer([]),source_binding={})

    def test_real_parser_tracer_retains_all_nine_physical_request_attempts(self):
        answers=[]
        for parsed in ({},[],response(self.wp)):
            answers+=['{','[',json.dumps(parsed,ensure_ascii=False)]
        physical=[]
        def transport(messages,**kwargs):physical.append(deepcopy(messages));return answers.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):receipt=self.diagnose(tracer)
            trace=[json.loads(line) for line in (Path(directory)/'llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
            prompts=[json.loads(line) for line in tracer.pfile.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(physical),9);self.assertEqual(len(prompts),3)
        self.assertEqual(len([r for r in trace if r['event']=='json_attempt']),9)
        self.assertEqual([r['output'] for r in prompts],[r['review'] for r in receipt['review']['protocol']['attempts']])
        review.validate_uncompiled_diagnostic(self.wp,self.candidate,receipt)


if __name__=='__main__':unittest.main()
