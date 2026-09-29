"""Offline reviewer protocol corrections; negative opinions are never retried."""
import os
import sys
import types
import socket
import json
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['OPENAI_API_KEY']='offline-dummy'
os.environ['MODEL']='offline-dummy'
def forbidden(*args,**kwargs):raise AssertionError('Network/provider prohibited')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
socket.create_connection=forbidden;socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:''
sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('REVIEW_TEST_REPO',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import openai
openai.OpenAI=forbidden;openai.AsyncOpenAI=forbidden
import config
config.chat=forbidden
if os.environ.get('REVIEW_TEST_MODULE'):
    import importlib.util
    spec=importlib.util.spec_from_file_location('pipeline.blueprint_feasibility',os.environ['REVIEW_TEST_MODULE'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from pipeline import blueprint_feasibility as review
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.run import Tracer as ProductionTracer
from execution_control import ExecutionStopped
from seed_world_selftest import fixture
from blueprint_feasibility_selftest import opinion,Tracer


class ReviewerProtocolTests(unittest.TestCase):
    def setUp(self):self.wp,*_=fixture()

    def test_multiple_invalid_then_accept_preserves_original_input_and_replies(self):
        before=deepcopy(self.wp);first=opinion(self.wp);first.pop('mechanism_checks')
        second=opinion(self.wp);second['mechanism_checks'][0]['status']=[]
        valid=opinion(self.wp);tracer=Tracer([first,second,valid])
        result=review.assess(self.wp,tracer,feedback={'finding':'original evidence'})
        self.assertEqual(self.wp,before);self.assertEqual(len(tracer.calls),3)
        self.assertEqual(result['decision'],'accept')
        self.assertEqual([a['review'] for a in result['protocol']['attempts']],[first,second,valid])
        original=deepcopy(tracer.calls[0]['payload'])
        self.assertEqual(result['input'],original);self.assertEqual(result['input_hash'],review.digest(original))
        for i,call in enumerate(tracer.calls[1:],1):
            payload=deepcopy(call['payload']);feedback=payload.pop('protocol_feedback')
            self.assertEqual(payload,original)
            self.assertEqual(feedback['previous_response'],[first,second][i-1])
            self.assertTrue(feedback['errors'])
        self.assertTrue(all(c['params']['retries']==3 for c in tracer.calls))

    def test_parsed_nonobjects_and_unhashable_nested_values_get_precise_feedback(self):
        malformed=[[],None,'review']
        for path in ('decision','status','mechanism_id','issues','reason'):
            raw=opinion(self.wp)
            if path in ('status','mechanism_id'):raw['mechanism_checks'][0][path]=[]
            else:raw[path]=[] if path!='issues' else [{'finding':[],'suggestion':None}]
            malformed.append(raw)
        for raw in malformed:
            with self.subTest(raw=raw):
                tracer=Tracer([raw,opinion(self.wp,'unresolved')]);result=review.assess(self.wp,tracer)
                self.assertEqual(result['decision'],'unresolved');self.assertEqual(len(tracer.calls),2)
                errors=tracer.calls[1]['payload']['protocol_feedback']['errors']
                self.assertTrue(errors);self.assertTrue(all(e['path'] and e['message'] for e in errors))

    def test_missing_duplicate_and_unknown_mechanisms_cannot_be_accepted(self):
        for mode in ('missing','duplicate','unknown'):
            raw=opinion(self.wp)
            if mode=='missing':raw['mechanism_checks']=[]
            elif mode=='duplicate':raw['mechanism_checks']*=2
            else:raw['mechanism_checks'][0]['mechanism_id']='invented'
            tracer=Tracer([raw,opinion(self.wp,'repair')]);result=review.assess(self.wp,tracer)
            self.assertEqual(result['decision'],'repair');self.assertEqual(len(tracer.calls),2)
            self.assertIn('mechanism_checks.mechanism_id',[e['path'] for e in result['protocol']['attempts'][0]['errors']])

    def test_contradictory_accept_can_be_corrected_to_negative_and_stops(self):
        for mode in ('blocked','issues'):
            raw=opinion(self.wp)
            if mode=='blocked':raw['mechanism_checks'][0]['status']='blocked'
            else:raw['issues']=[{'finding':'Actual conflict','suggestion':'Retain original scope'}]
            tracer=Tracer([raw,opinion(self.wp,'repair'),opinion(self.wp)])
            result=review.assess(self.wp,tracer)
            self.assertEqual(result['decision'],'repair');self.assertEqual(len(tracer.calls),2)
            self.assertIn('negative opinion',tracer.calls[1]['payload']['protocol_feedback']['instruction'])

    def test_valid_negative_opinions_return_immediately_without_searching_for_accept(self):
        for decision in ('repair','unresolved'):
            raw=opinion(self.wp,decision);tracer=Tracer([raw,opinion(self.wp)])
            result=review.assess(self.wp,tracer)
            self.assertEqual(result['review'],raw);self.assertEqual(len(tracer.calls),1)
            self.assertEqual(len(result['protocol']['attempts']),1)

    def test_repeated_invalid_exhausts_three_protocol_slots_with_evidence(self):
        raw=opinion(self.wp);raw['decision']=[];tracer=Tracer([raw]*4)
        with self.assertRaises(review.BlueprintReviewProtocolError) as got:review.assess(self.wp,tracer)
        self.assertEqual(len(tracer.calls),3)
        self.assertEqual(len(got.exception.protocol['attempts']),3)
        self.assertEqual(got.exception.protocol['max_physical_requests'],9)
        self.assertTrue(all(a['review']==raw and a['errors'] for a in got.exception.protocol['attempts']))

    def test_provider_sentinel_and_exception_are_not_protocol_retries(self):
        sentinel={'__error__':'provider unavailable','__error_metadata__':{'kind':'http_status_500','attempts':3}}
        tracer=Tracer([sentinel,opinion(self.wp)])
        with self.assertRaises(WorldBlueprintError) as got:review.assess(self.wp,tracer)
        self.assertEqual(got.exception.raw,sentinel);self.assertEqual(len(tracer.calls),1)
        calls=[]
        class Raising:
            def chat_json(_,step,messages,**kwargs):calls.append(1);raise RuntimeError('provider failure')
        with self.assertRaisesRegex(RuntimeError,'provider failure'):review.assess(self.wp,Raising())
        self.assertEqual(len(calls),1)

    def test_global_stop_on_second_request_does_not_trigger_third(self):
        stopped={'__error__':'budget stopped','__error_metadata__':{'global_stop':True,'kind':'estimated_budget_limit'}}
        tracer=Tracer([{},stopped,opinion(self.wp)])
        with self.assertRaises(ExecutionStopped) as got:review.assess(self.wp,tracer)
        self.assertEqual(got.exception.kind,'estimated_budget_limit');self.assertEqual(len(tracer.calls),2)
        self.assertEqual(got.exception.raw,stopped)

    def test_existing_global_stop_prevents_first_request(self):
        class Stopped:
            def check_global_stop(_):raise ExecutionStopped('already stopped',kind='call_limit')
            chat_json=staticmethod(forbidden)
        with self.assertRaises(ExecutionStopped):review.assess(self.wp,Stopped())

    def test_real_json_parser_and_tracer_bound_three_by_three_to_nine(self):
        answers=[]
        for parsed in ({'decision':[]},{'decision':'accept'},opinion(self.wp)):
            answers+=['{','[',json.dumps(parsed)]
        physical=[]
        def transport(messages,**kwargs):physical.append(deepcopy(messages));return answers.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):
                result=review.assess(self.wp,tracer)
            prompts=[json.loads(x) for x in tracer.pfile.read_text(encoding='utf-8').splitlines()]
            trace=[json.loads(x) for x in tracer.pfile.with_name('llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(physical),9);self.assertEqual(len(prompts),3)
        self.assertEqual(result['decision'],'accept')
        self.assertEqual([p['output'] for p in prompts],[a['review'] for a in result['protocol']['attempts']])
        self.assertEqual(len([r for r in trace if r['event']=='json_attempt']),9)
        self.assertTrue(all(p['params']['retries']==3 for p in prompts))
        self.assertNotIn('protocol_feedback',result['input'])

    def test_real_parser_budget_rejection_never_dispatches_another_physical_request(self):
        admission=[];physical=[]
        def transport(messages,**kwargs):
            admission.append(1)
            if len(admission)==4:raise ExecutionStopped('fixture budget exhausted',kind='estimated_budget_limit')
            physical.append(1)
            return '{' if len(physical)<3 else json.dumps({'decision':[]})
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(ExecutionStopped):
                review.assess(self.wp,tracer)
            self.assertEqual(tracer.n,2)
        self.assertEqual(len(admission),4);self.assertEqual(len(physical),3)

    def test_exhausted_json_syntax_does_not_restart_three_more_protocol_rounds(self):
        physical=[]
        def transport(messages,**kwargs):physical.append(1);return '{'
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(WorldBlueprintError):
                review.assess(self.wp,tracer)
            self.assertEqual(tracer.n,1)
        self.assertEqual(len(physical),3)


if __name__=='__main__':unittest.main()
