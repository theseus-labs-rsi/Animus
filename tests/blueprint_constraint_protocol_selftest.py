"""Constraint-author protocol corrections retain original business limits."""
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

os.environ.update(PYTHON_DOTENV_DISABLED='1', OPENAI_API_KEY='offline-dummy',
                  OPENAI_BASE_URL='http://offline.invalid', MODEL='offline-dummy')
def forbidden(*args, **kwargs): raise AssertionError('Network/provider prohibited')
socket.socket.connect=forbidden; socket.socket.connect_ex=forbidden
socket.create_connection=forbidden; socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv'); dotenv.load_dotenv=lambda *a, **k:False
dotenv.find_dotenv=lambda *a, **k:''; sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('CONSTRAINT_TEST_REPO', Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import openai
openai.OpenAI=forbidden; openai.AsyncOpenAI=forbidden
import config
config.chat=forbidden
from pipeline import blueprint_feasibility as review
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.run import Tracer as ProductionTracer
from execution_control import ExecutionStopped
from seed_world_selftest import fixture
from blueprint_feasibility_selftest import opinion, Tracer


class ConstraintProtocolTests(unittest.TestCase):
    def setUp(self):
        self.wp, *_ = fixture()
        field=next(f for t in self.wp['world_blueprint']['entity_types'] for f in t['fields'] if f['name']=='review')
        field['states']=['draft','confirmed','recheck']
        self.change={'reason':'Original task requires confirmation after recheck', 'edits':[
            {'entity_type':'report','field':'review','remove':['states'],'set':{},
             'reason':'Reversible business process; original seed remains fixed'}]}

    def test_array_entity_or_field_is_corrected_before_original_review(self):
        for key in ('entity_type','field'):
            with self.subTest(key=key):
                bad=deepcopy(self.change); bad['edits'][0][key]=[bad['edits'][0][key]]
                tracer=Tracer([bad,self.change,opinion(self.wp)]); audit={}; before=deepcopy(self.wp)
                result=review.revise_constraints(self.wp,tracer,{'reason':'Original evidence'},audit)
                self.assertEqual(self.wp,before); self.assertEqual(result['seed_contract'],before['seed_contract'])
                self.assertEqual([c['step'] for c in tracer.calls],['council.blueprint_constraint_repair']*2+['council.blueprint_feasibility'])
                protocol=audit['author_protocol']
                self.assertEqual([r['response'] for r in protocol['attempts']],[bad,self.change])
                self.assertIn('edits[0].'+key,[e['path'] for e in protocol['attempts'][0]['errors']])
                first=tracer.calls[0]['payload']; second=deepcopy(tracer.calls[1]['payload'])
                feedback=second.pop('protocol_feedback')
                self.assertEqual(first,second); self.assertEqual(feedback['previous_response'],bad)
                self.assertEqual(audit['author'],self.change)
                self.assertTrue(all(c['params']['retries']==3 for c in tracer.calls))

    def test_missing_and_unhashable_shapes_receive_field_errors(self):
        malformed=[[],None,{'reason':'No schema','edits':{}}, {'reason':[], 'edits':[]}]
        for key,value in (('entity_type',None),('field',{}),('remove',[{}]),('set',[]),('reason',[])):
            bad=deepcopy(self.change); bad['edits'][0][key]=value; malformed.append(bad)
        bad=deepcopy(self.change); del bad['edits'][0]['field']; malformed.append(bad)
        for bad in malformed:
            with self.subTest(bad=bad):
                tracer=Tracer([bad,self.change,opinion(self.wp)]); audit={}
                review.revise_constraints(self.wp,tracer,{},audit)
                self.assertTrue(audit['author_protocol']['attempts'][0]['errors'])
                self.assertEqual(len(tracer.calls),3)

    def test_valid_empty_edits_is_negative_and_not_retried(self):
        negative={'reason':'No supported repair within original business scope','edits':[]}
        tracer=Tracer([negative,self.change,opinion(self.wp)]); audit={}
        with self.assertRaisesRegex(WorldBlueprintError,'no supported constraint repair'):
            review.revise_constraints(self.wp,tracer,{},audit)
        self.assertEqual(len(tracer.calls),1); self.assertEqual(audit['author'],negative)
        self.assertEqual(audit['author_protocol']['attempts'][0]['errors'],[])

    def test_semantic_scope_and_valid_negative_reviewer_do_not_retry_author(self):
        before=deepcopy(self.wp)
        for kind in ('unknown','forbidden','duplicate','absent_constraint'):
            bad=deepcopy(self.change)
            if kind=='unknown':bad['edits'][0]['field']='unknown'
            elif kind=='forbidden':bad['edits'][0]['set']={'kind':'text'}
            elif kind=='duplicate':bad['edits']*=2
            else:bad['edits'][0]['remove']=['range']
            tracer=Tracer([bad,self.change]); audit={}
            with self.assertRaises(WorldBlueprintError):review.revise_constraints(self.wp,tracer,{},audit)
            self.assertEqual(len(tracer.calls),1); self.assertEqual(self.wp,before)
        tracer=Tracer([self.change,opinion(self.wp,'unresolved'),opinion(self.wp)])
        with self.assertRaisesRegex(WorldBlueprintError,'remains unresolved'):
            review.revise_constraints(self.wp,tracer,{}, {})
        self.assertEqual(len(tracer.calls),2); self.assertEqual(self.wp,before)

    def test_three_invalid_protocol_rounds_stop_with_all_replies(self):
        bad=deepcopy(self.change); bad['edits'][0]['entity_type']=[]
        tracer=Tracer([bad]*4); audit={}; before=deepcopy(self.wp)
        with self.assertRaises(review.BlueprintConstraintProtocolError) as got:
            review.revise_constraints(self.wp,tracer,{},audit)
        self.assertEqual(len(tracer.calls),3); self.assertEqual(self.wp,before)
        self.assertEqual(audit['author_protocol'],got.exception.protocol)
        self.assertEqual(audit['author_protocol']['max_physical_requests'],9)
        self.assertTrue(all(r['response']==bad and r['errors'] for r in audit['author_protocol']['attempts']))

    def test_provider_and_global_failures_are_not_protocol_feedback(self):
        for sentinel in ({'__error__':'provider failed'},
                         {'__error__':'budget stopped','__error_metadata__':{'global_stop':True,'kind':'estimated_budget_limit'}}):
            tracer=Tracer([sentinel,self.change]); audit={}
            with self.assertRaises((WorldBlueprintError,ExecutionStopped)):
                review.revise_constraints(self.wp,tracer,{},audit)
            self.assertEqual(len(tracer.calls),1)
            self.assertEqual(audit['author_protocol']['attempts'][0]['status'],'execution_failed')
        class Stopped:
            def check_global_stop(_):raise ExecutionStopped('already stopped',kind='call_limit')
            chat_json=staticmethod(forbidden)
        with self.assertRaises(ExecutionStopped):review.revise_constraints(self.wp,Stopped(),{}, {})
        class Raising:
            def chat_json(*a,**k):raise OSError('audit write failed')
        with self.assertRaisesRegex(OSError,'audit write failed'):
            review.revise_constraints(self.wp,Raising(),{}, {})

    def test_actual_parser_counts_nine_physical_author_requests_no_semantic_resubmit(self):
        bad=deepcopy(self.change); bad['edits'][0]['field']=[]
        negative={'reason':'Original scope has no justified repair','edits':[]}
        answers=[]
        for raw in (bad,bad,negative):answers+=['{','[',json.dumps(raw)]
        physical=[]
        def transport(messages,**kwargs):physical.append(deepcopy(messages)); return answers.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory))); audit={}
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):
                with self.assertRaisesRegex(WorldBlueprintError,'no supported constraint repair'):
                    review.revise_constraints(self.wp,tracer,{},audit)
            self.assertEqual(tracer.n,3)
            trace=[json.loads(x) for x in tracer.pfile.with_name('llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(physical),9); self.assertEqual(len([r for r in trace if r['event']=='json_attempt']),9)
        self.assertEqual(len(audit['author_protocol']['attempts']),3)

    def test_syntax_exhaustion_does_not_open_more_protocol_rounds(self):
        physical=[]
        def transport(*a,**k):physical.append(1); return '{'
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):
                with self.assertRaises(WorldBlueprintError):review.revise_constraints(self.wp,tracer,{}, {})
        self.assertEqual(len(physical),3)


if __name__=='__main__':unittest.main()
