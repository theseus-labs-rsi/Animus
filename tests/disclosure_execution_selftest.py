"""Disclosure execution boundary tests with a real Tracer and local fake provider.

Uses only small synthetic fixtures; no saved run artifacts or network.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from pipeline import disclosure as fixed
from pipeline.run import Tracer
from disclosure_selftest import data, raw_plan


def run(module, outputs, feedback=None):
    wp,ws,task=data(); calls=[]; values=list(outputs)
    def chat_json(messages, **kwargs):
        calls.append({'messages':deepcopy(messages),'params':deepcopy(kwargs)})
        out=values.pop(0)
        if isinstance(out,BaseException): raise out
        return deepcopy(out)
    with tempfile.TemporaryDirectory(prefix='real-tracer-offline-') as tmp:
        with patch.dict(sys.modules,{'config':SimpleNamespace(STRUCTURE_MODEL='offline-cheap',chat_json=chat_json)}):
            result=module.author_plan(wp,ws,Tracer(SimpleNamespace(dir=Path(tmp))),task,feedback)
        logged=[json.loads(line) for line in (Path(tmp)/'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    return result,calls,logged,ws


class Tests(unittest.TestCase):
    def setUp(self):
        self.guard=patch.object(socket.socket,'connect',side_effect=AssertionError('No network'))
        self.guard.start();self.addCleanup(self.guard.stop)
        self.good=raw_plan(data()[1])

    def test_after_provider_sentinel_stops_once_preserves_real_raw(self):
        for error in (TimeoutError('offline timeout'),ValueError('offline invalid JSON')):
            result,calls,logs,ws=run(fixed,[error,self.good])
            self.assertEqual(len(calls),1)
            self.assertEqual(result['status'],'error')
            self.assertEqual(result['attempts'][0]['status'],'execution_error')
            self.assertEqual(result['raw_output'],logs[0]['output'])
            self.assertFalse(ws.disclosure)

    def test_after_still_repairs_returned_shape_once(self):
        result,calls,logs,ws=run(fixed,[{'records':[]},self.good])
        self.assertEqual(len(calls),2)
        self.assertTrue(all(row['ok'] for row in logs))
        self.assertEqual(result['status'],'ready')
        self.assertFalse(fixed.validate_plan(ws))

    def test_error_on_second_dispatch_is_retained_without_third(self):
        result,calls,logs,ws=run(fixed,[{'records':[]},TimeoutError('second failed'),self.good])
        self.assertEqual(len(calls),2)
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['attempts'][1]['raw_output'],logs[1]['output'])
        self.assertEqual(result['attempts'][1]['status'],'execution_error')
        self.assertFalse(ws.disclosure)

    def feedback(self):
        return {'version':'original-world-semantics/v9','status':'failed',
            'raw_output':{'reason':'完整业务理由保持原字节'*40,'issues':[{'finding':'获取安排需要解释。'}]},
            'locations':{'issues':[{'source_value':'完整原证据'}]},'repair_targets':{'disclosure':True},
            'custom_business_extension':{'detail':'未知扩展字段必须保留'},
            'messages':[{'role':'user','content':'DUPLICATED_WORLD_SENTINEL'*100}],
            'input_snapshot':{'world':'DUPLICATED_WORLD_SENTINEL'*100},
            'binding':{'audit_hash':'binding'},'call':{'step':'world.semantic_review'},
            'previous':{'version':'original-world-semantics/v8','raw_output':{'reason':'保留上轮实际意见'},
                        'input_snapshot':{'world':'PREVIOUS_WORLD_SENTINEL'},'messages':[]}}

    def test_feedback_projection_preserves_semantics_and_binds_full_original(self):
        feedback=self.feedback()
        result,calls,logs,ws=run(fixed,[self.good],feedback)
        self.assertEqual(result['status'],'ready')
        payload=json.loads(calls[0]['messages'][1]['content'])
        projected=payload['feedback']
        self.assertEqual(projected['raw_output'],feedback['raw_output'])
        self.assertEqual(projected['locations'],feedback['locations'])
        self.assertEqual(projected['custom_business_extension'],feedback['custom_business_extension'])
        self.assertEqual(projected['previous']['raw_output'],feedback['previous']['raw_output'])
        self.assertNotIn('DUPLICATED_WORLD_SENTINEL',json.dumps(calls[0]['messages']))
        self.assertNotIn('PREVIOUS_WORLD_SENTINEL',json.dumps(calls[0]['messages']))
        self.assertEqual(ws.disclosure['source_inputs']['feedback'],feedback)
        self.assertEqual(ws.disclosure['binding']['feedback_hash'],fixed._hash(feedback))
        self.assertFalse(fixed.validate_plan(ws))

    def test_feedback_source_and_projected_reason_tampering_rejected(self):
        result,calls,logs,ws=run(fixed,[self.good],self.feedback());saved=deepcopy(ws.disclosure)
        for mode in ('source','projection'):
            ws.disclosure=deepcopy(saved)
            if mode=='source': ws.disclosure['source_inputs']['feedback']['messages'][0]['content']='edited archive'
            else: ws.disclosure['author_input']['feedback']['raw_output']['reason']='edited reason'
            ws.disclosure['binding']['messages_hash']=fixed._hash(fixed._messages(ws.disclosure['author_input']))
            ws.disclosure['plan_hash']=fixed._hash({k:v for k,v in ws.disclosure.items() if k!='plan_hash'})
            self.assertTrue(fixed.validate_plan(ws))

    def test_unknown_feedback_not_silently_reduced_and_old_version_not_migrated(self):
        feedback={'custom_business':'preserve','messages':['could be meaningful unknown format']}
        result,calls,logs,ws=run(fixed,[self.good],feedback)
        self.assertEqual(json.loads(calls[0]['messages'][1]['content'])['feedback'],feedback)
        self.assertFalse(fixed.validate_plan(ws))
        wp,oldws,task=data();oldws.disclosure=fixed.compile_plan(wp,oldws,self.good,task)
        oldws.disclosure['version']='original-public-disclosure/v2'
        oldws.disclosure['plan_hash']=fixed._hash({k:v for k,v in oldws.disclosure.items() if k!='plan_hash'})
        self.assertTrue(fixed.validate_plan(oldws))


if __name__ == '__main__':
    unittest.main(verbosity=2)
