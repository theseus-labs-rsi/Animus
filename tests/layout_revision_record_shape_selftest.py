"""Business edits preserve complete replacements and the original author quota."""
import os, socket, sys, types
from pathlib import Path
os.environ.update(PYTHON_DOTENV_DISABLED='1',OPENAI_API_KEY='offline-layout-dummy',MODEL='offline',OPENAI_BASE_URL='http://127.0.0.1:9')
def forbidden(*args,**kwargs):raise AssertionError('network/provider forbidden')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:''
sys.modules['dotenv']=dotenv
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import openai
openai.OpenAI=openai.AsyncOpenAI=forbidden
import unittest
from copy import deepcopy
from unittest.mock import patch
from pipeline import instance_plan as ip,layout_revision as lr
from instance_plan_selftest import setup_plan,ACCEPT
from layout_revision_selftest import Trace


class RecordShapeTests(unittest.TestCase):
    def setUp(self):
        self.wp,self.plan,*_=setup_plan()
        self.unit=self.plan['units'][0];self.uid=self.unit['unit_id']
        self.slot=self.unit['observations'][0]['entity']

    def revision(self,collection,value,index=0):
        row=self.unit[collection][index]
        return {'decision':'repair','reason':'fixture complete transaction',
            'changes':[{'unit_id':self.uid,'collection':collection,'operation':'replace',
                        'key':ip._record_key(collection,row),'value':deepcopy(value)}]}

    def missing_type(self):
        position=next(i for i,row in enumerate(self.unit['objects']) if row['slot']==self.slot)
        return self.revision('objects',{},position)

    def valid(self):
        return self.revision('publications',dict(self.unit['publications'][-1],session=5),-1)

    def assert_shape(self,raw,collection):
        before=deepcopy((self.wp,self.plan,raw))
        with self.assertRaises(ip.PlanConflict) as caught:lr.apply(self.wp,self.plan,raw)
        shapes=[f for f in caught.exception.findings if f['code']=='business_revision_record_shape']
        self.assertTrue(shapes)
        self.assertTrue(any(self.uid in f['affected_ids'] and f['evidence']['collection']==collection for f in shapes))
        self.assertEqual((self.wp,self.plan,raw),before)
        return shapes

    def test_missing_object_type_is_owned_and_never_inherited(self):
        raw=self.missing_type();shapes=self.assert_shape(raw,'objects')
        self.assertIn(self.slot,shapes[0]['affected_ids'])
        self.assertNotIn('type',raw['changes'][0]['value'])

    def test_partial_event_replace_requires_type_and_session(self):
        raw=self.revision('events',{'participants':deepcopy(self.unit['events'][0]['participants'])})
        shapes=self.assert_shape(raw,'events')
        self.assertIn('type must be str',shapes[0]['evidence']['issues'])
        self.assertIn('session must be int',shapes[0]['evidence']['issues'])

    def test_record_and_nested_type_matrix_keeps_author(self):
        cases=[('objects','type',[]),('relations','from',{}),('events','participants',{'report':[]}),
               ('observations','sessions',[[0]]),('observations','mechanism',{}),
               ('obligations','carrier',{'entities':'report-slot'}),
               ('obligations','carrier',{'entities':[[self.slot]]}),('publications','session',True)]
        for collection,field,value in cases:
            with self.subTest(collection=collection,field=field,value=value):
                row=deepcopy(self.unit[collection][0]);row[field]=value
                self.assert_shape(self.revision(collection,row),collection)

    def test_missing_projection_reference_is_structured(self):
        raw={'decision':'repair','reason':'fixture remove referenced object',
             'changes':[{'unit_id':self.uid,'collection':'objects','operation':'remove','key':self.slot}]}
        self.assert_shape(raw,'observations')

    def test_unknown_fields_cannot_silently_disappear(self):
        row=dict(self.unit['objects'][0],made_up_fact='do not discard')
        shapes=self.assert_shape(self.revision('objects',row),'objects')
        self.assertTrue(any('unsupported fields' in issue for f in shapes for issue in f['evidence']['issues']))

    def test_one_remaining_attempt_stays_consumed_and_no_reader_called(self):
        raw=self.missing_type();trace=Trace([raw]);audit={};saved=[]
        with self.assertRaises(ip.PlanConflict):
            lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,checkpoint=lambda a:saved.append(deepcopy(a)))
        self.assertEqual([r['step'] for r in trace.calls],['council.instance_plan_business_repair'])
        self.assertEqual(len(audit['business_revisions']),1)
        attempt=audit['business_revisions'][0]
        self.assertEqual(attempt['attempt'],1);self.assertEqual(attempt['status'],'compiler_rejected')
        self.assertEqual(attempt['proposal'],raw);self.assertTrue(attempt['findings'])
        self.assertEqual(audit['status'],'business_repair_required');self.assertEqual(saved[-1],audit)
        with self.assertRaises(ip.PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1)
        self.assertEqual(len(trace.calls),1)

    def test_remaining_attempt_can_correct_against_unchanged_original(self):
        raw=self.missing_type();trace=Trace([raw,self.valid(),ACCEPT]);audit={}
        result=lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=2)
        self.assertEqual([r['step'] for r in trace.calls],['council.instance_plan_business_repair',
            'council.instance_plan_business_repair','council.blueprint_feasibility'])
        self.assertEqual([a['attempt'] for a in audit['business_revisions']],[1,2])
        self.assertEqual(trace.calls[1]['payload']['revision_feedback']['previous_response'],raw)
        self.assertEqual(trace.calls[1]['payload']['proposal'],self.plan)
        self.assertEqual(audit['status'],'accepted');self.assertIn('business_instance_plan',result)

    def test_unresolved_keeps_one_attempt_and_stops(self):
        trace=Trace([{'decision':'unresolved','reason':'fixture explicit refusal'}]);audit={}
        with self.assertRaises(ip.PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1)
        self.assertEqual(len(trace.calls),1);self.assertEqual(audit['business_revisions'][0]['status'],'unresolved')

    def test_shape_gate_does_not_early_run_joint_compiler(self):
        with patch.object(ip,'compile_plan',wraps=ip.compile_plan) as compile_call:
            lr.apply(self.wp,self.plan,self.valid())
        self.assertEqual(compile_call.call_count,1)

    def test_static_relation_can_omit_session(self):
        row=deepcopy(self.unit['relations'][0]);row.pop('session')
        revised,proposal=lr.apply(self.wp,self.plan,self.revision('relations',row))
        self.assertIn('business_instance_plan',revised)
        self.assertNotIn('session',proposal['units'][0]['relations'][0])

    def test_optional_observation_mechanism_is_not_invented(self):
        for unit in self.plan['units']:
            for row in unit['observations']:row.pop('mechanism',None)
        revised,proposal=lr.apply(self.wp,self.plan,self.valid())
        self.assertIn('business_instance_plan',revised)
        self.assertTrue(all('mechanism' not in row for unit in proposal['units'] for row in unit['observations']))

    def test_optional_carrier_field_is_not_invented_or_rejected_as_null(self):
        for supplied_null in (False,True):
            with self.subTest(supplied_null=supplied_null):
                plan=deepcopy(self.plan)
                # L7 uses its numeric field; a structural L2 carrier has no field.
                plan['units'][0]['obligations'].append({'id':'fieldless-carrier','line':'L2_relational',
                    'carrier':{'entities':['report-slot','company-slot']}})
                for obligation in plan['units'][0]['obligations']:
                    if obligation['line']!='L7_consolidation':
                        obligation['carrier'].pop('field',None)
                        if supplied_null:obligation['carrier']['field']=None
                # This case isolates canonical projection; the other cases use
                # the real joint compiler and its original allocation contract.
                with patch.object(ip,'compile_plan',return_value={'business_instance_plan':{}}):
                    revised,proposal=lr.apply(self.wp,plan,self.valid())
                self.assertIn('business_instance_plan',revised)
                for obligation in proposal['units'][0]['obligations']:
                    if obligation['line']!='L7_consolidation':
                        self.assertEqual('field' in obligation['carrier'],supplied_null)
                        self.assertIsNone(obligation['carrier'].get('field'))


if __name__=='__main__':unittest.main()
