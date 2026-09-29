"""Global-deficit routing and durable bounded author attempts, offline only."""
import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['OPENAI_API_KEY']='offline-dummy'
os.environ['MODEL']='offline-dummy'
import socket
import sys
import types
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import threading
import unittest
from unittest.mock import patch

def forbidden(*args,**kwargs):raise AssertionError('Network/provider forbidden')
socket.socket.connect=forbidden
socket.socket.connect_ex=forbidden
socket.create_connection=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:''
sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('ROUTING_TEST_REPO',str(Path(__file__).resolve().parents[1])))
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import config
config.chat=forbidden;config.chat_json=forbidden
if os.environ.get('ROUTING_TEST_MODULE'):
    spec=importlib.util.spec_from_file_location('pipeline.plan_transactions',os.environ['ROUTING_TEST_MODULE'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from pipeline.plan_transactions import repair, _route_findings
from pipeline.instance_plan import PlanConflict, compile_plan, finding
from instance_plan_selftest import setup_plan

class StopBoundary(BaseException):pass

class RoutingTests(unittest.TestCase):
    def setup(self):
        wp,plan,*_=setup_plan()
        for uid in ('support-b','support-c'):
            plan['units'].append(dict(unit_id=uid,business_purpose='Independent support scope',objects=[],events=[],relations=[],observations=[],obligations=[],publications=[],depends_on=[]))
        return wp,plan

    def test_event_deficit_has_one_author_and_original_compiler_accepts_repair(self):
        wp,plan=self.setup();next(e for e in wp['world_blueprint']['event_types'] if e['id']=='reviewed')['min_count']=2
        before=deepcopy(plan);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);calls.append(p)
                self.assertEqual(p['assigned_instance_additions']['events'],{'reviewed':1})
                self.assertEqual(p['unit']['unit_id'],'report-case')
                event=dict(p['unit']['events'][-1],slot='report-case-review-one',session=1,caused_by='revision-zero')
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'add','key':event['slot'],'value':event}]}
        audit={};result=repair(wp,plan,{},Tracer(),audit)
        compile_plan(wp,result);self.assertEqual(len(calls),1);self.assertEqual(plan,before)

    def test_zero_relation_instances_are_assigned_and_repaired(self):
        wp,plan=self.setup();row=plan['units'][0]['relations'].pop();calls=[]
        row['slot']='report-case-issuer-slot'
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);calls.append(p)
                self.assertEqual(p['assigned_instance_additions']['relations'],{'issued_by':1})
                return {'decision':'repair','changes':[{'unit_id':p['unit']['unit_id'],'collection':'relations','operation':'add','key':row['slot'],'value':row}]}
        compile_plan(wp,repair(wp,plan,{},Tracer(),{}));self.assertEqual(len(calls),1)

    def test_object_and_exact_object_deficits_do_not_call_alias_author(self):
        for exact in (False,True):
            with self.subTest(exact=exact):
                wp,plan=self.setup();spec=next(t for t in wp['world_blueprint']['entity_types'] if t['id']=='report');spec['count']=2
                if exact:spec['cardinality_policy']='exact'
                calls=[]
                class Tracer:
                    def chat_json(_,step,messages,**kwargs):
                        calls.append(step);p=json.loads(messages[-1]['content'])
                        self.assertEqual(step,'council.instance_plan_unit_repair')
                        self.assertEqual(p['assigned_instance_additions']['objects'],{'report':1})
                        return {'decision':'repair','changes':[{'unit_id':p['unit']['unit_id'],'collection':'objects','operation':'add','key':'report-case-extra','value':{'type':'report'}}]}
                compile_plan(wp,repair(wp,plan,{},Tracer(),{}));self.assertEqual(len(calls),1)

    def test_subtype_repair_retains_line_count(self):
        wp,plan=self.setup();plan['units'][0]['obligations'][0].pop('subtype');calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);calls.append(p)
                self.assertEqual(p['assigned_subtype_minimum'],{'L7_consolidation':{'native_trend':1}})
                obligation=dict(p['unit']['obligations'][0],subtype='native_trend')
                return {'decision':'repair','changes':[{'unit_id':p['unit']['unit_id'],'collection':'obligations','operation':'replace','key':obligation['id'],'value':obligation}]}
        result=repair(wp,plan,{},Tracer(),{});compile_plan(wp,result)
        self.assertEqual(len(calls),1);self.assertEqual(sum(len(u['obligations']) for u in result['units']),1)

    def test_alias_reduction_can_precede_existing_object_deficit(self):
        wp,plan=self.setup();next(t for t in wp['world_blueprint']['entity_types'] if t['id']=='report')['count']=2
        next(t for t in wp['world_blueprint']['entity_types'] if t['id']=='company')['cardinality_policy']='exact'
        plan['units'][1]['objects']=[{'slot':'publisher-copy','type':'company'}]
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                calls.append(step)
                if step=='council.instance_resources':return {'decision':'repair','aliases':{'publisher-copy':'company-slot'}}
                p=json.loads(messages[-1]['content'])
                return {'decision':'repair','changes':[{'unit_id':p['unit']['unit_id'],'collection':'objects','operation':'add','key':'report-case-extra','value':{'type':'report'}}]}
        result=repair(wp,plan,{},Tracer(),{});compile_plan(wp,result)
        self.assertEqual(calls,['council.instance_resources','council.instance_plan_unit_repair'])

    def test_assigned_object_capacity_is_enforced_with_original_three_attempt_limit(self):
        wp,plan=self.setup();wp['supply_plan']['world_limits']['max_world_entities']=2
        plan['units'][0]['events'][-1]['session']=4;audit={};calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);calls.append(p)
                event=dict(p['unit']['events'][-1],session=3)
                return {'decision':'repair','changes':[
                    {'unit_id':'report-case','collection':'events','operation':'replace','key':event['slot'],'value':event},
                    {'unit_id':'report-case','collection':'objects','operation':'add','key':'report-case-extra-'+str(len(calls)),'value':{'type':'report'}}]}
        with self.assertRaises(PlanConflict) as caught:repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(caught.exception.findings[0]['code'],'resource_capacity')
        self.assertEqual(len(calls),3);self.assertEqual(len(audit['edit_attempts']['report-case']),3);self.assertNotIn('drafts',audit)

    def test_shared_type_deficit_not_broadcast_and_capacity_not_duplicated(self):
        wp,plan=self.setup();plan['units'][1]['events']=[dict(plan['units'][0]['events'][-1],slot='support-b-review',session=5)]
        next(e for e in wp['world_blueprint']['event_types'] if e['id']=='reviewed')['min_count']=3
        issue=finding('missing_declared_instances','minimum',['reviewed'],evidence={'required':3,'actual':2})
        routed=_route_findings(wp,plan,[issue])
        self.assertEqual(sum(bool(rows) for rows in routed[6].values()),1)
        self.assertEqual(sum(a['events'].get('reviewed',0) for a in routed[7].values()),1)
        self.assertEqual(sum(routed[9].values()),wp['supply_plan']['world_limits']['max_world_entities']-2)

    def test_unroutable_error_creates_no_empty_draft_or_call(self):
        wp,plan=self.setup();audit={}
        with patch('pipeline.instance_plan.compile_plan',side_effect=PlanConflict([finding('future_constraint','unsupported',['absent-type'])])):
            with self.assertRaises(PlanConflict) as caught:repair(wp,plan,{},types.SimpleNamespace(chat_json=forbidden),audit)
        self.assertEqual(caught.exception.findings[0]['code'],'repair_routing')
        self.assertNotIn('drafts',audit);self.assertNotIn('edit_attempts',audit)

    def test_unresolved_provider_and_exception_attempts_are_durable(self):
        for result,status in [({'decision':'unresolved','reason':'original constraints conflict'},'unresolved'),({'__error__':'offline failure'},'provider_error'),(ValueError('synthetic transport error'),'call_failed')]:
            with self.subTest(status=status):
                wp,plan=self.setup();plan['units'][0]['events'][-1]['session']=4;audit={};snapshots=[]
                class Tracer:
                    def chat_json(_,step,messages,**kwargs):
                        self.assertEqual(snapshots[-1]['edit_attempts']['report-case'][0]['status'],'requested')
                        if isinstance(result,Exception):raise result
                        return result
                with self.assertRaises((PlanConflict,RuntimeError,ValueError)):repair(wp,plan,{},Tracer(),audit,checkpoint=lambda a:snapshots.append(deepcopy(a)))
                self.assertEqual(audit['edit_attempts']['report-case'][0]['status'],status)
                self.assertEqual(snapshots[-1]['edit_attempts']['report-case'][0]['status'],status)

    def test_relation_schedule_reaches_original_relation_author(self):
        wp,plan=self.setup();relation=plan['units'][0]['relations'].pop();plan['units'][1]['relations']=[relation]
        issue=finding('relation_schedule','binding',['review-three','issued_by'],evidence={'relation':'issued_by','source':'report-slot','target':'company-slot'})
        routed=_route_findings(wp,plan,[issue])
        self.assertIn('report-case',routed[-1][0]['responsible_units']);self.assertIn('support-b',routed[-1][0]['responsible_units'])

    def test_persisted_unresolved_cannot_be_overwritten_or_retried(self):
        wp,plan=self.setup();plan['units'][0]['events'][-1]['session']=4
        audit={'edit_attempts':{'report-case':[{'attempt':1,'status':'unresolved','proposal':{'decision':'unresolved','reason':'original conflict'}}]}}
        before=deepcopy(audit)
        with self.assertRaises(PlanConflict) as caught:repair(wp,plan,{},types.SimpleNamespace(chat_json=forbidden),audit)
        self.assertEqual(caught.exception.findings[0]['code'],'plan_unresolved');self.assertEqual(audit,before)

    def test_successful_parallel_proposal_survives_sibling_unresolved(self):
        wp,plan=self.setup();plan['units'][0]['events'][-1]['session']=4
        plan['units'][1]['publications']=[{'fact_slot':'review-three','channel':'复核记录','session':2}]
        ready=threading.Event();audit={};snapshots=[]
        def checkpoint(a):
            snapshots.append(deepcopy(a))
            if 'report-case' in a.get('unit_proposals',{}):ready.set()
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);uid=p['unit']['unit_id']
                if uid=='support-b':
                    self.assertTrue(ready.wait(3));return {'decision':'unresolved','reason':'independent business conflict'}
                event=dict(p['unit']['events'][-1],session=3)
                return {'decision':'repair','changes':[{'unit_id':uid,'collection':'events','operation':'replace','key':event['slot'],'value':event}]}
        with self.assertRaises(PlanConflict):repair(wp,plan,{},Tracer(),audit,checkpoint=checkpoint)
        self.assertIn('report-case',audit['unit_proposals']);self.assertNotIn('drafts',audit)
        self.assertTrue(any('report-case' in s.get('unit_proposals',{}) for s in snapshots))

    def test_real_incident_reaches_one_original_author_without_provider(self):
        evidence=os.environ.get('ROUTING_REAL_EVIDENCE')
        if not evidence:self.skipTest('Original incident evidence not supplied')
        evidence=Path(evidence);wp=json.loads((evidence/'01_whitepaper.json').read_text(encoding='utf-8'))
        original_audit=json.loads((evidence/'01_instance_plan_audit.json').read_text(encoding='utf-8'))
        proposal=original_audit['attempts'][0]['proposal'];before=deepcopy(proposal);calls=[];audit={}
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);calls.append(p);raise StopBoundary()
        with self.assertRaises(StopBoundary):repair(wp,proposal,{},Tracer(),audit)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['assigned_instance_additions']['relations'],{'hero_investigation':1})
        self.assertTrue(any(e['type']=='hero_investigation' for e in calls[0]['unit']['relations']))
        self.assertEqual(before,proposal);self.assertNotIn('drafts',audit)

if __name__=='__main__':unittest.main()
