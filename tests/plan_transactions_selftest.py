"""Allocation identity, atomic edits and real-state observation regressions."""
import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ.setdefault('OPENAI_API_KEY','offline')
os.environ.setdefault('MODEL','offline')
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from copy import deepcopy
import unittest
import threading
from unittest.mock import patch
from pipeline.plan_transactions import apply_aliases,derived_dependencies,repair
from pipeline.instance_plan import PlanConflict
from instance_plan_selftest import setup_plan

class TransactionTests(unittest.TestCase):
    def test_valid_edit_with_remaining_compiler_conflict_returns_to_same_author(self):
        import json
        wp,plan,*_=setup_plan();plan['units'][0]['events'][-1]['session']=4
        original=deepcopy(plan);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                payload=json.loads(messages[-1]['content']);calls.append(payload)
                event=deepcopy(payload['unit']['events'][-1]);event['session']=5 if len(calls)==1 else 3
                if len(calls)==2:
                    self.assertEqual(payload['edit_feedback']['compiler_findings'][0]['code'],'causal_schedule')
                    self.assertEqual(payload['edit_feedback']['previous_response']['changes'][0]['value']['session'],5)
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':event}]}
        audit={};result=repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(len(calls),2);self.assertEqual(plan,original)
        self.assertEqual(result['units'][0]['events'][-1]['session'],3)
        self.assertEqual(audit['edit_attempts']['report-case'][0]['joint_findings'][0]['code'],'causal_schedule')

    def test_remaining_compiler_error_keeps_original_author_allowance(self):
        wp,plan,*_=setup_plan();plan['units'][0]['events'][-1]['session']=4
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                import json
                p=json.loads(messages[-1]['content']);calls.append(p)
                event=deepcopy(p['unit']['events'][-1]);event['session']+=1
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':event}]}
        audit={}
        with self.assertRaises(PlanConflict):repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(len(calls),3);self.assertEqual(len(audit['edit_attempts']['report-case']),3)
        # The first session-5 edit reaches joint compilation. Sessions beyond
        # the frozen calendar are now rejected inside the same local author
        # transaction instead of producing two additional malformed drafts.
        self.assertEqual(len(audit['drafts']),1)
        self.assertEqual([a['status'] for a in audit['edit_attempts']['report-case']],
                         ['joint_rejected','locally_rejected','locally_rejected'])

    def test_create_cannot_reset_exhausted_compiler_author_allowance(self):
        from pipeline.instance_plan import create
        wp,plan,*_=setup_plan();plan['units'][0]['events'][-1]['session']=4
        plan['units'] += [{'unit_id':name,'business_purpose':'Independent unchanged scope',
            'objects':[],'events':[],'relations':[],'observations':[],'obligations':[],'publications':[],'depends_on':[]}
            for name in ('support-a','support-b')]
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                import json
                p=json.loads(messages[-1]['content']);calls.append(p)
                event=deepcopy(p['unit']['events'][-1]);event['session']+=1
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':event}]}
        audit={}
        with patch('pipeline.blueprint_feasibility.assess') as review:
            with self.assertRaises(PlanConflict):create(wp,Tracer(),{},audit,initial_proposal=plan)
        self.assertEqual(len(calls),3);self.assertEqual(review.call_count,0)
        self.assertEqual(len(audit['transactions']),1)

    def test_natural_record_selector_retains_identity_and_can_update_public_time(self):
        from pipeline.instance_plan import apply_repair
        wp,plan,*_=setup_plan();u=plan['units'][0];publication=u['publications'][0]
        original=deepcopy(plan);updated=deepcopy(publication);updated['session']+=1
        changes={'decision':'repair','changes':[
            {'unit_id':u['unit_id'],'collection':'publications','operation':'replace',
             'key':'fact_slot|'+publication['fact_slot']+'|channel|'+publication['channel'],'value':updated},
            {'unit_id':u['unit_id'],'collection':'observations','operation':'replace',
             'selector':{'entity':u['observations'][0]['entity'],'field':u['observations'][0]['field']},
             'value':dict(u['observations'][0],sessions=[0,1,2,3])}]}
        result=apply_repair(plan,changes)
        self.assertEqual(plan,original);self.assertEqual(result['units'][0]['publications'][0],updated)
        self.assertEqual(result['units'][0]['objects'],original['units'][0]['objects'])
        ambiguous=deepcopy(plan);ambiguous['units'][0]['publications'].append(updated)
        with self.assertRaises(PlanConflict):apply_repair(ambiguous,changes)

    def test_invalid_edit_returns_exact_selector_feedback_before_joint_compile(self):
        wp,plan,*_=setup_plan();plan['units'][0]['events'][-1]['session']=4
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                import json
                payload=json.loads(messages[-1]['content']);calls.append(payload)
                if len(calls)==1:
                    return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'absent','value':{}}]}
                self.assertEqual(payload['edit_feedback']['compiler_findings'][0]['code'],'repair_target')
                self.assertEqual(payload['edit_catalog']['events'][-1]['selector'],{'slot':'review-three'})
                event=deepcopy(payload['unit']['events'][-1]);event['session']=3
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','selector':{'slot':'review-three'},'value':event}]}
        audit={};result=repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(len(calls),2);self.assertEqual(result['units'][0]['events'][-1]['session'],3)
        self.assertEqual(len(audit['edit_attempts']['report-case']),2)

    def setUp(self):
        self.wp,self.plan,*_=setup_plan()
        self.plan['units'].append({'unit_id':'second','business_purpose':'Same publisher issues a later record',
            'objects':[{'slot':'publisher-copy','type':'company'}],'events':[],
            'relations':[{'slot':'second-issuer','type':'issued_by','from':'report-slot','to':'publisher-copy','session':0}],
            'observations':[],'obligations':[],'publications':[],'depends_on':[]})

    def test_alias_references_all_original_records_survive(self):
        before=deepcopy(self.plan)
        merged=apply_aliases(self.plan,{'publisher-copy':'company-slot'},self.wp)
        self.assertEqual(self.plan,before)
        self.assertEqual(merged['units'][1]['relations'][0]['to'],'company-slot')
        self.assertEqual(merged['units'][1]['objects'],[])
        self.assertEqual(merged['units'][0]['events'],before['units'][0]['events'])
        self.assertEqual(merged['units'][0]['obligations'],before['units'][0]['obligations'])

    def test_cross_type_alias_rejected(self):
        with self.assertRaises(PlanConflict):apply_aliases(self.plan,{'publisher-copy':'report-slot'},self.wp)

    def test_returned_empty_aliases_get_actual_count_feedback_without_repeat(self):
        self.wp['supply_plan']['world_limits']['max_world_entities']=2
        self.plan['units'][1]['relations']=[]
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                import json
                p=json.loads(messages[-1]['content']);calls.append(p)
                self.assertEqual(p['actual_object_count'],3)
                self.assertEqual(p['mechanical_feedback']['compiler_findings'][0]['code'],'resource_cap')
                self.assertEqual(p['mechanical_feedback']['compiler_findings'][0]['evidence']['actual_object_count'],3)
                return {'decision':'repair','aliases':{'publisher-copy':'company-slot'}}
        audit={}
        result=repair(self.wp,self.plan,{},Tracer(),audit,resource_resume={'decision':'repair','aliases':{}})
        self.assertEqual(len(calls),1)
        self.assertTrue(audit['resource_attempts'][0]['revalidated_from_parent'])
        self.assertEqual(sum(len(u['objects']) for u in result['units']),2)

    def test_real_resource_unresolved_is_not_mechanically_retried(self):
        self.wp['supply_plan']['world_limits']['max_world_entities']=2
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                calls.append(step);return {'decision':'unresolved','reason':'Distinct roles exceed fixed capacity'}
        with self.assertRaises(PlanConflict) as error:repair(self.wp,self.plan,{},Tracer(),{})
        self.assertEqual(error.exception.findings[0]['code'],'plan_unresolved')
        self.assertEqual(len(calls),1)

    def test_compiler_correction_precedes_the_single_business_review(self):
        from pipeline.instance_plan import create
        from instance_plan_selftest import ACCEPT
        wp,plan,*_=setup_plan()
        plan['units'][0]['events'][-1]['session']=4
        plan['units'] += [{'unit_id':name,'business_purpose':'Existing bounded supporting scope',
            'objects':[],'events':[],'relations':[],'observations':[],'obligations':[],'publications':[],'depends_on':[]}
            for name in ('support-a','support-b')]
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                import json
                p=json.loads(messages[-1]['content']);calls.append(p)
                self.assertIn('causal_schedule',[f['code'] for f in p['findings']])
                event=deepcopy(p['unit']['events'][-1]);event['session']=4 if len(calls)==1 else 3
                return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':event}]}
        audit={}
        with patch('pipeline.blueprint_feasibility.assess',return_value=ACCEPT) as review:
            result=create(wp,Tracer(),{},audit,max_attempts=1,initial_proposal=plan)
        self.assertEqual(len(calls),2);self.assertEqual(review.call_count,1)
        self.assertEqual(result['business_instance_plan']['units'][0]['events'][-1]['session'],3)
        self.assertEqual(len(audit['transactions']),1)
        self.assertEqual(len(audit['transactions'][0]['edit_attempts']['report-case']),2)
        self.assertEqual(plan['units'][0]['events'][-1]['session'],4)

    def test_repeated_link_records_return_to_original_plan_transaction(self):
        from pipeline.instance_plan import compile_plan
        wp,plan,*_=setup_plan()
        original_minimum=wp['world_blueprint']['relation_types'][0]['min_count']
        for name in ('issuer-copy-a','issuer-copy-b'):
            plan['units'][0]['relations'].append(dict(plan['units'][0]['relations'][0],slot=name))
        with self.assertRaises(PlanConflict) as error:
            compile_plan(wp,derived_dependencies(plan))
        self.assertEqual(wp['world_blueprint']['relation_types'][0]['min_count'],original_minimum)
        self.assertEqual(sum(f['code']=='relation_compilation' for f in error.exception.findings),2)
        self.assertEqual(len(plan['units'][0]['relations']),3)

    def test_alias_cycle_rejected(self):
        with self.assertRaises(PlanConflict):apply_aliases(self.plan,{'publisher-copy':'company-slot','company-slot':'publisher-copy'},self.wp)

    def test_edit_key_defines_record_identity_without_duplicate_payload(self):
        from pipeline.instance_plan import apply_repair
        before=deepcopy(self.plan)
        result=apply_repair(self.plan,{'decision':'repair','changes':[{'unit_id':'second','collection':'objects','operation':'add','key':'new-report','value':{'type':'report'}}]})
        self.assertEqual(result['units'][1]['objects'][-1],{'slot':'new-report','type':'report'})
        self.assertEqual(self.plan,before)

    def test_conflicting_explicit_record_identity_is_rejected(self):
        from pipeline.instance_plan import apply_repair
        with self.assertRaises(PlanConflict):
            apply_repair(self.plan,{'decision':'repair','changes':[{'unit_id':'second','collection':'objects','operation':'add','key':'new-report','value':{'slot':'different','type':'report'}}]})

    def test_publication_add_uses_its_actual_natural_key(self):
        from pipeline.instance_plan import apply_repair
        row={'fact_slot':'review-three','channel':'复核记录','session':4}
        result=apply_repair(self.plan,{'decision':'repair','changes':[{'unit_id':'second','collection':'publications','operation':'add','key':'author-edit-label','value':row}]})
        self.assertEqual(result['units'][1]['publications'],[row])
        with self.assertRaises(PlanConflict):
            apply_repair(result,{'decision':'repair','changes':[{'unit_id':'second','collection':'publications','operation':'add','key':'another-label','value':row}]})

    def test_allocation_cannot_erase_original_minimum(self):
        self.wp['world_blueprint']['entity_types'][0]['count']=2
        with self.assertRaises(PlanConflict):apply_aliases(self.plan,{'publisher-copy':'company-slot'},self.wp)

    def test_dependency_is_actual_reference_projection(self):
        self.plan['units'][1]['depends_on']=['invented']
        projected=derived_dependencies(self.plan)
        self.assertEqual(projected['units'][1]['depends_on'],[])
        self.assertEqual(self.plan['units'][1]['depends_on'],['invented'])

    def test_real_causal_parent_still_orders_its_dependent_writer(self):
        self.plan['units'][1]['events']=[{'slot':'later-review','type':'reviewed','participants':{'report':'report-slot'},
            'session':3,'caused_by':'revision-two'}]
        projected=derived_dependencies(self.plan)
        self.assertEqual(projected['units'][1]['depends_on'],['report-case'])

    def test_unit_scope_violation_preserves_parent(self):
        self.plan['units'][0]['events'][-1]['session']=4
        original=deepcopy(self.plan)
        class Tracer:
            def chat_json(self,*args,**kwargs):
                return {'decision':'repair','changes':[{'unit_id':'second','collection':'events','operation':'remove','key':'review-three'}]}
        with self.assertRaises(PlanConflict):repair(self.wp,self.plan,{},Tracer(),{})
        self.assertEqual(self.plan,original)

    def test_parallel_edits_are_jointly_compiled_from_one_base(self):
        from pipeline.instance_plan import compile_plan
        wp,plan,*_=setup_plan()
        plan['units'][0]['events'][-1]['session']=4
        plan['units'].append({'unit_id':'second','business_purpose':'Publish the shared review',
            'objects':[],'events':[],'relations':[],'observations':[],'obligations':[],
            'publications':[{'fact_slot':'review-three','channel':'复核记录','session':2}],'depends_on':[]})
        barrier=threading.Barrier(2);calls=[]
        class Tracer:
            def chat_json(self,step,messages,**kwargs):
                import json
                payload=json.loads(messages[1]['content']);uid=payload['unit']['unit_id']
                calls.append(uid);barrier.wait(timeout=2)
                if uid=='report-case':
                    event=deepcopy(payload['unit']['events'][-1]);event['session']=3
                    changes=[{'unit_id':uid,'collection':'events','operation':'replace','key':'review-three','value':event}]
                else:
                    changes=[{'unit_id':uid,'collection':'publications','operation':'remove','key':'review-three|复核记录|2'},
                        {'unit_id':uid,'collection':'publications','operation':'add','key':'review-three|复核记录|3','value':{'fact_slot':'review-three','channel':'复核记录','session':3}}]
                return {'decision':'repair','changes':changes}
        before=deepcopy(plan);audit={};result=repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(set(calls),{'report-case','second'})
        self.assertEqual(plan,before)
        self.assertEqual(result['units'][1]['depends_on'],[])
        compile_plan(wp,result)

if __name__=='__main__':unittest.main()
