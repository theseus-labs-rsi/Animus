"""A business revision updates meaning and dependent carriers in one transaction."""
from copy import deepcopy
from pathlib import Path
import json,os,socket,sys,unittest
from unittest.mock import patch
os.environ.update(PYTHON_DOTENV_DISABLED='1',OPENAI_API_KEY='offline',MODEL='offline')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import config
from pipeline import instance_plan as ip,layout_revision as lr
from instance_plan_selftest import setup_plan,ACCEPT


class Trace:
    def __init__(self,replies):self.replies=list(replies);self.calls=[]
    def chat_json(self,step,messages,**kw):
        self.calls.append({'step':step,'payload':json.loads(messages[-1]['content'])})
        if not self.replies:raise AssertionError('Unexpected dispatch')
        return deepcopy(self.replies.pop(0))


class Tests(unittest.TestCase):
    def setUp(self):
        for obj,name in ((socket.socket,'connect'),(config,'chat'),(config,'chat_json')):
            guard=patch.object(obj,name,side_effect=AssertionError('No network'));guard.start();self.addCleanup(guard.stop)
        self.wp,self.plan,*_=setup_plan()
        fields=next(t['fields'] for t in self.wp['world_blueprint']['entity_types'] if t['id']=='report')
        original=deepcopy(next(f for f in fields if f['name']=='capital'));fields.append(dict(original,name='funding'))
        for event in self.wp['world_blueprint']['event_types']:
            for effect in list(event['effect_fields']):
                if effect['field']=='capital':event['effect_fields'].append(dict(effect,field='funding'))
        self.plan['units'][0]['observations'].append({'entity':'report-slot','field':'funding','sessions':[0,2],'mechanism':'Synonymous financial measure'})
        self.plan['units'][0]['obligations'][0]['carrier']['field']='funding'
        self.raw={'decision':'repair','reason':'Funding and capital denote the same financial measure',
            'field_aliases':[{'entity_type':'report','source_field':'funding','target_field':'capital','reason':'Keep the original seed term'}],
            'changes':[],'cohort_changes':[]}

    def test_alias_and_references_commit_together_without_mutating_raw(self):
        original=deepcopy((self.wp,self.plan,self.raw))
        result,proposal=lr.apply(self.wp,self.plan,self.raw)
        fields=next(t['fields'] for t in result['world_blueprint']['entity_types'] if t['id']=='report')
        self.assertNotIn('funding',[f['name'] for f in fields])
        self.assertEqual(proposal['units'][0]['obligations'][0]['carrier']['field'],'capital')
        observed=proposal['units'][0]['observations']
        self.assertEqual(len(observed),1);self.assertEqual(observed[0]['sessions'],list(range(6)))
        for event in result['world_blueprint']['event_types']:
            self.assertEqual(len(event['effect_fields']),len({(e['role'],e['field']) for e in event['effect_fields']}))
        self.assertEqual((self.wp,self.plan,self.raw),original)

    def test_authored_synonym_uses_seed_canonical_declaration_atomically(self):
        raw=deepcopy(self.raw);raw['field_aliases'][0].update(source_field='capital',target_field='funding')
        before=deepcopy((self.wp,self.plan))
        result,proposal=lr.apply(self.wp,self.plan,raw)
        self.assertEqual(proposal['units'][0]['obligations'][0]['carrier']['field'],'capital')
        fields=next(t['fields'] for t in result['world_blueprint']['entity_types'] if t['id']=='report')
        self.assertNotIn('funding',[f['name'] for f in fields])
        self.assertEqual((self.wp,self.plan),before)

    def test_alias_and_dependent_plan_edit_share_the_same_commit(self):
        raw=deepcopy(self.raw)
        original=self.plan['units'][0]['publications'][-1]
        raw['changes']=[{'unit_id':'report-case','collection':'publications','operation':'replace',
            'selector':{k:original[k] for k in ('fact_slot','channel','session')},
            'value':dict(original,session=5)}]
        result,proposal=lr.apply(self.wp,self.plan,raw)
        self.assertEqual(proposal['units'][0]['publications'][-1]['session'],5)
        self.assertEqual(result['business_instance_plan']['units'][0]['obligations'][0]['carrier']['field'],'capital')

    def test_joint_compiler_still_rejects_extra_demand(self):
        bad=deepcopy(self.raw);bad['changes']=[{'unit_id':'report-case','collection':'obligations','operation':'add','key':'extra',
             'value':dict(self.plan['units'][0]['obligations'][0],id='extra')}]
        with self.assertRaises(ip.PlanConflict) as error:lr.apply(self.wp,self.plan,bad)
        self.assertIn('demand_slots',[f['code'] for f in error.exception.findings])

    def test_create_routes_business_repair_through_single_transaction(self):
        feedback={'findings':[ip.finding('business_repair','Synonymous financial measures',kind='business')]}
        trace=Trace([self.raw,ACCEPT]);audit={}
        result=ip.create(self.wp,trace,feedback,audit,initial_proposal=self.plan,repair_required=True,max_attempts=3)
        self.assertEqual([r['step'] for r in trace.calls],['council.instance_plan_business_repair','council.blueprint_feasibility'])
        self.assertEqual(audit['status'],'accepted')
        self.assertEqual(len(audit['business_revisions']),1)
        self.assertEqual(result['blueprint_revisions'][-1]['author'],self.raw)

    def test_repeated_business_rejection_preserves_opinion_and_stops(self):
        negative=deepcopy(ACCEPT);negative.update(decision='repair',issues=[{'finding':'Meaning remains ambiguous','suggestion':'Define the term'}])
        trace=Trace([self.raw,negative,self.raw,negative]);audit={}
        with self.assertRaises(ip.PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit)
        self.assertEqual(len(trace.calls),4)
        self.assertEqual(len(audit['business_revisions']),2)
        self.assertEqual(audit['business_revisions'][-1]['business_review']['decision'],'repair')

    def test_compiler_feedback_retains_previous_proposal(self):
        malformed=deepcopy(self.raw);malformed['field_aliases'][0]['source_field']='unknown'
        trace=Trace([malformed,self.raw,ACCEPT]);audit={}
        result=lr.revise(self.wp,self.plan,{},trace,audit)
        self.assertEqual(trace.calls[1]['payload']['revision_feedback']['previous_response'],malformed)
        self.assertEqual(result['business_instance_plan']['units'][0]['obligations'][0]['carrier']['field'],'capital')

    def test_provider_error_is_not_a_semantic_retry(self):
        trace=Trace([{'__error__':'auth'}]);audit={}
        with self.assertRaises(RuntimeError):lr.revise(self.wp,self.plan,{},trace,audit)
        self.assertEqual(len(trace.calls),1)

    def test_real_revision_revalidation_does_not_dispatch_new_author(self):
        trace=Trace([ACCEPT]);audit={}
        lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,initial_revision=self.raw)
        self.assertEqual([x['step'] for x in trace.calls],['council.blueprint_feasibility'])
        self.assertTrue(audit['business_revisions'][0]['revalidated_from_parent'])
        self.assertEqual(audit['business_revisions'][0]['proposal'],self.raw)

    def declaration_transaction(self):
        raw=deepcopy(self.raw)
        declaration=deepcopy(next(e for e in self.wp['world_blueprint']['event_types'] if e['id']=='revision'))
        declaration.update(id='followup_revision',label='A declared later revision',min_count=1)
        raw['blueprint_additions']={'event_types':[declaration]}
        raw['changes']=[{'unit_id':'report-case','collection':'events','operation':'add','key':'revision-four',
            'value':{'slot':'revision-four','type':'followup_revision','participants':{'report':'report-slot'},'session':4}}]
        return raw

    def test_missing_business_process_commits_schema_and_facts_atomically(self):
        raw=self.declaration_transaction();before=deepcopy((self.wp,self.plan,raw))
        result,plan=lr.apply(self.wp,self.plan,raw)
        declared={e['id']:e for e in result['world_blueprint']['event_types']}
        self.assertIn('followup_revision',declared)
        self.assertEqual(next(e for e in plan['units'][0]['events'] if e['slot']=='revision-four')['type'],'followup_revision')
        self.assertEqual(result['seed_contract'],self.wp['seed_contract'])
        self.assertEqual(result['supply_plan']['final_allocation'],self.wp['supply_plan']['final_allocation'])
        self.assertEqual((self.wp,self.plan,raw),before)

    def test_schema_extension_cannot_replace_original_seed_declaration(self):
        raw=self.declaration_transaction();raw['blueprint_additions']['event_types'][0]['id']='revision'
        with self.assertRaises(ip.PlanConflict) as error:lr.apply(self.wp,self.plan,raw)
        self.assertEqual(error.exception.findings[0]['code'],'business_revision_declaration')

    def test_schema_extension_invalid_reference_rolls_back_entire_transaction(self):
        raw=self.declaration_transaction();raw['blueprint_additions']['event_types'][0]['effect_fields'][0]['field']='undeclared'
        before=deepcopy((self.wp,self.plan))
        with self.assertRaises(ip.PlanConflict) as error:lr.apply(self.wp,self.plan,raw)
        self.assertEqual(error.exception.findings[0]['code'],'blueprint_compilation')
        self.assertEqual((self.wp,self.plan),before)

    def test_schema_and_dependent_fact_use_original_joint_compiler(self):
        raw=self.declaration_transaction();raw['changes'][0]['value']['session']=10
        with self.assertRaises(ip.PlanConflict):lr.apply(self.wp,self.plan,raw)

    def test_schema_extension_does_not_open_seed_or_resource_edits(self):
        raw=self.declaration_transaction();raw['blueprint_additions']['temporal_model']={'n_sessions':10}
        with self.assertRaises(ip.PlanConflict) as error:lr.apply(self.wp,self.plan,raw)
        self.assertEqual(error.exception.findings[0]['code'],'business_revision_shape')

    def test_original_review_still_decides_new_declaration_meaning(self):
        raw=self.declaration_transaction();trace=Trace([raw,ACCEPT]);audit={}
        result=lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1)
        reviewed=trace.calls[1]['payload']['blueprint']
        self.assertIn('followup_revision',[e['id'] for e in reviewed['event_types']])
        self.assertEqual(result['blueprint_revisions'][-1]['author'],raw)


if __name__=='__main__':unittest.main()
