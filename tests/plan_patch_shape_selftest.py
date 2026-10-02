"""Complete local edits and settled checkpoint replay; no providers or sockets."""
import os
import sys
import types
import socket
from pathlib import Path
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['OPENAI_API_KEY']='offline-dummy'
os.environ['MODEL']='offline-dummy'
def forbidden(*a,**k):raise AssertionError('Network/provider prohibited')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden;socket.create_connection=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:''
sys.modules['dotenv']=dotenv
import importlib.util
ROOT=Path(os.environ.get('PATCH_TEST_REPO',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import config
config.chat=forbidden;config.chat_json=forbidden
if os.environ.get('PATCH_TEST_MODULE'):
    spec=importlib.util.spec_from_file_location('pipeline.plan_transactions',os.environ['PATCH_TEST_MODULE'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from pipeline.plan_transactions import validate_unit_patch,repair,_restore_batch,derived_dependencies,_route_findings
from pipeline.instance_plan import PlanConflict,compile_plan,apply_repair,finding
from instance_plan_selftest import setup_plan

class Boundary(BaseException):pass

class PatchShapeTests(unittest.TestCase):
    def setup(self):
        wp,plan,*_=setup_plan();return wp,plan

    def change(self,c,key,value,op='replace'):
        return {'decision':'repair','changes':[{'unit_id':'report-case','collection':c,'operation':op,'key':key,'value':value}]}

    def test_incomplete_event_replacement_rejected_atomically_with_original_schema(self):
        wp,plan=self.setup();before=deepcopy(plan)
        with self.assertRaises(PlanConflict) as got:
            validate_unit_patch(wp,plan,self.change('events','review-three',{'participants':{'report':'report-slot'}}),'report-case',{},6)
        f=got.exception.findings[0]
        self.assertEqual(f['affected_ids'],['report-case','review-three'])
        self.assertEqual(f['evidence']['original_record']['session'],3)
        self.assertEqual(f['evidence']['original_declared_schema']['id'],'reviewed')
        self.assertIn('session',f['evidence']['required_fields']);self.assertEqual(plan,before)

    def test_required_fields_across_record_collections(self):
        wp,plan=self.setup()
        cases=[('objects','report-slot',{},None),('relations','issuer-slot',{'type':'issued_by','to':'company-slot'},None),
            ('events','review-three',{'type':'reviewed','session':3},None),
            ('observations','report-slot|capital',{'entity':'report-slot','field':'capital'},None),
            ('obligations','trend-obligation',{'line':'L7_consolidation'},None),
            ('publications','review-three|复核记录|3',{'fact_slot':'review-three','channel':'复核记录','session':True},None),
            ('depends_on',None,['unknown'],None)]
        for c,key,value,_ in cases:
            with self.subTest(collection=c),self.assertRaises(PlanConflict):
                validate_unit_patch(wp,plan,self.change(c,key,value),'report-case',{},6)
        plan['cohorts']=[{'id':'cohort','basis':'shared business','members':['report-slot','company-slot']}]
        raw={'decision':'repair','cohort_changes':[{'operation':'replace','key':'cohort','value':{'members':['report-slot','company-slot']}}]}
        with self.assertRaises(PlanConflict):validate_unit_patch(wp,plan,raw,'report-case',{'cohort':'report-case'},6)

    def test_bad_nested_values_remain_structured_local_errors(self):
        wp,plan=self.setup()
        cases=[self.change('events','review-three',{'type':'reviewed','participants':{'report':{}},'session':3}),
            self.change('relations','issuer-slot',{'type':'issued_by','from':[],'to':'company-slot','session':0}),
            self.change('observations','report-slot|capital',{'entity':'report-slot','field':'capital','sessions':[{}]}),
            self.change('obligations','trend-obligation',{'line':{},'carrier':{'entities':[{}]}})]
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(PlanConflict):validate_unit_patch(wp,plan,raw,'report-case',{},6)

    def test_valid_complete_replacement_not_merged(self):
        wp,plan=self.setup();old=plan['units'][0]['events'][-1]
        # Omission of optional caused_by means it is removed; no implicit merge.
        value={'type':old['type'],'participants':deepcopy(old['participants']),'session':1}
        candidate=validate_unit_patch(wp,plan,self.change('events',old['slot'],value),'report-case',{},6)
        event=candidate['units'][0]['events'][-1]
        self.assertNotIn('caused_by',event);self.assertEqual(event['session'],1)
        self.assertEqual(event['slot'],'review-three');self.assertEqual(old['session'],3)

    def test_partial_then_complete_consumes_original_local_slots(self):
        wp,plan=self.setup();plan['units'][0]['events'][-1]['session']=4
        requests=[];snapshots=[];audit={}
        class Tracer:
            def chat_json(_,step,messages,**kw):
                p=json.loads(messages[-1]['content']);requests.append(p)
                if len(requests)==1:return self.change('events','review-three',{'participants':{'report':'report-slot'}})
                self.assertEqual(p['unit']['events'][-1]['session'],4)
                self.assertEqual(p['edit_feedback']['compiler_findings'][0]['code'],'repair_record_shape')
                return self.change('events','review-three',dict(p['unit']['events'][-1],session=3))
        result=repair(wp,plan,{},Tracer(),audit,checkpoint=lambda a:snapshots.append(deepcopy(a)))
        compile_plan(wp,result);self.assertEqual(len(requests),2)
        self.assertEqual([a['status'] for a in audit['edit_attempts']['report-case']],['locally_rejected','locally_compiled'])
        self.assertEqual(audit['drafts'][0]['iteration'],1)
        self.assertTrue(any(s['edit_attempts']['report-case'][0]['status']=='locally_rejected' for s in snapshots if s.get('edit_attempts')))

    def test_three_attempt_cap_no_mechanical_invention(self):
        wp,plan=self.setup();plan['units'][0]['events'][-1]['session']=4;calls=[];audit={}
        class Tracer:
            def chat_json(_,step,messages,**kw):
                calls.append(1)
                raw=self.change('events','review-three',{'participants':{'report':'report-slot'}});raw['reason']=str(len(calls));return raw
        with self.assertRaises(PlanConflict):repair(wp,plan,{},Tracer(),audit)
        self.assertEqual(len(calls),3);self.assertNotIn('drafts',audit)
        self.assertEqual(plan['units'][0]['events'][-1]['session'],4)

    def test_unknown_values_not_silently_projected_and_dependency_noop_rejected(self):
        wp,plan=self.setup()
        raw=self.change('objects','report-case-extra',{'type':'report','capital':100},'add')
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(wp,plan,raw,'report-case',{},6)
        self.assertIn('unsupported fields',str(got.exception))
        plan['units'].append({'unit_id':'other','business_purpose':'support','objects':[],'relations':[],'events':[],'observations':[],'obligations':[],'publications':[],'depends_on':[]})
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(wp,plan,self.change('depends_on',None,['other']),'report-case',{},6)
        self.assertEqual(got.exception.findings[0]['code'],'repair_no_change')

    def test_global_identity_collision_and_shared_deletion_keep_author(self):
        wp,plan=self.setup()
        raw=self.change('events','report-slot',{'type':'revision','participants':{'report':'report-slot'},'session':1},'add')
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(wp,plan,raw,'report-case',{},6)
        self.assertTrue(any(f['code']=='repair_identity_collision' for f in got.exception.findings))
        other={'unit_id':'other','business_purpose':'support','objects':[],'relations':[],'events':[],'observations':[],
               'obligations':[],'publications':[{'fact_slot':'review-three','channel':'复核记录','session':4}],'depends_on':[]}
        plan['units'].append(other)
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(wp,plan,self.change('events','review-three',None,'remove'),'report-case',{},6)
        self.assertEqual(got.exception.findings[0]['affected_ids'],['report-case','review-three'])
        self.assertEqual(got.exception.findings[0]['code'],'repair_shared_delete')

    def test_spare_capacity_fair_and_jointly_bounded(self):
        wp,plan=self.setup()
        for uid in ('support-a','support-b'):
            plan['units'].append({'unit_id':uid,'business_purpose':'independent support','objects':[],'relations':[],'events':[],
                'observations':[],'obligations':[],'publications':[],'depends_on':[]})
        findings=[finding('business_constraint','needs supporting object',[uid],kind='business') for uid in ('report-case','support-a','support-b')]
        route=_route_findings(wp,plan,findings);capacity=route[9]
        self.assertEqual(capacity,{'report-case':2,'support-a':2,'support-b':2})
        merged=deepcopy(plan)
        for uid,limit in capacity.items():
            raw={'decision':'repair','changes':[{'unit_id':uid,'collection':'objects','operation':'add','key':uid+'-extra-'+str(i),
                'value':{'type':'report'}} for i in range(limit)]}
            validate_unit_patch(wp,plan,raw,uid,{},limit);merged=apply_repair(merged,raw)
            raw['changes'].append({'unit_id':uid,'collection':'objects','operation':'add','key':uid+'-overflow','value':{'type':'report'}})
            with self.assertRaises(PlanConflict):validate_unit_patch(wp,plan,raw,uid,{},limit)
        self.assertEqual(sum(len(u['objects']) for u in merged['units']),8)

    def test_parallel_author_namespaces_cannot_claim_same_new_identity(self):
        wp,plan=self.setup();plan['units'].append({'unit_id':'report','business_purpose':'support','objects':[],'relations':[],'events':[],
                'observations':[],'obligations':[],'publications':[],'depends_on':[]})
        raw=self.change('objects','report-case-extra',{'type':'report'},'add')
        validate_unit_patch(wp,plan,raw,'report-case',{},3)
        raw['changes'][0]['unit_id']='report'
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(wp,plan,raw,'report',{},3)
        self.assertEqual(got.exception.findings[0]['code'],'repair_identity_namespace')

    def test_routing_record_shape_keeps_owner_with_malformed_references(self):
        wp,plan=self.setup();unit=plan['units'][0]
        unit['relations'][0].pop('from');unit['relations'][0]['to']={}
        unit['events'][0]['participants']={'report':{}}
        unit['observations'][0]['entity']={}
        unit['obligations'][0]['carrier']['entities']=[{}]
        unit['publications'][0]['fact_slot']={}
        issue=finding('record_shape','missing relation from',['report-case','issuer-slot'])
        result=_route_findings(wp,plan,[issue])
        self.assertEqual(result[-1][0]['responsible_units'],['report-case'])

class ActualPaidReplyReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('PATCH_REAL_EVIDENCE')
        if not path:raise unittest.SkipTest('Real paid incident evidence not supplied')
        path=Path(path)
        def read(name):return json.loads((path/name).read_text(encoding='utf-8-sig'))
        from pipeline.initial_plan_recovery import _prepared_whitepaper
        cls.wp=_prepared_whitepaper(read('01_whitepaper.json'))
        cls.proposal=read('01_instance_plan_audit.json')['attempts'][0]['proposal']
        cls.audit=read('01_initial_plan_recovery.json')['transaction']

    def test_original_response_fails_locally_with_real_slot_and_schema(self):
        audit=deepcopy(self.audit);raw=audit['unit_proposals']['u_institution_track'];base=audit['drafts'][0]['candidate']
        with self.assertRaises(PlanConflict) as got:validate_unit_patch(self.wp,base,raw,'u_institution_track',{},5)
        event=next(f for f in got.exception.findings if f['evidence']['collection']=='events')
        self.assertEqual(event['affected_ids'],['u_institution_track','ev_arm3'])
        self.assertIsNotNone(event['evidence']['original_declared_schema'])
        self.assertTrue(any('unsupported fields' in str(f) for f in got.exception.findings))

    def test_all_paid_responses_replayed_before_provider_and_static_conflict_retained(self):
        audit=deepcopy(self.audit);before=deepcopy(audit)
        base,iteration,findings,replies,pending,receipt=_restore_batch(self.wp,self.proposal,audit)
        self.assertEqual(iteration,2)
        self.assertEqual(receipt['joint_drafts_consumed'],2)
        self.assertEqual(receipt['max_attempts_per_author'],3)
        self.assertEqual(receipt['scheduler_version'],'original-author-slots/v1')
        self.assertEqual(set(pending),{'u_institution_track','u_missing_case'})
        self.assertTrue(any(f['code']=='scalar_binding_conflict' for f in pending['u_missing_case']))
        self.assertEqual(set(receipt['replayed_units']),{'u_seal_case','u_tower_case','u_missing_case'})
        self.assertEqual(audit,before)

    def test_resume_first_request_uses_second_original_slot_only_responsible_units(self):
        audit=deepcopy(self.audit);before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):calls.append(json.loads(messages[-1]['content']));raise Boundary()
        def sequential(fn,jobs,**kw):
            for job in jobs:
                result=fn(job)
                if kw.get('on_result'):kw['on_result'](result)
        with patch.object(config,'pmap',sequential),self.assertRaises(Boundary):
            repair(self.wp,self.proposal,{},Tracer(),audit,resume_checkpoint=True)
        self.assertEqual(len(calls),1);uid=calls[0]['unit']['unit_id']
        self.assertIn(uid,{'u_missing_case','u_institution_track'})
        self.assertEqual(audit['edit_attempts'][uid][-1]['attempt'],2)
        self.assertEqual(audit['edit_attempts'][uid][-1]['batch_iteration'],3)
        for name,rows in before['edit_attempts'].items():self.assertEqual(audit['edit_attempts'][name][:len(rows)],rows)
        self.assertEqual(audit['drafts'],before['drafts'])
        self.assertEqual(calls[0]['edit_feedback']['previous_response'],before['edit_attempts'][uid][-1]['proposal'])

    def test_tampered_candidate_or_missing_response_refuses_zero_calls(self):
        for mutation in ('draft','response','requested'):
            audit=deepcopy(self.audit)
            if mutation=='draft':audit['drafts'][0]['candidate']['units'][0]['business_purpose']='tampered'
            elif mutation=='response':audit['edit_attempts']['u_missing_case'][0].pop('proposal')
            else:audit['edit_attempts']['u_missing_case'][0]['status']='requested'
            with self.subTest(mutation=mutation),self.assertRaises(PlanConflict):
                repair(self.wp,self.proposal,{},types.SimpleNamespace(chat_json=forbidden),audit,resume_checkpoint=True)

    def test_exhausted_author_or_noncontiguous_history_cannot_restart(self):
        audit=deepcopy(self.audit);audit['drafts'][-1]['iteration']=3
        with self.assertRaises(PlanConflict) as got:_restore_batch(self.wp,self.proposal,audit)
        self.assertIn('not contiguous',str(got.exception))
        audit=deepcopy(self.audit);rows=audit['edit_attempts']['u_missing_case']
        for attempt in (2,3):rows.append(dict(deepcopy(rows[0]),attempt=attempt))
        with self.assertRaises(PlanConflict) as got:_restore_batch(self.wp,self.proposal,audit)
        self.assertIn('allowance exhausted',str(got.exception))

if __name__=='__main__':unittest.main()
