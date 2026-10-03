"""Original author quotas, append-only joint batches, and the real 18-call stop."""
import plan_patch_shape_selftest as guarded
import os
import json
import types
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from decimal import Decimal
from pipeline.plan_transactions import repair, _restore_batch, derived_dependencies, _route_findings
from pipeline.instance_plan import PlanConflict, compile_plan, apply_repair, finding
import config


class ActualEighteenCallScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get('PATCH_REAL_CHECKPOINT_EVIDENCE')
        if not source:raise unittest.SkipTest('Real 18-call checkpoint evidence not supplied')
        cls.path=Path(source)
        def read(name):return json.loads((cls.path/name).read_text(encoding='utf-8-sig'))
        from pipeline.initial_plan_recovery import _prepared_whitepaper
        cls.wp=_prepared_whitepaper(read('01_whitepaper.json'))
        cls.proposal=read('01_instance_plan_audit.json')['attempts'][0]['proposal']
        cls.record=read('01_initial_plan_recovery.json')
        cls.audit=cls.record['transaction']

    def test_real_latest_candidate_and_exact_construction_findings_replayed(self):
        before=deepcopy(self.audit)
        base,iteration,findings,replies,pending,receipt=_restore_batch(self.wp,self.proposal,self.audit)
        self.assertEqual(base,self.audit['drafts'][-1]['candidate'])
        self.assertEqual(iteration,3);self.assertEqual(receipt['joint_drafts_consumed'],3)
        self.assertEqual(receipt['scheduler_version'],'original-author-slots/v1')
        self.assertEqual(receipt['max_attempts_per_author'],3)
        self.assertEqual(replies,{})
        self.assertEqual(set(pending),{'u_tower_case','u_institution_track'})
        self.assertEqual(findings,self.record['findings'])
        with self.assertRaises(PlanConflict) as got:compile_plan(self.wp,base)
        self.assertEqual(got.exception.findings,findings)
        self.assertEqual(self.audit,before)

    def test_original_eighteen_costs_are_settled_and_unchanged(self):
        trace=[json.loads(line) for line in (self.path/'llm_attempts.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        requests=[r for r in trace if r['event']=='request']
        responses=[r for r in trace if r['event']=='response']
        self.assertEqual(len(requests),18);self.assertEqual(len(responses),18)
        total=sum((Decimal(r['response']['usage']['prompt_tokens'])*Decimal('.632')+
                   Decimal(r['response']['usage']['completion_tokens'])*Decimal('2.212'))/Decimal(1000000)
                  for r in responses)
        self.assertEqual(total,Decimal('.224414036'))

    def test_first_batch_only_original_tower_two_and_institution_three(self):
        audit=deepcopy(self.audit);before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                calls.append(json.loads(messages[-1]['content']));raise guarded.Boundary()
        def all_boundaries(fn,jobs,**kwargs):
            for job in jobs:
                try:fn(job)
                except guarded.Boundary:pass
            raise guarded.Boundary()
        with patch.object(config,'pmap',all_boundaries),self.assertRaises(guarded.Boundary):
            repair(self.wp,self.proposal,{},Tracer(),audit,resume_checkpoint=True)
        self.assertEqual({p['unit']['unit_id'] for p in calls},{'u_tower_case','u_institution_track'})
        for payload in calls:
            self.assertEqual(payload['current_candidate'],before['drafts'][-1]['candidate'])
            self.assertEqual(payload['blueprint'],self.wp['world_blueprint'])
            self.assertEqual(payload['requirements'],self.wp['supply_plan']['requirements'])
        for uid,slot in [('u_tower_case',2),('u_institution_track',3)]:
            self.assertEqual(audit['edit_attempts'][uid][-1]['attempt'],slot)
            self.assertEqual(audit['edit_attempts'][uid][-1]['batch_iteration'],4)
        for uid,rows in before['edit_attempts'].items():self.assertEqual(audit['edit_attempts'][uid][:len(rows)],rows)
        self.assertEqual(audit['drafts'],before['drafts'])
        # A boundary exception leaves a non-settled request and cannot be
        # called again merely by entering checkpoint recovery a second time.
        with self.assertRaises(PlanConflict):
            repair(self.wp,self.proposal,{},types.SimpleNamespace(chat_json=guarded.forbidden),audit,resume_checkpoint=True)

    def test_exhausted_responsible_author_stops_whole_batch_before_other_calls(self):
        audit=deepcopy(self.audit);rows=audit['edit_attempts']['u_institution_track']
        rows.append(dict(deepcopy(rows[-1]),attempt=3))
        before=deepcopy(audit)
        with self.assertRaises(PlanConflict) as got:
            repair(self.wp,self.proposal,{},types.SimpleNamespace(chat_json=guarded.forbidden),audit,resume_checkpoint=True)
        self.assertIn('allowance exhausted',str(got.exception));self.assertEqual(audit,before)

    def test_fourth_joint_draft_requires_new_responses_and_exhaustion_never_spends_sibling_slot(self):
        # Synthetic replies only change an observation's description and fail
        # the real compiler's same capability gates. This proves scheduling
        # and finite stopping; it is deliberately not a successful repair.
        audit=deepcopy(self.audit);before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                p=json.loads(messages[-1]['content']);uid=p['unit']['unit_id'];calls.append(uid)
                row=deepcopy(p['unit']['observations'][0]);row['mechanism']+=' (synthetic unchanged-capacity fixture)'
                return {'decision':'repair','changes':[{'unit_id':uid,'collection':'observations','operation':'replace',
                    'key':row['entity']+'|'+row['field'],'value':row}]}
        def sequential(fn,jobs,**kwargs):
            for job in jobs:
                value=fn(job)
                if kwargs.get('on_result'):kwargs['on_result'](value)
        with patch.object(config,'pmap',sequential),self.assertRaises(PlanConflict) as got:
            repair(self.wp,self.proposal,{},Tracer(),audit,resume_checkpoint=True)
        self.assertEqual(got.exception.findings[0]['code'],'repair_author_exhausted')
        self.assertEqual(set(calls),{'u_tower_case','u_institution_track'});self.assertEqual(len(calls),2)
        self.assertEqual([d['iteration'] for d in audit['drafts']],[1,2,3,4])
        self.assertEqual(audit['drafts'][:3],before['drafts'])
        self.assertEqual(audit['drafts'][-1]['author_slots_after']-audit['drafts'][-1]['author_slots_before'],2)
        self.assertEqual(audit['drafts'][-1]['findings'],self.record['findings'])
        self.assertEqual(len(audit['edit_attempts']['u_tower_case']),2)
        self.assertEqual(len(audit['edit_attempts']['u_institution_track']),3)
        for uid,rows in before['edit_attempts'].items():self.assertEqual(audit['edit_attempts'][uid][:len(rows)],rows)

    def test_ordinary_reentry_cannot_reset_saved_history(self):
        audit=deepcopy(self.audit);before=deepcopy(audit)
        with self.assertRaises(PlanConflict) as got:
            repair(self.wp,self.proposal,{},types.SimpleNamespace(chat_json=guarded.forbidden),audit)
        self.assertIn('explicit checkpoint',str(got.exception));self.assertEqual(audit,before)

    def test_false_rebase_and_reordered_batch_refused_without_requests(self):
        for change in ('base','iteration'):
            audit=deepcopy(self.audit)
            if change=='base':audit['checkpoint_revalidations'][0]['base_sha256']='0'*64
            else:audit['drafts'][-1]['iteration']=7
            with self.subTest(change=change),self.assertRaises(PlanConflict):
                repair(self.wp,self.proposal,{},types.SimpleNamespace(chat_json=guarded.forbidden),audit,resume_checkpoint=True)


class SyntheticSavedHistoryEngineTests(unittest.TestCase):
    """Synthetic business data, with real typed edits and the real compiler."""
    def setup_saved(self):
        wp,plan,*_=guarded.setup_plan();main=plan['units'][0]
        main['events'][-1]['session']=4
        def unit(uid):return {'unit_id':uid,'business_purpose':'Synthetic independent review scope','objects':[],
            'relations':[],'events':[],'observations':[],'obligations':[],'publications':[],'depends_on':[]}
        trend=unit('trend-case');trend['observations']=main.pop('observations');main['observations']=[]
        # Profit is owned by revision events. Initial period 0 plus revision
        # period 2 cannot satisfy the original four-point trend constructor.
        trend['observations'][0]['field']='profit'
        trend['obligations']=main.pop('obligations');main['obligations']=[]
        trend['obligations'][0]['carrier']['field']='profit'
        publisher=unit('publisher');publisher['publications']=[{'fact_slot':'company-slot','channel':'报告摘录','session':0}]
        plan['units'] += [trend,publisher]
        original=deepcopy(plan);working=derived_dependencies(plan)
        audit={'drafts':[],'routing':[],'edit_attempts':{'publisher':[]}}
        issue=finding('business_constraint','Synthetic publication scheduling request',['publisher'],kind='business')
        for iteration in range(1,4):
            base=deepcopy(working);route=_route_findings(wp,base,[issue])
            old=deepcopy(next(u for u in base['units'] if u['unit_id']=='publisher')['publications'][0])
            value=dict(old,session=iteration)
            raw={'decision':'repair','changes':[{'unit_id':'publisher','collection':'publications','operation':'replace',
                 'key':old['fact_slot']+'|'+old['channel']+'|'+str(old['session']),'value':value}]}
            working=derived_dependencies(apply_repair(base,raw))
            try:compile_plan(wp,working)
            except PlanConflict as error:remaining=deepcopy(error.findings)
            else:raise AssertionError('Synthetic initial causal conflict unexpectedly passed')
            audit['edit_attempts']['publisher'].append({'attempt':iteration,'batch_iteration':iteration,
                'status':'locally_compiled','proposal':deepcopy(raw)})
            audit['routing'].append({'iteration':iteration,'routes':route[-1],'assigned_new_object_slots':route[9]})
            audit['drafts'].append({'iteration':iteration,'candidate':deepcopy(working),
                'unit_proposals':{'publisher':deepcopy(raw)},'findings':remaining})
            audit['active_batch']={'iteration':iteration,'base':base,'findings':[issue],
                'unit_proposals':{'publisher':deepcopy(raw)},'status':'joint_rejected'}
        return wp,original,audit

    @staticmethod
    def sequential(fn,jobs,**kwargs):
        for job in jobs:
            value=fn(job)
            if kwargs.get('on_result'):kwargs['on_result'](value)

    def test_fourth_causal_repair_reveals_other_owner_then_fifth_compiles(self):
        wp,plan,audit=self.setup_saved();before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                payload=json.loads(messages[-1]['content']);calls.append(payload);u=payload['unit']
                if u['unit_id']=='report-case':
                    event=dict(u['events'][-1],session=3)
                    return {'decision':'repair','changes':[{'unit_id':'report-case','collection':'events',
                        'operation':'replace','key':event['slot'],'value':event}]}
                if u['unit_id']=='trend-case':
                    return {'decision':'repair','changes':[{'unit_id':'trend-case','collection':'events',
                        'operation':'add','key':'trend-case-revision-'+str(period),
                        'value':{'type':'revision','participants':{'report':'report-slot'},'session':period}}
                        for period in (1,4)]}
                raise AssertionError('Unexpected responsibility or repeated old publisher')
        with patch.object(config,'pmap',self.sequential):
            candidate=repair(wp,plan,{},Tracer(),audit,resume_checkpoint=True)
        compile_plan(wp,candidate)
        self.assertEqual([p['unit']['unit_id'] for p in calls],['report-case','trend-case'])
        self.assertEqual([d['iteration'] for d in audit['drafts']],[1,2,3,4,5])
        self.assertEqual(audit['drafts'][:3],before['drafts'])
        self.assertEqual(audit['edit_attempts']['publisher'],before['edit_attempts']['publisher'])
        self.assertEqual(audit['drafts'][-1]['status'],'jointly_compiled')
        self.assertTrue(all(d['author_slots_after']>d['author_slots_before'] for d in audit['drafts'][3:]))
        self.assertTrue(all(f['affected_ids']==['trend-obligation'] for f in audit['drafts'][3]['findings']))

    def test_explicit_unresolved_persists_and_never_creates_fourth_draft(self):
        wp,plan,audit=self.setup_saved();before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):calls.append(1);return {'decision':'unresolved','reason':'Synthetic original constraint conflict'}
        with patch.object(config,'pmap',self.sequential),self.assertRaises(PlanConflict):
            repair(wp,plan,{},Tracer(),audit,resume_checkpoint=True)
        self.assertEqual(len(calls),1);self.assertEqual(audit['drafts'],before['drafts'])
        self.assertEqual(audit['edit_attempts']['report-case'][0]['status'],'unresolved')
        with self.assertRaises(PlanConflict):
            repair(wp,plan,{},types.SimpleNamespace(chat_json=guarded.forbidden),audit,resume_checkpoint=True)

    def test_repeated_no_change_terminates_without_empty_fourth_draft(self):
        wp,plan,audit=self.setup_saved();before=deepcopy(audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):calls.append(1);return {'decision':'repair','changes':[]}
        with patch.object(config,'pmap',self.sequential),self.assertRaises(PlanConflict):
            repair(wp,plan,{},Tracer(),audit,resume_checkpoint=True)
        self.assertEqual(len(calls),2)
        self.assertEqual(audit['drafts'],before['drafts'])
        self.assertTrue(all(r['status']=='locally_rejected' for r in audit['edit_attempts']['report-case']))


if __name__=='__main__':unittest.main()
