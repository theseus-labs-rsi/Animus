"""Explicit cumulative business slots B2/B3; all responses are offline fixtures."""
import os,sys,json,tempfile,types,unittest,hashlib
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
ROOT=Path(os.environ.get('LAYOUT_TEST_REPO',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import layout_uncommitted_escalation_selftest as base  # Guards before repository/provider imports.
from pipeline import layout_revision as lr
from pipeline.capability_contract import digest
from pipeline.instance_plan import PlanConflict
from execution_control import ExecutionStopped
config=base.config

def binding(audit,plan,diagnostic):
    return {'version':'business-revision-checkpoint/v1','candidate_sha256':digest(plan),
        'diagnostic_sha256':digest(diagnostic),'business_revisions_sha256':digest(audit['business_revisions']),
        'completed_business_attempts':len(audit['business_revisions']),'max_business_attempts':3,
        'max_additional_physical_requests':36}

class BusinessCheckpointTests(unittest.TestCase):
    def setUp(self):
        fixture=base.UncompiledEscalationTests();fixture.setUp()
        self.wp,self.plan,self.good=fixture.wp,fixture.plan,fixture.revision
        self.diagnostic=fixture.receipt();self.bad=deepcopy(self.good);self.bad['changes'][0]['value']['session']=5
        self.audit={}
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},base.Trace([self.bad]),self.audit,
            max_attempts=1,uncompiled_diagnostic=self.diagnostic)
        self.b1=deepcopy(self.audit['business_revisions'][0]);self.original_diagnostic=deepcopy(self.audit['uncompiled_diagnostic'])
        self.third=deepcopy(self.good)
        publication=self.plan['units'][0]['publications'][0]
        self.third['changes'].append({'unit_id':'report-case','collection':'publications','operation':'replace',
            'selector':dict(publication),'value':dict(publication,session=5)})
    def resume(self,trace,**kwargs):
        return lr.revise(self.wp,self.plan,{},trace,self.audit,max_attempts=3,uncompiled_diagnostic=self.diagnostic,
            business_resume_checkpoint=binding(self.audit,self.plan,self.diagnostic),**kwargs)
    def unchanged_history(self):
        self.assertEqual(self.audit['business_revisions'][0],self.b1)
        self.assertEqual(self.audit['uncompiled_diagnostic'],self.original_diagnostic)
        self.assertEqual(self.audit['author_protocol']['max_business_attempts'],1)
    def test_b2_compiler_failure_then_b3_success_retains_entire_history(self):
        trace=base.Trace([self.bad,self.third,base.opinion(self.wp)])
        result=self.resume(trace)
        self.assertEqual([r['attempt'] for r in self.audit['business_revisions']],[1,2,3]);self.unchanged_history()
        self.assertEqual([c['payload']['authorized_business_continuation']['semantic_attempt'] for c in trace.calls if c['step']=='council.instance_plan_business_repair'],[2,3])
        self.assertEqual(trace.calls[0]['payload']['revision_feedback']['previous_response'],self.b1['proposal'])
        self.assertEqual(trace.calls[1]['payload']['revision_feedback']['findings'],self.audit['business_revisions'][1]['findings'])
        self.assertEqual(trace.calls[1]['payload']['business_failure_history'],self.audit['business_revisions'][:2])
        self.assertTrue(result['business_instance_plan']['plan_hash']);self.assertEqual(self.audit['status'],'accepted')
    def test_b2_accept_never_requests_b3(self):
        trace=base.Trace([self.good,base.opinion(self.wp)]);self.resume(trace)
        self.assertEqual(len(trace.calls),2);self.assertEqual(len(self.audit['business_revisions']),2);self.unchanged_history()
    def test_b2_final_repair_then_b3_author_change_final_accept(self):
        negative=base.opinion(self.wp,'repair');trace=base.Trace([self.good,negative,self.third,base.opinion(self.wp)])
        self.resume(trace);self.unchanged_history()
        self.assertEqual(len(trace.calls),4)
        self.assertEqual(trace.calls[2]['payload']['business_failure_history'][1]['business_review']['review'],negative)
        self.assertEqual(self.audit['business_revisions'][1]['status'],'business_review_rejected')
    def test_negative_review_cannot_be_retried_on_unchanged_candidate(self):
        trace=base.Trace([self.good,base.opinion(self.wp,'repair'),self.good])
        with self.assertRaises(PlanConflict) as got:self.resume(trace)
        self.assertEqual(len(trace.calls),3);self.assertEqual(got.exception.findings[0]['code'],'business_revision_no_progress')
        self.assertNotIn('business_review_call',self.audit['business_revisions'][2]);self.unchanged_history()
    def test_completed_b2_checkpoint_resumes_only_b3(self):
        def checkpoint(audit):
            rows=audit['business_revisions']
            if len(rows)==2 and rows[-1]['status']=='compiler_rejected':raise base.Boundary()
        first=base.Trace([self.bad])
        with self.assertRaises(base.Boundary):self.resume(first,checkpoint=checkpoint)
        before=deepcopy(self.audit['business_revisions']);second=base.Trace([self.third,base.opinion(self.wp)])
        self.resume(second)
        self.assertEqual(second.calls[0]['payload']['authorized_business_continuation']['semantic_attempt'],3)
        self.assertEqual(self.audit['business_revisions'][:2],before);self.unchanged_history()
    def test_interrupted_protocol_never_restarts_or_refunds_b2(self):
        with self.assertRaises(base.Boundary):self.resume(base.Trace([base.Boundary()]))
        self.assertEqual(self.audit['business_revisions'][-1]['attempt'],2)
        empty=base.Trace([])
        with self.assertRaises(PlanConflict):self.resume(empty)
        self.assertEqual(empty.calls,[]);self.unchanged_history()
    def test_valid_author_and_reviewer_unresolved_stop_immediately(self):
        for answers in ([{'decision':'unresolved','reason':'Cannot repair'}],[self.good,base.opinion(self.wp,'unresolved')]):
            original=deepcopy(self.audit);trace=base.Trace(answers)
            with self.assertRaises(PlanConflict):self.resume(trace)
            self.assertEqual(len(trace.calls),len(answers));self.assertEqual(len(self.audit['business_revisions']),2)
            with self.assertRaises(PlanConflict):self.resume(base.Trace([]))
            self.audit=original
    def test_all_three_consumed_reentry_zero_dispatch(self):
        trace=base.Trace([self.bad,self.bad])
        with self.assertRaises(PlanConflict):self.resume(trace)
        self.assertEqual(len(trace.calls),2);self.assertEqual(len(self.audit['business_revisions']),3)
        with self.assertRaises(PlanConflict):self.resume(base.Trace([]))
        self.unchanged_history()
    def test_candidate_diagnostic_rows_and_bounds_are_bound_before_mutation(self):
        for key in ('candidate_sha256','diagnostic_sha256','business_revisions_sha256','completed_business_attempts','max_business_attempts','max_additional_physical_requests'):
            proof=binding(self.audit,self.plan,self.diagnostic);proof[key]='changed';before=deepcopy(self.audit);trace=base.Trace([])
            with self.subTest(key=key),self.assertRaises(PlanConflict):
                lr.revise(self.wp,self.plan,{},trace,self.audit,max_attempts=3,uncompiled_diagnostic=self.diagnostic,business_resume_checkpoint=proof)
            self.assertEqual(trace.calls,[]);self.assertEqual(self.audit,before)
    def test_pending_final_review_is_not_treated_as_completed_negative(self):
        with self.assertRaises(base.Boundary):self.resume(base.Trace([self.good,base.Boundary()]))
        with self.assertRaises(PlanConflict):self.resume(base.Trace([]))
        self.unchanged_history()
    def test_protocol_corrections_do_not_consume_b3_and_sentinels_stop(self):
        trace=base.Trace([[],self.good,base.opinion(self.wp)]);self.resume(trace)
        self.assertEqual(len(self.audit['business_revisions']),2)
        self.assertEqual(len(self.audit['business_revisions'][1]['protocol_attempts']),2)
        self.unchanged_history()
    def test_new_stage_real_parser_upper_bound_is_36_physical_requests(self):
        answers=[]
        # Two semantic authors; each has three parsed protocol attempts, each
        # preceded by two malformed JSON attempts. Each final reviewer has the
        # same bounded syntax/protocol pattern and a distinct final decision.
        for author,decision in ((self.good,'repair'),(self.third,'accept')):
            for raw in ([],{'decision':[]},author):answers+=['{','[',json.dumps(raw)]
            for raw in ([],{'decision':[]},base.opinion(self.wp,decision)):answers+=['{','[',json.dumps(raw)]
        physical=[]
        def transport(messages,**kwargs):physical.append(deepcopy(messages));return answers.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            tracer=base.ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):self.resume(tracer)
            records=[json.loads(s) for s in tracer.pfile.read_text(encoding='utf-8').splitlines()]
            trace=[json.loads(s) for s in tracer.pfile.with_name('llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(physical),36);self.assertEqual(len(records),12)
        self.assertEqual(len([r for r in trace if r['event']=='json_attempt']),36)
        self.assertEqual(len([r for r in records if r['step']=='council.instance_plan_business_repair']),6)
        self.assertEqual(len([r for r in records if r['step']=='council.blueprint_feasibility']),6)
        self.assertEqual(self.audit['status'],'accepted');self.unchanged_history()
    def test_global_stop_inside_new_stage_never_advances_to_b3(self):
        stopped={'__error__':'new stage exhausted','__error_metadata__':{'global_stop':True,'kind':'call_limit'}}
        trace=base.Trace([[],stopped,self.good])
        with self.assertRaises(ExecutionStopped):self.resume(trace)
        self.assertEqual(len(trace.calls),2);self.assertEqual(len(self.audit['business_revisions']),2);self.unchanged_history()

class RealA6CheckpointBoundary(unittest.TestCase):
    def test_real_a6_b1_is_immutable_and_next_only_b2(self):
        evidence=os.environ.get('LAYOUT_REAL_A6_EVIDENCE')
        if not evidence:self.skipTest('Real A6 evidence not provided')
        path=Path(evidence)
        def read(name):return json.loads((path/name).read_text(encoding='utf-8'))
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        from pipeline.initial_plan_recovery import _prepared_whitepaper
        record=read('01_initial_plan_recovery.json');audit=deepcopy(record['business_revision']);before=deepcopy(audit)
        wp=_prepared_whitepaper(read('01_whitepaper.json'));diagnostic=audit['uncompiled_diagnostic'];candidate=diagnostic['review']['input']['uncompiled_candidate']
        trace=base.Trace([base.Boundary()])
        with self.assertRaises(base.Boundary):lr.revise(wp,candidate,{},trace,audit,max_attempts=3,uncompiled_diagnostic=diagnostic,
            business_resume_checkpoint=binding(audit,candidate,diagnostic))
        self.assertEqual(len(trace.calls),1);self.assertEqual(trace.calls[0]['step'],'council.instance_plan_business_repair')
        payload=trace.calls[0]['payload'];self.assertEqual(payload['authorized_business_continuation']['semantic_attempt'],2)
        self.assertEqual(payload['uncompiled_candidate'],candidate)
        self.assertEqual(payload['business_failure_history'],before['business_revisions'])
        self.assertEqual(payload['revision_feedback']['findings'],before['business_revisions'][0]['findings'])
        self.assertEqual(audit['business_revisions'][:1],before['business_revisions'])
        self.assertEqual(audit['uncompiled_diagnostic'],before['uncompiled_diagnostic'])
        self.assertEqual(payload['revision_feedback']['findings'][0]['evidence']['write_periods'],[])
        self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()})
        print('A6_NEXT_B2_AUTHOR_MESSAGES_UTF8_BYTES='+str(trace.calls[0]['message_utf8_bytes']))

if __name__=='__main__':unittest.main()
