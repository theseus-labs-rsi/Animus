"""Uncompiled diagnostic -> one original business author -> original final gates.

All model replies here are labeled synthetic protocol fixtures. Real A5 is
read only and establishes the candidate/boundary, never a real review opinion.
"""
import os,sys,socket,types,json,unittest,tempfile,hashlib
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
os.environ.update(PYTHON_DOTENV_DISABLED='1',OPENAI_API_KEY='offline-layout-dummy',MODEL='offline',OPENAI_BASE_URL='http://127.0.0.1:9')
def forbidden(*a,**k):raise AssertionError('Network/provider forbidden')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:'';sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('LAYOUT_TEST_REPO',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import openai
openai.OpenAI=openai.AsyncOpenAI=forbidden
import config
config.chat=forbidden
if os.environ.get('LAYOUT_TEST_MODULE'):
    import importlib.util
    spec=importlib.util.spec_from_file_location('pipeline.layout_revision',os.environ['LAYOUT_TEST_MODULE'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from pipeline import layout_revision as lr,blueprint_feasibility as bf
from pipeline.instance_plan import PlanConflict,compile_plan
from pipeline.capability_contract import digest
from instance_plan_selftest import setup_plan
from blueprint_feasibility_selftest import opinion
from execution_control import ExecutionStopped
from pipeline.run import Tracer as ProductionTracer

class Boundary(BaseException):pass
class Trace:
    def __init__(self,answers):self.answers=list(answers);self.calls=[]
    def chat_json(self,step,messages,**kwargs):
        self.calls.append({'step':step,'payload':json.loads(messages[-1]['content']),'kwargs':kwargs,
                           'message_utf8_bytes':len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))})
        if not self.answers:raise AssertionError('Unexpected request: '+step)
        result=self.answers.pop(0)
        if isinstance(result,BaseException):raise result
        if step=='council.blueprint_feasibility' and self.calls[-1]['payload'].get('phase')=='uncompiled_plan_escalation' and isinstance(result,dict) and 'decision' in result:
            result=deepcopy(result);result['responsibility_targets']=[]
            if result['decision']=='repair':
                result['responsibility_targets']=[{k:deepcopy(row[k]) for k in ('unit_id','collection','key','finding_indices')}
                    for row in self.calls[-1]['payload']['responsibility_catalog']]
                for row in result['responsibility_targets']:row['reason']='Synthetic original responsibility target fixture'
        return deepcopy(result)

def diagnose(wp,plan,trace,history=None):
    try:compile_plan(wp,plan)
    except PlanConflict as error:findings=error.findings
    else:raise AssertionError('Expected original compiler rejection')
    return bf.diagnose_uncompiled_plan(wp,plan,findings,trace,
        history=history or {'scope':'synthetic failure history fixture'},
        source_binding={'label':'synthetic offline source binding'},
        retry_policy_binding={'label':'synthetic offline one remaining business slot'},remaining_business_attempts=1)

class UncompiledEscalationTests(unittest.TestCase):
    def setUp(self):
        self.wp,self.plan,*_=setup_plan();self.plan['units'][0]['events'][-1]['session']=4
        event=deepcopy(self.plan['units'][0]['events'][-1]);event['session']=3
        self.revision={'decision':'repair','reason':'Synthetic restore original exact causal delay',
            'changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':event}]}
        self.history={'scope':'synthetic failure history fixture','failed_candidate_sha256':digest(self.plan)}
    def receipt(self,decision='repair'):
        return diagnose(self.wp,self.plan,Trace([opinion(self.wp,decision)]),self.history)
    def test_full_synthetic_diagnostic_author_compile_independent_review_chain(self):
        before=deepcopy((self.wp,self.plan));trace=Trace([opinion(self.wp,'repair'),self.revision,opinion(self.wp)])
        diagnostic=diagnose(self.wp,self.plan,trace,self.history);audit={};snapshots=[]
        result=lr.revise(self.wp,self.plan,{'failure_history':self.history},trace,audit,max_attempts=1,
            uncompiled_diagnostic=diagnostic,checkpoint=lambda a:snapshots.append(deepcopy(a)))
        self.assertEqual([c['step'] for c in trace.calls],['council.blueprint_feasibility','council.instance_plan_business_repair','council.blueprint_feasibility'])
        author=trace.calls[1]['payload'];self.assertNotIn('compiled_execution',author)
        self.assertEqual(author['uncompiled_candidate'],self.plan)
        self.assertEqual(author['input_mode'],'uncompiled_candidate_diagnosed')
        self.assertEqual(author['independent_diagnostic']['review'],diagnostic['review']['review'])
        self.assertNotIn('input',author['independent_diagnostic'])
        self.assertNotIn('proposal',author);self.assertNotIn('feedback',author)
        self.assertEqual(author['history'],self.history)
        self.assertEqual(author['requirements'],self.wp['supply_plan']['requirements'])
        self.assertEqual((self.wp,self.plan),before);self.assertEqual(audit['status'],'accepted')
        self.assertEqual(len(audit['business_revisions']),1)
        self.assertEqual(audit['business_revisions'][0]['business_review']['decision'],'accept')
        self.assertTrue(result['business_instance_plan']['plan_hash']);self.assertTrue(result['seed_audit']['passed'])
        self.assertTrue(any(a.get('business_revisions') and a['business_revisions'][0]['status']=='requested' for a in snapshots))
        self.assertEqual(snapshots[-1]['status'],'accepted')
    def test_default_entry_still_rejects_uncompiled_candidate_before_author(self):
        trace=Trace([])
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},trace,{},max_attempts=1)
        self.assertEqual(trace.calls,[])
    def test_accept_or_unresolved_diagnostic_cannot_invoke_business_author(self):
        for decision in ('accept','unresolved'):
            diagnostic=self.receipt(decision);trace=Trace([])
            with self.subTest(decision=decision),self.assertRaises(Exception):
                lr.revise(self.wp,self.plan,{},trace,{},max_attempts=1,uncompiled_diagnostic=diagnostic)
            self.assertEqual(trace.calls,[])
    def test_candidate_blueprint_seed_requirement_limit_changes_refuse_before_author(self):
        diagnostic=self.receipt()
        for mode in ('candidate','blueprint','seed','requirements','limits','receipt'):
            wp,plan,receipt=deepcopy(self.wp),deepcopy(self.plan),deepcopy(diagnostic);trace=Trace([])
            if mode=='candidate':plan['reason']='different authored plan'
            elif mode=='blueprint':wp['world_blueprint']['event_types'][0]['label']+=' changed'
            elif mode=='seed':wp['seed_contract']['mechanisms'][0]['description']='different mandatory task'
            elif mode=='requirements':wp['supply_plan']['requirements'][0]['independent_candidates']+=1
            elif mode=='limits':wp['supply_plan']['world_limits']['max_world_entities']+=1
            else:receipt['decision']='accept'
            with self.subTest(mode=mode),self.assertRaises(Exception):
                lr.revise(wp,plan,{},trace,{},max_attempts=1,uncompiled_diagnostic=receipt)
            self.assertEqual(trace.calls,[])
    def test_remaining_one_and_used_slot_and_unbound_reuse_are_hard_limits(self):
        diagnostic=self.receipt()
        for kwargs,audit in [({'max_attempts':2},{}),({'max_attempts':1,'initial_revision':self.revision},{}),
                             ({'max_attempts':1},{'business_revisions':[{'attempt':1,'status':'call_failed'}]})]:
            trace=Trace([]);before=deepcopy(audit)
            with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit,uncompiled_diagnostic=diagnostic,**kwargs)
            self.assertEqual(trace.calls,[]);self.assertEqual(audit,before)
    def test_failed_author_compile_has_no_final_review_and_no_extra_business_slot(self):
        diagnostic=self.receipt();raw=deepcopy(self.revision);raw['changes'][0]['value']['session']=5
        trace=Trace([raw]);audit={};before=deepcopy((self.wp,self.plan))
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(len(trace.calls),1);self.assertEqual(audit['business_revisions'][0]['status'],'compiler_rejected')
        self.assertEqual(len(audit['business_revisions']),1);self.assertEqual((self.wp,self.plan),before)
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},Trace([]),audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
    def test_final_independent_negative_preserved_and_not_committed(self):
        for decision in ('repair','unresolved'):
            diagnostic=self.receipt();trace=Trace([self.revision,opinion(self.wp,decision)]);audit={};before=deepcopy((self.wp,self.plan))
            with self.subTest(decision=decision),self.assertRaises(PlanConflict):
                lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
            self.assertEqual(len(trace.calls),2);self.assertEqual(len(audit['business_revisions']),1)
            self.assertEqual(audit['business_revisions'][0]['business_review']['decision'],decision)
            self.assertNotIn('after',audit);self.assertEqual((self.wp,self.plan),before)
    def test_author_unresolved_provider_and_global_stop_preserve_one_original_slot(self):
        cases=[({'decision':'unresolved','reason':'Cannot satisfy original scope'},PlanConflict),
               ({'__error__':'provider failed'},RuntimeError),
               ({'__error__':'budget','__error_metadata__':{'global_stop':True,'kind':'estimated_budget_limit'}},ExecutionStopped)]
        for raw,error in cases:
            diagnostic=self.receipt();trace=Trace([raw]);audit={}
            with self.subTest(raw=raw),self.assertRaises(error):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
            self.assertEqual(len(trace.calls),1);self.assertEqual(audit['business_revisions'][0]['proposal'],raw)
            with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},Trace([]),audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
    def test_boundary_request_was_checkpointed_and_reentry_does_not_dispatch(self):
        diagnostic=self.receipt();audit={};snapshots=[];trace=Trace([Boundary()])
        with self.assertRaises(Boundary):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic,checkpoint=lambda a:snapshots.append(deepcopy(a)))
        self.assertTrue(any(a.get('business_revisions') and a['business_revisions'][0]['status']=='requested' for a in snapshots))
        self.assertEqual(audit['business_revisions'][0]['status'],'call_failed')
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},Trace([]),audit,max_attempts=1,uncompiled_diagnostic=diagnostic)

    def test_bad_shapes_correct_within_one_business_slot_then_original_final_gates(self):
        diagnostic=self.receipt();bad=deepcopy(self.revision);bad['changes'][0]['value']={'session':3}
        trace=Trace([[],bad,self.revision,opinion(self.wp)]);audit={};snapshots=[]
        result=lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic,
            checkpoint=lambda a:snapshots.append(deepcopy(a)))
        row=audit['business_revisions'][0];self.assertEqual(len(audit['business_revisions']),1)
        self.assertEqual([p['status'] for p in row['protocol_attempts']],['protocol_rejected','protocol_rejected','protocol_valid'])
        self.assertEqual([p['proposal'] for p in row['protocol_attempts']],[[],bad,self.revision])
        self.assertEqual(trace.calls[1]['payload']['protocol_feedback']['previous_response'],[])
        self.assertEqual(trace.calls[2]['payload']['protocol_feedback']['previous_response'],bad)
        self.assertEqual(len(trace.calls),4);self.assertEqual(audit['status'],'accepted')
        self.assertEqual(audit['author_protocol']['max_physical_author_requests'],9)
        self.assertTrue(result['business_instance_plan']['plan_hash']);self.assertTrue(result['seed_audit']['passed'])
        for number in (1,2,3):
            self.assertTrue(any(len(a.get('business_revisions',[{}])[0].get('protocol_attempts',[]))==number
                and a['business_revisions'][0]['protocol_attempts'][-1]['status']=='requested' for a in snapshots if a.get('business_revisions')))

    def test_invalid_protocol_exhausts_three_preserves_raw_and_cannot_reenter(self):
        diagnostic=self.receipt();trace=Trace([None,None,None, self.revision]);audit={}
        with self.assertRaises(PlanConflict) as got:
            lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(len(trace.calls),3);self.assertEqual(got.exception.findings[0]['code'],'business_revision_protocol_exhausted')
        self.assertEqual(len(audit['business_revisions']),1)
        self.assertEqual([p['proposal'] for p in audit['business_revisions'][0]['protocol_attempts']],[None]*3)
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},Trace([]),audit,max_attempts=1,uncompiled_diagnostic=diagnostic)

    def test_protocol_compatibility_optional_field_and_alias_addition_records(self):
        revision=deepcopy(self.revision)
        revision['changes']=[{'unit_id':'report-case','collection':'obligations','operation':'replace','key':'fixture',
            'value':{'line':'L2','carrier':{'entities':['report-slot'],'field':None}}},
            {'unit_id':'report-case','collection':'observations','operation':'add',
             'value':{'entity':'report-slot','field':'review','sessions':[1],'mechanism':''}}]
        revision['field_aliases']=[{'entity_type':'report','source_field':'alias','target_field':'review','reason':'Synthetic synonym'}]
        revision['blueprint_additions']={'event_types':[{'id':'review_again','roles':{'record':'report'},
            'effect_fields':[{'role':'record','field':'review'}],'count':1}]}
        before=deepcopy(revision);self.assertEqual(lr._business_protocol_errors(self.wp,revision),[])
        self.assertEqual(revision,before)

    def test_valid_scope_failure_is_not_a_protocol_retry(self):
        diagnostic=self.receipt();raw=deepcopy(self.revision);raw['changes'][0]['unit_id']='unknown-original-unit'
        trace=Trace([raw,self.revision]);audit={}
        with self.assertRaises(PlanConflict):lr.revise(self.wp,self.plan,{},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(len(trace.calls),1);self.assertEqual(audit['business_revisions'][0]['protocol_attempts'][0]['status'],'protocol_valid')

    def test_real_parser_three_json_by_three_protocol_is_nine_author_requests(self):
        diagnostic=self.receipt();answers=[]
        for raw in ([],{'decision':[]},self.revision):answers+=['{','[',json.dumps(raw)]
        answers.append(json.dumps(opinion(self.wp)));physical=[];audit={}
        def transport(messages,**kwargs):physical.append(deepcopy(messages));return answers.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):
                lr.revise(self.wp,self.plan,{},tracer,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
            prompts=[json.loads(x) for x in tracer.pfile.read_text(encoding='utf-8').splitlines()]
            trace=[json.loads(x) for x in tracer.pfile.with_name('llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(physical),10) # Nine author requests plus one final independent review.
        self.assertEqual(len(prompts),4);self.assertEqual(len([p for p in trace if p['event']=='json_attempt']),10)
        self.assertEqual([p['output'] for p in prompts[:3]],[p['proposal'] for p in audit['business_revisions'][0]['protocol_attempts']])
        self.assertTrue(all(p['params']['retries']==3 for p in prompts))

    def test_real_parser_json_exhaustion_does_not_open_new_protocol(self):
        diagnostic=self.receipt();physical=[];audit={}
        def transport(messages,**kwargs):physical.append(1);return '{'
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(RuntimeError):
                lr.revise(self.wp,self.plan,{},tracer,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
            self.assertEqual(tracer.n,1)
        self.assertEqual(len(physical),3);self.assertEqual(len(audit['business_revisions'][0]['protocol_attempts']),1)

    def test_budget_rejection_after_badshape_stops_without_dispatch(self):
        diagnostic=self.receipt();admission=[];physical=[];audit={}
        def transport(messages,**kwargs):
            admission.append(1)
            if len(admission)==4:raise ExecutionStopped('Synthetic cumulative budget limit',kind='estimated_budget_limit')
            physical.append(1);return '{' if len(physical)<3 else '[]'
        with tempfile.TemporaryDirectory() as directory:
            tracer=ProductionTracer(types.SimpleNamespace(dir=Path(directory)))
            with patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(ExecutionStopped):
                lr.revise(self.wp,self.plan,{},tracer,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(len(admission),4);self.assertEqual(len(physical),3)
        self.assertEqual(len(audit['business_revisions']),1);self.assertEqual(len(audit['business_revisions'][0]['protocol_attempts']),2)

    def test_existing_global_stop_prevents_first_author_dispatch(self):
        diagnostic=self.receipt()
        class Stopped:
            def check_global_stop(_):raise ExecutionStopped('Stopped already',kind='call_limit')
            chat_json=staticmethod(forbidden)
        audit={}
        with self.assertRaises(ExecutionStopped):lr.revise(self.wp,self.plan,{},Stopped(),audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(audit['business_revisions'][0]['protocol_attempts'][0]['status'],'call_failed')

class RealA5UncompiledBoundary(unittest.TestCase):
    def test_real_candidate_with_labeled_synthetic_diagnosis_reaches_only_original_business_one(self):
        source=os.environ.get('LAYOUT_REAL_A5_EVIDENCE')
        if not source:self.skipTest('Real A5 evidence not supplied')
        path=Path(source)
        original_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        def read(name):return json.loads((path/name).read_text(encoding='utf-8'))
        from pipeline import initial_plan_recovery as recovery
        from tools.run_original_bc_smoke import implementation_hashes
        wp=recovery._prepared_whitepaper(read('01_whitepaper.json'));record=read('01_initial_plan_recovery.json')
        candidate=record['transaction']['drafts'][-1]['candidate'];before=deepcopy(record)
        with self.assertRaises(PlanConflict) as got:compile_plan(wp,candidate)
        self.assertEqual([f['code'] for f in got.exception.findings],['native_trend_temporal_capacity'])
        # This fake review tests binding/control flow only, not model diagnosis.
        request=recovery.build_checkpoint_receipt(path,implementation_hashes(),plan_retry_policy='separate-protocol-v2',
            plan_escalation='original-business-slot-v1')
        escalation=request['escalation']
        diagnostic=bf.diagnose_uncompiled_plan(wp,candidate,escalation['compiler_findings'],Trace([opinion(wp,'repair')]),
            history=escalation['failure_history'],source_binding={'run_id':recovery.RUN_ID,
                'source_migration':request['source_migration'],'seed_input':escalation['seed_input'],'seed_pack':escalation['seed_pack'],
                'delivery_target':escalation['delivery_target'],'manifest_config':read('manifest.json')['config'],
                'escalation_sha256':request['escalation_sha256']},
            retry_policy_binding={'policy':request['retry_policy'],'sha256':request['retry_policy_sha256']},
            remaining_business_attempts=1)
        trace=Trace([Boundary()]);audit={}
        with self.assertRaises(Boundary):lr.revise(wp,candidate,{'original_record':record},trace,audit,max_attempts=1,uncompiled_diagnostic=diagnostic)
        self.assertEqual(len(trace.calls),1);self.assertEqual(trace.calls[0]['step'],'council.instance_plan_business_repair')
        payload=trace.calls[0]['payload'];self.assertEqual(payload['uncompiled_candidate'],candidate)
        self.assertNotIn('compiled_execution',payload);self.assertEqual(record,before)
        self.assertEqual(len(audit['business_revisions']),1);self.assertEqual(audit['business_revisions'][0]['attempt'],1)
        self.assertEqual(original_sha,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()})
        print('A5_AUTHOR_MESSAGES_UTF8_BYTES='+str(trace.calls[0]['message_utf8_bytes']))

if __name__=='__main__':unittest.main()
