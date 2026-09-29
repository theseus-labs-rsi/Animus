"""Explicit protocol/semantic accounting; real captured A4 stays read-only."""
import os,sys,json,types,socket,tempfile,unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
os.environ.update(PYTHON_DOTENV_DISABLED='1',OPENAI_API_KEY='offline-dummy',MODEL='offline-dummy')
def forbidden(*a,**k):raise AssertionError('Network/provider forbidden')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden;socket.create_connection=forbidden;socket.getaddrinfo=forbidden
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:False;dotenv.find_dotenv=lambda *a,**k:'';sys.modules['dotenv']=dotenv
ROOT=Path(os.environ.get('PATCH_TEST_REPO',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import openai
openai.OpenAI=openai.AsyncOpenAI=forbidden
import config
config.chat=forbidden
if os.environ.get('PATCH_TEST_MODULE'):
    import importlib.util
    spec=importlib.util.spec_from_file_location('pipeline.plan_transactions',os.environ['PATCH_TEST_MODULE'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from pipeline import plan_transactions as pt
from pipeline.instance_plan import PlanConflict,compile_plan
from pipeline.run import Tracer as RealTracer
from pipeline.capability_contract import digest
from execution_control import ExecutionStopped
from instance_plan_selftest import setup_plan
class Boundary(BaseException):pass
def sequential(fn,jobs,**kwargs):
    for job in jobs:
        value=fn(job)
        if kwargs.get('on_result'):kwargs['on_result'](value)

class ProtocolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.wp,self.plan,*_=setup_plan();self.plan['units'][0]['events'][-1]['session']=4
        self.audit={};self.policy=pt.build_retry_policy_migration(self.audit,[])
        row=deepcopy(self.plan['units'][0]['events'][-1]);row['session']=3
        self.valid={'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':row}]}
    def run_answers(self,answers):
        calls=[];snapshots=[];outer=self
        class Tracer:
            def chat_json(_,step,messages,**kw):
                calls.append({'payload':json.loads(messages[-1]['content']),'kwargs':kw})
                self.assert_pending(outer.audit,snapshots)
                value=answers.pop(0)
                if isinstance(value,BaseException):raise value
                return deepcopy(value)
        with patch.object(config,'pmap',sequential):
            result=pt.repair(self.wp,self.plan,{},Tracer(),self.audit,retry_policy=self.policy,checkpoint=lambda a:snapshots.append(deepcopy(a)))
        return result,calls,snapshots
    def assert_pending(self,audit,snapshots):
        row=audit['edit_attempts']['report-case'][-1]
        self.assertEqual(row['protocol_attempts'][-1]['status'],'requested')
        self.assertEqual(snapshots[-1]['edit_attempts']['report-case'][-1],row)
    def test_null_unhashable_key_then_valid_is_one_semantic_and_three_protocol(self):
        malformed=deepcopy(self.valid);malformed['changes'][0]['key']={'slot':'review-three'}
        result,calls,snapshots=self.run_answers([None,malformed,self.valid])
        compile_plan(self.wp,result)
        rows=self.audit['edit_attempts']['report-case'];self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['semantic_attempt'],1);self.assertEqual(len(rows[0]['protocol_attempts']),3)
        self.assertEqual([p['proposal'] for p in rows[0]['protocol_attempts']],[None,malformed,self.valid])
        self.assertEqual(calls[2]['payload']['protocol_feedback']['errors'][0]['path'],'changes[0].key')
        self.assertTrue(all(c['kwargs']['retries']==3 for c in calls))
        self.assertEqual(pt.retry_policy_counts(self.audit,self.policy),{'report-case':1})
    def test_valid_semantic_failure_uses_next_semantic_not_protocol(self):
        bad=deepcopy(self.valid);bad['changes'][0]['value']['session']=5
        result,calls,_=self.run_answers([bad,self.valid]);compile_plan(self.wp,result)
        self.assertEqual([r['semantic_attempt'] for r in self.audit['edit_attempts']['report-case']],[1,2])
        self.assertEqual([len(r['protocol_attempts']) for r in self.audit['edit_attempts']['report-case']],[1,1])
        self.assertEqual(calls[1]['payload']['retry_policy']['semantic_attempt'],2)
        self.assertNotIn('protocol_feedback',calls[1]['payload'])
    def test_original_observation_and_publication_add_infer_key_from_value(self):
        raw={'decision':'repair','changes':[
            {'unit_id':'report-case','collection':'observations','operation':'add','value':{'entity':'company-slot','field':'capital','sessions':[0,1]}},
            {'unit_id':'report-case','collection':'publications','operation':'add','value':{'fact_slot':'revision-two','channel':'复核记录','session':2}}]}
        self.assertEqual(pt._protocol_errors(self.wp,raw),[])
        candidate=pt.validate_unit_patch(self.wp,self.plan,raw,'report-case',{},6)
        self.assertEqual(len(candidate['units'][0]['observations']),2)
        self.assertEqual(len(candidate['units'][0]['publications']),2)
    def test_three_invalid_protocols_stop_without_new_semantic(self):
        with self.assertRaises(PlanConflict) as got:self.run_answers([None,[],{'decision':[]},self.valid])
        self.assertEqual(got.exception.findings[0]['code'],'repair_protocol_exhausted')
        self.assertEqual(len(self.audit['edit_attempts']['report-case']),1);self.assertNotIn('drafts',self.audit)
        before=deepcopy(self.audit)
        with self.assertRaises(PlanConflict):pt.repair(self.wp,self.plan,{},types.SimpleNamespace(chat_json=forbidden),self.audit,retry_policy=self.policy,resume_checkpoint=True)
        self.assertEqual(self.audit,before)
    def test_unresolved_provider_and_global_stop_never_receive_protocol_retries(self):
        cases=[({'decision':'unresolved','reason':'Original requirements conflict'},PlanConflict,'unresolved'),
            ({'__error__':'provider unavailable'},RuntimeError,'provider_error'),
            ({'__error__':'budget','__error_metadata__':{'global_stop':True,'kind':'estimated_budget_limit'}},ExecutionStopped,'provider_error')]
        for raw,error,status in cases:
            with self.subTest(status=status):
                self.audit={};self.policy=pt.build_retry_policy_migration({},[])
                with self.assertRaises(error):self.run_answers([None,raw,self.valid])
                rows=self.audit['edit_attempts']['report-case'];self.assertEqual(len(rows),1)
                self.assertEqual(len(rows[0]['protocol_attempts']),2);self.assertEqual(rows[0]['status'],status)
    def test_pending_boundary_is_durable_and_no_request_reentry(self):
        with self.assertRaises(Boundary):self.run_answers([Boundary()])
        row=self.audit['edit_attempts']['report-case'][0];self.assertEqual(row['status'],'call_failed')
        before=deepcopy(self.audit)
        with self.assertRaises(PlanConflict):pt.repair(self.wp,self.plan,{},types.SimpleNamespace(chat_json=forbidden),self.audit,retry_policy=self.policy,resume_checkpoint=True)
        self.assertEqual(self.audit,before)
    def test_legacy_default_keeps_old_three_response_count(self):
        answers=[None,None,None];calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):calls.append(1);return answers.pop(0)
        with patch.object(config,'pmap',sequential),self.assertRaises(PlanConflict):pt.repair(self.wp,self.plan,{},Tracer(),self.audit)
        self.assertGreaterEqual(len(calls),2);self.assertTrue(all('protocol_attempts' not in r for r in self.audit['edit_attempts']['report-case']))
    def test_real_json_parser_three_by_three_has_nine_physical_requests(self):
        answers=[]
        for parsed in (None,{'decision':[]},self.valid):answers+=['{','[',json.dumps(parsed)]
        calls=[]
        def transport(messages,**kw):calls.append(1);return answers.pop(0)
        with tempfile.TemporaryDirectory() as folder:
            tracer=RealTracer(types.SimpleNamespace(dir=Path(folder)))
            with patch.object(config,'pmap',sequential),patch.object(config,'chat',transport),patch('time.sleep',lambda _:None):
                result=pt.repair(self.wp,self.plan,{},tracer,self.audit,retry_policy=self.policy)
            trace=[json.loads(l) for l in tracer.pfile.with_name('llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        compile_plan(self.wp,result);self.assertEqual(len(calls),9)
        self.assertEqual(sum(r['event']=='json_attempt' for r in trace),9)
        self.assertEqual(len(self.audit['edit_attempts']['report-case']),1)
    def test_budget_admission_four_rejects_without_fourth_dispatch(self):
        admission=[];physical=[]
        def transport(messages,**kw):
            admission.append(1)
            if len(admission)==4:raise ExecutionStopped('fixture budget exhausted',kind='estimated_budget_limit')
            physical.append(1);return '{' if len(physical)<3 else json.dumps({'decision':[]})
        with tempfile.TemporaryDirectory() as folder:
            tracer=RealTracer(types.SimpleNamespace(dir=Path(folder)))
            with patch.object(config,'pmap',sequential),patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(ExecutionStopped):
                pt.repair(self.wp,self.plan,{},tracer,self.audit,retry_policy=self.policy)
        self.assertEqual(len(admission),4);self.assertEqual(len(physical),3)
        self.assertEqual(len(self.audit['edit_attempts']['report-case'][0]['protocol_attempts']),2)
    def test_exhausted_json_does_not_start_next_protocol(self):
        calls=[]
        def transport(messages,**kw):calls.append(1);return '{'
        with tempfile.TemporaryDirectory() as folder:
            tracer=RealTracer(types.SimpleNamespace(dir=Path(folder)))
            with patch.object(config,'pmap',sequential),patch.object(config,'chat',transport),patch('time.sleep',lambda _:None),self.assertRaises(RuntimeError):
                pt.repair(self.wp,self.plan,{},tracer,self.audit,retry_policy=self.policy)
        self.assertEqual(len(calls),3);self.assertEqual(len(self.audit['edit_attempts']['report-case'][0]['protocol_attempts']),1)
    def test_state_capacity_context_is_generic_and_preserves_all_effects(self):
        bp=deepcopy(self.wp);plan=deepcopy(self.plan)
        entity=bp['world_blueprint']['entity_types'][1];entity['fields'].append({'name':'phase-X','kind':'status','states':['a','b']})
        kind=next(e for e in bp['world_blueprint']['event_types'] if e['id']=='reviewed')
        kind['effect_fields'].append({'role':'report','field':'phase-X'})
        facts=[{'code':'state_write_capacity','affected_ids':['report-slot','phase-X','report-case']}]
        context=pt._state_capacity_context(bp,plan,facts)[0]
        self.assertEqual(context['declared_states'],['a','b']);self.assertEqual(context['writer_count'],1)
        self.assertEqual({e['field'] for e in context['writers'][0]['all_effect_fields']}, {e['field'] for e in kind['effect_fields']})

class RealA4PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get('PATCH_REAL_A4_EVIDENCE')
        if not source:raise unittest.SkipTest('Real A4 checkpoint evidence not supplied')
        cls.path=Path(source)
        def read(n):return json.loads((cls.path/n).read_text(encoding='utf-8'))
        from pipeline.initial_plan_recovery import _prepared_whitepaper
        cls.wp=_prepared_whitepaper(read('01_whitepaper.json'));cls.original=read('01_instance_plan_audit.json')['attempts'][0]['proposal']
        cls.audit=read('01_initial_plan_recovery.json')['transaction']
        cls.policy=pt.build_retry_policy_migration(cls.audit,[{'unit_id':'u_tower_case','first_attempt':2,'second_attempt':3}])
    def test_exact_mapping_preserves_all_raw_rows_and_only_tower_gets_two_semantic(self):
        before=deepcopy(self.audit);counts=pt.retry_policy_counts(self.audit,self.policy)
        self.assertEqual(counts,{'u_main_case':1,'u_seal_case':1,'u_tower_case':2,'u_institution_track':3,'u_missing_case':2})
        self.assertEqual(self.audit,before);self.assertEqual(len(self.policy['historical_protocol_pairs']),1)
        with self.assertRaises(PlanConflict):pt._restore_batch(self.wp,self.original,self.audit)
        restored=pt._restore_batch(self.wp,self.original,self.audit,self.policy)
        self.assertEqual(restored[0],self.audit['drafts'][-1]['candidate']);self.assertEqual(set(restored[4]),{'u_tower_case'})
    def test_any_semantic_difference_or_catalog_or_hash_tamper_refuses_mapping(self):
        pair=[{'unit_id':'u_tower_case','first_attempt':2,'second_attempt':3}]
        for mode in ('value','batch','selector','prefix'):
            audit=deepcopy(self.audit)
            if mode=='value':audit['edit_attempts']['u_tower_case'][2]['proposal']['changes'][1]['value']['session']=0
            elif mode=='batch':audit['edit_attempts']['u_tower_case'][2]['batch_iteration']=5
            elif mode=='selector':
                for u in audit['drafts'][2]['candidate']['units']:
                    if u['unit_id']=='u_tower_case':u['observations'].append(deepcopy(u['observations'][0]))
            else:audit['edit_attempts']['u_seal_case'][0]['status']='joint_rejected'
            with self.subTest(mode=mode),self.assertRaises(PlanConflict):
                if mode=='prefix':pt.retry_policy_counts(audit,self.policy)
                else:pt.build_retry_policy_migration(audit,pair)
    def test_first_request_semantic_three_protocol_one_with_actual_capacity_not_key_feedback(self):
        audit=deepcopy(self.audit);calls=[];snapshots=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):calls.append(json.loads(messages[-1]['content']));raise Boundary()
        with patch.object(config,'pmap',sequential),self.assertRaises(Boundary):
            pt.repair(self.wp,self.original,{},Tracer(),audit,retry_policy=self.policy,resume_checkpoint=True,checkpoint=lambda a:snapshots.append(deepcopy(a)))
        self.assertEqual(len(calls),1);payload=calls[0]
        self.assertEqual(payload['unit']['unit_id'],'u_tower_case')
        self.assertEqual([f['code'] for f in payload['findings']],['state_write_capacity'])
        context=payload['mechanical_capacity_context'][0];self.assertEqual(context['writer_count'],5);self.assertEqual(context['distinct_state_count'],4)
        self.assertEqual(len(context['writers']),5)
        armed=[w for w in context['writers'] if w['event_type']=='device_armed']
        self.assertTrue(all({v['field'] for v in w['all_effect_fields']}=={'武装阶段','运行读数'} for w in armed))
        row=audit['edit_attempts']['u_tower_case'][-1]
        self.assertEqual((row['attempt'],row['semantic_attempt'],row['protocol_attempts'][0]['protocol_attempt']),(4,3,1))
        for uid,rows in self.audit['edit_attempts'].items():self.assertEqual(audit['edit_attempts'][uid][:len(rows)],rows)
        self.assertEqual(audit['drafts'],self.audit['drafts'])
        self.assertEqual(snapshots[-2]['edit_attempts']['u_tower_case'][-1]['protocol_attempts'][0]['status'],'call_failed')
        before=deepcopy(audit)
        with self.assertRaises(PlanConflict):pt.repair(self.wp,self.original,{},types.SimpleNamespace(chat_json=forbidden),audit,retry_policy=self.policy,resume_checkpoint=True)
        self.assertEqual(audit,before)
    def test_last_semantic_rejection_cannot_open_fourth_semantic_or_reset_pair(self):
        audit=deepcopy(self.audit);calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kw):
                calls.append(json.loads(messages[-1]['content']))
                return {'decision':'repair','changes':[]}
        with patch.object(config,'pmap',sequential),self.assertRaises(PlanConflict) as got:
            pt.repair(self.wp,self.original,{},Tracer(),audit,retry_policy=self.policy,resume_checkpoint=True)
        self.assertEqual(len(calls),1);self.assertEqual(got.exception.findings[0]['code'],'repair_no_change')
        self.assertEqual(pt.retry_policy_counts(audit,self.policy)['u_tower_case'],3)
        self.assertEqual(len(audit['edit_attempts']['u_tower_case']),4)
        before=deepcopy(audit)
        with self.assertRaises(PlanConflict):pt.repair(self.wp,self.original,{},types.SimpleNamespace(chat_json=forbidden),audit,retry_policy=self.policy,resume_checkpoint=True)
        self.assertEqual(audit,before)

if __name__=='__main__':unittest.main()
