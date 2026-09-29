"""Recorded billed syntax failures reopen only with explicit budget preservation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from original_smoke_json_retry_selftest import ReplaySDK
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke


class FormatRecoveryTests(unittest.TestCase):
    setUp=fixture.CorpusReasoningOverrideTests.setUp
    run_tool=fixture.CorpusReasoningOverrideTests.run_tool

    def evidence(self):
        profile={'model':'glm-5.3-flash','stopped':True,'admitted_calls':2,'max_calls':1200,
            'max_cny':12,'budget_consumed_cny':4.2,'unsettled_reservation_calls':1,'source_drift':[],
            'stop_reason':{'reason':'json_execution_failure','kind':'json_syntax','step':'world.instance.values'}}
        rows=[{'event':'request','call_id':'old','model':profile['model']},
            {'event':'request','call_id':'bad','model':profile['model'],'operation_id':'op','step':'world.instance.values'},
            {'event':'response','call_id':'bad','operation_id':'op','response':{
                'usage':{'prompt_tokens':100,'completion_tokens':20},
                'choices':[{'finish_reason':'stop','content':'{"bad":'}]}},
            {'event':'json_error','operation_id':'op','step':'world.instance.values','kind':'json_syntax',
                'error_type':'JSONDecodeError','raw_output':'{"bad":'}]
        return profile,rows

    def check(self,p,rows):
        path=self.directory/'trace.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
        return smoke.json_format_resume_evidence(p,path)

    def test_closed_billed_syntax_attempt_preserves_old_unknown_and_ledger(self):
        p,rows=self.evidence();before=deepcopy(p);proof=self.check(p,rows)
        self.assertEqual(p,before);self.assertEqual(proof['prior_admitted_calls'],2)
        self.assertEqual(proof['prior_budget_consumed_cny'],4.2)
        self.assertEqual(proof['unknown_reservations_preserved'],1)

    def test_auth_drift_exhaustion_and_wrong_model_remain_stopped(self):
        for key,value in [('max_calls',2),('max_cny',4.2),('source_drift',['config.py']),
                ('model','other'),('stop_reason',{'reason':'provider_failure','kind':'http_status_401'})]:
            p,rows=self.evidence();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(p,rows)

    def test_incomplete_unbilled_or_unbound_body_keeps_admission_closed(self):
        p,rows=self.evidence()
        for change in [{'usage':None},{'choices':[{'finish_reason':'length','content':'{"bad":'}]},
                {'choices':[{'finish_reason':'stop','content':'different'}]}]:
            bad=deepcopy(rows);bad[2]['response'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(p,bad)
        for missing in range(len(rows)):
            with self.subTest(missing=missing),self.assertRaises(ValueError):self.check(p,rows[:missing]+rows[missing+1:])
        rows[-1]['raw_output']='{}'
        with self.assertRaises(ValueError):self.check(p,rows)

    def test_running_or_terminal_replay_needs_no_reopening(self):
        p,rows=self.evidence();p['stopped']=False;self.assertIsNone(self.check(p,rows))

    def test_later_blocked_logical_calls_keep_the_actual_billed_failure_binding(self):
        p, rows = self.evidence()
        rows.append({'event':'json_error','step':'council.world_repair','operation_id':'blocked',
            'kind':'json_syntax','error_type':'RuntimeError','raw_output':'',
            'error':'wrapper already_stopped; no provider dispatch'})
        proof = self.check(p, rows)
        self.assertEqual(proof['error_call_ids'], ['bad'])
        self.assertEqual(proof['prior_admitted_calls'], 2)

    def test_real_wrapper_accounts_all_three_billed_invalid_responses(self):
        reply=fixture.SimpleNamespace(id='syntax',model='gpt-5.4-mini',
            usage=fixture.SimpleNamespace(prompt_tokens=100,completion_tokens=20,total_tokens=120),
            choices=[fixture.SimpleNamespace(index=0,finish_reason='stop',
                message=fixture.SimpleNamespace(content='{"bad":',refusal=None))])
        api=ReplaySDK([reply,reply,reply])
        def call(directory,cfg):
            fixture.Tracer(fixture.SimpleNamespace(dir=directory)).chat_json('world.instance.values',[],retries=3)
        with patch.object(fixture,'FakeSDK',return_value=api),patch('time.sleep'):
            result=self.run_tool(call,['--settle-reported-usage'])
        proof=smoke.json_format_resume_evidence(result.profile,result.directory/'llm_attempts.jsonl')
        self.assertEqual(result.profile['admitted_calls'],3)
        self.assertEqual(result.profile['settled_usage_calls'],3)
        self.assertEqual(len(proof['error_call_ids']),3)
        self.assertGreater(result.profile['reported_usage_estimate_cny'],0)

if __name__=='__main__':unittest.main(verbosity=2)
