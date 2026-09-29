"""Offline retry-policy checkpoint binding and wrapper option regression tests."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import initial_plan_recovery_selftest as guards
from pipeline import initial_plan_recovery as recovery
from tools import run_original_bc_smoke as smoke
from copy import deepcopy
import json
import unittest


class RetryBindingTests(unittest.TestCase):
    def operation(self, raw=None, uid='tower', operation='op', syntax=False):
        if raw is None: raw={'decision':'repair', 'changes':[]}
        step='council.instance_plan_unit_repair'
        messages=[{'role':'user','content':json.dumps({'unit':{'unit_id':uid}})}]
        events=[]
        def event(event_name, **kw):
            row={'event':event_name,'step':step,'operation_id':operation,**kw}
            events.append(row)
            return row
        original=deepcopy(messages)
        if syntax:
            event('json_attempt',attempt=1,attempt_id='try1')
            event('request',call_id='bad',messages=deepcopy(messages))
            event('response',call_id='bad',response={'choices':[{'content':'{bad'}]})
            event('json_error',attempt_id='try1',kind='json_syntax',error_type='JSONDecodeError',
                  retryable=True,retry_action='json_feedback',raw_output='{bad')
            event('json_retry_feedback',attempt_id='try1',next_attempt=2)
            messages += [{'role':'assistant','content':'{bad'}, {'role':'user','content':'fix syntax'}]
        event('json_attempt',attempt=2 if syntax else 1,attempt_id='final')
        event('request',call_id=operation,messages=deepcopy(messages))
        event('response',call_id=operation,response={'choices':[{'content':json.dumps(raw)}]})
        event('json_result',parsed=raw)
        return events,[{'step':step,'messages':original,'output':raw,'ok':True}]

    def bind(self, trace, prompts):
        return recovery._author_operation_bindings(trace,prompts,
            [r for r in trace if r['event']=='request'],[r for r in trace if r['event']=='response'])

    def test_syntax_retries_remain_distinct_paid_calls_one_author_response(self):
        trace,prompts=self.operation(syntax=True)
        bound=self.bind(trace,prompts)
        self.assertEqual(bound[0]['physical_call_ids'],['bad','op'])
        self.assertEqual(len(bound),1)

    def test_final_parse_failure_never_becomes_saved_reply(self):
        trace,prompts=self.operation(syntax=True)
        trace[-2]['response']['choices'][0]['content']='{bad'
        with self.assertRaises(ValueError):self.bind(trace,prompts)

    def test_complete_code_fence_matches_existing_strict_parser(self):
        trace,prompts=self.operation()
        trace[-2]['response']['choices'][0]['content']='```json\n'+trace[-2]['response']['choices'][0]['content']+'\n```'
        trace[-1]['parse_mode']='complete'
        self.assertEqual(len(self.bind(trace,prompts)),1)

    def test_substring_or_repaired_parse_does_not_relax_bound_strict_parser(self):
        trace,prompts=self.operation()
        trace[-1]['parse_mode']='repaired'
        with self.assertRaises(ValueError):self.bind(trace,prompts)

    def test_changed_original_payload_on_json_retry_rejected(self):
        trace,prompts=self.operation(syntax=True)
        next(r for r in trace if r['event']=='request' and r['call_id']=='op')['messages'][0]['content']='changed'
        with self.assertRaises(ValueError):self.bind(trace,prompts)

    def test_unproven_syntax_error_rejected(self):
        trace,prompts=self.operation(syntax=True)
        next(r for r in trace if r['event']=='json_error')['raw_output']='different'
        with self.assertRaises(ValueError):self.bind(trace,prompts)

    def test_syntax_chain_requires_one_final_result(self):
        trace,prompts=self.operation(syntax=True)
        trace.pop()
        with self.assertRaises(ValueError):self.bind(trace,prompts)

    def new_rows(self, exhausted=False):
        raw={'decision':'repair','changes':[]}
        protocol=[{'protocol_attempt':1,'status':'protocol_rejected','proposal':[]},
                  {'protocol_attempt':2,'status':'protocol_valid','proposal':raw}]
        if exhausted:
            protocol[1].update(status='protocol_rejected',proposal=None)
            protocol.append({'protocol_attempt':3,'status':'protocol_rejected','proposal':{}})
        policy={'version':'separate-author-protocol/v2','legacy_attempt_prefix':{},'historical_protocol_pairs':[]}
        row={'attempt':1,'semantic_attempt':1,'retry_policy':policy['version'],
             'status':'protocol_exhausted' if exhausted else 'joint_rejected',
             'protocol_attempts':protocol,'proposal':protocol[-1]['proposal']}
        bound=[]
        for n,part in enumerate(protocol):
            trace,prompts=self.operation(raw=part['proposal'],operation=str(n))
            payload={'unit':{'unit_id':'tower'},'retry_policy':{'version':policy['version'],
                'semantic_attempt':1,'max_semantic_attempts':3,'max_protocol_attempts':3,'max_json_attempts':3}}
            if n:payload['protocol_feedback']={'previous_response':protocol[n-1]['proposal']}
            message={'role':'user','content':json.dumps(payload)}
            for ev in trace:
                if ev['event']=='request':ev['messages']=[message]
            prompts[0]['messages']=[message]
            if part['proposal'] is None:
                for ev in trace:
                    if ev['event']=='response':ev['response']['choices'][0]['content']='null'
                    if ev['event']=='json_result':ev['parsed']=None
                prompts[0]['output']=None
            bound+=self.bind(trace,prompts)
        return {'tower':[row]},bound,policy

    def test_v2_bad_shape_protocol_replies_all_bind_to_original_semantic_slot(self):
        rows,bound,policy=self.new_rows()
        result=recovery._bind_checkpoint_author_rows(rows,bound,policy)
        self.assertEqual([r['protocol_attempt'] for r in result],[1,2])
        self.assertEqual([r['semantic_attempt'] for r in result],[1,1])

    def test_protocol_exhaustion_still_retains_three_paid_operations(self):
        rows,bound,policy=self.new_rows(exhausted=True)
        self.assertEqual(len(recovery._bind_checkpoint_author_rows(rows,bound,policy)),3)

    def test_requested_protocol_cannot_silently_receive_another_attempt(self):
        rows,bound,policy=self.new_rows()
        rows['tower'][0]['protocol_attempts'][-1]['status']='requested'
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound,policy)

    def test_protocol_reply_cannot_hide_unbound_operation(self):
        rows,bound,policy=self.new_rows()
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound+[deepcopy(bound[-1])],policy)

    def test_semantic_attempt_cannot_be_renumbered(self):
        rows,bound,policy=self.new_rows()
        rows['tower'][0]['semantic_attempt']=0
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound,policy)

    def test_parent_proposal_must_be_final_protocol_response(self):
        rows,bound,policy=self.new_rows()
        rows['tower'][0]['proposal']={'decision':'unresolved'}
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound,policy)

    def test_physical_semantic_counter_must_match_saved_slot(self):
        rows,bound,policy=self.new_rows()
        bound[-1]['request_payload']['retry_policy']['semantic_attempt']=2
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound,policy)

    def test_protocol_feedback_cannot_change_original_task(self):
        rows,bound,policy=self.new_rows()
        bound[-1]['request_payload']['findings']=['changed']
        with self.assertRaises(ValueError):recovery._bind_checkpoint_author_rows(rows,bound,policy)

    def test_policy_flag_requires_receipt_policy_and_source_digest(self):
        smoke._validate_plan_retry_policy_option(None,{})
        with self.assertRaises(ValueError):smoke._validate_plan_retry_policy_option('separate-protocol-v2',{})
        policy={'version':'separate-author-protocol/v2'}
        sha=recovery._digest(policy)
        permit={'plan_retry_policy':'separate-protocol-v2','retry_policy':policy,
                'retry_policy_sha256':sha,'source_migration':{'retry_policy_sha256':sha}}
        smoke._validate_plan_retry_policy_option('separate-protocol-v2',permit)
        with self.assertRaises(ValueError):smoke._validate_plan_retry_policy_option(None,permit)
        permit['retry_policy']['max_semantic_attempts']=4
        with self.assertRaises(ValueError):smoke._validate_plan_retry_policy_option('separate-protocol-v2',permit)


if __name__=='__main__':unittest.main(verbosity=2)
