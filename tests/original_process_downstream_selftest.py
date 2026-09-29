"""No-network counterexamples for original L3 process semantic routing.

Scripted model opinions test wiring only; they do not establish answer quality.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

from eval import judge, grading, multi_system, semantic_judge
from eval.answer_task_review import POLICY_VERSION, append_scoring_policy
from eval.provenance import reference_hash
from pipeline import grounding_review
from pipeline.semantic_review import review_questions, prepare_review
from semantic_judge_selftest import grade_output
from structured_review_fixture_helpers import StructuredScript

PROTOCOL = append_scoring_policy('仅依据这些公开材料。')
CORPUS = {'world': 'PRIVATE_CORPUS_WORLD', 'sessions': [{'session_id': 0, 'date': '2025-01-01',
    'docs': [{'doc_id': 'original-1', 'title': 'PRIVATE_TITLE', 'content': '公开记录明确需重新核对。',
              'fact_refs': ['PRIVATE_FACT_METADATA']}]}]}

def question():
    ref = {'answer': {'结论': '需要重新核对'}, 'rationale': '公开正文明确记录。'}
    witness = {'events': [{'id': 'PRIVATE_EVENT', 'effect': 'PRIVATE_WORLD_VALUE'}]}
    process = {'intent': '截至第1周，需要怎样处理？', 'at_session': 0,
               'witness_refs': {'event_ids': ['PRIVATE_EVENT']}, 'world_hash': 'PRIVATE_WORLD_HASH'}
    return {'qid': 'process-q', 'line': 'L3_process', 'capability': 'L3_process_trace',
        'entity': '公开对象', 'field': '', 'question': '截至第1周，需要怎样处理？',
        'gt': witness, 'reference_proposal': ref, 'evidence_sessions': [0],
        'world': 'PRIVATE_RAW_WORLD',
        'aux': {'process': process, 'scorer': 'semantic', 'gt_scope': 'canonical_witness_not_semantic_proof'},
        'question_contract': {'version': 1, 'answer_kind': 'structured', 'value_schema': {},
            'scoring_policy': POLICY_VERSION, 'scoring_scope': 'task_with_supporting_reasons',
            'render_policy': 'semantic_review', 'reference_authority': 'proposed_not_mechanically_proven',
            'gold': deepcopy(ref['answer']), 'reference_proposal': deepcopy(ref),
            'canonical_witness': deepcopy(witness), 'parameters': {'process': deepcopy(process)}}}

WP = {'quality_contract': {'scoring_policy': POLICY_VERSION,
    'public_semantic_review': True, 'isolated_reference_audit': True}}

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('No network allowed'))
        cls.network.start()

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()

    def test_legacy_scoring_explicitly_refuses_witness(self):
        q = question()
        self.assertTrue(grading.requires_semantic_grading(q))
        self.assertFalse(judge.is_judgeable(q))
        self.assertEqual(judge.judge_spec(q), (None, None, None))
        with patch.object(judge.config, 'chat_json', side_effect=AssertionError('No API')):
            result = judge.judge_record(q, json.dumps(q['gt']), use_llm=True)
        self.assertEqual(result['verdict'], 'unjudgeable')
        self.assertEqual(result['path'], 'semantic_mode_required')
        self.assertIsNone(result['correct'])

    def test_old_order_route_unchanged(self):
        q = {'capability': 'L3_order', 'gt': [{'field': '状态', 'value': '完成'}]}
        self.assertFalse(grading.requires_semantic_grading(q))
        self.assertEqual(judge.judge_spec(q)[0], 'order')
        self.assertTrue(judge.is_judgeable(q))



    def test_grounding_never_fills_missing_reference_from_private_witness(self):
        q = question(); del q['reference_proposal']
        for isolated in (True, False):
            with self.subTest(isolated=isolated), self.assertRaises(ValueError):
                grounding_review.candidates_with_evidence([q], CORPUS, isolated_reference=isolated)

    def test_grounding_keeps_explicit_reference_and_witness_separate(self):
        q = question(); before = deepcopy(q)
        [row] = grounding_review.candidates_with_evidence([q], CORPUS, isolated_reference=True)
        self.assertEqual(row['reference_proposal'], before['reference_proposal'])
        self.assertEqual(row['gt'], before['gt'])
        self.assertNotIn('reference_projection', row)
        self.assertEqual(q, before)

    def test_actual_three_roles_exclude_hidden_witness_from_model_payloads(self):
        script = StructuredScript()
        q = question()
        report = review_questions([q], CORPUS, PROTOCOL, chat_json=script,
            reviewer_model='offline', reference_auditor_model='offline', max_calls=3)
        self.assertEqual(report['calls_used'], 3)
        self.assertEqual(report['items'][0]['reference_proposal'], q['reference_proposal'])
        self.assertEqual(len(script.calls), 3)
        blind, auditor, arbiter = [call['payload'] for call in script.calls]
        self.assertEqual(set(blind), {'question', 'documents', 'public_protocol', 'input_scope'})
        for payload in (blind, auditor, arbiter):
            self.assertNotIn('PRIVATE_', json.dumps(payload))
        self.assertNotIn('reference_proposal', blind)
        self.assertEqual(auditor['original_reference'], q['reference_proposal'])
        self.assertNotIn('BLIND_ONLY_SENTINEL', json.dumps(auditor))
        self.assertEqual(arbiter['reference_proposal'], q['reference_proposal'])

    def test_reference_identity_changes_if_witness_process_or_reference_changes(self):
        q = question(); base = reference_hash(q)
        for field in ('gt', 'aux', 'reference_proposal', 'question_contract'):
            variant = deepcopy(q); variant[field] = {'changed': field}
            with self.subTest(field=field):
                self.assertNotEqual(reference_hash(variant), base)

    def test_semantic_judge_rejects_missing_policy_or_isolated_audit(self):
        q = question()
        for report, policy in [({}, None), ({}, POLICY_VERSION),
                               ({'binding': {'reference_auditor_model': 'offline'}}, None)]:
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'Process questions require'):
                semantic_judge.SemanticJudge(report, [q], CORPUS, PROTOCOL,
                    model='offline', chat_json=lambda *a, **k: self.fail('No API'), scoring_policy=policy)

    def test_direct_run_system_rejects_protocol_removal_before_solver_calls(self):
        with self.assertRaisesRegex(ValueError, 'Process questions require'):
            multi_system.run_system('offline', [question()], object(), protocol='', verbose=False)

    def test_full_semantic_grade_uses_natural_reference_and_no_witness(self):
        q = question(); script = StructuredScript()
        report = review_questions([q], CORPUS, PROTOCOL, chat_json=script,
            reviewer_model='offline', reference_auditor_model='offline', max_calls=3)
        calls = []
        def grade(step, messages, **params):
            calls.append(json.loads(messages[1]['content']))
            return grade_output(evidence=[], answer_task_review_response='OFFLINE direct method fixture.')
        grader = semantic_judge.SemanticJudge(report, [q], CORPUS, PROTOCOL,
            model='offline', chat_json=grade, scoring_policy=POLICY_VERSION, max_calls=1)
        result = grader(q, '需要重新核对。')
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['verdict'], 'correct')
        self.assertEqual(result['scoring_scope'], 'task_with_supporting_reasons')
        self.assertEqual(calls[0]['reference_review']['reference_proposal'], q['reference_proposal'])
        self.assertNotIn('PRIVATE_', json.dumps(calls[0]))
        # Source receipts bind even private witness metadata; changing it invalidates reuse.
        changed = deepcopy(q); changed['gt']['events'][0]['id'] = 'different'
        after = grader(changed, '需要重新核对。')
        self.assertEqual(after['verdict'], 'uncertain')
        self.assertEqual(len(calls), 1)



if __name__ == '__main__':
    unittest.main()
