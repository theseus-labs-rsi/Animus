"""No-network per-document review wiring tests; scripted opinions are not truth."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from pipeline import corpus_contract as cc
from pipeline import factory, render
from corpus_fixture_helpers import fixed_positive_review
from corpus_semantic_fidelity_selftest import Trace as RenderTrace, world
from corpus_review_format_repair_selftest import rehash
import config


def positive(payload):
    return fixed_positive_review([{'content': json.dumps(payload, ensure_ascii=False)}])


def uncertain(payload):
    raw = positive(payload)
    raw['verdict'] = 'fail'
    raw['document_reviews'][-1].update(status='uncertain', reason='固定意见：主体关系仍不确定。')
    return raw


def unsupported(payload):
    raw = positive(payload)
    raw['verdict'] = 'fail'
    raw['document_reviews'][-1].update(status='unsupported', reason='固定意见：本篇有额外错误。')
    raw['unsupported_claims'] = [{'doc_index': len(payload['documents']) - 1,
        'quote': payload['documents'][-1]['content'], 'reason': '固定离线错误意见，仅验证处理链路。'}]
    return raw


class Script:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if not self.replies:
            raise AssertionError('Unexpected additional review call')
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return deepcopy(reply(json.loads(messages[-1]['content'])) if callable(reply) else reply)


class DocumentReviewTests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(socket.socket, 'connect', side_effect=AssertionError('No network')),
                      patch.object(config, 'pmap', side_effect=lambda fn, xs, **kw: [fn(x) for x in xs])):
            guard.start(); self.addCleanup(guard.stop)
        self.ws = world()
        self.docs = [{'title': '阶段记录', 'content': '甲项目目前已进入收尾。'},
                     {'title': '归档', 'content': '本组记录在当日归档。'}]

    def review(self, *replies):
        trace = Script(*replies)
        result = cc.review_documents(trace, self.ws, 0, self.docs,
            requirements=cc.fidelity_requirements(self.ws, 0))
        return result, trace

    def test_each_document_visible_even_when_requirements_only_cite_first(self):
        report, trace = self.review(positive)
        self.assertEqual(report['status'], 'passed')
        payload = json.loads(trace.calls[0][1][-1]['content'])
        self.assertEqual([x['doc_index'] for x in payload['document_review_rows_to_complete']], [0, 1])
        self.assertEqual([x['doc_index'] for x in report['document_reviews']], [0, 1])
        self.assertEqual({e['doc_index'] for row in report['fidelity_coverage'] for e in row['evidence']}, {0})
        self.assertEqual((cc.VERSION, factory.CORPUS_RENDER_CONTRACT_VERSION), (11, 12))

    def test_missing_duplicate_invalid_index_and_empty_reason_are_format_errors(self):
        def reply(kind):
            def mutate(payload):
                raw = positive(payload)
                if kind == 'missing': raw['document_reviews'].pop()
                elif kind == 'duplicate': raw['document_reviews'][1]['doc_index'] = 0
                elif kind == 'bool': raw['document_reviews'][1]['doc_index'] = True
                elif kind == 'range': raw['document_reviews'][1]['doc_index'] = 20
                elif kind == 'reason': raw['document_reviews'][1]['reason'] = ' '
                else: raw.pop('document_reviews')
                return raw
            return mutate
        for kind in ('missing', 'duplicate', 'bool', 'range', 'reason', 'old_three_keys'):
            with self.subTest(kind=kind):
                fn = reply(kind); report, trace = self.review(fn, fn)
                self.assertEqual((report['status'], len(trace.calls)), ('error', 2))
                self.assertTrue(report['review_validation']['attempts'][0]['validation_error']['path'].startswith('/document_reviews'))

    def test_uncertain_without_claim_is_semantic_failure_once_and_not_certified(self):
        report, trace = self.review(uncertain)
        self.assertEqual((report['status'], len(trace.calls)), ('failed', 1))
        self.assertEqual(report['raw_output']['unsupported_claims'], [])
        self.assertEqual(report['issues'][0]['code'], 'document_not_supported')
        self.assertIsNone(report['review_validation']['attempts'][0]['validation_error'])
        with self.assertRaises(ValueError): cc.attach_receipts(deepcopy(self.docs), report, 0)

    def test_unsupported_needs_original_quote_and_cannot_also_claim_supported(self):
        for kind in ('no_claim', 'wrong_quote', 'supported_conflict'):
            def mutate(payload, kind=kind):
                raw = unsupported(payload)
                if kind == 'no_claim': raw['unsupported_claims'] = []
                elif kind == 'wrong_quote': raw['unsupported_claims'][0]['quote'] = '不存在的原句'
                else: raw['document_reviews'][-1]['status'] = 'supported'
                return raw
            with self.subTest(kind=kind):
                report, trace = self.review(mutate, mutate)
                self.assertEqual((report['status'], len(trace.calls)), ('error', 2))
        report, trace = self.review(unsupported)
        self.assertEqual((report['status'], len(trace.calls)), ('failed', 1))

    def test_overall_pass_cannot_ignore_uncertain_doc(self):
        def incorrect_aggregate(payload):
            raw = uncertain(payload); raw['verdict'] = 'pass'; return raw
        report, trace = self.review(incorrect_aggregate, incorrect_aggregate)
        self.assertEqual((report['status'], len(trace.calls)), ('error', 2))
        self.assertEqual(report['review_validation']['attempts'][0]['validation_error']['path'], '/verdict')

    def test_format_repair_preserves_original_inputs_and_both_raw_responses(self):
        def missing(payload):
            raw = positive(payload); raw.pop('document_reviews'); return raw
        report, trace = self.review(missing, positive)
        self.assertEqual((report['status'], len(trace.calls)), ('passed', 2))
        first, second = [json.loads(row[1][-1]['content']) for row in trace.calls]
        feedback = second.pop('format_repair')
        self.assertEqual(first, second)
        self.assertEqual(feedback['previous_response'], report['review_validation']['attempts'][0]['raw_output'])
        self.assertIn('document_reviews', feedback['instructions'])
        self.assertNotIn('完整三个键', feedback['instructions'])
        self.assertNotIn('document_reviews', feedback['previous_response'])
        self.assertIn('document_reviews', report['raw_output'])

    def test_provider_failure_stays_one_call_without_fake_semantic_verdict(self):
        report, trace = self.review(TimeoutError('offline timeout'))
        self.assertEqual((report['status'], len(trace.calls)), ('error', 1))
        self.assertIsNotNone(report['review_validation']['attempts'][0]['execution_error'])

    def test_original_author_feedback_receives_uncertain_doc_reason(self):
        trace = RenderTrace(outputs=[uncertain, positive])
        corpus = {'sessions': []}
        render.render_corpus({'quality_contract': {'corpus_review': True}}, self.ws, 0,
            trace, corpus, set(), lambda: None, log=lambda *_: None)
        steps = [row[0] for row in trace.calls]
        self.assertEqual(steps.count('render.signal'), 2)
        self.assertEqual(steps.count('corpus.review'), 2)
        rewritten_prompt = [row[1][-1]['content'] for row in trace.calls if row[0] == 'render.signal'][1]
        self.assertIn('document_not_supported', rewritten_prompt)
        self.assertIn('主体关系仍不确定', rewritten_prompt)
        self.assertEqual(cc.validate_corpus(self.ws, corpus)['status'], 'passed')

    def test_receipt_replay_rejects_removed_or_changed_document_opinion(self):
        report, _ = self.review(positive)
        docs = deepcopy(self.docs); cc.attach_receipts(docs, report, 0)
        self.assertTrue(all(cc._review_receipt_matches(self.ws, 0, doc) for doc in docs))
        for kind in ('missing', 'uncertain', 'old_version'):
            with self.subTest(kind=kind):
                doc = deepcopy(docs[1]); receipt = doc['quality_review']
                if kind == 'old_version': receipt['version'] = 7
                else:
                    raw = receipt['review_validation']['attempts'][-1]['raw_output']
                    if kind == 'missing': raw['document_reviews'].pop()
                    else: raw['document_reviews'][1]['status'] = 'uncertain'
                    rehash(receipt)
                self.assertFalse(cc._review_receipt_matches(self.ws, 0, doc))

    def test_full_corpus_validation_replays_one_shared_history_once(self):
        report, _ = self.review(positive)
        docs = deepcopy(self.docs); cc.attach_receipts(docs, report, 0)
        corpus = {'sessions': [{'session_id': 0, 'docs': docs}]}
        original = cc._replay_review_validation
        with patch.object(cc, '_replay_review_validation', wraps=original) as replay:
            cc.validate_corpus(self.ws, corpus)
        self.assertEqual(replay.call_count, 1)

    def test_report_mirror_does_not_override_bound_raw(self):
        report, _ = self.review(positive)
        report['document_reviews'][1]['reason'] = 'Changed opinion outside the actual raw response'
        with self.assertRaises(ValueError): cc.attach_receipts(deepcopy(self.docs), report, 0)


if __name__ == '__main__':
    unittest.main()
