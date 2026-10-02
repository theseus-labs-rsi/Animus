"""No-network subset delivery integration; scripted semantics are not quality evidence."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from original_grounding_selftest import fixture, opinions, ANSWER_PROTOCOL
from semantic_review_fixture_helpers import audit_output, attach_targets
from pipeline import grounding_review as gr, quality, factory


class Script:
    def __init__(self, modes=None):
        self.modes, self.calls, self.index = modes or {}, [], -1

    def __call__(self, step, messages, **params):
        self.calls.append(step)
        payload = json.loads(messages[-1]['content'])
        if step.endswith('blind_read'):
            self.index += 1
            if self.modes.get(self.index) == 'provider':
                raise TimeoutError('offline provider failure')
            return opinions().responses[0]
        if 'original_reference' in payload:
            raw = audit_output(payload['original_reference'])
            if self.modes.get(self.index) in ('audit_format', 'all_invalid'):
                raw['claims'][0]['evidence'] = [{'doc_id': 'd000001', 'field': 'content',
                    'location_scope': 'field', 'quote': '北溟保险', 'role': 'support', 'explanation': '固定格式反例'}]
            return raw
        raw = attach_targets(opinions().responses[1], payload['reference_proposal'])
        if self.modes.get(self.index) == 'adjudicate_format':
            raw['original_rationale_review']['limitations'] = '应该是字符串数组'
        return raw


class ParallelScript:
    """Thread-safe valid opinions plus an overlap witness."""
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, step, messages, **params):
        with self.lock:
            self.calls.append(step)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            payload = json.loads(messages[-1]['content'])
            if step.endswith('blind_read'):
                return opinions().responses[0]
            if 'original_reference' in payload:
                return audit_output(payload['original_reference'])
            return attach_targets(opinions().responses[1], payload['reference_proposal'])
        finally:
            with self.lock:
                self.active -= 1


class IsolationTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        self.wp, self.ws, q, self.corpus, self.protocol = fixture()
        self.wp['quality_contract']['isolated_reference_audit'] = True
        self.questions = [{**deepcopy(q), 'qid': q['qid'] + f'_{i}'} for i in range(5)]

    def run_review(self, modes=None):
        script = Script(modes)
        kept, report, review = gr.review_grounding(self.questions, self.corpus, self.protocol,
            chat_json=script, model='test', isolated_reference=True)
        return kept, report, review, script

    def validate(self, review):
        candidates = gr.candidates_with_evidence(self.questions, self.corpus, isolated_reference=True)
        return gr.validate_delivery(candidates, self.corpus, self.protocol, review)

    def release(self, kept, review, target=None, *, derived=False):
        with tempfile.TemporaryDirectory() as td:
            candidates = gr.candidates_with_evidence(self.questions, self.corpus, isolated_reference=True)
            source_final, routing = gr.selection(candidates, review)
            selected = {q['qid'] for q in kept}
            routing['scoped_excluded'] = [q['qid'] for q in source_final if q['qid'] not in selected]
            artifacts = {'01_whitepaper.json': self.wp, '02_world.json': self.ws.to_dict(),
                '04_questions.json': self.questions, '05_corpus.json': self.corpus,
                '06_grounded_questions.json': kept, '06_semantic_review.json': review,
                '06_grounding_report.json': routing,
                '00_about.json': {'answer_protocol': ANSWER_PROTOCOL},
                'manifest.json': {'algo': {'targetspec': target or {}}}}
            for name, value in artifacts.items():
                (Path(td) / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
            if derived:
                artifacts['manifest.json']['derived_from'] = {'operation': 'filter_all_selected_systems_correct',
                    'semantic_review': {'artifact': gr.REVIEW_ARTIFACT,
                        'sha256': hashlib.sha256((Path(td) / gr.REVIEW_ARTIFACT).read_bytes()).hexdigest(),
                        'scope': 'unchanged_complete_source_review', 'source_final_count': len(source_final),
                        'selected_qids': [q['qid'] for q in kept]}}
                (Path(td) / 'manifest.json').write_text(json.dumps(artifacts['manifest.json']), encoding='utf-8')
            return quality.evaluate_release(td)

    def test_five_full_three_selected_replay_and_quality_pass_without_rewriting_raw(self):
        kept, report, review, script = self.run_review({1: 'audit_format', 4: 'adjudicate_format'})
        self.assertEqual((len(kept), report['n_pending'], report['n_dropped']), (3, 2, 0))
        self.assertFalse(report['execution_complete']); self.assertTrue(report['delivery_safe'], report)
        original = deepcopy(review)
        self.assertTrue(self.validate(review)['delivery_safe'])
        gr.validate_current_review(kept, self.corpus, self.protocol, review)
        result = self.release(kept, review)
        self.assertTrue(result['eligible'], result['issues'])
        self.assertEqual(result['checks']['partition']['source_count'], 5)
        self.assertEqual(result['checks']['partition']['counts'], {
            'released': 3, 'rejected': 0, 'pending_review': 2, 'scoped_excluded': 0})
        self.assertEqual(review, original)
        self.assertEqual(len(review['items']), 5); self.assertEqual(len(script.calls), 15)

    def test_parallel_questions_overlap_and_merge_in_source_order(self):
        script = ParallelScript()
        kept, report, review = gr.review_grounding(
            self.questions, self.corpus, self.protocol, chat_json=script,
            model='test', isolated_reference=True, workers=4)
        self.assertGreaterEqual(script.max_active, 2)
        self.assertEqual(len(script.calls), 15)
        self.assertEqual([item['source_qid'] for item in review['items']],
                         [question['qid'] for question in self.questions])
        self.assertEqual([question['qid'] for question in kept],
                         [question['qid'] for question in self.questions])
        self.assertTrue(report['execution_complete'])
        self.assertTrue(report['delivery_safe'], report)

    def test_bounded_candidate_error_keeps_other_certified_questions_deliverable(self):
        _, _, review, _ = self.run_review({1: 'audit_format'})
        item = review['items'][1]
        execution = {'status': 'model_error', 'error_type': 'NameError',
                     'message': "name 'candidate_context' is not defined"}
        item['reference_audit']['raw_output'] = None
        item['reference_audit']['proposal'] = None
        item['reference_audit']['proposal_ready'] = False
        item['reference_audit']['execution'] = deepcopy(execution)
        item['stage_execution']['reference_audit'] = deepcopy(execution)
        for event in review['records']:
            if (event.get('candidate_id') == item['candidate_id']
                    and event.get('stage') == 'reference_audit'
                    and event.get('event') == 'finished'):
                event['output'] = None
                event['execution'] = deepcopy(execution)
        review['execution_accounting'] = {
            'execution_stopped': False, 'suppressed_after_failure': 0,
            'unit_failure_isolation': 'bounded-units/v1'}
        delivery = self.validate(review)
        self.assertTrue(delivery['delivery_safe'], delivery)
        self.assertEqual(delivery['selected_count'], 4)
        self.assertEqual(len(delivery['isolated_execution_failures']), 1)

    def test_blind_location_failure_is_recorded_without_overriding_two_independent_checks(self):
        class BadBlindLocation(Script):
            def __call__(self, step, messages, **params):
                raw = super().__call__(step, messages, **params)
                if step.endswith('blind_read'):
                    raw = deepcopy(raw)
                    raw['evidence'][0]['quote'] = '材料中不存在的盲读引用'
                return raw
        script = BadBlindLocation()
        kept, report, review = gr.review_grounding(
            self.questions[:1], self.corpus, self.protocol, chat_json=script,
            model='test', isolated_reference=True)
        item = review['items'][0]
        self.assertEqual(item['stage_evidence_location']['blind_read']['status'], 'failed')
        self.assertEqual(item['stage_evidence_location']['adjudicate']['status'], 'located')
        self.assertEqual(item['stage_evidence_location']['reference_audit']['status'], 'located')
        self.assertEqual(item['item_certification']['status'], 'certified')
        self.assertEqual(item['evidence_location'],
                         {'status': 'failed', 'failed_stages': ['blind_read']})
        self.assertEqual((len(kept), report['n_pending']), (1, 0))
        self.assertTrue(gr.validate_delivery(
            gr.candidates_with_evidence(self.questions[:1], self.corpus,
                                        isolated_reference=True),
            self.corpus, self.protocol, review)['delivery_safe'])

    def test_all_format_invalid_cannot_deliver(self):
        kept, report, review, _ = self.run_review({i: 'all_invalid' for i in range(5)})
        self.assertEqual(kept, []); self.assertFalse(report['delivery_safe'])
        self.assertFalse(self.release(kept, review)['eligible'])

    def test_original_factory_publishes_only_certified_subset_and_retains_full_review(self):
        artifacts = {'01_whitepaper.json': self.wp, '02_world.json': self.ws.to_dict(),
            '04_questions.json': self.questions, '05_corpus.json': self.corpus,
            '00_about.json': {'answer_protocol': ANSWER_PROTOCOL}}
        script = Script({1: 'audit_format', 4: 'adjudicate_format'})
        manifest = {'config': {}, 'algo': {'targetspec': {'min_questions': 3}}}
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        run = SimpleNamespace(dir=Path(temporary.name), manifest=manifest, tracer=SimpleNamespace(chat_json=script),
            has=lambda name: name in artifacts, read=lambda name: deepcopy(artifacts[name]),
            write=lambda name, value: artifacts.__setitem__(name, deepcopy(value)),
            set_algo=lambda **values: manifest['algo'].update(values), log=lambda *args: None)
        factory.stage_grounding(run)
        self.assertEqual(len(artifacts['06_grounded_questions.json']), 3)
        self.assertEqual(len(artifacts['06_semantic_review.json']['items']), 5)
        self.assertFalse(artifacts['06_grounding_report.json']['execution_complete'])
        self.assertTrue(artifacts['06_grounding_report.json']['delivery_safe'])
        self.assertEqual(manifest['algo']['grounding']['n_pending'], 2)
        self.assertEqual(manifest['algo']['met_status'], 'MET')

    def test_question_bound_provider_failure_does_not_suppress_later_candidates(self):
        kept, report, review, script = self.run_review({1: 'provider'})
        self.assertEqual(len(kept), 4); self.assertFalse(report['execution_stopped'])
        self.assertEqual(len(script.calls), 15)
        self.assertTrue(report['delivery_safe']); self.assertTrue(self.release(kept, review)['eligible'])

    def test_hard_budget_failure_still_stops_later_provider_calls(self):
        class HardBudget(Script):
            def __call__(self, step, messages, **params):
                if step.endswith('blind_read') and self.index == 0:
                    self.calls.append(step)
                    self.index += 1
                    raise RuntimeError(
                        'Original experiment model/call/estimated budget limit; no provider dispatch')
                return super().__call__(step, messages, **params)
        script = HardBudget()
        kept, report, _ = gr.review_grounding(
            self.questions, self.corpus, self.protocol, chat_json=script,
            model='test', isolated_reference=True)
        self.assertEqual(len(kept), 1)
        self.assertTrue(report['execution_stopped'])
        self.assertGreater(report['suppressed_after_failure'], 0)

    def test_global_audit_failure_cannot_deliver(self):
        _, _, review, _ = self.run_review({1: 'audit_format'})
        review['audit_error'] = {'message': 'write failed'}
        self.assertFalse(self.validate(review)['delivery_safe'])

    def test_checkpoint_sink_failure_retains_review_and_returns_unsafe_delivery(self):
        with tempfile.TemporaryDirectory() as td, \
             patch('pipeline.run._atomic_write_json', side_effect=OSError('checkpoint disk failure')):
            kept, report, review = gr.review_grounding(
                self.questions[:1], self.corpus, self.protocol,
                chat_json=Script(), model='test', isolated_reference=True,
                checkpoint_dir=Path(td))
        self.assertEqual(kept, [])
        self.assertTrue(report['execution_stopped'])
        self.assertFalse(report['delivery_safe'])
        self.assertEqual(review['audit_error']['error_type'], 'OSError')

    def test_missing_reordered_or_changed_binding_refused(self):
        _, _, original, _ = self.run_review({1: 'audit_format'})
        for kind in ('missing', 'order', 'binding'):
            review = deepcopy(original)
            if kind == 'missing': review['items'].pop()
            elif kind == 'order': review['items'].reverse()
            else: review['items'][0]['binding']['question_hash'] = 'changed'
            with self.subTest(kind=kind): self.assertFalse(self.validate(review)['delivery_safe'])

    def test_missing_changed_or_duplicate_pending_raw_trace_refused(self):
        _, _, original, _ = self.run_review({1: 'audit_format'})
        candidate = original['items'][1]['candidate_id']
        for kind in ('missing', 'changed', 'duplicate'):
            review = deepcopy(original)
            index = next(i for i, r in enumerate(review['records']) if r['candidate_id'] == candidate
                         and r['stage'] == 'reference_audit' and r['event'] == 'finished')
            if kind == 'missing': review['records'].pop(index)
            elif kind == 'changed': review['records'][index]['output'] = {}
            else: review['records'].append(deepcopy(review['records'][index]))
            with self.subTest(kind=kind): self.assertFalse(self.validate(review)['delivery_safe'])

    def test_pending_cannot_be_included_or_forged_as_certified(self):
        kept, _, review, _ = self.run_review({1: 'audit_format'})
        candidates = gr.candidates_with_evidence(self.questions, self.corpus, isolated_reference=True)
        self.assertFalse(self.release(candidates, review)['eligible'])
        review['items'][1]['review_state'] = 'completed'
        self.assertFalse(self.validate(review)['delivery_safe'])

    def test_minimum_and_per_line_contracts_count_selected_only(self):
        kept, _, review, _ = self.run_review({1: 'audit_format', 4: 'adjudicate_format'})
        for target in ({'min_questions': 4}, {'min_questions': 1, 'per_line_min': {'L1_timeline': 4}}):
            result = self.release(kept, review, target)
            self.assertTrue(result['eligible'], result['issues'])
            self.assertNotIn('delivery_target_unmet', [x['code'] for x in result['issues']])
            self.assertIn('delivery_target_unmet', [x['code'] for x in result['warnings']])
            warning = next(x for x in result['warnings'] if x['code'] == 'delivery_target_unmet')
            self.assertEqual(warning['effect'], 'planning_shortfall_only')
            self.assertFalse(result['checks']['coverage']['delivery_target_met'])
        met = self.release(kept, review, {'min_questions': 3, 'per_line_min': {'L1_timeline': 3}})
        self.assertTrue(met['eligible'])
        self.assertNotIn('delivery_target_unmet', [x['code'] for x in met['warnings']])
        self.assertTrue(met['checks']['coverage']['delivery_target_met'])

    def test_existing_filter_derivation_reuses_uncropped_full_review_with_pending_rows(self):
        kept, _, review, _ = self.run_review({1: 'audit_format', 4: 'adjudicate_format'})
        original = deepcopy(review)
        report = self.release(kept[:2], review, derived=True)
        self.assertTrue(report['eligible'], report['issues'])
        self.assertEqual(review, original)
        self.assertEqual(report['checks']['partition']['source_count'], 5)
        self.assertEqual(report['checks']['partition']['counts']['scoped_excluded'], 1)
        self.assertEqual(report['checks']['coverage']['final_count'], 2)

    def test_budget_or_unknown_stage_is_not_a_format_failure(self):
        _, _, original, _ = self.run_review({1: 'audit_format'})
        for status in ('call_budget_exhausted', 'input_limit', 'model_error', 'not_run', 'invented'):
            review = deepcopy(original); review['items'][1]['stage_execution']['reference_audit']['status'] = status
            with self.subTest(status=status): self.assertFalse(self.validate(review)['delivery_safe'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
