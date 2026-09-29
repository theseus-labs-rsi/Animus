"""Original review coverage scheduling, offline only; no semantic-quality claims."""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from pipeline import world_semantics as review
from pipeline.world_state import WorldState
from world_disclosure_record_review_selftest import planned


def empty_state():
    return {'issues': [], 'mechanism_coverage': [], 'disclosure_reviews': [],
            'note': '', 'revisions': []}


class PendingReply:
    """A marked offline reply fixture, never used as production input."""
    def __init__(self, folder=None):
        self.calls = []
        if folder is not None:
            self.pfile = Path(folder) / 'prompts.jsonl'

    def chat_json(self, step, messages, **params):
        payload = json.loads(messages[-1]['content'])
        self.calls.append(deepcopy(payload))
        work = payload['working_state']
        pending = work['coverage']['missing_disclosure_records']
        if pending:
            ids = {row['record_id'] for row in pending}
            contexts = [node['value'] for node in payload['exact_reads'].values()
                        if isinstance(node.get('value'), dict)
                        and node['value'].get('record_id') in ids]
            if not contexts and not payload['unread_ids']:
                raise AssertionError('Missing opinions must receive exact original joined contexts')
            return {'action': 'submit', 'issues': [], 'mechanism_coverage': [],
                    'disclosure_reviews': [{'record_id': row['record_id'],
                        'understanding': 'Offline transport fixture, not a model opinion.',
                        'refs': [row['record_ref_id']],
                        'reason': 'Offline fixture deliberately keeps the semantic judgment unresolved.',
                        'status': 'unresolved'} for row in contexts],
                    'note': 'Offline fixture submits only the explicitly pending original records.'}
        return {'action': 'finish', 'decision': 'unresolved',
                'reason': 'Offline fixture does not certify the business meaning.',
                'limitations': 'No real model calls in this test.',
                'repair_targets': {'intrinsic': [], 'structure': False, 'disclosure': False}}


class WorkQueueTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, 'connect', side_effect=AssertionError('Network prohibited')).start()
        self.wp, self.ws, self.task = planned()
        self.payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        self.window = review._scoped_review_context(self.payload)
        self.window.read = set(self.window.nodes)
        self.state = empty_state()

    def test_all_material_read_does_not_imply_all_opinions_submitted(self):
        working = review._review_request_state(self.payload, self.window, self.state)
        self.assertFalse(working['coverage']['complete'])
        self.assertIn('Use submit', working['required_next_action'])
        self.assertNotIn('Output finish now', working['required_next_action'])
        self.assertEqual(len(working['coverage']['missing_disclosure_records']), len(self.ws.disclosure['records']))

    def test_pending_context_is_original_and_does_not_mutate_world_or_submit_opinions(self):
        original = deepcopy(self.payload)
        original_read = deepcopy(self.window.read)
        review._review_request_state(self.payload, self.window, self.state)
        expected = {row['record_ref_id']: row for row in self.payload['disclosure_record_contexts']}
        for key, node in self.window.visible.items():
            self.assertEqual(node['value'], expected[key])
        self.assertEqual(self.payload, original)
        self.assertEqual(self.window.read, original_read)
        self.assertEqual(self.state['disclosure_reviews'], [])

    def test_final_parser_still_rejects_missing_opinions(self):
        raw = {'decision': 'accept', 'reason': 'Offline fixture.', 'limitations': 'Offline fixture.',
               'issues': [], 'mechanism_coverage': [], 'disclosure_reviews': [],
               'repair_targets': {'intrinsic': [], 'structure': False, 'disclosure': False}}
        with self.assertRaisesRegex(ValueError, 'read every disclosure record'):
            review._parse(raw, self.payload)
        result, advance = review._review_action(raw, self.window, self.state, self.payload)
        self.assertIsNone(result)
        self.assertTrue(advance)
        self.assertNotIn('pending_consolidated_final', self.state)

    def test_premature_finish_schedules_work_without_certifying(self):
        result, advance = review._review_action(
            {'action': 'finish', 'decision': 'accept'}, self.window, self.state, self.payload)
        self.assertIsNone(result)
        self.assertTrue(advance)
        self.assertEqual(self.state['disclosure_reviews'], [])

    def test_new_submission_cannot_be_erased_by_stale_provisional_final(self):
        self.state['pending_consolidated_final'] = {'disclosure_reviews': []}
        reader = PendingReply()
        working = review._review_request_state(self.payload, self.window, self.state)
        raw = reader.chat_json(review.STEP, self.window.messages('Offline fixture', working))
        self.window.mark_sent()
        review._review_action(raw, self.window, self.state, self.payload)
        self.assertNotIn('pending_consolidated_final', self.state)
        self.assertTrue(self.state['disclosure_reviews'])
        self.assertTrue(all(row['status'] == 'unresolved' for row in self.state['disclosure_reviews']))

    def test_missing_work_is_bounded_and_never_auto_marked_compatible(self):
        tracer = PendingReply()
        _, _, _, transcript, raw = review._scoped_review_session(
            self.wp, self.ws, self.task, None, None, 'offline-model', tracer=tracer)
        self.assertEqual(raw['decision'], 'unresolved')
        self.assertEqual(len(raw['disclosure_reviews']), len(self.ws.disclosure['records']))
        self.assertEqual(review._parse(raw, self.payload)['status'], 'unresolved')
        self.assertLess(len(transcript), 30)


class RealTranscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('WORLD_REVIEW_COVERAGE_FIXTURE')
        if not path:
            raise unittest.SkipTest('Optional archived real GPU transcript not supplied')
        cls.fixture = json.loads(gzip.decompress(Path(path).read_bytes()))

    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, 'connect', side_effect=AssertionError('Network prohibited')).start()
        self.f = deepcopy(self.fixture)
        self.model = self.f['checkpoint']['transcript'][0]['call']['params']['model']
        patch.dict(sys.modules, {'config': SimpleNamespace(REVIEWER_MODEL=self.model)}).start()
        self.ws = WorldState.from_dict(self.f['world'])

    def test_all_31_original_requests_replay_then_first_new_request_contains_15_concrete_gaps(self):
        calls = []
        class Boundary(Exception):
            pass
        class NoModel:
            def chat_json(_, step, messages, **params):
                calls.append(json.loads(messages[-1]['content']))
                raise Boundary('Exact offline boundary before the first new model request')
        records = []
        original = self.f['checkpoint']['transcript']
        with self.assertRaises(Boundary):
            review._scoped_review_session(self.f['wp'], self.ws, self.f['task'],
                self.f['report'].get('previous'), self.f['report'].get('author_responses'),
                self.model, tracer=NoModel(), records=records, resume=original,
                transport_resume_prefix=len(original))
        self.assertEqual(records[:31], original)
        self.assertEqual(len(calls), 1)
        working = calls[0]['working_state']
        self.assertEqual(len(working['disclosure_reviews']), 110)
        self.assertEqual(len(working['coverage']['missing_disclosure_records']), 15)
        self.assertIn('Use submit', working['required_next_action'])
        self.assertNotIn('Output finish now', working['required_next_action'])
        self.assertTrue(calls[0]['exact_reads'])

    def test_real_checkpoint_migration_retains_full_prefix_and_successful_opinions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '02_world_review_original.ckpt.json'
            path.write_text(json.dumps(self.f['checkpoint'], ensure_ascii=False), encoding='utf8')
            original_bytes = path.read_bytes()
            tracer = PendingReply(folder)
            report = review._review_agentic(self.f['wp'], self.ws, tracer, self.f['task'],
                self.f['report'].get('previous'), self.f['report'].get('author_responses'))
            self.assertEqual(report['status'], 'unresolved', report.get('error'))
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(report['transcript'][:31], self.f['checkpoint']['transcript'])
            self.assertLessEqual(len(tracer.calls), 5)
            self.assertEqual(len(report['disclosure_reviews']), 125)
            old = {row['record_id']: row for row in self.f['checkpoint']['transcript'][26]['raw_output']['disclosure_reviews']}
            current = {row['record_id']: row for row in report['disclosure_reviews']}
            for identity, row in old.items():
                self.assertEqual(current[identity], row)
            self.assertEqual(review.validate_review(report, self.f['wp'], self.ws, self.f['task']),
                             [{'code': 'world_review_not_passed', 'status': 'unresolved'}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
