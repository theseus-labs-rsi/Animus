"""Original world-stage publication and bounded repair wiring, without API calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import original_world_review_gate_selftest as fixtures
from original_world_review_gate_selftest import candidate
from pipeline import factory, disclosure, world_semantics
from pipeline.world_blueprint import WorldBlueprintError


class DisclosureGateTests(unittest.TestCase):
    builder = fixtures.GateTests.builder

    def setUp(self):
        fixtures.GateTests.setUp(self)
        wp = self.run.read('01_whitepaper.json')
        wp['quality_contract']['public_disclosure'] = True
        self.run.write('01_whitepaper.json', wp)

    def test_plan_is_written_after_preparation_before_original_review(self):
        order = []
        def prepare(wp, ws, log):
            ws.conflicts = [{'prepared': True}]
            order.append('prepare')
        def plan(wp, ws, tracer, **kwargs):
            self.assertEqual(ws.conflicts, [{'prepared': True}])
            ws.disclosure = {'test_marker': 'planned'}
            order.append('plan')
            return {'status': 'ready'}
        def review(wp, ws, tracer, **kwargs):
            self.assertEqual(ws.disclosure, {'test_marker': 'planned'})
            order.append('review')
            return {'status': 'passed'}
        with patch.object(factory, 'build_world', side_effect=self.builder), \
             patch.object(factory, '_prepare_lines', side_effect=prepare), \
             patch.object(disclosure, 'author_plan', side_effect=plan), \
             patch.object(disclosure, 'validate_plan', return_value=[]), \
             patch.object(world_semantics, 'review_world', side_effect=review), \
             patch.object(world_semantics, 'validate_review', return_value=[]):
            factory.stage_world(self.run)
        self.assertEqual(order, ['prepare', 'plan', 'review'])
        self.assertEqual(self.run.read('02_world.json')['disclosure']['test_marker'], 'planned')
        self.assertEqual(len(self.run.read('02_disclosure_plan_attempts.json')['attempts']), 1)


    def test_original_repair_replans_the_changed_world_and_reviews_it_again(self):
        values = []
        failure = {'status': 'failed', 'repair_targets': {'intrinsic': [], 'structure': True}}
        def plan(wp, ws, tracer, **kwargs):
            values.append(ws.entities['Project']['status'].latest_valid())
            self.assertEqual(kwargs['feedback'], failure if len(values) == 2 else None)
            return {'status': 'ready'}
        with patch.object(factory, 'build_world', side_effect=self.builder), \
             patch.object(factory, '_prepare_lines'), \
             patch.object(disclosure, 'author_plan', side_effect=plan), \
             patch.object(disclosure, 'validate_plan', return_value=[]), \
             patch.object(world_semantics, 'review_world', side_effect=[failure, {'status': 'passed'}]), \
             patch.object(world_semantics, 'validate_review', return_value=[]):
            factory.stage_world(self.run)
        self.assertEqual(values, ['pending', 'reviewed'])
        self.assertEqual(len(self.run.read('02_disclosure_plan_attempts.json')['attempts']), 2)

    def test_disclosure_cannot_run_without_original_semantic_review(self):
        self.run.manifest['config'] = {}
        self.run.write('01_whitepaper.json', {'quality_contract': {'public_disclosure': True}})
        with patch.object(factory, 'build_world') as build:
            with self.assertRaises(WorldBlueprintError):
                factory.stage_world(self.run)
        build.assert_not_called()

    def test_disclosure_only_repair_keeps_truth_and_reuses_original_review_loop(self):
        failure = {'status': 'failed', 'repair_targets': {
            'intrinsic': [], 'structure': False, 'disclosure': True}}
        with patch.object(factory, 'build_world', side_effect=self.builder) as build, \
             patch.object(factory, '_prepare_lines') as prepare, \
             patch.object(disclosure, 'author_plan', return_value={'status': 'ready'}) as plan, \
             patch.object(disclosure, 'validate_plan', return_value=[]), \
             patch.object(world_semantics, 'review_world', side_effect=[failure, {'status': 'passed'}]) as review, \
             patch.object(world_semantics, 'validate_review', return_value=[]):
            factory.stage_world(self.run)
        self.assertEqual((build.call_count, prepare.call_count, plan.call_count, review.call_count), (1, 1, 2, 2))
        self.assertEqual(plan.call_args.kwargs['feedback'], failure)
        self.assertEqual(self.run.read('02_world.json')['entities']['Project']['status'][0]['value'], 'pending')

    def test_resume_cannot_disable_required_review(self):
        self.run.manifest['config'] = {}
        wp = {'quality_contract': {'public_disclosure': True}}
        with self.assertRaises(WorldBlueprintError):
            factory._require_current_world_review(self.run, wp)


if __name__ == '__main__':
    unittest.main()
