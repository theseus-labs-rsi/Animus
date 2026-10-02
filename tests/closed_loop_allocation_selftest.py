"""Offline entity allocation regression for lower bounds and a fixed total cap.

The synthetic five-type fixture preserves the failed allocation shape without
loading experiment artifacts or requesting model output.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from pipeline import closed_loop, factory, run as runs
from pipeline.targetspec import TargetSpec
from pipeline.world_blueprint import WorldBlueprintError, relation_capacity


def fixture():
    counts = {'merchant': 3, 'review_ticket': 3, 'evidence_record': 3,
              'tip_source': 2, 'listing_entry': 3}
    return {
        'active_lines': [{'line': 'L1_timeline', 'weight': 1}],
        'domain_profile': {'field_schema': []},
        'shared_world_spec': {'entities': {'count': 14}, 'timeline': {'n_sessions': 4}},
        'seed_contract': {'blueprint_requirements': {'entity_types': [
            {'id': name, 'count': 3} for name in ('merchant', 'review_ticket', 'evidence_record')]}},
        'world_blueprint': {
            'entity_types': [{'id': name, 'count': count, 'primary': name == 'review_ticket',
                              'fields': []} for name, count in counts.items()],
            'relation_types': [], 'event_types': [], 'temporal_model': {'n_sessions': 4}}}


def populations(wp):
    return {row['id']: row['count'] for row in wp['world_blueprint']['entity_types']}


class ClosedLoopAllocationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')))
        self.stack.enter_context(patch.object(config, 'chat', side_effect=AssertionError('Provider forbidden')))
        self.stack.enter_context(patch.object(config, 'chat_json', side_effect=AssertionError('Provider forbidden')))

    def test_twelve_slots_fit_seed_floors_without_dropping_any_type(self):
        wp = fixture(); seed = deepcopy(wp['seed_contract'])
        closed_loop._scale_world_contract(wp, 12, 6)
        counts = populations(wp)
        self.assertEqual(sum(counts.values()), 12)
        self.assertEqual(set(counts), {'merchant', 'review_ticket', 'evidence_record', 'tip_source', 'listing_entry'})
        self.assertTrue(all(counts[name] >= 3 for name in ('merchant', 'review_ticket', 'evidence_record')))
        self.assertTrue(all(count >= 1 for count in counts.values()))
        self.assertEqual(wp['shared_world_spec']['entities']['count'], 12)
        self.assertEqual(wp['seed_contract'], seed)

    def test_infeasible_total_retains_minima_and_explicit_overshoot(self):
        wp = fixture()
        closed_loop._scale_world_contract(wp, 10, 6)
        self.assertEqual(sum(populations(wp).values()), 11)
        self.assertEqual(wp['shared_world_spec']['entities']['count'], 11)
        self.assertEqual([populations(wp)[name] for name in ('merchant', 'review_ticket', 'evidence_record')], [3, 3, 3])

    def test_exact_population_is_not_a_donor(self):
        wp = fixture()
        wp['world_blueprint']['entity_types'][3]['cardinality_policy'] = 'exact'
        closed_loop._scale_world_contract(wp, 12, 6)
        self.assertEqual(populations(wp), {'merchant': 3, 'review_ticket': 3, 'evidence_record': 3,
                                           'tip_source': 2, 'listing_entry': 1})
        self.assertEqual(wp['shared_world_spec']['entities']['count'], 12)

    def test_event_role_population_is_not_a_donor(self):
        wp = fixture()
        wp['world_blueprint']['event_types'] = [{'id': 'source_handoff',
            'roles': {'sender': 'tip_source', 'recipient': 'tip_source'}, 'min_count': 1}]
        closed_loop._scale_world_contract(wp, 12, 6)
        self.assertEqual(populations(wp), {'merchant': 3, 'review_ticket': 3, 'evidence_record': 3,
                                           'tip_source': 2, 'listing_entry': 1})
        self.assertEqual(closed_loop._ensure_event_role_capacity(wp), {})

    def test_relations_and_events_use_post_recovery_population_capacity(self):
        wp = fixture(); bp = wp['world_blueprint']
        bp['entity_types'][3]['cardinality_policy'] = 'exact'
        bp['entity_types'][4]['fields'] = [{'name': 'owner', 'kind': 'reference'}]
        bp['relation_types'] = [{'id': 'listing_owner', 'from_type': 'listing_entry',
            'to_type': 'merchant', 'field': 'owner', 'min_count': 3, 'temporal': False}]
        bp['event_types'] = [{'id': 'listing_update', 'roles': {'entry': 'listing_entry'}, 'min_count': 9}]
        closed_loop._scale_world_contract(wp, 12, 2)
        self.assertEqual(populations(wp)['listing_entry'], 1)
        self.assertEqual(bp['relation_types'][0]['min_count'], 1)
        self.assertEqual(bp['relation_types'][0]['min_count'], relation_capacity(bp, bp['relation_types'][0]))
        self.assertEqual(bp['event_types'][0]['min_count'], 2)
        self.assertEqual(wp['shared_world_spec']['timeline']['n_sessions'], 2)

    def test_repeated_scaling_uses_original_population_baseline(self):
        wp = fixture()
        closed_loop._scale_world_contract(wp, 12, 6)
        first = deepcopy(wp)
        closed_loop._scale_world_contract(wp, 12, 6)
        self.assertEqual(wp, first)
        closed_loop._scale_world_contract(wp, 20, 6)
        self.assertEqual(sum(populations(wp).values()), 20)
        closed_loop._scale_world_contract(wp, 12, 6)
        self.assertEqual(wp, first)

    def test_driver_reports_actual_total_and_cap_before_world_dispatch(self):
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(runs, 'RUNS_DIR', directory))
        run = runs.Run('office', 'allocation_cap', config_meta={
            'generation_contract': getattr(factory, 'GENERATION_CONTRACT', 'legacy')})
        run.log = lambda *_: None
        run.write('01_whitepaper.json', fixture())
        author = self.stack.enter_context(patch.object(factory, 'stage_world', side_effect=AssertionError('No model work for infeasible cap')))
        with self.assertRaises(WorldBlueprintError) as error:
            closed_loop.build_to_target(run, TargetSpec(min_questions=1, total_only=True, max_world_entities=10), max_rounds=1)
        message = str(error.exception)
        self.assertIn('11', message)
        self.assertIn('10', message)
        self.assertNotIn('Seed entity minima exceed', message)
        author.assert_not_called()
        self.assertEqual(run.tracer.n, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
