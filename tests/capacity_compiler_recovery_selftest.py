"""Bounded compiler feedback; no semantic refusal or budget retry."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from supply_driven_production_selftest import ProductionTransactionTests
from pipeline import production, factory, supply_capacity
from pipeline.world_blueprint import normalize_world_blueprint, WorldBlueprintError


class RecoveryTests(ProductionTransactionTests):
    def test_rejected_counts_return_to_architect_and_are_preserved(self):
        original = deepcopy(self.run.read('01_whitepaper.json'))
        calls = []
        def propose(wp, tracer, feedback, audit, *, checkpoint=None):
            calls.append(deepcopy(feedback))
            self.assertEqual(self.run.read('01_whitepaper.json'), original)
            if len(calls) == 1:
                audit['proposal'] = {'decision': 'revise', 'relation_min_counts': {'r': 4}}
                checkpoint(audit)
                raise WorldBlueprintError('world_blueprint 非法:\n- relation r min_count=4 超过标量 FK 最大容量 3')
            self.assertIn('compiler_error', feedback)
            self.assertEqual(len(self.run.read('11_revision_001.json')['compiler_recovery_attempts']),1)
            return self.revised(wp, tracer, feedback, audit, checkpoint=checkpoint)
        with patch.object(production, 'revise_capacity', side_effect=propose):
            production.prepare_revision(self.run, self.state, factory.STAGES,
                original['supply_plan']['candidate_allocation'], {'L7': 1})
        self.assertEqual(len(calls), 2)
        saved = self.run.read('11_revision_001.json')
        self.assertEqual(len(saved['compiler_recovery_attempts']), 1)
        self.assertEqual(self.state['round'], 2)
        self.assertEqual(self.run.read('experiment_profile.json')['admitted_calls'], 17)

    def test_mechanical_feedback_stops_after_three(self):
        def invalid(wp, tracer, feedback, audit, *, checkpoint=None):
            audit['proposal'] = {'decision': 'revise'}
            raise WorldBlueprintError('world_blueprint 非法:\nFK capacity')
        with patch.object(production, 'revise_capacity', side_effect=invalid) as call:
            with self.assertRaises(WorldBlueprintError):
                production.prepare_revision(self.run, self.state, factory.STAGES,
                    self.run.read('01_whitepaper.json')['supply_plan']['candidate_allocation'], {})
        self.assertEqual(call.call_count, 3)

    def test_semantic_refusal_is_not_retried(self):
        with patch.object(production, 'revise_capacity', side_effect=WorldBlueprintError('Capacity revision business feasibility unresolved')) as call:
            with self.assertRaises(WorldBlueprintError):
                production.prepare_revision(self.run, self.state, factory.STAGES,
                    self.run.read('01_whitepaper.json')['supply_plan']['candidate_allocation'], {})
        self.assertEqual(call.call_count, 1)

    def test_real_failed_proposal_remains_rejected(self):
        path = Path(__file__).resolve().parents[1] / 'output/supply_driven_fast_delta_recovery_20260926/capacity_failure_fixture.json'
        if not path.exists(): self.skipTest('GPU fixture stored separately')
        saved = json.loads(path.read_text())
        wp = deepcopy(saved['whitepaper']); raw = saved['audit']['proposal']
        for group, edits, field in [('entity_types','entity_counts','count'),('event_types','event_min_counts','min_count'),('relation_types','relation_min_counts','min_count')]:
            for row in wp['world_blueprint'][group]:row[field] = raw[edits].get(row['id'], row.get(field,0))
        with self.assertRaisesRegex(WorldBlueprintError, 'artifact_device'):
            normalize_world_blueprint(wp)


if __name__ == '__main__':
    unittest.main()
