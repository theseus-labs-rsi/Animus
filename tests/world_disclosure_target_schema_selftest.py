"""Offline output-contract regression. Model business judgment is not tested."""
from copy import deepcopy
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from pipeline import world_semantics as review
from world_disclosure_record_review_selftest import planned, opinion
from world_semantics_selftest import fixture, positive, FakeTracer

class DisclosureTargetSchemaTests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')),
                      patch.dict(sys.modules, {'config': SimpleNamespace(REVIEWER_MODEL='offline-mini')})):
            guard.start(); self.addCleanup(guard.stop)
        self.wp,self.ws,self.task=planned()
        self.raw=opinion(self.wp,self.ws,self.task)

    def call(self,raw):
        tracer=FakeTracer(deepcopy(raw))
        result=review.review_world(self.wp,self.ws,tracer,self.task)
        self.assertEqual(len(tracer.calls),1)
        self.assertEqual(result['raw_output'],raw)
        return result,tracer

    def test_actual_enabled_request_has_one_complete_explicit_route_template(self):
        self.raw['repair_targets']['disclosure']=False
        result,tracer=self.call(self.raw)
        self.assertEqual(result['status'],'passed')
        system=tracer.calls[0]['messages'][0]['content']
        marker='{\n  "mechanism_coverage":'
        self.assertEqual(system.count(marker),1)
        template=json.loads(system[system.index(marker):])
        self.assertEqual(set(template['repair_targets']),{'intrinsic','structure','disclosure'})
        self.assertIs(template['repair_targets']['disclosure'],False)
        self.assertEqual(set(template),set(self.raw))
        self.assertFalse(review.validate_review(result,self.wp,self.ws,self.task))

    def test_missing_route_is_still_error_and_explicit_route_uses_original_repair(self):
        self.raw['decision']='repair'
        self.raw['disclosure_reviews'][0].update(status='repair',reason='The original record requires a revised disclosure arrangement.')
        self.raw['repair_targets']={'intrinsic':[],'structure':False}
        invalid,_=self.call(self.raw)
        self.assertEqual(invalid['status'],'error')
        self.assertNotIn('disclosure',invalid['raw_output']['repair_targets'])
        corrected=deepcopy(self.raw); corrected['repair_targets']['disclosure']=True
        valid,_=self.call(corrected)
        self.assertEqual(valid['status'],'failed')
        self.assertEqual(valid['repair_targets'],corrected['repair_targets'])
        self.assertEqual(valid['disclosure_reviews'],self.raw['disclosure_reviews'])

    def test_legacy_six_keys_and_allowed_routes_are_unchanged(self):
        system=review._system(False)
        self.assertEqual(system,review.SYSTEM)
        template=json.loads('{"mechanism_coverage":'+system.split('{"mechanism_coverage":',1)[1])
        self.assertEqual(len(template),6)
        self.assertEqual(set(template['repair_targets']),{'intrinsic','structure'})
        wp,ws,task=fixture(False)
        self.assertEqual(review.review_world(wp,ws,FakeTracer(positive(False)),task)['status'],'passed')
        raw=positive(False);raw['repair_targets']['disclosure']=True
        self.assertEqual(review.review_world(wp,ws,FakeTracer(raw),task)['status'],'error')

if __name__=='__main__': unittest.main(verbosity=2)
