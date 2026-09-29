"""Disclosure recovery must not depend on a future question-stage artifact."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from original_world_review_gate_selftest import GateTests, candidate
from pipeline import disclosure, factory, order_warning, world_semantics


class DisclosureBeforeQuestionsTests(unittest.TestCase):
    def setUp(self):
        GateTests.setUp(self)
        wp = self.run.read("01_whitepaper.json")
        wp["quality_contract"]["public_disclosure"] = True
        self.run.write("01_whitepaper.json", wp)
        ws = candidate()
        ws.disclosure = {"plan_hash": "original-public-plan"}
        self.run.write("02_world.json", ws.to_dict())
        self.run.write(world_semantics.REVIEW_ARTIFACT, {"status": "passed"})
        proposal = {"status": "error", "items": []}
        warning = {"version": "order-generation-warning/v1", "status": "warning",
                   "release_eligible": False,
                   "binding": {"whitepaper_hash": order_warning.canonical_hash(wp),
                               "world_hash": order_warning.canonical_hash(ws.to_dict()),
                               "proposal_hash": order_warning.canonical_hash(proposal)}}
        self.run.write(order_warning.PROPOSAL_ARTIFACT, proposal)
        self.run.write(order_warning.WARNING_ARTIFACT, warning)
        self.original_warning = (self.run.dir / order_warning.WARNING_ARTIFACT).read_bytes()
        self.original_proposal = (self.run.dir / order_warning.PROPOSAL_ARTIFACT).read_bytes()

    def recover(self):
        with patch.object(disclosure, "validate_plan", return_value=[]), \
                patch.object(world_semantics, "validate_review", return_value=[]), \
                patch.object(world_semantics, "review_world", side_effect=AssertionError("No repeat review")):
            factory.stage_disclosure(self.run)
        self.assertTrue(self.run.has("02_disclosure_ready.json"))
        self.assertEqual((self.run.dir / order_warning.WARNING_ARTIFACT).read_bytes(), self.original_warning)
        self.assertEqual((self.run.dir / order_warning.PROPOSAL_ARTIFACT).read_bytes(), self.original_proposal)

    def test_before_questions_keeps_warning_and_removes_stale_scope(self):
        self.run.write(order_warning.RESOLUTION_ARTIFACT, {"stale": True})
        self.recover()
        self.assertFalse(self.run.has("04_questions.json"))
        self.assertFalse(self.run.has(order_warning.RESOLUTION_ARTIFACT))

    def test_existing_questions_keep_original_scope_semantics(self):
        questions = [{"qid": "q_l3", "line": "L3_process"},
                     {"qid": "q_l7", "line": "L7_consolidation"}]
        self.run.write("04_questions.json", questions)
        self.recover()
        expected = order_warning.build_resolution(
            self.run.read("01_whitepaper.json"), self.run.read("02_world.json"), questions,
            self.run.read(order_warning.WARNING_ARTIFACT), self.run.read(order_warning.PROPOSAL_ARTIFACT))
        self.assertEqual(self.run.read(order_warning.RESOLUTION_ARTIFACT), expected)
        self.assertEqual(expected["excluded_qids"], ["q_l3"])
        self.assertEqual(self.run.read("04_questions.json"), questions)

    def test_stale_warning_does_not_publish_a_scope(self):
        warning = deepcopy(self.run.read(order_warning.WARNING_ARTIFACT))
        warning["binding"]["world_hash"] = "stale"
        self.run.write(order_warning.WARNING_ARTIFACT, warning)
        self.original_warning = (self.run.dir / order_warning.WARNING_ARTIFACT).read_bytes()
        self.run.write("04_questions.json", [{"qid": "q", "line": "L3_process"}])
        self.run.write(order_warning.RESOLUTION_ARTIFACT, {"stale": True})
        self.recover()
        self.assertFalse(self.run.has(order_warning.RESOLUTION_ARTIFACT))

    def test_incomplete_questions_do_not_publish_a_scope(self):
        self.run.write("04_questions.json", [])
        self.recover()
        self.assertFalse(self.run.has(order_warning.RESOLUTION_ARTIFACT))


if __name__ == "__main__":
    unittest.main()
