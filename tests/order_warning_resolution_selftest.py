import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import order_warning


class OrderWarningResolutionTests(unittest.TestCase):
    def inputs(self, lines=("L1_timeline", "L3_process", "L2_relational")):
        wp = {"quality_contract": {}}
        world = {"entities": [], "events": []}
        proposal = {"status": "error", "items": []}
        questions = [{"qid": f"q{i}", "line": line, "capability": "x"}
                     for i, line in enumerate(lines)]
        warning = {"version": "order-generation-warning/v1", "status": "warning",
                   "release_eligible": False,
                   "binding": {"whitepaper_hash": order_warning.canonical_hash(wp),
                               "world_hash": order_warning.canonical_hash(world),
                               "proposal_hash": order_warning.canonical_hash(proposal)}}
        return wp, world, questions, warning, proposal

    def test_only_l3_questions_are_scoped(self):
        args = self.inputs()
        receipt = order_warning.build_resolution(*args)
        self.assertEqual(receipt["excluded_qids"], ["q1"])
        self.assertEqual(order_warning.validate_resolution(receipt, *args), [])

    def test_world_without_l3_keeps_all_questions(self):
        args = self.inputs(("L1_timeline", "L8_transition"))
        receipt = order_warning.build_resolution(*args)
        self.assertEqual(receipt["excluded_qids"], [])

    def test_changed_question_or_proposal_invalidates_receipt(self):
        wp, world, questions, warning, proposal = self.inputs()
        receipt = order_warning.build_resolution(wp, world, questions, warning, proposal)
        changed = json.loads(json.dumps(questions)); changed[0]["qid"] = "changed"
        self.assertTrue(order_warning.validate_resolution(
            receipt, wp, world, changed, warning, proposal))
        changed_proposal = {**proposal, "status": "completed"}
        self.assertTrue(order_warning.validate_resolution(
            receipt, wp, world, questions, warning, changed_proposal))

    def test_selected_l3_is_withheld_and_counts_recomputed(self):
        args = self.inputs()
        receipt = order_warning.build_resolution(*args)
        report = {"overall": {"n": 3, "grounded": 3, "survival": 1.0},
                  "by_line": {line: {"n": 1, "grounded": 1, "survival": 1.0}
                              for line in ("L1_timeline", "L3_process", "L2_relational")},
                  "by_capability": {"x": {"n": 3, "grounded": 3, "survival": 1.0}},
                  "limitations": []}
        kept, updated = order_warning.apply_resolution(args[2], report, receipt)
        self.assertEqual([q["qid"] for q in kept], ["q0", "q2"])
        self.assertEqual(updated["overall"]["grounded"], 2)
        self.assertEqual(updated["by_line"]["L3_process"]["grounded"], 0)
        self.assertEqual(updated["n_scoped_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
