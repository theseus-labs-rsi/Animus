"""Offline checks for the final lightweight stage-07 summary."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.quality import evaluate_release, quality_snapshot


def question(qid, line="L1_timeline"):
    return {
        "qid": qid,
        "question": f"question {qid}",
        "gt": {"value": qid},
        "line": line,
        "capability": "IE",
        "entity": "record",
        "field": "status",
    }


class ReleaseSummaryTests(unittest.TestCase):
    def make_run(self, *, released=("q1",), rejected=("q2",), pending=("q3",), scoped=()):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        all_qids = [*released, *rejected, *pending, *scoped]
        source = [question(qid, "L2_relational" if qid == "q2" else "L1_timeline")
                  for qid in all_qids]
        final = [row for row in source if row["qid"] in released]
        values = {
            "00_about.json": {"protocol": "fixture"},
            "02_world.json": {"entities": {}},
            "04_questions.json": source,
            "05_corpus.json": {"corpus": {"sessions": []}},
            "06_grounded_questions.json": final,
            "06_grounding_report.json": {
                "drops": [{"qid": qid} for qid in rejected],
                "pending": [{"qid": qid} for qid in pending],
                "order_warning_resolution": {"selected_withheld": list(scoped)},
            },
            "manifest.json": {"algo": {"targetspec": {
                "min_questions": 4, "per_line_min": {"L1_timeline": 3}}}},
        }
        for name, value in values.items():
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        return root

    def test_mixed_question_outcomes_release_the_passed_subset(self):
        root = self.make_run()
        result = evaluate_release(root)
        self.assertTrue(result["eligible"], result)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checks"]["partition"]["counts"], {
            "released": 1, "rejected": 1, "pending_review": 1, "scoped_excluded": 0})
        self.assertEqual({row["code"] for row in result["warnings"]}, {
            "delivery_target_unmet", "questions_rejected", "questions_pending_review"})

    def test_unresolved_upstream_review_keeps_partition_but_closes_release(self):
        root = self.make_run()
        (root / "02_world_review_warning.json").write_text(
            json.dumps({"release_eligible": False}), encoding="utf-8")
        result = evaluate_release(root)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["checks"]["partition"]["counts"]["released"], 1)
        self.assertIn({"code": "unresolved_generation_review",
                       "artifact": "02_world_review_warning.json"}, result["issues"])

    def test_scoped_exclusion_is_a_fourth_disjoint_status(self):
        root = self.make_run(scoped=("q4",))
        result = evaluate_release(root)
        self.assertTrue(result["eligible"], result)
        self.assertEqual(result["checks"]["partition"]["qids"]["scoped_excluded"], ["q4"])

    def test_overlap_or_missing_identity_refuses_the_summary(self):
        root = self.make_run()
        report = json.loads((root / "06_grounding_report.json").read_text(encoding="utf-8"))
        report["pending"].append({"qid": "q1"})
        (root / "06_grounding_report.json").write_text(json.dumps(report), encoding="utf-8")
        result = evaluate_release(root)
        self.assertFalse(result["eligible"])
        self.assertIn("question_status_overlap", {row["code"] for row in result["issues"]})

    def test_empty_released_subset_is_not_usable(self):
        root = self.make_run(released=(), rejected=("q1",), pending=())
        result = evaluate_release(root)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["eligible"])

    def test_receipt_becomes_stale_when_a_bound_artifact_changes(self):
        root = self.make_run()
        receipt = evaluate_release(root)
        (root / "07_release.json").write_text(json.dumps(receipt), encoding="utf-8")
        self.assertTrue(quality_snapshot(root)["eligible"])
        (root / "05_corpus.json").write_text('{"corpus":{"sessions":[{"session_id":0}]}}', encoding="utf-8")
        state = quality_snapshot(root)
        self.assertEqual(state["status"], "stale")
        self.assertFalse(state["eligible"])


if __name__ == "__main__":
    unittest.main()
