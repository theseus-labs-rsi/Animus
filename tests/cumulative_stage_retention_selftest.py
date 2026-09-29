"""Fault checks for the frozen cumulative stage inventory."""
import unittest
from unittest.mock import patch

from tools import audit_cumulative_stage_retention as retention


class CumulativeStageRetentionTest(unittest.TestCase):
    def test_seven_line_denominator_rejects_one_sided_drift(self):
        lines = ("L1_timeline", "L2_relational", "L3_process", "L5_conflict",
                 "L6_refusal", "L7_consolidation", "L8_transition")
        raw = [{"line": line, "qid": f"{line}-{index}"}
               for line in lines for index in range(6)]
        self.assertEqual(retention._orders(raw, list(raw))["effective"], 42)
        altered = list(raw)
        altered[0] = {"line": "L8_transition", "qid": altered[0]["qid"]}
        with self.assertRaisesRegex(ValueError, "denominator"):
            retention._orders(raw, altered)

    def test_administrative_success_cannot_erase_semantic_warning(self):
        sessions = [{"session_id": number, "docs": [{}] * (51 if number != 4 else 54)}
                    for number in (0, 2, 3, 4)]
        candidate = {"corpus": {"sessions": sessions}, "done_weeks": [0, 2, 3, 4]}
        checkpoint = {"corpus": {"sessions": sessions}, "done_weeks": [0, 2, 3, 4]}
        terminal = {"05_corpus_candidate.json": {"session_ids": [0, 2, 3, 4],
                    "documents": 207}, "05_corpus_warning.json": {"sha256": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "failed candidate"):
            retention._corpus(candidate, checkpoint,
                {"status": "warning", "release_eligible": True},
                {"status": "incomplete", "attempt_complete": False}, terminal)

    def test_spent_slot_cannot_become_unadmitted(self):
        key = "a" * 64
        prepared = {"public_plan_validated_without_changes": True,
            "groups": [{"legacy_key": key, "original_drafts": [
                {"position": 0, "saved_author_reply": True}],
                "prepared_slot_verified": True, "legacy_complete": False}]}
        prior = {"version": "legacy-signal-inheritance-audit/v1",
                 "provider_calls": 0,
                 "saved_reply_complete_messages_match_real_requests": True}
        old = {"key": key, "progress": {"drafts": [
            {"admitted": False, "author_output": "old"}]}}
        with patch.object(retention, "_bound_file", return_value=old):
            with self.assertRaisesRegex(ValueError, "position"):
                retention._slots(None, {}, prepared, prior)


if __name__ == "__main__":
    unittest.main()
