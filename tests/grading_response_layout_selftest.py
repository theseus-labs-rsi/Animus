"""Wire-layout compatibility only; no model semantic accuracy claims."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.grading_response_layout import normalize


class LayoutTests(unittest.TestCase):
    def test_canonical_is_unchanged(self):
        raw = {"requires_item_reassessment": False, "reassessment_reason": "",
               "answer_review": {"claims": ["Original free text"]}}
        value, audit = normalize(raw)
        self.assertEqual(value, raw)
        self.assertEqual(audit["actions"], [])
        self.assertIsNot(value, raw)

    def test_nested_copy_is_lossless_and_does_not_infer_from_verdict(self):
        raw = {"answer_verdict": "correct", "answer_review": {
            "requires_item_reassessment": True, "reassessment_reason": "Reference issue.",
            "claims": [{"claim": "Keep exact text", "evidence_indices": [2, 0]}]}}
        before = deepcopy(raw)
        value, audit = normalize(raw)
        self.assertIs(value["requires_item_reassessment"], True)
        for action in audit["actions"]:
            del value[action["to"][1:]]
        self.assertEqual(value, raw)
        self.assertEqual(raw, before)

    def test_identical_complete_pair_at_both_locations(self):
        pair = {"requires_item_reassessment": False, "reassessment_reason": "Explicit decision."}
        raw = {**pair, "answer_review": deepcopy(pair)}
        self.assertEqual(normalize(raw)[0], raw)

    def test_ambiguous_missing_or_wrong_types_are_never_healed(self):
        cases = [
            {},
            {"requires_item_reassessment": False},
            {"reassessment_reason": "No issue", "answer_review": {"requires_item_reassessment": False}},
            {"answer_review": {"requires_item_reassessment": "false", "reassessment_reason": "No issue"}},
            {"requires_item_reassessment": 0, "reassessment_reason": "No issue"},
            {"requires_item_reassessment": False, "reassessment_reason": None},
            {"answer_review": {"requires_item_reassessment": True, "reassessment_reason": " "}},
            {"requires_item_reassessment": False, "reassessment_reason": "No issue",
             "answer_review": {"requires_item_reassessment": True, "reassessment_reason": "Issue"}},
            {"requires_item_reassessment": False, "reassessment_reason": "No issue",
             "answer_review": {"requires_item_reassessment": False, "reassessment_reason": "Different reason"}},
            {"requires_item_reassessment": False, "reassessment_reason": "No issue",
             "answer_review": {"reassessment_reason": "Partial nested pair"}},
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
