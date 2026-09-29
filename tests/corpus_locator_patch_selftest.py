"""No-network locator correction tests; fake replies do not prove model accuracy."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-locator-only")
from pipeline import corpus_contract as cc
from pipeline.world_state import WorldState, Timeline, Op, SET, _date_of


class FakeTracer:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result(json.loads(messages[-1]["content"])) if callable(result) else result)


def opinion(payload, *, missing=False):
    return {"document_reviews": [{"doc_index": i, "status": "supported", "reason": "离线固定意见。"}
                                  for i in range(len(payload["documents"]))],
            "verdict": "fail" if missing else "pass", "unsupported_claims": [],
            "coverage": [{"requirement_id": r["requirement_id"],
                          "status": "missing" if missing and i else "supported",
                          "reason": "必须原样保留的语义意见。",
                          "evidence": [] if missing and i else [{"doc_index": 0, "quote": "甲项目已完成。"}]}
                         for i, r in enumerate(payload["requirements"])]}


def locator(payload):
    return {"locations": [{"path": item["path"], "doc_index": 0, "quote": "甲项目进行中。"}
                          for item in payload["format_repair"]["locations_to_correct"]]}


class LocatorTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)
        self.ws = WorldState(entities={"甲项目": {"状态": Timeline([Op(0, _date_of(0), SET, "进行中")])}}, n_sessions=1)
        self.docs = [{"title": "纪要", "content": "甲项目进行中。"},
                     {"title": "归档", "content": "材料登记完毕。"}]

    def review(self, first=opinion, second=locator):
        trace = FakeTracer([first, second])
        report = cc.review_documents(trace, self.ws, 0, self.docs,
                                     requirements=cc.fidelity_requirements(self.ws, 0))
        return report, trace

    def test_localized_only_two_calls_inputs_and_semantic_opinion_frozen(self):
        report, trace = self.review()
        self.assertEqual((report["status"], len(trace.calls)), ("passed", 2))
        first, second = [json.loads(c[1][-1]["content"]) for c in trace.calls]
        repair = second.pop("format_repair")
        self.assertEqual(second, first)
        self.assertEqual(repair["mode"], "locator_patch")
        self.assertEqual(trace.calls[1][1][0]["content"], cc.LOCATOR_REPAIR_SYSTEM)
        # JSON-object providers require an explicit JSON instruction, including
        # the narrow second request whose previous opinion has no such word.
        for _, messages, parameters in trace.calls:
            if parameters.get("response_format") == {"type": "json_object"}:
                self.assertIn("json", " ".join(m["content"] for m in messages).lower())
        self.assertEqual(trace.calls[0][2], trace.calls[1][2])
        expected = deepcopy(repair["previous_response"])
        expected["coverage"][0]["evidence"][0]["quote"] = "甲项目进行中。"
        self.assertEqual(report["raw_output"], expected)
        history = report["review_validation"]
        self.assertEqual(set(history["attempts"][1]["raw_output"]), {"locations"})
        self.assertEqual(history["attempts"][1]["resolved_output"], expected)
        docs = deepcopy(self.docs); cc.attach_receipts(docs, report, 0)
        self.assertTrue(cc._review_receipt_matches(self.ws, 0, docs[0]))

    def test_semantic_fail_stays_fail_and_missing_reason_is_unchanged(self):
        self.ws.entities["甲项目"]["市场"] = Timeline([Op(0, _date_of(0), SET, "尼泊尔")])
        report, trace = self.review(lambda p: opinion(p, missing=True))
        self.assertEqual((report["status"], len(trace.calls)), ("failed", 2))
        history = report["review_validation"]
        self.assertEqual(history["attempts"][0]["raw_output"]["coverage"][1], report["raw_output"]["coverage"][1])
        self.assertEqual(report["raw_output"]["verdict"], "fail")
        with self.assertRaises(ValueError): cc.attach_receipts(deepcopy(self.docs), report, 0)

    def test_patch_rejects_unknown_duplicate_missing_and_extra_paths(self):
        def two(p):
            raw = opinion(p); raw["coverage"][0]["evidence"].append({"doc_index": 0, "quote": "甲项目已经完成。"})
            return raw
        for kind in ("unknown", "duplicate", "missing", "extra"):
            def bad(p, kind=kind):
                raw = locator(p)
                if kind == "unknown": raw["locations"][0]["path"] = "/verdict"
                elif kind == "duplicate": raw["locations"][1]["path"] = raw["locations"][0]["path"]
                elif kind == "missing": raw["locations"].pop()
                else: raw["locations"].append(deepcopy(raw["locations"][0]))
                return raw
            with self.subTest(kind=kind):
                report, trace = self.review(two, bad)
                self.assertEqual((report["status"], len(trace.calls)), ("error", 2))

    def test_patch_rejects_all_semantic_keys(self):
        for key in ("verdict", "coverage", "document_reviews", "unsupported_claims", "reason"):
            def bad(p, key=key):
                raw = locator(p); raw[key] = "pass"; return raw
            with self.subTest(key=key):
                self.assertEqual(self.review(second=bad)[0]["status"], "error")
        def row_key(p):
            raw = locator(p); raw["locations"][0]["reason"] = "改理由"; return raw
        self.assertEqual(self.review(second=row_key)[0]["status"], "error")

    def test_noncontiguous_entity_rewrite_and_title_only_quote_still_fail(self):
        for quote in ("甲项目已完成。", "甲项目，进行中。", "乙项目进行中。", "纪要", ""):
            def bad(p, quote=quote):
                raw = locator(p); raw["locations"][0]["quote"] = quote; return raw
            with self.subTest(quote=quote):
                report, trace = self.review(second=bad)
                self.assertEqual((report["status"], len(trace.calls)), ("error", 2))

    def test_wrong_or_invalid_doc_index_still_fails(self):
        for index in (1, 9, -1, "0", True):
            def bad(p, index=index):
                raw = locator(p); raw["locations"][0]["doc_index"] = index; return raw
            with self.subTest(index=index):
                self.assertEqual(self.review(second=bad)[0]["status"], "error")

    def test_locator_may_correct_actual_doc_index_only_when_full_consistency_holds(self):
        def wrong_index(p):
            raw = opinion(p); raw["coverage"][0]["evidence"][0] = {"doc_index": 1, "quote": "甲项目进行中。"}; return raw
        self.assertEqual(self.review(wrong_index)[0]["status"], "passed")

    def test_non_location_shape_issue_uses_existing_full_opinion_repair(self):
        def structurally_bad(p):
            raw = opinion(p); raw["verdict"] = "wrong"; return raw
        def full(p):
            self.assertNotIn("mode", p["format_repair"])
            raw = opinion(p); raw["coverage"][0]["evidence"][0]["quote"] = "甲项目进行中。"; return raw
        self.assertEqual(self.review(structurally_bad, full)[0]["status"], "passed")

    def test_valid_semantic_fail_and_provider_failure_do_not_retry(self):
        def semantic_fail(p):
            raw = opinion(p); raw["verdict"] = "fail"
            raw["coverage"][0].update(status="missing", evidence=[]); return raw
        for first, status in ((semantic_fail, "failed"), (TimeoutError("offline deadline"), "error")):
            report, trace = self.review(first)
            self.assertEqual((report["status"], len(trace.calls)), (status, 1))

    def test_receipt_reconstructs_patch_and_rejects_forged_assembled_opinion(self):
        report, _ = self.review(); docs = deepcopy(self.docs); cc.attach_receipts(docs, report, 0)
        for field in ("patch", "resolved", "mode"):
            with self.subTest(field=field):
                bad = deepcopy(docs[0]); receipt = bad["quality_review"]; h = receipt["review_validation"]
                if field == "patch": h["attempts"][1]["raw_output"]["locations"][0]["path"] = "/verdict"
                elif field == "resolved": h["attempts"][1]["resolved_output"]["coverage"][0]["reason"] = "tampered"
                else: del h["attempts"][1]["resolved_output"]
                h["final_response_hash"] = cc.fingerprint(h["attempts"][1]["raw_output"])
                h["final_opinion_hash"] = cc.fingerprint(cc._saved_review_output(h["attempts"][1]))
                receipt["review_validation_hash"] = cc.fingerprint(h)
                self.assertFalse(cc._review_receipt_matches(self.ws, 0, bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
