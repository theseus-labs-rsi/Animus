"""Exact transport round trips and original review receipts; no model quality claim."""
from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import patch

import config
from pipeline import corpus_contract as cc, evidence_transport as wire
from corpus_review_format_repair_selftest import world, valid


class SharedTracer:
    def __init__(self, replies):
        self.replies, self.calls, self.logical_inputs = list(replies), [], []

    def chat_json(self, step, messages, **parameters):
        self.calls.append((step, deepcopy(messages), deepcopy(parameters)))
        payload = json.loads(messages[-1]["content"])
        if payload.get("encoding") == wire.SHARED:
            payload = wire.unpack(payload)
        self.logical_inputs.append(deepcopy(payload))
        if not self.replies:
            raise AssertionError("No additional reviewer attempt is allowed")
        reply = self.replies.pop(0)
        return deepcopy(reply(payload) if callable(reply) else reply)


class EvidenceTransportTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Provider/network forbidden"))
            guard.start(); self.addCleanup(guard.stop)

    def test_complete_nested_values_and_marker_objects_roundtrip_without_aliasing(self):
        fact = {"date": "2025-01-06", "value": "A complete value " * 20,
                "history": [{"session": 0, "value": "old"}, {"session": 2, "value": "new"}]}
        payload = {"CANON": {"facts": [fact, deepcopy(fact)], "future": {"date": "2025-01-20"}},
            "requirements": [{"target": deepcopy(fact)}],
            "markers": [{"$evidence": "e0"}, {"$literal": {"$evidence": "e1"}},
                        {"$literal": 0}], "scalars": [0, False, None, "", 5.25]}
        original = deepcopy(payload)
        encoded = wire.pack(payload)
        restored = wire.unpack(encoded)
        self.assertEqual(restored, original)
        self.assertEqual(json.dumps(restored, ensure_ascii=False), json.dumps(original, ensure_ascii=False))
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(payload)))
        restored["CANON"]["facts"][0]["value"] = "changed"
        self.assertEqual(restored["CANON"]["facts"][1], fact)
        self.assertEqual(payload, original)

    def test_bad_references_unused_values_and_unknown_versions_are_rejected(self):
        good = wire.pack({"facts": ["long value " * 20] * 2})
        missing = deepcopy(good)
        missing["values"].clear()
        unused = deepcopy(good)
        unused["values"]["hidden"] = "evidence without a payload reference"
        version = deepcopy(good)
        version["encoding"] = "unknown/v1"
        for value in (missing, unused, version, {"encoding": wire.SHARED}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wire.unpack(value)

    def test_inline_request_preserves_original_messages_and_shared_request_binds_both_views(self):
        payload = {"facts": ["long value " * 20] * 2}
        params = {"temperature": 0.0, "max_tokens": 8192}
        inline, _binding = wire.request("Original rules", payload, step="review", parameters=params)
        self.assertEqual(inline, [{"role": "system", "content": "Original rules"},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        shared, binding = wire.request("Original rules", payload, step="review",
                                       parameters=params, protocol=wire.SHARED)
        self.assertEqual(wire.unpack(json.loads(shared[-1]["content"])), payload)
        self.assertEqual(binding["wire_messages_hash"], wire.fingerprint(shared))
        self.assertEqual(binding["logical_input_hash"], wire.fingerprint(payload))
        self.assertTrue(shared[0]["content"].startswith("Original rules"))

    def review(self, protocol=wire.INLINE, reply=valid):
        ws = world()
        docs = [{"title": "Original document", "content": "甲项目目前进行中。"}]
        tracer = SharedTracer([reply])
        report = cc.review_documents(tracer, ws, 0, docs, requirements=cc.fidelity_requirements(ws, 0),
                                     evidence_transport=protocol)
        return ws, docs, tracer, report

    def test_shared_review_preserves_scope_and_uses_original_receipt_validation(self):
        ws, docs, tracer, report = self.review(wire.SHARED)
        _ws, _docs, inline, inline_report = self.review()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(tracer.logical_inputs, inline.logical_inputs)
        for key in ("raw_output", "context_hash", "requirements", "documents_hash"):
            self.assertEqual(report[key], inline_report[key])
        history = report["review_validation"]
        self.assertEqual(history["transport"], wire.SHARED)
        self.assertEqual(history["attempts"][0]["transport_binding"]["wire_messages_hash"],
                         wire.fingerprint(tracer.calls[0][1]))
        self.assertNotIn("transport", inline_report["review_validation"])
        self.assertNotIn("transport_binding", inline_report["review_validation"]["attempts"][0])
        cc.attach_receipts(docs, report, 0)
        self.assertTrue(cc._review_receipt_matches(ws, 0, docs[0]))
        bad = deepcopy(docs[0])
        bad["quality_review"]["review_validation"]["attempts"][0]["transport_binding"]["wire_messages_hash"] = "changed"
        bad["quality_review"]["review_validation_hash"] = cc.fingerprint(bad["quality_review"]["review_validation"])
        self.assertFalse(cc._review_receipt_matches(ws, 0, bad))

    def test_negative_opinion_and_format_attempt_limit_are_preserved(self):
        for status in ("missing", "incorrect", "ambiguous"):
            with self.subTest(status=status):
                _ws, _docs, tracer, report = self.review(wire.SHARED, lambda p: valid(p, status))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(tracer.calls), 1)
                self.assertEqual(report["raw_output"]["verdict"], "fail")
        ws = world()
        docs = [{"title": "Original document", "content": "甲项目目前进行中。"}]
        tracer = SharedTracer([{}, {}])
        report = cc.review_documents(tracer, ws, 0, docs, requirements=cc.fidelity_requirements(ws, 0),
                                     evidence_transport=wire.SHARED)
        self.assertEqual((report["status"], len(tracer.calls)), ("error", 2))
        self.assertIn("format_repair", tracer.logical_inputs[1])
        self.assertEqual(tracer.logical_inputs[1]["CANON"], tracer.logical_inputs[0]["CANON"])

    def test_old_history_cannot_be_relabelled_as_shared_without_actual_wire_binding(self):
        _ws, _docs, _tracer, report = self.review()
        history = deepcopy(report["review_validation"])
        history["transport"] = wire.SHARED
        with self.assertRaisesRegex(ValueError, "transport"):
            cc._replay_review_validation(history)

    def test_extra_document_claim_outside_requirement_keeps_its_negative_opinion(self):
        ws = world()
        docs = [{"title": "Required fact", "content": "甲项目目前进行中。"},
                {"title": "Extra claim", "content": "另一个主体已经结束。"}]
        def unsupported(payload):
            result = valid(payload)
            result["verdict"] = "fail"
            result["document_reviews"][1].update(status="unsupported", reason="Fixed negative test opinion")
            result["unsupported_claims"] = [{"doc_index": 1, "quote": docs[1]["content"],
                                             "reason": "Claim outside the required target has no support"}]
            return result
        tracer = SharedTracer([unsupported])
        report = cc.review_documents(tracer, ws, 0, docs, requirements=cc.fidelity_requirements(ws, 0),
                                     evidence_transport=wire.SHARED)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(tracer.logical_inputs[0]["documents"], docs)
        self.assertEqual(report["raw_output"]["unsupported_claims"][0]["doc_index"], 1)
        self.assertEqual(len(tracer.calls), 1)
        with self.assertRaises(ValueError):
            cc.attach_receipts(docs, report, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
