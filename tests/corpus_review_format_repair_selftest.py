"""Offline format-correction controls. Synthetic replies are not model quality evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-corpus-format-test")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline import corpus_contract as cc
from pipeline.world_state import WorldState, Timeline, Op, SET, _date_of
from pipeline.run import Tracer
from pipeline import render as renderer
import config


def world():
    ws = WorldState(entities={"甲项目": {"状态": Timeline([Op(0, _date_of(0), SET, "进行中")])}}, n_sessions=1)
    return ws


def valid(payload, status="supported"):
    return {"document_reviews": fixed_document_reviews(payload["documents"]),
            "verdict": "pass" if status == "supported" else "fail", "unsupported_claims": [],
            "coverage": [{"requirement_id": item["requirement_id"], "status": status,
                          "reason": "明确构造的离线意见，只检验执行，不证明语义正确率。",
                          "evidence": [] if status == "missing" else [{"doc_index": 0, "quote": payload["documents"][0]["content"]}]}
                         for item in payload["requirements"]]}


def rehash(receipt):
    history = receipt["review_validation"]
    previous = None
    for attempt in history["attempts"]:
        attempt["request_hash"] = cc._request_hash(cc._review_attempt_payload(history["input_payload"], previous))
        previous = attempt
    history["final_request_hash"] = previous["request_hash"]
    history["final_response_hash"] = cc.fingerprint(previous["raw_output"])
    history["final_opinion_hash"] = cc.fingerprint(cc._saved_review_output(previous))
    receipt["review_validation_hash"] = cc.fingerprint(history)


class FakeTracer:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if not self.replies:
            raise AssertionError("Unexpected extra call")
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result(json.loads(messages[-1]["content"])) if callable(result) else result)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.guards = [patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
                       patch.object(config, "_client", side_effect=AssertionError("Real provider initialization forbidden"), create=True)]
        for guard in self.guards:
            guard.start(); self.addCleanup(guard.stop)
        self.docs = [{"doc_id": "d0", "title": "纪要", "content": "甲项目目前进行中。"},
                     {"doc_id": "d1", "title": "归档", "content": "本组记录归档。"}]

    def review(self, replies):
        tracer = FakeTracer(replies)
        report = cc.review_documents(tracer, world(), 0, self.docs,
                                     requirements=cc.fidelity_requirements(world(), 0))
        return report, tracer

    def test_cross_document_splice_then_fake_reviewer_corrects_own_evidence(self):
        ws = world()
        ws.entities["乙项目"] = {"状态": Timeline([Op(0, _date_of(0), SET, "收尾")])}
        docs = [{"title": "甲记录", "content": "记录日期为2025-01-06；甲项目的状态为进行中。"},
                {"title": "乙记录", "content": "本期另行登记乙项目；状态为收尾。记录截至2025-01-06。"}]
        requirements = cc.fidelity_requirements(ws, 0)
        original = {"document_reviews": fixed_document_reviews(docs), "verdict": "pass", "unsupported_claims": [], "coverage": [
            {"requirement_id": "r1", "status": "supported", "reason": "离线原文定位对照。",
             "evidence": [{"doc_index": 0, "quote": "甲项目的状态为进行中"}]},
            {"requirement_id": "r2", "status": "supported", "reason": "离线拼接引文反例。",
             "evidence": [{"doc_index": 1, "quote": "记录日期为2025-01-06；状态为收尾"}]}]}
        corrected = deepcopy(original)
        corrected["coverage"][1]["evidence"][0]["quote"] = "状态为收尾"
        before = deepcopy((docs, original, ws.to_dict()))
        locator_patch = {"locations": [{"path": "/coverage/1/evidence/0", "doc_index": 1, "quote": "状态为收尾"}]}
        trace = FakeTracer([original, locator_patch])
        report = cc.review_documents(trace, ws, 0, docs, requirements=requirements)
        self.assertEqual((report["status"], len(trace.calls)), ("passed", 2))
        first, second = [json.loads(call[1][-1]["content"]) for call in trace.calls]
        feedback = second.pop("format_repair")
        self.assertEqual(first, second)
        self.assertEqual(first["documents"], docs)
        self.assertEqual(feedback["previous_response"], original)
        self.assertEqual(feedback["validation_error"]["path"], "/coverage/1/evidence/0/quote")
        self.assertEqual([call[0] for call in trace.calls], ["corpus.review", "corpus.review"])
        self.assertEqual(trace.calls[0][2], trace.calls[1][2])
        self.assertEqual((docs, original, ws.to_dict()), before)
        self.assertEqual([a["raw_output"] for a in report["review_validation"]["attempts"]], [original, locator_patch])
        self.assertEqual(report["review_validation"]["attempts"][1]["resolved_output"], corrected)
        cc.attach_receipts(docs, report, 0)
        self.assertTrue(all(cc._review_receipt_matches(ws, 0, doc) for doc in docs))

    def test_semantic_negative_is_single_call_and_raw_preserved(self):
        for status in ("missing", "incorrect", "ambiguous"):
            with self.subTest(status=status):
                report, trace = self.review([lambda p: valid(p, status)])
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(trace.calls), 1)
                self.assertIsNone(report["review_validation"]["attempts"][0]["validation_error"])

    def test_invalid_then_valid_negative_is_not_forced_to_pass(self):
        report, trace = self.review([{}, lambda p: valid(p, "missing")])
        self.assertEqual((report["status"], len(trace.calls)), ("failed", 2))
        self.assertEqual(report["raw_output"]["coverage"][0]["evidence"], [])
        def invalid_negative(p):
            raw = valid(p, "incorrect")
            raw["coverage"][0]["evidence"][0]["quote"] = "不存在的原文"
            return raw
        report, trace = self.review([invalid_negative, valid])
        self.assertEqual((report["status"], len(trace.calls)), ("error", 2))
        # Pure locator repair has no authority to replace the semantic opinion.
        self.assertEqual(report["review_validation"]["attempts"][0]["raw_output"]["verdict"], "fail")
        self.assertEqual(report["review_validation"]["attempts"][1]["raw_output"]["verdict"], "pass")
        self.assertEqual(report["review_validation"]["attempts"][1]["validation_error"]["path"], "/locations")

    def test_second_bad_evidence_remains_execution_error(self):
        def bad(p):
            obj = valid(p); obj["coverage"][0]["evidence"][0]["quote"] = "不存在的原文"
            return obj
        report, trace = self.review([bad, {"locations": [{"path": "/coverage/0/evidence/0", "doc_index": 0, "quote": "不存在的原文"}]}])
        self.assertEqual((report["status"], len(trace.calls)), ("error", 2))
        attempts = report["review_validation"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(a["validation_error"]["path"].endswith("/quote") for a in attempts))
        with self.assertRaises(ValueError): cc.attach_receipts(deepcopy(self.docs), report, 0)

    def test_deleting_evidence_cannot_make_supported_valid(self):
        def missing(p):
            obj = valid(p); obj["coverage"][0]["evidence"] = []
            return obj
        report, trace = self.review([missing, missing])
        self.assertEqual((report["status"], len(trace.calls)), ("error", 2))
        self.assertEqual(report["review_validation"]["attempts"][-1]["validation_error"]["path"], "/coverage/0/evidence")

    def test_transport_empty_or_unparsed_error_never_gets_format_retry(self):
        for failed in (TimeoutError("deadline"), "", None, {"__error__": "empty output: finish length"}):
            with self.subTest(failed=str(failed)):
                report, trace = self.review([failed])
                self.assertEqual((report["status"], len(trace.calls)), ("error", 1))
                self.assertIsNotNone(report["review_validation"]["attempts"][0]["execution_error"])

    def test_input_identity_error_cannot_enter_correction(self):
        trace = FakeTracer([])
        report = cc.review_documents(trace, world(), 0, self.docs,
            requirements=[{"requirement_id": "r1", "kind": "value", "target": {"invented": True}}])
        self.assertEqual((report["status"], len(trace.calls)), ("error", 0))

    def test_tracer_guard_exception_on_second_call_stops_and_keeps_first_raw(self):
        report, trace = self.review([{}, RuntimeError("source freeze changed")])
        self.assertEqual((report["status"], len(trace.calls)), ("error", 2))
        self.assertEqual(report["review_validation"]["attempts"][0]["raw_output"], {})
        self.assertIn("source freeze", report["review_validation"]["attempts"][1]["execution_error"]["message"])

    def test_receipts_replay_both_attempts_and_reject_changed_history(self):
        report, _ = self.review([{}, valid])
        docs = deepcopy(self.docs); cc.attach_receipts(docs, report, 0)
        self.assertTrue(cc._review_receipt_matches(world(), 0, docs[0]))
        for change in ("first_raw", "error_path", "payload", "final_raw", "admission"):
            with self.subTest(change=change):
                bad = deepcopy(docs[0]); r = bad["quality_review"]; h = r["review_validation"]
                if change == "first_raw": h["attempts"][0]["raw_output"] = valid(h["input_payload"])
                elif change == "error_path": h["attempts"][0]["validation_error"]["path"] = "/madeup"
                elif change == "payload": h["input_payload"]["documents"][0]["content"] += "Changed"
                elif change == "final_raw": h["attempts"][1]["raw_output"]["verdict"] = "fail"
                else: h["attempts"][0]["validation_error"] = None
                r["review_validation_hash"] = cc.fingerprint(h)
                self.assertFalse(cc._review_receipt_matches(world(), 0, bad), change)

    def test_final_mapped_fields_cannot_disagree_with_raw(self):
        report, _ = self.review([valid]); report["fidelity_coverage"][0]["reason"] = "edited"
        with self.assertRaises(ValueError): cc.attach_receipts(deepcopy(self.docs), report, 0)

    def test_replayed_rule_and_source_targets_match_their_full_canonical_declarations(self):
        for kind in ("rule", "source"):
            ws = world(); ws.entity_types = {"甲项目": "project"}
            ws.world_blueprint = {"entity_types": [{"id": "project", "noun": "项目", "fields": [
                {"name": "状态", "kind": "status", "states": ["立项", "进行中", "收尾"]}]}]}
            context = cc.canonical_context(ws, 0)
            rule_ids = []
            if kind == "rule": rule_ids = [context["public_stage_rules"][0]["rule_id"]]
            else:
                ws.conflicts = [{"session": 0, "entity": "甲项目", "field": "状态", "date": _date_of(0),
                                 "authoritative_source": "官方记录", "authoritative_value": "进行中"}]
                context["source_assertions"] = cc.authoritative_source_assertions(ws, 0)
            docs = [{"title": "阶段与来源", "content": "项目阶段为立项、进行中、收尾；官方记录甲项目目前进行中。"}]
            report = cc.review_documents(FakeTracer([valid]), ws, 0, docs, context=context,
                                         required_public_rule_ids=rule_ids)
            cc.attach_receipts(docs, report, 0)
            self.assertTrue(cc._review_receipt_matches(ws, 0, docs[0]))
            target = docs[0]["quality_review"]["review_validation"]["input_payload"]["requirements"][0]["target"]
            if kind == "rule": target["ordered_stages"] = ["收尾", "进行中", "立项"]
            else: target["value"] = "完成"
            rehash(docs[0]["quality_review"])
            self.assertFalse(cc._review_receipt_matches(ws, 0, docs[0]))

    def test_replay_rejects_execution_sentinel_even_with_other_valid_keys(self):
        report, _ = self.review([valid]); docs = deepcopy(self.docs)
        cc.attach_receipts(docs, report, 0)
        history = docs[0]["quality_review"]["review_validation"]
        history["attempts"][0]["raw_output"]["__error__"] = "provider error"
        rehash(docs[0]["quality_review"])
        self.assertFalse(cc._review_receipt_matches(world(), 0, docs[0]))

    def test_title_only_quote_or_wrong_doc_is_still_rejected(self):
        for kind in ("title", "wrong_index"):
            def bad(p):
                obj = valid(p)
                if kind == "title": obj["coverage"][0]["evidence"][0]["quote"] = p["documents"][0]["title"]
                else: obj["coverage"][0]["evidence"][0]["doc_index"] = 1
                return obj
            report, trace = self.review([bad, bad])
            self.assertEqual((report["status"], len(trace.calls)), ("error", 2))

    def test_title_claim_remains_allowed_while_coverage_is_body_only(self):
        def title_claim(p, index):
            obj = valid(p); obj["verdict"] = "fail"
            obj["document_reviews"] = fixed_document_reviews(p["documents"], unsupported_indices=[index])
            obj["unsupported_claims"] = [{"doc_index": index, "quote": p["documents"][0]["title"],
                                          "reason": "离线标题异议，程序只检查归属。"}]
            return obj
        def invalid_quote(p):
            raw = title_claim(p, 0)
            raw["unsupported_claims"][0]["quote"] = "不存在的标题"
            return raw
        report, trace = self.review([invalid_quote,
            {"locations": [{"path": "/unsupported_claims/0", "doc_index": 0, "quote": self.docs[0]["title"]}]}])
        self.assertEqual((report["status"], len(trace.calls)), ("failed", 2))
        feedback = json.loads(trace.calls[1][1][-1]["content"])["format_repair"]
        self.assertIn("标题或正文", feedback["instructions"])
        self.assertEqual(feedback["validation_error"]["path"], "/unsupported_claims/0/quote")

    def test_complete_support_group_is_still_required(self):
        report, _ = self.review([{}, valid]); docs = deepcopy(self.docs)
        cc.attach_receipts(docs, report, 0)
        full = {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": docs}]}
        self.assertEqual(cc.validate_corpus(world(), full)["status"], "passed")
        full["sessions"][0]["docs"].pop()
        self.assertEqual(cc.validate_corpus(world(), full)["status"], "failed")

    def test_real_original_tracer_records_both_calls_same_step(self):
        replies = [{}, valid]
        def offline_provider(messages, **kwargs):
            self.assertEqual(kwargs.get("response_format"), {"type": "json_object"})
            self.assertIs(kwargs.get("strict_json"), True)
            self.assertEqual(kwargs.get("retries"), 3)
            result = replies.pop(0)
            return result(json.loads(messages[-1]["content"])) if callable(result) else result
        with tempfile.TemporaryDirectory(prefix="offline-corpus-format-") as td, patch.object(config, "chat_json", side_effect=offline_provider):
            tracer = Tracer(SimpleNamespace(dir=Path(td)))
            report = cc.review_documents(tracer, world(), 0, self.docs,
                requirements=cc.fidelity_requirements(world(), 0))
            rows = [json.loads(line) for line in tracer.pfile.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["status"], "passed")
            self.assertEqual([r["step"] for r in rows], ["corpus.review", "corpus.review"])
            self.assertEqual(rows[0]["output"], {})
            self.assertEqual(rows[1]["output"], report["raw_output"])

    def test_real_render_keeps_correction_in_review_not_author_loop(self):
        counts = {"review": 0, "signal": 0}
        def provider(messages, **kwargs):
            if messages[0]["content"] == cc.REVIEW_SYSTEM:
                counts["review"] += 1
                return {} if counts["review"] == 1 else valid(json.loads(messages[-1]["content"]))
            if messages[0]["content"] == renderer.render("discriminate.quality_system"):
                return {"answers": [{"key": "q0", "answer": "进行中"}]}
            counts["signal"] += 1
            return {"docs": [{"title": "记录", "content": "甲项目目前进行中。"}]}
        with tempfile.TemporaryDirectory(prefix="offline-corpus-render-") as td, patch.object(config, "chat_json", side_effect=provider), patch.object(config, "pmap", side_effect=lambda fn, items, **kw: [fn(x) for x in items]):
            tracer = Tracer(SimpleNamespace(dir=Path(td))); result = {"sessions": []}
            renderer.render_corpus({"quality_contract": {"corpus_review": True}}, world(), 0,
                tracer, result, set(), lambda: None, log=lambda *_: None)
            self.assertEqual(counts, {"review": 2, "signal": 1})
            self.assertEqual(cc.validate_corpus(world(), result)["status"], "passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
