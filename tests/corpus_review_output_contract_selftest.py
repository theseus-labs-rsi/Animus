"""Offline unified reviewer-output regressions; fixed opinions are not semantic gold."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_source_assertion_selftest as fixtures
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline import corpus_contract as contract
from pipeline.world_state import Op, SET, Timeline, _date_of


class Tracer:
    def __init__(self, output):
        self.output, self.messages = output, None

    def chat_json(self, step, messages, **kwargs):
        self.messages = deepcopy(messages)
        return deepcopy(self.output)


class CorpusReviewOutputContractTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network prohibited"))
        guard.start(); self.addCleanup(guard.stop)
        self.ws = fixtures.world()
        self.ws.entities["甲项目"]["状态"] = Timeline([Op(0, _date_of(0), SET, "立项")])
        self.ws.entity_types = {"甲项目": "project"}
        self.ws.world_blueprint = {"entity_types": [{"id": "project", "noun": "项目", "fields": [
            {"name": "状态", "kind": "status", "states": ["立项", "进行中", "完成"]}]}]}
        self.context = fixtures.source_context(self.ws)
        self.rule_id = self.context["public_stage_rules"][0]["rule_id"]
        self.source_id = self.context["source_assertions"][0]["assertion_id"]
        self.docs = [{"content": f"{_date_of(0)}，甲项目负责人经办0，该信息来自甲项目签发记录。"},
                     {"content": f"{_date_of(0)}，项目的状态阶段顺序为立项、进行中、完成；实际更新可跳级。"}]

    def output(self):
        return {"document_reviews": fixed_document_reviews(self.docs), "verdict": "pass", "unsupported_claims": [],
                "coverage": [{"requirement_id": "r1", "status": "supported",
                    "reason": "Offline fixture: rule body reference.",
                    "evidence": [{"doc_index": 1, "quote": self.docs[1]["content"]}]},
                {"requirement_id": "r2", "status": "supported",
                    "reason": "Offline fixture: source body reference.",
                    "evidence": [{"doc_index": 0, "quote": self.docs[0]["content"]}]}]}

    def review(self, raw):
        tracer = Tracer(raw)
        report = contract.review_documents(tracer, self.ws, 0, self.docs, context=self.context,
                                           required_public_rule_ids=[self.rule_id])
        return report, tracer

    def test_actual_prompt_and_dynamic_template_have_one_complete_coverage(self):
        report, tracer = self.review(self.output())
        self.assertEqual(report["status"], "passed")
        system = tracer.messages[0]["content"]
        payload = json.loads(tracer.messages[-1]["content"])
        self.assertEqual([r["requirement_id"] for r in payload["coverage_rows_to_complete"]], ["r1", "r2"])
        self.assertTrue(all(r["status"] != "supported" for r in payload["coverage_rows_to_complete"]))
        self.assertIn("四个顶层键", system)
        self.assertEqual([r["kind"] for r in payload["requirements"]], ["public_rule", "public_source"])

    def test_source_and_rule_outcomes_are_both_required_for_pass(self):
        for index in (0, 1):
            with self.subTest(index=index):
                raw = self.output()
                raw["coverage"][index].update(status="missing", evidence=[])
                raw["verdict"] = "fail"
                report, _ = self.review(raw)
                self.assertEqual(report["status"], "failed", report)
                self.assertEqual(raw["unsupported_claims"], [])
                # Complete arrays cannot turn a negative coverage judgment into
                # approval just by leaving the top-level verdict at pass.
                raw["verdict"] = "pass"
                report, _ = self.review(raw)
                self.assertEqual(report["status"], "error")

    def test_old_shape_or_missing_required_row_still_errors(self):
        for omitted in (("document_reviews",), ("coverage",), ("unsupported_claims",), ("verdict",)):
            with self.subTest(omitted=omitted):
                raw = self.output()
                for key in omitted:
                    raw.pop(key)
                report, _ = self.review(raw)
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["raw_output"], raw)
                with self.assertRaises(ValueError):
                    contract.attach_receipts(deepcopy(self.docs), report, 0)

    def test_empty_target_arrays_are_valid_but_changed_prompt_receipt_is_stale(self):
        context = contract.canonical_context(self.ws, 0)
        raw = {"document_reviews": fixed_document_reviews([{}]), "verdict": "pass", "unsupported_claims": [], "coverage": []}
        docs = [{"doc_id": "d0", "content": "甲项目的状态为立项。"}]
        current = contract.review_documents(Tracer(raw), self.ws, 0, docs, context=context)
        self.assertEqual(current["status"], "passed")
        contract.attach_receipts(docs, current, 0)
        self.assertTrue(contract._review_receipt_matches(self.ws, 0, docs[0]))
        with patch.object(contract, "REVIEW_SYSTEM", "An earlier incompatible output template"):
            old = contract.review_documents(Tracer(raw), self.ws, 0, docs, context=context)
            contract.attach_receipts(docs, old, 0)
        self.assertFalse(contract._review_receipt_matches(self.ws, 0, docs[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
