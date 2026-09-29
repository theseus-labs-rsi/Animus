"""Offline binding regressions; scripted opinions are not semantic gold labels."""
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
from corpus_fixture_helpers import fixed_document_reviews
from pipeline.corpus_contract import (
    _review_receipt_matches, attach_receipts, public_stage_rules,
    review_documents, validate_corpus,
)
from pipeline.world_state import WorldState, _date_of


def world():
    ws = WorldState(entities={}, n_sessions=1)
    ws.world_blueprint = {
        "entity_types": [{"id": "project", "noun": "项目", "fields": [{
            "name": "状态", "kind": "status", "states": ["立项", "进行中", "完成"],
        }]}], "temporal_model": {"unit": "week", "step_days": 7},
    }
    return ws


def reviewed_group(ws, prefix, *, identical=False, single=False):
    bodies = (["2025-01-06，项目状态顺序为立项、进行中、完成。"] if single else [
        "2025-01-06，项目状态的第一段顺序为立项、进行中。",
        "2025-01-06，该项目阶段顺序中进行中之后为完成。",
    ])
    docs = [{"title": f"{prefix}-{0 if identical else i}",
             "content": bodies[0] if identical else body, "fact_refs": []}
            for i, body in enumerate(bodies)]
    rule_id = public_stage_rules(ws)[0]["rule_id"]

    class Review:
        def chat_json(self, _step, messages, **_kwargs):
            targets = json.loads(messages[-1]["content"])["requirements"]
            return {"document_reviews": fixed_document_reviews(docs), "verdict": "pass", "unsupported_claims": [],
                    "coverage": [{
                        "requirement_id": target["requirement_id"], "status": "supported",
                        "reason": "固定桩只检验证据组保全，不认证正文语义。",
                        "evidence": [{"doc_index": i, "quote": doc["content"]}
                                     for i, doc in enumerate(docs)],
                    } for target in targets]}

    report = review_documents(Review(), ws, 0, docs, required_public_rule_ids=[rule_id])
    assert report["status"] == "passed", report
    attach_receipts(docs, report, 0)
    # Mirror production: stable corpus IDs are assigned only after reviewing.
    for i, doc in enumerate(docs):
        doc["doc_id"] = f"s0_sig_{prefix}_{i}"
    return docs


def corpus(docs):
    return {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": deepcopy(docs)}]}


class PublicRuleSupportSetTests(unittest.TestCase):
    def setUp(self):
        self.ws = world()
        self.network_guard = patch.object(socket.socket, "connect",
                                         side_effect=AssertionError("network forbidden"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def test_joint_support_requires_each_original_document(self):
        docs = reviewed_group(self.ws, "joint")
        self.assertEqual(validate_corpus(self.ws, corpus(docs))["status"], "passed")
        for removed in range(2):
            with self.subTest(removed=removed):
                result = validate_corpus(self.ws, corpus([d for i, d in enumerate(docs) if i != removed]))
                self.assertEqual(result["status"], "failed")
                self.assertIn("missing_public_rule_material", {i["code"] for i in result["issues"]})

    def test_mutated_member_and_wrong_date_or_context_invalidate_group(self):
        docs = reviewed_group(self.ws, "mutation")
        for kind in ("title", "content", "date", "context"):
            candidate, ws = corpus(docs), deepcopy(self.ws)
            if kind in ("title", "content"):
                candidate["sessions"][0]["docs"][1][kind] += "changed"
            elif kind == "date":
                candidate["sessions"][0]["date"] = "2026-01-06"
            else:
                ws.world_blueprint["entity_types"][0]["fields"][0]["states"].append("归档")
            with self.subTest(kind=kind):
                self.assertEqual(validate_corpus(ws, candidate)["status"], "failed")

    def test_assignment_and_reordering_preserve_original_review_slots(self):
        docs = list(reversed(reviewed_group(self.ws, "order")))
        for index, doc in enumerate(docs):
            doc["doc_id"] = f"s0_sig_renumbered_{index}"
        self.assertEqual(validate_corpus(self.ws, corpus(docs))["status"], "passed")

    def test_identical_text_does_not_replace_a_missing_original_slot(self):
        docs = reviewed_group(self.ws, "duplicates", identical=True)
        self.assertEqual(validate_corpus(self.ws, corpus(docs))["status"], "passed")
        copied_first = deepcopy(docs[0])
        copied_first["doc_id"] = "s0_sig_another_id"
        self.assertEqual(validate_corpus(self.ws, corpus([docs[0], copied_first]))["status"], "failed")

    def test_alternative_complete_groups_are_not_mixed(self):
        first = reviewed_group(self.ws, "first")
        second = reviewed_group(self.ws, "second")
        for retained in (first, second, first + second, first + second[:1]):
            with self.subTest(retained=[d["doc_id"] for d in retained]):
                self.assertEqual(validate_corpus(self.ws, corpus(retained))["status"], "passed")
        self.assertEqual(validate_corpus(self.ws, corpus([first[0], second[1]]))["status"], "failed")

    def test_separately_reviewed_single_document_can_be_complete_alternative(self):
        incomplete = reviewed_group(self.ws, "incomplete")[:1]
        complete = reviewed_group(self.ws, "single", single=True)
        self.assertEqual(validate_corpus(self.ws, corpus(incomplete + complete))["status"], "passed")

    def test_old_unbound_rule_receipts_are_not_upgraded(self):
        docs = reviewed_group(self.ws, "old")
        for doc in docs:
            for key in ("review_document_index", "review_documents_hash", "public_rule_support_sets"):
                doc["quality_review"].pop(key)
        self.assertEqual(validate_corpus(self.ws, corpus(docs))["status"], "failed")

    def test_ordinary_receipts_do_not_require_rule_binding_fields(self):
        ws = WorldState(entities={}, n_sessions=1)
        docs = [{"doc_id": "s0_sig_ordinary", "content": "档案登记。", "fact_refs": []}]

        class Review:
            def chat_json(self, *_args, **_kwargs):
                return {"document_reviews": fixed_document_reviews(docs), "verdict": "pass", "unsupported_claims": [], "coverage": []}

        report = review_documents(Review(), ws, 0, docs)
        attach_receipts(docs, report, 0)
        self.assertNotIn("public_rule_support_sets", docs[0]["quality_review"])
        self.assertTrue(_review_receipt_matches(ws, 0, docs[0]))
        self.assertEqual(validate_corpus(ws, corpus(docs))["status"], "passed")


if __name__ == "__main__":
    unittest.main()
