"""Offline supply-loss regressions; fixture opinions are never re-scored here."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.answer_task_review import POLICY_VERSION
from pipeline.question_contract import build_question_contract, render_intent, validate_question
from pipeline.question_wording import SYSTEM, binding, validate_wording
from pipeline.render import phrase_questions

FIXTURE = ROOT / "output/showcase_quality_audit_20260926/run"
OK = {"verdict": "equivalent", "reason": "Offline scripted interface response.", "issues": []}


def ordering():
    # Audited question q_2361859c7c02bebf97ef2712: field-sort leaked chronology.
    return {"qid": "q_2361859c7c02bebf97ef2712", "line": "L3_process", "capability": "L3_order",
            "entity": "北墙哨钟", "field": "", "gt": [
                {"field": "权限登记", "value": "议会接管登记", "session": 8, "date": "2025-03-03", "op": "UPDATE"},
                {"field": "武装阶段", "value": "武装", "session": 9, "date": "2025-03-10", "op": "UPDATE"},
                {"field": "运行读数", "value": "74", "session": 10, "date": "2025-03-17", "op": "UPDATE"}],
            "evidence_sessions": [8, 9, 10], "aux": {"scorer": "kendall_tau"}}


class Tracer:
    def __init__(self):
        self.calls = []
    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), kwargs))
        if step != "phrase.review":
            raise AssertionError("Compiled L3 cards must not be rewritten by an author")
        return deepcopy(OK)


class SupplyWordingContracts(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_compiled_order_stable_unsorted_and_reference_unchanged(self):
        order = ordering()
        original = deepcopy(order)
        contract = build_question_contract(order)
        ids = [event["id"] for event in contract["event_refs"]]
        self.assertCountEqual(contract["event_display"]["event_ids"], ids)
        self.assertNotEqual(contract["event_display"]["event_ids"], ids)
        self.assertEqual(build_question_contract(order), contract)
        self.assertEqual(contract["gold"], original["gt"])
        self.assertEqual(order, original)
        self.assertEqual(render_intent(order, contract)[0], contract["canonical_question"])
        self.assertEqual(validate_question(contract["canonical_question"], contract), [])

    def test_real_render_semantic_and_material_modes_use_compiled_cards(self):
        for material_mode in (False, True):
            with self.subTest(material_mode=material_mode):
                order, tracer = ordering(), Tracer()
                wp = {"quality_contract": {"scoring_policy": POLICY_VERSION}}
                audit = {}
                kwargs = {"corpus": [{"session": 0, "documents": [{"id": "d000001", "content": "公开哨钟历史。"}]}],
                          "public_protocol": {"scope": "public"}} if material_mode else {}
                result = phrase_questions([order], wp, tracer, log=lambda *_: None, audit=audit, **kwargs)
                self.assertEqual(len(result), 1)
                question = result[0]
                self.assertEqual([call[0] for call in tracer.calls], ["phrase.review"])
                self.assertEqual(question["question"], question["question_contract"]["canonical_question"])
                self.assertEqual(question["gt"], order["gt"])
                self.assertEqual(question["evidence_sessions"], order["evidence_sessions"])
                self.assertEqual(validate_question(question), [])
                self.assertEqual(validate_wording(question), [])
                old_text = build_question_contract(order, contract_version=1)["canonical_question"]
                self.assertNotEqual(question["question"], old_text)
                self.assertEqual(validate_question(old_text, question["question_contract"])[0]["code"], "event_display_changed")

    def test_leakage_is_question_text_only_and_zero_based_time_is_explicit(self):
        order = {"line": "L1_timeline", "capability": "IE", "entity": "甲", "field": "负责人",
                 "gt": "张甲", "aux": {"at_week": 8, "time_unit": "周"}}
        contract = build_question_contract(order)
        self.assertEqual(contract["gold"], "张甲")
        self.assertIn("张甲", contract["forbidden_answer_values"])
        self.assertEqual(contract["answer_leakage_policy"]["scope"], "question_text_only")
        self.assertEqual(contract["query_time"]["ordinal"], 9)
        self.assertEqual(contract["time_coordinates"]["conversion"], "display_period=session_index+1")
        self.assertEqual(validate_question(contract["canonical_question"], contract), [])
        self.assertIn("参考答案出现在隐藏值列表中是正常的防泄露设置", SYSTEM)
        refusal = {"line": "L6_refusal", "capability": "ABS", "entity": "甲", "field": "负责人",
                   "gt": "INSUFFICIENT_EVIDENCE", "aux": {"refusal_type": "T2_window", "probe": {"at_week": 14}}}
        self.assertEqual(build_question_contract(refusal)["query_time"]["ordinal"], 14)

    def test_legacy_contract_and_review_receipt_remain_readable(self):
        order = ordering()
        contract = build_question_contract(order, contract_version=1)
        old = {**order, "question_contract": contract, "question": contract["canonical_question"]}
        old["question_validation"] = {"semantic_review": {"binding": binding(old["question"], contract),
            "status": "passed", "opinion": deepcopy(OK)}}
        self.assertEqual(old["question_validation"]["semantic_review"]["binding"]["version"], "original-question-wording/v3")
        self.assertEqual(validate_question(old), [])
        self.assertEqual(validate_wording(old), [])
        upgraded = deepcopy(old)
        upgraded["question_contract"] = build_question_contract(order)
        self.assertEqual(validate_wording(upgraded)[0]["code"], "missing_or_stale_wording_review")

    def test_judge_accepts_new_contract_with_identical_value_scoring(self):
        from eval.judge import is_judgeable, judge_record
        order = {"qid": "score-v2", "line": "L1_timeline", "capability": "KU", "entity": "甲",
                 "field": "负责人", "gt": "张甲", "aux": {}}
        records = []
        for version in (1, 2):
            contract = build_question_contract(order, contract_version=version)
            question = {**order, "question_contract": contract, "question": contract["canonical_question"]}
            self.assertEqual(validate_question(question), [])
            self.assertTrue(is_judgeable(question))
            records.append([judge_record(question, answer, use_llm=False)["correct"] for answer in ("张甲", "张乙")])
        self.assertEqual(records, [[True, False], [True, False]])
        question["question_contract"]["version"] = 999
        self.assertFalse(is_judgeable(question))

    @unittest.skipUnless(FIXTURE.exists(), "Optional frozen real Showcase artifacts")
    def test_real_six_unformed_keep_gold_and_explicit_question_only_contract(self):
        report = json.loads((FIXTURE / "04_wording_report.json").read_text(encoding="utf-8"))
        rows = [row for row in report["items"] if row["status"] == "unresolved"]
        self.assertEqual(len(rows), 6)
        for row in rows:
            with self.subTest(qid=row["source_qid"]):
                old = deepcopy(row["original_order"])
                policy = (old.get("question_contract") or {}).get("scoring_policy", POLICY_VERSION)
                contract = build_question_contract(old, {"quality_contract": {"scoring_policy": policy}})
                self.assertEqual(contract["gold"], old["gt"])
                self.assertEqual(contract["answer_leakage_policy"]["scope"], "question_text_only")
                self.assertEqual(row["original_order"], old)
                self.assertNotEqual(contract["contract_id"], old["question_contract"]["contract_id"])
        question = next(question for question in json.loads((FIXTURE / "04_questions.json").read_text(encoding="utf-8"))
                        if question["qid"] == ordering()["qid"])
        self.assertEqual(validate_wording(question), [])
        self.assertEqual(validate_question(question), [])
        self.assertEqual(question["gt"], ordering()["gt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
