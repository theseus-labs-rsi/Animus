"""Offline orchestration/authority tests, not a claim of LLM semantic accuracy."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.semantic_review import (prepare_review, review_questions, visible_documents,
                                     resolve_citations, locate_citations, review_certification,
                                     _validate_readable_response, _validate_response,
                                     CITATION_POLICY, CERTIFICATION_POLICY, ISOLATED_VERSION,
                                     ISOLATED_CERTIFICATION_POLICY, validate_isolated_completed,
                                     SemanticReviewAuditError)
from tools.review_semantics import _new_output, main


CORPUS = {"world": "PRIVATE_WORLD_SENTINEL", "sessions": [{
    "session_id": 3, "date": "2025-01-09", "private": "PRIVATE_SESSION_SENTINEL", "docs": [
        {"doc_id": "PRIVATE_SIGNAL_ID", "title": "欧洲客户记录", "content": "报告客户是北溟保险，所属市场为欧洲。",
         "is_filler": False, "refs": "PRIVATE_REFS_SENTINEL", "quality_review": "PRIVATE_REVIEW_SENTINEL"},
        {"doc_id": "PRIVATE_FILLER_ID", "title": "更正通知", "content": "之前提到的登记并不代表分析确认。",
         "is_filler": True, "is_conflict": True, "private": "PRIVATE_DOC_SENTINEL"}
    ]}]}
PROTOCOL = {"scope": "仅使用提供材料；信息不足时说明原因。", "version": 1}


def coverage(status="complete"):
    return {"status": status, "scope_conflict": False,
            "inspected_doc_ids": ["d000001", "d000002"], "limitations": []}


def evidence(quote="报告客户是北溟保险", field="content", doc_id="d000001"):
    return [{"doc_id": doc_id, "field": field, "quote": quote,
             "role": "support", "explanation": "客户名在公开正文中。"}]


def blind(**overrides):
    result = {"interpretation": "查询报告的客户是谁", "answer": "北溟保险", "answerability": "answerable",
              "major_requirements": ["给出报告客户身份"],
              "coverage": coverage(), "evidence": evidence(), "reasoning": "根据明确的客户表述。"}
    result.update(overrides)
    return result


def adjudication(**overrides):
    result = {"item_validity": "valid", "answerability": "answerable", "reference_status": "contradicted",
              "reviewed_answer": "北溟保险", "interpretation": "问题询问客户身份而不是所属市场。",
              "reviewed_rationale": "正文明确说明客户是北溟保险。",
              "major_requirements": ["给出报告客户身份"],
              "original_answer_review": [{"requirement": "给出报告客户身份", "assessment": "未完成",
                                          "explanation": "欧洲是市场，不是客户。"}],
              "original_rationale_review": {"status": "not_provided", "claims": [], "limitations": []},
              "review_findings": {"substantive_defects": ["原答案混淆客户与市场。"],
                                  "acceptable_brevity": [], "editorial_suggestions": []},
              "coverage": coverage(), "evidence": evidence(), "concerns": ["参考答案欧洲回答了另一问题。"],
              "reasoning": "题目可答，但参考提案不符合题意。"}
    result.update(overrides)
    return result


class Scripted:
    """Deliberately scripted semantic decisions; never calls a real model."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, step, messages, **kwargs):
        self.calls.append({"step": step, "messages": deepcopy(messages), "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("Unexpected model call")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


class SemanticReviewTests(unittest.TestCase):
    def run_review(self, script=None, questions=None, **kwargs):
        script = script or Scripted(blind(), adjudication())
        questions = questions if questions is not None else [{"question": "报告的客户是谁？", "gold": "欧洲"}]
        return review_questions(questions, CORPUS, PROTOCOL, chat_json=script,
                                reviewer_model="explicit-reviewer", reader_model="explicit-reader", **kwargs)

    def test_blind_view_excludes_reference_world_labels_and_original_ids(self):
        script = Scripted(blind(), adjudication())
        questions = [{"question": "报告的客户是谁？", "gold": "PRIVATE_GOLD_SENTINEL",
                      "qid": "PRIVATE_QID_SENTINEL", "line": "PRIVATE_LINE_SENTINEL",
                      "aux": {"gold": "PRIVATE_AUX_SENTINEL"}, "world": "PRIVATE_QUESTION_WORLD"}]
        report = self.run_review(script, questions)
        first = json.loads(script.calls[0]["messages"][1]["content"])
        self.assertEqual(set(first), {"question", "public_protocol", "documents", "input_scope"})
        self.assertNotIn("PRIVATE_", json.dumps(first))
        second = json.loads(script.calls[1]["messages"][1]["content"])
        self.assertEqual(second["reference_proposal"], "PRIVATE_GOLD_SENTINEL")
        self.assertNotIn("PRIVATE_WORLD", json.dumps(second))
        self.assertNotIn("PRIVATE_AUX", json.dumps(second))
        self.assertEqual(report["source_map"][0]["source_doc_id"], "PRIVATE_SIGNAL_ID")

    def test_structured_suggestion_does_not_replace_or_upgrade_original_reference(self):
        suggestion = {"customer": "北溟保险", "market": "欧洲", "events": [1, 2]}
        original = {"qid": "structured", "question": "报告的客户是谁？", "gt": {"customer": "欧洲"}}
        report = self.run_review(Scripted(blind(), adjudication(reviewed_answer=suggestion)), [original])
        item = report["items"][0]
        self.assertEqual(item["reviewed_answer"], suggestion)
        self.assertEqual(item["reference_status"], "contradicted")
        self.assertEqual(item["reference_proposal"]["answer"], original["gt"])
        self.assertEqual(original["gt"], {"customer": "欧洲"})
        self.assertEqual(item["stage_execution"]["adjudicate"]["status"], "ok")
        with self.assertRaises(ValueError):
            _validate_readable_response(adjudication(reviewed_answer=suggestion), "adjudicate",
                                        report["documents"], require_analysis=False)

    def test_review_suggestion_rejects_non_json_numbers(self):
        with self.assertRaises(ValueError):
            _validate_readable_response(adjudication(reviewed_answer={"value": float("nan")}),
                                        "adjudicate", visible_documents(CORPUS)[0], require_analysis=True)

    def test_titles_dates_sessions_and_filler_are_all_visible(self):
        docs, mapping = visible_documents(CORPUS, include_titles=True)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[1]["title"], "更正通知")
        self.assertEqual(docs[1]["content"], CORPUS["sessions"][0]["docs"][1]["content"])
        self.assertEqual(docs[1]["date"], "2025-01-09")
        self.assertEqual(docs[1]["session"], 3)
        self.assertEqual(set(docs[1]), {"doc_id", "title", "content", "date", "session"})
        self.assertEqual(mapping[1]["source_doc_id"], "PRIVATE_FILLER_ID")

    def test_default_view_excludes_titles_and_binds_opt_in(self):
        default = prepare_review(["问题"], CORPUS, PROTOCOL, reviewer_model="reviewer")
        titled = prepare_review(["问题"], CORPUS, PROTOCOL, reviewer_model="reviewer", include_titles=True)
        self.assertNotIn("title", default["documents"][0])
        self.assertEqual(default["binding"]["visible_view"], {"include_titles": False})
        self.assertNotEqual(default["binding"]["corpus_hash"], titled["binding"]["corpus_hash"])

    def test_sessions_sort_and_parent_metadata_match_solver_view(self):
        corpus = {"sessions": [
            {"session_id": "10", "date": "parent-ten", "docs": [{"content": "late", "date": "wrong-doc-date", "session": 99}]},
            {"session_id": "2", "date": "parent-two", "docs": [{"content": "early", "date": "also-wrong"}]}
        ]}
        docs, _ = visible_documents(corpus)
        self.assertEqual([(d["session"], d["date"], d["content"]) for d in docs],
                         [(2, "parent-two", "early"), (10, "parent-ten", "late")])

    def test_legacy_gt_values_and_missing_reference_are_distinguished(self):
        rows = [
            {"question": "身份？", "gt": "北溟保险", "aux": {"secret": "PRIVATE"}},
            {"question": "列出结果", "gt": {"甲": 1, "乙": 2}},
            {"question": "无参考"},
            {"question": "显式空参考", "reference_proposal": None}
        ]
        items = prepare_review(rows, CORPUS, PROTOCOL, reviewer_model="reviewer")["items"]
        self.assertEqual(items[0]["reference_proposal"], "北溟保险")
        self.assertEqual(items[1]["reference_proposal"], {"answer": {"甲": 1, "乙": 2}})
        self.assertFalse(items[2]["reference_provided"])
        self.assertTrue(items[3]["reference_provided"])
        self.assertNotEqual(items[2]["binding"]["proposal_hash"], items[3]["binding"]["proposal_hash"])

    def test_reference_presence_is_an_input_fact_not_a_model_choice(self):
        for questions, status in [([{"question": "客户是谁？", "gt": "北溟保险"}], "not_provided"),
                                  ([{"question": "客户是谁？"}], "supported")]:
            with self.subTest(status=status):
                item = self.run_review(Scripted(blind(), adjudication(reference_status=status)), questions)["items"][0]
                self.assertEqual(item["execution"]["status"], "invalid_review")
                self.assertEqual(item["review_state"], "pending")
        item = self.run_review(Scripted(blind(), adjudication(reference_status="not_provided")),
                               [{"question": "客户是谁？"}])["items"][0]
        self.assertEqual(item["review_state"], "completed")

    def test_legacy_abstention_encoding_is_only_an_adjudication_proposal(self):
        questions = [{"question": "分析确认状态能确定吗？", "gt": "INSUFFICIENT_EVIDENCE",
                      "question_contract": {"answer_kind": "abstention", "abstention_kind": "never_known",
                                            "lure": "PRIVATE_LURE_SENTINEL"},
                      "aux": {"world": "PRIVATE_WORLD_SENTINEL"}}]
        script = Scripted(blind(answer="无法确定", answerability="unanswerable"),
                          adjudication(answerability="unanswerable", reference_status="supported", reviewed_answer="无此项"))
        item = self.run_review(script, questions)["items"][0]
        blind_input = json.loads(script.calls[0]["messages"][1]["content"])
        adjudicator_input = json.loads(script.calls[1]["messages"][1]["content"])
        self.assertNotIn("INSUFFICIENT_EVIDENCE", json.dumps(blind_input))
        self.assertNotIn("reference_encoding", blind_input)
        self.assertEqual(adjudicator_input["reference_encoding"]["declared_subtype"]["code"], "never_known")
        self.assertNotIn("PRIVATE_", json.dumps(adjudicator_input))
        self.assertEqual(item["reference_proposal"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(item["reference_status"], "supported")
        changed = deepcopy(questions)
        changed[0]["question_contract"]["abstention_kind"] = "out_of_scope"
        other = prepare_review(changed, CORPUS, PROTOCOL, reviewer_model="explicit-reviewer", reader_model="explicit-reader")["items"][0]
        self.assertNotEqual(item["binding"]["encoding_hash"], other["binding"]["encoding_hash"])

    def test_new_literal_reference_is_not_silently_recast_as_legacy_code(self):
        item = prepare_review([{"question": "日志里写了哪个内部代码？", "reference_proposal": "INSUFFICIENT_EVIDENCE"}],
                              CORPUS, PROTOCOL, reviewer_model="reviewer")["items"][0]
        self.assertIsNone(item["reference_encoding"])

    def test_reference_can_be_overturned_without_invalidating_question(self):
        result = self.run_review()["items"][0]
        self.assertEqual(result["reference_proposal"], "欧洲")
        self.assertEqual(result["reference_status"], "contradicted")
        self.assertEqual(result["reviewed_answer"], "北溟保险")
        self.assertEqual(result["item_validity"], "valid")
        self.assertEqual(result["review_state"], "completed")

    def test_completing_reviewed_answer_does_not_upgrade_incomplete_reference(self):
        # Scripted decision checks preservation of the two distinct claims;
        # actual adherence to the revised prompt still needs model calibration.
        questions = [{"question": "给出客户名称，并说明是否记录了所属市场。", "gt": "欧洲"}]
        decision = adjudication(reference_status="unsupported",
                                reviewed_answer="客户为北溟保险；已记录所属市场为欧洲。",
                                concerns=["原提案只给出市场，遗漏客户名称。"],
                                reasoning="已写出的市场事实正确，但原提案未完整回答主要任务。")
        script = Scripted(blind(), decision)
        item = self.run_review(script, questions)["items"][0]
        self.assertEqual(item["reference_proposal"], "欧洲")
        self.assertEqual(item["reference_status"], "unsupported")
        self.assertEqual(item["reviewed_answer"], decision["reviewed_answer"])
        self.assertEqual(item["item_validity"], "valid")
        adjudicator_input = json.loads(script.calls[1]["messages"][1]["content"])
        self.assertEqual(adjudicator_input["reference_proposal"], "欧洲")

    def test_arbitrary_and_equivalent_questions_reach_models_without_old_gates(self):
        questions = [
            "截至第3天，蓝鲸保险2024年度客户留存分析报告所采用的来源，其报告期间是什么？",
            "截至第4天，蓝鲸保险2024年度客户留存分析报告的分析客户是谁？请同时说明是否记录了所属市场。",
            "第4天这个时间点不用考虑，问的是第1天：蓝鲸保险2024年度客户留存分析报告的分析客户的所属市场是什么？",
            "比较两份纪要的论证，给出最有依据的下一步行动并解释权衡。"
        ]
        script = Scripted(*[response for _ in questions for response in (blind(), adjudication())])
        report = self.run_review(script, questions)
        self.assertEqual(report["calls_used"], 8)
        for index, question in enumerate(questions):
            self.assertEqual(json.loads(script.calls[2 * index]["messages"][1]["content"])["question"], question)

    def test_reader_disagreement_is_not_automatic_bad_item(self):
        script = Scripted(blind(answer="不知道", answerability="unresolved", evidence=[]), adjudication())
        item = self.run_review(script)["items"][0]
        self.assertEqual(item["item_validity"], "valid")
        self.assertEqual(item["reviewed_answer"], "北溟保险")

    def test_valid_refusal_is_separate_from_item_validity(self):
        script = Scripted(blind(answer="材料不能确定", answerability="unanswerable", evidence=[]),
                          adjudication(answerability="unanswerable", reviewed_answer="材料不能确定",
                                       reference_status="supported", evidence=[], concerns=[]))
        item = self.run_review(script)["items"][0]
        self.assertEqual((item["item_validity"], item["answerability"]), ("valid", "unanswerable"))

    def test_semantic_negative_is_model_decision_not_program_keyword_test(self):
        verdict = adjudication(item_validity="invalid", answerability="unresolved", reviewed_answer=None,
                               reference_status="unsupported", concerns=["题面预设了材料中不存在的关联。"])
        item = self.run_review(Scripted(blind(), verdict))["items"][0]
        self.assertEqual(item["item_validity"], "invalid")
        self.assertEqual(item["review_state"], "pending")

    def test_unlocatable_quote_preserves_unverified_semantic_proposal(self):
        item = self.run_review(Scripted(blind(), adjudication(evidence=evidence("不存在的伪造句子"))))["items"][0]
        self.assertEqual(item["execution"]["status"], "ok")
        self.assertEqual(item["item_validity"], "valid")
        self.assertEqual(item["evidence_location"], {"status": "failed", "failed_stages": ["adjudicate"]})
        self.assertEqual(item["item_certification"]["status"], "unverified")
        self.assertEqual(item["adjudication"]["evidence"], evidence("不存在的伪造句子"))
        self.assertEqual(item["review_state"], "pending")

    def test_failed_blind_quote_continues_full_independent_review_without_repair(self):
        first = blind(evidence=evidence("客户是北溟……欧洲"))
        final = adjudication()
        script = Scripted(first, final)
        report = self.run_review(script)
        item = report["items"][0]
        self.assertEqual(report["calls_used"], 2)
        payload = json.loads(script.calls[1]["messages"][1]["content"])
        self.assertEqual(payload["documents"], report["documents"])
        self.assertEqual(payload["blind_read"], first)
        self.assertEqual(payload["reference_proposal"], "欧洲")
        self.assertEqual(payload["blind_read_evidence_location"]["status"], "failed")
        self.assertEqual(item["blind_read"], first)
        self.assertEqual(item["adjudication"], final)
        self.assertNotIn("location_scope", item["blind_read"]["evidence"][0])
        self.assertEqual(item["stage_resolved_evidence"]["blind_read"], [])
        self.assertEqual(item["stage_evidence_location"]["adjudicate"]["status"], "located")
        self.assertEqual(item["evidence_location"]["failed_stages"], ["blind_read"])
        self.assertEqual(item["semantic"], {"status": "resolved", "basis": "model_proposal",
                                             "correctness_verified": False})
        self.assertEqual(item["execution"]["status"], "ok")
        self.assertEqual(item["review_state"], "pending")
        self.assertEqual(item["item_certification"]["reasons"], ["blind_read:evidence_location_unverified"])
        finished = [r for r in report["records"] if r["event"] == "finished"]
        self.assertEqual([r["output"] for r in finished], [first, final])
        self.assertEqual(finished[0]["evidence_location"]["status"], "failed")

    def test_location_failure_keeps_budget_and_does_not_trigger_retry(self):
        script = Scripted(blind(evidence=evidence("bad quote")))
        item = self.run_review(script, max_calls=1)["items"][0]
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(script.calls[0]["kwargs"]["retries"], 1)
        self.assertEqual(item["execution"]["status"], "call_budget_exhausted")
        self.assertEqual(item["stage_execution"]["blind_read"]["status"], "ok")
        self.assertEqual(item["stage_execution"]["adjudicate"]["status"], "call_budget_exhausted")
        self.assertEqual(item["semantic"]["status"], "proposal_available")
        self.assertEqual(item["evidence_location"]["status"], "failed")
        self.assertIsNone(item["adjudication"])

    def test_transport_shape_and_location_failures_are_distinct(self):
        for output, status in ((RuntimeError("provider failed"), "model_error"),
                               (blind(major_requirements="not a list"), "invalid_review")):
            with self.subTest(status=status):
                script = Scripted(output)
                report = self.run_review(script)
                item = report["items"][0]
                self.assertEqual(len(script.calls), 1)
                self.assertEqual(item["execution"]["status"], status)
                self.assertEqual(item["evidence_location"]["status"], "not_checked")
                self.assertEqual(item["semantic"]["status"], "not_reviewed")
                self.assertEqual(item["item_validity"], "unresolved")
                self.assertEqual(item["review_state"], "pending")
                if status == "invalid_review":
                    self.assertEqual(report["records"][-1]["output"], output)

    def test_original_rationale_error_is_separate_from_correct_core_and_revision(self):
        proposal = {"answer": "北溟保险", "rationale": "客户身份由监管审批确认。"}
        revised = adjudication(reference_status="supported", reviewed_answer="北溟保险",
            reviewed_rationale="正文直接写明客户，未提供监管审批信息。",
            original_answer_review=[{"requirement": "客户是谁", "assessment": "完成",
                                     "explanation": "原答案正确给出客户。"}],
            original_rationale_review={"status": "reviewed", "claims": [{
                "claim": "监管审批确认身份", "assessment": "没有材料支持",
                "explanation": "正文只有客户表述；不能抬高来源权威。"}], "limitations": []},
            review_findings={"substantive_defects": ["理由捏造审批来源，但未改变本题客户身份答案。"],
                             "acceptable_brevity": [], "editorial_suggestions": []},
            concerns=["原理由有来源错误；核心答案仍成立。"], reasoning="分别评价答案任务和理由。")
        item = self.run_review(Scripted(blind(), revised), [
            {"question": "报告客户是谁？", "reference_proposal": proposal}])["items"][0]
        self.assertEqual(item["reference_proposal"], proposal)
        self.assertEqual(item["reference_status"], "supported")
        self.assertEqual(item["reviewed_rationale"], revised["reviewed_rationale"])
        self.assertNotEqual(item["reference_proposal"]["rationale"], item["reviewed_rationale"])
        self.assertEqual(item["original_rationale_review"], revised["original_rationale_review"])
        self.assertEqual(item["review_findings"], revised["review_findings"])
        self.assertEqual(item["review_state"], "completed")
        self.assertFalse(item["semantic"]["correctness_verified"])

    def test_brief_single_document_answer_is_not_programmatically_rejected(self):
        short = adjudication(reference_status="supported", reviewed_answer=None, reviewed_rationale=None,
            major_requirements=["给出客户名字"],
            original_answer_review=[{"requirement": "给出客户名字", "assessment": "已完成",
                                     "explanation": "只报名字已经回答问题。"}],
            review_findings={"substantive_defects": [], "acceptable_brevity": ["无需重述无关市场信息。"],
                             "editorial_suggestions": ["可选地补充正文出处。"]},
            concerns=[], reasoning="简单题和单篇证据不妨碍有效性。")
        item = self.run_review(Scripted(blind(), short), [
            {"question": "客户是谁？", "reference_proposal": {"answer": "北溟保险"}}])["items"][0]
        self.assertEqual(item["review_state"], "completed")
        self.assertEqual(item["item_certification"], {
            "status": "certified", "policy": CERTIFICATION_POLICY, "reasons": []})
        self.assertEqual(review_certification(item), item["item_certification"])
        self.assertEqual(len(item["major_requirements"]), 1)
        self.assertEqual(len(item["evidence"]), 1)

    def test_unfinished_rationale_review_is_explicitly_unverified(self):
        final = adjudication(original_rationale_review={"status": "unresolved", "claims": [],
                                                       "limitations": ["尚未核查原理由。"]})
        item = self.run_review(Scripted(blind(), final))["items"][0]
        self.assertEqual(item["execution"]["status"], "ok")
        self.assertEqual(item["evidence_location"]["status"], "located")
        self.assertEqual(item["reference_status"], "contradicted")
        self.assertEqual(item["review_state"], "pending")
        self.assertIn("original_rationale_review_unresolved", item["item_certification"]["reasons"])

    def test_citations_can_point_to_title_date_session_and_filler(self):
        citations = (evidence("更正通知", "title", "d000002") + evidence("2025-01-09", "date")
                     + evidence("3", "session") + evidence("并不代表分析确认", "content", "d000002"))
        item = self.run_review(Scripted(blind(evidence=citations), adjudication(evidence=citations)), include_titles=True)["items"][0]
        self.assertEqual(item["execution"]["status"], "ok")

    def test_incomplete_scope_remains_pending_without_bad_item_verdict(self):
        item = self.run_review(Scripted(blind(coverage=coverage("partial")), adjudication()))["items"][0]
        self.assertEqual(item["item_validity"], "valid")
        self.assertEqual(item["review_state"], "pending")
        partial = coverage()
        partial["inspected_doc_ids"] = ["nonexistent_document"]
        item = self.run_review(Scripted(blind(coverage=partial)))["items"][0]
        self.assertEqual(item["execution"]["status"], "invalid_review")

    def test_unknown_partial_and_explicit_scope_conflict_stay_pending(self):
        cases = [
            {"status": "unknown", "scope_conflict": False, "limitations": ["Cannot establish inspected scope"]},
            {"status": "partial", "scope_conflict": False, "inspected_doc_ids": ["d000001"], "limitations": ["Read a subset"]},
            {"status": "complete", "scope_conflict": True, "limitations": ["Unresolved conflict in scope claim"]}
        ]
        for claimed in cases:
            with self.subTest(claimed=claimed):
                script = Scripted(blind(coverage=claimed), adjudication())
                item = self.run_review(script)["items"][0]
                self.assertEqual(item["review_state"], "pending")
                self.assertEqual(item["execution"]["status"], "ok")
                self.assertEqual(item["blind_read"]["coverage"], claimed)
                self.assertEqual(len(script.calls), 2)

    def test_full_input_manifest_is_separate_from_eleven_evidence_citations(self):
        documents = [{"content": "A fact recorded here."} for _ in range(108)]
        citations = [{"doc_id": f"d{i:06d}", "field": "content", "quote": "A fact",
                      "role": "support", "explanation": "Relevant fact"} for i in range(1, 12)]
        claimed = {"status": "complete", "scope_conflict": False, "limitations": []}
        script = Scripted(blind(coverage=claimed, evidence=citations),
                          adjudication(coverage=claimed, evidence=citations))
        report = review_questions([{"question": "What is stated?", "gt": "A fact"}], documents, "Read all material",
                                  chat_json=script, reviewer_model="offline")
        self.assertEqual(report["items"][0]["review_state"], "completed")
        self.assertEqual(report["input_manifest"]["document_count"], 108)
        self.assertEqual(len(report["input_manifest"]["documents"]), 108)
        self.assertEqual(len(report["items"][0]["evidence"]), 11)
        self.assertEqual(report["items"][0]["coverage"], claimed)
        self.assertEqual(report["coverage_claim_source"], "model_self_report")
        self.assertEqual(json.loads(script.calls[0]["messages"][1]["content"])["input_scope"]["document_count"], 108)
        changed = deepcopy(documents)
        changed[-1]["content"] = "Different late evidence"
        other = prepare_review([{"question": "What is stated?", "gt": "A fact"}], changed, "Read all material", reviewer_model="offline")
        self.assertNotEqual(report["binding"]["input_manifest_hash"], other["binding"]["input_manifest_hash"])

    def test_partial_scope_requires_explicit_inspected_ids_but_never_full_echo(self):
        bad = {"status": "partial", "scope_conflict": False, "limitations": ["Subset read"]}
        item = self.run_review(Scripted(blind(coverage=bad)))["items"][0]
        self.assertEqual(item["execution"]["status"], "invalid_review")
        complete = {"status": "complete", "scope_conflict": False,
                    "inspected_doc_ids": ["d000001"], "limitations": ["Only key inspected IDs are listed; all documents were read"]}
        item = self.run_review(Scripted(blind(coverage=complete), adjudication(coverage=complete)))["items"][0]
        self.assertEqual(item["review_state"], "completed")

    def test_transport_failure_records_full_error_and_never_relabels_question(self):
        message = "transport failure " + "x" * 1800
        seen = []
        report = self.run_review(Scripted(TimeoutError(message)), record=seen.append)
        item = report["items"][0]
        self.assertEqual(item["execution"]["status"], "model_error")
        self.assertEqual(item["execution"]["message"], message)
        self.assertEqual(item["item_validity"], "unresolved")
        self.assertEqual(report["calls_used"], 1)
        self.assertEqual([x["event"] for x in seen], ["started", "finished"])
        self.assertEqual(seen[-1]["execution"]["message"], message)

    def test_error_envelope_and_non_json_output_remain_operational_failures(self):
        for response, status in [({"__error__": "provider unavailable"}, "model_error"),
                                 ("not json", "invalid_review"), (float("nan"), "invalid_review")]:
            with self.subTest(response=response):
                item = self.run_review(Scripted(response))["items"][0]
                self.assertEqual(item["execution"]["status"], status)
                self.assertEqual(item["item_validity"], "unresolved")

    def test_call_budget_is_global_bounded_and_single_attempt(self):
        script = Scripted(blind())
        report = self.run_review(script, ["题目一", "题目二"], max_calls=1)
        self.assertEqual(report["calls_used"], 1)
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(script.calls[0]["kwargs"]["retries"], 1)
        self.assertTrue(script.calls[0]["kwargs"]["strict_json"])
        self.assertTrue(all(item["execution"]["status"] == "call_budget_exhausted" for item in report["items"]))

    def test_input_limit_does_not_silently_truncate_or_invoke_model(self):
        script = Scripted()
        report = self.run_review(script, max_input_chars=10)
        self.assertEqual(report["calls_used"], 0)
        self.assertEqual(report["items"][0]["execution"]["status"], "input_limit")
        self.assertEqual(len(report["records"][0]["messages"]), 2)
        self.assertIn("之前提到的登记", report["records"][0]["messages"][1]["content"])

    def test_full_records_and_explicit_models_are_retained(self):
        long_reason = "证据说明" * 1000
        seen = []
        script = Scripted(blind(reasoning=long_reason), adjudication(reasoning=long_reason))
        report = self.run_review(script, record=seen.append)
        self.assertEqual(script.calls[0]["kwargs"]["model"], "explicit-reader")
        self.assertEqual(script.calls[1]["kwargs"]["model"], "explicit-reviewer")
        self.assertEqual(report["records"][-1]["output"]["reasoning"], long_reason)
        self.assertEqual(seen, report["records"])
        self.assertEqual(report["publication_effect"], "none")

    def test_sink_failure_stops_before_model_call(self):
        script = Scripted()
        def broken(_):
            raise OSError("Cannot persist audit")
        with self.assertRaises(OSError):
            self.run_review(script, record=broken)
        self.assertEqual(script.calls, [])

    def test_version_binding_changes_with_question_reference_corpus_protocol(self):
        def prepare(q="客户是谁？", gold="欧洲", corpus=None, protocol=None):
            return prepare_review([{"question": q, "gold": gold}], corpus or CORPUS,
                                  protocol or PROTOCOL, reviewer_model="reviewer", include_titles=True)["items"][0]["binding"]
        original = prepare()
        self.assertNotEqual(original["question_hash"], prepare(q="客户在哪里？")["question_hash"])
        self.assertNotEqual(original["reference_hash"], prepare(gold="北溟保险")["reference_hash"])
        modified = deepcopy(CORPUS)
        modified["sessions"][0]["docs"][1]["title"] = "新的标题"
        self.assertNotEqual(original["corpus_hash"], prepare(corpus=modified)["corpus_hash"])
        self.assertNotEqual(original["protocol_hash"], prepare(protocol="新公开范围")["protocol_hash"])

    def test_input_not_mutated_and_reference_private_keys_stripped(self):
        corpus, questions = deepcopy(CORPUS), [{"question": "客户是谁？", "reference_proposal": {
            "answer": "北溟保险", "rationale": "仅为作者提案", "world": "PRIVATE_REFERENCE_WORLD"}}]
        before = deepcopy((corpus, questions))
        report = prepare_review(questions, corpus, PROTOCOL, reviewer_model="reviewer")
        self.assertEqual((corpus, questions), before)
        self.assertEqual(set(report["items"][0]["reference_proposal"]), {"answer", "rationale"})

    def test_import_has_no_config_or_legacy_gate_side_effects(self):
        code = "import sys; import pipeline.semantic_review; print([m for m in ('config','pipeline.grounding','pipeline.question_contract','pipeline.lines') if m in sys.modules])"
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(proc.stdout.strip(), "[]")

    def test_cli_prepares_without_network_and_refuses_overwrite_or_run_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = {"questions": ["客户是谁？"], "corpus": CORPUS, "protocol": PROTOCOL}
            for name, payload in paths.items():
                (directory / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = directory / "review.json"
            args = ["--questions", str(directory / "questions.json"), "--corpus", str(directory / "corpus.json"),
                    "--protocol", str(directory / "protocol.json"), "--output", str(output), "--reviewer-model", "offline"]
            self.assertEqual(main(args), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual((report["mode"], report["calls_used"]), ("prepared", 0))
            with self.assertRaises(SystemExit):
                main(args)
            with self.assertRaises(SystemExit):
                main(args + ["--execute"])
            with self.assertRaises(ValueError):
                _new_output(ROOT / "output/runs/anything/new-review.json", [])

    def test_cli_about_protocol_matches_solver_public_text_only(self):
        from eval.provenance import load_public_protocol
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "questions.json").write_text(json.dumps(["题目"]), encoding="utf-8")
            (directory / "corpus.json").write_text(json.dumps(CORPUS), encoding="utf-8")
            about = {"answer_protocol": {"rules": ["只用公开材料"]}, "private_gold": "PRIVATE_ABOUT_SENTINEL"}
            (directory / "about.json").write_text(json.dumps(about), encoding="utf-8")
            output = directory / "review.json"
            self.assertEqual(main(["--questions", str(directory / "questions.json"),
                                   "--corpus", str(directory / "corpus.json"),
                                   "--protocol", str(directory / "about.json"),
                                   "--output", str(output), "--reviewer-model", "offline"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["public_protocol"], load_public_protocol(directory / "about.json"))
            self.assertNotIn("PRIVATE_ABOUT", report["public_protocol"])

    def test_cli_execute_records_layers_with_stub_provider_and_redacts_credentials(self):
        from llm_trace import emit
        responses = [blind(), adjudication(reasoning="valid explanation configured-secret")]
        def provider(messages, **params):
            response = responses.pop(0)
            emit("mock_provider_response", secrets=("configured-secret",), response=response)
            return response
        config = SimpleNamespace(chat_json=provider, _trace_secrets=lambda: ("configured-secret",))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"config": config}):
            directory = Path(tmp)
            for name, payload in {"questions": ["客户是谁？"], "corpus": CORPUS, "protocol": PROTOCOL}.items():
                (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            output = directory / "review.json"
            self.assertEqual(main(["--questions", str(directory / "questions.json"),
                                   "--corpus", str(directory / "corpus.json"),
                                   "--protocol", str(directory / "protocol.json"),
                                   "--output", str(output), "--reviewer-model", "stub",
                                   "--execute", "--max-calls", "2"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["calls_used"], 2)
            records = output.with_name(output.name + ".calls.jsonl").read_text(encoding="utf-8")
            attempts = output.with_name(output.name + ".attempts.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(records.splitlines()), 4)
            self.assertEqual(len(attempts.splitlines()), 2)
            self.assertNotIn("configured-secret", records + attempts + output.read_text(encoding="utf-8"))


ISOLATED_QUESTION = {"qid": "research-q", "question": "报告的客户是谁？", "reference_proposal": {
    "answer": "欧洲", "rationale": "欧洲是客户。"}}
ISOLATED_PROTOCOL = "仅根据公开材料回答实际题意。"


def reference_audit_response(**overrides):
    value = {"decision": "revise", "reason": "原提案把市场当成客户。",
        "task_requirements": ["给出客户身份"], "task_coverage": "回答了另一对象，需改正。",
        "claims": [{"reference_part": "answer", "source_quote": "欧洲", "assessment": "不成立",
                    "explanation": "正文区分客户与市场。", "evidence": evidence()}],
        "substantive_defects": ["客户与市场混淆"], "acceptable_brevity": [],
        "editorial_suggestions": [], "suggested_revision": "改为北溟保险。", "limitations": [],
        "evidence": evidence()}
    value.update(overrides)
    return value


def isolated_adjudication(**overrides):
    value = adjudication()
    value["reference_audit_response"] = "复核了原文和审计意见：欧洲表示市场，参考确实答错对象。"
    value["original_answer_review"][0].update(target_scope="quoted_text",
        reference_targets=[{"reference_part": "answer", "source_quote": "欧洲"}])
    value["original_rationale_review"] = {"status": "reviewed", "claims": [{
        "claim": "欧洲是客户", "assessment": "不成立", "explanation": "正文明确区分客户与市场。",
        "target_scope": "quoted_text", "reference_targets": [{"reference_part": "rationale",
                                                               "source_quote": "欧洲是客户。"}]}], "limitations": []}
    value.update(overrides)
    return value


class IsolatedReferenceTests(unittest.TestCase):
    def run_review(self, script=None, questions=None, **kwargs):
        return review_questions(questions or [deepcopy(ISOLATED_QUESTION)], CORPUS, ISOLATED_PROTOCOL,
            chat_json=script or Scripted(blind(), reference_audit_response(), isolated_adjudication()),
            reviewer_model="final-model", reader_model="blind-model", reference_auditor_model="audit-model", **kwargs)

    def test_three_fresh_contexts_target_isolation_and_exact_standalone_audit(self):
        from pipeline.reference_audit import audit_reference
        secret_blind = blind(answer="BLIND_ONLY_SENTINEL", reasoning="BLIND_REASON_SENTINEL")
        script = Scripted(secret_blind, reference_audit_response(), isolated_adjudication())
        before = deepcopy(ISOLATED_QUESTION), deepcopy(CORPUS)
        report = self.run_review(script)
        item = report["items"][0]
        self.assertEqual((ISOLATED_QUESTION, CORPUS), before)
        self.assertEqual(report["version"], ISOLATED_VERSION)
        self.assertEqual(report["calls_used"], 3)
        self.assertEqual([call["kwargs"]["model"] for call in script.calls],
                         ["blind-model", "audit-model", "final-model"])
        for call in script.calls:
            self.assertEqual(call["kwargs"]["retries"], 1)
            self.assertEqual(call["kwargs"]["max_tokens"], 4096)
            self.assertEqual(len(call["messages"]), 2)
        audit_input = json.loads(script.calls[1]["messages"][1]["content"])
        self.assertNotIn("BLIND_", json.dumps(audit_input))
        self.assertNotIn("PRIVATE_", json.dumps(audit_input))
        self.assertEqual(audit_input["original_reference"], ISOLATED_QUESTION["reference_proposal"])
        standalone = Scripted(reference_audit_response())
        audit_reference(ISOLATED_QUESTION, CORPUS, ISOLATED_PROTOCOL, model="audit-model",
                        chat_json=standalone, max_tokens=4096)
        self.assertEqual(script.calls[1], standalone.calls[0])
        final_input = json.loads(script.calls[2]["messages"][1]["content"])
        self.assertEqual(final_input["blind_read"], secret_blind)
        self.assertEqual(final_input["reference_audit"]["raw_output"], reference_audit_response())
        self.assertNotIn("messages", final_input["reference_audit"])
        self.assertNotIn("resolved_evidence", final_input["reference_audit"]["claim_locations"][0])
        self.assertEqual(item["review_state"], "completed")
        self.assertEqual(item["item_certification"]["policy"], ISOLATED_CERTIFICATION_POLICY)
        validated = validate_isolated_completed(item, report["documents"])
        self.assertEqual(validated["evidence_location"], item["stage_evidence_location"]["reference_audit"])

    def test_unsupported_input_modes_fail_before_provider(self):
        rows = [["question"], [{"question": "question", "gold": "answer"}],
                [{"question": "question", "reference_proposal": {"answer": "answer"}}],
                [{"question": "question", "reference_proposal": {"answer": "INSUFFICIENT_EVIDENCE", "rationale": ""},
                  "question_contract": {"answer_kind": "abstention"}}]]
        script = Scripted()
        for questions in rows:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                self.run_review(script, questions)
        for kwargs in ({"include_titles": True}, {"public_protocol": PROTOCOL}, {"reference_auditor_model": ""}):
            params = dict(reviewer_model="r", reference_auditor_model="a", public_protocol=ISOLATED_PROTOCOL)
            params.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_review([ISOLATED_QUESTION], CORPUS, **params)
        self.assertEqual(script.calls, [])

    def test_binding_changes_with_auditor_identity_and_prompt(self):
        from pipeline import reference_audit
        def prepared(model="audit-model"):
            return prepare_review([ISOLATED_QUESTION], CORPUS, ISOLATED_PROTOCOL,
                reviewer_model="r", reference_auditor_model=model)["binding"]
        original = prepared()
        self.assertNotEqual(original, prepared("other-model"))
        with patch.object(reference_audit, "SYSTEM", reference_audit.SYSTEM + " New review instruction"):
            changed = prepared()
        self.assertNotEqual(original["contract_hash"], changed["contract_hash"])
        self.assertNotEqual(original["reference_audit_prompt_hash"], changed["reference_audit_prompt_hash"])

    def test_global_budget_covers_all_three_roles_and_multiple_items(self):
        for limit in range(5):
            with self.subTest(limit=limit):
                script = Scripted(blind(), reference_audit_response(), isolated_adjudication(), blind())
                questions = [deepcopy(ISOLATED_QUESTION), {**deepcopy(ISOLATED_QUESTION), "qid": "second"}]
                report = self.run_review(script, questions, max_calls=limit)
                self.assertEqual(report["calls_used"], limit)
                self.assertEqual(len(script.calls), limit)
                self.assertEqual(report["items"][1]["review_state"], "pending")
                if limit < 3:
                    self.assertEqual(report["items"][0]["review_state"], "pending")
                self.assertEqual(report["items"][1]["reference_audit"]["execution"]["status"], "call_budget_exhausted")

    def test_input_cap_never_truncates_or_spends_hidden_call(self):
        script = Scripted()
        report = self.run_review(script, max_input_chars=10)
        self.assertEqual(report["calls_used"], 0)
        self.assertEqual(len(report["records"]), 3)
        self.assertTrue(all(event["event"] == "skipped" for event in report["records"]))
        for event in report["records"]:
            self.assertIn("报告客户是北溟保险", event["messages"][1]["content"])
        self.assertEqual(report["items"][0]["reference_audit"]["execution"]["status"], "input_limit")

    def test_auditor_revise_is_not_a_python_semantic_veto(self):
        final = isolated_adjudication(reference_status="supported",
            reference_audit_response="审计忽略了题意允许市场简称，我认为该简称在这道题可以成立。")
        item = self.run_review(Scripted(blind(), reference_audit_response(), final))["items"][0]
        # This deliberately fallible fixture checks authority, not that this defense is true.
        self.assertEqual(item["review_state"], "completed")
        self.assertEqual(item["reference_status"], "supported")
        self.assertFalse(item["semantic"]["correctness_verified"])

    def test_unresolved_audit_stays_pending_despite_final_confidence(self):
        report = self.run_review(Scripted(blind(), reference_audit_response(decision="unresolved"), isolated_adjudication()))
        item = report["items"][0]
        self.assertEqual(item["execution"]["status"], "ok")
        self.assertEqual(item["review_state"], "pending")
        self.assertIn("reference_audit:semantic_unresolved", item["item_certification"]["reasons"])
        with self.assertRaises(ValueError):
            validate_isolated_completed(item, report["documents"])

    def test_failed_audit_quote_is_not_repaired_or_hidden_from_final_agent(self):
        for failure in ("source", "evidence"):
            raw = reference_audit_response()
            if failure == "source":
                raw["claims"][0]["source_quote"] = "BLIND_ONLY_SENTINEL"
            else:
                raw["claims"][0]["evidence"] = evidence("nonexistent quote")
            script = Scripted(blind(answer="BLIND_ONLY_SENTINEL"), raw, isolated_adjudication())
            report = self.run_review(script)
            with self.subTest(failure=failure):
                item = report["items"][0]
                self.assertEqual(item["reference_audit"]["raw_output"], raw)
                self.assertEqual(item["stage_evidence_location"]["reference_audit"]["status"], "failed")
                self.assertEqual(item["review_state"], "pending")
                self.assertEqual(item["reference_status"], "contradicted")
                notice = json.loads(script.calls[2]["messages"][1]["content"])["reference_audit"]
                self.assertEqual(notice["raw_output"], raw)
                self.assertEqual(notice["execution"]["status"], "invalid_output")
                self.assertFalse(notice["proposal_ready"])
                with self.assertRaises(ValueError):
                    validate_isolated_completed(item, report["documents"])

    def test_transport_failure_is_visible_and_cannot_certify(self):
        script = Scripted(blind(), TimeoutError("offline transport failed"), isolated_adjudication())
        report = self.run_review(script)
        item = report["items"][0]
        self.assertEqual(report["calls_used"], 3)
        self.assertEqual(item["stage_execution"]["reference_audit"]["status"], "model_error")
        self.assertEqual(item["item_validity"], "valid")
        self.assertEqual(item["review_state"], "pending")
        self.assertIn("offline transport failed", script.calls[2]["messages"][1]["content"])

    def test_blind_failure_does_not_leak_or_block_independent_reference_audit(self):
        script = Scripted("unreadable blind", reference_audit_response(), isolated_adjudication())
        item = self.run_review(script)["items"][0]
        self.assertEqual(item["review_state"], "pending")
        self.assertEqual(item["stage_execution"]["reference_audit"]["status"], "ok")
        self.assertNotIn("unreadable blind", script.calls[1]["messages"][1]["content"])
        self.assertIn("unreadable blind", script.calls[2]["messages"][1]["content"])

    def test_target_attribution_rejects_blind_text_and_wrong_reference_part(self):
        for failure in ("blind_quote", "wrong_part", "missing_target", "missing_response"):
            final = isolated_adjudication()
            target = final["original_answer_review"][0]["reference_targets"][0]
            if failure == "blind_quote": target["source_quote"] = "北溟保险"
            if failure == "wrong_part": target["reference_part"] = "rationale"
            if failure == "missing_target": final["original_answer_review"][0]["reference_targets"] = []
            if failure == "missing_response": final.pop("reference_audit_response")
            with self.subTest(failure=failure):
                report = self.run_review(Scripted(blind(), reference_audit_response(), final))
                item = report["items"][0]
                self.assertEqual(item["execution"]["status"], "invalid_review")
                self.assertEqual(item["adjudicate_raw_output"], final)
                self.assertIsNone(item["adjudication"])
                self.assertEqual(item["review_state"], "pending")

    def test_omission_and_duplicate_spans_do_not_invent_unique_quote(self):
        question = deepcopy(ISOLATED_QUESTION)
        question["reference_proposal"]["answer"] = "欧洲，欧洲"
        final = isolated_adjudication()
        final["original_answer_review"].append({"requirement": "另一可能任务", "assessment": "遗漏",
            "explanation": "如果按该解释，还没有明确指出客户。", "target_scope": "omission", "reference_targets": []})
        report = self.run_review(Scripted(blind(), reference_audit_response(), final), [question])
        item = report["items"][0]
        self.assertEqual(item["review_state"], "completed")
        target = item["reference_target_locations"][0]
        self.assertEqual(target["location_status"], "ambiguous")
        self.assertEqual([{k: span[k] for k in ("start", "end")} for span in target["matching_spans"]],
                         [{"start": 0, "end": 2}, {"start": 3, "end": 5}])
        self.assertTrue(all(span["json_pointer"] == "" and span["source_kind"] == "string_value"
                            for span in target["matching_spans"]))
        audit_target = item["reference_audit"]["claim_locations"][0]
        self.assertNotIn("start", audit_target)
        self.assertEqual(audit_target["matching_spans"], target["matching_spans"])
        validate_isolated_completed(item, report["documents"])

    def test_no_claim_count_is_treated_as_semantic_coverage(self):
        report = self.run_review(Scripted(blind(), reference_audit_response(claims=[]), isolated_adjudication()))
        self.assertEqual(report["items"][0]["review_state"], "completed")
        validate_isolated_completed(report["items"][0], report["documents"])

    def test_nonempty_original_rationale_cannot_be_reported_absent(self):
        final = isolated_adjudication(original_rationale_review={
            "status": "not_provided", "claims": [], "limitations": []})
        report = self.run_review(Scripted(blind(), reference_audit_response(), final))
        item = report["items"][0]
        self.assertEqual(item["review_state"], "pending")
        self.assertEqual(item["execution"]["status"], "invalid_review")
        self.assertEqual(item["adjudicate_raw_output"], final)
        self.assertIsNone(item["adjudication"])
        # The program checks existence, never imposes a claim-count threshold.
        reviewed_empty = isolated_adjudication(original_rationale_review={
            "status": "reviewed", "claims": [], "limitations": []})
        self.assertEqual(self.run_review(Scripted(blind(), reference_audit_response(), reviewed_empty))
                         ["items"][0]["review_state"], "completed")

    def test_actually_empty_rationale_may_be_not_provided(self):
        for rationale in ("", " \n\t "):
            question = deepcopy(ISOLATED_QUESTION)
            question["reference_proposal"]["rationale"] = rationale
            final = isolated_adjudication(original_rationale_review={
                "status": "not_provided", "claims": [], "limitations": []})
            report = self.run_review(Scripted(blind(), reference_audit_response(), final), [question])
            with self.subTest(rationale=rationale):
                self.assertEqual(report["items"][0]["review_state"], "completed")
                validate_isolated_completed(report["items"][0], report["documents"])

    def test_audit_sink_failure_stops_calls_and_retains_complete_raw(self):
        for failure_event, expected_calls in (("started", 1), ("finished", 2)):
            script = Scripted(blind(), reference_audit_response(), isolated_adjudication())
            def record(event):
                if event["stage"] == "reference_audit" and event["event"] == failure_event:
                    raise OSError("Cannot save reference audit")
            with self.subTest(event=failure_event), self.assertRaises(SemanticReviewAuditError) as caught:
                self.run_review(script, record=record)
            report = caught.exception.report
            self.assertEqual(len(script.calls), expected_calls)
            self.assertEqual(report["calls_used"], expected_calls)
            item = report["items"][0]
            self.assertEqual(item["review_state"], "pending")
            self.assertEqual(item["reference_audit"]["execution"]["status"], "audit_error")
            if failure_event == "finished":
                self.assertEqual(item["reference_audit"]["raw_output"], reference_audit_response())
            self.assertIsNone(item["adjudication"])

    def test_completed_validator_rejects_audit_and_target_tampering(self):
        report = self.run_review()
        def alter_payload(item): item["reference_audit"]["inputs"]["question"] = "other question"
        def alter_raw(item): item["reference_audit"]["raw_output"]["claims"][0]["source_quote"] = "北溟保险"
        def alter_ref(item): item["reference_proposal"]["answer"] = "北溟保险"
        def alter_target(item): item["adjudication"]["original_answer_review"][0]["reference_targets"][0]["source_quote"] = "北溟保险"
        def alter_location(item): item["reference_audit"]["claim_locations"][0]["matching_spans"] = []
        def alter_protocol(item): item["reference_audit_public_protocol"] = "another protocol"
        def alter_reply(item): item["reference_audit_response"] = "different summary of the audit response"
        def alter_source(item): item["reference_audit"]["reference_audit_implementation_hash"] = "other implementation"
        for mutate in (alter_payload, alter_raw, alter_ref, alter_target, alter_location, alter_protocol,
                       alter_reply, alter_source):
            item = deepcopy(report["items"][0])
            mutate(item)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ValueError):
                validate_isolated_completed(item, report["documents"])


class CitationResolutionTests(unittest.TestCase):
    def test_strict_and_readable_interfaces_have_distinct_authority(self):
        docs, _ = visible_documents(CORPUS)
        bad_quote = blind(evidence=evidence("报告客户是……保险"))
        before = deepcopy(bad_quote)
        _validate_readable_response(bad_quote, "blind_read", docs, require_analysis=True)
        with self.assertRaisesRegex(ValueError, "verbatim"):
            _validate_response(bad_quote, "blind_read", docs, require_analysis=True)
        self.assertEqual(bad_quote, before)
        legacy = blind()
        del legacy["major_requirements"]
        _validate_response(legacy, "blind_read", docs)
        with self.assertRaisesRegex(ValueError, "major_requirements"):
            _validate_response(legacy, "blind_read", docs, require_analysis=True)

    def test_per_entry_location_failures_preserve_other_successes_and_original_input(self):
        docs, _ = visible_documents(CORPUS)
        citations = evidence("wrong quote") + evidence() + [None] + [{
            "doc_id": "d000002", "field": "content", "location_scope": "field", "role": "counterevidence",
            "explanation": "登记不能直接证明分析确认。"}] + evidence(doc_id="missing")
        before = deepcopy(citations)
        result = locate_citations(citations, docs)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([e["status"] for e in result["entries"]],
                         ["failed", "located", "failed", "located", "failed"])
        self.assertEqual([e["index"] for e in result["entries"]], list(range(5)))
        self.assertEqual([e["locator_source"] for e in result["resolved_evidence"]],
                         ["model_verbatim_quote", "program_resolved_field"])
        self.assertEqual(citations, before)
        self.assertNotIn("resolved", result["entries"][0])
        with self.assertRaises(ValueError):
            resolve_citations(citations, docs)

    def test_location_audit_does_not_treat_invalid_document_input_as_model_error(self):
        docs, _ = visible_documents(CORPUS)
        with self.assertRaisesRegex(ValueError, "unique"):
            locate_citations(evidence(), [docs[0], docs[0]])
        result = locate_citations([], docs)
        self.assertEqual(result, {"status": "located", "entries": [], "resolved_evidence": []})

    def setUp(self):
        self.docs, _ = visible_documents(CORPUS)
        self.field = {"doc_id": "d000001", "field": "content", "location_scope": "field",
                      "role": "support", "explanation": "The claim concerns this visible record."}

    def test_explicit_field_resolves_original_text_without_mutating_model_input(self):
        raw = [deepcopy(self.field)]
        before = deepcopy(raw), deepcopy(self.docs)
        located = resolve_citations(raw, self.docs)
        self.assertEqual((raw, self.docs), before)
        self.assertEqual(located[0]["resolved_text"], self.docs[0]["content"])
        self.assertEqual(located[0]["locator_source"], "program_resolved_field")
        self.assertNotIn("quote", raw[0])
        self.assertNotIn("supports", located[0])
        self.assertNotIn("verified_truth", located[0])

    def test_both_explicit_and_legacy_verbatim_quotes_remain_exact(self):
        legacy = evidence()
        explicit = [{**legacy[0], "location_scope": "quote"}]
        self.assertEqual(resolve_citations(legacy, self.docs), resolve_citations(explicit, self.docs))
        self.assertEqual(resolve_citations(legacy, self.docs)[0]["locator_source"], "model_verbatim_quote")
        for scope in (None, "quote", "field"):
            bad = {**legacy[0], "quote": "报告客户...欧洲"}
            if scope is not None:
                bad["location_scope"] = scope
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                resolve_citations([bad], self.docs)

    def test_field_mode_rejects_any_model_quote_and_forged_program_fields(self):
        for extra in ({"quote": None}, {"quote": self.docs[0]["content"]},
                      {"locator_source": "program_resolved_field"}, {"resolved_text": "fake"},
                      {"document_hash": "fake"}, {"field_hash": "fake"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                resolve_citations([{**self.field, **extra}], self.docs)

    def test_only_actual_visible_ids_fields_and_unique_locations_can_resolve(self):
        for changes in ({"doc_id": "PRIVATE_SIGNAL_ID"}, {"doc_id": "unknown"},
                        {"field": "title"}, {"field": "refs"}, {"location_scope": "fuzzy"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                resolve_citations([{**self.field, **changes}], self.docs)
        with self.assertRaises(ValueError):
            resolve_citations([self.field], [self.docs[0], deepcopy(self.docs[0])])
        titled, _ = visible_documents(CORPUS, include_titles=True)
        self.assertEqual(resolve_citations([{**self.field, "field": "title"}], titled)[0]["resolved_text"], "欧洲客户记录")
        self.assertEqual(resolve_citations([{**self.field, "field": "session"}], self.docs)[0]["resolved_text"], "3")

    def test_resolution_changes_when_bound_document_changes(self):
        original = resolve_citations([self.field], self.docs)[0]
        changed = deepcopy(self.docs)
        changed[0]["content"] = "Different evidence, potentially counterevidence."
        other = resolve_citations([self.field], changed)[0]
        self.assertNotEqual(original["document_hash"], other["document_hash"])
        self.assertNotEqual(original["field_hash"], other["field_hash"])
        self.assertEqual(other["resolved_text"], changed[0]["content"])

    def test_review_stores_raw_and_resolved_stages_separately_without_upgrading_truth(self):
        raw_blind, raw_decision = blind(evidence=[self.field]), adjudication(evidence=[self.field])
        report = review_questions([{"question": "客户是谁？", "gold": "欧洲"}], CORPUS, PROTOCOL,
            reviewer_model="offline", chat_json=Scripted(raw_blind, raw_decision))
        item = report["items"][0]
        self.assertEqual(report["version"], "semantic-question-shadow/v6")
        self.assertEqual(report["binding"]["citation_policy"], CITATION_POLICY)
        self.assertEqual(item["blind_read"], raw_blind)
        self.assertEqual(item["adjudication"], raw_decision)
        self.assertEqual(item["reference_status"], "contradicted")
        self.assertEqual(item["resolved_evidence"], resolve_citations([self.field], self.docs))
        self.assertEqual(set(item["stage_resolved_evidence"]), {"blind_read", "adjudicate"})
        finished = [event for event in report["records"] if event["event"] == "finished"]
        self.assertEqual(finished[0]["output"], raw_blind)
        self.assertEqual(finished[0]["resolved_evidence"], item["stage_resolved_evidence"]["blind_read"])

    def test_field_locator_does_not_hide_partial_scope_or_resolve_ambiguous_semantics(self):
        report = review_questions([{"question": "客户是谁？", "gold": "欧洲"}], CORPUS, PROTOCOL,
            reviewer_model="offline", chat_json=Scripted(blind(evidence=[self.field], coverage=coverage("partial")),
                adjudication(evidence=[self.field], item_validity="ambiguous")))
        self.assertEqual(report["items"][0]["review_state"], "pending")
        self.assertEqual(report["items"][0]["item_validity"], "ambiguous")


if __name__ == "__main__":
    unittest.main(verbosity=2)
