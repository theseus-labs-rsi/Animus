"""Offline invariants for bounded public review and revision proposals."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.agent_editing import (AuditRecordError, propose_material_revision,
                                    propose_question_revisions, review_materials, VERSION)
from pipeline.semantic_review import visible_documents


CORPUS = {"corpus": {"sessions": [
    {"session_id": 5, "date": "2025-01-08", "docs": [
        {"doc_id": "later", "title": "HIDDEN_TITLE", "content": "采用新材料仍需会签。",
         "quality_review": {"status": "passed"}, "is_filler": False}]},
    {"session_id": 1, "date": "2025-01-06", "docs": [
        {"doc_id": "earlier", "title": "内部标题", "content": "新材料已收到，尚未替换原依据。",
         "private": "HIDDEN_WRITER_WORLD"}]}]}, "release": {"eligible": True}}
PROTOCOL = "依据公开档案说明能够与不能确定的事项。"
QUESTIONS = [
    {"qid": "a", "question": "截至1月8日是否采用了新材料？",
     "reference_proposal": {"answer": "没有采用。", "rationale": "暂无会签记录。"},
     "quality_review": {"status": "passed"}, "aux": {"private": "HIDDEN_Q_WORLD"}},
    {"qid": "b", "question": "何时收到新材料？",
     "reference_proposal": {"answer": "1月6日已有收到记录。", "rationale": "档案第一条。"}}]
ISSUES = [{"issue_id": "i1", "qid": "a", "description": "未记录不能证明未发生。",
           "evidence": [{"doc_id": "d000001", "field": "content", "location_scope": "field",
                         "role": "support", "explanation": "原文保留采用状态的不确定性。"}],
           "gold": "HIDDEN_ISSUE_GOLD"}]
RESPONSES = [{"issue_id": "i1", "disposition": "addressed", "reason": "保留公开依据无法确定的范围。"}]


class Scripted:
    def __init__(self, value):
        self.value, self.calls = value, []

    def __call__(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": params})
        if isinstance(self.value, Exception):
            raise self.value
        return deepcopy(self.value)


def question_output():
    return {"decisions": [
        {"qid": "a", "action": "revise", "reason": "原参考把沉默写成未发生。",
         "reference_proposal": {"answer": "不能确定是否已采用。", "rationale": "只记录需会签，未见完成信息。"}},
        {"qid": "b", "action": "keep", "reason": "原参考有公开依据。"}],
        "issue_responses": deepcopy(RESPONSES)}


def material_output():
    documents, _ = visible_documents(CORPUS)
    return {"action": "revise", "reason": "澄清发言的证据范围。",
            "document_edits": [{"doc_id": documents[1]["doc_id"],
                                "content": "采用新材料仍需会签；本条不说明会签是否已经完成。"}],
            "issue_responses": deepcopy(RESPONSES)}


class EditingTests(unittest.TestCase):
    def questions(self, value=None, **kwargs):
        script = Scripted(question_output() if value is None else value)
        report = propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
            model="offline", chat_json=script, **kwargs)
        return report, script

    def materials(self, value=None, **kwargs):
        script = Scripted(material_output() if value is None else value)
        report = propose_material_revision(CORPUS, PROTOCOL, ISSUES,
            model="offline", chat_json=script, **kwargs)
        return report, script

    def test_question_reference_only_revision_preserves_identity_and_original(self):
        before = deepcopy((QUESTIONS, CORPUS, ISSUES))
        report, script = self.questions()
        self.assertEqual((QUESTIONS, CORPUS, ISSUES), before)
        self.assertEqual(report["execution"]["status"], "ok")
        self.assertTrue(report["proposal_ready"])
        self.assertTrue(report["candidate_ready"])
        self.assertEqual(report["candidate_validation"]["status"], "valid")
        self.assertEqual(report["editorial_completion"]["status"], "complete")
        self.assertTrue(report["editorial_completion"]["complete"])
        self.assertTrue(report["editorial_completion"]["issue_responses_complete"])
        self.assertEqual(report["proposed_questions"][0]["qid"], "a")
        self.assertEqual(report["proposed_questions"][0]["question"], QUESTIONS[0]["question"])
        self.assertNotEqual(report["variants"][0]["before_hash"], report["variants"][0]["after_hash"])
        self.assertEqual(report["quantity"]["new_question_count"], 0)
        self.assertEqual(report["original"], QUESTIONS)
        self.assertEqual(script.calls[0]["params"]["retries"], 1)
        self.assertEqual(report["calls_used"], 1)
        self.assertEqual(report["publication_effect"], "none")

    def test_fresh_question_projection_cannot_inherit_approval_or_private_world(self):
        report, script = self.questions()
        wire = script.calls[0]["messages"][1]["content"]
        for secret in ("HIDDEN_Q_WORLD", "HIDDEN_WRITER_WORLD", "HIDDEN_ISSUE_GOLD", "HIDDEN_TITLE"):
            self.assertNotIn(secret, wire)
        for q in report["proposed_questions"]:
            self.assertEqual(set(q), {"qid", "question", "reference_proposal"})
        self.assertFalse(report["variants"][0]["quality_approval"])

    def test_missing_duplicate_or_unknown_qid_invalidates_whole_batch(self):
        for mode in ("missing", "duplicate", "unknown"):
            with self.subTest(mode=mode):
                output = question_output()
                if mode == "missing": output["decisions"].pop()
                elif mode == "duplicate": output["decisions"].append(deepcopy(output["decisions"][0]))
                else: output["decisions"][1]["qid"] = "invented"
                report, _ = self.questions(output)
                self.assertEqual(report["execution"]["status"], "invalid_output")
                self.assertFalse(report["proposal_ready"])
                self.assertFalse(report["candidate_ready"])
                self.assertEqual(report["proposed_questions"], [])
                self.assertEqual(report["variants"], [])
                self.assertEqual(report["raw_output"], output)

    def test_each_issue_has_exactly_one_response(self):
        for values in ([], RESPONSES + RESPONSES,
                       [{**RESPONSES[0], "issue_id": "invented"}]):
            output = question_output()
            output["issue_responses"] = values
            report, _ = self.questions(output)
            self.assertFalse(report["proposal_ready"])
            self.assertIn("issue_response_identity_mismatch", report["format_issues"])

    def test_keep_drop_unresolved_cannot_smuggle_new_reference(self):
        for action in ("keep", "drop", "unresolved"):
            output = question_output()
            output["decisions"][0]["action"] = action
            report, _ = self.questions(output)
            self.assertFalse(report["proposal_ready"])

    def test_drop_and_unresolved_are_explicit_dispositions_not_new_items(self):
        output = {"decisions": [{"qid": "a", "action": "drop", "reason": "现有材料不能支持该问法。"},
                                  {"qid": "b", "action": "unresolved", "reason": "需要进一步核查。"}],
                  "issue_responses": [{"issue_id": "i1", "disposition": "deferred", "reason": "不能可靠修复。"}]}
        report, _ = self.questions(output)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposed_questions"], [])
        self.assertEqual(len(report["decisions"]), 2)
        self.assertEqual(report["quantity"]["by_action"]["unresolved"], 1)

    def test_author_can_dispute_criticism_without_inventing_a_change(self):
        output = {"action": "no_change", "reason": "这是合理角色意见差异。", "document_edits": [],
                  "issue_responses": [{"issue_id": "i1", "disposition": "disputed", "reason": "原文只是计划。"}]}
        report, _ = self.materials(output)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposal_state"], "no_change")
        self.assertIsNone(report["proposed_corpus"])
        self.assertEqual(report["variants"], [])

    def test_date_or_title_repair_outside_scope_can_remain_explicitly_unresolved(self):
        value = {"action": "unresolved", "reason": "需调整记录日期，本次只能改正文，不能可靠修复。",
                 "document_edits": [],
                 "issue_responses": [{"issue_id": "i1", "disposition": "deferred", "reason": "日期范围调整需派生新案例。"}]}
        report, _ = self.materials(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposal_state"], "unresolved")
        self.assertIsNone(report["proposed_corpus"])
        self.assertEqual(report["original"], CORPUS)

    def test_material_revision_preserves_source_order_ids_dates_titles_without_old_review(self):
        before = deepcopy(CORPUS)
        report, script = self.materials()
        self.assertEqual(CORPUS, before)
        self.assertTrue(report["proposal_ready"])
        sessions = report["proposed_corpus"]["corpus"]["sessions"]
        self.assertEqual([s["session_id"] for s in sessions], [5, 1])
        self.assertEqual(sessions[0]["date"], "2025-01-08")
        self.assertEqual(sessions[0]["docs"][0]["doc_id"], "later")
        self.assertEqual(sessions[0]["docs"][0]["title"], "HIDDEN_TITLE")
        self.assertEqual(sessions[0]["docs"][0]["content"], material_output()["document_edits"][0]["content"])
        self.assertEqual(sessions[1]["docs"][0]["content"], before["corpus"]["sessions"][1]["docs"][0]["content"])
        self.assertNotIn("quality_review", json.dumps(report["proposed_corpus"]))
        self.assertNotIn("eligible", json.dumps(report["proposed_corpus"]))
        payload = json.loads(script.calls[0]["messages"][1]["content"])
        self.assertNotIn("qid", payload["issues"][0])
        self.assertNotIn("HIDDEN_ISSUE_GOLD", json.dumps(payload))

    def test_material_unknown_duplicate_missing_ids_or_changed_metadata_are_rejected(self):
        for mode in ("missing_id", "unknown", "source_private_id", "duplicate", "date",
                     "session_type", "title", "empty_content", "protocol"):
            with self.subTest(mode=mode):
                value = material_output()
                if mode == "missing_id": value["document_edits"][0].pop("doc_id")
                elif mode == "unknown": value["document_edits"][0]["doc_id"] = "d999999"
                elif mode == "source_private_id": value["document_edits"][0]["doc_id"] = "later"
                elif mode == "duplicate": value["document_edits"].append(deepcopy(value["document_edits"][0]))
                elif mode == "date": value["document_edits"][0]["date"] = "2024-01-01"
                elif mode == "session_type": value["document_edits"][0]["session"] = True
                elif mode == "title": value["document_edits"][0]["title"] = "改标题"
                elif mode == "empty_content": value["document_edits"][0]["content"] = "   "
                else: value["public_protocol"] = "作者擅自更改协议"
                report, _ = self.materials(value)
                self.assertFalse(report["proposal_ready"])
                self.assertIsNone(report["proposed_corpus"])
                self.assertEqual(report["variants"], [])
                self.assertEqual(report["raw_output"], value)

    def test_no_op_revision_cannot_manufacture_a_new_version(self):
        value = material_output()
        old = visible_documents(CORPUS)[0][1]
        value["document_edits"] = [{"doc_id": old["doc_id"], "content": old["content"]}]
        report, _ = self.materials(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposal_state"], "no_change")
        self.assertEqual(report["effective_action"], "no_change")
        self.assertEqual(report["declared_action"], "revise")
        self.assertEqual(report["document_edits"], [])
        self.assertEqual(report["variants"], [])
        self.assertIsNone(report["proposed_corpus"])
        self.assertEqual(report["raw_output"], value)
        output = question_output()
        output["decisions"][0]["reference_proposal"] = deepcopy(QUESTIONS[0]["reference_proposal"])
        report, _ = self.questions(output)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["decisions"][0]["action"], "keep")
        self.assertEqual(report["decisions"][0]["declared_action"], "revise")
        self.assertEqual(report["decisions"][0]["change_status"], "no_change")
        self.assertEqual(report["variants"], [])
        self.assertEqual(report["raw_output"], output)
        self.assertEqual(report["normalized_output"]["decisions"][0]["action"], "keep")
        self.assertEqual(len(report["normalization_audit"]), 2)

    def test_sparse_v4_preserves_full_variant_audit_and_raw_edits(self):
        value = material_output()
        report, script = self.materials(value)
        self.assertEqual(VERSION, "agent-editing/v4")
        self.assertEqual(report["version"], VERSION)
        self.assertEqual(report["binding"]["version"], VERSION)
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(report["document_edits"], value["document_edits"])
        self.assertEqual(report["records"][-1]["output"], value)
        self.assertEqual(len(report["variants"][0]["before"]), 2)
        self.assertEqual(len(report["variants"][0]["after"]), 2)
        self.assertEqual(len(report["document_edits"]), 1)
        self.assertEqual(report["variants"][0]["before"][0], report["variants"][0]["after"][0])
        self.assertFalse(report["variants"][0]["quality_approval"])
        self.assertEqual(report["variants"][0]["review_state"], "pending_independent_material_review")
        payload = json.loads(script.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["documents"], visible_documents(CORPUS)[0])

    def test_edit_order_does_not_change_assembly_or_variant_identity(self):
        value = material_output()
        value["document_edits"].append({"doc_id": "d000001", "content": "新材料收到；采用状态另行核实。"})
        first, _ = self.materials(value)
        value["document_edits"].reverse()
        second, _ = self.materials(value)
        self.assertTrue(first["proposal_ready"])
        self.assertTrue(second["proposal_ready"])
        self.assertEqual(first["proposed_corpus"], second["proposed_corpus"])
        self.assertEqual(first["variants"], second["variants"])
        sessions = second["proposed_corpus"]["corpus"]["sessions"]
        self.assertEqual(sessions[0]["docs"][0]["content"], material_output()["document_edits"][0]["content"])
        self.assertEqual(sessions[1]["docs"][0]["content"], "新材料收到；采用状态另行核实。")

    def test_old_whole_document_contract_is_explicitly_rejected(self):
        old = {"action": "revise", "reason": "旧格式", "documents": visible_documents(CORPUS)[0],
               "issue_responses": deepcopy(RESPONSES)}
        for value in (old, {**material_output(), "documents": old["documents"]},
                      {"action": "no_change", "reason": "旧格式", "documents": None,
                       "issue_responses": deepcopy(RESPONSES)}):
            with self.subTest(action=value["action"]):
                report, _ = self.materials(value)
                self.assertFalse(report["proposal_ready"])
                self.assertIn("unexpected_material_revision_fields", report["format_issues"])
                self.assertEqual(report["raw_output"], value)
                self.assertIsNone(report["proposed_corpus"])

    def test_unchanged_echoes_projected_and_non_revision_cannot_carry_actual_edits(self):
        unchanged = {k: visible_documents(CORPUS)[0][0][k] for k in ("doc_id", "content")}
        value = material_output()
        value["document_edits"].append(unchanged)
        report, _ = self.materials(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["document_edits"], material_output()["document_edits"])
        self.assertEqual(len(report["variants"]), 1)
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(report["normalization_audit"][0]["operation"], "remove_no_op_document_edit")
        for action in ("no_change", "unresolved"):
            report, _ = self.materials({**material_output(), "action": action})
            self.assertFalse(report["proposal_ready"])
            self.assertIn("non_revision_must_not_supply_document_edits", report["format_issues"])
        for edits in (None, {}, "replace one body"):
            report, _ = self.materials({**material_output(), "document_edits": edits})
            self.assertFalse(report["proposal_ready"])
            self.assertIn("document_edits_must_be_list", report["format_issues"])

    def test_exact_date_session_echo_is_removed_without_changing_variant_or_source_identity(self):
        plain, _ = self.materials()
        value = material_output()
        old = visible_documents(CORPUS)[0][1]
        value["document_edits"][0].update(date=old["date"], session=old["session"])
        report, _ = self.materials(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["variants"], plain["variants"])
        self.assertEqual(report["proposed_corpus"], plain["proposed_corpus"])
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(report["normalized_output"], material_output())
        self.assertEqual(len(report["normalization_audit"]), 2)
        self.assertEqual(report["normalization_audit"][1]["source"]["path"], "/documents/1/session")

    def test_identity_echo_must_have_exact_json_type_and_existing_key(self):
        # Python's True == 1 and 1.0 == 1 must never prove an unchanged session.
        old = visible_documents(CORPUS)[0][0]
        for key, bad in (("session", True), ("session", 1.0), ("session", "1"),
                         ("date", old["date"] + " "), ("title", "内部标题"),
                         ("session_id", 1)):
            with self.subTest(key=key, bad=bad):
                value = material_output()
                value["document_edits"] = [{"doc_id": old["doc_id"], "content": old["content"], key: bad}]
                report, _ = self.materials(value)
                self.assertFalse(report["proposal_ready"])
                self.assertEqual(report["raw_output"], value)
                self.assertEqual(report["normalization_audit"], [])
        corpus = [{"content": "正文，无日期字段。"}]
        value = {"action": "revise", "reason": "回显未提供的日期", "issue_responses": [],
                 "document_edits": [{"doc_id": "d000001", "content": "新正文", "date": None}]}
        report = propose_material_revision(corpus, PROTOCOL, [], model="offline", chat_json=Scripted(value))
        self.assertFalse(report["proposal_ready"])

    def test_duplicate_unchanged_docs_remain_duplicate_before_no_op_projection(self):
        old = visible_documents(CORPUS)[0][0]
        echo = {k: old[k] for k in ("doc_id", "content")}
        for edits in ([echo, echo], [echo, {**echo, "content": "修订正文。"}],
                      [{**echo, "content": "修订正文。"}, echo]):
            value = {**material_output(), "document_edits": edits}
            report, _ = self.materials(value)
            self.assertFalse(report["proposal_ready"])
            self.assertIn("document_edits[1]:duplicate_doc_id", report["format_issues"])
            self.assertIsNone(report["proposed_corpus"])

    def test_empty_revise_is_audited_no_change_and_unchanged_non_revision_has_no_candidate(self):
        for action in ("revise", "no_change", "unresolved"):
            for edits in ([], [{k: visible_documents(CORPUS)[0][0][k] for k in ("doc_id", "content")}]):
                with self.subTest(action=action, edits=bool(edits)):
                    value = {**material_output(), "action": action, "document_edits": edits}
                    report, _ = self.materials(value)
                    self.assertTrue(report["proposal_ready"])
                    self.assertEqual(report["effective_action"], "no_change" if action == "revise" else action)
                    self.assertEqual(report["document_edits"], [])
                    self.assertEqual(report["variants"], [])
                    self.assertIsNone(report["proposed_corpus"])

    def test_unchanged_question_and_reference_echo_keep_original_values(self):
        value = question_output()
        value["decisions"][1].update(question=QUESTIONS[1]["question"],
                                      reference_proposal=deepcopy(QUESTIONS[1]["reference_proposal"]))
        report, _ = self.questions(value)
        plain, _ = self.questions()
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposed_questions"], plain["proposed_questions"])
        self.assertEqual(report["variants"], plain["variants"])
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(len(report["normalization_audit"]), 2)
        self.assertEqual(report["records"][-1]["normalization_audit"], report["normalization_audit"])

    def test_keep_full_empty_reference_placeholder_preserves_reference_and_records_distinct_audit(self):
        value = question_output()
        value["decisions"][1]["reference_proposal"] = {"answer": "", "rationale": ""}
        report, _ = self.questions(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposed_questions"][1]["reference_proposal"], QUESTIONS[1]["reference_proposal"])
        self.assertEqual(report["normalization_audit"][0]["operation"], "remove_explicit_keep_empty_reference_placeholder")
        self.assertEqual(report["raw_output"], value)
        # Missing reference stays absent: a placeholder does not manufacture one.
        questions = deepcopy(QUESTIONS)
        questions[1].pop("reference_proposal")
        report = propose_question_revisions(questions, CORPUS, PROTOCOL, ISSUES,
            model="offline", chat_json=Scripted(value))
        self.assertTrue(report["proposal_ready"])
        self.assertNotIn("reference_proposal", report["proposed_questions"][1])

    def test_conflicting_or_near_empty_keep_reference_is_never_silently_ignored(self):
        bad_refs = ({"answer": " ", "rationale": ""}, {"answer": "", "rationale": " "},
                    {"answer": "", "rationale": "新的理由"}, {"answer": "新答案", "rationale": ""},
                    {"answer": "", "rationale": "", "extra": ""}, {"answer": ""}, None,
                    {"answer": [], "rationale": ""})
        for ref in bad_refs:
            with self.subTest(ref=ref):
                value = question_output()
                value["decisions"][1]["reference_proposal"] = ref
                report, _ = self.questions(value)
                self.assertFalse(report["proposal_ready"])
                self.assertIn("non_revision_has_edit_fields:b", report["format_issues"])
                self.assertEqual(report["normalization_audit"], [])
                self.assertEqual(report["proposed_questions"], [])

    def test_revise_empty_reference_is_actual_proposal_not_keep_placeholder(self):
        value = question_output()
        value["decisions"][0]["reference_proposal"] = {"answer": "", "rationale": ""}
        report, _ = self.questions(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["decisions"][0]["action"], "revise")
        self.assertEqual(report["proposed_questions"][0]["reference_proposal"], {"answer": "", "rationale": ""})
        self.assertEqual(report["normalization_audit"], [])
        self.assertFalse(report["variants"][0]["quality_approval"])

    def test_no_edit_revise_with_only_missing_responses_retains_whole_candidate_and_incomplete_editorial_record(self):
        value = question_output()
        value["decisions"][0].pop("reference_proposal")
        value["issue_responses"] = []
        report, _ = self.questions(value)
        self.assertFalse(report["proposal_ready"])
        self.assertTrue(report["candidate_ready"])
        self.assertEqual(report["execution"]["status"], "editorial_incomplete")
        self.assertEqual(report["proposal_state"], "candidate_with_incomplete_editorial_response")
        self.assertEqual(report["format_issues"], ["issue_response_identity_mismatch"])
        self.assertEqual(report["issue_response_diagnostics"]["missing_issue_ids"], ["i1"])
        self.assertEqual(report["normalized_output"]["decisions"][0]["action"], "keep")
        self.assertEqual(report["normalization_audit"][0]["operation"], "revise_without_change_to_no_change")
        self.assertEqual(len(report["decisions"]), 2)
        self.assertEqual([d["action"] for d in report["decisions"]], ["keep", "keep"])
        self.assertEqual(report["proposed_questions"], report["inputs"]["questions"])
        self.assertFalse(report["editorial_completion"]["complete"])
        self.assertEqual(report["editorial_completion"]["status"], "incomplete")
        self.assertEqual(report["editorial_completion"]["missing_issue_ids"], ["i1"])
        self.assertFalse(report["editorial_completion"]["issue_responses_complete"])
        self.assertEqual(report["variants"], [])

    def test_actual_revision_and_keep_survive_missing_only_feedback_without_claiming_editor_completion(self):
        issues = ISSUES + [{"issue_id": "i2", "qid": "b", "description": "请复核收到日期的出处。"}]
        value = question_output()
        report = propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, issues,
            model="offline", chat_json=Scripted(value))
        self.assertTrue(report["candidate_ready"])
        self.assertFalse(report["proposal_ready"])
        self.assertEqual(report["candidate_validation"]["errors"], [])
        self.assertEqual(report["editorial_completion"]["missing_issue_ids"], ["i2"])
        self.assertEqual(report["editorial_completion"]["observed_issue_ids"], ["i1"])
        self.assertEqual(report["editorial_completion"]["status"], "incomplete")
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(report["proposed_questions"][0]["reference_proposal"], value["decisions"][0]["reference_proposal"])
        self.assertEqual(report["proposed_questions"][1], report["inputs"]["questions"][1])
        self.assertEqual(report["quantity"]["by_action"], {"keep": 1, "revise": 1, "drop": 0, "unresolved": 0})
        self.assertEqual(len(report["variants"]), 1)
        self.assertEqual(report["variants"][0]["review_state"], "pending_independent_question_review")
        self.assertFalse(report["variants"][0]["quality_approval"])
        self.assertFalse(report["records"][-1]["editorial_completion"]["complete"])
        self.assertTrue(report["records"][-1]["candidate_ready"])
        self.assertEqual(report["publication_effect"], "none")

    def test_missing_feedback_does_not_relax_qid_scope_or_actual_edit_shape(self):
        for mode in ("missing_qid", "duplicate_qid", "unknown_qid", "empty_question", "bad_reference", "injected_ready"):
            with self.subTest(mode=mode):
                value = question_output()
                value["issue_responses"] = []
                if mode == "missing_qid": value["decisions"].pop()
                elif mode == "duplicate_qid": value["decisions"].append(deepcopy(value["decisions"][0]))
                elif mode == "unknown_qid": value["decisions"][1]["qid"] = "new_qid"
                elif mode == "empty_question": value["decisions"][0]["question"] = ""
                elif mode == "bad_reference": value["decisions"][0]["reference_proposal"] = {"answer": "未提供rationale。"}
                else: value["candidate_ready"] = True
                report, _ = self.questions(value)
                self.assertFalse(report["candidate_ready"])
                self.assertFalse(report["proposal_ready"])
                self.assertEqual(report["execution"]["status"], "invalid_output")
                self.assertEqual(report["candidate_validation"]["status"], "invalid")
                self.assertTrue(report["candidate_validation"]["errors"])
                self.assertEqual(report["editorial_completion"]["status"], "invalid")
                self.assertFalse(report["editorial_completion"]["complete"])
                self.assertEqual(report["proposed_questions"], [])
                self.assertEqual(report["decisions"], [])
                self.assertEqual(report["variants"], [])
                self.assertEqual(report["raw_output"], value)

    def test_missing_feedback_never_swallows_a_nonempty_conflicting_keep_reference(self):
        value = question_output()
        value["decisions"][0]["action"] = "keep"
        value["issue_responses"] = []
        report, _ = self.questions(value)
        self.assertFalse(report["candidate_ready"])
        self.assertFalse(report["proposal_ready"])
        self.assertIn("non_revision_has_edit_fields:a", report["candidate_validation"]["errors"])
        self.assertEqual(report["proposed_questions"], [])
        self.assertEqual(report["normalized_output"]["decisions"][0]["reference_proposal"],
                         value["decisions"][0]["reference_proposal"])

    def test_unknown_duplicate_and_malformed_issue_responses_block_even_otherwise_valid_candidates(self):
        issues = ISSUES + [{"issue_id": "i2", "qid": "b", "description": "尚待回应。"}]
        for responses in ([{**RESPONSES[0], "issue_id": "not_supplied"}], RESPONSES + RESPONSES,
                          [{"issue_id": "i1", "addressed": "addressed", "reason": "错键不可猜。"}],
                          [{**RESPONSES[0], "reason": ""}], [{**RESPONSES[0], "disposition": "done"}],
                          ["i1"], None, {}):
            with self.subTest(responses=responses):
                value = {**question_output(), "issue_responses": responses}
                report = propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, issues,
                    model="offline", chat_json=Scripted(value))
                self.assertFalse(report["candidate_ready"])
                self.assertFalse(report["proposal_ready"])
                self.assertEqual(report["execution"]["status"], "invalid_output")
                self.assertEqual(report["editorial_completion"]["status"], "invalid")
                self.assertEqual(report["proposed_questions"], [])
                self.assertEqual(report["variants"], [])
                self.assertEqual(report["raw_output"], value)

    def test_material_missing_responses_remain_strict_without_question_candidate_interface(self):
        value = {**material_output(), "issue_responses": []}
        report, _ = self.materials(value)
        self.assertFalse(report["proposal_ready"])
        self.assertNotIn("candidate_ready", report)
        self.assertNotIn("editorial_completion", report)
        self.assertEqual(report["execution"]["status"], "invalid_output")
        self.assertIsNone(report["proposed_corpus"])
        self.assertEqual(report["variants"], [])

    def test_complete_responses_do_not_mark_invalid_question_edits_editorially_complete(self):
        value = question_output()
        value["decisions"][0]["reference_proposal"] = None
        report, _ = self.questions(value)
        self.assertFalse(report["candidate_ready"])
        self.assertFalse(report["editorial_completion"]["complete"])
        self.assertTrue(report["editorial_completion"]["issue_responses_complete"])
        self.assertEqual(report["editorial_completion"]["missing_issue_ids"], [])

    def test_readable_decisions_without_response_container_cannot_be_assumed_missing_only(self):
        value = question_output()
        value.pop("issue_responses")
        report, _ = self.questions(value)
        self.assertFalse(report["candidate_ready"])
        self.assertIn("issue_responses_must_be_list", report["candidate_validation"]["errors"])
        self.assertEqual(report["proposed_questions"], [])

    def test_malformed_decision_container_and_nonobject_returns_have_no_candidate(self):
        for value in ([question_output()], {**question_output(), "decisions": "not a list"}):
            report, _ = self.questions(value)
            self.assertFalse(report["candidate_ready"])
            self.assertEqual(report["candidate_validation"]["status"], "invalid")
            self.assertFalse(report["editorial_completion"]["complete"])
            self.assertEqual(report["editorial_completion"]["status"], "invalid")
            self.assertEqual(report["proposed_questions"], [])

    def test_no_issues_requires_explicit_empty_responses_but_counts_complete_when_valid(self):
        value = {**question_output(), "issue_responses": []}
        report = propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, [], model="offline", chat_json=Scripted(value))
        self.assertTrue(report["candidate_ready"])
        self.assertTrue(report["proposal_ready"])
        self.assertTrue(report["editorial_completion"]["complete"])
        self.assertTrue(report["editorial_completion"]["issue_responses_complete"])
        self.assertEqual(report["editorial_completion"]["expected_issue_ids"], [])

    def test_incomplete_editorial_candidate_audit_failure_revokes_readiness_and_retains_raw(self):
        value = {**question_output(), "issue_responses": []}
        def sink(event):
            if event["event"] == "finished": raise OSError("output disk unavailable")
        with self.assertRaises(AuditRecordError) as caught:
            propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
                model="offline", chat_json=Scripted(value), record=sink)
        report = caught.exception.report
        self.assertFalse(report["candidate_ready"])
        self.assertFalse(report["proposal_ready"])
        self.assertEqual(report["execution"]["status"], "audit_error")
        self.assertEqual(report["execution"]["prior_execution"]["status"], "editorial_incomplete")
        self.assertFalse(report["editorial_completion"]["complete"])
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(len(report["proposed_questions"]), 2)

    def test_wrong_response_key_is_not_guessed_or_misreported_as_missing_identity(self):
        value = material_output()
        value["issue_responses"] = [{"issue_id": "i1", "addressed": "addressed", "reason": "确有回应该身份。"}]
        value["document_edits"][0].update(date="2025-01-08", session=5)
        report, _ = self.materials(value)
        self.assertEqual(report["format_issues"], ["invalid_issue_response"])
        self.assertEqual(report["issue_response_diagnostics"]["missing_issue_ids"], [])
        self.assertEqual(report["issue_response_diagnostics"]["invalid_entries"],
                         [{"index": 0, "problems": ["invalid_disposition"]}])
        self.assertEqual(report["normalized_output"]["issue_responses"], value["issue_responses"])
        self.assertEqual(len(report["normalization_audit"]), 2)
        self.assertIsNone(report["proposed_corpus"])
        self.assertEqual(report["variants"], [])

    def test_duplicate_malformed_issue_id_still_detected_and_no_guessing_enum(self):
        value = question_output()
        value["issue_responses"] = [RESPONSES[0], {"issue_id": "i1", "disposition": ["addressed"], "reason": "解释。"}]
        report, _ = self.questions(value)
        self.assertIn("invalid_issue_response", report["format_issues"])
        self.assertIn("issue_response_identity_mismatch", report["format_issues"])
        self.assertEqual(report["issue_response_diagnostics"]["duplicate_issue_ids"], ["i1"])

    def test_normalization_does_not_trim_text_or_correct_rationale_citations(self):
        value = material_output()
        old = visible_documents(CORPUS)[0][1]
        value["document_edits"][0]["content"] = old["content"] + " "
        report, _ = self.materials(value)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["document_edits"], value["document_edits"])
        self.assertEqual(report["normalization_audit"], [])
        value = question_output()
        value["decisions"][0]["reference_proposal"]["rationale"] = "d000099 中不存在的引文。"
        report, _ = self.questions(value)
        self.assertEqual(report["proposed_questions"][0]["reference_proposal"], value["decisions"][0]["reference_proposal"])
        self.assertEqual(report["normalization_audit"], [])
        self.assertFalse(report["variants"][0]["quality_approval"])

    def test_material_review_sees_only_public_material_and_retains_disagreements(self):
        output = {"assessment": "保留合理分歧，建议澄清一处证据范围。",
                  "issues": [{k: v for k, v in ISSUES[0].items() if k not in {"qid", "gold"}}],
                  "role_differences": ["两位角色可有不同判断。"], "business_use_observations": ["会签有正常业务用途。"],
                  "limitations": [], "coverage": {"status": "complete", "limitations": []}}
        script = Scripted(output)
        report = review_materials(CORPUS, PROTOCOL, model="offline", chat_json=script)
        payload = json.loads(script.calls[0]["messages"][1]["content"])
        self.assertEqual(set(payload), {"documents", "public_protocol"})
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(report["proposal_state"], "review_proposed")
        self.assertEqual(report["coverage_claim_source"], "model_self_report")
        self.assertEqual(report["resolved_issue_evidence"][0]["evidence"][0]["resolved_text"],
                         visible_documents(CORPUS)[0][0]["content"])

    def test_invalid_material_review_quote_preserved_but_not_located_or_approved(self):
        output = {"assessment": "需要核实原文位置。", "issues": [{"issue_id": "i", "description": "可能不一致", "evidence": [
            {"doc_id": "d000001", "field": "content", "quote": "没有这句话", "role": "support", "explanation": "说明"}]}],
            "role_differences": [], "business_use_observations": [], "limitations": []}
        report = review_materials(CORPUS, PROTOCOL, model="offline", chat_json=Scripted(output))
        self.assertFalse(report["proposal_ready"])
        self.assertEqual(report["raw_output"], output)
        self.assertIn("citation_location_invalid", report["format_issues"][0])

    def test_review_prompt_names_actual_document_keys_without_repairing_section_title_fields(self):
        value = {"assessment": "需要核实。", "issues": [{"issue_id": "i", "description": "建议检查小节", "evidence": [
            {"doc_id": "d000001", "field": "采用来源", "location_scope": "field", "role": "support", "explanation": "小节名称。"}]}],
            "role_differences": [], "business_use_observations": [], "limitations": []}
        script = Scripted(value)
        report = review_materials(CORPUS, PROTOCOL, model="offline", chat_json=script)
        self.assertIn("实际存在的键content、date或session", script.calls[0]["messages"][0]["content"])
        self.assertIn("小节标题、业务字段名、段落名不是field", script.calls[0]["messages"][0]["content"])
        self.assertFalse(report["proposal_ready"])
        self.assertIn("citation_location_invalid", report["format_issues"][0])
        self.assertEqual(report["raw_output"], value)
        self.assertEqual(report["normalization_audit"], [])

    def test_review_severity_is_free_text_and_never_a_python_approval_condition(self):
        for description in ("严重阻断，应先修订。", "合理角色分歧，无需修改。"):
            output = {"assessment": description,
                      "issues": [{"issue_id": "i", "description": "需结合语境解释。", "severity": description,
                                  "suggested_response": "审阅者建议可被推翻。", "evidence": []}],
                      "role_differences": [], "business_use_observations": [], "limitations": []}
            report = review_materials(CORPUS, PROTOCOL, model="offline", chat_json=Scripted(output))
            self.assertTrue(report["proposal_ready"])
            self.assertEqual(report["issues"][0]["severity"], description)
            self.assertEqual(report["publication_effect"], "none")

    def test_question_variant_identity_changes_with_its_public_context(self):
        first, _ = self.questions()
        changed = deepcopy(CORPUS)
        changed["corpus"]["sessions"][0]["docs"][0]["content"] += "另有待补资料。"
        second = propose_question_revisions(QUESTIONS, changed, PROTOCOL, ISSUES,
            model="offline", chat_json=Scripted(question_output()))
        self.assertNotEqual(first["variants"][0]["variant_id"], second["variants"][0]["variant_id"])
        self.assertEqual(first["proposed_questions"], second["proposed_questions"])

    def test_budget_and_input_limits_prevent_any_call_without_truncation(self):
        for kwargs, expected in (({"max_calls": 0}, "call_budget_exhausted"),
                                  ({"max_input_chars": 1}, "input_limit")):
            report, script = self.questions(**kwargs)
            self.assertEqual(script.calls, [])
            self.assertEqual(report["calls_used"], 0)
            self.assertFalse(report["candidate_ready"])
            self.assertFalse(report["editorial_completion"]["complete"])
            self.assertEqual(report["candidate_validation"]["status"], "not_executed")
            self.assertEqual(report["execution"]["status"], expected)
            self.assertIn(CORPUS["corpus"]["sessions"][1]["docs"][0]["content"], report["messages"][1]["content"])

    def test_invalid_budgets_raise_before_provider_or_audit_sink(self):
        for kwargs in ({"max_calls": 2}, {"max_calls": True}, {"max_tokens": 0}, {"max_input_chars": -1}):
            script, events = Scripted(question_output()), []
            with self.assertRaises(ValueError):
                propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
                    model="offline", chat_json=script, record=events.append, **kwargs)
            self.assertEqual(script.calls, [])
            self.assertEqual(events, [])

    def test_provider_failure_is_not_a_semantic_rejection_or_retry(self):
        report, script = self.questions(RuntimeError("provider failed"))
        self.assertEqual(report["execution"]["status"], "model_error")
        self.assertFalse(report["candidate_ready"])
        self.assertFalse(report["editorial_completion"]["complete"])
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(report["decisions"], [])
        self.assertEqual([r["event"] for r in report["records"]], ["started", "finished"])

    def test_bad_semantic_enum_container_does_not_lose_raw_response(self):
        output = question_output()
        output["decisions"][0]["action"] = ["revise"]
        report, _ = self.questions(output)
        self.assertEqual(report["execution"]["status"], "invalid_output")
        self.assertEqual(report["raw_output"], output)
        self.assertEqual(report["records"][-1]["event"], "finished")

    def test_started_audit_failure_prevents_call_and_carries_event(self):
        script = Scripted(question_output())
        def sink(event): raise OSError("disk unavailable")
        with self.assertRaises(AuditRecordError) as caught:
            propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
                model="offline", chat_json=script, record=sink)
        self.assertEqual(script.calls, [])
        self.assertEqual(caught.exception.report["records"][-1]["event"], "started")
        self.assertEqual(caught.exception.report["calls_used"], 0)

    def test_finished_audit_failure_retains_returned_raw_output_and_decisions(self):
        script = Scripted(question_output())
        def sink(event):
            if event["event"] == "finished": raise OSError("disk unavailable")
        with self.assertRaises(AuditRecordError) as caught:
            propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
                model="offline", chat_json=script, record=sink)
        report = caught.exception.report
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(report["raw_output"], question_output())
        self.assertEqual(report["records"][-1]["output"], question_output())
        self.assertEqual(report["audit_error"]["event"], "finished")
        self.assertEqual(report["execution"]["status"], "audit_error")
        self.assertFalse(report["proposal_ready"])
        self.assertFalse(report["candidate_ready"])
        self.assertEqual(len(report["decisions"]), 2)

    def test_callback_and_provider_input_mutation_cannot_change_inputs(self):
        before = deepcopy((QUESTIONS, CORPUS, ISSUES))
        def provider(step, messages, **params):
            messages.clear()
            return question_output()
        def sink(event): event.clear()
        report = propose_question_revisions(QUESTIONS, CORPUS, PROTOCOL, ISSUES,
            model="offline", chat_json=provider, record=sink)
        self.assertEqual((QUESTIONS, CORPUS, ISSUES), before)
        self.assertTrue(report["proposal_ready"])
        self.assertEqual(len(report["messages"]), 2)
        self.assertEqual(report["records"][0]["event"], "started")

    def test_editor_cannot_smuggle_review_pass_flags(self):
        output = question_output()
        output["decisions"][0]["quality_approval"] = True
        report, _ = self.questions(output)
        self.assertFalse(report["proposal_ready"])
        self.assertEqual(report["variants"], [])


if __name__ == "__main__":
    unittest.main()
