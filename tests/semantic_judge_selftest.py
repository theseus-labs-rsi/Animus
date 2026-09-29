"""Offline information-flow and incomplete-result tests, not model accuracy tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.semantic_review import review_questions, REFERENCE_POLICY, CITATION_POLICY, resolve_citations
from eval.semantic_judge import SemanticJudge, SYSTEM, VERSION
from eval.grading import is_scored

Q = {"question": "报告发布日期是什么？", "gt": "2024-03-20"}
CORPUS = [{"content": "报告于2024年3月20日发布。"}]


def review_output(stage="adjudicate", **changes):
    value = {"interpretation": "报告发布日期", "answerability": "answerable",
             "major_requirements": ["回答发布日期"],
              "coverage": {"status": "complete", "scope_conflict": False, "inspected_doc_ids": ["d000001"], "limitations": []},
             "evidence": [{"doc_id": "d000001", "field": "content", "quote": "2024年3月20日",
                           "role": "support", "explanation": "发布日期"}],
             "reasoning": "正文记录该日期。"}
    if stage == "blind_read":
        value["answer"] = "2024-03-20"
    else:
        value.update(item_validity="valid", reference_status="supported", reviewed_answer="2024-03-20", concerns=[],
            reviewed_rationale="正文直接提供日期。",
            original_answer_review=[{"requirement": "回答发布日期", "assessment": "完成",
                                     "explanation": "答案符合正文。"}],
            original_rationale_review={"status": "not_provided", "claims": [], "limitations": []},
            review_findings={"substantive_defects": [], "acceptable_brevity": [], "editorial_suggestions": []})
    return {**value, **changes}


def report():
    return review_questions([Q], CORPUS, "回答日期", reviewer_model="offline-reader",
        chat_json=lambda step, messages, **kw: review_output("blind_read" if step.endswith("blind_read") else "adjudicate"))


def grade_output(**changes):
    result = review_output(answer_verdict="correct", format_compliance="compliant",
        additional_facts={"status": "not_assessed", "reason": "无附加要求"},
        answer_review={"primary_task": "根据公开记录回答报告发布日期。",
            "answer_meaning": "答案认为报告于2024年3月20日发布，没有额外时点断言。",
            "claims": [{"claim": "发布日期为2024年3月20日", "task_role": "primary",
                "assessment": "supported", "evidence_indices": [0], "explanation": "正文直接记录日期。"}],
            "reference_comparison": "既往解释与当前公开日期证据一致，未发现参考问题。"},
        requires_item_reassessment=False, reassessment_reason="没有需要同题重审的问题。")
    result.update(changes)
    if not result["evidence"]:
        for claim in result["answer_review"]["claims"]:
            claim["evidence_indices"] = []
    return result


class SemanticJudgeTest(unittest.TestCase):
    def make(self, output=None, reference=None, **kwargs):
        call = Mock(return_value=grade_output() if output is None else output)
        return SemanticJudge(reference or report(), [Q], CORPUS, "回答日期", model="offline-judge",
                             chat_json=call, **kwargs), call

    def test_natural_date_goes_to_model_with_full_answer(self):
        judge, call = self.make()
        pred = "文档没有说是3月21日。发布日期是2024年3月20日。"
        result = judge(Q, pred)
        self.assertTrue(result["correct"])
        self.assertTrue(is_scored({**Q, "pred": pred, "correct": True, "judgement": result}))
        payload = json.loads(call.call_args.args[1][1]["content"])
        self.assertEqual(payload["solver_answer"], pred)
        self.assertNotIn("question_contract", payload)
        self.assertEqual(call.call_args.kwargs["model"], "offline-judge")

    def test_even_exact_answer_gets_semantic_review(self):
        judge, call = self.make()
        judge(Q, Q["gt"])
        call.assert_called_once()

    def test_unknown_reference_and_partial_review_stay_unscored(self):
        for changes in ({"reference_status": "contradicted"}, {"item_validity": "ambiguous"},
                        {"answerability": "ambiguous"}, {"answerability": "unresolved"},
                        {"coverage": {"status": "partial", "scope_conflict": False, "inspected_doc_ids": [], "limitations": ["incomplete"]}},
                        {"coverage": {"status": "unknown", "scope_conflict": False, "limitations": ["Unknown scope"]}},
                        {"coverage": {"status": "complete", "scope_conflict": True, "limitations": ["Unresolved scope conflict"]}}):
            judge, _ = self.make(output=grade_output(**changes))
            result = judge(Q, "2024-03-20")
            self.assertEqual(result["verdict"], "uncertain")
            self.assertFalse(is_scored(result))

    def test_protocol_and_material_change_invalidate_reference(self):
        for corpus, protocol in [([{ "content": "改写了正文"}], "回答日期"), (CORPUS, "另一个口径")]:
            with self.assertRaises(ValueError):
                SemanticJudge(report(), [Q], corpus, protocol, model="offline", chat_json=Mock())

    def test_changed_reference_after_construction_is_not_scored(self):
        judge, call = self.make()
        result = judge({**Q, "gt": "2024-03-21"}, "2024-03-20")
        self.assertEqual(result["verdict"], "uncertain")
        call.assert_not_called()

    def test_answer_tamper_does_not_reuse_grade(self):
        judge, _ = self.make()
        result = judge(Q, "2024-03-20")
        self.assertFalse(is_scored({**Q, "pred": "2024-03-21", "correct": True, "judgement": result}))
        self.assertFalse(is_scored({**Q, "question": "另一道题？", "pred": "2024-03-20", "correct": True, "judgement": result}))

    def test_bad_citation_is_review_error_not_incorrect(self):
        out = grade_output()
        out["evidence"][0]["quote"] = "材料并没有这一句"
        judge, _ = self.make(output=out)
        result = judge(Q, "2024-03-20")
        self.assertEqual(result["verdict"], "error")
        self.assertIsNone(result["correct"])

    def test_budget_is_bounded_without_automatic_repair(self):
        judge, call = self.make(max_calls=1)
        judge(Q, "2024-03-20")
        self.assertEqual(judge(Q, "2024-03-20")["verdict"], "uncertain")
        call.assert_called_once()
        self.assertEqual(len(judge.events), 2)

    def test_disputed_original_gold_does_not_get_silently_replaced(self):
        reference = report()
        reference["items"][0]["reference_status"] = "contradicted"
        reference["items"][0]["reviewed_answer"] = "2024-03-21"
        reference["items"][0]["adjudication"]["reference_status"] = "contradicted"
        reference["items"][0]["adjudication"]["reviewed_answer"] = "2024-03-21"
        judge, call = self.make(reference=reference)
        self.assertEqual(judge(Q, "2024-03-21")["verdict"], "uncertain")
        call.assert_not_called()

    def test_stored_question_reference_and_encoding_payloads_must_match_binding(self):
        for field, value in (("question", "另一道题"), ("reference_proposal", "2099-12-31"),
                             ("reference_provided", False), ("reference_encoding", {"meaning": "forged"})):
            with self.subTest(field=field):
                reference = report()
                reference["items"][0][field] = value
                with self.assertRaises(ValueError):
                    self.make(reference=reference)

    def test_completed_summary_cannot_mask_partial_or_invalid_review_stages(self):
        for mutate in (
            lambda item: item["coverage"].update(status="partial"),
            lambda item: item["blind_read"]["coverage"].update(status="partial"),
            lambda item: item["adjudication"]["coverage"].update(scope_conflict=True),
            lambda item: item["adjudication"]["evidence"][0].update(quote="不存在"),
            lambda item: item.update(reference_status="contradicted"),
        ):
            reference = report()
            mutate(reference["items"][0])
            with self.assertRaises(ValueError):
                self.make(reference=reference)

    def test_original_reference_and_task_versions_are_bound_before_constructor(self):
        for changes in ({"question_version": "new"}, {"reference_version": "new"},
                         {"rubric": {"primary_task": "new requirement"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SemanticJudge(report(), [{**Q, **changes}], CORPUS, "回答日期",
                              model="offline", chat_json=Mock())

    def test_machine_input_manifest_and_full_visible_payload_are_bound(self):
        for mutate in (
            lambda stored: stored["input_manifest"].update(document_count=0),
            lambda stored: stored["documents"][0].update(content="different body"),
            lambda stored: stored.update(public_protocol="changed public rules"),
        ):
            reference = report()
            mutate(reference)
            with self.assertRaises(ValueError):
                self.make(reference=reference)

    def test_false_not_provided_from_judge_is_unscored_protocol_error(self):
        judge, call = self.make(output=grade_output(reference_status="not_provided"))
        result = judge(Q, Q["gt"])
        self.assertEqual(result["verdict"], "error")
        self.assertFalse(is_scored({**Q, "pred": Q["gt"], "correct": result["correct"], "judgement": result}))
        payload = json.loads(call.call_args.args[1][1]["content"])
        self.assertTrue(payload["reference_provided"])
        self.assertIn(REFERENCE_POLICY, call.call_args.args[1][0]["content"])

    def test_absent_reference_and_explicit_null_do_not_share_a_grade(self):
        question = {"question": Q["question"]}
        reference = review_questions([question], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output("blind_read") if step.endswith("blind_read")
                else review_output(reference_status="not_provided"))
        judge = SemanticJudge(reference, [question], CORPUS, "回答日期", model="offline-judge",
                              chat_json=lambda *a, **kw: grade_output(reference_status="not_provided"))
        result = judge(question, Q["gt"])
        valid = {**question, "pred": Q["gt"], "correct": True, "judgement": result}
        self.assertTrue(is_scored(valid))
        self.assertFalse(is_scored({**valid, "gt": None}))
        tampered = deepcopy(valid)
        tampered["judgement"]["reference_provided"] = True
        self.assertFalse(is_scored(tampered))

    def test_encoding_reaches_judge_without_hidden_reference_metadata(self):
        question = {"question": Q["question"], "gt": "INSUFFICIENT_EVIDENCE",
                    "question_contract": {"answer_kind": "abstention", "abstention_kind": "never_known",
                                          "lure": "HIDDEN_LURE"}}
        reference = review_questions([question], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output("blind_read") if step.endswith("blind_read")
                else review_output())
        call = Mock(return_value=grade_output())
        judge = SemanticJudge(reference, [question], CORPUS, "回答日期", model="offline-judge", chat_json=call)
        judge(question, "无法确定")
        payload = json.loads(call.call_args.args[1][1]["content"])
        self.assertEqual(payload["reference_encoding"]["declared_subtype"]["code"], "never_known")
        self.assertNotIn("HIDDEN_LURE", json.dumps(payload))

    def make_batch(self, questions, decisions=None):
        decisions = iter(decisions or [review_output() for _ in questions])
        reference = review_questions(questions, CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output("blind_read") if step.endswith("blind_read")
                else deepcopy(next(decisions)))
        call = Mock(return_value=grade_output())
        judge = SemanticJudge(reference, questions, CORPUS, "回答日期", model="offline-judge", chat_json=call)
        return judge, call, reference

    def test_same_question_different_references_are_independent(self):
        supported = {**Q, "qid": "same-source"}
        contradicted = {**supported, "gt": "2024-03-21"}
        judge, call, _ = self.make_batch([supported, contradicted],
            [review_output(), review_output(reference_status="contradicted")])
        self.assertEqual(judge(supported, Q["gt"])["verdict"], "correct")
        self.assertEqual(judge(contradicted, Q["gt"])["verdict"], "uncertain")
        call.assert_called_once()

    def test_full_review_can_serve_a_selected_question_subset(self):
        questions = [{**Q, "qid": "selected"}, {**Q, "qid": "not-selected"},
                     {**Q, "qid": "other", "question": "另一份报告哪天发布？"}]
        _, _, reference = self.make_batch(questions,
            [review_output(), review_output(reference_status="contradicted"), review_output()])
        call = Mock(return_value=grade_output())
        judge = SemanticJudge(reference, [questions[0]], CORPUS, "回答日期", model="offline-judge", chat_json=call)
        self.assertEqual(judge(questions[0], Q["gt"])["verdict"], "correct")
        self.assertEqual(judge(questions[1], Q["gt"])["verdict"], "uncertain")
        call.assert_called_once()

    def test_same_question_reference_different_source_ids_are_independent(self):
        supported, contradicted = {**Q, "qid": "first"}, {**Q, "qid": "second"}
        judge, call, _ = self.make_batch([supported, contradicted],
            [review_output(), review_output(reference_status="contradicted")])
        grade = judge(supported, Q["gt"])
        self.assertEqual(grade["verdict"], "correct")
        self.assertEqual(grade["source_qid"], "first")
        self.assertEqual(judge(contradicted, Q["gt"])["verdict"], "uncertain")
        call.assert_called_once()
        self.assertFalse(is_scored({**contradicted, "pred": Q["gt"], "correct": True, "judgement": grade}))

    def test_identical_duplicate_review_records_can_be_reused(self):
        question = {**Q, "qid": "duplicate"}
        judge, call, reference = self.make_batch([question, deepcopy(question)])
        before = deepcopy(reference)
        grade = judge(question, Q["gt"])
        self.assertEqual(grade["verdict"], "correct")
        self.assertEqual(grade["duplicate_reviews"]["status"], "identical_reviews")
        self.assertEqual(grade["duplicate_reviews"]["candidate_ids"], ["q000001", "q000002"])
        self.assertEqual(reference, before)
        call.assert_called_once()

    def test_conflicting_duplicate_reviews_only_pause_the_affected_question(self):
        duplicate = {**Q, "qid": "duplicate"}
        other = {**Q, "qid": "other", "question": "该报告哪天发布？"}
        for decisions in ([review_output(), review_output(reference_status="contradicted")],
                          [review_output(reference_status="contradicted"), review_output()]):
            with self.subTest(order=[d["reference_status"] for d in decisions]):
                judge, call, reference = self.make_batch([duplicate, deepcopy(duplicate), other],
                    [*decisions, review_output()])
                before = deepcopy(reference)
                grade = judge(duplicate, Q["gt"])
                self.assertEqual(grade["verdict"], "uncertain")
                self.assertIsNone(grade["correct"])
                self.assertEqual(grade["review_state"], "duplicate_review_pending")
                self.assertEqual(len(grade["duplicate_reviews"]["decision_hashes"]), 2)
                call.assert_not_called()
                self.assertEqual(judge(other, Q["gt"])["verdict"], "correct")
                call.assert_called_once()
                self.assertEqual(reference, before)

    def test_different_duplicate_reasoning_is_not_resolved_by_matching_labels(self):
        judge, call, _ = self.make_batch([Q, deepcopy(Q)],
            [review_output(), review_output(reasoning="A different evidence interpretation remains unresolved.")])
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["review_state"], "duplicate_review_pending")
        call.assert_not_called()

    def test_same_reference_hash_absent_and_null_candidates_remain_separate(self):
        absent, explicit = {"question": Q["question"]}, {"question": Q["question"], "gt": None}
        judge, call, _ = self.make_batch([absent, explicit],
            [review_output(reference_status="not_provided"), review_output(reference_status="unsupported")])
        call.return_value = grade_output(reference_status="not_provided")
        self.assertEqual(judge(absent, Q["gt"])["verdict"], "correct")
        self.assertEqual(judge(explicit, Q["gt"])["verdict"], "uncertain")
        call.assert_called_once()

    def test_source_id_changes_cannot_select_another_review(self):
        question = {**Q, "qid": 1}
        judge, call, reference = self.make_batch([question])
        self.assertEqual(judge({**question, "qid": True}, Q["gt"])["verdict"], "uncertain")
        call.assert_not_called()
        grade = judge(question, Q["gt"])
        self.assertFalse(is_scored({**question, "qid": True, "pred": Q["gt"], "correct": True, "judgement": grade}))
        for mutate in (lambda item: item.update(source_qid="different"), lambda item: item.pop("source_qid")):
            corrupted = deepcopy(reference)
            mutate(corrupted["items"][0])
            with self.assertRaises(ValueError):
                SemanticJudge(corrupted, [question], CORPUS, "回答日期", model="offline", chat_json=Mock())

    def test_judge_prompt_contains_a_self_contained_return_schema(self):
        schema, _ = json.JSONDecoder().raw_decode(SYSTEM.split("完整返回 JSON Schema：", 1)[1].lstrip())
        required = {"item_validity", "answerability", "reference_status", "reviewed_answer", "interpretation",
                    "coverage", "evidence", "concerns", "reasoning", "answer_verdict", "format_compliance", "additional_facts",
                    "answer_review", "requires_item_reassessment", "reassessment_reason"}
        self.assertEqual(set(schema["required"]), required)
        self.assertEqual(set(schema["properties"]), required)
        self.assertEqual(schema["properties"]["concerns"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(schema["properties"]["coverage"]["properties"]["scope_conflict"]["type"], "boolean")
        self.assertEqual(VERSION, "semantic-primary-v6")

    def test_malformed_concerns_remain_errors_with_diagnostic_audit(self):
        judge, call = self.make(output=grade_output(concerns="This is malformed, not a list."))
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "error")
        event = judge.events[-1]
        self.assertEqual(event["error"]["error_type"], "ValueError")
        self.assertEqual(event["error"]["message"], "concerns must be a string list")
        self.assertEqual(event["output"]["concerns"], "This is malformed, not a list.")
        call.assert_called_once()

    def test_provider_parse_error_is_persisted_without_retry(self):
        judge, call = self.make()
        call.side_effect = ValueError("Invalid control character at line 1 column 37")
        self.assertEqual(judge(Q, Q["gt"])["verdict"], "error")
        self.assertEqual(judge.events[-1]["error"], {"error_type": "ValueError",
            "message": "Invalid control character at line 1 column 37"})
        call.assert_called_once()

    def test_old_grader_version_is_not_a_current_score(self):
        judge, _ = self.make()
        grade = judge(Q, Q["gt"])
        for version in ("semantic-primary-v3", "semantic-primary-v4", "semantic-primary-v5"):
            old = {**grade, "version": version}
            self.assertFalse(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": old}))

    def field_evidence(self):
        return [{"doc_id": "d000001", "field": "content", "location_scope": "field",
                 "role": "support", "explanation": "公开记录直接记载发布日期。"}]

    def test_field_citation_grading_preserves_raw_and_records_program_resolved_text(self):
        output = grade_output(evidence=self.field_evidence())
        before = deepcopy(output)
        judge, call = self.make(output=output)
        grade = judge(Q, Q["gt"])
        self.assertTrue(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": grade}))
        self.assertEqual(grade["citation_policy"], CITATION_POLICY)
        self.assertEqual(grade["evidence"], before["evidence"])
        self.assertEqual(grade["resolved_evidence"][0]["resolved_text"], CORPUS[0]["content"])
        self.assertEqual(grade["resolved_evidence"][0]["locator_source"], "program_resolved_field")
        self.assertEqual(judge.events[-1]["output"], before)
        self.assertEqual(judge.events[-1]["resolved_evidence"], grade["resolved_evidence"])
        self.assertEqual(output, before)
        call.assert_called_once()

    def test_cached_field_resolution_is_checked_against_actual_bound_corpus(self):
        evidence = self.field_evidence()
        reference = review_questions([Q], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output(
                "blind_read" if step.endswith("blind_read") else "adjudicate", evidence=evidence))
        judge, _ = self.make(reference=reference)
        self.assertEqual(judge(Q, Q["gt"])["verdict"], "correct")
        for mutate in (
            lambda item: item["resolved_evidence"][0].update(resolved_text="forged"),
            lambda item: item["stage_resolved_evidence"]["blind_read"][0].update(field_hash="forged"),
            lambda item: item.pop("stage_resolved_evidence"),
        ):
            changed = deepcopy(reference)
            mutate(changed["items"][0])
            with self.assertRaises(ValueError):
                self.make(reference=changed)

    def test_citation_v4_review_is_not_implicitly_reinterpreted_as_v5(self):
        reference = report()
        reference["binding"]["version"] = "semantic-question-shadow/v4"
        reference["binding"].pop("citation_policy")
        with self.assertRaises(ValueError):
            self.make(reference=reference)

    def test_field_and_bad_quote_modes_cannot_be_silently_merged(self):
        for citation in (
            {**self.field_evidence()[0], "quote": "报告...发布。"},
            {**self.field_evidence()[0], "resolved_text": CORPUS[0]["content"]},
            {**self.field_evidence()[0], "field": "title"},
            {**review_output()["evidence"][0], "quote": "报告...发布。"},
        ):
            judge, call = self.make(output=grade_output(evidence=[citation]))
            grade = judge(Q, Q["gt"])
            self.assertEqual(grade["verdict"], "error")
            self.assertEqual(judge.events[-1]["output"]["evidence"], [citation])
            self.assertEqual(judge.events[-1]["resolved_evidence"], [])
            call.assert_called_once()

    def test_no_evidence_still_requires_explicit_empty_resolution_record(self):
        judge, _ = self.make(output=grade_output(evidence=[]))
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["resolved_evidence"], [])
        self.assertTrue(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": grade}))
        forged = deepcopy(grade)
        forged.pop("resolved_evidence")
        self.assertFalse(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": forged}))

    def test_complete_prior_interpretations_and_limitations_reach_each_grade_unchanged(self):
        blind = review_output("blind_read", interpretation="只问本报告首次公开发布日期，不是入库日期。",
            reasoning="没有登记完整性保证；所见日期不能推出其他活动不存在。",
            answer="现有材料记载2024年3月20日。",
            coverage={"status": "complete", "scope_conflict": False,
                      "limitations": ["全量提供不意味着全量现实事件。"], "inspected_doc_ids": []})
        adjudication = review_output(interpretation="原答案应保留可见材料范围。",
            reasoning="参考日期有依据，但不能扩张到没有其他发布日期。",
            concerns=["不能用本文档缺记录排除未收录事件。"],
            coverage={"status": "complete", "scope_conflict": False,
                      "limitations": ["不能证明归档外没有同日活动。"], "inspected_doc_ids": []},
            original_rationale_review={"status": "reviewed", "claims": [
                {"claim": "正文提供发布日期", "assessment": "有依据", "explanation": "日期明确。"}],
                "limitations": ["该理由不能建立完整登记假设。"]})
        reference = review_questions([Q], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: deepcopy(blind if step.endswith("blind_read") else adjudication))
        before = deepcopy(reference)
        judge, call = self.make(reference=reference, max_calls=2)
        predictions = ["3月20日，只限现有记录。", "不是3月21日；我撤回刚才的猜测，现有记录为3月20日。"]
        for prediction in predictions:
            grade = judge(Q, prediction)
            self.assertEqual(grade["verdict"], "correct")
            payload = json.loads(call.call_args.args[1][1]["content"])
            self.assertEqual(payload["solver_answer"], prediction)
            self.assertEqual(payload["prediction_independent_review"], blind)
            self.assertEqual(payload["reference_review"]["adjudication"], adjudication)
            self.assertEqual(payload["reference_review"]["reference_proposal"], Q["gt"])
            self.assertEqual(call.call_args.kwargs["retries"], 1)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(reference, before)

    def test_declared_reference_problem_requests_reassessment_without_rewriting_prediction(self):
        output = grade_output(reference_status="contradicted", reviewed_answer="另一日期",
            reasoning="原参考被公开记录反驳，需要版本修正。")
        judge, call = self.make(output=output)
        prediction = "2024年3月20日；这不排除未收录的其他事件。"
        grade = judge(Q, prediction)
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertTrue(grade["requires_item_reassessment"])
        event = grade["item_reassessment"]
        self.assertTrue(event["required"])
        self.assertEqual(event["scope"], "same_item_all_systems")
        self.assertIn("reference_status:contradicted", event["reasons"])
        for field in ("question_hash", "reference_hash", "source_qid", "review_hash", "input_manifest_hash"):
            self.assertEqual(event[field], grade[field])
        self.assertFalse(is_scored(grade))
        self.assertEqual(json.loads(call.call_args.args[1][1]["content"])["solver_answer"], prediction)
        self.assertEqual(judge.events[-1]["output"], output)

    def test_explicit_reassessment_request_overrides_nominal_correct_label(self):
        judge, _ = self.make(output=grade_output(requires_item_reassessment=True,
            reassessment_reason="新的证据路径揭示题目时间范围有歧义。"))
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertEqual(grade["item_reassessment"]["reasons"], ["grader_requested"])
        self.assertEqual(grade["reassessment_reason"], "新的证据路径揭示题目时间范围有歧义。")

    def test_solver_error_and_additional_fact_errors_do_not_request_item_reassessment(self):
        for primary_verdict in ("correct", "incorrect"):
            output = grade_output(answer_verdict=primary_verdict,
                additional_facts={"status": "contradicted", "reason": "附言的无关会议日期错，不改变所问发布日期。"})
            output["answer_review"]["claims"].append({"claim": "另一会议也在同日", "task_role": "additional",
                "assessment": "contradicted", "evidence_indices": [], "explanation": "无关会议日期有误。"})
            judge, _ = self.make(output=output)
            grade = judge(Q, "报告3月20日发布，另有会议日期。")
            self.assertEqual(grade["verdict"], primary_verdict)
            self.assertFalse(grade["requires_item_reassessment"])
            self.assertEqual(grade["item_reassessment"]["reasons"], [])
            self.assertEqual(grade["answer_review"], output["answer_review"])
            self.assertEqual(grade["additional_facts"]["status"], "contradicted")

    def test_claim_assessment_is_audit_not_a_hard_semantic_scoring_rule(self):
        output = grade_output()
        output["answer_review"]["claims"][0].update(assessment="uncertain", explanation="此处解释不足。")
        judge, _ = self.make(output=output)
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "correct")
        self.assertEqual(grade["answer_review"]["claims"][0]["assessment"], "uncertain")
        # This deliberately inconsistent mock proves only that Python does not
        # turn a claim label into semantic truth. Real calibration must find it.

    def test_bad_answer_audit_shape_is_error_not_solver_incorrect(self):
        mutations = (
            lambda value: value.pop("answer_review"),
            lambda value: value["answer_review"].update(answer_meaning=""),
            lambda value: value["answer_review"].update(claims=[]),
            lambda value: value["answer_review"]["claims"][0].update(evidence_indices=[True]),
            lambda value: value["answer_review"]["claims"][0].update(evidence_indices=[1]),
            lambda value: value["answer_review"]["claims"][0].update(evidence_indices=[0, 0]),
            lambda value: value["answer_review"]["claims"][0].update(task_role="untyped"),
            lambda value: value.update(requires_item_reassessment="false"),
            lambda value: value.update(requires_item_reassessment=True, reassessment_reason=""),
        )
        for mutate in mutations:
            output = grade_output()
            mutate(output)
            judge, call = self.make(output=output)
            grade = judge(Q, Q["gt"])
            self.assertEqual(grade["verdict"], "error")
            self.assertIsNone(grade["correct"])
            self.assertFalse(grade["requires_item_reassessment"])
            self.assertEqual(judge.events[-1]["output"], output)
            call.assert_called_once()

    def test_v5_review_is_historical_not_implicitly_certified_for_v6(self):
        for mutate in (
            lambda value: value.update(version="semantic-question-shadow/v5"),
            lambda value: value["binding"].update(version="semantic-question-shadow/v5"),
            lambda value: value["items"][0].pop("item_certification"),
            lambda value: value["items"][0]["blind_read"].pop("major_requirements"),
            lambda value: value["items"][0]["adjudication"].pop("original_rationale_review"),
        ):
            reference = report()
            mutate(reference)
            with self.assertRaises(ValueError):
                self.make(reference=reference)

    def test_empty_reason_is_valid_only_when_no_item_reassessment_is_requested(self):
        judge, call = self.make(output=grade_output(reassessment_reason=""))
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "correct")
        self.assertFalse(grade["requires_item_reassessment"])
        self.assertEqual(grade["reassessment_reason"], "")
        self.assertTrue(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": grade}))
        call.assert_called_once()
        for reason in ("", "   ", None):
            judge, call = self.make(output=grade_output(requires_item_reassessment=True, reassessment_reason=reason))
            self.assertEqual(judge(Q, Q["gt"])["verdict"], "error")
            call.assert_called_once()

    def test_pending_readable_proposal_with_bad_locator_cannot_enter_scoring(self):
        reference = review_questions([Q], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output("blind_read",
                evidence=[{"doc_id": "d000001", "field": "content", "quote": "不存在的引文",
                           "role": "support", "explanation": "错误定位。"}])
                if step.endswith("blind_read") else review_output())
        item = reference["items"][0]
        self.assertIsNotNone(item["adjudication"])
        self.assertEqual(item["semantic"]["status"], "resolved")
        self.assertEqual(item["item_certification"]["status"], "unverified")
        judge, call = self.make(reference=reference)
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertTrue(grade["requires_item_reassessment"])
        call.assert_not_called()
        item["review_state"] = "completed"
        with self.assertRaises(ValueError):
            self.make(reference=reference)

    def test_certification_summary_cannot_mask_stage_or_location_tampering(self):
        mutations = (
            lambda item: item["stage_execution"]["blind_read"].update(status="error"),
            lambda item: item["stage_evidence_location"]["blind_read"].update(status="located", entries=[]),
            lambda item: item["evidence_location"].update(failed_stages=["blind_read"]),
            lambda item: item["semantic"].update(status="proposal_available"),
            lambda item: item["item_certification"].update(policy="old"),
            lambda item: item["original_rationale_review"].update(status="unresolved"),
        )
        for mutate in mutations:
            reference = report()
            mutate(reference["items"][0])
            with self.assertRaises(ValueError):
                self.make(reference=reference)

    def test_full_prior_review_over_budget_is_not_summarized_or_silently_dropped(self):
        reference = review_questions([Q], CORPUS, "回答日期", reviewer_model="offline-reader",
            chat_json=lambda step, messages, **kw: review_output("blind_read" if step.endswith("blind_read")
                else "adjudicate", reasoning="完整限定和反证。" * 1200))
        judge, call = self.make(reference=reference, max_input_chars=len(SYSTEM) + 2000)
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertEqual(grade["coverage"]["status"], "partial")
        self.assertFalse(grade["requires_item_reassessment"])
        self.assertEqual(judge.calls_used, 0)
        call.assert_not_called()

    def test_grader_execution_failure_does_not_invent_a_semantic_item_dispute(self):
        judge, call = self.make()
        call.side_effect = TimeoutError("offline simulated provider timeout")
        grade = judge(Q, Q["gt"])
        self.assertEqual(grade["verdict"], "error")
        self.assertEqual(grade["execution_status"], "error")
        self.assertFalse(grade["requires_item_reassessment"])
        self.assertEqual(grade["item_reassessment"]["reasons"], [])
        self.assertEqual(judge.calls_used, 1)
        call.assert_called_once()

    def test_grading_contract_rejects_flagged_or_old_audit_records(self):
        judge, _ = self.make()
        grade = judge(Q, Q["gt"])
        for mutate in (
            lambda value: value.update(requires_item_reassessment=True),
            lambda value: value.pop("requires_item_reassessment"),
            lambda value: value.pop("answer_review"),
            lambda value: value.update(version="semantic-primary-v5"),
        ):
            changed = deepcopy(grade)
            mutate(changed)
            self.assertFalse(is_scored({**Q, "pred": Q["gt"], "correct": True, "judgement": changed}))


if __name__ == "__main__":
    unittest.main()
