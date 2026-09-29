"""Offline isolation, source attribution and one-attempt critic boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import answer_task_review as review
from pipeline.agent_editing import AuditRecordError


Q = {"qid": "q1", "question": "公开材料是否能证明办理完成？请解释。",
     "reference_proposal": {"answer": "HIDDEN_REFERENCE", "rationale": "HIDDEN_REASON"},
     "blind_read": "HIDDEN_BLIND", "previous_grade": "HIDDEN_GRADE"}
ANSWER = "无法确定。记录说拟办理，没有确认完成。"
CORPUS = [{"content": "明天拟办理，完成后另行通知。", "date": "2026-01-03",
           "title": "HIDDEN_TITLE", "author_notes": "HIDDEN_AUTHOR"}]
PROTOCOL = "只依据公开记录；不得把计划当已执行。"


def opinion():
    return {"task_interpretation": "确认是否完成并说明公开依据。",
        "answer_meaning": "回答保留未知，因为记录只记拟办。",
        "assessment": "这是可反驳的阅读意见。",
        "observations": [{"source_quote": "记录说拟办理", "task_relation": "给未知结论提供材料依据。",
            "assessment": "说明符合公开记录。", "explanation": "计划不是完成通知；不据此证明从未办理。",
            "evidence": [{"doc_id": "d000001", "field": "content", "location_scope": "field",
                          "role": "support", "explanation": "只见拟办计划。"}]}],
        "task_completion": "可以依据该范围回答未知；不补造确认记录。",
        "strongest_alternative": "可能后来办理，但材料未提供相应确认。",
        "limitations": [], "evidence": [],
        "coverage": {"status": "complete", "scope_conflict": False, "limitations": []}}


class AnswerTaskReviewTests(unittest.TestCase):
    def call(self, *, output=None, question=None, answer=ANSWER, corpus=None,
             protocol=PROTOCOL, fake=None, **kwargs):
        self.calls = []
        def invoke(step, messages, **params):
            self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
            if fake is not None:
                return fake(step, messages, **params)
            return deepcopy(opinion() if output is None else output)
        return review.review_answer_task(Q if question is None else question, answer,
            CORPUS if corpus is None else corpus, protocol, model="declared-model", chat_json=invoke, **kwargs)

    def test_only_actual_answer_and_public_task_reach_reader(self):
        before = deepcopy((Q, CORPUS))
        result = self.call()
        sent = self.calls[0]
        payload = json.loads(sent["messages"][1]["content"])
        self.assertNotIn("HIDDEN", json.dumps(result, ensure_ascii=False))
        self.assertEqual(payload["solver_answer"], ANSWER)
        self.assertEqual(payload["question"], Q["question"])
        self.assertEqual(payload["public_protocol"], review.append_scoring_policy(PROTOCOL))
        self.assertEqual((Q, CORPUS), before)
        self.assertEqual(sent["step"], "agent_editing.independent_answer_task_review")
        self.assertEqual(sent["params"], {"model": "declared-model", "temperature": 0.0,
            "max_tokens": 16384, "retries": 1, "strict_json": True})
        self.assertTrue(result["opinion_ready"])
        self.assertFalse(any(k in result for k in ("correct", "answer_verdict", "score")))

    def test_hidden_references_do_not_change_input_identity(self):
        baseline = self.call()
        question = deepcopy(Q); question["reference_proposal"] = {"answer": "another private answer"}
        changed = self.call(question=question)
        self.assertEqual(baseline["binding"], changed["binding"])
        self.assertEqual(baseline["answer_task_review_binding"], changed["answer_task_review_binding"])

    def test_actual_task_answer_protocol_and_body_each_bind_identity(self):
        original = self.call()
        changes = [dict(answer=ANSWER + " 另一种限定。"), dict(protocol=PROTOCOL + "补充范围。"),
                   dict(question={**Q, "question": "请只解释已知内容。"}),
                   dict(question={**Q, "qid": "q2"}), dict(corpus=[{"content": "后来明确完成。"}])]
        for args in changes:
            with self.subTest(args=args):
                changed = self.call(**args)
                self.assertNotEqual(original["binding"]["input_hash"], changed["binding"]["input_hash"])
                self.assertNotEqual(original["answer_task_review_binding"], changed["answer_task_review_binding"])

    def test_policy_exact_once_and_solver_critic_identity(self):
        augmented = review.append_scoring_policy(PROTOCOL)
        self.assertEqual(review.append_scoring_policy(augmented), augmented)
        policy = review.public_scoring_policy()
        self.assertEqual(policy["version"], "task-with-supporting-reasons/v1")
        self.assertEqual(augmented.count(policy["text"]), 1)
        a = self.call(protocol=PROTOCOL)
        b = self.call(protocol=augmented)
        self.assertEqual(a["binding"], b["binding"])
        self.assertEqual(a["answer_task_review_binding"]["policy_text_hash"], policy["text_hash"])
        self.assertEqual(a["public_scoring_policy"], policy)

    def test_conflicting_repeated_or_unversioned_policy_refused(self):
        augmented = review.append_scoring_policy(PROTOCOL)
        bad = [augmented + augmented, augmented.replace("task-with-supporting-reasons/v1", "other/v1"),
               augmented + "extra after frozen policy", PROTOCOL + review.POLICY_TEXT]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.call(protocol=value)
                self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError): self.call(policy_version="other/v1")
        self.assertEqual(self.calls, [])

    def test_claim_in_reference_or_question_is_not_solver_quote(self):
        for quote in ("HIDDEN_REFERENCE", Q["question"], "确定已经完成"):
            out = opinion(); out["observations"][0]["source_quote"] = quote
            result = self.call(output=out)
            self.assertFalse(result["opinion_ready"])
            self.assertEqual(result["raw_output"], out)
            self.assertEqual(result["opinion"], out)
            self.assertTrue(result["opinion_validation"]["shape_valid"])
            self.assertTrue(result["opinion_validation"]["location_errors"])

    def test_repeated_unicode_quote_retains_all_actual_locations(self):
        out = opinion(); out["observations"][0]["source_quote"] = "未确认"
        result = self.call(output=out, answer="甲未确认，乙已确认，丙未确认。")
        location = result["opinion_validation"]["observation_locations"][0]["answer_quote_location"]
        self.assertEqual(location["location_status"], "ambiguous")
        self.assertEqual([p["start"] for p in location["matching_spans"]], [1, 11])
        self.assertNotIn("start", location)
        self.assertTrue(result["opinion_ready"])

    def test_bad_public_quote_is_not_repaired_to_field(self):
        out = opinion()
        out["observations"][0]["evidence"][0].update(location_scope="quote", quote="已经完成")
        result = self.call(output=out)
        self.assertFalse(result["opinion_ready"])
        self.assertEqual(result["raw_output"], out)
        location = result["opinion_validation"]["observation_locations"][0]["evidence_location"]
        self.assertEqual(location["status"], "failed")
        self.assertEqual(location["resolved_evidence"], [])

    def test_one_failed_citation_keeps_other_successes_and_raw(self):
        out = opinion()
        out["observations"][0]["evidence"].append({"doc_id": "missing", "field": "content",
            "location_scope": "field", "role": "counterevidence", "explanation": "wrong id"})
        result = self.call(output=out)
        location = result["opinion_validation"]["observation_locations"][0]["evidence_location"]
        self.assertEqual([x["status"] for x in location["entries"]], ["located", "failed"])
        self.assertEqual(len(location["resolved_evidence"]), 1)
        self.assertEqual(result["raw_output"], out)

    def test_opinion_ready_is_not_semantic_acceptance_or_complete_coverage(self):
        out = opinion(); out["assessment"] = "该关键理由有实质错误。"
        out["task_completion"] = "任务范围尚未确定。"
        out["coverage"] = {"status": "partial", "scope_conflict": True,
                           "limitations": ["只能确认部分审查"], "inspected_doc_ids": ["d000001"]}
        result = self.call(output=out)
        self.assertTrue(result["opinion_ready"])
        self.assertEqual(result["opinion"], out)
        self.assertIn("not task completion", result["opinion_ready_scope"])

    def test_empty_answer_and_empty_observations_do_not_force_fake_claims(self):
        out = opinion(); out["observations"] = []
        out["answer_meaning"] = "原回答为空，未陈述结论。"
        result = self.call(output=out, answer="")
        self.assertTrue(result["opinion_ready"])
        self.assertEqual(result["opinion_validation"]["observation_locations"], [])

    def test_extra_explanation_or_verdict_stays_uninterpreted_raw_opinion(self):
        out = opinion()
        out.update(answer_verdict="incorrect", opinion_ready=False,
                   execution={"status": "approved"}, extra_explanation="另一种可质疑的说明。")
        out["observations"][0]["alternative_explanation"] = "保留这段额外解释。"
        result = self.call(output=out)
        self.assertTrue(result["opinion_ready"])
        self.assertEqual(result["execution"]["status"], "ok")
        self.assertEqual(result["raw_output"], out)
        self.assertEqual(result["opinion"], out)
        self.assertNotIn("answer_verdict", result)
        extra = result["opinion_validation"]["uninterpreted_fields"]
        self.assertIn("answer_verdict", extra["opinion"])
        self.assertEqual(extra["observations"], [{"observation_index": 0,
                         "fields": ["alternative_explanation"]}])

    def test_malformed_required_shape_never_promoted(self):
        variants = []
        for field, value in (("observations", "no"),
                             ("strongest_alternative", ""), ("limitations", "none"),
                             ("coverage", {"status": [], "scope_conflict": False, "limitations": []})):
            out = opinion(); out[field] = value; variants.append(out)
        for out in variants:
            with self.subTest(out=out):
                result = self.call(output=out)
                self.assertFalse(result["opinion_ready"])
                self.assertEqual(result["raw_output"], out)
                self.assertIsNone(result["opinion"])

    def test_zero_budget_and_real_input_limit_do_not_dispatch_or_truncate(self):
        for args, status in (({"max_calls": 0}, "call_budget_exhausted"),
                             ({"max_input_chars": 1}, "input_limit")):
            result = self.call(**args)
            self.assertEqual(self.calls, [])
            self.assertEqual(result["execution"]["status"], status)
            self.assertFalse(result["opinion_ready"])
            self.assertEqual(result["inputs"]["solver_answer"], ANSWER)
            self.assertEqual(result["inputs"]["documents"][0]["content"], CORPUS[0]["content"])
            self.assertEqual(result["records"][0]["event"], "skipped")

    def test_invalid_budget_cannot_request_more_than_once(self):
        for value in (-1, 2, True, 1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.call(max_calls=value)
                self.assertEqual(self.calls, [])

    def test_provider_failure_once_without_turning_answer_wrong(self):
        def fail(*a, **k): raise TimeoutError("deadline")
        result = self.call(fake=fail)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["execution"]["status"], "model_error")
        self.assertFalse(result["opinion_ready"])
        self.assertNotIn("correct", result)
        self.assertEqual(result["records"][-1]["event"], "finished")

    def test_invalid_provider_raw_and_error_object_preserved(self):
        for raw in ("not JSON", {"__error__": "provider failed"}):
            result = self.call(output=raw)
            self.assertEqual(result["raw_output"], raw)
            self.assertFalse(result["opinion_ready"])
            self.assertEqual(len(self.calls), 1)

    def test_start_record_failure_blocks_dispatch_and_keeps_audit(self):
        def fail(event): raise OSError("audit disk failure")
        with self.assertRaises(AuditRecordError) as caught: self.call(record=fail)
        result = caught.exception.report
        self.assertEqual(self.calls, [])
        self.assertEqual(result["calls_used"], 0)
        self.assertEqual(result["records"][0]["event"], "started")
        self.assertFalse(result["opinion_ready"])
        self.assertEqual(result["answer_task_review_version"], review.VERSION)

    def test_finished_record_failure_keeps_raw_but_withdraws_readiness(self):
        def fail(event):
            if event["event"] == "finished": raise OSError("audit disk failure")
        with self.assertRaises(AuditRecordError) as caught: self.call(record=fail)
        result = caught.exception.report
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["raw_output"], opinion())
        self.assertEqual(result["opinion"], opinion())
        self.assertEqual(result["execution"]["status"], "audit_error")
        self.assertFalse(result["opinion_ready"])


if __name__ == "__main__":
    unittest.main()
