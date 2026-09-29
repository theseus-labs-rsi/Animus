"""Offline A-policy grading path tests. Fixtures are not model accuracy evidence.

This file also runs against the unapplied staging tree: only its three owned
modules are substituted, and all provider functions are in-memory fixtures.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import threading
import unittest

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "eval/answer_task_review.py").is_file())
sys.path.insert(0, str(ROOT))
STAGED = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if STAGED != ROOT:
    import eval as package
    for name in ("grading", "semantic_judge", "reassessment"):
        setattr(package, name, load("eval." + name, STAGED / "eval" / (name + ".py")))

fixture = load("_answer_task_judge_fixture", ROOT / "tests/semantic_judge_selftest.py")
from eval import answer_task_review as reader
from eval.semantic_judge import (SemanticJudge, SYSTEM, VERSION, DIRECT_METHOD,
                                 ANSWER_TASK_JUDGE_VERSION, resolve_scoring_config)
from eval.grading import is_scored, answer_task_binding, valid_answer_task_grade
from eval.reassessment import ReassessmentTracker
from eval.provenance import record_provenance
from pipeline.semantic_review import review_questions, fingerprint
from pipeline.agent_editing import AuditRecordError

Q = {"qid": "q-public", "question": "报告发布日期是什么？", "reference_proposal": "2024-03-20"}
CORPUS = [{"content": "报告于2024年3月20日发布。", "title": "HIDDEN_TITLE", "notes": "HIDDEN_NOTES"}]
PROTOCOL = reader.append_scoring_policy("回答日期")
ANSWER = "报告于2024年3月20日发布。"


def reference(protocol=PROTOCOL):
    return review_questions([Q], CORPUS, protocol, reviewer_model="offline-reference",
        chat_json=lambda step, messages, **kw: fixture.review_output(
            "blind_read" if step.endswith("blind_read") else "adjudicate"))


def opinion():
    return {"task_interpretation": "回答报告发布日期。", "answer_meaning": "给出3月20日这一日期。",
        "assessment": "该回答符合原文。", "observations": [{"source_quote": "2024年3月20日",
            "task_relation": "主要答案。", "assessment": "原文支持。", "explanation": "日期相同。",
            "evidence": [{"doc_id": "d000001", "field": "content", "location_scope": "field",
                          "role": "support", "explanation": "正文明确日期。"}]}],
        "task_completion": "完成问题。", "strongest_alternative": "不能把其他记录日期混作发布日期。",
        "limitations": [], "evidence": [],
        "coverage": {"status": "complete", "scope_conflict": False, "limitations": []}}


def final_output(**changes):
    return fixture.grade_output(answer_task_review_response="原答给出的日期符合正文，读者所述异议不改变这项证据。", **changes)


class AnswerTaskJudgeTests(unittest.TestCase):
    def make(self, *, method=reader.VERSION, op=None, final=None, callback=None, report=None, protocol=PROTOCOL, **kw):
        self.calls = []
        def call(step, messages, **params):
            self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
            if callback:
                return callback(step, messages, **params)
            return deepcopy(opinion() if op is None else op) if step.startswith("agent_editing.") else deepcopy(
                final_output() if final is None else final)
        return SemanticJudge(reference() if report is None else report, [Q], CORPUS, protocol,
            model="offline-judge", chat_json=call, scoring_policy=reader.POLICY_VERSION,
            grading_method=method, **kw)

    def row(self, grade, answer=ANSWER):
        return {**Q, "pred": answer, "correct": grade["correct"], "judgement": grade,
                "evaluation_provenance": record_provenance(Q, grade["public_context"])}

    def test_resolver_exact_modes_without_provider_import(self):
        legacy = resolve_scoring_config()
        self.assertEqual(legacy["grade_version"], VERSION)
        self.assertEqual(legacy["final_prompt_hash"], fingerprint(SYSTEM))
        self.assertEqual(legacy["calls_per_answer"], 1)
        for method, count in ((DIRECT_METHOD, 1), (reader.VERSION, 2)):
            cfg = resolve_scoring_config(reader.POLICY_VERSION, method)
            self.assertEqual(cfg["grade_version"], ANSWER_TASK_JUDGE_VERSION)
            self.assertEqual(cfg["calls_per_answer"], count)
        for args in ((None, reader.VERSION), ("other", DIRECT_METHOD), (reader.POLICY_VERSION, "other")):
            with self.subTest(args=args), self.assertRaises(ValueError): resolve_scoring_config(*args)
        # Other suites may already have installed their config fixture. Check
        # import isolation in a fresh interpreter instead of depending on order.
        probe = subprocess.run([sys.executable, "-B", "-c",
            "import runpy,sys; d=runpy.run_path(sys.argv[1]); "
            "d['resolve_scoring_config'](d['reader'].POLICY_VERSION,d['reader'].VERSION); "
            "assert 'config' not in sys.modules; assert 'openai' not in sys.modules",
            str(HERE)], capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_legacy_message_and_grade_remain_the_original_path(self):
        # Frozen before the A-policy prompt reorganization. This guards the
        # actual historical wire string, not just equality to a new constant.
        self.assertEqual(hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
                         "add6fa3cae30c45dff942ac485574d856c6989f20f33103124a1e95e2235acf9")
        old = load("_original_semantic_judge", ROOT / "eval/semantic_judge.py") if STAGED != ROOT else None
        for klass in ([old.SemanticJudge, SemanticJudge] if old else [SemanticJudge]):
            captured = []
            judge = klass(fixture.report(), [fixture.Q], fixture.CORPUS, "回答日期", model="offline",
                chat_json=lambda step, messages, **kw: captured.append((step, messages, kw)) or fixture.grade_output())
            result = judge(fixture.Q, "2024-03-20")
            self.assertTrue(is_scored(result))
            self.assertEqual(result["version"], VERSION)
            self.assertNotIn("scoring_configuration", result)
            self.assertEqual(captured[0][1][0]["content"], SYSTEM)
            if klass is not SemanticJudge:
                original = (captured, result)
            elif old:
                self.assertEqual((captured, result), original)

    def test_exact_existing_policy_required_and_old_review_rejected(self):
        for protocol in ("回答日期", PROTOCOL + "extra", PROTOCOL + PROTOCOL):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError): self.make(protocol=protocol)
            self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError): self.make(report=reference("回答日期"))
        self.assertEqual(self.calls, [])

    def test_two_roles_exact_input_isolation_budget_and_identity(self):
        judge = self.make()
        grade = judge(Q, ANSWER)
        self.assertEqual([x["step"] for x in self.calls], ["agent_editing.independent_answer_task_review", "semantic_judge.answer"])
        self.assertEqual((judge.max_calls, judge.calls_used, judge._reserved_calls), (2, 2, 0))
        self.assertTrue(is_scored(self.row(grade)))
        public = json.loads(self.calls[0]["messages"][1]["content"])
        self.assertEqual(public["solver_answer"], ANSWER)
        self.assertEqual(public["public_protocol"], PROTOCOL)
        self.assertNotIn("reference_proposal", public)
        self.assertNotIn("HIDDEN", json.dumps(public))
        final = json.loads(self.calls[1]["messages"][1]["content"])
        self.assertEqual(final["solver_answer"], ANSWER)
        self.assertEqual(final["independent_answer_task_review"]["opinion"], opinion())
        for call in self.calls:
            self.assertEqual(call["params"], {"model": "offline-judge", "temperature": 0.0,
                "max_tokens": 4096, "retries": 1, "strict_json": True})
        self.assertEqual(grade["answer_task_review"]["raw_output"], opinion())
        self.assertEqual(grade["scoring_configuration"], judge.cache_context["scoring_configuration"])
        self.assertEqual(judge.cache_context["prompt_hash"], fingerprint(self.calls[1]["messages"][0]["content"]))

    def test_direct_A_is_one_call_and_has_separate_method_identity(self):
        direct = self.make(method=DIRECT_METHOD)
        grade = direct(Q, ANSWER)
        self.assertEqual(len(self.calls), 1)
        direct_payload = json.loads(self.calls[0]["messages"][1]["content"])
        self.assertIsNone(grade["answer_task_review"])
        self.assertTrue(is_scored(self.row(grade)))
        two = self.make()
        self.assertNotEqual(two.cache_context, direct.cache_context)
        self.assertNotIn("independent_answer_task_review", direct_payload)

    def test_A_actual_wire_selects_one_policy_before_one_complete_template(self):
        marker = "严格遵守以下完整返回 JSON Schema：\n"
        legacy_only = (
            "评分口径是题目要求的主要任务。阅读全文是为理解实际断言",
            "错误引用不自动使正确的主结论错误；先看引用/依据是否属于题目要求以及是否改变主结论。",
            "若题目要求证明且关键依据不成立，须按该公开要求评判，不把它当无关附言忽略。")
        for method in (DIRECT_METHOD, reader.VERSION):
            with self.subTest(method=method):
                judge = self.make(method=method)
                grade = judge(Q, ANSWER)
                system = self.calls[-1]["messages"][0]["content"]
                policy = reader.public_scoring_policy(reader.POLICY_VERSION)["text"]
                self.assertEqual(system.count(policy), 1)
                self.assertLess(system.index(policy), system.index("answer_review 是本次公开可审计"))
                self.assertNotIn(marker, system)
                for text in legacy_only:
                    self.assertIn(text, SYSTEM)
                    self.assertNotIn(text, system)
                template, _ = json.JSONDecoder().raw_decode("{\n" + system.split("返回 {\n", 1)[1])
                old_schema, _ = json.JSONDecoder().raw_decode(SYSTEM.split(marker, 1)[1])
                self.assertEqual(set(template), set(old_schema["required"]) | {"answer_task_review_response"})
                self.assertEqual(template["evidence"][0]["location_scope"], "field")
                self.assertEqual(set(template["answer_review"]), set(old_schema["properties"]["answer_review"]["required"]))
                self.assertEqual(fingerprint(system), judge.cache_context["prompt_hash"])
                self.assertEqual(fingerprint(system), grade["scoring_configuration"]["final_prompt_hash"])
                self.assertNotEqual(fingerprint(system),
                    "e31ea861b5d77643acb42d5ba4bff88ab05bc6f8bf47f6fa8dc23450ed6693be")

    def test_claim_role_or_assessment_does_not_programmatically_assign_A_score(self):
        # These deliberately conflicting fixtures test the transport contract,
        # not the semantic merit of any verdict or a model's accuracy.
        for role in ("primary", "additional"):
            for verdict in ("correct", "incorrect"):
                with self.subTest(role=role, verdict=verdict):
                    output = final_output(answer_verdict=verdict)
                    output["answer_review"]["claims"][0].update(
                        task_role=role, assessment="contradicted", explanation="离线固定语义意见。")
                    judge = self.make(method=DIRECT_METHOD, final=output)
                    grade = judge(Q, ANSWER)
                    self.assertEqual(grade["verdict"], verdict)
                    self.assertEqual(len(self.calls), 1)

    def test_partial_or_disputed_reader_has_no_automatic_veto(self):
        op = opinion()
        op["assessment"] = "有风险，可能错误。"
        op["coverage"] = {"status": "partial", "scope_conflict": True,
                          "limitations": ["独立读者仍有异议"], "inspected_doc_ids": ["d000001"]}
        judge = self.make(op=op)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "correct")
        self.assertTrue(is_scored(self.row(grade)))
        self.assertEqual(len(self.calls), 2)

    def test_final_unresolved_scope_is_uncertain_and_requests_reassessment(self):
        final = final_output(coverage={"status": "complete", "scope_conflict": True, "limitations": ["范围未决"]})
        judge = self.make(final=final)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertTrue(grade["requires_item_reassessment"])
        self.assertFalse(is_scored(self.row(grade)))

    def test_invalid_reader_shape_or_quote_preserves_raw_and_never_calls_final(self):
        outputs = ["bad JSON", {"__error__": "failure"}]
        for field, value in (("source_quote", "未写过这句话"), ("explanation", "")):
            op = opinion(); op["observations"][0][field] = value; outputs.append(op)
        for op in outputs:
            with self.subTest(op=op):
                judge = self.make(op=op)
                grade = judge(Q, ANSWER)
                self.assertEqual(grade["verdict"], "error")
                self.assertEqual(grade["answer_task_review"]["raw_output"], op)
                self.assertEqual((len(self.calls), judge.calls_used, judge._reserved_calls), (1, 1, 0))
                self.assertFalse(is_scored(self.row(grade)))

    def test_reader_provider_failure_retains_one_attempt(self):
        def fail(*a, **kw): raise TimeoutError("offline")
        judge = self.make(callback=fail)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "error")
        self.assertEqual(grade["answer_task_review"]["execution"]["status"], "model_error")
        self.assertEqual((judge.calls_used, judge._reserved_calls), (1, 0))

    def test_pair_admission_does_not_spend_an_orphan_reader_call(self):
        judge = self.make(max_calls=1)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertEqual((self.calls, judge.calls_used, judge._reserved_calls), ([], 0, 0))

    def test_pair_admission_is_atomic_for_concurrent_answers(self):
        entered, release = threading.Event(), threading.Event()
        def call(step, messages, **kw):
            if step.startswith("agent_editing"):
                entered.set(); self.assertTrue(release.wait(5)); return opinion()
            return final_output()
        judge = self.make(callback=call, max_calls=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(judge, Q, ANSWER)
            self.assertTrue(entered.wait(5))
            second = pool.submit(judge, Q, ANSWER).result(timeout=5)
            self.assertEqual(second["verdict"], "uncertain")
            release.set()
            self.assertEqual(first.result(timeout=5)["verdict"], "correct")
        self.assertEqual((len(self.calls), judge.calls_used, judge._reserved_calls), (2, 2, 0))

    def test_existing_pending_review_skips_both_roles(self):
        rep = reference(); rep["items"][0]["review_state"] = "pending"
        judge = self.make(report=rep)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "uncertain")
        self.assertTrue(grade["requires_item_reassessment"])
        self.assertEqual(self.calls, [])
        tracker = ReassessmentTracker(grade["public_context"], grade["review_hash"])
        self.assertTrue(tracker.observe([self.row(grade)], system="one"))

    def test_reader_audit_failure_raises_with_raw_and_releases_reservation(self):
        for stage, expected_calls in (("started", 0), ("finished", 1)):
            def fail(event):
                if event["event"] == stage: raise OSError("offline disk failure")
            judge = self.make(record=fail)
            with self.assertRaises(AuditRecordError) as caught: judge(Q, ANSWER)
            self.assertEqual((len(self.calls), judge.calls_used, judge._reserved_calls), (expected_calls, expected_calls, 0))
            self.assertEqual(caught.exception.report["raw_output"], opinion() if expected_calls else None)

    def test_final_response_contract_and_raw_preserved_on_error(self):
        for value in (None, {}, "", "   "):
            final = final_output(); final["answer_task_review_response"] = value
            judge = self.make(final=final)
            grade = judge(Q, ANSWER)
            self.assertEqual(grade["verdict"], "error")
            self.assertEqual(judge.events[-1]["output"], final)
            self.assertEqual(len(self.calls), 2)

    def test_final_provider_or_audit_failure_not_scored(self):
        def call(step, messages, **kw):
            if step.startswith("agent_editing"): return opinion()
            raise TimeoutError("final failure")
        judge = self.make(callback=call)
        self.assertEqual(judge(Q, ANSWER)["verdict"], "error")
        self.assertEqual(judge.calls_used, 2)
        def audit(event):
            if event.get("step") == "semantic_judge.answer" and event["event"] == "finished": raise OSError("disk")
        judge = self.make(record=audit)
        with self.assertRaises(OSError): judge(Q, ANSWER)
        self.assertEqual(judge.events[-1]["output"], final_output())
        self.assertEqual(judge._reserved_calls, 0)

    def test_input_budgets_do_not_truncate_or_dispatch_final_after_large_opinion(self):
        judge = self.make(max_input_chars=1)
        self.assertFalse(is_scored(judge(Q, ANSWER)))
        self.assertEqual(self.calls, [])
        huge = opinion(); huge["assessment"] = "保留完整意见。" * 10000
        judge = self.make(op=huge, max_input_chars=30000)
        grade = judge(Q, ANSWER)
        self.assertEqual(grade["verdict"], "error")
        self.assertEqual(grade["answer_task_review"]["raw_output"], huge)
        self.assertEqual((len(self.calls), judge._reserved_calls), (1, 0))

    def test_new_record_binding_rejects_answer_context_policy_method_and_opinion_tamper(self):
        judge = self.make()
        good = self.row(judge(Q, ANSWER))
        self.assertTrue(is_scored(good))
        mutations = [lambda r: r.update(pred="另一个回答"),
            lambda r: r["evaluation_provenance"].update(protocol_hash="0"*64),
            lambda r: r["judgement"]["scoring_configuration"].update(scoring_policy="other"),
            lambda r: r["judgement"]["scoring_configuration"].update(grading_method=DIRECT_METHOD),
            lambda r: r["judgement"]["answer_task_review"]["raw_output"].update(assessment="changed"),
            lambda r: r["judgement"].update(answer_task_binding={})]
        for mutate in mutations:
            row = deepcopy(good); mutate(row)
            self.assertFalse(is_scored(row))
        row = deepcopy(good)
        report = row["judgement"]["answer_task_review"]
        report["raw_output"]["observations"][0]["source_quote"] = "不在原答"
        report["opinion"] = deepcopy(report["raw_output"])
        row["judgement"]["answer_task_binding"] = answer_task_binding(row["judgement"])
        self.assertFalse(is_scored(row))
        row = deepcopy(good)
        report = row["judgement"]["answer_task_review"]
        report["inputs"]["solver_answer"] = "另一原答"
        row["judgement"]["answer_task_binding"] = answer_task_binding(row["judgement"])
        self.assertFalse(is_scored(row))

    def test_same_item_dispute_quarantines_earlier_new_contract_score(self):
        # Both systems grade against one actual reference-review artifact.
        # Calling reference() twice creates distinct reviews when latency differs.
        frozen_report = reference()
        good_judge = self.make(report=deepcopy(frozen_report)); good = self.row(good_judge(Q, ANSWER))
        disputed = final_output(requires_item_reassessment=True, reassessment_reason="题目范围未决。")
        bad_judge = self.make(report=deepcopy(frozen_report), final=disputed); bad = self.row(bad_judge(Q, ANSWER))
        tracker = ReassessmentTracker(good["judgement"]["public_context"], good["judgement"]["review_hash"])
        summary = tracker.close({"one": [good], "two": [bad]})
        self.assertEqual(summary["n_disputed_items"], 1)
        self.assertFalse(is_scored(good)); self.assertFalse(is_scored(bad))
        self.assertEqual(good["original_judgement"]["verdict"], "correct")
        self.assertEqual(good["pred"], ANSWER)

    def test_different_review_dispute_does_not_quarantine_prior_score(self):
        from itertools import count
        from unittest.mock import patch
        with patch("pipeline.semantic_review.time.monotonic", return_value=0.0):
            first = reference()
        with patch("pipeline.semantic_review.time.monotonic", side_effect=count()):
            second = reference()
        good_judge = self.make(report=first); good = self.row(good_judge(Q, ANSWER))
        disputed = final_output(requires_item_reassessment=True, reassessment_reason="新审阅范围未决。")
        bad_judge = self.make(report=second, final=disputed); bad = self.row(bad_judge(Q, ANSWER))
        self.assertNotEqual(good["judgement"]["review_hash"], bad["judgement"]["review_hash"])
        self.assertTrue(valid_answer_task_grade(bad["judgement"], bad, require_opinion=False))
        tracker = ReassessmentTracker(good["judgement"]["public_context"], good["judgement"]["review_hash"])
        summary = tracker.close({"one": [good], "two": [bad]})
        self.assertEqual(summary["n_disputed_items"], 0)
        self.assertTrue(is_scored(good)); self.assertFalse(is_scored(bad))
        self.assertNotIn("original_judgement", good)


if __name__ == "__main__":
    unittest.main()
