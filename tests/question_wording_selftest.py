"""Offline original phrasing and release receipt tests, not semantic accuracy."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import socket
import sys
import threading
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]
from original_grounding_selftest import fixture
from pipeline.question_contract import validate_question
from pipeline.question_wording import AUTHOR_SYSTEM, review_wording, validate_wording
from pipeline.render import phrase_questions
import config


class Tracer:
    def __init__(self, responses):
        self.responses = list(responses); self.calls = []
    def chat_json(self, step, messages, **kw):
        self.calls.append((step, messages, kw))
        return self.responses.pop(0)


OK = {"verdict": "equivalent", "reason": "The intended object and time are retained.", "issues": []}


class WordingTests(unittest.TestCase):
    def setUp(self):
        blocked = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def test_natural_paraphrase_not_rejected_for_missing_exact_slots(self):
        wp, _, q, _, _ = fixture()
        tracer = Tracer([{"question": "第一期里，这份测试报告服务于哪一家客户？"}, OK])
        qs = phrase_questions([q], wp, tracer, log=lambda *a: None)
        self.assertEqual(qs[0]["question"], "第一期里，这份测试报告服务于哪一家客户？")
        self.assertEqual(validate_question(qs[0]), [])
        self.assertEqual(validate_wording(qs[0]), [])
        self.assertEqual([c[0] for c in tracer.calls], ["phrase", "phrase.review"])

    def test_lost_time_repairs_original_wording_without_rewriting_gold(self):
        wp, _, q, _, _ = fixture()
        before = deepcopy(q)
        feedback = {"verdict": "revise", "reason": "The requested first period was lost.", "issues": ["time omitted"]}
        repaired = "在第一期，这份测试报告对应的客户是哪家？"
        tracer = Tracer([{"question": "测试报告的客户是谁？"},
            feedback, {"question": repaired}, OK])
        audit = {}
        qs = phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit)
        self.assertEqual(qs[0]["question"], repaired)
        self.assertEqual(qs[0]["gt"], q["gt"])
        self.assertEqual(q, before)
        self.assertEqual(qs[0]["question_contract"], before["question_contract"])
        self.assertEqual(len(qs[0]["question_validation"]["review_history"]), 2)
        self.assertEqual(validate_wording(qs[0]), [])
        self.assertEqual([c[0] for c in tracer.calls], ["phrase", "phrase.review", "phrase", "phrase.review"])
        request = json.loads(tracer.calls[2][1][-1]["content"])
        self.assertEqual(request["original_intent"], q["question_contract"]["canonical_question"])
        self.assertEqual(request["hidden_values"], q["question_contract"]["hidden_values"])
        self.assertEqual(request["candidate_question"], "测试报告的客户是谁？")
        self.assertEqual(request["review_feedback"], feedback)
        self.assertEqual(tracer.calls[0][1][0]["content"], AUTHOR_SYSTEM)
        self.assertEqual(tracer.calls[2][1][0]["content"], AUTHOR_SYSTEM)
        self.assertEqual(audit["items"][0]["attempts"][1]["source"], "author_repair")

    def test_provider_failure_does_not_trigger_content_rewrite(self):
        wp, _, q, _, _ = fixture()
        tracer = Tracer([{"question": "题目"}, {"__error__": "transport unavailable"}])
        with self.assertRaises(RuntimeError):
            phrase_questions([q], wp, tracer, log=lambda *a: None)
        self.assertEqual(len(tracer.calls), 2)

    def test_modified_text_or_contract_invalidates_semantic_receipt(self):
        _, _, q, _, _ = fixture()
        self.assertEqual(validate_wording(q), [])
        q["question"] += " 也请预测明年。"
        self.assertEqual(validate_wording(q)[0]["code"], "missing_or_stale_wording_review")

    def test_unresolved_candidates_preserve_original_denominator_and_both_texts(self):
        wp, _, q, _, _ = fixture()
        before = deepcopy(q)
        negative = {"verdict": "unresolved", "reason": "The scope remains ambiguous.", "issues": ["scope"]}
        revised = "这份报告的服务对象是哪家客户？"
        tracer = Tracer([{"question": "这份报告服务于谁？"}, negative, {"question": revised}, negative])
        audit = {}
        self.assertEqual(phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit), [])
        self.assertEqual(q, before)
        self.assertEqual(audit["execution_status"], "completed")
        self.assertEqual(audit["original_count"], 1)
        self.assertEqual(audit["returned_qids"], [])
        item = audit["items"][0]
        self.assertEqual(item["original_order"], before)
        self.assertEqual(item["status"], "unresolved")
        self.assertFalse(item["selected"])
        self.assertEqual([a["question"] for a in item["attempts"]],
                         ["这份报告服务于谁？", revised])
        self.assertTrue(all(a["review"]["opinion"] == negative for a in item["attempts"]))
        self.assertEqual(item["candidate"]["gt"], before["gt"])
        self.assertEqual(item["candidate"]["question"], revised)
        self.assertEqual(item["author_output"], {"question": "这份报告服务于谁？"})
        self.assertEqual(item["repair_author_output"], {"question": revised})
        self.assertEqual(len(tracer.calls), 4)  # No third author attempt or silent template fallback.

    def test_repair_author_failure_keeps_first_candidate_and_raw_without_second_review(self):
        wp, _, q, _, _ = fixture()
        negative = {"verdict": "revise", "reason": "time omitted", "issues": ["time"]}
        for raw in ({"__error__": "provider failed"}, {"question": ""}, ["invalid author output"]):
            with self.subTest(raw=raw):
                tracer = Tracer([{"question": "测试报告的客户是谁？"}, negative, raw])
                audit = {}
                with self.assertRaises(RuntimeError):
                    phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit)
                row = audit["items"][0]
                self.assertEqual(row["status"], "execution_error")
                self.assertEqual(row["repair_author_output"], raw)
                self.assertEqual(row["candidate"]["question"], "测试报告的客户是谁？")
                self.assertEqual(row["attempts"][0]["review"]["opinion"], negative)
                self.assertEqual(len(row["attempts"]), 1)
                self.assertEqual([c[0] for c in tracer.calls], ["phrase", "phrase.review", "phrase"])
                self.assertEqual(audit["returned_qids"], [])

    def test_repair_review_failure_keeps_both_authored_candidates(self):
        wp, _, q, _, _ = fixture()
        negative = {"verdict": "revise", "reason": "time omitted", "issues": ["time"]}
        bad_review = {"verdict": "unknown", "reason": "provider raw"}
        tracer = Tracer([{"question": "客户是谁？"}, negative, {"question": "第一期客户是谁？"}, bad_review])
        audit = {}
        with self.assertRaises(RuntimeError):
            phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit)
        row = audit["items"][0]
        self.assertEqual([a["question"] for a in row["attempts"]], ["客户是谁？", "第一期客户是谁？"])
        self.assertEqual(row["failed_review"]["raw_output"], bad_review)
        self.assertEqual(row["attempts"][1]["review"]["status"], "execution_error")
        self.assertFalse(row["selected"])
        self.assertEqual(len(tracer.calls), 4)

    def test_canonical_looking_authored_text_is_still_revised_by_author_if_not_equivalent(self):
        wp, _, q, _, _ = fixture()
        canonical = q["question_contract"]["canonical_question"]
        negative = {"verdict": "unresolved", "reason": "Canonical intent can still be unclear.", "issues": ["scope"]}
        tracer = Tracer([{"question": canonical}, negative, {"question": "第一期该测试报告服务于哪家客户？"}, OK])
        result = phrase_questions([q], wp, tracer, log=lambda *a: None)
        self.assertEqual(len(tracer.calls), 4)
        self.assertEqual(result[0]["question"], "第一期该测试报告服务于哪家客户？")

    def test_old_reviewer_receipt_cannot_be_silently_reused(self):
        _, _, q, _, _ = fixture()
        q["question_validation"]["semantic_review"]["binding"]["version"] = "original-question-wording/v1"
        self.assertEqual(validate_wording(q), [{"code": "missing_or_stale_wording_review"}])

    def test_invalid_review_retains_original_raw_and_all_pending_candidates(self):
        wp, _, q, _, _ = fixture()
        other = deepcopy(q); other["qid"] += "_other"
        raw = {"verdict": "wrong shape", "reason": "original raw opinion"}
        tracer = Tracer([{"question": "题目"}, raw])
        audit = {}
        # The real pmap also stops returning on the first failed future. A queued
        # worker must not be allowed to dispatch after that failure was observed.
        def drain(fn, items, workers):
            errors = []
            for item in items:
                try:
                    fn(item)
                except Exception as exc:
                    errors.append(exc)
            raise errors[0]
        with patch.object(config, "pmap", side_effect=drain), self.assertRaises(RuntimeError):
            phrase_questions([q, other], wp, tracer, log=lambda *a: None, audit=audit)
        self.assertEqual(len(tracer.calls), 2)
        self.assertEqual(audit["original_count"], 2)
        self.assertEqual(audit["execution_status"], "failed")
        self.assertEqual(audit["returned_qids"], [])
        first, second = audit["items"]
        self.assertEqual(first["failed_review"]["raw_output"], raw)
        self.assertEqual(first["attempts"][0]["question"], "题目")
        self.assertEqual(first["status"], "execution_error")
        self.assertEqual(second["status"], "not_run_after_execution_failure")
        self.assertEqual(second["calls_admitted"], 0)
        self.assertEqual(second["original_order"], other)

    def test_author_error_is_audited_without_review_or_fallback(self):
        wp, _, q, _, _ = fixture()
        raw = {"__error__": "offline provider failure"}
        tracer, audit = Tracer([raw]), {}
        with self.assertRaises(RuntimeError):
            phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit)
        self.assertEqual([c[0] for c in tracer.calls], ["phrase"])
        self.assertEqual(audit["items"][0]["author_output"], raw)
        self.assertEqual(audit["items"][0]["status"], "execution_error")
        self.assertEqual(audit["items"][0]["original_order"]["gt"], q["gt"])
        # A caller may reuse its output container, but not the prior execution's
        # failure state or recorded candidates. This is a separate explicit run.
        tracer.responses = [{"question": q["question_contract"]["canonical_question"]}, OK]
        phrase_questions([q], wp, tracer, log=lambda *a: None, audit=audit)
        self.assertEqual(audit["execution_status"], "completed")
        self.assertNotIn("execution_failure", audit)
        self.assertEqual(audit["counts"], {"passed": 1})

    def test_queued_inflight_author_cannot_start_review_after_peer_failure(self):
        wp, _, q, _, _ = fixture()
        other = deepcopy(q); other["qid"] += "_other"; other["entity"] = "第二报告"
        second_started, release_second = threading.Event(), threading.Event()
        calls, returns = [], []
        class ConcurrentTracer:
            def chat_json(self, step, messages, **kwargs):
                is_second = "第二报告" in messages[-1]["content"]
                calls.append((step, is_second))
                if step == "phrase.review":
                    return {"__error__": "first reviewer failed"}
                if is_second:
                    second_started.set()
                    if not release_second.wait(5):
                        raise AssertionError("second request was not released")
                    returns.append("second author completed")
                    return {"question": "第二报告第一期服务于谁？"}
                if not second_started.wait(5):
                    raise AssertionError("second request did not start")
                return {"question": "测试报告第一期服务于谁？"}
        def pmap(fn, items, workers):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = [pool.submit(fn, item) for item in items]
                try:
                    first.result(timeout=8)
                except RuntimeError:
                    release_second.set()
                    with self.assertRaises(RuntimeError):
                        second.result(timeout=8)
                    raise
        audit = {}
        with patch.object(config, "pmap", side_effect=pmap), self.assertRaises(RuntimeError):
            phrase_questions([q, other], wp, ConcurrentTracer(), log=lambda *a: None, audit=audit)
        self.assertEqual(calls.count(("phrase", False)), 1)
        self.assertEqual(calls.count(("phrase", True)), 1)
        self.assertEqual([c for c in calls if c[0] == "phrase.review"], [("phrase.review", False)])
        self.assertEqual(returns, ["second author completed"])
        self.assertEqual(audit["items"][1]["author_output"]["question"], "第二报告第一期服务于谁？")
        self.assertEqual(audit["original_count"], 2)

    def test_legacy_template_callers_keep_their_zero_call_behavior(self):
        wp, _, q, _, _ = fixture()
        wp.pop("quality_contract")
        tracer = Tracer([])
        result = phrase_questions([q], wp, tracer, log=lambda *a: None)
        self.assertEqual(len(result), 1)
        self.assertEqual(tracer.calls, [])
        self.assertEqual(result[0]["gt"], q["gt"])
        self.assertEqual(result[0]["question_validation"]["mode"], "canonical_template")

    def test_invalid_cached_opinion_is_unresolved_without_exception(self):
        _, _, q, _, _ = fixture()
        q["question_validation"]["semantic_review"]["opinion"] = ["malformed"]
        self.assertEqual(validate_wording(q), [{"code": "unresolved_wording_review"}])


if __name__ == "__main__":
    unittest.main()
