"""Real renderer/reviewer/tracer regressions with provider and sockets blocked."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-corpus-review")
os.environ.setdefault("MODEL", "offline-corpus-review")

import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline.corpus_contract import (CorpusReviewExecutionError, REVIEW_SYSTEM, LOCATOR_REPAIR_SYSTEM,
                                     attach_receipts, canonical_context, fingerprint,
                                     fidelity_requirements, review_documents, validate_corpus)
from pipeline.prompts import render
from pipeline.render import render_corpus
from pipeline.run import Tracer
from pipeline.world_state import Op, SET, UPDATE, Timeline, WorldState, _date_of


def world(names=("测试报告",)):
    return WorldState(entities={name: {"状态": Timeline([
        Op(0, _date_of(0), SET, "待接收")])} for name in names}, n_sessions=1)


def future_world():
    return WorldState(entities={"测试报告": {"状态": Timeline([
        Op(0, _date_of(0), SET, "待接收"),
        Op(1, _date_of(1), UPDATE, "已登记", "待接收")])}}, n_sessions=2)


def opinion(payload, claims=()):
    """Fixed test judgments over the actual short-ID request, never a real judge."""
    return {"document_reviews": fixed_document_reviews(payload["documents"],
                unsupported_indices=[claim["doc_index"] for claim in claims]),
            "verdict": "fail" if claims else "pass", "unsupported_claims": list(claims),
            "coverage": [{"requirement_id": item["requirement_id"], "status": "supported",
                          "reason": "离线桩验证执行与正文凭据绑定。",
                          "evidence": [{"doc_index": 0, "quote": payload["documents"][0]["content"]}]}
                         for item in payload["requirements"]]}


def accept_review(_step, messages, **_kwargs):
    return opinion(json.loads(messages[-1]["content"]))


class CorpusReviewExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.tracer = Tracer(SimpleNamespace(dir=self.directory))
        self.corpus, self.done, self.saved = {"sessions": []}, set(), []
        self.review_count = 0
        self.signal_count = 0
        self.review_responses = []
        self.signal_suffixes = []
        patch.object(config, "chat_json", side_effect=self.provider_json).start()
        patch.object(config, "chat", return_value="外围归档记录。").start()

    def provider_json(self, messages, **kwargs):
        system = messages[0]["content"]
        if system in (REVIEW_SYSTEM, LOCATOR_REPAIR_SYSTEM):
            self.review_count += 1
            response = self.review_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            if callable(response):
                return response(json.loads(messages[-1]["content"]))
            return deepcopy(response)
        if system == render("discriminate.quality_system"):
            return {"answers": [{"key": "q0", "answer": "待接收"}]}
        self.signal_count += 1
        suffix = self.signal_suffixes.pop(0) if self.signal_suffixes else ""
        return {"docs": [{"title": "当期记录", "type": "纪要",
                          "content": "测试报告的状态为待接收。" + suffix}]}

    def run_render(self, ws=None):
        return render_corpus(
            {"quality_contract": {"corpus_review": True},
             "domain_profile": {"doc_genres": ["纪要"]}},
            ws or world(), 0, self.tracer, self.corpus, self.done,
            save_cb=lambda: self.saved.append(deepcopy(self.corpus)), log=lambda *_: None)

    def trace(self):
        return [json.loads(line) for line in self.tracer.pfile.read_text(encoding="utf-8").splitlines()]

    def assert_stopped_without_repair(self, review_calls=1):
        self.assertEqual([record["step"] for record in self.trace()],
                         ["render.signal", "render.discriminate"] + ["corpus.review"] * review_calls)
        self.assertEqual(self.signal_count, 1)
        self.assertEqual(self.corpus, {"sessions": []})
        self.assertEqual(self.done, set())
        self.assertEqual(self.saved, [])

    def test_provider_failure_preserves_real_trace_and_stops(self):
        self.review_responses = [TimeoutError("offline provider deadline")]
        with self.assertRaises(CorpusReviewExecutionError) as caught:
            self.run_render()
        self.assert_stopped_without_repair()
        self.assertFalse(self.trace()[-1]["ok"])
        self.assertIn("offline provider deadline", caught.exception.report["raw_output"]["__error__"])
        self.assertEqual(caught.exception.report["raw_output"], self.trace()[-1]["output"])

    def test_malformed_review_keeps_raw_but_never_rewrites_content(self):
        raw = {"verdict": "pass", "unsupported_claims": [{"unexpected": "shape"}],
               "original_note": "preserve this output"}
        self.review_responses = [raw, deepcopy(raw)]
        with self.assertRaises(CorpusReviewExecutionError) as caught:
            self.run_render()
        self.assert_stopped_without_repair(review_calls=2)
        self.assertEqual(caught.exception.report["raw_output"], raw)
        self.assertEqual(self.trace()[-1]["output"], raw)
        attempts = caught.exception.report["review_validation"]["attempts"]
        self.assertEqual([item["raw_output"] for item in attempts], [raw, raw])
        self.assertTrue(all(item["validation_error"] for item in attempts))

    def test_unlocatable_review_is_execution_error_not_author_feedback(self):
        self.review_responses = [lambda payload: opinion(payload, [
            {"doc_index": 0, "quote": "原文没有这句话", "reason": "missing evidence"}]),
            {"locations": [{"path": "/unsupported_claims/0", "doc_index": 0, "quote": "原文没有这句话"}]}]
        with self.assertRaises(CorpusReviewExecutionError):
            self.run_render()
        self.assert_stopped_without_repair(review_calls=2)

    def test_evidenced_negative_review_still_repairs_and_binds_final_content(self):
        self.signal_suffixes = ["该报告获得奖项。", ""]
        self.review_responses = [
            lambda payload: opinion(payload, [
                {"doc_index": 0, "quote": "该报告获得奖项。", "reason": "CANON没有获奖记录"}]), opinion]
        self.run_render()
        self.assertEqual((self.signal_count, self.review_count), (2, 2))
        signal_calls = [record for record in self.trace() if record["step"] == "render.signal"]
        self.assertIn("CANON没有获奖记录", signal_calls[1]["user"])
        self.assertNotIn("review_error", signal_calls[1]["user"])
        self.assertEqual(validate_corpus(world(), self.corpus)["status"], "passed")
        signals = [d for s in self.corpus["sessions"] for d in s["docs"] if not d.get("is_filler")]
        self.assertEqual([d["content"] for d in signals], ["测试报告的状态为待接收。"])

    def test_completed_negative_review_exhausts_existing_content_budget(self):
        negative = lambda payload: opinion(payload, [
            {"doc_index": 0, "quote": "该报告获得奖项。", "reason": "CANON没有获奖记录"}])
        self.signal_suffixes = ["该报告获得奖项。"] * 4
        self.review_responses = [negative] * 4
        with self.assertRaises(RuntimeError) as caught:
            self.run_render()
        self.assertNotIsInstance(caught.exception, CorpusReviewExecutionError)
        self.assertEqual((self.signal_count, self.review_count), (4, 4))
        self.assertEqual(self.corpus, {"sessions": []})

    def test_concurrent_inflight_response_is_logged_but_no_followup_dispatched(self):
        names = ("甲项目", "乙项目", "丙项目")
        first_started, second_started, release_second = (threading.Event() for _ in range(3))

        def provider(messages, **kwargs):
            system = messages[0]["content"]
            if system in (REVIEW_SYSTEM, LOCATOR_REPAIR_SYSTEM):
                return {"verdict": "malformed", "original": "failed reviewer"}
            if system == render("discriminate.quality_system"):
                return {"answers": [{"key": f"q{i}", "answer": "待接收"} for i in range(2)]}
            # The shared canonical context contains all names. The per-group facts
            # occur before it, so split there to identify these two admitted calls.
            group = messages[-1]["content"].split("【生成与审阅共享")[0]
            if "甲项目" in group:
                first_started.set()
                self.assertTrue(second_started.wait(5))
                content = "甲项目的状态为待接收。乙项目的状态为待接收。"
            else:
                second_started.set()
                self.assertTrue(first_started.wait(5))
                self.assertTrue(release_second.wait(5))
                content = "丙项目的状态为待接收。"
            return {"docs": [{"content": content}]}

        def controlled_pmap(fn, items, workers=6):
            items = list(items)
            if len(items) != 2:
                return [fn(item) for item in items]
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = [pool.submit(fn, item) for item in items]
                try:
                    first.result(timeout=8)
                except CorpusReviewExecutionError:
                    release_second.set()
                    with self.assertRaises(CorpusReviewExecutionError):
                        second.result(timeout=8)
                    raise

        with patch.object(config, "chat_json", side_effect=provider), \
                patch.object(config, "pmap", side_effect=controlled_pmap):
            with self.assertRaises(CorpusReviewExecutionError):
                self.run_render(world(names))
        steps = [record["step"] for record in self.trace()]
        self.assertEqual(steps.count("render.signal"), 2)
        self.assertEqual(steps.count("render.discriminate"), 1)
        self.assertEqual(steps.count("corpus.review"), 2)  # one bounded format correction
        self.assertEqual(steps[-1], "render.signal")  # preserved in-flight response
        self.assertEqual(self.corpus, {"sessions": []})

    def test_failed_period_is_checkpointed_as_missing_while_other_periods_finish(self):
        ws = WorldState(entities={"测试报告": {"状态": Timeline([
            Op(0, _date_of(0), SET, "待接收")])}}, n_sessions=2)

        def provider(messages, **kwargs):
            system = messages[0]["content"]
            if system in (REVIEW_SYSTEM, LOCATOR_REPAIR_SYSTEM):
                payload = json.loads(messages[-1]["content"])
                if payload["CANON"]["as_of"]["session"] == 0:
                    raise TimeoutError("period-local offline provider deadline")
                return opinion(payload)
            if system == render("discriminate.quality_system"):
                return {"answers": [{"key": "q0", "answer": "待接收"}]}
            return {"docs": [{"title": "当期记录", "type": "纪要",
                              "content": "测试报告的状态为待接收。"}]}

        with patch.object(config, "chat_json", side_effect=provider):
            with self.assertRaises(CorpusReviewExecutionError):
                self.run_render(ws)

        self.assertEqual(self.done, {1})
        self.assertEqual([item["session_id"] for item in self.corpus["sessions"]], [1])
        self.assertTrue(self.saved)

    def test_as_of_world_change_invalidates_previous_receipt(self):
        ws = world()
        docs = [{"doc_id": "d1", "content": "测试报告状态为待接收。"}]
        reviewer = SimpleNamespace(chat_json=accept_review)
        report = review_documents(reviewer, ws, 0, docs, requirements=fidelity_requirements(ws, 0))
        attach_receipts(docs, report, 0)
        corpus = {"sessions": [{"session_id": 0, "docs": docs}]}
        self.assertEqual(validate_corpus(ws, corpus)["status"], "passed")
        ws.entities["测试报告"]["状态"] = Timeline([Op(0, _date_of(0), SET, "已登记")])
        self.assertIn("missing_or_stale_document_review",
                      {issue["code"] for issue in validate_corpus(ws, corpus)["issues"]})

    def test_changed_body_cannot_receive_original_review_receipt(self):
        docs = [{"title": "原标题", "content": "测试报告状态为待接收。"}]
        reviewer = SimpleNamespace(chat_json=accept_review)
        report = review_documents(reviewer, world(), 0, docs)
        for key in ("content", "title"):
            changed = deepcopy(docs)
            changed[0][key] += "更改"
            with self.assertRaises(ValueError):
                attach_receipts(changed, report, 0)

    def test_public_session_date_must_match_the_reviewed_as_of_period(self):
        ws = world()
        docs = [{"doc_id": "d1", "content": "测试报告状态为待接收。"}]
        reviewer = SimpleNamespace(chat_json=accept_review)
        attach_receipts(docs, review_documents(reviewer, ws, 0, docs,
                       requirements=fidelity_requirements(ws, 0)), 0)
        session = {"session_id": 0, "docs": docs}
        corpus = {"sessions": [session]}
        self.assertEqual(validate_corpus(ws, corpus)["status"], "passed")  # old missing-date objects
        session["date"] = _date_of(0)
        self.assertEqual(validate_corpus(ws, corpus)["status"], "passed")
        session["date"] = "2039-12-31"
        # The same wrong public date now also invalidates the fact coverage
        # group; retain the original date failure and assert both new failures.
        self.assertEqual({issue["code"] for issue in validate_corpus(ws, corpus)["issues"]},
                         {"session_date_mismatch", "invalid_fidelity_material", "missing_fidelity_material"})

    def check_review_input(self, text, response, *, hints_expected, include_fidelity=False):
        ws = future_world()
        docs = [{"doc_id": "d0", "content": text}]
        self.review_responses = [lambda payload: opinion(payload, response["unsupported_claims"])]
        report = review_documents(self.tracer, ws, 0, docs,
            requirements=fidelity_requirements(ws, 0) if include_fidelity else ())
        record = self.trace()[-1]
        self.assertEqual(record["step"], "corpus.review")
        payload = json.loads(record["user"])
        self.assertEqual(payload["documents"][0]["content"], text)
        self.assertEqual(bool(payload["future_claim_hints"]), hints_expected)
        self.assertEqual(payload["future_claim_hints"], report["diagnostics"]["future_claim_hints"])
        self.assertEqual(report["status"], "passed" if response["verdict"] == "pass" else "failed")
        self.assertEqual(self.review_count, 1)
        return ws, docs, report

    def test_reported_mistake_hint_cannot_override_semantic_review(self):
        text = "审阅员误写“测试报告状态为已登记”，这条错误记录已退回。测试报告当前状态仍为待接收。"
        ws, docs, report = self.check_review_input(
            text, {"verdict": "pass", "unsupported_claims": []}, hints_expected=True, include_fidelity=True)
        attach_receipts(docs, report, 0)
        later = [{"doc_id": "d1", "content": "测试报告状态为已登记。"}]
        self.review_responses = [opinion]
        attach_receipts(later, review_documents(self.tracer, ws, 1, later,
                       requirements=fidelity_requirements(ws, 1)), 1)
        checked = validate_corpus(ws, {"sessions": [
            {"session_id": 0, "docs": docs}, {"session_id": 1, "docs": later}]})
        self.assertEqual(checked["status"], "passed")
        self.assertTrue(checked["diagnostics"]["future_claim_hints"])
        self.assertEqual(checked["issues"], [])

    def test_prohibition_is_reviewed_in_full_despite_lexical_hit(self):
        self.check_review_input("请勿把测试报告状态登记为已登记。",
                                {"verdict": "pass", "unsupported_claims": []}, hints_expected=True)

    def test_unrelated_negation_does_not_skip_semantic_review(self):
        text = "测试报告状态为已登记；资料员否认的是收到纸质副本。"
        self.check_review_input(text, {"verdict": "fail", "unsupported_claims": [
            {"doc_index": 0, "quote": "测试报告状态为已登记", "reason": "本期只有待接收记录"}]},
            hints_expected=False)

    def test_changed_future_hint_invalidates_receipt_even_with_same_as_of_context(self):
        ws, docs, report = self.check_review_input(
            "审阅员误写“测试报告状态为已登记”，这条错误记录已退回。测试报告当前状态仍为待接收。",
            {"verdict": "pass", "unsupported_claims": []}, hints_expected=True, include_fidelity=True)
        attach_receipts(docs, report, 0)
        ws.entities["测试报告"]["状态"] = Timeline([
            Op(0, _date_of(0), SET, "待接收"),
            Op(1, _date_of(1), UPDATE, "已接收", "待接收")])
        self.assertEqual(fingerprint(canonical_context(ws, 0)), report["context_hash"])
        checked = validate_corpus(ws, {"sessions": [{"session_id": 0, "docs": docs}]})
        self.assertIn({"code": "missing_or_stale_document_review", "doc_id": "d0"}, checked["issues"])

    def test_old_lexical_policy_receipt_is_not_promoted_to_semantic_policy(self):
        docs = [{"doc_id": "d1", "content": "测试报告状态为待接收。"}]
        self.review_responses = [opinion]
        attach_receipts(docs, review_documents(self.tracer, world(), 0, docs,
                       requirements=fidelity_requirements(world(), 0)), 0)
        docs[0]["quality_review"]["version"] = 1
        checked = validate_corpus(world(), {"sessions": [{"session_id": 0, "docs": docs}]})
        self.assertIn({"code": "missing_or_stale_document_review", "doc_id": "d1"}, checked["issues"])


if __name__ == "__main__":
    unittest.main()
