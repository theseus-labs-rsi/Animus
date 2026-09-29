"""Real renderer/phrasing checkpoints with only model services replaced."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("OPENAI_API_KEY", "offline-no-network")
os.environ.setdefault("MODEL", "offline-test")

import config
from pipeline.corpus_contract import CorpusReviewExecutionError, REVIEW_SYSTEM, validate_corpus
from pipeline.render import (MATERIAL_PLAN_SYSTEM, MaterialAuthoringStopped,
                             _CorpusReviewCallGuard, phrase_questions, render_corpus)
from pipeline.run import Tracer
from pipeline.question_wording import authoring_binding, validate_authoring
from pipeline.prompts import render
from corpus_review_execution_selftest import world, future_world, opinion
from original_grounding_selftest import fixture
from question_wording_selftest import OK, Tracer as WordingTracer


PLAN = {"documents": [{"purpose": "记录报告状态", "sources": ["fact:0"],
                       "time_context": "只记录当期状态及本期公开时间"}]}


class MaterialPlanningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.tracer = Tracer(SimpleNamespace(dir=self.directory))
        self.calls = []
        self.plan_output = deepcopy(PLAN)
        self.signal_interrupt = False
        self.review_timeout = False
        patch.object(config, "chat_json", side_effect=self.provider).start()
        patch.object(config, "chat", side_effect=AssertionError("No filler expected")).start()
        self.corpus, self.done = {"sessions": []}, set()
        self.orders = None
        self.wp = {"quality_contract": {"corpus_review": True},
                   "generation_contract": {"version": "generation-first/v2", "material_first": True},
                   "domain_profile": {"doc_genres": ["纪要"]}}

    def provider(self, messages, **kwargs):
        system = messages[0]["content"]
        self.calls.append((system, deepcopy(messages), kwargs))
        if system == MATERIAL_PLAN_SYSTEM:
            if isinstance(self.plan_output, BaseException):
                raise self.plan_output
            return deepcopy(self.plan_output)
        if system == render("discriminate.quality_system"):
            return {"answers": [{"key": "q0", "answer": "待接收"}]}
        if system == REVIEW_SYSTEM:
            if self.review_timeout:
                raise TimeoutError("local review timeout")
            return opinion(json.loads(messages[-1]["content"]))
        if self.signal_interrupt:
            raise KeyboardInterrupt("after planning; author dispatch interrupted")
        return {"docs": [{"title": "当期记录", "type": "纪要", "content": "测试报告的状态为待接收。"}]}

    def run_render(self):
        render_corpus(self.wp, world(), 0, self.tracer, self.corpus, self.done,
                      save_cb=lambda: None, log=lambda *_: None, orders=self.orders)

    def checkpoint(self):
        files = list((self.directory / "05_signal_checkpoints").glob("*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

    def plan_calls(self):
        return [call for call in self.calls if call[0] == MATERIAL_PLAN_SYSTEM]

    def inherited_group(self, *, lost_reply=False):
        """A local v1-shaped checkpoint with the exact preceding author task."""
        from pipeline.semantic_review import fingerprint
        captured = []
        def capture(_directory, **kwargs):
            captured.append(kwargs)
            return None
        with patch("pipeline.signal_inheritance.inherit_progress", side_effect=capture):
            self.run_render()
        self.assertEqual(len(captured), 1)
        current_path = next((self.directory / "05_signal_checkpoints").glob("*.json"))
        state = json.loads(current_path.read_text(encoding="utf-8"))
        original = deepcopy(state)
        old_key = "e" * 64
        original["key"] = old_key
        original.pop("acceptance_binding")
        slot = original["progress"]["drafts"][0]
        request = slot.pop("author_request")
        slot.pop("subcalls", None)
        slot.pop("author_transport_binding", None)
        if lost_reply:
            for key in ("author_output", "blind_answers", "quality_review"):
                slot.pop(key, None)
            original.pop("documents")
            original.pop("hash")
        original["progress_hash"] = fingerprint(original["progress"])
        old_bytes = (json.dumps(original, ensure_ascii=False) + "\n").encode("utf-8")
        current_path.unlink()
        (current_path.parent / (old_key + ".json")).write_bytes(old_bytes)
        task = captured[0]
        output_hash = (None if lost_reply else
                       fingerprint(state["progress"]["drafts"][0]["author_output"]))
        manifest = {"version": "signal-slot-inheritance/v1", "model": config.MODEL,
            "reviewer_model": config.REVIEWER_MODEL,
            "discriminator_model": config.DISCRIMINATOR_MODEL,
            "groups": [{"key": task["key"], "legacy_key": old_key,
                "group_hash": fingerprint(task["group"]),
                "task_hash": fingerprint(task["task"]),
                "material_needs_hash": fingerprint(task["material_needs"]),
                "legacy_sha256": hashlib.sha256(old_bytes).hexdigest(),
                "drafts": [{"position": 0, "messages_hash": fingerprint(request["messages"]),
                    "parameters": request["parameters"], "saved_author_reply": not lost_reply,
                    "author_output_hash": output_hash,
                    "matching_original_physical_requests": 0 if lost_reply else 1}]}]}
        manifest["hash"] = fingerprint(manifest)
        (self.directory / "05_signal_inheritance.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.corpus, self.done = {"sessions": []}, set()
        self.calls.clear()
        return manifest

    def test_explicit_old_slot_reuses_reply_without_rebuying_author_or_review(self):
        manifest = self.inherited_group()
        self.run_render()
        self.assertEqual(self.calls, [])
        checkpoint = json.loads((self.directory / "05_signal_checkpoints" /
            (manifest["groups"][0]["key"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["progress"]["legacy_inheritance"]["new_author_positions"], 0)
        self.assertEqual(len(checkpoint["progress"]["drafts"]), 1)
        self.assertTrue(self.corpus["sessions"])

    def test_old_admitted_position_without_reply_is_not_dispatched_again(self):
        manifest = self.inherited_group(lost_reply=True)
        self.run_render()
        authors = [call for call in self.calls if call[0] not in
                   {MATERIAL_PLAN_SYSTEM, REVIEW_SYSTEM, render("discriminate.quality_system")}]
        self.assertEqual(len(authors), 1)
        self.assertEqual(authors[0][2]["temperature"], 0.2)
        checkpoint = json.loads((self.directory / "05_signal_checkpoints" /
            (manifest["groups"][0]["key"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(len(checkpoint["progress"]["drafts"]), 2)
        self.assertNotIn("author_output", checkpoint["progress"]["drafts"][0])

    def test_changed_old_author_input_refuses_inheritance_before_model_call(self):
        from pipeline.semantic_review import fingerprint
        manifest = self.inherited_group()
        manifest["groups"][0]["drafts"][0]["messages_hash"] = "0" * 64
        manifest["hash"] = fingerprint({k: v for k, v in manifest.items() if k != "hash"})
        (self.directory / "05_signal_inheritance.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete messages"):
            self.run_render()
        self.assertEqual(self.calls, [])

    def test_inherited_review_cannot_be_reused_after_model_binding_changes(self):
        self.inherited_group()
        self.run_render()
        self.corpus, self.done = {"sessions": []}, set()
        self.calls.clear()
        with patch.object(config, "REVIEWER_MODEL", "different-reviewer"):
            with self.assertRaisesRegex(ValueError, "explicit migration"):
                self.run_render()
        self.assertEqual(self.calls, [])

    def test_unknown_task_in_inheritance_manifest_does_not_open_new_slots(self):
        from pipeline.semantic_review import fingerprint
        manifest = self.inherited_group()
        manifest["groups"][0]["key"] = "0" * 64
        manifest["hash"] = fingerprint({k: v for k, v in manifest.items() if k != "hash"})
        (self.directory / "05_signal_inheritance.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No uniquely inherited slot"):
            self.run_render()
        self.assertEqual(self.calls, [])

    def test_manifest_rejects_a_legacy_path_outside_checkpoint_store(self):
        from pipeline.semantic_review import fingerprint
        manifest = self.inherited_group()
        manifest["groups"][0]["legacy_key"] = "../../.env"
        manifest["hash"] = fingerprint({k: v for k, v in manifest.items() if k != "hash"})
        (self.directory / "05_signal_inheritance.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "historical slot more than once"):
            self.run_render()
        self.assertEqual(self.calls, [])

    def test_plan_is_single_physical_request_and_body_has_plan(self):
        self.run_render()
        self.assertEqual(len(self.plan_calls()), 1)
        self.assertEqual(self.plan_calls()[0][2]["retries"], 1)
        self.assertTrue(self.plan_calls()[0][2]["strict_json"])
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "planned")
        self.assertEqual(validate_corpus(world(), self.corpus)["status"], "passed")
        author = next(call for call in self.calls if call[0] not in
                      {MATERIAL_PLAN_SYSTEM, REVIEW_SYSTEM, render("discriminate.quality_system")})
        self.assertIn("记录报告状态", author[1][-1]["content"])

    def test_plan_receives_relevant_order_needs_without_private_answers(self):
        base = {"line": "L1_timeline", "capability": "IE", "entity": "测试报告", "field": "状态",
                "evidence_sessions": [0], "gt": "PRIVATE_GOLD",
                "aux": {"private": "PRIVATE_AUX"}, "canonicalanswer": "PRIVATE_CANONICAL",
                "privatewitness": "PRIVATE_WITNESS",
                "question_contract": {"capability_purpose": "在当期公开记录中定位状态",
                                      "gold": "PRIVATE_CONTRACT_GOLD"}}
        self.orders = [base, {**base, "entity": "另一报告"}, {**base, "evidence_sessions": [1]}]
        self.run_render()
        payload = json.loads(self.plan_calls()[0][1][-1]["content"])
        self.assertEqual(payload["material_needs"], [{"line": "L1_timeline", "capability": "IE",
            "entity": "测试报告", "field": "状态", "evidence_sessions": [0],
            "capability_purpose": "在当期公开记录中定位状态"}])
        self.assertNotIn("PRIVATE_", self.plan_calls()[0][1][-1]["content"])
        self.assertEqual(len(self.plan_calls()), 1)

    def test_changed_order_need_invalidates_planning_checkpoint(self):
        self.orders = [{"line": "L1_timeline", "capability": "IE", "entity": "测试报告",
                        "field": "状态", "evidence_sessions": [0]}]
        self.run_render()
        self.corpus, self.done = {"sessions": []}, set()
        self.orders[0]["capability"] = "TR"
        self.run_render()
        self.assertEqual(len(self.plan_calls()), 2)
        self.assertEqual(len(list((self.directory / "05_signal_checkpoints").glob("*.json"))), 2)
        changed = json.loads(self.plan_calls()[-1][1][-1]["content"])
        self.assertEqual(changed["material_needs"][0]["capability"], "TR")

    def test_malformed_plan_falls_back_without_extra_call(self):
        self.plan_output = {"documents": "wrong"}
        self.run_render()
        self.assertEqual(len(self.plan_calls()), 1)
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "fallback")
        self.assertTrue(self.corpus["sessions"])

    def test_unknown_source_falls_back_without_inventing_business_value(self):
        self.plan_output["documents"][0]["sources"] = ["fact:999"]
        self.run_render()
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "fallback")
        self.assertEqual(validate_corpus(world(), self.corpus)["status"], "passed")

    def test_optional_local_timeout_falls_back(self):
        self.plan_output = TimeoutError("optional planning timed out")
        self.run_render()
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "fallback")

    def test_budget_failure_stops_before_body(self):
        self.plan_output = RuntimeError("Original experiment model/call/estimated budget limit; no provider dispatch")
        with self.assertRaises(RuntimeError):
            self.run_render()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.corpus, {"sessions": []})
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "failed")

    def test_authentication_failure_stops_before_body(self):
        error = RuntimeError("invalid key")
        error.kind = "http_status_401"
        self.plan_output = error
        with self.assertRaises(RuntimeError):
            self.run_render()
        self.assertEqual(len(self.calls), 1)

    def test_global_guard_blocks_json_and_filler_after_auth_failure(self):
        class Provider:
            def __init__(self):
                self.calls = []
            def chat_json(self, *args, **kwargs):
                self.calls.append("json")
                return {"__error__": "invalid key", "__error_metadata__": {"kind": "http_status_401"}}
            def chat_text(self, *args, **kwargs):
                self.calls.append("text")
                return "new filler"
        provider = Provider()
        guard = _CorpusReviewCallGuard(provider, stop_global=True)
        with self.assertRaises(MaterialAuthoringStopped):
            guard.chat_json("render.signal", [])
        with self.assertRaises(MaterialAuthoringStopped):
            guard.chat_json("corpus.review", [])
        with self.assertRaises(MaterialAuthoringStopped):
            guard.chat_text("render.filler", [])
        self.assertEqual(provider.calls, ["json"])

    def test_parallel_business_failure_does_not_mask_later_global_stop(self):
        original_provider = self.provider

        def provider(messages, **kwargs):
            if messages[0]["content"] == MATERIAL_PLAN_SYSTEM:
                payload = json.loads(messages[-1]["content"])
                if payload["session"] == 1:
                    raise RuntimeError("Original experiment model/call/estimated budget limit; no provider dispatch")
            if messages[0]["content"] == REVIEW_SYSTEM:
                payload = json.loads(messages[-1]["content"])
                return opinion(payload, [{"doc_index": 0,
                    "quote": payload["documents"][0]["content"], "reason": "offline negative business opinion"}])
            return original_provider(messages, **kwargs)

        def drain_map(function, values, **kwargs):
            # A ThreadPoolExecutor drains admitted siblings before rethrowing
            # its first failed future. Keep that ordering deterministic here.
            first_failure = None
            result = []
            for value in values:
                try:
                    result.append(function(value))
                except Exception as error:
                    first_failure = first_failure or error
            if first_failure is not None:
                raise first_failure
            return result

        with patch.object(config, "chat_json", side_effect=provider), \
                patch.object(config, "pmap", side_effect=drain_map):
            with self.assertRaises(MaterialAuthoringStopped) as caught:
                render_corpus(self.wp, future_world(), 0, self.tracer, self.corpus, self.done,
                              save_cb=lambda: None, log=lambda *_: None)
        self.assertIn("estimated budget limit", str(caught.exception))
        self.assertIn("4 轮", str(caught.exception.__cause__))
        self.assertEqual(self.corpus, {"sessions": []})
        self.assertEqual(self.done, set())
        checkpoints = [json.loads(x.read_text(encoding="utf8")) for x in
                       (self.directory / "05_signal_checkpoints").glob("*.json")]
        self.assertTrue(any(len(x["progress"].get("drafts", [])) == 4 for x in checkpoints))

    def test_parallel_business_failure_keeps_original_error_without_global_stop(self):
        def drain_map(function, values, **kwargs):
            result = [function(value) for value in values]
            raise RuntimeError("original period quality failure")

        with patch.object(config, "pmap", side_effect=drain_map):
            with self.assertRaisesRegex(RuntimeError, "original period quality failure") as caught:
                self.run_render()
        self.assertNotIsInstance(caught.exception, MaterialAuthoringStopped)

    def test_interrupted_plan_does_not_receive_another_allowance(self):
        self.plan_output = KeyboardInterrupt("lost planning response")
        with self.assertRaises(KeyboardInterrupt):
            self.run_render()
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "admitted")
        self.plan_output = deepcopy(PLAN)
        self.run_render()
        self.assertEqual(len(self.plan_calls()), 1)
        self.assertEqual(self.checkpoint()["progress"]["planning"]["status"], "fallback")

    def test_interrupted_body_reuses_plan_and_consumes_original_slot(self):
        self.signal_interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_render()
        self.signal_interrupt = False
        self.run_render()
        self.assertEqual(len(self.plan_calls()), 1)
        drafts = self.checkpoint()["progress"]["drafts"]
        self.assertEqual(len(drafts), 2)
        self.assertNotIn("author_output", drafts[0])
        self.assertIn("author_output", drafts[1])

    def test_review_failure_resume_retains_body_and_blind_read(self):
        self.review_timeout = True
        with self.assertRaises(CorpusReviewExecutionError):
            self.run_render()
        first = len(self.calls)
        self.review_timeout = False
        with self.assertRaises(CorpusReviewExecutionError):
            self.run_render()
        self.assertEqual(len(self.calls), first)
        self.assertEqual(len(self.plan_calls()), 1)

    def test_complete_group_resume_performs_no_model_calls(self):
        self.run_render()
        before = len(self.calls)
        self.corpus, self.done = {"sessions": []}, set()
        self.run_render()
        self.assertEqual(len(self.calls), before)

    def test_four_draft_slots_cannot_reset_on_resume(self):
        self.signal_interrupt = True
        for _ in range(4):
            with self.assertRaises(KeyboardInterrupt):
                self.run_render()
        before = len(self.calls)
        self.signal_interrupt = False
        with self.assertRaises(RuntimeError):
            self.run_render()
        self.assertEqual(len(self.calls), before)
        self.assertEqual(len(self.checkpoint()["progress"]["drafts"]), 4)


class MaterialQuestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "questions.ckpt.json"
        blocked = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        blocked.start()
        self.addCleanup(blocked.stop)
        self.wp, _, self.order, self.corpus, self.protocol = fixture()
        self.author_output = {"question": "第一期里，这份测试报告服务于哪一家客户？"}

    def phrase(self, tracer, corpus=None, protocol=None, order=None, audit=None):
        return phrase_questions([order or self.order], self.wp, tracer, log=lambda *_: None,
            checkpoint_path=self.path, corpus=self.corpus if corpus is None else corpus,
            public_protocol=self.protocol if protocol is None else protocol, audit=audit)

    def test_real_body_and_public_protocol_are_in_author_request(self):
        tracer = WordingTracer([self.author_output, OK])
        result = self.phrase(tracer)
        payload = json.loads(tracer.calls[0][1][-1]["content"])
        self.assertEqual(payload["public_protocol"], self.protocol)
        self.assertTrue(payload["material_context"]["documents"])
        self.assertEqual(result[0]["gt"], self.order["gt"])
        self.assertEqual(validate_authoring(result[0], self.corpus, self.protocol), [])
        self.assertFalse(result[0]["question_validation"]["authoring"]["semantic_sufficiency_verified"])

    def test_missing_material_note_adds_no_gate_or_author_retry(self):
        tracer = WordingTracer([self.author_output, OK])
        result = self.phrase(tracer)
        self.assertEqual(len(result), 1)
        self.assertEqual([call[0] for call in tracer.calls], ["phrase", "phrase.review"])
        self.assertEqual(result[0]["question_validation"]["authoring"]["gap_kind"], "author_note_missing")

    def test_new_public_evidence_keeps_words_and_old_read_history(self):
        first = self.phrase(WordingTracer([self.author_output, OK]))
        changed = deepcopy(self.corpus)
        changed["corpus"]["sessions"][0]["docs"].append({"content": "新增公开反证，原结论需要重新审查。"})
        tracer = WordingTracer([])
        audit = {}
        second = self.phrase(tracer, corpus=changed, audit=audit)
        self.assertEqual(first, second)
        self.assertEqual(tracer.calls, [])
        old = second[0]["question_validation"]["authoring"]
        self.assertNotEqual(old["corpus_hash"], audit["source_binding"]["corpus_hash"])
        self.assertEqual(validate_authoring(second[0], changed, self.protocol), [])

    def test_internal_metadata_does_not_change_public_binding(self):
        before = authoring_binding(self.corpus, self.protocol)
        changed = deepcopy(self.corpus)
        changed["internal_audit"] = {"new": True}
        changed["corpus"]["sessions"][0]["docs"][0]["private_receipt"] = "changed"
        self.assertEqual(before, authoring_binding(changed, self.protocol))

    def test_reference_change_reauthors_candidate(self):
        self.phrase(WordingTracer([self.author_output, OK]))
        changed = deepcopy(self.order)
        changed["gt"] = "另一客户"
        tracer = WordingTracer([self.author_output, OK])
        result = self.phrase(tracer, order=changed)
        self.assertEqual(len(tracer.calls), 2)
        self.assertEqual(result[0]["gt"], "另一客户")

    def test_protocol_change_reauthors_candidate(self):
        self.phrase(WordingTracer([self.author_output, OK]))
        changed = self.protocol + "新公开规则"
        tracer = WordingTracer([self.author_output, OK])
        self.phrase(tracer, protocol=changed)
        self.assertEqual(len(tracer.calls), 2)

    def test_empty_note_is_not_evidence_sufficiency_claim(self):
        result = self.phrase(WordingTracer([self.author_output, OK]))
        receipt = result[0]["question_validation"]["authoring"]
        self.assertEqual(receipt["status"], "not_assessed")
        self.assertFalse(receipt["semantic_sufficiency_verified"])

    def test_completed_resume_is_zero_call(self):
        self.phrase(WordingTracer([self.author_output, OK]))
        tracer = WordingTracer([])
        self.assertEqual(len(self.phrase(tracer)), 1)
        self.assertEqual(tracer.calls, [])

    def test_completed_author_is_saved_before_interrupted_review(self):
        class Interrupting(WordingTracer):
            def chat_json(self, step, messages, **kwargs):
                if step == "phrase.review":
                    raise KeyboardInterrupt("review interrupted")
                return super().chat_json(step, messages, **kwargs)
        first = Interrupting([self.author_output])
        with self.assertRaises(KeyboardInterrupt):
            self.phrase(first)
        self.assertEqual(len(first.calls), 1)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        entries = next(iter(saved["items"].values()))["row"]["call_cache"].values()
        self.assertEqual(sum("output" in entry for entry in entries), 1)
        # Unknown review dispatch remains pending. It cannot silently gain an
        # extra retry on each automatic resume or cause a repeat author call.
        resumed = WordingTracer([])
        with self.assertRaisesRegex(RuntimeError, "No question wording completed"):
            self.phrase(resumed)
        self.assertEqual(resumed.calls, [])

    def test_author_read_allowance_stays_bounded_and_resume_does_not_reset(self):
        read = {"action": "read", "requests": [{"doc_id": "d000001", "start": 0}]}
        tracer = WordingTracer([read, read, read])
        with self.assertRaisesRegex(RuntimeError, "No question wording completed"):
            self.phrase(tracer)
        self.assertEqual(len(tracer.calls), 3)
        resumed = WordingTracer([])
        with self.assertRaisesRegex(RuntimeError, "No question wording completed"):
            self.phrase(resumed)
        self.assertEqual(resumed.calls, [])

    def test_budget_failure_is_global_and_preserves_original_order(self):
        tracer = WordingTracer([{"__error__": "Original experiment model/call/estimated budget limit; no provider dispatch",
                                "__error_metadata__": {"kind": "RuntimeError"}}])
        before = deepcopy(self.order)
        with self.assertRaises(RuntimeError):
            self.phrase(tracer)
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(self.order, before)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(next(iter(saved["items"].values()))["row"]["failure"])


if __name__ == "__main__":
    unittest.main()
