"""Real text tracer and renderer: durable bodies, bounded retries, strict reuse."""
from copy import deepcopy
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
from pipeline import render as renderer
from pipeline.render import (_FillerWriter, FillerExecutionError, MaterialAuthoringStopped,
                             _CorpusReviewCallGuard, _top_up_haystack)
from pipeline.run import Tracer, _atomic_write_json
from pipeline.semantic_review import fingerprint
from pipeline.corpus_contract import REVIEW_SYSTEM
from pipeline.prompts import render
from corpus_review_execution_selftest import world, opinion


def temporary_connection(kind="http_connect_error"):
    error = RuntimeError("temporary connection lost")
    error.kind, error.retryable = kind, True
    return error


class FillerResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.tracer = Tracer(SimpleNamespace(dir=self.directory))
        self.messages = [{"role": "system", "content": "Only background prose"},
                         {"role": "user", "content": "Session 1, slot 1"}]
        self.body = "外围行政记录。" * 100
        self.provider = patch.object(config, "chat", return_value=self.body).start()

    def write(self, messages=None, blocked=()):
        return _FillerWriter(self.tracer)(self.tracer, messages or self.messages, blocked)

    def records(self):
        return [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((self.directory / "05_filler_checkpoints").glob("*.json"))]

    def save_record(self, record):
        record["hash"] = fingerprint(record["state"])
        _atomic_write_json(self.directory / "05_filler_checkpoints" / (record["key"] + ".json"), record)

    def test_success_survives_new_renderer_without_a_second_call(self):
        first = self.write()
        self.assertEqual(self.records()[0]["state"]["output"], self.body)
        self.assertEqual(self.write(), first)
        self.assertEqual(self.provider.call_count, 1)

    def test_temporary_connection_retries_and_persists_each_attempt(self):
        self.provider.side_effect = [temporary_connection(), temporary_connection("http_protocol_error"), self.body]
        self.assertEqual(self.write()[0]["content"], self.body)
        self.assertEqual(len(self.records()[0]["state"]["attempts"]), 3)
        self.write()
        self.assertEqual(self.provider.call_count, 3)

    def test_three_failures_stay_exhausted_across_resume(self):
        self.provider.side_effect = temporary_connection()
        for _ in range(2):
            with self.assertRaisesRegex(FillerExecutionError, "3 total attempts"):
                self.write()
        self.assertEqual(self.provider.call_count, 3)
        self.assertEqual(len(self.records()[0]["state"]["attempts"]), 3)

    def test_interrupted_dispatch_consumes_a_slot_on_resume(self):
        self.provider.side_effect = KeyboardInterrupt("process interrupted")
        with self.assertRaises(KeyboardInterrupt):
            self.write()
        self.assertEqual(self.records()[0]["state"]["attempts"], [{"status": "admitted"}])
        self.provider.side_effect = temporary_connection()
        with self.assertRaises(FillerExecutionError):
            self.write()
        self.assertEqual(self.provider.call_count, 3)

    def test_raw_success_saved_before_final_checkpoint_update_is_reusable(self):
        self.write()
        record = self.records()[0]
        record["state"] = {"attempts": record["state"]["attempts"]}
        self.save_record(record)
        self.assertEqual(self.write()[0]["content"], self.body)
        self.assertEqual(self.provider.call_count, 1)

    def test_authentication_and_budget_stop_globally_without_retry(self):
        auth = RuntimeError("invalid key")
        auth.kind = "http_status_401"
        budget = RuntimeError("Original experiment model/call/estimated budget limit; no provider dispatch")
        for index, failure in enumerate((auth, budget)):
            self.provider.side_effect = failure
            guard = _CorpusReviewCallGuard(self.tracer, stop_global=True)
            messages = [*self.messages, {"role": "user", "content": str(index)}]
            with self.assertRaises(MaterialAuthoringStopped):
                _FillerWriter(guard)(guard, messages, ())
            with self.assertRaises(MaterialAuthoringStopped):
                guard.chat_text("render.filler", messages)
        self.assertEqual(self.provider.call_count, 2)
        self.assertTrue(all(len(record["state"]["attempts"]) == 1 for record in self.records()))

    def test_content_rejection_has_no_transport_retry(self):
        self.provider.return_value = "禁止专名的公告"
        self.assertEqual(self.write(blocked=("禁止专名",)), [])
        self.assertEqual(self.write(blocked=("禁止专名",)), [])
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(self.records()[0]["state"]["status"], "rejected")

    def test_prompt_model_and_blocked_changes_do_not_use_existing_checkpoint(self):
        self.write()
        self.write([*self.messages, {"role": "user", "content": "Changed input"}])
        with patch.object(config, "MODEL", "other-model"):
            self.write()
        self.write(blocked=("新增专名",))
        self.assertEqual(self.provider.call_count, 4)
        self.assertEqual(len(self.records()), 4)

    def make_legacy_trace(self, *, model=None, params=None, body=None):
        self.tracer.pfile.with_name("experiment_profile.json").write_text(
            json.dumps({"model": model or config.MODEL}), encoding="utf-8")
        record = {"i": 17, "step": "render.filler", "ok": True,
                  "messages": self.messages, "output": body or self.body,
                  "params": params or {"temperature": .9, "max_tokens": renderer.FILLER_TEXT_MAX_TOKENS}}
        self.tracer.pfile.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def test_exact_legacy_body_recovered_without_provider_call(self):
        self.make_legacy_trace()
        self.assertEqual(self.write()[0]["content"], self.body)
        self.provider.assert_not_called()
        state = self.records()[0]["state"]
        self.assertEqual(state["source"], "exact_prompt_trace")
        self.assertEqual(state["attempts"][0]["source_trace_index"], 17)

    def test_legacy_model_and_prompt_mismatches_do_not_reuse(self):
        self.make_legacy_trace(model="another-model")
        self.write()
        self.assertEqual(self.provider.call_count, 1)
        messages = [*self.messages, {"role": "user", "content": "Different task"}]
        self.write(messages)
        self.assertEqual(self.provider.call_count, 2)

    def test_legacy_parameter_mismatch_does_not_reuse(self):
        self.make_legacy_trace(params={"temperature": .9, "max_tokens": 100})
        self.write()
        self.assertEqual(self.provider.call_count, 1)

    def test_unknown_legacy_model_identity_does_not_reuse(self):
        self.make_legacy_trace()
        self.tracer.pfile.with_name("experiment_profile.json").write_text("{", encoding="utf-8")
        self.write()
        self.assertEqual(self.provider.call_count, 1)

    def test_legacy_connection_attempts_count_toward_the_same_three_slot_limit(self):
        self.make_legacy_trace()
        record = json.loads(self.tracer.pfile.read_text(encoding="utf-8"))
        record.update(ok=False, output={"__error__": "temporary connection", "__error_metadata__": {
            "kind": "http_connect_error", "retryable": True}})
        self.tracer.pfile.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
        self.provider.side_effect = temporary_connection()
        with self.assertRaises(FillerExecutionError):
            self.write()
        with self.assertRaises(FillerExecutionError):
            self.write()
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(len(self.records()[0]["state"]["attempts"]), 3)

    def test_changed_saved_output_hash_is_rejected_without_a_call(self):
        self.write()
        path = next((self.directory / "05_filler_checkpoints").glob("*.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["state"]["output"] = "changed body"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            self.write()
        self.assertEqual(self.provider.call_count, 1)

    def test_legacy_body_is_revalidated_against_frozen_names(self):
        self.make_legacy_trace(body="禁止专名的公告")
        self.assertEqual(self.write(blocked=("禁止专名",))[0]["content"], self.body)
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(len(self.records()[0]["state"]["attempts"]), 2)

    def test_topup_sibling_global_failure_preserves_completed_body(self):
        corpus = {"sessions": [{"session_id": 0, "date": "2026-01-01",
                               "docs": [{"doc_id": "signal", "content": "正文" * 50}]}]}
        self.provider.side_effect = [self.body,
            RuntimeError("Original experiment model/call/estimated budget limit; no provider dispatch")]
        saved = []
        with self.assertRaises(MaterialAuthoringStopped):
            _top_up_haystack(corpus, 0, 9, self.tracer, "system", (), "周",
                            lambda: saved.append(deepcopy(corpus)), lambda *_: None)
        self.assertEqual(len(corpus["sessions"][0]["docs"]), 2)
        self.assertEqual(saved[-1], corpus)
        self.assertEqual(sum(record["state"].get("status") == "accepted" for record in self.records()), 1)

    def test_real_period_resume_reuses_signal_and_successful_filler(self):
        wp = {"quality_contract": {"corpus_review": True},
              "generation_contract": {"version": "generation-first/v2", "material_first": True},
              "domain_profile": {"doc_genres": ["纪要"]}}
        def json_provider(messages, **_kwargs):
            system = messages[0]["content"]
            if system == renderer.MATERIAL_PLAN_SYSTEM:
                return {"documents": []}
            if system == render("discriminate.quality_system"):
                return {"answers": [{"key": "q0", "answer": "待接收"}]}
            if system == REVIEW_SYSTEM:
                return opinion(json.loads(messages[-1]["content"]))
            return {"docs": [{"title": "记录", "type": "纪要", "content": "测试报告的状态为待接收。"}]}
        corpus, done = {"sessions": []}, set()
        with (patch.object(config, "chat_json", side_effect=json_provider) as signal,
              patch.object(config, "pmap", side_effect=lambda fn, values, **_kwargs: [fn(v) for v in values])):
            self.provider.side_effect = [self.body,
                RuntimeError("Original experiment model/call/estimated budget limit; no provider dispatch")]
            with self.assertRaises(MaterialAuthoringStopped):
                renderer.render_corpus(wp, world(), 1600, self.tracer, corpus, done,
                                       lambda: None, lambda *_: None)
            self.assertEqual(corpus["sessions"], [])
            original_signal_calls = signal.call_count
            self.assertEqual(sum(record["state"].get("status") == "accepted" for record in self.records()), 1)
            self.provider.side_effect = None
            self.provider.return_value = self.body * 3
            renderer.render_corpus(wp, world(), 1600, self.tracer, corpus, done,
                                   lambda: None, lambda *_: None)
            self.assertEqual(signal.call_count, original_signal_calls)
            self.assertEqual(self.provider.call_count, 3)
            self.assertEqual(done, {0})
            self.assertEqual(len(corpus["sessions"][0]["docs"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
