"""Explicit provider content rejection: no same-slot retry, bounded omission, retained liability."""
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
os.environ.setdefault("OPENAI_API_KEY", "offline")
os.environ.setdefault("MODEL", "offline-test")
import config
from pipeline.render import _FillerWriter, FillerContentRejected, FillerExecutionError
from pipeline.run import Tracer
from pipeline.filler_failure import is_content_rejection
from corpus_token_render_selftest import CorpusTokenRenderTests
from tools.run_original_bc_smoke import filler_content_resume_evidence, recoverable_unit_failure
from original_smoke_text_recovery_selftest import TextRecoveryTests


def rejected():
    return {"__error__": "The request failed because the input may contain sensitive information.",
            "__error_metadata__": {"kind": "http_status_400", "call_id": "blocked"}}


class ProviderContentTests(unittest.TestCase):
    def test_only_explicit_content_rejection_is_local(self):
        error = RuntimeError("input may contain sensitive information")
        error.kind = "http_status_400"
        self.assertTrue(recoverable_unit_failure("render.filler", error))
        self.assertFalse(recoverable_unit_failure("render.signal", error))
        for raw in ({"kind": "http_status_400", "error": "invalid parameter"},
                    {"kind": "http_status_401", "error": "SensitiveContentDetected"}):
            self.assertFalse(is_content_rejection(raw))

    def test_real_writer_preserves_blocked_checkpoint_and_never_resends_it(self):
        with tempfile.TemporaryDirectory() as name, patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            tracer = Tracer(SimpleNamespace(dir=Path(name)))
            messages = [{"role": "user", "content": "background slot"}]
            with patch.object(config, "chat", return_value=rejected()) as provider:
                for _ in range(2):
                    with self.assertRaises(FillerContentRejected):
                        _FillerWriter(tracer)(tracer, messages, ())
                self.assertEqual(provider.call_count, 1)
            row = json.loads(next((Path(name) / "05_filler_checkpoints").glob("*.json")).read_text())
            self.assertEqual(row["state"]["status"], "provider_content_rejected")
            self.assertEqual(len(row["state"]["attempts"]), 1)
            self.assertEqual(row["state"]["attempts"][0]["output"], rejected())

    def test_resume_needs_actual_call_error_and_keeps_budget_unknown(self):
        profile = {"stopped": True, "admitted_calls": 1, "max_calls": 10,
                   "budget_consumed_cny": 0.1, "max_cny": 1, "model": "offline-test",
                   "unsettled_reservation_calls": 1, "stop_reason": {
                       "reason": "provider_failure", "step": "render.filler", "kind": "http_status_400", "call_id": "a"}}
        rows = [{"event": "request", "call_id": "a", "model": "offline-test"},
                {"event": "call_error", "call_id": "a", "step": "render.filler", "kind": "http_status_400",
                 "error": "SensitiveContentDetected", "local_cleanup_confirmed": True}]
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "trace.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            original = deepcopy(profile)
            result = filler_content_resume_evidence(profile, path)
            self.assertEqual(profile, original)
            self.assertEqual(result["unknown_reservations_preserved"], 1)
            for mutation in ("invalid parameter", "invalid key"):
                rows[-1]["error"] = mutation
                path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    filler_content_resume_evidence(profile, path)


class WrapperContentTests(unittest.TestCase):
    setUp = TextRecoveryTests.setUp
    run_tool = TextRecoveryTests.run_tool
    text_calls = TextRecoveryTests.text_calls

    def test_actual_wrapper_retains_failed_call_and_allows_next_background_slot(self):
        error = RuntimeError("SensitiveContentDetected")
        error.kind = "http_status_400"
        result, answers = self.text_calls([error, "unrelated background prose"])
        self.assertIsNone(result.error)
        self.assertEqual(len(result.api.calls), 2)
        self.assertFalse(result.profile["stopped"])
        self.assertEqual(result.profile["admitted_calls"], 2)
        self.assertEqual(result.profile["unsettled_reservation_calls"], 2)
        self.assertIn("__error__", answers[0])
        self.assertEqual(answers[1], "unrelated background prose")

    def test_other_400_stays_globally_closed(self):
        error = RuntimeError("invalid request parameter")
        error.kind = "http_status_400"
        result, _ = self.text_calls([error, "must not dispatch"])
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])


class TokenOmissionTests(CorpusTokenRenderTests):
    def test_rejected_only_pending_batch_does_not_stop_remaining_topup(self):
        seen = []
        fail = [True]
        def writer(tracer, messages, blocked):
            user = messages[-1]["content"]
            seen.append(user)
            if "token-1-" in user:
                if fail[0]:
                    raise RuntimeError("offline interruption before content classification")
                raise FillerContentRejected(rejected())
            return [{"content": "外围档案。" * 30}]
        with self.assertRaisesRegex(RuntimeError, "offline interruption"):
            self.expand(target=1800, writer=writer)
        self.assertEqual(len(self.state["topup"]["pending_jobs"]), 1)
        saved = deepcopy(self.corpus)
        fail[0] = False
        self.expand(target=1800, writer=writer)
        self.assertTrue(self.state["target_met"])
        self.assertEqual(len(self.state["topup"]["provider_rejected_jobs"]), 1)
        self.assertEqual(self.corpus["sessions"][0]["docs"][:2], saved["sessions"][0]["docs"])
        self.assertGreater(len(seen), 3)

    def test_rejected_slot_omitted_and_successful_sibling_preserved(self):
        seen = []
        def writer(tracer, messages, blocked):
            user = messages[-1]["content"]
            seen.append(user)
            if "token-1-" in user:
                raise FillerContentRejected(rejected())
            return [{"content": "外围档案。" * 30}]
        self.expand(target=1800, writer=writer)
        self.assertTrue(self.state["target_met"])
        self.assertEqual(len(self.state["topup"]["provider_rejected_jobs"]), 1)
        self.assertEqual(sum("token-1-" in prompt for prompt in seen), 1)
        calls = len(seen)
        self.expand(target=1800, writer=writer)
        self.assertEqual(len(seen), calls)

    def test_common_content_failure_stops_without_more_calls_after_resume(self):
        seen = []
        def writer(tracer, messages, blocked):
            seen.append(messages)
            raise FillerContentRejected(rejected())
        with self.assertRaises(FillerExecutionError):
            self.expand(target=8000, writer=writer)
        calls = len(seen)
        with self.assertRaises(FillerExecutionError):
            self.expand(target=8000, writer=writer)
        self.assertEqual(len(seen), calls)
        self.assertGreater(len(self.state["topup"]["provider_rejected_jobs"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
