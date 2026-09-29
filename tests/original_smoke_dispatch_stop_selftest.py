"""Deterministic real-wrapper queued dispatch tests with fake SDK and no network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke
fixture.smoke = smoke
Tracer, SimpleNamespace = fixture.Tracer, fixture.SimpleNamespace


class ObservedLock:
    """Observe an actual waiter without adding timing sleeps or parallel SDK use."""
    def __init__(self):
        self.lock = threading.RLock()
        self.waiting = threading.Event()
        self.owner = None
        self.depth = 0
        self.reentries = 0

    def __enter__(self):
        ident = threading.get_ident()
        if self.owner is not None and self.owner != ident:
            self.waiting.set()
        if self.owner == ident:
            self.reentries += 1
        self.lock.acquire()
        self.owner = ident
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        if not self.depth:
            self.owner = None
        self.lock.release()


class QueuedSDK(fixture.FakeSDK):
    def __init__(self, outcome):
        super().__init__()
        self.outcome = outcome
        self.entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()

    def with_options(self, **options):
        def create(**kwargs):
            with self.lock:
                self.calls.append({"options": deepcopy(options), "kwargs": deepcopy(kwargs)})
                index = len(self.calls)
            if index == 1:
                self.entered.set()
                if not self.release_first.wait(timeout=5):
                    raise AssertionError("Test did not release first SDK call")
                if self.outcome == "provider_error":
                    raise RuntimeError("Offline queued provider failure")
                if self.outcome == "empty_length":
                    return SimpleNamespace(id="empty", model=kwargs["model"], usage=None,
                        choices=[SimpleNamespace(index=0, finish_reason="length",
                            message=SimpleNamespace(content="", refusal=None))])
                if self.outcome in ("malformed_json", "semantic_fail"):
                    content = '{"ok":true}}' if self.outcome == "malformed_json" else '{"verdict":"fail"}'
                    return SimpleNamespace(id="returned-body", model=kwargs["model"], usage=None,
                        choices=[SimpleNamespace(index=0, finish_reason="stop",
                            message=SimpleNamespace(content=content, refusal=None))])
            if index == 2:
                self.second_entered.set()
            return SimpleNamespace(id="ok", model=kwargs["model"], usage=None,
                choices=[SimpleNamespace(index=0, finish_reason="stop",
                    message=SimpleNamespace(content='{"ok":true}', refusal=None))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class DispatchStopTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def exercise_queue(self, outcome, *, raw_first=False, attempts=1):
        api = QueuedSDK(outcome)
        observed = {}
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = (pool.submit(cfg.chat, [], max_tokens=4096) if raw_first else
                         pool.submit(tracer.chat_json, "world.semantic_review", [], max_tokens=4096))
                self.assertTrue(api.entered.wait(timeout=3))
                second = pool.submit(tracer.chat_json, "world.disclosure_plan", [], max_tokens=4096)
                try:
                    self.assertTrue(api.second_entered.wait(timeout=3),
                                    "Second caller must dispatch while the first is in flight")
                    profile = json.loads((directory / "experiment_profile.json").read_text(encoding="utf-8"))
                    self.assertEqual((len(api.calls), profile["admitted_calls"]), (2, 2))
                    observed["first_reservation"] = profile["reserved_upper_estimate_cny"]
                finally:
                    api.release_first.set()
                try:
                    first_answer = first.result(timeout=5)
                except RuntimeError as exc:
                    if not raw_first:
                        raise
                    first_answer = exc
                observed["answers"] = [first_answer, second.result(timeout=5)]
        with patch.object(fixture, "FakeSDK", return_value=api):
            result = self.run_tool(calls, ["--world-review-reasoning-effort", "medium",
                "--json-attempts", str(attempts), "--llm-concurrency", "2"])
        return result, observed

    def test_first_provider_failure_closes_future_admission_but_keeps_inflight_result(self):
        result, observed = self.exercise_queue("provider_error")
        self.assertIsNone(result.error)
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"], len(result.requests)), (2,2,2))
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", observed["answers"][0])
        self.assertEqual(observed["answers"][1], {"ok": True})
        self.assertEqual(result.requests[0]["step"], "world.semantic_review")

    def test_local_truncation_preserves_failed_call_and_allows_next_bounded_unit(self):
        result, observed = self.exercise_queue("empty_length")
        self.assertIsNone(result.error)
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"], len(result.requests)), (2,2,2))
        self.assertFalse(result.profile["stopped"])
        self.assertEqual(result.profile["reserved_upper_estimate_cny"], observed["first_reservation"])
        self.assertIn("__error__", observed["answers"][0])
        self.assertEqual(observed["answers"][1], {"ok": True})

    def test_local_malformed_json_preserves_failure_without_blocking_other_unit(self):
        result, observed = self.exercise_queue("malformed_json")
        self.assertIsNone(result.error)
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"], len(result.requests)), (2,2,2))
        self.assertFalse(result.profile["stopped"])
        self.assertEqual(result.profile["reserved_upper_estimate_cny"], observed["first_reservation"])
        self.assertIn("__error__", observed["answers"][0])
        self.assertEqual(observed["answers"][1], {"ok": True})
        responses = [row for row in result.rows if row.get("event") == "response"]
        self.assertEqual(responses[0]["response"]["choices"][0]["finish_reason"], "stop")
        errors = [row for row in result.rows if row.get("event") == "json_error"]
        self.assertTrue(any(row.get("error_type") == "JSONDecodeError" and
                            row.get("raw_output") == '{"ok":true}}' for row in errors))
        self.assertTrue(all(row["max_attempts"] == 1 for row in result.rows if row.get("event") == "json_attempt"))

    def test_recoverable_empty_reply_finishes_retry_before_admitting_waiting_request(self):
        result, observed = self.exercise_queue("empty_length", attempts=3)
        self.assertIsNone(result.error)
        self.assertEqual(observed["answers"], [{"ok": True}, {"ok": True}])
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"]), (3, 3))
        self.assertEqual(sorted(r["step"] for r in result.requests),
            sorted(["world.semantic_review", "world.semantic_review", "world.disclosure_plan"]))
        self.assertFalse(result.profile["stopped"])

    def test_semantic_negative_json_is_not_an_execution_error_or_stop(self):
        result, observed = self.exercise_queue("semantic_fail")
        self.assertIsNone(result.error)
        self.assertEqual(observed["answers"], [{"verdict":"fail"}, {"ok":True}])
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"]), (2,2))
        self.assertFalse(result.profile["stopped"])

    def test_raw_chat_failure_stops_future_admission_after_inflight_request(self):
        result, observed = self.exercise_queue("provider_error", raw_first=True)
        self.assertIsNone(result.error)
        self.assertIsInstance(observed["answers"][0], RuntimeError)
        self.assertEqual(observed["answers"][1], {"ok": True})
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"]), (2,2))
        self.assertTrue(result.profile["stopped"])

    def test_successful_first_call_admits_second_on_its_turn_with_separate_step_transport(self):
        result, observed = self.exercise_queue("success")
        self.assertIsNone(result.error)
        self.assertEqual(observed["answers"], [{"ok":True}, {"ok":True}])
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"], len(result.requests)), (2,2,2))
        self.assertFalse(result.profile["stopped"])
        self.assertAlmostEqual(result.profile["reserved_upper_estimate_cny"], observed["first_reservation"])
        self.assertEqual([r["kwargs"]["reasoning_effort"] for r in result.api.calls], ["medium", "low"])
        self.assertTrue(all(r["options"] == {"timeout":150, "max_retries":0} for r in result.api.calls))
        self.assertIsNone(smoke._TRACER_STEP.get())


if __name__ == "__main__":
    unittest.main(verbosity=2)
