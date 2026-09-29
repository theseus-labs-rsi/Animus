"""Offline wire, accounting and routing checks for bounded production batches."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from original_smoke_json_retry_selftest import ReplaySDK
from tools import run_original_bc_smoke as smoke


class ProductionTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def response(self, usage, body='{"ok":true}', finish="stop"):
        return fixture.SimpleNamespace(id="offline", model="gpt-5.4-mini", usage=usage,
            choices=[fixture.SimpleNamespace(index=0, finish_reason=finish,
                message=fixture.SimpleNamespace(content=body, refusal=None))])

    def test_existing_factory_receives_quantity_mode_without_small_cap(self):
        captured = []
        result = self.run_tool(lambda *_: captured.extend(sys.argv), ["--min-questions", "50",
            "--total-only", "--max-world-entities", "36", "--time-span-weeks", "12"])
        self.assertIsNone(result.error)
        self.assertNotIn("--question-budget", captured)
        self.assertEqual("50", captured[captured.index("--min-questions") + 1])
        self.assertIn("--total-only", captured)
        self.assertEqual([], result.api.calls)

    def test_usage_settlement_allows_next_call_only_after_response(self):
        usage = fixture.SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        sdk = ReplaySDK([self.response(usage)] * 2)
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            for _ in range(2):
                self.assertEqual({"ok": True}, tracer.chat_json("world.batch", [], max_tokens=4096))
        with patch.object(fixture, "FakeSDK", return_value=sdk):
            result = self.run_tool(calls, ["--settle-reported-usage", "--max-cny", "0.12"])
        self.assertIsNone(result.error)
        self.assertEqual(2, len(sdk.calls))
        self.assertEqual(2, result.profile["settled_usage_calls"])
        self.assertGreater(result.profile["reserved_upper_estimate_cny"], 0.12)
        self.assertAlmostEqual(0.000525, result.profile["budget_consumed_cny"])
        self.assertEqual(0, result.profile["unsettled_reservation_calls"])

    def test_fresh_wrapper_forwards_zero_haystack_ratio(self):
        captured = []
        result = self.run_tool(lambda *_: captured.extend(sys.argv), ["--haystack-ratio", "0"])
        self.assertIsNone(result.error)
        self.assertEqual(captured[captured.index("--haystack-ratio") + 1], "0.0")
        self.assertEqual(result.api.calls, [])

    def test_missing_usage_keeps_reservation_and_blocks_next_request(self):
        answers = []
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            for _ in range(2):
                answers.append(tracer.chat_json("world.batch", [], max_tokens=4096))
        result = self.run_tool(calls, ["--settle-reported-usage", "--max-cny", "0.12"])
        self.assertEqual(1, len(result.api.calls))
        self.assertEqual(1, result.profile["unsettled_reservation_calls"])
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", answers[-1])

    def test_billed_truncated_reply_is_settled_before_retry(self):
        usage = fixture.SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        sdk = ReplaySDK([self.response(usage, "", "length"), self.response(usage)])
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            self.assertEqual({"ok": True}, tracer.chat_json("world.batch", [], max_tokens=4096))
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--settle-reported-usage", "--max-cny", "0.20"])
        self.assertIsNone(result.error)
        self.assertEqual(2, result.profile["settled_usage_calls"])
        self.assertEqual(2, len(sdk.calls))
        self.assertEqual([call["kwargs"]["max_completion_tokens"] for call in sdk.calls], [4096, 8192])

    def test_larger_retry_reservation_is_checked_after_settling_truncated_usage(self):
        usage = fixture.SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        sdk = ReplaySDK([self.response(usage, "", "length")])
        answers = []
        def calls(directory, cfg):
            answers.append(fixture.Tracer(fixture.SimpleNamespace(dir=directory)).chat_json(
                "world.structure", [], max_tokens=4096))
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--settle-reported-usage", "--max-cny", "0.12"])
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(result.profile["settled_usage_calls"], 1)
        self.assertEqual(result.profile["unsettled_reservation_calls"], 0)
        self.assertAlmostEqual(result.profile["reported_usage_estimate_cny"], 0.0002625)
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", answers[0])

    def test_timeout_keeps_reservation_and_no_second_provider_request(self):
        sdk = ReplaySDK([TimeoutError("offline deadline")])
        def calls(directory, cfg):
            result = fixture.Tracer(fixture.SimpleNamespace(dir=directory)).chat_json("world.structure", [])
            self.assertIn("__error__", result)
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--settle-reported-usage", "--json-attempts", "5"])
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(result.profile["unsettled_reservation_calls"], 1)
        self.assertEqual(result.profile["reported_usage_estimate_cny"], 0)
        self.assertEqual(result.profile["budget_consumed_cny"], result.profile["reserved_upper_estimate_cny"])

    def test_invalid_usage_never_releases_reservation(self):
        for usage in (None, {}, {"prompt_tokens": -1, "completion_tokens": 1},
                      {"prompt_tokens": True, "completion_tokens": 1},
                      {"prompt_tokens": 10, "completion_tokens": "1"}):
            self.assertIsNone(smoke.usage_cost(usage, 3.75, 22.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
