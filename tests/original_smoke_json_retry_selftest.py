"""Actual bounded wrapper retries format errors while accounting every dispatch."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke
fixture.smoke = smoke
Tracer, SimpleNamespace = fixture.Tracer, fixture.SimpleNamespace


class ReplaySDK(fixture.FakeSDK):
    def __init__(self, bodies):
        super().__init__()
        self.bodies = list(bodies)

    def with_options(self, **options):
        def create(**kwargs):
            with self.lock:
                self.calls.append({"options": deepcopy(options), "kwargs": deepcopy(kwargs)})
                if not self.bodies:
                    raise AssertionError("Unexpected extra provider call")
                body = self.bodies.pop(0)
            if isinstance(body, Exception):
                raise body
            if isinstance(body, SimpleNamespace):
                return body
            return SimpleNamespace(id="offline", model=kwargs["model"], usage=None,
                choices=[SimpleNamespace(index=0, finish_reason="stop",
                    message=SimpleNamespace(content=body, refusal=None))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class RetryTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def exercise(self, bodies, *, args=(), followup=False):
        sdk = ReplaySDK(bodies)
        answers = []
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            answers.append(tracer.chat_json("world.batch", [{"role": "user", "content": "Return JSON"}],
                                           max_tokens=4096, response_format={"type": "json_object"}))
            if followup:
                answers.append(tracer.chat_json("world.structure", [], max_tokens=4096))
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, list(args))
        return result, answers

    def test_default_retries_then_succeeds_and_preserves_trace(self):
        result, answers = self.exercise(['{"entities":', '{"entities":[]}'])
        self.assertEqual(answers, [{"entities": []}])
        self.assertEqual(result.profile["json_attempts"], 3)
        self.assertEqual(result.profile["admitted_calls"], 2)
        self.assertFalse(result.profile["stopped"])
        self.assertEqual(len(result.api.calls), 2)
        self.assertEqual(len([r for r in result.rows if r.get("event") == "json_error"]), 1)
        self.assertEqual(len([r for r in result.rows if r.get("event") == "json_result"]), 1)
        self.assertGreater(len(result.api.calls[1]["kwargs"]["messages"]),
                           len(result.api.calls[0]["kwargs"]["messages"]))

    def test_three_failures_stop_followup_without_fourth_dispatch(self):
        result, answers = self.exercise(['{"bad":'] * 3, followup=True)
        self.assertEqual(result.profile["admitted_calls"], 3)
        self.assertTrue(result.profile["stopped"])
        self.assertTrue(all("__error__" in answer for answer in answers))
        self.assertEqual(len(result.requests), 3)

    def test_call_budget_is_rechecked_before_retry(self):
        result, answers = self.exercise(['{"bad":'], args=("--max-calls", "1"))
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", answers[0])

    def test_semantic_rejection_is_returned_without_retry(self):
        result, answers = self.exercise(['{"verdict":"fail"}'])
        self.assertEqual(answers, [{"verdict": "fail"}])
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertFalse(result.profile["stopped"])

    def test_empty_length_recovers_and_next_logical_request_can_dispatch(self):
        empty = SimpleNamespace(id="empty", model="gpt-5.4-mini", usage=None,
            choices=[SimpleNamespace(index=0, finish_reason="length",
                message=SimpleNamespace(content="", refusal=None))])
        result, answers = self.exercise([empty, '{"ok":true}', '{"next":true}'], followup=True)
        self.assertEqual(answers, [{"ok": True}, {"next": True}])
        self.assertEqual(result.profile["admitted_calls"], 3)
        self.assertFalse(result.profile["stopped"])
        self.assertFalse(smoke._JSON_REQUEST_ACTIVE.get())

    def test_connection_retries_are_counted_and_can_recover(self):
        result, answers = self.exercise([ConnectionError("offline reset"), '{"ok":true}'])
        self.assertEqual(answers, [{"ok": True}])
        self.assertEqual(result.profile["admitted_calls"], 2)
        self.assertFalse(result.profile["stopped"])

    def test_transient_exhaustion_stops_followup(self):
        result, answers = self.exercise([ConnectionError("offline reset")] * 3, followup=True)
        self.assertEqual(result.profile["admitted_calls"], 3)
        self.assertTrue(result.profile["stopped"])
        self.assertTrue(all("__error__" in answer for answer in answers))
        self.assertFalse(smoke._JSON_REQUEST_ACTIVE.get())

    def test_transient_retry_respects_call_budget(self):
        result, answers = self.exercise([ConnectionError("offline reset")], args=("--max-calls", "1"))
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", answers[0])

    def test_nonretryable_provider_error_stops_immediately(self):
        result, answers = self.exercise([RuntimeError("offline permanent error")], followup=True)
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        self.assertTrue(all("__error__" in answer for answer in answers))

    def test_deadline_stops_without_repeating_or_dispatching_followup(self):
        result, answers = self.exercise([TimeoutError("offline total deadline")], followup=True)
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        self.assertTrue(all("__error__" in answer for answer in answers))

    def test_explicit_single_attempt_is_not_expanded_by_runner(self):
        sdk = ReplaySDK(['{"bad":'])
        def calls(directory, cfg):
            answer = Tracer(SimpleNamespace(dir=directory)).chat_json(
                "world.joint_plan", [], retries=1, max_tokens=4096)
            self.assertIn("__error__", answer)
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--json-attempts", "5"])
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual([r["max_attempts"] for r in result.rows if r["event"] == "json_attempt"], [1])

    def test_truncation_at_runner_ceiling_has_no_unchanged_retry(self):
        empty = SimpleNamespace(id="empty", model="gpt-5.4-mini", usage=None,
            choices=[SimpleNamespace(index=0, finish_reason="length",
                message=SimpleNamespace(content="", refusal=None))])
        result, answers = self.exercise([empty], args=("--max-output-tokens", "4096", "--json-attempts", "5"))
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        self.assertIn("__error__", answers[0])


if __name__ == "__main__":
    unittest.main()
