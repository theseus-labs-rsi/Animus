"""Narrow author-truncation recovery retains all original dispatch budgets."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from original_smoke_json_retry_selftest import ReplaySDK


def truncated(usage=None):
    return fixture.SimpleNamespace(id="truncated", model="gpt-5.4-mini", usage=usage,
        choices=[fixture.SimpleNamespace(index=0, finish_reason="length",
            message=fixture.SimpleNamespace(content="", refusal=None))])


class AgentRecoveryTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def exercise(self, responses, *, extra=(), first_step="world.agent.write", first_model=None):
        sdk, answers = ReplaySDK(responses), []
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            answers.append(tracer.chat_json(first_step, [{"role": "user", "content": "Write original broad unit"}],
                                            model=first_model, max_tokens=4096, retries=1))
            answers.append(tracer.chat_json("world.agent.plan", [{"role": "user", "content": "Split the failed unit"}],
                                            max_tokens=4096, retries=1))
            answers.append(tracer.chat_json("world.agent.write", [{"role": "user", "content": "Write smaller revised unit"}],
                                            max_tokens=4096, retries=1))
        with patch.object(fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--json-attempts", "5", *extra])
        return result, answers

    def test_author_length_failure_can_return_to_planner_and_revised_writer(self):
        result, answers = self.exercise([truncated(), '{"action":"split"}', '{"entities":[]}'])
        self.assertIsNone(result.error)
        self.assertEqual(answers[0]["__error_metadata__"]["kind"], "output_truncated")
        self.assertEqual(answers[1:], [{"action": "split"}, {"entities": []}])
        self.assertEqual(result.profile["admitted_calls"], 3)
        self.assertFalse(result.profile["stopped"])
        self.assertEqual(len(result.profile["deferred_agent_author_failures"]), 1)
        self.assertEqual(result.profile["logical_failure_policy"]["version"], "agent-author-replan/v1")
        self.assertEqual([row["step"] for row in result.requests], ["world.agent.write", "world.agent.plan", "world.agent.write"])
        self.assertEqual([row["max_attempts"] for row in result.rows if row["event"] == "json_attempt"], [1, 1, 1])
        self.assertNotEqual(result.api.calls[0]["kwargs"]["messages"], result.api.calls[2]["kwargs"]["messages"])

    def test_call_limit_prevents_replanning_dispatch_after_length(self):
        result, answers = self.exercise([truncated()], extra=("--max-calls", "1"))
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])
        self.assertTrue(all("__error__" in answer for answer in answers))

    def test_budget_limit_prevents_replanning_dispatch_after_length(self):
        result, answers = self.exercise([truncated()], extra=("--settle-reported-usage", "--max-cny", "0.12"))
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.profile["unsettled_reservation_calls"], 1)
        self.assertTrue(all("__error__" in answer for answer in answers))

    def test_reported_overspend_stop_is_never_cleared_by_recovery(self):
        usage = fixture.SimpleNamespace(prompt_tokens=10000, completion_tokens=10000, total_tokens=20000)
        result, answers = self.exercise([truncated(usage)], extra=("--settle-reported-usage", "--max-cny", "0.12"))
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.profile["deferred_agent_author_failures"], [])
        self.assertGreater(result.profile["budget_consumed_cny"], 0.12)

    def test_timeout_and_empty_output_never_receive_exception(self):
        empty = fixture.SimpleNamespace(id="empty", model="gpt-5.4-mini", usage=None,
            choices=[fixture.SimpleNamespace(index=0, finish_reason="stop",
                message=fixture.SimpleNamespace(content="", refusal=None))])
        for response in (TimeoutError("offline deadline"), empty):
            with self.subTest(response=response):
                result, answers = self.exercise([response])
                self.assertEqual(len(result.api.calls), 1)
                self.assertTrue(result.profile["stopped"])
                self.assertEqual(result.profile["deferred_agent_author_failures"], [])
                self.assertTrue(all("__error__" in answer for answer in answers))

    def test_non_agent_and_near_match_steps_cannot_recover(self):
        for step in ("world.structure", "world.agent.write.extra", "world.agent.plan"):
            with self.subTest(step=step):
                result, answers = self.exercise([truncated()], first_step=step)
                self.assertEqual(len(result.api.calls), 1)
                self.assertTrue(result.profile["stopped"])
                self.assertEqual(result.profile["deferred_agent_author_failures"], [])
                self.assertTrue(all("__error__" in answer for answer in answers))

    def test_model_mismatch_never_dispatches_or_recovers(self):
        result, answers = self.exercise([], first_model="expensive-model")
        self.assertEqual(result.api.calls, [])
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.profile["deferred_agent_author_failures"], [])
        self.assertTrue(all("__error__" in answer for answer in answers))

    def test_forged_error_without_response_cannot_unlock_followups(self):
        def calls(directory, cfg):
            error = cfg.CompletionOutputError("offline truncated without returned response", kind="output_truncated")
            sdk = ReplaySDK([error])
            original = cfg._perform_request
            from llm_offline_executor import install
            install(cfg, sdk)
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            first = tracer.chat_json("world.agent.write", [], retries=1)
            self.assertFalse(first["__error_metadata__"]["response_received"])
            second = tracer.chat_json("world.agent.plan", [], retries=1)
            self.assertIn("__error__", second)
            self.assertEqual(len(sdk.calls), 1)
            cfg._perform_request = original
        result = self.run_tool(calls)
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.profile["deferred_agent_author_failures"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
