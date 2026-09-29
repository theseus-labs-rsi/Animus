"""Offline regression tests for bounded JSON syntax retries; no API calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-test-key")
os.environ.setdefault("MODEL", "offline-test-model")
import config
from llm_trace import trace_scope
from llm_offline_executor import install


def completion(text, *, reason="stop", refusal=None):
    return SimpleNamespace(id="offline-custom", model="offline-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=8192, total_tokens=8202,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=8192)),
        choices=[SimpleNamespace(index=0, finish_reason=reason,
            message=SimpleNamespace(content=text, refusal=refusal))])


def status_error(status, *, code=None):
    return config._openai.APIStatusError("offline status error",
        response=config.httpx.Response(status, request=config.httpx.Request("POST", "https://offline.invalid")),
        body={"error": {"code": code}})


class JsonRetryTests(unittest.TestCase):
    def exercise(self, replies, **kwargs):
        requests = []
        values = iter(replies)

        def create(**request):
            requests.append(deepcopy(request))
            value = next(values)
            if isinstance(value, Exception):
                raise value
            if hasattr(value, "choices"):
                return value
            return SimpleNamespace(
                id=f"offline-{len(requests)}", model=request["model"],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                choices=[SimpleNamespace(index=0, finish_reason="stop",
                    message=SimpleNamespace(content=value, refusal=None))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [{"role": "system", "content": "Follow the original business rules."},
                    {"role": "user", "content": "Return a JSON object with verdict and reasons."}]
        before = deepcopy(messages)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            original = config._perform_request
            install(config, client)
            with patch.object(config, "_perform_request", config._perform_request), trace_scope(path, "json-retry"):
                try:
                    result = config.chat_json(messages, strict_json=kwargs.pop("strict_json", True), retry_delay_base=0,
                                              model="offline-model", **kwargs)
                    error = None
                except Exception as exc:
                    result, error = None, exc
            config._perform_request = original
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(messages, before)
        return result, error, requests, records

    def test_bad_then_good_preserves_task_and_adds_precise_feedback(self):
        bad = '{"verdict": "pass",}'
        result, error, requests, records = self.exercise([bad, '{"verdict":"pass"}'])
        self.assertIsNone(error)
        self.assertEqual(result, {"verdict": "pass"})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["messages"][:2], requests[0]["messages"])
        self.assertEqual(requests[1]["messages"][2], {"role": "assistant", "content": bad})
        self.assertIn("第 1 行", requests[1]["messages"][3]["content"])
        self.assertIn("Expecting property name enclosed in double quotes", requests[1]["messages"][3]["content"])
        self.assertEqual([r["model"] for r in requests], ["offline-model"] * 2)
        self.assertEqual([r["event"] for r in records], [
            "json_attempt", "request", "response", "json_error", "json_retry_feedback",
            "json_attempt", "request", "response", "json_result"])
        self.assertEqual(records[3]["raw_output"], bad)
        for start in (0, 5):
            self.assertEqual(records[start + 1]["call_id"], records[start + 2]["call_id"])
        self.assertEqual(records[0]["attempt_id"], records[3]["attempt_id"])
        self.assertEqual(records[5]["attempt_id"], records[8]["attempt_id"])

    def test_three_attempts_exhausted_with_complete_trace(self):
        bad = ['{"attempt":1,}', '{"attempt":2,}', '{"attempt":3,}']
        result, error, requests, records = self.exercise(bad)
        self.assertIsNone(result)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 3)
        self.assertEqual([len(r["messages"]) for r in requests], [2, 4, 4])
        self.assertEqual(requests[2]["messages"][2]["content"], bad[1])
        self.assertEqual([r["raw_output"] for r in records if r["event"] == "json_error"], bad)
        for event in ("json_attempt", "request", "response", "json_error"):
            self.assertEqual(sum(r["event"] == event for r in records), 3)
        self.assertEqual(sum(r["event"] == "json_retry_feedback" for r in records), 2)

    def test_semantic_failure_is_returned_without_retry(self):
        value = {"verdict": "fail", "reasons": ["Missing evidence"]}
        result, error, requests, records = self.exercise([json.dumps(value)])
        self.assertIsNone(error)
        self.assertEqual(result, value)
        self.assertEqual(len(requests), 1)
        self.assertFalse(any(r["event"] == "json_retry_feedback" for r in records))

    def test_explicit_single_attempt_keeps_original_behavior(self):
        result, error, requests, records = self.exercise(['{"verdict": "pass",}'], retries=1)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(requests[0]["messages"]), 2)
        self.assertEqual([r["event"] for r in records],
                         ["json_attempt", "request", "response", "json_error"])

    def test_feedback_can_be_disabled(self):
        _, error, requests, records = self.exercise(['{"x":1,}', '{"x":1}'],
                                                   retry_parse_feedback=False)
        self.assertIsNone(error)
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertFalse(any(r["event"] == "json_retry_feedback" for r in records))

    def test_feedback_output_is_bounded(self):
        bad = '{"x":"' + 'x' * 50000
        _, error, requests, records = self.exercise([bad, '{"x":1}'])
        self.assertIsNone(error)
        self.assertLess(len(requests[1]["messages"][2]["content"]), 8300)
        self.assertIn("头尾片段", requests[1]["messages"][3]["content"])
        feedback = next(r for r in records if r["event"] == "json_retry_feedback")
        self.assertTrue(feedback["excerpt_truncated"])
        self.assertEqual(feedback["raw_output_chars"], len(bad))
        self.assertEqual(next(r for r in records if r["event"] == "json_error")["raw_output"], bad)

    def test_network_retry_does_not_create_syntax_feedback(self):
        _, error, requests, records = self.exercise([ConnectionError("offline reset"), '{"x":1}'])
        self.assertIsNone(error)
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertFalse(any(r["event"] == "json_retry_feedback" for r in records))
        self.assertEqual(sum(r["event"] == "call_error" for r in records), 1)

    def test_empty_or_truncated_completion_without_larger_allowance_stops(self):
        for reply in (completion(""), completion("", reason="length"),
                      completion('{"x":1}', reason="length")):
            with self.subTest(reason=reply.choices[0].finish_reason, text=reply.choices[0].message.content):
                result, error, requests, records = self.exercise([reply, '{"x":1}'])
                self.assertIsNone(result)
                self.assertIsInstance(error, config.ChatJSONError)
                self.assertEqual(len(requests), 1)
                failures = [r for r in records if r["event"] == "json_error"]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["error_type"], "CompletionOutputError")
                self.assertFalse(failures[0]["retryable"])
                self.assertIn("reasoning_tokens=8192", failures[0]["error"])

    def test_transient_transport_statuses_retry(self):
        values = [ConnectionError("connection reset"), config.httpx.ConnectTimeout("connect timeout"),
                  config.httpx.ConnectError("connect"),
                  config._openai.APIConnectionError(request=config.httpx.Request("POST", "https://offline.invalid")),
                  *[status_error(status) for status in (408, 429, 500, 502, 503)]]
        for failure in values:
            with self.subTest(failure=repr(failure)):
                self.assertTrue(config.is_retryable_chat_error(failure))
                result, error, requests, _ = self.exercise([failure, '{"x":1}'])
                self.assertIsNone(error)
                self.assertEqual(result, {"x": 1})
                self.assertEqual(len(requests), 2)

    def test_transient_exhaustion_is_bounded_and_preserves_cause(self):
        for failure in (ConnectionError("connection reset"), status_error(503)):
            with self.subTest(failure=repr(failure)):
                _, error, requests, records = self.exercise([failure] * 3)
                self.assertEqual(len(requests), 3)
                self.assertIs(error.__cause__, failure)
                self.assertEqual(sum(r["event"] == "json_error" for r in records), 3)
        _, error, requests, _ = self.exercise([completion("", reason="length")] * 3)
        self.assertEqual(len(requests), 1)
        self.assertIsInstance(error.__cause__, config.CompletionOutputError)

    def test_length_increases_effective_cap_once_and_stops_at_upper_bound(self):
        _, error, requests, records = self.exercise([completion("", reason="length")] * 3,
                                                   max_tokens=4096, max_output_tokens=8192, retries=5)
        self.assertEqual([r["max_tokens"] for r in requests], [4096, 8192])
        self.assertEqual(error.kind, "output_truncated")
        self.assertEqual(error.token_cap, 8192)
        self.assertEqual(error.attempts, 2)
        self.assertEqual([r["retry_action"] for r in records if r["event"] == "json_error"],
                         ["increase_output_cap", "stop"])

    def test_length_with_larger_bound_can_recover_without_changing_model(self):
        result, error, requests, _ = self.exercise([completion("", reason="length"), '{"x":1}'],
                                                  max_tokens=4096, max_output_tokens=8192)
        self.assertIsNone(error)
        self.assertEqual(result, {"x": 1})
        self.assertEqual([r["max_tokens"] for r in requests], [4096, 8192])
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual({r["model"] for r in requests}, {"offline-model"})

    def test_total_deadline_and_read_timeout_do_not_repeat_work(self):
        values = [TimeoutError("deadline"), config.httpx.ReadTimeout("read"), config.httpx.WriteTimeout("write"),
                  config.CallDeadlineError("deadline", phase="awaiting_http_headers", deadline_s=600)]
        for failure in values:
            with self.subTest(failure=failure):
                _, error, requests, _ = self.exercise([failure, '{"x":1}'])
                self.assertEqual(len(requests), 1)
                self.assertFalse(config.is_retryable_chat_error(failure))
                self.assertIs(error.__cause__, failure)

    def test_refusal_and_filter_never_retry(self):
        replies = [completion("", refusal="refused"), completion('{"x":1}', refusal="refused"),
                   completion("", reason="content_filter"), completion("", reason="length", refusal="refused")]
        for reply in replies:
            with self.subTest(reply=reply):
                _, error, requests, records = self.exercise([reply, '{"x":1}'])
                self.assertEqual(len(requests), 1)
                self.assertIsInstance(error.__cause__, config.CompletionRefusalError)
                self.assertFalse(config.is_retryable_chat_error(error.__cause__))
                self.assertFalse(next(r for r in records if r["event"] == "json_error")["retryable"])

    def test_auth_configuration_budget_and_audit_errors_do_not_retry(self):
        values = [status_error(401), status_error(403), status_error(400), status_error(404),
                  status_error(429, code="insufficient_quota"),
                  ValueError("unsupported request parameter"), RuntimeError("budget exhausted"),
                  PermissionError("audit path cannot be written")]
        for failure in values:
            with self.subTest(failure=repr(failure)):
                self.assertFalse(config.is_retryable_chat_error(failure))
                _, error, requests, records = self.exercise([failure, '{"x":1}'])
                self.assertEqual(len(requests), 1)
                self.assertIs(error.__cause__, failure)
                self.assertFalse(next(r for r in records if r["event"] == "json_error")["retryable"])

    def test_optional_repair_missing_or_failed_preserves_syntax_retry(self):
        def failed_repair(*args, **kwargs):
            raise RuntimeError("offline repair failed")

        for module, error_type in ((None, "ModuleNotFoundError"),
                                  (SimpleNamespace(repair_json=failed_repair), "RuntimeError")):
            with self.subTest(error_type=error_type), patch.dict(sys.modules, {"json_repair": module}):
                result, error, requests, records = self.exercise(['{"x":1,}', '{"x":1}'], strict_json=False)
                self.assertIsNone(error)
                self.assertEqual(result, {"x": 1})
                self.assertEqual(len(requests), 2)
                repair_error = next(r for r in records if r["event"] == "json_repair_error")
                self.assertEqual(repair_error["error_type"], error_type)
                parse_error = next(r for r in records if r["event"] == "json_error")
                self.assertEqual(parse_error["error_type"], "JSONDecodeError")
                self.assertTrue(parse_error["retryable"])
                self.assertIn("Expecting property name", requests[1]["messages"][-1]["content"])

    def test_repaired_result_audit_failure_is_not_swallowed_or_retried(self):
        original_emit = config.emit
        failure = PermissionError("offline audit write failed")

        def audit(event, **kwargs):
            if event == "json_result":
                raise failure
            return original_emit(event, **kwargs)

        module = SimpleNamespace(repair_json=lambda *a, **kw: {"x": 1})
        with patch.dict(sys.modules, {"json_repair": module}), patch.object(config, "emit", audit):
            _, error, requests, records = self.exercise(['{"x":1,}', '{"x":1}'], strict_json=False)
        self.assertEqual(len(requests), 1)
        self.assertIs(error.__cause__, failure)
        self.assertFalse(any(r["event"] == "json_repair_error" for r in records))
        self.assertFalse(next(r for r in records if r["event"] == "json_error")["retryable"])

    def test_empty_choices_stop_without_spending_further_calls(self):
        for choices in ([], None):
            with self.subTest(choices=choices):
                response = completion("")
                response.choices = choices
                result, error, requests, records = self.exercise([response, '{"x":1}'])
                self.assertIsInstance(error, config.ChatJSONError)
                self.assertIsNone(result)
                self.assertEqual(len(requests), 1)
                self.assertEqual(next(r for r in records if r["event"] == "response")["response"]["choices"], [])
                self.assertEqual(next(r for r in records if r["event"] == "json_error")["error_type"],
                                 "CompletionOutputError")
                _, error, requests, records = self.exercise([response] * 3)
                self.assertEqual(len(requests), 1)
                self.assertIsInstance(error.__cause__, config.CompletionOutputError)
                self.assertEqual(sum(r["event"] == "json_error" for r in records), 1)


if __name__ == "__main__":
    unittest.main()
