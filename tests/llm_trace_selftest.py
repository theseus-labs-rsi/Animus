"""Offline transport/trace tests; no dotenv read and no network requests."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_trace import trace_scope
from llm_offline_executor import install
from pipeline.run import Tracer


def response(text, reason="stop"):
    return types.SimpleNamespace(id="fake-response", model="fake-provider-model",
        choices=[types.SimpleNamespace(index=0, message=types.SimpleNamespace(content=text), finish_reason=reason)],
        usage=types.SimpleNamespace(prompt_tokens=8, completion_tokens=12, total_tokens=20))


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "attempts.jsonl"
        spec = importlib.util.spec_from_file_location("offline_trace_config", ROOT / "config.py")
        self.config = importlib.util.module_from_spec(spec)
        self.api = Mock()
        with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda *a: None),
                "openai": types.SimpleNamespace(OpenAI=lambda **kw: self.api)}), patch.dict(os.environ,
                {"OPENAI_API_KEY": "offline-secret-token", "MODEL": "offline-model"}):
            spec.loader.exec_module(self.config)
        install(self.config, self.api)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        return [json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines()]

    def test_complete_responses_parse_failure_and_retry(self):
        long_output = "x" * 1400
        self.api.chat.completions.create.side_effect = [response("broken"), response(json.dumps({"text": long_output}))]
        with trace_scope(self.path, "review.test"):
            result = self.config.chat_json([{"role": "user", "content": "input"}], strict_json=True,
                                          retries=2, retry_delay_base=0)
        self.assertEqual(result["text"], long_output)
        rows = self.rows()
        self.assertEqual(sum(r["event"] == "request" for r in rows), 2)
        self.assertEqual(sum(r["event"] == "json_error" for r in rows), 1)
        responses = [r for r in rows if r["event"] == "response"]
        self.assertEqual(responses[1]["response"]["choices"][0]["content"], json.dumps({"text": long_output}))
        self.assertEqual(responses[1]["response"]["usage"]["total_tokens"], 20)

    def test_failed_transport_is_not_parse_success(self):
        self.api.chat.completions.create.side_effect = TimeoutError("fake timeout")
        with trace_scope(self.path, "review.timeout"), self.assertRaises(ValueError):
            self.config.chat_json([], retries=1, strict_json=True)
        self.assertEqual(sum(r["event"] == "call_error" for r in self.rows()), 1)
        self.assertFalse(any(r["event"] == "json_result" for r in self.rows()))

    def test_default_sampling_parameters_remain_unchanged(self):
        self.api.chat.completions.create.return_value = response('{"ok":true}')
        with trace_scope(self.path, "transport.defaults"):
            self.assertEqual(self.config.chat_json([], retries=1, strict_json=True), {"ok": True})
        sent = self.api.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["top_p"], 1.0)
        self.assertEqual(sent["temperature"], 0.7)
        self.assertNotIn("response_format", sent)
        request = next(row for row in self.rows() if row["event"] == "request")
        self.assertEqual(request["parameters"]["top_p"], sent["top_p"])
        self.assertNotIn("response_format", request["parameters"])

    def test_chat_explicit_none_omits_top_p_from_wire_and_trace(self):
        self.api.chat.completions.create.return_value = response("ok")
        with trace_scope(self.path, "transport.omitted"):
            self.assertEqual(self.config.chat([], temperature=0.0, top_p=None), "ok")
        sent = self.api.chat.completions.create.call_args.kwargs
        self.assertNotIn("top_p", sent)
        self.assertEqual(sent["temperature"], 0.0)
        request = next(row for row in self.rows() if row["event"] == "request")
        self.assertNotIn("top_p", request["parameters"])

    def test_chat_json_forwards_explicit_none_and_numeric_top_p(self):
        self.api.chat.completions.create.return_value = response('{"ok":true}')
        with trace_scope(self.path, "transport.json"):
            self.config.chat_json([], retries=1, strict_json=True, top_p=None)
            self.config.chat_json([], retries=1, strict_json=True, top_p=0.8)
        first, second = self.api.chat.completions.create.call_args_list
        self.assertNotIn("top_p", first.kwargs)
        self.assertEqual(second.kwargs["top_p"], 0.8)
        requests = [row for row in self.rows() if row["event"] == "request"]
        self.assertNotIn("top_p", requests[0]["parameters"])
        self.assertEqual(requests[1]["parameters"]["top_p"], 0.8)

    def test_omission_does_not_infer_model_or_retry_failed_request(self):
        self.api.chat.completions.create.side_effect = ValueError("provider rejected request")
        with trace_scope(self.path, "transport.rejected"), self.assertRaises(ValueError):
            self.config.chat_json([], model="arbitrary-provider-model", retries=1, strict_json=True, top_p=None)
        self.assertEqual(self.api.chat.completions.create.call_count, 1)
        sent = self.api.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["model"], "arbitrary-provider-model")
        self.assertNotIn("top_p", sent)
        self.assertEqual(sum(row["event"] == "request" for row in self.rows()), 1)
        self.assertFalse(any(row["event"] == "json_result" for row in self.rows()))

    def test_explicit_response_format_reaches_chat_and_chat_json_unchanged(self):
        self.api.chat.completions.create.return_value = response('{"ok":true}')
        mode = {"type": "json_object"}
        with trace_scope(self.path, "transport.json_mode"):
            self.config.chat([], top_p=None, response_format=mode)
            self.assertEqual(self.config.chat_json([], retries=1, strict_json=True,
                top_p=None, response_format=mode), {"ok": True})
        self.assertEqual(mode, {"type": "json_object"})
        for call in self.api.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["response_format"], mode)
            self.assertNotIn("top_p", call.kwargs)
        for row in self.rows():
            if row["event"] == "request":
                self.assertEqual(row["parameters"]["response_format"], mode)
                self.assertNotIn("top_p", row["parameters"])

    def test_json_mode_never_certifies_or_repairs_invalid_response(self):
        self.api.chat.completions.create.return_value = response('{"answer":"state is "confirmed""}')
        with trace_scope(self.path, "transport.bad_json_mode"), self.assertRaises(ValueError):
            self.config.chat_json([], retries=1, strict_json=True, top_p=None,
                                  response_format={"type": "json_object"})
        self.assertEqual(self.api.chat.completions.create.call_count, 1)
        self.assertEqual(sum(row["event"] == "response" for row in self.rows()), 1)
        self.assertFalse(any(row["event"] == "json_result" for row in self.rows()))

    def test_unsupported_response_format_is_preserved_without_fallback(self):
        self.api.chat.completions.create.side_effect = ValueError("unsupported response_format")
        with trace_scope(self.path, "transport.unsupported_mode"), self.assertRaises(ValueError):
            self.config.chat_json([], retries=1, strict_json=True, top_p=None,
                                  response_format={"type": "json_object"})
        self.assertEqual(self.api.chat.completions.create.call_count, 1)
        self.assertEqual(self.api.chat.completions.create.call_args.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(sum(row["event"] == "call_error" for row in self.rows()), 1)

    def test_truncated_nonempty_output_is_error(self):
        self.api.chat.completions.create.return_value = response('{"answer":"looks complete"}', "length")
        with trace_scope(self.path, "review.truncated"), self.assertRaises(ValueError):
            self.config.chat_json([], retries=1, strict_json=True)
        self.assertTrue(any(r["event"] == "response" for r in self.rows()))
        self.assertFalse(any(r["event"] == "json_result" for r in self.rows()))

    def test_both_tracer_interfaces_preserve_safe_truncation_metadata(self):
        self.api.chat.completions.create.return_value = response('{"partial":', "length")
        tracer = Tracer(types.SimpleNamespace(dir=Path(self.temp.name)))
        with patch.dict(sys.modules, {"config": self.config}):
            results = [tracer.chat_json("world.structure", [], retries=1, max_tokens=4096),
                       tracer.chat_text("world.text", [], max_tokens=4096)]
        for result in results:
            metadata = result["__error_metadata__"]
            self.assertEqual(metadata["kind"], "output_truncated")
            self.assertEqual(metadata["token_cap"], 4096)
            self.assertEqual(metadata["usage_status"], "reported")
            self.assertTrue(metadata["response_received"])
            self.assertNotIn("messages", metadata)
            self.assertNotIn("response", metadata)

    def test_tracer_returned_error_redacts_credentials_before_truncation(self):
        self.api.chat.completions.create.side_effect = ValueError("offline-secret-token config error")
        tracer = Tracer(types.SimpleNamespace(dir=Path(self.temp.name)))
        with patch.dict(sys.modules, {"config": self.config}), patch.dict(os.environ, {"OPENAI_API_KEY": "offline-secret-token"}):
            result = tracer.chat_text("world.text", [])
        self.assertNotIn("offline-secret-token", json.dumps(result))
        self.assertIn("[redacted]", result["__error__"])

    def test_credentials_redacted_and_thread_scope_propagates(self):
        self.api.chat.completions.create.return_value = response("ok offline-secret-token")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-secret-token"}), trace_scope(self.path, "parent"):
            self.config.pmap(lambda i: self.config.chat([{"role": "user", "content": f"{i} offline-secret-token"}]), range(3), workers=3)
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("offline-secret-token", text)
        self.assertIn("[redacted]", text)
        requests = [r for r in self.rows() if r["event"] == "request"]
        self.assertEqual(len(requests), 3)
        self.assertEqual(len({r["operation_id"] for r in requests}), 1)
        self.assertEqual(len({r["call_id"] for r in requests}), 3)


if __name__ == "__main__":
    unittest.main()
