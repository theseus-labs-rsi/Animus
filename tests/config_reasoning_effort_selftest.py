"""Offline request-boundary checks for the opt-in ordinary reasoning setting."""
from copy import deepcopy
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
from llm_transport import original_run_transport, VERSION
from pipeline.run import _env_snapshot


class ReasoningEffortTests(unittest.TestCase):
    def load_config(self, effort=None):
        env = {"OPENAI_API_KEY": "offline-only", "MODEL": "glm-5.3-flash",
               "LLM_MIN_COMPLETION_TOKENS": "8192", "LLM_HTTP_READ_TIMEOUT_S": "17",
               "LLM_DEADLINE_S": "19"}
        if effort is not None:
            env["LLM_REASONING_EFFORT"] = effort
        spec = importlib.util.spec_from_file_location("offline_reasoning_config", ROOT / "config.py")
        cfg = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {
                "dotenv": types.SimpleNamespace(load_dotenv=lambda *args: None)}):
            spec.loader.exec_module(cfg)
        cfg._perform_request = Mock(return_value=types.SimpleNamespace(
            id="offline", model="glm-5.3-flash", usage=None,
            choices=[types.SimpleNamespace(index=0, finish_reason="stop",
                message=types.SimpleNamespace(content='{"ok":true}', refusal=None))]))
        return cfg

    def test_unset_and_blank_preserve_legacy_request(self):
        for effort in (None, "", "  "):
            with self.subTest(effort=effort):
                cfg = self.load_config(effort)
                cfg.chat([], model="arbitrary-provider-model", max_tokens=4096)
                sent = cfg._perform_request.call_args.kwargs
                self.assertEqual(sent["parameters"], {"temperature": 0.7, "top_p": 1.0, "max_tokens": 8192})
                self.assertEqual(sent["http_timeout"].as_dict(), {"connect": 15.0, "read": 17.0, "write": 30.0, "pool": 15.0})
                self.assertLessEqual(sent["deadline_s"], 19)
                self.assertGreater(sent["deadline_s"], 18)
                self.assertFalse(sent["explicit_profile"])

    def test_low_reaches_text_json_and_trace_without_other_wire_changes(self):
        cfg = self.load_config("low")
        messages = [{"role": "user", "content": "Original business task"}]
        before = deepcopy(messages)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with trace_scope(path, "reasoning-test"):
                cfg.chat(messages, max_tokens=4096, top_p=None)
                self.assertEqual(cfg.chat_json(messages, max_tokens=4096, top_p=None,
                    retries=1, strict_json=True), {"ok": True})
            requests = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                        if json.loads(line)["event"] == "request"]
        self.assertEqual(messages, before)
        self.assertEqual(len(requests), 2)
        for call in cfg._perform_request.call_args_list:
            sent = call.kwargs
            self.assertEqual(sent["model"], "glm-5.3-flash")
            self.assertEqual(sent["messages"], before)
            self.assertEqual(sent["parameters"], {"temperature": 0.7, "max_tokens": 8192, "reasoning_effort": "low"})
            self.assertEqual(sent["http_timeout"].read, 17)
            self.assertFalse(sent["explicit_profile"])
        self.assertTrue(all(row["parameters"]["reasoning_effort"] == "low" for row in requests))

    def test_explicit_transport_precedes_even_invalid_environment_setting(self):
        cfg = self.load_config("unsupported-effort")
        cfg.chat_json([], model="glm-5.3-flash", retries=1,
                      transport=original_run_transport("glm-5.3-flash", "high", 23))
        high = cfg._perform_request.call_args.kwargs
        self.assertEqual(high["parameters"]["reasoning_effort"], "high")
        self.assertEqual(high["parameters"]["max_tokens"], 4096)
        self.assertEqual(high["http_timeout"], 23)
        cfg.chat([], model="unlisted-model", transport={"version": VERSION,
            "profiles": {}, "model_profiles": {}, "default_profile": "legacy"})
        legacy = cfg._perform_request.call_args.kwargs
        self.assertNotIn("reasoning_effort", legacy["parameters"])
        self.assertEqual(legacy["parameters"]["max_tokens"], 8192)

    def test_unsupported_effective_model_or_effort_fails_before_request(self):
        for effort, model in (("low", "unlisted-model"), ("low", "glm-4.7-nothinking"),
                              ("medium", "glm-5.3-flash"), ("max", "gpt-5.4-mini")):
            with self.subTest(effort=effort, model=model):
                cfg = self.load_config(effort)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.jsonl"
                    with trace_scope(path, "invalid-reasoning"), self.assertRaisesRegex(ValueError, "LLM_REASONING_EFFORT"):
                        cfg.chat_json([], model=model, retries=3, retry_delay_base=0)
                    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                    self.assertFalse(any(row["event"] == "request" for row in rows))
                cfg._perform_request.assert_not_called()

    def test_snapshot_records_setting_and_explicit_model_is_validated(self):
        cfg = self.load_config("max")
        cfg.MODEL = "unlisted-default"
        cfg.chat([], model="glm-5.3-flash")
        self.assertEqual(cfg._perform_request.call_args.kwargs["parameters"]["reasoning_effort"], "max")
        with patch.dict(sys.modules, {"config": cfg}):
            snapshot = _env_snapshot()
        self.assertEqual(snapshot["reasoning_effort"], "max")
        self.assertEqual(snapshot["model"], "unlisted-default")
        self.assertNotIn("api_key", snapshot)


if __name__ == "__main__":
    unittest.main()
