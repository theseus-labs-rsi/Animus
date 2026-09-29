"""Offline actual-wrapper/wire tests for a step-scoped experiment setting."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_original_bc_smoke as smoke
from pipeline.run import Tracer
from llm_offline_executor import install


class FakeSDK:
    def __init__(self, failure=False):
        self.calls = []
        self.lock = threading.Lock()
        self.failure = failure
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        raise AssertionError("Explicit transport must clone client with SDK retries disabled")

    def with_options(self, **options):
        def create(**kwargs):
            with self.lock:
                self.calls.append({"options": deepcopy(options), "kwargs": deepcopy(kwargs)})
            if self.failure:
                raise RuntimeError("Offline injected SDK failure")
            return SimpleNamespace(id="offline", model=kwargs["model"], usage=None,
                choices=[SimpleNamespace(index=0, finish_reason="stop",
                    message=SimpleNamespace(content='{"ok":true}', refusal=None))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def load_config(api):
    spec = importlib.util.spec_from_file_location("offline_smoke_config", ROOT / "config.py")
    cfg = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda *a: None),
            "openai": SimpleNamespace(OpenAI=lambda **kw: api)}), patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-only", "OPENAI_BASE_URL": "https://offline.invalid/v1",
            "MODEL": "offline-model", "LLM_MIN_COMPLETION_TOKENS": "99999"}):
        spec.loader.exec_module(cfg)
    install(cfg, api)
    return cfg


class CorpusReasoningOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.counter = 0
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def run_tool(self, callback, extra=(), *, sdk_failure=False, tracer_method=None):
        self.counter += 1
        name = "case" + str(self.counter)
        directory = self.directory / "output/runs" / name
        api = FakeSDK(failure=sdk_failure)
        cfg = load_config(api)
        source_hashes = {"tools/run_original_bc_smoke.py": hashlib.sha256(
            (ROOT / "tools/run_original_bc_smoke.py").read_bytes()).hexdigest()}
        argv = ["smoke", "--run", name, "--max-calls", "12", "--max-cny", "50", *extra]
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"config": cfg}))
            from pipeline import factory
            stack.enter_context(patch.object(smoke, "ROOT", self.directory))
            stack.enter_context(patch.object(smoke, "implementation_hashes", return_value=source_hashes))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.dict(factory.SCENARIOS))
            stack.enter_context(patch.object(factory, "main", side_effect=lambda: callback(directory, cfg)))
            if tracer_method is not None:
                stack.enter_context(patch.object(Tracer, "chat_json", tracer_method))
            initial_method = Tracer.chat_json
            try:
                smoke.main()
            except BaseException as exc:
                error = exc
            self.assertIs(Tracer.chat_json, initial_method)
            self.assertIsNone(smoke._TRACER_STEP.get())
        profile_path = directory / "experiment_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else None
        rows_path = directory / "llm_attempts.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()] if rows_path.exists() else []
        return SimpleNamespace(directory=directory, api=api, config=cfg, error=error, profile=profile,
            rows=rows, requests=[r for r in rows if r.get("event") == "request"])

    def test_exact_step_override_real_wire_and_shared_budget(self):
        steps = ["world.structure", "render.signal", "render.discriminate", "corpus.review",
                 "grounding.reference", "corpus.review.extra", "CORPUS.REVIEW"]
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            for step in steps:
                # Deliberate role-looking prose cannot activate the override.
                self.assertEqual(tracer.chat_json(step, [{"role": "user", "content": "corpus.review " + step}],
                    max_tokens=4096), {"ok": True})
            self.assertEqual(cfg.chat_json([{"role": "user", "content": "corpus.review"}]), {"ok": True})
        result = self.run_tool(calls, ["--corpus-review-reasoning-effort", "medium"])
        self.assertIsNone(result.error)
        wire = [call["kwargs"] for call in result.api.calls]
        self.assertEqual([call["reasoning_effort"] for call in wire],
                         ["medium" if step == "corpus.review" else "low" for step in steps] + ["low"])
        self.assertTrue(all(call["max_completion_tokens"] == 4096 and call["model"] == "gpt-5.4-mini"
                            and "temperature" not in call and "top_p" not in call for call in wire))
        self.assertTrue(all(call["options"] == {"timeout": 150, "max_retries": 0} for call in result.api.calls))
        self.assertEqual(result.profile["admitted_calls"], len(result.api.calls))
        self.assertEqual(result.profile["admitted_calls"], 8)
        self.assertGreater(result.profile["reserved_upper_estimate_cny"], 0)
        self.assertEqual(result.profile["transport"]["profiles"]["low"]["reasoning_effort"], "low")
        self.assertEqual(result.profile["step_transport_overrides"]["corpus.review"]["reasoning_effort"], "medium")
        self.assertEqual(result.profile["source_drift"], [])
        self.assertEqual([r["parameters"]["reasoning_effort"] for r in result.requests],
                         ["medium" if step == "corpus.review" else "low" for step in steps])
        self.assertTrue(all(r["max_attempts"] == 3 for r in result.rows if r.get("event") == "json_attempt"))

    def test_default_none_preserves_global_settings_and_no_override(self):
        for setting in ("low", "medium"):
            with self.subTest(setting=setting):
                def calls(directory, cfg):
                    tracer = Tracer(SimpleNamespace(dir=directory))
                    tracer.chat_json("corpus.review", [], max_tokens=4096)
                    tracer.chat_json("render.signal", [], max_tokens=8192)
                result = self.run_tool(calls, ["--reasoning-effort", setting])
                self.assertIsNone(result.error)
                self.assertEqual(result.profile["step_transport_overrides"], {})
                self.assertEqual([c["kwargs"]["reasoning_effort"] for c in result.api.calls], [setting] * 2)
                self.assertEqual([c["kwargs"]["max_completion_tokens"] for c in result.api.calls], [4096, 8192])

    def test_explicit_low_overrides_medium_only_on_exact_step(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            tracer.chat_json("corpus.review", [])
            tracer.chat_json("world.review", [])
        result = self.run_tool(calls, ["--reasoning-effort", "medium", "--corpus-review-reasoning-effort", "low"])
        self.assertIsNone(result.error)
        self.assertEqual([c["kwargs"]["reasoning_effort"] for c in result.api.calls], ["low", "medium"])

    def test_glm_flag_rejected_before_directory_or_dispatch(self):
        for setting in ("low", "medium"):
            result = self.run_tool(lambda *a: self.fail("Factory must not execute"),
                ["--model", "glm-4.7-nothinking", "--corpus-review-reasoning-effort", setting])
            self.assertIsInstance(result.error, SystemExit)
            self.assertEqual(result.error.code, 2)
            self.assertFalse(result.directory.exists())
            self.assertEqual(result.api.calls, [])

    def test_glm_without_flag_keeps_original_transport(self):
        def calls(directory, cfg):
            Tracer(SimpleNamespace(dir=directory)).chat_json("corpus.review", [], max_tokens=4096)
        result = self.run_tool(calls, ["--model", "glm-4.7-nothinking"])
        self.assertIsNone(result.error)
        wire = result.api.calls[0]["kwargs"]
        self.assertEqual(wire["max_tokens"], 4096)
        self.assertNotIn("reasoning_effort", wire)
        self.assertEqual(result.profile["step_transport_overrides"], {})

    def test_context_and_tracer_restored_if_original_method_raises(self):
        def raising(self, step, messages, **kwargs):
            self_test.assertEqual(smoke._TRACER_STEP.get(), step)
            raise RuntimeError("Offline tracer boundary failure")
        self_test = self
        def calls(directory, cfg):
            Tracer(SimpleNamespace(dir=directory)).chat_json("corpus.review", [])
        result = self.run_tool(calls, ["--corpus-review-reasoning-effort", "medium"], tracer_method=raising)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(result.profile["execution_status"], "failed")
        self.assertEqual(result.profile["admitted_calls"], 0)
        self.assertEqual(result.api.calls, [])

    def test_sdk_failure_stops_same_budget_and_context_is_reset(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            self.assertIn("__error__", tracer.chat_json("corpus.review", []))
            self.assertIsNone(smoke._TRACER_STEP.get())
            self.assertIn("__error__", tracer.chat_json("render.signal", []))
        result = self.run_tool(calls, ["--corpus-review-reasoning-effort", "medium"], sdk_failure=True)
        self.assertEqual(len(result.api.calls), 1)
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(len(result.requests), 1)
        self.assertEqual(result.requests[0]["parameters"]["reasoning_effort"], "medium")

    def test_call_and_money_limits_still_prevent_provider_dispatch(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            first = tracer.chat_json("corpus.review", [])
            second = tracer.chat_json("render.signal", [])
            self.assertIn("__error__", second)
        result = self.run_tool(calls, ["--max-calls", "1", "--corpus-review-reasoning-effort", "medium"])
        self.assertEqual(len(result.api.calls), 1)
        self.assertEqual(result.profile["admitted_calls"], 1)
        self.assertTrue(result.profile["stopped"])
        result = self.run_tool(calls, ["--max-cny", "0.001", "--corpus-review-reasoning-effort", "medium"])
        self.assertEqual(result.api.calls, [])
        self.assertEqual(result.profile["admitted_calls"], 0)
        self.assertTrue(result.profile["stopped"])

    def test_concurrent_call_context_does_not_cross_steps(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            barrier = threading.Barrier(2)
            def invoke(step):
                barrier.wait(timeout=3)
                return tracer.chat_json(step, [{"role": "user", "content": step}])
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(list(pool.map(invoke, ["corpus.review", "render.signal"])), [{"ok": True}] * 2)
        result = self.run_tool(calls, ["--corpus-review-reasoning-effort", "medium"])
        self.assertIsNone(result.error)
        settings = {call["kwargs"]["messages"][0]["content"]: call["kwargs"]["reasoning_effort"]
                    for call in result.api.calls}
        self.assertEqual(settings, {"corpus.review": "medium", "render.signal": "low"})
        self.assertEqual(result.profile["admitted_calls"], 2)

    def test_override_never_admits_a_different_model(self):
        def calls(directory, cfg):
            value = Tracer(SimpleNamespace(dir=directory)).chat_json("corpus.review", [], model="unapproved-model")
            self.assertIn("__error__", value)
        result = self.run_tool(calls, ["--corpus-review-reasoning-effort", "medium"])
        self.assertEqual(result.api.calls, [])
        self.assertEqual(result.profile["admitted_calls"], 0)
        self.assertTrue(result.profile["stopped"])


if __name__ == "__main__":
    unittest.main()
