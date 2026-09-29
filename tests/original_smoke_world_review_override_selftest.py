"""Actual wrapper/transport tests; fake SDK, network blocked, no model verdicts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke
fixture.smoke = smoke
Tracer, SimpleNamespace = fixture.Tracer, fixture.SimpleNamespace


class WorldReviewOverrideTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def test_world_exact_match_preserves_authors_disclosure_and_plain_calls(self):
        steps = ["world.structure", "world.disclosure_plan", "world.intrinsic",
                 "world.semantic_review", "world.semantic_review.extra", "WORLD.SEMANTIC_REVIEW",
                 "world.review", "corpus.review"]
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            for step in steps:
                self.assertEqual(tracer.chat_json(step, [{"role": "user", "content": "world.semantic_review"}],
                    max_tokens=3072), {"ok": True})
            cfg.chat_json([{"role": "user", "content": "world.semantic_review"}], max_tokens=3072)
        result = self.run_tool(calls, ["--world-review-reasoning-effort", "medium"])
        self.assertIsNone(result.error)
        self.assertEqual([r["kwargs"]["reasoning_effort"] for r in result.api.calls],
            ["medium" if s == "world.semantic_review" else "low" for s in steps] + ["low"])
        self.assertTrue(all(r["kwargs"]["max_completion_tokens"] == 3072 for r in result.api.calls))
        self.assertTrue(all(r["options"] == {"timeout": 150, "max_retries": 0} for r in result.api.calls))
        self.assertTrue(all(r["kwargs"]["model"] == "gpt-5.4-mini" for r in result.api.calls))
        self.assertEqual(result.profile["admitted_calls"], 9)
        self.assertEqual(result.profile["source_drift"], [])
        self.assertEqual(set(result.profile["step_transport_overrides"]), {"world.semantic_review"})
        setting = result.profile["step_transport_overrides"]["world.semantic_review"]
        self.assertEqual(setting["reasoning_effort"], "medium")
        self.assertEqual(setting["max_tokens"], "unchanged caller limit")
        self.assertEqual(setting["transport"]["profiles"]["low"]["deadline_seconds"], 150)
        self.assertTrue(all(r["max_attempts"] == 3 for r in result.rows if r.get("event") == "json_attempt"))

    def test_world_and_corpus_overrides_coexist_with_original_corpus_token_cap(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            for step in ["world.semantic_review", "corpus.review", "world.structure", "world.disclosure_plan"]:
                tracer.chat_json(step, [], max_tokens=4096)
        result = self.run_tool(calls, ["--world-review-reasoning-effort", "medium",
            "--corpus-review-reasoning-effort", "medium", "--corpus-review-max-tokens", "8192"])
        self.assertIsNone(result.error)
        self.assertEqual([(r["kwargs"]["reasoning_effort"], r["kwargs"]["max_completion_tokens"])
                          for r in result.api.calls], [("medium",4096),("medium",8192),("low",4096),("low",4096)])
        self.assertEqual(set(result.profile["step_transport_overrides"]), {"world.semantic_review", "corpus.review"})
        self.assertEqual(result.profile["transport"]["profiles"]["low"]["reasoning_effort"], "low")

    def test_default_and_explicit_low_keep_independent_global_and_corpus_settings(self):
        def calls(directory, cfg):
            for step in ["world.semantic_review", "world.structure", "corpus.review"]:
                Tracer(SimpleNamespace(dir=directory)).chat_json(step, [])
        default = self.run_tool(calls)
        self.assertIsNone(default.error)
        self.assertEqual(default.profile["step_transport_overrides"], {})
        self.assertEqual([r["kwargs"]["reasoning_effort"] for r in default.api.calls], ["low"]*3)
        controlled = self.run_tool(calls, ["--reasoning-effort", "medium", "--world-review-reasoning-effort", "low"])
        self.assertIsNone(controlled.error)
        self.assertEqual([r["kwargs"]["reasoning_effort"] for r in controlled.api.calls], ["low","medium","medium"])

    def test_glm_and_invalid_setting_rejected_before_artifact_or_provider(self):
        for flags in [["--model", "glm-4.7-nothinking", "--world-review-reasoning-effort", "low"],
                      ["--model", "glm-4.7-nothinking", "--world-review-reasoning-effort", "medium"],
                      ["--world-review-reasoning-effort", "high"]]:
            with self.subTest(flags=flags):
                result = self.run_tool(lambda *_: self.fail("Factory must not execute"), flags)
                self.assertIsInstance(result.error, SystemExit)
                self.assertEqual(result.error.code, 2)
                self.assertFalse(result.directory.exists())
                self.assertEqual(result.api.calls, [])

    def test_reservation_and_call_money_limits_are_unchanged(self):
        def one(directory, cfg):
            Tracer(SimpleNamespace(dir=directory)).chat_json("world.semantic_review", [], max_tokens=4096)
        baseline = self.run_tool(one)
        medium = self.run_tool(one, ["--world-review-reasoning-effort", "medium"])
        self.assertEqual(baseline.profile["reserved_upper_estimate_cny"], medium.profile["reserved_upper_estimate_cny"])
        self.assertEqual(medium.profile["admitted_calls"], 1)
        def two(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            tracer.chat_json("world.semantic_review", [])
            self.assertIn("__error__", tracer.chat_json("world.structure", []))
        limited = self.run_tool(two, ["--world-review-reasoning-effort", "medium", "--max-calls", "1"])
        self.assertEqual((len(limited.api.calls), limited.profile["admitted_calls"]), (1,1))
        self.assertTrue(limited.profile["stopped"])
        money = self.run_tool(one, ["--world-review-reasoning-effort", "medium", "--max-cny", "0.001"])
        self.assertEqual((len(money.api.calls), money.profile["admitted_calls"]), (0,0))
        self.assertTrue(money.profile["stopped"])

    def test_exception_restores_context_and_original_tracer(self):
        def raising(self, step, messages, **kw):
            if smoke._TRACER_STEP.get() != step:
                raise AssertionError("Missing step scope")
            raise RuntimeError("Offline boundary failure")
        result = self.run_tool(lambda directory, cfg: Tracer(SimpleNamespace(dir=directory)).chat_json(
            "world.semantic_review", []), ["--world-review-reasoning-effort", "medium"], tracer_method=raising)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(result.profile["execution_status"], "failed")
        self.assertEqual(result.profile["admitted_calls"], 0)
        self.assertIsNone(smoke._TRACER_STEP.get())

    def test_provider_error_keeps_shared_stop_no_retry(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            self.assertIn("__error__", tracer.chat_json("world.semantic_review", []))
            self.assertIsNone(smoke._TRACER_STEP.get())
            self.assertIn("__error__", tracer.chat_json("world.disclosure_plan", []))
        result = self.run_tool(calls, ["--world-review-reasoning-effort", "medium"], sdk_failure=True)
        self.assertEqual((len(result.api.calls), result.profile["admitted_calls"]), (1,1))
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.api.calls[0]["kwargs"]["reasoning_effort"], "medium")

    def test_concurrent_world_corpus_and_author_scopes_do_not_leak(self):
        steps = ["world.semantic_review", "corpus.review", "world.disclosure_plan"]
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory)); barrier = threading.Barrier(3)
            def invoke(step):
                barrier.wait(timeout=3)
                return tracer.chat_json(step, [{"role": "user", "content": step}])
            with ThreadPoolExecutor(max_workers=3) as pool:
                self.assertEqual(list(pool.map(invoke, steps)), [{"ok": True}]*3)
        result = self.run_tool(calls, ["--world-review-reasoning-effort", "medium",
                                     "--corpus-review-reasoning-effort", "low"])
        self.assertIsNone(result.error)
        self.assertEqual({r["kwargs"]["messages"][0]["content"]:r["kwargs"]["reasoning_effort"] for r in result.api.calls},
                         {"world.semantic_review":"medium","corpus.review":"low","world.disclosure_plan":"low"})
        self.assertEqual(result.profile["admitted_calls"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
