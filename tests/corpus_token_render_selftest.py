"""Token-sized corpus expansion through real renderer and factory, no model I/O."""
from copy import deepcopy
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import create_autospec, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("OPENAI_API_KEY", "offline-token-render")
os.environ.setdefault("MODEL", "offline-token-render")
from pipeline.corpus_tokens import CorpusTokenCounter
from pipeline.render import _top_up_haystack_tokens, CorpusTokenTargetUnmet, render_corpus
from pipeline import factory, run as run_module
from pipeline import render as renderer
from pipeline.world_state import WorldState
from incremental_quality_selftest import world, reviewed_session


class CorpusTokenRenderTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = CorpusTokenCounter()

    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.corpus = {"sessions": [{"session_id": 0, "date": "2026-01-01", "docs": [
            {"doc_id": "core", "content": "事实" * 50}]}]}
        self.state, self.receipts = {}, []
        self.tracer = create_autospec(run_module.Tracer, instance=True)
        self.tracer.n = 0
        self.tracer.chat_text.return_value = "外围行政档案。" * 200

    def expand(self, target=2500, ratio=0, writer=None):
        return _top_up_haystack_tokens(
            self.corpus, target, ratio, self.tracer, "system", ["冻结专名"], "周",
            lambda: None, self.state, lambda value: self.receipts.append(value),
            self.counter, lambda *_: None, filler_writer=writer)

    def test_token_target_and_exact_completed_resume(self):
        original = deepcopy(self.corpus["sessions"][0]["docs"][0])
        self.expand()
        self.assertGreaterEqual(self.counter.measure(self.corpus)["total"]["tokens"], 2500)
        self.assertTrue(self.state["target_met"])
        self.assertEqual(self.corpus["sessions"][0]["docs"][0], original)
        calls = self.tracer.chat_text.call_count
        self.expand()
        self.assertEqual(calls, self.tracer.chat_text.call_count)
        self.assertTrue(self.receipts[0]["deficit_tokens"])
        self.assertTrue(self.receipts[-1]["target_met"])

    def test_ratio_uses_token_totals(self):
        self.expand(target=0, ratio=9)
        measured = self.counter.measure(self.corpus)
        self.assertGreaterEqual(measured["filler"]["tokens"], 9 * measured["core"]["tokens"])

    def test_empty_and_low_yield_stop_across_resume(self):
        for body in ("", "字"):
            with self.subTest(body=body):
                self.state.clear()
                self.corpus["sessions"][0]["docs"] = self.corpus["sessions"][0]["docs"][:1]
                self.tracer.chat_text.reset_mock()
                self.tracer.chat_text.return_value = body
                with self.assertRaises(CorpusTokenTargetUnmet):
                    self.expand()
                calls = self.tracer.chat_text.call_count
                self.assertLessEqual(calls, 16)
                with self.assertRaises(CorpusTokenTargetUnmet):
                    self.expand()
                self.assertEqual(calls, self.tracer.chat_text.call_count)

    def test_failed_job_keeps_exact_prompt_and_successful_sibling(self):
        seen, fail = [], [True]
        def writer(tracer, messages, blocked):
            user = messages[-1]["content"]
            seen.append(user)
            if "token-1-" in user and fail[0]:
                raise RuntimeError("offline transport")
            return [{"content": "外围档案。" * 30}]
        with self.assertRaisesRegex(RuntimeError, "offline transport"):
            self.expand(target=1800, writer=writer)
        saved = deepcopy(self.corpus)
        pending = deepcopy(self.state["topup"]["pending_jobs"])
        self.assertEqual(len(pending), 1)
        self.assertEqual(len(saved["sessions"][0]["docs"]), 2)
        failed_prompt = pending[0]["user"]
        fail[0] = False
        self.expand(target=1800, writer=writer)
        self.assertEqual(seen.count(failed_prompt), 2)
        self.assertEqual(self.corpus["sessions"][0]["docs"][:2], saved["sessions"][0]["docs"])
        ids = [doc["doc_id"] for doc in self.corpus["sessions"][0]["docs"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_size_change_reuses_text_and_resets_only_new_target_plan(self):
        self.expand(target=300)
        old = deepcopy(self.corpus)
        previous_hash = self.state["plan_hash"]
        self.expand(target=3000)
        self.assertNotEqual(self.state["plan_hash"], previous_hash)
        self.assertEqual(self.corpus["sessions"][0]["docs"][:len(old["sessions"][0]["docs"])],
                         old["sessions"][0]["docs"])

    def test_full_renderer_ignores_legacy_character_target_in_token_mode(self):
        ws = world()
        current = {"sessions": [reviewed_session(ws, sid) for sid in range(2)]}
        render_corpus({"domain_profile": {}, "quality_contract": {"corpus_review": True}},
                      ws, 100_000_000, self.tracer, current, {0, 1}, lambda: None,
                      lambda *_: None, corpus_token_target=2500, token_state=self.state,
                      token_save_cb=self.receipts.append, haystack_ratio=0)
        self.assertGreaterEqual(self.counter.measure(current)["total"]["tokens"], 2500)
        self.assertLess(self.counter.measure(current)["total"]["characters"], 100_000_000)
        self.tracer.chat_json.assert_not_called()

    def test_fresh_draft_uses_small_samples_then_real_token_measurement(self):
        ws = WorldState(entities={}, n_sessions=2)
        current, completed = {"sessions": []}, set()
        render_corpus({"domain_profile": {}}, ws, 100_000_000,
                      self.tracer, current, completed, lambda: None, lambda *_: None,
                      corpus_token_target=300, token_state=self.state,
                      token_save_cb=self.receipts.append, haystack_ratio=0)
        self.assertEqual(completed, {0, 1})
        self.assertEqual(self.tracer.chat_text.call_count, 2)
        self.assertTrue(self.receipts[0]["target_met"])
        self.assertEqual(self.receipts[0]["measurement"]["total"]["documents"], 2)

    def test_process_exit_after_text_responses_replays_exact_checkpoints(self):
        import config
        with tempfile.TemporaryDirectory() as directory:
            self.tracer = run_module.Tracer(SimpleNamespace(dir=Path(directory)))
            with patch.object(config, "chat", return_value="外围行政档案。" * 200) as provider:
                def interrupted_map(function, values, **kwargs):
                    result = [function(value) for value in values]
                    raise KeyboardInterrupt("process exit before corpus merge")
                with patch.object(config, "pmap", side_effect=interrupted_map):
                    with self.assertRaises(KeyboardInterrupt):
                        self.expand(target=1400)
                saved_state = deepcopy(self.receipts[-1])
                self.assertEqual(len(saved_state["topup"]["pending_jobs"]), 2)
                self.assertEqual(len(self.corpus["sessions"][0]["docs"]), 1)
                calls = provider.call_count
                self.state = saved_state
                self.expand(target=1400)
                self.assertTrue(self.state["target_met"])
                self.assertEqual(provider.call_count, calls)

    def test_factory_budget_stop_keeps_published_corpus_and_pending_jobs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(factory, "diversity_report", return_value={}):
            run = run_module.Run("office", "token-budget", config_meta={
                "target_tokens": 0, "corpus_token_target": 2000,
                "corpus_tokenizer": "cl100k_base@0.12.0", "haystack_ratio": 0})
            run.log = lambda *_: None
            run.tracer = self.tracer
            run.write("01_whitepaper.json", {"domain_profile": {}})
            ws = world()
            run.write("02_world.json", ws.to_dict())
            original = {"corpus": {"sessions": [reviewed_session(ws, sid) for sid in range(2)]},
                        "done_weeks": [0, 1]}
            run.write("05_corpus.json", original)
            def stopped_writer(tracer, messages, blocked):
                if "token-1-" in messages[-1]["content"]:
                    error = RuntimeError("budget stop fixture")
                    error.global_stop = True
                    raise error
                return [{"content": "外围行政档案。" * 30}]
            with patch.object(renderer, "_FillerWriter", return_value=stopped_writer):
                with self.assertRaisesRegex(RuntimeError, "budget stop fixture"):
                    factory.stage_corpus(run)
            self.assertEqual(run.read("05_corpus.json"), original)
            report = run.read("05_corpus_token_scale.json")
            checkpoint = run.read(factory.CORPUS_CKPT)
            self.assertFalse(report["attempt_complete"])
            self.assertFalse(report["target_met"])
            self.assertGreater(report["candidate_measurement"]["total"]["tokens"], report["total_tokens"])
            pending = checkpoint["token_state"]["topup"]["pending_jobs"]
            self.assertEqual(len(pending), 1)
            saved = deepcopy(checkpoint["corpus"])
            factory.stage_corpus(run)
            self.assertTrue(run.read("05_corpus_token_scale.json")["target_met"])
            final = run.read("05_corpus.json")["corpus"]
            for sid, session in enumerate(saved["sessions"]):
                self.assertEqual(final["sessions"][sid]["docs"][:len(session["docs"])], session["docs"])

    def test_factory_writes_bound_token_report_and_reuses_published_body(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(factory, "diversity_report", return_value={}):
            run = run_module.Run("office", "token-size", config_meta={
                "target_tokens": 100_000_000, "corpus_token_target": 2500,
                "corpus_tokenizer": "cl100k_base", "haystack_ratio": 0})
            run.log = lambda *_: None
            run.tracer = self.tracer
            run.write("01_whitepaper.json", {"domain_profile": {}})
            ws = world()
            run.write("02_world.json", ws.to_dict())
            initial = {"corpus": {"sessions": [reviewed_session(ws, sid) for sid in range(2)]},
                       "done_weeks": [0, 1]}
            run.write("05_corpus.json", initial)
            factory.stage_corpus(run)
            report = run.read("05_corpus_token_scale.json")
            self.assertTrue(report["target_met"])
            self.assertTrue(report["attempt_complete"])
            self.assertEqual(report["binding"], factory._corpus_scale_binding(run))
            self.assertEqual(report["measurement"]["total"]["tokens"],
                             self.counter.measure(run.read("05_corpus.json"))["total"]["tokens"])
            with patch.object(factory, "_require_current_corpus_review", return_value=None):
                self.assertTrue(factory._corpus_is_current(run))
                damaged = dict(report)
                damaged["target_tokens"] += 1
                run.write("05_corpus_token_scale.json", damaged)
                self.assertFalse(factory._corpus_is_current(run))
                run.write("05_corpus_token_scale.json", report)
            for sid, session in enumerate(initial["corpus"]["sessions"]):
                self.assertEqual(run.read("05_corpus.json")["corpus"]["sessions"][sid]["docs"][0], session["docs"][0])
            calls = self.tracer.chat_text.call_count
            factory.stage_corpus(run)
            self.assertEqual(calls, self.tracer.chat_text.call_count)

    def test_formal_material_shortfall_is_not_filled_by_haystack(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(factory, "diversity_report", return_value={}):
            run = run_module.Run("office", "core-short", config_meta={
                "target_tokens": 0, "corpus_token_target": 2500,
                "core_corpus_token_target": 1000,
                "corpus_tokenizer": "cl100k_base@0.12.0", "haystack_ratio": 0})
            run.log = lambda *_: None
            run.tracer = self.tracer
            run.write("01_whitepaper.json", {"domain_profile": {}})
            ws = world()
            run.write("02_world.json", ws.to_dict())
            run.write("05_corpus.json", {"corpus": {"sessions": [reviewed_session(ws, sid)
                for sid in range(2)]}, "done_weeks": [0, 1]})
            factory.stage_corpus(run)
            report = run.read("05_corpus_token_scale.json")
            self.assertGreaterEqual(report["measurement"]["total"]["tokens"], 2500)
            self.assertLess(report["measurement"]["core"]["tokens"], 1000)
            self.assertFalse(report["core_target_met"])
            self.assertFalse(report["target_met"])
            self.assertEqual(report["warning"], "formal_core_token_target_unmet")
            with patch.object(factory, "_require_current_corpus_review", return_value=None):
                self.assertFalse(factory._corpus_is_current(run))


if __name__ == "__main__":
    main(verbosity=2)
