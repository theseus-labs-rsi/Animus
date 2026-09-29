"""Offline accepted-prose expansion, bounded failure, and resume integration."""
from copy import deepcopy
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import create_autospec, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-haystack")
os.environ.setdefault("MODEL", "offline-haystack")
from pipeline.render import corpus_scale, _top_up_haystack, render_corpus
from pipeline import factory, run as run_module
from tests.incremental_quality_selftest import world, reviewed_session


class HaystackTests(TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.corpus = {"sessions": [{"session_id": 0, "date": "2026-01-01", "docs": [
            {"doc_id": "signal", "content": "事实" * 50, "fact_refs": ["original"]}]}]}
        self.original = deepcopy(self.corpus["sessions"][0]["docs"][0])
        self.tracer = create_autospec(run_module.Tracer, instance=True)
        self.tracer.n = 0
        self.tracer.chat_text.return_value = "周边行政安排。" * 200
        self.saved = []

    def expand(self, target=0, ratio=9):
        _top_up_haystack(self.corpus, target, ratio, self.tracer, "system", ["冻结专名"], "周",
                        lambda: self.saved.append(deepcopy(self.corpus)), lambda *_: None)

    def test_ratio_increases_actual_prose_and_resume_makes_no_duplicate_calls(self):
        self.expand()
        self.assertGreaterEqual(corpus_scale(self.corpus, 0, 9)["filler_share"], .9)
        self.assertEqual(self.corpus["sessions"][0]["docs"][0], self.original)
        calls = self.tracer.chat_text.call_count
        self.expand()
        self.assertEqual(self.tracer.chat_text.call_count, calls)
        self.assertTrue(self.saved)

    def test_absolute_floor_and_ratio_zero(self):
        self.expand(target=2500, ratio=0)
        self.assertTrue(corpus_scale(self.corpus, 2500, 0)["target_met"])
        self.assertGreaterEqual(corpus_scale(self.corpus)["total_chars"], 2500)

    def test_zero_targets_do_not_generate(self):
        self.expand(target=0, ratio=0)
        self.tracer.chat_text.assert_not_called()

    def test_rejected_batch_stops_and_does_not_invent_accepted_prose(self):
        self.tracer.chat_text.return_value = "冻结专名的公告"
        self.expand()
        self.assertEqual(self.tracer.chat_text.call_count, 2)
        self.assertEqual(self.corpus["sessions"][0]["docs"], [self.original])
        self.assertFalse(corpus_scale(self.corpus, 0, 9)["target_met"])

    def test_failed_sibling_preserves_success_and_resume_only_fills_deficit(self):
        self.tracer.chat_text.side_effect = ["周边材料" * 100, {"__error__": "budget exhausted"}]
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            self.expand()
        self.assertEqual(self.saved[-1], self.corpus)
        self.assertEqual(len(self.corpus["sessions"][0]["docs"]), 2)
        self.tracer.chat_text.side_effect = None
        self.expand()
        self.assertTrue(corpus_scale(self.corpus, 0, 9)["target_met"])
        ids = [d["doc_id"] for d in self.corpus["sessions"][0]["docs"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_short_output_has_finite_allowance(self):
        self.tracer.chat_text.return_value = "字"
        self.expand()
        self.assertLessEqual(self.tracer.chat_text.call_count, 12)
        self.assertFalse(corpus_scale(self.corpus, 0, 9)["target_met"])

    def test_invalid_ratio(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.expand(ratio=value)

    def test_real_tracer_accepts_the_prompt_and_checkpoints_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)):
            run = run_module.Run("office", "real-tracer")
            self.tracer = run.tracer
            with patch("config.chat", return_value="周边行政安排。" * 200) as provider:
                self.expand()
            self.assertTrue(corpus_scale(self.corpus, 0, 9)["target_met"])
            self.assertEqual(run.tracer.n, provider.call_count)
            self.assertTrue((run.dir / "prompts.jsonl").is_file())

    def test_factory_size_change_reuses_body_and_invalidates_old_scale(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(factory, "diversity_report", return_value={}):
            run = run_module.Run("office", "haystack-resume", config_meta={"target_tokens": 0, "haystack_ratio": 0})
            run.log = lambda *_: None
            run.tracer = self.tracer
            run.write("01_whitepaper.json", {"domain_profile": {}})
            ws = world()
            run.write("02_world.json", ws.to_dict())
            material = {"corpus": {"sessions": [reviewed_session(ws, sid) for sid in range(2)]},
                        "done_weeks": [0, 1]}
            run.write("05_corpus.json", material)
            initial = corpus_scale(material["corpus"], 0, 0)
            run.write("05_corpus_scale.json", {**initial, "binding": factory._corpus_scale_binding(run)})
            self.assertTrue(factory._corpus_is_current(run))
            run.manifest["config"].update(haystack_ratio=9, target_tokens=4000)
            self.assertFalse(factory._corpus_is_current(run))
            factory.stage_corpus(run)
            current = run.read("05_corpus.json")["corpus"]
            for index, session in enumerate(material["corpus"]["sessions"]):
                self.assertEqual(current["sessions"][index]["docs"][0], session["docs"][0])
            self.assertTrue(run.read("05_corpus_scale.json")["target_met"])
            self.assertTrue(factory._corpus_is_current(run))
            self.tracer.chat_json.assert_not_called()

    def test_completed_periods_are_used_by_real_renderer(self):
        ws = world()
        corpus = {"sessions": [reviewed_session(ws, sid) for sid in range(2)]}
        original = deepcopy(corpus)
        render_corpus({"domain_profile": {}, "quality_contract": {"corpus_review": True}}, ws, 4000,
                      self.tracer, corpus, {0, 1}, lambda: None, lambda *_: None, haystack_ratio=9)
        for index, session in enumerate(original["sessions"]):
            self.assertEqual(corpus["sessions"][index]["docs"][0], session["docs"][0])
        self.assertTrue(corpus_scale(corpus, 4000, 9)["target_met"])
        self.tracer.chat_json.assert_not_called()

    def test_legacy_complete_corpus_and_interrupted_expansion_preserve_every_document(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(factory, "diversity_report", return_value={}):
            run = run_module.Run("office", "legacy-resume", config_meta={"target_tokens": 4000, "haystack_ratio": 9})
            run.log = lambda *_: None
            run.tracer = self.tracer
            run.write("01_whitepaper.json", {"domain_profile": {}})
            ws = world()
            run.write("02_world.json", ws.to_dict())
            original = {"corpus": {"sessions": [reviewed_session(ws, sid) for sid in range(2)]}, "done_weeks": [0, 1]}
            run.write("05_corpus.json", original)
            self.tracer.chat_text.side_effect = ["周边行政安排。" * 200, {"__error__": "offline transport"},
                                                "周边公告。" * 200, "周边活动。" * 200, "周边活动。" * 200]
            with self.assertRaisesRegex(RuntimeError, "offline transport"):
                factory.stage_corpus(run)
            self.assertEqual(run.read("05_corpus.json"), original)
            saved = run.read(factory.CORPUS_CKPT)
            self.assertEqual(saved["done_weeks"], [0, 1])
            self.assertGreater(corpus_scale(saved["corpus"])["documents"], 2)
            self.assertFalse(run.read("05_corpus_scale.json")["attempt_complete"])
            self.assertEqual(run.read("05_corpus_scale.json")["retry_from_checkpoint"], "--force --from corpus")
            self.tracer.chat_text.side_effect = None
            factory.stage_corpus(run)
            self.assertTrue(run.read("05_corpus_scale.json")["target_met"])
            final = run.read("05_corpus.json")["corpus"]
            for index, session in enumerate(saved["corpus"]["sessions"]):
                self.assertEqual(final["sessions"][index]["docs"][:len(session["docs"])], session["docs"])
            self.tracer.chat_json.assert_not_called()


if __name__ == "__main__":
    main(verbosity=2)
