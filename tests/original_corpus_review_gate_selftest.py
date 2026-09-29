"""Original corpus freshness routing and direct grounding guards; no API calls.

Local validator doubles isolate orchestration from semantic receipt internals.
One test also exercises the real validator on an unreviewed signal document.
"""
from dataclasses import replace
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import corpus_contract, factory, grounding, grounding_review
from pipeline import run as run_module
from pipeline.run import Run, drive
from pipeline.world_state import WorldState, Timeline, Op, SET


def candidate_world():
    return WorldState(
        entities={"示例项目": {"状态": Timeline([Op(0, "2025-01-06", SET, "立项")])}},
        n_sessions=1,
    ).to_dict()


def grounding_result():
    return {
        "execution_complete": True, "delivery_safe": True,
        "overall": {"n": 0, "grounded": 0, "survival": 0},
        "by_line": {}, "by_capability": {}, "n_dropped": 0,
    }


class CorpusReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for p in (
            patch.object(run_module, "RUNS_DIR", Path(self.temp.name)),
            patch.object(socket.socket, "connect", side_effect=AssertionError("offline test")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("offline test")),
            patch.object(run_module.Tracer, "chat_json", side_effect=AssertionError("no model calls")),
            patch.object(run_module.Tracer, "chat_text", side_effect=AssertionError("no model calls")),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.run = Run("office", "offline_corpus_freshness")
        self.run.log = lambda *args: None
        self.run.write(factory.ART["whitepaper"], {"quality_contract": {"corpus_review": True}})
        self.run.write(factory.ART["world"], candidate_world())
        self.run.write(factory.ART["questions"], [])
        self.run.write(factory.ART["corpus"], {"fixture_current": False})
        self.run.write("00_about.json", {"public_protocol": "离线接线测试"})

    @staticmethod
    def audit(ws, corpus):
        if corpus.get("fixture_current"):
            return {"status": "passed", "issues": []}
        return {"status": "failed", "issues": [{"code": "fixture_stale_receipt"}]}

    def completed_stages(self):
        """Use the actual dependency graph/hooks, replacing only stage work."""
        self.calls = []
        stages = []
        for stage in factory.STAGES:
            def work(run, name=stage.name, artifact=stage.artifact):
                self.calls.append(name)
                run.write(artifact, {"fixture_current": True, "written_by": name})
            stages.append(replace(stage, fn=work,
                                  is_current=(lambda _: True)
                                  if stage.name in ("disclosure", "quality") else stage.is_current))
            if not self.run.has(stage.artifact):
                self.run.write(stage.artifact, {})
            self.run.mark(stage.name, stage.artifact, 0)
        return stages

    def test_corpus_stage_uses_original_freshness_hook(self):
        stage = next(stage for stage in factory.STAGES if stage.name == "corpus")
        self.assertIs(stage.is_current, factory._corpus_is_current)
        with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit) as audit:
            self.assertFalse(stage.is_current(self.run))
            self.run.write(factory.ART["corpus"], {"fixture_current": True})
            self.assertTrue(stage.is_current(self.run))
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(audit.call_args.args[0].to_dict(), candidate_world())

    def test_direct_grounding_rejects_stale_corpus_before_either_reviewer(self):
        for semantic in (False, True):
            with self.subTest(public_semantic_review=semantic):
                self.run.write(factory.ART["whitepaper"], {"quality_contract": {
                    "corpus_review": True, "public_semantic_review": semantic}})
                with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit), \
                     patch.object(grounding, "run_grounding") as lexical, \
                     patch.object(grounding_review, "review_grounding") as semantic_review:
                    with self.assertRaisesRegex(ValueError, "corpus"):
                        factory.stage_grounding(self.run)
                    lexical.assert_not_called()
                    semantic_review.assert_not_called()
                self.assertFalse(self.run.has(factory.ART["grounding"]))

    def test_direct_valid_corpus_reaches_declared_grounding_path(self):
        self.run.write(factory.ART["corpus"], {"fixture_current": True})
        for semantic in (False, True):
            with self.subTest(public_semantic_review=semantic):
                self.run.write(factory.ART["whitepaper"], {"quality_contract": {
                    "corpus_review": True, "public_semantic_review": semantic}})
                with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit) as audit, \
                     patch.object(grounding, "run_grounding", return_value=([], grounding_result())) as lexical, \
                     patch.object(grounding_review, "review_grounding",
                                  return_value=([], grounding_result(), {})) as semantic_review:
                    factory.stage_grounding(self.run)
                    audit.assert_called_once()
                    self.assertEqual(semantic_review.call_count, int(semantic))
                    self.assertEqual(lexical.call_count, int(not semantic))
                self.assertEqual(self.run.read(factory.ART["grounding"]), [])

    def test_real_validator_rejects_unreviewed_signal_before_model_calls(self):
        self.run.write(factory.ART["corpus"], {"sessions": [{"session_id": 0,
            "date": "2025-01-06", "docs": [{"doc_id": "unreviewed", "title": "项目记录",
                                          "content": "示例项目状态为立项。"}]}]})
        with patch.object(grounding, "run_grounding") as lexical, \
             patch.object(grounding_review, "review_grounding") as semantic:
            with self.assertRaisesRegex(ValueError, "corpus"):
                factory.stage_grounding(self.run)
            lexical.assert_not_called()
            semantic.assert_not_called()
        self.assertFalse(factory._corpus_is_current(self.run))

    def test_missing_or_malformed_artifacts_are_not_current(self):
        for artifact, value in ((factory.ART["corpus"], None),
                                (factory.ART["corpus"], "not-json"),
                                (factory.ART["world"], "[]")):
            with self.subTest(artifact=artifact, value=value):
                path = self.run.dir / artifact
                previous = path.read_bytes()
                try:
                    if value is None:
                        path.unlink()
                    else:
                        path.write_text(value, encoding="utf-8")
                    self.assertFalse(factory._corpus_is_current(self.run))
                finally:
                    path.write_bytes(previous)

    def test_legacy_whitepapers_do_not_acquire_review_obligation(self):
        for wp in ({}, {"quality_contract": {}}, {"quality_contract": {"corpus_review": False}}):
            with self.subTest(whitepaper=wp):
                self.run.write(factory.ART["whitepaper"], wp)
                with patch.object(corpus_contract, "validate_corpus", side_effect=AssertionError("legacy")) as audit, \
                     patch.object(grounding, "run_grounding", return_value=([], grounding_result())) as lexical:
                    self.assertTrue(factory._corpus_is_current(self.run))
                    factory.stage_grounding(self.run)
                    audit.assert_not_called()
                    lexical.assert_called_once()

    def test_midstream_restart_rejects_stale_corpus_before_any_work(self):
        stages = self.completed_stages()
        self.run.manifest["stages"]["grounding"]["done"] = False
        self.run._save_manifest()
        with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit):
            with self.assertRaisesRegex(SystemExit, "corpus"):
                drive(self.run, stages, from_stage="grounding")
        self.assertEqual(self.calls, [])

    def test_done_grounding_and_quality_cannot_skip_stale_corpus(self):
        stages = self.completed_stages()
        with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit):
            for target in ("grounding", "quality"):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(SystemExit, "corpus"):
                        drive(self.run, stages, only=target)
        self.assertEqual(self.calls, [])

    def test_stale_corpus_rebuild_invalidates_only_dependent_stages(self):
        stages = self.completed_stages()
        question_bytes = (self.run.dir / factory.ART["questions"]).read_bytes()
        with patch.object(corpus_contract, "validate_corpus", side_effect=self.audit):
            drive(self.run, stages, only="corpus")
            self.assertEqual(self.calls, ["corpus"])
            self.assertTrue(self.run.is_done("questions"))
            self.assertTrue(self.run.is_done("corpus"))
            for target in ("grounding", "quality"):
                self.assertFalse(self.run.is_done(target))
                self.assertEqual(self.run.manifest["stages"][target]["invalidated_by"], "corpus")
            drive(self.run, stages, from_stage="grounding")
        self.assertEqual(self.calls, ["corpus", "grounding", "quality"])
        self.assertEqual((self.run.dir / factory.ART["questions"]).read_bytes(), question_bytes)

    def test_done_legacy_run_still_skips_without_validating_corpus(self):
        stages = self.completed_stages()
        self.run.write(factory.ART["whitepaper"], {})
        with patch.object(corpus_contract, "validate_corpus", side_effect=AssertionError("legacy")) as audit:
            drive(self.run, stages, from_stage="grounding")
            audit.assert_not_called()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
