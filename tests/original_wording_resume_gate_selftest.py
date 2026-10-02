"""Original questions/grounding resume boundary; local opinions, zero API calls."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.answer_task_review import POLICY_VERSION
from pipeline import factory, grounding, grounding_review, question_wording
from pipeline import run as run_module
from pipeline.question_contract import attach_question_contract, bind_question_world
from pipeline.run import Run, drive
from pipeline.world_state import WorldState, Timeline, Op, SET


def grounding_result(questions):
    return {"execution_complete": True, "delivery_safe": True,
            "overall": {"n": len(questions), "grounded": len(questions), "survival": 1},
            "by_line": {}, "by_capability": {}, "n_dropped": 0}


class WordingResumeGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for guard in (
            patch.object(run_module, "RUNS_DIR", Path(self.temp.name)),
            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")),
            patch.object(run_module.Tracer, "chat_json", side_effect=AssertionError("model forbidden")),
            patch.object(run_module.Tracer, "chat_text", side_effect=AssertionError("model forbidden")),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.run = Run("office", "offline_wording_resume")
        self.run.log = lambda *args: None
        self.wp = {"quality_contract": {"scoring_policy": POLICY_VERSION,
                                         "public_semantic_review": True}}
        self.ws = WorldState({"测试项目": {"负责人": Timeline([
            Op(0, "2025-01-06", SET, "李明")])}}, n_sessions=1)
        self.question = attach_question_contract(bind_question_world({
            "line": "L1_timeline", "capability": "IE", "entity": "测试项目", "field": "负责人",
            "gt": {"value": "李明", "at_week": 0}, "aux": {"at_week": 0},
            "evidence_sessions": [0]}, self.ws), self.wp)
        self.question["question"] = self.question["question_contract"]["canonical_question"]
        # Produce a current receipt through the actual wording API, using a
        # fixed local opinion. This tests binding, not the opinion's semantics.
        receipt = question_wording.review_wording(
            self.question["question"], self.question["question_contract"],
            chat_json=lambda *a, **k: {"verdict": "equivalent", "issues": [],
                                      "reason": "Local fixture preserves the requested scope."},
            model="offline-fixture")
        self.question["question_validation"] = {"semantic_review": receipt}
        self.run.write(factory.ART["whitepaper"], self.wp)
        self.run.write(factory.ART["world"], self.ws.to_dict())
        self.run.write(factory.ART["orders"], [self.question])
        self.run.write(factory.ART["questions"], [self.question])
        self.run.write(factory.ART["corpus"], {"sessions": []})
        self.run.write("00_about.json", {"answer_protocol": factory._answer_protocol(self.wp)})

    def outdated(self):
        old = deepcopy(self.question)
        old["question_validation"]["semantic_review"]["binding"]["version"] = "original-question-wording/v1"
        self.assertNotEqual(question_wording.VERSION, "original-question-wording/v1")
        return old

    def completed_stages(self, *, real_questions=False):
        self.calls = []
        stages = []
        for stage in factory.STAGES:
            def work(run, name=stage.name, artifact=stage.artifact):
                self.calls.append(name)
                run.write(artifact, {"written_by": name})
            function = stage.fn if real_questions and stage.name == "questions" else work
            stages.append(replace(stage, fn=function,
                is_current=(lambda _: True)
                if stage.name in ("disclosure", "grounding", "quality") else stage.is_current))
            if not self.run.has(stage.artifact):
                self.run.write(stage.artifact, {})
            self.run.mark(stage.name, stage.artifact, 0)
        return stages

    def test_registered_hook_validates_every_original_question(self):
        stage = next(s for s in factory.STAGES if s.name == "questions")
        self.assertIs(stage.is_current, factory._questions_is_current)
        questions = [deepcopy(self.question), self.outdated(), deepcopy(self.question)]
        for i, question in enumerate(questions):
            question["qid"] = f"q{i}"
        self.run.write(factory.ART["questions"], questions)
        with patch.object(question_wording, "validate_wording", wraps=question_wording.validate_wording) as check:
            self.assertFalse(stage.is_current(self.run))
        self.assertEqual([call.args[0] for call in check.call_args_list], questions)
        self.assertEqual(self.run.read(factory.ART["questions"]), questions)

    def test_missing_stale_or_unpassed_receipt_stops_before_grounding_calls(self):
        for mode in ("missing", "old_version", "text", "contract", "pending"):
            bad = deepcopy(self.question)
            if mode == "missing": bad.pop("question_validation")
            elif mode == "old_version": bad = self.outdated()
            elif mode == "text": bad["question"] += " 并预测下一期。"
            elif mode == "contract": bad["question_contract"]["canonical_question"] += " changed"
            else:
                receipt = bad["question_validation"]["semantic_review"]
                receipt["status"] = "pending"
                receipt["opinion"].update(verdict="unresolved", issues=["scope"])
            for semantic in (False, True):
                with self.subTest(mode=mode, semantic=semantic):
                    wp = deepcopy(self.wp)
                    wp["quality_contract"]["public_semantic_review"] = semantic
                    self.run.write(factory.ART["whitepaper"], wp)
                    self.run.write(factory.ART["questions"], [bad])
                    with patch.object(grounding, "run_grounding") as lexical, \
                         patch.object(grounding_review, "review_grounding") as model:
                        with self.assertRaisesRegex(ValueError, "questions"):
                            factory.stage_grounding(self.run)
                        lexical.assert_not_called()
                        model.assert_not_called()
                    self.assertFalse(self.run.has(factory.ART["grounding"]))
                    self.assertEqual(self.run.read(factory.ART["questions"]), [bad])

    def test_current_receipt_reaches_declared_grounding_route(self):
        questions = [self.question]
        for semantic in (False, True):
            with self.subTest(semantic=semantic):
                wp = deepcopy(self.wp)
                wp["quality_contract"]["public_semantic_review"] = semantic
                self.run.write(factory.ART["whitepaper"], wp)
                with patch.object(grounding, "run_grounding",
                                  return_value=(questions, grounding_result(questions))) as lexical, \
                     patch.object(grounding_review, "review_grounding",
                                  return_value=(questions, grounding_result(questions), {})) as model:
                    self.assertTrue(factory._questions_is_current(self.run))
                    factory.stage_grounding(self.run)
                    self.assertEqual(model.call_count, int(semantic))
                    self.assertEqual(lexical.call_count, int(not semantic))
                self.assertEqual(self.run.read(factory.ART["grounding"]), questions)

    def test_empty_current_question_list_does_not_invent_candidates(self):
        self.run.write(factory.ART["questions"], [])
        with patch.object(question_wording, "validate_wording", wraps=question_wording.validate_wording) as check:
            self.assertTrue(factory._questions_is_current(self.run))
            check.assert_not_called()

    def test_missing_or_malformed_question_artifact_not_current(self):
        path = self.run.dir / factory.ART["questions"]
        original = path.read_bytes()
        for content in (None, "not-json", "{}", "[42]"):
            with self.subTest(content=content):
                try:
                    if content is None: path.unlink()
                    else: path.write_text(content, encoding="utf-8")
                    self.assertFalse(factory._questions_is_current(self.run))
                finally:
                    path.write_bytes(original)

    def test_legacy_does_not_acquire_wording_review_requirement(self):
        self.run.write(factory.ART["questions"], [self.outdated()])
        for wp in ({}, {"quality_contract": {}}, {"quality_contract": {"scoring_policy": None}}):
            with self.subTest(wp=wp):
                self.run.write(factory.ART["whitepaper"], wp)
                with patch.object(question_wording, "validate_wording", side_effect=AssertionError("legacy")) as check, \
                     patch.object(grounding, "run_grounding", return_value=([], grounding_result([]))) as lexical:
                    self.assertTrue(factory._questions_is_current(self.run))
                    factory.stage_grounding(self.run)
                    check.assert_not_called()
                    lexical.assert_called_once()

    def test_midstream_or_done_descendants_cannot_skip_stale_questions(self):
        stages = self.completed_stages()
        self.run.write(factory.ART["questions"], [self.outdated()])
        for options in ({"from_stage": "grounding"}, {"only": "grounding"}, {"only": "quality"}):
            with self.subTest(options=options), self.assertRaisesRegex(SystemExit, "questions|grounding"):
                drive(self.run, stages, **options)
        self.assertEqual(self.calls, [])

    def test_old_receipt_reruns_original_questions_and_invalidates_only_descendants(self):
        stages = self.completed_stages(real_questions=True)
        self.run.write(factory.ART["questions"], [self.outdated()])
        corpus_before = (self.run.dir / factory.ART["corpus"]).read_bytes()
        def rephrase(orders, wp, tracer, log, *, audit, checkpoint_path, corpus, public_protocol):
            self.assertEqual(corpus, self.run.read(factory.ART["corpus"]))
            self.assertIsInstance(public_protocol, str)
            self.assertEqual(checkpoint_path, self.run.dir / "04_wording.ckpt.json")
            self.assertEqual(orders, [self.question])
            audit.update(original_count=1, returned_qids=[self.question["qid"]], execution_status="completed")
            return [deepcopy(self.question)]
        with patch.object(factory, "phrase_questions", side_effect=rephrase) as original_author:
            drive(self.run, stages, only="questions")
            original_author.assert_called_once()
        self.assertEqual(self.run.read(factory.ART["questions"]), [self.question])
        self.assertEqual(self.run.read("04_wording_report.json")["original_count"], 1)
        self.assertTrue(self.run.is_done("questions"))
        for name in ("input", "whitepaper", "world", "orders", "well_posed", "corpus"):
            self.assertTrue(self.run.is_done(name), name)
        for name in ("grounding", "quality"):
            self.assertFalse(self.run.is_done(name))
            self.assertEqual(self.run.manifest["stages"][name]["invalidated_by"], "questions")
        self.assertEqual((self.run.dir / factory.ART["corpus"]).read_bytes(), corpus_before)
        with patch.object(factory, "phrase_questions", side_effect=AssertionError("current stage must skip")):
            drive(self.run, stages, only="questions")
        drive(self.run, stages, from_stage="grounding")
        self.assertEqual(self.calls, ["grounding", "quality"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
