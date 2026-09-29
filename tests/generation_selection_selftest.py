"""Offline production-tail tests: actual driver, filtering and standard export."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.grading import JUDGE_VERSION
from eval.provenance import record_provenance
from pipeline import calibration as c
from pipeline import run as run_module
from pipeline.run import Run, Stage, drive
from pipeline.quality import evaluate_release


SETTINGS = {"systems": ["A", "C"], "answering_model": "offline-solver", "judge_model": "offline-judge",
            "max_judge_calls": 100, "workers": 2, "timeout_s": 60, "keep_easy_ratio": .2, "seed": 7}


class GenerationSelectionTest(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(run_module, "RUNS_DIR", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.run = Run("test-domain", "a world", config_meta={"calibration": deepcopy(SETTINGS)})
        self.questions = [{"qid": f"q{i}", "question": f"第1期项目{i}的值？", "gt": {"value": f"v{i}"},
                           "line": "L1_timeline" if i < 6 else "L2_relational", "capability": "IE"}
                          for i in range(10)]
        bad = {**self.questions[0], "qid": "bad", "question": "invalid question"}
        for name, value in {
            "00_about.json": {"public_protocol": "只依据提供的材料回答。"},
            "01_whitepaper.json": {},
            "02_world.json": {"entities": {}},
            "04_questions.json": [*self.questions, bad],
            "05_corpus.json": {"corpus": {"sessions": [{"session_id": 1, "date": "2026-01-01",
                "docs": [{"doc_id": "d1", "type": "note", "content": "完整语料，包括干扰内容"}]}]}},
            "06_grounded_questions.json": self.questions,
            "06_grounding_report.json": {"drops": [{"qid": "bad"}], "pending": []},
            "06_semantic_review.json": {"offline_fixture": True},
        }.items():
            self.run.write(name, value)
        self.run.write("07_release.json", evaluate_release(self.run.dir))
        self.stages = [Stage("quality", [], lambda run: None, "07_release.json"),
                       Stage("calibration", ["quality"], c.stage_calibration, c.CALIBRATION_ARTIFACT,
                             is_current=c.calibration_is_current),
                       Stage("selection", ["calibration"], c.stage_selection, c.SELECTION_ARTIFACT,
                             is_current=c.selection_is_current)]

    def results(self, run, directory, settings, *, all_correct=False):
        directory.mkdir(parents=True, exist_ok=True)
        result = {"systems": settings["systems"], "results": {}}
        for system in settings["systems"]:
            rows = []
            for i, q in enumerate(run.read("06_grounded_questions.json")):
                correct = True if all_correct else False if i == 8 and system == "C" else None if i == 9 and system == "C" else True
                grade = {"version": JUDGE_VERSION, "verdict": "correct" if correct else "error" if correct is None else "incorrect",
                         "correct": correct, "path": "offline_test", "reason": "fixture"}
                rows.append({**deepcopy(q), "pred": "fixture answer", "correct": correct,
                             "judgement": grade, "evaluation_provenance": record_provenance(q, c._context(run))})
            result["results"][system] = {"records": rows}
        path = directory / "results.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return path

    def source_hashes(self):
        return {n: hashlib.sha256((self.run.dir / n).read_bytes()).hexdigest()
                for n in [*c.INPUTS, "04_questions.json", "06_grounding_report.json"]}

    def change_settings(self, **changes):
        self.run.manifest["config"]["calibration"].update(changes)
        self.run._save_manifest()

    def test_tail_filters_exports_and_preserves_all_sources(self):
        original = self.source_hashes()
        with patch.object(c, "execute_calibration", side_effect=self.results) as execute:
            drive(self.run, self.stages)
            execute.assert_called_once()
        report = self.run.read(c.SELECTION_ARTIFACT)
        self.assertEqual(report["counts"]["input"], 10)
        self.assertEqual(report["counts"]["all_correct"], 8)
        self.assertEqual(report["counts"]["kept_easy_sample"], 3)
        self.assertEqual(report["counts"]["kept"], 5)
        self.assertEqual(report["counts"]["kept_incomplete"], 1)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(self.source_hashes(), original)
        self.assertTrue(c.selection_is_current(self.run))
        package = self.run.dir / report["benchmark"]
        public = json.loads((package / "public/questions.json").read_text(encoding="utf-8"))
        refs = json.loads((package / "references/questions.json").read_text(encoding="utf-8"))
        self.assertEqual([q["qid"] for q in public], [q["qid"] for q in refs])
        self.assertTrue({"q8", "q9"}.issubset({q["qid"] for q in public}))
        self.assertNotIn("bad", {q["qid"] for q in public})
        self.assertTrue(all("gt" not in q and "answer" not in q for q in public))
        self.assertEqual(json.loads((package / "public/material.json").read_text(encoding="utf-8"))[0]["content"],
                         "完整语料，包括干扰内容")

    def test_resume_and_ratio_change_do_not_repeat_answers(self):
        with patch.object(c, "execute_calibration", side_effect=self.results) as execute:
            drive(self.run, self.stages)
            identity = c.calibration_identity(self.run)
            previous_package = self.run.dir / self.run.read(c.SELECTION_ARTIFACT)["benchmark"]
            drive(self.run, self.stages)
            self.assertEqual(execute.call_count, 1)
            self.change_settings(keep_easy_ratio=0)
            self.assertEqual(c.calibration_identity(self.run), identity)
            drive(self.run, self.stages, from_stage="selection")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(self.run.read(c.SELECTION_ARTIFACT)["counts"]["kept"], 2)
            self.assertTrue(previous_package.is_dir())

    def test_changed_inputs_require_new_calibration(self):
        with patch.object(c, "execute_calibration", side_effect=self.results):
            drive(self.run, self.stages)
        self.run.write("02_world.json", {"entities": {}, "changed": True})
        self.assertFalse(c.calibration_is_current(self.run))
        with self.assertRaises(SystemExit):
            drive(self.run, self.stages, from_stage="selection")

    def test_calibration_failure_retains_sources_and_can_resume(self):
        original = self.source_hashes()
        with patch.object(c, "execute_calibration", side_effect=RuntimeError("offline interruption")):
            with self.assertRaises(RuntimeError):
                drive(self.run, self.stages)
        self.assertEqual(self.source_hashes(), original)
        self.assertFalse(self.run.has(c.SELECTION_ARTIFACT))
        with patch.object(c, "execute_calibration", side_effect=self.results):
            drive(self.run, self.stages, from_stage="calibration")
        self.assertTrue(c.selection_is_current(self.run))

    def test_remove_all_easy_can_deliver_an_explicit_empty_result(self):
        self.change_settings(keep_easy_ratio=0)
        with patch.object(c, "execute_calibration", side_effect=lambda *a: self.results(*a, all_correct=True)):
            drive(self.run, self.stages)
        report = self.run.read(c.SELECTION_ARTIFACT)
        self.assertEqual(report["status"], "empty")
        self.assertIsNone(report["benchmark"])
        self.assertEqual(self.run.read(c.SELECTED_ARTIFACT), [])
        self.assertTrue(c.selection_is_current(self.run))

    def test_export_corruption_invalidates_selection_only(self):
        with patch.object(c, "execute_calibration", side_effect=self.results):
            drive(self.run, self.stages)
        package = self.run.dir / self.run.read(c.SELECTION_ARTIFACT)["benchmark"]
        (package / "public/material.json").write_text("[]", encoding="utf-8")
        self.assertFalse(c.selection_is_current(self.run))
        self.assertTrue(c.calibration_is_current(self.run))

    def test_explicit_settings_reject_missing_models_and_duplicate_aliases(self):
        for changes in ({"judge_model": ""}, {"systems": ["A", "simplemem"]},
                        {"systems": ["A", "unknown"]}, {"max_judge_calls": 0},
                        {"keep_easy_ratio": 1.1}, {"keep_easy_ratio": "0.2"}, {"unexpected": True}):
            path = self.root / "settings.json"
            path.write_text(json.dumps({**SETTINGS, **changes}), encoding="utf-8")
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.load_config(path)
        path.write_text(json.dumps({**SETTINGS, "systems": [" a ", "c"]}), encoding="utf-8")
        self.assertEqual(c.load_config(path)["systems"], ["A", "C"])

    def test_subprocess_is_bounded_uses_explicit_models_and_semantic_grading(self):
        with patch.object(c.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as proc:
            c.execute_calibration(self.run, self.root / "eval with spaces", SETTINGS)
        args, kwargs = proc.call_args
        self.assertIn("--output-dir", args[0])
        self.assertIn("semantic", args[0])
        self.assertEqual(kwargs["env"]["MODEL"], SETTINGS["answering_model"])
        self.assertEqual(kwargs["env"]["JUDGE_MODEL"], SETTINGS["judge_model"])
        self.assertEqual(kwargs["timeout"], SETTINGS["timeout_s"])

    def test_factory_registers_tail_and_quantity_completion_invokes_it_once(self):
        from pipeline import factory
        stages = factory.generation_stages(self.run)
        self.assertEqual([s.name for s in stages[-2:]], ["calibration", "selection"])
        with patch.object(factory, "drive") as drv:
            factory.finish_generation(self.run)
            self.assertEqual(drv.call_args.args[2], "quality")
            factory.finish_generation(self.run, "quality")
            self.assertEqual(drv.call_count, 1)

    def test_finalization_refreshes_old_summary_without_replaying_generator_receipts(self):
        from pipeline import factory
        original = {name: digest for name, digest in self.source_hashes().items() if name != "07_release.json"}
        drive(self.run, self.stages, to_stage="quality")
        self.run.write("07_release.json", {"version": "old_summary", "status": "passed"})
        stages = factory.generation_stages(self.run, finalize_only=True)
        self.assertEqual([s.name for s in stages], ["quality", "calibration", "selection"])
        with patch.object(c, "execute_calibration", side_effect=self.results):
            drive(self.run, stages, from_stage="quality")
        self.assertTrue(c.selection_is_current(self.run))
        self.assertEqual({name: digest for name, digest in self.source_hashes().items()
                          if name != "07_release.json"}, original)

    def test_real_evaluator_and_semantic_grades_feed_generation_export(self):
        # Reuse the evaluator's network-disabled fixture. Only provider answers
        # and memory retrieval are fakes; main(), SemanticJudge, result loading,
        # selection, and standard export all execute their real implementations.
        from tests.multi_system_policy_selftest import OriginalPolicyCLI
        fixture = OriginalPolicyCLI()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.change_settings(systems=["fake1", "fake2"], keep_easy_ratio=1)
        for name, path in (("00_about.json", fixture.about), ("05_corpus.json", fixture.material),
                           ("06_grounded_questions.json", fixture.bench),
                           ("06_semantic_review.json", fixture.review)):
            self.run.write(name, json.loads(path.read_text(encoding="utf-8")))
        self.run.write("04_questions.json", fixture.questions)
        self.run.write("06_grounding_report.json", {"drops": [], "pending": []})
        self.run.write("07_release.json", evaluate_release(self.run.dir))

        def evaluate(run, directory, settings):
            fixture.probe = fixture.module._EvalProbe(output_dir=directory)
            fixture.execute(["--output-dir", str(directory)], filter_enabled=False)
            return directory / "results.json"

        with patch.object(c, "execute_calibration", side_effect=evaluate):
            drive(self.run, self.stages)
        report = self.run.read(c.SELECTION_ARTIFACT)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"]["all_correct"], 1)
        self.assertEqual(report["counts"]["kept"], 1)
        self.assertEqual(len(fixture.calls), 2)
        self.assertTrue(c.selection_is_current(self.run))


if __name__ == "__main__":
    main()
