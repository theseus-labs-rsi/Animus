"""Release integration: actual factory tail, four harnesses, judge and export.

Native subprocesses and judge HTTP are the only simulated service boundaries.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests import generation_selection_selftest as generation_fixture
from agent_harnesses.artifacts import load_benchmark_questions
from pipeline import calibration as c, factory
from pipeline.run import drive


class ReleasePipelineTest(TestCase):
    def setUp(self):
        self.fixture = generation_fixture.GenerationSelectionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.settings = c.load_config(ROOT / "examples/release_four.json")
        self.run.manifest["config"]["calibration"] = deepcopy(self.settings)
        self.run._save_manifest()
        self.calls = []
        self.fail_judging = False
        self.patches = [
            patch("agent_harnesses.tracks.native.preflight_system", side_effect=self.preflight),
            patch("agent_harnesses.execution.subprocess.run", side_effect=self.process),
            patch("config.chat", side_effect=self.judge_http),
        ]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def preflight(self, system, model):
        return {"ok": True, "errors": [], "binary": "offline-cli", "binary_version": "test",
                "adapter": system.implementation["adapter"], "model": model["model_id"],
                "protocol_style": model["protocol_style"], "runtime_options": {}}

    def process(self, command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="offline-test\n")
        plan = json.loads(Path(command[command.index("--plan") + 1]).read_text(encoding="utf-8"))
        path = Path(plan["output_dir"]) / "results.jsonl"
        existing = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        done = {r["question_index"] for r in existing if r["status"] == "completed"}
        with path.open("a", encoding="utf-8") as stream:
            for i, q in enumerate(load_benchmark_questions(Path(plan["benchmark"]["path"]))):
                if i in done:
                    continue
                self.calls.append((plan["target_id"], i))
                answer = "incorrect text" if i == 8 and plan["target_id"] == self.settings["systems"][-1] else f"v{i}"
                stream.write(json.dumps({"question_index": i, "question_id": q["qid"], "answer": answer,
                                        "status": "completed", "error_type": None, "returncode": 0}) + "\n")
        return SimpleNamespace(returncode=0)

    def judge_http(self, *args, **kwargs):
        if self.fail_judging:
            raise RuntimeError("offline provider unavailable")
        return '{"correct": false, "reason": "different value"}'

    def tail(self):
        return factory.generation_stages(self.run, finalize_only=True)

    def test_full_tail_real_judge_selects_exact_intersection_and_preserves_source(self):
        before = self.fixture.source_hashes()
        drive(self.run, self.tail())
        report = self.run.read(c.SELECTION_ARTIFACT)
        self.assertEqual(report["counts"]["removed_easy"], 9)
        self.assertEqual(report["counts"]["kept"], 1)
        self.assertTrue(report["release_ready"])
        self.assertEqual(self.run.read(c.SELECTED_ARTIFACT)[0]["qid"], "q8")
        self.assertEqual(len(self.calls), 40)
        self.assertEqual({k: v for k, v in before.items() if k != "07_release.json"},
                         {k: v for k, v in self.fixture.source_hashes().items() if k != "07_release.json"})
        package = self.run.dir / report["benchmark"]
        self.assertEqual(len(json.loads((package / "public/questions.json").read_text())), 1)
        drive(self.run, self.tail())
        self.assertEqual(len(self.calls), 40)
        self.assertEqual(self.run.read(c.SELECTION_ARTIFACT)["benchmark"], report["benchmark"])

    def test_partial_resume_retries_judging_without_repeating_athletes(self):
        self.fail_judging = True
        # A provider failure is surfaced without waiting through real backoff.
        with patch("time.sleep"):
            drive(self.run, self.tail())
        report = self.run.read(c.SELECTION_ARTIFACT)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["counts"]["kept_incomplete"], 1)
        self.assertIsNone(report["benchmark"])
        self.assertFalse(report["release_ready"])
        self.assertFalse(c.calibration_can_skip(self.run))
        self.fail_judging = False
        drive(self.run, self.tail(), from_stage="calibration")
        self.assertEqual(len(self.calls), 40)
        self.assertTrue(self.run.read(c.SELECTION_ARTIFACT)["release_ready"])

    def test_interrupted_input_export_resumes_without_partial_input(self):
        from pipeline import benchmark_export
        real = benchmark_export.export_selected
        def interrupted(run, questions, destination, selection):
            destination.mkdir(parents=True)
            raise OSError("offline disk interruption")
        with patch.object(benchmark_export, "export_selected", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "disk interruption"):
                drive(self.run, self.tail())
        self.assertEqual(self.calls, [])
        self.assertFalse(list((self.run.dir / "calibration").glob("native-*/evaluation_input")))
        with patch.object(benchmark_export, "export_selected", side_effect=real):
            drive(self.run, self.tail(), from_stage="calibration")
        self.assertTrue(self.run.read(c.SELECTION_ARTIFACT)["release_ready"])

    def test_generation_prefix_stays_identical(self):
        stages = factory.generation_stages(self.run)
        self.assertEqual(stages[:-2], factory.STAGES)
        self.assertEqual([s.name for s in stages[-2:]], ["calibration", "selection"])
        for value in ({"keep_easy_ratio": .2}, {"keep_easy_ratio": "0"}, {"targets": self.settings["targets"][:3]}):
            path = self.fixture.root / "invalid.json"
            path.write_text(json.dumps({**json.loads((ROOT / "examples/release_four.json").read_text()), **value}))
            with self.assertRaises(ValueError):
                c.load_config(path)

    def test_selection_only_cannot_reuse_stale_calibration(self):
        drive(self.run, self.tail())
        self.run.manifest["config"]["calibration"]["judge_model"] = "changed-judge"
        self.run._save_manifest()
        self.assertFalse(c.selection_is_current(self.run))
        with self.assertRaisesRegex(ValueError, "calibration before selecting"):
            drive(self.run, self.tail(), only="selection")
        self.assertEqual(len(self.calls), 40)

    def test_expanded_corpus_invalidates_scores_from_old_material(self):
        drive(self.run, self.tail())
        corpus = self.run.read("05_corpus.json")
        body = corpus.get("corpus", corpus)
        body["sessions"][0]["docs"].append({"doc_id": "extra-haystack", "content": "新增周边材料", "is_filler": True})
        self.run.write("05_corpus.json", corpus)
        self.assertFalse(c.calibration_is_current(self.run))
        self.assertFalse(c.selection_is_current(self.run))


if __name__ == "__main__":
    main()
