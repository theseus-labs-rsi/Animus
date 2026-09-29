"""Offline native-result import and existing all-correct-filter integration."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase, main

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harnesses.artifacts import inspect_benchmark
from agent_harnesses.config import BenchmarkRef
from eval.grading import JUDGE_VERSION
from eval.question_filter import filter_questions, load_results
from pipeline.benchmark_export import export_selected, public_view
from pipeline.native_results import import_native_results


class RunFixture:
    def __init__(self, directory):
        self.dir = directory
        directory.mkdir()
        self.run_id, self.scenario, self.manifest = "current-run", "fixture", {"config": {}}

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.dir / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class NativeResultImportTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = RunFixture(self.root / "run")
        self.questions = [{"qid": f"q{i}", "question": f"项目{i}的状态？", "line": "L1_timeline",
                           "capability": "IE", "gt": {"value": f"v{i}"}, "aux": {}} for i in range(3)]
        self.run.write("06_grounded_questions.json", self.questions[:2])
        self.run.write("00_about.json", {"public_protocol": "请根据提供材料回答。"})
        self.run.write("02_world.json", {"entities": {}})
        self.run.write("05_corpus.json", {"corpus": {"sessions": [{"session_id": 1, "date": "2026-01-01",
                       "docs": [{"doc_id": "d1", "content": "项目0状态v0。项目1状态v1。"}]}]}})
        self.benchmark = self.root / "source"
        export_selected(self.run, self.questions, self.benchmark, {})
        self.files = inspect_benchmark(BenchmarkRef(self.benchmark, False, "as_provided")).files
        self.targets = [{"id": f"athlete{i}", "system": "native.codex" if i < 2 else "native.dsh",
                         "model_id": f"model{i}", "model_label": f"model{i}", "endpoint_profile": "TEST",
                         "protocol_style": "responses" if i < 2 else "openai_chat"} for i in range(4)]
        self.settings = {"backend": "native_four", "systems": [t["id"] for t in self.targets], "targets": self.targets}
        self.dirs = {}
        for target in self.targets:
            path = self.root / target["id"]
            path.mkdir()
            self.dirs[target["id"]] = path
            adapter = target["system"].removeprefix("native.")
            plan = {"target_id": target["id"], "system_id": target["system"], "runner": "native_cli", "track": "native",
                    "model": {"label": target["model_label"], "model_id": target["model_id"],
                              "endpoint_profile": target["endpoint_profile"], "protocol_style": target["protocol_style"]},
                    "runtime": {"adapter": adapter, "model": target["model_id"], "protocol_style": target["protocol_style"]},
                    "benchmark": {"schema": "memory-bench-standard-light/v1", "files": self.files, "path": "/different/computer/source"}}
            self.write_json(path / "run_plan.json", plan)
            executions = [{"question_index": i, "question_id": q["qid"], "answer": f"v{i}", "status": "completed",
                           "error_type": None, "returncode": 0} for i, q in enumerate(self.questions)]
            judgements = [{**row, "correct": True, "judgeable": True,
                           "reference_answer": self.questions[i]["gt"]["value"],
                           "truth_source": "references/questions.json.answer",
                           "judge": {"version": {"judge_sha256": "original-source-judge"}, "error": None}}
                          for i, row in enumerate(executions)]
            self.write_lines(path / "results.jsonl", executions)
            self.write_lines(path / "judged.jsonl", judgements)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def write_lines(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def rows(self, name="judged.jsonl"):
        return [json.loads(line) for line in (self.dirs["athlete0"] / name).read_text(encoding="utf-8").splitlines()]

    def import_(self):
        return import_native_results(self.run, self.benchmark, self.dirs, self.root / "normalized", self.settings)

    def filtered(self):
        path = self.import_()
        data = json.loads(path.read_text(encoding="utf-8"))
        results = load_results(aggregate=path, systems=self.settings["systems"])
        return filter_questions(self.run.read("06_grounded_questions.json"), results,
                                expected_context=data["evaluation_context"]), data

    def test_matching_superset_removes_only_current_all_correct_questions(self):
        (kept, report), data = self.filtered()
        self.assertEqual(kept, [])
        self.assertEqual(report["counts"]["removed_easy"], 2)
        self.assertEqual(len(data["results"]["athlete0"]["records"]), 2)
        row = data["results"]["athlete0"]["records"][0]
        self.assertEqual(row["judgement"]["version"], JUDGE_VERSION)
        self.assertFalse(row["judgement"]["regraded"])
        self.assertEqual(row["native_judge"]["version"]["judge_sha256"], "original-source-judge")
        self.assertEqual(self.import_(), self.root / "normalized/results.json")

    def test_standard_public_view_matches_published_protocol_without_extra_rules(self):
        self.run.write("00_about.json", {"answer_protocol": {"version": 7,
            "scoring_policy": "task-with-supporting-reasons/v1", "scoring_scope": "task_with_supporting_reasons",
            "rules": ["第一条。", "第二条。"], "gold_sentinel_map": {"INSUFFICIENT": "查无此记录"}}})
        material, protocol = public_view(self.run)
        expected = ("answer_protocol_version: 7\nscoring_policy: task-with-supporting-reasons/v1\n"
                    "scoring_scope: task_with_supporting_reasons\n\nrules:\n1. 第一条。\n2. 第二条。\n"
                    "\ngold_sentinel_map:\n- INSUFFICIENT: 查无此记录\n")
        self.assertEqual(protocol.encode("utf-8"), expected.encode("utf-8"))
        self.assertEqual(material[0]["doc_id"], "d000001")
        self.assertEqual(self.run.read("05_corpus.json")["corpus"]["sessions"][0]["docs"][0]["doc_id"], "d1")
        self.run.write("00_about.json", {"public_protocol": "verbatim\n  exact text\n"})
        self.assertEqual(public_view(self.run)[1], "verbatim\n  exact text\n")

    def test_wrong_corpus_or_protocol_is_fatal(self):
        for name, value in [("05_corpus.json", {"sessions": [{"session_id": 1, "date": "2026-01-01",
                "docs": [{"doc_id": "d1", "content": "changed"}]}]}),
                ("00_about.json", {"public_protocol": "Changed protocol"})]:
            original = self.run.read(name)
            with self.subTest(name=name):
                self.run.write(name, value)
                with self.assertRaisesRegex(ValueError, "corpus/protocol"):
                    self.import_()
                self.run.write(name, original)

    def test_changed_current_qid_question_and_answer_are_fatal(self):
        for field, value in [("qid", "unknown"), ("question", "changed"), ("gt", {"value": "changed"})]:
            with self.subTest(field=field):
                rows = deepcopy(self.questions[:2]); rows[0][field] = value
                self.run.write("06_grounded_questions.json", rows)
                with self.assertRaises(ValueError):
                    self.import_()

    def test_wrong_native_model_or_source_hash_is_fatal(self):
        path = self.dirs["athlete0"] / "run_plan.json"
        original = json.loads(path.read_text())
        for section, key, value in [("model", "model_id", "wrong"), ("runtime", "model", "wrong"),
                                     ("benchmark", "files", {})]:
            with self.subTest(section=section):
                plan = deepcopy(original); plan[section][key] = value; self.write_json(path, plan)
                with self.assertRaises(ValueError):
                    self.import_()

    def test_source_reference_or_latest_answer_mismatch_is_fatal(self):
        original = self.rows()
        for field, value in [("reference_answer", "wrong"), ("answer", "wrong"), ("status", "failed"),
                             ("error_type", "timeout"), ("question_id", "unknown")]:
            with self.subTest(field=field):
                rows = deepcopy(original); rows[0][field] = value
                self.write_lines(self.dirs["athlete0"] / "judged.jsonl", rows)
                with self.assertRaises(ValueError):
                    self.import_()

    def test_duplicate_scored_rows_or_eligible_qids_are_fatal(self):
        rows = self.rows()
        self.write_lines(self.dirs["athlete0"] / "judged.jsonl", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.import_()
        self.write_lines(self.dirs["athlete0"] / "judged.jsonl", rows)
        self.run.write("06_grounded_questions.json", [self.questions[0], self.questions[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.import_()

    def test_missing_target_keeps_all_questions(self):
        del self.dirs["athlete0"]
        (kept, report), _ = self.filtered()
        self.assertEqual(len(kept), 2)
        self.assertEqual(report["counts"]["kept_incomplete"], 2)

    def test_missing_one_score_keeps_only_that_question(self):
        self.write_lines(self.dirs["athlete0"] / "judged.jsonl", self.rows()[1:])
        (kept, report), _ = self.filtered()
        self.assertEqual([q["qid"] for q in kept], ["q0"])
        self.assertEqual(report["counts"]["kept_incomplete"], 1)

    def test_partial_then_complete_updates_only_owned_normalized_result(self):
        target = self.dirs.pop("athlete0")
        (kept, _), _ = self.filtered()
        self.assertEqual(len(kept), 2)
        original = (target / "results.jsonl").read_bytes()
        self.dirs["athlete0"] = target
        (kept, report), _ = self.filtered()
        self.assertEqual(kept, [])
        self.assertEqual(report["counts"]["removed_easy"], 2)
        self.assertEqual((target / "results.jsonl").read_bytes(), original)
        self.write_json(self.root / "normalized/results.json", {"unrelated": True})
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.import_()

    def test_failed_empty_judge_error_and_contaminated_answers_keep_question(self):
        original_execution, original_judged = self.rows("results.jsonl"), self.rows()
        changes = [{"status": "failed", "error_type": "timeout", "answer": None},
                   {"answer": ""}, {"contaminated": True},
                   {"judge": {"version": {"judge_sha256": "x"}, "error": "judge timed out"}}]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                execution, judged = deepcopy(original_execution), deepcopy(original_judged)
                judged[0].update(change)
                execution[0].update({k: v for k, v in change.items() if k != "judge"})
                self.write_lines(self.dirs["athlete0"] / "results.jsonl", execution)
                self.write_lines(self.dirs["athlete0"] / "judged.jsonl", judged)
                path = import_native_results(self.run, self.benchmark, self.dirs, self.root / f"failed{index}", self.settings)
                data = json.loads(path.read_text(encoding="utf-8"))
                kept, report = filter_questions(self.questions[:2], load_results(aggregate=path), expected_context=data["evaluation_context"])
                self.assertEqual([q["qid"] for q in kept], ["q0"])

    def test_append_retry_uses_latest_answer_and_rejects_stale_grade(self):
        rows = self.rows("results.jsonl")
        rows.append({**rows[0], "answer": "new answer"})
        self.write_lines(self.dirs["athlete0"] / "results.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.import_()
        judged = self.rows(); judged[0]["answer"] = "new answer"; judged[0]["correct"] = False
        self.write_lines(self.dirs["athlete0"] / "judged.jsonl", judged)
        (kept, report), _ = self.filtered()
        self.assertEqual([q["qid"] for q in kept], ["q0"])
        self.assertEqual(report["counts"]["kept_not_all_correct"], 1)

    def test_unknown_target_and_scores_without_plan_are_fatal(self):
        self.dirs["unrequested"] = self.root / "extra"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.import_()
        del self.dirs["unrequested"]
        (self.dirs["athlete0"] / "run_plan.json").unlink()
        with self.assertRaisesRegex(ValueError, "lack"):
            self.import_()

    def test_process_trace_never_receives_fabricated_semantic_grade(self):
        from pipeline.native_results import _normalized
        from eval.provenance import make_evaluation_context
        q = {**self.questions[0], "capability": "L3_process_trace"}
        row = _normalized(q, self.rows("results.jsonl")[0], self.rows()[0],
                          make_evaluation_context([], "protocol"), self.targets[0], {})
        self.assertIsNone(row["correct"])
        self.assertEqual(row["judgement"]["verdict"], "unjudgeable")
        self.assertEqual(row["execution_status"], "incomplete")


if __name__ == "__main__":
    main()
