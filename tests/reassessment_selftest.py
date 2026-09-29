"""Offline semantic dispute closure through the real runner, cache and filter."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_contract_selftest import load_runner, judge
from semantic_judge_selftest import CORPUS, Q, review_output, grade_output
from eval import qa_cache
from eval.grading import is_scored
from eval.provenance import make_evaluation_context, record_provenance
from eval.question_filter import filter_questions
from eval.reassessment import ReassessmentTracker, closure_path
from eval.semantic_judge import SemanticJudge
from pipeline.semantic_review import review_questions

PROTOCOL = "回答日期"


def reference_report(questions):
    return review_questions(questions, CORPUS, PROTOCOL, reviewer_model="offline-reviewer",
        chat_json=lambda step, messages, **kw: review_output("blind_read" if step.endswith("blind_read") else "adjudicate"))


class ReassessmentTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        self.questions = [{**Q, "qid": "same-text-a"}, {**Q, "qid": "same-text-b"}]
        self.context = make_evaluation_context([(0, "", CORPUS[0]["content"])], PROTOCOL)
        self.report = reference_report(self.questions)
        self.call = Mock(return_value=grade_output())
        self.grader = SemanticJudge(self.report, self.questions, CORPUS, PROTOCOL,
            model="offline-grader", chat_json=self.call, max_calls=50)
        self.review_hash = self.grader.cache_context["reference_review_hash"]

    def row(self, q=None, *, dispute=False):
        q = q or self.questions[0]
        self.call.return_value = grade_output(**({"reference_status": "contradicted",
            "answer_verdict": "uncertain", "requires_item_reassessment": True,
            "reassessment_reason": "离线预设：发现原参考理由需要同题重新审查。"} if dispute else {}))
        pred = "报告发布日期是2024年3月20日。"
        grade = self.grader(q, pred)
        return {**q, "pred": pred, "mode": "semantic", "judgeable": True,
                "execution_status": "ok", "result_scope": "research_only",
                "evaluation_provenance": record_provenance(q, self.context),
                "judgement": grade, "correct": grade["correct"]}

    def test_late_trigger_closes_earlier_scores_preserves_evidence_and_recomputes_aggregate(self):
        early, other = self.row(), self.row(self.questions[1])
        early["_resumed"] = True
        trigger = self.row(dispute=True)
        after = self.row()
        original = deepcopy(early)
        self.assertTrue(is_scored(early))
        results = {"A": {"records": [early, other]}, "B": {"records": [trigger]},
                   "C": {"records": [after]}}
        tracker = ReassessmentTracker(self.context, self.review_hash)
        closure = self.runner.close_semantic_reassessment(results, tracker)
        self.assertEqual(closure["n_disputed_items"], 1)
        self.assertEqual(closure["quarantined_by_system"], {"A": 1, "B": 1, "C": 1})
        self.assertEqual(results["A"]["agg"]["overall"], {"n": 1, "correct": 1, "acc": 1.0})
        self.assertEqual(results["A"]["agg"]["n_unscored"], 1)
        self.assertIsNone(results["B"]["agg"]["overall"]["acc"])
        self.assertEqual(results["B"]["agg"]["n_uncertain"], 1)
        self.assertEqual(early["original_judgement"], original["judgement"])
        self.assertEqual(early["judgement"]["evidence"], original["judgement"]["evidence"])
        self.assertEqual(early["judgement"]["resolved_evidence"], original["judgement"]["resolved_evidence"])
        for field in ("pred", "evaluation_provenance", "result_scope", "execution_status"):
            self.assertEqual(early[field], original[field])
        self.assertFalse(is_scored(early))
        self.assertTrue(is_scored(other))
        self.assertEqual(early["reassessment"]["trigger_systems"], ["B"])
        first = deepcopy(early)
        self.runner.close_semantic_reassessment(results, tracker)
        self.assertEqual(early, first)
        self.assertEqual(len(tracker.snapshot()["items"][0]["triggers"]), 1)
        kept, filtered = filter_questions(self.questions[:1], {"A": [early], "B": [trigger]},
                                          expected_context=self.context)
        self.assertEqual(kept, self.questions[:1])
        self.assertEqual(filtered["counts"]["all_correct"], 0)

    def test_other_qid_context_reference_presence_and_review_are_isolated(self):
        base, trigger = self.row(), self.row(dispute=True)
        other_qid = self.row(self.questions[1])
        other_context = deepcopy(base)
        context = make_evaluation_context([(0, "", "different material")], PROTOCOL)
        other_context["evaluation_provenance"] = record_provenance(other_context, context)
        other_reference = deepcopy(base)
        other_reference["gt"] = "new reference"
        other_reference["evaluation_provenance"] = record_provenance(other_reference, self.context)
        other_review = deepcopy(base)
        other_review["judgement"]["review_hash"] = "f" * 64
        legacy = deepcopy(base)
        legacy.pop("mode")
        legacy["judgement"] = judge.judgement("correct", "offline", "legacy fixture")
        legacy["correct"] = True
        rows = [base, other_qid, other_context, other_reference, other_review, legacy]
        unchanged = deepcopy(rows[1:])
        tracker = ReassessmentTracker(self.context, self.review_hash)
        tracker.close({"A": rows, "B": [trigger]})
        self.assertTrue(base["quarantined"])
        self.assertEqual(rows[1:], unchanged)

    def test_absent_reference_and_explicit_null_are_different_item_keys(self):
        questions = [{"qid": "no-ref", "question": Q["question"]},
                     {"qid": "no-ref", "question": Q["question"], "reference_proposal": None}]
        rows = []
        for q in questions:
            grade = deepcopy(self.row()["judgement"])
            from eval.provenance import question_hash, reference_hash
            grade.update(question_hash=question_hash(q), reference_hash=reference_hash(q),
                         source_qid=q["qid"], reference_provided="reference_proposal" in q)
            rows.append({**q, "mode": "semantic", "evaluation_provenance": record_provenance(q, self.context),
                         "judgement": grade, "correct": True, "pred": "same answer"})
        rows[0]["judgement"]["requires_item_reassessment"] = True
        tracker = ReassessmentTracker(self.context, self.review_hash)
        tracker.close({"A": [rows[0]], "B": [rows[1]]})
        self.assertTrue(rows[0]["quarantined"])
        self.assertNotIn("quarantined", rows[1])

    def test_invalid_current_identity_cannot_trigger_other_items_by_question_text(self):
        row = self.row(dispute=True)
        row["judgement"]["source_qid"] = "forged"
        tracker = ReassessmentTracker(self.context, self.review_hash)
        with self.assertRaisesRegex(ValueError, "source qid"):
            tracker.observe([row], system="A")
        self.assertEqual(tracker.snapshot()["n_disputed_items"], 0)

    def test_trigger_is_durable_before_final_close_and_concurrent_observers_do_not_lose_sources(self):
        trigger = self.row(dispute=True)
        with tempfile.TemporaryDirectory() as directory:
            path = closure_path(directory, self.context, self.review_hash)
            tracker = ReassessmentTracker(self.context, self.review_hash, path=path)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda n: tracker.observe([deepcopy(trigger)], system=f"system-{n}"), range(8)))
            restored = ReassessmentTracker(self.context, self.review_hash, path=path)
            self.assertTrue(restored.is_disputed(self.row()))
            self.assertEqual(len(restored.snapshot()["items"][0]["triggers"]), 8)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            with self.assertRaisesRegex(ValueError, "different input/review scope"):
                ReassessmentTracker(self.context, "a" * 64, path=path)

    def test_atomic_storage_failure_is_visible_and_does_not_certify_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = ReassessmentTracker(self.context, self.review_hash,
                path=closure_path(directory, self.context, self.review_hash))
            with patch("eval.reassessment.os.replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
                tracker.observe([self.row(dispute=True)], system="A")
            self.assertTrue(tracker.is_disputed(self.row()))
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    def test_real_cache_cannot_revive_current_dispute_but_new_review_reuses_prediction_and_regrades(self):
        q = self.questions[0]
        system = Mock()
        system.retrieve.return_value = CORPUS[0]["content"]
        system.get_diagnostics.return_value = {}
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)), \
                patch.object(self.runner, "unified_answer", return_value="报告发布日期是2024年3月20日。") as answer:
            path = closure_path(directory, self.context, self.review_hash)
            tracker = ReassessmentTracker(self.context, self.review_hash, path=path)
            kwargs = {"verbose": False, "bench_id": "cache-regression", "protocol": PROTOCOL,
                      "cache_context": {"evaluation_context": self.context}, "judge_fn": self.grader,
                      "reassessment": tracker}
            self.call.return_value = grade_output()
            early = self.runner.run_system("A", [q], system, **kwargs)
            self.assertTrue(is_scored(early[0]))
            self.call.return_value = grade_output(reference_status="contradicted", answer_verdict="uncertain",
                requires_item_reassessment=True, reassessment_reason="公开依据引出参考争议。")
            late = self.runner.run_system("B", [q], system, **kwargs)
            results = {"A": {"records": early}, "B": {"records": late}}
            self.runner.close_semantic_reassessment(results, tracker)
            self.assertFalse(is_scored(early[0]))
            count = self.call.call_count
            restored = ReassessmentTracker(self.context, self.review_hash, path=path)
            resumed = self.runner.run_system("A", [q], system, **{**kwargs, "reassessment": restored})
            self.assertEqual(self.call.call_count, count)
            self.assertEqual(answer.call_count, 2)
            self.assertTrue(resumed[0]["_prediction_resumed"])
            self.assertTrue(resumed[0]["_resumed"])
            self.assertTrue(resumed[0]["original_judgement"]["correct"])
            self.assertFalse(is_scored(resumed[0]))
            fresh_q = {**q, "gt": "2024年3月20日"}
            fresh_report = reference_report([fresh_q])
            fresh_call = Mock(return_value=grade_output())
            fresh_grader = SemanticJudge(fresh_report, [fresh_q], CORPUS, PROTOCOL,
                model="offline-grader", chat_json=fresh_call)
            fresh_hash = fresh_grader.cache_context["reference_review_hash"]
            self.assertNotEqual(fresh_hash, self.review_hash)
            fresh_tracker = ReassessmentTracker(self.context, fresh_hash,
                path=closure_path(directory, self.context, fresh_hash))
            new = self.runner.run_system("A", [fresh_q], system,
                **{**kwargs, "judge_fn": fresh_grader, "reassessment": fresh_tracker})
            self.assertTrue(new[0]["_prediction_resumed"])
            self.assertNotIn("_resumed", new[0])
            self.assertTrue(is_scored(new[0]))
            fresh_call.assert_called_once()
            self.assertEqual(answer.call_count, 2)
            self.assertEqual(fresh_tracker.snapshot()["n_disputed_items"], 0)
            self.assertTrue(path.exists())

    def test_pre_closure_policy_grades_are_regraded_without_reanswering(self):
        q = self.questions[0]
        system = Mock()
        system.retrieve.return_value = CORPUS[0]["content"]
        system.get_diagnostics.return_value = {}
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)), \
                patch.object(self.runner, "unified_answer", return_value="2024年3月20日") as answer:
            kwargs = {"verbose": False, "bench_id": "migration", "protocol": PROTOCOL,
                      "cache_context": {"evaluation_context": self.context}, "judge_fn": self.grader}
            self.runner.run_system("A", [q], system, **kwargs)
            self.assertEqual(self.call.call_count, 1)
            tracker = ReassessmentTracker(self.context, self.review_hash)
            migrated = self.runner.run_system("A", [q], system, **kwargs, reassessment=tracker)
            self.assertEqual(self.call.call_count, 2)
            answer.assert_called_once()
            self.assertTrue(migrated[0]["_prediction_resumed"])
            self.assertNotIn("_resumed", migrated[0])
            self.assertTrue(is_scored(migrated[0]))

    def test_main_closes_disputes_before_table_export_report_and_filter(self):
        early, trigger = self.row(), self.row(dispute=True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bench, corpus, reference = [folder / name for name in ("bench.json", "corpus.json", "review.json")]
            bench.write_text(json.dumps(self.questions), encoding="utf-8")
            corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
            reference.write_text(json.dumps(self.report), encoding="utf-8")
            probe = Mock()
            probe.run_dir = folder
            sequence = []
            def run_system(name, *args, **kwargs):
                sequence.append("run-" + name)
                return [deepcopy(early if name == "A" else trigger)]
            def check_results(results, systems, *args, **kwargs):
                self.assertEqual(sequence[:2], ["run-A", "run-B"])
                for result in results.values():
                    self.assertEqual(result["agg"]["overall"]["n"], 0)
                    self.assertIsNone(result["agg"]["overall"]["acc"])
                    self.assertFalse(is_scored(result["records"][0]))
            def filter_export(path, records, output, **kwargs):
                self.assertTrue(all(not is_scored(rows[0]) for rows in records.values()))
                exported = json.loads((folder / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(exported["reassessment_closure"]["n_disputed_items"], 1)
                self.assertEqual(exported["n_scored_by_system"], {"A": 0, "B": 0})
                return {"counts": {"all_correct": 0, "removed_easy": 0, "kept": 2}}
            argv = ["multi_system", "--bench", str(bench), "--corpus", str(corpus),
                    "--systems", "A,B", "--judge-mode", "semantic", "--semantic-review", str(reference),
                    "--judge-model", "offline", "--allow-unverified", "--filter-easy"]
            with patch.object(sys, "argv", argv), patch.object(qa_cache, "CACHE_DIR", folder / "cache"), \
                    patch.object(self.runner, "_EvalProbe", return_value=probe), \
                    patch.object(self.runner, "load_protocol", return_value=PROTOCOL), \
                    patch.object(self.runner, "load_corpus", return_value=[(0, "", CORPUS[0]["content"])]), \
                    patch.object(self.runner, "run_system", side_effect=run_system), \
                    patch.object(self.runner, "prepare_systems", return_value=({"A": None, "B": None},
                        {"A": {"status": "ok"}, "B": {"status": "ok"}})), \
                    patch.object(self.runner, "trace_scope", side_effect=lambda *a, **k: nullcontext()), \
                    patch.object(self.runner, "print_table", side_effect=check_results) as table, \
                    patch.object(self.runner, "write_report", side_effect=check_results) as report, \
                    patch.object(self.runner, "export_filtered_benchmark", side_effect=filter_export) as easy, \
                    patch("eval.semantic_judge.SemanticJudge", return_value=self.grader), \
                    patch.dict(sys.modules, {"pipeline.quality": types.SimpleNamespace(
                        require_release=lambda *a, **k: {"override": True}),
                        "eval.memory_systems": types.SimpleNamespace(make_system=Mock())}):
                self.runner.main()
            table.assert_called_once()
            report.assert_called_once()
            easy.assert_called_once()
            self.assertEqual(probe.sys_done.call_count, 2)
            self.assertTrue(all(call.args[1]["overall"]["n"] == 0 for call in probe.sys_done.call_args_list))


if __name__ == "__main__":
    unittest.main(verbosity=2)
