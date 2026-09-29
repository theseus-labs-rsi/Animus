"""Offline regressions for result identity and separate answer/grade caches."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import qa_cache
from eval.grading import JUDGE_VERSION
from eval.provenance import (load_public_protocol, load_visible_corpus, make_evaluation_context,
                             preserve_result_contexts, provenance_issue, record_provenance,
                             valid_context)
from eval.question_filter import filter_questions, load_results


def question():
    return {"question": "哪个市场？", "line": "L1_timeline", "capability": "IE", "gt": "亚太",
            "aux": {"at_week": 0}, "question_version": 1, "reference_version": 1}


def scored(q, context):
    return {**deepcopy(q), "pred": "亚太", "correct": True,
            "evaluation_provenance": record_provenance(q, context),
            "judgement": {"version": JUDGE_VERSION, "verdict": "correct", "correct": True,
                          "path": "offline_fixture", "reason": "explicit fixture"}}


class ProvenanceTest(unittest.TestCase):
    def test_explicit_protocol_round_trips_without_legacy_domain_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "about.json"
            for protocol in ("依据正文判断，说明不确定性。\n截止日含当日。", ""):
                path.write_text(json.dumps({"public_protocol": protocol, "private": "not evidence"}), encoding="utf-8")
                self.assertEqual(load_public_protocol(path), protocol)
            for bad in ({"public_protocol": None},
                        {"public_protocol": "exact", "answer_protocol": {"rules": []}}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_public_protocol(path)

    def setUp(self):
        self.q = question()
        self.context = make_evaluation_context([(0, "2026-09-16", "客户市场为亚太")], "公开答题约定")
        self.row = scored(self.q, self.context)

    def test_exact_input_is_eligible_for_filtering(self):
        self.assertTrue(valid_context(self.context))
        self.assertIsNone(provenance_issue(self.row, self.q, self.context))
        kept, report = filter_questions([self.q], {s: [self.row] for s in ("a", "b")},
                                         expected_context=self.context)
        self.assertEqual(kept, [])
        self.assertEqual(report["counts"]["all_correct"], 1)

    def test_changed_visible_inputs_are_incomplete_not_easy(self):
        variants = [make_evaluation_context([(0, "2026-09-16", "客户市场为欧洲")], "公开答题约定"),
                    make_evaluation_context([(0, "2026-09-16", "客户市场为亚太")], "另一个约定"), None]
        for expected in variants:
            kept, report = filter_questions([self.q], {s: [self.row] for s in ("a", "b")},
                                             expected_context=expected)
            self.assertEqual(kept, [self.q])
            self.assertEqual(report["counts"]["kept_incomplete"], 1)

    def test_question_and_reference_versions_bound(self):
        for field, value in [("question", "什么市场？"), ("gt", "欧洲"), ("aux", {"at_week": 2}),
                             ("question_version", 2), ("reference_version", 2),
                             ("rubric", "必须解释"), ("gold", "欧洲"),
                             ("reference_proposal", {"answer": "欧洲", "rationale": "new basis"})]:
            changed = {**self.q, field: value}
            self.assertIsNotNone(provenance_issue(self.row, changed, self.context), field)

    def test_legacy_loading_preserves_but_does_not_create_identity(self):
        legacy = {k: v for k, v in self.row.items() if k != "evaluation_provenance"}
        rows = preserve_result_contexts([legacy], {"corpus": "now-changed.json", "model": "old"})
        self.assertNotIn("evaluation_provenance", rows[0])
        self.assertEqual(rows[0]["_result_legacy_context"][0]["corpus"], "now-changed.json")
        self.assertEqual(provenance_issue(rows[0], self.q, self.context),
                         "missing_or_invalid_evaluation_provenance")

    def test_envelope_conflict_is_not_erased_by_flattening(self):
        different = make_evaluation_context([], "")
        rows = preserve_result_contexts([self.row], {"evaluation_context": self.context},
                                        {"evaluation_context": different})
        self.assertEqual(len(rows[0]["_result_contexts"]), 2)
        self.assertEqual(provenance_issue(rows[0], self.q, self.context), "result_envelope_context_mismatch")

    def test_research_scope_survives_even_empty_envelope_rows(self):
        for rows in ([self.row], []):
            loaded = preserve_result_contexts(rows, {"result_scope": "research_only"})
            self.assertIn("research_only", loaded.result_scopes)
            _, report = filter_questions([self.q], {"a": loaded, "b": [self.row]},
                                         expected_context=self.context)
            self.assertEqual(report["result_scope"], "research_only")
            if not rows:
                self.assertEqual(report["counts"]["kept_incomplete"], 1)

    def test_corrupted_context_digest_is_not_accepted(self):
        changed = deepcopy(self.row)
        changed["evaluation_provenance"]["corpus_hash"] = "f" * 64
        self.assertEqual(provenance_issue(changed, self.q, self.context),
                         "missing_or_invalid_evaluation_provenance")

    def test_malformed_envelope_and_execution_failure_remain_incomplete(self):
        for change in ({"_result_contexts": None}, {"execution_status": "error"}):
            row = {**self.row, **change}
            kept, report = filter_questions([self.q], {"a": [self.row], "b": [row]},
                                             expected_context=self.context)
            self.assertEqual(kept, [self.q])
            self.assertEqual(report["counts"]["kept_incomplete"], 1)

    def test_single_system_json_and_aggregate_retain_context(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name in ("a", "b"):
                p = Path(directory) / f"{name}.json"
                p.write_text(json.dumps({"evaluation_context": self.context, "records": [self.row]}), encoding="utf-8")
                paths[name] = p
            loaded = load_results(system_files=paths)
            for rows in loaded.values():
                self.assertEqual(rows[0]["_result_contexts"], [self.context])
                self.assertIsNone(provenance_issue(rows[0], self.q, self.context))

    def test_actual_harness_view_ignores_hidden_labels_not_body_or_date(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "corpus.json"
            data = {"corpus": {"sessions": [
                {"session_id": 2, "date": "later", "docs": [{"content": "second", "title": "hidden"}]},
                {"session_id": 0, "date": "earlier", "docs": [
                    {"content": "first", "is_filler": True}, {"content": "next", "fact_refs": ["truth"]}]}]}}
            p.write_text(json.dumps(data), encoding="utf-8")
            docs = load_visible_corpus(p)
            self.assertEqual(docs, [(0, "earlier", "first"), (0, "earlier", "next"), (2, "later", "second")])
            data["corpus"]["sessions"][0]["docs"][0]["title"] = "another hidden title"
            p.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_visible_corpus(p), docs)
            data["corpus"]["sessions"][0]["date"] = "changed"
            p.write_text(json.dumps(data), encoding="utf-8")
            self.assertNotEqual(load_visible_corpus(p), docs)

    def test_protocol_identity_is_exact_injected_rules_not_private_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "about.json"
            p.write_text(json.dumps({"answer_protocol": {"rules": ["只答结论"]}, "private": "old"}), encoding="utf-8")
            public = load_public_protocol(p)
            p.write_text(json.dumps({"answer_protocol": {"rules": ["只答结论"]}, "private": "new"}), encoding="utf-8")
            self.assertEqual(load_public_protocol(p), public)
            p.write_text(json.dumps({"answer_protocol": {"rules": ["解释依据"]}}), encoding="utf-8")
            self.assertNotEqual(load_public_protocol(p), public)
            p.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_public_protocol(p)


class CacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.q = question()
        self.evaluation = make_evaluation_context([(0, "date", "material")], "rules")
        self.context = {"evaluation_context": self.evaluation, "solver": {"model": "solver1", "prompt": "s1"},
                        "grading": {"model": "judge1", "prompt_hash": "p1", "policy": "semantic1"}}

    def test_judge_changes_reuse_prediction_but_never_grade(self):
        for field in ("model", "prompt_hash", "policy"):
            other = deepcopy(self.context)
            other["grading"][field] = "changed"
            self.assertEqual(qa_cache.bench_id([self.q], self.context), qa_cache.bench_id([self.q], other))
            self.assertEqual(qa_cache.qhash(self.q, self.context, grading=False),
                             qa_cache.qhash(self.q, other, grading=False))
            self.assertNotEqual(qa_cache.qhash(self.q, self.context, prediction="亚太"),
                                qa_cache.qhash(self.q, other, prediction="亚太"))

    def test_reference_revisions_reuse_prediction_only(self):
        for field, value in (("gt", "欧洲"), ("gold", "欧洲"),
                             ("reference_proposal", {"answer": "欧洲"})):
            other = {**self.q, field: value, "reference_version": 2}
            self.assertEqual(qa_cache.bench_id([self.q], self.context), qa_cache.bench_id([other], self.context))
            self.assertEqual(qa_cache.qhash(self.q, self.context, grading=False),
                             qa_cache.qhash(other, self.context, grading=False))
            self.assertNotEqual(qa_cache.qhash(self.q, self.context, prediction="亚太"),
                                qa_cache.qhash(other, self.context, prediction="亚太"))

    def test_solver_or_material_changes_cannot_reuse_answers(self):
        for key, value in [("solver", {"model": "other"}),
                           ("evaluation_context", make_evaluation_context([], "different"))]:
            other = {**self.context, key: value}
            self.assertNotEqual(qa_cache.qhash(self.q, self.context, grading=False),
                                qa_cache.qhash(self.q, other, grading=False))

    def test_grade_key_binds_full_answer_not_old_scores(self):
        first = qa_cache.qhash(self.q, self.context, prediction="亚太")
        self.assertNotEqual(first, qa_cache.qhash(self.q, self.context, prediction="不是亚太"))
        metadata = {**self.q, "pred": "old", "correct": False, "judgement": {"verdict": "incorrect"}, "_qh": "old"}
        self.assertEqual(first, qa_cache.qhash(metadata, self.context, prediction="亚太"))

    def test_duplicate_text_keeps_individual_reference_review_identity(self):
        for first, second in (({**self.q, "qid": "a"}, {**self.q, "qid": "b"}),
                              ({"question": "自然题"}, {"question": "自然题", "reference_proposal": None})):
            self.assertEqual(qa_cache.qhash(first, self.context, grading=False),
                             qa_cache.qhash(second, self.context, grading=False))
            self.assertNotEqual(qa_cache.qhash(first, self.context, prediction="回答"),
                                qa_cache.qhash(second, self.context, prediction="回答"))

    def test_cache_roundtrip_and_judge_only_regrade(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)):
            bid = qa_cache.bench_id([self.q], self.context)
            row = scored(self.q, self.evaluation)
            row["_qh"] = qa_cache.qhash(self.q, self.context, prediction=row["pred"])
            pk = qa_cache.qhash(self.q, self.context, grading=False)
            qa_cache.append_prediction(bid, "system", {**row, "_qh": pk})
            qa_cache.append(bid, "system", row)
            other = {**self.context, "grading": {"model": "judge2"}}
            new_bid = qa_cache.bench_id([self.q], other)
            self.assertIn(qa_cache.qhash(self.q, other, grading=False), qa_cache.load_predictions(new_bid, "system"))
            self.assertNotIn(qa_cache.qhash(self.q, other, prediction=row["pred"]), qa_cache.load(new_bid, "system"))
            self.assertIn(row["_qh"], qa_cache.load(bid, "system"))

    def test_error_and_unknown_verdicts_never_become_cached_grades(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)):
            for verdict in ("unknown", "uncertain", "error", "unjudgeable"):
                row = scored(self.q, self.evaluation)
                row.update(_qh=verdict, correct=None)
                row["judgement"].update(verdict=verdict, correct=None)
                qa_cache.append("bench", "sys", row)
            self.assertEqual(qa_cache.load("bench", "sys"), {})

    def test_corrupted_saved_answer_cannot_reuse_old_grade(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)):
            row = scored(self.q, self.evaluation)
            row["_qh"] = "fixture"
            qa_cache.append("bench", "sys", row)
            path = qa_cache._path("bench", "sys")
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["pred"] = "a changed answer"
            path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(qa_cache.load("bench", "sys"), {})

    def test_execution_fault_is_not_a_prediction(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)):
            row = {"_qh": "key", "pred": "信息不足", "execution_status": "error"}
            qa_cache.append_prediction("bench", "sys", row)
            self.assertEqual(qa_cache.load_predictions("bench", "sys"), {})


if __name__ == "__main__":
    unittest.main()
