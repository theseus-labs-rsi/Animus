"""A released L7 comparison subset cannot satisfy the original native S1 quota."""
from __future__ import annotations

import os
import socket
import sys
import types

# Install every offline boundary before importing any pipeline/provider module.
os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-dummy",
                  OPENAI_BASE_URL="http://127.0.0.1:9/v1", MODEL="offline-dummy")


def _forbidden(*args, **kwargs):
    raise AssertionError("Network/provider forbidden in native release selftest")


socket.socket.connect = _forbidden
socket.socket.connect_ex = _forbidden
socket.create_connection = _forbidden
socket.getaddrinfo = _forbidden
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules["dotenv"] = dotenv
import openai
openai.OpenAI = _forbidden
openai.AsyncOpenAI = _forbidden

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import factory, production
from pipeline.capability_contract import requirements
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from pipeline.quality import evaluate_release, quality_snapshot
from pipeline.supply import author_brief

L7 = "L7_consolidation"


class FileRun:
    def __init__(self, root, target):
        self.dir = root
        self.run_id = root.name
        self.manifest = {"config": {"delivery_target": target, "question_budget": 6,
            "production_control": {"version": "supply-driven/v5"},
            "acceptance_scope": {"version": 1, "count_stage": "generation_quality"}}}

    def has(self, name):
        return (self.dir / name).is_file()

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.dir / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def question(qid, *, native=False, planted=False, entity=None):
    aux = {"sub": "S1_trend" if native or planted else "S2_compare"}
    if native:
        aux["supply_origin"] = "native_numeric_trajectory/v1"
    return {"qid": qid, "line": L7, "capability": L7, "entity": entity or qid,
            "field": "volume", "evidence_sessions": [0, 1, 2, 3], "aux": aux,
            "question": "Synthetic question " + qid, "gt": "up"}


class NativeReleaseTests(unittest.TestCase):
    def make_run(self, rows=None, *, released=("compare-a", "compare-b"), rejected=("native",), pending=()):
        folder = tempfile.TemporaryDirectory(prefix="native-release-")
        self.addCleanup(folder.cleanup)
        target = {"version": 1, "final_questions": 2, "count_stage": "selection_complete",
                  "requested_lines": [L7], "per_line_min": {L7: 2}, "per_line_max": {L7: 2},
                  "corpus_tokens": 4000, "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 1}
        run = FileRun(Path(folder.name), target)
        rows = rows or [question("compare-a"), question("compare-b"), question("native", native=True)]
        plan = author_brief(target, 6)
        plan.update(version=2, instance_policy="business-instance-plan/v3", requirements=[requirements(L7, 6)],
                    applicability=[{"line": L7, "applicable": True, "implemented": True, "reason": "Fixture"}])
        values = {"01_whitepaper.json": {"supply_plan": plan}, "03_raw_orders.json": rows,
                  "03_orders.json": rows, "04_questions.json": rows,
                  "00_about.json": {"protocol": "fixture"}, "02_world.json": {"entities": {}},
                  "05_corpus.json": {"corpus": {"sessions": []}}, "05_corpus_token_scale.json": {"target_met": True},
                  "06_grounded_questions.json": [row for row in rows if row["qid"] in released],
                  "06_grounding_report.json": {"drops": [{"qid": qid} for qid in rejected],
                                               "pending": [{"qid": qid} for qid in pending]},
                  "manifest.json": run.manifest,
                  "experiment_profile.json": {"model": "offline-dummy", "max_calls": 5000, "max_cny": 100,
                                              "admitted_calls": 11, "budget_consumed_cny": 0.13}}
        for name, value in values.items():
            run.write(name, value)
        run.write("07_release.json", evaluate_release(run.dir))
        self.assertTrue(quality_snapshot(run.dir)["eligible"])
        return run

    def outcome(self, run, *, selected=False):
        # Material acceptance is independent of the native partition under test.
        with patch.object(factory, "_corpus_is_current", return_value=True):
            return production.outcome(run, selected=selected)

    def test_rejected_native_does_not_hide_behind_two_released_comparisons(self):
        run = self.make_run()
        result = self.outcome(run)
        self.assertEqual(result["counts"][L7], 2)
        self.assertTrue(result["report"]["quality_eligible"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["deficits"], {L7: 1})
        self.assertEqual(result["native_trend"]["qids"], [])
        self.assertEqual(result["report"]["by_line"][0]["primary"]["rejected"], 1)

    def test_pending_native_does_not_count_as_released(self):
        result = self.outcome(self.make_run(rejected=(), pending=("native",)))
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_trend"]["actual"], 0)

    def test_released_native_and_comparison_satisfy_both_contracts(self):
        result = self.outcome(self.make_run(released=("compare-a", "native"), rejected=("compare-b",)))
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["native_trend"]["qids"], ["native"])
        self.assertEqual(result["native_trend"]["actual"], 1)

    def test_legacy_planted_s1_is_not_native_supply(self):
        rows = [question("compare-a"), question("compare-b"), question("native", planted=True)]
        result = self.outcome(self.make_run(rows, released=("compare-a", "native"), rejected=("compare-b",)))
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_trend"]["actual"], 0)

    def test_question_subtype_cannot_relabel_a_comparison_order(self):
        run = self.make_run(released=("compare-a", "native"), rejected=("compare-b",))
        orders = run.read("03_orders.json")
        orders[-1]["aux"] = {"sub": "S2_compare"}
        run.write("03_orders.json", orders)
        result = self.outcome(run)
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_trend"]["source_mismatches"], ["native"])

    def test_stale_release_never_retains_native_credit(self):
        run = self.make_run(released=("compare-a", "native"), rejected=("compare-b",))
        run.write("05_corpus.json", {"changed": True})
        result = self.outcome(run)
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_trend"]["actual"], 0)

    def test_duplicate_native_family_does_not_fill_a_larger_frozen_requirement(self):
        rows = [question("native-a", native=True, entity="same"), question("native-b", native=True, entity="same")]
        run = self.make_run(rows, released=("native-a", "native-b"), rejected=())
        wp = run.read("01_whitepaper.json")
        wp["supply_plan"]["requirements"][0]["subtype_minimum"]["native_trend"] = 2
        run.write("01_whitepaper.json", wp)
        result = self.outcome(run)
        self.assertEqual(result["counts"][L7], 2)
        self.assertEqual(result["native_trend"]["actual"], 1)
        self.assertFalse(result["passed"])

    def test_original_controller_cannot_mark_comparison_only_run_generation_ready(self):
        run = self.make_run()
        ledger = run.read("experiment_profile.json")
        token = enter_bounded_run(run.run_id)
        try:
            with patch.object(factory, "_corpus_is_current", return_value=True), \
                    patch.object(production, "ensure_instance_plan"), patch.object(production, "drive"):
                state = production._produce(run, to_stage="quality")
        finally:
            leave_bounded_run(token)
        self.assertEqual(state["status"], "round_limit")
        self.assertFalse(state["result"]["passed"])
        self.assertEqual(run.read("experiment_profile.json"), ledger)

    def test_selected_subset_must_retain_the_released_native(self):
        from eval.question_filter import question_key
        from pipeline import calibration
        run = self.make_run(released=("compare-a", "compare-b", "native"), rejected=())
        rows = run.read("06_grounded_questions.json")
        systems = ["a", "b", "c", "d"]
        run.write("09_selected_questions.json", rows[:2])
        items = [{"qid": row["qid"], "line": L7, "capability": L7, "question": row["question"],
                  "key": question_key(row), "correct": {name: row["qid"] == "native" or name != "a" for name in systems},
                  "issues": {}, "disposition": "removed_easy" if row["qid"] == "native" else "kept_not_all_correct"}
                 for row in rows]
        run.write("09_selection.json", {"schema_version": 3, "systems": systems,
            "rule": "all_selected_systems_correct", "keep_easy_ratio": 0,
            "result_scope": "validated_input_identity", "items": items})
        with patch.object(calibration, "selection_is_current", return_value=True):
            result = self.outcome(run, selected=True)
        self.assertTrue(result["report"]["selection_complete"])
        self.assertEqual(result["counts"][L7], 2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_trend"]["actual"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
