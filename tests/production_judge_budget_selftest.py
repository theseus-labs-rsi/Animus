"""Cross-context production caps through real native plans/scoring; no API or CLI."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("OPENAI_API_KEY", "offline-production-budget")
os.environ.setdefault("MODEL", "offline-production-budget")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import native_evaluation_selftest as fixtures
from pipeline import calibration as calibration
from pipeline import native_evaluation as native

CONTROL = {"version": "supply-driven/v3", "max_rounds": 4}


class SharedBudgetTest(unittest.TestCase):
    def test_new_objects_reload_and_freeze_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "production" / "judge_budget.json"
            first = native._Budget(path, 2, shared=True)
            second = native._Budget(path, 2, shared=True)
            first.reserve()
            second.reserve()
            with self.assertRaisesRegex(RuntimeError, "production judge call budget exhausted"):
                first.reserve()
            with self.assertRaisesRegex(ValueError, "cap/ledger binding changed"):
                native._Budget(path, 3, shared=True)
            with self.assertRaisesRegex(ValueError, "cap/ledger binding changed"):
                native._Budget(path, 1, shared=True)
            record = json.loads(path.read_text())
            self.assertEqual(record["reserved_calls"], 2)
            self.assertEqual(set(record), {"version", "max_judge_calls", "reserved_calls", "policy"})

    def test_independent_threads_do_not_over_admit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_budget.json"
            budgets = [native._Budget(path, 10, shared=True) for _ in range(20)]
            def attempt(budget):
                try:
                    budget.reserve()
                    return True
                except RuntimeError:
                    return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                admitted = list(pool.map(attempt, budgets))
            self.assertEqual(sum(admitted), 10)
            self.assertEqual(json.loads(path.read_text())["reserved_calls"], 10)

    def test_malformed_reserved_count_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_budget.json"
            native._Budget(path, 2, shared=True)
            record = json.loads(path.read_text())
            record["reserved_calls"] = True
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "reservation count"):
                native._Budget(path, 2, shared=True)

    def test_legacy_per_context_budgets_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / str(i) / "judge_budget.json" for i in range(2)]
            for path in paths:
                native._Budget(path, 1).reserve()
            self.assertEqual([json.loads(p.read_text())["reserved_calls"] for p in paths], [1, 1])


class ProductionNativeChainTest(fixtures.NativeEvaluationTest):
    def test_shared_json_retry_reservations_persist(self):
        import config
        judge = fixtures.FixtureJudge(Mock(return_value="invalid-json"))
        provider = judge._module.config.chat
        judge._module.config.chat_json = config.chat_json
        path = self.root / "production" / "judge_budget.json"
        budget = native._Budget(path, 1, shared=True)
        wrapped = native._bounded_judge(judge, self.settings, budget)
        with self.assertRaisesRegex(config.ChatJSONError, "budget exhausted"):
            wrapped._module.config.chat_json([], retries=3, strict_json=True, retry_delay_base=0)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(json.loads(path.read_text())["reserved_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            native._Budget(path, 1, shared=True).reserve()

    def test_four_contexts_share_cap_and_keep_separate_answers(self):
        self.settings["max_judge_calls"] = 5
        ledger = self.root / "production" / "judge_budget.json"
        context_outputs = []
        for index in range(4):
            questions = [{**q, "question": f"context-{index}: {q['question']}"} for q in self.questions]
            refs = [{"qid": q["qid"], "answer": f"v{i}", "answer_raw": f"v{i}",
                     "answer_projection": "raw_gt", "quality_status": "released"}
                    for i, q in enumerate(questions)]
            bench = fixtures.build_standard_light(self.root / f"benchmark-{index}", questions, refs)
            folder = self.root / f"calibration-{index}"
            context_outputs.append(native.execute_native_evaluation(bench, folder, self.settings,
                shared_judge_budget=ledger))
            self.assertFalse((folder / "judge_budget.json").exists())
        self.assertEqual(self.provider.call_count, 5)
        self.assertEqual(len(self.answer_calls), 32)
        self.assertEqual(json.loads(ledger.read_text())["reserved_calls"], 5)
        self.assertEqual(len({str(p) for outputs in context_outputs for p in outputs.values()}), 16)
        # A fifth context is refused before any athlete dispatch when the cap changes.
        self.settings["max_judge_calls"] = 6
        with self.assertRaisesRegex(ValueError, "cap/ledger binding changed"):
            native.execute_native_evaluation(self.benchmark, self.root / "changed-cap", self.settings,
                shared_judge_budget=ledger)
        self.assertEqual(self.provider.call_count, 5)
        self.assertEqual(len(self.answer_calls), 32)

    def test_calibration_passes_explicit_run_ledger(self):
        folder = self.root / "context-calibration"
        (folder / "evaluation_input").mkdir(parents=True)
        fake_run = SimpleNamespace(dir=self.root, manifest={"config": {"production_control": CONTROL}},
            read=lambda name: [{"qid": "q0", "line": "L1_timeline", "capability": "IE", "gt": "v0"}])
        result = self.root / "results.json"
        with patch.object(native, "execute_native_evaluation", return_value={"athlete-0": self.root}) as execute, \
             patch("pipeline.native_results.import_native_results", return_value=result):
            self.assertEqual(calibration.execute_calibration(fake_run, folder, self.settings), result)
        self.assertEqual(execute.call_args.kwargs,
                         {"shared_judge_budget": self.root / "production" / "judge_budget.json"})


class ProductionSupportTest(unittest.TestCase):
    def test_dsh_and_codex_supported_and_cost_boundary_reported(self):
        settings = {"backend": "native_four", "max_judge_calls": 6,
                    "targets": [{"system": "native.codex"}, {"system": "native.codex"},
                                {"system": "native.dsh"}, {"system": "native.dsh"}]}
        report = calibration.production_calibration_support(settings, CONTROL)
        self.assertTrue(report["supported"])
        self.assertEqual(report["judge_budget_scope"], "all_production_rounds")
        self.assertFalse(report["athlete_total_cost_bounded"])
        self.assertFalse(report["athlete_internal_calls_bounded"])

    def test_unsupported_legacy_backend_fails_before_creating_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "not-created"
            run = SimpleNamespace(manifest={"config": {"production_control": CONTROL}})
            with patch.object(calibration.subprocess, "run") as call:
                with self.assertRaisesRegex(ValueError, "production_calibration_unsupported_backend"):
                    calibration.execute_calibration(run, folder, {"max_judge_calls": 6})
            call.assert_not_called()
            self.assertFalse(folder.exists())

    def test_imported_results_have_no_new_calls(self):
        report = calibration.production_calibration_support(
            {"backend": "native_four", "max_judge_calls": 6, "result_dirs": {}}, CONTROL)
        self.assertEqual(report["judge_budget_scope"], "no_new_calls")

    def test_unknown_production_version_and_runner_fail(self):
        with self.assertRaisesRegex(ValueError, "invalid_control"):
            calibration.production_calibration_support({}, {"version": "unexpected"})
        with self.assertRaisesRegex(ValueError, "unsupported_athletes"):
            calibration.production_calibration_support(
                {"backend": "native_four", "max_judge_calls": 6,
                 "targets": [{"system": "offline.scripted"}] * 4}, CONTROL)

    def test_empty_stage_does_not_skip_unsupported_production_preflight(self):
        run = SimpleNamespace(manifest={"config": {"production_control": CONTROL,
                                                   "calibration": {"max_judge_calls": 6}}})
        with patch.object(calibration, "calibration_identity") as identity:
            with self.assertRaisesRegex(ValueError, "unsupported_backend"):
                calibration.stage_calibration(run)
        identity.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
