"""Offline integration for early supply feedback and fixed per-line reserves."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from pipeline import central_office, factory, supply
from pipeline.lines import run_lines
from pipeline.world_agent import generate_world
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part
from seed_world_selftest import fixture


def target(lines=None, final=4):
    return {"version": 1, "final_questions": final, "count_stage": "selection_complete",
            "requested_lines": lines or ["L1_timeline", "L3_process"], "per_line_min": {},
            "per_line_max": {}, "corpus_tokens": 1000, "tokenizer": "cl100k_base@0.12.0",
            "max_supply_rounds": 1}


class SupplyPipelineTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network/model"))
            guard.start(); self.addCleanup(guard.stop)
        self.wp, self.world, self.table = fixture()
        self.wp["active_lines"] = [{"line": "L1_timeline", "weight": 100}, {"line": "L3_process", "weight": .01}]
        self.wp["line_mapping"] = [{"line": x, "applicable": True, "instantiation": "Real field history"} for x in target()["requested_lines"]]
        supply.attach_plan(self.wp, target(), 10)

    def test_equal_allocation_precedes_legacy_model_weights(self):
        stats = {}
        run_lines(self.wp, self.world, lambda *_: None, question_budget=10, stats=stats)
        self.assertEqual([r["allocated"] for r in stats["lines"]], [5, 5])
        self.assertEqual(stats["shortfall"], 10-stats["selected"])
        self.assertTrue(all(r["selected"] <= 5 for r in stats["lines"]))

    def test_unsupported_lines_keep_their_shortfall(self):
        t = target(["L1_timeline", "L9_induction"])
        self.wp["line_mapping"].append({"line": "L9_induction", "applicable": True, "instantiation": "Proposed induction"})
        supply.attach_plan(self.wp, t, 10)
        stats = {}
        run_lines(self.wp, self.world, lambda *_: None, question_budget=10, stats=stats)
        l9 = next(row for row in stats["lines"] if row["line"] == "L9_induction")
        self.assertEqual(l9["allocated"], 5)
        self.assertEqual(l9["selected"], 0)
        self.assertIn("pending", l9["feasibility_reason"])

    def test_world_preview_has_no_side_effects(self):
        before = deepcopy(self.world.to_dict())
        observed = supply.inventory(self.wp, self.world)
        self.assertEqual(before, self.world.to_dict())
        self.assertEqual(sum(row["selected"] for row in observed["lines"]), observed["selected"])

    def test_l7_preview_includes_final_deterministic_imprint(self):
        wp, world, _ = fixture()
        # The seed contract protects the numeric fixture field; remove that
        # constraint to exercise the ordinary L7 imprint path.
        wp.pop("seed_contract")
        wp["line_mapping"] = [{"line": "L7_consolidation", "applicable": True,
                               "instantiation": "Independent numeric trend"}]
        supply.attach_plan(wp, target(["L7_consolidation"], 1), candidate_budget=1)
        before = deepcopy(world.to_dict())
        observed = supply.inventory(wp, world)
        self.assertEqual((observed["selected"], observed["shortfall"]), (1, 0))
        self.assertEqual(world.to_dict(), before)
        self.assertFalse(getattr(world, "_trended_fields", []))

    def test_bounded_supplement_preserves_valid_short_world_and_resume(self):
        full = table_part(self.table, [0, 1], True)
        trace = ScriptedTracer([(PLAN, write("one")), (WRITE, full), (PLAN, FINISH), (PLAN, FINISH)])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "world.json"
            result, meta = generate_world(self.wp, trace, checkpoint_path=checkpoint, log=lambda *_: None)
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["supply_supplement_rounds"], 1)
            self.assertEqual(len(meta["supply_observations"]), 2)
            self.assertGreater(meta["supply_warning"]["shortfall"], 0)
            self.assertIn("supply_shortfall", trace.calls[-1]["input"]["observation"])
            again = ScriptedTracer([])
            replay, _ = generate_world(self.wp, again, checkpoint_path=checkpoint, log=lambda *_: None)
            self.assertEqual(replay, result)
            self.assertFalse(again.calls)

    def test_new_mode_map_checks_all_lines_and_legacy_seven_stays(self):
        rows = [{"line": line, "applicable": False} for line in central_office._MAP_LINE_IDS]
        self.assertFalse(central_office._map_issues({"per_line": rows}))
        from pipeline.delivery_target import LINE_IDS
        self.assertTrue(central_office._map_issues({"per_line": rows}, LINE_IDS))
        rows += [{"line": line, "applicable": False} for line in LINE_IDS[7:]]
        self.assertFalse(central_office._map_issues({"per_line": rows}, LINE_IDS))

    def test_explicit_budget_under_final_goal_rejected(self):
        with self.assertRaises(ValueError):
            supply.author_brief(target(final=500), 100)

    def test_sourced_line_retention_controls_candidate_reserves(self):
        lines = ["L1_timeline", "L3_process"]
        rates = {line: {"rate": rate, "source": "fixture evaluated four-athlete run",
                        "sample_size": 100} for line, rate in zip(lines, (.5, .25))}
        brief = supply.author_brief(target(lines, final=8), survival_rates=rates)
        self.assertEqual(brief["candidate_allocation"], {"L1_timeline": 10, "L3_process": 20})
        self.assertEqual(brief["candidate_budget"], 30)
        self.assertEqual(brief["reserve_basis"], "sourced_retention_rates")
        with self.assertRaises(ValueError):
            supply.author_brief(target(lines, final=8), 20, survival_rates=rates)
        with self.assertRaises(ValueError):
            supply.author_brief(target(lines, final=8), survival_rates={lines[0]: rates[lines[0]]})

    def test_candidate_budget_drift_rejected(self):
        with self.assertRaises(ValueError):
            run_lines(self.wp, self.world, question_budget=11)

    def test_generation_report_counts_candidates_without_claiming_selection(self):
        generation_target = {"version": 2, "count_stage": "generation_quality",
            "candidate_questions": 2, "requested_lines": ["L1_timeline", "L3_process"],
            "corpus_tokens": 100, "core_tokens": 10, "filler_ratio": 9,
            "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 2}
        class Run:
            manifest = {"config": {"delivery_target": generation_target}}
            dir = Path("unused")
            artifacts = {"01_whitepaper.json": {},
                "03_orders.json": [{"line": "L1_timeline"}, {"line": "L3_process"}],
                "04_questions.json": [{"line": "L1_timeline"}, {"line": "L3_process"}],
                "06_grounded_questions.json": [{"line": "L1_timeline"}],
                "03_capacity_gate.json": {"passed": True},
                "05_corpus_token_scale.json": {"target_met": True}}
            def has(self, name): return name in self.artifacts or name == "07_release.json"
            def read(self, name): return self.artifacts[name]
            def write(self, name, value): self.artifacts[name] = value
        run = Run()
        with patch("pipeline.quality.quality_snapshot", return_value={"eligible": True}), \
             patch("pipeline.factory._corpus_is_current", return_value=True):
            report = supply.write_delivery_report(run)
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["grounded_count"], 1)
        self.assertTrue(report["candidate_target_met"])
        self.assertEqual(report["selection_scope"], "not_run")
        run.artifacts["03_capacity_gate.json"] = {"passed": False}
        with patch("pipeline.quality.quality_snapshot", return_value={"eligible": True}), \
             patch("pipeline.factory._corpus_is_current", return_value=True):
            self.assertFalse(supply.write_delivery_report(run)["candidate_target_met"])


if __name__ == "__main__":
    unittest.main()
