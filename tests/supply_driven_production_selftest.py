"""Real fixture gates and synthetic original-driver production transactions.

Network is disabled. Synthetic quality opinions test orchestration, not quality.
"""
import os
os.environ.setdefault("OPENAI_API_KEY", "offline-test")
os.environ.setdefault("MODEL", "offline-model")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import config
from pipeline import factory, production, supply, supply_capacity, run as run_module
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from pipeline.run import Run, Stage
from pipeline.world_state import WorldState
from pipeline.world_agent import generate_world
from seed_world_selftest import fixture
from supply_pipeline_selftest import target
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part
import release_pipeline_selftest as release_fixture


class CapacityGateTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network/model"))
            guard.start(); self.addCleanup(guard.stop)
        self.wp, self.world, self.table = fixture()
        self.wp["line_mapping"] = [{"line": line, "applicable": True, "instantiation": "Actual test carrier"}
                                   for line in ("L1_timeline", "L3_process", "L5_conflict", "L7_consolidation", "L8_transition")]

    def test_impossible_state_carriers_rejected_before_world(self):
        brief = supply.author_brief(target(["L8_transition"], 10), 30)
        report = supply_capacity.blueprint_capacity_report(self.wp["world_blueprint"], brief)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["rows"][0]["schema_upper_bound"], 0)
        with self.assertRaises(ValueError):
            supply.attach_plan(self.wp, target(["L8_transition"], 10), 30, capacity_driven=True)

    def test_l5_default_three_is_replaced_by_its_candidate_allocation(self):
        supply.attach_plan(self.wp, target(["L5_conflict"], 1), 2, capacity_driven=True)
        self.assertEqual(self.wp["domain_profile"]["l5_max_conflicts"], 2)
        self.assertEqual(self.wp["supply_plan"]["version"], 2)

    def test_real_showcase_orders_fail_before_corpus(self):
        folder = Path(__file__).resolve().parents[1] / "output/showcase_quality_audit_20260926/run"
        if not folder.exists():
            self.skipTest("Optional historical fixture")
        with tempfile.TemporaryDirectory() as temp, patch.object(run_module, "RUNS_DIR", Path(temp)):
            run = Run("fixture", "real-gate")
            wp = json.loads((folder / "01_whitepaper.json").read_text(encoding="utf-8"))
            wp["supply_plan"]["version"] = 2
            run.write("01_whitepaper.json", wp)
            for name in ("02_world.json", "03_orders.json"):
                run.write(name, json.loads((folder / name).read_text(encoding="utf-8")))
            with self.assertRaises(supply_capacity.CapacityReviewRequested):
                factory.stage_corpus(run)
            receipt = run.read("03_capacity_gate.json")
            self.assertEqual(receipt["deficits"]["L5_conflict"], 9)
            self.assertEqual(receipt["deficits"]["L7_consolidation"], 10)
            self.assertEqual(receipt["deficits"]["L8_transition"], 11)
            self.assertFalse(run.has("05_corpus.json"))

    def test_no_op_finish_routes_to_quantity_architect(self):
        supply.attach_plan(self.wp, target(["L1_timeline", "L3_process"], 4), 10, capacity_driven=True)
        trace = ScriptedTracer([(PLAN, write("one")), (WRITE, table_part(self.table, [0, 1], True)),
                                (PLAN, FINISH), (PLAN, FINISH)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "world.json"
            with self.assertRaises(supply_capacity.CapacityReviewRequested):
                generate_world(self.wp, trace, checkpoint_path=path, log=lambda *_: None)
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "blueprint_review_requested")
            self.assertTrue(state["upstream_request"]["unchanged_world"])
            self.assertNotIn("supply_warning", state)

    def test_no_easy_line_padding_after_filter(self):
        t = target(["L1_timeline", "L7_consolidation"], 4)
        questions = [{"qid": f"q{i}", "line": "L1_timeline", "capability": "IE"} for i in range(8)]
        questions += [{"qid": "hard", "line": "L7_consolidation", "capability": "S1"}]
        report = {"items": [{"qid": q["qid"], "disposition": "kept_not_all_correct"} for q in questions]}
        selected, receipt = supply.balanced_selection_subset(questions, report, t, strict_allocation=True)
        self.assertEqual(receipt["selected_by_line"], {"L1_timeline": 2, "L7_consolidation": 1})
        self.assertEqual(len(selected), 3)

    def test_capacity_architect_revises_actual_schema_with_original_business_review(self):
        supply.attach_plan(self.wp, target(["L7_consolidation"], 1), 2, capacity_driven=True)
        before = deepcopy(self.wp)
        proposal = {"decision": "revise", "reason": "Add an independent revision carrier",
                    "entity_counts": {"report": 2}, "event_min_counts": {"revision": 4},
                    "relation_min_counts": {"issued_by": 2},
                    "line_constructions": [{"line": "L7_consolidation", "construction": "Two independent report trajectories"}]}
        review = {"decision": "accept", "reason": "Original mechanism remains executable",
                  "mechanism_checks": [{"mechanism_id": "revision_review", "walkthrough": "Each report revision causes its review", "status": "feasible"}], "issues": []}
        trace = ScriptedTracer([("council.supply_capacity", proposal), ("council.blueprint_feasibility", review)])
        audit = {}
        revised = supply_capacity.revise_capacity(self.wp, trace, {"actual_gap": 1}, audit)
        self.assertEqual(self.wp, before)
        self.assertEqual(revised["world_blueprint"]["entity_types"][1]["count"], 2)
        self.assertEqual(revised["seed_contract"], before["seed_contract"])
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual([call["step"] for call in trace.calls], ["council.supply_capacity", "council.blueprint_feasibility"])

    def test_capacity_revision_rejects_no_change_and_authorized_limit_breach(self):
        supply.attach_plan(self.wp, target(["L7_consolidation"], 1), 2,
                           world_limits={"max_world_entities": 2}, capacity_driven=True)
        proposal = {"decision": "revise", "reason": "Proposal", "entity_counts": {},
                    "event_min_counts": {}, "relation_min_counts": {},
                    "line_constructions": [{"line": "L7_consolidation", "construction": "Unchanged carriers"}]}
        for counts in ({}, {"report": 2}):
            with self.subTest(counts=counts):
                proposal["entity_counts"] = counts
                trace = ScriptedTracer([("council.supply_capacity", proposal)])
                with self.assertRaises(supply_capacity.WorldBlueprintError):
                    supply_capacity.revise_capacity(self.wp, trace, {}, {})
                self.assertEqual(len(trace.calls), 1)


class ProductionTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        guard = patch.object(run_module, "RUNS_DIR", Path(self.temp.name)); guard.start(); self.addCleanup(guard.stop)
        self.t = target(["L1_timeline", "L7_consolidation"], 4); self.t["max_supply_rounds"] = 2
        self.cfg = {"delivery_target": self.t, "question_budget": 6,
                    "production_control": {"version": supply_capacity.VERSION}}
        self.run = Run("fixture", "controller", config_meta=self.cfg)
        bounded_token = enter_bounded_run(self.run.run_id)
        self.addCleanup(leave_bounded_run, bounded_token)
        wp, world, _ = fixture()
        wp["world_blueprint"]["entity_types"][1]["count"] = 2
        wp["line_mapping"] = [{"line": line, "applicable": True, "instantiation": "Real carrier"} for line in self.t["requested_lines"]]
        supply.attach_plan(wp, self.t, 6, capacity_driven=True)
        self.run.write("01_whitepaper.json", wp)
        self.run.write("02_world.json", world.to_dict())
        self.run.write("03_orders.json", [])
        self.run.write("05_corpus.json", {"old": "retained"})
        self.run.write("experiment_profile.json", {"admitted_calls": 17, "unknown": 2})
        self.run.tracer.pfile.write_text('{"i":1,"step":"old"}\n', encoding="utf-8")
        self.run.tracer.n = 1
        self.state = {"version": supply_capacity.VERSION, "identity": production.frozen_identity(self.run),
                      "round": 1, "status": "running", "history": []}

    def revised(self, wp, tracer, feedback, audit, *, checkpoint=None):
        new = deepcopy(wp)
        new["world_blueprint"]["entity_types"][1]["count"] += 1
        audit.update(status="accepted", before=deepcopy(wp), after=new, feedback=deepcopy(feedback),
                     implementation=supply_capacity.revision_implementation(), after_hash=supply_capacity.digest(new))
        if checkpoint is not None:checkpoint(audit)
        return new

    def result(self, l7=1, errors=()):
        rows = [{"line": line, "primary": {"effective_orders": 3}} for line in self.t["requested_lines"]]
        return {"counts": {"L1_timeline": 2, "L7_consolidation": l7}, "count_stage": "generation_quality",
                "deficits": {"L7_consolidation": 2-l7} if l7 < 2 else {},
                "review_execution_errors": list(errors), "passed": l7 >= 2,
                "report": {"by_line": rows, "quality_eligible": True, "selection_complete": True,
                           "corpus_target": {"current": True, "measurement": {"target_met": True}}}}

    def test_line_specific_losses_increase_only_that_reserve(self):
        plan = self.run.read("01_whitepaper.json")["supply_plan"]
        self.assertEqual(production.next_reserve(plan, self.result()), {"L1_timeline": 3, "L7_consolidation": 6})
        with self.assertRaises(ValueError):
            production.next_reserve(plan, self.result(l7=0))

    def test_accepted_capacity_revision_retires_the_previous_resume_binding(self):
        name = "02_world_agent_parent.json"
        self.run.write(name, {"old_world": True})
        self.run.manifest["derived_from"] = {"operation":"engineering_recovery",
            "world_construction_checkpoints":[name]}
        self.run._save_manifest()
        original_ledger = (self.run.dir/"experiment_profile.json").read_bytes()
        with patch.object(production, "revise_capacity", side_effect=self.revised):
            production.prepare_revision(self.run, self.state, factory.STAGES,
                {"L1_timeline":3,"L7_consolidation":6}, {"line":"L7"})
        derived = self.run.manifest["derived_from"]
        self.assertNotIn("world_construction_checkpoints", derived)
        self.assertEqual(derived["operation"], "engineering_recovery")
        entry = derived["world_construction_checkpoint_history"][0]
        self.assertEqual(entry["archived_round"], 1)
        self.assertIn(name, entry["files"])
        self.assertNotEqual(entry["old_whitepaper_sha256"], entry["new_whitepaper_sha256"])
        self.assertTrue((self.run.dir/"production/round-001"/name).exists())
        self.assertEqual((self.run.dir/"experiment_profile.json").read_bytes(), original_ledger)

    def test_same_input_keeps_its_active_resume_binding(self):
        manifest={"derived_from":{"world_construction_checkpoints":["cp"]}}
        before=deepcopy(manifest)
        receipt={"files":{"01_whitepaper.json":"same","cp":"checkpoint"}}
        self.assertIsNone(production.retire_checkpoint_reuse(manifest,receipt,2,"same"))
        self.assertEqual(manifest,before)

    def test_unarchived_checkpoint_keeps_its_resume_protection(self):
        manifest={"derived_from":{"world_construction_checkpoints":["cp"]}}
        before=deepcopy(manifest)
        with self.assertRaisesRegex(ValueError,"missing from the archived round"):
            production.retire_checkpoint_reuse(manifest,{"files":{"01_whitepaper.json":"old"}},2,"new")
        self.assertEqual(manifest,before)

    def test_install_interruption_replays_without_erasing_ledger(self):
        allocation = {"L1_timeline": 3, "L7_consolidation": 6}
        original = self.run.write
        def fail_once(name, value):
            if name == "01_supply_plan.json":
                raise OSError("simulated installation interruption")
            return original(name, value)
        with patch.object(production, "revise_capacity", side_effect=self.revised) as revise, patch.object(self.run, "write", side_effect=fail_once):
            with self.assertRaises(OSError):
                production.prepare_revision(self.run, self.state, factory.STAGES, allocation, {"line": "L7"})
            self.assertEqual(revise.call_count, 1)
        saved = self.run.read(production.STATE)
        self.assertEqual(saved["status"], "installing_revision")
        with self.run.stage_write_lock("test_resume"):
            production.install_revision(self.run, saved, factory.STAGES)
        self.assertEqual(saved["round"], 2)
        self.assertEqual(self.run.read("01_supply_plan.json")["candidate_allocation"], allocation)
        self.assertEqual(self.run.read("experiment_profile.json"), {"admitted_calls": 17, "unknown": 2})
        self.assertEqual(self.run.tracer.pfile.read_text(encoding="utf-8"), '{"i":1,"step":"old"}\n')
        archived = self.run.dir / "production/round-001/05_corpus.json"
        self.assertEqual(json.loads(archived.read_text(encoding="utf-8")), {"old": "retained"})

    def test_successful_capacity_audit_reused_after_archive_interruption(self):
        allocation = {"L1_timeline": 3, "L7_consolidation": 6}
        with patch.object(production, "revise_capacity", side_effect=self.revised) as revise:
            with patch.object(production, "archive_round", side_effect=OSError("interruption after accepted plan")):
                with self.assertRaises(OSError):
                    production.prepare_revision(self.run, self.state, factory.STAGES, allocation, {"line": "L7"})
            production.prepare_revision(self.run, self.state, factory.STAGES, allocation, {"line": "L7"})
            self.assertEqual(revise.call_count, 1)

    def test_controller_refills_quality_loss_using_original_drive(self):
        def write_artifact(name):
            return lambda run: run.write(name, {"fixture": True})
        stages = [Stage("input", [], write_artifact("00_input.json"), "00_input.json"),
                  Stage("whitepaper", ["input"], lambda run: None, "01_whitepaper.json"),
                  Stage("world", ["whitepaper"], write_artifact("02_world.json"), "02_world.json"),
                  Stage("well_posed", ["world"], write_artifact("03_orders.json"), "03_orders.json"),
                  Stage("corpus", ["well_posed"], write_artifact("05_corpus.json"), "05_corpus.json"),
                  Stage("quality", ["corpus"], write_artifact("07_release.json"), "07_release.json")]
        with patch.object(factory, "STAGES", stages), patch.object(factory, "generation_stages", return_value=stages), \
                patch.object(production, "outcome", side_effect=[self.result(), self.result(l7=2)]), \
                patch.object(production, "revise_capacity", side_effect=self.revised):
            state = production.produce(self.run, to_stage="quality")
        self.assertEqual(state["round"], 2)
        self.assertEqual(state["status"], "generation_ready_awaiting_selection")
        self.assertEqual(self.run.read("01_supply_plan.json")["candidate_allocation"]["L7_consolidation"], 6)
        self.assertTrue((self.run.dir / "production/round-001/07_release.json").exists())

    def test_controller_refills_after_complete_athlete_filter(self):
        from pipeline.calibration import load_config
        root = Path(__file__).resolve().parents[1]
        self.run.manifest["config"]["calibration"] = load_config(root / "examples/release_four.json")
        self.run._save_manifest()
        original = list(factory.STAGES)
        extra = [Stage("calibration", ["quality"], lambda run: None, "08_calibration.json"),
                 Stage("selection", ["calibration"], lambda run: None, "09_selection.json")]
        full = self.result(l7=2)
        short = self.result(l7=1); short["count_stage"] = "selection_complete"
        delivered = self.result(l7=2); delivered["count_stage"] = "selection_complete"
        with patch.object(factory, "generation_stages", return_value=original+extra), \
                patch.object(production, "drive"), \
                patch.object(production, "outcome", side_effect=[full, short, full, delivered]), \
                patch.object(production, "revise_capacity", side_effect=self.revised) as revise:
            state = production.produce(self.run, to_stage="selection")
        self.assertEqual(state["status"], "delivered")
        self.assertEqual(state["round"], 2)
        self.assertEqual(revise.call_args.args[2]["stage"], "selection_complete")
        self.assertEqual(revise.call_args.args[0]["supply_plan"]["candidate_allocation"]["L7_consolidation"], 6)

    def test_review_execution_failure_stops_quantity_regeneration(self):
        with patch.object(production, "drive"), patch.object(production, "outcome", return_value=self.result(errors=["bad-reader"])), \
                patch.object(production, "revise_capacity", side_effect=AssertionError("No quantity retry")):
            state = production.produce(self.run, to_stage="quality")
        self.assertEqual(state["status"], "review_recovery_required")
        self.assertEqual(state["round"], 1)

    def test_world_capacity_request_uses_the_same_production_round_budget(self):
        calls = []
        def drive(run, stages, from_stage=None, to_stage=None, *args, **kwargs):
            calls.append(from_stage)
            if from_stage == "world" and calls.count("world") == 1:
                raise supply_capacity.CapacityReviewRequested({"reason": "Actual world lacks independent trajectories"})
        with patch.object(production, "drive", side_effect=drive), \
                patch.object(production, "outcome", return_value=self.result(l7=2)), \
                patch.object(production, "revise_capacity", side_effect=self.revised) as revised:
            state = production.produce(self.run, to_stage="quality")
        self.assertEqual(state["round"], 2)
        self.assertEqual(state["status"], "generation_ready_awaiting_selection")
        self.assertEqual(revised.call_count, 1)
        self.assertEqual(calls.count("world"), 2)

    def test_all_installation_commit_boundaries_replay(self):
        allocation = {"L1_timeline": 3, "L7_consolidation": 6}
        for boundary in ("unlink", "manifest", "state_commit"):
            with self.subTest(boundary=boundary):
                self.setUp()
                original_write, original_unlink = self.run.write, Path.unlink
                def interrupted_write(name, value):
                    if boundary == "state_commit" and name == production.STATE and value["status"] == "running":
                        raise OSError("state commit interruption")
                    return original_write(name, value)
                def interrupted_unlink(path, *args, **kwargs):
                    if boundary == "unlink" and path.name == "05_corpus.json":
                        raise OSError("partial deletion interruption")
                    return original_unlink(path, *args, **kwargs)
                with patch.object(production, "revise_capacity", side_effect=self.revised), \
                        patch.object(self.run, "write", side_effect=interrupted_write), \
                        patch.object(Path, "unlink", interrupted_unlink):
                    if boundary == "manifest":
                        with patch.object(self.run, "_save_manifest", side_effect=OSError("manifest interruption")):
                            with self.assertRaises(OSError):
                                production.prepare_revision(self.run, self.state, factory.STAGES, allocation, {})
                    else:
                        with self.assertRaises(OSError):
                            production.prepare_revision(self.run, self.state, factory.STAGES, allocation, {})
                saved = self.run.read(production.STATE)
                self.assertEqual(saved["status"], "installing_revision")
                with self.run.stage_write_lock("zero_call_replay"):
                    production.install_revision(self.run, saved, factory.STAGES)
                self.assertEqual(saved["round"], 2)
                self.assertEqual(self.run.tracer.n, 1)

    def test_shared_physical_budget_cannot_reset_or_change_cap(self):
        profile = {"model": "offline-model", "max_calls": 100, "max_cny": 5,
                   "admitted_calls": 17, "budget_consumed_cny": 1}
        self.run.write("experiment_profile.json", profile)
        production.observe_budget(self.run, self.state)
        profile["admitted_calls"] = 0
        self.run.write("experiment_profile.json", profile)
        with self.assertRaises(ValueError):
            production.observe_budget(self.run, self.state)
        profile.update(admitted_calls=17, max_calls=200)
        self.run.write("experiment_profile.json", profile)
        with self.assertRaises(ValueError):
            production.observe_budget(self.run, self.state)


class CompleteShortSelectionTest(unittest.TestCase):
    def test_actual_native_scores_and_selection_remain_current_with_line_shortfall(self):
        from pipeline import calibration
        fixture = release_fixture.ReleasePipelineTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        run = fixture.run
        t = target(["L1_timeline", "L2_relational"], 4)
        run.manifest["config"].update(delivery_target=t, production_control={"version": supply_capacity.VERSION})
        run._save_manifest()
        support = [{"line": line, "applicable": True, "implemented": True, "reason": "Synthetic fixture"} for line in t["requested_lines"]]
        brief = supply.author_brief(t, 12)
        run.write("01_whitepaper.json", {"supply_plan": {**brief, "version": 2, "applicability": support}})
        run.write("03_raw_orders.json", run.read("04_questions.json"))
        run.write("03_orders.json", run.read("04_questions.json"))
        run.write("05_corpus_token_scale.json", {"target_met": True})
        with patch.object(factory, "_corpus_is_current", return_value=True):
            run_module.drive(run, fixture.tail())
            selected = run.read(calibration.SELECTION_ARTIFACT)
            self.assertEqual(selected["status"], "complete")
            self.assertFalse(selected["release_ready"])
            self.assertIsNone(selected["benchmark"])
            self.assertEqual(selected["exported_question_count"], 0)
            self.assertIn("per-line delivery shortfall", selected["note"])
            self.assertTrue(calibration.selection_is_current(run))
            result = production.outcome(run, selected=True)
        self.assertTrue(result["report"]["selection_complete"])
        self.assertEqual(result["deficits"], {"L1_timeline": 2, "L2_relational": 1})
        self.assertEqual(len(fixture.calls), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
