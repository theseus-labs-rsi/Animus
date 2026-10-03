"""Initial author failures, sibling commits and upstream design recovery.

Real compilers run throughout. Provider replies are scripted and all network
access is forbidden. The GPU fixture keeps the original v4 failed values.
"""
import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-disabled")
os.environ.setdefault("MODEL", "offline-disabled")
from copy import deepcopy
import gzip
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from execution_control import ExecutionStopped
from pipeline import instance_executor as executor, instance_plan, world_agent, world_gen
from pipeline.construction import compact_feedback
from pipeline.supply_capacity import CapacityReviewRequested
from instance_plan_selftest import setup_plan, authored_values, ACCEPT
from world_agent_selftest import ScriptedTracer


class InitialRecoveryTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Network/provider forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        self.base, self.plan, _, self.full, _ = setup_plan()
        self.wp = instance_plan.compile_plan(self.base, self.plan)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.json"

    def generate(self, trace, **kw):
        return executor.generate_world(self.wp, trace, checkpoint_path=self.path, log=lambda *_:None, **kw)

    def values(self):
        return authored_values(self.wp, self.wp["business_instance_plan"]["units"][0], self.full)

    def test_repeated_initial_failure_is_durable_redesign_with_no_calls_on_resume(self):
        bad = {"objects":{}, "event_values":{}}
        trace = ScriptedTracer([("world.instance.values", bad)] * 2)
        with self.assertRaises(CapacityReviewRequested) as caught:
            self.generate(trace)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "capacity_review_requested")
        self.assertEqual(saved["layout_request"], caught.exception.evidence)
        self.assertEqual(saved["steps"], 2)
        self.assertEqual(saved["units"], [])
        self.assertEqual(saved["log"][-1]["status"], "design_review_requested")
        self.assertTrue(caught.exception.evidence["findings"][0]["evidence"]["compiler_issues"])
        before = self.path.read_bytes()
        with self.assertRaises(CapacityReviewRequested):
            self.generate(ScriptedTracer([]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_distinct_invalid_replies_exhaust_local_allowance_and_reach_design(self):
        self.wp["world_generation"] = {"max_steps":40, "value_repair_attempts":6}
        trace = ScriptedTracer([("world.instance.values", {
            "objects":{}, "event_values":{}, "issue_responses":[{"response":str(n)}]}) for n in range(6)])
        with self.assertRaises(CapacityReviewRequested) as caught:
            self.generate(trace)
        self.assertEqual(len(trace.calls), 6)
        self.assertEqual(caught.exception.evidence["findings"][0]["evidence"]["author_response"]["issue_responses"], [{"response":"5"}])

    def test_explicit_unresolved_is_a_legal_response_and_preserves_reason(self):
        reply = {"decision":"unresolved", "reason":"Three fixed changes cannot fit two irreversible states"}
        trace = ScriptedTracer([("world.instance.values", reply)])
        with self.assertRaises(CapacityReviewRequested) as caught:
            self.generate(trace)
        alternatives = trace.calls[0]["input"]["response_schema"]["oneOf"]
        self.assertEqual(alternatives[1]["properties"]["decision"]["const"], "unresolved")
        evidence = caught.exception.evidence["findings"][0]["evidence"]
        self.assertEqual(evidence["author_response"], reply)
        self.assertEqual(evidence["compiler_issues"], [reply["reason"]])
        self.assertEqual(len(trace.calls), 1)

    def test_identity_allocation_exhaustion_reaches_designer(self):
        self.plan["execution_policy"] = "shared-identities/v1"
        self.wp = instance_plan.compile_plan(self.base, self.plan)
        trace = ScriptedTracer([("world.instance.identities", {"identities":{}})] * 3)
        with self.assertRaises(CapacityReviewRequested) as caught:
            self.generate(trace)
        self.assertEqual(len(trace.calls), 3)
        self.assertEqual(caught.exception.evidence["findings"][0]["code"], "identity_allocation_unresolved")
        self.assertEqual(json.loads(self.path.read_text())["status"], "capacity_review_requested")

    def test_no_value_unit_compiler_exception_reaches_designer(self):
        self.plan["units"].append({"unit_id":"observe", "business_purpose":"Observe the already compiled report",
            "depends_on":["report-case"], "objects":[],
            "observations":deepcopy(self.plan["units"][0]["observations"])})
        self.wp = instance_plan.compile_plan(self.base, self.plan)
        original = executor.materialize
        def reject_read_only(task, raw):
            if task["unit_id"] == "observe":
                raise executor.WorldBlueprintError("Compiled observation references an invalid business field")
            return original(task, raw)
        with patch.object(executor, "materialize", side_effect=reject_read_only):
            with self.assertRaises(CapacityReviewRequested) as caught:
                self.generate(ScriptedTracer([("world.instance.values", self.values())]))
        self.assertEqual(caught.exception.evidence["findings"][0]["affected_ids"], ["observe"])
        self.assertIn("invalid business field", str(caught.exception.evidence))

    def parallel_case(self):
        case = self.plan["units"][0]
        company = case["objects"].pop(0)
        case["depends_on"] = ["foundation"]
        self.plan["units"] = [
            {"unit_id":"foundation", "business_purpose":"Company identity", "objects":[company]},
            {"unit_id":"independent", "business_purpose":"Independent company", "objects":[{"slot":"other-company", "type":"company"}]}, case]
        self.wp = instance_plan.compile_plan(self.base, self.plan)

    def assert_sibling_committed(self, failure, expected_error):
        self.parallel_case()
        barrier = threading.Barrier(2)
        failure_saved = threading.Event()
        original_save = world_agent._save
        calls = []
        def save(path, state):
            original_save(path, state)
            if state.get("layout_request") or any(r.get("status") == "execution_error" for r in state["log"]):
                failure_saved.set()
        class Tracer:
            def chat_json(_, step, messages, **kwargs):
                uid = json.loads(messages[-1]["content"])["task"]["unit_id"]
                calls.append(uid); barrier.wait(timeout=5)
                if uid == "foundation":
                    return deepcopy(failure)
                if not failure_saved.wait(timeout=5):
                    raise AssertionError("Sibling failure was not saved")
                return {"objects":{"other-company":{"name":"Beryl", "fields":{
                    "sector":{"type":"stable", "value":"insurance"}}}}, "event_values":{}}
        with patch.object(config, "LLM_CONCURRENCY", 2), patch.object(world_agent, "_save", side_effect=save):
            with self.assertRaises(expected_error):
                self.generate(Tracer())
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual([r["unit_id"] for r in saved["units"]], ["independent"])
        self.assertEqual(set(calls), {"foundation", "independent"})
        self.assertEqual(saved["units"][0]["raw"]["entities"][0]["name"], "Beryl")
        return saved

    def test_successful_sibling_is_committed_after_design_request_before_failure_propagates(self):
        saved = self.assert_sibling_committed({"decision":"unresolved", "reason":"Fixed layout conflict"}, CapacityReviewRequested)
        self.assertEqual(saved["status"], "capacity_review_requested")

    def test_global_provider_stop_remains_fatal_and_retains_finished_sibling(self):
        saved = self.assert_sibling_committed({"__error__":"Provider denied authentication",
            "__error_metadata__":{"kind":"http_status_401", "global_stop":True}}, ExecutionStopped)
        self.assertEqual(saved["status"], "error")
        self.assertNotIn("layout_request", saved)

    def test_complete_compiler_failure_reaches_designer_with_accepted_units_preserved(self):
        original = world_agent._compiled
        def reject_complete(table, blueprint, existing, complete, wp=None):
            world, applied, initial, issues = original(table, blueprint, existing, complete, wp)
            return world, applied, initial, issues + (["Cross-unit final constraint conflict"] if complete else [])
        with patch.object(world_agent, "_compiled", side_effect=reject_complete):
            with self.assertRaises(CapacityReviewRequested) as caught:
                self.generate(ScriptedTracer([("world.instance.values", self.values())]))
        self.assertEqual(caught.exception.evidence["findings"][0]["code"], "complete_world_compilation")
        self.assertEqual(len(json.loads(self.path.read_text())["units"]), 1)

    def test_finalization_failure_reaches_the_same_designer(self):
        self.wp["world_generation"] = {"strategy":"agentic"}
        original = world_gen.assemble_world
        def reject_final(*args, **kwargs):
            world, issues = original(*args, **kwargs)
            return world, issues + ["Final compiler conflict"]
        with patch.object(world_gen, "assemble_world", side_effect=reject_final):
            with self.assertRaises(CapacityReviewRequested) as caught:
                world_gen.build_world(self.wp, ScriptedTracer([("world.instance.values", self.values())]), log=lambda *_:None)
        self.assertEqual(caught.exception.evidence["findings"][0]["code"], "world_finalization")

    def test_compact_feedback_keeps_local_compiler_error_and_overall_shortage(self):
        local = {"code":"fixed_layout_values_unresolved", "evidence":{"compiler_issues":["state reversal"]}}
        shortage = {"code":"native_trend_shortage"}
        evidence = {"phase":"business_layout", "findings":[local],
            "instance_fulfillment":{"findings":[shortage], "world_hash":"original-world"}}
        before = deepcopy(evidence)
        compact = compact_feedback(evidence)
        self.assertEqual(compact["findings"], [local, shortage])
        self.assertEqual(evidence, before)

    def test_original_controller_redesigns_after_initial_failure_and_finishes_world(self):
        from pipeline import factory, joint_design, production, run as run_module
        from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
        from pipeline.run import Run, Stage
        revised_plan = deepcopy(self.plan)
        revised_plan["units"][0]["business_purpose"] += "; keep revision changes compatible with the ordered states"
        trace = ScriptedTracer([
            ("council.blueprint_feasibility", ACCEPT),
            ("world.instance.values", {"objects":{}, "event_values":{}}),
            ("world.instance.values", {"objects":{}, "event_values":{}}),
            ("council.joint_design_revision", {"decision":"replace_plan", "reason":"Rework the rejected fixed layout"}),
            ("council.instance_plan_replacement", revised_plan),
            ("council.blueprint_feasibility", ACCEPT),
            ("world.instance.values", self.values())])
        with patch.object(run_module, "RUNS_DIR", Path(self.temp.name)):
            run = Run("fixture", "initial-recovery", config_meta={
                "production_control":{"version":"supply-driven/v6"}})
        run.tracer.chat_json = trace.chat_json
        bounded = enter_bounded_run(run.run_id)
        self.addCleanup(leave_bounded_run, bounded)
        self.base["_council_views"] = {"world_author_attempts":1}
        self.base["world_generation"] = {"strategy":"agentic", "value_repair_attempts":6}
        audit = {"version":joint_design.VERSION, "status":"initial_draft_ready"}
        paper, receipt = joint_design.accept(run, self.base, None, None, audit,
            lambda row:run.write("01_joint_design_audit.json", row), initial_proposal=self.plan)
        run.write("01_whitepaper.json", paper)
        run.write("01_joint_design_receipt.json", receipt)
        def build(run):
            world = world_gen.build_world(run.read("01_whitepaper.json"), run.tracer,
                checkpoint_path=run.dir/"02_world_agent_fixture.json", log=lambda *_:None)
            run.write("02_world.json", world.to_dict())
        stages = [Stage("input", [], lambda r:r.write("00_input.json", {}), "00_input.json"),
            Stage("whitepaper", ["input"], lambda r:None, "01_whitepaper.json"),
            Stage("world", ["whitepaper"], build, "02_world.json"),
            Stage("well_posed", ["world"], lambda r:None, "03_orders.json")]
        with patch.object(factory, "STAGES", stages), patch.object(factory, "generation_stages", return_value=stages):
            result = production.produce(run, to_stage="world")
        self.assertEqual(result["round"], 2)
        self.assertEqual(result["status"], "paused_at_stage")
        self.assertEqual(run.read("01_joint_design_audit.json")["used_revisions"], 1)
        self.assertEqual(run.manifest["stages"]["world"]["status"], "succeeded")
        archived = run.dir/"production/round-001/02_world_agent_fixture.json"
        self.assertEqual(json.loads(archived.read_text())["status"], "capacity_review_requested")
        compiler_failure = result["history"][0]["capacity_deficit"]["findings"][0]
        self.assertEqual(compiler_failure["code"], "fixed_layout_values_unresolved")
        feedback = trace.calls[3]["input"]
        self.assertIn("world_layout_unresolved", str(feedback))
        self.assertIn("compiler_issues", str(feedback))
        self.assertEqual(trace.script, [])

    @unittest.skipUnless((Path(__file__).resolve().parent / "fixtures" / 'showcase_v4_initial_author_failure.json.gz').is_file(), "Local historical experiment fixture is not distributed")
    def test_real_v4_replay_preserves_ready_sibling_and_routes_repeated_initial_values(self):
        fixture = json.loads(gzip.decompress((ROOT / "tests/fixtures/showcase_v4_initial_author_failure.json.gz").read_bytes()))
        self.wp = deepcopy(fixture["whitepaper"])
        state = deepcopy(fixture["checkpoint"])
        # Test-only migration of the old input schema into the new protocol;
        # original author replies, facts and failures remain unchanged.
        state["binding"] = world_agent._binding(self.wp, None)
        for row in state["log"]:
            if "task" in row.get("input", {}):
                row["input"]["response_schema"] = executor.response_contract(row["input"]["task"])
        world_agent._save(self.path, state)
        trace = ScriptedTracer([])
        with patch.object(config, "LLM_CONCURRENCY", 16):
            with self.assertRaises(CapacityReviewRequested) as caught:
                self.generate(trace)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(trace.calls, [])
        self.assertEqual({r["unit_id"] for r in saved["units"]}, {"U1_mainline_false_registry", "U2_forgery_provenance"})
        finding = caught.exception.evidence["findings"][0]
        self.assertEqual(finding["affected_ids"], ["U3_missing_clerk_ink_contract"])
        self.assertIn("ev_ink_custody_s4", str(finding["evidence"]["compiler_issues"]))


if __name__ == "__main__": unittest.main()
