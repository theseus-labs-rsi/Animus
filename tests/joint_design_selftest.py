"""Offline transaction gates for the v6 unpublished joint design."""
import os
os.environ.setdefault("OPENAI_API_KEY", "offline-test")
os.environ.setdefault("MODEL", "offline-model")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from copy import deepcopy
from pathlib import Path
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import config
from pipeline import joint_design, factory
from pipeline.capability_contract import digest
from pipeline.instance_plan import PlanConflict
from pipeline.lines.L3_process import ProcessLine
from pipeline.lines.L6_refusal import RefusalLine
from instance_plan_selftest import ACCEPT, setup_plan
from world_agent_selftest import ScriptedTracer
from seed_contract_selftest import fixture as seed_fixture, blueprint as seed_blueprint, CouncilTracer
from pipeline.seed_pack import seed_input
from pipeline.central_office import central_office


class JointDesignTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Offline test forbids network"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, self.plan, _, _, _ = setup_plan()
        self.wp["_council_views"] = {"world_author_attempts": 1}

    def run_joint(self, replies):
        tracer = ScriptedTracer(replies)
        run = SimpleNamespace(tracer=tracer)
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        checkpoints = []
        accepted, receipt = joint_design.accept(run, self.wp, None, None,
                                                 audit, lambda row: checkpoints.append(row))
        return accepted, receipt, audit, tracer, checkpoints

    def test_initial_design_is_published_only_after_full_compiler_and_review(self):
        accepted, receipt, audit, tracer, checkpoints = self.run_joint([
            ("council.instance_plan", self.plan),
            ("council.blueprint_feasibility", ACCEPT)])
        self.assertEqual(receipt["status"], "ready_for_world_probe")
        self.assertEqual(receipt["whitepaper_hash"], digest(accepted))
        self.assertEqual(receipt["checks"]["actual_source_supply"], "blocked_by_world")
        self.assertEqual(audit["used_revisions"], 0)
        self.assertEqual(audit["checks"][0]["compiler"], "passed")
        self.assertEqual([row["step"] for row in audit["calls"]], ["council.instance_plan"])
        self.assertEqual(checkpoints[-1]["status"], "accepted")
        self.assertEqual(len(tracer.calls), 2)

    def test_fixed_initial_proposal_still_gets_full_review_without_author_call(self):
        tracer = ScriptedTracer([("council.blueprint_feasibility", ACCEPT)])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        accepted, receipt = joint_design.accept(
            SimpleNamespace(tracer=tracer), self.wp, None, None, audit,
            lambda _row: None, initial_proposal=self.plan)
        self.assertEqual(audit["fixture_initial_proposal_hash"], digest(self.plan))
        self.assertEqual(audit["calls"], [])
        self.assertEqual(audit["checks"][0]["compiler"], "passed")
        self.assertEqual(receipt["whitepaper_hash"], digest(accepted))
        self.assertEqual(len(tracer.calls), 1)

    def test_unambiguous_session_field_patch_expands_to_full_record(self):
        event = self.plan["units"][0]["events"][-1]
        revision = {"decision": "repair", "changes": [{
            "unit_id": self.plan["units"][0]["unit_id"], "collection": "events",
            "operation": "replace", "selector": f"events[slot={event['slot']}].session",
            "value": event["session"] + 1}]}
        result, evidence = joint_design.normalize_record_field_edits(self.plan, revision)
        self.assertEqual(result["changes"][0]["selector"], {"slot": event["slot"]})
        self.assertEqual(result["changes"][0]["value"]["session"], event["session"] + 1)
        self.assertEqual(revision["changes"][0]["value"], event["session"] + 1)
        self.assertEqual(len(evidence), 1)
        duplicate = deepcopy(self.plan)
        duplicate["units"][0]["events"].append(deepcopy(event))
        with self.assertRaises(PlanConflict):
            joint_design.normalize_record_field_edits(duplicate, revision)

    def test_historical_joint_patch_cohort_shorthand_reaches_full_compiler(self):
        root = Path(__file__).resolve().parents[1] / "output/fresh_plan_failure_after_escalation_v5_20260928"
        if not root.exists():
            self.skipTest("Optional immutable historical artifacts")
        from tools.run_design_fixture_cell import load_fixture
        from pipeline.layout_revision import _business_protocol_errors, apply
        wp, proposal, _ = load_fixture(root)
        revision = {"decision": "repair", "reason": "Move one trend requirement to a natural comparison",
            "field_aliases": [], "blueprint_additions": {},
            "changes": [{"unit_id": "u_tower_case", "collection": "obligations",
                "operation": "replace", "selector": {"id": "L7-2"},
                "value": {"id": "L7-2", "line": "L7_consolidation", "subtype": "compare",
                    "carrier": {"entities": ["inv_a", "inv_d"], "field": "证据完备程度"}}}],
            "cohort_changes": [{"operation": "add", "cohort": {
                "id": "cohort_investigations", "basis": "All investigations by type",
                "members": ["inv_a", "inv_b", "inv_c", "inv_d"]}}]}
        fixed, evidence = joint_design.normalize_record_field_edits(proposal, revision)
        self.assertEqual(fixed["cohort_changes"][0]["key"], "cohort_investigations")
        self.assertEqual(fixed["cohort_changes"][0]["value"]["members"],
                         ["inv_a", "inv_b", "inv_c", "inv_d"])
        self.assertNotIn("cohort", fixed["cohort_changes"][0])
        self.assertEqual(evidence[0]["collection"], "cohorts")
        self.assertEqual(_business_protocol_errors(wp, fixed), [])
        with self.assertRaises(PlanConflict) as error:
            apply(wp, proposal, fixed)
        self.assertEqual({row["code"] for row in error.exception.findings}, {"cohort_layout"})

    def test_bad_patch_wire_feedback_identifies_transaction_and_preserves_candidate(self):
        from pipeline.layout_revision import _business_protocol_errors
        bad = {"decision": "repair", "reason": "Wrong inference", "changes": [{
            "unit_id": "report-case", "collection": "objects", "operation": "replace",
            "selector": {"slot": "report-slot"},
            "value": {"slot": "report-slot", "type": "report", "key": "invented"}}],
            "cohort_changes": []}
        errors = _business_protocol_errors(self.wp, bad)
        self.assertTrue(any(row["path"] == "changes[0].value" and
                            "Unsupported record fields: key" in row["message"] for row in errors))
        original = {"code": "cohort_layout", "affected_ids": ["cohort_council_devices"]}
        audit = {"checks": [], "actions": [{"decision": "patch", "status": "rejected",
            "reason": "Invented key", "findings": []}],
            "calls": [{"step": "council.joint_design_revision", "response": {
                "decision": "patch", "revision": bad}}], "used_revisions": 1,
            "open_design_findings": [original]}
        feedback = joint_design._feedback(self.wp, self.plan, [], audit)
        self.assertIn("revision_transaction", feedback["feedback_origin"])
        self.assertEqual(feedback["last_rejected_revision"], bad)
        self.assertEqual(feedback["open_design_findings"], [original])
        self.assertEqual(feedback["declared_object_minima"], {"company": 1, "report": 1})
        self.assertEqual(feedback["allocated_object_counts"], {"company": 1, "report": 1})

    def test_rejected_schedule_uses_one_shared_patch_and_recompiles_everything(self):
        broken = deepcopy(self.plan)
        broken["units"][0]["events"][-1]["session"] = 4
        repair = {"decision": "patch", "reason": "Restore the causal review delay",
                  "revision": {"decision": "repair", "reason": "The review follows revision by one period",
                               "changes": [{"unit_id": "report-case", "collection": "events",
                                            "operation": "replace", "key": "review-three",
                                            "value": deepcopy(self.plan["units"][0]["events"][-1])}],
                               "cohort_changes": []}}
        accepted, receipt, audit, tracer, _ = self.run_joint([
            ("council.instance_plan", broken),
            ("council.joint_design_revision", repair),
            ("council.blueprint_feasibility", ACCEPT)])
        self.assertEqual(accepted["business_instance_plan"]["units"][0]["events"][-1]["session"], 3)
        self.assertEqual(audit["used_revisions"], 1)
        self.assertEqual([row["compiler"] for row in audit["checks"]], ["rejected", "passed"])
        self.assertEqual(receipt["business_instance_plan_hash"],
                         digest(accepted["business_instance_plan"]))
        self.assertEqual(len(tracer.calls), 3)

    def test_two_rejected_atomic_patches_trigger_a_full_plan_rewrite(self):
        broken = deepcopy(self.plan)
        broken["units"][0]["events"][-1]["session"] = 4
        wrong = deepcopy(self.plan["units"][0]["events"][-1])
        wrong["slot"] = "wrong-identity"
        invalid = {"decision": "patch", "reason": "Attempted event repair",
            "revision": {"decision": "repair", "reason": "Attempted event repair",
                "changes": [{"unit_id": "report-case", "collection": "events",
                    "operation": "replace", "key": "review-three", "value": wrong}],
                "cohort_changes": []}}
        accepted, _, audit, tracer, _ = self.run_joint([
            ("council.instance_plan", broken),
            ("council.joint_design_revision", invalid),
            ("council.joint_design_revision", invalid),
            ("council.instance_plan_replacement", self.plan),
            ("council.blueprint_feasibility", ACCEPT)])
        self.assertEqual(audit["used_revisions"], 3)
        self.assertEqual([row["status"] for row in audit["actions"]],
                         ["rejected", "rejected", "applied"])
        self.assertEqual(audit["actions"][-1]["source"],
                         "controller_rejected_patch_fallback")
        self.assertEqual([row["step"] for row in audit["calls"]],
                         ["council.instance_plan", "council.joint_design_revision",
                          "council.joint_design_revision", "council.instance_plan_replacement"])
        self.assertEqual(accepted["business_instance_plan"]["units"][0]["events"][-1]["session"], 3)
        self.assertEqual(len(tracer.calls), 5)

    def test_repeated_complete_candidate_stops_without_another_model_call(self):
        broken = deepcopy(self.plan)
        broken["units"][0]["events"][-1]["session"] = 4
        action = {"decision": "replace_plan", "reason": "Try rebuilding the same plan"}
        tracer = ScriptedTracer([("council.instance_plan", broken),
                                ("council.joint_design_revision", action),
                                ("council.instance_plan_replacement", broken)])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with self.assertRaises(PlanConflict) as error:
            joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None,
                                audit, lambda _row: None)
        self.assertEqual(error.exception.findings[0]["code"], "joint_cycle")
        self.assertEqual(len(tracer.calls), 3)

    def test_materialized_shortfall_spends_the_same_remaining_revision_budget(self):
        accepted, receipt, audit, _, _ = self.run_joint([
            ("council.instance_plan", self.plan),
            ("council.blueprint_feasibility", ACCEPT)])
        changed = deepcopy(self.plan["units"][0]["observations"][0])
        changed["sessions"] = [0, 1, 2, 3, 4]
        patch_reply = {"decision": "patch", "reason": "Change the observation schedule",
                       "revision": {"decision": "repair", "reason": "Observe a sufficient revised range",
                                    "changes": [{"unit_id": "report-case", "collection": "observations",
                                                 "operation": "replace", "selector": {"entity": "report-slot",
                                                                                   "field": "capital"},
                                                 "value": changed}], "cohort_changes": []}}
        files = {"01_whitepaper.json": accepted}
        run = SimpleNamespace(tracer=ScriptedTracer([
            ("council.joint_design_revision", patch_reply),
            ("council.blueprint_feasibility", ACCEPT)]),
            read=lambda name: files[name])
        second, second_receipt = joint_design.accept(
            run, audit["draft"], None, None, audit, lambda _row: None,
            source_findings=[{"code": "actual_supply_shortfall", "kind": "design",
                              "message": "One original line is short", "affected_ids": ["L7_consolidation"]}])
        self.assertNotEqual(receipt["whitepaper_hash"], second_receipt["whitepaper_hash"])
        self.assertEqual(second_receipt["used_revisions"], 1)
        self.assertEqual(second["business_instance_plan"]["units"][0]["observations"][0]["sessions"],
                         [0, 1, 2, 3, 4])

    def test_l3_reads_do_not_invent_event_writes(self):
        bp = {"entity_types": [{"id": "case", "fields": [{"name": "status", "kind": "category"}]}],
              "event_types": [{"id": "change", "roles": {"subject": "case"},
                               "effect_fields": [{"role": "subject", "field": "status"}]}]}
        issues = ProcessLine().construction_issues(
            {"entities": ["case_a"], "field": "status"}, bp,
            {"case_a": {"type": "case"}}, {},
            [{"entity": "case_a", "field": "status", "sessions": [0, 1, 2, 3]}])
        self.assertEqual(issues[0]["code"], "process_periods")

    def test_l6_absent_focus_target_is_not_a_static_error(self):
        bp = {"entity_types": [{"id": "person", "fields": [{"name": "name"}]},
                               {"id": "company", "fields": [{"name": "revenue"}]}]}
        line = RefusalLine()
        self.assertEqual(line.carrier_field_issues(
            {"entities": ["person_a"], "field": "revenue"}, bp,
            {"person_a": {"type": "person"}}), [])
        self.assertEqual(line.carrier_field_issues(
            {"entities": ["person_a"], "field": "undeclared"}, bp,
            {"person_a": {"type": "person"}})[0]["code"], "undeclared_carrier_field")

    def test_downstream_requires_matching_design_and_real_supply_receipts(self):
        wp = deepcopy(self.wp)
        wp["business_instance_plan"] = self.plan
        world = {"entities": []}
        actual = {"passed": True}
        records = {
            "01_joint_design_receipt.json": {"status": "ready_for_world_probe",
                "whitepaper_hash": digest(wp), "supply_plan_hash": digest(wp["supply_plan"]),
                "business_instance_plan_hash": digest(self.plan)},
            "02_source_supply_gate.json": {"status": "source_supply_ready",
                "whitepaper_hash": digest(wp), "world_hash": digest(world),
                "actual_supply_hash": digest(actual)},
            "02_instance_fulfillment.json": actual}
        run = SimpleNamespace(manifest={"config": {"production_control": {"version": "supply-driven/v6"}}},
                              has=lambda key: key in records, read=lambda key: records[key])
        factory._require_v6_supply_gate(run, wp, world)
        records["02_instance_fulfillment.json"] = {"passed": False}
        with self.assertRaisesRegex(ValueError, "stale or changed"):
            factory._require_v6_supply_gate(run, wp, world)

    def test_world_blueprint_review_returns_to_shared_design_controller(self):
        from pipeline.blueprint_feasibility import BlueprintReviewRequested
        from pipeline.supply_capacity import CapacityReviewRequested
        run = SimpleNamespace(manifest={"config": {"production_control": {"version": "supply-driven/v6"}}},
                              read=lambda _name: self.wp, has=lambda _name: False)
        with patch("pipeline.production.ensure_instance_plan"), patch.object(
                factory, "_stage_world_once", side_effect=BlueprintReviewRequested(
                    {"reason": "A world field cannot be authored"})):
            with self.assertRaises(CapacityReviewRequested) as error:
                factory.stage_world(run)
        self.assertEqual(error.exception.evidence["phase"], "world_blueprint_review")
        self.assertEqual(error.exception.evidence["blueprint_review"]["reason"],
                         "A world field cannot be authored")

    def test_schema_replacement_reruns_mapping_without_old_business_opinion(self):
        pack = seed_fixture()
        desc, few = seed_input(pack)
        tracer = CouncilTracer(seed_blueprint(pack))
        first = central_office(desc, few, tracer, log=lambda *_: None, seed_pack=pack,
                               max_world_attempts=1, defer_business_review=True)
        second = central_office(desc, few, tracer, log=lambda *_: None, seed_pack=pack,
                                max_world_attempts=1, revision_of=first,
                                design_feedback={"finding": "Replace uncommitted schema"},
                                defer_business_review=True)
        steps = [step for step, _ in tracer.calls]
        self.assertEqual(steps.count("council.map"), 2)
        self.assertEqual(steps.count("council.world_repair"), 1)
        self.assertNotIn("council.blueprint_feasibility", steps)
        self.assertEqual(second["_council_views"]["world_author_attempts"], 2)

    def test_all_recorded_failed_drafts_still_face_the_full_compiler(self):
        root = Path(__file__).resolve().parents[1] / "output/fresh_plan_failure_after_escalation_v5_20260928"
        if not root.exists():
            self.skipTest("Optional immutable historical artifacts")
        import json
        from pipeline.instance_plan import compile_plan
        wp = json.loads((root / "01_whitepaper.json").read_text(encoding="utf-8"))
        recovery = json.loads((root / "01_initial_plan_recovery.json").read_text(encoding="utf-8"))
        expected = ["relation_schedule", "record_shape", "native_trend_temporal_capacity",
                    "state_write_capacity", "native_trend_temporal_capacity"]
        for draft, code in zip(recovery["transaction"]["drafts"], expected, strict=True):
            with self.assertRaises(PlanConflict) as error:
                compile_plan(wp, draft["candidate"])
            self.assertIn(code, [row["code"] for row in error.exception.findings])

    def test_e1_fixture_selects_fifth_complete_draft_with_source_hashes(self):
        root = Path(__file__).resolve().parents[1] / "output/fresh_plan_failure_after_escalation_v5_20260928"
        if not root.exists():
            self.skipTest("Optional immutable historical artifacts")
        from tools.run_design_fixture_cell import load_fixture
        from pipeline.instance_plan import compile_plan
        wp, proposal, receipt = load_fixture(root)
        self.assertEqual(receipt["historical_draft_count"], 5)
        self.assertEqual(receipt["fifth_candidate_hash"], digest(proposal))
        self.assertEqual(len(receipt["whitepaper_sha256"]), 64)
        with self.assertRaises(PlanConflict) as error:
            compile_plan(wp, proposal)
        self.assertIn("native_trend_temporal_capacity",
                      [row["code"] for row in error.exception.findings])
        self.assertIn("cohort_layout",
                      [row["code"] for row in error.exception.findings])
        feedback = joint_design._feedback(wp, proposal, error.exception.findings,
                                          {"checks": [], "actions": [], "used_revisions": 0})
        self.assertEqual(feedback["affected_record_owners"]["L7-2"],
                         [{"unit_id": "u_tower_case", "collection": "obligations"}])

    def test_curated_seed_projection_restores_required_structure_without_new_facts(self):
        root = Path(__file__).resolve().parents[1] / "output/fresh_plan_failure_after_escalation_v5_20260928"
        if not root.exists():
            self.skipTest("Optional immutable historical artifacts")
        import json
        from pipeline.seed_pack import project_required_blueprint, validate_seed_blueprint
        seed = json.loads((root / "00_seed_pack.json").read_text(encoding="utf-8"))
        calls = [json.loads(row) for row in (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                 if row.strip()]
        source = calls[5]["output"]
        frozen = deepcopy(source)
        projected, edits = project_required_blueprint(source, seed)
        self.assertTrue(edits)
        self.assertEqual(source, frozen)
        self.assertTrue(validate_seed_blueprint(projected, seed)["passed"])


if __name__ == "__main__":
    unittest.main()
