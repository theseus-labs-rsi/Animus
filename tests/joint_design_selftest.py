"""Offline transaction gates for the v6 unpublished joint design."""
import os
os.environ.setdefault("OPENAI_API_KEY", "offline-test")
os.environ.setdefault("MODEL", "offline-model")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from copy import deepcopy
from pathlib import Path
import json
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import config
from pipeline import joint_design, factory
from pipeline.capability_contract import digest
from pipeline.instance_plan import PlanConflict
from pipeline.blueprint_feasibility import BlueprintReviewProtocolError
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

    def test_unresolved_business_review_returns_feedback_for_bounded_revision(self):
        audit = {"candidate_hashes": [], "checks": []}
        opinion = {"decision": "unresolved", "input_hash": "fixture-input",
                   "review": {"reason": "The business timeline is uncertain"}}
        with patch.object(joint_design, "_review", return_value=opinion):
            revised, findings = joint_design._candidate(
                self.wp, self.plan, None, audit, lambda _row: None)
        self.assertIsNone(revised)
        self.assertEqual(findings[0]["code"], "business_unresolved")
        self.assertEqual(audit["checks"][0]["business_review"], "unresolved")

    def test_revision_author_unresolved_uses_same_bounded_schema_redesign(self):
        accepted, _, audit, _, _ = self.run_joint([
            ("council.instance_plan", self.plan), ("council.blueprint_feasibility", ACCEPT)])
        diagnosis = {"decision":"unresolved", "reason":"More independent state carriers are needed"}
        changed = deepcopy(self.plan)
        changed["units"][0]["observations"][0]["sessions"] = [0,1,2,3,4]
        run = SimpleNamespace(read=lambda _name:accepted, tracer=ScriptedTracer([
            ("council.joint_design_revision", diagnosis),
            ("council.instance_plan_replacement", changed),
            ("council.blueprint_feasibility", ACCEPT)]))
        with patch.object(joint_design, "_replacement", return_value=deepcopy(self.wp)) as redesign:
            paper, receipt = joint_design.accept(run, audit["draft"], None, None, audit, lambda _row:None,
                source_findings=[{"code":"actual_supply_shortfall", "kind":"design"}])
        self.assertEqual(redesign.call_args.args[2]["revision_author_diagnosis"], diagnosis)
        self.assertEqual(audit["actions"][-1]["source"], "controller_revision_unresolved")
        self.assertEqual(receipt["used_revisions"], 1)
        self.assertEqual(receipt["status"], "ready_for_world_probe")

    def test_plan_unresolved_redesigns_from_initial_replacement_and_world_probe(self):
        diagnosis = {"decision": "unresolved", "reason": "Schema has too few independent carriers",
                     "constraint_conflicts": [{"line": "L7_consolidation", "required": 2,
                                               "available": 1}]}
        complete = deepcopy(self.plan)
        complete["reason"] = "Complete plan after the world architect reconsidered the layout"
        replace_plan = {"decision": "replace_plan", "reason": "Rebuild the instance layout"}
        for origin in ("initial", "replacement", "world_probe"):
            with self.subTest(origin=origin):
                source = None
                if origin == "world_probe":
                    accepted, _, audit, _, _ = self.run_joint([
                        ("council.instance_plan", self.plan),
                        ("council.blueprint_feasibility", ACCEPT)])
                    source = [{"code": "actual_supply_shortfall", "kind": "design",
                               "message": "Materialized independent carriers are short", "affected_ids": []}]
                    script = [("council.joint_design_revision", replace_plan),
                              ("council.instance_plan_replacement", diagnosis)]
                    run = SimpleNamespace(read=lambda name: accepted)
                else:
                    audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
                    run = SimpleNamespace()
                    if origin == "initial":
                        script = [("council.instance_plan", diagnosis)]
                    else:
                        broken = deepcopy(self.plan)
                        broken["units"][0]["events"][-1]["session"] = 4
                        script = [("council.instance_plan", broken),
                                  ("council.joint_design_revision", replace_plan),
                                  ("council.instance_plan_replacement", diagnosis)]
                run.tracer = ScriptedTracer(script + [
                    ("council.instance_plan_replacement", complete),
                    ("council.blueprint_feasibility", ACCEPT)])
                with patch.object(joint_design, "_replacement", return_value=deepcopy(self.wp)) as redesign, \
                     patch.object(joint_design, "compile_plan", wraps=joint_design.compile_plan) as compile_:
                    paper, receipt = joint_design.accept(run, self.wp, None, None, audit,
                        lambda _row: None, source_findings=source)
                redesign.assert_called_once()
                feedback = redesign.call_args.args[2]
                self.assertEqual(feedback["proposal"], diagnosis)
                self.assertEqual(feedback["current_findings"][0]["evidence"], diagnosis)
                self.assertEqual(feedback["check_coverage"]["compiler"], "not_run")
                self.assertEqual(feedback["check_coverage"]["business_review"], "not_run")
                self.assertIn("author claims", feedback["schema_redesign_instruction"])
                if origin == "initial":
                    self.assertIsNone(feedback["last_executed_check"])
                else:
                    self.assertEqual(feedback["last_executed_check"], audit["checks"][0])
                    self.assertEqual(feedback["last_executed_check"]["compiler"],
                                     "rejected" if origin == "replacement" else "passed")
                    if origin == "replacement":
                        self.assertEqual(feedback["last_executed_check"]["findings"],
                                         audit["checks"][0]["findings"])
                self.assertEqual(compile_.call_count, 2 if origin == "replacement" else 1)
                checked = audit["checks"][-2]
                self.assertEqual((checked["compiler"], checked["business_review"]), ("not_run", "not_run"))
                self.assertEqual(checked["findings"][0]["code"], "plan_unresolved")
                self.assertEqual(audit["actions"][-1]["source"], "controller_plan_unresolved")
                self.assertEqual(audit["used_revisions"], 1 if origin == "initial" else 2)
                self.assertEqual(audit["checks"][-1]["compiler"], "passed")
                self.assertEqual(audit["checks"][-1]["business_review"], "accept")
                self.assertEqual(receipt["status"], "ready_for_world_probe")
                self.assertTrue(paper["business_instance_plan"])
                self.assertEqual(run.tracer.script, [])

    def test_repeated_plan_unresolved_uses_shared_allowance_then_blueprint_fallback(self):
        diagnosis = {"decision": "unresolved", "reason": "Independent carrier conflict",
                     "constraint_conflicts": ["The required trajectories do not fit"]}
        for limit in (0, 2):
            with self.subTest(limit=limit):
                audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
                tracer = ScriptedTracer([("council.instance_plan", diagnosis)] +
                    [("council.instance_plan_replacement", diagnosis)] * limit)
                with patch.object(joint_design, "MAX_REVISIONS", limit), \
                     patch.object(joint_design, "_replacement", return_value=deepcopy(self.wp)) as redesign, \
                     patch.object(joint_design, "compile_plan") as compile_, \
                     patch.object(joint_design, "_review") as review:
                    paper, receipt = joint_design.accept(SimpleNamespace(tracer=tracer),
                        self.wp, None, None, audit, lambda _row: None, allow_degraded=True)
                self.assertEqual(redesign.call_count, limit)
                compile_.assert_not_called()
                review.assert_not_called()
                self.assertEqual(audit["used_revisions"], limit)
                self.assertEqual(len(audit["checks"]), limit + 1)
                self.assertEqual(audit["proposal"], diagnosis)
                self.assertEqual(receipt["unexecuted_proposal_hash"], digest(diagnosis))
                self.assertEqual(receipt["unresolved_findings"][0]["evidence"], diagnosis)
                self.assertEqual(receipt["status"], "exploratory_after_design_limit")
                self.assertFalse(receipt["release_eligible"])
                self.assertEqual(paper["world_blueprint"], self.wp["world_blueprint"])
                self.assertNotIn("business_instance_plan", paper)
                self.assertNotIn("instance_policy", paper["supply_plan"])
                self.assertEqual(tracer.script, [])

    def test_plan_unresolved_with_zero_allowance_obeys_strict_mode(self):
        tracer = ScriptedTracer([("council.instance_plan",
                                 {"decision": "unresolved", "reason": "Carrier shortage"})])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with patch.object(joint_design, "MAX_REVISIONS", 0), \
             patch.object(joint_design, "_replacement") as redesign:
            with self.assertRaises(PlanConflict) as error:
                joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None,
                    audit, lambda _row: None)
        redesign.assert_not_called()
        self.assertEqual(error.exception.findings[0]["code"], "joint_revisions_exhausted")
        self.assertEqual(audit["status"], "revisions_exhausted")

    def test_plan_transport_failure_keeps_original_schema_and_diagnostic(self):
        diagnosis = {"decision": "unresolved", "reason": "Expand the business carriers"}
        replacement = deepcopy(self.wp)
        replacement["world_blueprint"]["replacement_marker"] = True
        failure = config.ChatJSONError("Plan request exhausted its transport attempts",
            cause=config.CallDeadlineError("headers stalled", phase="awaiting_http_headers", deadline_s=600),
            attempts=5, token_cap=65536)
        tracer = ScriptedTracer([("council.instance_plan", diagnosis),
                                 ("council.instance_plan_replacement", failure)])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        checkpoints = []
        with patch.object(joint_design, "_replacement", return_value=replacement):
            with self.assertRaises(config.ChatJSONError) as error:
                joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None,
                    audit, lambda row: checkpoints.append(deepcopy(row)), allow_degraded=True)
        self.assertIs(error.exception, failure)
        self.assertEqual(audit["draft"], self.wp)
        self.assertEqual(audit["proposal"], diagnosis)
        self.assertEqual(audit["used_revisions"], 1)
        self.assertEqual(audit["actions"][-1]["status"], "call_failed")
        self.assertEqual(checkpoints[-1]["draft"], self.wp)
        self.assertEqual(checkpoints[-1]["proposal"], diagnosis)
        self.assertEqual(tracer.script, [])

    def test_tracer_timeout_sentinel_preserves_cause_and_stops_design_revisions(self):
        from execution_control import ExecutionStopped
        from pipeline.run import Tracer
        diagnosis = {"decision": "unresolved", "reason": "Expand the business carriers"}
        failure = {"__error__": "headers stalled",
                   "__error_metadata__": {"kind": "total_deadline", "attempts": 5, "call_id": "failed"}}
        replacement = deepcopy(self.wp)
        replacement["world_blueprint"]["replacement_marker"] = True
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(config, "chat_json", side_effect=[diagnosis, failure]) as provider, \
             patch.object(joint_design, "_replacement", return_value=replacement):
            tracer = Tracer(SimpleNamespace(dir=Path(temporary)))
            with self.assertRaises(ExecutionStopped) as error:
                joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None,
                    audit, lambda _row: None, allow_degraded=True)
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(error.exception.kind, "total_deadline")
        self.assertEqual(error.exception.raw, failure)
        self.assertEqual(audit["draft"], self.wp)
        self.assertEqual(audit["proposal"], diagnosis)
        self.assertEqual(audit["used_revisions"], 1)
        self.assertEqual(audit["actions"][-1]["status"], "call_failed")
        self.assertEqual(audit["calls"][-1]["status"], "provider_error")
        self.assertEqual(audit["calls"][-1]["response"], failure)
        self.assertEqual([call["step"] for call in audit["calls"]],
                         ["council.instance_plan", "council.instance_plan_replacement"])

    def test_world_redesign_council_timeout_keeps_failure_without_another_revision(self):
        from pipeline.central_office import CouncilExecutionError, _require_call_result
        diagnosis = {"decision": "unresolved", "reason": "Expand the business carriers"}
        failure = {"__error__": "headers stalled",
                   "__error_metadata__": {"kind": "total_deadline", "attempts": 5, "call_id": "world-failed"}}
        tracer = ScriptedTracer([("council.instance_plan", diagnosis)])
        run = SimpleNamespace(tracer=tracer, manifest={"config": {}}, log=lambda *_: None,
                              read=lambda name: {"description": "Original seed", "few_shot": []})
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}

        def fail_world_author(*args, **kwargs):
            return _require_call_result(failure, "council.world_repair")

        with patch("pipeline.central_office.central_office", side_effect=fail_world_author) as author:
            with self.assertRaises(CouncilExecutionError) as error:
                joint_design.accept(run, self.wp, None, None, audit, lambda _row: None,
                                    allow_degraded=True)
        author.assert_called_once()
        self.assertEqual(error.exception.step, "council.world_repair")
        self.assertEqual(error.exception.result, failure)
        self.assertEqual(audit["draft"], self.wp)
        self.assertEqual(audit["proposal"], diagnosis)
        self.assertEqual(audit["used_revisions"], 1)
        self.assertEqual(len(audit["actions"]), 1)
        self.assertEqual(audit["actions"][0]["status"], "call_failed")
        self.assertEqual(audit["actions"][0]["call_failure"], failure)
        self.assertEqual([call["step"] for call in tracer.calls], ["council.instance_plan"])

    def test_rejected_world_redesign_retries_against_original_diagnostic(self):
        diagnosis = {"decision": "unresolved", "reason": "Need another natural carrier",
                     "constraint_conflicts": [{"type": "report", "required": 2}]}
        rejected = PlanConflict([{"code": "schema_rejected", "message": "Replacement changed a seed role",
                                  "affected_ids": []}])
        with patch.object(joint_design, "_replacement", side_effect=[rejected, deepcopy(self.wp)]) as redesign:
            _, receipt, audit, tracer, _ = self.run_joint([
                ("council.instance_plan", diagnosis),
                ("council.instance_plan_replacement", self.plan),
                ("council.blueprint_feasibility", ACCEPT)])
        self.assertEqual(redesign.call_count, 2)
        retry = redesign.call_args_list[1]
        self.assertEqual(retry.args[1], self.wp)
        self.assertEqual(retry.args[2]["proposal"], diagnosis)
        self.assertEqual(retry.args[2]["current_findings"][0]["code"], "schema_rejected")
        self.assertEqual(retry.args[2]["open_design_findings"][0]["evidence"], diagnosis)
        self.assertEqual([row["decision"] for row in audit["actions"]], ["replace_draft", "replace_draft"])
        self.assertEqual([row["status"] for row in audit["actions"]], ["rejected", "applied"])
        self.assertEqual(receipt["used_revisions"], 2)
        self.assertEqual(tracer.script, [])

    def test_failed_plan_after_new_schema_keeps_the_prior_pair_for_next_redesign(self):
        diagnosis = {"decision": "unresolved", "reason": "Need a different business layout"}
        replacement = deepcopy(self.wp)
        replacement["world_blueprint"]["replacement_marker"] = "unfinished transaction"
        with patch.object(joint_design, "_replacement", side_effect=[replacement, deepcopy(self.wp)]) as redesign:
            _, receipt, audit, tracer, _ = self.run_joint([
                ("council.instance_plan", diagnosis),
                ("council.instance_plan_replacement", ValueError("New plan request was invalid")),
                ("council.instance_plan_replacement", self.plan),
                ("council.blueprint_feasibility", ACCEPT)])
        second = redesign.call_args_list[1]
        self.assertEqual(second.args[1], self.wp)
        self.assertEqual(second.args[2]["schema"], self.wp["world_blueprint"])
        self.assertEqual(second.args[2]["proposal"], diagnosis)
        self.assertEqual(audit["actions"][0]["before_hash"], audit["actions"][1]["before_hash"])
        self.assertEqual([row["status"] for row in audit["actions"]], ["rejected", "applied"])
        self.assertEqual(receipt["used_revisions"], 2)
        self.assertEqual(tracer.script, [])

    def test_schema_replacement_rebinds_original_contracts_target_and_limits(self):
        original = deepcopy(self.wp)
        for field in ("world_generation", "generation_contract", "quality_contract"):
            original[field] = {"preserved": field}
        replacement = deepcopy(original)
        replacement["delivery_target"] = {"candidate_questions": 9999}
        replacement["supply_plan"] = {"world_limits": {"max_world_entities": 9999}}
        cfg = {"delivery_target": original["delivery_target"], "question_budget": 1,
               "single_pass_world_limits": original["supply_plan"]["world_limits"]}
        run = SimpleNamespace(read=lambda name: {"description": "seed business", "few_shot": []},
                              manifest={"config": cfg}, tracer=object(), log=lambda *_: None)
        pack, feedback, brief = {"id": "original-seed"}, {"proposal": {"decision": "unresolved"}}, {"target": "frozen"}
        with patch("pipeline.central_office.central_office", return_value=replacement) as author, \
             patch.object(factory, "validate_seed_identity") as identity:
            result = joint_design._replacement(run, original, feedback, pack, brief)
        identity.assert_called_once_with(run, replacement)
        self.assertEqual(author.call_args.kwargs["seed_pack"], pack)
        self.assertEqual(author.call_args.kwargs["delivery_brief"], brief)
        self.assertEqual(author.call_args.kwargs["design_feedback"], feedback)
        self.assertEqual(result["seed_contract"], original["seed_contract"])
        self.assertEqual(result["delivery_target"], original["delivery_target"])
        self.assertEqual(result["supply_plan"]["requirements"], original["supply_plan"]["requirements"])
        self.assertEqual(result["supply_plan"]["world_limits"], original["supply_plan"]["world_limits"])
        for field in ("world_generation", "generation_contract", "quality_contract"):
            self.assertEqual(result[field], original[field])

    def test_exhausted_business_review_protocol_keeps_exploratory_blueprint(self):
        protocol = {"attempts": [{"errors": [{"path": "decision", "message": "invalid"}]}]}
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with patch.object(joint_design, "_review",
                          side_effect=BlueprintReviewProtocolError(protocol)):
            paper, receipt = joint_design.accept(
                SimpleNamespace(tracer=ScriptedTracer([])), self.wp, None, None,
                audit, lambda _row: None, initial_proposal=self.plan,
                allow_degraded=True)
        self.assertEqual(receipt["status"], "exploratory_after_review_limit")
        self.assertFalse(receipt["release_eligible"])
        self.assertEqual(receipt["used_revisions"], 0)
        self.assertEqual(receipt["unresolved_findings"][0]["code"],
                         "business_review_protocol_exhausted")
        self.assertEqual(audit["checks"][0]["business_review"], "protocol_exhausted")
        self.assertNotIn("business_instance_plan", paper)
        self.assertTrue(joint_design.exploratory_after_limit(
            SimpleNamespace(has=lambda _name: True, read=lambda _name: receipt), paper))

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

    def test_transport_resume_reissues_exact_failed_revision_without_resetting_allowance(self):
        finding = {"code": "joint_revision_wire", "affected_ids": [], "message": "Invalid edit"}
        key = digest({"schema": self.wp["world_blueprint"], "proposal": self.plan})
        prior_revision = {"decision": "repair", "changes": []}
        audit = {"version": joint_design.VERSION, "status": "designing", "draft": deepcopy(self.wp),
                 "proposal": deepcopy(self.plan), "used_revisions": 4,
                 "checks": [{"candidate_hash": key, "compiler": "rejected"}],
                 "candidate_hashes": [key], "reviews": [], "open_design_findings": [finding],
                 "actions": [{"decision": "patch", "status": "rejected", "before_hash": key,
                              "reason": "Invalid edit", "findings": [finding]}],
                 "calls": [{"step": "council.joint_design_revision", "status": "response_received",
                            "response": {"decision": "patch", "revision": prior_revision}}]}
        payload = joint_design._feedback(self.wp, self.plan, [finding], audit)
        audit["calls"].append({"step": "council.joint_design_revision", "status": "provider_error",
                               "request_hash": digest(payload), "response": {"__error__": "timeout"}})
        tracer = ScriptedTracer([("council.joint_design_revision",
                                  {"__error__": "Synthetic transport stop"})])
        from execution_control import ExecutionStopped
        with self.assertRaises(ExecutionStopped):
            joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None, audit,
                                lambda _row: None, resume_unpublished=True)
        self.assertEqual(audit["used_revisions"], 4)
        self.assertEqual(len(audit["checks"]), 1)
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(digest(tracer.calls[0]["input"]), digest(payload))

    def test_unpublished_factory_resume_requires_bound_explicit_permit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scenario = {"description": "offline", "few_shot": []}
            cfg = {"production_control": {"version": "supply-driven/v6"},
                   "delivery_target": "offline-target", "question_budget": 1}
            audit = {"version": joint_design.VERSION, "identity": digest({"input": scenario, "config": cfg}),
                     "status": "designing", "draft": self.wp}
            raw = json.dumps(audit).encode()
            (directory / "01_joint_design_audit.json").write_bytes(raw)
            writes = {}
            run = SimpleNamespace(run_id="offline-run", dir=directory, scenario="office",
                manifest={"config": cfg}, read=lambda name: scenario if name == "00_input.json" else audit,
                has=lambda name: name == "01_joint_design_audit.json",
                write=lambda name, value: writes.update({name: value}), set_algo=lambda **_kw: None)
            with patch.object(factory, "validate_seed_input", return_value=None):
                with self.assertRaisesRegex(ValueError, "explicit transport recovery"):
                    factory.stage_whitepaper(run)
                permit = {"run_id": run.run_id,
                          "joint_design_checkpoint_sha256": __import__("hashlib").sha256(raw).hexdigest()}
                with patch.object(factory, "JOINT_DESIGN_RECOVERY", permit), \
                     patch("pipeline.supply.author_brief", return_value={"candidate_allocation": {}}), \
                     patch("pipeline.supply_capacity.requirements", return_value={}), \
                     patch("pipeline.joint_design.accept", return_value=(self.wp, {"status": "ready"})) as accept, \
                     patch.object(factory, "validate_seed_identity"):
                    factory.stage_whitepaper(run)
                self.assertTrue(accept.call_args.kwargs["resume_unpublished"])
                self.assertIn("01_whitepaper.json", writes)

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

    def test_revision_limit_publishes_bound_blueprint_only_experiment(self):
        broken = deepcopy(self.plan)
        broken["units"][0]["events"][-1]["session"] = 4
        tracer = ScriptedTracer([("council.instance_plan", broken)])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with patch.object(joint_design, "MAX_REVISIONS", 0):
            paper, receipt = joint_design.accept(
                SimpleNamespace(tracer=tracer), self.wp, None, None, audit,
                lambda _row: None, allow_degraded=True)
        self.assertEqual(audit["status"], "degraded")
        self.assertEqual(receipt["status"], "exploratory_after_design_limit")
        self.assertFalse(receipt["release_eligible"])
        self.assertEqual(receipt["whitepaper_hash"], digest(paper))
        self.assertNotIn("business_instance_plan", paper)
        self.assertNotIn("instance_policy", paper["supply_plan"])
        self.assertTrue(receipt["unresolved_findings"])
        self.assertEqual(len(tracer.calls), 1)

    def test_repeated_complete_candidate_returns_to_structure_without_duplicate_review(self):
        broken = deepcopy(self.plan)
        broken["units"][0]["events"][-1]["session"] = 4
        action = {"decision": "replace_plan", "reason": "Try rebuilding the same plan"}
        tracer = ScriptedTracer([("council.instance_plan", broken),
                                ("council.joint_design_revision", action),
                                ("council.instance_plan_replacement", broken),
                                ("council.instance_plan_replacement", self.plan),
                                ("council.blueprint_feasibility", ACCEPT)])
        audit = {"version": joint_design.VERSION, "status": "initial_draft_ready"}
        with patch.object(joint_design, "_replacement", return_value=deepcopy(self.wp)) as redesign:
            _, receipt = joint_design.accept(SimpleNamespace(tracer=tracer), self.wp, None, None,
                                            audit, lambda _row: None)
        self.assertEqual(redesign.call_args.args[2]["current_findings"][0]["code"], "joint_cycle")
        self.assertEqual(receipt["used_revisions"], 2)
        self.assertEqual(len(audit["checks"]), 2)
        self.assertEqual(len(tracer.calls), 5)

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

    def test_world_review_limit_allows_only_bound_exploratory_supply(self):
        from pipeline import world_semantics
        wp = deepcopy(self.wp)
        wp["business_instance_plan"] = self.plan
        world = {"entities": []}
        actual = {"passed": True}
        review = {"status": "error"}
        warning = {"release_eligible": False, "binding": "current-world"}
        records = {
            "00_input.json": {"description": "offline"},
            "01_joint_design_audit.json": {"used_revisions": 11},
            "01_joint_design_receipt.json": {"status": "ready_for_world_probe",
                "whitepaper_hash": digest(wp)},
            "02_instance_fulfillment.json": actual,
            world_semantics.REVIEW_ARTIFACT: review,
            world_semantics.WARNING_ARTIFACT: warning,
            "02_source_supply_gate.json": {
                "status": "exploratory_source_observed", "release_eligible": False,
                "whitepaper_hash": digest(wp), "world_hash": digest(world),
                "actual_supply_hash": digest(actual), "world_review_warning": True,
                "world_review_warning_hash": digest(warning)},
        }
        run = SimpleNamespace(
            manifest={"config": {"production_control": {"version": "supply-driven/v6"}}},
            has=lambda name: name in records, read=lambda name: records[name])
        with patch.object(factory.WorldState, "from_dict", return_value=object()), patch.object(
                world_semantics, "validate_generation_warning", return_value=[]):
            factory._require_v6_supply_gate(run, wp, world)
            records["02_source_supply_gate.json"]["world_review_warning_hash"] = "stale"
            with self.assertRaisesRegex(ValueError, "stale or changed"):
                factory._require_v6_supply_gate(run, wp, world)
            records["02_source_supply_gate.json"]["world_review_warning_hash"] = digest(warning)
            records.pop(world_semantics.WARNING_ARTIFACT)
            with self.assertRaisesRegex(ValueError, "stale or changed"):
                factory._require_v6_supply_gate(run, wp, world)

    def test_exploratory_source_and_order_shortfall_continue_without_certification(self):
        from pipeline import production, supply_capacity
        paper = deepcopy(self.wp)
        paper["supply_plan"].pop("instance_policy", None)
        paper["supply_plan"]["candidate_allocation"] = {"L3_process": 2}
        world = {"entities": []}
        records = {
            "01_whitepaper.json": paper, "02_world.json": world,
            "03_orders.json": [{"line": "L3_process"}],
            "01_joint_design_receipt.json": {
                "status": "exploratory_after_design_limit", "fallback": "blueprint_only",
                "release_eligible": False, "whitepaper_hash": digest(paper),
                "supply_plan_hash": digest(paper["supply_plan"]),
                "schema_hash": digest(paper["world_blueprint"])},
            "02_source_supply_gate.json": {"status": "exploratory_source_observed",
                "release_eligible": False, "whitepaper_hash": digest(paper),
                "world_hash": digest(world), "actual_supply_hash": None},
        }
        run = SimpleNamespace(
            manifest={"config": {"production_control": {"version": "supply-driven/v6"}}},
            has=lambda name: name in records, read=lambda name: records[name],
            write=lambda name, value: records.__setitem__(name, value))
        self.assertTrue(joint_design.exploratory_after_limit(run, paper))
        production.ensure_instance_plan(run)
        factory._require_v6_supply_gate(run, paper, world)
        supply_capacity.require_order_capacity(run)
        self.assertFalse(records["03_capacity_gate.json"]["passed"])
        self.assertFalse(records["03_capacity_gate.json"]["release_eligible"])
        records["02_source_supply_gate.json"]["world_hash"] = "tampered"
        with self.assertRaisesRegex(ValueError, "stale or changed"):
            factory._require_v6_supply_gate(run, paper, world)

    def test_exploratory_world_commits_a_degraded_source_receipt_without_plan(self):
        paper = deepcopy(self.wp)
        paper.pop("seed_contract", None)
        paper["supply_plan"].pop("instance_policy", None)
        records = {
            "01_whitepaper.json": paper,
            "01_joint_design_audit.json": {"used_revisions": joint_design.MAX_REVISIONS},
            "01_joint_design_receipt.json": {
                "status": "exploratory_after_design_limit", "fallback": "blueprint_only",
                "release_eligible": False, "whitepaper_hash": digest(paper),
                "supply_plan_hash": digest(paper["supply_plan"]),
                "schema_hash": digest(paper["world_blueprint"]),
                "unresolved_findings": [{"code": "unfilled_carrier"}]},
        }
        world = SimpleNamespace(entities={}, n_sessions=6, to_dict=lambda: {"entities": []})
        run = SimpleNamespace(
            scenario="synthetic", dir=Path("."), tracer=None,
            manifest={"config": {"production_control": {"version": "supply-driven/v6"}}},
            has=lambda name: name in records, read=lambda name: records[name],
            log=lambda *_args: None)
        published = {}
        with patch.object(factory, "validate_seed_identity"), patch.object(
                factory, "validate_seed_world", return_value={"passed": True}), patch.object(
                factory, "build_world", return_value=world), patch.object(
                factory, "_prepare_lines"), patch.object(
                factory, "_publish_world_bundle", side_effect=lambda _run, bundle, *_args, **_kwargs:
                published.update(bundle)), patch("pipeline.world_semantics.enabled", return_value=False), patch(
                "pipeline.disclosure.enabled", return_value=False):
            factory._stage_world_once(run)
        self.assertEqual(published["02_source_supply_gate.json"]["status"],
                         "exploratory_source_observed")
        self.assertFalse(published["02_source_supply_gate.json"]["release_eligible"])
        self.assertEqual(published["02_source_supply_gate.json"]["unresolved_design_findings"],
                         [{"code": "unfilled_carrier"}])

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
