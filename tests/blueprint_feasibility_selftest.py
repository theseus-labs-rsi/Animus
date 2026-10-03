"""Offline upstream routing, constraint preservation and real state validation."""
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
from pipeline import blueprint_feasibility as review, factory
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, validate
from pipeline.world_agent import generate_world, _context, _index, _unit_issues
from seed_world_selftest import fixture
from world_agent_factory_selftest import LocalRun
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, write, table_part


def opinion(wp, decision="accept"):
    return {"decision": decision, "reason": "Check the complete revision/review cycle",
            "mechanism_checks": [{"mechanism_id": m["id"], "walkthrough": "Confirmed record is revised, reviewed and confirmed again",
                                  "status": "feasible" if decision == "accept" else "blocked"}
                                 for m in wp["seed_contract"]["mechanisms"]],
            "issues": [] if decision == "accept" else [{"finding": "One-way declaration blocks required recheck",
                                                       "suggestion": "Use the existing reversible status field"}]}


class Tracer:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "payload": json.loads(messages[1]["content"]), "params": params})
        if not self.answers:
            raise AssertionError("Unexpected model call: " + step)
        return deepcopy(self.answers.pop(0))


class Tests(unittest.TestCase):
    def setUp(self):
        self.wp, self.world, self.table = fixture()
        self.field = next(f for t in self.wp["world_blueprint"]["entity_types"] for f in t["fields"] if f["name"] == "review")
        self.field["states"] = ["draft", "confirmed", "recheck"]
        self.change = {"reason": "Task needs confirmation after recheck", "edits": [
            {"entity_type": "report", "field": "review", "remove": ["states"], "set": {},
             "reason": "Reversible operational status, not an irreversible lifecycle"}]}
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network"))
            guard.start(); self.addCleanup(guard.stop)

    def test_incomplete_or_contradictory_acceptance_cannot_pass(self):
        for change in (lambda r: r.update(mechanism_checks=[]), lambda r: r["mechanism_checks"][0].update(status="blocked")):
            raw = opinion(self.wp); change(raw)
            with self.assertRaises(WorldBlueprintError):
                review.assess(self.wp, Tracer([raw,raw,raw]))

    def test_negative_decision_is_preserved_and_not_retried(self):
        tracer = Tracer([opinion(self.wp, "repair")])
        result = review.assess(self.wp, tracer)
        self.assertEqual(result["decision"], "repair")
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(tracer.calls[0]["params"]["response_format"], {"type": "json_object"})
        self.assertIn("states", json.dumps(result["input"]))

    def test_causal_review_contract_matches_actual_optional_trigger(self):
        from pipeline.world_state import assemble_world, causal_execution_contract
        wp, _world, table = fixture()
        before = deepcopy(table)
        _actual, issues = assemble_world(table,blueprint=wp["world_blueprint"],include_shape_diagnostics=False)
        self.assertEqual(issues,[])
        contract = causal_execution_contract(wp["world_blueprint"],table["events"])
        self.assertFalse(contract["every_trigger_instance_requires_effect"])
        self.assertFalse(contract["automatic_effect_event_generation"])
        self.assertEqual(contract["complete_world_minimum_witnesses_per_rule"],1)
        self.assertTrue(all(contract["candidate_witnesses"].values()))
        linked = {e.get("caused_by") for e in table["events"]}
        self.assertTrue(any(e["type"]==wp["world_blueprint"]["causal_rules"][0]["trigger_event"] and e["id"] not in linked for e in table["events"]))
        self.assertEqual(table,before)

    def test_wrong_declared_edge_and_missing_rule_witness_still_fail(self):
        from pipeline.world_state import assemble_world, causal_execution_contract
        wp, _world, table = fixture()
        child = next(e for e in table["events"] if e.get("caused_by"))
        child["session"] += 1
        contract = causal_execution_contract(wp["world_blueprint"],table["events"])
        self.assertFalse(any(contract["candidate_witnesses"].values()))
        _actual, issues = assemble_world(table,blueprint=wp["world_blueprint"],include_shape_diagnostics=False)
        self.assertTrue(any('caused_by' in x for x in issues))
        self.assertTrue(any('没有 caused_by' in x for x in issues))

    def test_review_receives_original_rules_and_execution_policy(self):
        original=deepcopy(self.wp)
        trace=Tracer([opinion(self.wp,'repair')])
        result=review.assess(self.wp,trace)
        self.assertEqual(result['decision'],'repair')
        self.assertEqual(trace.calls[0]['payload']['mechanical_execution']['rules'],self.wp['world_blueprint']['causal_rules'])
        self.assertEqual(self.wp,original)

    def test_review_receives_compiled_global_identity_and_local_capability_scope(self):
        from instance_plan_selftest import setup_plan
        from pipeline.instance_plan import compile_plan
        wp, plan, *_ = setup_plan()
        company = plan['units'][0]['objects'].pop(0)
        plan['units'].insert(0, {'unit_id':'foundation', 'business_purpose':'Define the shared company', 'objects':[company]})
        plan['units'][1]['depends_on'] = ['foundation']
        revised = compile_plan(wp,plan)
        before = deepcopy(revised)
        tracer = Tracer([opinion(revised,'repair')])
        result = review.assess(revised,tracer)
        self.assertEqual(result['decision'],'repair')
        self.assertEqual(len(tracer.calls),1)
        contract = tracer.calls[0]['payload']['instance_execution']
        objects = {o['slot']:o for o in contract['objects']}
        self.assertEqual(len(objects),2)
        self.assertEqual(objects['company-slot']['definition_unit'],'foundation')
        self.assertEqual(objects['company-slot']['referencing_units'],['report-case'])
        self.assertEqual(contract['capability_scope']['obligations'][0]['carrier']['entities'],['report-slot'])
        self.assertEqual(len(contract['capability_scope']['obligations']),1)
        self.assertEqual(contract['declared_object_minima'],{'company':1,'report':1})
        self.assertEqual(contract['allocated_object_counts'],{'company':1,'report':1})
        self.assertIn({'entity_type':'report','field':'capital'},
                      contract['numeric_field_ownership']['intrinsic_observable'])
        self.assertIn({'entity_type':'report','field':'profit'},
                      contract['numeric_field_ownership']['event_owned'])
        self.assertEqual(contract['plan_hash'],revised['business_instance_plan']['plan_hash'])
        self.assertEqual(revised,before)

    def test_execution_projection_does_not_replace_plan_validation(self):
        from instance_plan_selftest import setup_plan
        from pipeline.instance_plan import compile_plan, PlanConflict
        wp, plan, *_ = setup_plan()
        plan['units'][0]['relations'][0]['to'] = 'undefined-company'
        with self.assertRaises(PlanConflict):
            compile_plan(wp,plan)

    def test_whitepaper_repairs_semantic_conflict_before_capability_mapping(self):
        from pipeline.central_office import central_office
        from seed_contract_selftest import CouncilTracer, fixture as seed_fixture, blueprint
        pack = seed_fixture()
        class RepairTracer(CouncilTracer):
            def chat_json(self, tag, messages, **kwargs):
                result = super().chat_json(tag, messages, **kwargs)
                if tag == "council.world":
                    result["world_blueprint"]["entity_types"][0]["fields"].append(
                        {"name": "cycle_status", "kind": "status", "states": ["confirmed", "recheck"]})
                if tag == "council.blueprint_feasibility" and sum(t == tag for t, _ in self.calls) == 1:
                    result.update(decision="repair", issues=[{"finding": "cycle_status prevents reconfirmation",
                        "suggestion": "Remove the unsupported one-way declaration"}])
                    result["mechanism_checks"][0]["status"] = "blocked"
                return result
        tracer = RepairTracer(blueprint(pack))
        result = central_office(pack["task"]["objective"], [], tracer, log=lambda *_: None, seed_pack=pack)
        tags = [tag for tag, _ in tracer.calls]
        self.assertEqual([tag for tag in tags if tag in {"council.world", "council.world_repair", "council.blueprint_feasibility", "council.map"}],
                         ["council.world", "council.blueprint_feasibility", "council.world_repair", "council.blueprint_feasibility", "council.map"])
        self.assertEqual([item["decision"] for item in result["_council_views"]["blueprint_feasibility"]], ["repair", "accept"])
        self.assertIn("cycle_status", next(payload for tag, payload in tracer.calls if tag == "council.world_repair"))

    def test_revision_changes_only_named_optional_constraint_then_rechecks(self):
        original = deepcopy(self.wp)
        tracer, audit = Tracer([self.change, opinion(self.wp)]), {}
        repaired = review.revise_constraints(self.wp, tracer, {"reason": "recheck cannot finish"}, audit)
        expected = deepcopy(original["world_blueprint"])
        next(f for t in expected["entity_types"] for f in t["fields"] if f["name"] == "review").pop("states")
        self.assertEqual(repaired["world_blueprint"], expected)
        self.assertEqual(self.wp, original)
        self.assertEqual(repaired["seed_contract"], original["seed_contract"])
        self.assertTrue(repaired["seed_audit"]["passed"])
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual([c["step"] for c in tracer.calls], ["council.blueprint_constraint_repair", "council.blueprint_feasibility"])

    def test_seed_required_constraint_cannot_be_removed(self):
        from pipeline.seed_pack import seed_contract
        pack = deepcopy(self.wp["seed_contract"])
        for key in ("digest", "contract_digest"):
            pack.pop(key, None)
        next(f for t in pack["blueprint_requirements"]["entity_types"]
             for f in t["fields"] if f["name"] == "review")["states"] = self.field["states"][:]
        self.wp["seed_contract"] = seed_contract(pack)
        with self.assertRaisesRegex(WorldBlueprintError, "violates seed"):
            review.revise_constraints(self.wp, Tracer([self.change]), {}, {})

    def test_unapproved_fields_and_semantic_rejection_preserve_original(self):
        original = deepcopy(self.wp)
        bad = deepcopy(self.change); bad["edits"][0]["set"] = {"kind": "text"}
        with self.assertRaises(WorldBlueprintError):
            review.revise_constraints(self.wp, Tracer([bad]), {}, {})
        with self.assertRaisesRegex(WorldBlueprintError, "remains unresolved"):
            review.revise_constraints(self.wp, Tracer([self.change, opinion(self.wp, "unresolved")]), {}, {})
        self.assertEqual(self.wp, original)

    def test_real_validator_accepts_cycle_only_without_one_way_declaration(self):
        ws = WorldState({"Record": {"review": Timeline([
            Op(0, "2025-01-01", SET, "confirmed"),
            Op(1, "2025-01-02", UPDATE, "recheck", "confirmed"),
            Op(2, "2025-01-03", UPDATE, "confirmed", "recheck")])}}, n_sessions=3)
        before = {"field_schema": [self.field], "state_machines": [{"field": "review", "states": self.field["states"]}]}
        self.assertTrue(any(x["type"] == "illegal_transition" for x in validate(ws, {}, before)))
        repaired = review.revise_constraints(self.wp, Tracer([self.change, opinion(self.wp)]), {}, {})
        self.assertFalse(any(x["type"] == "illegal_transition" for x in validate(ws, {}, repaired["domain_profile"])))
        # A separately declared irreversible lifecycle still rejects its reversal.
        self.assertTrue(any(x["type"] == "illegal_transition" for x in validate(ws, {}, before)))

    def test_planner_routes_explicit_request_with_rejected_proposal(self):
        full = table_part(self.table, [0, 1], True)
        full["entities"][1]["fields"]["review"] = {"type": "stable", "value": "confirmed"}
        tracer = ScriptedTracer([(PLAN, write("bad")), (WRITE, full),
            (PLAN, {"action": "request_blueprint_review", "reason": "Declaration conflicts with required recheck"})])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            with self.assertRaises(review.BlueprintReviewRequested) as raised:
                generate_world(self.wp, tracer, checkpoint_path=checkpoint, log=lambda *_: None)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "blueprint_review_requested")
        self.assertEqual(raised.exception.evidence["last_observation"]["rejected_proposal"], full)
        self.assertIn("review", raised.exception.evidence["prior_failures"][0]["issues"][0])

    def test_relation_index_and_duplicate_feedback_have_actual_ids(self):
        context = _context(self.wp, self.wp["world_blueprint"])
        self.assertEqual(context["field_write_routes"]["report"]["structure_fields"]["issuer"][0]["via"], "relation")
        self.assertEqual(_index(self.world, self.wp["world_blueprint"])["relations"][0]["id"], "r1")
        raw = {"entities": [], "relations": deepcopy(self.world.relations), "events": [], "initial_states": []}
        issues = _unit_issues(raw, {"entities": [{"name": name} for name in self.world.entities], "events": []}, self.world, self.wp["world_blueprint"])
        self.assertTrue(any("duplicate" in s and "r1" in s and "Yearbook" in s for s in issues))

    def run_factory(self, fail_after=False, published=False):
        self.wp["world_generation"] = {"strategy": "agentic", "version": 2, "blueprint_repair_attempts": 1}
        directory = self.enterContext(tempfile.TemporaryDirectory())
        run = LocalRun(directory, self.wp)
        if published:
            run.write("02_world.json", self.world.to_dict())
        run.tracer = Tracer([self.change, opinion(self.wp)])
        calls = []
        def author(wp, *args, **kwargs):
            calls.append(deepcopy(wp))
            if len(calls) == 1:
                raise review.BlueprintReviewRequested({"reason": "Required recheck violates states"})
            if fail_after:
                raise WorldBlueprintError("Later world validation failed")
            return deepcopy(self.world)
        with patch.object(factory, "build_world", side_effect=author), patch.object(factory, "validate_seed_identity"), patch.object(factory, "_prepare_lines"):
            error = None
            try:
                factory.stage_world(run)
            except WorldBlueprintError as exc:
                error = exc
        return run, calls, error

    def test_actual_factory_restarts_new_world_after_reviewed_revision(self):
        run, calls, error = self.run_factory()
        self.assertIsNone(error)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["world_blueprint"], calls[1]["world_blueprint"])
        self.assertEqual(run.read("01_whitepaper.json"), calls[1])
        self.assertTrue(run.has("02_world.json"))
        self.assertEqual(len(run.read("02_blueprint_revision_attempts.json")["attempts"]), 1)

    def test_later_failure_restores_contract_and_retains_revision_audit(self):
        run, _, error = self.run_factory(fail_after=True)
        self.assertIsInstance(error, WorldBlueprintError)
        self.assertEqual(run.read("01_whitepaper.json"), self.wp)
        self.assertFalse(run.has("02_world.json"))
        self.assertTrue(run.has("02_blueprint_revision_attempts.json"))

    def test_published_world_cannot_have_its_definition_rewritten(self):
        run, calls, error = self.run_factory(published=True)
        self.assertIsInstance(error, review.BlueprintReviewRequested)
        self.assertEqual(len(calls), 1)
        self.assertFalse(run.tracer.calls)
        self.assertEqual(run.read("01_whitepaper.json"), self.wp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
