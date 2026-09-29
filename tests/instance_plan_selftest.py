"""Executable layout, actual original supply and recovery invariants, offline."""
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
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import config
from pipeline import instance_plan as ip, production, supply, run as run_module
from pipeline.capability_contract import digest, family_key, actual_capacity
from pipeline.delivery_target import AcceptanceScope
from pipeline.run import Run
from pipeline.world_agent import generate_world
from pipeline.world_state import assemble_world
from pipeline.instance_executor import task, materialize
from seed_world_selftest import fixture
from supply_pipeline_selftest import target
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part


def setup_plan():
    wp, _world, table = fixture()
    wp["line_mapping"] = [{"line":"L7_consolidation","applicable":True,"instantiation":"Original numerical history"}]
    supply.attach_plan(wp,target(["L7_consolidation"],1),1,{"max_world_entities":8,"time_span_weeks":6},capacity_driven=True,instance_driven=True)
    plan = {"decision":"plan","reason":"Annual report revision and review", "units":[{
        "unit_id":"report-case","business_purpose":"Company issues a report and reviews its revision", "depends_on":[],
        "objects":[{"slot":"company-slot","type":"company"},{"slot":"report-slot","type":"report"}],
        "relations":[{"slot":"issuer-slot","type":"issued_by","from":"report-slot","to":"company-slot","session":0}],
        "events":[{"slot":"revision-zero","type":"revision","participants":{"report":"report-slot"},"session":0},
                  {"slot":"revision-two","type":"revision","participants":{"report":"report-slot"},"session":2},
                  {"slot":"review-three","type":"reviewed","participants":{"report":"report-slot"},"session":3,"caused_by":"revision-two"}],
        "observations":[{"entity":"report-slot","field":"capital","sessions":list(range(6)),"mechanism":"Annual capital observations with revisions"}],
        "obligations":[{"id":"trend-obligation","line":"L7_consolidation","subtype":"native_trend","carrier":{"entities":["report-slot"],"field":"capital"}}],
        "publications":[{"fact_slot":"review-three","channel":"复核记录","session":3}]}],"cohorts":[]}
    values = [100,150,110,170,140,190]
    table["entities"][1]["fields"]["capital"]["trajectory"] = [{"session":s,"value":v} for s,v in enumerate(values)]
    full = table_part(table,[0,1],True)
    full["slot_mapping"] = {"company-slot":"Aster","report-slot":"Yearbook","issuer-slot":"r1",
        "revision-zero":"e0","revision-two":"e1","review-three":"e2"}
    action = write("actual-case") | {"instance_units":["report-case"]}
    return wp,plan,table,full,action


ACCEPT = {"decision":"accept","reason":"Revision and review have compatible original roles and time",
    "mechanism_checks":[{"mechanism_id":"revision_review","walkthrough":"Revision causes review one period later","status":"feasible"}],"issues":[]}


def authored_values(wp, unit, full):
    """Project original fixture values; generated layout is independently assembled."""
    assigned = task(wp, unit, {})
    actual = full["slot_mapping"]
    entities = {e["name"]: e for e in full["entities"]}
    events = {e["id"]: e for e in full["events"]}
    return {"objects": {o["slot"]: {"name": actual[o["slot"]], "fields": deepcopy(entities[actual[o["slot"]]]["fields"]),
        "initial_values": {r["field"]: r["value"] for r in full["initial_states"] if r["entity"] == actual[o["slot"]]}}
        for o in assigned["objects"]}, "event_values": {e["slot"]: {role: {f["name"]: next(v["set"] for v in events[actual[e["slot"]]]["effects"]
            if v["entity"] == actual[e["participants"][role]] and v["field"] == f["name"]) for f in fs}
            for role, fs in e["value_fields"].items()} for e in assigned["events"]}}


class InstancePlanTests(unittest.TestCase):
    def setUp(self):
        for obj,name in ((socket.socket,"connect"),(config,"chat"),(config,"chat_json")):
            guard = patch.object(obj,name,side_effect=AssertionError("Network/provider forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        self.wp,self.plan,self.table,self.full,self.action = setup_plan()

    def compiled(self):
        return ip.compile_plan(self.wp,self.plan)

    def reject(self,edit,code):
        proposal = deepcopy(self.plan); edit(proposal)
        with self.assertRaises(ip.PlanConflict) as error:
            ip.compile_plan(self.wp,proposal)
        self.assertIn(code,[f["code"] for f in error.exception.findings])

    def test_original_world_and_actual_trend_certified(self):
        wp = self.compiled()
        values = authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)
        trace = ScriptedTracer([("world.instance.values",values)])
        table,state = generate_world(wp,trace,log=lambda *_:None)
        world,issues = assemble_world(table,blueprint=wp["world_blueprint"],include_shape_diagnostics=False)
        self.assertEqual(issues,[])
        receipt = ip.fulfillment(wp,world,state)
        self.assertTrue(receipt["passed"],receipt["findings"])
        self.assertEqual(len(receipt["assignments"]),1)
        self.assertEqual(state["status"],"completed")
        self.assertEqual(state["units"][0]["instance_mapping"]["report-slot"], "Yearbook")
        self.assertEqual(len(trace.calls), 1)
        self.assertIn("task", trace.calls[0]["input"])

    def test_author_content_cannot_replace_event_schedule(self):
        wp = self.compiled(); values = authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)
        wrong = deepcopy(values); wrong["events"] = [{"session":4}]
        trace = ScriptedTracer([("world.instance.values",wrong),("world.instance.values",values)])
        _,state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(len(state["units"]),1)
        self.assertEqual(state["units"][0]["raw"]["events"][-1]["session"], 3)
        self.assertIn("correction",trace.calls[1]["input"])

    def test_old_planner_action_does_not_become_world_content(self):
        wp = self.compiled()
        values = authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)
        trace = ScriptedTracer([("world.instance.values",write("ignored")),("world.instance.values",values)])
        _,state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(len(state["units"]),1)
        self.assertEqual(len(trace.calls),2)
        details=json.loads(trace.calls[1]["input"]["correction"]["compiler_issues"][0].split(": ",1)[1])
        self.assertEqual(details["missing_keys"],["objects","event_values"])
        self.assertEqual(details["path"],"")

    def test_compiled_dependency_delivers_actual_prior_facts(self):
        case = self.plan["units"][0]
        company = case["objects"].pop(0)
        self.plan["units"].insert(0,{"unit_id":"foundation","business_purpose":"Issue the original company identity","objects":[company]})
        case["depends_on"] = ["foundation"]
        wp = self.compiled()
        first = table_part(self.table,[0]); first["slot_mapping"] = {"company-slot":"Aster"}
        second = deepcopy(self.full); second["entities"] = second["entities"][1:]; second["slot_mapping"].pop("company-slot")
        action = write("foundation-actual") | {"instance_units":["foundation"]}
        first_values = {"objects":{"company-slot":{"name":"Aster","fields":first["entities"][0]["fields"],"initial_values":{}}},"event_values":{}}
        values = authored_values(wp, wp["business_instance_plan"]["units"][1], self.full)
        trace = ScriptedTracer([("world.instance.values",first_values),("world.instance.values",values)])
        _,state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(len(state["units"]),2)
        self.assertEqual(trace.calls[1]["input"]["dependency_world"]["entities"][0]["name"],"Aster")

    def test_original_selector_keeps_frozen_native_subtype(self):
        from pipeline.lines import run_lines
        wp = self.compiled(); world,_ = assemble_world(self.table,blueprint=wp["world_blueprint"],include_shape_diagnostics=False)
        orders = run_lines(wp,world,lambda *_:None,question_budget=1)
        self.assertEqual(orders[0]["aux"]["sub"],"S1_trend")

    def test_completed_initial_plan_reuses_audit_without_call(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run_module,"RUNS_DIR",Path(temp)):
            run = Run("fixture","plan-audit",config_meta={"production_control":{"version":ip.PRODUCTION_VERSION}})
            run.write("01_whitepaper.json",self.wp)
            trace = ScriptedTracer([("council.instance_plan",self.plan),("council.blueprint_feasibility",ACCEPT)])
            run.tracer = trace
            production.ensure_instance_plan(run)
            calls = len(trace.calls)
            production.ensure_instance_plan(run)
            self.assertEqual(len(trace.calls),calls)
            self.assertEqual(run.read("01_instance_plan_audit.json")["status"],"accepted")

    def test_no_hidden_answers(self):
        self.reject(lambda p:p["units"][0].update(answer="上升"),"answer_in_plan")

    def test_missing_identity(self):
        self.reject(lambda p:p["units"][0]["objects"].pop(),"missing_declared_objects")

    def test_same_period_write(self):
        self.reject(lambda p:p["units"][0]["events"][1].update(session=0),"same_period_write")

    def test_causal_delay(self):
        self.reject(lambda p:p["units"][0]["events"][-1].update(session=4),"causal_schedule")

    def test_causal_feedback_contains_exact_pair_and_rule(self):
        proposal=deepcopy(self.plan);proposal['units'][0]['events'][-1]['session']=4
        with self.assertRaises(ip.PlanConflict) as error:ip.compile_plan(self.wp,proposal)
        evidence=next(f['evidence'] for f in error.exception.findings if f['code']=='causal_schedule')
        self.assertEqual(evidence['actual_delay_sessions'],2)
        self.assertEqual(evidence['applicable_rules'][0]['delay_sessions'],1)
        self.assertEqual(evidence['parent']['slot'],'revision-two')
        self.assertEqual(evidence['effect']['slot'],'review-three')

    def test_exact_compiled_proposal_reaches_review_before_any_new_plan(self):
        original=deepcopy(self.plan);trace=ScriptedTracer([('council.blueprint_feasibility',ACCEPT)])
        audit={};revised=ip.create(self.wp,trace,{'previous_review':{'decision':'repair'}},audit,initial_proposal=self.plan)
        self.assertEqual(audit['status'],'accepted')
        self.assertEqual(audit['attempts'],[])
        self.assertEqual(self.plan,original)
        rule_id=self.wp['world_blueprint']['causal_rules'][0]['id']
        self.assertEqual(trace.calls[0]['input']['mechanical_execution']['candidate_witnesses'][rule_id][0]['effect'],'review-three')
        self.assertEqual(revised['business_instance_plan']['units'][0]['events'],self.plan['units'][0]['events'])

    def test_compiled_recovery_preserves_real_negative_review(self):
        negative=deepcopy(ACCEPT);negative['decision']='unresolved';negative['mechanism_checks'][0]['status']='unresolved'
        negative['issues']=[{'finding':'Original task is unresolved','suggestion':'Keep original constraints'}]
        trace=ScriptedTracer([('council.blueprint_feasibility',negative)]);audit={}
        with self.assertRaises(ip.PlanConflict):ip.create(self.wp,trace,{},audit,initial_proposal=self.plan)
        self.assertEqual(audit['status'],'design_unresolved')
        self.assertEqual(audit['restored_business_review']['review'],negative)
        self.assertEqual(len(trace.calls),1)

    def test_review_only_recovery_adds_no_author_repairs(self):
        negative=deepcopy(ACCEPT);negative['decision']='repair'
        negative['issues']=[{'finding':'Real original business constraint','suggestion':'Keep the issue'}]
        trace=ScriptedTracer([('council.blueprint_feasibility',negative)]);audit={}
        with self.assertRaises(ip.PlanConflict):ip.create(self.wp,trace,{},audit,initial_proposal=self.plan,max_attempts=0)
        self.assertEqual(audit['status'],'business_repair_required')
        self.assertEqual(len(trace.calls),1)
        self.assertEqual(audit['attempts'],[])

    def test_review_only_recovery_requires_structurally_valid_original(self):
        bad=deepcopy(self.plan);bad['units'][0]['events'][-1]['session']=4
        trace=ScriptedTracer([]);audit={}
        with self.assertRaises(ip.PlanConflict):ip.create(self.wp,trace,{},audit,initial_proposal=bad,max_attempts=0)
        self.assertEqual(audit['status'],'restored_structure_failed')
        self.assertEqual(trace.calls,[])

    def test_no_unknown_parent(self):
        self.reject(lambda p:p["units"][0]["events"][-1].update(caused_by="unknown"),"causal_parent_missing")

    def test_publication_cannot_precede_fact(self):
        self.reject(lambda p:p["units"][0]["publications"][0].update(session=2),"publication_before_fact")

    def test_publication_channel(self):
        self.reject(lambda p:p["units"][0]["publications"][0].update(channel="unknown"),"publication_shape")

    def test_observation_needs_original_writer(self):
        self.reject(lambda p:(p["units"][0]["observations"][0].update(field="profit"),p["units"][0]["obligations"][0]["carrier"].update(field="profit")),"native_trend_temporal_capacity")

    def test_observation_reads_original_persistent_state(self):
        self.plan["units"][0]["observations"].append({"entity":"report-slot","field":"profit","sessions":[3,4,5],"mechanism":"Read the reviewed profit until the next revision"})
        revised = self.compiled()
        self.assertEqual(revised["business_instance_plan"]["units"][0]["observations"][-1]["sessions"],[3,4,5])
        world,issues=assemble_world(self.table,blueprint=revised["world_blueprint"],include_shape_diagnostics=False)
        self.assertEqual(issues,[])
        history=world.entities["Yearbook"]["profit"]
        self.assertIsNotNone(history.value_at_session(3))
        self.assertEqual(history.value_at_session(3),history.value_at_session(5))

    def test_demand_individual_slots(self):
        self.reject(lambda p:p["units"][0]["obligations"].clear(),"demand_slots")

    def test_native_trend_subtype_required(self):
        self.reject(lambda p:p["units"][0]["obligations"][0].pop("subtype"),"subtype_slots")

    def test_invalid_session(self):
        self.reject(lambda p:p["units"][0]["events"][0].update(session=True),"session_out_of_range")

    def test_duplicate_slot(self):
        self.reject(lambda p:p["units"][0]["objects"][1].update(slot="company-slot"),"slot_identity")

    def test_dependency_cycle(self):
        self.reject(lambda p:p["units"][0]["depends_on"].append("report-case"),"unit_dependencies")

    def test_entity_cap(self):
        self.wp["supply_plan"]["world_limits"]["max_world_entities"] = 1
        self.reject(lambda p:None,"entity_cap")

    def test_exact_identity(self):
        self.wp["world_blueprint"]["entity_types"][0]["cardinality_policy"] = "exact"
        self.reject(lambda p:p["units"][0]["objects"].append({"slot":"company-two","type":"company"}),"exact_identity_count")

    def test_scalar_binding_capacity_is_joint(self):
        def edit(p):
            p["units"][0]["objects"].append({"slot":"company-two","type":"company"})
            p["units"][0]["relations"].append({"slot":"issuer-two","type":"issued_by","from":"report-slot","to":"company-two","session":0})
        self.reject(edit,"scalar_binding_conflict")

    def test_duplicate_relation_ids_do_not_create_extra_canonical_supply(self):
        self.reject(lambda p:p['units'][0]['relations'].append(
            dict(p['units'][0]['relations'][0],slot='duplicate-issuer')), 'relation_compilation')

    def test_relation_failure_reports_independent_trend_deficit_in_same_pass(self):
        proposal = deepcopy(self.plan)
        duplicate = dict(proposal['units'][0]['relations'][0], slot='duplicate-issuer')
        proposal['units'][0]['relations'].append(duplicate)
        proposal['units'][0]['observations'][0]['field'] = 'profit'
        proposal['units'][0]['obligations'][0]['carrier']['field'] = 'profit'
        with self.assertRaises(ip.PlanConflict) as error:
            ip.compile_plan(self.wp, proposal)
        self.assertEqual({row['code'] for row in error.exception.findings},
                         {'relation_compilation', 'native_trend_temporal_capacity'})

    def test_unchanged_temporal_fk_is_rejected_before_value_authorship(self):
        wp, proposal = self.temporal_binding_plan()
        row = proposal['units'][1]['relations'][0]
        row.update(to='company-slot', session=4)
        with self.assertRaises(ip.PlanConflict) as error:
            ip.compile_plan(wp, proposal)
        rejected = next(f for f in error.exception.findings if f['code']=='relation_compilation')
        self.assertEqual(rejected['evidence']['proposed']['id'], 'issuer-update')
        self.assertIsNone(rejected['evidence']['canonical'])
        self.assertTrue(rejected['evidence']['compiler_issues'])

    def test_monotonic_business_field_not_trend_carrier(self):
        self.wp["world_blueprint"]["entity_types"][1]["fields"][-1]["monotonic"] = "up"
        self.reject(lambda p:None,"native_trend_plan")

    def test_original_relation_binding_required(self):
        self.wp["world_blueprint"]["event_types"][-1].update(roles={"report":"report","issuer":"company"},relation_bindings=[{"relation":"issued_by","from_role":"report","to_role":"issuer"}])
        self.plan["units"][0]["events"][-1]["participants"]["issuer"] = "company-slot"
        self.reject(lambda p:p["units"][0]["relations"].clear(),"missing_declared_instances")

    def temporal_binding_plan(self):
        wp, _, _ = fixture(relation_bindings=True, temporal_relation=True)
        wp["line_mapping"] = deepcopy(self.wp["line_mapping"])
        supply.attach_plan(wp,target(["L7_consolidation"],1),1,{"max_world_entities":8,"time_span_weeks":6},capacity_driven=True,instance_driven=True)
        proposal = deepcopy(self.plan)
        first = proposal["units"][0]
        first["events"][-1]["participants"]["issuer"] = "company-slot"
        proposal["units"].append({"unit_id":"later-case","business_purpose":"Issuer changes before the original review",
            "depends_on":["report-case"], "objects":[{"slot":"second-company","type":"company"}],
            "relations":[{"slot":"issuer-update","type":"issued_by","from":"report-slot","to":"second-company","session":2}]})
        return wp, proposal

    def test_later_unit_relation_overwrite_rejected_before_value_authorship(self):
        wp, proposal = self.temporal_binding_plan()
        with self.assertRaises(ip.PlanConflict) as error:
            ip.compile_plan(wp,proposal)
        proof = next(row for row in error.exception.findings if row["code"] == "relation_schedule")
        self.assertEqual(proof["evidence"]["event"], "review-three")
        self.assertEqual(proof["evidence"]["required_value"], "company-slot")
        self.assertEqual(proof["evidence"]["actual_value"], "second-company")
        self.assertEqual(proof["evidence"]["session"], 3)

    def test_later_relation_does_not_invalidate_an_earlier_event(self):
        wp, proposal = self.temporal_binding_plan()
        proposal["units"][1]["relations"][0]["session"] = 4
        self.assertEqual(len(ip.compile_plan(wp,proposal)["business_instance_plan"]["units"]), 2)

    def test_event_using_the_current_relation_passes_same_joint_rule(self):
        wp, proposal = self.temporal_binding_plan()
        # The changed issuer owns its own unit; the original review can be
        # located in that later unit without a cyclic dependency.
        event = proposal["units"][0]["events"].pop()
        event["participants"]["issuer"] = "second-company"
        proposal["units"][0]["publications"].clear()
        proposal["units"][1]["events"] = [event]
        self.assertEqual(len(ip.compile_plan(wp,proposal)["business_instance_plan"]["units"]), 2)

    def test_installed_structural_conflict_returns_upstream_without_model_call(self):
        from pipeline.supply_capacity import CapacityReviewRequested
        wp, proposal = self.temporal_binding_plan()
        # A checkpoint from an older implementation can contain the bad
        # schedule. Execution routes this immutable structure to plan repair.
        wp["business_instance_plan"] = {"version":ip.VERSION, "units":proposal["units"], "cohorts":[], "reason":proposal["reason"]}
        tracer = ScriptedTracer([])
        with self.assertRaises(CapacityReviewRequested) as error:
            generate_world(wp,tracer,log=lambda *_:None)
        self.assertEqual(error.exception.evidence["phase"], "business_layout")
        self.assertEqual(tracer.calls, [])

    def test_different_slots_cannot_alias(self):
        wp = self.compiled(); proposal = authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)
        proposal["objects"]["report-slot"]["name"] = "Aster"
        with self.assertRaises(Exception): materialize(task(wp, wp["business_instance_plan"]["units"][0], {}), proposal)

    def test_unplanned_event_not_accepted(self):
        wp = self.compiled(); proposal = authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)
        proposal["event_values"]["extra"] = {}
        with self.assertRaises(Exception): materialize(task(wp, wp["business_instance_plan"]["units"][0], {}), proposal)

    def test_realized_objects_with_missing_actual_trend_fail(self):
        wp = self.compiled()
        world,_ = assemble_world(fixture()[2],blueprint=wp["world_blueprint"],include_shape_diagnostics=False)
        state = {"units":[{"instance_units":["report-case"],"instance_mapping":self.full["slot_mapping"]}]}
        receipt = ip.fulfillment(wp,world,state)
        self.assertFalse(receipt["passed"])
        self.assertIn("actual_native_trend_shortfall",[f["code"] for f in receipt["findings"]])
        self.assertIn("obligation_unrealized",[f["code"] for f in receipt["planned_fulfillment"]["findings"]])

    def test_valid_alternative_carrier_satisfies_supply_without_claiming_plan_realized(self):
        wp = self.compiled()
        world, _ = assemble_world(self.table, blueprint=wp["world_blueprint"], include_shape_diagnostics=False)
        state = {"units":[{"instance_units":["report-case"],"instance_mapping":self.full["slot_mapping"]}]}
        wp["business_instance_plan"]["units"][0]["obligations"][0]["carrier"]["field"] = "profit"
        receipt = ip.fulfillment(wp, world, state)
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertFalse(receipt["planned_fulfillment"]["passed"])
        self.assertEqual(receipt["verification_policy"], "actual-independent-supply/v1")
        self.assertEqual(receipt["actual_supply"]["L7_consolidation"]["required"], 1)
        self.assertGreaterEqual(receipt["native_trend"]["actual"], 1)

    def test_alternative_supply_does_not_cover_a_larger_same_line_reserve(self):
        wp = self.compiled()
        world, _ = assemble_world(self.table, blueprint=wp["world_blueprint"], include_shape_diagnostics=False)
        state = {"units":[{"instance_units":["report-case"],"instance_mapping":self.full["slot_mapping"]}]}
        wp["supply_plan"]["candidate_allocation"]["L7_consolidation"] = 100
        receipt = ip.fulfillment(wp, world, state)
        self.assertFalse(receipt["passed"])
        self.assertIn("actual_supply_shortfall", [f["code"] for f in receipt["findings"]])

    def test_prior_comparison_group_must_match_natural_partition(self):
        self.plan["cohorts"] = [{"id":"fake-group","basis":"Prior label","members":["company-slot","report-slot"]}]
        self.plan['units'][0]['obligations'][0].update(subtype='compare',carrier={'entities':['company-slot','report-slot']})
        self.wp['supply_plan']['requirements'][0]['subtype_minimum']={}
        with self.assertRaises(ip.PlanConflict) as error:
            self.compiled()
        self.assertIn("cohort_layout", [f["code"] for f in error.exception.findings])

    def test_business_repair_updates_layout_through_original_review(self):
        repair = deepcopy(ACCEPT); repair["decision"] = "repair"
        repair["issues"] = [{"finding":"Plan needs a later disclosure","suggestion":"Revise the public schedule"}]
        changes = {"decision":"repair","changes":[
            {"unit_id":"report-case","collection":"publications","operation":"remove","key":"review-three|复核记录|3"},
            {"unit_id":"report-case","collection":"publications","operation":"add","key":"review-three|复核记录|4",
             "value":{"fact_slot":"review-three","channel":"复核记录","session":4}}]}
        trace = ScriptedTracer([("council.instance_plan",self.plan),("council.blueprint_feasibility",repair),
            ("council.instance_plan_business_repair",changes),("council.blueprint_feasibility",ACCEPT)])
        audit = {}; revised = ip.create(self.wp,trace,{"actual":"original deficit"},audit)
        self.assertEqual(audit["status"],"accepted")
        self.assertEqual(trace.calls[2]["input"]["feedback"]["findings"][0]["kind"],"business")
        self.assertEqual(revised['business_instance_plan']['units'][0]['publications'][-1]['session'],4)
        self.assertEqual(audit['business_revisions'][0]['business_review']['decision'],'accept')
        self.assertEqual(revised["business_instance_plan"]["plan_hash"],digest({k:v for k,v in revised["business_instance_plan"].items() if k != "plan_hash"}))

    def test_keyed_repair_preserves_other_fact_rows(self):
        change={"decision":"repair","changes":[{"unit_id":"report-case","collection":"events","operation":"replace","key":"revision-two",
            "value":deepcopy(self.plan["units"][0]["events"][1]) | {"session":1}}]}
        original=deepcopy(self.plan);result=ip.apply_repair(self.plan,change)
        self.assertEqual(self.plan,original)
        self.assertEqual(result["units"][0]["objects"],original["units"][0]["objects"])
        self.assertEqual(result["units"][0]["events"][0],original["units"][0]["events"][0])
        self.assertEqual(result["units"][0]["obligations"],original["units"][0]["obligations"])
        with self.assertRaises(ip.PlanConflict) as e:ip.compile_plan(self.wp,result)
        self.assertIn('causal_schedule',[f['code'] for f in e.exception.findings])

    def test_repair_cannot_replace_the_whole_plan(self):
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','units':[]})
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,self.plan)

    def test_repair_cannot_alias_existing_slots(self):
        value=deepcopy(self.plan['units'][0]['objects'][0])
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','changes':[
            {'unit_id':'report-case','collection':'objects','operation':'add','key':'company-slot','value':value}]})

    def test_repair_wrong_target_is_atomic(self):
        before=deepcopy(self.plan)
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','changes':[
            {'unit_id':'report-case','collection':'events','operation':'remove','key':'revision-two'},
            {'unit_id':'missing','collection':'events','operation':'remove','key':'review-three'}]})
        self.assertEqual(before,self.plan)

    def test_repair_no_op_cannot_reassess_to_pass(self):
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','changes':[]})
        value=deepcopy(self.plan['units'][0]['objects'][0])
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','changes':[
            {'unit_id':'report-case','collection':'objects','operation':'replace','key':'company-slot','value':value}]})

    def test_repair_duplicate_target_is_rejected(self):
        change={'unit_id':'report-case','collection':'events','operation':'remove','key':'revision-two'}
        with self.assertRaises(ip.PlanConflict):ip.apply_repair(self.plan,{'decision':'repair','changes':[change,change]})

    def test_structural_error_uses_keyed_change_not_new_full_plan(self):
        bad=deepcopy(self.plan);bad['units'][0]['events'][-1]['session']=4
        fix={'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':self.plan['units'][0]['events'][-1]}]}
        trace=ScriptedTracer([('council.instance_plan',bad),('council.instance_plan_repair',fix),('council.blueprint_feasibility',ACCEPT)])
        audit={};revised=ip.create(self.wp,trace,{},audit)
        self.assertEqual(audit['status'],'accepted');self.assertIn('change_set',audit['attempts'][1])
        self.assertEqual(revised['business_instance_plan']['units'][0]['objects'],self.plan['units'][0]['objects'])

    def test_derived_repair_revalidates_exact_parent_proposal(self):
        bad=deepcopy(self.plan);bad['units'][0]['events'][-1]['session']=4
        fix={'decision':'repair','changes':[{'unit_id':'report-case','collection':'events','operation':'replace','key':'review-three','value':self.plan['units'][0]['events'][-1]}]}
        trace=ScriptedTracer([('council.instance_plan_repair',fix),('council.blueprint_feasibility',ACCEPT)])
        audit={};ip.create(self.wp,trace,{'diagnosed_parent':'immutable'},audit,initial_proposal=bad)
        self.assertEqual(audit['recovery_proposal_hash'],digest(bad))
        self.assertEqual(trace.calls[0]['input']['feedback']['rejected_plan'],bad)

    def test_global_provider_error_does_not_consume_plan_retries(self):
        trace = ScriptedTracer([("council.instance_plan",{"__error__":"HTTP 429"})]); audit = {}
        with self.assertRaises(RuntimeError): ip.create(self.wp,trace,{},audit)
        self.assertEqual(len(trace.calls),1); self.assertEqual(audit["status"],"provider_error")

    def test_family_identity_ignores_presentation_permutation(self):
        first = {"line":"L7_consolidation","entity":"X","field":"rate","aux":{"candidates":["X","Y"],"qid":"one"}}
        second = deepcopy(first); second["aux"].update(candidates=["Y","X"],qid="two")
        self.assertEqual(family_key(first),family_key(second))

    def test_acceptance_scope_separate_from_delivery(self):
        self.assertEqual(AcceptanceScope.from_dict({"version":1,"count_stage":"generation_quality"}).count_stage,"generation_quality")
        for bad in ({"version":True,"count_stage":"generation_quality"},{"version":1,"count_stage":"grounding"}):
            with self.assertRaises(ValueError): AcceptanceScope.from_dict(bad)

    def test_state_cas_and_idempotent_recovery(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run_module,"RUNS_DIR",Path(temp)):
            run = Run("fixture","lease-state")
            state = {"version":ip.PRODUCTION_VERSION,"round":1,"status":"ready","generation_budget":{"count":0}}
            production.save_state(run,state); original = (run.dir / production.STATE).read_bytes()
            production.save_state(run,state)
            self.assertEqual(original,(run.dir / production.STATE).read_bytes())
            state["generation_budget"]["count"] = 1
            production.save_state(run,state); self.assertEqual(state["state_revision"],1)
            stale = deepcopy(state)
            state["status"] = "completed"; production.save_state(run,state)
            with self.assertRaises(ValueError): production.save_state(run,stale)

    def test_production_lease_rejects_second_controller(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run_module,"RUNS_DIR",Path(temp)):
            run = Run("fixture","lease-exclusive")
            with run.stage_write_lock("first",lock_name=".production.lock"):
                with self.assertRaises(RuntimeError):
                    with run.stage_write_lock("second",lock_name=".production.lock"): pass


if __name__ == "__main__": unittest.main()
