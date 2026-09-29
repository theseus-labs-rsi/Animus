"""Real seven-line failure and local value correction through original kernels."""
from copy import deepcopy
from pathlib import Path
import gzip
import json
import os
import socket
import sys
import unittest
from unittest.mock import patch
os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-test", MODEL="offline-model")
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from instance_plan_selftest import setup_plan, authored_values, ScriptedTracer
from pipeline import instance_plan as ip
from pipeline.capability_contract import actual_capacity, digest
from pipeline.construction import compile_rows, resolve, compact_feedback
from pipeline.instance_executor import assembled_facts
from pipeline.world_agent import generate_world, _compiled
from pipeline.lines import prepare_lines
from pipeline.lines.L7_consolidation import _native_sequence_reason
import config


class ConstructionIntegration(unittest.TestCase):
    def test_intrinsic_sampling_is_derived_without_values_or_repair_calls(self):
        wp,plan,*_=setup_plan();plan['units'][0]['observations']=[]
        original=deepcopy(plan)
        revised=ip.compile_plan(wp,plan)
        self.assertEqual(plan,original)
        compiled=revised['business_instance_plan']
        self.assertEqual(compiled['units'][0]['observations'][0]['sessions'],list(range(6)))
        self.assertEqual(compiled['observation_lowering'][0]['original_observation_periods'],[])
        self.assertFalse(compiled['observation_lowering'][0]['facts_created'])
        self.assertEqual(compiled['units'][0]['events'],plan['units'][0]['events'])
        self.assertNotIn('value',compiled['units'][0]['observations'][0])

    def test_event_route_with_missing_write_periods_keeps_original_constraint(self):
        wp,plan,*_=setup_plan();plan['units'][0]['observations']=[]
        plan['units'][0]['obligations'][0]['carrier']['field']='profit'
        with self.assertRaises(ip.PlanConflict) as error:ip.compile_plan(wp,plan)
        self.assertIn('native_trend_plan',[f['code'] for f in error.exception.findings])

    def test_unused_cohort_is_retained_without_claiming_comparison_capacity(self):
        wp,plan,*_=setup_plan()
        plan['cohorts']=[{'id':'unused','basis':'Original allocation retained as historical context',
            'members':['company-slot','report-slot']}]
        compiled=ip.compile_plan(wp,plan)['business_instance_plan']
        self.assertEqual(compiled['cohorts'],plan['cohorts'])
        self.assertEqual(compiled['inactive_cohort_ids'],['unused'])
        bad=deepcopy(plan);bad['units'][0]['obligations'][0].update(subtype='compare',carrier={'entities':['company-slot','report-slot']})
        wp['supply_plan']['requirements'][0]['subtype_minimum']={}
        with self.assertRaises(ip.PlanConflict) as error:ip.compile_plan(wp,bad)
        self.assertIn('cohort_layout',[f['code'] for f in error.exception.findings])

    def test_shared_foreign_demand_reaches_its_actual_value_writer(self):
        from pipeline.construction import writer_units, author_requirements
        wp,plan,*_=setup_plan();wp=ip.compile_plan(wp,plan)
        row=deepcopy(wp['business_instance_plan']['construction'][0]);row['unit_id']='reader'
        wp['business_instance_plan']['units'].append({'unit_id':'reader','objects':[],'events':[],
            'observations':[],'obligations':[],'relations':[],'publications':[]})
        wp['business_instance_plan']['construction']=[row]
        self.assertEqual(writer_units(wp,row),['report-case'])
        self.assertEqual(author_requirements(wp,'reader'),[])
        requirements=author_requirements(wp,'report-case')
        self.assertEqual(requirements[0]['id'],row['id'])
        self.assertEqual(requirements[0]['planned_observations'][0]['sessions'],list(range(6)))

    def test_heterogeneous_relation_path_uses_line_owned_field_and_scope(self):
        from pipeline.lines.L2_relational import RelationalLine
        line = RelationalLine()
        blueprint = {'entity_types': [{'id': 'report', 'fields': [{'name': 'publisher'}]},
                      {'id': 'company', 'fields': [{'name': 'director'}]}]}
        objects = {'r': {'type': 'report'}, 'c': {'type': 'company'}}
        carrier = {'entities': ['r', 'c'], 'field': 'director'}
        self.assertEqual(line.carrier_field_issues(carrier, blueprint, objects), [])
        support = {'entity': 'Report', 'field': 'publisher→director', 'aux': {
            'path': ['publisher', 'director'], 'resolved_relation_edges': [{'from': 'Report', 'to': 'Company'}]}}
        self.assertTrue(line.matches_carrier(carrier, {'Report', 'Company'}, support))
        self.assertFalse(line.matches_carrier(carrier, {'Report'}, support))
        self.assertFalse(line.matches_carrier({'field': 'absent'}, {'Report', 'Company'}, support))
        self.assertEqual(line.carrier_field_issues({'entities': ['r', 'c'], 'field': 'absent'}, blueprint, objects)[0]['code'], 'undeclared_carrier_field')

    def test_process_carrier_is_checked_against_actual_event_fields(self):
        from pipeline.lines.L3_process import ProcessLine
        line = ProcessLine()
        support = {'entity': 'Report', 'field': '', 'aux': {'events': [{'field': 'status'}, {'field': 'owner'}]}}
        self.assertTrue(line.matches_carrier({'field': 'status'}, {'Report'}, support))
        self.assertFalse(line.matches_carrier({'field': 'absent'}, {'Report'}, support))

    def test_planned_observation_reads_persisted_value_without_new_write(self):
        wp, state, world = self.real()
        unit = wp['business_instance_plan']['units'][0]
        obj = unit['objects'][0]['slot']
        name = ip.active_mapping(state)[obj]
        field, timeline = next(iter(world.entities[name].items()))
        from pipeline.world_state import Timeline, Op, SET, DELETE
        world.entities[name][field] = Timeline([Op(0, '2025-01-01', SET, 'existing value')])
        unit['observations'] = [{'entity': obj, 'field': field, 'sessions': [0, 1, 2], 'mechanism': 'Read the current business state'}]
        proof = ip.fulfillment(wp, world, state)
        self.assertFalse(any(f['code'] == 'observation_unrealized' and f['affected_ids'] == [obj, field] for f in proof['planned_fulfillment']['findings']))
        world.entities[name][field].ops.append(Op(2, '2025-01-15', DELETE))
        proof = ip.fulfillment(wp, world, state)
        self.assertTrue(any(f['code'] == 'observation_unrealized' and f['affected_ids'] == [obj, field] for f in proof['planned_fulfillment']['findings']))

    def setUp(self):
        for obj, name in ((socket.socket,"connect"), (config,"chat"), (config,"chat_json")):
            guard = patch.object(obj,name,side_effect=AssertionError("Offline integration forbids provider/network"))
            guard.start(); self.addCleanup(guard.stop)

    def real(self):
        fixture = Path(__file__).parent / "fixtures/seven_line_construction_failure.json.gz"
        if not fixture.exists():
            self.skipTest("Local historical construction fixture is not distributed")
        data = json.loads(gzip.decompress(fixture.read_bytes()))
        wp, state = data["whitepaper"], data["state"]
        world, _, _, issues = _compiled(assembled_facts(wp,state["units"],complete=True),wp["world_blueprint"],None,True,wp)
        self.assertEqual(issues, [])
        return wp, state, world

    def test_unedited_seven_line_fixture_retains_planned_failure_with_actual_supply(self):
        wp, state, world = self.real()
        proof = ip.fulfillment(wp,world,state)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["verification_policy"], "actual-independent-supply/v1")
        self.assertFalse(proof["planned_fulfillment"]["passed"])
        # Explicit text declarations now include a formerly misclassified date-bearing claim.
        self.assertEqual(len(proof["planned_fulfillment"]["findings"]), 9)
        self.assertEqual(proof["findings"], [])
        self.assertEqual(len(proof["actual_supply"]), 7)
        for row in proof["actual_supply"].values():
            self.assertTrue(row["passed"])
            self.assertEqual(row["required"], 6)
            self.assertEqual(len(set(row["family_keys"])), 6)
        self.assertGreaterEqual(proof["native_trend"]["actual"], 1)
        selected = {r["line"]:r["selected"] for r in proof["capacity"]["original_statistics"]["lines"] if r["allocated"]}
        self.assertEqual(len(selected), 7)
        self.assertEqual(set(selected.values()), {6})

    def test_real_l3_and_compare_errors_are_rejected_before_value_generation(self):
        wp, state, world = self.real()
        _, issues = compile_rows(wp["business_instance_plan"],wp["world_blueprint"])
        from pipeline.lines.L3_process import ProcessLine
        mapping = ip.active_mapping(state)
        self.assertEqual(ProcessLine().value_issues(world,{"entities":[mapping[e] for e in ("inv_gate","rec_telemetry","rec_gatelogs") ]})[0]["code"],"process_locator")
        edited = deepcopy(wp["business_instance_plan"])
        for u in edited["units"]:
            for o in u["obligations"]:
                if o["line"] == "L7_consolidation": o["subtype"] = "undefined_subtype"
        _, issues = compile_rows(edited,wp["world_blueprint"])
        self.assertIn("consolidation_subtype", {r["code"] for r in issues})

    def test_l5_constructor_consumes_real_planned_fields(self):
        wp, state, world = self.real()
        plan = wp["business_instance_plan"]
        rows, _ = compile_rows(plan,wp["world_blueprint"])
        plan["construction"] = [r for r in rows if r["line"] == "L5_conflict"]
        world.supply_construction = resolve(wp,ip.active_mapping(state))
        before = digest({"entities":world.to_dict()["entities"],"relations":world.relations,"events":world.events})
        prepare_lines(wp,world,lambda *_:None)
        self.assertEqual(len(world.conflicts),6)
        self.assertEqual({r["field"] for r in world.conflicts},{"公开主张"})
        self.assertEqual(before,digest({"entities":world.to_dict()["entities"],"relations":world.relations,"events":world.events}))
        cap = actual_capacity(wp,world)
        l5 = [c for c in cap["certificates"] if c["line"] == "L5_conflict"]
        self.assertEqual(len(l5),6)
        self.assertTrue(all(c["answer_matches"] for c in l5))

    def test_actual_bad_trend_returns_to_its_value_author_and_passes(self):
        wp, proposal, _, full, _ = setup_plan()
        wp = ip.compile_plan(wp,proposal)
        good = authored_values(wp,wp["business_instance_plan"]["units"][0],full)
        bad = deepcopy(good)
        bad["objects"]["report-slot"]["fields"]["capital"]["trajectory"] = [
            {"session":s,"value":v} for s,v in enumerate([100,150,110,150,110,105])]
        trace = ScriptedTracer([("world.instance.values",bad),("world.instance.values",good)])
        _, state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(len(trace.calls),2)
        self.assertEqual(len(state["local_repairs"]),1)
        repair = trace.calls[1]["input"]["local_repair"]
        self.assertEqual(repair["issues"][0]["code"],"native_trajectory")
        self.assertEqual(repair["issues"][0]["actual"],{"Yearbook":"unclear_net_direction"})
        self.assertEqual(trace.calls[0]["input"]["construction_requirements"][0]["conditions"]["native_trend"]["net_fraction_of_range"],.3)
        self.assertEqual(state["units"][0]["instance_mapping"]["report-slot"],"Yearbook")
        self.assertEqual(state["fulfillment"]["findings"],[])

    def test_feedback_keeps_only_affected_evidence(self):
        wp, state, world = self.real()
        proof = ip.fulfillment(wp,world,state)
        small = compact_feedback({"phase":"actual_instance_fulfillment","proof":proof})
        self.assertEqual(small["findings"],proof["findings"])
        self.assertLess(len(json.dumps(small,ensure_ascii=False)),40000)
        self.assertNotIn("certificates",json.dumps(small))


if __name__ == "__main__": unittest.main()
