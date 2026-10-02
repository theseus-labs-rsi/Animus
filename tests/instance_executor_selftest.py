"""Original compiler, concurrent dispatch and checkpoint recovery, entirely offline."""
from copy import deepcopy
from pathlib import Path
import json
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from instance_plan_selftest import setup_plan, authored_values, ScriptedTracer, ACCEPT
from pipeline import instance_plan as ip
from pipeline.world_agent import generate_world, _digest, _save
from pipeline.world_blueprint import WorldBlueprintError
import config


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Provider/network forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        self.wp, self.plan, _, self.full, _ = setup_plan()

    def values(self, wp):
        return authored_values(wp, wp["business_instance_plan"]["units"][0], self.full)

    def test_completed_world_resume_uses_no_model_calls(self):
        wp = ip.compile_plan(self.wp, self.plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            table, state = generate_world(wp, ScriptedTracer([("world.instance.values", self.values(wp))]), checkpoint_path=path, log=lambda *_:None)
            again, resumed = generate_world(wp, ScriptedTracer([]), checkpoint_path=path, log=lambda *_:None)
            self.assertEqual(table, again); self.assertEqual(state, resumed)

    def test_read_only_obligation_and_publication_unit_uses_no_value_author(self):
        case = self.plan["units"][0]
        summary = {"unit_id": "summary-only", "business_purpose": "Read the completed report history",
            "depends_on": [case["unit_id"]], "objects": [], "relations": [], "events": [],
            "observations": [], "obligations": case.pop("obligations"),
            "publications": case.pop("publications")}
        self.plan["units"].append(summary)
        wp = ip.compile_plan(self.wp, self.plan)
        trace = ScriptedTracer([("world.instance.values", self.values(wp))])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            table, state = generate_world(wp, trace, checkpoint_path=path, log=lambda *_:None)
            self.assertEqual(state["status"], "completed")
            self.assertTrue(state["fulfillment"]["passed"])
            self.assertEqual([row["unit_id"] for row in state["units"]],
                             ["report-case", "summary-only"])
            summary_row = state["units"][-1]
            self.assertEqual(summary_row["execution"], "deterministic_no_values/v1")
            self.assertEqual(summary_row["raw"],
                             {"entities": [], "relations": [], "events": [], "initial_states": []})
            self.assertEqual(len(trace.calls), 1)
            self.assertEqual(len(table["entities"]), 2)
            publications = ip.publication_obligations(wp, state)["rows"]
            self.assertEqual([(row["unit_id"], row["actual_fact_id"])
                              for row in publications], [("summary-only", "review-three")])
            resumed_trace = ScriptedTracer([])
            resumed, recovered = generate_world(wp, resumed_trace, checkpoint_path=path,
                                                log=lambda *_:None)
            self.assertEqual(resumed, table)
            self.assertEqual(recovered, state)
            self.assertEqual(resumed_trace.calls, [])

    def test_compiler_rejects_a_unit_without_facts_or_read_obligations(self):
        self.plan["units"].append({"unit_id": "empty", "business_purpose": "Nothing to do",
            "depends_on": ["report-case"], "objects": [], "relations": [], "events": [],
            "observations": [], "obligations": [], "publications": []})
        with self.assertRaises(ip.PlanConflict) as caught:
            ip.compile_plan(self.wp, self.plan)
        self.assertIn("empty_unit", {item["code"] for item in caught.exception.findings})

    def test_shared_identities_allow_reciprocal_references_and_parallel_values(self):
        from pipeline.plan_transactions import derived_dependencies
        case = self.plan["units"][0]
        company = case["objects"].pop(0)
        self.plan["units"].append({"unit_id":"company-case", "business_purpose":"Publisher observes its report",
            "objects":[company], "observations":deepcopy(case["observations"]),
            "relations":[], "events":[], "obligations":[], "publications":[], "depends_on":[]})
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        self.assertTrue(all(not u["depends_on"] for u in wp["business_instance_plan"]["units"]))
        names = {"company-slot":"Aster", "report-slot":"Yearbook"}
        values = {u["unit_id"]:authored_values(wp,u,self.full) for u in wp["business_instance_plan"]["units"]}
        barrier = threading.Barrier(2); calls = []; lock = threading.Lock()
        class ParallelTracer:
            def chat_json(_, step, messages, **kw):
                payload = json.loads(messages[-1]["content"])
                with lock: calls.append(step)
                if step == "world.instance.identities": return {"identities":names}
                self.assertEqual(step,"world.instance.values")
                self.assertEqual(payload["task"]["known_identities"],names)
                for slot,contract in payload["response_schema"]["oneOf"][0]["properties"]["objects"]["properties"].items():
                    self.assertEqual(contract["properties"]["name"]["const"],names[slot])
                barrier.wait(timeout=5)
                return values[payload["task"]["unit_id"]]
        with tempfile.TemporaryDirectory() as temp, patch.object(config,"LLM_CONCURRENCY",16):
            path = Path(temp)/"checkpoint.json"
            table,state = generate_world(wp,ParallelTracer(),checkpoint_path=path,log=lambda *_:None)
            self.assertEqual(state["status"],"completed")
            self.assertTrue(state["fulfillment"]["passed"])
            self.assertEqual(state["identity_registry"],names)
            self.assertEqual(len(table["entities"]),2)
            self.assertTrue(all(e["fields"] for e in table["entities"]))
            again,resumed = generate_world(wp,ScriptedTracer([]),checkpoint_path=path,log=lambda *_:None)
            self.assertEqual(again,table); self.assertEqual(resumed,state)
        self.assertEqual(calls.count("world.instance.identities"),1)
        self.assertEqual(calls.count("world.instance.values"),2)

    def test_other_units_session_zero_event_reserves_initial_field(self):
        from pipeline.plan_transactions import derived_dependencies
        from pipeline.instance_executor import task

        case = self.plan["units"][0]
        objects = case["objects"]
        case["objects"] = []
        self.plan["units"].append({"unit_id":"value-origin", "business_purpose":"Own report and company",
            "objects":objects, "events":[], "relations":[]})
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        units = {unit["unit_id"]:unit for unit in wp["business_instance_plan"]["units"]}
        assigned = task(wp, units["value-origin"], {})
        report = next(obj for obj in assigned["objects"] if obj["slot"] == "report-slot")
        self.assertEqual([field["name"] for field in report["initial_fields"]], ["review"])

        names = {"company-slot":"Aster", "report-slot":"Yearbook"}
        values = {uid:authored_values(wp, unit, self.full) for uid, unit in units.items()}
        class Tracer:
            def chat_json(_, step, messages, **kwargs):
                if step == "world.instance.identities":
                    return {"identities":names}
                return values[json.loads(messages[-1]["content"])["task"]["unit_id"]]
        table, state = generate_world(wp, Tracer(), log=lambda *_:None)
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["fulfillment"]["passed"])
        self.assertEqual(len(table["events"]), 3)

    def test_author_receives_one_scalar_shape_for_initial_and_event_values(self):
        wp = ip.compile_plan(self.wp, self.plan)
        trace = ScriptedTracer([("world.instance.values", self.values(wp))])
        generate_world(wp, trace, log=lambda *_:None)
        shape = trace.calls[0]["input"]["response_schema"]["oneOf"][0]["properties"]
        request = trace.calls[0]["input"]["task"]
        for obj in request["objects"]:
            contract = shape["objects"]["properties"][obj["slot"]]["properties"]
            self.assertEqual(set(contract["fields"]["properties"]), {f["name"] for f in obj["intrinsic_fields"]})
            self.assertFalse(contract["fields"]["additionalProperties"])
            for field in contract["initial_values"]["properties"].values():
                self.assertEqual(field, {"type":["string", "number"]})
        for event in shape["event_values"]["properties"].values():
            for role in event["properties"].values():
                for field in role["properties"].values():
                    self.assertEqual(field, {"type":["string", "number"]})

    def test_value_author_receives_the_original_fact_time_contract(self):
        from pipeline.world_agent import _context
        from pipeline.world_state import intrinsic_fact_time_contract, WorldState
        wp = ip.compile_plan(self.wp, self.plan)
        trace = ScriptedTracer([("world.instance.values", self.values(wp))])
        table, _state = generate_world(wp, trace, log=lambda *_:None)
        request = trace.calls[0]["input"]
        world = WorldState(world_blueprint=wp["world_blueprint"])
        expected = intrinsic_fact_time_contract(world.date_of_session)
        self.assertEqual(request["fact_time_contract"], expected)
        self.assertEqual(request["fact_time_contract"], _context(wp, wp["world_blueprint"])["fact_time_contract"])
        self.assertEqual(request["calendar"][0], {k: expected["stable"][k] for k in ("session", "date")})
        self.assertEqual(len(trace.calls), 1)
        self.assertEqual(table["entities"], self.full["entities"])

    def test_shared_initial_value_failure_is_corrected_before_commit(self):
        from pipeline.plan_transactions import derived_dependencies
        self.plan["units"][0]["events"][0]["session"] = 1
        wp = ip.compile_plan(self.wp,derived_dependencies(self.plan))
        good = self.values(wp)
        good["objects"]["report-slot"]["initial_values"]["profit"] = 0
        bad = deepcopy(good)
        bad["objects"]["report-slot"]["initial_values"]["profit"] = "2025-01-10 (unverified)"
        trace = ScriptedTracer([("world.instance.identities",{"identities":{"company-slot":"Aster","report-slot":"Yearbook"}}),
            ("world.instance.values",bad),("world.instance.values",good)])
        _table,state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(state["status"],"completed")
        self.assertEqual(state["steps"],3)
        self.assertEqual(state["log"][0]["status"],"rejected")
        error=trace.calls[2]["input"]["correction"]["compiler_issues"][0]
        self.assertIn('"entity": "Yearbook"',error)
        self.assertIn('"field": "profit"',error)
        self.assertEqual(trace.calls[2]["input"]["correction"]["original_response"],bad)
        self.assertEqual(state["units"][0]["author_output"],good)

    def test_shared_event_noop_returns_to_author_before_commit(self):
        from pipeline.plan_transactions import derived_dependencies
        self.plan['units'][0]['events'][0]['session'] = 1
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        good = self.values(wp)
        good['objects']['report-slot']['initial_values']['profit'] = 0
        bad = deepcopy(good)
        bad['objects']['report-slot']['initial_values']['profit'] = 200
        trace = ScriptedTracer([('world.instance.identities', {'identities':{'company-slot':'Aster','report-slot':'Yearbook'}}),
            ('world.instance.values',bad), ('world.instance.values',good)])
        _table, state = generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(state['status'],'completed')
        self.assertEqual(len(trace.calls),3)
        evidence = json.loads(trace.calls[-1]['input']['correction']['compiler_issues'][0])['evidence']
        self.assertEqual(evidence['proposed']['id'],'revision-zero')
        self.assertIsNone(evidence['canonical'])
        self.assertTrue(evidence['compiler_issues'])
        self.assertEqual(trace.calls[-1]['input']['correction']['original_response'],bad)

    def test_deferred_cross_unit_noop_repairs_its_value_owner(self):
        from pipeline.plan_transactions import derived_dependencies
        first = self.plan['units'][0]
        first['events'][0]['session'] = 1
        objects = first['objects']; first['objects'] = []
        self.plan['units'].append({'unit_id':'value-origin', 'business_purpose':'Original intrinsic values and initial state',
            'objects':objects, 'events':[], 'relations':[]})
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        by_unit = {u['unit_id']:authored_values(wp,u,self.full) for u in wp['business_instance_plan']['units']}
        by_unit['value-origin']['objects']['report-slot']['initial_values']['profit'] = 200
        before = deepcopy(by_unit)
        calls=[]
        class Tracer:
            def chat_json(_,step,messages,**kwargs):
                payload=json.loads(messages[-1]['content']);calls.append(payload)
                if step=='world.instance.identities':return {'identities':{'company-slot':'Aster','report-slot':'Yearbook'}}
                reply=deepcopy(by_unit[payload['task']['unit_id']])
                if payload.get('local_repair'):
                    self.assertEqual(payload['task']['unit_id'],'report-case')
                    self.assertEqual(payload['local_repair']['previous_response'],before['report-case'])
                    write = next(row for row in payload['event_write_context'] if row['field'] == 'profit')
                    self.assertEqual(write['entity'], 'Yearbook')
                    self.assertIsNotNone(write['known_history'])
                    self.assertIn('event-caused new value', write['meaning'])
                    reply['event_values']['revision-zero']['report']['profit']=210
                return reply
        with patch.object(config,'LLM_CONCURRENCY',16):
            _table,state=generate_world(wp,Tracer(),log=lambda *_:None)
        self.assertEqual(state['status'],'completed')
        self.assertEqual(len(calls),4)
        self.assertEqual(len(state['local_repairs']),1)
        origin=next(u for u in state['units'] if u['unit_id']=='value-origin')
        self.assertEqual(origin['author_output'],before['value-origin'])
        self.assertEqual(state['local_repairs'][0]['findings'][0]['carrier'],{'entities':['Yearbook'],'field':'profit'})

    def test_event_history_waits_for_prior_writes_and_initial_owner(self):
        from pipeline.instance_executor import assembled_facts
        bp={'relation_types':[], 'event_types':[{'id':'write','effect_fields':[{'role':'object','field':'value'}]}]}
        wp={'world_blueprint':bp,'business_instance_plan':{'units':[
            {'objects':[{'slot':'o'}],'relations':[],'events':[{'slot':'e1','type':'write','participants':{'object':'o'},'session':1}]},
            {'objects':[],'relations':[],'events':[{'slot':'e3','type':'write','participants':{'object':'o'},'session':3}]}]}}
        def facts(event, owns=False):
            return {'raw':{'entities':[{'name':'o','fields':{}}] if owns else [],'relations':[],'initial_states':[],
                'events':[{'id':event,'session':1 if event=='e1' else 3,'participants':{'object':'o'},
                    'effects':[{'entity':'o','field':'value','set':1}]}]}}
        late=facts('e3')
        self.assertEqual(assembled_facts(wp,[late])['events'],[])
        early=facts('e1',True)
        self.assertEqual({e['id'] for e in assembled_facts(wp,[late,early])['events']},{'e1','e3'})
        self.assertEqual(assembled_facts(wp,[late],complete=True)['events'],late['raw']['events'])

    def test_returned_invalid_initial_state_resume_dispatches_correction(self):
        from pipeline.plan_transactions import derived_dependencies
        self.plan["units"][0]["events"][0]["session"] = 1
        wp = ip.compile_plan(self.wp,derived_dependencies(self.plan))
        good = self.values(wp)
        good["objects"]["report-slot"]["initial_values"]["profit"] = 0
        bad = deepcopy(good)
        bad["objects"]["report-slot"]["initial_values"]["profit"] = "invalid"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            class Interrupted(BaseException):pass
            class StopAfterReply:
                calls=[]
                def chat_json(_,step,messages,**kw):
                    if step=='world.instance.identities':return {"identities":{"company-slot":"Aster","report-slot":"Yearbook"}}
                    raise Interrupted()
            with self.assertRaises(Interrupted):generate_world(wp,StopAfterReply(),checkpoint_path=path,log=lambda *_:None)
            saved=json.loads(path.read_text(encoding='utf8'))
            saved['log'][-1].update(raw_output=bad,status='returned')
            _save(path,saved)
            original=deepcopy(saved['log'])
            trace=ScriptedTracer([('world.instance.values',good)])
            _,state=generate_world(wp,trace,checkpoint_path=path,log=lambda *_:None)
            self.assertEqual(len(trace.calls),1)
            self.assertEqual(trace.calls[0]['input']['correction']['original_response'],bad)
            self.assertEqual(state['log'][:len(original)],original)
            self.assertEqual(len(state['revalidation_findings']),1)
            self.assertEqual(state['steps'],3)

    def test_shared_invalid_intrinsic_values_receive_original_compiler_feedback(self):
        from pipeline.plan_transactions import derived_dependencies
        wp = ip.compile_plan(self.wp,derived_dependencies(self.plan))
        values = self.values(wp)
        values["objects"]["report-slot"]["fields"]["capital"]["trajectory"][0]["value"] = "invalid numeric value"
        trace = ScriptedTracer([("world.instance.identities",{"identities":{"company-slot":"Aster","report-slot":"Yearbook"}}),
            ("world.instance.values",values),("world.instance.values",self.values(wp))])
        _table,state=generate_world(wp,trace,log=lambda *_:None)
        self.assertEqual(state['status'],'completed')
        self.assertIn('numeric',trace.calls[2]['input']['correction']['compiler_issues'][0])

    def test_event_owned_fields_have_an_empty_intrinsic_response_map(self):
        from pipeline.instance_executor import task, author_contract, materialize
        obj = {"slot":"event-owned", "type":"report", "intrinsic_fields":[],
            "initial_fields":[{"name":"capital", "kind":"numeric"}]}
        request = {"objects":[obj], "events":[], "relations":[], "known_identities":{}}
        shape = author_contract(request)["properties"]["objects"]["properties"][obj["slot"]]["properties"]["fields"]
        self.assertEqual(shape["properties"], {})
        self.assertFalse(shape["additionalProperties"])
        values = {"objects":{obj["slot"]:{"name":"Opening report", "fields":{},
            "initial_values":{"capital":100}}}, "event_values":{}}
        values["objects"][obj["slot"]]["fields"][obj["initial_fields"][0]["name"]] = {"type":"stable", "value":100}
        with self.assertRaises(WorldBlueprintError) as raised:
            materialize(request, values)
        details = json.loads(str(raised.exception).split(": ",1)[1])
        self.assertEqual(details["path"], "/objects/"+obj["slot"]+"/fields")
        self.assertEqual(details["expected_keys"], [])
        self.assertEqual(details["unexpected_keys"], [obj["initial_fields"][0]["name"]])

    def test_missing_event_value_identifies_the_exact_role_and_field(self):
        from pipeline.instance_executor import task, materialize
        wp = ip.compile_plan(self.wp, self.plan)
        request = task(wp, wp["business_instance_plan"]["units"][0], {})
        values = self.values(wp)
        event = request["events"][0]; role = next(iter(event["value_fields"]))
        field = event["value_fields"][role][0]["name"]
        values["event_values"][event["slot"]][role].pop(field)
        with self.assertRaises(WorldBlueprintError) as raised:
            materialize(request, values)
        details = json.loads(str(raised.exception).split(": ",1)[1])
        self.assertEqual(details["path"], "/event_values/"+event["slot"]+"/"+role)
        self.assertEqual(details["missing_keys"], [field])

    def test_interleaved_relation_histories_wait_for_missing_predecessors(self):
        from pipeline.instance_executor import assembled_facts
        wp={"world_blueprint":{"entity_types":[
            {"id":"owner","fields":[{"name":"active","kind":"fk","ref_type":"case"}]},
            {"id":"case","fields":[]}], "relation_types":[
                {"id":"active_case","from_type":"owner","to_type":"case","field":"active","temporal":True}]},
            "business_instance_plan":{"units":[
                {"objects":[{"slot":"o"},{"slot":"a"}], "events":[{"slot":"late"},{"slot":"child"},{"slot":"early"}],
                 "relations":[{"slot":"a1","type":"active_case","from":"o","to":"a","session":1},
                              {"slot":"a3","type":"active_case","from":"o","to":"a","session":3}]},
                {"objects":[{"slot":"b"}], "events":[],
                 "relations":[{"slot":"b2","type":"active_case","from":"o","to":"b","session":2}]}]}}
        def row(spec):return {"id":spec["slot"],**{k:v for k,v in spec.items() if k!='slot'}}
        first={"raw":{"entities":[],"events":[
            {"id":"early","session":1,"participants":{"actor":"o"}},
            {"id":"late","session":3,"participants":{"actor":"o"}},
            {"id":"child","session":4,"participants":{"case":"a"},"caused_by":"late"}],"initial_states":[],
            "relations":[row(r) for r in wp['business_instance_plan']['units'][0]['relations']]}}
        before=deepcopy(first)
        self.assertEqual([r['id'] for r in assembled_facts(wp,[first])['relations']],['a1'])
        self.assertEqual([e['id'] for e in assembled_facts(wp,[first])['events']],['early'])
        self.assertEqual(first,before)
        second={"raw":{"entities":[],"events":[],"initial_states":[],
            "relations":[row(wp['business_instance_plan']['units'][1]['relations'][0])]}}
        self.assertEqual(set(r['id'] for r in assembled_facts(wp,[first,second])['relations']),{'a1','a3','b2'})
        self.assertEqual(assembled_facts(wp,[first,second])['events'],first['raw']['events'])
        self.assertEqual(assembled_facts(wp,[first],complete=True),first['raw'])

    def test_returned_reply_survives_interruption_before_commit(self):
        wp = ip.compile_plan(self.wp, self.plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            expected, state = generate_world(wp, ScriptedTracer([("world.instance.values", self.values(wp))]), checkpoint_path=path, log=lambda *_:None)
            # Exact process boundary after provider reply persisted, before commit.
            state["units"] = []; state["status"] = "building"
            _save(path, state)
            table, resumed = generate_world(wp, ScriptedTracer([]), checkpoint_path=path, log=lambda *_:None)
            self.assertEqual(expected, table); self.assertEqual(resumed["steps"], 1)

    def test_recompiled_reply_preserves_the_original_rejection_record(self):
        wp = ip.compile_plan(self.wp, self.plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            expected, state = generate_world(wp, ScriptedTracer([("world.instance.values", self.values(wp))]), checkpoint_path=path, log=lambda *_:None)
            state["units"] = []; state["status"] = "building"
            state["log"][0].update(status="rejected", issues=["old incremental execution error"])
            original_log = deepcopy(state["log"])
            _save(path, state)
            table, resumed = generate_world(wp, ScriptedTracer([]), checkpoint_path=path, log=lambda *_:None)
            self.assertEqual(table, expected)
            self.assertEqual(resumed["steps"], 1)
            self.assertEqual(resumed["log"], original_log)
            receipt = resumed["units"][0]["revalidated_from"]
            self.assertEqual(receipt["old_status"], "rejected")
            self.assertEqual(receipt["old_issues"], original_log[0]["issues"])
            self.assertEqual(receipt["current_issues"], [])
            self.assertEqual(receipt["original_raw_hash"], _digest(original_log[0]["raw_output"]))

    def test_independent_units_execute_concurrently_and_dependency_waits(self):
        case = self.plan["units"][0]
        company = case["objects"].pop(0)
        case["depends_on"] = ["foundation"]
        self.plan["units"] = [
            {"unit_id":"foundation", "business_purpose":"Company identity", "objects":[company]},
            {"unit_id":"independent", "business_purpose":"Independent company identity", "objects":[{"slot":"other-company", "type":"company"}]}, case]
        wp = ip.compile_plan(self.wp, self.plan)
        case_values = authored_values(wp, wp["business_instance_plan"]["units"][2], self.full)
        barrier = threading.Barrier(2); calls = []; lock = threading.Lock()
        class ParallelTracer:
            def chat_json(_, step, messages, **kw):
                payload = json.loads(messages[-1]["content"]); unit = payload["task"]["unit_id"]
                with lock: calls.append(unit)
                if unit == "report-case":
                    self.assertEqual(set(calls), {"foundation", "independent", "report-case"})
                    self.assertEqual(payload["task"]["known_identities"]["company-slot"], "Aster")
                    return case_values
                barrier.wait(timeout=5)
                slot = "company-slot" if unit == "foundation" else "other-company"
                return {"objects":{slot:{"name":"Aster" if unit == "foundation" else "Beryl", "fields":{"sector":{"type":"stable","value":"insurance"}}}}, "event_values":{}}
        with patch.object(config, "LLM_CONCURRENCY", 16):
            _, state = generate_world(wp, ParallelTracer(), log=lambda *_:None)
        self.assertEqual(state["status"], "completed"); self.assertEqual(len(calls), 3)

    def test_semantic_revision_keeps_original_feedback_and_author_response(self):
        wp = ip.compile_plan(self.wp, self.plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            generate_world(wp, ScriptedTracer([("world.instance.values", self.values(wp))]), checkpoint_path=path, log=lambda *_:None)
            feedback = {"issues":[{"issue_id":"original-review-1", "finding":"Explain the company's original sector"}],
                "repair_targets":{"intrinsic":[{"entity":"Aster", "fields":["sector"]}], "structure":False}}
            values = self.values(wp); values["issue_responses"] = [{"issue_id":"original-review-1", "disposition":"disputed", "response":"The company's original sector is insurance."}]
            trace = ScriptedTracer([("world.instance.values", values)])
            _, state = generate_world(wp, trace, checkpoint_path=path, feedback=feedback, log=lambda *_:None)
            self.assertEqual(trace.calls[0]["input"]["feedback"], feedback)
            self.assertEqual(state["issue_responses"], values["issue_responses"])
            self.assertEqual(len(state["retired_units"]), 1)

    def test_structural_change_invalidates_cached_reply(self):
        wp = ip.compile_plan(self.wp, self.plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            generate_world(wp, ScriptedTracer([("world.instance.values", self.values(wp))]), checkpoint_path=path, log=lambda *_:None)
            wp["business_instance_plan"]["units"][0]["business_purpose"] += " changed"
            with self.assertRaisesRegex(WorldBlueprintError, "binding mismatch"):
                generate_world(wp, ScriptedTracer([]), checkpoint_path=path, log=lambda *_:None)

    def test_provider_failure_keeps_raw_error_and_adds_no_content_retry(self):
        wp = ip.compile_plan(self.wp, self.plan)
        trace = ScriptedTracer([("world.instance.values", {"__error__":"HTTP 429"})])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                generate_world(wp, trace, checkpoint_path=path, log=lambda *_:None)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["units"], []); self.assertEqual(saved["log"][0]["raw_output"]["__error__"], "HTTP 429")
            self.assertEqual(len(trace.calls), 1)

    def test_content_corrections_share_original_stage_budget(self):
        wp = ip.compile_plan(self.wp, self.plan)
        wp["world_generation"] = {"max_steps": 4}
        values = self.values(wp)
        trace = ScriptedTracer([("world.instance.values", {"objects":{}, "event_values":{}, "issue_responses":[{"response":str(n)}]}) for n in range(3)]
            + [("world.instance.values", values)])
        _, state = generate_world(wp, trace, log=lambda *_:None)
        self.assertEqual(state["steps"], 4)
        self.assertEqual(len(trace.calls), 4)
        wp["world_generation"]["max_steps"] = 3
        exhausted = ScriptedTracer([("world.instance.values", {"objects":{}, "event_values":{}, "issue_responses":[{"response":str(n)}]}) for n in range(3)])
        from pipeline.supply_capacity import CapacityReviewRequested
        with self.assertRaises(CapacityReviewRequested):
            generate_world(wp, exhausted, log=lambda *_:None)
        self.assertEqual(len(exhausted.calls), 3)

    def test_identical_initial_content_requests_redesign_without_spending_the_whole_stage(self):
        from pipeline.supply_capacity import CapacityReviewRequested
        wp = ip.compile_plan(self.wp, self.plan)
        trace = ScriptedTracer([("world.instance.values", {"objects":{}, "event_values":{}})] * 3)
        with self.assertRaises(CapacityReviewRequested) as caught:
            generate_world(wp, trace, log=lambda *_:None)
        self.assertEqual(len(trace.calls), 2)
        self.assertEqual(caught.exception.evidence["findings"][0]["code"], "fixed_layout_values_unresolved")

    def test_repeated_original_event_rejection_routes_to_existing_layout_transaction(self):
        from pipeline.plan_transactions import derived_dependencies
        from pipeline.supply_capacity import CapacityReviewRequested
        self.plan['units'][0]['events'][0]['session'] = 1
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        bad = self.values(wp)
        bad['objects']['report-slot']['initial_values']['profit'] = 200
        names = {'company-slot':'Aster', 'report-slot':'Yearbook'}
        trace = ScriptedTracer([('world.instance.identities', {'identities':names}),
            ('world.instance.values', bad), ('world.instance.values', bad)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.json'
            with self.assertRaises(CapacityReviewRequested) as caught:
                generate_world(wp, trace, checkpoint_path=path, log=lambda *_:None)
            saved = json.loads(path.read_text(encoding='utf8'))
            finding = caught.exception.evidence['findings'][0]
            self.assertEqual(finding['kind'], 'business')
            self.assertEqual(finding['evidence']['author_response'], bad)
            self.assertEqual(finding['evidence']['projection'][0]['id'], 'revision-zero')
            self.assertEqual(saved['layout_request'], caught.exception.evidence)
            self.assertEqual([r['raw_output'] for r in saved['log']], [bad, bad])
            self.assertEqual(saved['units'], [])
            self.assertEqual(len(trace.calls), 3)
            # Re-entering the same bound checkpoint must hand off the saved
            # compiler failure without repurchasing the failed value request.
            before = deepcopy(saved['log'])
            resumed = ScriptedTracer([])
            with self.assertRaises(CapacityReviewRequested):
                generate_world(wp, resumed, checkpoint_path=path, log=lambda *_:None)
            after = json.loads(path.read_text(encoding='utf8'))
            self.assertEqual(after['log'], before)
            self.assertEqual(after['steps'], saved['steps'])
            self.assertEqual(resumed.calls, [])

    def test_bounded_event_corrections_escalate_without_extending_value_allowance(self):
        from pipeline.plan_transactions import derived_dependencies
        from pipeline.supply_capacity import CapacityReviewRequested
        self.plan['units'][0]['events'][0]['session'] = 1
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        wp['world_generation'] = {'max_steps':3}
        bad = self.values(wp)
        bad['objects']['report-slot']['initial_values']['profit'] = 200
        changed = deepcopy(bad)
        changed['issue_responses'] = [{'issue_id':'projection', 'disposition':'disputed',
            'response':'The fixed event repeats its value; ask the layout owner.'}]
        trace = ScriptedTracer([('world.instance.identities', {'identities':{
            'company-slot':'Aster', 'report-slot':'Yearbook'}}),
            ('world.instance.values', bad), ('world.instance.values', changed)])
        with self.assertRaises(CapacityReviewRequested) as caught:
            generate_world(wp, trace, log=lambda *_:None)
        self.assertEqual(len(trace.calls), 3)
        self.assertEqual(caught.exception.evidence['findings'][0]['evidence']['author_response'], changed)

    def test_explicit_resume_does_not_treat_failed_provider_result_as_cached_content(self):
        wp = ip.compile_plan(self.wp, self.plan)
        failure = {"__error__":"Empty truncated reply", "__error_metadata__":
            {"kind":"output_truncated","response_received":True,"token_cap":16384}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.json"
            first = ScriptedTracer([("world.instance.values",failure)])
            with self.assertRaisesRegex(RuntimeError,"Empty truncated reply"):
                generate_world(wp,first,checkpoint_path=path,log=lambda *_:None)
            before = json.loads(path.read_text(encoding="utf-8"))
            resumed = ScriptedTracer([("world.instance.values",self.values(wp))])
            _, state = generate_world(wp,resumed,checkpoint_path=path,log=lambda *_:None)
            self.assertEqual(state["status"],"completed")
            self.assertEqual(state["log"][0],before["log"][0])
            self.assertEqual(state["steps"],2)
            self.assertEqual(len(first.calls),1)
            self.assertEqual(len(resumed.calls),1)

    def test_capacity_revision_edits_before_reviewing_changed_layout(self):
        fix = {"decision":"repair", "changes":[{"unit_id":"report-case", "collection":"objects", "operation":"replace", "key":"company-slot", "value":{"slot":"company-slot", "type":"company", "purpose":"Improved business evidence"}}]}
        trace = ScriptedTracer([("council.instance_plan_repair", fix), ("council.blueprint_feasibility", ACCEPT)])
        audit = {}; ip.create(self.wp, trace, {"findings":[{"code":"actual_capacity", "actual":0}]}, audit, initial_proposal=self.plan, repair_required=True)
        self.assertEqual([x["step"] for x in trace.calls], ["council.instance_plan_repair", "council.blueprint_feasibility"])

    def test_capacity_revision_preserves_installed_shared_execution_policy(self):
        from pipeline.plan_transactions import derived_dependencies
        from pipeline.supply_capacity import revise_capacity
        wp = ip.compile_plan(self.wp, derived_dependencies(self.plan))
        fix = {'decision':'repair', 'changes':[{'unit_id':'report-case', 'collection':'observations',
            'operation':'replace', 'selector':{'entity':'report-slot', 'field':'capital'},
            'value':{'entity':'report-slot', 'field':'capital', 'sessions':[0,1,2,3,4],
                'mechanism':'Revised observation window'}}]}
        trace = ScriptedTracer([('council.instance_plan_business_repair',fix),
            ('council.blueprint_feasibility',ACCEPT)])
        result = revise_capacity(wp,trace,{'findings':[{'kind':'business','code':'fixed_layout_values_unresolved'}]}, {})
        self.assertEqual(trace.calls[0]['input']['proposal']['execution_policy'], 'shared-identities/v1')
        self.assertEqual(result['business_instance_plan']['execution_policy'], 'shared-identities/v1')


if __name__ == "__main__": unittest.main()
