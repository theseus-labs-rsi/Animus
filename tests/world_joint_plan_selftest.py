"""Original build_world joint-author integration; no network or provider use."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from pipeline import world_joint_plan as joint
from pipeline.world_gen import build_world, _world_digest
from pipeline.world_blueprint import WorldBlueprintError
from seed_world_selftest import fixture, FixtureTracer
from world_structure_context_selftest import section, TableTracer


class Tracer(FixtureTracer):
    def __init__(self, table, baseline=None, plan=None):
        super().__init__(table)
        self.baseline = baseline
        self.plan = plan
        self.params = []

    def chat_json(self, step, messages, **params):
        self.params.append((step, deepcopy(params)))
        if step == "world.joint_plan" and self.plan is not None:
            self.calls.append((step, deepcopy(messages)))
            if isinstance(self.plan, Exception):
                raise self.plan
            return deepcopy(self.plan)
        out = super().chat_json(step, messages, **params)
        if step == "world.structure" and self.baseline is not None:
            out["initial_states"] = deepcopy(self.baseline)
        return out


class WorldJointPlanTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network/provider allowed"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, self.existing, self.table = fixture()

    def build(self, tracer=None, **kw):
        tracer = tracer or Tracer(self.table)
        draft = {}
        world = build_world(self.wp, tracer, log=lambda *_: None, draft_out=draft, **kw)
        return world, draft, tracer

    def test_initial_value_error_identifies_real_field_and_preserves_original_rule(self):
        rows = [{"entity":"Yearbook", "field":"review", "session":0,
                 "value":{"type":"text", "value":"pending"}}]
        original = deepcopy(self.table)
        with self.assertRaises(WorldBlueprintError) as error:
            joint.apply_initial_states(self.table, {"initial_states":rows}, self.wp["world_blueprint"])
        self.assertIn("nonempty scalar", str(error.exception))
        detail = json.loads(str(error.exception).split("; initial_state=", 1)[1])
        self.assertEqual(detail["entity"], "Yearbook")
        self.assertEqual(detail["field"], "review")
        self.assertEqual(detail["received_value"], rows[0]["value"])
        self.assertEqual(self.table, original)
        rows[0]["value"] = "pending"
        entities, saved = joint.apply_initial_states(self.table, {"initial_states":rows}, self.wp["world_blueprint"])
        self.assertEqual(saved, rows)
        self.assertEqual(next(e for e in entities if e["name"]=="Yearbook")["fields"]["review"],
                         {"type":"stable", "value":"pending"})

    def test_real_builder_shared_complete_task_calendar_and_no_private_inputs(self):
        contract = self.wp["seed_contract"]
        contract["task"]["instructions"] = ("完整冻结任务\n" * 300).rstrip()
        contract["sources"][0]["path"] = "SECRET_SOURCE"
        from pipeline.seed_pack import _digest
        contract["contract_digest"] = _digest({k:v for k,v in contract.items() if k!="contract_digest"})
        self.wp["_council_views"] = {"review":"SECRET_REVIEW"}
        extended = deepcopy(self.wp)
        extended["seed_contract"]["task"]["rubric"] = "SECRET_RUBRIC"
        extended["seed_contract"]["mechanisms"][0]["review"] = "SECRET_REVIEW"
        self.assertNotIn("SECRET_", json.dumps(joint.make_context(
            extended, self.wp["world_blueprint"], [])))
        before = deepcopy(self.wp)
        world, draft, tracer = self.build()
        self.assertEqual(self.wp, before)
        self.assertEqual([s for s, _ in tracer.calls],
                         ["world.joint_plan", "world.batch", "world.batch", "world.structure"])
        for step, messages in tracer.calls[1:]:
            common = section(messages[1]["content"], "【共同作者上下文：冻结任务、日历与同一联合提案】")
            self.assertEqual(common["context"], draft["joint_plan"]["input"])
            self.assertEqual(common["proposal"], draft["joint_plan"]["raw_output"])
            self.assertEqual(common["plan_hash"], draft["joint_plan"]["plan_hash"])
        sent = json.dumps(tracer.calls, ensure_ascii=False)
        self.assertNotIn("SECRET_", sent)
        self.assertIn(contract["task"]["instructions"], tracer.calls[0][1][1]["content"].replace('\\n','\n'))
        calendar = draft["joint_plan"]["input"]["calendar"]
        self.assertEqual(calendar, [{"session":s,"date":world.date_of_session(s)} for s in range(6)])
        params = tracer.params[0][1]
        self.assertEqual(params["model"], config.STRUCTURE_MODEL)
        self.assertEqual(params["retries"], 1)
        self.assertIs(params["strict_json"], True)
        self.assertEqual(params["response_format"], {"type": "json_object"})

    def test_subsequent_batch_sees_actual_prior_identity_facts_and_optional_purpose(self):
        self.table["entities"][0]["purpose"] = "Holder of this report; proposal only."
        _, _, tracer = self.build()
        batches = [m for s,m in tracer.calls if s == "world.batch"]
        actual = section(batches[1][1]["content"], "【此前已生成的实际对象与内在事实；purpose 仅为作者用途说明】")
        self.assertEqual(actual, [self.table["entities"][0]])

    def test_stable_objects_and_proposal_not_compiled_as_facts(self):
        plan = {"plan":"UNCOMPILED_PLAN: Keep the sector stable; optional switches stay untriggered.",
                "limitations":"This does not certify business quality."}
        world, draft, _ = self.build(Tracer(self.table, plan=plan))
        self.assertEqual(len(world.timeline("Aster", "sector").ops), 1)
        self.assertNotIn("UNCOMPILED_PLAN", json.dumps(world.to_dict()))
        self.assertEqual(draft["joint_plan"]["raw_output"], plan)

    def test_structure_baseline_uses_existing_compiler_and_real_later_change(self):
        rows = [{"entity":"Yearbook","field":"review","session":0,"value":"pending"}]
        world, draft, _ = self.build(Tracer(self.table, baseline=rows))
        self.assertEqual([(o.session,o.value) for o in world.timeline("Yearbook","review").ops],
                         [(0,"pending"),(3,"checked")])
        self.assertEqual(draft["initial_states"], rows)
        self.assertEqual(draft["merged"]["entities"][1]["fields"]["review"],
                         {"type":"stable","value":"pending"})

    def test_invalid_baseline_scope_duplicate_and_same_event_slot_stop(self):
        good = {"entity":"Yearbook","field":"review","session":0,"value":"pending"}
        cases = [[{**good,"field":"issuer"}], [{**good,"field":"capital"}],
                 [{**good,"session":1}], [{**good,"session":False}], [good,good],
                 [{**good,"entity":"missing"}], [{**good,"value":None}],
                 [{**good,"value":{}}], [{**good,"field":"profit","value":"200"}]]
        for rows in cases:
            with self.subTest(rows=rows):
                tracer, draft = Tracer(self.table, baseline=rows), {}
                with self.assertRaises(ValueError):
                    build_world(self.wp,tracer,log=lambda *_:None,draft_out=draft)
                self.assertEqual(draft["status"],"error")
                self.assertEqual(sum(s == "world.structure" for s,_ in tracer.calls),3)
                self.assertEqual(len(draft["structure_attempts"]),3)
                self.assertTrue(all(r["status"]=="invalid_initial_state" for r in draft["structure_attempts"]))

    def test_baseline_value_schema_date_states_numeric_and_nonfinite(self):
        bp = deepcopy(self.wp["world_blueprint"])
        review = next(f for f in bp["entity_types"][1]["fields"] if f["name"] == "review")
        for decl,value in [({"kind":"status","states":["pending","checked"]},"other"),
                           ({"kind":"numeric","range":[0,10]},11),
                           ({"kind":"numeric"},float('nan')),
                           ({"kind":"date"},"2025-02-30")]:
            review.update(decl)
            if "states" not in decl:review.pop("states",None)
            if "range" not in decl:review.pop("range",None)
            with self.subTest(decl=decl):
                with self.assertRaises(ValueError):
                    joint.apply_initial_states(self.table,{"initial_states":[{
                        "entity":"Yearbook","field":"review","session":0,"value":value}]},bp)

    def test_incremental_noop_makes_no_plan_or_other_call(self):
        tracer = Tracer(self.table)
        build_world(self.wp,tracer,existing=self.existing,log=lambda *_:None)
        self.assertEqual(tracer.calls,[])

    def test_incremental_existing_facts_visible_but_baseline_cannot_overwrite(self):
        self.wp["world_blueprint"]["entity_types"][0]["count"] = 2
        table = deepcopy(self.table)
        table["entities"] = [{"name":"Beryl","type":"company","fields":{
            "sector":{"type":"stable","value":"insurance"}}}]
        class Incremental(TableTracer):
            def chat_json(inner,step,messages,**params):
                result=super().chat_json(step,messages,**params)
                if step=="world.structure":result["initial_states"]=[{
                    "entity":"Yearbook","field":"review","session":0,"value":"pending"}]
                return result
        original=deepcopy(self.existing.to_dict());tracer=Incremental(table)
        with self.assertRaisesRegex(WorldBlueprintError,"new entities"):
            build_world(self.wp,tracer,existing=self.existing,log=lambda *_:None)
        self.assertEqual(self.existing.to_dict(),original)
        context=json.loads(tracer.calls[0][1][1]["content"])
        self.assertEqual(context["existing_world"]["entities"],original["entities"])

    def test_semantic_repair_reuses_exact_plan_and_baseline_without_planner(self):
        baseline=[{"entity":"Yearbook","field":"review","session":0,"value":"pending"}]
        _,draft,_=self.build(Tracer(self.table,baseline=baseline))
        tracer=Tracer(self.table);out={}
        world=build_world(self.wp,tracer,log=lambda *_:None,draft_out=out,repair_input={
            "draft":draft,"feedback":{"issues":["fallible"]},
            "targets":{"intrinsic":[],"structure":True},"max_calls":1})
        self.assertEqual([s for s,_ in tracer.calls],["world.structure"])
        self.assertEqual(out["joint_plan"],draft["joint_plan"])
        self.assertEqual(out["initial_states"],baseline)
        self.assertEqual(world.to_dict(),draft["candidate_world"])

    def test_tampered_raw_and_current_source_drift_reject_before_repair_call(self):
        _,draft,_=self.build()
        changed=deepcopy(draft);changed["joint_plan"]["raw_output"]["plan"]="changed"
        changed["draft_hash"]=_world_digest({k:v for k,v in changed.items() if k!="draft_hash"})
        repair=lambda d:{"draft":d,"feedback":"opinion","targets":{"intrinsic":[],"structure":True}}
        tracer=Tracer(self.table)
        with self.assertRaisesRegex(WorldBlueprintError,"joint"):
            build_world(self.wp,tracer,repair_input=repair(changed),log=lambda *_:None)
        self.assertEqual(tracer.calls,[])
        with patch.object(joint,"implementation_hashes",return_value={"changed":"source"}):
            with self.assertRaisesRegex(WorldBlueprintError,"binding"):
                build_world(self.wp,tracer,repair_input=repair(draft),log=lambda *_:None)
        self.assertEqual(tracer.calls,[])

    def test_planner_execution_stops_once_shape_has_one_bounded_correction(self):
        for answer in ({"plan":"missing limits"},{"__error__":"provider failure"},OSError("blocked")):
            tracer=Tracer(self.table,plan=answer);draft={}
            with self.subTest(answer=answer):
                with self.assertRaises(WorldBlueprintError):
                    build_world(self.wp,tracer,draft_out=draft,log=lambda *_:None)
                count = 2 if answer == {"plan":"missing limits"} else 1
                self.assertEqual([s for s,_ in tracer.calls],["world.joint_plan"] * count)
                self.assertEqual(draft["joint_plan"]["status"],"error")
                self.assertTrue(draft["joint_plan"]["plan_hash"])
                self.assertEqual(len(draft["joint_plan"]["attempts"]), count)

    def test_unseeded_keeps_original_calls_and_structure_prompt(self):
        self.wp.pop("seed_contract")
        _,draft,tracer=self.build()
        self.assertIsNone(draft["joint_plan"])
        self.assertNotIn("world.joint_plan",[s for s,_ in tracer.calls])
        system=next(m[0]["content"] for s,m in tracer.calls if s=="world.structure")
        self.assertNotIn("initial_states",system)

    def test_partial_batch_keeps_original_names_and_missing_count_without_replanning(self):
        self.wp["world_blueprint"]["entity_types"][0]["count"] = 2
        second={"name":"Beryl","type":"company","fields":{
            "sector":{"type":"stable","value":"insurance"}},"purpose":"Second stable company."}
        table=deepcopy(self.table)
        class Partial(FixtureTracer):
            def chat_json(inner,step,messages,**params):
                if step=="world.batch":
                    i=sum(s==step for s,_ in inner.calls)
                    inner.calls.append((step,deepcopy(messages)))
                    return {"entities":[deepcopy([table["entities"][0],second,table["entities"][1]][i])]}
                return super().chat_json(step,messages,**params)
        tracer=Partial(table)
        world,_,_=self.build(tracer)
        self.assertEqual(sum(s=="world.joint_plan" for s,_ in tracer.calls),1)
        batches=[m[1]["content"] for s,m in tracer.calls if s=="world.batch"]
        self.assertIn("设计 1 个",batches[1])
        prior=section(batches[1],"【此前已生成的实际对象与内在事实；purpose 仅为作者用途说明】")
        self.assertEqual(prior[0]["name"],"Aster")
        self.assertEqual(set(world.entities),{"Aster","Beryl","Yearbook"})

    def test_source_change_on_last_response_fails_instead_of_returning_world(self):
        hashes=joint.implementation_hashes()
        self.assertIn("prompts.py", hashes)
        for source in ("world_joint_plan.py", "prompts.py"):
            with self.subTest(source=source):
                current=deepcopy(hashes)
                class Drifting(Tracer):
                    def chat_json(inner,step,messages,**params):
                        result=super().chat_json(step,messages,**params)
                        if step=="world.structure":current[source]="changed"
                        return result
                tracer=Drifting(self.table);draft={}
                with patch.object(joint,"implementation_hashes",side_effect=lambda:deepcopy(current)):
                    with self.assertRaisesRegex(WorldBlueprintError,"drift"):
                        build_world(self.wp,tracer,log=lambda *_:None,draft_out=draft)
                self.assertEqual(draft["status"],"error")
                self.assertEqual(len(tracer.calls),4)

    def test_baseline_does_not_relax_event_noop_rule(self):
        rows=[{"entity":"Yearbook","field":"review","session":0,"value":"checked"}]
        tracer=Tracer(self.table,baseline=rows)
        with self.assertRaises(WorldBlueprintError):
            self.build(tracer)
        self.assertEqual(sum(s=="world.structure" for s,_ in tracer.calls),3)

    def test_explicit_empty_baselines_can_remove_candidate_baseline_in_structure_repair(self):
        rows=[{"entity":"Yearbook","field":"review","session":0,"value":"pending"}]
        _,draft,_=self.build(Tracer(self.table,baseline=rows))
        tracer=Tracer(self.table,baseline=[]);out={}
        world=build_world(self.wp,tracer,log=lambda *_:None,draft_out=out,repair_input={
            "draft":draft,"feedback":"Baseline need not be asserted in this candidate.",
            "targets":{"intrinsic":[],"structure":True},"max_calls":1})
        self.assertEqual(out["initial_states"],[])
        self.assertEqual([(o.session,o.value) for o in world.timeline("Yearbook","review").ops],[(3,"checked")])
        self.assertIn("initial_states",out["repair_log"][0]["allowed_structure"])
        self.assertIn("未发布候选",tracer.calls[0][1][1]["content"])

    def test_baseline_cannot_substitute_required_event_or_causal_witness(self):
        table=deepcopy(self.table)
        table["events"]=[e for e in table["events"] if e["type"]!="reviewed"]
        rows=[{"entity":"Yearbook","field":"review","session":0,"value":"checked"}]
        tracer=Tracer(table,baseline=rows)
        with self.assertRaisesRegex(WorldBlueprintError,"event|causal"):
            self.build(tracer)
        self.assertEqual(sum(s=="world.structure" for s,_ in tracer.calls),3)

    def test_bad_initial_state_then_corrected_structure_uses_same_bounded_loop(self):
        conflict=[{"entity":"Yearbook","field":"profit","session":0,"value":"200"}]
        class Corrected(Tracer):
            def chat_json(inner,step,messages,**params):
                result=super().chat_json(step,messages,**params)
                if step=="world.structure":
                    result["initial_states"]=conflict if sum(s==step for s,_ in inner.calls)==1 else []
                return result
        world,draft,tracer=self.build(Corrected(self.table))
        for key in ("entities","relations","events","n_sessions"):
            self.assertEqual(world.to_dict()[key],self.existing.to_dict()[key])
        self.assertEqual(sum(s=="world.joint_plan" for s,_ in tracer.calls),1)
        calls=[m for s,m in tracer.calls if s=="world.structure"]
        self.assertEqual(len(calls),2)
        self.assertIn("上轮初态机械校验失败",calls[1][1]["content"])
        self.assertEqual(draft["structure_attempts"][0]["raw_output"]["initial_states"],conflict)
        self.assertEqual(draft["structure_attempts"][0]["status"],"invalid_initial_state")
        self.assertEqual(draft["initial_states"],[])
        self.assertEqual(draft["merged"]["entities"],self.table["entities"])

    def test_later_baseline_errors_keep_prior_candidate_but_never_return_it(self):
        baseline=[{"entity":"Yearbook","field":"review","session":0,"value":"pending"}]
        conflict=[{"entity":"Yearbook","field":"profit","session":0,"value":"200"}]
        class LaterInvalid(Tracer):
            def chat_json(inner,step,messages,**params):
                result=super().chat_json(step,messages,**params)
                if step=="world.structure":
                    if sum(s==step for s,_ in inner.calls)==1:
                        result["initial_states"]=baseline
                        result["events"]=result["events"][:2]
                    else:result["initial_states"]=conflict
                return result
        draft={};tracer=LaterInvalid(self.table)
        with self.assertRaisesRegex(WorldBlueprintError,"initial-state validation exhausted"):
            build_world(self.wp,tracer,log=lambda *_:None,draft_out=draft)
        self.assertEqual(draft["status"],"error")
        self.assertNotIn("candidate_world",draft)
        self.assertEqual(draft["merged"]["events"],self.table["events"][:2])
        self.assertEqual(draft["initial_states"],baseline)
        self.assertNotIn("profit",draft["merged"]["entities"][1]["fields"])
        self.assertEqual([r["status"] for r in draft["structure_attempts"]],
                         ["returned","invalid_initial_state","invalid_initial_state"])

    def test_transport_error_still_stops_before_any_structure_feedback_retry(self):
        class Failed(Tracer):
            def chat_json(inner,step,messages,**params):
                result=super().chat_json(step,messages,**params)
                return {"__error__":"provider failed"} if step=="world.structure" else result
        tracer=Failed(self.table);draft={}
        with self.assertRaisesRegex(WorldBlueprintError,"调用失败"):
            build_world(self.wp,tracer,log=lambda *_:None,draft_out=draft)
        self.assertEqual(sum(s=="world.structure" for s,_ in tracer.calls),1)
        self.assertEqual(draft["status"],"error")


if __name__=="__main__":unittest.main(verbosity=2)
