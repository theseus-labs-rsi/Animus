"""Actual original author requests; fake responses, no network or provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from pipeline.world_gen import build_world
from pipeline.world_state import assemble_world
from pipeline.world_blueprint import WorldBlueprintError
from world_identity_policy_selftest import case, LocalTracer, LegacyTracer, NAMES
import world_identity_policy_selftest as identity_fixture

DECLARATIONS = "【冻结白皮书的类型与关系声明；仅供跨对象理解，不扩大本批字段权限】"
PUBLISHED = "【已发布对象及 canonical 字段时间线；只读，不得修改】"
PRIOR = "【此前已生成的实际对象与内在事实；purpose 仅为作者用途说明】"


def section(text, marker):
    return json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]


class WorldAuthorContextTests(unittest.TestCase):
    def setUp(self):
        for guard in (
            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")),
            patch.object(config, "chat", side_effect=AssertionError("provider forbidden")),
            patch.object(config, "chat_json", side_effect=AssertionError("provider forbidden")),
            patch.object(config, "pmap", side_effect=lambda fn, items, workers: [fn(x) for x in items]),
        ):
            guard.start(); self.addCleanup(guard.stop)

    def build(self, wp, table, **kw):
        tracer = LocalTracer(table)
        world = build_world(wp, tracer, log=lambda *_: None, **kw)
        return world, tracer

    def test_nonseed_typed_real_request_has_prior_identity_facts_and_frozen_relations(self):
        wp, table = case(seeded=False)
        wp["_council_views"] = {"opinion": "PRIVATE_OPINION_SENTINEL"}
        wp["rubric"] = "PRIVATE_RUBRIC_SENTINEL"
        table["entities"][0]["purpose"] = "A fallible author purpose, not a canonical field."
        original_wp, original_table = deepcopy(wp), deepcopy(table)
        world, tracer = self.build(wp, table)
        self.assertEqual([s for s, _, _ in tracer.calls], ["world.batch", "world.batch", "world.structure"])
        user = tracer.calls[1][1][1]["content"]
        self.assertEqual(section(user, PRIOR), [table["entities"][0]])
        self.assertEqual(section(user, DECLARATIONS), {k: world.world_blueprint[k] for k in ("entity_types", "relation_types")})
        self.assertEqual(section(user, PUBLISHED), [])
        self.assertIn("引用已有对象不等于新建同名实体", user)
        self.assertNotIn("禁止复用或换类型冒用", user)
        self.assertNotIn("PRIVATE_", json.dumps(tracer.calls, ensure_ascii=False))
        self.assertEqual(wp, original_wp); self.assertEqual(table, original_table)
        self.assertNotIn("A fallible author purpose", json.dumps(world.to_dict()))

    def test_empty_intrinsic_objects_remain_visible_with_their_actual_type(self):
        wp, table = case(seeded=False)
        # An entity whose fields are not populated yet still has a usable name
        # and type, as in the actual office report/appointment generation.
        wp["world_blueprint"]["entity_types"][0]["fields"] = []
        table["entities"][0]["fields"] = {}
        _, tracer = self.build(wp, table)
        prior = section(tracer.calls[1][1][1]["content"], PRIOR)
        self.assertEqual(prior, [{"name": "青禾公司", "type": "company", "fields": {}}])

    def test_seeded_path_keeps_one_shared_plan_and_original_call_count(self):
        wp, table = case()
        world, tracer = self.build(wp, table)
        self.assertEqual([s for s, _, _ in tracer.calls], ["world.joint_plan", "world.batch", "world.batch", "world.structure"])
        user = tracer.calls[2][1][1]["content"]
        self.assertEqual(section(user, PRIOR), [table["entities"][0]])
        self.assertIn("共同作者上下文", user)
        self.assertEqual(world.timeline(NAMES[0], "issuer").value_at_session(0), "青禾公司")

    def test_incremental_nonseed_sees_published_canonical_timelines_without_rewriting_them(self):
        old_wp, old_table = case(NAMES[:1], seeded=False)
        existing, issues = assemble_world(old_table, blueprint=old_wp["world_blueprint"])
        self.assertEqual(issues, [])
        original = deepcopy(existing.to_dict())
        wp, table = case(seeded=False)
        world, tracer = self.build(wp, table, existing=existing)
        self.assertEqual([s for s, _, _ in tracer.calls], ["world.batch", "world.structure"])
        user = tracer.calls[0][1][1]["content"]
        published = section(user, PUBLISHED)
        self.assertEqual(published, [{"name": n, "type": original["entity_types"][n], "timelines": f}
                                     for n, f in original["entities"].items()])
        self.assertEqual(section(user, PRIOR), [])
        actual = world.to_dict()
        for name, fields in original["entities"].items():
            self.assertEqual(actual["entities"][name], fields)
        for key in ("relations", "events"):
            current = {r["id"]: r for r in actual[key]}
            for row in original[key]: self.assertEqual(current[row["id"]], row)

    def test_quality_only_legacy_schema_gets_prior_round_context_without_planner(self):
        wp = identity_fixture.WorldIdentityTests.legacy_wp()
        wp["quality_contract"] = {"world_semantic_review": True}
        class Partial(LegacyTracer):
            def chat_json(inner, step, messages, **params):
                response = super().chat_json(step, messages, **params)
                response["entities"] = [response["entities"][len(inner.calls) - 1]]
                return response
        tracer = Partial(NAMES)
        world = build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual(len(tracer.calls), 2)
        prior = section(tracer.calls[1][1][1]["content"], PRIOR)
        self.assertEqual(prior[0]["name"], NAMES[0])
        self.assertEqual(prior[0]["type"], "legacy_entity")
        self.assertEqual(set(world.entities), set(NAMES))

    def test_same_round_siblings_only_see_round_start_not_future_responses(self):
        wp, table = case([f"青禾公司年报{2020+i}" for i in range(10)], seeded=False)
        class Batches(LocalTracer):
            def __init__(inner, data): super().__init__(data); inner.report_batch = 0
            def chat_json(inner, step, messages, **params):
                result = super().chat_json(step, messages, **params)
                if step == "world.batch" and "type id=report" in messages[1]["content"]:
                    start = inner.report_batch * 8; inner.report_batch += 1
                    result["entities"] = result["entities"][start:start+8]
                return result
        tracer = Batches(table); world = build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual([s for s, _, _ in tracer.calls], ["world.batch", "world.batch", "world.batch", "world.structure"])
        self.assertEqual(len(world.entities), 11)
        reports = [m for step, m, _ in tracer.calls if step == "world.batch" and "type id=report" in m[1]["content"]]
        self.assertEqual([section(m[1]["content"], PRIOR) for m in reports], [[table["entities"][0]]] * 2)

    def test_legacy_plain_request_preserves_old_context_and_name_guard(self):
        tracer = LegacyTracer(["记录甲", "记录乙"])
        build_world(identity_fixture.WorldIdentityTests.legacy_wp(), tracer, log=lambda *_: None)
        user = tracer.calls[0][1][1]["content"]
        for marker in (DECLARATIONS, PUBLISHED, PRIOR): self.assertNotIn(marker, user)
        self.assertIn("【全世界已占用专名，禁止复用或换类型冒用】[]", user)
        self.assertEqual(len(tracer.calls), 1)

    def test_context_does_not_allow_renaming_during_structure_repair(self):
        wp, table = case(seeded=False); draft = {}
        self.build(wp, table, draft_out=draft)
        before = deepcopy(draft)
        class Rename(LocalTracer):
            def chat_json(inner, step, messages, **params):
                result = super().chat_json(step, messages, **params)
                if step == "world.structure": result["entities"] = [{"name": "changed identity"}]
                return result
        tracer = Rename(table)
        with self.assertRaisesRegex(WorldBlueprintError, "non-structural inputs"):
            build_world(wp, tracer, log=lambda *_: None, repair_input={"draft": draft,
                "feedback": {"issues": ["Identity mismatch"]}, "targets": {"intrinsic": [], "structure": True}, "max_calls": 1})
        self.assertEqual([s for s, _, _ in tracer.calls], ["world.structure"])
        self.assertEqual(draft, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
