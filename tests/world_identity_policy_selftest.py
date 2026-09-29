"""Original author identity/stability routing; all model responses are local."""
from copy import deepcopy
import hashlib
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from seed_world_selftest import fixture
from pipeline.world_gen import build_world, _world_system
from pipeline.world_state import assemble_world, _strip_disambig
from pipeline.world_blueprint import WorldBlueprintError


NAMES = ["青禾公司年报2023", "青禾公司年报2024"]


def substitute(obj, old, new):
    if isinstance(obj, dict):
        return {k: substitute(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, old, new) for v in obj]
    return new if obj == old else obj


def case(names=NAMES, *, seeded=True):
    wp, _, original = fixture()
    original = substitute(original, "Aster", "青禾公司")
    wp["active_lines"] = []
    wp["world_blueprint"]["entity_types"][1]["count"] = len(names)
    if not seeded:
        wp.pop("seed_contract")
    table = {"entities": [deepcopy(original["entities"][0])], "relations": [], "events": []}
    for i, name in enumerate(names):
        entity = substitute(original["entities"][1], "Yearbook", name)
        entity["fields"]["capital"] = {"type": "stable", "value": "140"}
        table["entities"].append(entity)
        relation = substitute(original["relations"][0], "Yearbook", name)
        relation["id"] = f"r{i}"
        table["relations"].append(relation)
        for row in original["events"]:
            event = substitute(row, "Yearbook", name)
            event["id"] = f"{row['id']}_{i}"
            if "caused_by" in event:
                event["caused_by"] = f"{row['caused_by']}_{i}"
            table["events"].append(event)
    return wp, table


class LocalTracer:
    def __init__(self, table, *, wrong_type=False):
        self.table, self.calls, self.wrong_type = table, [], wrong_type

    def chat_json(self, step, messages, **params):
        self.calls.append((step, deepcopy(messages), deepcopy(params)))
        if step == "world.joint_plan":
            return {"plan": "Local fixture: one issuer, two distinct reports; fixed capital and separate revision events.",
                    "limitations": "Offline control flow only, not a semantic quality judgment."}
        if step == "world.batch":
            tid = "company" if "type id=company" in messages[-1]["content"] else "report"
            rows = [deepcopy(e) for e in self.table["entities"] if e["type"] == tid]
            if self.wrong_type and tid == "report":
                for row in rows:
                    row["type"] = "company"
            return {"entities": rows}
        if step == "world.structure":
            return {k: deepcopy(self.table[k]) for k in ("relations", "events")}
        raise AssertionError("Unexpected local step: " + step)


class LegacyTracer:
    def __init__(self, names):
        self.names, self.calls = names, []

    def chat_json(self, step, messages, **params):
        self.calls.append((step, deepcopy(messages), deepcopy(params)))
        if step != "world.batch":
            raise AssertionError("Unexpected local step: " + step)
        return {"entities": [{"name": n, "fields": {"capital": {"type": "stable", "value": "140"}}}
                             for n in self.names]}


class WorldIdentityTests(unittest.TestCase):
    def setUp(self):
        for guard in (
            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")),
            patch.object(config, "chat", side_effect=AssertionError("provider forbidden")),
            patch.object(config, "chat_json", side_effect=AssertionError("provider forbidden")),
            patch.object(config, "pmap", side_effect=lambda fn, items, workers: [fn(x) for x in items]),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def build(self, wp, table, **kwargs):
        tracer = LocalTracer(table)
        result = build_world(wp, tracer, log=lambda *_: None, **kwargs)
        return result, tracer

    @staticmethod
    def legacy_wp():
        return {"domain_profile": {"entity_noun": "报告", "field_schema": [
                    {"name": "capital", "kind": "numeric", "unit": "元"}]},
                "shared_world_spec": {"entities": {"count": 2}, "timeline": {"n_sessions": 4}}}

    def test_seeded_suffix_year_names_survive_with_only_a_diagnostic(self):
        wp, table = case()
        original = deepcopy(table)
        world, tracer = self.build(wp, table)
        self.assertTrue(set(NAMES).issubset(world.entities))
        self.assertEqual([c[0] for c in tracer.calls],
                         ["world.joint_plan", "world.batch", "world.batch", "world.structure"])
        hints = [d for d in world.generation_diagnostics if d["type"] == "similar_name_stem"]
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["entities"], NAMES)
        self.assertEqual(hints[0]["severity"], "diagnostic")
        self.assertIn("not_identity_error", hints[0]["scope"])
        for name in NAMES:
            self.assertEqual(world.timeline(name, "issuer").value_at_session(0), "青禾公司")
        self.assertEqual(table, original)
        self.assertEqual(_strip_disambig(NAMES[0]), _strip_disambig(NAMES[1]))

    def test_unseeded_typed_path_allows_distinct_parenthesized_versions(self):
        names = ["青禾记录（版本甲）", "青禾记录（版本乙）"]
        wp, table = case(names, seeded=False)
        world, tracer = self.build(wp, table)
        self.assertTrue(set(names).issubset(world.entities))
        self.assertNotIn("world.joint_plan", [c[0] for c in tracer.calls])
        self.assertTrue(any(d["type"] == "similar_name_stem" for d in world.generation_diagnostics))

    def test_exact_duplicate_names_still_exhaust_original_count_limit(self):
        wp, table = case([NAMES[0], NAMES[0]])
        tracer, draft = LocalTracer(table), {}
        with self.assertRaisesRegex(WorldBlueprintError, "实例不足:1/2"):
            build_world(wp, tracer, log=lambda *_: None, draft_out=draft)
        self.assertEqual([c[0] for c in tracer.calls].count("world.batch"), 5)
        self.assertNotIn("world.structure", [c[0] for c in tracer.calls])
        self.assertEqual(draft["status"], "error")

    def test_wrong_entity_type_and_invalid_event_reference_are_not_relaxed(self):
        wp, table = case()
        tracer = LocalTracer(table, wrong_type=True)
        with self.assertRaisesRegex(WorldBlueprintError, "实例不足:0/2"):
            build_world(wp, tracer, log=lambda *_: None)
        changed = deepcopy(table)
        changed["events"][0]["participants"]["report"] = "不存在的对象"
        tracer = LocalTracer(changed)
        with self.assertRaises(WorldBlueprintError):
            build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual([c[0] for c in tracer.calls].count("world.structure"), 3)

    def test_incremental_new_version_preserves_all_published_entities_and_operations(self):
        old_wp, old_table = case(NAMES[:1])
        old, issues = assemble_world(old_table, blueprint=old_wp["world_blueprint"])
        self.assertEqual(issues, [])
        before = deepcopy(old.to_dict())
        wp, table = case()
        world, tracer = self.build(wp, table, existing=old)
        self.assertIs(world, old)
        for name, fields in before["entities"].items():
            self.assertEqual(world.to_dict()["entities"][name], fields)
            self.assertEqual(world.entity_types[name], before["entity_types"][name])
        for key in ("relations", "events"):
            actual = {row["id"]: row for row in world.to_dict()[key]}
            for row in before[key]:
                self.assertEqual(actual[row["id"]], row)
        self.assertIn(NAMES[1], world.entities)
        self.assertEqual([c[0] for c in tracer.calls], ["world.joint_plan", "world.batch", "world.structure"])

    def test_stable_numeric_values_stay_fixed_and_legitimate_changes_remain_allowed(self):
        wp, table = case()
        world, _ = self.build(wp, table)
        for name in NAMES:
            self.assertEqual(len(world.timeline(name, "capital").ops), 1)
            self.assertEqual([world.timeline(name, "capital").value_at_session(s) for s in range(6)], ["140"] * 6)
        table["entities"][1]["fields"]["capital"] = {"type": "evolving", "trajectory": [
            {"session": 0, "value": "140"}, {"session": 2, "value": "160"}]}
        changed, _ = self.build(wp, table)
        self.assertEqual([o.value for o in changed.timeline(NAMES[0], "capital").ops], ["140", "160"])

    def test_neutral_policy_reaches_actual_original_author_request(self):
        wp, table = case()
        _, tracer = self.build(wp, table)
        for step, messages, _ in tracer.calls:
            if step != "world.batch":
                continue
            system, user = messages[0]["content"], messages[1]["content"]
            self.assertIn("时间推进本身不要求改变取值", system)
            self.assertIn("合法共享主体、名称主干、年份或版本系列均允许", system)
            self.assertIn("保持稳定或自然变化", user)
            self.assertNotIn("近重名整条作废", system)
            self.assertNotIn("$world_scope", system)
            self.assertNotIn("$identity_policy", system)

    def test_legacy_name_filter_and_exact_original_system_remain(self):
        wp = self.legacy_wp()
        tracer = LegacyTracer(NAMES)
        with self.assertRaisesRegex(WorldBlueprintError, "实例不足:1/2"):
            build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual(len(tracer.calls), 4)
        original_hash = "8810c23c43dcc20c338b6a6ccceae2c0270983e5c577c00752721f5ebdefbd55"
        system = _world_system(wp["domain_profile"], "legacy_entity", "week")
        self.assertEqual(hashlib.sha256(system.encode()).hexdigest(), original_hash)
        self.assertIn("近重名整条作废", tracer.calls[0][1][0]["content"])
        self.assertIn("按领域自然演化；不为题型强制中间峰谷", tracer.calls[0][1][1]["content"])

    def test_quality_opt_in_on_legacy_schema_uses_new_identity_policy(self):
        wp = self.legacy_wp()
        wp["quality_contract"] = {"scoring_policy": "task-with-supporting-reasons/v1"}
        tracer = LegacyTracer(NAMES)
        world = build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual(set(world.entities), set(NAMES))
        self.assertEqual(len(tracer.calls), 1)
        self.assertIn("时间推进本身不要求改变取值", tracer.calls[0][1][0]["content"])
        self.assertTrue(any(d["type"] == "similar_name_stem" for d in world.generation_diagnostics))

    def test_saved_candidate_repair_replays_diagnostics_without_changing_identity(self):
        wp, table = case()
        draft = {}
        original, _ = self.build(wp, table, draft_out=draft)
        tracer, repaired = LocalTracer(table), {}
        world = build_world(wp, tracer, log=lambda *_: None, draft_out=repaired, repair_input={
            "draft": draft, "feedback": {"issues": ["Fallible identity concern."]},
            "targets": {"intrinsic": [], "structure": True}, "max_calls": 1})
        self.assertEqual([c[0] for c in tracer.calls], ["world.structure"])
        self.assertEqual(world.to_dict(), original.to_dict())
        self.assertEqual(repaired["candidate_world"], draft["candidate_world"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
