"""Offline seed/world integration tests; all model responses are local fixtures."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.seed_pack import attach_seed_contract
from pipeline.seed_world import (SeedWorldError, seed_entity_prompt, seed_protected_fields, seed_world_report,
                                 validate_seed_world)
from pipeline.world_gen import _wire_declared_causality, build_world, imprint_structure
from pipeline.world_state import WorldState, Op, UPDATE, assemble_world
from pipeline.world_blueprint import WorldBlueprintError


def fixture(*, shared_roles=False, relation_bindings=False, temporal_relation=False):
    blueprint = {
        "version": 1,
        "entity_types": [
            {"id": "company", "noun": "公司", "primary": True, "count": 1,
             "fields": [{"name": "sector", "kind": "category"}]},
            {"id": "report", "noun": "报告", "count": 1,
             "fields": [{"name": "issuer", "kind": "reference"},
                        {"name": "profit", "kind": "numeric"},
                        {"name": "review", "kind": "status"},
                        {"name": "capital", "kind": "numeric"}]},
        ],
        "relation_types": [{"id": "issued_by", "from_type": "report", "to_type": "company",
                            "field": "issuer", "temporal": False, "min_count": 1}],
        "event_types": [
            {"id": "revision", "label": "发布修订", "roles": {"report": "report"},
             "effect_fields": [{"role": "report", "field": "profit"}], "min_count": 2},
            {"id": "reviewed", "label": "完成复核", "roles": {"report": "report"},
             "effect_fields": [{"role": "report", "field": "review"}], "min_count": 1},
        ],
        "causal_rules": [{"id": "review_revision", "trigger_event": "revision",
                          "effect_event": "reviewed", "delay_sessions": 1}],
        "temporal_model": {"unit": "day", "n_sessions": 6, "cadence": "daily", "step_days": 1},
        "evidence_channels": ["报告摘录", "复核记录"],
    }
    if shared_roles:
        blueprint["causal_rules"][0]["shared_roles"] = ["report"]
    if temporal_relation:
        blueprint["relation_types"][0]["temporal"] = True
    if relation_bindings:
        blueprint["event_types"][1]["roles"]["issuer"] = "company"
        blueprint["event_types"][1]["relation_bindings"] = [
            {"relation": "issued_by", "from_role": "report", "to_role": "issuer"}]
    requirements = deepcopy(blueprint)
    requirements.pop("version")
    requirements["temporal_model"] = {"min_sessions": 6}
    pack = {
        "schema_version": 1, "seed_id": "fixture_revision", "family": "insurance",
        "title": "Synthetic revision fixture", "description": "Test fixture, not real source data.",
        "sources": [{"id": "fixture", "path": "tests/seed_world_selftest.py", "sha256": "0" * 64,
                     "role": "builder_only"}],
        "task": {"objective": "Track revision and review", "instructions": "Keep canonical revisions."},
        "exemplars": [],
        "mechanisms": [{"id": "revision_review", "description": "Revision causes review.",
                        "source_refs": [{"source_id": "fixture", "locator": "fixture()"}],
                        "required_entity_types": ["company", "report"],
                        "required_relation_types": ["issued_by"],
                        "required_event_types": ["revision", "reviewed"],
                        "required_causal_rules": ["review_revision"]}],
        "blueprint_requirements": requirements,
    }
    wp = attach_seed_contract({"world_blueprint": blueprint, "domain_profile": {},
                               "active_lines": [{"line": "L7_consolidation", "weight": 1}]}, pack)
    table = {
        "entities": [{"name": "Aster", "type": "company", "fields": {
            "sector": {"type": "stable", "value": "insurance"}}},
                     {"name": "Yearbook", "type": "report", "fields": {
                         "capital": {"type": "evolving", "trajectory": [
                             {"session": session, "value": value}
                             for session, value in enumerate([100, 110, 120, 130, 140, 150])]}}}],
        "relations": [{"id": "r1", "type": "issued_by", "from": "Yearbook", "to": "Aster", "session": 0}],
        "events": [{"id": "e0", "type": "revision", "session": 0,
                    "participants": {"report": "Yearbook"},
                    "effects": [{"entity": "Yearbook", "field": "profit", "set": "200"}]},
                   {"id": "e1", "type": "revision", "session": 2,
                    "participants": {"report": "Yearbook"},
                    "effects": [{"entity": "Yearbook", "field": "profit", "set": "180"}]},
                   {"id": "e2", "type": "reviewed", "session": 3, "caused_by": "e1",
                    "participants": {"report": "Yearbook"},
                    "effects": [{"entity": "Yearbook", "field": "review", "set": "checked"}]}],
    }
    if relation_bindings:
        table["events"][-1]["participants"]["issuer"] = "Aster"
    world, issues = assemble_world(table, blueprint=blueprint)
    assert not [issue for issue in issues if not "数值轨迹单调" in issue], issues
    return wp, world, table


class FixtureTracer:
    def __init__(self, table, bad_first=False, always_bad=False):
        self.table = table
        self.bad_first = bad_first
        self.always_bad = always_bad
        self.calls = []

    def chat_json(self, name, messages, **kwargs):
        self.calls.append((name, deepcopy(messages)))
        if name == "world.joint_plan":
            return {"plan": "Keep existing identities and stable sector; revise the report and then review it.",
                    "limitations": "Offline fixture; no semantic acceptance claimed."}
        if name == "world.batch":
            # world.batch is serial per type in this two-entity fixture.
            index = sum(call[0] == name for call in self.calls) - 1
            return {"entities": [deepcopy(self.table["entities"][index])]}
        if name == "world.structure":
            result = {key: deepcopy(self.table[key]) for key in ("relations", "events")}
            count = sum(call[0] == name for call in self.calls)
            if self.always_bad or self.bad_first and count == 1:
                result["events"] = result["events"][:2]
            return result
        raise AssertionError("Unexpected model call: " + name)


class SeedWorldTests(unittest.TestCase):
    def setUp(self):
        self.wp, self.world, self.table = fixture()

    def test_actual_witnesses_and_roundtrip(self):
        report = validate_seed_world(self.wp, self.world)
        self.assertTrue(report["passed"])
        self.assertEqual(report["mechanism_coverage"][0]["events"]["revision"], ["e0", "e1"])
        self.assertEqual(report["mechanism_coverage"][0]["causal_rules"]["review_revision"],
                         [{"trigger": "e1", "effect": "e2"}])
        self.assertTrue(validate_seed_world(self.wp, WorldState.from_dict(self.world.to_dict()))["passed"])

    def test_unseeded_legacy_is_noop(self):
        self.assertEqual(validate_seed_world({}, WorldState())["applied"], False)

    def test_metadata_alone_cannot_satisfy_seed(self):
        self.world.entities = {}
        with self.assertRaises(SeedWorldError):
            validate_seed_world(self.wp, self.world)

    def test_missing_field_timeline_rejected(self):
        del self.world.entities["Yearbook"]["capital"]
        with self.assertRaisesRegex(SeedWorldError, "no canonical values"):
            validate_seed_world(self.wp, self.world)

    def test_deleted_event_and_stale_effect_rejected(self):
        self.world.events.pop()
        with self.assertRaises(SeedWorldError):
            validate_seed_world(self.wp, self.world)

    def test_effect_disagreement_rejected(self):
        self.world.entities["Yearbook"]["profit"].ops[-1].value = "999"
        with self.assertRaisesRegex(SeedWorldError, "canonical witness"):
            validate_seed_world(self.wp, self.world)

    def test_orphan_overwrite_rejected(self):
        self.world.entities["Yearbook"]["profit"].ops.append(
            Op(4, self.world.date_of_session(4), UPDATE, "999", "180"))
        with self.assertRaisesRegex(SeedWorldError, "no relation/event witness"):
            validate_seed_world(self.wp, self.world)

    def test_causal_metadata_must_link_real_events(self):
        self.world.events[-1]["caused_by"] = "missing"
        with self.assertRaisesRegex(SeedWorldError, "no executable caused_by witness"):
            validate_seed_world(self.wp, self.world)

    def test_relation_needs_actual_fk(self):
        self.world.entities["Yearbook"]["issuer"].ops[0].value = "Other"
        with self.assertRaisesRegex(SeedWorldError, "canonical FK witness"):
            validate_seed_world(self.wp, self.world)

    def test_real_entity_types_and_count_required(self):
        self.world.entity_types["Yearbook"] = "company"
        with self.assertRaisesRegex(SeedWorldError, "actual count"):
            validate_seed_world(self.wp, self.world)

    def test_time_and_world_blueprint_cannot_drift(self):
        self.world.n_sessions = 2
        self.world.world_blueprint["evidence_channels"] = []
        report = seed_world_report(self.wp, self.world)
        self.assertFalse(report["passed"])
        self.assertTrue(any("temporal_model" in issue for issue in report["issues"]))
        self.assertTrue(any("evidence_channels" in issue for issue in report["issues"]))

    def test_effect_date_is_checked(self):
        self.world.entities["Yearbook"]["profit"].ops[-1].date = "2099-01-01"
        with self.assertRaises(SeedWorldError):
            validate_seed_world(self.wp, self.world)

    def test_seed_intrinsic_values_are_not_trend_imprinted(self):
        before = deepcopy(self.world.to_dict())
        count = imprint_structure(self.world, log=lambda *args: None, profile={
            "seed_protected_fields": sorted(seed_protected_fields(self.wp))})
        self.assertEqual(count, 0)
        self.assertEqual(self.world.to_dict()["entities"], before["entities"])
        # Prove the fixture would otherwise be eligible for trend manufacture.
        ordinary = WorldState.from_dict(before)
        self.assertEqual(imprint_structure(ordinary, log=lambda *args: None), 1)

    def test_build_repairs_structure_preserves_natural_seed_trajectory(self):
        tracer = FixtureTracer(self.table, bad_first=True)
        world = build_world(self.wp, tracer, log=lambda *args: None)
        self.assertTrue(validate_seed_world(self.wp, world)["passed"])
        self.assertEqual([op.value for op in world.entities["Yearbook"]["capital"].ops],
                         ["100", "110", "120", "130", "140", "150"])
        structure_calls = [messages for name, messages in tracer.calls if name == "world.structure"]
        self.assertEqual(len(structure_calls), 2)
        self.assertIn("真实任务种子约束", structure_calls[-1][-1]["content"])
        self.assertIn("上轮机械校验失败", structure_calls[-1][-1]["content"])

    def test_repair_exhaustion_refuses_world(self):
        with self.assertRaises(WorldBlueprintError):
            build_world(self.wp, FixtureTracer(self.table, always_bad=True), log=lambda *args: None)

    def test_incremental_noop_still_validates_seed(self):
        self.world.events.pop()
        with self.assertRaises(SeedWorldError):
            build_world(self.wp, FixtureTracer(self.table), existing=self.world, log=lambda *args: None)

    def test_shared_roles_accept_same_business_object(self):
        wp, world, _ = fixture(shared_roles=True)
        self.assertTrue(validate_seed_world(wp, world)["passed"])

    def test_shared_roles_reject_cross_object_chain(self):
        wp, world, _ = fixture(shared_roles=True)
        # Both events still have valid typed participants and real effects;
        # only their business-object identity is wrong.
        world.entity_types["Other report"] = "report"
        world.entities["Other report"] = {
            "capital": deepcopy(world.entities["Yearbook"]["capital"]),
            "review": world.entities["Yearbook"].pop("review")}
        world.events[-1]["participants"]["report"] = "Other report"
        world.events[-1]["effects"][0]["entity"] = "Other report"
        with self.assertRaisesRegex(SeedWorldError, "shared_roles mismatch"):
            validate_seed_world(wp, world)

    def test_relation_binding_accepts_actual_issuer(self):
        wp, world, table = fixture(relation_bindings=True)
        self.assertTrue(validate_seed_world(wp, world)["passed"])
        tracer = FixtureTracer(table)
        self.assertTrue(validate_seed_world(wp, build_world(wp, tracer, log=lambda *args: None))["passed"])
        prompt = next(messages[-1]["content"] for name, messages in tracer.calls if name == "world.structure")
        self.assertIn("relation_bindings", prompt)
        self.assertIn("shared_roles", prompt)

    def test_entity_prompts_share_plan_but_keep_intrinsic_only_ownership(self):
        wp, _, table = fixture(shared_roles=True, relation_bindings=True)
        tracer = FixtureTracer(table)
        build_world(wp, tracer, log=lambda *args: None)
        batches = [messages for name, messages in tracer.calls if name == "world.batch"]
        self.assertEqual(len(batches), 2)
        for messages in batches:
            text = "\n".join(message["content"] for message in messages)
            self.assertIn("Track revision and review", text)
            self.assertIn("Revision causes review.", text)
            self.assertIn("本阶段只生成当前类型的内在字段", text)
            self.assertIn("共同作者上下文", text)
            self.assertIn(wp["seed_contract"]["task"]["instructions"], text)
            self.assertNotIn("tests/seed_world_selftest.py", text)
            self.assertNotIn("默认非单调", text)
        # The report entity schema advertises capital, not the event-owned
        # profit/review or the relation-owned issuer fields.
        self.assertIn("capital(numeric)", batches[1][0]["content"])
        self.assertNotIn("profit(numeric)", batches[1][0]["content"])
        self.assertNotIn("review(status)", batches[1][0]["content"])
        structure = next(messages for name, messages in tracer.calls if name == "world.structure")
        self.assertIn("caused_by", structure[-1]["content"])
        self.assertIn("relation_bindings", structure[-1]["content"])

    def test_entity_context_is_type_scoped_and_absent_without_seed(self):
        self.assertEqual(seed_entity_prompt({}, "report"), "")
        contract = deepcopy(self.wp["seed_contract"])
        contract["mechanisms"].append({"description": "UNRELATED_MECHANISM",
                                        "required_entity_types": ["other_type"]})
        contract["task"]["instructions"] = "FULL_TASK_INSTRUCTIONS"
        hint = seed_entity_prompt({"seed_contract": contract}, "report")
        self.assertIn("Revision causes review.", hint)
        self.assertNotIn("UNRELATED_MECHANISM", hint)
        self.assertNotIn("FULL_TASK_INSTRUCTIONS", hint)

    def test_relation_binding_rejects_other_valid_issuer(self):
        wp, world, _ = fixture(relation_bindings=True)
        world.entity_types["Other issuer"] = "company"
        world.entities["Other issuer"] = deepcopy(world.entities["Aster"])
        world.events[-1]["participants"]["issuer"] = "Other issuer"
        with self.assertRaisesRegex(SeedWorldError, "relation binding issued_by"):
            validate_seed_world(wp, world)

    def test_relation_binding_uses_event_time_not_latest_or_future_fk(self):
        wp, world, _ = fixture(relation_bindings=True, temporal_relation=True)
        world.entity_types["Future issuer"] = "company"
        world.entities["Future issuer"] = deepcopy(world.entities["Aster"])
        world.relations.append({"id": "r2", "type": "issued_by", "from": "Yearbook",
                                "to": "Future issuer", "session": 4})
        world.entities["Yearbook"]["issuer"].ops.append(
            Op(4, world.date_of_session(4), UPDATE, "Future issuer", "Aster"))
        # The review at session 3 legitimately refers to Aster even though the
        # latest issuer is now different; a future FK is not available at 3.
        self.assertTrue(validate_seed_world(wp, world)["passed"])
        world.events[-1]["participants"]["issuer"] = "Future issuer"
        with self.assertRaisesRegex(SeedWorldError, "relation binding issued_by"):
            validate_seed_world(wp, world)

    def test_causal_rewiring_uses_matching_business_object(self):
        wp, _, table = fixture(shared_roles=True)
        foreign_parent = deepcopy(table["events"][1])
        foreign_parent["id"] = "a_foreign"
        foreign_parent["participants"]["report"] = "Other report"
        structure = {"events": [foreign_parent] + deepcopy(table["events"])}
        structure["events"][-1]["caused_by"] = "a_foreign"
        self.assertEqual(_wire_declared_causality(structure, wp["world_blueprint"]), 1)
        self.assertEqual(structure["events"][-1]["caused_by"], "e1")
        structure["events"] = [foreign_parent, structure["events"][-1]]
        structure["events"][-1].pop("caused_by")
        self.assertEqual(_wire_declared_causality(structure, wp["world_blueprint"]), 0)
        self.assertNotIn("caused_by", structure["events"][-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
