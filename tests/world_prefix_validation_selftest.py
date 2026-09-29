"""Partial world compilation retains local legality while deferring coverage."""
from copy import deepcopy
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.seed_pack import attach_seed_contract
from pipeline.seed_world import seed_world_issues, seed_world_report, validate_seed_world, SeedWorldError
from pipeline.world_state import assemble_world, Timeline, Op, SET


def fixture():
    blueprint = {
        "version": 1,
        "entity_types": [
            {"id": "company", "noun": "公司", "primary": True, "count": 2, "cardinality_policy": "exact",
             "fields": [{"name": "sector", "kind": "category"}]},
            {"id": "report", "noun": "报告", "count": 1,
             "fields": [{"name": "issuer", "kind": "reference"},
                        {"name": "capital", "kind": "numeric", "range": [0, 1000]},
                        {"name": "profit", "kind": "numeric", "range": [0, 1000]},
                        {"name": "review", "kind": "status", "states": ["unchecked", "checked"]}]}],
        "relation_types": [{"id": "issued_by", "from_type": "report", "to_type": "company",
                            "field": "issuer", "temporal": False, "min_count": 1}],
        "event_types": [
            {"id": "revision", "label": "修订", "roles": {"report": "report"},
             "effect_fields": [{"role": "report", "field": "profit"}], "min_count": 2},
            {"id": "reviewed", "label": "复核", "roles": {"report": "report", "issuer": "company"},
             "effect_fields": [{"role": "report", "field": "review"}], "min_count": 1,
             "relation_bindings": [{"relation": "issued_by", "from_role": "report", "to_role": "issuer"}]}],
        "causal_rules": [{"id": "revision_review", "trigger_event": "revision", "effect_event": "reviewed",
                          "delay_sessions": 1, "shared_roles": ["report"]}],
        "temporal_model": {"unit": "day", "n_sessions": 6, "cadence": "daily", "step_days": 1},
        "evidence_channels": ["报告"]}
    requirements = deepcopy(blueprint); requirements.pop("version")
    requirements["temporal_model"] = {"min_sessions": 6}
    requirements["entity_types"][0]["cardinality_policy"] = "exact"
    pack = {"schema_version": 1, "seed_id": "prefix_fixture", "family": "insurance", "title": "Prefix fixture",
        "description": "Offline generated fixture.",
        "sources": [{"id": "fixture", "path": "tests/world_prefix_validation_selftest.py", "sha256": "0" * 64,
                     "role": "builder_only"}],
        "task": {"objective": "Follow revisions", "instructions": "Preserve their recorded relations."},
        "exemplars": [], "blueprint_requirements": requirements,
        "mechanisms": [{"id": "review", "description": "Revision followed by review.",
            "source_refs": [{"source_id": "fixture", "locator": "fixture"}],
            "required_entity_types": ["company", "report"], "required_relation_types": ["issued_by"],
            "required_event_types": ["revision", "reviewed"], "required_causal_rules": ["revision_review"]}]}
    wp = attach_seed_contract({"world_blueprint": blueprint, "domain_profile": {}}, pack)
    table = {"entities": [
        {"name": "A", "type": "company", "fields": {"sector": {"type": "stable", "value": "insurance"}}},
        {"name": "B", "type": "company", "fields": {"sector": {"type": "stable", "value": "finance"}}},
        {"name": "R", "type": "report", "fields": {"capital": {"type": "stable", "value": "100"}}}],
        "relations": [{"id": "r1", "type": "issued_by", "from": "R", "to": "A", "session": 0}],
        "events": [
            {"id": "e0", "type": "revision", "session": 0, "participants": {"report": "R"},
             "effects": [{"entity": "R", "field": "profit", "set": "200"}]},
            {"id": "e1", "type": "revision", "session": 2, "participants": {"report": "R"},
             "effects": [{"entity": "R", "field": "profit", "set": "180"}]},
            {"id": "e2", "type": "reviewed", "session": 3, "participants": {"report": "R", "issuer": "A"},
             "effects": [{"entity": "R", "field": "review", "set": "checked"}], "caused_by": "e1"}]}
    return wp, table


class PrefixValidationTests(unittest.TestCase):
    def setUp(self):
        self.wp, self.table = fixture()
        self.blueprint = self.wp["world_blueprint"]
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def assemble(self, table, **kw):
        return assemble_world(table, blueprint=self.blueprint, include_shape_diagnostics=False, **kw)

    def prefix(self):
        value = deepcopy(self.table)
        value["entities"].pop(1)
        value["relations"], value["events"] = [], []
        return value

    def test_valid_incomplete_prefix_defers_only_coverage_and_full_mode_rejects(self):
        table = self.prefix()
        before = deepcopy(table)
        world, issues = self.assemble(table, require_complete=False)
        self.assertEqual(issues, [])
        report = seed_world_report(self.wp, world, require_complete=False)
        self.assertTrue(report["passed"], report["issues"])
        self.assertFalse(report["require_complete"])
        self.assertEqual(seed_world_issues(self.wp, world, require_complete=False), [])
        self.assertEqual(table, before)
        _, full_issues = self.assemble(table)
        for category in ("entity type company", "relation type issued_by", "event type revision", "causal rule revision_review"):
            self.assertTrue(any(category in issue for issue in full_issues), full_issues)
        self.assertFalse(seed_world_report(self.wp, world)["passed"])
        with self.assertRaises(SeedWorldError):
            validate_seed_world(self.wp, world)

    def test_complete_default_and_explicit_mode_are_equal(self):
        implicit, issues = assemble_world(self.table, blueprint=self.blueprint)
        explicit, other = assemble_world(self.table, blueprint=self.blueprint, require_complete=True,
                                          include_shape_diagnostics=True)
        self.assertEqual(issues, [])
        self.assertEqual(issues, other)
        self.assertEqual(implicit.to_dict(), explicit.to_dict())
        self.assertTrue(validate_seed_world(self.wp, implicit)["passed"])
        self.assertEqual(seed_world_report(self.wp, implicit), seed_world_report(self.wp, implicit, require_complete=True))

    def test_prefix_bad_relation_target_and_type_remain_errors(self):
        for field, value in (("to", "missing"), ("to", "R"), ("session", 1), ("type", "unknown")):
            with self.subTest(field=field, value=value):
                table = deepcopy(self.table); table["events"] = []
                table["relations"][0][field] = value
                _, issues = self.assemble(table, require_complete=False)
                self.assertTrue(issues)

    def test_prefix_bad_effect_schema_range_and_time_remain_errors(self):
        for mutation in (lambda e: e["effects"][0].update(field="undeclared"),
                         lambda e: e["effects"][0].update(set="2000"),
                         lambda e: e.update(session=6),
                         lambda e: e["participants"].update(report="missing")):
            table = deepcopy(self.table); table["events"] = table["events"][:1]
            mutation(table["events"][0])
            _, issues = self.assemble(table, require_complete=False)
            self.assertTrue(issues)

    def test_prefix_asserted_bad_causal_reference_and_delay_are_rejected(self):
        for parent in ("missing", "e0", "e2"):
            table = deepcopy(self.table); table["events"][-1]["caused_by"] = parent
            _, issues = self.assemble(table, require_complete=False)
            self.assertTrue(any("caused_by" in issue for issue in issues), issues)
            world, clean_issues = self.assemble(self.table)
            self.assertEqual(clean_issues, [])
            world.events[-1]["caused_by"] = parent
            report = seed_world_report(self.wp, world, require_complete=False)
            self.assertFalse(report["passed"])
            self.assertTrue(any("invalid asserted caused_by" in issue for issue in report["issues"]))

    def test_prefix_seed_relation_binding_and_canonical_effects_still_checked(self):
        for mutation in (lambda w: w.events[-1]["participants"].update(issuer="B"),
                         lambda w: setattr(w.entities["R"]["profit"].ops[-1], "value", "777"),
                         lambda w: setattr(w.entities["R"]["profit"].ops[-1], "date", "2099-01-01")):
            world, _ = self.assemble(self.table)
            mutation(world)
            self.assertFalse(seed_world_report(self.wp, world, require_complete=False)["passed"])

    def test_prefix_shared_roles_does_not_accept_foreign_parent(self):
        table = deepcopy(self.table)
        table["entities"].append({"name": "R2", "type": "report", "fields": {
            "capital": {"type": "stable", "value": "150"}}})
        table["events"].append({"id": "foreign_revision", "type": "revision", "session": 2,
            "participants": {"report": "R2"}, "effects": [{"entity": "R2", "field": "profit", "set": "80"}]})
        table["events"][2]["caused_by"] = "foreign_revision"
        world, issues = self.assemble(table, require_complete=False)
        self.assertEqual(issues, [])
        report = seed_world_report(self.wp, world, require_complete=False)
        self.assertFalse(report["passed"])
        self.assertTrue(any("invalid asserted caused_by" in issue for issue in report["issues"]))

    def test_prefix_missing_existing_intrinsic_field_and_exact_overflow_rejected(self):
        table = self.prefix(); del table["entities"][0]["fields"]["sector"]
        world, issues = self.assemble(table, require_complete=False)
        self.assertTrue(any("内在字段" in issue for issue in issues))
        self.assertFalse(seed_world_report(self.wp, world, require_complete=False)["passed"])
        world, _ = self.assemble(self.table)
        world.entities["C"] = deepcopy(world.entities["A"]); world.entity_types["C"] = "company"
        self.assertTrue(any("exact" in issue for issue in seed_world_issues(self.wp, world, require_complete=False)))

    def test_seed_prefix_cannot_hide_invalid_objects_behind_deferred_coverage(self):
        for mutation in (
                lambda w: w.entity_types.update(R="undeclared"),
                lambda w: w.relations.append({"id": "extra", "type": "undeclared", "from": "R", "to": "A", "session": 0}),
                lambda w: w.events.append({"id": "extra", "type": "undeclared", "session": 2, "participants": {}, "effects": []}),
                lambda w: w.entities["R"].update(profit=Timeline([Op(2, w.date_of_session(2), SET, "111")]))):
            world, issues = self.assemble(self.prefix(), require_complete=False)
            self.assertEqual(issues, [])
            mutation(world)
            self.assertFalse(seed_world_report(self.wp, world, require_complete=False)["passed"])

    def test_incremental_prefix_references_prior_entities_without_mutating_existing(self):
        prefix, issues = self.assemble(self.prefix(), require_complete=False)
        self.assertEqual(issues, [])
        before = prefix.to_dict()
        delta = {"entities": [deepcopy(self.table["entities"][1])],
                 "relations": deepcopy(self.table["relations"]), "events": deepcopy(self.table["events"])}
        complete, issues = self.assemble(delta, existing=prefix, require_complete=True)
        self.assertEqual(issues, [])
        self.assertEqual(prefix.to_dict(), before)
        self.assertTrue(validate_seed_world(self.wp, complete)["passed"])

    def test_shape_option_omits_only_trend_diagnostics(self):
        table = deepcopy(self.table)
        table["entities"][-1]["fields"]["capital"] = {"type": "evolving", "trajectory": [
            {"session": 0, "value": "100"}, {"session": 1, "value": "110"}, {"session": 2, "value": "120"}]}
        default, issues = assemble_world(table, blueprint=self.blueprint)
        quiet, quiet_issues = self.assemble(table)
        self.assertTrue(any("数值轨迹单调" in issue for issue in issues))
        self.assertEqual(quiet_issues, [])
        self.assertEqual(default.to_dict(), quiet.to_dict())
        table["entities"][-1]["fields"]["capital"]["trajectory"][-1]["value"] = "2000"
        _, still_invalid = self.assemble(table)
        self.assertTrue(any("越界" in issue for issue in still_invalid))

    def test_shape_option_defers_single_value_evolving_hint(self):
        table = deepcopy(self.table)
        table["entities"][-1]["fields"]["capital"] = {"type": "evolving", "trajectory": [
            {"session": 0, "value": "100"}]}
        default, issues = assemble_world(table, blueprint=self.blueprint)
        quiet, quiet_issues = self.assemble(table)
        self.assertTrue(any("只有 1 个值" in issue for issue in issues))
        self.assertEqual(quiet_issues, [])
        self.assertEqual(default.to_dict(), quiet.to_dict())
        table["entities"][-1]["fields"]["capital"]["trajectory"][0]["value"] = "2000"
        _, invalid = self.assemble(table)
        self.assertTrue(any("越界" in issue for issue in invalid))

    def test_partial_options_are_keyword_only_and_boolean(self):
        with self.assertRaises(TypeError):
            assemble_world({}, "2025-01-06", 7, self.blueprint, None, False)
        for flag in (None, 0, "false"):
            with self.assertRaises(ValueError):
                self.assemble({}, require_complete=flag)
            with self.assertRaises(ValueError):
                seed_world_report(self.wp, self.assemble(self.table)[0], require_complete=flag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
