"""Original typed entity-author routing; local replies and no network."""
from copy import deepcopy
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import config
from pipeline.world_gen import build_world, _intrinsic_proposal_issues
from pipeline.world_blueprint import WorldBlueprintError
from seed_world_selftest import FixtureTracer, fixture


class Script(FixtureTracer):
    def __init__(self, table, batches):
        super().__init__(table)
        self.batches = deepcopy(batches)

    def chat_json(self, step, messages, **kwargs):
        if step != "world.batch":
            return super().chat_json(step, messages, **kwargs)
        self.calls.append((step, deepcopy(messages)))
        if not self.batches:
            raise AssertionError("Unexpected additional entity-author call")
        return self.batches.pop(0)


class IntrinsicRetryTests(unittest.TestCase):
    def setUp(self):
        for target, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(target, name, side_effect=AssertionError("Network prohibited"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, self.expected, self.table = fixture()
        self.good = deepcopy(self.table["entities"][0])
        self.report = deepcopy(self.table["entities"][1])
        self.bad = deepcopy(self.good)
        self.bad["fields"]["sector"] = {"type": "insurance"}

    def build(self, batches):
        tracer = Script(self.table, batches)
        draft = {}
        world = build_world(self.wp, tracer, log=lambda *_: None, draft_out=draft)
        return world, tracer, draft

    def test_real_legal_value_in_type_shape_is_rejected_without_value_invention(self):
        entity = {"name": "青禾数码服务社场地备案证明副本", "type": "permit_doc",
                  "fields": {"证照类别": {"type": "备案证明"}}}
        bp = deepcopy(self.wp["world_blueprint"])
        bp["entity_types"].append({"id": "permit_doc", "noun": "证照材料",
                                   "count": 1, "fields": [{"name": "证照类别", "kind": "category"}]})
        original = deepcopy(entity)
        issues = _intrinsic_proposal_issues(entity, bp)
        self.assertTrue(any("缺少本类型内在字段" in x for x in issues))
        self.assertEqual(entity, original)

    def test_same_author_gets_raw_bad_field_and_error_then_reuses_same_identity(self):
        world, tracer, draft = self.build([
            {"entities": [self.bad]}, {"entities": [self.good]}, {"entities": [self.report]}])
        for key in ("entities", "relations", "events", "entity_types"):
            self.assertEqual(world.to_dict()[key], self.expected.to_dict()[key])
        batches = [m for s, m in tracer.calls if s == "world.batch"]
        self.assertEqual(len(batches), 3)
        self.assertIn('"type": "insurance"', batches[1][1]["content"])
        self.assertIn("缺少本类型内在字段", batches[1][1]["content"])
        self.assertEqual(sum(s == "world.structure" for s, _ in tracer.calls), 1)
        self.assertEqual(draft["intrinsic_rejections"][0]["proposal"], self.bad)

    def test_missing_required_field_uses_same_feedback_loop(self):
        bad = deepcopy(self.good)
        bad["fields"] = {}
        _, tracer, draft = self.build([
            {"entities": [bad]}, {"entities": [self.good]}, {"entities": [self.report]}])
        self.assertIn("sector", draft["intrinsic_rejections"][0]["issues"][0])
        self.assertEqual(sum(s == "world.structure" for s, _ in tracer.calls), 1)

    def test_exhaustion_is_four_original_author_rounds_and_zero_structure_calls(self):
        tracer = Script(self.table, [{"entities": [self.bad]}] * 4)
        draft = {}
        with self.assertRaisesRegex(WorldBlueprintError, "4 轮"):
            build_world(self.wp, tracer, log=lambda *_: None, draft_out=draft)
        self.assertEqual(sum(s == "world.batch" for s, _ in tracer.calls), 4)
        self.assertFalse(any(s == "world.structure" for s, _ in tracer.calls))
        self.assertEqual(len(draft["intrinsic_rejections"]), 4)

    def test_accepted_sibling_is_kept_and_only_missing_capacity_requested(self):
        self.wp["world_blueprint"]["entity_types"][0]["count"] = 2
        good2 = deepcopy(self.good)
        good2["name"] = "Beryl"
        bad2 = deepcopy(good2)
        bad2["fields"]["sector"] = {"type": "insurance"}
        world, tracer, draft = self.build([
            {"entities": [self.good, bad2]}, {"entities": [good2]}, {"entities": [self.report]}])
        self.assertEqual(set(world.entities), {"Aster", "Beryl", "Yearbook"})
        self.assertEqual(draft["merged"]["entities"][:2], [self.good, good2])
        batches = [m for s, m in tracer.calls if s == "world.batch"]
        self.assertIn('"Aster"', batches[1][1]["content"])
        self.assertEqual(sum(s == "world.structure" for s, _ in tracer.calls), 1)

    def test_valid_evolving_field_and_event_owned_omissions_are_not_rejected(self):
        self.assertEqual(_intrinsic_proposal_issues(self.report, self.wp["world_blueprint"]), [])
        world, tracer, draft = self.build([{"entities": [self.good]}, {"entities": [self.report]}])
        self.assertEqual(world.timeline("Yearbook", "capital").set_values(),
                         self.expected.timeline("Yearbook", "capital").set_values())
        self.assertNotIn("intrinsic_rejections", draft)
        self.assertEqual(sum(s == "world.batch" for s, _ in tracer.calls), 2)

    def test_existing_numeric_contract_error_routes_to_intrinsic_author(self):
        bad = deepcopy(self.report)
        bad["fields"]["capital"] = {"type": "stable", "value": "not a number"}
        self.assertTrue(any("numeric" in x for x in _intrinsic_proposal_issues(bad, self.wp["world_blueprint"])))

    def test_provider_execution_failure_is_not_retried_as_a_bad_entity(self):
        tracer = Script(self.table, [{"__error__": "provider stopped"}])
        with self.assertRaisesRegex(WorldBlueprintError, "调用失败"):
            build_world(self.wp, tracer, log=lambda *_: None)
        self.assertEqual(sum(s == "world.batch" for s, _ in tracer.calls), 1)
        self.assertFalse(any(s == "world.structure" for s, _ in tracer.calls))


if __name__ == "__main__":
    unittest.main()
