"""Original build_world request integration checks; local responses only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
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
from pipeline.seed_world import seed_world_prompt, validate_seed_world
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_gen import build_world
from seed_world_selftest import FixtureTracer, fixture


def section(text, marker):
    return json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]


class TableTracer(FixtureTracer):
    def chat_json(self, name, messages, **kwargs):
        if name == "world.batch":
            index = sum(call[0] == name for call in self.calls)
            self.calls.append((name, deepcopy(messages)))
            type_id = list(dict.fromkeys(e["type"] for e in self.table["entities"]))[index]
            return {"entities": deepcopy([e for e in self.table["entities"] if e["type"] == type_id])}
        return super().chat_json(name, messages, **kwargs)


class WorldStructureContextTests(unittest.TestCase):
    def setUp(self):
        # Any accidental real provider call fails this test, including helpers.
        for target, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            mocked = patch.object(target, name, side_effect=AssertionError("Network/API prohibited"))
            mocked.start()
            self.addCleanup(mocked.stop)

    def build(self, *, bad_first=False):
        wp, _, table = fixture()
        tracer = FixtureTracer(table, bad_first=bad_first)
        original = deepcopy((wp, table))
        world = build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual((wp, table), original)
        calls = [messages for step, messages in tracer.calls if step == "world.structure"]
        return wp, table, world, tracer, calls

    def test_actual_request_contains_complete_intrinsic_trajectories_and_calendar(self):
        _, table, world, _, calls = self.build()
        text = calls[0][1]["content"]
        entities = section(text, "【typed entities】")
        self.assertEqual({e["name"]: e["intrinsic_fields"] for e in entities},
                         {e["name"]: e["fields"] for e in table["entities"]})
        self.assertTrue(all(e["initial_state"] == {} for e in entities))
        self.assertEqual(section(text, "【session 对应实际日期】"),
                         [{"session": s, "date": world.date_of_session(s)}
                          for s in range(world.n_sessions)])
        self.assertEqual(section(text, "【已有 canonical 结构；空数组表示没有已有实例】"),
                         {key: [] for key in ("relations", "events", "cascades", "absent_fields")})

    def test_actual_request_has_frozen_business_context_without_private_source_records(self):
        wp, _, _, tracer, calls = self.build()
        text = calls[0][1]["content"]
        context = section(text, "【已冻结的任务与业务机制背景】")
        self.assertEqual(context["task"], wp["seed_contract"]["task"])
        self.assertEqual(context["mechanisms"][0]["description"], "Revision causes review.")
        self.assertNotIn("source_refs", context["mechanisms"][0])
        self.assertNotIn("tests/seed_world_selftest.py", text)
        self.assertNotIn('"sources"', text)
        self.assertNotIn('"exemplars"', text)
        # Shared task context is wider; writable field ownership stays narrow.
        for step, messages in tracer.calls:
            if step == "world.batch":
                self.assertIn(wp["seed_contract"]["task"]["instructions"], messages[1]["content"])

    def test_business_projection_is_allowlisted_and_not_truncated(self):
        wp, _, _ = fixture()
        contract = wp["seed_contract"]
        contract["task"]["instructions"] = "完整业务说明\n" * 200
        contract["task"]["rubric"] = "PRIVATE_RUBRIC_SENTINEL"
        contract["mechanisms"][0]["private_note"] = "PRIVATE_OPINION_SENTINEL"
        text = seed_world_prompt(wp)
        context = section(text, "【已冻结的任务与业务机制背景】")
        self.assertEqual(context["task"]["instructions"], contract["task"]["instructions"])
        self.assertNotIn("PRIVATE_", text)
        self.assertEqual(seed_world_prompt({}), "")

    def test_bounded_structure_repair_keeps_same_fact_and_task_context(self):
        _, _, world, tracer, calls = self.build(bad_first=True)
        self.assertEqual(len(calls), 2)
        for marker in ("【typed entities】", "【session 对应实际日期】", "【已冻结的任务与业务机制背景】"):
            self.assertEqual(section(calls[0][1]["content"], marker),
                             section(calls[1][1]["content"], marker))
        self.assertIn("上轮机械校验失败", calls[1][1]["content"])
        self.assertEqual([step for step, _ in tracer.calls],
                         ["world.joint_plan", "world.batch", "world.batch", "world.structure", "world.structure"])
        self.assertEqual(len(world.relations), 1)  # No code-created switch for difficulty.

    def test_minimum_is_not_a_cap_and_extra_valid_relation_keeps_existing_schema_checks(self):
        wp, _, table = fixture(temporal_relation=True)
        wp["world_blueprint"]["entity_types"][0]["count"] = 2
        table["entities"].insert(1, {"name": "Beryl", "type": "company", "fields": {
            "sector": {"type": "stable", "value": "insurance"}}})
        table["relations"].append({"id": "r2", "type": "issued_by", "from": "Yearbook",
                                   "to": "Beryl", "session": 4})
        tracer = TableTracer(table)
        world = build_world(wp, tracer, log=lambda *_: None)
        self.assertTrue(validate_seed_world(wp, world)["passed"])
        self.assertEqual(len(world.relations), 2)
        system = next(messages[0]["content"] for step, messages in tracer.calls if step == "world.structure")
        self.assertIn("这是最低数量，不是恰好数量或上限", system)
        self.assertNotIn("恰好生成 min_count", system)
        self.assertEqual(world.timeline("Yearbook", "issuer").value_at_session(4), "Beryl")

    def test_minimum_violation_still_exhausts_original_three_structure_attempts(self):
        wp, _, table = fixture()
        tracer = FixtureTracer(table, always_bad=True)
        with self.assertRaises(WorldBlueprintError):
            build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual(sum(step == "world.structure" for step, _ in tracer.calls), 3)

    def test_augment_request_has_full_existing_operations_and_canonical_structure(self):
        wp, existing, table = fixture()
        wp["world_blueprint"]["entity_types"][0]["count"] = 2
        existing.conflicts = [{"private": "CONFLICT_SIDECHANNEL_SENTINEL"}]
        existing.generation_diagnostics = [{"private": "DIFFICULTY_HINT_SENTINEL"}]
        original = deepcopy(existing.to_dict())
        table["entities"] = [{"name": "Beryl", "type": "company", "fields": {
            "sector": {"type": "stable", "value": "insurance"}}}]
        tracer = TableTracer(table)
        build_world(wp, tracer, existing=existing, log=lambda *_: None)
        text = next(messages[1]["content"] for step, messages in tracer.calls if step == "world.structure")
        entities = section(text, "【typed entities】")
        self.assertEqual({e["name"]: e["canonical_fields"] for e in entities if "canonical_fields" in e},
                         original["entities"])
        self.assertEqual(section(text, "【已有 canonical 结构；空数组表示没有已有实例】"),
                         {key: original[key] for key in ("relations", "events", "cascades", "absent_fields")})
        self.assertEqual(next(e for e in entities if e["name"] == "Yearbook")["initial_state"],
                         {"profit": "200"})
        self.assertNotIn("SIDECHANNEL_SENTINEL", text)
        self.assertNotIn("DIFFICULTY_HINT_SENTINEL", text)
        self.assertEqual({e: existing.to_dict()["entities"][e] for e in original["entities"]},
                         original["entities"])

    @unittest.skipUnless((ROOT / "output/runs/bc_insurance_seed_world_v1_20260918/prompts.jsonl").exists(),
                         "Optional real local input absent")
    def test_real_insurance_responses_replayed_only_to_capture_new_request_not_quality(self):
        case = ROOT / "output/runs/bc_insurance_seed_world_v1_20260918"
        paths = [case / "01_whitepaper.json", case / "prompts.jsonl"]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        wp = json.loads(paths[0].read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
        rows = [row for row in rows if row["step"] in ("world.batch", "world.structure")]
        class Replay:
            def __init__(self): self.calls = []
            def chat_json(self, step, messages, **kwargs):
                if step == "world.joint_plan":
                    self.calls.append((step, deepcopy(messages)))
                    return {"plan": "Offline synthetic planning input for old response replay.",
                            "limitations": "Not a real model plan or quality improvement evidence."}
                row = rows[sum(s != "world.joint_plan" for s, _ in self.calls)]
                self.calls.append((step, deepcopy(messages)))
                if step != row["step"]:
                    raise AssertionError("Unexpected repair or extra call")
                return deepcopy(row["output"])
        tracer = Replay()
        build_world(wp, tracer, log=lambda *_: None)
        text = next(messages[1]["content"] for step, messages in tracer.calls if step == "world.structure")
        self.assertEqual({e["name"]: e["intrinsic_fields"] for e in section(text, "【typed entities】")},
                         {e["name"]: e["fields"] for row in rows if row["step"] == "world.batch"
                          for e in row["output"]["entities"]})
        self.assertEqual(section(text, "【已冻结的任务与业务机制背景】")["task"], wp["seed_contract"]["task"])
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths], before)
        # Old raw responses are not a new semantic improvement experiment.


if __name__ == "__main__":
    unittest.main(verbosity=2)
