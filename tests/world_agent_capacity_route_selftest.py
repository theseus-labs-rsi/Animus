"""Actual world-agent actions route quantity requests to the capacity author."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from pipeline import supply
from pipeline.blueprint_feasibility import BlueprintReviewRequested
from pipeline.supply_capacity import CapacityReviewRequested
from pipeline.world_agent import generate_world
from seed_world_selftest import fixture
from supply_pipeline_selftest import target
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part

REQUEST = {"action": "request_blueprint_review", "reason": "Observed candidate shortage requires more independent carriers",
           "suggestion": "Reassess the declared object counts against line reserves"}


class CapacityRouteTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network/model"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, self.world, self.table = fixture()
        self.wp["active_lines"] = [{"line": "L1_timeline", "weight": 1}, {"line": "L3_process", "weight": 1}]
        self.wp["line_mapping"] = [{"line": line, "applicable": True, "instantiation": "Actual field histories"}
                                   for line in target()["requested_lines"]]
        supply.attach_plan(self.wp, target(), 10)
        self.full = table_part(self.table, [0, 1], True)

    def execute(self, version, script):
        wp = deepcopy(self.wp)
        wp["supply_plan"]["version"] = version
        before = deepcopy(wp)
        tracer = ScriptedTracer(script)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "world.json"
            with self.assertRaises(BlueprintReviewRequested) as raised:
                generate_world(wp, tracer, checkpoint_path=checkpoint, log=lambda *_: None)
            state = json.loads(checkpoint.read_text("utf-8"))
        self.assertEqual(wp, before)
        self.assertFalse(tracer.script)
        self.assertEqual(state["status"], "blueprint_review_requested")
        self.assertEqual(state["upstream_request"], raised.exception.evidence)
        return raised.exception, state, tracer

    def shortage_script(self):
        return [(PLAN, write("one")), (WRITE, self.full), (PLAN, FINISH), (PLAN, REQUEST)]

    def test_v2_explicit_supply_request_uses_capacity_route(self):
        error, state, tracer = self.execute(2, self.shortage_script())
        self.assertIs(type(error), CapacityReviewRequested)
        self.assertIn("supply_shortfall", error.evidence["last_observation"])
        self.assertGreater(state["supply_observations"][-1]["shortfall"], 0)
        self.assertEqual(len(state["units"]), 1)
        self.assertEqual(state["units"][0]["raw"], self.full)
        self.assertEqual(tracer.calls[-1]["input"]["observation"], error.evidence["last_observation"])

    def test_v1_same_supply_feedback_keeps_original_route(self):
        error, state, _ = self.execute(1, self.shortage_script())
        self.assertIs(type(error), BlueprintReviewRequested)
        self.assertIn("supply_shortfall", error.evidence["last_observation"])
        self.assertEqual(len(state["units"]), 1)

    def test_v2_without_supply_observation_keeps_business_route(self):
        error, state, _ = self.execute(2, [(PLAN, REQUEST)])
        self.assertIs(type(error), BlueprintReviewRequested)
        self.assertIsNone(error.evidence["last_observation"])
        self.assertEqual(state["units"], [])

    def test_v2_rejected_business_proposal_keeps_business_route(self):
        bad = deepcopy(self.full)
        bad["entities"][1]["fields"]["review"] = {"type": "stable", "value": "confirmed"}
        error, state, _ = self.execute(2, [(PLAN, write("bad")), (WRITE, bad), (PLAN, REQUEST)])
        self.assertIs(type(error), BlueprintReviewRequested)
        self.assertIn("rejected_proposal", error.evidence["last_observation"])
        self.assertEqual(state["units"], [])
        self.assertTrue(error.evidence["prior_failures"])


if __name__ == "__main__":
    unittest.main()
