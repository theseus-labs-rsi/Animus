"""Opt-in L7 author-owned numeric histories; no model or fixture mutations."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from pipeline.lines.L7_consolidation import ConsolidationLine
from pipeline.world_state import WorldState, Timeline, Op


def world(values, *, kind="numeric", sessions=None):
    sessions = sessions or list(range(len(values)))
    return WorldState(
        entities={"A": {"score": Timeline([
            Op(session, f"2025-01-{index + 1:02d}", "SET" if index == 0 else "UPDATE",
               str(value), None if index == 0 else str(values[index - 1]))
            for index, (session, value) in enumerate(zip(sessions, values))])}},
        n_sessions=max(sessions) + 1,
        entity_types={"A": "case"},
        world_blueprint={"entity_types": [{"id": "case", "fields": [{"name": "score", "kind": kind}]}]})


NEW = {"supply_plan": {"version": 2}, "domain_profile": {"supply_policy": "capacity_contract/v2"}}


class NativeTrendTests(unittest.TestCase):
    def setUp(self):
        self.line = ConsolidationLine()

    def trends(self, ws, wp=NEW):
        return [order for order in self.line.enumerate(ws, 100, wp)
                if order["aux"]["sub"] == "S1_trend"]

    def test_observed_history_and_gold(self):
        ws = world([10, 20, 30, 40, 35])
        before = deepcopy(ws.to_dict())
        orders = self.trends(ws)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["gt"], "上升")
        self.assertEqual(orders[0]["evidence_sessions"], list(range(5)))
        self.assertEqual(self.line.gt(ws, orders[0]), "上升")
        self.assertEqual(self.line.well_posed(orders[0], ws), ("well_posed", ""))
        self.assertEqual(ws.to_dict(), before)
        self.assertEqual(getattr(ws, "_trended_fields", []), [])

    def test_legacy_remains_planted_only(self):
        ws = world([10, 20, 30, 40, 35])
        self.assertEqual(self.trends(ws, {}), [])
        self.assertFalse(self.line.feasible(ws, {})[0])
        self.assertTrue(self.line.feasible(ws, NEW["domain_profile"])[0])
        ws._trended_fields = [("A", "score")]
        old = self.trends(ws, {})
        self.assertEqual(len(old), 1)
        self.assertNotIn("supply_origin", old[0]["aux"])
        self.assertEqual(self.trends(ws), old)

    def test_declared_field_and_numeric_values_required(self):
        self.assertEqual(self.trends(world([10, 20, 30, 25], kind="text")), [])
        self.assertEqual(self.trends(world([10, 20, "unknown", 40, 35])), [])

    def test_noise_and_short_sequence(self):
        for values in ([10, 20, 15, 11], [10, 10, 10, 10], [10, 20, 15]):
            self.assertEqual(self.trends(world(values)), [])

    def test_local_direction_shortcut(self):
        self.assertEqual(self.trends(world([10, 20, 30, 40, 50])), [])
        # Preserve the original rule: a reverse middle segment qualifies too.
        self.assertEqual(len(self.trends(world([50, 40, 58, 45, 62]))), 1)

    def test_complete_temporal_sequence(self):
        self.assertEqual(self.trends(world([10, 20, 30, 25], sessions=[0, 0, 1, 1])), [])
        ws = world([10, 20, 30, 25])
        ws.entities["A"]["score"].ops.append(Op(4, "2025-01-05", "EXPIRE"))
        self.assertEqual(self.trends(ws), [])

    def test_generation_contract_marker(self):
        ws = world([10, 20, 30, 25])
        contract = {"generation_contract": {"version": "supply-driven/v3", "material_first": True}}
        self.assertEqual(len(self.trends(ws, contract)), 1)
        self.assertEqual(self.trends(ws, {"supply_plan": {"version": 1}}), [])

    def test_stale_native_order_revalidates_history(self):
        ws = world([10, 20, 30, 25])
        order = self.trends(ws)[0]
        ws.world_blueprint["entity_types"][0]["fields"][0]["kind"] = "text"
        self.assertEqual(self.line.well_posed(order, ws)[0], "drop")

    def test_actual_showcase_world(self):
        source = Path(__file__).resolve().parents[1] / "output/showcase_quality_audit_20260926/run"
        if not (source / "02_world.json").exists():
            self.skipTest("Historical Showcase world is not bundled in clean clones")
        raw = (source / "02_world.json").read_bytes()
        ws = WorldState.from_dict(json.loads(raw))
        wp = json.loads((source / "01_whitepaper.json").read_bytes())
        before = deepcopy(ws.to_dict())
        old = self.line.enumerate(ws, 100, wp)
        self.assertEqual(len(old), 2)
        upgraded = deepcopy(wp)
        upgraded["supply_plan"]["version"] = 2
        upgraded["domain_profile"]["supply_policy"] = "capacity_contract/v2"
        new = self.line.enumerate(ws, 100, upgraded)
        self.assertEqual(len(new), 3)
        actual = [order for order in new if order["aux"]["sub"] == "S1_trend"]
        self.assertEqual([(o["entity"], o["field"], o["gt"]) for o in actual],
                         [("沈砚青", "信任程度", "上升")])
        self.assertEqual([o for o in new if o["aux"]["sub"] == "S2_compare"], old)
        self.assertTrue(all(self.line.well_posed(order, ws)[0] == "well_posed" for order in new))
        self.assertEqual(ws.to_dict(), before)
        self.assertEqual((source / "02_world.json").read_bytes(), raw)
        capacity = self.line.native_trend_capacity(ws)
        self.assertEqual(capacity["eligible"], 1)
        self.assertEqual(len(capacity["fields"]), 14)
        print("Actual Showcase L7: legacy=2, native=3; new S1=沈砚青.信任程度 [50,40,58,45,62] -> 上升")


if __name__ == "__main__":
    unittest.main()
