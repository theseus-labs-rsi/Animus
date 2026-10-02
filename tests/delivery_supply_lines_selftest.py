"""Opt-in L3/L7 supply from disjoint frozen-world evidence, without model calls."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.delivery_target import DeliveryTarget
from pipeline.lines.L3_process import ProcessLine, enumerate_l3_orders
from pipeline.lines.L7_consolidation import ConsolidationLine
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, EXPIRE


def paper():
    return {"delivery_target": DeliveryTarget.from_dict({
        "version": 1, "final_questions": 20, "count_stage": "selection_complete",
        "requested_lines": ["L3_process", "L7_consolidation"], "per_line_min": {}, "per_line_max": {},
        "corpus_tokens": 10000, "tokenizer": "offline@1", "max_supply_rounds": 2}).to_dict()}


def stamp(session):
    return (date(2025, 1, 6) + timedelta(days=7 * session)).isoformat()


def history(values):
    return Timeline([Op(session, stamp(session), SET if index == 0 else UPDATE, value,
                        values[index - 1][1] if index else None)
                     for index, (session, value) in enumerate(values)])


def episodes():
    fields = {name: history([(0, name + "-初始"), (index + 1, name + "-首轮"), (index + 4, name + "-次轮")])
              for index, name in enumerate(("阶段", "记录", "负责人"))}
    events = []
    for index, field in enumerate(("阶段", "记录", "负责人", "阶段", "记录", "负责人")):
        row = {"id": f"e{index + 1}", "type": "change", "session": index + 1,
               "participants": {"subject": "对象甲"},
               "effects": [{"entity": "对象甲", "field": field, "set": field + ("-首轮" if index < 3 else "-次轮")}]}
        if index not in (0, 3):
            row["caused_by"] = f"e{index}"
        events.append(row)
    return WorldState({"对象甲": fields}, n_sessions=7, entity_types={"对象甲": "case"}, events=events)


def changes(count, prefix):
    return history([(session, f"{prefix}{session}") for session in range(count + 1)])


def cohort_world():
    values = {"甲": 5, "乙": 1, "丙": 4, "丁": 1}
    entities = {name: {"状态": changes(count, name), "归属": history([(0, "组一" if name in ("甲", "乙") else "组二")])}
                for name, count in values.items()}
    entities.update({"组一": {}, "组二": {}})
    return WorldState(entities, n_sessions=6,
        entity_types={name: "case" if name in values else "group" for name in entities},
        world_blueprint={"entity_types": [{"id": "case", "fields": [{"name": "归属", "kind": "reference"}, {"name": "状态", "kind": "category"}]},
                                         {"id": "group", "fields": []}]})


class DeliverySupplyLinesTests(unittest.TestCase):
    def test_l3_two_real_episodes_supply_two_disjoint_orders(self):
        world = episodes()
        before = deepcopy(world.to_dict())
        line = ProcessLine()
        self.assertEqual(len(line.enumerate(world)), 1)
        orders = line.enumerate(world, wp=paper())
        self.assertEqual(len(orders), 2)
        self.assertEqual([o["aux"]["supply_episode"]["root_event_id"] for o in orders], ["e1", "e4"])
        self.assertEqual([o["evidence_sessions"] for o in orders], [[1, 2, 3], [4, 5, 6]])
        for order in orders:
            self.assertEqual(line.well_posed(order, world), ("well_posed", ""))
            self.assertEqual(line.gt(world, order), order["gt"])
            self.assertNotIn("e1", line.intent(order)[0])
        self.assertEqual(world.to_dict(), before)
        self.assertEqual(len(line.enumerate(world, target=1, wp=paper())), 1)
        self.assertEqual(line.enumerate(world, target=0, wp=paper()), [])

    def test_l3_long_single_episode_is_one_task(self):
        world = episodes()
        world.events[3]["caused_by"] = "e3"
        orders = ProcessLine().enumerate(world, target=1000, wp=paper())
        self.assertEqual(len(orders), 1)
        self.assertGreaterEqual(len(orders[0]["gt"]), 3)
        self.assertLessEqual(len(orders[0]["gt"]), 4)

    def test_l3_unlinked_changes_keep_original_single_group(self):
        world = episodes()
        for event in world.events:
            event.pop("caused_by", None)
        line = ProcessLine()
        self.assertEqual(line.enumerate(world, wp=paper()), line.enumerate(world))
        world.events = []
        self.assertEqual(line.enumerate(world, wp=paper()), line.enumerate(world))

    def test_l3_invented_effect_and_repeated_values_add_no_episode(self):
        world = episodes()
        world.events[4]["effects"][0]["set"] = "并未发生的变动"
        orders = ProcessLine().enumerate(world, wp=paper())
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["evidence_sessions"], [1, 2, 3])
        world = episodes()
        world.entities["对象甲"]["记录"].ops[-1].value = "记录-首轮"
        world.events[4]["effects"][0]["set"] = "记录-首轮"
        self.assertEqual(len(ProcessLine().enumerate(world, wp=paper())), 1)

    def test_l3_card_reordering_and_world_changes_fail_original_gate(self):
        world = episodes()
        line = ProcessLine()
        order = line.enumerate(world, wp=paper())[1]
        invalid = deepcopy(order)
        invalid["gt"] = list(reversed(invalid["gt"]))
        self.assertEqual(line.well_posed(invalid, world)[0], "drop")
        world.entities["对象甲"]["阶段"].ops[-1].value = "新事实"
        self.assertEqual(line.well_posed(order, world)[0], "drop")

    def test_l3_missing_opt_in_preserves_original_payload(self):
        world = episodes()
        expected = [{"line": "L3_process", "capability": "L3_order", "entity": o.entity, "field": o.field,
                     "gt": o.gt, "evidence_sessions": o.evidence_sessions, "aux": o.aux}
                    for o in enumerate_l3_orders(world)]
        self.assertEqual(ProcessLine().enumerate(world, wp={}), expected)

    def test_l7_reference_partition_yields_independent_groups(self):
        world = cohort_world()
        before = deepcopy(world.to_dict())
        line = ConsolidationLine()
        original = line.enumerate(world)
        self.assertEqual(len(original), 1)
        orders = line.enumerate(world, wp=paper())
        self.assertEqual(len(orders), 2)
        self.assertEqual({o["gt"] for o in orders}, {"甲", "丙"})
        candidate_sets = [set(o["aux"]["candidates"]) for o in orders]
        self.assertFalse(candidate_sets[0] & candidate_sets[1])
        for order in orders:
            self.assertEqual(line.gt(world, order), order["gt"])
            self.assertEqual(line.well_posed(order, world), ("well_posed", ""))
            intent = line.intent(order)[0]
            self.assertTrue(all(name in intent for name in order["aux"]["candidates"]))
            self.assertEqual(order["aux"]["supply_cohort"]["basis"], "stable_reference")
        self.assertEqual(world.to_dict(), before)

    def test_l7_types_are_natural_independent_groups(self):
        world = cohort_world()
        for name in ("甲", "乙", "丙", "丁"):
            world.entities[name].pop("归属")
            world.entity_types[name] = "case_A" if name in ("甲", "乙") else "case_B"
        orders = ConsolidationLine().enumerate(world, wp=paper())
        self.assertEqual(len(orders), 2)
        self.assertEqual({o["aux"]["supply_cohort"]["entity_type"] for o in orders}, {"case_A", "case_B"})

    def test_l7_dynamic_or_undeclared_owner_does_not_create_partition(self):
        world = cohort_world()
        world.entities["乙"]["归属"] = history([(0, "组一"), (3, "组二")])
        self.assertEqual(len([c for c in ConsolidationLine._natural_cohorts(world) if len(c["members"]) >= 2]), 2)
        # The two groups above are the original case/group types; no owner split.
        self.assertTrue(all(c["basis"] == "entity_type" for c in ConsolidationLine._natural_cohorts(world)))
        world = cohort_world()
        world.world_blueprint["entity_types"][0]["fields"][0]["kind"] = "text"
        self.assertTrue(all(c["basis"] == "entity_type" for c in ConsolidationLine._natural_cohorts(world)))
        world = cohort_world()
        world.entities["乙"]["归属"].ops.append(Op(3, stamp(3), EXPIRE, None, "组一"))
        self.assertTrue(all(c["basis"] == "entity_type" for c in ConsolidationLine._natural_cohorts(world)))

    def test_l7_one_cohort_does_not_multiply_combinations(self):
        world = cohort_world()
        for name in ("丙", "丁"):
            world.entities[name]["归属"] = history([(0, "组一")])
        orders = ConsolidationLine().enumerate(world, target=1000, wp=paper())
        self.assertEqual(len(orders), 1)

    def test_l7_independent_trend_fields_keep_existing_full_span_truth(self):
        world = WorldState({"对象": {"金额": history([(0, "10"), (1, "25"), (2, "40"), (3, "30")]),
                                     "耗时": history([(0, "40"), (1, "25"), (2, "10"), (3, "20")])}}, n_sessions=4)
        world._trended_fields = [("对象", "金额"), ("对象", "耗时")]
        line = ConsolidationLine()
        self.assertEqual(len(line.enumerate(world)), 1)
        self.assertEqual(len(line.enumerate(world, wp=paper())), 2)
        for order in line.enumerate(world, wp=paper()):
            self.assertEqual(line.gt(world, order), order["gt"])
            self.assertEqual(line.well_posed(order, world), ("well_posed", ""))
            self.assertEqual(order["evidence_sessions"], [0, 1, 2, 3])
        world._trended_fields = []
        self.assertEqual(line.enumerate(world, wp=paper()), [])

    def test_l7_original_gate_rejects_tampered_winner(self):
        world = cohort_world()
        line = ConsolidationLine()
        order = line.enumerate(world, wp=paper())[0]
        order["gt"] = "不在候选中的对象"
        self.assertEqual(line.well_posed(order, world)[0], "drop")

    def test_invalid_delivery_contract_does_not_silently_enable(self):
        for line, world in ((ProcessLine(), episodes()), (ConsolidationLine(), cohort_world())):
            with self.assertRaises(ValueError):
                line.enumerate(world, wp={"delivery_target": {"final_questions": 500}})


if __name__ == "__main__":
    unittest.main()
