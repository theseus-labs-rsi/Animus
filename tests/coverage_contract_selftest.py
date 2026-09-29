"""Offline regressions for real relation edges, supply and held-out leakage."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.lines import run_lines
from pipeline.lines.L2_relational import relation_path_metrics, RelationalLine
from pipeline.lines.L7_consolidation import ConsolidationLine
from pipeline.lines.L9_induction import InductionLine, heldout_leakage
from pipeline.question_contract import attach_question_contract, validate_question
from pipeline.render import phrase_questions
from pipeline.grounding import run_grounding
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE


def series(values):
    return Timeline([Op(i, f"2025-01-{i + 1:02d}", SET if i == 0 else UPDATE, value,
                        values[i - 1] if i else None) for i, value in enumerate(values)])


class CoverageTests(unittest.TestCase):
    def test_identity_field_does_not_add_a_relation_edge(self):
        ws = WorldState({"组甲": {"名称": series(["组甲"]), "负责人": series(["成员乙"])},
                         "成员乙": {"职责": series(["审阅"]), "上级": series(["成员丙"])},
                         "成员丙": {"职责": series(["核准"])}}, n_sessions=1)
        result = relation_path_metrics(ws, "组甲", ["名称", "负责人", "职责"], 0)
        self.assertEqual((result["field_reads"], result["relation_hops"], result["identity_reads"], result["distinct_entities"]), (3, 1, 1, 2))
        self.assertEqual(result["difficulty_basis"], "legacy_field_reads")
        genuine = relation_path_metrics(ws, "组甲", ["负责人", "上级", "职责"], 0)
        self.assertEqual((genuine["field_reads"], genuine["relation_hops"], genuine["identity_reads"]), (3, 2, 0))

    def test_typed_reference_declaration_is_reported_separately_from_resolved_edges(self):
        ws = WorldState({"组甲": {"标签": series(["成员乙"])}, "成员乙": {"职责": series(["审阅"])}},
                        n_sessions=1, entity_types={"组甲": "team", "成员乙": "person"},
                        world_blueprint={"entity_types": [{"id": "team", "fields": [{"name": "标签", "kind": "text"}]}]})
        metrics = relation_path_metrics(ws, "组甲", ["标签", "职责"], 0)
        self.assertEqual((metrics["relation_hops"], metrics["declared_reference_hops"], metrics["undeclared_relation_hops"]), (1, 0, 1))
        ws.world_blueprint["entity_types"][0]["fields"][0]["kind"] = "reference"
        metrics = relation_path_metrics(ws, "组甲", ["标签", "职责"], 0)
        self.assertEqual((metrics["relation_hops"], metrics["declared_reference_hops"], metrics["undeclared_relation_hops"]), (1, 1, 0))

    def test_relation_numeric_gold_preserves_percent_dimension(self):
        ws = WorldState({"组甲": {"关联": series(["报告乙"])}, "报告乙": {"系数": series(["0.5"])}}, n_sessions=1)
        order = {"line": "L2_relational", "capability": "L2_multihop", "entity": "组甲", "field": "关联→系数",
                 "gt": "0.5%", "aux": {"path": ["关联", "系数"], "at_week": 0}, "evidence_sessions": [0]}
        self.assertEqual(RelationalLine().well_posed(order, ws)[0], "drop")
        order["gt"] = "0.50"
        self.assertEqual(RelationalLine().well_posed(order, ws)[0], "well_posed")

    def test_l7_feasibility_agrees_with_actual_supply(self):
        line = ConsolidationLine()
        ws = WorldState({"甲": {"指标": series([10, 20, 30, 40])}, "乙": {"指标": series([10, 20, 30, 40])}}, n_sessions=4)
        self.assertFalse(line.feasible(ws, {})[0])
        self.assertEqual(line.enumerate(ws), [])
        ws.entities["乙"]["指标"] = series([10])
        self.assertTrue(line.feasible(ws, {})[0])
        self.assertEqual(line.enumerate(ws)[0]["gt"], "甲")
        ws = WorldState({"甲": {"指标": series([10, 20, 17, 30])}}, n_sessions=4)
        self.assertFalse(line.feasible(ws, {})[0])
        ws._trended_fields = [("甲", "指标")]
        self.assertTrue(line.feasible(ws, {})[0])
        self.assertEqual(line.enumerate(ws)[0]["gt"], "上升")
        self.assertEqual(line.enumerate(ws, 0), [])

    def test_l9_known_heldout_pair_is_blocked_but_unrelated_numbers_survive(self):
        line = InductionLine()
        ws = WorldState({}, n_sessions=3)
        line.prepare(ws, {})
        order = line.enumerate(ws)[0]
        docs = [{"content": f"{i['inst_id']} 的处置是 {i['surface_action']}。"} for i in ws.rule_instances]
        self.assertEqual(line.ground(order, docs)[0], "grounded")
        for leak in ("全新的工单响应时长47分钟，按既有规则应升级为紧急工单。",
                     "工单响应时长为47.0分钟，标记为紧急工单。"):
            self.assertTrue(heldout_leakage(order, [leak]))
            self.assertEqual(line.ground(order, docs, leak)[0], "drop")
        for benign in ("第47份工单已升级为紧急工单，响应时长52分钟。",
                       "工单响应时长147分钟，升级为紧急工单。",
                       "工单响应时长47.5分钟，升级为紧急工单。",
                       "报告响应时长47分钟，已上报主管。",
                       "工单预算47万元，响应时长52分钟，升级为紧急工单。",
                       "工单响应时长47分钟。另一个工单升级为紧急工单。",
                       "工单响应时长47分钟，但没有决定是否升级为紧急工单。"):
            with self.subTest(text=benign):
                self.assertFalse(heldout_leakage(order, [benign]))
                self.assertEqual(line.ground(order, docs, benign)[0], "grounded")

    def test_l9_contract_includes_probe_value_and_uses_no_model_call(self):
        line, ws = InductionLine(), WorldState({}, n_sessions=3)
        line.prepare(ws, {})
        order = line.enumerate(ws)[0]
        first = attach_question_contract(order)
        changed = deepcopy(order)
        changed["aux"]["x_star"] = 48
        self.assertNotEqual(first["qid"], attach_question_contract(changed)["qid"])
        class NoCalls:
            def chat_json(self, *args, **kwargs):
                raise AssertionError("deterministic line must not call a model")
        rows = phrase_questions([order], {}, NoCalls(), log=lambda *_: None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(validate_question(rows[0]), [])
        self.assertEqual(rows[0]["question_validation"]["llm_calls"], 0)

    def test_grounding_entry_checks_filler_for_heldout_pair(self):
        line, ws = InductionLine(), WorldState({}, n_sessions=3)
        line.prepare(ws, {})
        order = line.enumerate(ws)[0]
        sessions = [{"session_id": session, "docs": [
            {"doc_id": f"witness_{i['inst_id']}", "is_filler": False,
             "content": f"{i['inst_id']} 的处置是 {i['surface_action']}。"}
            for i in ws.rule_instances if i["session"] == session]} for session in range(3)]
        corpus = {"sessions": sessions}
        filler = {"doc_id": "filler_probe", "is_filler": True,
                  "content": "全新的工单响应时长47分钟，按既有规则应升级为紧急工单。"}
        sessions[0]["docs"].append(filler)
        kept, _ = run_grounding([order], corpus)
        self.assertEqual(kept, [])
        filler["content"] = "今天共登记47份工单，均需等待后续分派。"
        kept, _ = run_grounding([order], corpus)
        self.assertEqual(len(kept), 1)


class FakeLine:
    auto_activate = False
    def __init__(self, line_id, count, feasible=True, bad_first=0):
        self.id, self.count, self.possible, self.bad_first = line_id, count, feasible, bad_first
        self.requested = []
    def feasible(self, *_):
        return self.possible, "fixture supply"
    def enumerate(self, ws, target, wp):
        self.requested.append(target)
        return [{"line": self.id, "capability": "A" if i % 2 else "B", "entity": f"实体{i}", "field": "值",
                 "gt": str(i), "aux": {"bad": i < self.bad_first}} for i in range(min(target, self.count))]
    def well_posed(self, order, ws):
        return ("drop", "invalid candidate") if order["aux"]["bad"] else ("well_posed", "")


class BudgetTests(unittest.TestCase):
    def execute(self, lines, weights, budget=None, **kwargs):
        wp = {"active_lines": [{"line": line.id, "weight": weight} for line, weight in zip(lines, weights)],
              "capability_targets": {"total_q": 4}}
        stats = {}
        with patch("pipeline.lines.LINES", lines):
            rows = run_lines(wp, None, log=lambda *_: None, question_budget=budget, stats=stats, **kwargs)
        return rows, stats

    def test_weighted_total_budget_and_eligible_before_selection(self):
        lines = [FakeLine("L1_timeline", 100, bad_first=8), FakeLine("L6_refusal", 100)]
        rows, stats = self.execute(lines, [3, 1], 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["allocated"] for row in stats["lines"]], [6, 2])
        self.assertEqual([row["selected"] for row in stats["lines"]], [6, 2])
        self.assertTrue(all(not order["aux"]["bad"] for order in rows))
        self.assertEqual(stats["lines"][0]["rejected"], {"invalid candidate": 8})
        self.assertEqual(stats["legacy_total_q"], 4)

    def test_missing_supply_never_fills_from_refusal_line(self):
        lines = [FakeLine("L1_timeline", 100, feasible=False), FakeLine("L6_refusal", 5000)]
        rows, stats = self.execute(lines, [3, 1], 8)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["shortfall"], 6)
        self.assertEqual(stats["lines"][0]["shortfall_reason"], "infeasible_substrate")
        self.assertTrue(stats["lines"][1]["pool_at_limit"])
        self.assertEqual(lines[1].requested, [1000])

    def test_legacy_and_explicit_quotas_keep_previous_semantics(self):
        lines = [FakeLine("L1_timeline", 10), FakeLine("L6_refusal", 10)]
        rows, stats = self.execute(lines, [3, 1])
        self.assertEqual(len(rows), 8)
        self.assertEqual(stats["mode"], "legacy_per_line")
        rows, stats = self.execute(lines, [3, 1], quotas={"L1_timeline": 2, "L6_refusal": 3})
        self.assertEqual(len(rows), 5)
        self.assertEqual(stats["mode"], "explicit_quotas")
        with self.assertRaises(ValueError):
            self.execute(lines, [3, 1], 8, quotas={})

    def test_empty_zero_invalid_weights_and_budget(self):
        lines = [FakeLine("L1_timeline", 10)]
        self.assertEqual(self.execute(lines, [1], 0)[0], [])
        self.assertEqual(lines[0].requested, [])
        self.assertEqual(self.execute(lines, [0], 8)[1]["shortfall"], 8)
        self.assertEqual(self.execute([], [], 8)[0], [])
        for budget in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                self.execute(lines, [1], budget)
        for weight in (-1, float("nan"), float("inf"), "1"):
            with self.assertRaises(ValueError):
                self.execute(lines, [weight], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
