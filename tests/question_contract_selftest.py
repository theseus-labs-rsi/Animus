"""Offline regressions for typed extrema and final question contracts.

Fixtures preserve the failures in insurance Q20/23/27/30, Q65/72/109/127,
Q50/51 and the source-vs-name question, with synthetic entity labels.
Expected dates and period numbers are independent constants, not gt_mr output.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.lines.L1_timeline import TimelineLine, _candidates
from pipeline.lines.L3_process import ProcessLine
from pipeline.lines.L5_conflict import ConflictLine
from pipeline.question_contract import attach_question_contract, build_question_contract, validate_question, bind_question_world
from pipeline.render import phrase_questions
from pipeline.value_types import ValueComparisonError, comparison_keys, parse_date, parse_number
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, INSUFFICIENT, _to_num, gt_mr


def timeline(values):
    return Timeline([Op(i, f"2025-01-{1 + i * 7:02d}", SET if i == 0 else UPDATE,
                        value, values[i - 1] if i else None) for i, value in enumerate(values)])


def world_for(values, kind=None, unit=None):
    schema = {"name": "指标", "kind": kind, **({"unit": unit} if unit else {})}
    return WorldState({"报告甲": {"指标": timeline(values)}}, n_sessions=len(values),
                      entity_types={"报告甲": "report"} if kind else {},
                      world_blueprint={"entity_types": [{"id": "report", "fields": [schema]}]} if kind else {})


def conflict_order():
    return {"line": "L5_conflict", "capability": "L5_conflict", "entity": "成员甲", "field": "直属负责人",
            "gt": "审阅员甲", "aux": {"session": 6, "ans_kind": "person", "time_unit": "周",
                "authoritative_source": "官方通报", "rumor_source": "外部传闻",
                "authoritative_value": "审阅员甲", "rumor_value": "审阅员乙", "rule": "source_reliability"}}


class FixedTracer:
    def __init__(self, question):
        self.question = question
        self.calls = 0

    def chat_json(self, *_args, **_kwargs):
        self.calls += 1
        return {"question": self.question}


class TypedComparisonTests(unittest.TestCase):
    def test_four_real_date_shapes_have_independent_maxima(self):
        samples = [
            (["2023-03-10", "2023-04-15", "2023-05-20", "2023-06-30"], "2023-06-30"),
            (["2022-01-15", "2022-02-20", "2022-03-10", "2022-03-31"], "2022-03-31"),
            (["2023-12-05", "2023-12-15", "2023-12-24", "2023-12-31"], "2023-12-31"),
            (["2023-08-01", "2023-08-15", "2023-09-05", "2023-09-25"], "2023-09-25"),
        ]
        for values, expected in samples:
            with self.subTest(values=values):
                ws = world_for(values, "date")
                self.assertEqual(gt_mr(ws, "报告甲", "指标")["value"], expected)
                bad = {"capability": "MR", "entity": "报告甲", "field": "指标",
                       "gt": {"value": values[0], "session": 0, "date": "2025-01-01", "agg": "max"},
                       "aux": {"agg": "max"}, "evidence_sessions": [0, 1, 2, 3]}
                self.assertEqual(TimelineLine().well_posed(bad, ws)[0], "drop")
                good = deepcopy(bad)
                good["gt"] = {"value": expected, "session": 3, "date": "2025-01-22", "agg": "max"}
                self.assertEqual(TimelineLine().well_posed(good, ws)[0], "well_posed")
                candidates = _candidates(ws, "MR")
                self.assertEqual({o.aux["ans_kind"] for o in candidates}, {"date"})
                self.assertEqual(len(candidates), 2)

    def test_dates_and_malformed_dates_never_become_years(self):
        for value in ("2023-09-25", "2023-02-30", "2023/9/25"):
            self.assertIsNone(_to_num(value))
        self.assertEqual(parse_date("2024-02-29").day, 29)
        for value in ("2023-02-29", "2023-13-01", "2023-02", "2023-2-01", "2023-01-01尾注"):
            with self.subTest(value=value), self.assertRaises(ValueComparisonError):
                parse_date(value)

    def test_invalid_member_invalidates_entire_extremum(self):
        for values in (["2023-01-01", "2023-02-30"], ["2023-01-01", "2023"]):
            ws = world_for(values, "date")
            self.assertEqual(gt_mr(ws, "报告甲", "指标"), {"value": INSUFFICIENT})
            self.assertEqual(_candidates(ws, "MR"), [])

    def test_legacy_adaptation_is_strict_and_schema_has_priority(self):
        ws = world_for(["2023-12-30", "2024-01-01"])
        self.assertEqual(gt_mr(ws, "报告甲", "指标")["value"], "2024-01-01")
        ws = world_for(["2023-12-30", "2024-01-01"], "category")
        self.assertEqual(gt_mr(ws, "报告甲", "指标", schema={"kind": "date"}), {"value": INSUFFICIENT})
        ws = world_for(["版本1", "版本2"])
        self.assertEqual(_candidates(ws, "MR"), [])

    def test_numbers_units_ties_and_invalid_aggregator(self):
        self.assertEqual(gt_mr(world_for(["9,000元", "1.2万元"], "numeric", "元"), "报告甲", "指标")["value"], "1.2万元")
        self.assertEqual(gt_mr(world_for(["100", "120"], "numeric", "NPR million"), "报告甲", "指标")["value"], "120")
        self.assertEqual(gt_mr(world_for(["10次", "15次", "15次"]), "报告甲", "指标")["session"], 1)
        self.assertEqual(gt_mr(world_for(["10", "15"]), "报告甲", "指标", "average"), {"value": INSUFFICIENT})
        for values in (["9元", "10次"], ["1,20", "200"], ["约100", "110"], ["NaN", "1"], [True, 2]):
            with self.subTest(values=values), self.assertRaises(ValueComparisonError):
                comparison_keys(values)

    def test_percent_gold_does_not_equal_unscaled_number(self):
        ws = world_for(["0.25", "0.5"], "numeric")
        order = {"line": "L1_timeline", "capability": "MR", "entity": "报告甲", "field": "指标",
                 "gt": {"value": "0.5%", "session": 1, "date": "2025-01-08", "agg": "max"},
                 "aux": {"agg": "max"}, "evidence_sessions": [0, 1]}
        self.assertEqual(TimelineLine().well_posed(order, ws)[0], "drop")
        order["gt"]["value"] = "0.50"
        self.assertEqual(TimelineLine().well_posed(order, ws)[0], "well_posed")


class QuestionContractTests(unittest.TestCase):
    def phrase(self, order, proposed):
        rows = phrase_questions([order], {}, FixedTracer(proposed), log=lambda *_: None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(validate_question(rows[0]), [])
        return rows[0]

    def test_source_selection_question_cannot_keep_person_gold(self):
        order = conflict_order()
        bad = "截至第7周，成员甲的直属负责人应该认定为官方通报里的还是外部传闻里的？"
        contract = build_question_contract(order)
        self.assertTrue(validate_question(bad, contract))
        result = self.phrase(order, bad)
        self.assertIn("最终取值", result["question"])
        self.assertIn("第7周", result["question"])
        self.assertNotIn("审阅员甲", result["question"])
        self.assertEqual(result["gt"], "审阅员甲")

    def test_four_window_question_shapes_restore_time_without_revealing_refusal(self):
        for field in ("版本性质", "来源发布日期", "报告期间", "覆盖期间"):
            order = {"line": "L6_refusal", "capability": "L6_refusal", "entity": "报告甲", "field": field,
                     "gt": INSUFFICIENT, "aux": {"refusal_type": "T2_window", "probe": {"at_week": 6}, "time_unit": "周"}}
            result = self.phrase(order, f"报告甲的{field}是什么？")
            self.assertIn("第 6 周", result["question"])
            self.assertNotIn(INSUFFICIENT, result["question"])
            self.assertEqual(result["question_contract"]["abstention_kind"], "out_of_scope")
            bad_issues = validate_question(f"报告甲的{field}是什么？", result["question_contract"])
            self.assertTrue(any(i["code"] == "missing_time" for i in bad_issues))

    def test_sorting_keeps_target_values_including_business_dates(self):
        events = [
            {"field": "版本性质", "value": "修改稿", "session": 1, "date": "2023-04-15", "op": UPDATE},
            {"field": "来源发布日期", "value": "2023-05-20", "session": 2, "date": "2023-05-20", "op": UPDATE},
            {"field": "税后利润", "value": "128", "session": 3, "date": "2023-06-30", "op": UPDATE},
        ]
        order = {"line": "L3_process", "capability": "L3_order", "entity": "报告甲", "field": "", "gt": events,
                 "aux": {"events": deepcopy(events)}}
        for bad in ("请把报告甲的来源发布日期、版本性质和税后利润变更排序。",
                    "请把报告甲的版本性质变为修改稿、来源发布日期被修改、税后利润变为128按先后排序。"):
            result = self.phrase(order, bad)
            for value in ("修改稿", "2023-05-20", "128"):
                self.assertIn(value, result["question"])
            self.assertEqual(result["question_contract"]["answer_kind"], "order")
        self.assertNotIn("2023-05-20", ProcessLine().intent(order)[1])

    def test_good_template_and_presentation_variants_survive(self):
        order = conflict_order()
        canonical = build_question_contract(order)["canonical_question"]
        candidate = canonical.replace("，", ", ").replace("？", "?")
        self.assertEqual(validate_question(candidate, build_question_contract(order)), [])
        result = self.phrase(order, candidate)
        self.assertEqual(result["question"], canonical)
        self.assertEqual(result["question_validation"]["rewrite_issues"], [])
        self.assertEqual(result["question_validation"]["llm_calls"], 0)

    def test_explicit_gold_and_truncated_entity_are_rejected(self):
        order = conflict_order()
        order["entity"] = "测试部门成员甲"
        contract = build_question_contract(order)
        leaked = contract["canonical_question"] + "答案就是审阅员甲。"
        self.assertIn("answer_leak", {i["code"] for i in validate_question(leaked, contract)})
        bad = contract["canonical_question"].replace("测试部门成员甲", "测试部门")
        self.assertIn("missing_entity", {i["code"] for i in validate_question(bad, contract)})

    def test_invalid_fallback_is_rejected_without_model_call(self):
        order = conflict_order()
        tracer = FixedTracer("whatever")
        with patch.object(ConflictLine, "intent", return_value=("成员甲的直属负责人是什么？答案是审阅员甲。", ["审阅员甲"])):
            self.assertEqual(phrase_questions([order], {}, tracer, log=lambda *_: None), [])
        self.assertEqual(tracer.calls, 0)
        order["aux"]["session"] = None
        self.assertEqual(phrase_questions([order], {}, tracer, log=lambda *_: None), [])

    def test_qids_and_contracts_are_stable_and_gold_sensitive(self):
        order = conflict_order()
        original = deepcopy(order)
        a = attach_question_contract(order)
        b = attach_question_contract({**order, "question": "different prose"})
        self.assertEqual(a["qid"], b["qid"])
        self.assertEqual(order, original)
        self.assertNotEqual(a["qid"], attach_question_contract({**order, "gt": "different answer"})["qid"])
        self.assertEqual(attach_question_contract({**order, "qid": "legacy-Q1"})["qid"], "legacy-Q1")
        a["question"] = a["question_contract"]["canonical_question"]
        a["gt"] = "篡改答案"
        self.assertIn("contract_mismatch", {issue["code"] for issue in validate_question(a)})

    def test_scoring_slots_only_use_explicit_aliases(self):
        order = {"line": "L1_timeline", "capability": "KU", "entity": "报告甲", "field": "状态", "gt": "已结算", "aux": {}}
        wp = {"domain_profile": {"field_schema": [{"name": "状态", "kind": "status", "allowed_aliases": {"已结算": ["结算完毕"]}}]}}
        contract = build_question_contract(order, wp)
        self.assertEqual(contract["answer_kind"], "enum")
        self.assertEqual(contract["allowed_aliases"], ["结算完毕"])
        self.assertNotIn("已完成", contract["allowed_aliases"])
        self.assertEqual(contract["scoring_scope"], "primary_answer")
        frozen = attach_question_contract(order, wp)
        frozen["question"] = contract["canonical_question"]
        self.assertEqual(validate_question(frozen), [])
        for key, value in (("answer_kind", "abstention"), ("allowed_aliases", ["已完成"]), ("render_policy", "protected_slots")):
            changed = deepcopy(frozen)
            changed["question_contract"][key] = value
            self.assertIn("contract_mismatch", {issue["code"] for issue in validate_question(changed)})

    def test_event_sign_and_decimal_are_semantic(self):
        events = [{"field": field, "value": value, "session": i, "date": f"2025-01-{i + 1:02d}", "op": UPDATE}
                  for i, (field, value) in enumerate((("利润", "-12.5"), ("状态", "审批中"), ("人数", "15")))]
        order = {"line": "L3_process", "capability": "L3_order", "entity": "项目甲", "field": "", "gt": events,
                 "aux": {"events": events}}
        contract = build_question_contract(order)
        self.assertEqual(validate_question(contract["canonical_question"], contract), [])
        for bad in ("12.5", "-125"):
            self.assertTrue(validate_question(contract["canonical_question"].replace("-12.5", bad), contract))

    def test_manual_duration_has_complete_oracle_contract_and_boundary_evidence(self):
        ws = WorldState({"项目甲": {"状态": Timeline([
            Op(0, "2025-01-01", SET, "审批中"), Op(2, "2025-01-15", UPDATE, "执行中", "审批中")])}}, n_sessions=4)
        candidate = _candidates(ws, "DURATION")[0]
        order = {"line": "L1_timeline", "capability": "DURATION", "entity": "项目甲", "field": "状态",
                 "gt": candidate.gt, "aux": candidate.aux, "evidence_sessions": candidate.evidence_sessions}
        expected = {"value": "审批中", "weeks": 2, "start": 0, "end": 2}
        line = TimelineLine()
        self.assertEqual(line.gt(ws, order), expected)
        self.assertEqual(line.well_posed(order, ws)[0], "well_posed")
        contract = build_question_contract(order)
        self.assertEqual((contract["answer_kind"], contract["value_kind"], contract["value_schema"]["unit"]), ("value", "duration", "周"))
        self.assertEqual(validate_question(contract["canonical_question"], contract), [])
        for changed in ({**order, "gt": {**expected, "weeks": 3}}, {**order, "evidence_sessions": [0, 2]}):
            self.assertEqual(line.well_posed(changed, ws)[0], "drop")
        docs = [{"session": 0, "content": "项目甲的状态为审批中。"}, {"session": 2, "content": "项目甲的状态改为执行中。"}]
        self.assertEqual(line.ground(order, docs)[0], "grounded")
        self.assertEqual(line.ground(order, docs[:1])[0], "drop")
        ws.entities["项目甲"]["状态"].ops.append(Op(3, "2025-01-22", UPDATE, "审批中", "执行中"))
        self.assertEqual(line.well_posed(order, ws)[0], "drop")
        self.assertNotIn("DURATION", TimelineLine._CAP_WEIGHT)

    def test_schema_binding_uses_entity_type_and_rejects_aux_alias_override(self):
        schema_a = {"name": "状态", "kind": "status", "allowed_aliases": {"已登记": ["待接收"]}}
        schema_b = {"name": "状态", "kind": "status", "allowed_aliases": {"已登记": []}}
        blueprint = {"entity_types": [{"id": "a", "fields": [schema_a]}, {"id": "b", "fields": [schema_b]}]}
        ws = WorldState({"报告甲": {"状态": timeline(["待接收", "已登记"])}}, n_sessions=2,
                        entity_types={"报告甲": "b"}, world_blueprint=blueprint)
        wp = {"world_blueprint": blueprint}
        order = {"line": "L1_timeline", "capability": "KU", "entity": "报告甲", "field": "状态", "gt": "已登记",
                 "allowed_aliases": ["待接收"], "aux": {"value_schema": schema_a}}
        unresolved = build_question_contract(order, wp)
        self.assertIn("invalid_schema", {i["code"] for i in validate_question(unresolved["canonical_question"], unresolved)})
        bound = attach_question_contract(bind_question_world(order, ws), wp)
        self.assertEqual(bound["question_contract"]["allowed_aliases"], [])
        bound["question"] = bound["question_contract"]["canonical_question"]
        self.assertEqual(validate_question(bound), [])

    def test_relation_answer_schema_is_from_terminal_entity(self):
        blueprint = {"entity_types": [
            {"id": "team", "fields": [{"name": "状态", "kind": "status", "allowed_aliases": {"已登记": ["待接收"]}}]},
            {"id": "report", "fields": [{"name": "状态", "kind": "status", "allowed_aliases": {"已登记": []}}]}]}
        ws = WorldState({"组甲": {"关联报告": timeline(["报告乙"])}, "报告乙": {"状态": timeline(["已登记"])}},
                        n_sessions=1, entity_types={"组甲": "team", "报告乙": "report"}, world_blueprint=blueprint)
        order = {"line": "L2_relational", "capability": "L2_multihop", "entity": "组甲", "field": "关联报告→状态",
                 "gt": "已登记", "aux": {"path": ["关联报告", "状态"], "at_week": 0, "time_unit": "周"}}
        bound = attach_question_contract(bind_question_world(order, ws), {"world_blueprint": blueprint})
        self.assertEqual(bound["entity_type"], "team")
        self.assertEqual(bound["answer_entity_type"], "report")
        self.assertEqual(bound["answer_field"], "状态")
        self.assertEqual(bound["question_contract"]["allowed_aliases"], [])
        self.assertEqual(bound["question_contract"]["value_kind"], "status")


if __name__ == "__main__":
    unittest.main(verbosity=2)
