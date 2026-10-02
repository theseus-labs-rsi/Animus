"""Offline regressions through the real judge, runner, caches and easy filter."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
config = types.ModuleType("config")
config.MODEL = "offline-model"
config.chat_json = Mock(side_effect=AssertionError("unexpected model call"))
config.chat = Mock(side_effect=AssertionError("unexpected model call"))
config.pmap = lambda f, rows, **kw: [f(row) for row in rows]
with patch.dict(sys.modules, {"config": config}):
    spec = importlib.util.spec_from_file_location("offline_real_judge", ROOT / "eval/judge.py")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
from eval import qa_cache
from eval.question_filter import filter_questions
from eval.grading import JUDGE_VERSION, is_scored
from eval.provenance import make_evaluation_context, record_provenance


def question(**changes):
    return {"qid": "test-q", "line": "L1_timeline", "capability": "KU", "field": "负责人",
            "question": "负责人是谁？", "gt": "孔雪", "aux": {}, **changes}


def contract(kind="value", **changes):
    return {"version": 1, "answer_kind": kind, "allowed_aliases": [], "scoring_scope": "primary_answer",
            "abstention_kind": None, **changes}


def row(q, grade, pred="孔雪"):
    return {**q, "pred": pred, "judgeable": True, "judgement": grade, "correct": grade["correct"]}


def load_runner():
    spec = importlib.util.spec_from_file_location("offline_real_multi", ROOT / "eval/multi_system.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"config": config, "eval.judge": judge,
        "eval.memory_interface": types.SimpleNamespace(EmbedMemory=object, _chunk=Mock(), build_embed_memory=Mock()),
        "eval.embed_cache": types.SimpleNamespace(cached_embed=Mock(), cache_size=Mock())}):
        spec.loader.exec_module(module)
    return module


class JudgeContractTest(unittest.TestCase):
    def setUp(self):
        config.chat_json.reset_mock(side_effect=True)
        config.chat_json.side_effect = AssertionError("unexpected model call")

    def test_three_real_glm_false_positives(self):
        fixtures = json.loads((ROOT / "tests/fixtures/judge_historical_glm.json").read_text(encoding="utf-8"))
        self.assertEqual(len(fixtures), 3)
        for fixture in fixtures:
            with self.subTest(key=fixture["key"]):
                grade = judge.judge_record(fixture["question"], fixture["pred"])
                self.assertTrue(fixture["historic_correct"])
                self.assertIs(grade["correct"], False)
                self.assertTrue(is_scored(grade))
        config.chat_json.assert_not_called()

    def test_partial_names_negation_and_reasoning_cannot_match(self):
        for pred in ["孔", "不是孔雪，是张三", "<think>可能孔雪</think>张三"]:
            self.assertFalse(judge.judge_answer(question(), pred, use_llm=False))
        self.assertTrue(judge.judge_answer(question(), "答案：孔雪。", use_llm=False))

    def test_state_requires_explicit_alias(self):
        q = question(gt="已结算", field="当前任务状态", question_contract=contract("enum", value_kind="status"))
        self.assertFalse(judge.judge_answer(q, "已完成"))
        q["question_contract"]["allowed_aliases"] = ["已完成"]
        self.assertTrue(judge.judge_answer(q, "已完成"))

    def test_time_has_full_week_and_date_boundaries(self):
        q = question(capability="TR", gt={"to": "倾听陪伴", "week": 3, "date": "2025-01-20"}, question_contract=contract("time"))
        for pred in ["第13周", "倾听陪伴", "2025", "2025-01"]:
            self.assertFalse(judge.judge_answer(q, pred))
        for pred in ["3", "第3周", "2025-01-20", "2025年1月20日"]:
            self.assertTrue(judge.judge_answer(q, pred))
        self.assertFalse(judge.literal_match("12", ["1.2"]))

    def test_generated_duration_and_next_state_shapes(self):
        from pipeline.lines.L1_timeline import _candidates
        from pipeline.lines.L8_transition import TransitionLine
        from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE
        ws = WorldState({"工单甲": {"流程": Timeline([Op(0, "2025-01-06", SET, "新建"),
                            Op(2, "2025-01-20", UPDATE, "处理中", "新建")])}}, n_sessions=4)
        duration = _candidates(ws, "DURATION")[0]
        q = {"line": "L1_timeline", "capability": duration.capability, "entity": duration.entity,
             "field": duration.field, "gt": duration.gt, "aux": {**duration.aux, "time_unit": "周"},
             "question": "新建持续多少周？", "question_contract": contract("value", value_kind="duration")}
        self.assertEqual(q["gt"]["weeks"], 2)
        self.assertTrue(judge.is_judgeable(q))
        for answer in ("2", "2周", "持续了2周", "答案：2.0周"):
            self.assertTrue(judge.judge_answer(q, answer))
        for answer in ("12周", "第2周", "2天", "14天", "新建", "不是2周"):
            self.assertFalse(judge.judge_answer(q, answer))
        wp = {"domain_profile": {"state_machines": [{"field": "流程", "states": ["新建", "处理中", "已解决", "关闭"]}]}}
        transition = TransitionLine().enumerate(ws, wp=wp)[0]
        transition["question"] = TransitionLine().intent(transition)[0]
        self.assertTrue(judge.is_judgeable(transition))
        self.assertTrue(judge.judge_answer(transition, "已解决"))
        for answer in ("处理中", "已完成", "新建→处理中→已解决→关闭"):
            self.assertFalse(judge.judge_answer(transition, answer))
        config.chat_json.assert_not_called()

    def test_time_parentheses_compare_every_complete_component(self):
        from agent_harnesses.scoring import _standard_scoring_fields
        q = question(capability="TR", question="第一次变化出现在第几期？")
        q.update(_standard_scoring_fields(q, {"answer": {"week": 7, "date": "2025-02-17"}}))
        for answer in ("第7周（第7期）。", "第7期（第7周）。", "第7周（2025-02-17）。",
                       "2025年2月17日（第7期）", "第7期(2025/02/17)", "第7周（7）"):
            grade = judge.judge_record(q, answer)
            self.assertTrue(grade["correct"], answer)
            self.assertEqual(grade["path"], "time_composite")
        for answer in ("第7周（第6期）。", "第7期（第8周）。", "第7周（2025-02-18）。",
                       "第7周（2025-02）。", "第7周（7天）。", "第17期", "2025-02-18",
                       "不是第7期", "第7期或第8期"):
            self.assertFalse(judge.judge_answer(q, answer), answer)
        config.chat_json.assert_not_called()

    def test_explanatory_time_uses_semantics_with_numbering_context(self):
        from agent_harnesses.scoring import _standard_scoring_fields
        q = question(capability="TR", question="第一次变化出现在第几期？")
        q.update(_standard_scoring_fields(q, {"answer": {"week": 10, "date": "2025-03-10"}}))
        config.chat_json.side_effect = None
        for answer, expected in (
            ("第10周（2025-03-10，证据充分性由待补证更新为部分充分）。", True),
            ("第10周（session=9）。", True),
            ("第10周（期数9）。", False),
            ("不是第10周，是第11周。", False),
            ("可能第10周，也可能第11周。", False),
        ):
            config.chat_json.return_value = {"correct": expected, "reason": "offline semantic fixture"}
            grade = judge.judge_record(q, answer)
            self.assertIs(grade["correct"], expected)
            self.assertEqual(grade["path"], "llm_primary")
            messages = config.chat_json.call_args.args[0]
            data = json.loads(messages[1]["content"])
            self.assertEqual(data["target"]["canonical_period"], 10)
            self.assertEqual(data["target"]["canonical_date"], "2025-03-10")
            self.assertEqual(data["target"]["numbering"]["canonical_session"], 9)
            self.assertIn("期数本身不能减1", messages[0]["content"])
        # Internal questions have their own declared time unit, no implicit
        # standard-light session conversion is added to their grading context.
        q.pop("_benchmark_schema")
        judge.judge_record(q, "发生在第10周，并完成归档。")
        data = json.loads(config.chat_json.call_args.args[0][1]["content"])
        self.assertNotIn("numbering", data["target"])

    def test_time_semantic_failure_is_unscored(self):
        q = question(capability="TR", gt={"week": 10, "date": "2025-03-10"},
                     question_contract=contract("time"))
        config.chat_json.side_effect = TimeoutError("offline timeout")
        grade = judge.judge_record(q, "发生在第10周，随后完成归档。")
        self.assertEqual(grade["verdict"], "error")
        self.assertIsNone(grade["correct"])
        config.chat_json.reset_mock()
        grade = judge.judge_record(q, "发生在第10周，随后完成归档。", use_llm=False)
        self.assertFalse(grade["correct"])
        config.chat_json.assert_not_called()

    def test_tr_uses_declared_period_unit_only(self):
        for unit in ("周", "日", "天", "章"):
            q = question(capability="TR", gt={"to": "倾听陪伴", "week": 3, "date": "2025-01-20"},
                         aux={"time_unit": unit}, question_contract=contract("time"))
            for answer in ("3", f"第3{unit}", f"3{unit}", "2025年1月20日"):
                self.assertTrue(judge.judge_answer(q, answer), (unit, answer))
            for answer in ("3.0", "3.5", "答案大概3", "第13" + unit, "2025", "2025-01", "倾听陪伴"):
                self.assertFalse(judge.judge_answer(q, answer), (unit, answer))
            for other in set(("周", "日", "天", "章")) - {unit}:
                self.assertFalse(judge.judge_answer(q, "第3" + other), (unit, other))
            q.pop("question_contract")
            self.assertFalse(judge.judge_answer(q, "3"))
        config.chat_json.assert_not_called()

    def test_complete_typed_numeric_equivalence_and_dimensions(self):
        for gold, unit, positives, negatives in (
            ("0.5", None, ("0.50", "0.5", ".5", "5e-1"), ("0.5%", "50%", "0. 5", "0.5元", "不是0.5")),
            ("0.5", "%", ("0.5%", "0.50", "0.500%"), ("50%", "0.005", "0.5元")),
            ("10000元", "元", ("1万元", "10000", "10,000元"), ("10000美元", "1元", "1万元或2万元")),
            ("1万元", "万元", ("10000元", "1", "1.00万元"), ("10000", "1美元")),
        ):
            q = question(gt=gold, field="金额", question_contract=contract("value", value_kind="numeric", value_schema={"kind": "numeric", "unit": unit}))
            for answer in positives:
                self.assertTrue(judge.judge_answer(q, answer), (gold, unit, answer))
            for answer in negatives:
                self.assertFalse(judge.judge_answer(q, answer), (gold, unit, answer))
        config.chat_json.assert_not_called()

    def test_complete_calendar_date_equivalence_for_value_answers(self):
        for capability in ("IE", "MR", "KU"):
            gold = "2024-03-20" if capability == "KU" else {"value": "2024-03-20"}
            q = question(capability=capability, gt=gold, field="到期日",
                         question_contract=contract("value", value_kind="date", value_schema={"kind": "date"}))
            for answer in ("2024年3月20日", "2024-3-20", "2024/03/20", "2024-03-20"):
                self.assertTrue(judge.judge_answer(q, answer), (capability, answer))
            for answer in ("2024", "2024-03", "2024-03-21", "2024年2月30日", "2024-03-20或2024-03-21"):
                self.assertFalse(judge.judge_answer(q, answer), (capability, answer))
        self.assertTrue(judge.judge_answer(question(gt="为民", question_contract=contract("enum")), "为民"))
        config.chat_json.assert_not_called()

    def test_abstention_kind_and_contradictory_value(self):
        q = question(capability="L6_refusal", gt=None, question_contract=contract("abstention", abstention_kind="never_known"))
        self.assertTrue(judge.judge_answer(q, "无此项/查无此记录", use_llm=False))
        self.assertFalse(judge.judge_answer(q, "已停止统计", use_llm=False))
        self.assertFalse(judge.judge_answer(q, "查无此记录，但答案是张三", use_llm=False))
        q.update(capability="FORGET", question_contract=contract("abstention", abstention_kind="forgotten"))
        self.assertTrue(judge.judge_answer(q, "已停止统计", use_llm=False))
        self.assertFalse(judge.judge_answer(q, "查无此记录", use_llm=False))

    def test_llm_timeout_and_schema_are_not_false(self):
        q = question()
        for response in [TimeoutError("offline timeout"), {"correct": "false", "reason": "x"},
                         {"correct": 0, "reason": "x"}, {"correct": False}, {"reason": "x"}]:
            config.chat_json.side_effect = response if isinstance(response, Exception) else None
            config.chat_json.return_value = response
            grade = judge.judge_record(q, "负责者名字的另一种说法")
            self.assertEqual(grade["verdict"], "error")
            self.assertIsNone(grade["correct"])
            with self.assertRaises(judge.JudgeError):
                judge.judge_answer(q, "负责者名字的另一种说法")

    def test_order_extra_dates_do_not_change_primary_scope(self):
        q = question(capability="L3_order", gt=[{"field": "压力", "value": 9, "date": "2025-01-27"},
                {"field": "倾诉重点", "value": "转移注意", "date": "2025-03-24"}], question_contract=contract("order"))
        config.chat_json.side_effect = None
        config.chat_json.return_value = {"correct": True, "reason": "相对顺序正确"}
        grade = judge.judge_record(q, "先压力9，再倾诉重点转移注意（2025-02-17）")
        self.assertTrue(grade["correct"])
        self.assertEqual(grade["additional_facts"]["status"], "not_assessed")
        self.assertIn("附带日期", config.chat_json.call_args.args[0][0]["content"])

    def test_unknown_contract_is_unjudgeable(self):
        for kind in ("mystery", "set", "structured"):
            grade = judge.judge_record(question(question_contract=contract(kind)), "孔雪")
            self.assertEqual(grade["verdict"], "unjudgeable")
            self.assertIsNone(grade["correct"])

    def test_malformed_question_contract_is_diagnostic(self):
        malformed = [None, "question", question(question_contract="bad"), question(aux="bad"),
                     question(strict_scoring="bad"), question(question_contract=contract(value_schema="bad")),
                     question(aux={"time_unit": 7}), question(capability="IE", gt="bad")]
        for q in malformed:
            self.assertFalse(judge.is_judgeable(q), q)
            grade = judge.judge_record(q, "孔雪")
            self.assertEqual(grade["verdict"], "unjudgeable", q)
            self.assertIsNone(grade["correct"])
        config.chat_json.assert_not_called()

    def test_protocol_v5_third_refusal_and_legacy_compatibility(self):
        runner = load_runner()
        from agent_harnesses import artifacts
        protocol = {"version": 4, "rules": ["short answer"], "gold_sentinel_map":
                    {"INSUFFICIENT": "无此项/查无", "forgotten=true": "已停止统计"}}
        with tempfile.TemporaryDirectory() as td:
            about_path = Path(td) / "00_about.json"
            for version in (4, 5):
                protocol["version"] = version
                if version == 5:
                    protocol["gold_sentinel_map"]["out_of_scope"] = "信息不足/不在记录范围内"
                about = {"answer_protocol": protocol}
                about_path.write_text(json.dumps(about, ensure_ascii=False), encoding="utf-8")
                rendered = runner.load_protocol(about_path)
                self.assertEqual(rendered, artifacts.load_benchmark_protocol(Path(td)))
                self.assertEqual("超出记录时间范围" in rendered, version == 5)
                self.assertIn("已停统", rendered)
        q = question(capability="L6_refusal", gt=None, question_contract=contract("abstention", abstention_kind="out_of_scope"))
        self.assertTrue(judge.judge_answer(q, "信息不足/不在记录范围内", use_llm=False))

    def test_cache_identity_and_failed_grade_rejection(self):
        q = question(question_contract=contract())
        base = qa_cache.qhash(q)
        for field in ("question", "gt", "aux", "question_contract"):
            self.assertNotEqual(base, qa_cache.qhash({**q, field: "changed"}))
        for field in ("model", "corpus_hash", "protocol"):
            self.assertNotEqual(qa_cache.bench_id([q], {field: "a"}), qa_cache.bench_id([q], {field: "b"}))
        with tempfile.TemporaryDirectory() as td, patch.object(qa_cache, "CACHE_DIR", Path(td)):
            for verdict in ("error", "unjudgeable"):
                qa_cache.append("b", "s", {**row(q, judge.judgement(verdict, "test", "failure")), "_qh": base})
            self.assertEqual(qa_cache.load("b", "s"), {})
            qa_cache.append("b", "s", {**row(q, judge.judge_record(q, "孔雪")), "_qh": base})
            self.assertIn(base, qa_cache.load("b", "s"))
            with patch.object(qa_cache, "JUDGE_VERSION", "next"):
                self.assertNotEqual(base, qa_cache.qhash(q))

    def test_runner_timeout_rejudges_cached_prediction_without_difficulty(self):
        runner = load_runner()
        q = question()
        system = Mock()
        system.retrieve.return_value = "context"
        system.get_diagnostics.return_value = {}
        evaluation_context = make_evaluation_context([(0, "2026-01-01", "context")], "")
        cache_context = {"evaluation_context": evaluation_context}
        with tempfile.TemporaryDirectory() as td, patch.object(qa_cache, "CACHE_DIR", Path(td)), \
             patch.object(runner, "unified_answer", return_value="负责者名字的另一种说法") as solve:
            config.chat_json.side_effect = TimeoutError("offline timeout")
            first = runner.run_system("A", [q], system, verbose=False, bench_id="b", cache_context=cache_context)
            self.assertEqual(first[0]["judgement"]["verdict"], "error")
            self.assertEqual(runner.aggregate(first)["overall"]["n"], 0)
            disc = runner.discrimination_summary({"A": {"records": first, "agg": runner.aggregate(first)}}, ["A"])
            self.assertEqual(disc["status"], "incomplete")
            self.assertIsNone(disc["overall"]["A"])
            self.assertIsNone(disc["headroom"])
            peer = row(q, judge.judge_record(q, "孔雪"))
            peer["evaluation_provenance"] = record_provenance(q, evaluation_context)
            kept, report = filter_questions([q], {"A": first, "B": [peer]}, expected_context=evaluation_context)
            self.assertEqual(kept, [q])
            self.assertEqual(report["items"][0]["disposition"], "kept_incomplete")
            config.chat_json.side_effect = None
            config.chat_json.return_value = {"correct": False, "reason": "答案不同"}
            second = runner.run_system("A", [q], system, verbose=False, bench_id="b", cache_context=cache_context)
            self.assertIs(second[0]["correct"], False)
            self.assertTrue(second[0]["_prediction_resumed"])
            self.assertEqual(solve.call_count, 1)
            self.assertEqual(runner.aggregate(second)["overall"]["n"], 1)
            _, report = filter_questions([q], {"A": second, "B": [peer]}, expected_context=evaluation_context)
            self.assertEqual(report["items"][0]["disposition"], "kept_not_all_correct")
            third = runner.run_system("A", [q], system, verbose=False, bench_id="b", cache_context=cache_context)
            self.assertTrue(third[0]["_resumed"])
            runner.run_system("A", [q], system, verbose=False, bench_id="b", cache_context={"corpus_hash": "changed"})
            self.assertEqual(solve.call_count, 2)

    def test_legacy_or_inconsistent_labels_are_incomplete(self):
        q = question()
        peer = row(q, judge.judge_record(q, "孔雪"))
        for bad in [{**q, "pred": "孔雪", "correct": True}, {**peer, "correct": False},
                    row(q, {**peer["judgement"], "version": "old"}),
                    row(q, {**peer["judgement"], "reason": ""})]:
            _, report = filter_questions([q], {"a": [peer], "b": [bad]})
            self.assertEqual(report["counts"]["kept_incomplete"], 1)


if __name__ == "__main__":
    unittest.main()
