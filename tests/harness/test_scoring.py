from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harnesses.config import REPOSITORY_ROOT, ConfigurationError
from agent_harnesses.judging import (
    FACTORY_ROOT_ENV,
    load_judge,
    resolve_factory_root,
)
from agent_harnesses.scoring import _standard_scoring_fields, aggregate, read_results, score_run
from standard_light_fixture import build_standard_light


def _qid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


QUESTIONS = [
    {"line": "L1", "capability": "IE", "entity": "e0", "field": "f0",
     "question": "q0?", "gt": {"value": "42"}, "aux": {}},
    {"line": "L1", "capability": "KU", "entity": "e1", "field": "f1",
     "question": "q1?", "gt": "blue", "aux": {}},
    {"line": "L2", "capability": "IE", "entity": "e2", "field": "f2",
     "question": "q2?", "gt": {"value": "7"}, "aux": {}},
    {"line": "L2", "capability": "UNKNOWN_CAP", "entity": "e3", "field": "f3",
     "question": "q3?", "gt": None, "aux": {}},
]


class FakeJudge:
    version = {"factory_root": "fake", "factory_commit": None, "judge_sha256": "deadbeef"}

    def is_judgeable(self, q):
        return q.get("capability") in {"IE", "KU"}

    def judge_mode(self, q):
        return "value" if self.is_judgeable(q) else None

    def gold_display(self, q):
        return q.get("gt")

    def judge_answer(self, q, pred, use_llm=True):
        gt = q.get("gt")
        gold = gt.get("value") if isinstance(gt, dict) else gt
        return str(gold) == pred

    def judge_l2_partial(self, q, pred, use_llm=True):
        return 0.0

    def classify_refusal(self, pred, lure=None):
        return "other"


def _record(index: int, *, answer=None, error_type=None) -> dict:
    return {
        "schema": "agent-harnesses.result/v1",
        "question_id": _qid(QUESTIONS[index]["question"]),
        "question_index": index,
        "status": "failed" if error_type else "completed",
        "answer": answer,
        "judgeable": False,
        "correct": None,
        "error_type": error_type,
        "returncode": 1 if error_type else 0,
        "elapsed_s": 0.1,
        "raw_stdout": f"raw/{index:04d}.stdout.txt",
        "raw_stderr": f"raw/{index:04d}.stderr.txt",
    }


def _build_out_dir(root: Path) -> Path:
    run_dir = root / "benchmark"
    run_dir.mkdir()
    (run_dir / "06_grounded_questions.json").write_text(
        json.dumps({"questions": QUESTIONS}, ensure_ascii=False), encoding="utf-8"
    )
    out = root / "out"
    (out / "raw").mkdir(parents=True)
    plan = {
        "schema": "agent-harnesses.run-plan/v1",
        "experiment_id": "t-exp",
        "system_id": "t-sys",
        "runner": "native_cli",
        "system": {"implementation": {"adapter": "claude"}},
        "runtime": {"adapter": "claude", "model": "deepseek-v4-flash"},
        "benchmark": {"path": str(run_dir)},
        "protocol": {"scoring": "disabled_for_chain_smoke"},
        "execution": {"mode": "smoke", "limit": 4},
    }
    (out / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    records = [
        _record(0, answer="42"),
        _record(1, answer="red"),
        _record(2, error_type="timeout"),
        _record(3, answer="whatever"),
    ]
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (out / "raw" / "0000.stdout.txt").write_text(
        '{"result":"42","usage":{"input_tokens":1000,"output_tokens":100}}',
        encoding="utf-8",
    )
    return out


class ScoringTests(unittest.TestCase):
    def test_standard_light_period_answers_match_public_protocol(self):
        public = {"capability": "TR", "question": "首次变化在第几期？请回答期数。"}
        reference = {
            "answer": {"week": 10, "date": "2025-03-10", "to": "已完成"},
            "answer_raw": {"week": 99},
        }
        question = {**public, **_standard_scoring_fields(public, reference)}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "MODEL": "test-model"}):
            judge = load_judge(REPOSITORY_ROOT)
            with mock.patch.object(judge._module.config, "chat_json", side_effect=AssertionError("no API calls")):
                for answer in ("第10期", "第10期。", "10期", "第10周", "10周", "第 10 周", "10", "2025-03-10"):
                    with self.subTest(answer=answer):
                        self.assertTrue(judge.judge_answer(question, answer, use_llm=True))
                for answer in ("第9期", "第11期", "第99期", "第10天", "10个月", "2025-03-11", "第10期或第11期", "不是第10期", "已完成"):
                    with self.subTest(answer=answer):
                        self.assertFalse(judge.judge_answer(question, answer, use_llm=True))

    def test_standard_light_scores_all_quality_statuses_from_answer_truth(self):
        questions = [
            {"qid": "q-release", "question": "状态？", "line": "L1", "capability": "IE", "quality_status": "released"},
            {"qid": "q-reject", "question": "未知项？", "line": "L6", "capability": "L6_refusal", "quality_status": "rejected"},
            {"qid": "q-pending", "question": "下一状态？", "line": "L8", "capability": "L8_next", "quality_status": "pending_review"},
        ]
        references = [
            {"qid": "q-release", "answer": "进行中", "answer_raw": {"value": "进行中", "at_week": 1}, "answer_projection": "gt/value", "quality_status": "released"},
            {"qid": "q-reject", "answer": "查无此记录", "answer_raw": "INSUFFICIENT_EVIDENCE", "answer_projection": "abstention:never_known", "quality_status": "rejected"},
            {"qid": "q-pending", "answer": "已完成", "answer_raw": "已完成", "answer_projection": "raw_gt", "quality_status": "pending_review"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = build_standard_light(root / "benchmark", questions, references)
            out = root / "out"
            out.mkdir()
            plan = {
                "experiment_id": "standard-test",
                "system_id": "fake",
                "runner": "native_cli",
                "system": {"implementation": {"adapter": "claude"}},
                "runtime": {"adapter": "claude", "model": "fake-model"},
                "benchmark": {"path": str(benchmark)},
                "protocol": {"scoring": "enabled"},
            }
            (out / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            answers = ["进行中", "查无此记录", "已完成"]
            rows = [
                {
                    "schema": "agent-harnesses.result/v1",
                    "question_id": question["qid"],
                    "question_index": index,
                    "quality_status": question["quality_status"],
                    "status": "completed",
                    "answer": answers[index],
                    "judgeable": False,
                    "correct": None,
                    "error_type": None,
                }
                for index, question in enumerate(questions)
            ]
            (out / "results.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": "test-key", "MODEL": "test-model"}
            ):
                judge = load_judge(REPOSITORY_ROOT)
                summary = score_run(out, judge, use_llm=False)
            self.assertEqual(summary["aggregate"]["overall"]["n_total"], 3)
            self.assertEqual(summary["aggregate"]["overall"]["n_correct"], 3)
            self.assertEqual(
                set(summary["aggregate"]["by_quality_status"]),
                {"released", "rejected", "pending_review"},
            )
            judged = [
                json.loads(line)
                for line in (out / "judged.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(
                    row["truth_source"] == "references/questions.json.answer"
                    for row in judged
                )
            )
            self.assertEqual(judged[0]["reference_answer"], "进行中")

    def test_score_run_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_out_dir(Path(tmp))
            summary = score_run(out, FakeJudge(), use_llm=False)

            rows = {
                r["question_index"]: r
                for r in (
                    json.loads(line)
                    for line in (out / "judged.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(rows[0]["schema"], "agent-harnesses.result/v2")
            self.assertTrue(rows[0]["correct"])
            self.assertEqual(rows[0]["judge"]["version"]["judge_sha256"], "deadbeef")
            self.assertFalse(rows[1]["correct"])
            # 基础设施失败：不计错、不删分母
            self.assertIsNone(rows[2]["correct"])
            self.assertTrue(rows[2]["judgeable"])
            self.assertEqual(rows[2]["error_type"], "timeout")
            # 不可判分题显式暴露
            self.assertFalse(rows[3]["judgeable"])
            self.assertIsNone(rows[3]["correct"])
            # usage 回填 + 成本
            self.assertEqual(rows[0]["usage"]["prompt_tokens"], 1000)
            self.assertAlmostEqual(rows[0]["cost_cny"], (1000 * 1.0 + 100 * 2.0) / 1_000_000)
            self.assertIsNone(rows[1]["usage"])

            overall = summary["aggregate"]["overall"]
            self.assertEqual(overall["n_total"], 4)
            self.assertEqual(overall["n_judgeable"], 3)
            self.assertEqual(overall["n_judged"], 2)
            self.assertEqual(overall["n_correct"], 1)
            self.assertEqual(overall["n_infra_failed"], 1)
            self.assertEqual(overall["accuracy"], 0.25)
            self.assertEqual(overall["accuracy_over_judged"], 0.5)
            self.assertTrue(summary["advisory"])
            self.assertTrue((out / "score_summary.json").is_file())
            self.assertTrue((out / "score_summary.md").is_file())

    def test_score_run_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_out_dir(Path(tmp))
            score_run(out, FakeJudge(), use_llm=False)
            with self.assertRaises(ConfigurationError):
                score_run(out, FakeJudge(), use_llm=False)
            score_run(out, FakeJudge(), use_llm=False, force=True)

    def test_score_run_detects_question_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_out_dir(Path(tmp))
            lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            rec = json.loads(lines[0])
            rec["question_id"] = "bogus"
            lines[0] = json.dumps(rec, ensure_ascii=False)
            (out / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                score_run(out, FakeJudge(), use_llm=False)

    def test_read_results_dedupes_by_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_out_dir(Path(tmp))
            retry = _record(2, answer="7")
            with (out / "results.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(retry, ensure_ascii=False) + "\n")
            records = read_results(out)
            self.assertEqual(len(records), 4)
            self.assertEqual(records[2]["status"], "completed")
            self.assertEqual(records[2]["answer"], "7")

    def test_aggregate_groups_by_line_and_capability(self):
        records = [
            {"line": "L1", "capability": "IE", "judgeable": True, "correct": True},
            {"line": "L1", "capability": "KU", "judgeable": True, "correct": False},
            {"line": "L2", "capability": "IE", "judgeable": True, "correct": None, "error_type": "timeout"},
        ]
        agg = aggregate(records)
        self.assertEqual(agg["by_line"]["L1"]["n_correct"], 1)
        self.assertEqual(agg["by_line"]["L2"]["n_infra_failed"], 1)
        self.assertEqual(agg["by_capability"]["IE"]["n_total"], 2)


class JudgingTests(unittest.TestCase):
    def test_resolve_factory_root_defaults_to_this_repository(self):
        """合并后 judge 与评测控制面同仓库，无需外部 factory_root 配置。"""
        old = os.environ.pop(FACTORY_ROOT_ENV, None)
        try:
            self.assertEqual(resolve_factory_root({}), REPOSITORY_ROOT.resolve())
        finally:
            if old is not None:
                os.environ[FACTORY_ROOT_ENV] = old

    def test_resolve_factory_root_rejects_path_without_judge(self):
        old = os.environ.pop(FACTORY_ROOT_ENV, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ConfigurationError):
                    resolve_factory_root({"factory_root": tmp})
        finally:
            if old is not None:
                os.environ[FACTORY_ROOT_ENV] = old

    def test_load_judge_from_fake_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval").mkdir()
            (root / "eval" / "judge.py").write_text(
                "def judge_spec(q):\n"
                "    return ('value', [str(q.get('gt'))], None)\n"
                "def is_judgeable(q):\n"
                "    return True\n"
                "def gold_display(q):\n"
                "    return q.get('gt')\n"
                "def judge_answer(q, pred, use_llm=True):\n"
                "    return str(q.get('gt')) == pred\n"
                "def judge_l2_partial(q, pred, use_llm=True):\n"
                "    return 0.5\n"
                "def classify_refusal(pred, lure=None):\n"
                "    return 'other'\n",
                encoding="utf-8",
            )
            resolved = resolve_factory_root({"factory_root": str(root)})
            judge = load_judge(resolved)
            self.assertTrue(judge.judge_answer({"gt": "42"}, "42"))
            self.assertFalse(judge.judge_answer({"gt": "42"}, "41"))
            self.assertEqual(judge.judge_mode({"gt": "42"}), "value")
            self.assertEqual(len(judge.version["judge_sha256"]), 16)
            self.assertIsNone(judge.version["factory_commit"])


if __name__ == "__main__":
    unittest.main()
