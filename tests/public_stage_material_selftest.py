"""Original corpus/L8 wiring regression. All reviewers are fixed offline doubles.

These tests establish transport, repair, provenance and isolation behavior, not
the accuracy of a real model's semantic reading.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-public-stage-test")

import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline.corpus_contract import (CorpusReviewExecutionError, attach_receipts,
    canonical_context, public_rule_coverage_issues, public_stage_rules,
    review_documents, validate_corpus)
from pipeline.lines.L8_transition import TransitionLine
from pipeline.render import (MaterialRejected, _render_public_stage_material,
                             _sanitize_corpus, render_corpus)
from pipeline.world_state import Op, SET, UPDATE, Timeline, WorldState, _date_of


def world():
    ws = WorldState(entities={"岚桥项目": {"状态": Timeline([
        Op(0, _date_of(0), SET, "进行中"),
        Op(1, _date_of(1), UPDATE, "收尾", "进行中")])}}, n_sessions=2)
    ws.entity_types = {"岚桥项目": "project"}
    ws.world_blueprint = {"entity_types": [{"id": "project", "noun": "项目",
        "fields": [{"name": "状态", "kind": "status", "states": ["立项", "进行中", "收尾", "完成"]}]}],
        "temporal_model": {"unit": "week", "step_days": 7}}
    return ws


class FakeTracer:
    def __init__(self, failures=(), error=False):
        self.calls = []
        self.failures = list(failures)
        self.error = error
        self.facts = []

    def chat_text(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages)))
        return "外围仓库档案整理。"

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages)))
        user = messages[-1]["content"]
        if step == "render.signal" and "【冻结的公共阶段定义】" in user:
            rules = json.loads(user.split("【冻结的公共阶段定义】", 1)[1].splitlines()[0])
            date = user.split("【文档日期】", 1)[1].splitlines()[0]
            return {"docs": [{"title": "阶段说明", "content":
                f"{date}，{rule['entity_noun']}的{rule['field']}阶段顺序为"
                + "、".join(rule["ordered_stages"]) + "。实际记录可跳级，不可倒退。"} for rule in rules]}
        if step == "render.signal":
            self.facts = json.JSONDecoder().raw_decode(user[user.index('[{"entity":'):])[0]
            return {"docs": [{"content": "。".join(
                f"{fact['entity']}的{fact['field']}为{fact['value']}" for fact in self.facts) + "。"}]}
        if step == "render.discriminate":
            queries = json.JSONDecoder().raw_decode(user[user.index('[{"key":'):])[0]
            values = {(fact["entity"], fact["field"]): fact["value"] for fact in self.facts}
            return {"answers": [{"key": q["key"], "answer": values[q["entity"], q["field"]]} for q in queries]}
        if step == "corpus.review":
            data = json.loads(user)
            required = [item for item in data["requirements"] if item["kind"] == "public_rule"]
            if required and self.error:
                return {"verdict": "pass", "unsupported_claims": [], "coverage": []}
            failed = bool(required and self.failures and self.failures.pop(0))
            coverage = [{"requirement_id": item["requirement_id"], "status": "missing" if failed else "supported",
                         "reason": "固定桩：正文未表达适用顺序" if failed else "固定桩：完整正文已表达规则",
                         "evidence": [] if failed else [{"doc_index": 0,
                             "quote": data["documents"][0]["content"]}]}
                        for item in data["requirements"]]
            return {"document_reviews": fixed_document_reviews(data["documents"]),
                    "verdict": "fail" if failed else "pass", "unsupported_claims": [],
                    "coverage": coverage}
        raise AssertionError(step)


class PublicStageMaterialTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.object(config, "chat_json", side_effect=AssertionError("provider forbidden")).start()
        patch.object(config, "chat", side_effect=AssertionError("provider forbidden")).start()
        patch.object(config, "pmap", side_effect=lambda fn, items, **kw: [fn(item) for item in items]).start()

    def render(self, tracer=None):
        ws, corpus, done = world(), {"sessions": []}, set()
        tracer = tracer or FakeTracer()
        render_corpus({"domain_profile": {}, "quality_contract": {"corpus_review": True}},
                      ws, 0, tracer, corpus, done, lambda: None, log=lambda *_: None)
        return ws, corpus, done, tracer

    def test_original_render_publishes_body_and_survives_sanitize(self):
        ws, corpus, done, tracer = self.render()
        rule_docs = [d for s in corpus["sessions"] for d in s["docs"] if d.get("public_rule_refs")]
        self.assertEqual(len(rule_docs), 1)
        self.assertEqual(rule_docs[0]["fact_refs"], [])
        self.assertIn("2025-01-06", rule_docs[0]["content"])
        self.assertIn("项目的状态阶段顺序为立项、进行中、收尾、完成", rule_docs[0]["content"])
        self.assertNotIn("岚桥项目", rule_docs[0]["content"])
        self.assertEqual(done, {0, 1})
        self.assertEqual(validate_corpus(ws, corpus)["status"], "passed")
        frozen = deepcopy(corpus)
        self.assertFalse(_render_public_stage_material({}, ws, tracer, corpus))
        self.assertEqual(_sanitize_corpus(corpus, ws)["dropped_unref"], 0)
        self.assertEqual(corpus, frozen)
        authors = [m[-1]["content"] for step, m in tracer.calls
                   if step == "render.signal" and "【冻结的公共阶段定义】" in m[-1]["content"]]
        self.assertEqual(len(authors), 1)
        self.assertNotIn("岚桥项目", authors[0])
        self.assertNotIn('"gt"', authors[0])
        self.assertNotIn('"cur_state"', authors[0])

    def test_deleting_rule_material_fails_public_coverage(self):
        ws, corpus, _, _ = self.render()
        corpus["sessions"][0]["docs"] = [d for d in corpus["sessions"][0]["docs"] if not d.get("public_rule_refs")]
        self.assertIn("missing_public_rule_material", {x["code"] for x in validate_corpus(ws, corpus)["issues"]})

    def test_missing_semantics_repairs_existing_author_without_gold(self):
        tracer = FakeTracer(failures=[True, False])
        ws, corpus, _, _ = self.render(tracer)
        authors = [m[-1]["content"] for step, m in tracer.calls
                   if step == "render.signal" and "【冻结的公共阶段定义】" in m[-1]["content"]]
        self.assertEqual(len(authors), 2)
        self.assertIn("正文未表达适用顺序", authors[-1])
        self.assertNotIn("岚桥项目", authors[-1])
        self.assertEqual(public_rule_coverage_issues(ws, corpus), [])

    def test_malformed_review_is_execution_failure_without_rule_document(self):
        ws, corpus, tracer = world(), {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": []}]}, FakeTracer(error=True)
        with self.assertRaises(CorpusReviewExecutionError):
            _render_public_stage_material({}, ws, tracer, corpus)
        self.assertEqual(corpus["sessions"][0]["docs"], [])
        # A parsed malformed opinion gets the existing single format correction; neither attempt publishes a rule.
        self.assertEqual([step for step, _ in tracer.calls], ["render.signal", "corpus.review", "corpus.review"])

    def test_negative_semantics_exhausts_budget_without_publishing(self):
        ws, corpus, tracer = world(), {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": []}]}, FakeTracer(failures=[True] * 4)
        with self.assertRaises(RuntimeError):
            _render_public_stage_material({}, ws, tracer, corpus)
        self.assertEqual(corpus["sessions"][0]["docs"], [])
        self.assertEqual(len(tracer.calls), 8)

    def test_exploratory_rule_review_exhaustion_keeps_attempts_for_downstream(self):
        ws = world()
        corpus = {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": []}]}
        tracer = FakeTracer(failures=[True] * 4)
        with self.assertRaises(MaterialRejected) as caught:
            _render_public_stage_material({}, ws, tracer, corpus,
                                          allow_exploratory_review=True)
        self.assertEqual(len(caught.exception.report["drafts"]), 4)
        self.assertEqual(caught.exception.report["task"]["kind"], "public_stage_rules")
        self.assertEqual(corpus["sessions"][0]["docs"], [])
        self.assertEqual(len(tracer.calls), 8)

    def test_reference_tags_are_not_semantic_evidence(self):
        ws, corpus, _, _ = self.render()
        original = deepcopy(corpus)
        for mutation in ("body", "refs", "date", "source"):
            candidate, changed_ws = deepcopy(original), deepcopy(ws)
            doc = next(d for d in candidate["sessions"][0]["docs"] if d.get("public_rule_refs"))
            if mutation == "body": doc["content"] = "阶段另行通知。"
            if mutation == "refs": doc["public_rule_refs"] = ["invented_rule"]
            if mutation == "date": candidate["sessions"][0]["date"] = "2039-01-01"
            if mutation == "source": changed_ws.world_blueprint["entity_types"][0]["fields"][0]["states"].insert(3, "复核")
            with self.subTest(mutation=mutation):
                self.assertEqual(validate_corpus(changed_ws, candidate)["status"], "failed")

    def test_title_only_coverage_is_rejected(self):
        ws = world()
        rule_id = public_stage_rules(ws)[0]["rule_id"]
        class TitleOnly:
            def chat_json(self, _step, messages, **_kw):
                target = json.loads(messages[-1]["content"])["requirements"][0]
                return {"document_reviews": fixed_document_reviews([{}]), "verdict": "pass", "unsupported_claims": [], "coverage": [
                    {"requirement_id": target["requirement_id"], "status": "supported", "reason": "fixture",
                     "evidence": [{"doc_index": 0, "quote": "阶段顺序在标题"}]}]}
        result = review_documents(TitleOnly(), ws, 0,
            [{"title": "阶段顺序在标题", "content": "仅有一段空泛介绍。"}], required_public_rule_ids=[rule_id])
        self.assertEqual(result["status"], "error")

    def test_legacy_and_non_lifecycle_fields_add_no_rule_calls(self):
        ws = world()
        field = ws.world_blueprint["entity_types"][0]["fields"][0]
        variants = [{}, {"kind": "status"}, {"kind": "status", "states": ["单一类别"]},
                    {"kind": "category", "states": ["甲类", "乙类"]}]
        class NoCalls:
            def chat_json(self, *a, **k): raise AssertionError("unexpected rule call")
        for variant in variants:
            field.clear(); field.update({"name": "状态", **variant})
            self.assertEqual(public_stage_rules(ws), [])
            self.assertFalse(_render_public_stage_material({}, ws, NoCalls(), {"sessions": []}))

    def test_l8_uses_entity_type_and_never_flat_profile_override(self):
        ws = world()
        ws.entities["乙工单"] = {"状态": Timeline([Op(0, _date_of(0), SET, "收尾")])}
        ws.entity_types["乙工单"] = "ticket"
        ws.world_blueprint["entity_types"].append({"id": "ticket", "noun": "工单", "fields": [
            {"name": "状态", "kind": "status", "states": ["进行中", "收尾", "复核", "关闭"]}]})
        profile = {"state_machines": [{"field": "状态", "states": ["收尾", "错误的统一后继"]}]}
        line = TransitionLine()
        orders = {o["entity"]: o for o in line.enumerate(ws, wp={"domain_profile": profile})}
        self.assertEqual(orders["岚桥项目"]["gt"], "完成")
        self.assertEqual(orders["乙工单"]["gt"], "复核")
        for order in orders.values():
            self.assertEqual(line.gt(ws, order), order["gt"])
            self.assertEqual(line.well_posed(order, ws)[0], "well_posed")
        corrupted = deepcopy(orders["乙工单"])
        corrupted["aux"]["states"] = orders["岚桥项目"]["aux"]["states"]
        corrupted["gt"] = "完成"
        self.assertEqual(line.gt(ws, corrupted), "复核")
        self.assertEqual(line.well_posed(corrupted, ws)[0], "drop")
        wording, _ = line.intent(orders["岚桥项目"])
        self.assertIn("紧邻", wording)
        self.assertIn("不是预测下一次实际状态变更", wording)
        self.assertNotIn("完成", wording)
        self.assertEqual(len(public_stage_rules(ws)), 2)


if __name__ == "__main__":
    unittest.main()
