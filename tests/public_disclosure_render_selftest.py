"""Offline integration checks; scripted opinions are not LLM quality evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("OPENAI_API_KEY", "offline-disclosure-render")
import config
from pipeline import corpus_contract as cc, disclosure, render as renderer
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, EXPIRE, _date_of
from corpus_fixture_helpers import fixed_document_reviews

WP = {"quality_contract": {"public_disclosure": True, "corpus_review": True}}


def world():
    return WorldState(entities={"甲项目": {
        "状态": Timeline([Op(0, _date_of(0), SET, "旧甲"), Op(1, _date_of(1), UPDATE, "新乙", "旧甲")]),
        "隐藏参数": Timeline([Op(0, _date_of(0), SET, "PRIVATE_SENTINEL_7901")])}}, n_sessions=3)


def positive(payload):
    return {"document_reviews": fixed_document_reviews(payload["documents"]),
            "verdict": "pass", "unsupported_claims": [],
            "coverage": [{"requirement_id": row["requirement_id"], "status": "supported",
                "evidence": [{"doc_index": 0, "quote": payload["documents"][0]["content"]}],
                "reason": "固定离线意见，只验证接线和收据，不证明语义。"} for row in payload["requirements"]]}


class Trace:
    def __init__(self, raw_plan=None):
        self.raw_plan = raw_plan
        self.calls = []

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if step == disclosure.STEP:
            return deepcopy(self.raw_plan)
        if step == "render.signal":
            return {"docs": [{"title": "离线资料", "content": "本条离线正文仅用于检查调用、引用及上下文传递。"}]}
        if step == "render.discriminate":
            text = messages[-1]["content"]
            queries, _ = json.JSONDecoder().raw_decode(text[text.index('[{"key"'):])
            return {"answers": [{"key": row["key"], "answer": "离线盲读"} for row in queries]}
        if step == "corpus.review":
            return positive(json.loads(messages[-1]["content"]))
        raise AssertionError(step)

    def chat_text(self, *args, **kwargs):
        raise AssertionError("No filler required in these controls")


def planned(ws, groups):
    """groups=[(session,[catalogue entries], optional channel)]."""
    inventory = disclosure.catalogue(ws)
    fixed = disclosure._fixed(ws, inventory)
    used = {ref for row in fixed for ref in row["refs"]}
    records = []
    for group in groups:
        session, items = group[:2]
        refs = [item["ref"] for item in items]
        used.update(refs)
        records.append({"session": session, "refs": refs,
            "channel": group[2] if len(group) > 2 else "回顾记录",
            "acquisition_context": "离线安排：在本期通过回顾记录披露所列原时点信息。"})
    raw = {"records": records, "undisclosed": [x["ref"] for x in inventory if x["ref"] not in used],
           "reason": "离线结构测试，不证明业务安排合理。"}
    report = disclosure.author_plan(WP, ws, Trace(raw), task_input={"description": "离线用例"})
    assert report["status"] == "ready", report
    assert not disclosure.validate_plan(ws), disclosure.validate_plan(ws)
    return ws


def status_refs(ws):
    return [item for item in disclosure.catalogue(ws) if item["kind"] == "fact" and item["field"] == "状态"]


def render_all(ws):
    tracer = Trace()
    corpus = {"sessions": []}
    with patch.object(config, "pmap", side_effect=lambda fn, items, workers=6: [fn(item) for item in items]):
        renderer.render_corpus(WP, ws, 0, tracer, corpus, set(), lambda: None, lambda *args: None)
    return corpus, tracer


class PublicDisclosureRenderTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def test_delayed_truth_is_not_opening_publication(self):
        ws = world(); before = ws.to_dict()
        planned(ws, [(2, [status_refs(ws)[0]])])
        self.assertEqual(renderer._session_facts(ws, 0), [])
        self.assertEqual(cc.fidelity_requirements(ws, 0), [])
        target = cc.fidelity_requirements(ws, 2)[0]["target"]
        self.assertEqual((target["fact_session"], target["disclosure_session"], target["value"]), (0, 2, "旧甲"))
        self.assertEqual({k: v for k, v in ws.to_dict().items() if k != "disclosure"}, before)
        self.assertNotIn("旧甲", json.dumps(cc.canonical_context(ws, 0), ensure_ascii=False))

    def test_declared_plan_missing_stops_direct_renderer_before_calls(self):
        tracer = Trace()
        with self.assertRaisesRegex(ValueError, "plan is missing"):
            renderer.render_corpus(WP, world(), 0, tracer, {"sessions": []}, set(),
                                   lambda: None, lambda *args: None)
        self.assertEqual(tracer.calls, [])
        ws = world(); planned(ws, [])
        ws.disclosure["plan_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "plan is invalid"):
            renderer.render_corpus({"quality_contract": {"corpus_review": True}}, ws, 0, tracer,
                                   {"sessions": []}, set(), lambda: None, lambda *args: None)
        self.assertEqual(tracer.calls, [])

    def test_history_keeps_old_and_new_without_private_values(self):
        ws = world(); planned(ws, [(2, status_refs(ws))])
        context = cc.canonical_context(ws, 2)
        self.assertEqual([row["value"] for row in context["facts"]], ["旧甲", "新乙"])
        self.assertNotIn("PRIVATE_SENTINEL_7901", json.dumps(context, ensure_ascii=False))
        self.assertEqual(cc.explicit_future_claims(ws, 0, "甲项目隐藏参数为PRIVATE_SENTINEL_7901。"), [])

    def test_actual_render_shares_projection_and_temporal_blind_queries(self):
        ws = world(); planned(ws, [(2, status_refs(ws))])
        corpus, trace = render_all(ws)
        author = next(messages[-1]["content"] for step, messages, _ in trace.calls if step == "render.signal")
        marker = "【生成与审阅共享的截至时点事实；不得把未知业务状态写成已发生】\n"
        author_context, _ = json.JSONDecoder().raw_decode(author.split(marker, 1)[1])
        review = next(json.loads(messages[-1]["content"]) for step, messages, _ in trace.calls if step == "corpus.review")
        self.assertEqual(author_context, review["CANON"])
        self.assertEqual([row["fact_session"] for row in review["blind_reads"]], [0, 1])
        self.assertEqual(len({row["target_ref"] for row in review["blind_reads"]}), 2)
        blind = next(messages for step, messages, _ in trace.calls if step == "render.discriminate")
        # This scripted body intentionally contains neither expected value.
        self.assertNotIn("旧甲", json.dumps(blind, ensure_ascii=False))
        self.assertNotIn("新乙", json.dumps(blind, ensure_ascii=False))
        self.assertNotIn("PRIVATE_SENTINEL_7901", json.dumps(trace.calls, ensure_ascii=False))
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "passed")
        changed = deepcopy(review["blind_reads"]); changed[0]["fact_session"] = 1
        with self.assertRaises(ValueError): cc._validate_blind_reads(changed, cc.fidelity_requirements(ws, 2))

    def test_repeat_publication_is_independent_obligation(self):
        ws = world(); ref = status_refs(ws)[0]; planned(ws, [(1, [ref]), (2, [ref])])
        first, second = cc.fidelity_requirements(ws, 1)[0], cc.fidelity_requirements(ws, 2)[0]
        self.assertNotEqual(cc._fidelity_key(first), cc._fidelity_key(second))
        corpus, _ = render_all(ws)
        corpus["sessions"][2]["docs"] = []
        self.assertTrue(any(row["code"] == "missing_fidelity_material" and row["session"] == 2
                            for row in cc.fidelity_coverage_issues(ws, corpus)))

    def test_delayed_stop_keeps_effective_time_in_author_review_and_blind_query(self):
        ws = WorldState(entities={"甲项目": {"状态": Timeline([
            Op(0, _date_of(0), SET, "旧甲"),
            Op(1, _date_of(1), EXPIRE, prev="旧甲")])}}, n_sessions=3)
        stopped = next(item for item in status_refs(ws) if item["stopped"])
        planned(ws, [(2, [stopped])])
        before = ws.to_dict()["entities"]
        corpus, trace = render_all(ws)
        author = next(messages for step, messages, _ in trace.calls if step == "render.signal")
        self.assertNotIn("自本期起", author[0]["content"])
        self.assertIn("原生效时点", author[0]["content"])
        review = next(json.loads(messages[-1]["content"]) for step, messages, _ in trace.calls
                      if step == "corpus.review")
        fact = review["CANON"]["facts"][0]
        self.assertEqual((fact["fact_session"], fact["fact_date"], fact["disclosure_session"]),
                         (1, _date_of(1), 2))
        target = next(row["target"] for row in review["requirements"] if row["kind"] == "stop")
        self.assertEqual((target["fact_session"], target["fact_date"]), (1, _date_of(1)))
        query = review["blind_reads"][0]
        self.assertEqual((query["fact_session"], query["fact_date"]), (1, _date_of(1)))
        blind = next(messages for step, messages, _ in trace.calls if step == "render.discriminate")
        queries, _ = json.JSONDecoder().raw_decode(
            blind[-1]["content"][blind[-1]["content"].index('[{"key"'):])
        self.assertEqual((queries[0]["fact_session"], queries[0]["fact_date"]), (1, _date_of(1)))
        self.assertEqual(ws.to_dict()["entities"], before)
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "passed")

    def test_event_only_group_does_not_require_same_period_fact(self):
        ws = world()
        ws.events = [{"id": "PRIVATE_ANCESTOR", "type": "private_action", "session": 0,
            "participants": {"project": "甲项目"},
            "effects": [{"entity": "甲项目", "field": "私有动作", "set": "PARENT_PRIVATE_DETAIL"}]},
            {"id": "evt", "type": "meeting", "session": 1, "date": _date_of(1),
             "participants": {"project": "甲项目"}, "effects": [], "caused_by": "PRIVATE_ANCESTOR"}]
        event = next(x for x in disclosure.catalogue(ws) if x["kind"] == "event" and x["event"]["id"] == "evt")
        planned(ws, [(2, [event])])
        corpus, trace = render_all(ws)
        self.assertEqual(len([c for c in trace.calls if c[0] == "render.signal"]), 1)
        self.assertEqual(cc.fidelity_requirements(ws, 2)[0]["kind"], "event")
        # The declared causal link is public with this event. Its opaque target
        # ID does not recursively publish that parent's private account.
        self.assertIn("PRIVATE_ANCESTOR", json.dumps(trace.calls, ensure_ascii=False))
        self.assertNotIn("PARENT_PRIVATE_DETAIL", json.dumps(trace.calls, ensure_ascii=False))
        self.assertEqual([event["id"] for event in cc.canonical_context(ws, 2)["events"]], ["evt"])
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "passed")

    def test_private_story_function_does_not_enter_author(self):
        ws = world()
        ws.narrative = {"private": "UNREVEALED_STORY_OUTCOME"}
        planned(ws, [(2, [status_refs(ws)[0]])])
        with patch.object(renderer, "replay_story_ledger", return_value=[{
                "scene_id": "x", "session": 0, "event_refs": [], "dramatic_function": "UNREVEALED_STORY_OUTCOME"}]):
            corpus, trace = render_all(ws)
        self.assertNotIn("UNREVEALED_STORY_OUTCOME", json.dumps(trace.calls, ensure_ascii=False))

    def test_never_published_does_not_force_missing_fidelity(self):
        ws = world(); planned(ws, [])
        corpus, trace = render_all(ws)
        self.assertEqual(trace.calls, [])
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "passed")
        # This certifies no questions; original public grounding must still
        # reject references lacking readable support.

    def test_private_custom_context_rejected_before_review(self):
        ws = world(); planned(ws, [(2, [status_refs(ws)[0]])])
        context = cc.canonical_context(ws, 2); context["private_truth"] = "PRIVATE_SENTINEL_7901"
        trace = Trace()
        report = cc.review_documents(trace, ws, 2, [{"content": "资料"}], context=context,
                                     requirements=cc.fidelity_requirements(ws, 2))
        self.assertEqual((report["status"], trace.calls), ("error", []))

    def test_plan_change_invalidates_receipt_and_sanitize_has_no_literal_fallback(self):
        ws = world(); refs = status_refs(ws); planned(ws, [(2, [refs[0]])])
        corpus, _ = render_all(ws)
        doc = corpus["sessions"][2]["docs"][0]
        self.assertTrue(cc._review_receipt_matches(ws, 2, doc))
        planned(ws, [(2, [refs[0]], "另一合法但不同的公开途径")])
        self.assertFalse(cc._review_receipt_matches(ws, 2, doc))
        with self.assertRaises(ValueError): renderer._sanitize_corpus(corpus, ws, semantic=True)
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "failed")

    def test_fixed_channels_still_require_plan_integrity(self):
        ws = world()
        ws.conflicts = [{"entity": "甲项目", "field": "状态", "session": 1, "date": _date_of(1),
            "authoritative_value": "新乙", "authoritative_source": "官方通报",
            "rumor_value": "传闻丙", "rumor_source": "未经核实的外部传闻"}]
        ws.sensitive = [{"entity": "甲项目", "field": "测试口令", "value": "synthetic_secret_123", "session": 1}]
        ws.rule_instances = [{"inst_id": "i1", "session": 1, "trigger_field": "测试量", "x": 3,
                              "unit": "次", "surface_action": "登记"}]
        planned(ws, [])
        self.assertEqual([x["value"] for x in renderer._session_facts(ws, 1)], ["新乙"])
        self.assertEqual(len(cc.authoritative_source_assertions(ws, 1)), 1)
        corpus, _ = render_all(ws)
        docs = corpus["sessions"][1]["docs"]
        self.assertTrue(all(any(doc.get(role) for doc in docs) for role in
                            ("is_conflict", "is_sensitive", "is_rule_instance")))
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "passed")
        ws.disclosure["records"] = []
        self.assertEqual(cc.validate_corpus(ws, corpus)["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
