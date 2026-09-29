"""Network-free integration controls; scripted opinions do not prove LLM accuracy."""
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
os.environ.setdefault("OPENAI_API_KEY", "offline-corpus-fidelity")
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline import corpus_contract as cc
from pipeline.prompts import render
from pipeline.render import (_corpus_system, _discriminate_many, _sanitize_corpus,
                             _strict_eq, _attach_story_provenance, render_corpus)
from pipeline.world_state import WorldState, Timeline, Op, SET, EXPIRE, _date_of


def world(*, rich=False):
    ws = WorldState(entities={"甲项目": {"阶段": Timeline([Op(0, _date_of(0), SET, "收尾")])}}, n_sessions=1)
    ws.entity_types = {"甲项目": "project"}
    ws.world_blueprint = {"entity_types": [{"id": "project", "fields": [
        {"name": "阶段", "kind": "status"}, {"name": "预算", "kind": "number", "unit": "万元"}]}]}
    if rich:
        for field, value in (("次数", 0), ("启用", False)):
            ws.entities["甲项目"][field] = Timeline([Op(0, _date_of(0), SET, value)])
        ws.entities["甲项目"]["统计"] = Timeline([Op(0, _date_of(0), EXPIRE, None)])
        ws.events = [{"id": "e1", "type": "transfer", "session": 0,
            "participants": {"project": "甲项目", "team": "乙团队"},
            "effects": [{"entity": "甲项目", "field": "负责人", "set": "乙团队"}]}]
    return ws


def output(payload, *, status="supported", index=0):
    return {"document_reviews": fixed_document_reviews(payload["documents"]),
            "verdict": "pass" if status == "supported" else "fail", "unsupported_claims": [],
            "coverage": [{"requirement_id": target["requirement_id"], "status": status,
                "evidence": ([{"doc_index": index, "quote": payload["documents"][index]["content"]}]
                             if status != "missing" else []),
                "reason": "固定离线意见，只验证消息/执行/证据保全。"}
                for target in payload["requirements"]]}


class Trace:
    def __init__(self, outputs=None, *, status="supported", bodies=None):
        self.calls, self.outputs, self.status = [], list(outputs or []), status
        self.bodies = list(bodies or ["甲项目目前已进入收尾。"])
        self.signal_count = 0

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if step == "render.signal":
            body = self.bodies[min(self.signal_count, len(self.bodies)-1)]
            self.signal_count += 1
            return {"docs": [{"title": "记录", "content": body}]}
        if step == "render.discriminate":
            text = messages[-1]["content"]
            begin = text.index("[{\"key\"")
            queries, _ = json.JSONDecoder().raw_decode(text[begin:])
            return {"answers": [{"key": q["key"], "answer": "进入收尾"} for q in queries]}
        if step == "corpus.review":
            payload = json.loads(messages[-1]["content"])
            if self.outputs:
                raw = self.outputs.pop(0)
                if isinstance(raw, Exception):
                    raise raw
                return raw(payload) if callable(raw) else deepcopy(raw)
            return output(payload, status=self.status)
        raise AssertionError(step)


def reviewed(ws, *, prefix="a", identical=False, single=False):
    bodies = ["甲项目目前已进入收尾。"] if single else ["甲项目目前已进入收尾。", "本组记录在当日归档。"]
    docs = [{"title": "同题" if identical else f"{prefix}{i}", "content": bodies[0] if identical else body}
            for i, body in enumerate(bodies)]
    trace = Trace()
    report = cc.review_documents(trace, ws, 0, docs, requirements=cc.fidelity_requirements(ws, 0),
        blind_reads=[{"key": "q0", "entity": "甲项目", "field": "阶段", "answer": "进入收尾"}])
    assert report["status"] == "passed", report
    cc.attach_receipts(docs, report, 0)
    for i, doc in enumerate(docs): doc["doc_id"] = f"s0_sig_{prefix}{i}"
    return docs, report, trace


def corpus(docs):
    return {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": deepcopy(docs)}]}


class FidelityTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)
        guard = patch.object(config, "pmap", side_effect=lambda fn, xs, **kw: [fn(x) for x in xs])
        guard.start(); self.addCleanup(guard.stop)

    def run_render(self, trace, *, semantic=True):
        result = {"sessions": []}
        render_corpus({"quality_contract": {"corpus_review": semantic}}, world(), 0, trace,
                      result, set(), lambda: None, log=lambda *_: None)
        return result

    def test_real_render_natural_equivalence_single_review_and_no_gold_in_blind(self):
        trace = Trace()
        result = self.run_render(trace)
        self.assertEqual([x[0] for x in trace.calls], ["render.signal", "render.discriminate", "corpus.review"])
        blind = trace.calls[1][1]
        self.assertEqual(blind[0]["content"], render("discriminate.quality_system"))
        self.assertNotIn('"value"', blind[-1]["content"])
        self.assertNotIn("万元", blind[-1]["content"])
        self.assertNotIn("requirements", blind[-1]["content"])
        payload = json.loads(trace.calls[2][1][-1]["content"])
        self.assertEqual(payload["blind_reads"][0]["answer"], "进入收尾")
        self.assertEqual(payload["requirements"][0]["target"]["value"], "收尾")
        self.assertEqual(payload["CANON"]["field_schemas"][0]["fields"][1]["unit"], "万元")
        self.assertTrue(payload["lexical_diagnostics"])
        self.assertEqual(cc.validate_corpus(world(), result)["status"], "passed")
        self.assertIn("目前", result["sessions"][0]["docs"][0]["content"])
        self.assertEqual(result["sessions"][0]["docs"][0]["fact_refs"], ["甲项目.阶段"])

    def test_zero_false_stop_event_targets_and_original_values_are_not_rewritten(self):
        ws = world(rich=True); before = ws.to_dict()
        req = cc.fidelity_requirements(ws, 0)
        self.assertEqual([r["kind"] for r in req], ["value", "value", "value", "stop", "event"])
        self.assertEqual(req[1]["target"]["value"], 0)
        self.assertIs(req[2]["target"]["value"], False)
        self.assertEqual([r["requirement_id"] for r in req], [f"r{i}" for i in range(1, 6)])
        self.assertEqual(ws.to_dict(), before)

    def test_single_array_maps_source_rule_and_fidelity_without_model_long_ids(self):
        ws = world(); ws.world_blueprint["entity_types"][0]["fields"][0]["states"] = ["开展", "收尾"]
        ws.conflicts = [{"entity": "甲项目", "field": "阶段", "session": 0, "date": _date_of(0),
                         "authoritative_source": "正式通报", "authoritative_value": "收尾"}]
        context = cc.canonical_context(ws, 0)
        context["source_assertions"] = cc.authoritative_source_assertions(ws, 0)
        self.assertTrue(context["source_assertions"])
        trace = Trace(); doc = {"content": "甲项目处于收尾，见正式通报。流程为开展、收尾。"}
        report = cc.review_documents(trace, ws, 0, [doc], context=context,
            requirements=cc.fidelity_requirements(ws, 0),
            required_public_rule_ids=[context["public_stage_rules"][0]["rule_id"]])
        self.assertEqual(report["status"], "passed", report)
        payload = json.loads(trace.calls[0][1][-1]["content"])
        self.assertEqual([r["requirement_id"] for r in payload["requirements"]], ["r1", "r2", "r3"])
        self.assertEqual([len(report[k]) for k in ("fidelity_coverage", "public_rule_coverage", "public_source_coverage")], [1, 1, 1])
        self.assertEqual(set(report["raw_output"]), {"document_reviews", "verdict", "unsupported_claims", "coverage"})

    def test_negative_or_ambiguous_is_completed_semantic_feedback_not_error(self):
        for status in ("missing", "incorrect", "ambiguous"):
            with self.subTest(status=status):
                trace = Trace(status=status)
                report = cc.review_documents(trace, world(), 0, [{"content": "甲项目尚无结论。"}],
                    requirements=cc.fidelity_requirements(world(), 0))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["fidelity_coverage"][0]["status"], status)
                self.assertEqual(report["raw_output"]["coverage"][0]["status"], status)

    def test_negative_repairs_existing_path_and_binds_final_body_only(self):
        trace = Trace(outputs=[lambda p: output(p, status="incorrect"), lambda p: output(p)],
                      bodies=["甲项目没有进入收尾。", "甲项目目前已进入收尾。"])
        result = self.run_render(trace)
        self.assertEqual([r[0] for r in trace.calls].count("render.signal"), 2)
        self.assertEqual([r[0] for r in trace.calls].count("corpus.review"), 2)
        self.assertIn("fidelity_not_supported", trace.calls[3][1][-1]["content"])
        self.assertNotIn("逐字照抄", trace.calls[3][1][-1]["content"])
        self.assertEqual(cc.validate_corpus(world(), result)["status"], "passed")

    def test_bad_shape_or_provider_error_stops_before_rewrite_and_keeps_raw(self):
        for raw in ({"verdict": "pass", "unsupported_claims": [], "coverage": []}, TimeoutError("deadline")):
            trace = Trace(outputs=[raw, deepcopy(raw)] if isinstance(raw, dict) else [raw])
            with self.assertRaises(cc.CorpusReviewExecutionError) as caught:
                self.run_render(trace)
            self.assertEqual(len(trace.calls), 4 if isinstance(raw, dict) else 3)
            if isinstance(raw, dict): self.assertEqual(caught.exception.report["raw_output"], raw)
            else: self.assertEqual(caught.exception.report["error_type"], "TimeoutError")

    def test_signal_or_blind_execution_error_never_reaches_another_role(self):
        for failing_step in ("render.signal", "render.discriminate"):
            class Broken(Trace):
                def chat_json(self, step, messages, **kwargs):
                    if step == failing_step:
                        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
                        return {"__error__": "fixed offline failure"}
                    return super().chat_json(step, messages, **kwargs)
            trace = Broken()
            with self.assertRaises(RuntimeError): self.run_render(trace)
            self.assertEqual(trace.calls[-1][0], failing_step)
            self.assertNotIn("corpus.review", [row[0] for row in trace.calls])

    def test_unknown_blind_answer_is_not_replaced_or_automatic_failure(self):
        class Unknown(Trace):
            def chat_json(self, step, messages, **kwargs):
                result = super().chat_json(step, messages, **kwargs)
                if step == "render.discriminate":
                    result["answers"][0]["answer"] = "不确定是否已经进入该阶段"
                return result
        trace = Unknown(); result = self.run_render(trace)
        payload = json.loads(trace.calls[-1][1][-1]["content"])
        self.assertEqual(payload["blind_reads"][0]["answer"], "不确定是否已经进入该阶段")
        self.assertEqual(len(trace.calls), 3)
        self.assertEqual(cc.validate_corpus(world(), result)["status"], "passed")

    def test_coverage_shape_location_and_old_raw_do_not_auto_upgrade(self):
        target = [{"content": "甲项目已进入收尾。"}]
        def mutate(kind):
            def response(payload):
                raw = output(payload)
                if kind == "old": return {"verdict": "pass", "unsupported_claims": [], "public_rule_coverage": [], "public_source_coverage": []}
                if kind == "duplicate": raw["coverage"] *= 2
                if kind == "unknown": raw["coverage"][0]["requirement_id"] = "r999"
                if kind == "quote": raw["coverage"][0]["evidence"][0]["quote"] = "编造的原文"
                if kind == "missing_quote": raw["coverage"][0]["evidence"][0].pop("quote")
                if kind == "disagreement": raw["verdict"] = "fail"
                return raw
            return response
        for kind in ("old", "duplicate", "unknown", "quote", "missing_quote", "disagreement"):
            report = cc.review_documents(Trace(outputs=[mutate(kind), mutate(kind)]), world(), 0, target,
                                         requirements=cc.fidelity_requirements(world(), 0))
            self.assertEqual(report["status"], "error", kind)
            self.assertIsNotNone(report["raw_output"])

    def test_invalid_target_rejected_before_provider(self):
        req = cc.fidelity_requirements(world(), 0); req[0]["target"]["value"] = "完成"
        trace = Trace(); report = cc.review_documents(trace, world(), 0, [], requirements=req)
        self.assertEqual(report["status"], "error")
        self.assertEqual(trace.calls, [])

    def test_raw_world_event_is_normalized_only_for_declared_label(self):
        ws = world(rich=True)
        expected = cc.fidelity_requirements(ws, 0, facts=[])
        self.assertEqual(cc.fidelity_requirements(ws, 0, facts=[], events=ws.events), expected)
        changed = deepcopy(ws.events); changed[0]["label"] = "invented label"
        with self.assertRaises(ValueError): cc.fidelity_requirements(ws, 0, facts=[], events=changed)
        changed = deepcopy(ws.events); changed[0]["participants"]["team"] = "丙团队"
        with self.assertRaises(ValueError): cc.fidelity_requirements(ws, 0, facts=[], events=changed)

    def test_uncited_group_member_survives_sanitize_without_invented_refs(self):
        docs, _, _ = reviewed(world()); candidate = corpus(docs)
        stats = _sanitize_corpus(candidate, world(), semantic=True)
        self.assertEqual(stats["dropped_unref"], 0)
        self.assertEqual(candidate["sessions"][0]["docs"][1]["fact_refs"], [])
        self.assertEqual(cc.validate_corpus(world(), candidate)["status"], "passed")

    def test_any_deleted_group_member_or_all_target_docs_fail(self):
        docs, _, _ = reviewed(world())
        for kept in (docs[:1], docs[1:], []):
            issues = cc.validate_corpus(world(), corpus(kept))["issues"]
            self.assertIn("missing_fidelity_material", {item["code"] for item in issues})

    def test_duplicate_slot_and_partial_groups_do_not_prove_coverage(self):
        docs, _, _ = reviewed(world(), identical=True)
        clone = deepcopy(docs[0]); clone["doc_id"] += "copy"
        self.assertEqual(cc.validate_corpus(world(), corpus([docs[0], clone]))["status"], "failed")
        first, _, _ = reviewed(world(), prefix="first")
        second, _, _ = reviewed(world(), prefix="second")
        self.assertEqual(cc.validate_corpus(world(), corpus([first[0], second[1]]))["status"], "failed")
        alternative, _, _ = reviewed(world(), prefix="single", single=True)
        self.assertEqual(cc.validate_corpus(world(), corpus(first[:1] + alternative))["status"], "passed")

    def test_reorder_and_document_id_assignment_preserve_original_slots(self):
        docs, _, _ = reviewed(world()); docs.reverse()
        for i, doc in enumerate(docs): doc["doc_id"] = f"s0_sig_renumbered{i}"
        self.assertEqual(cc.validate_corpus(world(), corpus(docs))["status"], "passed")

    def test_bound_changes_and_empty_coverage_cannot_silently_pass_release(self):
        docs, _, _ = reviewed(world())
        for kind in ("body", "title", "date", "blind", "requirement", "version", "schema", "group", "coverage"):
            ws, candidate = world(), corpus(docs); doc = candidate["sessions"][0]["docs"][0]
            receipt = doc["quality_review"]
            if kind == "body": doc["content"] += "改动"
            elif kind == "title": doc["title"] += "改动"
            elif kind == "date": candidate["sessions"][0]["date"] = "2025-02-01"
            elif kind == "blind": receipt["fidelity"]["blind_reads"][0]["answer"] = "完成"
            elif kind == "requirement": receipt["fidelity"]["requirements"][0]["target"]["value"] = "完成"
            elif kind == "version": receipt["version"] -= 1
            elif kind == "schema": ws.world_blueprint["entity_types"][0]["fields"][1]["unit"] = "元"
            elif kind == "group": receipt["fidelity_support_sets"] = {}
            else: receipt["fidelity"]["coverage"] = []
            self.assertEqual(cc.validate_corpus(ws, candidate)["status"], "failed", kind)

    def test_supportedness_only_receipt_cannot_claim_ordinary_coverage(self):
        docs = [{"doc_id": "s0_sig_0", "content": "甲项目已进入收尾。"}]
        report = cc.review_documents(Trace(), world(), 0, docs)
        self.assertEqual(report["status"], "passed")
        cc.attach_receipts(docs, report, 0)
        self.assertEqual(cc.validate_corpus(world(), corpus(docs))["status"], "failed")

    def test_event_provenance_comes_from_valid_evidence_not_word_cooccurrence(self):
        ws = world(rich=True); doc = {"doc_id": "s0_sig_0", "content": "这次交接由乙接手，甲的工作交给该团队。"}
        report = cc.review_documents(Trace(), ws, 0, [doc], requirements=cc.fidelity_requirements(ws, 0))
        cc.attach_receipts([doc], report, 0)
        _attach_story_provenance([doc], ws.events, [{"scene_id": "scene1", "event_refs": ["e1"]}], ws=ws, session=0)
        self.assertEqual(doc["event_refs"], ["e1"])
        self.assertEqual(doc["scene_refs"], ["scene1"])
        doc["content"] += "changed"
        _attach_story_provenance([doc], ws.events, [], ws=ws, session=0)
        self.assertEqual(doc["event_refs"], [])

    def test_legacy_prompt_and_strict_comparison_are_preserved(self):
        self.assertFalse(_strict_eq("进入收尾", "收尾"))
        self.assertFalse(_strict_eq("78%", "0.78"))
        trace = Trace(); _discriminate_many(["甲项目阶段为收尾。"], [{"key": "q0", "entity": "甲项目", "field": "阶段"}], trace)
        self.assertEqual(trace.calls[0][1][0]["content"], render("discriminate.system"))
        self.assertIn("逐字", trace.calls[0][1][-1]["content"])
        self.assertIn("防剧透·硬约束", _corpus_system({}))
        self.assertNotIn("防剧透·硬约束", _corpus_system({}, semantic=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
