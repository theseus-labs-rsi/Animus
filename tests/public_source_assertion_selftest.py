"""Offline original-renderer source binding checks; scripted reviews are not gold."""
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
os.environ.setdefault("OPENAI_API_KEY", "offline-source-binding")
os.environ.setdefault("MODEL", "offline-source-binding")

import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline.corpus_contract import (
    VERSION, CorpusReviewExecutionError, authoritative_source_assertions,
    attach_receipts, canonical_context, fidelity_requirements, public_source_coverage_issues,
    review_documents, validate_corpus,
)
from pipeline.render import _sanitize_corpus, render_corpus
from pipeline.world_state import Op, SET, Timeline, WorldState, _date_of


def world(names=("甲项目",), *, periods=1):
    ws = WorldState(entities={name: {"负责人": Timeline([
        Op(0, _date_of(0), SET, f"经办{i}")])} for i, name in enumerate(names)},
        n_sessions=periods)
    ws.conflicts = [
        {"entity": name, "field": "负责人", "session": 0, "date": _date_of(0),
         "authoritative_value": f"经办{i}", "authoritative_source": f"{name}签发记录",
         "rumor_value": "PRIVATE_RUMOR_DO_NOT_PASS", "rumor_source": "群聊转述",
         "rule": "source_reliability", "gt": "PRIVATE_GOLD_DO_NOT_PASS"}
        for i, name in enumerate(names)
    ]
    return ws


def source_context(ws, keys=None, session=0):
    context = canonical_context(ws, session)
    context["source_assertions"] = authoritative_source_assertions(ws, session, keys)
    return context


def opinion(payload, status="supported"):
    docs = payload["documents"]
    return {"document_reviews": fixed_document_reviews(payload["documents"]),
            "verdict": "pass" if status == "supported" else "fail", "unsupported_claims": [],
            "coverage": [
                {"requirement_id": item["requirement_id"],
                 "status": status if item["kind"] == "public_source" else "supported",
                 "reason": "离线意见桩只验证接线和收据，不证明语义正确。",
                 "evidence": ([] if status == "missing" and item["kind"] == "public_source" else [
                     {"doc_index": i, "quote": doc["content"]} for i, doc in enumerate(docs)
                     if item["kind"] == "public_source" or i == 0])}
                for item in payload["requirements"]]}


def source_only_opinion(context, docs, status="supported"):
    """Fixtures that deliberately test source review without fact coverage."""
    return opinion({"documents": docs, "requirements": [
        {"requirement_id": f"r{i + 1}", "kind": "public_source", "target": item}
        for i, item in enumerate(context["source_assertions"])]}, status)


class Review:
    def __init__(self, response=None):
        self.response, self.calls = response, []

    def chat_json(self, step, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        self.calls.append((step, deepcopy(payload)))
        if isinstance(self.response, Exception):
            raise self.response
        if callable(self.response):
            return self.response(payload)
        return deepcopy(self.response if self.response is not None else opinion(payload))


def reviewed_group(ws, name="group", *, identical=False, single=False):
    docs = [{"title": name, "content": "甲项目负责人经办0。"}]
    if not single:
        docs.append({"title": name if identical else name + "-来源",
                     "content": docs[0]["content"] if identical else "上述记录由甲项目签发记录提供。"})
    report = review_documents(Review(), ws, 0, docs, context=source_context(ws),
                              requirements=fidelity_requirements(ws, 0))
    assert report["status"] == "passed", report
    attach_receipts(docs, report, 0)
    for index, doc in enumerate(docs):
        doc["doc_id"] = f"s0_sig_{name}_{index}"
    return docs


def corpus(docs):
    return {"sessions": [{"session_id": 0, "date": _date_of(0), "docs": deepcopy(docs)}]}


class RendererTrace:
    """Provider-free fake at the existing tracer seam, using real render/review."""
    def __init__(self, ws, reviews=()):
        self.ws, self.reviews, self.calls = ws, list(reviews), []

    def chat_json(self, step, messages, **kwargs):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": kwargs})
        user = messages[-1]["content"]
        if step == "render.signal":
            facts = json.JSONDecoder().raw_decode(user[user.index('[{"entity":'):])[0]
            body = "。".join(f"{item['entity']}的{item['field']}为{item['value']}" for item in facts) + "。"
            # The fake author's output deliberately does not manufacture a
            # source sentence. The scripted semantic reviewer owns its judgment.
            return {"docs": [{"content": body}]}
        if step == "render.discriminate":
            queries = json.JSONDecoder().raw_decode(user[user.index('[{"key":'):])[0]
            return {"answers": [{"key": q["key"], "answer": self.ws.entities[q["entity"]][q["field"]].value_at_session(0)}
                                for q in queries]}
        if step == "corpus.review":
            payload = json.loads(user)
            result = self.reviews.pop(0) if self.reviews else "supported"
            if isinstance(result, Exception):
                raise result
            if isinstance(result, dict):
                return deepcopy(result)
            return opinion(payload, result)
        raise AssertionError(step)

    def chat_text(self, *args, **kwargs):
        raise AssertionError("zero-filler fixture must not call text provider")


class PublicSourceAssertionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.object(config, "chat_json", side_effect=AssertionError("provider forbidden")).start()
        patch.object(config, "chat", side_effect=AssertionError("provider forbidden")).start()
        patch.object(config, "pmap", side_effect=lambda fn, items, **kw: [fn(item) for item in items]).start()

    def render(self, ws, tracer, *, quality=True, existing=None, done=None):
        result = existing if existing is not None else {"sessions": []}
        render_corpus({"quality_contract": {"corpus_review": quality}, "domain_profile": {}},
                      ws, 0, tracer, result, done if done is not None else set(),
                      save_cb=lambda: None, log=lambda *_: None)
        return result

    def test_projection_is_exact_group_and_period_not_gold_or_rumor(self):
        ws = world(("甲项目", "乙项目"), periods=2)
        later = {**ws.conflicts[0], "session": 1, "date": _date_of(1),
                 "authoritative_source": "后期来源"}
        ws.conflicts.append(later)
        frozen = deepcopy(ws.to_dict())
        items = authoritative_source_assertions(ws, 0, {("甲项目", "负责人")})
        self.assertEqual(len(items), 1)
        self.assertEqual({k: items[0][k] for k in ("entity", "field", "session", "date", "value", "source")},
                         {"entity": "甲项目", "field": "负责人", "session": 0, "date": _date_of(0),
                          "value": "经办0", "source": "甲项目签发记录"})
        encoded = json.dumps(items, ensure_ascii=False)
        for hidden in ("乙项目", "后期来源", "PRIVATE_", "source_reliability"):
            self.assertNotIn(hidden, encoded)
        self.assertEqual(authoritative_source_assertions(ws, 0, {("甲项目", "不存在字段")}), [])
        self.assertEqual(ws.to_dict(), frozen)

    def test_real_renderer_writer_and_reviewer_receive_identical_filtered_contexts(self):
        ws = world(("甲项目", "乙项目", "丙项目"))
        tracer = RendererTrace(ws)
        original = deepcopy(ws.to_dict())
        result = self.render(ws, tracer)
        writers = [x for x in tracer.calls if x["step"] == "render.signal"]
        reviewers = [json.loads(x["messages"][-1]["content"]) for x in tracer.calls if x["step"] == "corpus.review"]
        self.assertEqual(len(writers), 2)
        self.assertEqual(len(reviewers), 2)
        for writer, reviewer in zip(writers, reviewers):
            text = writer["messages"][-1]["content"].split("【生成与审阅共享的截至时点事实；不得把未知业务状态写成已发生】\n")[1]
            context = json.JSONDecoder().raw_decode(text)[0]
            self.assertEqual(context, reviewer["CANON"])
            self.assertNotIn("PRIVATE_", json.dumps(context))
        self.assertEqual([{x["entity"] for x in p["CANON"]["source_assertions"]} for p in reviewers],
                         [{"甲项目", "乙项目"}, {"丙项目"}])
        self.assertEqual(validate_corpus(ws, result)["status"], "passed")
        self.assertEqual(ws.to_dict(), original)

    def test_legacy_renderer_adds_no_source_requirement_or_review(self):
        ws = world()
        tracer = RendererTrace(ws)
        result = self.render(ws, tracer, quality=False)
        self.assertTrue(result["sessions"])
        self.assertNotIn("corpus.review", [c["step"] for c in tracer.calls])
        for call in tracer.calls:
            self.assertNotIn("source_assertions", call["messages"][-1]["content"])

    def test_missing_source_is_semantic_feedback_in_existing_bounded_author_loop(self):
        ws = world()
        tracer = RendererTrace(ws, ["missing", "supported"])
        self.render(ws, tracer)
        writers = [c for c in tracer.calls if c["step"] == "render.signal"]
        self.assertEqual(len(writers), 2)
        self.assertIn("public_source_not_supported", writers[1]["messages"][-1]["content"])

    def test_missing_source_exhausts_original_four_attempt_limit(self):
        ws = world()
        tracer = RendererTrace(ws, ["missing"] * 4)
        with self.assertRaises(RuntimeError):
            self.render(ws, tracer)
        self.assertEqual([x["step"] for x in tracer.calls].count("render.signal"), 4)
        self.assertEqual([x["step"] for x in tracer.calls].count("corpus.review"), 4)

    def test_malformed_or_provider_review_stops_without_content_retry(self):
        for bad in ({"verdict": "pass", "unsupported_claims": []}, TimeoutError("offline timeout")):
            with self.subTest(bad=type(bad).__name__):
                ws = world()
                tracer = RendererTrace(ws, [bad, deepcopy(bad)] if isinstance(bad, dict) else [bad])
                with self.assertRaises(CorpusReviewExecutionError) as caught:
                    self.render(ws, tracer)
                self.assertEqual([x["step"] for x in tracer.calls],
                                 ["render.signal", "render.discriminate"]
                                 + ["corpus.review"] * (2 if isinstance(bad, dict) else 1))
                self.assertEqual(caught.exception.report["status"], "error")
                if isinstance(bad, dict):
                    self.assertEqual(caught.exception.report["raw_output"], bad)
                    attempts = caught.exception.report["review_validation"]["attempts"]
                    self.assertEqual([a["raw_output"] for a in attempts], [bad, bad])

    def test_supported_source_requires_body_quote_not_title(self):
        ws = world()
        context = source_context(ws)
        docs = [{"title": "甲项目签发记录记负责人经办0", "content": "档案整理。"}]
        raw = source_only_opinion(context, docs)
        raw["coverage"][0]["evidence"] = [{"doc_index": 0, "quote": docs[0]["title"]}]
        report = review_documents(Review(raw), ws, 0, docs, context=context)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["raw_output"], raw)

    def test_missing_repeated_or_changed_requirement_rejected_before_call(self):
        ws = world()
        base = source_context(ws)
        for mode in ("repeat", "source", "time", "entity", "value"):
            context = deepcopy(base)
            if mode == "repeat":
                context["source_assertions"] *= 2
            else:
                key = "session" if mode == "time" else mode
                context["source_assertions"][0][key] = 99 if key == "session" else "changed"
            reviewer = Review()
            report = review_documents(reviewer, ws, 0, [{"content": "记录。"}], context=context)
            self.assertEqual(report["status"], "error", mode)
            self.assertEqual(reviewer.calls, [], mode)

    def test_model_judgment_not_lexical_presence_decides_source_support(self):
        ws = world(); context = source_context(ws)
        docs = [{"content": "甲项目负责人经办0、甲项目签发记录仅列为词语例子，并未建立来源关系。"}]
        negative = source_only_opinion(context, docs, "missing")
        report = review_documents(Review(negative), ws, 0, docs, context=context)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["issues"][0]["code"], "public_source_not_supported")
        # Binding validation also does not pretend to prove a scripted positive
        # semantically correct: exact quotations plus an LLM opinion are receipts.
        docs = [{"content": "前述记录的签发者在此处得到对应说明。"}]
        report = review_documents(Review(), ws, 0, docs, context=context)
        self.assertEqual(report["status"], "passed")

    def test_all_original_group_members_required_even_uncited_member(self):
        ws = world(); docs = reviewed_group(ws)
        context = source_context(ws)
        def first_document_only(payload):
            raw = opinion(payload)
            for row in raw["coverage"]:
                row["evidence"] = [{"doc_index": 0, "quote": docs[0]["content"]}]
            return raw
        attach_receipts(docs, review_documents(Review(first_document_only), ws, 0, docs,
            context=context, requirements=fidelity_requirements(ws, 0)), 0)
        self.assertEqual(validate_corpus(ws, corpus(docs))["status"], "passed")
        for kept in (docs[:1], docs[1:]):
            self.assertIn("missing_public_source_material",
                          {x["code"] for x in validate_corpus(ws, corpus(kept))["issues"]})

    def test_different_partial_groups_cannot_be_combined_but_complete_alternative_can(self):
        ws = world(); first = reviewed_group(ws, "first"); second = reviewed_group(ws, "second")
        mixed = corpus([first[0], second[1]])
        self.assertTrue(public_source_coverage_issues(ws, mixed))
        alternative = reviewed_group(ws, "single", single=True)
        self.assertEqual(public_source_coverage_issues(ws, corpus(first[:1] + alternative)), [])

    def test_identical_text_cannot_replace_missing_original_slot(self):
        ws = world(); docs = reviewed_group(ws, identical=True)
        self.assertEqual(validate_corpus(ws, corpus(docs))["status"], "passed")
        duplicate = deepcopy(docs[0]); duplicate["doc_id"] = "s0_sig_duplicate"
        self.assertEqual(validate_corpus(ws, corpus([docs[0], duplicate]))["status"], "failed")

    def test_group_renumbering_and_reordering_is_safe_but_changes_are_stale(self):
        ws = world(); docs = reviewed_group(ws)
        reordered = list(reversed(deepcopy(docs)))
        for i, doc in enumerate(reordered): doc["doc_id"] = f"s0_sig_new_{i}"
        self.assertEqual(validate_corpus(ws, corpus(reordered))["status"], "passed")
        for mutation in ("content", "title", "date", "source", "requirements", "version", "group"):
            copy_ws, candidate = deepcopy(ws), corpus(docs)
            target = candidate["sessions"][0]["docs"][0]
            if mutation in ("content", "title"):
                target[mutation] = target.get(mutation, "") + "changed"
            elif mutation == "date": candidate["sessions"][0]["date"] = "2030-01-01"
            elif mutation == "source": copy_ws.conflicts[0]["authoritative_source"] = "另一来源"
            elif mutation == "requirements": target["quality_review"]["source_assertions"] = []
            elif mutation == "version": target["quality_review"]["version"] = VERSION - 1
            else: target["quality_review"].pop("public_source_support_sets")
            self.assertEqual(validate_corpus(copy_ws, candidate)["status"], "failed", mutation)

    def test_plain_supportedness_receipt_does_not_claim_missing_source_coverage(self):
        ws = world(); docs = [{"doc_id": "s0_sig_0", "content": "甲项目签发记录记甲项目负责人经办0。"}]
        attach_receipts(docs, review_documents(Review(), ws, 0, docs), 0)
        self.assertEqual(validate_corpus(ws, corpus(docs))["status"], "failed")
        # A completed checkpoint cannot skip straight to a new source-certified
        # corpus. It fails locally and explicitly requires a whole-group review.
        trace = RendererTrace(ws)
        with self.assertRaisesRegex(RuntimeError, "complete new group review"):
            self.render(ws, trace, existing=corpus(docs), done={0})
        self.assertEqual(trace.calls, [])

    def test_sanitizer_preserves_source_only_member_without_inventing_fact_refs(self):
        ws = world(); docs = reviewed_group(ws)
        docs[1]["content"] = "本组记录出自签发件。"
        report = review_documents(Review(), ws, 0, docs, context=source_context(ws),
                                  requirements=fidelity_requirements(ws, 0))
        attach_receipts(docs, report, 0)
        candidate = corpus(docs)
        stats = _sanitize_corpus(candidate, ws)
        self.assertEqual(stats["dropped_unref"], 0)
        self.assertEqual(candidate["sessions"][0]["docs"][1]["fact_refs"], [])
        self.assertEqual(validate_corpus(ws, candidate)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
