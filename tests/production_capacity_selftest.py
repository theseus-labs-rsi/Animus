"""Capacity, interruption and current-receipt checks; all provider calls are fake."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from pipeline import process_batches, process_proposals, paged_read, render, grounding_review
from process_proposals_selftest import fixture as process_fixture
from original_grounding_selftest import fixture, opinions
import config


class Tests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_process_batches_resume_completed_proposal_then_finish(self):
        wp, ws, proposal = process_fixture()
        calls = []
        def first(step, messages, **kw):
            calls.append(messages)
            if len(calls) == 1:
                return {"action": "submit", "proposals": [proposal], "reason": "local"}
            raise TimeoutError("injected interruption")
        path = self.path / "process.json"
        report = process_proposals.propose_process_orders(wp, ws, target=9, model="fake", chat_json=first, checkpoint_path=path)
        self.assertEqual(report["status"], "error", report)
        self.assertEqual(len(report["orders"]), 1, report)
        second_calls = []
        def second(*args, **kw):
            second_calls.append(args)
            return {"action": "finish", "reason": "only one supported proposal"}
        resumed = process_proposals.propose_process_orders(wp, ws, target=9, model="fake", chat_json=second, checkpoint_path=path)
        self.assertEqual(resumed["status"], "completed", resumed)
        self.assertEqual(len(second_calls), 1)
        self.assertEqual(process_proposals.validate_process_report(resumed, wp, ws), resumed["orders"])
        bad = deepcopy(resumed); bad["orders"][0]["gt"] = {}
        with self.assertRaises(ValueError):
            process_proposals.validate_process_report(bad, wp, ws)

    def test_process_unread_witness_cannot_be_certified(self):
        wp, ws, proposal = process_fixture()
        window = process_batches.context(wp, ws)
        with self.assertRaises(ValueError):
            process_batches._known_witness(proposal, ws, window)

    def test_real_large_world_request_fits_window(self):
        root = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not root.exists(): self.skipTest("optional historical fixture")
        from pipeline.world_state import WorldState
        from pipeline import disclosure
        from disclosure_batches_selftest import DirectFixture
        wp = json.loads((root / "01_whitepaper.json").read_text(encoding="utf-8"))
        ws = WorldState.from_dict(json.loads((root / "02_world_candidate.json").read_text(encoding="utf-8")))
        task = json.loads((root / "00_input.json").read_text(encoding="utf-8"))
        saved = json.loads(next(root.glob("02_disclosure_checkpoint_*.json")).read_text(encoding="utf-8"))
        replies = [p["output"] for p in saved["parts"]] + [a["raw_output"] for a in saved["attempts"] if a["part"] == 3][:2]
        result = disclosure.author_plan(wp, ws, DirectFixture(lambda body, n: deepcopy(replies[n-1])), task)
        self.assertEqual(result["status"], "ready")
        full = process_proposals.prepare_process_proposals(wp, ws, target=137, model="fake")
        self.assertGreater(sum(len(m["content"]) for m in full["messages"]), 200000)
        window = process_batches.context(wp, ws); process_batches._next_window(window)
        sizes = []
        while window.read != set(window.nodes):
            if not window.visible:
                pending = next(k for k in window.nodes if k not in window.read)
                window.control({"action": "inspect", "ids": [pending], "part": len(window.parts_read.get(pending, set()))})
            messages = window.messages(process_proposals.SYSTEM + process_batches.RULES, {})
            sizes.append(sum(len(m["content"]) for m in messages))
            window.mark_sent(); process_batches._next_window(window)
        self.assertLess(max(sizes), 100000)
        print(json.dumps({"large_process_full_chars": sum(len(m["content"]) for m in full["messages"]),
                          "max_window_chars": max(sizes), "all_nodes_read": len(window.read)}))

    def test_paged_read_preserves_all_long_document_parts_and_negative_opinion(self):
        documents = [{"doc_id": "d1", "content": "公开原文" * 40000}, {"doc_id": "d2", "content": "末尾反证"}]
        messages = [{"role": "system", "content": "Original independent review role"},
                    {"role": "user", "content": json.dumps({"documents": documents, "question": "task"}, ensure_ascii=False)}]
        seen, calls = set(), []
        def reader(step, request, **params):
            calls.append(request)
            body = json.loads(request[-1]["content"])
            seen.update(body["exact_reads"])
            if body["unread_ids"]:
                return {"action": "continue", "notes": "Counterevidence must be retained."}
            return {"action": "finish", "opinion": {"verdict": "unresolved", "reason": "counterevidence"}}
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader, model="fake")
        self.assertEqual(output["verdict"], "unresolved")
        self.assertEqual(seen, {"d1", "d2"})
        self.assertLess(max(sum(len(m["content"]) for m in req) for req in calls), paged_read.LIMIT)
        paged_read.validate(output, documents)
        changed = deepcopy(documents); changed[1]["content"] = "changed"
        with self.assertRaises(ValueError): paged_read.validate(output, changed)
        forged = deepcopy(output); forged["verdict"] = "passed"
        with self.assertRaises(ValueError): paged_read.validate(forged, documents)

    def test_paged_reader_cannot_finish_before_all_pages(self):
        messages = [{"role": "system", "content": "review"}, {"role": "user", "content": json.dumps({
            "documents": [{"doc_id": "d1", "content": "x" * 250000}]})}]
        with self.assertRaisesRegex(ValueError, "invalid paged"):
            paged_read.call("semantic_review.blind_read", messages,
                chat_json=lambda *a, **kw: {"action": "finish", "opinion": {"verdict": "passed"}})

    def test_paged_reader_repeated_directory_actions_are_bounded(self):
        messages = [{"role": "system", "content": "review"}, {"role": "user", "content": json.dumps({
            "documents": [{"doc_id": "d1", "content": "x" * 150000}]})}]
        calls = [0]
        def loop(*args, **kwargs):
            calls[0] += 1
            return {"action": "index", "offset": 0}
        with self.assertRaisesRegex(ValueError, "no cumulative progress"):
            paged_read.call("semantic_review.blind_read", messages, chat_json=loop)
        self.assertGreater(calls[0], 4)

    def test_process_author_repeated_directory_actions_are_bounded(self):
        wp, ws, _ = process_fixture()
        calls = [0]
        def loop(*args, **kwargs):
            calls[0] += 1
            return {"action": "index", "offset": 0}
        report = process_batches.propose(wp, ws, target=9, chat_json=loop, model="fake")
        self.assertEqual(report["status"], "error")
        self.assertIn("no cumulative progress", report["error"])
        self.assertLessEqual(calls[0], 8)

    def test_wording_one_error_does_not_discard_other_item_and_resume_skips_success(self):
        wp, _, q, _, _ = fixture()
        other = deepcopy(q); other["qid"] += "_other"
        ok = {"verdict": "equivalent", "reason": "fixture", "issues": []}
        replies = [{"__error__": "temporary"}, {"question": q["question"]}, ok]
        class Tracer:
            def chat_json(self, *a, **kw): return replies.pop(0)
        checkpoint, audit = self.path / "wording.json", {}
        sequential = lambda fn, items, workers: [fn(item) for item in items]
        with patch.object(config, "pmap", side_effect=sequential):
            result = render.phrase_questions([q, other], wp, Tracer(), log=lambda *a: None,
                audit=audit, checkpoint_path=checkpoint)
            self.assertEqual(len(result), 1)
            self.assertEqual(audit["counts"], {"execution_error": 1, "passed": 1})
            replies.extend([{ "question": q["question"]}, ok])
            result = render.phrase_questions([q, other], wp, Tracer(), log=lambda *a: None,
                audit=audit, checkpoint_path=checkpoint)
        self.assertEqual(len(result), 2)
        self.assertEqual(replies, [])
        self.assertTrue(audit["items"][1]["resumed"])

    def test_grounding_completed_roles_resume_without_model_calls(self):
        _, _, q, corpus, protocol = fixture()
        cache = self.path / "review"
        kept, report, review = grounding_review.review_grounding([q], corpus, protocol,
            chat_json=opinions(), model="test", checkpoint_dir=cache)
        self.assertEqual(len(kept), 1, report)
        def forbidden(*a, **kw): raise AssertionError("Completed review must replay")
        kept, report, review = grounding_review.review_grounding([q], corpus, protocol,
            chat_json=forbidden, model="test", checkpoint_dir=cache)
        self.assertEqual(len(kept), 1, report)
        self.assertEqual(report["caller_invocations"], 0)
        grounding_review.validate_current_review(kept, corpus, protocol, review)

    def test_grounding_local_transport_failure_is_pending_and_other_question_completes(self):
        from grounding_candidate_isolation_selftest import Script
        _, _, q, corpus, protocol = fixture()
        questions = [q, {**deepcopy(q), "qid": q["qid"] + "_other"}]
        base = Script()
        def reader(step, messages, **kw):
            if step.endswith("blind_read") and base.index == -1:
                base.index += 1
                return {"__error__": "Malformed JSON after bounded retries", "__error_metadata__": {"kind": "json_syntax"}}
            return base(step, messages, **kw)
        kept, report, review = grounding_review.review_grounding(questions, corpus, protocol,
            chat_json=reader, model="test", isolated_reference=True, checkpoint_dir=self.path / "partial")
        self.assertEqual(len(kept), 1, report)
        self.assertEqual(report["n_pending"], 1)
        self.assertTrue(report["delivery_safe"], report)
        self.assertEqual(len(report["delivery_validation"]["isolated_execution_failures"]), 1)
        grounding_review.validate_current_review(kept, corpus, protocol, review)

    def test_grounding_invalid_paging_actions_are_local_to_one_question(self):
        from grounding_candidate_isolation_selftest import Script
        from pipeline import paged_read
        _, _, q, corpus, protocol = fixture()
        questions = [q, {**deepcopy(q), "qid": q["qid"] + "_other"}]
        script = Script()
        calls = [0]
        def bounded(step, messages, *, chat_json, **params):
            calls[0] += 1
            if calls[0] == 1:
                raise paged_read.PagedReadProtocolError(
                    "paged_invalid_action", "four invalid paging actions")
            return chat_json(step, messages, **params)
        with patch.object(paged_read, "call", side_effect=bounded):
            kept, report, review = grounding_review.review_grounding(
                questions, corpus, protocol, chat_json=script, model="test",
                isolated_reference=True, checkpoint_dir=self.path / "paged_local")
        self.assertEqual([questions[1]["qid"]], [row["qid"] for row in kept])
        self.assertEqual(report["n_pending"], 1)
        self.assertFalse(report["execution_stopped"])
        self.assertTrue(report["delivery_safe"], report)
        failure = report["delivery_validation"]["isolated_execution_failures"]
        self.assertEqual((failure[0]["qid"], failure[0]["stage"]),
                         (questions[0]["qid"], "blind_read"))
        grounding_review.validate_current_review(kept, corpus, protocol, review)

    def test_grounding_review_scope_keeps_declared_session_and_later_entity_mentions(self):
        from pipeline.semantic_review import prepare_review
        _, _, q, _, protocol = fixture()
        q = {**q, "entity": "测试报告", "evidence_sessions": [0]}
        corpus = {"sessions": [
            {"session_id": 0, "docs": [
                {"doc_id": "early", "content": "测试报告在本期登记。"},
                {"doc_id": "same_session_filler", "content": "本期无关材料", "is_filler": True}]},
            {"session_id": 1, "docs": [{"doc_id": "unrelated", "content": "完全无关的公开记录。"}]},
            {"session_id": 2, "docs": [{"doc_id": "retrospective", "content": "回顾测试报告的客户记录。"}]},
        ]}
        candidate = grounding_review.candidates_with_evidence([q], corpus)[0]
        self.assertEqual(candidate["semantic_scope_doc_ids"], ["early", "retrospective"])
        self.assertEqual(candidate["candidate_evidence_doc_ids"], ["early", "retrospective"])
        receipt = candidate["semantic_scope_receipt"]
        self.assertEqual(receipt["full_public_document_count"], 4)
        self.assertEqual(receipt["matched_document_count"], 2)
        prepared = prepare_review([candidate], corpus, protocol,
                                  reviewer_model="test", reader_model="test")
        scope = prepared["items"][0]["document_scope"]
        self.assertEqual(scope["source_doc_ids"], candidate["semantic_scope_doc_ids"])
        self.assertEqual(scope["document_count"], 2)
        self.assertEqual(len(prepared["documents"]), 4)
        self.assertEqual(scope["selection_receipt"], receipt)

    def test_paged_reader_ignores_directory_actions_while_unread_material_remains(self):
        documents = [{"doc_id": "d1", "content": "x" * 130000},
                     {"doc_id": "d2", "content": "末尾反证"}]
        messages = [{"role": "system", "content": "review"},
                    {"role": "user", "content": json.dumps({
                        "documents": documents, "question": "task"}, ensure_ascii=False)}]
        calls = [0]
        def reader(step, request, **params):
            calls[0] += 1
            body = json.loads(request[-1]["content"])
            if body["unread_ids"]:
                return {"action": "index", "offset": 0}
            return {"action": "finish", "opinion": {"verdict": "unresolved", "reason": "counterevidence"}}
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader, model="fake")
        self.assertEqual(output["verdict"], "unresolved")
        self.assertGreater(calls[0], 1)

    def test_long_corpus_paging_replays_all_three_original_roles_and_cache(self):
        from semantic_review_fixture_helpers import audit_output, attach_targets
        _, _, q, corpus, protocol = fixture()
        corpus["corpus"]["sessions"][0]["docs"][0]["content"] += "无关公开材料。" * 20000
        calls = []
        def reader(step, messages, **kw):
            calls.append(step)
            body = json.loads(messages[-1]["content"])
            if body.get("unread_ids"):
                return {"action": "continue", "notes": "保留首段的客户证据，并已读到当前页。"}
            context = body["requirements"]
            if step.endswith("blind_read"):
                opinion = opinions().responses[0]
            elif "original_reference" in context:
                opinion = audit_output(context["original_reference"])
            else:
                opinion = attach_targets(opinions().responses[1], context["reference_proposal"])
            return {"action": "finish", "opinion": opinion}
        cache = self.path / "paged_review"
        kept, report, review = grounding_review.review_grounding([q], corpus, protocol,
            chat_json=reader, model="test", isolated_reference=True, checkpoint_dir=cache)
        self.assertEqual(len(kept), 1, report)
        self.assertTrue(report["delivery_safe"], report)
        self.assertGreater(len(calls), 3)
        grounding_review.validate_current_review(kept, corpus, protocol, review)
        kept, report, review = grounding_review.review_grounding([q], corpus, protocol,
            chat_json=lambda *a, **kw: self.fail("Completed paged opinions must replay"),
            model="test", isolated_reference=True, checkpoint_dir=cache)
        self.assertEqual(len(kept), 1, report)
        self.assertEqual(report["caller_invocations"], 0)

    def test_corpus_completed_groups_replay_without_calls(self):
        from public_disclosure_render_selftest import world, planned, status_refs, Trace, WP
        ws = world(); planned(ws, [(2, status_refs(ws))])
        tracer = Trace(); tracer.pfile = self.path / "prompts.jsonl"
        sequential = lambda fn, items, workers=8: [fn(item) for item in items]
        with patch.object(config, "pmap", side_effect=sequential):
            first = {"sessions": []}
            render.render_corpus(WP, ws, 0, tracer, first, set(), lambda: None, lambda *a: None)
            self.assertTrue(list((self.path / "05_signal_checkpoints").glob("*.json")))
            second = {"sessions": []}
            with patch.object(tracer, "chat_json", side_effect=AssertionError("Completed group must be reused")):
                render.render_corpus(WP, ws, 0, tracer, second, set(), lambda: None, lambda *a: None)
        self.assertEqual(first, second)

    def test_world_review_reads_and_opinions_resume(self):
        from world_semantics_selftest import fixture as world_fixture
        from world_scoped_agents_selftest import ScopedAgentFixture
        from pipeline import world_semantics, disclosure
        wp, ws, task = world_fixture()
        wp["world_generation"] = {"strategy": "agentic"}; wp["quality_contract"]["public_disclosure"] = True
        ws.conflicts = []
        disclosure.author_plan(wp, ws, ScopedAgentFixture(), task)
        reader = ScopedAgentFixture(); reader.pfile = self.path / "prompts.jsonl"
        original = reader.chat_json
        count = [0]
        def interrupted(*a, **kw):
            count[0] += 1
            if count[0] == 3: return {"__error__": "injected interruption"}
            return original(*a, **kw)
        with patch.object(reader, "chat_json", side_effect=interrupted):
            failed = world_semantics.review_world(wp, ws, reader, task)
        self.assertEqual(failed["status"], "error")
        self.assertTrue(list(self.path.glob("02_world_review_*.ckpt.json")))
        before = len(reader.calls)
        result = world_semantics.review_world(wp, ws, reader, task)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(world_semantics.validate_review(result, wp, ws, task), [])
        self.assertEqual(len(result["transcript"]), len(reader.calls) - before + 2)

    def test_batch_prepares_world_reuse_without_dispatch(self):
        from tools import run_original_bc_batch as batch
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists(): self.skipTest("optional local world")
        args = batch.parser().parse_args(["--batch", "offline_capacity_resume", "--output-dir", str(self.path / "batch"),
            "--seeds", "insurance", "--source-run", str(source), "--min-questions", "200", "--total-only",
            "--model", "glm-5.3-flash", "--time-span-weeks", "24"])
        directory, plan = batch.prepare(args)
        command = plan["runs"][0]["command"]
        self.assertIn("--reuse-world-checkpoint", command)
        self.assertNotIn("--seed-pack", command)
        self.assertEqual(plan["reuse_source"]["directory"], str(source.resolve()))
        self.assertEqual(batch.validate(directory), plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
