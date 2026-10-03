"""Reader wire recovery preserves opinions, exact pages and historical receipts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import paged_read, paged_read_v2, paged_read_v3


class Tests(unittest.TestCase):
    def messages(self):
        docs = [{"doc_id": "d1", "content": "a" * 110000}, {"doc_id": "d2", "content": "counterevidence"}]
        return [{"role": "system", "content": "independent reader"},
                {"role": "user", "content": json.dumps({"documents": docs})}], docs

    def opinion(self):
        return {"answer": "uncertain", "answerability": "unresolved", "coverage": {"status": "partial"},
                "evidence": [], "reasoning": "counterevidence", "major_requirements": ["task"]}

    def test_flat_opinion_after_complete_delivery_preserved_and_replayed(self):
        messages, docs = self.messages(); original = self.opinion()
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            return {"action": "continue", "notes": "uncertain"} if body["unread_ids"] else deepcopy(original)
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader)
        self.assertEqual({k: v for k, v in output.items() if k != "_paged_read"}, original)
        self.assertEqual(output["_paged_read"]["transcript"][-1]["raw_output"], original)
        paged_read.validate(output, docs)
        output["coverage"]["status"] = "complete"
        with self.assertRaises(ValueError): paged_read.validate(output, docs)

    def test_flat_opinion_before_all_pages_keeps_bounded_failure(self):
        messages, _ = self.messages()
        with self.assertRaises(paged_read.PagedReadProtocolError):
            paged_read.call("semantic_review.blind_read", messages, chat_json=lambda *a, **kw: self.opinion())

    def test_reference_auditor_flat_envelope_preserves_unresolved_decision(self):
        messages, docs = self.messages()
        result = {"decision": "unresolved", "reason": "incomplete interpretation", "task_requirements": [],
                  "claims": [], "task_coverage": "unknown", "limitations": ["uncertain"], "evidence": []}
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            return {"action": "continue", "notes": "unknown"} if body["unread_ids"] else result
        output = paged_read.call("agent_editing.independent_reference_audit", messages, chat_json=reader)
        self.assertEqual(output["decision"], "unresolved")
        paged_read.validate(output, docs)

    def test_doc_ids_alias_still_checks_membership_and_original_reply(self):
        messages, docs = self.messages(); seen = []; inspected = []
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"]); seen.append(body)
            if body["unread_ids"]: return {"action": "continue", "notes": "read"}
            if not inspected:
                inspected.append(True)
                return {"action": "inspect", "doc_ids": ["d2"]}
            return {"action": "finish", "opinion": self.opinion()}
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader)
        paged_read.validate(output, docs)
        self.assertTrue(any(r["raw_output"].get("doc_ids") == ["d2"] for r in output["_paged_read"]["transcript"]))

    def test_invalid_alias_does_not_deliver_unknown_documents(self):
        messages, _ = self.messages()
        with self.assertRaisesRegex(paged_read.PagedReadProtocolError, "Invalid IDs"):
            paged_read.call("semantic_review.blind_read", messages,
                chat_json=lambda *a, **kw: {"action": "inspect", "doc_ids": ["invented"]})

    def test_complete_delivery_hint_breaks_continue_loop_without_changing_opinion(self):
        messages, docs = self.messages()
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            work = body["working_state"]
            if body["unread_ids"]:
                return {"action": "continue", "notes": "unresolved counterevidence"}
            self.assertIn("action=finish", work["required_next_action"])
            return {"action": "finish", "opinion": self.opinion()}
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader)
        self.assertEqual(output["coverage"], {"status": "partial"})
        paged_read.validate(output, docs)

    def test_inspect_after_delivery_really_resends_requested_original(self):
        messages, docs = self.messages()
        reread = False
        def reader(step, request, **kw):
            nonlocal reread
            body = json.loads(request[-1]["content"])
            if body["unread_ids"]:
                return {"action": "continue", "notes": "read"}
            if not reread:
                reread = True
                return {"action": "inspect", "doc_ids": ["d2"]}
            self.assertEqual(body["exact_reads"]["d2"]["value"], docs[1])
            return {"action": "finish", "opinion": self.opinion()}
        output = paged_read.call("semantic_review.blind_read", messages, chat_json=reader)
        paged_read.validate(output, docs)

    def test_v3_receipt_replays_with_its_original_transport(self):
        messages, docs = self.messages()
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            return ({"action": "continue", "notes": "read"} if body["unread_ids"] else
                    {"action": "finish", "opinion": self.opinion()})
        output = paged_read_v3.call("semantic_review.blind_read", messages, chat_json=reader)
        before = deepcopy(output)
        paged_read.validate(output, docs)
        self.assertEqual(output, before)

    def test_legacy_receipt_stays_exact_and_unchanged(self):
        messages, docs = self.messages()
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            return {"action": "continue", "notes": "read"} if body["unread_ids"] else {"action": "finish", "opinion": self.opinion()}
        output = paged_read_v2.call("semantic_review.blind_read", messages, chat_json=reader)
        before = deepcopy(output); paged_read.validate(output, docs)
        self.assertEqual(before, output)

    def test_legacy_role_cache_reused_without_provider_or_opinion_change(self):
        self.check_legacy_role_cache(paged_read_v2)

    def test_v3_role_cache_reused_without_provider_or_opinion_change(self):
        self.check_legacy_role_cache(paged_read_v3)

    def check_legacy_role_cache(self, legacy):
        from pipeline import grounding_review
        from original_grounding_selftest import fixture, opinions
        from semantic_review_fixture_helpers import audit_output, attach_targets
        _, _, question, corpus, protocol = fixture()
        corpus["corpus"]["sessions"][0]["docs"][0]["content"] += "公开背景。" * 25000
        def reader(step, request, **kw):
            body = json.loads(request[-1]["content"])
            if body.get("unread_ids"): return {"action": "continue", "notes": "retain evidence"}
            context = body["requirements"]
            if step.endswith("blind_read"): result = opinions().responses[0]
            elif "original_reference" in context: result = audit_output(context["original_reference"])
            else: result = attach_targets(opinions().responses[1], context["reference_proposal"])
            return {"action": "finish", "opinion": result}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(paged_read, "__file__", legacy.__file__), patch.object(paged_read, "call", legacy.call):
                kept, report, old = grounding_review.review_grounding([question], corpus, protocol,
                    chat_json=reader, model="test", isolated_reference=True, checkpoint_dir=directory)
            def forbidden(*a, **kw): self.fail("Historical successful role must be reused")
            kept, report, current = grounding_review.review_grounding([question], corpus, protocol,
                chat_json=forbidden, model="test", isolated_reference=True, checkpoint_dir=directory)
            self.assertEqual(report["caller_invocations"], 0)
            self.assertEqual(len(kept), 1)
            for role in ("blind_read_raw_output", "adjudicate_raw_output"):
                self.assertEqual(old["items"][0][role], current["items"][0][role])
            self.assertEqual(old["items"][0]["reference_audit"]["raw_output"], current["items"][0]["reference_audit"]["raw_output"])


if __name__ == "__main__": unittest.main(verbosity=2)
