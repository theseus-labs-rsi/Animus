"""Offline full-public-input cache invalidation, including out-of-scope additions.

Fixed reviewer opinions exercise execution and caching only. They do not prove
semantic quality or claim that scoped readers saw a document outside their scope.
"""
from copy import deepcopy
from pathlib import Path
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("MODEL", "offline-model")
os.environ.setdefault("OPENAI_API_KEY", "offline-selftest")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")

import config
from pipeline import factory
from pipeline.benchmark_export import public_view
from pipeline.grounding_review import review_grounding
from original_grounding_selftest import fixture
from semantic_review_selftest import blind, adjudication
from semantic_review_fixture_helpers import audit_output, attach_targets


STEPS = ["semantic_review.blind_read", "agent_editing.independent_reference_audit",
         "semantic_review.adjudicate"]


class InputRun:
    """Read-only run view for the production factory's public-input identity."""
    def __init__(self, wp, question, corpus):
        self.manifest = {"config": {}}
        self.values = {"01_whitepaper.json": wp, "04_questions.json": [question],
                       "05_corpus.json": corpus,
                       "00_about.json": {"answer_protocol": factory.ANSWER_PROTOCOL}}

    def read(self, name):
        return deepcopy(self.values[name])


class GenerationGroundingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="generation-grounding-cache-")
        self.addCleanup(self.temp.cleanup)
        for target, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                             (config, "chat"), (config, "chat_json")):
            guard = patch.object(target, name, side_effect=AssertionError("Network/API prohibited"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, _, self.question, self.corpus, self.protocol = fixture()
        self.wp["generation_contract"] = {"version": "generation-first/v2", "material_first": True}
        self.run = InputRun(self.wp, self.question, self.corpus)
        self.original_question = deepcopy(self.question)
        self.calls = []

    def opinion(self, step, messages, **kwargs):
        self.calls.append(step)
        reference = {"answer": "北溟保险", "rationale": ""}
        if step == STEPS[1]:
            return audit_output(reference)
        result = (blind() if step == STEPS[0] else
                  attach_targets(adjudication(reference_status="supported"), reference))
        result["evidence"] = []
        result["coverage"]["inspected_doc_ids"] = ["d000001"]
        return result

    def review(self, *, namespace=True):
        options = ({"cache_namespace": factory._grounding_source_binding(self.run)["public_hash"]}
                   if namespace else {})
        return review_grounding([self.question], self.corpus, self.protocol,
            chat_json=self.opinion, model="offline", isolated_reference=True,
            checkpoint_dir=Path(self.temp.name), **options)

    def test_out_of_scope_public_append_rechecks_all_roles_but_metadata_reuses(self):
        before_view = public_view(self.run)
        self.review()
        self.assertEqual(self.calls, STEPS)
        self.calls.clear()

        # This document deliberately lacks the question's literal entity and
        # lives outside its declared period. The old reference-audit message is
        # unchanged, so message-only cache identity incorrectly reused it.
        self.corpus["corpus"]["sessions"].append({"session_id": 1, "date": "2025-01-13",
            "docs": [{"doc_id": "extra", "content": "An operational notice revises all earlier records."}]})
        after_view = public_view(self.run)
        self.assertNotEqual(before_view, after_view)
        _, _, report = self.review()
        self.assertEqual(self.calls, STEPS)
        self.assertEqual(report["items"][0]["document_scope"]["document_count"], 1)
        self.assertEqual(report["input_manifest"]["document_count"], 2)
        self.calls.clear()

        self.corpus["internal_note"] = "Resume bookkeeping, absent from public inputs."
        self.corpus["corpus"]["sessions"][0]["docs"][0]["internal_note"] = "Reviewer queue note."
        self.assertEqual(public_view(self.run), after_view)
        self.review()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.question, self.original_question)

    def test_legacy_default_keeps_completed_message_cache_reusable(self):
        self.review(namespace=False)
        self.assertEqual(self.calls, STEPS)
        self.calls.clear()
        self.review(namespace=False)
        self.assertEqual(self.calls, [])

    def test_v2_namespace_does_not_claim_legacy_message_cache_is_current(self):
        self.review(namespace=False)
        self.calls.clear()
        self.review(namespace=True)
        self.assertEqual(self.calls, STEPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
