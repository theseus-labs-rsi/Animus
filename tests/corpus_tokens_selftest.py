"""Real tokenizer counts, stale-cache rejection and token-based batch planning.

Run with tiktoken==0.12.0 and preloaded cl100k_base encoding. No model calls.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import metadata
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.corpus_tokens import (
    CorpusTokenCounter, TokenizerDependencyError, plan_filler_batch,
)


def corpus(*documents):
    return {"sessions": [{"session_id": 0, "docs": list(documents)}]}


class CorpusTokensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = CorpusTokenCounter()

    def fixture(self):
        return corpus({"doc_id": "core", "content": "hello world"},
                      {"doc_id": "filler", "content": "中文证据。\n", "is_filler": True})

    def test_real_tokenizer_and_body_only(self):
        value = self.fixture()
        value["sessions"][0]["docs"][0]["title"] = "a very long excluded title" * 100
        value["protocol"] = "excluded protocol"
        measured = self.counter.measure(value)
        self.assertEqual(measured["tokenizer"]["encoding"], "cl100k_base")
        self.assertEqual(measured["tokenizer"]["version"], "0.12.0")
        self.assertEqual(measured["core"], {"documents": 1, "tokens": 2, "characters": 11})
        self.assertEqual(measured["filler"]["tokens"], self.counter.count_text("中文证据。\n"))
        self.assertEqual(measured["total"]["tokens"], 2 + measured["filler"]["tokens"])
        self.assertEqual(measured["total"]["characters"], 17)

    def test_document_boundaries_are_counted_separately(self):
        measured = self.counter.measure(corpus({"content": "a"}, {"content": "b"}))
        self.assertEqual(measured["total"]["tokens"], 2)
        self.assertEqual(self.counter.count_text("ab"), 1)

    def test_special_token_spelling_is_ordinary_text(self):
        text = "<|endoftext|>"
        self.assertGreater(self.counter.count_text(text), 1)
        self.assertEqual(self.counter.count_text(text),
                         len(self.counter._encoding.encode(text, disallowed_special=())))

    def test_cache_roundtrip_reuses_counts_without_encoding(self):
        first = self.counter.measure(self.fixture())
        with patch.object(self.counter, "count_text", side_effect=AssertionError("must reuse")):
            second = self.counter.measure(self.fixture(), cache=first["cache"])
        self.assertEqual(second["encoded_documents"], 0)
        self.assertEqual(second["cache_reused_documents"], 2)
        self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])
        self.assertEqual(first["cache"], second["cache"])

    def test_changed_body_recounts_even_same_character_length(self):
        first = self.counter.measure(corpus({"doc_id": "d", "content": "hello world"}))
        second = self.counter.measure(corpus({"doc_id": "d", "content": "!!!!!!!????"}), first["cache"])
        self.assertEqual(second["encoded_documents"], 1)
        self.assertNotEqual(first["corpus_sha256"], second["corpus_sha256"])
        self.assertEqual(second["total"]["tokens"], self.counter.count_text("!!!!!!!????"))

    def test_damaged_cache_recounts_and_preserves_input(self):
        first = self.counter.measure(self.fixture())
        cache = deepcopy(first["cache"])
        next(iter(cache["entries"].values()))["tokens"] = 999999
        original = deepcopy(cache)
        second = self.counter.measure(self.fixture(), cache)
        self.assertEqual(second["encoded_documents"], 1)
        self.assertEqual(second["total"], first["total"])
        self.assertEqual(cache, original)

    def test_tokenizer_changed_cache_recounts(self):
        first = self.counter.measure(self.fixture())
        cache = deepcopy(first["cache"])
        cache["tokenizer"]["version"] = "stale"
        second = self.counter.measure(self.fixture(), cache)
        self.assertEqual(second["encoded_documents"], 2)

    def test_category_change_reaggregates_cached_body(self):
        value = self.fixture()
        first = self.counter.measure(value)
        value["sessions"][0]["docs"][0]["is_filler"] = True
        second = self.counter.measure(value, first["cache"])
        self.assertEqual(second["encoded_documents"], 0)
        self.assertEqual(second["core"]["tokens"], 0)
        self.assertEqual(second["filler"]["tokens"], second["total"]["tokens"])
        self.assertNotEqual(first["corpus_sha256"], second["corpus_sha256"])

    def test_wrapper_empty_body_and_duplicate_bodies(self):
        measured = self.counter.measure({"corpus": corpus({"content": ""}, {"content": "x"}, {"content": "x"})})
        self.assertEqual(measured["total"], {"documents": 3, "tokens": 2, "characters": 2})
        self.assertEqual(measured["encoded_documents"], 2)
        self.assertEqual(measured["cache_reused_documents"], 1)

    def test_invalid_corpus_inputs_fail(self):
        for value in ([], {}, {"sessions": [{}]}, corpus({}), corpus({"content": 42}),
                      corpus({"content": "ok", "is_filler": "false"})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.counter.measure(value)

    def test_missing_dependency_has_no_fallback(self):
        with patch("pipeline.corpus_tokens.metadata.version",
                   side_effect=metadata.PackageNotFoundError("tiktoken")):
            with self.assertRaisesRegex(TokenizerDependencyError, "Install tiktoken==0.12.0"):
                CorpusTokenCounter()

    def test_version_mismatch_and_unknown_encoding_fail(self):
        with patch("pipeline.corpus_tokens.metadata.version", return_value="0.11.0"):
            with self.assertRaisesRegex(TokenizerDependencyError, "installed=0.11.0"):
                CorpusTokenCounter()
        with self.assertRaises(TokenizerDependencyError):
            CorpusTokenCounter("does_not_exist")

    def test_plan_exact_remainder_and_cap(self):
        measured = self.counter.measure(corpus({"content": "hello world"},
                                               {"content": "foo bar", "is_filler": True}))
        plan = plan_filler_batch(measured, 9, max_documents=2)
        self.assertEqual(plan["remaining_filler_tokens"], 5)
        self.assertEqual(plan["observed_filler_mean_tokens"], 2)
        self.assertEqual(plan["next_batch_documents"], 2)
        self.assertFalse(plan["target_met"])

    def test_plan_ratio_can_raise_total_and_stops_when_met(self):
        measured = self.counter.measure(corpus({"content": "hello world"},
                                               {"content": "foo bar", "is_filler": True}))
        plan = plan_filler_batch(measured, 4, haystack_ratio=9)
        self.assertEqual(plan["required_total_tokens"], 20)
        self.assertEqual(plan["remaining_filler_tokens"], 16)
        done = plan_filler_batch(measured, 4, haystack_ratio=1)
        self.assertTrue(done["target_met"])
        self.assertEqual(done["next_batch_documents"], 0)

    def test_empty_filler_uses_explicit_estimate(self):
        measured = self.counter.measure(corpus())
        plan = plan_filler_batch(measured, 2000, default_document_tokens=800)
        self.assertEqual(plan["next_batch_documents"], 3)
        self.assertIsNone(plan["observed_filler_mean_tokens"])
        self.assertEqual(plan["next_document_token_hint"], 667)

    def test_invalid_plans_rejected(self):
        measured = self.counter.measure(self.fixture())
        for kwargs in ({"target_tokens": True}, {"target_tokens": -1},
                       {"target_tokens": 1, "haystack_ratio": float("nan")},
                       {"target_tokens": 1, "max_documents": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_filler_batch(measured, **kwargs)
        bad = deepcopy(measured)
        bad["total"]["tokens"] += 1
        with self.assertRaises(ValueError):
            plan_filler_batch(bad, 1)


if __name__ == "__main__":
    unittest.main()
