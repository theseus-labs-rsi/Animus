"""Offline integration of the real corpus stage with mocked rendering boundaries.

These checks exercise real files, checkpoint identities and receipt validation.
The fixed reviewer only supplies fixtures; no claim is made about model accuracy.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OPENAI_API_KEY"] = "offline-incremental-quality"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
os.environ["MODEL"] = "offline-model"

from pipeline import factory
from pipeline import run as run_module
from pipeline.corpus_contract import attach_receipts, review_documents, validate_corpus, fidelity_requirements
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_positive_review
from pipeline.world_state import Op, SET, Timeline, WorldState, _date_of


class PassReviewer:
    def chat_json(self, step, *args, **kwargs):
        assert step == "corpus.review"
        return fixed_positive_review(args[0])


def world(expanded=False):
    entities = {"旧部门": {"状态": Timeline([Op(0, _date_of(0), SET, "已开业", None)])}}
    if expanded:
        entities["新部门"] = {"状态": Timeline([Op(0, _date_of(0), SET, "已开业", None)])}
    return WorldState(entities=entities, n_sessions=2)


def reviewed_session(ws, sid, *, doc_id=None):
    docs = [{"doc_id": doc_id or f"signal-{sid}", "title": "当期登记",
             "content": "旧部门状态为已开业。"}]
    report = review_documents(PassReviewer(), ws, sid, docs, requirements=fidelity_requirements(ws, sid))
    attach_receipts(docs, report, sid)
    return {"session_id": sid, "date": _date_of(sid), "docs": docs}


class IncrementalQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(run_module, "RUNS_DIR", Path(self.temporary.name)).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.object(factory, "diversity_report", return_value={}).start()
        self.run = run_module.Run("office", "office__incremental-quality",
                                  config_meta={"target_tokens": 0, "render_only": ["新部门"]})
        self.run.log = lambda *args, **kwargs: None
        self.run.write(factory.ART["whitepaper"], {"domain_profile": {}})
        self.run.write(factory.ART["world"], world().to_dict())
        self.run.write(factory.ART["questions"], [{"qid": "frozen-question", "gt": {"value": "已开业"}}])
        self.old = {"corpus": {"sessions": [reviewed_session(world(), sid) for sid in range(2)]}, "done_weeks": [0, 1]}
        self.run.write(factory.ART["corpus"], self.old)

    def frozen_inputs(self):
        return {name: (self.run.dir / factory.ART[name]).read_bytes()
                for name in ("whitepaper", "world", "questions")}

    def assert_inputs_unchanged(self, original):
        self.assertEqual(self.frozen_inputs(), original)

    def test_changed_context_refresh_failure_preserves_old_final(self):
        self.run.write(factory.ART["world"], world(True).to_dict())
        inputs = self.frozen_inputs()
        old_bytes = (self.run.dir / factory.ART["corpus"]).read_bytes()
        def interrupted(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertEqual(corpus, {"sessions": []})
            self.assertEqual(done, set())
            self.assertIsNone(scope["only_entities"])
            self.assertIsNone(scope["only_entity_sessions"])
            corpus["sessions"].append(reviewed_session(ws, 0))
            done.add(0)
            save()
            raise RuntimeError("fixture interruption")
        with patch.object(factory, "render_corpus", interrupted), self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            factory.stage_corpus(self.run)
        self.assertEqual((self.run.dir / factory.ART["corpus"]).read_bytes(), old_bytes)
        self.assertTrue(self.run.has(factory.CORPUS_CKPT))
        self.assertTrue(self.run.has(factory.CORPUS_WARNING))
        self.assertTrue(self.run.has("05_corpus_candidate.json"))
        self.assertEqual(self.run.manifest["algo"]["render_strategy"]["effective"], "full")
        self.assertIn("missing_or_stale_document_review", self.run.manifest["algo"]["render_strategy"]["refresh_reasons"])
        self.assert_inputs_unchanged(inputs)

    def test_full_refresh_resumes_its_matching_checkpoint(self):
        self.run.write(factory.ART["world"], world(True).to_dict())
        inputs = self.frozen_inputs()
        def interrupted(wp, ws, target, tracer, corpus, done, save, log, **scope):
            corpus["sessions"].append(reviewed_session(ws, 0, doc_id="resumed-0"))
            done.add(0)
            save()
            raise RuntimeError("fixture interruption")
        with patch.object(factory, "render_corpus", interrupted), self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            factory.stage_corpus(self.run)
        def resume(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertIsNone(scope["only_entities"])
            self.assertEqual(done, {0})
            self.assertEqual(corpus["sessions"][0]["docs"][0]["doc_id"], "resumed-0")
            corpus["sessions"].append(reviewed_session(ws, 1, doc_id="resumed-1"))
            done.add(1)
            save()
        with patch.object(factory, "render_corpus", resume):
            factory.stage_corpus(self.run)
        final = self.run.read(factory.ART["corpus"])
        self.assertEqual(validate_corpus(world(True), final)["status"], "passed")
        self.assertEqual(final["done_weeks"], [0, 1])
        self.assertFalse(self.run.has(factory.CORPUS_CKPT))
        self.assertFalse(self.run.has(factory.CORPUS_WARNING))
        self.assertFalse(self.run.has("05_corpus_candidate.json"))
        self.assert_inputs_unchanged(inputs)

    def test_v6_review_limit_retains_partial_corpus_for_question_stage(self):
        from pipeline.render import MaterialRejected
        self.run.manifest["config"].pop("render_only")
        self.run.manifest["config"]["production_control"] = {"version": "supply-driven/v6"}
        self.run.write(factory.ART["whitepaper"], {
            "domain_profile": {}, "quality_contract": {"corpus_review": True}})
        (self.run.dir / factory.ART["corpus"]).unlink()
        def review_exhausted(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertTrue(scope["allow_exploratory_review"])
            corpus["sessions"].append(reviewed_session(ws, 0))
            done.add(0)
            save()
            raise MaterialRejected(1, {"group": "fixture"}, [{"draft": 1}], ["unsupported"])
        with patch.object(factory, "render_corpus", review_exhausted), patch.object(
                factory, "_require_v6_supply_gate"):
            factory.stage_corpus(self.run)
        self.assertEqual(self.run.read(factory.ART["corpus"])["done_weeks"], [0])
        self.assertEqual(self.run.read(factory.CORPUS_WARNING)["continuation"], "review_limit")
        self.assertFalse(self.run.read(factory.CORPUS_WARNING)["release_eligible"])
        self.assertTrue(factory._corpus_is_current(self.run))
        self.run.write("05_corpus_candidate.json", {"corpus": {"sessions": []}, "done_weeks": []})
        self.assertFalse(factory._corpus_is_current(self.run))

    def test_full_refresh_resumes_exact_bound_partial_candidate_without_checkpoint(self):
        self.run.manifest["config"].pop("render_only")
        (self.run.dir / factory.ART["corpus"]).unlink()
        candidate = {"corpus": {"sessions": [reviewed_session(world(), 0)]},
                     "done_weeks": [0]}
        self.run.write("05_corpus_candidate.json", candidate)
        whitepaper = self.run.read(factory.ART["whitepaper"])
        frozen_world = self.run.read(factory.ART["world"])
        self.run.write(factory.CORPUS_WARNING, {
            "version": "corpus-generation-warning/v1",
            "status": "warning",
            "release_eligible": False,
            "binding": {
                "whitepaper_hash": factory._canonical_hash(whitepaper),
                "world_hash": factory._canonical_hash(frozen_world),
                "corpus_hash": factory._canonical_hash(candidate),
                "candidate_hash": factory._canonical_hash(candidate),
            },
        })

        def resume(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertEqual(done, {0})
            self.assertEqual([s["session_id"] for s in corpus["sessions"]], [0])
            corpus["sessions"].append(reviewed_session(ws, 1))
            done.add(1)
            save()

        with patch.object(factory, "render_corpus", resume):
            factory.stage_corpus(self.run)
        final = self.run.read(factory.ART["corpus"])
        self.assertEqual(final["done_weeks"], [0, 1])
        self.assertEqual([s["session_id"] for s in final["corpus"]["sessions"]], [0, 1])
        self.assertFalse(self.run.has(factory.CORPUS_WARNING))
        self.assertFalse(self.run.has("05_corpus_candidate.json"))

    def test_same_world_delta_keeps_existing_documents(self):
        inputs = self.frozen_inputs()
        def append(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertEqual(scope["only_entities"], {"新部门"})
            self.assertEqual(corpus, self.old["corpus"])
            self.assertEqual(done, {0, 1})
            corpus["sessions"][0]["docs"].extend(reviewed_session(ws, 0, doc_id="additional-witness")["docs"])
            save()
        with patch.object(factory, "render_corpus", append):
            factory.stage_corpus(self.run)
        final = self.run.read(factory.ART["corpus"])
        for index, session in enumerate(self.old["corpus"]["sessions"]):
            self.assertEqual(final["corpus"]["sessions"][index]["docs"][:len(session["docs"])], session["docs"])
        self.assertEqual(self.run.manifest["algo"]["render_strategy"]["effective"], "delta")
        self.assertEqual(validate_corpus(world(), final)["status"], "passed")
        self.assert_inputs_unchanged(inputs)

    def test_same_world_delta_ignores_unrelated_checkpoint_uses_final(self):
        self.run.write(factory.CORPUS_CKPT, {"identity": "unrelated-checkpoint", "corpus": {"sessions": []}, "done_weeks": []})
        def inspect(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertEqual(corpus, self.old["corpus"])
            self.assertEqual(done, {0, 1})
            self.assertEqual(scope["only_entities"], {"新部门"})
        with patch.object(factory, "render_corpus", inspect):
            factory.stage_corpus(self.run)
        self.assertEqual(self.run.read(factory.ART["corpus"]), self.old)
        self.assertFalse(self.run.has(factory.CORPUS_CKPT))

    def test_corrupt_optional_checkpoint_falls_back_to_published_corpus(self):
        (self.run.dir / factory.CORPUS_CKPT).write_text('{"identity":', encoding="utf-8")
        def inspect(wp, ws, target, tracer, corpus, done, save, log, **scope):
            self.assertEqual(corpus, self.old["corpus"])
            self.assertEqual(done, {0, 1})
            self.assertEqual(scope["only_entities"], {"新部门"})
        with patch.object(factory, "render_corpus", inspect):
            factory.stage_corpus(self.run)
        self.assertEqual(self.run.read(factory.ART["corpus"]), self.old)
        self.assertFalse(self.run.has(factory.CORPUS_CKPT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
