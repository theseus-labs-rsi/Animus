"""Actual disclosure/reviewer paths with offline replies; no quality claims."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from pipeline import disclosure, world_semantics as review
from world_scoped_agents_selftest import ScopedAgentFixture
from world_semantics_selftest import fixture


class ReplyFixture(ScopedAgentFixture):
    def __init__(self, path, *, repair=False, channel=None):
        super().__init__()
        self.pfile = path / "prompts.jsonl"
        self.repair, self.channel = repair, channel

    def chat_json(self, step, messages, **params):
        raw = super().chat_json(step, messages, **params)
        if step == disclosure.STEP and raw.get("action") == "arrange" and self.channel:
            for record in raw["records"]:
                record["channel"] = self.channel
        if step == review.STEP and self.repair:
            if raw.get("action") == "submit" and raw.get("disclosure_reviews"):
                raw["disclosure_reviews"][0].update(status="repair", reason="Offline disclosure repair request.")
            if raw.get("action") == "finish":
                raw["decision"] = "repair"
                raw["repair_targets"]["disclosure"] = True
        return raw


class NoCall:
    def __init__(self, path):
        self.pfile = path / "prompts.jsonl"
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Completed same-input checkpoint must not call the provider")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(STRUCTURE_MODEL="cheap-author", REVIEWER_MODEL="cheap-review")}).start()
        self.wp, self.ws, self.task = fixture()
        self.wp["world_generation"] = {"strategy": "agentic"}
        self.wp["quality_contract"]["public_disclosure"] = True
        self.ws.conflicts = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.assertEqual(disclosure.author_plan(self.wp, self.ws, ReplyFixture(self.path), self.task)["status"], "ready")

    def binding(self, **changes):
        args = {"wp": self.wp, "ws": self.ws, "task_input": self.task,
                "previous": None, "author_responses": None, "model": "cheap-review"}
        args.update(changes)
        return review._hash(review._review_checkpoint_inputs(**args))

    def test_actual_disclosure_repair_new_plan_then_review_and_exact_resume(self):
        first = review.review_world(self.wp, self.ws, ReplyFixture(self.path, repair=True), self.task)
        self.assertEqual(first["status"], "failed", first)
        self.assertTrue(first["repair_targets"]["disclosure"])
        first_path = next(self.path.glob("02_world_review_*.ckpt.json"))
        first_bytes = first_path.read_bytes()
        first_binding = json.loads(first_bytes)["input_binding"]
        old_plan = deepcopy(self.ws.disclosure)
        author = ReplyFixture(self.path, channel="Corrected offline original-record channel")
        revised = disclosure.author_plan(self.wp, self.ws, author, self.task, feedback=first)
        self.assertEqual(revised["status"], "ready", revised)
        self.assertEqual(disclosure.validate_plan(self.ws), [])
        self.assertNotEqual(self.ws.disclosure, old_plan)
        reader = ReplyFixture(self.path)
        second = review.review_world(self.wp, self.ws, reader, self.task, previous=first)
        self.assertEqual(second["status"], "passed", second)
        self.assertEqual(review.validate_review(second, self.wp, self.ws, self.task), [])
        self.assertGreater(len(reader.calls), 0)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        checkpoints = list(self.path.glob("02_world_review_*.ckpt.json"))
        self.assertEqual(len(checkpoints), 2)
        self.assertNotEqual(self.binding(previous=first), first_binding)
        resumed = NoCall(self.path)
        third = review.review_world(self.wp, self.ws, resumed, self.task, previous=first)
        self.assertEqual(third["status"], "passed", third)
        self.assertEqual(resumed.calls, 0)
        self.assertEqual(third["transcript"], second["transcript"])

    def test_cross_implementation_requires_full_input_binding(self):
        first = review.review_world(self.wp, self.ws, ReplyFixture(self.path), self.task)
        self.assertEqual(first["status"], "passed", first)
        exact_path = next(self.path.glob("02_world_review_*.ckpt.json"))
        original = json.loads(exact_path.read_text(encoding="utf-8"))
        exact_path.unlink()
        old_path = self.path / "02_world_review_old_implementation.ckpt.json"
        old = deepcopy(original); old["identity"] = "prior-implementation"
        old_path.write_text(json.dumps(old), encoding="utf-8")
        current = NoCall(self.path)
        # Modern transcripts carry their original nonlegacy replay semantics.
        old["legacy_resume_prefix"] = 0
        old_path.write_text(json.dumps(old), encoding="utf-8")
        same = review.review_world(self.wp, self.ws, current, self.task)
        self.assertEqual(same["status"], "passed", same)
        self.assertEqual(current.calls, 0)
        for path in self.path.glob("02_world_review_*.ckpt.json"):
            if path != old_path:
                path.unlink()
        old.pop("input_binding")
        old_path.write_text(json.dumps(old), encoding="utf-8")
        old_bytes = old_path.read_bytes()
        fresh = ReplyFixture(self.path)
        report = review.review_world(self.wp, self.ws, fresh, self.task)
        self.assertEqual(report["status"], "passed", report)
        self.assertGreater(len(fresh.calls), 0)
        self.assertEqual(old_path.read_bytes(), old_bytes)

    def test_all_semantic_inputs_have_distinct_binding(self):
        original = self.binding()
        changed_world = deepcopy(self.ws)
        changed_world.disclosure["records"][0]["channel"] += " changed"
        changed_wp = deepcopy(self.wp); changed_wp["rubric"] += " changed"
        changed_task = deepcopy(self.task); changed_task["private"] += " changed"
        for changes in ({"wp": changed_wp}, {"ws": changed_world}, {"task_input": changed_task},
                        {"previous": {"status": "failed"}}, {"author_responses": {"response": "changed"}},
                        {"model": "changed-review-model"}):
            with self.subTest(changes=list(changes)):
                self.assertNotEqual(self.binding(**changes), original)

    def test_downstream_evidence_has_new_agentic_identity_and_exact_resume(self):
        previous = review.review_world(self.wp, self.ws, ReplyFixture(self.path), self.task)
        self.assertEqual(previous["status"], "passed", previous)
        old_path = next(self.path.glob("02_world_review_*.ckpt.json"))
        old_bytes = old_path.read_bytes()
        body = {"version": "verified-downstream-evidence/v1", "offline_only": True,
            "new_provider_calls": 0, "new_world_opinion": False,
            "paid_resume_authorization": False,
            "world_hash": review._hash(self.ws.to_dict()),
            "whitepaper_hash": review._hash(self.wp),
            "original_task_hash": review._hash(self.task),
            "historical_passed_review_hash": review._hash(previous),
            "historical_world_review_status": "passed",
            "historical_negative_public_review": {"status": "failed"},
            "downstream_negative_reviews": [
                {"call_id": "offline-fixture-call", "negative_opinion": {"verdict": "fail"}}]}
        evidence = {**body, "envelope_hash": review._hash(body)}
        reader = ReplyFixture(self.path)
        report = review.review_world(self.wp, self.ws, reader, self.task,
                                     previous=previous, downstream_evidence=evidence)
        self.assertEqual(report["status"], "passed", report)  # scripted flow only
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task), [])
        self.assertGreater(len(reader.calls), 0)
        self.assertEqual(old_path.read_bytes(), old_bytes)
        self.assertEqual(len(list(self.path.glob("02_world_review_*.ckpt.json"))), 2)
        replay = NoCall(self.path)
        same = review.review_world(self.wp, self.ws, replay, self.task,
                                   previous=previous, downstream_evidence=evidence)
        self.assertEqual(same["status"], "passed", same)
        self.assertEqual(replay.calls, 0)
        self.assertEqual(same["transcript"], report["transcript"])

    def test_unbound_longer_checkpoint_cannot_override_bound_compatible_one(self):
        first = review.review_world(self.wp, self.ws, ReplyFixture(self.path), self.task)
        self.assertEqual(first["status"], "passed", first)
        exact_path = next(self.path.glob("02_world_review_*.ckpt.json"))
        old = json.loads(exact_path.read_text(encoding="utf-8")); exact_path.unlink()
        old["identity"] = "old-parser"; old["legacy_resume_prefix"] = 0
        (self.path / "02_world_review_compatible.ckpt.json").write_text(json.dumps(old), encoding="utf-8")
        unrelated = deepcopy(old)
        unrelated["input_binding"] = self.binding(previous={"status": "failed"})
        unrelated["transcript"].append(deepcopy(unrelated["transcript"][-1]))
        unrelated["hash"] = review._hash(unrelated["transcript"])
        (self.path / "02_world_review_longer_stale.ckpt.json").write_text(json.dumps(unrelated), encoding="utf-8")
        no_call = NoCall(self.path)
        report = review.review_world(self.wp, self.ws, no_call, self.task)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(no_call.calls, 0)

    def test_same_input_incompatible_parser_prefix_starts_current_review(self):
        first = review.review_world(self.wp, self.ws, ReplyFixture(self.path), self.task)
        self.assertEqual(first["status"], "passed", first)
        exact = next(self.path.glob("02_world_review_*.ckpt.json"))
        old = json.loads(exact.read_text(encoding="utf-8")); exact.unlink()
        old["identity"] = "prior-parser-with-different-first-message"
        old["transcript"][0]["messages_hash"] = "0" * 64
        old["hash"] = review._hash(old["transcript"])
        preserved = self.path / "02_world_review_incompatible_parser.ckpt.json"
        preserved.write_text(json.dumps(old), encoding="utf-8")
        original = preserved.read_bytes()
        tracer = ReplyFixture(self.path)
        report = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "passed", report)
        self.assertGreater(len(tracer.calls), 0)
        self.assertEqual(preserved.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
