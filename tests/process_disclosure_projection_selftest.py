"""Offline original process author input/binding tests; no semantic claims."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from pipeline import process_proposals as helper
from process_proposals_selftest import fixture, Fake


def plan():
    return {"version": "frozen-plan-fixture", "catalogue": [
        {"ref": "f1", "kind": "fact", "entity": "Analysis", "field": "state", "value": "pending"}],
        "records": [{"id": "d1", "session": 2, "refs": ["f1"],
                     "channel": "补交记录", "acquisition_context": "第三期才公开第二期状态。"}],
        "undisclosed": ["e1"], "raw_output": {"reason": "保留原安排理由和原输出", "records": []},
        "plan_hash": "frozen-plan-id", "future_semantic_field": {"must_keep": True},
        "author_input": {"messages": ["INTERNAL_AUTHOR_HISTORY"]},
        "source_inputs": {"private_receipt": "INTERNAL_SOURCE_SNAPSHOT"},
        "binding": {"source_hash": "INTERNAL_PLAN_BINDING"}, "call": {"id": "INTERNAL_CALL_ID"}}


class Tests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)
        self.wp, self.ws, self.proposal = fixture()
        self.ws.disclosure = plan()

    def propose(self, **kwargs):
        fake = Fake({"proposals": [self.proposal], "reason": "离线接线意见"})
        result = helper.propose_process_orders(self.wp, self.ws, target=2,
            chat_json=fake, model="offline-mini", **kwargs)
        return result, fake

    def test_actual_request_preserves_all_semantics_without_mutating_world(self):
        before = deepcopy(self.ws.to_dict())
        report, fake = self.propose()
        self.assertEqual(report["status"], "completed")
        payload = json.loads(fake.calls[0][1][-1]["content"])
        expected = deepcopy(before)
        for key in ("author_input", "source_inputs", "binding", "call"):
            expected["disclosure"].pop(key)
        self.assertEqual(payload["world"], expected)
        self.assertEqual(self.ws.to_dict(), before)
        self.assertEqual(report["binding"]["world_hash"], helper._hash(before))
        self.assertEqual(report["binding"]["messages_hash"], helper._hash(fake.calls[0][1]))
        self.assertEqual(helper.validate_process_report(report, self.wp, self.ws), report["orders"])
        self.assertEqual(fake.calls[0][2]["retries"], 1)
        self.assertEqual(report["calls_used"], 1)
        self.assertEqual(len(fake.calls), 1)

    def test_large_audit_envelope_no_longer_blocks_same_bounded_author_call(self):
        self.ws.disclosure["author_input"]["messages"] = ["audit" * 60000]
        self.assertGreater(len(json.dumps(self.ws.to_dict())), 200000)
        report, fake = self.propose()
        self.assertEqual(report["status"], "completed")
        self.assertLess(sum(len(m["content"]) for m in fake.calls[0][1]), 200000)
        self.assertEqual(report["binding"]["world_hash"], helper._hash(self.ws.to_dict()))

    def test_removed_audit_metadata_still_invalidates_full_world_and_order_binding(self):
        report, _ = self.propose()
        old_messages = deepcopy(report["messages"])
        self.ws.disclosure["call"]["id"] = "changed-history"
        prepared = helper.prepare_process_proposals(self.wp, self.ws, target=2, model="offline-mini")
        self.assertEqual(prepared["messages"], old_messages)
        self.assertNotEqual(prepared["binding"]["world_hash"], report["binding"]["world_hash"])
        with self.assertRaises(ValueError): helper.validate_process_report(report, self.wp, self.ws)
        with self.assertRaises(ValueError): helper.validate_process_order(report["orders"][0], self.ws)

    def test_every_plan_semantic_component_stays_in_request_and_binding(self):
        baseline = helper.prepare_process_proposals(self.wp, self.ws, target=2, model="offline-mini")
        for key in ("records", "catalogue", "undisclosed", "raw_output", "future_semantic_field"):
            with self.subTest(key=key):
                saved = deepcopy(self.ws.disclosure[key])
                self.ws.disclosure[key] = {"changed": key}
                changed = helper.prepare_process_proposals(self.wp, self.ws, target=2, model="offline-mini")
                self.assertNotEqual(changed["messages"], baseline["messages"])
                self.assertNotEqual(changed["binding"]["world_hash"], baseline["binding"]["world_hash"])
                self.ws.disclosure[key] = saved

    def test_mutated_record_in_preserved_model_input_cannot_replay(self):
        report, _ = self.propose()
        payload = json.loads(report["messages"][-1]["content"])
        payload["world"]["disclosure"]["records"][0]["acquisition_context"] = "changed"
        report["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False)
        report["binding"]["messages_hash"] = helper._hash(report["messages"])
        with self.assertRaises(ValueError): helper.validate_process_report(report, self.wp, self.ws)

    def test_actual_business_payload_still_obeys_original_input_cap(self):
        self.ws.disclosure["records"][0]["acquisition_context"] = "business-context" * 20000
        report, fake = self.propose()
        self.assertEqual(report["status"], "error")
        self.assertIn("exceeds budget", report["error"])
        self.assertEqual((report["calls_used"], len(fake.calls)), (0, 0))

    def test_old_proposal_version_does_not_upgrade_silently(self):
        report, _ = self.propose()
        report["version"] = "original-process-proposal/v1"
        with self.assertRaises(ValueError): helper.validate_process_report(report, self.wp, self.ws)

    def test_no_disclosure_world_keeps_complete_legacy_author_payload(self):
        self.ws.disclosure = {}
        report, fake = self.propose()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(json.loads(fake.calls[0][1][-1]["content"])["world"], self.ws.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
