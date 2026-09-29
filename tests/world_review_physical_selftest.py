"""A saved world opinion must have new, settled physical reviewer calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.world_review_physical import audit_new_review, digest


def line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class NewWorldReviewPhysicalTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace_path = self.root / "llm_attempts.jsonl"
        self.prompt_path = self.root / "prompts.jsonl"
        self.old_trace = line({"event": "request", "call_id": "old", "step": "corpus.review"})
        self.old_prompt = line({"step": "corpus.review", "output": {"verdict": "fail"}})
        self.trace_path.write_bytes(self.old_trace)
        self.prompt_path.write_bytes(self.old_prompt)
        self.messages = [{"role": "user", "content": "inspect world and evidence"}]
        self.params = {"model": "dummy", "temperature": 0, "max_tokens": 16384,
                       "response_format": {"type": "json_object"}}
        self.output = {"decision": "repair", "reason": "offline scripted fixture"}
        self.review = {"strategy": "agentic", "transcript": [{
            "messages_hash": digest(self.messages),
            "call": {"step": "world.semantic_review", "params": self.params},
            "raw_output": self.output}]}
        self.request = {"event": "request", "step": "world.semantic_review",
                        "call_id": "new", "operation_id": "op-new", "messages": self.messages,
                        "model": "dummy", "parameters": self.params}
        self.response = {"event": "response", "step": "world.semantic_review",
                         "call_id": "new", "operation_id": "op-new",
                         "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}}
        self.result = {"event": "json_result", "step": "world.semantic_review",
                       "operation_id": "op-new", "parsed": self.output}
        self.prompt = {"step": "world.semantic_review", "messages": self.messages,
                       "output": self.output,
                       "params": {key: value for key, value in self.params.items()
                                  if key in ("model", "temperature", "max_tokens", "retries",
                                             "strict_json", "complete_containers")}}

    def write_tail(self, *rows):
        self.trace_path.write_bytes(self.old_trace + b"".join(map(line, rows)))
        self.prompt_path.write_bytes(self.old_prompt + line(self.prompt))

    def audit(self):
        return audit_new_review(self.review, self.trace_path, self.prompt_path,
            original_trace_bytes=len(self.old_trace),
            original_trace_sha256=hashlib.sha256(self.old_trace).hexdigest(),
            original_prompt_bytes=len(self.old_prompt),
            original_prompt_sha256=hashlib.sha256(self.old_prompt).hexdigest())

    def test_new_review_is_bound_to_usage_result_and_prompt(self):
        self.write_tail(self.request, self.response, self.result)
        proof = self.audit()
        self.assertEqual(proof["new_call_ids"], ["new"])
        self.assertEqual(proof["usage_prompt_tokens"], 10)
        self.assertFalse(proof["business_truth_certified"])

    def test_historical_replay_alone_is_not_a_new_opinion(self):
        with self.assertRaisesRegex(ValueError, "new reviewer message"):
            self.audit()

    def test_reused_historical_call_id_is_rejected(self):
        request = deepcopy(self.request); request["call_id"] = "old"
        self.write_tail(request, self.response, self.result)
        with self.assertRaisesRegex(ValueError, "historical"):
            self.audit()

    def test_changed_request_or_missing_usage_is_rejected(self):
        request = deepcopy(self.request); request["messages"] = [{"role": "user", "content": "changed"}]
        self.write_tail(request, self.response, self.result)
        with self.assertRaisesRegex(ValueError, "messages differ"):
            self.audit()
        response = deepcopy(self.response); response["response"] = {}
        self.write_tail(self.request, response, self.result)
        with self.assertRaisesRegex(ValueError, "settled usage"):
            self.audit()

    def test_changed_parsed_output_or_prompt_is_rejected(self):
        result = deepcopy(self.result); result["parsed"] = {"decision": "accept"}
        self.write_tail(self.request, self.response, result)
        with self.assertRaisesRegex(ValueError, "parsed JSON differs"):
            self.audit()
        self.write_tail(self.request, self.response, self.result)
        self.prompt_path.write_bytes(self.old_prompt + line({**self.prompt, "output": {"decision": "accept"}}))
        with self.assertRaisesRegex(ValueError, "prompt log differs"):
            self.audit()

    def test_non_review_request_before_opinion_is_rejected(self):
        extra = {"event": "request", "step": "render.signal", "call_id": "other"}
        self.write_tail(extra, self.request, self.response, self.result)
        with self.assertRaisesRegex(ValueError, "non-review provider request"):
            self.audit()

    def test_response_operation_or_prompt_parameters_cannot_change(self):
        response = deepcopy(self.response); response["operation_id"] = "different"
        self.write_tail(self.request, response, self.result)
        with self.assertRaisesRegex(ValueError, "physical operation"):
            self.audit()
        self.write_tail(self.request, self.response, self.result)
        changed = deepcopy(self.prompt); changed["params"]["max_tokens"] = 1
        self.prompt_path.write_bytes(self.old_prompt + line(changed))
        with self.assertRaisesRegex(ValueError, "prompt log differs"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
