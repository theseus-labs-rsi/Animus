"""Bind a new independent world opinion to calls after a frozen trace prefix.

Semantic receipt replay does not prove a provider call happened. This read-only
check requires the historical trace and prompt log to be byte-identical prefixes
and matches each new reviewer message, call parameters, usage, parsed result,
and prompt-log output. It never creates an opinion or changes admission.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


VERSION = "post-corpus-world-review-physical/v1"
STEP = "world.semantic_review"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _prefix_and_tail(path, prefix_bytes, prefix_sha256):
    if type(prefix_bytes) is not int or prefix_bytes < 0 or not isinstance(prefix_sha256, str):
        raise ValueError("Frozen trace prefix identity is malformed")
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        position = 0
        while position < prefix_bytes:
            raw = stream.readline()
            position += len(raw)
            if not raw or position > prefix_bytes or not raw.endswith(b"\n"):
                raise ValueError("Frozen trace prefix ends inside a record")
            result.update(raw)
            yield True, json.loads(raw.decode("utf-8"))
        if result.hexdigest() != prefix_sha256:
            raise ValueError("Frozen trace prefix changed")
        for raw in stream:
            if raw.strip():
                yield False, json.loads(raw.decode("utf-8"))


def _wanted_calls(review):
    if review.get("strategy") == "agentic":
        rows = review.get("transcript")
        if not isinstance(rows, list) or not rows:
            raise ValueError("New reviewer transcript is missing")
    else:
        rows = [{"messages_hash": digest(review.get("messages")),
                 "call": review.get("call"), "raw_output": review.get("raw_output")}]
    wanted = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("messages_hash"), str):
            raise ValueError("New reviewer transcript is malformed")
        message_hash = row["messages_hash"]
        call = row.get("call") or {}
        if (message_hash in wanted or call.get("step") != STEP
                or not isinstance(call.get("params"), dict)
                or row.get("raw_output") is None):
            raise ValueError("New reviewer call identity is incomplete or repeated")
        wanted[message_hash] = row
    return wanted


def audit_new_review(review, trace_path, prompt_path, *,
                     original_trace_bytes, original_trace_sha256,
                     original_prompt_bytes, original_prompt_sha256):
    wanted = _wanted_calls(review)
    original_ids = set()
    requests = {}
    by_message = defaultdict(list)
    responses = {}
    parsed = {}
    for historical, row in _prefix_and_tail(trace_path, original_trace_bytes, original_trace_sha256):
        event = row.get("event")
        if historical:
            if event == "request":
                original_ids.add(row.get("call_id"))
            continue
        if event == "request":
            if row.get("step") != STEP:
                raise ValueError("A non-review provider request preceded the new world opinion")
            message_hash = digest(row.get("messages"))
            if message_hash not in wanted:
                raise ValueError("New reviewer messages differ from the saved opinion")
            params = wanted[message_hash]["call"]["params"]
            actual = row.get("parameters") or {}
            if (row.get("model") != params.get("model")
                    or any(actual.get(key) != params.get(key)
                           for key in ("temperature", "max_tokens", "response_format"))):
                raise ValueError("New physical reviewer parameters differ from the saved call")
            call_id = row.get("call_id")
            operation_id = row.get("operation_id")
            if (not isinstance(call_id, str) or not call_id or call_id in original_ids
                    or call_id in requests or not isinstance(operation_id, str) or not operation_id):
                raise ValueError("New reviewer request ID is missing, repeated, or historical")
            requests[call_id] = (message_hash, operation_id)
            by_message[message_hash].append(call_id)
        elif event == "response":
            call_id = row.get("call_id")
            if call_id not in requests or call_id in responses:
                raise ValueError("New reviewer response has no unique new request")
            if row.get("step") != STEP or row.get("operation_id") != requests[call_id][1]:
                raise ValueError("New reviewer response changed its physical operation")
            usage = (row.get("response") or {}).get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if (type(prompt) is not int or type(completion) is not int
                    or prompt < 0 or completion < 0):
                raise ValueError("New reviewer request lacks settled usage")
            responses[call_id] = (prompt, completion)
        elif event == "json_result":
            operation_id = row.get("operation_id")
            if (row.get("step") != STEP
                    or operation_id not in {item[1] for item in requests.values()}
                    or operation_id in parsed):
                raise ValueError("Repeated parsed reviewer result")
            parsed[operation_id] = row.get("parsed")
    if (not requests or set(requests) != set(responses)
            or set(by_message) != set(wanted)):
        raise ValueError("Every new reviewer message needs settled physical evidence")
    for message_hash, call_ids in by_message.items():
        operations = {requests[call_id][1] for call_id in call_ids}
        expected = wanted[message_hash]["raw_output"]
        if not any(parsed.get(operation_id) == expected for operation_id in operations):
            raise ValueError("New parsed JSON differs from the saved reviewer output")
    prompt_rows = {}
    for historical, row in _prefix_and_tail(prompt_path, original_prompt_bytes, original_prompt_sha256):
        if historical:
            continue
        if row.get("step") != STEP:
            raise ValueError("A non-review prompt was logged before the new opinion")
        message_hash = digest(row.get("messages"))
        expected_params = wanted.get(message_hash, {}).get("call", {}).get("params", {})
        logged_params = {key: value for key, value in expected_params.items()
                         if key in ("model", "temperature", "max_tokens", "retries",
                                    "strict_json", "complete_containers")}
        if (message_hash not in wanted or message_hash in prompt_rows
                or row.get("output") != wanted[message_hash]["raw_output"]
                or row.get("params") != logged_params):
            raise ValueError("New reviewer prompt log differs from the saved output")
        prompt_rows[message_hash] = row
    if set(prompt_rows) != set(wanted):
        raise ValueError("New reviewer prompt log is incomplete")
    return {"version": VERSION, "review_sha256": digest(review),
            "original_trace_sha256": original_trace_sha256,
            "original_prompt_sha256": original_prompt_sha256,
            "new_trace_sha256": sha_file(trace_path),
            "new_prompt_sha256": sha_file(prompt_path),
            "new_call_ids": sorted(requests), "new_physical_requests": len(requests),
            "new_logical_review_messages": len(wanted),
            "usage_prompt_tokens": sum(item[0] for item in responses.values()),
            "usage_completion_tokens": sum(item[1] for item in responses.values()),
            "historical_call_id_reused": False, "provider_calls_by_this_audit": 0,
            "business_truth_certified": False, "paid_resume_authorization": False}
