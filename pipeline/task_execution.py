"""Durable subcalls for one bound logical task, independent of its business policy."""
from copy import deepcopy
import hashlib
import json

VERSION = "task-call-journal/v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class UnfinishedCall(RuntimeError):
    """A dispatched call without a saved reply consumes its existing allowance."""
    kind = "reply_not_persisted"


class RecordedCallFailure(RuntimeError):
    def __init__(self, raw):
        self.raw = deepcopy(raw)
        super().__init__(raw["__error__"])


class JournaledTracer:
    def __init__(self, tracer, state, save):
        self._tracer, self._state, self._save = tracer, state, save
        self.pfile = getattr(tracer, "pfile", None)
        self._position = 0
        self._calls = None
        if state and state.get("version") != VERSION:
            raise ValueError("Unsupported task call journal")
        state.setdefault("version", VERSION)
        state.setdefault("bindings", {})

    def chat_json(self, step, messages, **parameters):
        return self._call("chat_json", step, messages, parameters)

    def chat_text(self, step, messages, **parameters):
        return self._call("chat_text", step, messages, parameters)

    def _call(self, method, step, messages, parameters):
        key = _hash({"method": method, "step": step, "messages": messages, "parameters": parameters})
        index = self._position
        self._position += 1
        # The first complete request identifies this logical review/read.
        # Changed input owns a new branch; the previous calls are never relabeled.
        if index == 0:
            self._calls = self._state["bindings"].setdefault(key, [])
        calls = self._calls
        if index < len(calls):
            entry = calls[index]
            if entry["input_hash"] != key:
                raise ValueError("Task subcall input changed; original reply cannot certify it")
            if entry["status"] == "admitted":
                raise UnfinishedCall(f"Saved {step} subcall has no persisted reply; no duplicate dispatch")
            if entry["output_hash"] != _hash(entry["output"]):
                raise ValueError("Task subcall reply changed")
            if entry["status"] == "raised":
                raise RecordedCallFailure(entry["output"])
            return deepcopy(entry["output"])
        entry = {"step": step, "input_hash": key, "status": "admitted"}
        calls.append(entry)
        self._save()
        try:
            output = getattr(self._tracer, method)(step, messages, **parameters)
        except Exception as error:
            from llm_trace import failure_record
            output = failure_record(error)
            entry.update(status="raised", output=output, output_hash=_hash(output))
            self._save()
            raise
        entry.update(status="returned", output=deepcopy(output), output_hash=_hash(output))
        self._save()
        return output
