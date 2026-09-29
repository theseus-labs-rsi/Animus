"""Observe SDK boundaries whose callers may suppress dependency exceptions."""
from __future__ import annotations

import json
import threading
from functools import wraps

from .base import MemoryExecutionError, execution_stage, require_response


class SDKGuard:
    """Keep failures even when an SDK catches them and returns an empty result.

    Calls on one adapter are serialized so SDK worker-thread failures cannot be
    assigned to another question. This changes concurrency, not retrieval logic.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.stage = "init"
        self.failures = {}

    def watch(self, obj, name, validate=None):
        original = getattr(obj, name, None)
        if not callable(original) or getattr(original, "_memory_guard", None) is self:
            return

        @wraps(original)
        def observed(*args, **kwargs):
            key = (id(obj), name, _call_value(args), _call_value(kwargs))
            try:
                with execution_stage(self.stage, dependency=name):
                    value = original(*args, **kwargs)
                    if validate is not None:
                        validate(value, args, kwargs, self.stage)
                    # A successful retry of the same operation is recovery.
                    self.failures.pop(key, None)
                    return value
            except MemoryExecutionError as exc:
                self.failures[key] = exc
                raise

        observed._memory_guard = self
        setattr(obj, name, observed)

    def run(self, stage, function, *args, **kwargs):
        with self._lock:
            self.stage, self.failures = stage, {}
            with execution_stage(stage):
                value = function(*args, **kwargs)
                if self.failures:
                    # A different fallback may have recovered a failed batch,
                    # but a normal SDK return alone does not prove completeness.
                    raise MemoryExecutionError(stage, "dependency_completion_unverified", {
                        "failures": [error.as_dict() for error in self.failures.values()]})
                return value


def _call_value(value):
    """Snapshot ordinary SDK arguments in memory; never log their contents."""
    if value is None or type(value) in (str, bytes, bool, int, float):
        return (type(value), value)
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_call_value(v) for v in value))
    if isinstance(value, dict):
        return (dict, frozenset((_call_value(k), _call_value(v)) for k, v in value.items()))
    # Do not guess equality for mutable SDK objects or numpy arrays.
    return object()


def validate_json_completion(value, args, kwargs, stage):
    """Validate the requested transport format, never the meaning of its text."""
    fmt = kwargs.get("response_format")
    if not fmt or fmt.get("type") not in ("json_object", "json_schema"):
        return
    require_response(isinstance(value, str) and bool(value.strip()), stage,
                     component="llm_json")
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise MemoryExecutionError(stage, "invalid_json", {"component": "llm"}) from exc
    require_response(isinstance(parsed, dict), stage, component="llm_json")
    schema = (fmt.get("json_schema") or {}).get("schema")
    if schema:
        _check_schema(parsed, schema, stage)


def _check_schema(value, schema, stage):
    # The pinned A-Mem requests only these JSON types; no semantic predicates.
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    if kind in types:
        require_response(isinstance(value, types[kind]) and
                         not (kind in ("integer", "number") and isinstance(value, bool)),
                         stage, component="llm_schema")
    if isinstance(value, dict):
        require_response(all(k in value for k in schema.get("required", [])),
                         stage, component="llm_schema")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _check_schema(value[key], sub, stage)
    if isinstance(value, list) and "items" in schema:
        for item in value:
            _check_schema(item, schema["items"], stage)


def validate_chroma_search(value, args, kwargs, stage):
    require_response(isinstance(value, dict) and isinstance(value.get("ids"), list)
                     and len(value["ids"]) == 1 and isinstance(value["ids"][0], list),
                     stage, component="chroma_search")
    ids = value["ids"][0]
    if ids:
        metadata = value.get("metadatas")
        require_response(isinstance(metadata, list) and len(metadata) == 1
                         and isinstance(metadata[0], list) and len(metadata[0]) == len(ids)
                         and all(isinstance(m, dict) for m in metadata[0]),
                         stage, component="chroma_metadata")
