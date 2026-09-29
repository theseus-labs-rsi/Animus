"""Complete local model-call records; no API client or configuration side effects."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import json
import threading
import time
import uuid

_scope = ContextVar("llm_trace_scope", default=None)
_write_lock = threading.Lock()


def redact(value, secrets=()):
    """Remove configured credentials, without changing ordinary benchmark facts."""
    if isinstance(value, dict):
        return {key: ("[redacted]" if str(key).lower() in {
            "api_key", "authorization", "access_token", "api_token", "password"
        } else redact(item, secrets)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if isinstance(secret, str) and secret:
                value = value.replace(secret, "[redacted]")
    return value


def failure_record(exc, *, secrets=()):
    """Preserve a small, allowlisted error contract for agents and old callers.

    No request body, provider response, credential, or traceback is copied.
    The existing __error__ sentinel remains available to every original stage.
    """
    metadata = {"version": 1, "error_type": type(exc).__name__}
    for field in ("kind", "retryable", "attempts", "token_cap", "input_chars", "phase", "call_id",
                  "deadline_s", "local_cleanup_confirmed", "remote_cancellation", "usage_status",
                  "response_received"):
        value = getattr(exc, field, None)
        if type(value) in (str, int, float, bool):
            metadata[field] = value
    return {"__error__": redact(str(exc), secrets)[:120], "__error_metadata__": redact(metadata, secrets)}


@contextmanager
def trace_scope(path, step, *, metadata=None):
    """Bind one logical operation; child operations retain an explicit parent ID."""
    parent = _scope.get()
    scope = {"path": str(Path(path)), "step": step, "operation_id": uuid.uuid4().hex,
             "parent_operation_id": parent.get("operation_id") if parent else None,
             "metadata": metadata or {}}
    token = _scope.set(scope)
    try:
        yield scope["operation_id"]
    finally:
        _scope.reset(token)


def emit(event, *, secrets=(), **fields):
    scope = _scope.get()
    if scope is None:
        return
    record = {"trace_version": 1, "ts": time.time(),
              **{k: v for k, v in scope.items() if k != "path"},
              "event": event, **fields}
    text = json.dumps(redact(record, secrets), ensure_ascii=False, allow_nan=False)
    path = Path(scope["path"])
    # A trace write failure is observable; do not silently certify an unrecorded review.
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
