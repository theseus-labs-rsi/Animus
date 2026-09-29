"""Shared call admission and bounded parallel execution; no provider imports."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from copy import deepcopy
import threading


class ExecutionStopped(RuntimeError):
    global_stop = True

    def __init__(self, message, *, kind="execution_stopped", raw=None):
        self.kind = kind
        self.raw = deepcopy(raw)
        super().__init__(message)


def global_failure(error):
    """Classify structured failures and their causes, never business verdict text."""
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            if "__error__" not in value:
                continue
            metadata = value.get("__error_metadata__")
            metadata = metadata if isinstance(metadata, dict) else {}
            if metadata.get("global_stop") or metadata.get("kind") in {
                    "http_status_401", "http_status_402", "http_status_403",
                    "model_limit", "call_limit", "estimated_budget_limit", "source_drift"}:
                return True
            message = value.get("__error__", "")
        else:
            if getattr(value, "global_stop", False):
                return True
            if getattr(value, "kind", None) in {"http_status_401", "http_status_402", "http_status_403"}:
                return True
            status = getattr(value, "status_code", None)
            if status in (401, 402, 403):
                return True
            pending.extend((getattr(value, "raw", None), getattr(value, "__cause__", None),
                            getattr(value, "__context__", None)))
            message = str(value)
        # Compatibility for frozen traces made before structured stop metadata.
        if str(message).startswith("Original experiment model/call/estimated budget limit; no provider dispatch"):
            return True
    return False


def global_stop_exception(error):
    """Return the original global failure or a structured stop for a tracer sentinel."""
    if not global_failure(error):
        return None
    if isinstance(error, BaseException):
        return error
    if not isinstance(error, dict):
        return ExecutionStopped(str(error))
    metadata = error.get("__error_metadata__")
    kind = metadata.get("kind") if isinstance(metadata, dict) else None
    return ExecutionStopped(str(error["__error__"]), kind=kind or "execution_stopped", raw=error)


class StopState:
    def __init__(self):
        self._lock = threading.Lock()
        self._failure = None

    def check(self):
        with self._lock:
            error = self._failure
        if error is not None:
            raise error

    def record(self, error):
        with self._lock:
            if self._failure is None:
                self._failure = error
            return self._failure


_scope = ContextVar("execution_stop_state", default=None)


def check_stop():
    state = _scope.get()
    if state is not None:
        state.check()


def observe_failure(error):
    state = _scope.get()
    if state is not None and global_failure(error):
        failure = error if isinstance(error, BaseException) else ExecutionStopped(
            str(error.get("__error__", "Provider execution stopped")), raw=error)
        state.record(failure)


class CallGate(StopState):
    """One admission boundary for nested authors/readers; network calls stay parallel."""
    def __init__(self, tracer, *, stop_global=False, failure_factory=None):
        super().__init__()
        self._tracer = tracer
        self.pfile = getattr(tracer, "pfile", None)
        self._stop_global = stop_global
        self._failure_factory = failure_factory or (lambda raw: ExecutionStopped(
            str(raw.get("__error__", "Provider execution stopped")), raw=raw))

    def fail(self, error):
        observe_failure(error)
        raise self.record(error)

    def chat_json(self, *args, **kwargs):
        return self._call(self._tracer.chat_json, *args, **kwargs)

    def chat_text(self, *args, **kwargs):
        return self._call(self._tracer.chat_text, *args, **kwargs)

    def _call(self, method, *args, **kwargs):
        self.check()
        check_stop()
        try:
            result = method(*args, **kwargs)
        except Exception as error:
            if self._stop_global and global_failure(error):
                from llm_trace import failure_record
                stopped = self._failure_factory(failure_record(error))
                stopped.__cause__ = error
                self.fail(stopped)
            raise
        if self._stop_global and global_failure(result):
            self.fail(self._failure_factory(result))
        return result


def bounded_map(fn, items, workers=6, *, on_result=None):
    """Ordered results, bounded admission, prompt failure observation, drained siblings.

    Only running work is admitted. On failure no new items start; already running
    work can persist its results. Provider-wide failures stop follow-up calls in
    nested maps as well. Content rejection stays local to the caller's policy.
    on_result commits each successful result in the collecting thread before
    admitting more work or propagating a sibling failure.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    iterator = iter(items)
    results, active, errors = [], {}, []
    token = _scope.set(StopState()) if _scope.get() is None else None
    try:
        check_stop()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def admit():
                check_stop()
                try:
                    item = next(iterator)
                except StopIteration:
                    return False
                index = len(results)
                results.append(None)
                active[pool.submit(copy_context().run, fn, item)] = index
                return True

            try:
                for _ in range(workers):
                    if not admit():
                        break
            except BaseException as error:
                errors.append(error)
                observe_failure(error)
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda future: active[future]):
                    index = active.pop(future)
                    try:
                        results[index] = future.result()
                        if on_result is not None:
                            on_result(results[index])
                    except BaseException as error:
                        errors.append(error)
                        observe_failure(error)
                if not errors:
                    try:
                        for _ in range(workers - len(active)):
                            if not admit():
                                break
                    except BaseException as error:
                        errors.append(error)
                        observe_failure(error)
            if errors:
                # The late global stop is more actionable than an earlier local
                # failure. Both remain linked; no admitted result is discarded.
                primary = next((e for e in errors if global_failure(e)
                                and not isinstance(e, ExecutionStopped)),
                               next((e for e in errors if global_failure(e)), errors[0]))
                previous = next((e for e in errors if not global_failure(e)), errors[0])
                if primary is not previous:
                    raise primary from previous
                raise primary
        check_stop()
        return results
    finally:
        if token is not None:
            _scope.reset(token)
