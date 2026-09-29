"""
eval.memory_systems.base — 记忆系统统一接口。

所有被测系统(内置 baseline + 外部 mem0/zep/memOS)都实现此接口,
评测外壳通过它做 ingest → retrieve → 统一答题,隔离"记忆能力"。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
import hashlib


def configuration_fingerprint(value: str | None) -> str | None:
    """Bind an endpoint without placing URLs or embedded credentials in results."""
    return hashlib.sha256(value.encode()).hexdigest() if value else None


class MemoryExecutionError(RuntimeError):
    """An execution failure, never an answer or a successful empty search.

    Details contain counters/status codes, not provider response bodies or secrets.
    A failed write may have taken effect remotely; completed_docs is only a lower
    bound confirmed before the failing operation, not a rollback guarantee.
    """

    def __init__(self, stage: str, code: str, details: dict | None = None):
        self.stage = stage
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"memory {stage}: {code}")

    def as_dict(self) -> dict:
        return {"stage": self.stage, "code": self.code, "details": self.details}


@contextmanager
def execution_stage(stage: str, **details):
    """Preserve machine-readable failures without echoing arbitrary SDK errors."""
    try:
        yield
    except MemoryExecutionError as exc:
        exc.details = {**details, **exc.details}
        raise
    except Exception as exc:
        info = {**details, "cause_type": type(exc).__name__}
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            info["http_status"] = status
        raise MemoryExecutionError(stage, "operation_failed", info) from exc


def ingest_receipt(n_docs: int, *, completion: str = "completed", **details) -> dict:
    """Counts documents processed/accepted, not facts an LLM chose to retain."""
    return {"status": "ok", "n_docs": n_docs, "requested_docs": n_docs,
            "completion": completion, **details}


def require_response(condition: bool, stage: str, **details) -> None:
    if not condition:
        raise MemoryExecutionError(stage, "invalid_response", details)


class MemorySystem(ABC):

    @classmethod
    def preflight(cls, **kwargs) -> dict:
        """Side-effect-free dependency/service check for the control plane.

        External adapters override this method. Returned values must be safe to
        persist in ``run_plan.json``: versions and endpoint fingerprints are
        allowed; credentials, provider payloads, and random namespace IDs are not.
        """
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "memory_runtime": {"adapter": cls.__name__, "configuration": "repository"},
        }

    def evaluation_config(self) -> dict:
        """Stable instance settings for cache identity; never return secrets/IDs.

        Custom adapters must override this and declare their effective settings.
        An undeclared configuration is not a safe basis for reusable answers.
        """
        return {"configuration_status": "undeclared"}

    @abstractmethod
    def ingest_session(self, session: dict) -> dict:
        """逐 session 喂入语料。
        session = {"session_id": int, "date": "YYYY-MM-DD", "docs": [str, ...]}
        成功返回 ingest_receipt；失败抛 MemoryExecutionError。
        completion=accepted 仅表示已接受，不能替代 finalize 完成确认。"""

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 5) -> str:
        """对一个问题做原生检索,返回拼好的 context 字符串。
        query budget = 1:多跳系统在内部完成,对外仍是一次调用。"""

    def get_retrieved_context(self) -> str:
        """R-check:返回最近一次 retrieve 的 context。默认缓存实现。"""
        return getattr(self, "_last_context", "")

    @abstractmethod
    def get_memory_snapshot(self) -> dict:
        """W-check:返回全部记忆状态(审计用)。"""

    @abstractmethod
    def reset(self) -> None:
        """清空全部状态。"""

    def finalize_ingest(self, on_progress=None) -> dict | None:
        """所有 session ingest 完后调用一次。
        on_progress(done, total): 可选进度回调。默认 no-op。
        可返回限定验证范围的完成凭据；runner 应保留该 dict。"""

    def cleanup(self) -> dict:
        """Release run-scoped resources without creating a replacement namespace.

        ``reset()`` prepares an instance for reuse and some external adapters
        therefore allocate a fresh store. The runner needs a different end-of-run
        operation that only removes the current namespace. In-process baselines
        have no persistent resources, so the default is an explicit no-op receipt.
        """
        return {"status": "ok", "completion": "not_applicable"}

    def get_diagnostics(self) -> dict:
        """最近一次 retrieve 的诊断信息(如 bridge 实体)。默认空。"""
        return {}
