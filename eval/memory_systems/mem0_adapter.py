"""
eval.memory_systems.mem0_adapter — mem0 记忆系统 adapter。

依赖:pip install mem0ai qdrant-client
服务:docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

Embedding 用本地 BAAI/bge-small-zh-v1.5 (512维,huggingface provider,不走 API)。
LLM 事实抽取走 INGEST_LLM_BASE_URL(若设了),否则走 OPENAI_BASE_URL (DMXAPI)。
"""
from __future__ import annotations
import importlib.metadata
import importlib.util
import os, sys, uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.memory_systems.base import (MemorySystem, execution_stage, ingest_receipt,
                                      require_response, configuration_fingerprint)
from eval.memory_systems.execution import SDKGuard, validate_json_completion
from eval.public_context import header


MEM0_ADAPTER_VERSION = "mem0-oss-v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSIONS = 512
REQUIRED_PACKAGE_VERSIONS = {
    "mem0ai": "2.0.7",
    "qdrant-client": "1.18.0",
    "sentence-transformers": "6.1.0",
}


class Mem0Adapter(MemorySystem):

    @classmethod
    def preflight(cls, **kwargs) -> dict:
        """Validate pinned dependencies, explicit model identity, and Qdrant."""
        errors: list[str] = []
        warnings: list[str] = []
        packages: dict[str, str] = {}
        for module, distribution in (
            ("mem0", "mem0ai"),
            ("qdrant_client", "qdrant-client"),
            ("sentence_transformers", "sentence-transformers"),
        ):
            if importlib.util.find_spec(module) is None:
                errors.append(f"缺少 Python 依赖 {distribution}")
                continue
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"无法确定 Python 依赖版本 {distribution}")
                continue
            expected = REQUIRED_PACKAGE_VERSIONS[distribution]
            if packages[distribution] != expected:
                errors.append(
                    f"{distribution} 版本必须是 {expected}，当前 {packages[distribution]}"
                )

        top_k = kwargs.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            errors.append("Mem0 top_k 必须是正整数")
        embedding_model = str(kwargs.get("embedding_model") or "").strip()
        if embedding_model != DEFAULT_EMBEDDING_MODEL:
            errors.append(
                f"Mem0 v1 embedding_model 必须固定为 {DEFAULT_EMBEDDING_MODEL}"
            )
        llm_model = str(kwargs.get("llm_model") or "").strip()
        llm_base_url = str(kwargs.get("llm_base_url") or "").strip()
        llm_api_key = str(kwargs.get("llm_api_key") or "").strip()
        if not llm_model:
            errors.append("Mem0 internal_model.model_id 不能为空")
        if not llm_base_url or not llm_api_key:
            errors.append("Mem0 internal_model.endpoint_profile 缺少 endpoint 或凭证")

        host = str(kwargs.get("qdrant_host") or "").strip()
        port = kwargs.get("qdrant_port")
        expected_service_version = str(
            kwargs.get("qdrant_expected_version") or ""
        ).strip()
        service_version = None
        if not host or isinstance(port, bool) or not isinstance(port, int) or port <= 0:
            errors.append("Mem0 Qdrant host/port 配置无效")
        elif "qdrant-client" in packages:
            try:
                response = httpx.get(f"http://{host}:{port}/", timeout=3.0)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("invalid Qdrant metadata")
                raw_version = payload.get("version")
                if isinstance(raw_version, str) and raw_version.strip():
                    service_version = raw_version.strip()
                    if (
                        expected_service_version
                        and service_version != expected_service_version
                    ):
                        errors.append(
                            "Qdrant 服务版本必须是 "
                            f"{expected_service_version}，当前 {service_version}"
                        )
                else:
                    warnings.append("Qdrant 可达，但服务版本未暴露")
            except Exception as exc:  # no provider body or endpoint is persisted
                errors.append(f"Qdrant 不可达或响应无效（{type(exc).__name__}）")

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "memory_runtime": {
                "adapter": "mem0",
                "adapter_version": MEM0_ADAPTER_VERSION,
                "package_versions": packages,
                "internal_model": {
                    "model_id": llm_model or None,
                    "endpoint_fingerprint": configuration_fingerprint(llm_base_url),
                },
                "embedding_model": embedding_model or None,
                "vector_store": {
                    "provider": "qdrant",
                    "endpoint_fingerprint": configuration_fingerprint(
                        f"{host}:{port}" if host and isinstance(port, int) else None
                    ),
                    "service_version": service_version,
                    "expected_service_version": expected_service_version or None,
                },
            },
        }

    def __init__(
        self,
        top_k: int = 20,
        *,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_prefix: str = "mem0_eval",
        **kwargs,
    ):
        self.top_k = top_k
        self._qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self._qdrant_port = int(qdrant_port or os.getenv("QDRANT_PORT", "6333"))
        self._llm_model = llm_model or os.getenv("INGEST_LLM_MODEL") or os.getenv("MODEL")
        self._llm_base_url = (
            llm_base_url
            or os.getenv("INGEST_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self._llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self._embedding_model = embedding_model
        self._collection_prefix = collection_prefix
        self._collection = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
        self._last_context = ""
        self._guard = SDKGuard()
        with execution_stage("init"):
            self._rebuild_memory()

    def _rebuild_memory(self):
        from mem0 import Memory

        cfg = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": self._qdrant_host, "port": self._qdrant_port,
                    "collection_name": self._collection,
                    "embedding_model_dims": EMBEDDING_DIMENSIONS,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": self._embedding_model},
            },
        }

        if self._llm_api_key and self._llm_base_url and self._llm_model:
            cfg["llm"] = {
                "provider": "openai",
                "config": {
                    "model": self._llm_model,
                    "api_key": self._llm_api_key,
                    "openai_base_url": self._llm_base_url,
                },
            }

        self.memory = Memory.from_config(cfg)
        active_llm = getattr(self.memory.llm, "config", None)
        actual_model = getattr(active_llm, "model", None)
        actual_base = getattr(active_llm, "openai_base_url", None)
        self._evaluation_llm = {
            "provider": cfg.get("llm", {}).get("provider", "sdk_default"),
            "model": (actual_model if isinstance(actual_model, str) else
                      cfg.get("llm", {}).get("config", {}).get("model", "not_exposed")),
            "endpoint_fingerprint": configuration_fingerprint(
                actual_base if isinstance(actual_base, str) else
                cfg.get("llm", {}).get("config", {}).get("openai_base_url")),
            "sdk_defaults": "not_exposed"}
        # 注入 socket 超时：mem0 的 OpenAIConfig 不支持 timeout 参数，
        # DMXAPI 死连接会让 generate_response 永久挂起。
        self._inject_llm_timeout()
        self._guard.watch(self.memory.llm, "generate_response", validate_json_completion)
        for attr, methods in {
            "embedding_model": ("embed", "embed_batch"),
            "vector_store": ("insert", "update", "delete", "search", "get", "list"),
            "db": ("add_history", "batch_add_history", "save_messages"),
        }.items():
            component = getattr(self.memory, attr, None)
            if component is not None:
                for method in methods:
                    self._guard.watch(component, method)
        # 关 thinking 不在 adapter 里各自做了：ingest LLM 统一走 embed_server 网关
        # (INGEST_LLM_BASE_URL → :9800),no-think / 剥 <think> / 超时重试 都在网关
        # 一处实现(services/embed_server.py)。mem0 指向网关即自动 no-think,见
        # services/run_eval.sh。

    def _inject_llm_timeout(self):
        """给 mem0 内部 OpenAI client 补上 socket 超时，防 DMXAPI 死连接钉死进程。"""
        llm = self.memory.llm
        client = getattr(llm, "client", None)
        if client is not None and hasattr(client, "_client"):
            client._client._timeout = httpx.Timeout(60.0, connect=10.0)

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "top_k": self.top_k,
                "llm": dict(self._evaluation_llm),
                "adapter_version": MEM0_ADAPTER_VERSION,
                "embedding_provider": "huggingface", "embedding_model": self._embedding_model,
                "embedding_dimensions": EMBEDDING_DIMENSIONS, "vector_provider": "qdrant",
                "vector_endpoint_fingerprint": configuration_fingerprint(
                    f"{self._qdrant_host}:{self._qdrant_port}"),
                "sdk_operations": "serialized"}

    @staticmethod
    def _results(value, stage):
        require_response(isinstance(value, dict) and isinstance(value.get("results"), list), stage)
        require_response(all(isinstance(row, dict) for row in value["results"]), stage)
        return value["results"]

    def ingest_session(self, session: dict) -> dict:
        sid, date = session["session_id"], session["date"]
        n = 0
        for doc in session["docs"]:
            with execution_stage("ingest", requested_docs=len(session["docs"]), completed_docs=n):
                result = self._guard.run("ingest", self.memory.add,
                                         header(sid, date) + doc, user_id=self._user_id)
                self._results(result, "ingest")
            n += 1
        return ingest_receipt(n)

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._last_context = ""
        with execution_stage("retrieve"):
            result = self._guard.run("retrieve", self.memory.search, query=question,
                filters={"user_id": self._user_id}, limit=top_k or self.top_k)
            memories = self._results(result, "retrieve")
            lines = []
            for m in memories:
                require_response(isinstance(m.get("memory"), str), "retrieve")
                lines.append(f"[score={m.get('score', 0):.2f}] {m['memory']}")
            self._last_context = "\n".join(lines) or "(无检索结果)"
            return self._last_context

    def get_memory_snapshot(self) -> dict:
        with execution_stage("snapshot"):
            result = self._guard.run("snapshot", self.memory.get_all,
                                     filters={"user_id": self._user_id})
            memories = self._results(result, "snapshot")
            text = "\n".join(m["memory"] for m in memories)
            return {"text": text or "(empty)", "n_memories": len(memories)}

    def _delete_collection(self, stage: str) -> dict:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=self._qdrant_host, port=self._qdrant_port)
        try:
            qc.delete_collection(self._collection)
        finally:
            close = getattr(qc, "close", None)
            if callable(close):
                close()
        return {"status": "ok", "completion": "deleted", "scope": "qdrant_collection"}

    def cleanup(self) -> dict:
        """Delete this run's collection without allocating a replacement."""
        with execution_stage("cleanup"):
            receipt = self._delete_collection("cleanup")
            self._last_context = ""
            return receipt

    def reset(self) -> None:
        with execution_stage("reset"):
            self._delete_collection("reset")
            self._collection = f"{self._collection_prefix}_{uuid.uuid4().hex[:8]}"
            self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
            self._last_context = ""
            self._rebuild_memory()
