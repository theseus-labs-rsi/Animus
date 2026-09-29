"""
eval.memory_systems.simplemem — 朴素 dense RAG(= 现有 EmbedMemory 包装)。

对应原系统 A(SingleShotRAG):top-k 向量检索 → 拼片段。
零外部基础设施的对照基线。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.memory_systems.base import MemorySystem, execution_stage, ingest_receipt
from eval.memory_interface import EmbedMemory, _chunk, build_embed_memory
from eval.embed_cache import cached_embed


class SimpleMem(MemorySystem):

    def __init__(self, top_k: int = 3, embed_mem: EmbedMemory = None):
        self.top_k = top_k
        self._mem = embed_mem
        self._docs_buf: list = []
        self._last_context = ""

    def evaluation_config(self) -> dict:
        memory_config = (self._mem.evaluation_config() if self._mem is not None
                         and callable(getattr(self._mem, "evaluation_config", None))
                         else {"configuration_status": "undeclared"})
        return {"configuration_status": memory_config.get("configuration_status", "undeclared"),
                "top_k": self.top_k, "embedding": memory_config}

    def ingest_session(self, session: dict) -> dict:
        sid = session["session_id"]
        date = session["date"]
        n = 0
        for doc in session["docs"]:
            self._docs_buf.append((int(sid), date, doc))
            n += 1
        return ingest_receipt(n, completion="accepted")

    def finalize_ingest(self, on_progress=None) -> None:
        if self._mem is not None:
            return
        with execution_stage("finalize"):
            self._mem = build_embed_memory(self._docs_buf, on_progress=on_progress)

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._last_context = ""
        k = top_k or self.top_k
        with execution_stage("retrieve"):
            snippets = self._mem.retrieve(question, top_k=k)
        if not snippets:
            ctx = "(无检索结果)"
        else:
            ctx = "\n\n".join(f"[片段{i+1}] {s}" for i, s in enumerate(snippets))
        self._last_context = ctx
        return ctx

    def get_memory_snapshot(self) -> dict:
        if not self._mem:
            return {"text": "", "n_chunks": 0}
        return {"text": "\n---\n".join(self._mem._docs), "n_chunks": len(self._mem._docs)}

    def reset(self) -> None:
        self._mem = None
        self._docs_buf = []
        self._last_context = ""

    def chunk_count(self) -> int:
        return len(self._mem._docs) if self._mem else 0
