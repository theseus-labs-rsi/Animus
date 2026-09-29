"""
eval.memory_interface — 本地 bge embedding + EmbedMemory(baseline dense 检索)。

EmbedMemory:机制等同 simpleMem — 本地向量 + numpy 余弦,无外部 DB。
  ★ 关键:retrieve 是【选择性 top-k】,这正是信号竞争(M3)赖以触发的检索瓶颈:
    旧值在多 period 文档高频出现 → top-k 更可能命中旧值文档 → R1 合成出旧值(M3)。

统一接口定义在 eval.memory_systems.base.MemorySystem。
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from eval.public_context import header
from eval.embed_cache import cached_embed


# ── 统一本地 embedder(bge-small-zh-v1.5, 512维)─────────────────────────────
# 全系统(baseline/mem0/zep/amem)统一走本地 embed:① 同底座 = 公平(消除 embedder 混淆);
# ② 本地零延迟 = 躲开 DMXAPI embedding 的延迟/抖动(此前 mem0/zep 内部 embedder 因此卡死)。
# DMXAPI 只留给文本 LLM(答题/抽取/笔记分析)。纯编码(不加 bge 查询指令)→ 与各外部
# 系统内部用法一致(它们没法替自己加指令),跨系统口径统一。
EMBED_MODEL = "bge-small-zh-v1.5"
_LOCAL_HF_ID = "BAAI/bge-small-zh-v1.5"
_local_embedder = None
import threading as _threading
_embedder_lock = _threading.Lock()
# 见 embed_texts 的注释：torch CPU encode 并发会 segfault，必须串行。
_encode_lock = _threading.Lock()


def _get_embedder():
    global _local_embedder
    if _local_embedder is None:
        with _embedder_lock:
            if _local_embedder is None:
                from sentence_transformers import SentenceTransformer
                _local_embedder = SentenceTransformer(_LOCAL_HF_ID)
    return _local_embedder


def embed_texts(texts: list, model: str = EMBED_MODEL) -> list:
    """本地 bge-small-zh-v1.5 编码(512维, L2归一化),返回 list[np.ndarray(float32)]。

    encode 必须串行：本机 torch 2.14 + macOS arm64 下，多线程并发调用
    `SentenceTransformer.encode` 会 SIGSEGV（实测 221 docs / workers=3 必崩）。
    embedding 是纯 CPU 计算，串行化只牺牲吞吐、不影响结果，且让并发的收益留给
    网络型阶段（LLM 调用）。
    """
    if not texts:
        return []
    with _encode_lock:
        emb = _get_embedder()
        arr = emb.encode(list(texts), normalize_embeddings=True,
                         show_progress_bar=False, batch_size=64)
    return [np.asarray(v, dtype=np.float32) for v in arr]


def _chunk(text: str, max_chars: int = 220) -> list:
    """按段落/句子粗切片。细切片能放大信号竞争(高频旧值切出更多 chunk)。"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    # 先按句号/换行切,再贪心合并到 max_chars
    import re
    parts = re.split(r"(?<=[。！？\n])", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text]


class EmbedMemory:
    """本地 dense 检索记忆(机制 = simpleMem,被 SimpleMem adapter 内部使用)。"""

    def __init__(self, model: str = EMBED_MODEL, chunk: bool = True,
                 chunk_chars: int = 220):
        self.model = model
        self.chunk = chunk
        self.chunk_chars = chunk_chars
        self.reset()

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "model": self.model,
                "encoder_hf_id": _LOCAL_HF_ID, "normalize_embeddings": True,
                "chunk": self.chunk, "chunk_chars": self.chunk_chars}

    def reset(self) -> None:
        self._docs: list = []     # list[str] 片段文本
        self._meta: list = []     # list[dict]
        self._vecs: list = []     # list[np.ndarray]

    def ingest(self, text: str, doc_id: str = "", metadata: Optional[dict] = None,
               text_prefix: str = "") -> None:
        """ingest 文档。text_prefix 会拼到【每个 chunk】前(用于注入日期/周期表头,
        让检索片段自带时间戳 → 公平地给 baseline 做 recency 推理的机会)。"""
        pieces = _chunk(text, self.chunk_chars) if self.chunk else [text]
        if not pieces:
            return
        pieces = [text_prefix + p for p in pieces]
        vecs = embed_texts(pieces, self.model)
        if len(vecs) != len(pieces):
            # Do not silently zip a partial batch into a successful index.
            from eval.memory_systems.base import MemoryExecutionError
            raise MemoryExecutionError("ingest", "embedding_count_mismatch",
                                       {"expected": len(pieces), "received": len(vecs)})
        for piece, v in zip(pieces, vecs):
            self._docs.append(piece)
            self._meta.append({**(metadata or {}), "doc_id": doc_id})
            self._vecs.append(v)

    def _topk(self, query: str, top_k: int):
        """返回 (idx_array, sims_array)；库空时返回 (None, None)。"""
        if not self._vecs:
            return None, None
        from eval.embed_cache import cached_embed
        qv = cached_embed([query], self.model)[0]
        M = np.stack(self._vecs)
        sims = (M @ qv) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(qv) + 1e-8) + 1e-8)
        idx = np.argsort(-sims)[:top_k]
        return idx, sims

    def retrieve(self, query: str, top_k: int = 5) -> list:
        idx, _ = self._topk(query, top_k)
        if idx is None:
            return []
        return [self._docs[i] for i in idx]

    def retrieve_with_meta(self, query: str, top_k: int = 5) -> list:
        """诊断用:返回 [(text, meta, score)]。"""
        idx, sims = self._topk(query, top_k)
        if idx is None:
            return []
        return [(self._docs[i], self._meta[i], float(sims[i])) for i in idx]


def build_embed_memory(docs: list, workers: int = 3,
                       on_progress=None) -> EmbedMemory:
    """并行 embed 所有 doc(带周期/日期表头),再按序装进 EmbedMemory。

    对 embedding 端点抖动有韧性:低并发首轮 + 失败篇【串行补漏】+ 补到底仍失败才抛
    (残缺索引污染检索,不静默跳过)。检索 order-independent,装入顺序不影响正确性。
    on_progress(done, total): 可选回调,每完成一篇 embed 调一次。

    原先定义在 eval.multi_system(legacy driver)里;移到本模块是为了让
    eval.memory_systems.* 不再为它 import 整个 legacy 评测编排。multi_system
    仍然 re-export 同名函数,外部调用方不受影响。
    """
    mem = EmbedMemory(chunk=True)
    mem.reset()
    _cnt_lk = threading.Lock()
    _cnt = [0]
    total = len(docs)

    def _prep(item):
        sid, date, content = item
        return [header(sid, date) + p
                for p in (_chunk(content, mem.chunk_chars) if mem.chunk else [content]) if p]

    def _try(item):
        pieces = _prep(item)
        if not pieces:
            with _cnt_lk:
                _cnt[0] += 1
                if on_progress:
                    on_progress(_cnt[0], total)
            return [item, [], []]            # 空文档:无 piece
        try:
            r = [item, pieces, cached_embed(pieces, mem.model)]   # 命中走盘缓存,未命中嵌+落盘
            if len(r[2]) != len(pieces):
                raise ValueError("embedding response count differs from requested chunks")
        except Exception:
            return [item, pieces, None]      # 失败标记,稍后串行补
        with _cnt_lk:
            _cnt[0] += 1
            if on_progress:
                on_progress(_cnt[0], total)
        return r

    results = config.pmap(_try, docs, workers=workers)

    # 串行补漏(并发失败的,降速逐篇重试;最多 2 轮)
    for rnd in range(2):
        failed = [r for r in results if r[2] is None]
        if not failed:
            break
        print(f"[embed] 第{rnd+1}轮串行补漏 {len(failed)} 篇 ...")
        for r in failed:
            try:
                r[2] = cached_embed(r[1], mem.model)
                if len(r[2]) != len(r[1]):
                    r[2] = None
                    raise ValueError("embedding response count differs from requested chunks")
            except Exception:
                pass
            else:
                with _cnt_lk:
                    _cnt[0] += 1
                    if on_progress:
                        on_progress(_cnt[0], total)
    still = [r[0][0] for r in results if r[2] is None]
    if still:
        raise RuntimeError(
            f"embed ingest:{len(still)} 篇补漏后仍失败(embedding 端点不稳),建议稍后重试。失败周期={still[:10]}")

    for item, pieces, vecs in results:
        sid = item[0]
        for piece, v in zip(pieces, vecs):
            mem._docs.append(piece)
            mem._meta.append({"period_idx": sid, "doc_id": f"s{sid}"})
            mem._vecs.append(v)
    return mem

