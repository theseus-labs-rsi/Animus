"""
eval.embed_cache — 内容寻址的 embedding 磁盘缓存(断点续传)。

为什么要它:语料(289 篇)每轮重嵌,既慢又把命运绑死在 embedding 端点健康上——
DMXAPI 一抖,整轮 ingest 被拖垮(实测两次)。本模块把 embedding 按 (model, 文本) 的
sha1 永久存盘:嵌一次永久复用;端点抖动下可分多次把缓存喂满,之后秒载、且与端点健康解耦。

用法:透明包在 embed_texts 外。cached_embed(texts, model):
  - 命中缓存的直接返回;只对【未命中】的批量调 embed_texts。
  - 成功后【立即原子落盘】(tmp + rename)→ 即便随后崩溃/被杀,已嵌的不丢。
  - 未命中批 embed 失败 → 抛给调用方(已缓存的不受影响,下次续传)。
存储:output/eval/_embed_cache/<model>.npz —— hashes(str[N]) + vecs(float32[N,dim])。
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "output" / "eval" / "_embed_cache"

_lk = threading.Lock()
_mem = {}   # model -> {key(hex): np.ndarray}  进程内缓存,避免每次读盘


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def _path(model: str) -> Path:
    return CACHE_DIR / f"{model.replace('/', '_')}.npz"


def _load_locked(model: str) -> dict:
    """读盘载入(只首次);调用方须已持 _lk。"""
    if model in _mem:
        return _mem[model]
    d = {}
    p = _path(model)
    if p.exists():
        try:
            z = np.load(p, allow_pickle=False)
            H, V = z["hashes"], z["vecs"]
            d = {str(H[i]): V[i] for i in range(len(H))}
        except Exception as e:
            import logging
            logging.warning("[embed_cache] 缓存文件损坏,将重建: %s: %s", p, e)
            d = {}
    _mem[model] = d
    return d


def _save_locked(model: str) -> None:
    """原子落盘(tmp + rename);调用方须已持 _lk。"""
    d = _mem.get(model) or {}
    if not d:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        H = np.array(list(d.keys()))
        V = np.stack(list(d.values()))
        p = _path(model)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, hashes=H, vecs=V)   # 传文件对象,避免 savez 自动补 .npz
        tmp.rename(p)
    except Exception as e:
        import logging
        logging.warning("[embed_cache] 落盘失败(进程内缓存仍可用): %s", e)


def cached_embed(texts: list, model: str = "bge-small-zh-v1.5") -> list:
    """返回 texts 的 embedding(list[np.ndarray],与输入同序)。
    命中走缓存;未命中批量嵌一次并立即落盘。未命中批失败则抛(已缓存不丢,可续传)。"""
    if not texts:
        return []
    # 1) 持锁:载入 + 算未命中(快操作)
    with _lk:
        d = _load_locked(model)
        keys = [_key(model, t) for t in texts]
        miss_idx = [i for i, k in enumerate(keys) if k not in d]
        miss_texts = [texts[i] for i in miss_idx]
    # 2) 不持锁:网络嵌入(可能抛 → 交给调用方,缓存不受影响)
    if miss_texts:
        from eval.memory_interface import embed_texts   # 懒导入,破循环依赖
        new = embed_texts(miss_texts, model)
        # 3) 持锁:写入 + 落盘
        with _lk:
            d = _mem[model]
            for j, i in enumerate(miss_idx):
                d[keys[i]] = new[j]
            _save_locked(model)
    with _lk:
        d = _mem[model]
        return [d[k] for k in keys]


def cache_size(model: str = "bge-small-zh-v1.5") -> int:
    """已缓存的文本条数(供报告/调试)。"""
    with _lk:
        return len(_load_locked(model))


def reset_mem_cache() -> None:
    """清空进程内缓存(测试用;不删盘)。"""
    with _lk:
        _mem.clear()
