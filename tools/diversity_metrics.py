"""
tools.diversity_metrics —— 语料多样性 / 跨场景区分度的【可证伪硬指标】。

按 DataMorgana(2501.12789)+ InfoSynth(2601.00575)+ Shaib 2403.00553 的定义实现:
  词汇/句法层(纯代码、无网络、语言无关):
    - NDG   n-gram 多样性 = Σ_{n=1..4} 去重n-gram/总n-gram(越高越多样,(0,4])
    - IDO   跨文档 n-gram 重叠率(越高=越同质,[0,1])—— 自重复的稳健连续版
    - CR    gzip 压缩比 = len(raw)/len(gzip)(越高=越冗余=越不多样)
  语义层(需句向量 embed_fn,可选):
    - HS    同质化分 = 文档向量两两平均余弦(越高=越同质);用 1−HS 当语义多样性
  跨场景区分度:
    - 两场景语料的 JS 散度(基于 n-gram 分布);+ bootstrap 置信区间

中文按【字符 n-gram】算(语言无关、零依赖);bootstrap 给 95% CI。
用法:python -m tools.diversity_metrics            # 自检
      python -m tools.diversity_metrics <corpus.json>  # 跑某语料(取 docs[].content)
"""
from __future__ import annotations
import gzip
import json
import math
import sys
from collections import Counter


def _char_ngrams(s: str, n: int):
    s = "".join(str(s).split())                      # 去空白,纯字符序列
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else []


def ndg(docs, max_n: int = 4) -> float:
    """n-gram 多样性:各阶 去重/总 之和。越高越多样(理论上限 = max_n)。"""
    total = 0.0
    blob = "".join("".join(str(d).split()) for d in docs)
    for n in range(1, max_n + 1):
        grams = [blob[i:i + n] for i in range(len(blob) - n + 1)]
        if grams:
            total += len(set(grams)) / len(grams)
    return round(total, 4)


def inter_doc_overlap(docs, n: int = 5) -> float:
    """跨文档 n-gram 重叠率:每篇 n-gram 里"也出现在别的文档"的占比,再对文档平均。越高=越同质。"""
    sets = [set(_char_ngrams(d, n)) for d in docs]
    if len(sets) < 2:
        return 0.0
    df = Counter()                                    # 每个 n-gram 出现在多少篇文档
    for s in sets:
        for g in s:
            df[g] += 1
    ratios = []
    for s in sets:
        if not s:
            continue
        shared = sum(1 for g in s if df[g] >= 2)
        ratios.append(shared / len(s))
    return round(sum(ratios) / len(ratios), 4) if ratios else 0.0


def gzip_cr(docs) -> float:
    """gzip 压缩比:越高=越冗余=越不多样。"""
    blob = "\n".join(str(d) for d in docs).encode("utf-8")
    if not blob:
        return 0.0
    return round(len(blob) / max(1, len(gzip.compress(blob, 6))), 4)


def homogenization_score(embeddings) -> float:
    """HS = 文档向量两两平均余弦(L2 归一后 = (‖Σê‖²−N)/(N(N−1)))。越高=越同质。需 numpy。"""
    import numpy as np
    E = np.asarray(embeddings, dtype=float)
    if E.ndim != 2 or len(E) < 2:
        return 0.0
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    N = len(E)
    s = E.sum(axis=0)
    return round(float((s @ s - N) / (N * (N - 1))), 4)


def _ngram_dist(docs, n: int = 3) -> Counter:
    c = Counter()
    for d in docs:
        c.update(_char_ngrams(d, n))
    return c


def js_divergence(docs_a, docs_b, n: int = 3) -> float:
    """两语料 n-gram 分布的 Jensen-Shannon 散度 ∈[0,1](log2)。越高=两场景越可分=区分度越强。"""
    pa, pb = _ngram_dist(docs_a, n), _ngram_dist(docs_b, n)
    ta, tb = sum(pa.values()) or 1, sum(pb.values()) or 1
    keys = set(pa) | set(pb)
    js = 0.0
    for k in keys:
        a, b = pa[k] / ta, pb[k] / tb
        m = (a + b) / 2
        if a > 0:
            js += 0.5 * a * math.log2(a / m)
        if b > 0:
            js += 0.5 * b * math.log2(b / m)
    return round(js, 4)


def bootstrap_ci(docs, metric_fn, n_boot: int = 300, seed: int = 0, frac: float = 0.8):
    """★无放回子采样估 95% CI(百分位法)。
    不用经典 with-replacement:复制文档会人为制造重复,把 IDO/CR/HS 这类【对重复敏感】的
    冗余指标系统性抬高(point 反而低于 CI 即此故)。子采样保持"无重复",量的是"换一批文档指标稳不稳"。"""
    import random
    rng = random.Random(seed)
    N = len(docs)
    m = max(2, int(frac * N))
    if N <= 2:
        return {"point": round(metric_fn(docs), 4), "ci": None}
    vals = sorted(metric_fn(rng.sample(docs, m)) for _ in range(n_boot))
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return {"point": round(metric_fn(docs), 4), "ci": [round(lo, 4), round(hi, 4)]}


def report(docs, embed_fn=None, n_boot: int = 200) -> dict:
    """一次性出全部词汇/句法指标(带 CI);若给 embed_fn 再补语义 HS。"""
    r = {
        "n_docs": len(docs),
        "NDG_n-gram多样性(越高越好)": bootstrap_ci(docs, ndg, n_boot),
        "IDO_跨文档重叠(越低越好)": bootstrap_ci(docs, inter_doc_overlap, n_boot),
        "CR_gzip压缩比(越低越好)": bootstrap_ci(docs, gzip_cr, n_boot),
    }
    if embed_fn is not None:
        try:
            embs = embed_fn(docs)
            r["HS_语义同质化(越低越好);语义多样性=1−HS"] = homogenization_score(embs)
        except Exception as e:
            r["HS_语义"] = f"(跳过:{str(e)[:60]})"
    return r


def _self_test() -> bool:
    diverse = ["今天开了产品评审会,讨论了下个季度的路线图和优先级排序。",
               "财务部通报本月报销新规,差旅标准上调,需提前申请。",
               "机房昨夜空调故障触发告警,值班同学已切换备用制冷。",
               "新同事入职培训安排在周三下午,包含安全与合规两讲。",
               "客户A续约谈判进入尾声,法务正在审最后一版合同条款。"]
    repetitive = ["本周例会通知:请大家准时参加本周例会。",
                  "本周例会通知:请大家准时参加本周例会,谢谢。",
                  "本周例会通知:请各位准时参加本周例会。",
                  "本周例会通知:请大家按时参加本周例会。",
                  "本周例会通知:请大家准时参加本周的例会。"]
    checks = []
    nd_d, nd_r = ndg(diverse), ndg(repetitive)
    checks.append(("NDG: 多样>重复", nd_d > nd_r))
    ido_d, ido_r = inter_doc_overlap(diverse), inter_doc_overlap(repetitive)
    checks.append(("IDO: 多样<重复", ido_d < ido_r))
    cr_d, cr_r = gzip_cr(diverse), gzip_cr(repetitive)
    checks.append(("CR: 多样<重复", cr_d < cr_r))
    js_same = js_divergence(diverse, diverse)
    js_diff = js_divergence(diverse, repetitive)
    checks.append(("JS: 自比≈0 < 异比", js_same < js_diff and js_same < 0.05))
    # HS:用假向量(多样=近正交,重复=近平行)
    try:
        import numpy as np
        div_emb = list(np.eye(5))                     # 正交 → HS≈0
        rep_emb = [np.array([1, 0.99, 0, 0, 0])] * 5  # 近平行 → HS≈1
        checks.append(("HS: 正交<平行", homogenization_score(div_emb) < homogenization_score(rep_emb)))
    except ImportError:
        pass
    print(f"  多样语料: NDG={nd_d} IDO={ido_d} CR={cr_d}")
    print(f"  重复语料: NDG={nd_r} IDO={ido_r} CR={cr_r}")
    print(f"  JS(自比/异比): {js_same} / {js_diff}")
    npass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"\n[diversity_metrics self-test] {npass}/{len(checks)} PASS")
    return npass == len(checks)


def _load_corpus_docs(path: str):
    obj = json.load(open(path, encoding="utf-8"))
    sessions = obj.get("corpus", obj).get("sessions", []) if isinstance(obj, dict) else []
    return [d.get("content", "") for s in sessions for d in s.get("docs", [])]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        docs = _load_corpus_docs(sys.argv[1])
        print(f"语料 {sys.argv[1]}:{len(docs)} 篇")
        print(json.dumps(report(docs), ensure_ascii=False, indent=2))
    else:
        sys.exit(0 if _self_test() else 1)
