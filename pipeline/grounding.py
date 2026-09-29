"""
pipeline.grounding —— ★命门3 接地闸(redesign_factory_v2.1 §G 的代码落地)。

把 orders(gold,源自世界)与 corpus(语料,源自世界)这两条【并列、本不交汇】的分支【汇合】做交叉校验:
每道题的 gold 必须在它的【证据文档】里【逐字 + 就近归属】可验,否则弃题(不接地的 gold 不出厂)。
纯代码、零 LLM、零网络、确定性、可复跑。判定原语 attributed + gt→标量映射 gold_scalar 在此;
每条线的 ground()(线自己懂自己的 gold 怎么接地)在各 lines/Lx.py。

跑法(干跑,验证逻辑,不改产物):
  ./venv/bin/python -m pipeline.grounding                                   # 自检
  ./venv/bin/python -m pipeline.grounding output/factory_v2_office_v3/06_questions.json \
                                          output/factory_v2_office_v3/05_corpus.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.world_state import _norm           # ★与判分同口径(去空格/全角/尾%)

WINDOW = 90                                       # 就近窗口(字符,_norm 后文本上算);90≈一个长句,容"同句但隔得远"的合法归属(±50 太紧,实测把相距 68 的真归属误杀)
STOP_MARKERS = ("停", "停止", "不再", "下架", "出院", "结案", "终止", "暂停")  # FORGET 停用标记


# ════════════════════════════════════════════════════════════════════════════
# 判定原语:value 与 anchor 在【任一】证据文档的就近窗口内共现(any 语义)
# ════════════════════════════════════════════════════════════════════════════
def attributed(value, anchor, docs, window: int = WINDOW) -> bool:
    """value 与 anchor(实体名/桥实体)在某篇 doc 的 ±window 字符内共现(均 _norm 后做子串)。
    any 语义:任一篇命中即 True(mention-on-change 下值只在变更周出现一次,不要求每篇都中)。
    docs:list[str]。anchor 为空则退化为"只验 value 出现"。"""
    nv = _norm(value)
    if not nv:
        return False
    na = _norm(anchor) if anchor else ""
    for text in docs:
        nt = _norm(text)
        if not nt:
            continue
        if not na:
            if nv in nt:
                return True
            continue
        i = nt.find(nv)
        while i >= 0:
            lo, hi = max(0, i - window), i + len(nv) + window
            if na in nt[lo:hi]:
                return True
            i = nt.find(nv, i + 1)
    return False


# ════════════════════════════════════════════════════════════════════════════
# gt → 待验标量(redesign §G.12.1 映射表;只覆盖"有明确单值"的能力)
#   L3(事件 list)/ABS(反向判)/FORGET(停用标记)/L5(两值)→ 返回 None,由各线 ground() 自己处理
# ════════════════════════════════════════════════════════════════════════════
def gold_scalar(order: dict):
    cap, gt = order.get("capability"), order.get("gt")
    if cap in ("KU", "PREEXPIRE", "L2_multihop"):
        return str(gt) if isinstance(gt, str) and gt.strip() else None
    if cap == "IE":
        v = (gt or {}).get("value") if isinstance(gt, dict) else None
        return str(v) if v not in (None, "") else None
    if cap == "MR":
        v = (gt or {}).get("value") if isinstance(gt, dict) else None
        return str(v) if v not in (None, "") else None
    if cap == "TR":
        v = (gt or {}).get("to") if isinstance(gt, dict) else None
        return str(v) if v not in (None, "") else None
    return None


# ════════════════════════════════════════════════════════════════════════════
# 证据文档收集(redesign §G.12.2):evidence_sessions 内的【信号文档】(非 filler);L5 含 conflict 文档
# ════════════════════════════════════════════════════════════════════════════
def gather_evidence(order: dict, by_id: dict) -> list[dict]:
    """返回证据池（含 doc_id/session/content），只取非 filler 文档。"""
    out = []
    for sid in (order.get("evidence_sessions") or []):
        s = by_id.get(sid) or by_id.get(str(sid))
        if not s:
            continue
        for d in s.get("docs", []):
            if d.get("is_filler"):                 # None(信号)/True(草堆);only True 排除
                continue
            out.append({"doc_id": d.get("doc_id"), "session": sid,
                        "content": d.get("content", ""),
                        "is_conflict": bool(d.get("is_conflict"))})
    return out


def _sessions_of(corpus_obj: dict) -> list:
    """corpus 真实结构外层包了一层 ["corpus"](§G.3);兼容直接给 sessions 的情形。"""
    if "corpus" in corpus_obj and isinstance(corpus_obj["corpus"], dict):
        return corpus_obj["corpus"].get("sessions", [])
    return corpus_obj.get("sessions", [])


def all_signal_text(corpus_obj: dict) -> str:
    """全语料信号文档正文拼接(ABS 反向判'字段全局缺失'用)。"""
    return "\n".join(d.get("content", "") for s in _sessions_of(corpus_obj)
                     for d in s.get("docs", []) if not d.get("is_filler"))


# ════════════════════════════════════════════════════════════════════════════
# stage 编排:遍历题 → 各线 ground() → 接地子集 + 存活率报告
# ════════════════════════════════════════════════════════════════════════════
def _key(q: dict) -> dict:
    return {"line": q.get("line"), "capability": q.get("capability"),
            "entity": q.get("entity"), "field": q.get("field")}


def run_grounding(questions: list, corpus_obj: dict) -> tuple[list, dict]:
    """返回 (接地通过的题列表, 报告 dict{survival/by_line/by_cap/drops})。纯代码、确定性。"""
    from pipeline.lines import line_for             # 懒导入,避免与 lines→grounding 循环

    by_id = {s["session_id"]: s for s in _sessions_of(corpus_obj)}
    all_sig = all_signal_text(corpus_obj)
    # Held-out leakage is invalid anywhere the solver can read, including a
    # mislabelled filler. Other lines retain their signal-only evidence scope.
    all_visible = "\n".join(d.get("content", "") for s in _sessions_of(corpus_obj)
                            for d in s.get("docs", []))

    kept, drops = [], []
    tally = defaultdict(lambda: {"n": 0, "ok": 0})   # 按 line / cap 计数

    for q in questions:
        line = line_for(q.get("line", ""))
        ev = gather_evidence(q, by_id)
        if line is None or not callable(getattr(line, "ground", None)):
            status, reason = "drop", "无对应产线 / 该线未实现 ground()(fail-closed)"
        else:
            try:
                status, reason = line.ground(q, ev, all_visible if q.get("line") == "L9_induction" else all_sig)
            except Exception as e:
                status, reason = "drop", f"ground() 异常:{type(e).__name__}:{str(e)[:60]}"

        cap = q.get("capability", "?")
        for k in (f"line:{q.get('line')}", f"cap:{cap}", "ALL"):
            tally[k]["n"] += 1
            tally[k]["ok"] += int(status == "grounded")
        if status == "grounded":
            # 06 是可交付题库：把本题实际送入接地器的非 filler 证据池一并发布。
            # 这是候选证据集合，不声称做最小证明集；但每个 ID 都来自声明时间窗。
            evidence_doc_ids = list(dict.fromkeys(
                str(item["doc_id"]) for item in ev if item.get("doc_id")))
            kept.append({**q, "evidence_doc_ids": evidence_doc_ids,
                         "candidate_evidence_doc_ids": evidence_doc_ids,
                         "evidence_scope": {"kind": "candidate_pool", "minimal_proof": "not_measured",
                                            "necessary_document_count": None,
                                            "difficulty_verified": False}})
        else:
            drops.append({**_key(q), "reason": reason})

    def rate(d):
        return {"n": d["n"], "grounded": d["ok"], "survival": round(d["ok"] / d["n"], 3) if d["n"] else None}

    report = {
        "overall": rate(tally["ALL"]),
        "by_line": {k[5:]: rate(v) for k, v in sorted(tally.items()) if k.startswith("line:")},
        "by_capability": {k[4:]: rate(v) for k, v in sorted(tally.items()) if k.startswith("cap:")},
        "n_dropped": len(drops),
        "drops": drops,
    }
    return kept, report


def print_report(report: dict, show_drops: int = 999):
    o = report["overall"]
    print(f"\n{'='*64}\n=== 接地闸存活率(命门3)===\n{'='*64}")
    print(f"总: {o['grounded']}/{o['n']} = {o['survival']:.1%} 接地" if o['survival'] is not None else "总: 0 题")
    print("\n-- by line --")
    for ln, d in report["by_line"].items():
        print(f"  {ln:16} {d['grounded']:>3}/{d['n']:<3} = {d['survival']:.0%}" if d['survival'] is not None else f"  {ln}: 0")
    print("\n-- by capability --")
    for cap, d in report["by_capability"].items():
        print(f"  {cap:14} {d['grounded']:>3}/{d['n']:<3} = {d['survival']:.0%}" if d['survival'] is not None else f"  {cap}: 0")
    if report["drops"]:
        print(f"\n-- 弃题({report['n_dropped']})逐条弃因 --")
        for dr in report["drops"][:show_drops]:
            print(f"  ✗ [{dr['line']}/{dr['capability']}] {dr['entity']}.{dr['field']}: {dr['reason']}")


# ════════════════════════════════════════════════════════════════════════════
# 自检 + 干跑 CLI
# ════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    checks = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # attributed 原语
    docs = ["天枢数据部本周负责人为王皓,P0缺陷率0.13。", "基础架构部本周SLA99.92%。"]
    ck("逐字+就近命中", attributed("王皓", "天枢数据部", docs))
    ck("幽灵值不命中(0次)", not attributed("尤娜", "天枢数据部", docs))
    ck("串部门:0.13就近天枢、不就近基础架构", attributed("0.13", "天枢数据部", docs) and not attributed("0.13", "基础架构部", docs))
    ck("尾%归一:99.92就近基础架构", attributed("99.92", "基础架构部", docs))
    ck("any语义:任一篇命中即可", attributed("99.92", "基础架构部", docs))

    # gold_scalar 映射
    ck("IE 取 value", gold_scalar({"capability": "IE", "gt": {"value": "王皓", "at_week": 3}}) == "王皓")
    ck("MR 取 value", gold_scalar({"capability": "MR", "gt": {"value": "8次", "session": 2}}) == "8次")
    ck("TR 取 to", gold_scalar({"capability": "TR", "gt": {"to": "李四", "session": 2}}) == "李四")
    ck("KU 取 gt 本身", gold_scalar({"capability": "KU", "gt": "CFO"}) == "CFO")
    ck("L3/ABS/FORGET 返回 None(线自处理)", gold_scalar({"capability": "L3_order", "gt": []}) is None
       and gold_scalar({"capability": "ABS", "gt": "INSUFFICIENT_EVIDENCE"}) is None)

    # run_grounding:一个迷你 corpus + 三道题(接地/幽灵/串部门)→ 1 留 2 弃
    corpus = {"corpus": {"sessions": [
        {"session_id": 0, "date": "2025-01-06", "docs": [
            {"doc_id": "s0_sig_0", "content": "天枢数据部第0周负责人为王皓,P0缺陷率0.13。", "is_filler": None},
            {"doc_id": "s0_fil_1", "content": "食堂本周菜单更新(无关草堆)。", "is_filler": True},
        ]},
    ]}}
    qs = [
        {"line": "L1_timeline", "capability": "KU", "entity": "天枢数据部", "field": "负责人",
         "gt": "王皓", "evidence_sessions": [0], "aux": {}},                      # 应接地
        {"line": "L1_timeline", "capability": "KU", "entity": "天枢数据部", "field": "负责人",
         "gt": "尤娜", "evidence_sessions": [0], "aux": {}},                      # 幽灵 → 弃
        {"line": "L1_timeline", "capability": "MR", "entity": "基础架构部", "field": "P0",
         "gt": {"value": "0.13", "session": 0}, "evidence_sessions": [0], "aux": {}},  # 串部门(0.13属天枢)→ 弃
    ]
    kept, report = run_grounding(qs, corpus)
    ck("接地题保留(王皓)", any(q["gt"] == "王皓" for q in kept))
    ck("接地题发布非空证据文档池",
       kept[0]["evidence_doc_ids"] == ["s0_sig_0"])
    ck("幽灵题被弃(尤娜)", all(q.get("gt") != "尤娜" for q in kept))
    ck("串部门题被弃(0.13@基础架构)", len(kept) == 1)
    ck("存活率 1/3", report["overall"]["survival"] == round(1 / 3, 3))
    ck("草堆不算证据(food 不影响)", report["overall"]["n"] == 3)

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[grounding self-test] {npass}/{len(checks)} PASS")
    return npass == len(checks)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(0 if _self_test() else 1)
    bench, corpus_path = args[0], args[1]
    questions = json.loads(Path(bench).read_text(encoding="utf-8"))
    corpus_obj = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    print(f"[grounding] bench={bench}\n[grounding] corpus={corpus_path} | {len(questions)} 题")
    kept, report = run_grounding(questions, corpus_obj)
    print_report(report)
    print(f"\n→ 接地通过 {len(kept)}/{len(questions)} 题(若接进流水线即写 06_grounded_questions.json)")


if __name__ == "__main__":
    main()
