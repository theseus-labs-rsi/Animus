"""
pipeline.well_posed —— 边 A 闸(题面↔答案【良定义】)编排,与 pipeline.grounding(边 B)对称。

验证三角(redesign_factory_v2.3.md §V):一道好题 = (题面, 答案, 语料) 三边一致。
  · 边 B = 接地闸(gold↔语料,gold 能否在语料逐字+就近找到)—— pipeline.grounding,渲染后跑;
  · 边 A = 良定义闸(题面↔答案,gold 是否该题在世界里唯一正确解)—— 本模块,【出题前】跑,不碰 corpus。
两闸正交、串联才是完整护城河(§V.2)。

逐题派发到各线 line.well_posed(order, ws)(纯代码、零 LLM、确定性);各线判据见 docs/anchors/edge_a/*.md
(经对抗式 QA 实测背书:删了误杀型代理 roles_of/W5,留了真不变量,源头修复后闸≈防回归)。
"""
from __future__ import annotations
from collections import defaultdict


def _key(o: dict) -> dict:
    return {"line": o.get("line"), "capability": o.get("capability"),
            "entity": o.get("entity"), "field": o.get("field"), "gt": o.get("gt")}


def run_well_posed(orders: list, ws) -> tuple[list, dict]:
    """返回 (良定义通过的 order 列表, 报告 dict{pass_rate/by_line/drops})。纯代码、确定性。
    与 run_grounding 同构:per-line 派发 + (status,reason) + 存活率统计;弃因逐条留报告。"""
    from pipeline.lines import line_for                  # 懒导入,避免 lines→well_posed 循环

    kept, drops = [], []
    tally = defaultdict(lambda: {"n": 0, "ok": 0})

    for o in orders:
        line = line_for(o.get("line", ""))
        if line is None:
            status, reason = "drop", "无对应产线(fail-closed)"
        else:
            try:
                status, reason = line.well_posed(o, ws)
            except Exception as e:
                status, reason = "drop", f"well_posed 异常:{type(e).__name__}:{str(e)[:60]}"

        for k in (f"line:{o.get('line')}", "ALL"):
            tally[k]["n"] += 1
            tally[k]["ok"] += int(status == "well_posed")
        if status == "well_posed":
            kept.append(o)
        else:
            drops.append({**_key(o), "reason": reason})

    def rate(d):
        return {"n": d["n"], "well_posed": d["ok"],
                "pass_rate": round(d["ok"] / d["n"], 3) if d["n"] else None}

    report = {
        "overall": rate(tally["ALL"]),
        "by_line": {k[5:]: rate(v) for k, v in sorted(tally.items()) if k.startswith("line:")},
        "n_dropped": len(drops),
        "drops": drops,
    }
    return kept, report


# 自检:python -m pipeline.well_posed
if __name__ == "__main__":
    import sys
    from pipeline.world_state import build_demo_world
    from pipeline.lines import line_for

    ws = build_demo_world()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 真实枚举(已过源头修复)→ 跑边 A 闸,应几乎全过(源头已挡 ill-posed)
    orders = []
    for lid in ("L1_timeline", "L3_process"):
        ln = line_for(lid)
        if ln:
            orders += ln.enumerate(ws)
    kept, report = run_well_posed(orders, ws)
    ck("orders 非空(demo 世界有题)", len(orders) >= 1)
    ck("报告结构完整", set(report) == {"overall", "by_line", "n_dropped", "drops"})
    ck("overall.n == 输入题数", report["overall"]["n"] == len(orders))
    ck("kept + dropped == 输入", len(kept) + report["n_dropped"] == len(orders))
    ck("新鲜 order 几乎全过边 A 闸(源头已修)", report["overall"]["pass_rate"] is None or report["overall"]["pass_rate"] >= 0.9)

    # 注入一道 ill-posed L3(复现值绕过源头,手构)→ 必被对应线 well_posed drop
    bad = {"line": "L3_process", "capability": "L3_order", "entity": "AI工程部", "field": "",
           "gt": [{"field": "P0缺陷率", "value": "不存在的鬼值", "session": 1, "date": "2025-01-13", "op": "UPDATE"},
                  {"field": "Oncall", "value": "9", "session": 2, "date": "2025-01-20", "op": "UPDATE"},
                  {"field": "负责人", "value": "李四", "session": 3, "date": "2025-01-27", "op": "UPDATE"}],
           "evidence_sessions": [1, 2, 3], "aux": {"events": []}}
    kept2, report2 = run_well_posed([bad], ws)
    ck("手构 ill-posed(悬空值)被边 A 闸 drop", report2["n_dropped"] == 1 and not kept2)

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[well_posed self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
