"""
pipeline.lines —— 产线注册表(redesign 宪法/纲领的代码落地,唯一真源)。

  LINES    = 已建成的产线【实例】(自带 prepare/enumerate/gt/intent;run_lines 遍历它派发)。
  PLANNED  = 纲领规划但未落地的坐标(只元数据,无代码)——让 taxonomy_prose 给中央议会【完整 L1–L7 菜单】。
  taxonomy_prose / line_for / implemented_ids = 派生工具,别处都从这里取。

加一条线:写 lines/Lx.py 的 ProductionLine 子类 + 在 LINES 加一行。别处不动。
"""
from __future__ import annotations
import re
import math
import json
from collections import Counter

from pipeline.lines.base import ProductionLine, Order
from pipeline.lines.L1_timeline import TimelineLine
from pipeline.lines.L2_relational import RelationalLine
from pipeline.lines.L3_process import ProcessLine
from pipeline.lines.L4_preference import PreferenceLine
from pipeline.lines.L5_conflict import ConflictLine
from pipeline.lines.L6_refusal import RefusalLine
from pipeline.lines.L7_consolidation import ConsolidationLine
from pipeline.lines.L8_transition import TransitionLine
from pipeline.lines.L9_induction import InductionLine
from pipeline.lines.L10_admission import AdmissionLine

# ── 已建成的产线(实例)──
# ★L8_transition:结构原型【状态机】招牌线,feasible 只读取世界蓝图自然声明的 state_machines；
#   不按场景名决定，也不为激活能力事后补造生命周期。
# ★L9_induction:条件归纳招牌线,auto_activate——prepare 声明阶跃规则 + 注入执行实例(ws.rule_instances)后即触发;
#   从【单条情境→动作】实例归纳 IF-THEN 函数,并在【未见 trigger 值 x*】上外推(function-on-unseen)。
#   gold=apply_rule 纯查表;闭选项 MC EM 避开 llm_judge;canon 层防检索捷径;唯一性闸保 gold 硬(与 L8 换皮无关)。
# ★L10_admission:写入期【非泄露】招牌线,auto_activate——prepare 注入敏感基质(ws.sensitive)后即触发,
#   与 L6 读取期拒答正交(L6 考"别瞎编缺席值",L10 考"记得敏感原文也别逐字吐")。
LINES: list[ProductionLine] = [TimelineLine(), RelationalLine(), ProcessLine(), PreferenceLine(),
                               ConflictLine(), RefusalLine(), ConsolidationLine(), TransitionLine(),
                               InductionLine(), AdmissionLine()]

# ── 规划但未落地的坐标(只元数据,给议会看完整菜单;建成后从这里删、移进 LINES)──
# ★L1–L7 七条线全部落地。新规划线在此登记(只元数据),建成后移进 LINES。
PLANNED: list[tuple[str, str, str]] = []

INVARIANTS: list[str] = [
    "每条产线必须自带【代码可算的 ground truth】——没有代码 gt 的产线不准入厂(守护城河)。",
    "所有产线共享【同一个世界基质】,不得各造各的(防语料自相矛盾)。",
]


def taxonomy_prose() -> str:
    """给中央议会提示词用:列出完整 L1–L7 坐标(已建 LINES + 规划 PLANNED),让 LLM 照抄规范 id。"""
    items = [(l.id, l.title, l.memory) for l in LINES] + list(PLANNED)
    return " / ".join(f"{i}({t}:{m})" for (i, t, m) in items)


def implemented_ids() -> list[str]:
    return [l.id for l in LINES]


def line_for(active_id: str) -> ProductionLine | None:
    """鲁棒匹配:议会可能吐 'L2_relations_multi_hop' 等变体 → 先精确、再按 L<数字> 前缀兜。
    只解析【已建成】的产线;规划中(L3–L7)的坐标返回 None(run_lines 据此跳过)。"""
    if not active_id:
        return None
    exact = next((l for l in LINES if l.id == active_id), None)
    if exact:
        return exact
    m = re.match(r"[Ll](\d+)", active_id.strip())
    if m:
        pref = "L" + m.group(1)
        return next((l for l in LINES if l.id.startswith(pref + "_")), None)
    return None


# ── 能力依赖 DAG(Skill-it 依赖图思想的【忠实落地】)───────────────────────────
def dependency_graph() -> dict[str, list[str]]:
    """{line_id: [它依赖的世界基质特征 / 前置线]} —— 每条产线"靠世界里的哪种基质才能产题"。

    【诚实声明范围】Skill-it 的依赖图本是为【在线训练数据混合】服务的(按各技能学习
    速度动态重配采样权重)。我们是【评测生成器、不训练模型】,那个在线混合红利用不上、
    不去造它。此图只服务两件事:
      (1) 激活可行性判断 —— 基质不满足时,把"产线静默产 0 单"显式化、可日志化(见 feasible_lines);
      (2) 论文里的【能力依赖结构图】—— 一张可读的 DAG,讲清各能力坐落在哪层基质之上。
    边 = ProductionLine.requires 声明(每条线自报基质需求);L1 无依赖 = 根。"""
    return {l.id: list(l.requires) for l in LINES}


def feasible_lines(ws, profile: dict) -> list[tuple[str, bool, str]]:
    """遍历 LINES 调各自 feasible,返回 [(line_id, 能否产题, 一句人读原因)]。
    给编排/日志用:激活某线前先问它"当前世界喂得饱吗",替代 enumerate 默默吐空单。"""
    return [(l.id, *l.feasible(ws, profile)) for l in LINES]


# ── 注册表级产线编排(stage_world / stage_orders / 闭环 driver 共用;原 run_factory_v2 搬入)─────
# ★派发:run_lines 遍历白皮书激活线 → line.enumerate 点菜(配额驱动 + feasible 门控)。
#   叠基质:prepare_lines 各线把所需基质叠进共享世界(命门1)。两者都只依赖本表的 line_for。
def _budgeted_lines(wp, ws, budget: int, log, stats: dict | None, *, process_orders=None) -> list[dict]:
    """Allocate a total budget before selection; never refill a short line with L6.

    Candidate enumeration is bounded and its limit is reported. A pool at the
    limit is not advertised as the exhaustive supply. Invalid and duplicate
    candidates cannot consume a selected slot.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("question_budget must be a nonnegative integer")
    profile = wp.get("domain_profile", {})
    planned, seen, unknown = [], set(), []
    for item in wp.get("active_lines", []):
        line = line_for(item.get("line"))
        if line is None:
            unknown.append(item.get("line")); continue
        if line.id in seen:
            continue
        seen.add(line.id)
        weight = item.get("weight", 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid active-line weight for {line.id}: {weight!r}")
        planned.append((line, float(weight), False))
    for line in LINES:
        if getattr(line, "auto_activate", False) and line.id not in seen and line.feasible(ws, profile)[0]:
            planned.append((line, 1.0, True))
            seen.add(line.id)
    total_weight = sum(weight for _, weight, _ in planned)
    # Normalize before multiplying to avoid overflow for finite large weights.
    if not math.isfinite(total_weight):
        largest = max((weight for _, weight, _ in planned), default=1)
        planned = [(line, weight / largest, auto) for line, weight, auto in planned]
        total_weight = sum(weight for _, weight, _ in planned)
    shares = [budget * (weight / total_weight) if total_weight else 0 for _, weight, _ in planned]
    allocations = [math.floor(share) for share in shares]
    remainder = budget - sum(allocations) if total_weight else 0
    priority = sorted(range(len(planned)), key=lambda i: (-(shares[i] - allocations[i]), i))
    for index in priority[:remainder]:
        allocations[index] += 1
    supply_plan = wp.get("supply_plan") if wp.get("delivery_target") else None
    support = {}
    if supply_plan:
        if supply_plan["candidate_budget"] != budget:
            raise ValueError("Delivery candidate budget differs from the frozen supply plan")
        support = {row["line"]: row for row in supply_plan["applicability"]}
        planned = [(line_for(lid), 1.0, False) for lid in supply_plan["candidate_allocation"]]
        if any(line is None for line, _, _ in planned):
            raise ValueError("Delivery supply plan contains an unknown line")
        allocations = [supply_plan["candidate_allocation"][line.id] for line, _, _ in planned]
    limit = max(1000, min(10000, budget * 20))
    report = {"mode": "total_budget", "question_budget": budget,
              "legacy_total_q": wp.get("capability_targets", {}).get("total_q"),
              "candidate_limit_per_line": limit, "selection": "eligible_capability_round_robin",
              "unimplemented_lines": unknown, "lines": []}
    selected = []
    for (line, weight, auto), allocated in zip(planned, allocations):
        feasible, reason = line.feasible(ws, profile)
        declared = support.get(line.id)
        if declared and not (declared["applicable"] and declared["implemented"]):
            feasible, reason = False, declared["reason"]
        extra = process_orders if line.id == "L3_process" and process_orders is not None else []
        if extra and not feasible:
            feasible, reason = True, "Bounded typed process proposals are structurally available; semantics await review"
        row = {"line": line.id, "weight": weight, "auto_activated": auto, "allocated": allocated,
               "feasible": feasible, "feasibility_reason": reason, "candidates": 0, "eligible": 0,
               "selected": 0, "duplicate_candidates": 0, "rejected": {}, "pool_at_limit": False}
        eligible, identities, rejected = [], set(), Counter()
        if feasible and allocated > 0:
            candidates = line.enumerate(ws, limit, wp) + list(extra)
            row["candidates"] = len(candidates)
            row["pool_at_limit"] = len(candidates) >= limit
            for order in candidates:
                status, why = line.well_posed(order, ws)
                if status != "well_posed":
                    rejected[why or status] += 1
                    continue
                # Exclude render/provenance metadata from identity; qid semantics
                # remain the responsibility of question_contract downstream.
                identity = json.dumps({key: order.get(key) for key in
                    ("line", "capability", "entity", "field", "gt", "aux")},
                    sort_keys=True, ensure_ascii=False)
                if (supply_plan or {}).get("instance_policy"):
                    from pipeline.capability_contract import verify_order
                    certificate = verify_order(line,ws,order)
                    if not certificate["passed"]:
                        rejected["original_answer_certificate_failed"] += 1
                        continue
                    identity = certificate["family_key"]
                if identity in identities:
                    row["duplicate_candidates"] += 1
                    continue
                identities.add(identity)
                eligible.append(order)
            pools = {}
            for order in eligible:
                pools.setdefault(order.get("capability", ""), []).append(order)
            picked, offset = [], 0
            if (supply_plan or {}).get("instance_policy"):
                from pipeline.capability_contract import family_key
                requirement = next((r for r in supply_plan["requirements"] if r["line"] == line.id),{})
                subtype_codes = {"native_trend":"S1_trend","compare":"S2_compare"}
                for subtype,minimum in requirement.get("subtype_minimum",{}).items():
                    choices = [o for o in eligible if (o.get("aux") or {}).get("sub") == subtype_codes[subtype]]
                    picked.extend(choices[:minimum])
                used = {family_key(o) for o in picked}
                pools = {cap:[o for o in pool if family_key(o) not in used] for cap,pool in pools.items()}
            while len(picked) < allocated:
                round_items = [pool[offset] for pool in pools.values() if len(pool) > offset]
                if not round_items:
                    break
                picked.extend(round_items[:allocated - len(picked)])
                offset += 1
            selected.extend(picked)
            row["selected"] = len(picked)
        if process_orders is not None and line.id == "L3_process":
            row["process_proposals"] = {"structurally_bound": len(extra),
                "selected": sum(o.get("capability") == "L3_process_trace" for o in selected),
                "semantic_status": "not_reviewed"}
        row["eligible"] = len(eligible)
        row["rejected"] = dict(rejected)
        row["shortfall"] = allocated - row["selected"]
        row["shortfall_reason"] = ("infeasible_substrate" if not feasible else
            "bounded_candidate_supply" if row["pool_at_limit"] else "eligible_supply_exhausted") if row["shortfall"] else None
        report["lines"].append(row)
    report["selected"] = len(selected)
    report["shortfall"] = budget - len(selected)
    if not total_weight:
        report["shortfall_reason"] = "no_positive_weight_lines"
    if stats is not None:
        stats.clear(); stats.update(report)
    log(f"  总题预算 {budget}: 选中 {len(selected)}, 缺口 {report['shortfall']}; 各线缺口不转移")
    for row in report["lines"]:
        log(f"    {row['line']}: 配额 {row['allocated']}, 合法候选 {row['eligible']}, 选中 {row['selected']}"
            + (f", 缺口原因 {row['shortfall_reason']}" if row["shortfall"] else ""))
    return selected


def run_lines(wp, ws, log=print, quotas=None, *, question_budget=None, stats=None, process_proposals=None) -> list[dict]:
    """按白皮书激活的产线,line_for 鲁棒匹配 → 调 line.enumerate 点菜。
    ★配额驱动(closed_loop_targetspec §7):`quotas={line_id: 配额}`(由 invert_rate 反推),缺省回退 total_q。
    ★feasible 接线(盲审 B6:零件原本没装上):基质喂不饱的线跳过 + 日志,不无效产 0 单、不死循环。"""
    process_orders = None
    if process_proposals is not None:
        from pipeline.process_proposals import validate_process_report
        process_orders = validate_process_report(process_proposals, wp, ws)
    if question_budget is not None:
        if quotas is not None:
            raise ValueError("question_budget and per-line quotas are mutually exclusive")
        return _budgeted_lines(wp, ws, question_budget, log, stats, process_orders=process_orders)
    active = [l.get("line") for l in wp.get("active_lines", [])]
    profile = wp.get("domain_profile", {})
    default_target = int(wp.get("capability_targets", {}).get("total_q", 40))
    quotas = quotas or {}
    orders_out, fired, skipped, seen = [], [], [], set()
    for aid in active:
        line = line_for(aid)                       # ★鲁棒匹配 LLM 的 id 变体 → 已建产线实例(未建返回 None)
        if line is None:
            skipped.append(f"{aid}(规划中/未建)"); continue
        if line.id in seen:
            continue
        seen.add(line.id)
        ok, why = line.feasible(ws, profile)       # ★基质可行性:不满足 → 跳过(避免无效产 0 单 / 死循环)
        extra = process_orders if line.id == "L3_process" and process_orders is not None else []
        if extra and not ok:
            ok, why = True, "Bounded typed process proposals available"
        if not ok:
            skipped.append(f"{line.id}(基质不足:{why})"); continue
        q = int(quotas.get(line.id, default_target))   # ★该线配额(over-provision 已由 invert_rate 算好)
        got = line.enumerate(ws, q, wp)            # 各产线自带代码 gt(护城河);wp 供 L2 等取关系
        if extra:
            # Only the explicit proposal path mixes capabilities. Keep the
            # original line quota; sufficient old sorting supply must not hide
            # every newly authored process candidate.
            pools, picked, offset = {}, [], 0
            for order in got + list(extra):
                pools.setdefault(order.get("capability", ""), []).append(order)
            while len(picked) < max(q, 0):
                round_items = [pool[offset] for pool in pools.values() if len(pool) > offset]
                if not round_items:
                    break
                picked.extend(round_items[:q - len(picked)])
                offset += 1
            got = picked
        orders_out.extend(got)
        fired.append(f"{line.id}:{len(got)}单/配额{q}")
    # ★结构原型自动激活:auto_activate 线(如 L8 状态机)只要世界里有其基质(feasible)就触发,
    #   不依赖议会是否在 active_lines 列它 ——「场景结构分叉」应由【世界有没有该结构】决定,而非 LLM 判定。
    for line in LINES:
        if getattr(line, "auto_activate", False) and line.id not in seen:
            ok, why = line.feasible(ws, profile)
            if ok:
                seen.add(line.id)
                q = int(quotas.get(line.id, default_target))
                got = line.enumerate(ws, q, wp)
                orders_out.extend(got)
                fired.append(f"{line.id}:{len(got)}单/配额{q}(结构自动激活)")
    log(f"  产线已跑: {fired or '无'}")
    if skipped:
        log(f"  ⓘ 白皮书激活但跳过: {skipped}")
    if stats is not None:
        stats.clear()
        stats.update({"mode": "explicit_quotas" if quotas else "legacy_per_line",
                      "legacy_total_q": default_target, "selected": len(orders_out),
                      "fired": fired, "skipped": skipped})
        if process_orders is not None:
            stats["process_proposals"] = {"structurally_bound": len(process_orders),
                "selected": sum(o.get("capability") == "L3_process_trace" for o in orders_out),
                "semantic_status": "not_reviewed"}
    return orders_out


def prepare_lines(wp, ws, log=print):
    """各激活产线把所需基质叠进【共享世界】(命门1):L2=加人员实体、L5=注矛盾,其它默认 no-op。
    stage_world 与闭环 driver 共用,避免两处各写一遍 prepare 循环。"""
    blueprint = getattr(ws, "world_blueprint", None) or {}
    if blueprint and not blueprint.get("legacy_adapter"):
        # 显式世界蓝图已经冻结领域本体与动力学。一般能力线不能再注入字段或改轨迹；
        # 但允许显式声明 typed_overlay_safe 的产线，从冻结真值派生不改 canonical 的评测证据侧信道。
        profile = {**wp.get("domain_profile", {}), "supply_construction": getattr(ws, "supply_construction", {})}
        seen, overlays = set(), []
        for aid in [item.get("line") for item in wp.get("active_lines", [])]:
            line = line_for(aid)
            if not line or line.id in seen or not getattr(line, "typed_overlay_safe", False):
                continue
            seen.add(line.id)
            note = line.prepare(ws, profile)
            if note:
                overlays.append(line.id)
                log(note)
        if overlays:
            log(f"  ✓ typed world 已冻结:仅派生评测证据侧信道 {overlays}，canonical 不变")
        else:
            log("  ✓ typed world 已冻结:能力线只读映射，canonical 不变")
        return
    profile = {**wp.get("domain_profile", {}), "supply_construction": getattr(ws, "supply_construction", {})}
    seen = set()
    for aid in [l.get("line") for l in wp.get("active_lines", [])]:
        line = line_for(aid)
        if line and line.id not in seen:
            seen.add(line.id)
            note = line.prepare(ws, profile)
            if note:
                log(note)
    # ★结构原型自动激活线(如 L10 敏感注入):即便议会未列进 active_lines,也要 prepare 其基质,
    #   否则 feasible 恒 False、run_lines 的 auto_activate 无从触发(与 run_lines 同款兜底)。
    for line in LINES:
        if getattr(line, "auto_activate", False) and line.id not in seen:
            seen.add(line.id)
            note = line.prepare(ws, profile)
            if note:
                log(note)


def _selftest() -> int:
    """注册表自检(由 __main__.py 触发:python -m pipeline.lines)。返回失败数(0 = 全 PASS)。"""
    from pipeline.world_state import assemble_world, build_demo_world

    print("taxonomy_prose():\n ", taxonomy_prose())
    print("\nimplemented:", implemented_ids())
    print("\n鲁棒匹配:")
    for v in ["L1_timeline", "L2_relations_multi_hop", "L7_consolidation_summary", "L6_refusal", "L9_unknown"]:
        s = line_for(v)
        print(f"  {v:30} → {s.id if s else None}")

    print("\n依赖图 dependency_graph():")
    dg = dependency_graph()
    for lid, reqs in dg.items():
        print(f"  {lid:16} requires {reqs or '∅(根)'}")

    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 不变量:每条 LINES 必须能实例化(= 四个抽象方法都实现了 = 自带 gt)
    bad = [l.id for l in LINES if not (callable(getattr(l, "gt", None)) and callable(getattr(l, "enumerate", None)))]
    ck("每条线自带 gt+enumerate", not bad)

    # ── 依赖图结构不变量 ──
    ck("依赖图覆盖所有 LINES", set(dg) == {l.id for l in LINES})
    ck("L1 无依赖(根)", dg.get("L1_timeline") == [])
    ck("L2 依赖 soft_fk_path", dg.get("L2_relational") == ["soft_fk_path"])
    ck("L3 依赖 multi_event_timelines", dg.get("L3_process") == ["multi_event_timelines"])
    ck("L5 依赖 text_fields", dg.get("L5_conflict") == ["text_fields"])

    # ── feasible:每条有依赖的线,在【满足/不满足基质】两个小世界上分别 True/False ──
    def feas(line_id, ws, profile):
        return next(ok for (lid, ok, _) in feasible_lines(ws, profile) if lid == line_id)

    demo = build_demo_world()                                  # L1/L3 标准世界:AI工程部 6 个跨周事件

    # L1 根线:任何世界恒可行(哪怕空世界)
    empty_ws, _ = assemble_world({"entities": []})
    ck("L1 feasible 恒 True(空世界亦然)", feas("L1_timeline", empty_ws, {}) is True)

    # L2:profile 2 person 字段 → True;仅 1 个 → False(口径同 prepare)
    p2 = {"field_schema": [{"name": "负责人", "kind": "person"}, {"name": "汇报对象", "kind": "person"}]}
    p1 = {"field_schema": [{"name": "负责人", "kind": "person"}]}
    ck("L2 feasible: 2 person 字段 → True", feas("L2_relational", empty_ws, p2) is True)
    ck("L2 feasible: 1 person 字段 → False", feas("L2_relational", empty_ws, p1) is False)

    # L3:有 ≥3 跨周事件的实体 → True;单 SET 实体(0 事件) → False
    lone_ws, _ = assemble_world({"entities": [
        {"name": "孤部", "fields": {"状态": {"type": "stable", "value": "正常"}}}]})
    ck("L3 feasible: 多事件世界 → True", feas("L3_process", demo, {}) is True)
    ck("L3 feasible: 单事件世界 → False", feas("L3_process", lone_ws, {}) is False)

    # L5:有文本类字段 → True;全数值字段 → False(口径同 inject_conflicts)
    text_ws, _ = assemble_world({"entities": [
        {"name": "搜索部", "fields": {"负责人": {"type": "stable", "value": "李娜"}}}]})
    num_ws, _ = assemble_world({"entities": [
        {"name": "指标部", "fields": {"营收": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "10"}, {"session": 2, "value": "20"}]}}}]})
    ck("L5 feasible: 文本字段世界 → True", feas("L5_conflict", text_ws, {"field_schema": [{"name": "负责人", "kind": "person"}]}) is True)
    ck("L5 feasible: 全数值世界 → False", feas("L5_conflict", num_ws, {"field_schema": [{"name": "营收", "kind": "numeric"}]}) is False)

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"\n不变量检查: {npass}/{len(checks)} PASS")
    return len(checks) - npass


# 直接 `python __init__.py` 时也可跑(常规入口是 python -m pipeline.lines → __main__.py)
if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
