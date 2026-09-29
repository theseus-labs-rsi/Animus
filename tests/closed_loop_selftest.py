"""
闭环旋钮 driver 的【纯逻辑】离线自检(无 LLM):
  · 5 个决策纯函数:_orders_by_line / _order_deficit / _grow_for_supply / _floor_status /
    _scale_world_contract;
  · 一个【控制流编排】仿真:用合成供给表跑 driver 的循环骨架,验证
      ①供不上 → 长世界 → 终能补齐;②floor 不达 → 纠偏;不可行线 → 永久 UNMET 不空转。
driver 本体(build_to_target)调真 LLM stage 逻辑,端到端另用小规模真跑验证(no-mock)。
这里只锁【循环的大脑】是对的。

跑:./venv/bin/python tests/closed_loop_selftest.py
"""
from __future__ import annotations
import sys, math, dataclasses
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.closed_loop import (_orders_by_line, _order_deficit, _order_shortfall, _grow_for_supply,
                                  _ensure_l7_capacity,
                                  _floor_status, _resolve_per_line_contract,
                                  _scale_world_contract)
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.targetspec import TargetSpec, WorldParams, invert_rate, N_ENT_CLAMP, N_SESS_CLAMP, DEFAULT_SLACK

checks: list[tuple[bool, str]] = []


def ck(name, cond):
    checks.append((bool(cond), name))


# ── _orders_by_line ──────────────────────────────────────────────────────────
orders = [{"line": "L1_timeline"}] * 8 + [{"line": "L2_relational"}] * 4 + [{"line": "L3_process"}] * 5
obl = _orders_by_line(orders)
ck("_orders_by_line 计数正确", obl == {"L1_timeline": 8, "L2_relational": 4, "L3_process": 5})
ck("_orders_by_line 空输入 → {}", _orders_by_line([]) == {})

# ── _order_deficit:只比较硬下限；不可行线不计、达标线不计 ────────────────
target = {"L1_timeline": 10, "L2_relational": 6, "L3_process": 5, "L5_conflict": 3}
produced = {"L1_timeline": 8, "L2_relational": 6, "L3_process": 2}
feasible = {"L1_timeline", "L2_relational", "L3_process"}               # L5 不可行(无文本字段)
defi = _order_deficit(produced, target, feasible)
ck("_order_deficit:硬下限差 2 也必须进赤字", defi.get("L1_timeline") == 2)
ck("_order_deficit:硬下限差 3 进赤字", defi.get("L3_process") == 3)
ck("_order_deficit:配额已满足的线不进赤字(L2)", "L2_relational" not in defi)
ck("_order_deficit:不可行线不进赤字(L5,扩世界也没用)", "L5_conflict" not in defi)
ck("_order_deficit:全达标 → 空赤字", _order_deficit({"L1_timeline": 10}, {"L1_timeline": 10}, {"L1_timeline"}) == {})
# 当前真实受控样本：软配额 L3=16 只产 13，但硬下限 L3=5，绝不能重建世界。
controlled_produced = {
    "L1_timeline": 8, "L2_relational": 7, "L3_process": 13,
    "L5_conflict": 7, "L6_refusal": 7, "L7_consolidation": 8,
}
controlled_floor = {
    "L1_timeline": 4, "L2_relational": 4, "L3_process": 5,
    "L5_conflict": 4, "L6_refusal": 3, "L7_consolidation": 4,
}
ck("_order_deficit:真实样本已过硬下限，不追软 slack 重建",
   _order_deficit(controlled_produced, controlled_floor, set(controlled_floor)) == {})
ck("_order_shortfall:最终硬闸包含不可行线，不能进入不可能成功的 corpus",
   _order_shortfall({"L1_timeline": 4, "L7_consolidation": 3},
                    {"L1_timeline": 4, "L7_consolidation": 4})
   == {"L7_consolidation": 1})

# 显式配额是调用者契约：规范化可以，静默丢线不可以。
ck("_resolve_per_line_contract:可规范化 L1 简写",
   _resolve_per_line_contract(["L1_timeline"], {"L1": 3}) == {"L1_timeline": 3})
try:
    _resolve_per_line_contract(["L1_timeline"], {"L7_consolidation": 4})
except WorldBlueprintError:
    _inactive_floor_rejected = True
else:
    _inactive_floor_rejected = False
ck("_resolve_per_line_contract:显式 L7 未激活时 fail-closed",
   _inactive_floor_rejected)

# ── _grow_for_supply:有赤字才长、夹 clamp、无赤字原样 ───────────────────────────
p0 = WorldParams(n_entities=12, n_sessions=10, quota_L1=10, max_n_conflicts=3, target_orders=target)
p1 = _grow_for_supply(p0, {"L3_process": 3})
ck("_grow_for_supply:有赤字 → n_entities 增大", p1.n_entities > p0.n_entities)
ck("_grow_for_supply:有赤字 → n_sessions 增大", p1.n_sessions > p0.n_sessions)
ck("_grow_for_supply:增量轮可锁住时间轴",
   _grow_for_supply(p0, {"L3_process": 3}, grow_sessions=False).n_sessions == p0.n_sessions)
ck("_grow_for_supply:target_orders 不动(长的是供给侧,不是配额)", p1.target_orders == p0.target_orders)
ck("_grow_for_supply:无赤字 → 原样返回", _grow_for_supply(p0, {}) is p0)
p_big = WorldParams(n_entities=N_ENT_CLAMP[1], n_sessions=N_SESS_CLAMP[1], quota_L1=1, max_n_conflicts=1, target_orders={})
p_big2 = _grow_for_supply(p_big, {"L1_timeline": 99})
ck("_grow_for_supply:已到 clamp 上限 → 不越界", p_big2.n_entities == N_ENT_CLAMP[1] and p_big2.n_sessions == N_SESS_CLAMP[1])

# ── _scale_world_contract:只固定显式 exact 类型，结构密度同步增长 ──────────
typed_wp = {
    "shared_world_spec": {
        "entities": {"count": 999},
        "timeline": {"n_sessions": 999},
    },
    "world_blueprint": {
        "entity_types": [
            {"id": "player", "count": 2, "primary": True, "cardinality_policy": "exact"},
            {"id": "boss", "count": 3},
            {"id": "equipment", "count": 5},
        ],
        "relation_types": [
            {"id": "equips", "min_count": 1},
            {"id": "drops", "min_count": 2},
            {"id": "optional_link", "min_count": 0},
        ],
        "event_types": [
            {"id": "defeat_boss", "min_count": 2},
            {"id": "acquire_item", "min_count": 3},
        ],
        "temporal_model": {"n_sessions": 8},
    },
}
_scale_world_contract(typed_wp, n_entities=20, n_sessions=12)
scaled_bp = typed_wp["world_blueprint"]
scaled_counts = {item["id"]: item["count"] for item in scaled_bp["entity_types"]}
scaled_rel_min = {item["id"]: item["min_count"] for item in scaled_bp["relation_types"]}
scaled_event_min = {item["id"]: item["min_count"] for item in scaled_bp["event_types"]}
ck("_scale_world_contract:exact 主角固定为 2，外围实体按原比例扩到总数 20",
   scaled_counts == {"player": 2, "boss": 7, "equipment": 11})
ck("_scale_world_contract:relation min_count 随实体规模同比放大且 0 保持可选",
   scaled_rel_min == {"equips": 2, "drops": 4, "optional_link": 0})
ck("_scale_world_contract:event min_count 随实体规模同比放大",
   scaled_event_min == {"defeat_boss": 4, "acquire_item": 6})
ck("_scale_world_contract:typed blueprint 与 legacy 镜像使用实际实体/session 总量",
   scaled_bp["temporal_model"]["n_sessions"] == 12
   and typed_wp["shared_world_spec"]["entities"]["count"] == 20
   and typed_wp["shared_world_spec"]["timeline"]["n_sessions"] == 12)

# 闭环发布必须把 01/02 当成同一事务：候选可变，但已发布合同快照不可被原地污染。
published_contract = deepcopy(typed_wp)
candidate_contract = deepcopy(typed_wp)
_scale_world_contract(candidate_contract, n_entities=18, n_sessions=9)
ck("闭环合同事务:候选扩容不污染此前已发布白皮书快照",
   published_contract["world_blueprint"]["temporal_model"]["n_sessions"] == 12
   and candidate_contract["world_blueprint"]["temporal_model"]["n_sessions"] == 9)

# 缩放目标来自闭环旋钮：小目标也必须能压回小世界；比例锚始终取首次白皮书。
_scale_world_contract(typed_wp, n_entities=7, n_sessions=6)
ck("_scale_world_contract:较小目标仍固定 exact 类型，外围按比例缩小且每类至少一个",
   sum(item["count"] for item in scaled_bp["entity_types"]) == 7
   and scaled_bp["entity_types"][0]["count"] == 2
   and all(item["count"] >= 1 for item in scaled_bp["entity_types"])
   and {item["id"]: item["min_count"] for item in scaled_bp["relation_types"]}
       == {"equips": 1, "drops": 2, "optional_link": 0}
   and {item["id"]: item["min_count"] for item in scaled_bp["event_types"]}
       == {"defeat_boss": 2, "acquire_item": 3}
   and scaled_bp["temporal_model"]["n_sessions"] == 6)

# target-owned static 标量 FK：未标 exact 的 primary 维持旧缩放行为。
reverse_owner_scale_wp = {
    "shared_world_spec": {},
    "world_blueprint": {
        "entity_types": [
            {"id": "source", "count": 9, "primary": True, "fields": [{"name": "标签", "kind": "text"}]},
            {"id": "target", "count": 1, "fields": [{"name": "来源", "kind": "reference"}]},
        ],
        "relation_types": [{
            "id": "points", "from_type": "source", "to_type": "target",
            "field": "来源", "temporal": False, "min_count": 1,
        }],
        "event_types": [],
        "temporal_model": {"n_sessions": 4},
    },
}
_scale_world_contract(reverse_owner_scale_wp, n_entities=15, n_sessions=4)
ck("_scale_world_contract:未标 exact 的 primary 仍按比例扩容",
   reverse_owner_scale_wp["world_blueprint"]["entity_types"][0]["count"] == 14
   and reverse_owner_scale_wp["world_blueprint"]["entity_types"][1]["count"] == 1
   and reverse_owner_scale_wp["world_blueprint"]["relation_types"][0]["min_count"] == 1)

# 只有未标 exact 的 primary 类型时仍保持历史扩容语义。
incremental_scale_wp = {
    "shared_world_spec": {},
    "world_blueprint": {
        "entity_types": [{"id": "x", "count": 10, "primary": True}],
        "relation_types": [],
        "event_types": [{"id": "tick", "min_count": 1}],
        "temporal_model": {"n_sessions": 4},
    },
}
for target_size in range(11, 16):
    _scale_world_contract(incremental_scale_wp, n_entities=target_size, n_sessions=4)
ck("_scale_world_contract:未标 exact 的单 primary 维持旧扩容语义且不累积 ceil",
   incremental_scale_wp["world_blueprint"]["entity_types"][0]["count"] == 15
   and incremental_scale_wp["world_blueprint"]["event_types"][0]["min_count"] == 2)

# 单一主人公显式 exact：闭环目标再大也不能复制第二个主角。
exact_character_wp = {
    "shared_world_spec": {},
    "world_blueprint": {
        "entity_types": [{"id": "player_character", "count": 1, "primary": True,
                          "cardinality_policy": "exact"}],
        "relation_types": [],
        "event_types": [{"id": "story_beat", "min_count": 1}],
        "temporal_model": {"n_sessions": 4},
    },
}
_scale_world_contract(exact_character_wp, n_entities=15, n_sessions=4, narrative=True)
ck("_scale_world_contract:exact 单主角不被闭环复制",
   exact_character_wp["world_blueprint"]["entity_types"][0]["count"] == 1
   and exact_character_wp["shared_world_spec"]["entities"]["count"] == 1
   and exact_character_wp["world_blueprint"]["event_types"][0]["min_count"] == 1)

event_capacity_wp = {
    "shared_world_spec": {},
    "world_blueprint": {
        "entity_types": [{"id": "player", "count": 1, "primary": True,
                          "cardinality_policy": "exact"}],
        "relation_types": [],
        "event_types": [{"id": "tick", "roles": {"actor": "player"},
                         "min_count": 8}],
        "temporal_model": {"n_sessions": 2},
    },
}
_scale_world_contract(event_capacity_wp, n_entities=100, n_sessions=2)
ck("_scale_world_contract:event min_count 不超过角色组合×session 的实例容量",
   event_capacity_wp["world_blueprint"]["event_types"][0]["min_count"] == 2)

l7_capacity_wp = {
    "domain_profile": {},
    "world_blueprint": {
        "entity_types": [
            {"id": "tool", "count": 1, "fields": [
                {"name": "success_rate", "kind": "numeric", "range": [0, 1]},
                {"name": "timeout_count", "kind": "numeric", "monotonic": "up"},
            ]},
            {"id": "invocation", "count": 9, "fields": [
                {"name": "retry_count", "kind": "numeric", "monotonic": "up"},
            ]},
        ],
        "relation_types": [],
        "event_types": [],
        "temporal_model": {"n_sessions": 10},
    },
}
l7_capacity = _ensure_l7_capacity(l7_capacity_wp, 4)
l7_counts = {item["id"]: item["count"]
             for item in l7_capacity_wp["world_blueprint"]["entity_types"]}
ck("_ensure_l7_capacity:按硬 floor 把自然趋势容量从 1 提到 4",
   l7_capacity == {"before": 1, "after": 4, "moved": {"tool": 3, "invocation": -3}})
ck("_ensure_l7_capacity:只重分配类型人口，总实体数不膨胀",
   l7_counts == {"tool": 4, "invocation": 6})
ck("_ensure_l7_capacity:累计数值不冒充可塑趋势，imprint 上限同步 floor",
   l7_capacity_wp["domain_profile"]["l7_max_trends"] == 4)

# ── _floor_status:达标/可行未达(growable)/不可行未达(permanent)/总数闸 ─────────
spec = TargetSpec(min_questions=20, per_line_min={"L1_timeline": 8, "L2_relational": 6, "L3_process": 5})

# (a) 全达标且总数够 → met
by_line_ok = {"L1_timeline": {"grounded": 9}, "L2_relational": {"grounded": 6}, "L3_process": {"grounded": 6}}
met, plf, grow, perm = _floor_status(by_line_ok, {"grounded": 21}, spec, {"L1_timeline", "L2_relational", "L3_process"})
ck("_floor_status (a) 全达标+总数够 → met=True", met and not grow and not perm)
ck("_floor_status (a) per_line_final 报实际接地数", plf == {"L1_timeline": 9, "L2_relational": 6, "L3_process": 6})

# (b) 可行线 L3 低于 floor → growable, 未达
by_line_b = {"L1_timeline": {"grounded": 9}, "L2_relational": {"grounded": 6}, "L3_process": {"grounded": 2}}
met, plf, grow, perm = _floor_status(by_line_b, {"grounded": 17}, spec, {"L1_timeline", "L2_relational", "L3_process"})
ck("_floor_status (b) 可行线未达 → growable 非空、met=False", (not met) and grow == ["L3_process:2/5"] and not perm)

# (c) L3 不可行(不在 feasible) 且未达 → permanent(不空转)
met, plf, grow, perm = _floor_status(by_line_b, {"grounded": 17}, spec, {"L1_timeline", "L2_relational"})
ck("_floor_status (c) 不可行线未达 → permanent 非空、不进 growable", (not met) and perm == ["L3_process:2/5"] and not grow)

# (d) 各线 floor 都满足但【总数】< min_questions → 仍未达(总数闸)
spec_hi = TargetSpec(min_questions=30, per_line_min={"L1_timeline": 8, "L2_relational": 6, "L3_process": 5})
met, plf, grow, perm = _floor_status(by_line_ok, {"grounded": 21}, spec_hi, {"L1_timeline", "L2_relational", "L3_process"})
ck("_floor_status (d) 逐线达但总数<min_q → met=False(总数闸)", not met)

# ── 控制流编排仿真:用合成供给表跑 driver 的循环骨架(不碰 LLM)─────────────────
# 合成"世界供给模型":给定 (n_entities, n_sessions) → 各线能产多少 order(单调随规模增)。
def synth_supply(p: WorldParams):
    return {
        "L1_timeline": p.n_sessions * 2,                       # L1 供给 ∝ 周数
        "L2_relational": p.n_entities // 2,                    # L2 ∝ 人员(≈实体/2)
        "L3_process": int(p.n_entities * 0.5),                 # L3 ∝ 实体
    }


def synth_ground(produced: dict, survival: dict):
    return {lid: {"grounded": int(n * survival.get(lid, 0.8)), "survival": survival.get(lid, 0.8)}
            for lid, n in produced.items()}


def drive_sim(spec: TargetSpec, true_survival: dict, feasible: set, max_rounds=2, order_subrounds=3):
    """复刻 build_to_target 的【决策骨架】(用合成供给替换真 LLM stage),返回 (status, rounds, last_by_line)。"""
    s = sum(spec.per_line_min.values())                       # 镜像 driver:总下限盈余摊进各线 floor
    if 0 < s < spec.min_questions:
        scale = spec.min_questions / s
        spec.per_line_min = {k: math.ceil(v * scale) for k, v in spec.per_line_min.items()}
    params = invert_rate(spec)
    by_line = {}
    for rnd in range(1, max_rounds + 1):
        quotas = dict(params.target_orders)
        # ① 供给环:供不上 → 长世界,多子轮。produced = min(世界可用, 配额)(忠实 enumerate:供给且配额双限)
        for sub in range(1, order_subrounds + 1):
            cap = synth_supply(params)
            produced = {k: min(cap[k], quotas[k]) for k in cap if k in feasible}
            deficit = _order_deficit(produced, quotas, feasible)
            if not deficit or sub == order_subrounds:
                break
            params = _grow_for_supply(params, deficit)
        # 渲+接地(合成):真实 survival 决定接地数
        by_line = synth_ground(produced, true_survival)
        overall = {"grounded": sum(v["grounded"] for v in by_line.values())}
        met, plf, grow, perm = _floor_status(by_line, overall, spec, feasible)
        if met:
            return "MET", rnd, by_line
        if perm:                                              # floor 落在不可行线 → 永远达不到,fail-open,不空转
            return "UNMET_permanent", rnd, by_line
        if rnd < max_rounds:                                  # ② 按实测 survival 放大
            measured = {lid: v["survival"] for lid, v in by_line.items()}
            params = invert_rate(spec, survival=measured, slack=DEFAULT_SLACK * 1.3)
            params = dataclasses.replace(params, n_entities=max(params.n_entities, 12), n_sessions=max(params.n_sessions, 10))
    return "UNMET_exhausted", max_rounds, by_line

# 场景1:survival 与保守先验一致(甚至更好)→ 一轮内 ①供给环补齐 → MET
# (min_q = Σfloor = 19,盈余摊派为 no-op,纯测控制流)
spec1 = TargetSpec(min_questions=19, per_line_min={"L1_timeline": 8, "L2_relational": 6, "L3_process": 5})
st1, r1, bl1 = drive_sim(spec1, {"L1_timeline": .9, "L2_relational": .9, "L3_process": .6},
                         {"L1_timeline", "L2_relational", "L3_process"})
ck("控制流①:供给充足+survival 不差 → MET", st1 == "MET")

# 场景2:第一轮 survival 比先验差很多(L3 真实0.25)→ ②纠偏放大后第二轮 MET
st2, r2, bl2 = drive_sim(spec1, {"L1_timeline": .9, "L2_relational": .9, "L3_process": .25},
                         {"L1_timeline", "L2_relational", "L3_process"}, max_rounds=3)
ck("控制流②:首轮 survival 偏低 → 纠偏环放大后达标(MET)", st2 == "MET")
ck("控制流②:纠偏确实多花了轮次(>1)", r2 > 1)

# 场景3:L3 设了 floor 但不可行(feasible 不含 L3)→ 永久 UNMET,且只跑 1 轮(不空转耗尽)
st3, r3, bl3 = drive_sim(spec1, {"L1_timeline": .9, "L2_relational": .9},
                         {"L1_timeline", "L2_relational"}, max_rounds=3)
ck("控制流③:floor 线不可行 → UNMET_permanent", st3 == "UNMET_permanent")
ck("控制流③:不可行 → 第1轮即收手,不空转到 max_rounds", r3 == 1)

# 场景4:floor 高到 clamp 也供不满(survival 极低)→ 耗尽 max_rounds 仍 UNMET(上层必须 fail-closed)
spec4 = TargetSpec(min_questions=200, per_line_min={"L1_timeline": 90, "L2_relational": 90})
st4, r4, bl4 = drive_sim(spec4, {"L1_timeline": .2, "L2_relational": .2},
                         {"L1_timeline", "L2_relational"}, max_rounds=2)
ck("控制流④:floor 远超 clamp 上限 → 有界返回 UNMET 供上层判失败",
   st4 == "UNMET_exhausted" and r4 == 2)

# 场景5:总下限盈余摊派 —— min_q 大于 Σfloor 时,摊派后 Σfloor' ≥ min_q(消掉"总数单独差"歧义态)
spec5 = TargetSpec(min_questions=30, per_line_min={"L1_timeline": 8, "L2_relational": 6, "L3_process": 5})  # Σ=19<30
_s = sum(spec5.per_line_min.values())
_scaled = {k: math.ceil(v * spec5.min_questions / _s) for k, v in spec5.per_line_min.items()}
ck("控制流⑤:盈余摊派后 Σfloor' ≥ min_questions(逐线达 ⟹ 总数达)", sum(_scaled.values()) >= spec5.min_questions)

# ── 汇总 ──────────────────────────────────────────────────────────────────────
npass = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    if not ok:
        print(f"  ✗ {name}")
print(f"[closed_loop self-test] {npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
