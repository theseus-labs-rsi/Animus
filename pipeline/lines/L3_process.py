"""
pipeline.lines.L3_process —— L3 过程序列产线(单能力线:L3_order)。

世界基质 = 状态机的【变更事件流】本身(免费:每个 UPDATE/EXPIRE/DELETE op = 一个事件,
gt_event_order 已按 (date,session) 排好真值时序)。这条线从 L1 接手原来临时挂着的 ORDER。
  点菜 = 每实体取 ≥3 个【跨字段·跨周】事件(信息量最大,不退化成单字段值流水);
  gt   = 用 gt_event_order 把选中卡片重排(可重算 → 过校验闸);
  出题 = 给【按字段名排序(非时序!)】的事件卡片,问发生先后(不泄漏时序、不给日期);
  评分 = Kendall-τ(序相关),非 EM —— scorer 写进 aux 供评测侧取用。
"""
from __future__ import annotations

from collections import Counter
from pipeline.lines.base import ProductionLine, Order
from pipeline.world_state import WorldState, gt_event_order, _norm, UPDATE, EXPIRE, DELETE

MIN_EVENTS = 3                                          # 一道排序题至少要 3 个事件才有料


def _value_history_count(ws: WorldState, ent: str) -> Counter:
    """(field, _norm(value)) → 该取值在实体【全值史·含初始 SET】出现几次 = 读者在语料里看得见它几次。
    >1 ⇒ "「字段」变为该值"无唯一可定位周(指代歧义)。
    ★这是 L3 "可定位"判据的【唯一真相源】:源头点菜(_locatable_events)与边A闸(well_posed I2)共用同一函数,
      杜绝两处定义漂移 —— run160053 Q31 正是源头与闸【都】只数变更流、【都】漏初始 SET → 双双漏放
      (值班 wk1 初设 3 / wk6 改回 3,读者两处看见 3,却没被判为复现)。计数域必须 = 读者真看得见的取值序列。"""
    vc: Counter = Counter()
    for fname, tl in ws.entities.get(ent, {}).items():
        for (_s, _d, v) in tl.set_values():             # set_values = 该字段所有 SET+UPDATE 取值(含初始 SET)
            if v:
                vc[(fname, _norm(v))] += 1
    return vc


def _locatable_events(ws: WorldState, ent: str) -> list[dict]:
    """该实体的变更事件,滤掉【复现值】(全值史里同(字段,值)出现 >1 次)→ 源头杜绝 ill-posed 排序。
    · 按 (field, value) 计数,不按 field:同字段不同值(报表数=25 / =15)各自唯一可定位,保留;
    · 停用事件(value 空)是终态、天然唯一,保留。"""
    vc = _value_history_count(ws, ent)                  # 与 well_posed I2 同一口径
    return [e for e in gt_event_order(ws, ent)          # 排序题只点 UPDATE/EXPIRE/DELETE
            if (not e.get("value")) or vc[(e["field"], _norm(e.get("value")))] == 1]


# ════════════════════════════════════════════════════════════════════════════
# 点菜枚举(从 L1_timeline._candidates 的 ORDER 分支接手,行为不变 + 修泄漏)
# ════════════════════════════════════════════════════════════════════════════
def enumerate_l3_orders(ws: WorldState, max_events: int = 4) -> list[Order]:
    """对每个实体取一组【跨字段·跨周】变更事件(≥3),烘焙真值时序 = 一道排序题。
    pass1:字段、周都唯一(信息量最大);不够再 pass2:放宽到仅跨周;取前 max_events 个。
    返回 L3_order 订单(gt=有序事件 list;aux.events=同一组卡片,gt() 据此从世界重排)。"""
    out: list[Order] = []
    for ent in ws.entities:
        events = _locatable_events(ws, ent)                # ★只从【唯一可定位】事件点菜(跳过复现值,源头杜绝 ill-posed 排序)
        picked, seen_sess, seen_field = [], set(), set()
        for e in events:                                   # pass1:字段、周都唯一
            if e["session"] not in seen_sess and e["field"] not in seen_field:
                picked.append(e); seen_sess.add(e["session"]); seen_field.add(e["field"])
        if len(picked) < MIN_EVENTS:                       # pass2:不够再放宽到仅跨周
            for e in events:
                if e["session"] not in seen_sess:
                    picked.append(e); seen_sess.add(e["session"])
        if len(picked) >= MIN_EVENTS:
            sub = sorted(picked, key=lambda e: (e["date"], e["session"]))[:max_events]
            out.append(Order("L3_order", ent, "", gt=sub,
                             evidence_sessions=sorted({e["session"] for e in sub}),
                             aux={"scorer": "kendall_tau", "n_fields": len({e["field"] for e in sub}),
                                  "events": [dict(e) for e in sub]}))
    return out


def _episode_orders(ws: WorldState, max_events: int = 4) -> list[Order]:
    """One disjoint card set per actual causal episode and focus entity.

    Causal roots come from the frozen world's existing ``caused_by`` edges.
    Every card must exactly match a canonical event effect and the unchanged
    timeline locator rules. Long episodes remain one task; no sliding windows
    or combinations of the same facts are manufactured to increase supply.
    """
    legacy = {order.entity: order for order in enumerate_l3_orders(ws, max_events)}
    event_rows = list(getattr(ws, "events", []) or [])
    events = {event.get("id"): event for event in event_rows if isinstance(event, dict) and event.get("id")}
    if len(events) != len(event_rows):
        return list(legacy.values())
    roots = {}
    for event_id in sorted(events):
        trail, current = set(), event_id
        while current in events and current not in trail:
            trail.add(current)
            parent = events[current].get("caused_by")
            if not parent:
                roots[event_id] = current
                break
            current = parent
    locators = {entity: {(card["field"], _norm(card.get("value")), card["session"]): card
                        for card in _locatable_events(ws, entity)} for entity in ws.entities}
    groups = {}
    for event_id in sorted(roots):
        event, root = events[event_id], roots[event_id]
        for effect in event.get("effects", []):
            if not isinstance(effect, dict):
                continue
            entity = effect.get("entity")
            locator = (effect.get("field"), _norm(effect.get("set", effect.get("value"))), event.get("session"))
            card = locators.get(entity, {}).get(locator)
            if card is not None:
                group = groups.setdefault((entity, root), {"cards": {}, "event_ids": set()})
                group["cards"][locator] = card
                group["event_ids"].add(event_id)
    pools, used = {}, set()
    for (entity, root), group in sorted(groups.items(), key=lambda item: (
            item[0][0], min(card["session"] for card in item[1]["cards"].values()), item[0][1])):
        cards = sorted(group["cards"].values(), key=lambda card: (card["date"], card["session"], card["field"]))
        picked, sessions, fields = [], set(), set()
        for card in cards:
            if card["session"] not in sessions and card["field"] not in fields:
                picked.append(card); sessions.add(card["session"]); fields.add(card["field"])
        if len(picked) < MIN_EVENTS:
            for card in cards:
                if card["session"] not in sessions:
                    picked.append(card); sessions.add(card["session"])
        picked = sorted(picked, key=lambda card: (card["date"], card["session"]))[:max_events]
        identities = {(entity, card["field"], _norm(card.get("value")), card["session"]) for card in picked}
        if len(picked) < MIN_EVENTS or used.intersection(identities):
            continue
        used.update(identities)
        pools.setdefault(entity, []).append(Order("L3_order", entity, "", gt=picked,
            evidence_sessions=sorted({card["session"] for card in picked}),
            aux={"scorer": "kendall_tau", "n_fields": len({card["field"] for card in picked}),
                 "events": [dict(card) for card in picked],
                 "supply_episode": {"version": 1, "basis": "frozen_causal_component",
                                    "root_event_id": root, "event_ids": sorted(group["event_ids"])}}))
    for entity, order in legacy.items():
        if entity not in pools:
            pools[entity] = [order]
    # Give each entity its first independent episode before its next one.
    return [pool[index] for index in range(max((len(pool) for pool in pools.values()), default=0))
            for _entity, pool in sorted(pools.items()) if index < len(pool)]


# ════════════════════════════════════════════════════════════════════════════
# L3 产线
# ════════════════════════════════════════════════════════════════════════════
class ProcessLine(ProductionLine):
    def matches_carrier(self, carrier, entities, support):
        events = (support.get('aux') or {}).get('events', [])
        return (support.get('entity') in entities and
                (not carrier.get('field') or any(e.get('field') == carrier['field'] for e in events)))

    def construction_spec(self):
        return {"process": {"minimum_distinct_change_periods": MIN_EVENTS,
                "focus": "one_entity_per_episode", "locator": "unique_field_value_in_complete_history",
                "initial_set_is_change": False}}

    def construction_issues(self, carrier, blueprint, objects, events, observations):
        from pipeline.world_agent import field_write_routes

        etypes = {e["id"]: e for e in blueprint["event_types"]}
        routes = field_write_routes(blueprint)
        periods = {}
        for event in events.values():
            for effect in etypes.get(event.get("type"), {}).get("effect_fields", []):
                entity = event.get("participants", {}).get(effect["role"])
                if event.get("session", 0) > 0:
                    periods.setdefault(entity, set()).add(event["session"])
        for observation in observations:
            entity, field = observation.get("entity"), observation.get("field")
            kind = objects.get(entity, {}).get("type")
            # Reading an event/relation-owned field does not create a change.
            # Intrinsic observations are only opportunities; actual distinct,
            # locatable changes remain the original line's world-time gate.
            if kind in routes and field in routes[kind]["intrinsic_fields"]:
                periods.setdefault(entity, set()).update(
                    s for s in observation.get("sessions", []) if s > 0)
        if not any(len(periods.get(entity, set())) >= MIN_EVENTS for entity in carrier["entities"]):
            return [{"code": "process_periods", "actual": {e: sorted(periods.get(e, set())) for e in carrier["entities"]},
                     "required": MIN_EVENTS, "message": "Allocate at least three distinct change periods on one focus entity"}]
        return []

    def value_issues(self, world, carrier):
        periods = {entity: sorted({e["session"] for e in _locatable_events(world, entity)})
                   for entity in carrier["entities"] if entity in world.entities}
        return [] if any(len(p) >= MIN_EVENTS for p in periods.values()) else [
            {"code": "process_locator", "actual": periods, "required": MIN_EVENTS,
             "message": "Use at least three distinct change periods on the entity; different fields may contribute. Initial SET is not a change, but its value participates in the uniqueness check against the full history. Repeated values cannot locate a unique change."}]
    id = "L3_process"
    title = "过程序列"
    memory = "跨字段事件的时序重建(把散落各文档的变更按发生先后排序)"
    gt_substrate = "状态机变更事件流 + gt_event_order 重排"
    implemented = True
    requires: list[str] = ["multi_event_timelines"]  # 需某实体携带 ≥3 个跨周变更事件,否则无序可排

    # prepare = 默认 no-op:复用共享世界,变更 op 本身就是事件日志(基质免费)。

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """看有没有实体能凑够 MIN_EVENTS 个跨周事件(口径同 enumerate:pass2 放宽后按去重 session 计)。
        基质免费(不需 prepare),但世界里的变更事件太少 → 这条线照样产 0 单。"""
        for ent in ws.entities:
            n = len({e["session"] for e in _locatable_events(ws, ent)})    # 口径同 enumerate:只数唯一可定位事件,防 feasible 说 yes 却产 0
            if n >= MIN_EVENTS:
                return True, f"实体「{ent}」有 {n} 个唯一可定位跨周事件(≥{MIN_EVENTS},可排序)"
        return False, f"无实体凑够 {MIN_EVENTS} 个唯一可定位跨周事件(复现值已剔除)"

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        delivery = isinstance(wp, dict) and "delivery_target" in wp
        if delivery:
            from pipeline.delivery_target import DeliveryTarget
            DeliveryTarget.from_dict(wp["delivery_target"])
        out = []
        if delivery and target <= 0:
            return out
        orders = _episode_orders(ws) if delivery else enumerate_l3_orders(ws)
        for o in orders:
            order = {"line": self.id, "capability": "L3_order", "entity": o.entity,
                     "field": o.field, "gt": o.gt, "evidence_sessions": o.evidence_sessions, "aux": o.aux}
            if delivery and self.well_posed(order, ws)[0] != "well_posed":
                continue
            out.append(order)
        return out[:target]

    def gt(self, ws, o: dict):
        """护城河:用 gt_event_order 重新从世界算出真值时序,再过滤到本题选中的卡片,
        保持时序 = enumerate 烘焙的有序 list(见自检校验闸)。不信 aux 里的日期,真从世界重算。"""
        if o.get("capability") == "L3_process_trace":
            from pipeline.process_proposals import validate_process_order
            return validate_process_order(o, ws)
        ent = o["entity"]
        cards = (o.get("aux") or {}).get("events") or o.get("gt") or []
        keys = {(c["field"], str(c.get("value")), c.get("session")) for c in cards}
        return [e for e in gt_event_order(ws, ent)
                if (e["field"], str(e.get("value")), e["session"]) in keys]

    def intent(self, o: dict) -> tuple[str, list]:
        if o.get("capability") == "L3_process_trace":
            return o["aux"]["process"]["intent"], []
        ent, gt = o.get("entity", ""), (o.get("gt") or [])
        # ★按【字段名】排序呈现(非时序!)防把正确顺序直接喂给模型;问的就是真时序
        shown = sorted(gt, key=lambda e: str(e.get("field", "")))
        evs = "、".join((f"「{e.get('field','')}」变为 {e.get('value')}" if e.get("value")
                         else f"「{e.get('field','')}」停止统计") for e in shown)
        s = f"请把【{ent}】发生的下列事件按发生时间先后排序：{evs}。"
        # 业务字段的日期值是事件身份的一部分，不能当成发生时间删掉。
        hide = [str(e["date"]) for e in gt if e.get("date") and str(e["date"]) not in evs]
        return s, hide

    # ── ★边 A 闸:良定义(docs/anchors/edge_a/L3_well_posed.md,已 QA)──
    def well_posed(self, order: dict, ws) -> tuple:
        """gold 是这道排序题在【世界】里唯一、合法、可复算的解吗?纯代码、零 LLM、不碰 corpus。
        序/存在性信 gt_event_order(ws,ent)(与 gt() 同源,非 aux 烘焙日期);I2 复现计数信全值史
        _value_history_count(含初始 SET,与源头 _locatable_events 同一函数)。I0–I6 见设计 §2/§3。
        现线唯一真高发病 = I2(同(字段,值)复现 → 指代不唯一);源头 _locatable_events 已挡,
        I2 现为防回归。I1/I3/I4/I5 实测 0 触发(纯兜底)。I2 按 (field,value) 判、不按 field 去重
        (同字段不同唯一值各自可定位,必放行 —— 反误杀守则 WP3)。"""
        if order.get("capability") == "L3_process_trace":
            from pipeline.process_proposals import validate_process_order
            try:
                validate_process_order(order, ws)
            except (ValueError, TypeError, KeyError) as exc:
                return ("drop", f"Typed process witness invalid: {exc}")
            return ("well_posed", "Structural witness only; natural task/reference await semantic review")
        ent = order.get("entity", "")
        E = order.get("gt") or []                          # 已是“有序事件 list”(呈现序 = gold 序)

        # ── I0 长度 ───────────────────────────────────────────────
        if len(E) < MIN_EVENTS:
            return ("drop", f"序列过短(n={len(E)}<{MIN_EVENTS}),无序可排")

        # ── 从世界重建真值事件链(唯一真相源)────────────────────
        W = gt_event_order(ws, ent)                        # 已按 (date,session) 升序
        if not W:
            return ("drop", f"实体「{ent}」世界里无任何变更事件")

        # real:(field,_norm(value),op) → [真实 change-session…];停用 value=None → 身份 (field,"",op)
        real: dict = {}
        pos_in_W: dict = {}                                # (field,_norm(value),op,session) → W 真序下标
        for idx, w in enumerate(W):
            k = (w["field"], _norm(w["value"]), w["op"])
            real.setdefault(k, []).append(w["session"])
            pos_in_W[(*k, w["session"])] = idx

        # ── 为 E 每个事件解析它对应的【唯一】真实 session(I1 存在性 + I2 唯一可定位)──
        vhist = _value_history_count(ws, ent)              # ★I2 口径=全值史(与源头 _locatable_events 同一函数,含初始 SET)
        resolved = []                                      # [(e, true_session, true_idx)]
        for e in E:
            is_stop = e.get("op") in (EXPIRE, DELETE) or e.get("value") in (None, "")
            op = e.get("op") or (EXPIRE if is_stop else UPDATE)
            k = (e.get("field"), "" if is_stop else _norm(e.get("value")), op)
            sessions = real.get(k)
            # I1 存在性:世界里压根没这次变更(悬空值 / 该字段无此停用 op)
            if not sessions:
                return ("drop", f"事件「{e.get('field')}={e.get('value')}」(op={op})"
                                f"世界无此变更(悬空值/物理不存在)")
            # I2 唯一可定位值:非停用值按【全值史·含初始 SET】计数(= vhist,与源头同口径);停用事件天然单次,按变更流判。
            #   run160053 Q31:旧实现只数变更流、漏初始 SET → "初值=X + 改回 X" 漏放。停用值=""走身份 (field,"",op)。
            n_loc = len(set(sessions)) if is_stop else vhist[(e.get("field"), _norm(e.get("value")))]
            if n_loc > 1:
                from pipeline.world_state import week_label
                tl = ws.entities.get(ent, {}).get(e.get("field"))
                wk = (sorted(set(sessions)) if is_stop else
                      sorted({_s for (_s, _d, v) in (tl.set_values() if tl else []) if _norm(v) == _norm(e.get("value"))}))
                return ("drop", f"事件「{e.get('field')}={e.get('value')}」无唯一可定位周"
                                f"(出现于周 {[week_label(s) for s in wk]},指代歧义)")
            s = sessions[0]
            resolved.append((e, s, pos_in_W[(*k, s)]))

        # ── I4 无平手:被选事件真实 (date,session) 两两不同(用 date+session,防跨字段同周不同日)──
        seen: dict = {}
        for (e, s, idx) in resolved:
            w = W[idx]; key = (w["date"], w["session"])
            if key in seen:
                return ("drop", f"事件「{e.get('field')}」与「{seen[key]}」同周({w['date']}),"
                                f"全序不唯一 → 非良定义")
            seen[key] = e.get("field")

        # ── I3 方向/顺序一致:gold 序(E)== 按真实时序下标升序 ──────
        true_order = [e for (e, s, idx) in sorted(resolved, key=lambda r: r[2])]
        if [id(x) for x in true_order] != [id(e) for e in E]:
            return ("drop", "gold 顺序与世界真序不一致(时序倒置)")

        # ── I5 停用位置自洽(I3 在停用上的投影;不写死“末位”,只给可读诊断 reason)──
        #   I3 通过后 true_order==E,故此环恒过;保留为独立断言,停用题被错排时给指向性 reason。
        true_pos = {id(e): i for i, e in enumerate(true_order)}
        for gold_i, e in enumerate(E):
            if e.get("op") in (EXPIRE, DELETE) or e.get("value") in (None, ""):
                true_i = true_pos[id(e)]
                if gold_i != true_i:
                    return ("drop", f"停用事件「{e.get('field')}」错位"
                                    f"(gold 第{gold_i}位,真实时序第{true_i}位)")

        # ── I6 护城河:gt() 重算 == 烘焙序(与既有“gt()重算==烘焙”校验闸合流)──
        if self.gt(ws, order) != E:
            return ("drop", "gt() 重算 ≠ 烘焙序(护城河漏:选择集编码不稳)")

        return ("well_posed", "")

    def ground(self, order, evidence_docs, all_signal_text=""):
        """接地(§G.5):gt 里【每个事件】的值都须就近实体、在各自 session 文档内;一个不接地 → 整题弃。
        停用(value 为空)事件 v0 不验值(留 v1)。"""
        if order.get("capability") == "L3_process_trace":
            return ("drop", "Typed process natural reference requires public semantic grounding")
        from pipeline.grounding import attributed
        ent = order.get("entity", "")
        by_sess = {}
        for d in evidence_docs:
            by_sess.setdefault(d["session"], []).append(d["content"])
        events = order.get("gt") or []
        for e in events:
            val, sess = e.get("value"), e.get("session")
            if val in (None, ""):
                continue
            docs = by_sess.get(sess) or [d["content"] for d in evidence_docs]
            if not attributed(val, ent, docs):
                return ("drop", f"事件「{e.get('field')}={val}」@s{sess} 未就近「{ent}」(幽灵/未渲)")
        return ("grounded", f"全部 {len(events)} 个事件就近「{ent}」")


# 自检:python -m pipeline.lines.L3_process
if __name__ == "__main__":
    import sys
    from pipeline.world_state import (build_demo_world, WorldState, Timeline, Op,
                                       SET, UPDATE, EXPIRE, _date_of)
    line = ProcessLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # demo 世界:AI工程部有 P0(1.8%@s1,2.8%@s3,停@s5)/Oncall(15@s2,9@s4)/负责人(李四@s2)多事件
    ws = build_demo_world()
    orders = line.enumerate(ws)
    ck("枚举出≥1 道排序题", len(orders) >= 1)
    ck("每题 gt 是 ≥3 有序事件", all(isinstance(o["gt"], list) and len(o["gt"]) >= 3 for o in orders))
    o0 = orders[0]
    seq = [(e["field"], e["session"]) for e in o0["gt"]]
    ck("gt 按 (date,session) 升序", seq == sorted(seq, key=lambda x: x[1]))
    ck("优先跨字段(>1 个不同字段)", o0["aux"]["n_fields"] >= 2)
    ck("跨周(证据≥3 周)", len(o0["evidence_sessions"]) >= 3)
    ck("scorer=kendall_tau", o0["aux"].get("scorer") == "kendall_tau")

    # ★护城河校验闸:gt() 重算 == enumerate 烘焙(L1 里 ORDER 进不了闸,L3 能)
    ck("校验闸:gt() 重算 == 烘焙", all(line.gt(ws, o) == o["gt"] for o in orders))

    # 出题不泄漏:题面不含任何日期、且事件【非时序】呈现(防直接抄答案顺序)
    intent, hide = line.intent(o0)
    ck("intent 隐藏了日期", all(d in hide for d in [e["date"] for e in o0["gt"]]))
    ck("题面不含日期串", all(d not in intent for d in [e["date"] for e in o0["gt"]]))
    # 呈现顺序 = 字段名序,应 ≠ 真时序(除非巧合,用一个保证错位的世界再验)
    rng_ws = WorldState({"X": {
        "甲字段": Timeline([Op(0, "2025-01-06", SET, "a0", None), Op(3, "2025-01-27", UPDATE, "a1", "a0")]),
        "乙字段": Timeline([Op(0, "2025-01-06", SET, "b0", None), Op(1, "2025-01-13", UPDATE, "b1", "b0")]),
        "丙字段": Timeline([Op(0, "2025-01-06", SET, "c0", None), Op(2, "2025-01-20", UPDATE, "c1", "c0")]),
    }}, n_sessions=4)
    ro = line.enumerate(rng_ws)[0]
    # 真时序:乙@s1 → 丙@s2 → 甲@s3;题面按【字段名 Unicode 序】呈现 → 与真时序错位 = 不泄漏
    intent_str = line.intent(ro)[0]
    shown_order = sorted(["甲字段", "乙字段", "丙字段"], key=lambda f: intent_str.index(f))
    true_order = [e["field"] for e in ro["gt"]]
    ck("题面呈现序(字段名序)≠ 真时序", shown_order != true_order)
    ck("错位世界也过校验闸", line.gt(rng_ws, ro) == ro["gt"])

    # ★源头修复:复现值(同(字段,值)出现>1次)必须被 enumerate 剔除(无唯一可定位周 → ill-posed)
    recur_ws = WorldState({"Y": {
        "值班":  Timeline([Op(0, "2025-01-06", SET, "8", None), Op(1, "2025-01-13", UPDATE, "9", "8"),
                          Op(2, "2025-01-20", UPDATE, "7", "9"), Op(4, "2025-02-03", UPDATE, "9", "7")]),  # 值=9 复现(s1,s4)
        "缺陷率": Timeline([Op(0, "2025-01-06", SET, "1%", None), Op(3, "2025-01-27", UPDATE, "2%", "1%")]),
        "负责人": Timeline([Op(0, "2025-01-06", SET, "张三", None), Op(5, "2025-02-10", UPDATE, "李四", "张三")]),
    }}, n_sessions=6)
    rec_orders = line.enumerate(recur_ws)
    rec_vals = [(e["field"], str(e.get("value"))) for o in rec_orders for e in o["gt"]]
    ck("剔除复现值后仍凑够 3 唯一事件、产出题", len(rec_orders) >= 1)
    ck("复现值(值班=9)被源头剔除、不进 gt", ("值班", "9") not in rec_vals)
    ck("唯一值(值班=7)保留", ("值班", "7") in rec_vals)
    ck("剔除后仍过校验闸", all(line.gt(recur_ws, o) == o["gt"] for o in rec_orders))

    # ════════════════════════════════════════════════════════════════════════
    # ★边 A 闸 well_posed() 自检(设计 §7;离线、零 LLM、零 corpus)
    #   well/ill 各例手算期望,断言 (status, reason 命中)。守 WP2 中游停用放行、
    #   WP3 同字段异唯一值放行(反"字段去重"误杀)、IP5 复现值 drop(证 I2 真在工作)。
    # ════════════════════════════════════════════════════════════════════════
    _d = lambda s: _date_of(s)                              # 同 enumerate/world 日历(2025-01-06 + 7d/session)

    def _W(field_ops, n=10):                                # {field: [(s,op,val,prev),...]}
        return WorldState({"E": {f: Timeline([Op(s, _d(s), op, v, p) for (s, op, v, p) in steps])
                                 for f, steps in field_ops.items()}}, n_sessions=n)

    def _ord(events):                                       # 手构 L3_order(绕过 enumerate,直塞病态序)
        return {"line": "L3_process", "capability": "L3_order", "entity": "E", "field": "",
                "gt": [dict(e) for e in events],
                "evidence_sessions": sorted({e["session"] for e in events}),
                "aux": {"scorer": "kendall_tau", "events": [dict(e) for e in events]}}

    def _ev(field, value, s, op=UPDATE):                    # 事件卡(date 由 session 派生,与世界同尺)
        return {"field": field, "value": value, "session": s, "date": _d(s), "op": op}

    # WP1 ✔well:三跨字段唯一事件(P0@s1, SLA@s3, Onc@s4)
    wp1 = _W({"P0":  [(0, SET, "2.5", None), (1, UPDATE, "1.8", "2.5")],
              "SLA": [(0, SET, "99", None), (3, UPDATE, "98", "99")],
              "Onc": [(0, SET, "8", None), (4, UPDATE, "9", "8")]})
    o1 = line.enumerate(wp1)[0]
    ck("WP1 well-posed(三唯一事件全过)", line.well_posed(o1, wp1) == ("well_posed", ""))

    # WP2 ✔well(中游停用):SLA 在 s3 EXPIRE,夹在 P0@s1 与 Onc@s4 之间 → I5 必放行(反"末位特判")
    wp2 = _W({"P0":  [(0, SET, "2.5", None), (1, UPDATE, "1.8", "2.5")],
              "SLA": [(0, SET, "99", None), (3, EXPIRE, None, "99")],
              "Onc": [(0, SET, "8", None), (4, UPDATE, "9", "8")]})
    o2 = line.enumerate(wp2)[0]
    ck("WP2 中游停用必放行(不写死末位)", line.well_posed(o2, wp2)[0] == "well_posed")
    ck("WP2 选中集确含中游 EXPIRE", any(e.get("op") == EXPIRE for e in o2["gt"]))

    # WP3 ✔well(同字段重复 + 异唯一值):报表=25@s2/=15@s4、准确率=92@s3/=88@s5 → 反"字段去重"误杀
    wp3 = _W({"报表":  [(0, SET, "20", None), (2, UPDATE, "25", "20"), (4, UPDATE, "15", "25")],
              "准确率": [(0, SET, "90", None), (3, UPDATE, "92", "90"), (5, UPDATE, "88", "92")]})
    o3 = _ord([_ev("报表", "25", 2), _ev("准确率", "92", 3), _ev("报表", "15", 4), _ev("准确率", "88", 5)])
    ck("WP3 同字段异唯一值必放行(反字段去重误杀)", line.well_posed(o3, wp3) == ("well_posed", ""))

    # IP1 ✗悬空值(Q42):世界 8→9→7 无 5,题面排"值班变为 5" → I1
    ip1_ws = _W({"值班":   [(0, SET, "8", None), (1, UPDATE, "9", "8"), (2, UPDATE, "7", "9")],
                 "缺陷率": [(0, SET, "1", None), (3, UPDATE, "2", "1")],
                 "负责人": [(0, SET, "张三", None), (4, UPDATE, "李四", "张三")]})
    ip1 = _ord([_ev("值班", "5", 1), _ev("缺陷率", "2", 3), _ev("负责人", "李四", 4)])
    st, why = line.well_posed(ip1, ip1_ws)
    ck("IP1 悬空值 drop(I1)", st == "drop")
    ck("IP1 reason 命中『世界无此变更』", "世界无此变更" in why)

    # IP2 ✗倒序(Q38 前):停用@s9 被排到首位,真实最末 → I3(I5 给诊断)
    ip2_ws = _W({"P0":  [(0, SET, "1", None), (1, UPDATE, "2", "1")],
                 "SLA": [(0, SET, "99", None), (9, EXPIRE, None, "99")],
                 "宕机": [(0, SET, "30", None), (2, UPDATE, "40", "30")]})
    ip2 = _ord([_ev("SLA", None, 9, op=EXPIRE), _ev("P0", "2", 1), _ev("宕机", "40", 2)])  # 故意倒序
    st, why = line.well_posed(ip2, ip2_ws)
    ck("IP2 时序倒置 drop(I3/I5)", st == "drop")
    ck("IP2 reason 命中倒置/错位", ("倒置" in why) or ("错位" in why))

    # IP3 ✗无该停用(Q38 后):SLA 全程仅 SET/UPDATE,题面塞"SLA 停统计" → I1
    ip3_ws = _W({"SLA": [(0, SET, "99", None), (2, UPDATE, "98", "99")],
                 "P0":  [(0, SET, "1", None), (1, UPDATE, "2", "1")],
                 "宕机": [(0, SET, "30", None), (4, UPDATE, "40", "30")]})
    ip3 = _ord([_ev("P0", "2", 1), _ev("SLA", None, 5, op=EXPIRE), _ev("宕机", "40", 4)])
    st, why = line.well_posed(ip3, ip3_ws)
    ck("IP3 无该停用 drop(I1)", st == "drop" and "世界无此变更" in why)

    # IP4 ✗平手:A、B 同在 s2 变更 → 全序不唯一 → I4
    ip4_ws = _W({"A": [(0, SET, "a0", None), (2, UPDATE, "a1", "a0")],
                 "B": [(0, SET, "b0", None), (2, UPDATE, "b1", "b0")],
                 "C": [(0, SET, "c0", None), (4, UPDATE, "c1", "c0")]})
    ip4 = _ord([_ev("A", "a1", 2), _ev("B", "b1", 2), _ev("C", "c1", 4)])
    st, why = line.well_posed(ip4, ip4_ws)
    ck("IP4 同周平手 drop(I4)", st == "drop")
    ck("IP4 reason 命中『全序不唯一』", "全序不唯一" in why)

    # IP5 ✗值复现指代不唯一(★现线唯一真高发):值班 8→9(s1)→7(s2)→9(s3),"变为 9"两落点 → I2
    ip5_ws = _W({"值班":   [(0, SET, "8", None), (1, UPDATE, "9", "8"), (2, UPDATE, "7", "9"), (3, UPDATE, "9", "7")],
                 "缺陷率": [(0, SET, "1", None), (4, UPDATE, "2", "1")],
                 "负责人": [(0, SET, "张三", None), (5, UPDATE, "李四", "张三")]})
    ip5 = _ord([_ev("值班", "9", 1), _ev("缺陷率", "2", 4), _ev("负责人", "李四", 5)])
    st, why = line.well_posed(ip5, ip5_ws)
    ck("IP5 复现值 drop(I2,证 I2 真工作非只堵字面5)", st == "drop")
    ck("IP5 reason 命中『无唯一可定位周』", "无唯一可定位周" in why)

    # IP6 ✗【初值复现】(run160053 Q31:IP5 漏的那类):值班 wk0 初设=3 → s1 改 5 → s6 又改回 3。
    #   "变为 3" 在初设(s0)与 s6 两处可落 → 指代不唯一。★IP5 用两个 UPDATE(都在变更流里),
    #   IP6 故意让复现值【等于初始 SET】—— 旧实现只数变更流、漏初始 SET → 源头与闸【双双漏放】(这正是 Q31)。
    ip6_ws = _W({"值班":   [(0, SET, "3", None), (1, UPDATE, "5", "3"), (6, UPDATE, "3", "5")],   # 3 = 初值,s6 改回
                 "缺陷率": [(0, SET, "1", None), (4, UPDATE, "2", "1")],
                 "负责人": [(0, SET, "张三", None), (5, UPDATE, "李四", "张三")]})
    # ① 源头:enumerate 必须剔掉"值班=3"(初值复现),不进任何 gt
    ip6_orders = line.enumerate(ip6_ws)
    ip6_vals = [(e["field"], str(e.get("value"))) for o in ip6_orders for e in o["gt"]]
    ck("IP6 源头:初值复现『值班=3』被 enumerate 剔除", ("值班", "3") not in ip6_vals)
    ck("IP6 源头:唯一值『值班=5』保留", ("值班", "5") in ip6_vals)
    # ② 闸:手塞含"值班→3@s6"的坏序,well_posed 必 drop(证闸与源头同口径、不再漏放)
    ip6 = _ord([_ev("值班", "3", 6), _ev("缺陷率", "2", 4), _ev("负责人", "李四", 5)])
    st, why = line.well_posed(ip6, ip6_ws)
    ck("IP6 闸:初值复现 drop(I2 全值史口径,堵 Q31)", st == "drop")
    ck("IP6 reason 命中『无唯一可定位周』且列出初设周", "无唯一可定位周" in why)

    # I0:长度过短 drop
    ip0 = _ord([_ev("P0", "1.8", 1), _ev("SLA", "98", 3)])
    ck("I0 序列过短 drop", line.well_posed(ip0, wp1)[0] == "drop")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L3_process self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
