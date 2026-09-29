"""
pipeline.lines.L8_transition —— L8 状态机转移产线(单能力线:L8_next）。

世界基质 = 白皮书 domain_profile.state_machines 声明的【单向状态序】字段
(如 工单状态:新建→处理中→已解决→关闭;案件状态:立案→…→结案）。世界已强制该字段沿声明序
单调推进(world_state.validate 的 illegal_transition 闸:倒流/出界算缺陷、定向重修）。
  点菜 = 每个状态机字段 × 每个【当前非终态】实体 → 一道"阶段序紧邻后继"题;
  gt   = 声明序里【当前态的后继】(代码从 states 索引算,过校验闸）;
  出题 = 不给当前态名、不在题面列完整状态表 → 结合实体记录与公开流程说明查找紧邻后继;
  能力 = 当前状态记忆 + 公开阶段规则应用，不宣称仅由稀疏轨迹归纳完整流程。
         实际记录允许跳级；阶段序的紧邻后继不等于预测下一次实际更新。

v1 只做【下一合法状态】(值答案,套现有判分,零协议改动）;
【非法转移声明】(判定答案"成立/不成立",牵动答案协议 + judge)留 v1.5。
"""
from __future__ import annotations

from pipeline.lines.base import ProductionLine
from pipeline.world_state import WorldState, _norm


def _state_machines(profile: dict) -> dict:
    """白皮书声明的状态机:{field: [按推进顺序的状态名…]}。
    只取【≥2 状态、且 _norm 后无重复】的合法声明(与 world_state.validate 同口径,防 idx 塌缩误判倒流）。"""
    sm: dict[str, list] = {}
    for m in (profile or {}).get("state_machines") or []:
        if not isinstance(m, dict):
            continue
        f, sts = m.get("field"), m.get("states")
        if f and isinstance(sts, list) and len(sts) >= 2:
            sts = [str(x) for x in sts]
            if len({_norm(x) for x in sts}) == len(sts):
                sm[f] = sts
    return sm


def _current_state(ws: WorldState, ent: str, field: str):
    """实体在该状态机字段的【当前态】= latest_valid(与 carry-forward 同口径)。无则 None。"""
    tl = ws.entities.get(ent, {}).get(field)
    return tl.latest_valid() if tl else None


def _has_typed_rules(ws) -> bool:
    blueprint = getattr(ws, "world_blueprint", None) or {}
    return bool(blueprint.get("entity_types") and not blueprint.get("legacy_adapter"))


def _entity_state_machines(ws, entity: str, profile=None) -> dict:
    """Typed rules keep their owner; only legacy worlds use the flat mirror."""
    if not _has_typed_rules(ws):
        return _state_machines(profile or {})
    entity_type = (getattr(ws, "entity_types", None) or {}).get(entity)
    declarations = [item for item in ws.world_blueprint["entity_types"]
                    if item.get("id") == entity_type]
    if len(declarations) != 1:
        return {}
    return _state_machines({"state_machines": [
        {"field": field.get("name"), "states": field.get("states")}
        for field in declarations[0].get("fields", []) if field.get("kind") == "status"]})


def _order_states(ws, order):
    if _has_typed_rules(ws):
        return _entity_state_machines(ws, order.get("entity", "")).get(order.get("field"), [])
    return [str(value) for value in (order.get("aux") or {}).get("states", [])]


class TransitionLine(ProductionLine):
    id = "L8_transition"
    title = "状态机转移"
    memory = "当前状态记忆与公开阶段顺序应用(定位当前态，再找序列中紧邻的后一阶段)"
    gt_substrate = "白皮书 state_machines 声明序 + 实体当前态索引(代码算后继)"
    implemented = True
    requires: list[str] = ["state_machine_fields"]   # 需 ≥1 状态机字段且有【非终态】实体
    # ★结构原型:基质在(feasible)就自动激活,不靠议会把它列进 active_lines。
    #   "场景结构分叉"应由【世界有没有状态机】决定,而非 LLM 判定 → 杜绝盲审病根①的同质化判断层。
    auto_activate = True

    # prepare = 默认 no-op:状态机字段本就在共享世界里(world stage 已按声明序约束推进),基质免费。

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """有状态机字段、且至少一个实体当前处于【非终态】(有唯一后继)→ 可激活。
        这条 feasible 正是【世界结构分叉】的开关：无状态序声明就直接 False，不为出题补造状态机。"""
        has_rules = False
        for ent, flds in ws.entities.items():
            for f, sts in _entity_state_machines(ws, ent, profile).items():
                has_rules = True
                last, allset = _norm(sts[-1]), {_norm(x) for x in sts}
                if f in flds:
                    cur = _current_state(ws, ent, f)
                    if cur and _norm(cur) in allset and _norm(cur) != last:
                        return (True, f"状态机「{f}」有非终态实体「{ent}」(当前={cur})可问下一阶段")
        if not has_rules:
            return (False, "白皮书未声明适用于实体的阶段顺序→ 本线不激活")
        return (False, "有状态机字段,但无【当前处于非终态】的实体(都已到终态/出界)")

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        profile = (wp or {}).get("domain_profile", {}) if isinstance(wp, dict) else {}
        out: list[dict] = []
        for ent, flds in ws.entities.items():
            for f, sts in _entity_state_machines(ws, ent, profile).items():
                idx_of = {_norm(s): i for i, s in enumerate(sts)}
                if f not in flds:
                    continue
                cur = _current_state(ws, ent, f)
                if not cur:
                    continue
                i = idx_of.get(_norm(cur))
                if i is None or i >= len(sts) - 1:        # 出界 / 已终态 → 无唯一后继,跳过
                    continue
                gold = sts[i + 1]
                tl = flds.get(f)                          # 当前态被设到哪个 session(证据:系统得先找到 X 当前阶段)
                sv = [s for (s, _d, v) in (tl.set_values() if tl else []) if _norm(v) == _norm(cur)]
                ev = max(sv) if sv else None
                out.append({"line": self.id, "capability": "L8_next", "entity": ent,
                            "field": f, "gt": gold,
                            "evidence_sessions": [ev] if ev is not None else [],
                            "aux": {"states": sts, "cur_state": cur, "cur_idx": i}})
                if len(out) >= target:
                    return out
        return out

    def gt(self, ws, o: dict):
        """护城河:从声明序 + 实体【当前态】重算后继(不信 aux.gt,真从世界取当前态再索引)。"""
        sts = _order_states(ws, o)
        cur = _current_state(ws, o.get("entity", ""), o.get("field", ""))
        if not sts or not cur:
            return None if _has_typed_rules(ws) else o.get("gt")
        idx_of = {_norm(s): i for i, s in enumerate(sts)}
        i = idx_of.get(_norm(cur))
        if i is None or i >= len(sts) - 1:
            return o.get("gt")
        return sts[i + 1]

    def well_posed(self, order: dict, ws) -> tuple:
        """边 A 闸:后继是该实体"下一合法阶段"在世界里【唯一、合法、可复算】的解吗?纯代码、零 LLM、不碰 corpus。
        I0 声明合法(≥2 态、_norm 无重复)/ I1 当前态在声明序内 / I2 当前态非终态(有唯一后继)/
        I3 gold==声明序后继 / I4 护城河 gt() 重算==烘焙。"""
        aux = order.get("aux") or {}
        ent, fld = order.get("entity", ""), order.get("field", "")
        sts = _order_states(ws, order)
        if _has_typed_rules(ws) and list(aux.get("states") or []) != sts:
            return ("drop", "订单阶段顺序与该实体类型的冻结定义不一致")
        # I0 声明合法
        if len(sts) < 2 or len({_norm(x) for x in sts}) != len(sts):
            return ("drop", f"状态机声明非法(<2 态或 _norm 重复):{sts}")
        idx_of = {_norm(s): i for i, s in enumerate(sts)}
        cur = _current_state(ws, ent, fld)
        # I1 当前态在声明序内
        if not cur or _norm(cur) not in idx_of:
            return ("drop", f"实体「{ent}」当前态 {cur!r} 不在状态机声明序内(出界/无值)")
        i = idx_of[_norm(cur)]
        # I2 当前态非终态
        if i >= len(sts) - 1:
            return ("drop", f"当前态 {cur!r} 已是终态,无唯一【下一阶段】")
        # I3 gold == 声明序后继
        gold = sts[i + 1]
        if _norm(order.get("gt")) != _norm(gold):
            return ("drop", f"gold {order.get('gt')!r} ≠ 声明序后继 {gold!r}")
        # I4 护城河:gt() 重算 == 烘焙
        if _norm(self.gt(ws, order)) != _norm(gold):
            return ("drop", "gt() 重算 ≠ 烘焙后继(护城河漏)")
        return ("well_posed", "")

    def ground(self, order, evidence_docs, all_signal_text=""):
        """接地:gold(下一阶段)是【预测性结构答案】、X 未必到达 → 不验 gold 在不在语料;
        改验【X 的当前态】就近归属(系统得先在语料里找到 X 现处何阶段,才能推下一步)。"""
        from pipeline.grounding import attributed
        ent = order.get("entity", "")
        cur = (order.get("aux") or {}).get("cur_state")
        if not cur:
            return ("drop", "fail-closed:无 cur_state 可验")
        if attributed(cur, ent, [d["content"] for d in evidence_docs]):
            return ("grounded", f"X 当前态「{cur}」就近归属「{ent}」(可据此推下一阶段)")
        return ("drop", f"X 当前态「{cur}」未就近归属「{ent}」(系统无从知道 X 现处何阶段)")

    def intent(self, o: dict) -> tuple[str, list]:
        ent, fld, gold = o.get("entity", ""), o.get("field", ""), o.get("gt", "")
        s = (f"根据记录确定【{ent}】的「{fld}」所处阶段，再对照公开的适用阶段顺序，"
             f"在该顺序中紧邻这个阶段的后一阶段是什么？"
             f"问的是阶段序中的相邻位置，不是预测下一次实际状态变更。"
             f"★只回一个阶段名;不要回当前阶段、也不要把整条流程列出来。")
        return s, [str(gold)]                              # 后继(答案)别泄漏进题面


# 自检:python -m pipeline.lines.L8_transition
if __name__ == "__main__":
    import sys
    from pipeline.world_state import Timeline, Op, SET, UPDATE, _date_of

    line = TransitionLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    STATES = ["新建", "处理中", "已解决", "关闭"]
    WP = {"domain_profile": {"state_machines": [{"field": "工单状态", "states": STATES}]}}
    _d = lambda s: _date_of(s)

    def _tl(steps):     # steps = [(session, value)],首个为 SET 其余 UPDATE
        ops = []
        for k, (s, v) in enumerate(steps):
            ops.append(Op(s, _d(s), SET if k == 0 else UPDATE, v, steps[k - 1][1] if k else None))
        return Timeline(ops)

    ws = WorldState({
        "工单A": {"工单状态": _tl([(0, "新建"), (2, "处理中")])},            # 当前=处理中(idx1)→ 后继=已解决
        "工单B": {"工单状态": _tl([(0, "新建"), (1, "处理中"), (3, "已解决"), (5, "关闭")])},  # 当前=关闭(终态)→ 无题
        "工单C": {"工单状态": _tl([(0, "新建")])},                          # 当前=新建(idx0)→ 后继=处理中
    }, n_sessions=6)

    # ── feasible ──
    ok, why = line.feasible(ws, WP["domain_profile"])
    ck("feasible: 有状态机+非终态实体 → True", ok)
    ck("feasible: 无 state_machines → False(场景分叉开关)", not line.feasible(ws, {})[0])

    # ── enumerate ──
    orders = line.enumerate(ws, wp=WP)
    by_ent = {o["entity"]: o for o in orders}
    ck("枚举:终态实体(工单B)不出题", "工单B" not in by_ent)
    ck("枚举:工单A 后继=已解决", by_ent.get("工单A", {}).get("gt") == "已解决")
    ck("枚举:工单C 后继=处理中", by_ent.get("工单C", {}).get("gt") == "处理中")
    ck("枚举:capability=L8_next", all(o["capability"] == "L8_next" for o in orders))
    ck("枚举:证据指向当前态所在 session", by_ent.get("工单A", {}).get("evidence_sessions") == [2])

    # ── 护城河:gt() 重算 == 烘焙 ──
    ck("校验闸:gt() 重算 == 烘焙", all(_norm(line.gt(ws, o)) == _norm(o["gt"]) for o in orders))

    # ── well_posed ──
    ck("well_posed: 工单A 合法 → 过", line.well_posed(by_ent["工单A"], ws) == ("well_posed", ""))
    # 手构 ill-posed:I2 终态(把工单B 硬塞成题)
    badB = {"line": "L8_transition", "capability": "L8_next", "entity": "工单B", "field": "工单状态",
            "gt": "关闭", "aux": {"states": STATES, "cur_state": "关闭", "cur_idx": 3}}
    st, why = line.well_posed(badB, ws)
    ck("well_posed: 终态实体 drop(I2)", st == "drop" and "终态" in why)
    # I1 当前态出界(实体当前态不在声明序)
    ws2 = WorldState({"工单X": {"工单状态": _tl([(0, "新建"), (1, "已归档")])}}, n_sessions=3)  # 已归档不在声明序
    badX = {"entity": "工单X", "field": "工单状态", "gt": "处理中",
            "aux": {"states": STATES, "cur_state": "已归档", "cur_idx": 0}}
    ck("well_posed: 当前态出界 drop(I1)", line.well_posed(badX, ws2)[0] == "drop")
    # I3 gold 锚错(后继填错)
    badGold = dict(by_ent["工单A"]); badGold["gt"] = "关闭"
    ck("well_posed: gold≠声明序后继 drop(I3)", line.well_posed(badGold, ws)[0] == "drop")

    # ── ground ──
    docs_ok = [{"content": "工单A 本期工单状态:处理中,负责客服小林。", "session": 2}]
    docs_no = [{"content": "工单A 客户满意度 4.6。", "session": 2}]
    ck("ground: 当前态就近 → grounded", line.ground(by_ent["工单A"], docs_ok)[0] == "grounded")
    ck("ground: 当前态未现 → drop", line.ground(by_ent["工单A"], docs_no)[0] == "drop")

    # ── intent:不泄漏后继(答案),不列完整状态表 ──
    intent, hide = line.intent(by_ent["工单A"])
    ck("intent: 隐藏后继答案", "已解决" in hide)
    ck("intent: 题面不含后继(答案)", "已解决" not in intent)
    ck("intent: 题面不列完整状态表(不泄漏顺序)", not all(s in intent for s in STATES))

    npass = sum(1 for ok_, _ in checks if ok_)
    for ok_, name in checks:
        if not ok_:
            print(f"  ✗ {name}")
    print(f"[L8_transition self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
