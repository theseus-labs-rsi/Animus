"""
pipeline.lines.L1_timeline —— L1 时间线产线。

多能力捆绑线:IE/KU/TR/MR/FORGET/DURATION/PREEXPIRE/ABS。点菜枚举(_candidates)+ 按 capability 派发的 gt()/intent()。
宪法 note:ORDER 已拆给 L3_process;CONFLICT 已下线(冲突归 L5_conflict 的跨来源裁决——L1 旧 CONFLICT 只是"同字段跨周出现过≠值"= 合法演化、非矛盾,易与 L5 混,故退役);ABS(拒答)保留在 L1。本线收敛为纯"时点/最新/聚合/时长 + 拒答"。
"""
from __future__ import annotations
from datetime import datetime, timedelta
import random

from pipeline.lines.base import ProductionLine, Order, field_kind, interrogative, sample_field_value
from pipeline.world_state import (
    WorldState, _norm, _to_num, _date_of, week_label,
    SET, UPDATE, DELETE, EXPIRE, INVALID, INSUFFICIENT,
    gt_ie, gt_ku, gt_mr, gt_forget, gt_pre_expire, gt_duration,
    stable_query_weeks,
)


def _after(date: str, days: int = 5) -> str:
    return (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════════════════════
# 点菜枚举(从 order_gen._candidates / generate_orders 逐字搬迁,行为不变)
# ════════════════════════════════════════════════════════════════════════════
def _candidates(ws: WorldState, cap: str, wp=None) -> list[Order]:
    cap = cap.upper()
    out: list[Order] = []

    # ABS:不依赖具体 entity,从 absent_fields 取
    if cap == "ABS":
        for f in ws.absent_fields:
            out.append(Order("ABS", "", f, gt=INSUFFICIENT, evidence_sessions=[]))
        return out

    for ent, flds in ws.entities.items():
        for fld, tl in flds.items():
            sv = tl.set_values()                       # [(session, date, value)] 时间序
            sessions = [s for (s, _, _) in sv]
            distinct = {_norm(v) for (_, _, v) in sv}

            if cap == "IE":
                # 时点切片:某【过去周】X 是多少。mention-on-change 下取"非最新"的中间值
                # → 第 N 周没复述该值, 必须回忆最近一次变更, 且不能用最新值(考 recency 抗性)
                latest = tl.latest_valid()
                for (N, cs, v) in stable_query_weeks(ws, ent, fld):
                    if _norm(v) != _norm(latest):
                        out.append(Order("IE", ent, fld, gt={"value": v, "at_week": N},
                                         question_date=ws.date_of_session(N),
                                         evidence_sessions=list(range(cs, N + 1)),
                                         aux={"at_week": N, "value": v, "framing": "decision_time"}))

            elif cap == "MR":
                from pipeline.value_types import ValueComparisonError, comparison_keys, field_schema
                try:
                    schema = field_schema(ws, ent, fld, wp)
                    value_kind, _ = comparison_keys([v for _, _, v in sv], schema)
                except ValueComparisonError:
                    continue
                if len(sv) >= 2:
                    for agg in ("max", "min"):
                        out.append(Order("MR", ent, fld, gt=gt_mr(ws, ent, fld, agg, schema=schema),
                                         evidence_sessions=sessions,
                                         aux={"agg": agg, "value_schema": schema,
                                              "ans_kind": value_kind}))

            elif cap == "TR":
                chs = [o for o in tl.change_ops() if o.op in (SET, UPDATE) and o.prev is not None]
                if chs:
                    o = chs[0]                          # ★ 只取首次变化(消除"变了多次"的歧义)
                    prior = [s for s in sessions if s < o.session]
                    if prior:
                        out.append(Order("TR", ent, fld,
                                         # ★FixA:gold 不 ship 裸 0-based session(下游会拿它格式化"第N周"→off-by-one);
                                         #   只留 week(1-based,人类口径)+ date。内部锚 session 由 evidence_sessions / week-1 派生。
                                         gt={"week": week_label(o.session), "date": o.date,
                                             "from": o.prev, "to": o.value, "ordinal": "首次"},
                                         evidence_sessions=sorted(set(prior + [o.session])),
                                         aux={"to_value": o.value, "from_value": o.prev, "ordinal": "首次"}))

            elif cap == "KU":
                # 最新值:取最后一次变更【之后】的稳定周问"现在是多少"(那几周没复述 → 必须跨周回忆)
                latest = tl.latest_valid()
                if latest not in (INVALID, INSUFFICIENT) and len(distinct) >= 2:  # ★ 真更新过才算 KU
                    sq = [(N, cs) for (N, cs, v) in stable_query_weeks(ws, ent, fld)
                          if _norm(v) == _norm(latest)]
                    if sq:
                        N, cs = sq[-1]                  # 最后一个稳定周 = 真·最新参照点
                        out.append(Order("KU", ent, fld, gt=latest, question_date=ws.date_of_session(N),
                                         evidence_sessions=list(range(cs, N + 1)), aux={"at_week": N}))

            elif cap == "FORGET":
                exp = [o for o in tl._sorted() if o.op in (DELETE, EXPIRE)]
                if exp:
                    e0 = exp[0]
                    qd = _after(e0.date)
                    prior = [s for s in sessions if s < e0.session]
                    out.append(Order("FORGET", ent, fld, gt=gt_forget(ws, ent, fld, qd),
                                     question_date=qd,
                                     evidence_sessions=sorted(set(prior + [e0.session]))))

            elif cap == "DURATION":
                # 某值持续几周(选"被取代过"的值 = 有明确起止, ≥2 周才有意义)
                for (_s, _d, v) in sv:
                    dr = gt_duration(ws, ent, fld, v)
                    if isinstance(dr, dict) and dr.get("weeks", 0) >= 2 and dr["end"] < (ws.n_sessions or 99):
                        end_op = next(op for op in tl._sorted() if op.session == dr["end"])
                        out.append(Order("DURATION", ent, fld, gt=dr,
                                         evidence_sessions=list(range(dr["start"], dr["end"] + 1)),  # ★全跨度:含中间周,让模型能确认"没变过"
                                         aux={"value": v, "duration_end_op": end_op.op,
                                              "duration_end_value": end_op.value}))
                        break

            elif cap == "PREEXPIRE":
                # 复合 FORGET→IE:已停统计字段在停掉【前】的最后值
                pe = gt_pre_expire(ws, ent, fld)
                if isinstance(pe, dict) and pe.get("value"):
                    es = pe["expire_session"]
                    prior = [s for s in sessions if s < es]
                    if prior:
                        out.append(Order("PREEXPIRE", ent, fld, gt=pe["value"],
                                         evidence_sessions=sorted(set(prior + [es])),
                                         aux={"expire_session": es}))
    return out


# ════════════════════════════════════════════════════════════════════════════
# L1 产线
# ════════════════════════════════════════════════════════════════════════════
class TimelineLine(ProductionLine):
    id = "L1_timeline"
    title = "时间线"
    memory = "时点切片/最新值/何时变/数值聚合/时长"
    gt_substrate = "状态机 + state-diff 切片"
    implemented = True
    requires: list[str] = []                 # 根线:只需基础世界(随时间演化的字段),恒可行
    # feasible 继承基类默认(恒 True)。

    # ★配额驱动(取代写死 _PLAN(29)):这是各能力的【配比权重】、不是硬上限;
    #   enumerate 按权重把 `target` 配额分到各能力(每能力不超其候选供给),不足从有余的能力补。
    #   (closed_loop_targetspec_design §7:L1 配额由 TargetSpec.invert_rate 反推,白皮书=被执行的合同。)
    _CAP_WEIGHT = {"IE": 6, "KU": 6, "TR": 5, "MR": 4, "FORGET": 3, "PREEXPIRE": 2, "ABS": 3}

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        """按 `_CAP_WEIGHT` 比例把 `target` 配额分到各能力(供给受限处从余量补);target = L1 配额
        (反推来;缺省 200 = 尽量全产)。耗尽仍不足 target = 世界供给到顶 → run_lines 据此判 'short' 去长世界。"""
        rng = random.Random(0)
        cands = {c: _candidates(ws, c, wp) for c in self._CAP_WEIGHT}
        for c in cands:
            rng.shuffle(cands[c])
        total_w = sum(self._CAP_WEIGHT.values())
        caps_by_w = [c for c, _ in sorted(self._CAP_WEIGHT.items(), key=lambda kv: -kv[1])]  # ★按权重降序遍历:
        taken = {c: 0 for c in cands}                          #   小 target(<能力数)被 [:target] 截断时保住【高权重】能力,与 dict 序解耦
        picked: list = []
        for c in caps_by_w:                                    # 第一轮:按权重比例分配
            n_c = min(len(cands[c]), max(1, round(target * self._CAP_WEIGHT[c] / total_w)))
            picked += cands[c][:n_c]
            taken[c] = n_c
        while len(picked) < target:                            # 不足 target:从仍有余的能力轮取补(仍高权重优先)
            progressed = False
            for c in caps_by_w:
                if len(picked) >= target:
                    break
                if taken[c] < len(cands[c]):
                    picked.append(cands[c][taken[c]]); taken[c] += 1; progressed = True
            if not progressed:
                break                                          # 全耗尽 = 世界供给上限
        profile = (wp or {}).get("domain_profile", {})         # ★Fix2:stamp 答案类型(疑问词单一真源)
        out = []
        for o in picked[:target]:
            aux = dict(o.aux)
            from pipeline.value_types import field_schema
            declaration = field_schema(ws, o.entity, o.field, wp)
            if declaration:
                aux["value_schema"] = declaration
            aux.setdefault("ans_kind", declaration.get("kind") or field_kind(o.field, sample_field_value(ws, o.entity, o.field), profile))
            aux["time_unit"] = ws.period_unit()
            out.append({"line": self.id, "capability": o.capability, "entity": o.entity,
                        "entity_type": ws.entity_types.get(o.entity), "field": o.field,
                        "gt": o.gt, "evidence_sessions": o.evidence_sessions, "aux": aux})
        return out

    def gt(self, ws, o: dict):
        """护城河:按 capability 派发,用代码重算答案(与 enumerate 烘焙逐字段相等,见自检的校验闸)。"""
        cap, ent, fld, aux = o["capability"], o["entity"], o["field"], (o.get("aux") or {})
        if cap == "IE":
            N = aux.get("at_week")
            return {"value": gt_ie(ws, ent, fld, N), "at_week": N}
        if cap == "KU":
            return gt_ku(ws, ent, fld)
        if cap == "MR":
            return gt_mr(ws, ent, fld, aux.get("agg", "max"), schema=aux.get("value_schema"))
        if cap == "DURATION":
            return gt_duration(ws, ent, fld, aux.get("value"))
        if cap == "TR":
            tl = ws.timeline(ent, fld)
            chs = [op for op in tl.change_ops() if op.op in (SET, UPDATE) and op.prev is not None] if tl else []
            if chs:
                op = chs[0]
                return {"week": week_label(op.session), "date": op.date,
                        "from": op.prev, "to": op.value, "ordinal": "首次"}    # ★FixA:无裸 session,week=1-based 唯一周表示
            return INSUFFICIENT
        if cap == "FORGET":
            qd = o.get("question_date")          # 题面 dict 不保留 question_date;缺则回放烘焙值(不强行重算)
            return gt_forget(ws, ent, fld, qd) if qd else o.get("gt")
        if cap == "PREEXPIRE":
            pe = gt_pre_expire(ws, ent, fld)
            return pe["value"] if isinstance(pe, dict) else pe
        if cap == "ABS":
            return INSUFFICIENT
        return o.get("gt")                 # 兜底:enumerate 烘焙的即真值

    def intent(self, o: dict) -> tuple[str, list]:
        cap, ent, fld, aux, gt = (o.get("capability"), o.get("entity", ""), o.get("field", ""),
                                  o.get("aux", {}) or {}, o.get("gt"))
        scalar = gt.get("value") if isinstance(gt, dict) else gt
        hide = [str(scalar)] if scalar is not None else []
        q = interrogative(aux.get("ans_kind"))     # ★Fix2:疑问词由字段 kind 派生(person→是谁/number→是多少/其它→是什么)
        unit = aux.get("time_unit") or "周"
        if cap == "IE":
            s = f"第 {week_label(aux.get('at_week'))} {unit}时，【{ent}】的「{fld}」{q}？"
        elif cap == "KU":
            s = f"截至最新一期，【{ent}】的「{fld}」{q}？"
        elif cap == "TR":
            s = f"【{ent}】的「{fld}」首次发生变化是在第几{unit}？请回答期数。"
        elif cap == "MR":
            if aux.get("ans_kind") == "date" or (aux.get("value_schema") or {}).get("kind") == "date":
                s = f"在全部记录中，【{ent}】的「{fld}」{'最晚' if aux.get('agg') == 'max' else '最早'}日期是哪一天？"
            else:
                s = f"在全部记录中，【{ent}】的「{fld}」{'最大' if aux.get('agg') == 'max' else '最小'}值是多少？"
        elif cap == "PREEXPIRE":
            s = f"【{ent}】的「{fld}」在停止统计前最后一次记录的值{q}？"
        elif cap == "FORGET":
            s = f"截至最新一期，【{ent}】的「{fld}」{q}？"
        elif cap == "ABS":
            subject = f"【{ent}】" if ent else "现有记录"
            s = f"{subject}的「{fld}」{q}？"
        elif cap == "DURATION":
            s = f"【{ent}】的「{fld}」保持为「{aux.get('value')}」持续了多少{unit}？"
            hide = []
        else:
            s = f"问 {ent} 的「{fld}」是多少。"
        return s, hide


    def ground(self, order, evidence_docs, all_signal_text=""):
        """接地(§G.5/§G.12):IE/KU/MR/TR/PREEXPIRE 值就近实体(MR/TR 锚到 gt.session);
        FORGET 验停用标记就近;ABS 反向验字段全局缺失。"""
        from pipeline.grounding import gold_scalar, attributed, STOP_MARKERS
        from pipeline.world_state import _norm
        cap, ent = order.get("capability"), order.get("entity", "")
        docs = [d["content"] for d in evidence_docs]
        if cap == "DURATION":
            gt, aux = order.get("gt") or {}, order.get("aux") or {}
            start_docs = [d["content"] for d in evidence_docs if d.get("session") == gt.get("start")]
            end_docs = [d["content"] for d in evidence_docs if d.get("session") == gt.get("end")]
            if not attributed(aux.get("value"), ent, start_docs):
                return "drop", "DURATION 起点目标值无正文见证"
            if aux.get("duration_end_op") in (DELETE, EXPIRE):
                ended = any(attributed(marker, ent, end_docs) for marker in STOP_MARKERS)
            else:
                ended = attributed(aux.get("duration_end_value"), ent, end_docs)
            return ("grounded", "DURATION 起止事件均有正文见证") if ended else ("drop", "DURATION 终点取代/停用事件无正文见证")
        if cap == "ABS":                                 # 反向:字段全语料不出现 = 缺失为真
            fld = order.get("field", "")
            if fld and _norm(fld) in _norm(all_signal_text):
                return ("drop", f"ABS 不成立:字段「{fld}」在语料里出现了(并非缺失)")
            return ("grounded", f"ABS:字段「{fld}」全语料未现(缺失为真)")
        if cap == "FORGET":                              # order 无 expire 周字段 → 扫任一证据文档找停用标记
            for d in docs:
                nd = _norm(d)
                if _norm(ent) in nd and any(_norm(m) in nd for m in STOP_MARKERS):
                    return ("grounded", f"FORGET:「{ent}」+停用标记就近")
            return ("drop", f"FORGET:无停用标记就近「{ent}」")
        if cap in ("MR", "TR"):                          # 锚到 gt 标注的那一 session
            gt = order.get("gt")
            sess = gt.get("session") if isinstance(gt, dict) else None       # MR 仍带 session
            if sess is None and isinstance(gt, dict) and gt.get("week") is not None:
                sess = gt["week"] - 1                    # ★FixA:TR 已去裸 session,由 week(1-based)-1 派生内部锚
            if sess is not None:
                docs = [d["content"] for d in evidence_docs if d["session"] == sess] or docs
        s = gold_scalar(order)
        if not s:
            return ("drop", f"fail-closed:{cap} 无待验标量")
        if attributed(s, ent, docs):
            return ("grounded", f"「{cap}」'{s}' 就近「{ent}」")
        return ("drop", f"「{cap}」'{s}' 未就近「{ent}」")

    # ── ★边 A 闸(良定义 well_posed)· 详设 docs/anchors/edge_a/L1_well_posed.md(已 QA 版)──
    def well_posed(self, order: dict, ws) -> tuple:
        """gold 是该订单意图在【世界】里【唯一、合法、可复算】的答案吗?是 → ('well_posed','');否 → ('drop',reason)。
        纯代码、确定性、零 LLM、绝不碰 corpus;只用 order 字段 + ws,复用产线 gt_* 同口径函数。
        实现 W0/W1/W2/W3/W4(降级·IE-only)/W6/W7/W8/W9;W5(极值平手)实测证伪已撤下、不实现。"""
        cap = order.get("capability")
        ent, fld = order.get("entity", ""), order.get("field", "")
        gt, aux = order.get("gt"), (order.get("aux") or {})
        ev = order.get("evidence_sessions") or []

        def _eq(a, b) -> bool:
            from pipeline.value_types import field_schema, values_equal, ValueComparisonError
            try:
                declaration = field_schema(ws, ent, fld) or aux.get("value_schema") or {}
                return values_equal(a, b, declaration)
            except ValueComparisonError:
                return False

        def _is_sentinel(v) -> bool:
            return v in (INVALID, INSUFFICIENT, None, "") or _norm(v) == ""

        # ── W0 结构完整(ABS 不需 entity/timeline)──
        if cap != "ABS":
            tl = ws.timeline(ent, fld)
            if tl is None:
                return ("drop", f"well_posed:结构缺失:{ent}.{fld} 世界无此时间线")

        # ── IE —— 某周(session)生效值 ──
        if cap == "IE":
            N = aux.get("at_week")
            if N is None or not isinstance(N, int):                          # W4 结构:周锚缺失/非整数(fail-closed)
                return ("drop", "well_posed:IE 题面无周锚(at_week 缺失/非整数)")
            if not (0 <= N < (ws.n_sessions or 10**9)):                      # W4 结构:越界(实测 0 触发,纯兜底)
                return ("drop", f"well_posed:IE 周锚越界:at_week={N} 不在 [0,{ws.n_sessions})")
            g = gt.get("value") if isinstance(gt, dict) else None            # W3 类型:IE gt 必须 {value,at_week}
            if g is None:
                return ("drop", f"well_posed:IE 类型错配:gt 非 {{value,at_week}},gt={gt}")
            r = gt_ie(ws, ent, fld, N)                                       # = tl.value_at_session(N)
            if _is_sentinel(r):                                              # W2:该周无合法值(fail-closed)
                return ("drop", f"well_posed:IE gold 悬空:第{N}周(session)处值={r}")
            if not _eq(r, g):                                               # W1 ★主力:重算==烘焙
                return ("drop", f"well_posed:IE gt 与世界不符:重算={r} 烘焙={g} @session{N}")
            if gt.get("at_week") != N:                                      # W4:gt 内周锚与 aux 同源(单一真源残留校验)
                return ("drop", f"well_posed:IE 周锚不一致:gt.at_week={gt.get('at_week')} aux.at_week={N}")
            if N not in ev:                                                 # W9:锚周可读(fail-closed)
                return ("drop", f"well_posed:IE 证据缺锚:session{N} 不在 evidence_sessions")
            return ("well_posed", "")

        # ── KU —— 最新有效值(无显式周锚;不查 at_week)──
        if cap == "KU":
            r = gt_ku(ws, ent, fld)                                          # = tl.latest_valid()
            if _is_sentinel(r):                                              # W2:最新已失效/无 → 应走 FORGET
                return ("drop", f"well_posed:KU gold 悬空:最新值={r}(应走 FORGET 或字段无效)")
            if isinstance(gt, dict):                                         # W3:KU gt 必须是纯标量
                return ("drop", f"well_posed:KU 类型错配:gt 应为单值标量,得 dict={gt}")
            if not _eq(r, gt):                                              # W1
                return ("drop", f"well_posed:KU gt 与世界不符:重算={r} 烘焙={gt}")
            distinct = {_norm(v) for (_, _, v) in ws.timeline(ent, fld).set_values()}
            if len(distinct) < 2:                                          # 复核"真更新过"(否则 trivial,非记忆题)
                return ("drop", f"well_posed:KU 退化:{ent}.{fld} 全程单值,无'更新'语义")
            return ("well_posed", "")

        # ── MR —— 极值(治 Q19;★不查平手 W5,实测证伪撤下)──
        if cap == "MR":
            agg = aux.get("agg", "max")
            if agg not in ("max", "min"):
                return ("drop", f"well_posed:MR agg 非法:{agg}")
            if not isinstance(gt, dict) or "value" not in gt or "session" not in gt:  # W3
                return ("drop", f"well_posed:MR 类型错配:gt 应为 {{value,session,...}},得 {gt}")
            r = gt_mr(ws, ent, fld, agg, schema=aux.get("value_schema"))
            if _is_sentinel(r.get("value")):                               # W2
                return ("drop", "well_posed:MR gold 悬空:无数值候选")
            if not (_eq(r["value"], gt["value"]) and r["session"] == gt["session"]
                    and r.get("agg") == gt.get("agg")):                     # W1 ★主力:治 Q19(陈旧/错算极值)
                return ("drop", f"well_posed:MR gt 与世界不符:重算={r} 烘焙={gt}")
            # ── W5(极值平手)撤下:实测 10.3% 误杀 / 0 真覆盖 —— 题面问"值"非"哪周",平手时值仍唯一。
            if gt["session"] not in ev:                                    # W9:证据可读(fail-closed)
                return ("drop", f"well_posed:MR 证据缺锚:极值 session{gt['session']} 不在 evidence")
            return ("well_posed", "")

        if cap == "DURATION":
            target = aux.get("value")
            r = gt_duration(ws, ent, fld, target)
            if not isinstance(gt, dict) or not isinstance(r, dict):
                return "drop", "well_posed:DURATION 目标值或结构缺失"
            if any(not isinstance(gt.get(key), int) or isinstance(gt.get(key), bool) for key in ("start", "end", "weeks")):
                return "drop", "well_posed:DURATION 起止与时长必须是整数"
            if gt != r or r["weeks"] < 2 or not (0 <= r["start"] < r["end"] < ws.n_sessions):
                return "drop", "well_posed:DURATION 起止/时长与世界不符或区间未闭合"
            occurrences = [op for op in tl._sorted() if op.op in (SET, UPDATE) and _eq(op.value, target)]
            if len(occurrences) != 1:
                return "drop", "well_posed:DURATION 目标值出现多个区间，题面未消歧"
            end_ops = [op for op in tl._sorted() if op.session == r["end"]]
            if len(end_ops) != 1 or end_ops[0].op not in (UPDATE, SET, DELETE, EXPIRE):
                return "drop", "well_posed:DURATION 终点事件不唯一或不受支持"
            end = end_ops[0]
            if aux.get("duration_end_op") != end.op or aux.get("duration_end_value") != end.value:
                return "drop", "well_posed:DURATION 终点见证与世界不符"
            if not set(range(r["start"], r["end"] + 1)).issubset(ev):
                return "drop", "well_posed:DURATION 证据未覆盖完整起止区间"
            return "well_posed", ""

        # ── TR —— 首次变更(哪一周)──
        if cap == "TR":
            tl = ws.timeline(ent, fld)
            chs = [op for op in tl.change_ops() if op.op in (SET, UPDATE) and op.prev is not None]
            if not chs:                                                    # W6:无变更
                return ("drop", f"well_posed:TR 首次变更不成立:{ent}.{fld} 无 SET/UPDATE 变更")
            op = chs[0]
            if not isinstance(gt, dict) or "week" not in gt or "to" not in gt:  # W3(FixA:周字段是 week,1-based)
                return ("drop", f"well_posed:TR 类型错配:gt 应为 {{week,to,...}},得 {gt}")
            if not (op.session == gt["week"] - 1 and _eq(op.value, gt["to"]) and _eq(op.prev, gt.get("from"))):  # W1(week-1=内部session)
                return ("drop", f"well_posed:TR gt 与世界不符:首变={op} 烘焙={gt}")
            prior = [s for (s, _, _) in tl.set_values() if s < op.session]
            if not prior:                                                  # W6:无"变化前"周 → "首次变化"无参照
                return ("drop", f"well_posed:TR 首变前无周:session{op.session} 之前无该字段值")
            if op.session not in ev:                                       # W9
                return ("drop", f"well_posed:TR 证据缺锚:首变 session{op.session} 不在 evidence")
            return ("well_posed", "")

        # ── FORGET —— 遗忘/停用(时效良定义是关键;order dict 无 question_date → 锚 latest)──
        if cap == "FORGET":
            tl = ws.timeline(ent, fld)
            has_stop = any(op.op in (DELETE, EXPIRE) for op in tl.ops)
            if not has_stop:                                               # W7:世界根本没停用 → FORGET 不成立
                return ("drop", f"well_posed:FORGET 停用不成立:{ent}.{fld} 无 EXPIRE/DELETE op")
            if not (isinstance(gt, dict) and "forgotten" in gt):           # W3
                return ("drop", f"well_posed:FORGET 类型错配:gt 应含 forgotten,得 {gt}")
            qd = order.get("question_date")                                # 当前 enumerate 不落进 dict → 退化锚 latest
            if qd:                                                         # W7:切片时点确已停
                v = tl.value_at(qd)
                if (v == INVALID) != bool(gt.get("forgotten")):
                    return ("drop", f"well_posed:FORGET 时效不符:@{qd} 切片={'停' if v == INVALID else '在'} gt.forgotten={gt.get('forgotten')}")
            else:                                                          # 无 question_date:锚"截至最新"(与 intent 对齐)
                lv = tl.latest_valid()                                     # 治 run190554 停-复活型坏 FORGET(W1 抓不到)
                if (lv == INVALID) != bool(gt.get("forgotten")):
                    return ("drop", f"well_posed:FORGET 时效不符:最新={'失效' if lv == INVALID else '有效'} gt.forgotten={gt.get('forgotten')}")
            return ("well_posed", "")

        # ── PREEXPIRE —— 过期前值 ──
        if cap == "PREEXPIRE":
            pe = gt_pre_expire(ws, ent, fld)                               # {value, expire_session} 或 INSUFFICIENT
            if not isinstance(pe, dict) or _is_sentinel(pe.get("value")):  # W7 + W2
                return ("drop", f"well_posed:PREEXPIRE 停用前无值:{ent}.{fld} 无 EXPIRE 或停前值空")
            if isinstance(gt, dict):                                       # W3:停前值是单标量
                return ("drop", f"well_posed:PREEXPIRE 类型错配:gt 应为单值,得 dict={gt}")
            if not _eq(pe["value"], gt):                                  # W1
                return ("drop", f"well_posed:PREEXPIRE gt 与世界不符:停前={pe['value']} 烘焙={gt}")
            es = aux.get("expire_session", pe["expire_session"])
            if es not in ev:                                              # W9
                return ("drop", f"well_posed:PREEXPIRE 证据缺锚:expire session{es} 不在 evidence")
            return ("well_posed", "")

        # ── ABS —— 缺失(拒答):对世界判(边 B 才对 corpus 判)──
        if cap == "ABS":
            if ws.has_field(fld):                                          # W8:世界里其实有这个字段 → 拒答不成立
                return ("drop", f"well_posed:ABS 不成立:字段「{fld}」在世界存在(非缺失)")
            if _norm(gt) != _norm(INSUFFICIENT):                          # W3:ABS gold 必须是拒答哨兵
                return ("drop", f"well_posed:ABS 类型错配:gt 应为 {INSUFFICIENT},得 {gt}")
            return ("well_posed", "")

        return ("drop", f"well_posed:未知能力 {cap}(fail-closed)")


# 自检:python -m pipeline.lines.L1_timeline
if __name__ == "__main__":
    import sys
    from pipeline.world_state import build_demo_world
    ws = build_demo_world()
    line = TimelineLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    by_cap = {c: _candidates(ws, c) for c in
              ["IE", "MR", "TR", "KU", "FORGET", "ABS", "DURATION", "PREEXPIRE"]}
    for c in by_cap:
        ck(f"{c} 有候选订单", len(by_cap[c]) > 0)
    for c, orders in by_cap.items():
        if c == "ABS":
            continue
        ck(f"{c} 全跨文档(≥2 session)", all(len(o.evidence_sessions) >= 2 for o in orders))

    ku = {o.field: o.gt for o in by_cap["KU"]}
    ck("KU 负责人=李四", _norm(ku.get("负责人")) == _norm("李四"))
    ck("KU Oncall=9次", _norm(ku.get("Oncall数")) == _norm("9次"))
    ck("KU 不含已 EXPIRE 的 P0缺陷率", "P0缺陷率" not in ku)
    ie_zfz = [o for o in by_cap["IE"] if o.field == "负责人"]
    ck("IE 负责人取到中间值张三", any(_norm(o.gt["value"]) == _norm("张三") for o in ie_zfz))
    ck("IE 不取最新值李四", all(_norm(o.gt["value"]) != _norm("李四") for o in ie_zfz))
    pe = {o.field: o.gt for o in by_cap["PREEXPIRE"]}
    ck("PREEXPIRE P0停前最后=2.8%", _norm(pe.get("P0缺陷率")) == _norm("2.8%"))
    mr = {(o.field, o.aux["agg"]): o.gt for o in by_cap["MR"]}
    ck("MR max Oncall=15次@s2", _norm(mr[("Oncall数", "max")]["value"]) == _norm("15次") and mr[("Oncall数", "max")]["session"] == 2)
    ck("MR min Oncall=9次@s4", _norm(mr[("Oncall数", "min")]["value"]) == _norm("9次") and mr[("Oncall数", "min")]["session"] == 4)
    tr = {o.aux["to_value"]: o for o in by_cap["TR"] if o.field == "负责人"}
    ck("TR 负责人→李四 @第3周(session2→week_label=3)", "李四" in tr and tr["李四"].gt["week"] == 3)
    # ★Fix1 时间单一真源:TR 答案是"哪一周",gt.week 必须是 week_label(session)(1-based,与语料同源),不是裸 session
    # ★FixA:TR gold 只 ship 1-based week + date,【绝不 ship 裸 0-based session】(下游会拿它格式化"第N周"→off-by-one)
    ck("TR gold 无裸 session 键(杜绝下游误读为周)", all("session" not in o.gt for o in by_cap["TR"]))
    ck("TR gold.week 为 1-based 整数", all(isinstance(o.gt.get("week"), int) and o.gt["week"] >= 1 for o in by_cap["TR"]))
    forget = {o.field: o for o in by_cap["FORGET"]}
    ck("FORGET P0 已遗忘", "P0缺陷率" in forget and forget["P0缺陷率"].gt.get("forgotten") is True)

    # ★护城河校验闸:gt() 重算 == enumerate 烘焙(可重算的能力;FORGET 需 question_date、ORDER 选择编码进答案 → 不入闸)
    RECOMP = {"IE", "KU", "MR", "TR", "PREEXPIRE", "ABS"}
    orders = line.enumerate(ws)
    ck("校验闸:gt() 重算 == 烘焙(7 能力)",
       all(line.gt(ws, o) == o["gt"] for o in orders if o["capability"] in RECOMP))

    # ★Fix2:疑问词由 ans_kind 派生(person→是谁/number→是多少/其它→是什么),不写死
    def _q_of(cap, kind):
        return line.intent({"capability": cap, "entity": "X", "field": "某字段", "gt": "v",
                            "aux": {"ans_kind": kind, "at_week": 1}})[0]
    ck("Fix2:IE+person → 题面'是谁'", "是谁" in _q_of("IE", "person"))
    ck("Fix2:IE+numeric(真实schema词表)→ '是多少'且非'是谁'", "是多少" in _q_of("IE", "numeric") and "是谁" not in _q_of("IE", "numeric"))
    ck("Fix2:KU+status → 题面'是什么'", "是什么" in _q_of("KU", "status"))
    ck("Fix2:enumerate 给每单 stamp ans_kind", all("ans_kind" in o["aux"] for o in orders))

    # ════════════════════════════════════════════════════════════════════════
    # ★边 A 闸 well_posed 自检(详设 §7;含撤下 W5 不再误杀 / 停-复活 drop / IE 越界不一致)
    # ════════════════════════════════════════════════════════════════════════
    from pipeline.world_state import Timeline, Op, WorldState as _WS, SET as _S, UPDATE as _U, EXPIRE as _E, INSUFFICIENT as _INS

    def ckwp(name, order, want_status, want_kw=""):
        st, rs = line.well_posed(order, ws)
        ok = (st == want_status) and (want_kw in rs)
        if not ok:
            print(f"  ✗ [wp] {name}: got=({st!r},{rs!r}) want=({want_status},kw={want_kw!r})")
        checks.append((ok, f"[wp] {name}"))

    E = "AI工程部"
    # ── well-posed(应 PASS)──
    ckwp("§7.1 IE 负责人@s0=张三",
         {"capability": "IE", "entity": E, "field": "负责人", "gt": {"value": "张三", "at_week": 0},
          "aux": {"at_week": 0}, "evidence_sessions": [0, 1]}, "well_posed")
    ckwp("§7.2 MR Oncall max=15@s2",
         {"capability": "MR", "entity": E, "field": "Oncall数", "gt": {"value": "15次", "session": 2, "agg": "max"},
          "aux": {"agg": "max"}, "evidence_sessions": [0, 2, 4]}, "well_posed")
    ckwp("§7.3 PREEXPIRE P0 停前=2.8%",
         {"capability": "PREEXPIRE", "entity": E, "field": "P0缺陷率", "gt": "2.8%",
          "aux": {"expire_session": 5}, "evidence_sessions": [3, 5]}, "well_posed")

    # ★§7.4 MR 平手不再误杀(撤下 W5 的反例):Oncall 10→15→15(s1=s2=15 并列),max gt 锚 s1
    tie = _WS({"T": {"f": Timeline([Op(0, "2025-01-06", _S, "10次", None),
                                    Op(1, "2025-01-13", _U, "15次", "10次"),
                                    Op(2, "2025-01-20", _U, "15次", "15次")])}}, n_sessions=3)
    st_tie, _ = line.well_posed(
        {"capability": "MR", "entity": "T", "field": "f", "gt": gt_mr(tie, "T", "f", "max"),
         "aux": {"agg": "max"}, "evidence_sessions": [0, 1, 2]}, tie)
    ck("[wp] §7.4 MR 平手不误杀(撤下 W5)", st_tie == "well_posed")
    st_tie2, rs_tie2 = line.well_posed(                       # 对照:gt.value 改错 → W1 drop
        {"capability": "MR", "entity": "T", "field": "f", "gt": {"value": "16次", "session": 1, "agg": "max"},
         "aux": {"agg": "max"}, "evidence_sessions": [0, 1, 2]}, tie)
    ck("[wp] §7.4 对照 MR 错值被 W1 drop", st_tie2 == "drop" and "gt 与世界不符" in rs_tie2)

    # ── ill-posed(应 drop,reason 含关键词)──
    ckwp("§7.5 Q19 复刻 MR gt 错算→W1",
         {"capability": "MR", "entity": E, "field": "Oncall数", "gt": {"value": "16次", "session": 2, "agg": "max"},
          "aux": {"agg": "max"}, "evidence_sessions": [0, 2, 4]}, "drop", "gt 与世界不符")
    ckwp("§7.6a IE 周锚越界→W4",
         {"capability": "IE", "entity": E, "field": "负责人", "gt": {"value": "张三", "at_week": 9},
          "aux": {"at_week": 9}, "evidence_sessions": [0, 1]}, "drop", "周锚越界")
    ckwp("§7.6b IE 周锚不一致→W4",
         {"capability": "IE", "entity": E, "field": "负责人", "gt": {"value": "张三", "at_week": 1},
          "aux": {"at_week": 0}, "evidence_sessions": [0, 1]}, "drop", "周锚不一致")
    ckwp("§7.7 ABS 不成立(字段其实存在)→W8",
         {"capability": "ABS", "entity": "", "field": "负责人", "gt": _INS,
          "aux": {}, "evidence_sessions": []}, "drop", "ABS 不成立")
    ckwp("§7.7 对照 ABS 季度营收 真缺失",
         {"capability": "ABS", "entity": "", "field": "季度营收", "gt": _INS,
          "aux": {}, "evidence_sessions": []}, "well_posed")

    # ★§7.8 FORGET 停-复活(实测真坏题型):s0 EXPIRE 后 s1 复活到有效值,gt 却 forgotten=True
    revive = _WS({"D": {"sla": Timeline([Op(0, "2025-01-06", _S, "99.5", None),
                                         Op(1, "2025-01-13", _E, None, "99.5"),
                                         Op(2, "2025-01-20", _S, "99.5", None)])}}, n_sessions=3)
    st_rv, rs_rv = line.well_posed(
        {"capability": "FORGET", "entity": "D", "field": "sla", "gt": {"forgotten": True, "value": None},
         "aux": {}, "evidence_sessions": [0, 1, 2]}, revive)
    ck("[wp] §7.8 FORGET 停-复活被 W7 drop", st_rv == "drop" and "FORGET 时效不符" in rs_rv)

    ckwp("§7.9 TR 无变更(稳定字段)→W6",
         {"capability": "TR", "entity": E, "field": "部门代号", "gt": {"session": 0, "to": "ENG"},
          "aux": {}, "evidence_sessions": [0]}, "drop", "首次变更不成立")
    ckwp("§7.10 FORGET 无停用(负责人未 EXPIRE)→W7",
         {"capability": "FORGET", "entity": E, "field": "负责人", "gt": {"forgotten": True, "value": None},
          "aux": {}, "evidence_sessions": [0, 2]}, "drop", "停用不成立")

    # 烟测:demo 世界全 enumerate 穿闸不崩 + 全 well_posed(实测当前 demo 28 单 100%)
    demo_orders = line.enumerate(ws)
    ck("[wp] 烟测:demo enumerate 全 well_posed",
       all(line.well_posed(o, ws)[0] == "well_posed" for o in demo_orders))

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L1_timeline self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
