"""
pipeline.lines.L6_refusal —— L6 抗虚构拒答产线(单能力线:L6_refusal)。设计见 docs/anchors/L6_refusal_design.md。

【它和 L1.ABS 的区别】ABS = 字段【全局缺失】(没人有)→ 干净的"无此项"。
  L6 = 语料里【摆着一个像样的诱饵、且诱饵就近探针实体本身】,考"宁可说不知道,也别瞎编"(记忆系统最致命的幻觉)。
  ★rebalance(本版根治 0/42 命中诱饵的旧病):旧 T1【张冠张李】把诱饵挂到【别的实体】→ 探针检索空 → 反射式"查无"trivial;
    新设三型全部把诱饵【就近探针实体自己】,逼系统在【E 确实在场、料很像】时仍守住边界:
    R1 邻字段(T1_adjacent):E 有别的字段(邻字段值就近 E)但【没目标字段 F】→ 诱去拿 E 自己的邻字段值当 F 的答案。
    R2 假前提(T3_premise,最强):E 的 F 曾经有、【现在到期/删了且其后无恢复】→ 现在时强预设"E 现在的 F 是多少",诱去答停前值。
    R3 时间窗外(T2_window,最弱/兜底):语料只到第 n 周,问第 (n+2) 周(E 窗内真有 F,更诱去外推)。
  gold 一律 = 拒答哨兵 INSUFFICIENT(由 value_at/latest_valid 机械判);型别只进 aux 供分桶。

【为什么近乎零依赖】① 三型诱饵都是世界里的【正常内容】(邻字段值/停前值/窗内值)→ 渲染器照常渲,render 不改;
  ② gold_scalar 对未知 cap 默认返 None → 本线自己 ground()(骑 ABS 反向判缺席的先例),grounding 核心不改;
  ③ 拒答哨兵已在 ANSWER_PROTOCOL,protocol 不改。

★配额:enumerate 按 [R2,R1,R3] 加权轮转采样(R2 最强优先、R3 最弱兜底),取代旧 shuffle-rich-first + (entity,type) dedup
  ——旧法在小 target 下被单型(最弱 T1)独占,正是 0/42 病根;轮转保证小配额也【强型优先、三型都在场】。
"""
from __future__ import annotations
import random
import re
from collections import defaultdict

from pipeline.lines.base import ProductionLine, field_kind, interrogative
from pipeline.world_state import (WorldState, SET, UPDATE, EXPIRE, DELETE, INVALID, INSUFFICIENT,
                                   gt_pre_expire, _norm)

_AT_WEEK_OFFSET = 2          # 窗外问"第 (n_sessions + 2) 周":n=7 → 第9周,不踩"第8周像 0-index"的歧义
_SEED = 20260608

# ★[R2,R1,R3] 加权轮转(强型优先、弱型兜底):R2 假前提最强、R1 邻字段次之、R3 窗外最弱
_ROTATION = ["T3_premise", "T1_adjacent", "T2_window"]
_WEIGHT = {"T3_premise": 3, "T1_adjacent": 2, "T2_window": 1}


def field_ownership_mentions(entity: str, field: str, texts: list[str]) -> list[dict]:
    """Conservative three-way attribution of nearby field mentions.

    The old 90-character window only proposes candidates. A mention is cleared
    only by an explicit absence statement or a separately named owner in the
    same clause. Unresolved pronouns/ownership stay ambiguous and fail closed.
    This intentionally recognizes a small grammar, not arbitrary coreference.
    """
    from pipeline.grounding import WINDOW
    entity, field = _norm(entity), _norm(field)
    if not entity or not field:
        return []
    findings = []
    boundary = re.compile(r"[。！？!?；;\n]")
    modifiers = r"(?:本周|本期|当前|最新)?"
    for text in texts:
        normalized = _norm(text)
        breaks = [-1] + [match.start() for match in boundary.finditer(normalized)] + [len(normalized)]
        for match in re.finditer(re.escape(field), normalized):
            position = match.start()
            window = normalized[max(0, position - WINDOW):match.end() + WINDOW]
            if entity not in window:
                continue
            start = max(point for point in breaks if point < position) + 1
            end = min(point for point in breaks if point >= match.end())
            prefix, suffix = normalized[start:position], normalized[match.end():end]
            status, reason = "ambiguous", "field_owner_unresolved"
            if re.fullmatch(r"[」】\"]?(?:在[^，,；;。]{0,12})?(?:尚无记录|暂无记录|未记录|未提供|没有记录|无记录)(?:[，,](?:不得据此|不能据此)[^；;。]*)?", suffix):
                status, reason = "cleared", "explicit_absence"
            elif re.search(re.escape(entity) + r"[」】》\"]?(?:的|[，,]其)?" + modifiers + r"[「【\"]?$", prefix):
                if re.match(r"[」】\"]?(?:为|是|登记为|调整为|更新为|记录为|[:：])[^，,；;。]+", suffix) and not suffix.startswith(("是什么", "为何")):
                    status, reason = "target", "explicit_target_owner"
            else:
                # E.g. “报告乙的字段”. Never infer an owner from “其/该…”.
                owner_match = re.search(r"([^，,:：；;。！？!?\n]{2,80})的" + modifiers + r"[「【\"]?$", prefix)
                owner = owner_match.group(1).strip("《》「」【】") if owner_match else ""
                generic = {"客户", "公司", "集团", "企业", "部门", "报告", "双方", "他们", "保险公司", "保险客户", "保险集团"}
                if owner and owner not in generic and entity not in owner and not owner.startswith(("其", "该", "本", "上述", "相关", "同一", "此", "非", "除")):
                    status, reason = "cleared", "explicit_other_owner"
                else:
                    # A single explicitly titled subject can retain ownership
                    # through its predicate and “其”, but not across semicolons
                    # or another literal mention of the probed entity.
                    owners = re.findall(r"《([^《》]+)》", prefix)
                    if len(owners) == 1 and entity not in prefix:
                        after_owner = prefix.split("》", 1)[1]
                        if re.fullmatch(r"(?:的|进行[^，,]{0,20}[，,]其|以该[^，,]{0,20}为客户[，,]|[^，,]{0,20}[，,]其)?[「【\"]?", after_owner):
                            status, reason = "cleared", "explicit_titled_owner"
            clause = normalized[start:end]
            if re.search(r"但|然而|实际|其实|而是|并非|不是|不属于|不能认定|不认为|不表示|不代表", clause):
                status, reason = "ambiguous", "contrast_or_negated_ownership"
            findings.append({"status": status, "reason": reason, "clause": clause})
    return findings


def _has_field(ws: WorldState, ent: str, fld: str) -> bool:
    """实体 ent 在世界里是否【曾有】fld 的有效取值(任一 session 的 SET/UPDATE)。
    用于:R1 探针须【从无】此字段(考"E 在场却没这一项")。"""
    tl = ws.entities.get(ent, {}).get(fld)
    return bool(tl) and any(v for (_s, _d, v) in tl.set_values())


def _has_other_field(ws: WorldState, ent: str, fld: str) -> bool:
    """E 是否【除 fld 外】还有≥1 个字段有有效值(R1 的命门:否则 E 根本不在场 = 退化成纯 ABS,不是拒答)。"""
    for g, tl in ws.entities.get(ent, {}).items():
        if g == fld:
            continue
        if any(v for (_s, _d, v) in tl.set_values()):
            return True
    return False


def _stop_no_resume(tl):
    """该字段是否【停统且其后从未恢复】。是 → 返回 (停用 session, 停前值);否 → None。
    R2 假前提的命门:有 EXPIRE/DELETE,且其后无任何 SET/UPDATE(若有 = 恢复了 = 前提成立 = 有答案)。"""
    if tl is None:
        return None
    ops = tl._sorted()
    stop_i = next((i for i, o in enumerate(ops) if o.op in (EXPIRE, DELETE)), None)
    if stop_i is None:
        return None
    if any(o.op in (SET, UPDATE) for o in ops[stop_i + 1:]):    # 其后有恢复 → 不是假前提
        return None
    return (ops[stop_i].session, ops[stop_i].prev)


def _field_owners(ws: WorldState) -> dict:
    """{字段名: {拥有该字段有效值的实体集}}(R1 找'目标字段 F 是世界里真实字段、但探针 E 没有'用)。"""
    owners = defaultdict(set)
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            if any(v for (_s, _d, v) in tl.set_values()):
                owners[fname].add(ent)
    return owners


# ════════════════════════════════════════════════════════════════════════════
# L6 产线
# ════════════════════════════════════════════════════════════════════════════
class RefusalLine(ProductionLine):
    id = "L6_refusal"
    title = "拒答边界"
    memory = "知道记忆边界,诱饵【就近探针实体】也拒答(抗虚构)"
    gt_substrate = "世界结构反向判定(邻字段在场但目标字段缺席 / 停统无恢复 / 窗外)+ 诱饵=世界真实内容"
    implemented = True
    requires: list[str] = []      # R3 窗外恒可行 → 近根线(R1/R2 富类型有则更强,见 feasible)

    def carrier_field_issues(self, carrier, blueprint, objects):
        """A refusal probe may lack the target field on the focus object.

        T1_adjacent requires that the target field exists elsewhere in the
        world while the focus owns a different, real lure field. The actual
        absence and lure are certified by the original enumerator and
        well-posed check after world authorship.
        """
        field = carrier.get("field")
        if not field:
            return []
        declared = {item["name"] for kind in blueprint["entity_types"]
                    for item in kind["fields"]}
        if field not in declared:
            return [{"code": "undeclared_carrier_field",
                     "message": "The refusal target must be a real field somewhere in the world",
                     "field": field}]
        return []

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """恒可行(R3 窗外任何世界都能产);报告 R1/R2 富类型可用性,便于日志看清这一轮 L6 有没有对抗料。"""
        owners = _field_owners(ws)
        all_fields = {f for flds in ws.entities.values() for f in flds}
        has_r1 = any((ent not in owners[f]) and _has_other_field(ws, ent, f)
                     for ent in ws.entities for f in all_fields if owners[f])
        has_r2 = any(_stop_no_resume(tl) for flds in ws.entities.values() for tl in flds.values())
        tags = ["R3窗外"] + (["R1邻字段"] if has_r1 else []) + (["R2假前提"] if has_r2 else [])
        return True, f"可产拒答型:{'/'.join(tags)}"

    # prepare:无 —— L6 不造新世界,只从既有世界挑拒答机会(诱饵是世界正常内容,渲染器照常渲)

    # ── orders stage:[R2,R1,R3] 加权轮转采样(强型优先、三型都在场)──
    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        profile = (wp or {}).get("domain_profile", {})
        rng = random.Random(_SEED)
        pools = {
            "T3_premise": self._enum_premise(ws, profile),     # R2 最强
            "T1_adjacent": self._enum_adjacent(ws, profile),   # R1
            "T2_window": self._enum_window(ws, profile),       # R3 最弱/兜底
        }
        for k in pools:
            rng.shuffle(pools[k])
        idx = {k: 0 for k in pools}
        out: list[dict] = []
        while len(out) < target:
            progressed = False
            for k in _ROTATION:                                # 每轮:R2 取 3、R1 取 2、R3 取 1(加权)
                for _ in range(_WEIGHT[k]):
                    if len(out) >= target:
                        break
                    if idx[k] < len(pools[k]):
                        out.append(pools[k][idx[k]]); idx[k] += 1; progressed = True
            if not progressed:
                break                                          # 三池耗尽 = 世界供给上限
        for order in out:
            (order.get("aux") or {})["time_unit"] = ws.period_unit()
        return out

    def _order(self, ent, fld, rtype, ans_kind, lure, evidence, reason, probe=None):
        aux = {"refusal_type": rtype, "ans_kind": ans_kind, "lure": lure, "reason": reason}
        if probe:
            aux["probe"] = probe
        return {"line": self.id, "capability": "L6_refusal", "entity": ent, "field": fld,
                "gt": INSUFFICIENT, "evidence_sessions": sorted(set(e for e in evidence if e is not None)),
                "aux": aux}

    def _enum_adjacent(self, ws, profile) -> list[dict]:
        """R1:探针 E 缺目标字段 F(F 是世界里真实字段、别的实体有),但 E【有别的字段】→
        诱去拿 E 自己的【邻字段值】当 F 的答案。诱饵值就近 E 本身(与旧'张冠李戴挂到别实体'相反)。"""
        owners = _field_owners(ws)
        out = []
        for ent in ws.entities:
            adj = []                                            # E 自己有值的邻字段 [(字段名, set_values)]
            for g, tl in ws.entities[ent].items():
                gv = [(s, d, v) for (s, d, v) in tl.set_values() if v]
                if gv:
                    adj.append((g, gv))
            if not adj:
                continue                                        # E 无任何在场字段 → 退化纯 ABS,跳过
            adj.sort(key=lambda x: x[0])
            for fname, holders in sorted(owners.items()):
                if ent in holders:                              # E 自己有该字段 → 非拒答
                    continue
                cand = [(g, gv) for (g, gv) in adj if g != fname]
                if not cand:
                    continue
                g, gv = cand[0]                                 # 取 E 的第一个邻字段做诱饵源
                sess, _d, val = gv[len(gv) // 2]                # 中段代表值当诱饵(就近 E)
                hv = [v for (_s, _d2, v) in ws.entities[sorted(holders)[0]][fname].set_values() if v]
                fsample = hv[len(hv) // 2] if hv else None      # F 的代表值仅用于定 ans_kind(疑问词按 F 的型)
                out.append(self._order(
                    ent, fname, "T1_adjacent", field_kind(fname, fsample, profile),
                    lure={"entity": ent, "value": val, "field": g},
                    evidence=[s for (s, _, _) in gv],
                    reason=f"「{ent}」从无「{fname}」记录;'{val}' 是它自己的「{g}」(邻字段诱饵,就近 E)"))
        return out

    def _enum_premise(self, ws, profile) -> list[dict]:
        """R2(最强):F 停统且其后无恢复 → 现在时问"E 现在的 F"(假前提),诱去答停前值(就近 E)。"""
        out = []
        for ent, flds in ws.entities.items():
            for fname, tl in flds.items():
                stop = _stop_no_resume(tl)
                if stop is None:
                    continue
                stop_sess, prestop_val = stop
                if not prestop_val:
                    continue
                out.append(self._order(
                    ent, fname, "T3_premise", field_kind(fname, prestop_val, profile),
                    lure={"entity": ent, "value": prestop_val},
                    evidence=[stop_sess] + [s for (s, _, v) in tl.set_values() if _norm(v) == _norm(prestop_val)],
                    reason=f"「{ent}」的「{fname}」在第{stop_sess}周停统、其后从未恢复(假前提)"))
        return out

    def _enum_window(self, ws, profile) -> list[dict]:
        """R3(最弱/兜底):语料只到第 n 周,问第 (n+OFFSET) 周(E 窗内真有 F → 更诱去外推)。"""
        n = ws.n_sessions or 0
        if n <= 0:
            return []
        at_week = n + _AT_WEEK_OFFSET                               # 1-based:n=7 → 第9周
        out = []
        for ent, flds in ws.entities.items():
            for fname, tl in flds.items():
                sv = [(s, d, v) for (s, d, v) in tl.set_values() if v]
                if not sv:
                    continue
                out.append(self._order(
                    ent, fname, "T2_window", field_kind(fname, sv[-1][2], profile),
                    lure={"entity": ent, "value": sv[len(sv) // 2][2]},
                    evidence=[s for (s, _, _) in sv],
                    reason=f"语料只到第{n}周,问第{at_week}周(窗外)",
                    probe={"at_week": at_week}))
        return out

    # ── ★护城河:每型重新验"确实答不出"(全由 value_at/latest_valid 机械判)。成立→拒答哨兵;不成立→非哨兵(well_posed 据此 drop)──
    def gt(self, ws, order: dict):
        aux = order.get("aux") or {}
        t, ent, fld = aux.get("refusal_type"), order.get("entity"), order.get("field")
        if t == "T1_adjacent":
            # R1:E 无目标字段 F、且 E 有别的字段在场 → 拒答(有 F = 可答;无别字段 = 退化 ABS,不是 L6 拒答)
            return (INSUFFICIENT if (not _has_field(ws, ent, fld) and _has_other_field(ws, ent, fld))
                    else "__HAS_FIELD__")
        if t == "T3_premise":
            # R2:F 的最新有效值 = INVALID(= 停统无恢复)→ 拒答(value_at(latest) 机械判)
            tl = ws.entities.get(ent, {}).get(fld)
            return INSUFFICIENT if (tl is not None and tl.latest_valid() == INVALID) else "__RESUMED__"
        if t == "T2_window":
            at_week = (aux.get("probe") or {}).get("at_week")
            n = ws.n_sessions or 0
            return INSUFFICIENT if (isinstance(at_week, int) and at_week > n) else "__IN_WINDOW__"
        return "__UNKNOWN_TYPE__"

    # ── ★边 A 闸:良定义 well_posed —— 专杀【假拒答】(标了拒答其实有合法答案)──
    def well_posed(self, order: dict, ws) -> tuple:
        aux = order.get("aux") or {}
        t, ent, fld = aux.get("refusal_type"), order.get("entity"), order.get("field")

        # W0 gold 必须是拒答哨兵 + gt() 重算一致(护城河:订单与世界脱钩则 drop)
        if _norm(order.get("gt")) != _norm(INSUFFICIENT):
            return ("drop", f"L6 类型错配:gt 应为拒答哨兵 {INSUFFICIENT},得 {order.get('gt')!r}")
        if _norm(self.gt(ws, order)) != _norm(INSUFFICIENT):
            return ("drop", "假拒答:gt() 重算【确实有答案】(拒答前提不成立)")

        if t == "T1_adjacent":
            if _has_field(ws, ent, fld):                        # 探针其实有 → 非拒答
                return ("drop", f"假拒答:「{ent}」其实有「{fld}」记录(非缺失)")
            if not _has_other_field(ws, ent, fld):              # ★E 无别的在场字段 → 退化纯 ABS(非'诱饵就近 E'的拒答)
                return ("drop", f"退化纯ABS:「{ent}」除「{fld}」外无任何在场字段(E 不在场,诱不动)")
            return ("well_posed", "")

        if t == "T2_window":
            at_week = (aux.get("probe") or {}).get("at_week")
            n = ws.n_sessions or 0
            if not (isinstance(at_week, int) and at_week > n):
                return ("drop", f"非窗外:at_week={at_week} 未超过语料 {n} 周")
            if not _has_field(ws, ent, fld):                    # 窗内本就无此字段 = 退化成 ABS,非"窗外"
                return ("drop", f"窗外退化:「{ent}」窗内本无「{fld}」(应归 ABS)")
            return ("well_posed", "")

        if t == "T3_premise":
            stop = _stop_no_resume(ws.entities.get(ent, {}).get(fld))
            if stop is None:
                return ("drop", f"前提其实成立:「{ent}」的「{fld}」非'停统无恢复'(有答案)")
            if not stop[1]:                                     # 停前值为空 → 诱不动
                return ("drop", f"停前值为空:「{ent}」的「{fld}」停统前无值(诱不动)")
            return ("well_posed", "")

        return ("drop", f"未知拒答型 '{t}'(fail-closed)")

    # ── questions stage:长得像普通题(这就是诱饵);疑问词走 ans_kind;诱饵留语料,题面不藏 ──
    def intent(self, order: dict) -> tuple[str, list]:
        aux = order.get("aux") or {}
        t, ent, fld = aux.get("refusal_type"), order.get("entity", ""), order.get("field", "")
        q = interrogative(aux.get("ans_kind"))
        unit = aux.get("time_unit") or "周"
        if t == "T2_window":
            at_week = (aux.get("probe") or {}).get("at_week")
            s = f"第 {at_week} {unit},{ent} 的「{fld}」{q}"           # at_week 已是 1-based 序号
        elif t == "T3_premise":
            s = f"{ent} 现在的「{fld}」{q}"                          # ★现在时强预设(诱去答停前值)
        else:  # T1_adjacent(邻字段,也是兜底)
            s = f"{ent} 的「{fld}」{q}"
        return s + "？", []    # 时间/现在时与普通字段查询均为合同中的必需槽位。

    # ── ★命门3 接地:验【诱饵就近探针 E】(题够诱、E 在场)+【真答案不误接 F】(拒答成立)。骑 ABS 反向判先例 ──
    def ground(self, order, evidence_docs, all_signal_text=""):
        from pipeline.grounding import attributed, STOP_MARKERS
        aux = order.get("aux") or {}
        t, ent, fld = aux.get("refusal_type"), order.get("entity", ""), order.get("field", "")
        docs = [d["content"] for d in evidence_docs]
        lure = aux.get("lure") or {}
        lv = lure.get("value")

        if t == "T1_adjacent":
            # Presence is lexical; field ownership requires a separate check.
            if not attributed(lv, ent, docs):
                return ("drop", f"E 未丰富在场:邻字段诱饵 '{lv}' 未就近「{ent}」(题不够诱)")
            visible = docs + (all_signal_text.splitlines() if all_signal_text else [])
            ownership = field_ownership_mentions(ent, fld, visible)
            if any(item["status"] == "target" for item in ownership):
                return ("drop", f"拒答与正文冲突:正文明确把「{fld}」归属于「{ent}」")
            if any(item["status"] == "ambiguous" for item in ownership):
                return ("drop", f"归属歧义待审:「{fld}」与「{ent}」就近共现，无法唯一确定字段主体")
            return ("grounded", f"邻字段诱饵 '{lv}' 就近「{ent}」；目标字段无就近断言，或已明确归属其他主体/未记录")

        if t == "T2_window":
            if not attributed(lv, ent, docs):
                return ("drop", f"窗内值 '{lv}' 未就近「{ent}」(题不够诱)")
            return ("grounded", f"窗内值 '{lv}' 就近「{ent}」;窗外第{(aux.get('probe') or {}).get('at_week')}周语料结构性无")

        if t == "T3_premise":
            # 停前诱饵值【就近 E】+ 停用标记【就近 E】(读者看得出'E 的 F 已停统'→'现在的 F'是假前提)
            if not attributed(lv, ent, docs):
                return ("drop", f"停前诱饵 '{lv}' 未就近「{ent}」(题不够诱)")
            if not any(_norm(ent) in _norm(x) and any(m in _norm(x) for m in STOP_MARKERS) for x in docs):
                return ("drop", f"停用标记未就近「{ent}」(读者看不出'已停统',假前提不成立)")
            return ("grounded", f"停前值 '{lv}' 就近「{ent}」+ 停用标记就近「{ent}」('现在的 F'是假前提)")

        return ("drop", f"未知拒答型 '{t}'(fail-closed)")


# 自检:python -m pipeline.lines.L6_refusal
if __name__ == "__main__":
    import sys
    from pipeline.world_state import assemble_world, Timeline, Op, _date_of
    line = RefusalLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 世界:工程部 有P0缺陷率、无负责人(R1:问工程部负责人 → 诱饵=它自己的P0邻字段值);
    #       数据部 有负责人;风控部 的"季度KPI"在 s2 停统、其后无恢复(R2 假前提);n_sessions=5
    table = {"entities": [
        {"name": "工程部", "fields": {"P0缺陷率": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "2.5"}, {"session": 2, "value": "1.8"}]}}},
        {"name": "数据部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "李四"}, {"session": 3, "value": "王强"}]}}},
        {"name": "风控部", "fields": {"季度KPI": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "90"}, {"session": 2, "value": None}]}}},  # s2 EXPIRE 无恢复
    ], "n_sessions": 5}
    profile = {"field_schema": [{"name": "负责人", "kind": "person"},
                                {"name": "P0缺陷率", "kind": "numeric"}, {"name": "季度KPI", "kind": "numeric"}]}
    ws, _ = assemble_world(table)

    # ── feasible / enumerate ──
    ok, why = line.feasible(ws, profile)
    ck("feasible 恒 True", ok is True)
    ck("feasible 报告含 R1/R2/R3", all(x in why for x in ["R1", "R2", "R3"]))

    orders = line.enumerate(ws, target=50, wp={"domain_profile": profile})
    by_type = defaultdict(list)
    for o in orders:
        by_type[o["aux"]["refusal_type"]].append(o)
    ck("产出三型都有", all(by_type[t] for t in ["T1_adjacent", "T2_window", "T3_premise"]))
    ck("所有 gt 都是拒答哨兵", all(o["gt"] == INSUFFICIENT for o in orders))
    ck("所有订单 cap=L6_refusal", all(o["capability"] == "L6_refusal" for o in orders))

    # ★rebalance 核心:诱饵实体【恒 == 探针实体自己】(不再挂到别的实体 → 探针检索不再空)
    ck("★诱饵就近探针本身(lure.entity==entity,全型)",
       all(o["aux"]["lure"]["entity"] == o["entity"] for o in orders))

    # R1:应有"工程部的负责人"(工程部无负责人、有P0缺陷率邻字段)
    r1 = next((o for o in by_type["T1_adjacent"] if o["entity"] == "工程部" and o["field"] == "负责人"), None)
    ck("R1 命中:问工程部的负责人(它没有、但有P0邻字段)", r1 is not None)
    ck("R1 诱饵实体=工程部自己、值是其P0缺陷率之一", r1 and r1["aux"]["lure"]["entity"] == "工程部"
       and r1["aux"]["lure"]["value"] in {"2.5", "1.8"})
    ck("R1 疑问词=是谁(按目标字段 F=负责人 的 person 型)", r1 and "是谁" in line.intent(r1)[0])
    ck("R1 题面不剧透诱饵(长得像普通题)",
       r1 and r1["aux"]["lure"]["value"] not in line.intent(r1)[0] and line.intent(r1)[1] == [])

    # R2:风控部.季度KPI —— 现在时强预设 + 诱饵=停前值90
    r2 = next((o for o in by_type["T3_premise"] if o["entity"] == "风控部"), None)
    ck("R2 命中:风控部.季度KPI(停统无恢复)", r2 is not None)
    ck("R2 诱饵=停前值90", r2 and r2["aux"]["lure"]["value"] == "90")
    ck("R2 intent 现在时强预设('现在')", r2 and "现在" in line.intent(r2)[0])
    ck("R2 题面不剧透停前值90", r2 and "90" not in line.intent(r2)[0])

    # R3:at_week = 1-based = n_sessions+2;题面"第N周"(★验 off-by-one 不复发)
    r3 = by_type["T2_window"][0]
    _exp_week = ws.n_sessions + _AT_WEEK_OFFSET
    ck(f"R3 at_week = n_sessions+2 (={_exp_week})", r3["aux"]["probe"]["at_week"] == _exp_week)
    ck("R3 题面是 1-based 周号(非裸 session)", f"第 {_exp_week} 周" in line.intent(r3)[0])

    # ── 护城河:gt() 重算 == 拒答哨兵(gold 由世界 value_at/latest_valid 机械算)──
    ck("校验闸:gt() 重算全 == 拒答哨兵", all(line.gt(ws, o) == INSUFFICIENT for o in orders))
    ck("R2 gt 由 latest_valid==INVALID 判", ws.timeline("风控部", "季度KPI").latest_valid() == INVALID)

    # ── ★配额 rebalance:小 target 下【强型 R2 优先、不被最弱型独占】──
    small = line.enumerate(ws, target=3, wp={"domain_profile": profile})
    ck("小 target 首题=R2(强型优先)", small and small[0]["aux"]["refusal_type"] == "T3_premise")
    ck("小 target 未被最弱 R3 独占", not all(o["aux"]["refusal_type"] == "T2_window" for o in small))

    # ════════════════════════════════════════════════════════════════════════
    # ★边 A 闸 well_posed:WP 必过;IP 各触发对应 drop
    # ════════════════════════════════════════════════════════════════════════
    ck("WP-R1 well-posed", line.well_posed(r1, ws) == ("well_posed", ""))
    ck("WP-R2 well-posed", line.well_posed(r2, ws) == ("well_posed", ""))
    ck("WP-R3 well-posed", line.well_posed(r3, ws) == ("well_posed", ""))

    # IP1 假拒答(R1):探针其实有该字段 → drop(用"数据部的负责人",数据部真有负责人)
    ip_has = line._order("数据部", "负责人", "T1_adjacent", "person",
                         lure={"entity": "数据部", "value": "李四", "field": "负责人"}, evidence=[0], reason="伪造")
    ck("IP1 假拒答(探针其实有该字段)→drop", line.well_posed(ip_has, ws)[0] == "drop")

    # IP2 退化纯ABS(R1):E 除目标字段外无任何在场字段 → drop
    #   构造实体"空部":唯一字段被 EXPIRE 掉(无 set 值),问它缺的另一字段
    empty_ws = WorldState({"空部": {"某项": Timeline([Op(0, _date_of(0), EXPIRE, None, None)])},
                           "他部": {"负责人": Timeline([Op(0, _date_of(0), SET, "钱七", None)])}}, n_sessions=3)
    ip_abs = line._order("空部", "负责人", "T1_adjacent", "person",
                         lure={"entity": "空部", "value": "x", "field": "某项"}, evidence=[0], reason="退化")
    ck("IP2 退化纯ABS(E 无任何在场字段)→drop", line.well_posed(ip_abs, empty_ws)[0] == "drop")

    # IP3 非窗外(R3):at_week ≤ n → drop
    ip_inwin = line._order("数据部", "负责人", "T2_window", "person",
                           lure={"entity": "数据部", "value": "李四"}, evidence=[0],
                           reason="非窗外", probe={"at_week": 3})    # 3 ≤ 5
    ck("IP3 at_week≤n(非窗外)→drop", line.well_posed(ip_inwin, ws)[0] == "drop")

    # IP4 前提其实成立(R2):字段停后又恢复 → drop
    resume_table = {"entities": [{"name": "甲", "fields": {"f": {"type": "evolving",
        "trajectory": [{"session": 0, "value": "a"}, {"session": 1, "value": None}, {"session": 3, "value": "b"}]}}}],
        "n_sessions": 5}   # s1 停、s3 又 SET=b → 恢复了
    resume_ws, _ = assemble_world(resume_table)
    ip_resume = line._order("甲", "f", "T3_premise", "text",
                            lure={"entity": "甲", "value": "a"}, evidence=[0], reason="其实恢复了")
    ck("IP4 停后又恢复(前提成立)→drop", line.well_posed(ip_resume, resume_ws)[0] == "drop")

    # ── ground():R1 翻转 / R2 停用标记就近 ──
    r1_lv = r1["aux"]["lure"]["value"]
    docs_r1_ok = [{"session": 0, "content": f"工程部本周P0缺陷率{r1_lv},交付正常。", "is_conflict": False}]
    ck("ground R1:E 在场(P0就近工程部)+ 目标字段名'负责人'不就近 → grounded",
       line.ground(r1, docs_r1_ok)[0] == "grounded")
    docs_r1_bad = [{"session": 0, "content": f"工程部本周负责人为李四,P0缺陷率{r1_lv}。", "is_conflict": False}]
    ck("ground R1:目标字段名'负责人'就近工程部 → 假拒答 drop",
       line.ground(r1, docs_r1_bad)[0] == "drop")
    docs_r1_absent = [{"session": 0, "content": "某无关内容(工程部未在场)。", "is_conflict": False}]
    ck("ground R1:E 未丰富在场(诱饵未就近)→ drop", line.ground(r1, docs_r1_absent)[0] == "drop")
    docs_r2 = [{"session": 0, "content": "风控部季度KPI为90。", "is_conflict": False},
               {"session": 2, "content": "风控部季度KPI本周起停止统计。", "is_conflict": False}]
    ck("ground R2:停前值90就近风控部+停用标记就近 → grounded", line.ground(r2, docs_r2)[0] == "grounded")
    docs_r2_nomarker = [{"session": 0, "content": "风控部季度KPI为90。", "is_conflict": False}]
    ck("ground R2:无停用标记就近 → drop", line.ground(r2, docs_r2_nomarker)[0] == "drop")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L6_refusal self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
