"""
pipeline.lines.L7_consolidation —— L7 长跨度趋势归纳产线。设计见 docs/anchors/L7_consolidation_design.md。

【它和 L1.MR / L3 的区别】MR=数值聚合(加总=多少钱,单值算术);L3=事件排序(按先后排好)。
  L7 = 把散落全程的事实整合成一个【方向/对比判断】(格式塔),且【最近一期会误导你】(抗 recency、抗逐字)。
  v0 两型(都 code-falsifiable、gold 是人类口径类别/名字、答案侧无周号):
    S1 趋势方向:某数值字段全程整体【上升/下降】(确定性分类器 + margin;模棱两可一律跳,不产软 gold)。
    S2 跨实体对比:给定候选集,哪个实体某字段【变动最频繁】(argmax change_ops + margin 防平手)。
  【v1 记此,未做】S3 阶段对比(前k周 vs 后m周变了哪些字段)——gold 是【字段名集合】,集合型接地/判分脆,
    易引入歧义,故 v0 不做(本着"源头做对、不贪多、不造软 gold")。

★命脉(设计 §6):趋势天生"软"。NET_FRAC 是 v0 默认值,DoD 要求【真实序列实测"代码标签 vs 人类标签分歧率"定标】
  (复用 L4 众数-margin 那套);分歧高 = 这型不可靠,像 W5/roles_of 一样毙掉。当前为离线自检默认值,未经真实定标。
★签名:S1 强制【末段反向】(抗 recency)——纯单调到底的序列"看最近一段"就蒙对、退化成 KU,well_posed 必 drop。
"""
from __future__ import annotations
import random

from pipeline.lines.base import ProductionLine
from pipeline.world_state import WorldState, _to_num, _norm, _strip_disambig

NET_FRAC = 0.30           # ★趋势=【首末净方向】(对齐 ANSWER_PROTOCOL v3),|末−首| ≥ NET_FRAC×摆幅 才算清晰;否则波动→不产
T_MIN = 4                 # 趋势至少 4 个数值点(太短无"长跨度"可言)
CMP_MARGIN = 2            # 对比:冠军 change 数 − 亚军 ≥ 2(防平手)
CMP_MAX_CAND = 4          # 对比候选集上限(题面列出,自足可答)
GROUND_MIN_PTS = 3        # S1 接地:≥3 个序列值就近实体(证据够散才能看出趋势)
_SEED = 20260608


def _numeric_seq(tl) -> list:
    """该字段按时间序的数值序列 [num,...](mention-on-change 下 = 读者在语料里看见的取值序)。"""
    if tl is None:
        return []
    out = []
    for (_s, _d, v) in tl.set_values():
        n = _to_num(v)
        if n is not None:
            out.append(n)
    return out


def _trend_label(nums: list):
    """★确定性趋势分类器 = 【首末净方向】(对齐 ANSWER_PROTOCOL v3"趋势=首末净方向"):
    净位移 |末−首| ≥ NET_FRAC×摆幅(max−min)→ 上升/下降;否则 None(净位移太小=波动/不清晰,不产软 gold)。
    ★旧版用"升降步数差"是协议未声明前的代理:对'单调+末反转'恰好一致,但对'中段深谷/末段突变'等
      形状会与协议打架(174925 趋势形状模板化的修复要多形状,逼对齐到协议本身的口径)。"""
    if len(nums) < T_MIN:
        return None
    span = max(nums) - min(nums)
    if span <= 0:
        return None
    net = nums[-1] - nums[0]
    if net >= NET_FRAC * span:
        return "上升"
    if -net >= NET_FRAC * span:
        return "下降"
    return None


def _anti_recency(nums: list, label: str) -> bool:
    """★抗"看局部"签名:序列里【至少一处局部反向】于整体净方向(末段反挫 / 中段深谷 / 前段平缓相反…)。
    保证"只看某个局部窗口"会判错 → 逼跨全程整合(不是只看最近/只看某段)。形状无关,与多形状 imprint 配套。"""
    if len(nums) < 2 or label not in ("上升", "下降"):
        return False
    up = label == "上升"
    return any((nums[i] < nums[i - 1]) if up else (nums[i] > nums[i - 1]) for i in range(1, len(nums)))


def _change_count(tl) -> int:
    """该字段【真值变动】次数(world_state.change_ops:删/过期 + 新值≠旧值;初始 SET 不算)= "变动频繁度"。"""
    return len(tl.change_ops()) if tl else 0


# ════════════════════════════════════════════════════════════════════════════
# L7 产线
# ════════════════════════════════════════════════════════════════════════════
class ConsolidationLine(ProductionLine):
    id = "L7_consolidation"
    title = "巩固摘要"
    memory = "长跨度趋势/对比归纳(整段方向、谁变最多;非逐字)"
    gt_substrate = "数值序列趋势分类器 + change_ops 计数(代码可算,确定性)"
    implemented = True
    requires: list[str] = ["multi_event_timelines"]   # 需演化字段(趋势/变动才有料),口径同 L3

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """Use the actual candidate constructors, including trend and margin rules."""
        n_s1, n_s2 = len(self._enum_trend(ws)), len(self._enum_compare(ws))
        tags = ([f"S1趋势={n_s1}"] if n_s1 else []) + ([f"S2对比={n_s2}"] if n_s2 else [])
        if tags:
            return True, f"可产归纳型:{'/'.join(tags)}"
        return False, "趋势候选=0、比较候选=0：未满足已声明趋势准入或变更次数差距；不修改世界补造供给"

    # prepare:无 —— L7 不造新世界,只对既有演化字段做整段归纳

    @staticmethod
    def _field_candidates(ws) -> dict:
        """{字段名: [拥有该字段的实体名…]}(S2 找'同字段跨实体'对比集用)。"""
        owners: dict = {}
        for ent, flds in ws.entities.items():
            for fname in flds:
                owners.setdefault(fname, []).append(ent)
        return owners

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        if target <= 0:
            return []
        rng = random.Random(_SEED)
        out = self._enum_trend(ws) + self._enum_compare(ws)
        rng.shuffle(out)
        # 同(实体,型)只出一题:控冗余、促覆盖
        picked, seen = [], set()
        for o in out:
            key = (o["entity"], o["aux"]["sub"])
            if key in seen:
                continue
            seen.add(key); picked.append(o)
            if len(picked) >= target:
                break
        return picked

    def _enum_trend(self, ws) -> list[dict]:
        """S1:只对【imprint 真正种下趋势的字段】(ws._trended_fields)出题。
        ★审计 HIGH 根治:旧版扫【所有】数值字段跑 _trend_label,而'首末净方向'对纯噪声放行率 62.7%
          → 非 imprint 的 LLM 噪声字段(如 4 点 [57,56,68,10])照产 gold=下降 的软题,违反本线契约
          '模棱两可一律跳'。改为只骑 planted 趋势(噪声字段根本不判),从根上杜绝软 gold——契合
          'LLM 给噪声、代码种结构、L7 骑结构'的设计本意,非去调分类器阈值猜噪声(那才是打地鼠)。"""
        trended = set(getattr(ws, "_trended_fields", []) or [])
        out = []
        for (ent, fname) in sorted(trended):
            tl = ws.entities.get(ent, {}).get(fname)
            if tl is None:
                continue
            nums = _numeric_seq(tl)
            label = _trend_label(nums)
            if label and _anti_recency(nums, label):
                out.append({"line": self.id, "capability": "L7_consolidation", "entity": ent, "field": fname,
                            "gt": label,
                            "evidence_sessions": sorted({s for (s, _d, v) in tl.set_values() if _to_num(v) is not None}),
                            "aux": {"sub": "S1_trend", "n_points": len(nums)}})
        return out

    def _enum_compare(self, ws) -> list[dict]:
        """S2:同字段跨实体,候选集里谁变动最频繁(argmax change_ops + margin)。"""
        out = []
        for fname, ents in self._field_candidates(ws).items():
            if len(ents) < 2:
                continue
            counts = sorted(((ent, _change_count(ws.entities[ent][fname])) for ent in ents),
                            key=lambda x: (-x[1], x[0]))
            if not counts or counts[0][1] < 1:
                continue
            # 不能固定取 top-k：若前几名并列 7 次，但后面有 4 次的实体，
            # top-k 会误判“无明确赢家”。题面的候选集本来就是枚举器选定的子集，
            # 因此从同字段拥有者中选“一个最高者 + 差距达 margin 的对手”，仍保证唯一 gold。
            winner = counts[0]
            opponents = [item for item in counts[1:]
                         if winner[1] - item[1] >= CMP_MARGIN]
            if not opponents:
                continue
            cand = [winner] + opponents[:CMP_MAX_CAND - 1]
            winner = cand[0][0]
            cand_names = sorted(e for e, _ in cand)
            evid = sorted({o.session for e, _ in cand for o in ws.entities[e][fname].change_ops()})
            winner_values, seen_values = [], set()
            for _session, _date, value in ws.entities[winner][fname].set_values():
                normalized = _norm(value)
                if normalized and normalized not in seen_values:
                    seen_values.add(normalized)
                    winner_values.append(str(value))
            out.append({"line": self.id, "capability": "L7_consolidation", "entity": winner, "field": fname,
                        "gt": winner,
                        "evidence_sessions": evid,
                        "aux": {"sub": "S2_compare", "candidates": cand_names,
                                "counts": {e: c for e, c in cand},
                                "winner_values": winner_values}})
        return out

    # ── ★护城河:从世界重算趋势标签 / argmax 赢家,== 烘焙则过校验闸 ──
    def gt(self, ws, order: dict):
        aux = order.get("aux") or {}
        sub, ent, fld = aux.get("sub"), order.get("entity"), order.get("field")
        if sub == "S1_trend":
            return _trend_label(_numeric_seq(ws.entities.get(ent, {}).get(fld)))   # 上升/下降/None
        if sub == "S2_compare":
            cands = aux.get("candidates") or []
            ranked = sorted(((e, _change_count(ws.entities.get(e, {}).get(fld))) for e in cands),
                            key=lambda x: (-x[1], x[0]))
            return ranked[0][0] if ranked else None
        return None

    # ── ★边 A 闸:良定义 well_posed ──
    def well_posed(self, order: dict, ws) -> tuple:
        aux = order.get("aux") or {}
        sub, ent, fld = aux.get("sub"), order.get("entity"), order.get("field")

        if sub == "S1_trend":
            nums = _numeric_seq(ws.entities.get(ent, {}).get(fld))
            label = _trend_label(nums)
            if label is None:                             # _trend_label 即【首末净方向】判定(≥NET_FRAC×摆幅),与 ANSWER_PROTOCOL v3 同口径
                return ("drop", f"无清晰趋势:「{ent}.{fld}」首末净位移 < {NET_FRAC:.0%}×摆幅(模棱两可,不产软 gold)")
            if _norm(label) != _norm(order.get("gt")):    # gold 必须 == 世界重算的首末净方向(护城河)
                return ("drop", f"gold 锚错:gt={order.get('gt')!r} ≠ 世界重算趋势 {label!r}")
            if not _anti_recency(nums, label):            # ∃局部反向:逼跨全程整合(只看某局部窗口会判错)
                return ("drop", f"抗 recency 不成立:「{ent}.{fld}」无局部反向(看局部即泄漏趋势 → 退化 KU)")
            # 注:旧版此处另有"net_label==label"独立闸;_trend_label 改首末净方向后它恒成立=死分支,已删(审计点名死码)。
            return ("well_posed", "")

        if sub == "S2_compare":
            cands = aux.get("candidates") or []
            if len(cands) < 2:
                return ("drop", "对比候选 < 2(无可比)")
            ranked = sorted(((e, _change_count(ws.entities.get(e, {}).get(fld))) for e in cands),
                            key=lambda x: (-x[1], x[0]))
            if ranked[0][1] - ranked[1][1] < CMP_MARGIN:
                return ("drop", f"对比无明确赢家:冠亚军变动数差 < {CMP_MARGIN}(平手,gold 不唯一)")
            if _norm(ranked[0][0]) != _norm(order.get("gt")):
                return ("drop", f"gold 锚错:gt={order.get('gt')!r} ≠ 重算赢家 {ranked[0][0]!r}")
            if ranked[0][1] < 1:
                return ("drop", "赢家变动数 0(全程没变,'变动最频繁'无意义)")
            return ("well_posed", "")

        return ("drop", f"未知归纳型 '{sub}'(fail-closed)")

    # ── questions stage:强制"看整段";趋势给固定二选一候选(type-critical 表面锁死);抗 recency 不给提示 ──
    def intent(self, order: dict) -> tuple[str, list]:
        aux = order.get("aux") or {}
        sub, ent, fld = aux.get("sub"), order.get("entity", ""), order.get("field", "")
        if sub == "S1_trend":
            s = (f"综合【{ent}】全程「{fld}」的变化,整体趋势是上升还是下降?"
                 f"——看整段走向、别只看最近一两期;只回:上升 / 下降。")
            return s, []                       # 趋势结论不在语料,无需藏;每期值留作证据
        if sub == "S2_compare":
            cands = "、".join(aux.get("candidates") or [])
            s = (f"全程来看,在【{cands}】中,哪一个的「{fld}」变动最频繁?"
                 f"——综合整段变更次数判断,只回一个名字。")
            return s, []
        return f"{ent} 的「{fld}」", []

    # ── ★命门3 接地:S1 验序列值够散(趋势可见);S2 验赢家变动可见(≥2 个不同值就近赢家)──
    def ground(self, order, evidence_docs, all_signal_text=""):
        from pipeline.grounding import attributed
        aux = order.get("aux") or {}
        sub, ent, fld = aux.get("sub"), order.get("entity", ""), order.get("field")
        docs = [d["content"] for d in evidence_docs]

        if sub == "S1_trend":
            # 接地不持有 ws:数"证据文档里就近实体的【不同数值】个数"——≥GROUND_MIN_PTS 才算趋势可见。
            seen = set()
            for d in docs:
                seen |= _num_tokens(_norm(d))
            grounded_pts = sum(1 for tok in seen if attributed(tok, ent, docs))
            if grounded_pts < GROUND_MIN_PTS:
                return ("drop", f"趋势证据不足:仅 {grounded_pts} 个数值就近「{ent}」(< {GROUND_MIN_PTS},趋势不可见)")
            return ("grounded", f"{grounded_pts} 个数值就近「{ent}」(趋势可见)")

        if sub == "S2_compare":
            values = list(dict.fromkeys(str(v) for v in aux.get("winner_values", []) if _norm(v)))
            grounded_values = [value for value in values if attributed(value, ent, docs)]
            if len(grounded_values) < 2:
                return ("drop", f"赢家变动不可见:「{ent}.{fld}」的真实字段值就近仅 "
                                f"{len(grounded_values)} 个(< 2,看不出'变动频繁')")
            return ("grounded", f"赢家「{ent}」的「{fld}」≥2 个真实字段值就近(变动可见)")

        return ("drop", f"未知归纳型 '{sub}'(fail-closed)")


# ── ground 辅助(不持有 ws,只从证据文本判) ───────────────────────────────
def _num_tokens(text: str) -> set:
    """从 _norm 文本里抽出数值 token(整数/小数),供 S1 接地数"出现了几个数值"。"""
    import re
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# 自检:python -m pipeline.lines.L7_consolidation
if __name__ == "__main__":
    import sys
    from pipeline.world_state import assemble_world
    line = ConsolidationLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 世界:
    #  研发部.缺陷数:10→20→30→40→35  = 清晰上升 + 末点下挫(抗 recency)→ S1 上升
    #  负责人(person)在 3 部门变动数 = 3 / 1 / 0 → S2 赢家=研发部,margin=2
    table = {"entities": [
        {"name": "研发部", "fields": {
            "缺陷数": {"type": "evolving", "trajectory": [
                {"session": 0, "value": "10"}, {"session": 1, "value": "20"},
                {"session": 2, "value": "30"}, {"session": 3, "value": "40"}, {"session": 4, "value": "35"}]},
            "负责人": {"type": "evolving", "trajectory": [
                {"session": 0, "value": "李明"}, {"session": 1, "value": "王芳"},
                {"session": 2, "value": "张伟"}, {"session": 3, "value": "陈静"}]}}},      # 变动 3 次
        {"name": "测试部", "fields": {
            "负责人": {"type": "evolving", "trajectory": [
                {"session": 0, "value": "刘洋"}, {"session": 2, "value": "赵敏"}]}}},        # 变动 1 次
        {"name": "运维部", "fields": {
            "负责人": {"type": "stable", "value": "孙浩"}}},                                  # 变动 0 次
    ], "n_sessions": 5}
    ws, _ = assemble_world(table)
    ws._trended_fields = [("研发部", "缺陷数")]   # ★模拟 imprint 已种趋势(L7 只骑 planted 字段;真实路径由 world_gen.imprint 设)

    ok, why = line.feasible(ws, {})
    ck("feasible True(有 S1+S2)", ok is True and "S1" in why and "S2" in why)

    orders = line.enumerate(ws, target=50)
    by_sub = {"S1_trend": [], "S2_compare": []}
    for o in orders:
        by_sub[o["aux"]["sub"]].append(o)
    ck("产出 S1+S2 两型", by_sub["S1_trend"] and by_sub["S2_compare"])

    # S1:研发部.缺陷数 = 上升
    s1 = next((o for o in by_sub["S1_trend"] if o["entity"] == "研发部" and o["field"] == "缺陷数"), None)
    ck("S1 命中研发部.缺陷数", s1 is not None)
    ck("S1 gold=上升", s1 and s1["gt"] == "上升")
    ck("S1 题面给二选一候选(上升/下降)", s1 and "上升还是下降" in line.intent(s1)[0])
    ck("S1 题面强制看整段、压 recency", s1 and "别只看最近" in line.intent(s1)[0])
    ck("S1 护城河:gt() 重算==上升", s1 and line.gt(ws, s1) == "上升")
    ck("S1 well_posed 过", s1 and line.well_posed(s1, ws) == ("well_posed", ""))

    # S2:负责人 变动对比 → 赢家研发部
    s2 = next((o for o in by_sub["S2_compare"] if o["field"] == "负责人"), None)
    ck("S2 命中负责人对比", s2 is not None)
    ck("S2 gold=研发部(变动最多)", s2 and s2["gt"] == "研发部")
    ck("S2 候选含三部门", s2 and set(s2["aux"]["candidates"]) == {"研发部", "测试部", "运维部"})
    ck("S2 题面列出候选 + 只回一个名字", s2 and "研发部" in line.intent(s2)[0] and "只回一个名字" in line.intent(s2)[0])
    ck("S2 护城河:gt() 重算==研发部", s2 and line.gt(ws, s2) == "研发部")
    ck("S2 well_posed 过", s2 and line.well_posed(s2, ws) == ("well_posed", ""))

    # ── IP:well_posed 必 drop 的坏题 ──
    # IP1 波动(首末净位移过小 < NET_FRAC×摆幅):10→20→15→11,首=10 末=11 净移1、摆幅10 → _trend_label=None
    #   ★注:按协议"趋势=首末净方向",纯振荡但首末分明(如 10→20→10→20,末=max)是【上升】、可裁,不再算波动;
    #     只有【首末几乎持平】才算无清晰趋势——这正是 v3 协议与旧 ups-downs 代理的差别。
    flat_ws, _ = assemble_world({"entities": [{"name": "A部", "fields": {"x": {"type": "evolving",
        "trajectory": [{"session": 0, "value": "10"}, {"session": 1, "value": "20"},
                       {"session": 2, "value": "15"}, {"session": 3, "value": "11"}]}}}], "n_sessions": 4})
    ip1 = {"line": line.id, "capability": "L7_consolidation", "entity": "A部", "field": "x",
           "gt": "上升", "aux": {"sub": "S1_trend"}, "evidence_sessions": [0, 1, 2, 3]}
    ck("IP1 首末持平→drop(净位移过小)", line.well_posed(ip1, flat_ws)[0] == "drop")
    # IP1b 纯振荡但首末分明(末=max):按协议=上升、可裁(不再误杀为波动)
    osc_ws, _ = assemble_world({"entities": [{"name": "B部", "fields": {"y": {"type": "evolving",
        "trajectory": [{"session": 0, "value": "10"}, {"session": 1, "value": "20"},
                       {"session": 2, "value": "10"}, {"session": 3, "value": "20"}]}}}], "n_sessions": 4})
    ip1b = {"line": line.id, "capability": "L7_consolidation", "entity": "B部", "field": "y",
            "gt": "上升", "aux": {"sub": "S1_trend"}, "evidence_sessions": [0, 1, 2, 3]}
    ck("IP1b 振荡但首末分明→上升可裁(协议=首末净方向)", line.well_posed(ip1b, osc_ws) == ("well_posed", ""))

    # IP2 纯单调到底(抗 recency 失败):10→20→30→40→50 → 上升但末段同向
    mono_ws, _ = assemble_world({"entities": [{"name": "B部", "fields": {"y": {"type": "evolving",
        "trajectory": [{"session": 0, "value": "10"}, {"session": 1, "value": "20"}, {"session": 2, "value": "30"},
                       {"session": 3, "value": "40"}, {"session": 4, "value": "50"}]}}}], "n_sessions": 5})
    ip2 = {"line": line.id, "capability": "L7_consolidation", "entity": "B部", "field": "y",
           "gt": "上升", "aux": {"sub": "S1_trend"}, "evidence_sessions": [0, 1, 2, 3, 4]}
    st, why2 = line.well_posed(ip2, mono_ws)
    ck("IP2 纯单调到底→drop(抗recency失败)", st == "drop" and "recency" in why2)

    # IP3 对比平手:两部门变动数相同 → margin 不足
    tie_ws, _ = assemble_world({"entities": [
        {"name": "甲", "fields": {"f": {"type": "evolving", "trajectory": [
            {"session": 0, "value": "a"}, {"session": 1, "value": "b"}]}}},
        {"name": "乙", "fields": {"f": {"type": "evolving", "trajectory": [
            {"session": 0, "value": "c"}, {"session": 1, "value": "d"}]}}},
    ], "n_sessions": 3})
    ip3 = {"line": line.id, "capability": "L7_consolidation", "entity": "甲", "field": "f",
           "gt": "甲", "aux": {"sub": "S2_compare", "candidates": ["甲", "乙"]}, "evidence_sessions": [0, 1]}
    ck("IP3 对比平手→drop(无明确赢家)", line.well_posed(ip3, tie_ws)[0] == "drop")

    # ── ground:S1 序列值够散 → grounded;S2 赢家≥2不同值 → grounded ──
    s1_docs = [{"session": 0, "content": "研发部本周缺陷数10。", "is_conflict": False},
               {"session": 2, "content": "研发部本周缺陷数30。", "is_conflict": False},
               {"session": 4, "content": "研发部本周缺陷数35。", "is_conflict": False}]
    ck("ground S1:3个数值就近研发部 → grounded", s1 and line.ground(s1, s1_docs)[0] == "grounded")
    s1_thin = [{"session": 0, "content": "研发部本周缺陷数10。", "is_conflict": False}]
    ck("ground S1:仅1个数值 → drop(趋势不可见)", s1 and line.ground(s1, s1_thin)[0] == "drop")
    s2_docs = [{"session": 0, "content": "研发部负责人李明。", "is_conflict": False},
               {"session": 2, "content": "研发部负责人张伟。", "is_conflict": False}]
    # S2 接地数"数值",负责人是人名非数值 → 这里改用一个数值字段验 S2 接地逻辑
    s2_num_docs = [{"session": 0, "content": "研发部指标 3。", "is_conflict": False},
                   {"session": 1, "content": "研发部指标 7。", "is_conflict": False}]
    s2_num = {"line": line.id, "capability": "L7_consolidation", "entity": "研发部", "field": "指标",
              "gt": "研发部", "aux": {"sub": "S2_compare", "candidates": ["研发部", "测试部"],
                                          "winner_values": ["3", "7"]}, "evidence_sessions": [0, 1]}
    ck("ground S2:赢家≥2个真实字段值就近 → grounded", line.ground(s2_num, s2_num_docs)[0] == "grounded")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L7_consolidation self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
