"""
pipeline.lines.L2_relational —— L2 关系多跳产线(单能力线:L2_multihop)。

世界基质 = 软外键 over 状态机(prepare 把人名字段值提升为人员实体,从 run_factory 搬来);
点菜 = 数据驱动发现 2 跳路径 + 枚举唯一可解、优先跨周的实例;gt = 时序图遍历(gt_multihop)。
桥实体在 aux,出题须隐藏(防单文档抄近路)。
"""
from __future__ import annotations

from pipeline.lines.base import ProductionLine, Order, field_kind, interrogative
from pipeline.world_state import (
    WorldState, Timeline, Op, _date_of, week_label, _norm,
    SET, UPDATE, INVALID, INSUFFICIENT, gt_multihop,
)


# ════════════════════════════════════════════════════════════════════════════
# 点菜枚举(从 order_gen.enumerate_l2_orders / discover_fk_paths 逐字搬迁,行为不变)
# ════════════════════════════════════════════════════════════════════════════
def enumerate_l2_orders(ws: WorldState, paths, at_weeks=None, max_per_path: int = 8) -> list[Order]:
    """★L2 路径枚举:对每条软外键 field-path,为每个合法起点找一个【唯一可解、优先跨周】的 2 跳实例。
    paths:[["负责人","汇报对象"], …](来自白皮书 relation_schema)。我们的字段单值 → 解天然唯一。
    返回 L2_relational 订单(gt=终点答案;aux 带 path/at_week/bridge/cross_week/hops)。桥实体在 aux,出题须隐藏。"""
    out: list[Order] = []
    weeks = list(at_weeks) if at_weeks is not None else list(range(ws.n_sessions or 0))
    for path in paths:
        if len(path) < 2:
            continue
        starts = [e for e, flds in ws.entities.items() if path[0] in flds]   # 有首字段=合法起点
        seen, picked = set(), 0
        for start in starts:
            if picked >= max_per_path:
                break
            best = None                                  # (w, mh, cross);优先 cross_week=True
            for w in weeks:
                mh = gt_multihop(ws, start, path, w)
                if mh.get("answer") in (INSUFFICIENT, INVALID, None):
                    continue
                ev = mh.get("path_evidence", [])
                if len(ev) < 2:                          # 必须真 2 跳(中间桥实体可解)
                    continue
                cross = ev[0]["set_session"] != ev[-1]["set_session"]
                if best is None or (cross and not best[2]):
                    best = (w, mh, cross)
                if cross:
                    break
            if best is None:
                continue
            w, mh, cross = best
            ev = mh["path_evidence"]
            key = (start, tuple(path), mh["answer"])
            if key in seen:
                continue
            seen.add(key)
            out.append(Order("L2_relational", start, "→".join(path), gt=mh["answer"],
                             evidence_sessions=sorted({e["set_session"] for e in ev}),
                             question_date=ws.date_of_session(w),
                             aux={"path": path, "at_week": w, "bridge": ev[0]["value"],
                                  "cross_week": cross, "hops": len(path)}))
            picked += 1
    return out


def discover_fk_paths(ws: WorldState, hops: int = 2) -> list[list[str]]:
    """★数据驱动发现软外键路径(不依赖松散的关系描述,直接看世界):
    某字段任一取值若是另一实体的名 = 该字段是一条边。返回去重的 hops 跳【字段路径模板】。
    hops==2:f1(起点→桥)+ f2(桥的任意字段=答案)。
    hops==3:f1(起点→桥1)+ f2(桥1→桥2,必须仍是 FK 边)+ f3(桥2 的任意字段=答案)——桥实体再走一跳。"""
    keys = set(ws.entities)

    def fk_targets(ent, fld):                       # 该 (实体,字段) 指向的实体集合
        tl = ws.entities.get(ent, {}).get(fld)
        return {v for (_s, _d, v) in tl.set_values() if v in keys} if tl else set()

    def fk_fields(ent):
        return [f for f in ws.entities.get(ent, {}) if fk_targets(ent, f)]

    patterns = set()
    for x in ws.entities:
        for f1 in fk_fields(x):
            for y in fk_targets(x, f1):
                if hops == 2:
                    for f2 in ws.entities.get(y, {}):       # hop2:桥实体 y 的任意字段(末跳=答案)
                        patterns.add((f1, f2))
                elif hops == 3:
                    for f2 in fk_fields(y):                 # hop2 必须仍是 FK 边(桥1→桥2)
                        for z in fk_targets(y, f2):
                            for f3 in ws.entities.get(z, {}):   # hop3:桥2 的任意字段(末跳=答案)
                                patterns.add((f1, f2, f3))
    # set 的遍历顺序受 PYTHONHASHSEED 影响；若不排序，后续 target 截断会让同一
    # 冻结世界在不同进程中得到不同订单与 floor 结果。
    return [list(p) for p in sorted(patterns)]


def _l2_difficulty(hops: int, cross_week: bool) -> str:
    """★纯结构难度分档(全由 path/证据结构算,零 LLM):
      T1 = 2 跳·单周桥(桥稳定,证据同周)     —— 最易;
      T2 = 2 跳·跨周桥(桥换过人,须锁周回忆)  —— 中;
      T3 = 3 跳(桥实体再走一跳)              —— 最难。"""
    if hops >= 3:
        return "T3"
    return "T2" if cross_week else "T1"


def relation_path_metrics(ws: WorldState, start: str, path: list[str], at_week: int) -> dict:
    """Measure resolved reads, without treating identity reads as graph edges.

    These are structural metrics, not a claim that every cited document is
    necessary. Existing hops/difficulty fields remain historical definitions.
    """
    from pipeline.value_types import field_schema
    evidence = gt_multihop(ws, start, path, at_week).get("path_evidence", [])
    typed = bool(ws.world_blueprint and not ws.world_blueprint.get("legacy_adapter"))
    identities, edges, visited = [], [], {start}
    for index, step in enumerate(evidence):
        source, target = step["entity"], step["value"]
        visited.add(source)
        if target == source:
            identities.append(index)
            continue
        declaration = field_schema(ws, source, step["field"]) if typed else {}
        if target in ws.entities:
            edges.append({"from": source, "field": step["field"], "to": target,
                          "declared_reference": declaration.get("kind") == "reference" if typed else None})
            visited.add(target)
    sessions = sorted({step["set_session"] for index, step in enumerate(evidence) if index not in identities})
    return {"path_metrics_version": 1, "field_reads": len(evidence), "relation_hops": len(edges),
            "declared_reference_hops": sum(edge["declared_reference"] is True for edge in edges) if typed else None,
            "undeclared_relation_hops": sum(edge["declared_reference"] is False for edge in edges) if typed else None,
            "identity_reads": len(identities), "identity_read_indices": identities,
            "distinct_entities": len(visited), "resolved_relation_edges": edges,
            "non_identity_read_sessions": sessions,
            "relation_basis": "resolved_entity_values",
            "difficulty_basis": "legacy_field_reads"}


def _apportion_by_tier(tagged: list[dict], ratio: dict, target: int) -> list[dict]:
    """按档配额(如 T1:T2:T3=3:3:2)把 target 分到各难度档;某档供给不足从余档补(高配额优先)。
    与 L1 配额驱动同款:ratio 是【配比权重】非硬上限。tagged 每项 aux.difficulty 已 stamp。"""
    pools: dict[str, list] = {t: [] for t in ratio}
    for o in tagged:
        d = (o.get("aux") or {}).get("difficulty")
        pools.setdefault(d, []).append(o)
    total_w = sum(ratio.values()) or 1
    tiers = [t for t, _ in sorted(ratio.items(), key=lambda kv: -kv[1])]   # 高配额优先
    taken = {t: 0 for t in pools}
    picked: list = []
    for t in tiers:                                                        # 第一轮:按配额比例
        n_t = min(len(pools.get(t, [])), max(1, round(target * ratio[t] / total_w)))
        picked += pools[t][:n_t]; taken[t] = n_t
    while len(picked) < target:                                           # 补足:从仍有余的档轮取
        progressed = False
        for t in list(pools):
            if len(picked) >= target:
                break
            if taken[t] < len(pools[t]):
                picked.append(pools[t][taken[t]]); taken[t] += 1; progressed = True
        if not progressed:
            break
    return picked[:target]


# ════════════════════════════════════════════════════════════════════════════
# L2 产线
# ════════════════════════════════════════════════════════════════════════════
class RelationalLine(ProductionLine):
    def carrier_field_issues(self, carrier, blueprint, objects):
        # A path field belongs to one of its nodes, rather than every node.
        fields = {t['id']: {f['name'] for f in t['fields']} for t in blueprint['entity_types']}
        if carrier.get('field') and not any(carrier['field'] in fields[objects[e]['type']] for e in carrier['entities']):
            return [{'code': 'undeclared_carrier_field', 'message': 'Choose a declared field on the relation path', 'field': carrier['field']}]
        return []

    def matches_carrier(self, carrier, entities, support):
        path = (support.get('aux') or {}).get('path', [])
        edges = (support.get('aux') or {}).get('resolved_relation_edges', [])
        nodes = {edge[key] for edge in edges for key in ('from', 'to')}
        return (support.get('entity') in entities and nodes <= entities and
                (not carrier.get('field') or carrier['field'] in path))

    id = "L2_relational"
    title = "关系多跳"
    memory = "跨实体 N 跳遍历与聚合(A→B→C)"
    gt_substrate = "软外键 + 时序图遍历(gt_multihop)"
    implemented = True
    requires: list[str] = ["soft_fk_path"]       # blueprint 关系优先；legacy 才用 ≥2 person 字段合成 FK 链

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """优先认蓝图已编译的真实软外键；历史扁平世界再检查 person 字段兜底。"""
        paths = discover_fk_paths(ws, hops=2)
        if paths:
            return True, f"world blueprint 已编译 {len(paths)} 种 2 跳软外键路径"
        if getattr(ws, "world_blueprint", None) and not ws.world_blueprint.get("legacy_adapter"):
            return False, "typed world 没有可遍历软外键路径；禁止回退合成人员关系"
        pf = [f["name"] for f in (profile.get("field_schema") or []) if f.get("kind") == "person"]
        if len(pf) >= 2:
            return True, f"person 字段 {len(pf)} 个(「{pf[0]}」→人员→「{pf[1]}」可成链)"
        return False, f"person 字段仅 {len(pf)} 个(<2),无法造软外键链"

    def prepare(self, ws, profile: dict):
        """★世界基质:把【人名类字段】的值提升为【人员实体】,并给人员一个【下一跳字段】(时变,制造跨周2跳)。
        引用完整性由代码保证:dept.负责人 的值 = 真实存在的人员实体名。≥2 个 person 字段才构成链。
        (从 run_factory._augment_relations 搬来;归位到 L2 → 基质按产线激活备料。)

        ★根因修复(反词面短路):不再盲取 next_field=pf[1]。先算【起点实体族】的原生字段并集;
          若 pf[1] 与起点原生字段【碰撞】(如案件本身也有「督导合伙人」)→ 铸造一个【人专属】末跳字段名
          (pf[1]+'·直属上级')只赋给人员实体 → 起点(案件)根本没有该字段 → 词面短路无处可抄。
        ★上级池单射度:seniors 由 difficulty 控(profile.l2_senior_pool_frac,缺省近单射=全体人员)。
          近单射(池≈全体)→ 每人分到基本唯一的上级 → 桥/答案可消歧(治"张宇是所有人上级不可消歧")。"""
        existing_paths = discover_fk_paths(ws, hops=2)
        if existing_paths:
            if getattr(ws, "relations", None):
                return f"  ★关系基质:直接使用 world blueprint 编译的软外键路径×{len(existing_paths)}(不再合成人员关系)"
            return None                              # legacy 世界已有人为/历史 FK，幂等 no-op
        if getattr(ws, "world_blueprint", None) and not ws.world_blueprint.get("legacy_adapter"):
            return None                              # typed world 的关系只能来自 blueprint compiler
        pf = [f["name"] for f in (profile.get("field_schema") or []) if f.get("kind") == "person"]
        if len(pf) < 2:
            return None
        fk_field = pf[0]                               # 负责人(dept→person FK)
        from pipeline.world_state import _strip_disambig
        # ★起点实体族(有 fk_field 的实体)的原生字段并集 —— 用于探测末跳字段碰撞
        start_fields = set()
        for ent, flds in ws.entities.items():
            if fk_field in flds:
                start_fields |= set(flds.keys())
        # ★碰撞则铸人专属末跳字段名(只赋人员实体,起点无此字段→无短路);否则用原字段名
        next_field = pf[1] if pf[1] not in start_fields else f"{pf[1]}·直属上级"
        persons, seen_base = [], {_strip_disambig(e) for e in ws.entities}   # ★Fix3:人名也防表面塌缩(主干已被实体/已收人名占用 → 跳)
        for ent, flds in list(ws.entities.items()):
            tl = flds.get(fk_field)
            if not tl:
                continue
            for (_s, _d, v) in tl.set_values():
                v = str(v)
                if v and v not in ws.entities and _strip_disambig(v) not in seen_base:
                    persons.append(v); seen_base.add(_strip_disambig(v))
        persons = sorted(set(persons))
        if len(persons) < 2:                          # 造不出"人→上级人"的链(需 ≥2 个不同人名)
            return None
        # ★下一跳值 = 另一个【真实人员名】,不再写死 CTO/CEO 职级(审计 ★2,根因修复):
        #   ① 域无关(medical 的「会诊上级」= 医生名而非 CTO);② 消别名漂移(gold=名、语料也渲名,不再 CEO↔刘总);
        #   ③ 答案空间多样(不再挤在 5 个 CXO token、防瞎猜)。
        # ★上级池:近单射(默认 frac=1.0=全体)→ i→i+1 的循环移位映射 ≈ 排列(单射)→ 桥/答案唯一可消歧。
        #   frac 越小(T3 难档)→ 池越窄 → 多人共享上级 → 桥不可消歧(制造干扰)。由白皮书 difficulty 旋钮控,gold 零改。
        frac = profile.get("l2_senior_pool_frac", 1.0)
        try:
            frac = min(1.0, max(0.0, float(frac)))
        except (TypeError, ValueError):
            frac = 1.0
        k = max(2, round(len(persons) * frac))
        seniors = persons[:k]
        n, ch = (ws.n_sessions or 10), max(1, (ws.n_sessions or 10) // 2)
        for i, p in enumerate(persons):
            cands = [s for s in seniors if s != p] or [x for x in persons if x != p]
            m1, m2 = cands[i % len(cands)], cands[(i + 1) % len(cands)]
            ops = [Op(0, ws.date_of_session(0), SET, m1, None)]
            if m2 != m1:                               # 中段换上级 → 2跳证据落不同周(cross_week)
                ops.append(Op(ch, ws.date_of_session(ch), UPDATE, m2, m1))
            ws.entities[p] = {next_field: Timeline(ops)}
        note = f"  ★关系增强:+{len(persons)} 人员实体(「{next_field}」=真实人员名·时变),软外键链 「{fk_field}」→人员→「{next_field}」打通"
        if next_field != pf[1]:
            note += f"(★末跳字段与起点原生「{pf[1]}」碰撞→铸人专属名,反词面短路)"
        return note

    # ★按档配额(纯结构):T1:T2:T3 = 3:3:2(配比权重,非硬上限;供给不足从余档补)。
    _TIER_RATIO = {"T1": 3, "T2": 3, "T3": 2}

    def _answer_siblings(self, ws, order, path) -> int:
        """★distractor_n(纯结构):末跳字段上,与本链答案【同解】的其他实体数
        (= 有多少人也报给同一个上级/指向同一答案)→ 共指干扰规模。近单射世界≈0。"""
        last = path[-1] if path else None
        aux = order.aux if hasattr(order, "aux") else (order.get("aux") or {})
        w, ans = aux.get("at_week"), (order.gt if hasattr(order, "gt") else order.get("gt"))
        n = 0                                         # 末跳字段上,@w 取值=本链答案的实体数(= 有多少人也报给同一上级)
        for e, flds in ws.entities.items():
            tl = flds.get(last)
            if tl and _norm(tl.value_at_session(w)) == _norm(ans):
                n += 1
        return max(0, n - 1)                          # 减 1:本链自身的直接前驱不算干扰

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        """★2 跳 + 3 跳一起枚举;每单按结构 stamp difficulty/hops/bridge_hidden/distractor_n;按档配额(3:3:2)发题。"""
        profile = (wp or {}).get("domain_profile", {})
        raw = enumerate_l2_orders(ws, discover_fk_paths(ws, hops=2)) \
            + enumerate_l2_orders(ws, discover_fk_paths(ws, hops=3))
        tagged, seen = [], set()
        for o in raw:
            aux = dict(o.aux)
            path = aux.get("path") or []
            key = (o.entity, tuple(path), o.gt, aux.get("at_week"))
            if key in seen:                                 # 去重(2/3 跳可能重复某些起点)
                continue
            seen.add(key)
            hops = len(path)
            cross = bool(aux.get("cross_week"))
            last = path[-1] if path else None               # ★Fix2:末跳字段的 kind 决定疑问词(样本值=gt 答案)
            aux["hops"] = hops
            aux["difficulty"] = _l2_difficulty(hops, cross)
            aux.update(relation_path_metrics(ws, o.entity, path, aux.get("at_week")))
            aux["bridge_hidden"] = aux["difficulty"] != "T1"    # ★T2/T3 桥须隐藏(渲染侧钩子:桥事实不与起点同现)
            aux["distractor_n"] = self._answer_siblings(ws, o, path)
            from pipeline.value_types import field_schema
            evidence = gt_multihop(ws, o.entity, path, aux.get("at_week")).get("path_evidence", [])
            terminal = evidence[-1] if evidence else {}
            declaration = field_schema(ws, terminal.get("entity"), last, wp)
            if declaration:
                aux["value_schema"] = declaration
            aux.setdefault("ans_kind", declaration.get("kind") or field_kind(last, o.gt, profile))
            aux["time_unit"] = ws.period_unit()
            tagged.append({"line": self.id, "capability": "L2_multihop", "entity": o.entity,
                           "entity_type": ws.entity_types.get(o.entity),
                           "answer_entity_type": ws.entity_types.get(terminal.get("entity")), "answer_field": last,
                           "field": o.field, "gt": o.gt, "evidence_sessions": o.evidence_sessions, "aux": aux})
        # 同一条确定性良定义合同最终还会在全局边 A 闸复核；这里先过滤，避免已知
        # 会被拒绝的候选占掉 target 名额，造成“全池 66 条合法却只选中 3 条”的假短缺。
        eligible = [order for order in tagged
                    if self.well_posed(order, ws)[0] == "well_posed"]
        return _apportion_by_tier(eligible, self._TIER_RATIO, target)

    def gt(self, ws, o: dict):
        """护城河:时序软外键图遍历,与 enumerate 烘焙逐字段相等(见自检校验闸)。"""
        aux = o.get("aux") or {}
        return gt_multihop(ws, o["entity"], aux.get("path"), aux.get("at_week"))["answer"]

    def intent(self, o: dict) -> tuple[str, list]:
        ent, fld, aux, gt = o.get("entity", ""), o.get("field", ""), o.get("aux", {}) or {}, o.get("gt")
        path = list(aux.get("path") or [fld])
        hops = len(path)
        aw = aux.get("at_week")
        # ★§V-A 良定义:时变关系链题面必须带时间锚；中间节点统一称“所指对象”，不假定它是人。
        q = interrogative(aux.get("ans_kind"))     # ★Fix2:末跳疑问词由 path[-1] 的 kind 派生(治"管理跨度是谁")
        unit = aux.get("time_unit") or "周"
        anchor = f"第{week_label(aw)}{unit}" if aw is not None else ""
        wk = f"截至{anchor}(以该时点的状态为准)," if anchor else ""
        # intent 同时是 phraser 失败时的公开题面，因此自身必须是一句可直接发布的
        # 问题，不能混入“必须点明/绝不点名”等给模型看的制作指令。
        chain = f"【{ent}】"
        for field in path[:-1]:
            chain += f"的「{field}」所指对象"
        s = f"{wk}{chain}的「{path[-1]}」{q}？"
        hide = [str(gt)]
        if aux.get("bridge"):
            hide.append(str(aux["bridge"]))            # L2 桥实体必须隐藏
        return s, hide


    def ground(self, order, evidence_docs, all_signal_text=""):
        """接地(§G.5):末跳答案(gt 纯串)就近【桥实体 aux.bridge】;comparison 题无桥则退化为起点实体。"""
        from pipeline.grounding import gold_scalar, attributed
        s = gold_scalar(order)
        if not s:
            return ("drop", "fail-closed:L2 无待验标量")
        aux = order.get("aux") or {}
        anchor = aux.get("bridge") or order.get("entity", "")
        if attributed(s, anchor, [d["content"] for d in evidence_docs]):
            return ("grounded", f"末值 '{s}' 就近桥实体「{anchor}」")
        return ("drop", f"末值 '{s}' 未就近桥实体「{anchor}」(别名悬空/未渲)")

    def well_posed(self, order: dict, ws) -> tuple:
        """★边 A 闸(题面↔答案【良定义】),与 ground() 对称。设计见 docs/anchors/edge_a/L2_well_posed.md(已 QA 版)。
        断言:order 声明的坐标 (start, path, W) 在世界里【唯一锁死】gold。纯代码、确定性、零 LLM、绝不碰 corpus。
        落地 INV-0/1/2/3(INV-4 桥唯一、INV-5 起点自带末跳 均经实测证伪【撤下】,见下与设计 §6)。"""
        aux = order.get("aux") or {}
        start = order.get("entity", "")
        path = aux.get("path")
        W = aux.get("at_week")
        gold = order.get("gt")

        # ── INV-0 结构前置(path≥2 / hops 自洽 / gold 非空非哨兵)──
        if not isinstance(path, list) or len(path) < 2:
            return ("drop", "ill-formed:path<2 跳(L2 必须真多跳)")
        if aux.get("hops") not in (None, len(path)):
            return ("drop", f"ill-formed:hops({aux.get('hops')})≠path 长({len(path)})")
        if not (isinstance(gold, str) and gold.strip()) or gold in (INSUFFICIENT, INVALID):
            return ("drop", "ill-formed:gold 空/INSUFFICIENT(L2 必须有唯一答案)")

        # ── ★INV-6 反短路(纯结构):末跳字段若也是【起点原生字段】→ 词面短路 ──
        #   题面"X 的负责人的『末跳字段』",若 X 本身也有『末跳字段』,系统只需抄 X 自己的同名字段即可
        #   得到一个(错的)值,完全绕过多跳遍历(根因:末跳与起点原生字段同名 → 全员答起点自己的值)。
        #   prepare 已通过【碰撞则铸人专属末跳字段名】杜绝此形;此闸兜底,遇碰撞形【一律 drop】。
        if path[-1] in ws.entities.get(start, {}):
            return ("drop", f"短路:末跳字段「{path[-1]}」是起点「{start}」原生字段(词面短路,绕过多跳)")

        # ── INV-1 必须带唯一周锚(治"无周锚→横跳多解")──
        weeks = set(ws.sessions())
        if W is None or not isinstance(W, int) or W not in weeks:
            return ("drop", "无周锚:时变关系链不带合法 at_week,逐周多值 gold 不唯一")

        # ── INV-5 撤下(实测 16/16 误杀,connectivity≠ambiguity 同 roles_of)──
        #   题面"X部门负责人的汇报对象"自然解析唯一(=那个负责人的汇报对象,非部门自己的);
        #   起点同名末跳字段不使【本题】二义——要二义得问"X部门的汇报对象"(不带"负责人"=另一道题)。
        #   §V-A 已强制题面带全链("负责人的…"),故"起点自带末跳字段"不构成边 A 病。详见设计 §2 INV-5 + §6。

        # ── INV-2 + INV-3:在【单一 W】重算全链、核 gold(命门2:复用 gt_multihop,与产线 gt() 同口径)──
        mh = gt_multihop(ws, start, path, W)
        if mh.get("answer") in (INSUFFICIENT, INVALID, None) \
                or len(mh.get("path_evidence", [])) < len(path):
            return ("drop", f"链在第{W}周断裂于 {mh.get('broke_at')}(该周此关系不成立)")
        from pipeline.value_types import field_schema, values_equal, ValueComparisonError
        terminal = mh["path_evidence"][-1]
        try:
            declaration = field_schema(ws, terminal["entity"], terminal["field"]) or aux.get("value_schema") or {}
            equal = values_equal(mh["answer"], gold, declaration)
        except ValueComparisonError:
            equal = False
        if not equal:
            return ("drop", f"gold 与世界@第{W}周重算不符:世界='{mh['answer']}' gold='{gold}'(查无实据/锚不统一)")

        # ── INV-4 撤下:桥指代唯一已由【名键结构(无同名实体)+ INV-2 链不断 + INV-3 唯一重算】共同保证 ──
        return ("well_posed", "")


# 自检:python -m pipeline.lines.L2_relational
if __name__ == "__main__":
    import sys
    from pipeline.world_state import assemble_world
    line = RelationalLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    rel_table = {"entities": [
        {"name": "搜索部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "李娜"}, {"session": 3, "value": "王强"}]}}},
        {"name": "推荐部", "fields": {"负责人": {"type": "stable", "value": "周明"}}},
        {"name": "李娜", "fields": {"汇报对象": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "CTO"}, {"session": 2, "value": "CVP"}]}}},
        {"name": "王强", "fields": {"汇报对象": {"type": "stable", "value": "CEO"}}},
        {"name": "周明", "fields": {"汇报对象": {"type": "stable", "value": "CTO"}}},
    ]}
    relws, _ = assemble_world(rel_table)
    paths = discover_fk_paths(relws)
    l2 = enumerate_l2_orders(relws, [["负责人", "汇报对象"]])
    ck("软外键路径发现含 [负责人,汇报对象]", ["负责人", "汇报对象"] in paths)
    ck("软外键路径按字段元组稳定排序", paths == sorted(paths))
    ck("L2 枚举出≥2 订单", len(l2) >= 2)
    ck("L2 全是 2 跳关系题", all(o.capability == "L2_relational" and o.aux.get("hops") == 2 for o in l2))
    ck("L2 答案非空/非 INSUFFICIENT", all(o.gt not in (INSUFFICIENT, INVALID, None) for o in l2))
    ck("L2 aux 带 bridge(桥实体)", all(o.aux.get("bridge") for o in l2))
    ck("L2 至少一题跨周拼接(证据≥2周)", any(len(o.evidence_sessions) >= 2 for o in l2))
    ck("L2 桥实体≠答案(真2跳,非单查)", all(o.aux.get("bridge") != o.gt for o in l2))

    # ★护城河校验闸:gt() 重算 == enumerate 烘焙
    orders = line.enumerate(relws)
    ck("enumerate 配额前已过滤自身确定性坏题",
       all(line.well_posed(o, relws)[0] == "well_posed" for o in orders))
    ck("校验闸:gt() 重算 == 烘焙", all(line.gt(relws, o) == o["gt"] for o in orders))
    # ★§V-A 良定义:时变多跳题,题面必须带周锚(否则 gold 不唯一)
    ck("§V-A:L2 题面带周锚(单一真源 week_label=s+1)",
       all(f"第{week_label(o['aux'].get('at_week'))}周" in line.intent(o)[0] for o in orders if o["aux"].get("at_week") is not None))
    # ★Fix2:末跳疑问词由 path[-1] 的 kind 派生(治"管理跨度是谁")
    def _l2q(kind):
        return line.intent({"line": "L2_relational", "capability": "L2_multihop", "entity": "X", "field": "f",
                            "gt": "v", "aux": {"path": ["负责人", "末跳"], "at_week": 0, "ans_kind": kind}})[0]
    ck("Fix2:L2 末跳 numeric(真实schema词表)→ '是多少'且非'是谁'", "是多少" in _l2q("numeric") and "是谁" not in _l2q("numeric"))
    ck("Fix2:L2 末跳 person → '是谁'", "是谁" in _l2q("person"))
    ck("L2 intent 本身可公开，不泄漏内部制作指令",
       not any(marker in _l2q("person") for marker in ("★", "必须点明", "绝不点名", "提问:")))

    # prepare:office 型 profile(2 person 字段)应增强出人员实体
    base_table = {"entities": [
        {"name": "支付部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "陈一"}, {"session": 2, "value": "孙二"}]}}},
    ]}
    bws, _ = assemble_world(base_table)
    note = line.prepare(bws, {"field_schema": [{"name": "负责人", "kind": "person"}, {"name": "汇报对象", "kind": "person"}]})
    ck("prepare 增强出人员实体(陈一/孙二)", "陈一" in bws.entities and "孙二" in bws.entities)
    ck("prepare 人员带【汇报对象】字段", all("汇报对象" in bws.entities[p] for p in ["陈一", "孙二"]))
    ck("prepare <2 person 字段时 no-op", line.prepare(bws, {"field_schema": [{"name": "负责人", "kind": "person"}]}) is None)
    ck("prepare 近单射:每人上级各异(排列)→ 桥可消歧",
       bws.entities["陈一"]["汇报对象"].value_at_session(0) != bws.entities["孙二"]["汇报对象"].value_at_session(0))

    # ★根因修复:起点原生字段与 pf[1] 碰撞 → prepare 铸【人专属】末跳字段名,案件无此字段(反短路)
    collide_table = {"entities": [
        {"name": "案件1", "fields": {
            "负责人": {"type": "evolving", "trajectory": [{"session": 0, "value": "甲"}, {"session": 2, "value": "乙"}]},
            "督导合伙人": {"type": "stable", "value": "王雷"}}},     # ★案件原生就有「督导合伙人」= 碰撞源
    ]}
    cws, _ = assemble_world(collide_table)
    cprof = {"field_schema": [{"name": "负责人", "kind": "person"}, {"name": "督导合伙人", "kind": "person"}]}
    cnote = line.prepare(cws, cprof)
    ck("prepare 碰撞→铸人专属末跳字段「督导合伙人·直属上级」", "督导合伙人·直属上级" in cws.entities.get("甲", {}))
    ck("prepare 碰撞→起点(案件1)无该末跳字段(无短路)", "督导合伙人·直属上级" not in cws.entities.get("案件1", {}))
    ck("prepare 碰撞→人员实体不复用起点原生「督导合伙人」名", "督导合伙人" not in cws.entities.get("甲", {}))
    # 碰撞世界枚举出的题:案件起点·末跳=人专属字段 → 过 INV-6(不短路);
    #   人-起点链(起点自带同名末跳字段)才被 INV-6 正确 drop —— 这正是短路兜底。
    corders = line.enumerate(cws, wp={"domain_profile": cprof})
    ck("碰撞世界:案件起点题(末跳人专属)过 well_posed(不短路)",
       any(o["entity"] == "案件1" and line.well_posed(o, cws)[0] == "well_posed" for o in corders))
    bad_collide = {"line": "L2_relational", "capability": "L2_multihop", "entity": "案件1",
                   "field": "负责人→督导合伙人", "gt": "王雷", "evidence_sessions": [0],
                   "aux": {"path": ["负责人", "督导合伙人"], "at_week": 0, "hops": 2}}
    rSC = line.well_posed(bad_collide, cws)
    ck("INV-6:人为拼碰撞路径(末跳=起点原生字段)→ drop 短路", rSC[0] == "drop" and "短路" in rSC[1])

    # ════════════════════════════════════════════════════════════════════════
    # ★3 跳发现 + 难度分档 + 配额(纯结构)自检
    # ════════════════════════════════════════════════════════════════════════
    # 造 3 跳世界:部门→负责人(人)→直属上级(人)→再上级(人);人链可再走一跳 = 3 跳。
    chain_table = {"entities": [
        {"name": "A部", "fields": {"负责人": {"type": "stable", "value": "p1"}}},
        {"name": "B部", "fields": {"负责人": {"type": "stable", "value": "p2"}}},
        {"name": "p1", "fields": {"直属上级": {"type": "stable", "value": "p2"}}},
        {"name": "p2", "fields": {"直属上级": {"type": "stable", "value": "p3"}}},
        {"name": "p3", "fields": {"直属上级": {"type": "stable", "value": "p4"}}},
        {"name": "p4", "fields": {"职级": {"type": "stable", "value": "总监"}}},
    ]}
    chws, _ = assemble_world(chain_table)
    p3 = discover_fk_paths(chws, hops=3)
    ck("discover_fk_paths hops=3 找到 3 跳路径", any(len(p) == 3 for p in p3))
    ck("discover_fk_paths hops=3 含 [负责人,直属上级,直属上级]", ["负责人", "直属上级", "直属上级"] in p3)
    corders3 = line.enumerate(chws, wp={"domain_profile": {}})
    ck("enumerate 每单 stamp difficulty/hops/bridge_hidden/distractor_n",
       all(all(k in o["aux"] for k in ("difficulty", "hops", "bridge_hidden", "distractor_n")) for o in corders3))
    ck("enumerate 出 3 跳(T3)题", any(o["aux"]["difficulty"] == "T3" and o["aux"]["hops"] == 3 for o in corders3))
    ck("enumerate 出【可过闸】的 3 跳(T3)题(案件起点,非短路)",
       any(o["aux"]["hops"] == 3 and line.well_posed(o, chws)[0] == "well_posed" for o in corders3))
    ck("_l2_difficulty 分档:2跳单周=T1 / 2跳跨周=T2 / 3跳=T3",
       _l2_difficulty(2, False) == "T1" and _l2_difficulty(2, True) == "T2" and _l2_difficulty(3, False) == "T3")
    ck("bridge_hidden:T1=False,T2/T3=True",
       all((o["aux"]["bridge_hidden"] is False) == (o["aux"]["difficulty"] == "T1") for o in corders3))
    # 配额:合成 tagged 验 3:3:2 分配(每档供给充足时按比例)
    def _mk(d):
        return {"aux": {"difficulty": d}}
    synth = [_mk("T1")] * 10 + [_mk("T2")] * 10 + [_mk("T3")] * 10
    got = _apportion_by_tier(synth, {"T1": 3, "T2": 3, "T3": 2}, 8)
    from collections import Counter as _C
    dist = _C(o["aux"]["difficulty"] for o in got)
    ck("配额 3:3:2 @target8 → T1=3,T2=3,T3=2", dist["T1"] == 3 and dist["T2"] == 3 and dist["T3"] == 2)
    ck("配额:某档供给不足从余档补足 target",
       len(_apportion_by_tier([_mk("T1")] * 2 + [_mk("T2")] * 10, {"T1": 3, "T2": 3, "T3": 2}, 8)) == 8)

    # ════════════════════════════════════════════════════════════════════════
    # ★边 A 闸 well_posed() 自检(设计 §7:well-posed A/B + ill-posed C–H)
    #   造小世界直接拼 Op/Timeline,手搓 order,断言返回值 + reason 命中对应不变量。
    # ════════════════════════════════════════════════════════════════════════
    def WL(*steps):  # (session, op, value, prev) → Timeline
        return Timeline([Op(s, _date_of(s), op, v, p) for (s, op, v, p) in steps])

    def od(entity, path, at_w, gt, bridge=None, hops=2):
        return {"line": "L2_relational", "capability": "L2_multihop", "entity": entity,
                "field": "→".join(path), "gt": gt, "evidence_sessions": [0],
                "aux": {"path": path, "at_week": at_w, "bridge": bridge, "hops": hops}}

    P = ["负责人", "汇报对象"]
    # 公共小世界:部门→负责人→人员→汇报对象;负责人 w3 换人(李娜→王强)。
    # 逐周真值(gt_multihop 实算):w0–2→CTO(经李娜),w3–5→CEO(经王强);李娜 w5 改值在桥换人后休眠。
    base = WorldState({
        "搜索部": {"负责人": WL((0, SET, "李娜", None), (3, UPDATE, "王强", "李娜"))},
        "李娜":   {"汇报对象": WL((0, SET, "CTO", None), (5, UPDATE, "CVP", "CTO"))},
        "王强":   {"汇报对象": WL((0, SET, "CEO", None))},
    }, n_sessions=6)

    # ── well-posed(应 PASS)──
    ck("[wp] A 锚w0 链通·gold=重算·桥唯一 → well_posed",
       line.well_posed(od("搜索部", P, 0, "CTO", "李娜"), base)[0] == "well_posed")
    ck("[wp] B 锚w5 桥已换王强→CEO(单一W一致)→ well_posed",
       line.well_posed(od("搜索部", P, 5, "CEO", "王强"), base)[0] == "well_posed")

    # ── ill-posed(应 drop,且 reason 命中对应不变量)──
    rC = line.well_posed(od("搜索部", P, None, "CTO", "李娜"), base)
    ck("[wp] C 无周锚(Q02/03/05)→ INV-1 drop", rC[0] == "drop" and "无周锚" in rC[1])
    rD = line.well_posed(od("搜索部", P, 0, "COO", "李娜"), base)   # 世界只有 CTO
    ck("[wp] D gold查无实据(Q33 COO)→ INV-3 drop", rD[0] == "drop" and "查无实据" in rD[1])
    rE = line.well_posed(od("搜索部", P, 5, "CTO", "李娜"), base)   # 声明 W=5 却填 w0 旧值
    ck("[wp] E 混锚(Q32:W5 填 w0 旧值)→ INV-3 drop", rE[0] == "drop" and "查无实据" in rE[1])

    # F) 桥"一人多身份"【不再误杀】(原 INV-4 反例):赵一既数据部负责人、又新人甲导师 FK 值
    reuse = WorldState({
        "数据部": {"负责人": WL((0, SET, "赵一", None))},
        "新人甲": {"导师":   WL((0, SET, "赵一", None))},   # 赵一第 2 种 FK 身份(roles=2)
        "赵一":   {"汇报对象": WL((0, SET, "COO", None))},
    }, n_sessions=2)
    ck("[wp] F1 多角色但 gold 对 → 放行(撤 roles_of 不误杀)",
       line.well_posed(od("数据部", P, 0, "COO", "赵一"), reuse)[0] == "well_posed")
    rF2 = line.well_posed(od("数据部", P, 0, "CTO", "赵一"), reuse)  # 同世界但 gold 错
    ck("[wp] F2 多角色 gold 错 → INV-3 drop(Q27 真缺陷在此边)",
       rF2[0] == "drop" and "查无实据" in rF2[1])

    # G) ★INV-6 反短路:起点自带【与末跳同名】字段 → 词面短路(全员抄起点自己的「汇报对象」=李四,绕过多跳)→ 必 drop。
    #    (这正是根因坏题型:末跳字段=起点原生字段;INV-6 当初就会 drop 掉全部 7 题。prepare 侧已靠铸人专属字段名杜绝。)
    dual = WorldState({
        "前端部": {"负责人": WL((0, SET, "张三", None)), "汇报对象": WL((0, SET, "李四", None))},  # 部门自带汇报对象=李四
        "张三":   {"汇报对象": WL((0, SET, "CFO", None))},
    }, n_sessions=2)
    rG = line.well_posed(od("前端部", P, 0, "CFO", "张三"), dual)
    ck("[wp] G 起点自带同名末跳字段 → INV-6 反短路 drop", rG[0] == "drop" and "短路" in rG[1])

    # H) 链断(末跳实体在 W 无该字段)→ INV-2
    broke = WorldState({"销售部": {"负责人": WL((0, SET, "孤儿", None))}}, n_sessions=2)
    rH = line.well_posed(od("销售部", P, 0, "X", "孤儿"), broke)
    ck("[wp] H 链断(孤儿无汇报对象)→ INV-2 drop", rH[0] == "drop" and "断裂" in rH[1])

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L2_relational self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
