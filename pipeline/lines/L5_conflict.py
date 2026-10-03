"""
pipeline.lines.L5_conflict —— L5 冲突可信产线(单能力线:L5_conflict)。

世界基质 = 跨来源矛盾【侧信道】ws.conflicts:prepare 给【已有权威值】的文本字段,在那一周追加
一条【低可信"小道"值】(canonical 时间线不动 → L1/L2/L3 的 gt 不被污染;小道值由渲染器另吐
一篇低可信来源文档,与权威信号文档同周并存 = 语料里真出现矛盾)。
  gt   = 按裁决规则重算:rule="source_reliability" → 信权威源(gt = 权威值);可重算 → 过校验闸。
  出题 = 不泄漏两值(都埋在语料里),逼记忆系统【检索两边 + 按来源可靠度裁决】。
  区分度 = 朴素 top-k 只取更高频/更相似那条(常是小道)→ 答错(信号竞争升级版,带显式裁决规则);
           会"检索两边再按来源裁决"的系统才对。这正是 M3 的可证伪化。

【v1 路线(未落地,记此)】
  · "标不确定":对【本无权威记录】之事(absent 字段)注入两条互斥小道 → gt=无法确定,考"知道自己不知道";
  · "新近裁决":两份都权威、只是先后不同 → 取新 ≈ L1.KU(已覆盖),不在本线重复造。
"""
from __future__ import annotations
import random

from pipeline.lines.base import ProductionLine, Order
from pipeline.world_state import WorldState, INVALID, INSUFFICIENT, _to_num, _norm, week_label

AUTHORITATIVE_SRC = "官方通报"
RUMOR_SRCS = ["内部群聊转述", "未经核实的外部传闻", "走廊里的口耳相传"]
TEXT_KINDS = ("person", "status", "category", "text", "string")  # 声明为文本的字段优先使用其类型

# ── 边 A 良定义闸的真相源(单一来源,派生自上方来源常量;见 L5_well_posed.md §3)──
# 可靠度档:数值越大越可信。当前两档;将来加"准官方/已核实"等档只改此表,闸自动生效。
_RELIABILITY_TIERS: dict[str, int] = {
    AUTHORITATIVE_SRC: 2,                       # 官方通报
    **{src: 0 for src in RUMOR_SRCS},          # 各类传闻/小道,同档
    # 策展型世界可明确区分“一手/物证”与“受污染官方记录”。
    "独立一手记录": 3,
    "独立物证链": 3,
    "受污染的官方记录": 0,
}
_KNOWN_RULES = {"source_reliability"}          # 闸能证伪良定义的裁决规则(fail-closed:未知规则一律 drop)


def reliability_rank(src: str):
    """来源 → 可靠度档(整数)。未登记来源返回 None(= 不可比 → I6 判 drop,fail-closed)。"""
    return _RELIABILITY_TIERS.get(src)


def _is_text_field(kinds: dict, fname: str, tl) -> bool:
    """文本类字段判定(矛盾注入的着床点):profile 声明为 person/status/category,
    或时间线取值全非数值(数值字段拼小道值不像"张冠李戴",故排除)。feasible 与注入共用此口径。"""
    if kinds.get(fname) in TEXT_KINDS:
        return True
    vals = [v for (_s, _d, v) in tl.set_values()]
    return bool(vals) and all(_to_num(v) is None for v in vals)


def has_text_field(ws: WorldState, profile: dict) -> bool:
    """世界里是否存在【可着床矛盾的文本类字段】= L5 的基质可行性(口径同 inject_conflicts)。"""
    kinds = {f.get("name"): f.get("kind") for f in (profile.get("field_schema") or [])}
    return any(_is_text_field(kinds, fname, tl)
               for flds in ws.entities.values() for fname, tl in flds.items())


def inject_conflicts(ws: WorldState, profile: dict, max_n: int = 3, seed: int = 20260605) -> list[dict]:
    """挑【文本类(person/status/category)、且有权威值】的字段,注入跨来源矛盾(source_reliability)。
    小道错误值 = 同字段【全局值池】里 ≠ 权威值的另一真实取值(张冠李戴,看似合理而非瞎编)。"""
    rng = random.Random(seed)
    kinds = {f.get("name"): f.get("kind") for f in (profile.get("field_schema") or [])}

    def is_text(fname, tl):
        return _is_text_field(kinds, fname, tl)

    cands, pool = [], {}
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            if is_text(fname, tl):
                cands.append((ent, fname, tl))
                pool.setdefault(fname, set()).update(str(v) for (_s, _d, v) in tl.set_values() if v)
    rng.shuffle(cands)

    planned = [r for r in profile.get("supply_construction", {}).get("rows", []) if r["line"] == "L5_conflict"]
    if planned:
        available = {}
        for ent, fname, tl in cands:
            own = {_norm(v) for _s, _d, v in tl.set_values() if v}
            if any(_norm(v) not in own and _norm(v) != _norm(ent) for v in pool.get(fname, ())):
                available[(ent, fname)] = (ent, fname, tl)
        choices = {r["id"]: [key for key in available if key[0] in r["carrier"]["entities"] and key[1] == r["carrier"].get("field")] for r in planned}
        assignments = {}
        def assign(demand, seen):
            for key in choices[demand]:
                if key in seen:
                    continue
                seen.add(key)
                if key not in assignments or assign(assignments[key], seen):
                    assignments[key] = demand
                    return True
            return False
        for requirement in sorted(planned, key=lambda r: len(choices[r["id"]])):
            assign(requirement["id"], set())
        cands = [available[key] for key in assignments]

    conflicts: list[dict] = []
    for ent, fname, tl in cands:
        if len(conflicts) >= max_n:
            break
        sv = [(s, d, v) for (s, d, v) in tl.set_values() if v]
        if not sv:
            continue
        s, d, auth = sv[len(sv) // 2]                      # 中段权威值(避开端点,稳)
        own = {_norm(v) for (_s, _d, v) in sv}             # ★本实体该字段【所有周】真值
        # 小道值必须来自【别的实体】,不能是本实体别周真值——否则"传闻"实为本实体过去/未来真值,
        # 矛盾退化成合法时效演化(好系统会按 recency 给竞争正解,gold 不再唯一可辩护)。见 L5 well_posed I7。
        # 排己:传闻值不得 == 主体专名(否则题面既要点主语又要藏该值 → 悬空代词,见 well_posed I8)
        wrong = sorted(v for v in pool.get(fname, ()) if _norm(v) not in own and _norm(v) != _norm(ent))
        if not wrong:
            continue
        conflicts.append({
            "entity": ent, "field": fname, "session": s, "date": d,
            "authoritative_value": auth, "authoritative_source": AUTHORITATIVE_SRC,
            "rumor_value": rng.choice(wrong), "rumor_source": rng.choice(RUMOR_SRCS),
            "rule": "source_reliability", "gt": auth,
        })
    return conflicts


# ════════════════════════════════════════════════════════════════════════════
# L5 产线
# ════════════════════════════════════════════════════════════════════════════
class ConflictLine(ProductionLine):
    def construction_spec(self):
        return {"conflict": {"rule": "source_reliability", "text_field_required": True,
                "rumor_from_other_entity": True, "exclude_all_canonical_values_of_focus": True,
                "canonical_unchanged": True}}

    def construction_issues(self, carrier, blueprint, objects, events, observations):
        types = {t["id"]: t for t in blueprint["entity_types"]}
        kinds = [f.get("kind") for e in carrier["entities"] for f in types[objects[e]["type"]]["fields"] if f["name"] == carrier.get("field")]
        return [] if kinds and all(k in ("text", "string", "person", "status", "category") for k in kinds) else [
            {"code": "conflict_field", "message": "Select a declared text, person, status or category field for source_reliability", "field": carrier.get("field")}]

    def value_issues(self, world, carrier):
        profile = {"field_schema": [f for t in world.world_blueprint.get("entity_types", []) for f in t["fields"]],
                   "supply_construction": {"rows": [{"id": "value_check", "line": self.id, "carrier": carrier}]}}
        available = inject_conflicts(world, profile, max_n=len(carrier["entities"]))
        return [] if available else [{"code": "conflict_value_pool", "message": "Give another entity a distinct business-grounded value of the same field; keep canonical history consistent", "field": carrier.get("field")}]
    id = "L5_conflict"
    title = "冲突可信"
    memory = "跨来源矛盾检测 + 按来源可靠度裁决"
    gt_substrate = "矛盾注入侧信道 ws.conflicts + 裁决规则(代码可算)"
    implemented = True
    # conflicts 只保存“低可信说法 + 裁决规则”，不改 canonical 时间线。
    typed_overlay_safe = True
    requires: list[str] = ["text_fields"]    # 需文本类字段(person/status/category)作矛盾着床点

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """看世界里有没有可着床矛盾的文本类字段(口径同 inject_conflicts);无 → prepare 注 0 条、产 0 单。
        prepare 之前的轻量判定:只判"有无文本字段",不预跑整套注入(值池配对在 prepare 内兜)。"""
        if has_text_field(ws, profile):
            return True, "存在文本类字段(person/status/category),可注入跨来源矛盾"
        return False, "无文本类字段(全数值),矛盾无处着床"

    def prepare(self, ws, profile: dict):
        """世界基质:注入跨来源矛盾到 ws.conflicts(canonical 不动)。幂等(--force 重跑不叠加)。
        ★注入条数 max_n 由 profile["l5_max_conflicts"] 决定(闭环 invert_rate 反推的配额;缺省 3),取代写死。"""
        if getattr(ws, "conflicts", None) and not profile.get("supply_construction", {}).get("rows"):
            return None
        ws.conflicts = inject_conflicts(ws, profile, max_n=int(profile.get("l5_max_conflicts", 3)))
        if ws.conflicts:
            return f"  ★冲突注入:+{len(ws.conflicts)} 条跨来源矛盾(canonical 不动,小道值另渲低可信文档)"
        return None

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        out = []
        for c in (getattr(ws, "conflicts", None) or [])[:target]:
            out.append({"line": self.id, "capability": "L5_conflict", "entity": c["entity"],
                        "field": c["field"], "gt": c["gt"], "evidence_sessions": [c["session"]],
                        "aux": {"session": c["session"], "rule": c["rule"],
                                "authoritative_value": c.get("authoritative_value"),
                                "authoritative_source": c.get("authoritative_source"),
                                "rumor_value": c.get("rumor_value"),
                                "rumor_source": c.get("rumor_source"),
                                "time_unit": ws.period_unit()}})
        return out

    def gt(self, ws, o: dict):
        """护城河:从 ws.conflicts 重新查到本题矛盾,按裁决规则重算(与烘焙相等,见自检校验闸)。"""
        ent, fld, s = o["entity"], o["field"], (o.get("aux") or {}).get("session")
        c = next((c for c in (getattr(ws, "conflicts", None) or [])
                  if c["entity"] == ent and c["field"] == fld and c.get("session") == s), None)
        if c is None:
            return o.get("gt")
        if c["rule"] == "source_reliability":
            return c["authoritative_value"]                # 规则:信权威源
        return o.get("gt")

    # ── ★边 A 闸:良定义 well_posed(I0–I8,设计见 docs/anchors/edge_a/L5_well_posed.md §4)──
    def well_posed(self, order: dict, ws) -> tuple:
        """边 A 良定义闸:order 的 gold(权威值)是否为该矛盾题在世界里【唯一、合法】的答案,
        且传闻是真干扰。纯代码、确定性、零 LLM、绝不碰 corpus。与 ground() 对称。
        返回 ("well_posed","") 或 ("drop", reason)。reason 取首个失败项(顺序 I0→I3→I8→I4→I7)。"""
        aux = order.get("aux") or {}
        ent, fld, s = order.get("entity"), order.get("field"), aux.get("session")
        auth, rumor = aux.get("authoritative_value"), aux.get("rumor_value")
        asrc, rsrc = aux.get("authoritative_source"), aux.get("rumor_source")

        # I0 矛盾可溯源:回 ws.conflicts 查回本题矛盾(与 gt() 同口径)
        c = next((c for c in (getattr(ws, "conflicts", None) or [])
                  if c["entity"] == ent and c["field"] == fld and c.get("session") == s), None)
        if c is None:
            return ("drop", "矛盾不可溯源:ws.conflicts 无 (ent,fld,session) 条目(order 与世界脱钩)")

        # I1 order/世界一致:aux 四值须与 ws.conflicts 一致(防出题旁路世界真相)
        for k in ("authoritative_value", "rumor_value", "authoritative_source", "rumor_source"):
            if _norm(aux.get(k)) != _norm(c.get(k)):
                return ("drop", f"order/世界不一致:aux.{k} 与 ws.conflicts 不符(出题旁路了世界真相)")

        # I2 规则可裁:未知裁决规则 fail-closed(闸无法证伪其良定义)
        rule = c.get("rule")
        if rule not in _KNOWN_RULES:
            return ("drop", f"未知裁决规则 '{rule}':闸无法证伪其良定义(fail-closed)")

        # ── 按 rule 分派各自的良定义不变量(当前仅 source_reliability;加规则 = 加一个分支)──
        if rule == "source_reliability":
            # I3 真矛盾
            if _norm(auth) == _norm(rumor):
                return ("drop", f"假矛盾:权威值与传闻值相等({auth!r}),无冲突")

            # I8 主体可命名:主体专名不得 == 任一隐藏值(auth/rumor)。否则题面要【既点主语又藏该值】,
            #    phrase 只能把主语降级成"他/该案"代词 → 题面悬空、无法作答(裸代词病题)。源头已由
            #    inject_conflicts 排己防生成,此处 fail-closed 兜住(将来改 inject / 加裁决规则也不漏)。
            if _norm(ent) in (_norm(auth), _norm(rumor)):
                return ("drop", f"主体不可命名:主体专名 {ent!r} 撞隐藏值(auth={auth!r}/rumor={rumor!r}),题面会失去指代")

            # I4 唯一合法权威解:世界在该 session 的 canonical = 唯一、合法、== 权威值(canonical 当裁判)
            canon = self._canonical_at(ws, ent, fld, s)
            if canon in (INVALID, INSUFFICIENT, None, ""):
                return ("drop", f"权威解不合法:session {s} 世界 canonical={canon!r}(已停用/从未出现)")
            if _norm(canon) != _norm(auth):
                return ("drop", f"权威解不唯一:session {s} canonical={canon!r} ≠ 权威值 {auth!r}")

            # I5 gold 锚定正确:order.gt 与 gt() 重算都 == 权威值(复用护城河,确定性)
            gt_recomputed = self.gt(ws, order)
            if _norm(order.get("gt")) != _norm(auth) or _norm(gt_recomputed) != _norm(auth):
                return ("drop", f"gold 锚错:gt={order.get('gt')!r} ≠ 该 session 权威值 {auth!r}(可能锚到传闻或别周)")

            # I6 可靠度严格可分:权威源档位严格 > 传闻源档位(两源都须登记,否则 fail-closed)
            ra, rr = reliability_rank(asrc), reliability_rank(rsrc)
            if ra is None or rr is None or not (ra > rr):
                return ("drop", f"可靠度不可分:权威源 '{asrc}' 未严格高于传闻源 '{rsrc}'(读者无从甄别)")

            # I7 传闻非等价真相:传闻值不得在该 (ent,fld) canonical 时间线任一 session 成立(源头已修,防回归)
            if self._appears_in_canonical(ws, ent, fld, rumor):
                return ("drop", f"传闻非干扰:传闻值 {rumor!r} 在 canonical 时间线确出现过(它是真值不是干扰)")

            return ("well_posed", "")

        return ("drop", f"裁决规则 '{rule}' 无对应良定义检查(fail-closed)")   # 兜底

    @staticmethod
    def _canonical_at(ws, ent, fld, s) -> str:
        """世界在该 session 的 canonical 值(复用 value_at_session + 哨兵,不碰 corpus)。"""
        tl = ws.timeline(ent, fld)
        return tl.value_at_session(s) if tl else INSUFFICIENT

    @staticmethod
    def _appears_in_canonical(ws, ent, fld, value) -> bool:
        """value 是否在该 (ent,fld) canonical 时间线的【任一】有效取值里出现过(_norm 比较,复用 set_values)。"""
        tl = ws.timeline(ent, fld)
        if not tl:
            return False
        return any(_norm(v) == _norm(value) for (_s, _d, v) in tl.set_values())

    def intent(self, o: dict) -> tuple[str, list]:
        ent, fld, aux = o.get("entity", ""), o.get("field", ""), (o.get("aux") or {})
        # ★FixD:锁周——官方值会跨周漂移,题面必须锚到【矛盾发生那一周】,否则"官方值"= 哪周不唯一,
        #   合理系统答"最新官方值"会被误判(D 首点:Q117/136 等)。gold = 该周官方值,与此锚一致。
        sess = aux.get("session")
        unit = aux.get("time_unit") or "周"
        wk = f"截至第{week_label(sess)}{unit}，" if sess is not None else ""
        s = (f"{wk}关于【{ent}】的「{fld}」，来自【{aux.get('authoritative_source', '正式记录')}】"
             f"和【{aux.get('rumor_source', '小道消息')}】的说法不一致。"
             f"按来源可靠度裁决，该时点「{fld}」的最终取值是什么？请回答具体值。")
        hide = [str(aux.get("authoritative_value")), str(aux.get("rumor_value"))]   # 两值都在语料,题面不剧透
        return s, hide


    def ground(self, order, evidence_docs, all_signal_text=""):
        """接地(§G.5):权威值就近实体(信号文档)且 小道值现于矛盾文档——两边都渲了才算真有冲突。"""
        from pipeline.grounding import attributed
        ent, aux = order.get("entity", ""), (order.get("aux") or {})
        auth, rumor = aux.get("authoritative_value"), aux.get("rumor_value")
        sig = [d["content"] for d in evidence_docs if not d["is_conflict"]]
        conf = [d["content"] for d in evidence_docs if d["is_conflict"]]
        ok_a = attributed(auth, ent, sig)
        ok_r = attributed(rumor, ent, conf or sig)
        if ok_a and ok_r:
            return ("grounded", "权威值+小道值两边都渲了")
        miss = []
        if not ok_a:
            miss.append(f"权威值'{auth}'未就近「{ent}」(信号文档)")
        if not ok_r:
            miss.append(f"小道值'{rumor}'未现于矛盾文档")
        return ("drop", "; ".join(miss))


# 自检:python -m pipeline.lines.L5_conflict
if __name__ == "__main__":
    import sys
    from pipeline.world_state import assemble_world
    line = ConflictLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 世界:3 部门,负责人(person)各有取值 → 值池够供"张冠李戴"的小道错误值
    table = {"entities": [
        {"name": "搜索部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "李娜"}, {"session": 3, "value": "王强"}]}}},
        {"name": "推荐部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "周明"}, {"session": 2, "value": "赵岩"}]}}},
        {"name": "风控部", "fields": {"负责人": {"type": "stable", "value": "孙宇"}}},
    ], "absent_fields": ["季度营收"]}
    ws, _ = assemble_world(table)
    profile = {"field_schema": [{"name": "负责人", "kind": "person"}]}
    pool = {"李娜", "王强", "周明", "赵岩", "孙宇"}

    note = line.prepare(ws, profile)
    ck("prepare 注入了矛盾", len(ws.conflicts) >= 1)
    ck("prepare 返回日志含'冲突注入'", isinstance(note, str) and "冲突注入" in note)
    ck("prepare 幂等(再调不叠加)", line.prepare(ws, profile) is None and len(ws.conflicts) >= 1)
    ck("canonical 时间线未被污染(负责人仍是真值)",
       _norm(ws.timeline("搜索部", "负责人").latest_valid()) == _norm("王强"))

    orders = line.enumerate(ws)
    ck("枚举数 == 矛盾条数", len(orders) == len(ws.conflicts))
    ck("订单都是 L5_conflict", all(o["capability"] == "L5_conflict" for o in orders))
    for c in ws.conflicts:
        ck(f"[{c['entity']}.{c['field']}] 小道值≠权威值(真矛盾)",
           _norm(c["rumor_value"]) != _norm(c["authoritative_value"]))
        ck(f"[{c['entity']}.{c['field']}] 小道值是池内真实取值(张冠李戴,非瞎编)", c["rumor_value"] in pool)
        ck(f"[{c['entity']}.{c['field']}] gt=权威值(信官方)", _norm(c["gt"]) == _norm(c["authoritative_value"]))
        own_c = {_norm(v) for (_s, _d, v) in ws.timeline(c["entity"], c["field"]).set_values() if v}
        ck(f"[{c['entity']}.{c['field']}] 小道值非本实体别周真值(排己,防矛盾退化为合法演化)",
           _norm(c["rumor_value"]) not in own_c)

    # ★护城河校验闸:gt() 从 ws.conflicts 重算 == enumerate 烘焙
    ck("校验闸:gt() 重算 == 烘焙", all(line.gt(ws, o) == o["gt"] for o in orders))

    # 序列化往返:conflicts 须随世界存活(corpus stage 从盘读回 ws 才能渲小道文档)
    ws2 = WorldState.from_dict(ws.to_dict())
    ck("conflicts 过序列化往返且 gt 仍对",
       len(ws2.conflicts) == len(ws.conflicts) and line.gt(ws2, orders[0]) == orders[0]["gt"])

    # 出题不泄漏:权威值 + 小道值都不进题面
    intent, hide = line.intent(orders[0])
    c0 = ws.conflicts[0]
    ck("intent 隐藏权威值+小道值", c0["authoritative_value"] in hide and c0["rumor_value"] in hide)
    ck("题面不含两值", c0["authoritative_value"] not in intent and c0["rumor_value"] not in intent)
    ck("题面点明来源可靠度并要求具体值",
       "可靠度" in intent and c0["authoritative_source"] in intent
       and c0["rumor_source"] in intent and "具体值" in intent)
    # ★FixD:题面必须锚到矛盾那一周(防"未锁周 + 官方值漂移"→合理系统答最新官方被误判)
    from pipeline.world_state import week_label as _wl
    ck("FixD:L5 题面带周锚(截至第N周)",
       all(f"第{_wl(o['aux'].get('session'))}周" in line.intent(o)[0] for o in orders if o['aux'].get('session') is not None))

    # ════════════════════════════════════════════════════════════════════════
    # ★边 A 闸 well_posed(I0–I7)离线自检(设计见 docs/anchors/edge_a/L5_well_posed.md §8)
    #   造小 WorldState + 手写 ws.conflicts + 构 order(模拟 enumerate 输出),断言返回值 + reason 命中。
    #   零 corpus、零 LLM、确定性。well-posed 必过;ill-posed 各触发对应不变量 drop。
    # ════════════════════════════════════════════════════════════════════════
    def _mk(ws_table, conflicts, order):
        w, _ = assemble_world(ws_table)
        w.conflicts = conflicts
        return ConflictLine().well_posed(order, w)

    def _conf(ent, fld, s, auth, rumor, asrc=AUTHORITATIVE_SRC, rsrc=RUMOR_SRCS[0]):
        return {"entity": ent, "field": fld, "session": s, "date": "2025-01-01",
                "authoritative_value": auth, "authoritative_source": asrc,
                "rumor_value": rumor, "rumor_source": rsrc, "rule": "source_reliability", "gt": auth}

    def _order(ent, fld, s, auth, rumor, asrc=AUTHORITATIVE_SRC, rsrc=RUMOR_SRCS[0], gt=None, rule="source_reliability"):
        return {"line": "L5_conflict", "capability": "L5_conflict", "entity": ent, "field": fld,
                "gt": auth if gt is None else gt,
                "aux": {"session": s, "rule": rule,
                        "authoritative_value": auth, "authoritative_source": asrc,
                        "rumor_value": rumor, "rumor_source": rsrc}}

    # 良定义底座:搜索部.负责人 s0=李娜→s3=王强;周明是推荐部的人(本时间线从没出现)= 真干扰
    wp_table = {"entities": [
        {"name": "搜索部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "李娜"}, {"session": 3, "value": "王强"}]}}},
        {"name": "推荐部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "周明"}, {"session": 2, "value": "赵岩"}]}}},
    ]}

    def ckwp(name, got, want_status, want_kw=""):
        st, rs = got
        ck(f"[well_posed] {name}", st == want_status and (want_kw in rs))

    # 用例 1:well-posed(canonical@s3=王强=权威,传闻=周明跨实体,官方源 vs 群聊源)→ 全过
    ckwp("1 well-posed 全过",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "周明")],
             _order("搜索部", "负责人", 3, "王强", "周明")), "well_posed", "")
    # 用例 2:假矛盾 auth==rumor → I3
    ckwp("2 假矛盾(I3)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "王强")],
             _order("搜索部", "负责人", 3, "王强", "王强")), "drop", "假矛盾")
    # 用例 3:可靠度不可分(asrc 与 rsrc 都填 RUMOR_SRCS[0],同档)→ I6
    ckwp("3 可靠度不可分(I6)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "周明", asrc=RUMOR_SRCS[0], rsrc=RUMOR_SRCS[0])],
             _order("搜索部", "负责人", 3, "王强", "周明", asrc=RUMOR_SRCS[0], rsrc=RUMOR_SRCS[0])), "drop", "可靠度不可分")
    # 用例 4:权威解不合法(矛盾挂在字段已 EXPIRE 的 session,value_at_session=INVALID)→ I4
    wp_expire = {"entities": [{"name": "搜索部", "fields": {"负责人": {"type": "evolving",
        "trajectory": [{"session": 0, "value": "李娜"}, {"session": 3, "value": "王强"},
                       {"session": 5, "value": None}]}}},  # s5 EXPIRE → s5 canonical=INVALID
        {"name": "推荐部", "fields": {"负责人": {"type": "evolving",
            "trajectory": [{"session": 0, "value": "周明"}]}}}]}
    ckwp("4 权威解不合法/已EXPIRE(I4)",
         _mk(wp_expire, [_conf("搜索部", "负责人", 5, "王强", "周明")],
             _order("搜索部", "负责人", 5, "王强", "周明")), "drop", "权威解不合法")
    # 用例 5:gold 锚错(order.gt 填成 rumor 值)→ I5
    ckwp("5 gold 锚错(I5)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "周明")],
             _order("搜索部", "负责人", 3, "王强", "周明", gt="周明")), "drop", "gold 锚错")
    # 用例 6:传闻非干扰(rumor=李娜,搜索部.负责人 s0 的真值,本实体别周值)→ I7
    ckwp("6 传闻非干扰/本实体别周真值(I7)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "李娜")],
             _order("搜索部", "负责人", 3, "王强", "李娜")), "drop", "传闻非干扰")
    # 用例 7:未知规则 rule=recency → I2(fail-closed)
    c_rec = _conf("搜索部", "负责人", 3, "王强", "周明"); c_rec["rule"] = "recency"
    ckwp("7 未知规则 recency(I2 fail-closed)",
         _mk(wp_table, [c_rec], _order("搜索部", "负责人", 3, "王强", "周明", rule="recency")), "drop", "未知裁决规则")
    # 补:I0 溯源(order 的 session 在 ws.conflicts 查不回)→ drop
    ckwp("8 矛盾不可溯源(I0)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "周明")],
             _order("搜索部", "负责人", 4, "王强", "周明")), "drop", "不可溯源")
    # 补:I1 一致(aux 旁路世界:aux.rumor_value 与 ws.conflicts 不符)→ drop
    ckwp("9 order/世界不一致(I1)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "周明")],
             _order("搜索部", "负责人", 3, "王强", "赵岩")), "drop", "order/世界不一致")
    # 补:I8 主体不可命名(传闻值 == 主体专名)→ drop(根治"他的督导合伙人"类悬空代词)
    ckwp("10 主体不可命名/传闻==主体(I8)",
         _mk(wp_table, [_conf("搜索部", "负责人", 3, "王强", "搜索部")],
             _order("搜索部", "负责人", 3, "王强", "搜索部")), "drop", "主体不可命名")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L5_conflict self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
