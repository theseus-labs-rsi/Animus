"""
pipeline.lines.L4_preference —— L4 偏好隐式产线(单能力线:L4_preference)。

能力:从【散落的每周选择行为】反推那个【从未明说】的稳定偏好(= 选择众数)。
签名:故意让【众数 ≠ 最近一次】——答"最近一次"(recency)= 错,逼跨文档聚合。
设计:docs/anchors/L4_preference_design.md(按"被坑5波后"纪律,gold出厂即类别值/闸条实测定标/无补丁)。

基质来源:白皮书 domain_profile.preference_axis = {field, options}(议会出,域无关;prepare 读它确定性构造有偏选择流)。
"""
from __future__ import annotations
from collections import Counter
import random

from pipeline.lines.base import ProductionLine, Order, field_kind, interrogative
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, _date_of, _norm

K_MIN = 5        # 至少 5 期才算"稳定偏好"(非巧合)
MARGIN = 2       # 众数须领先亚军 ≥2(够读者稳定聚合;★v0 默认,落地须真实数据定标)
RATIO = 0.6      # 偏好项占比(prepare 构造用)


def _mode(values: list) -> tuple:
    """众数 + 计数:返回 (mode, n_mode, n_runner, unique)。unique=是否唯一众数。"""
    c = Counter(_norm(v) for v in values if v not in (None, ""))
    if not c:
        return (None, 0, 0, False)
    ranked = c.most_common()
    n_mode = ranked[0][1]
    n_runner = ranked[1][1] if len(ranked) > 1 else 0
    unique = (n_runner < n_mode)                    # 无并列冠军
    # 取回原始(非归一)众数串:第一个 _norm 命中 ranked[0][0] 的原值
    mode_norm = ranked[0][0]
    mode_raw = next((v for v in values if _norm(v) == mode_norm), mode_norm)
    return (mode_raw, n_mode, n_runner, unique)


def _build_choice_seq(options: list, n: int, seed: int, pref_idx: int = 0) -> tuple:
    """确定性构造有偏选择流:偏好项 pref≈RATIO·n、其余打散、★末位≠pref(抗 recency)。
    保证【唯一众数 + 领先 ≥MARGIN(n 够大时)+ latest≠pref】= 源头即良定义。
    ★pref_idx:偏好项【跨实体轮转】(患者 i → 第 i%k 个选项),否则全注 options[0] → 一组题 gold 全同、
      零区分度(run110317 盲审实锤:14 题全=门诊复诊)。轮转让被问者众数分散、≠全局众数。"""
    rng = random.Random(seed)
    pref = options[pref_idx % len(options)]
    others = [o for o in options if o != pref] or options
    n_pref = max(MARGIN + 1, round(RATIO * n))
    n_pref = min(n_pref, n - 1)                      # 至少留 1 期给非 pref(保证末位可≠pref)
    seq = [pref] * n_pref + [others[i % len(others)] for i in range(n - n_pref)]
    rng.shuffle(seq)
    if seq[-1] == pref:                              # ★末位≠pref
        for j in range(len(seq) - 2, -1, -1):
            if seq[j] != pref:
                seq[-1], seq[j] = seq[j], seq[-1]
                break
    return seq, pref


class PreferenceLine(ProductionLine):
    id = "L4_preference"
    title = "偏好隐式"
    memory = "从散落选择行为反推稳定偏好(从不明说)"
    gt_substrate = "周度有偏选择流 + 众数(代码可算)"
    implemented = True
    requires: list[str] = ["preference_axis"]       # 需白皮书给【选择/偏好维度】

    CHOICE_FIELD_TAG = "倾向"                         # 注入字段名后缀(与原字段区分)

    def _choice_field(self, field: str) -> str:
        """注入字段名:轴名 + TAG;★轴名本身已以 TAG 结尾则不再叠(014559:轴'本期处理策略倾向'
        被拼成'…倾向倾向',渲染 LLM 把叠词塌缩回单'倾向',与世界原字段混同 → Q30 类污染)。
        配套:central_office 已把轴字段(±'倾向'变体)从 field_schema 剔除,世界生成不再造竞争时间线。"""
        return field if field.endswith(self.CHOICE_FIELD_TAG) else f"{field}{self.CHOICE_FIELD_TAG}"

    def _axis(self, profile: dict):
        ax = (profile or {}).get("preference_axis") or {}
        field = ax.get("field")
        options = [o for o in (ax.get("options") or []) if o]
        # ★表面互不近似(坑#7):选项去主干塌缩
        from pipeline.world_state import _strip_disambig
        seen, opts = set(), []
        for o in options:
            b = _strip_disambig(o)
            if b not in seen:
                seen.add(b); opts.append(o)
        return field, opts

    def _typed_owner(self, ws, profile: dict, field: str) -> str | None:
        """返回显式蓝图中偏好字段的唯一 owner；legacy 世界返回 None。"""
        blueprint = getattr(ws, "world_blueprint", None) or {}
        if not blueprint or blueprint.get("legacy_adapter"):
            return None
        owners = [t.get("id") for t in blueprint.get("entity_types", [])
                  if any(f.get("name") == field for f in (t.get("fields") or []))]
        requested = ((profile or {}).get("preference_axis") or {}).get("entity_type")
        if requested in owners:
            return requested
        return owners[0] if len(owners) == 1 else ""

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        field, opts = self._axis(profile)
        if not field or len(opts) < 2:
            return False, "白皮书无 preference_axis(或选项<2),无偏好维度可考"
        if (ws.n_sessions or 0) < K_MIN:
            return False, f"周数 {ws.n_sessions} < K_MIN({K_MIN}),不够推稳定偏好"
        owner = self._typed_owner(ws, profile, field)
        if owner == "":
            return False, f"typed blueprint 中偏好字段「{field}」无唯一 entity_type owner"
        if owner:
            candidates = [flds.get(field) for entity, flds in ws.entities.items()
                          if ws.entity_types.get(entity) == owner and flds.get(field)]
            if not any(len(tl.set_values()) >= K_MIN for tl in candidates):
                return False, f"typed world 的 {owner}.{field} 没有足够自然选择记录，能力线不得事后注入"
        return True, f"偏好轴「{field}」{len(opts)} 选项 × {ws.n_sessions} 周"

    def prepare(self, ws, profile: dict):
        """注入【周度有偏选择流】到主体实体(非人员:有≥2 字段者),字段名=「{轴}{倾向}」。幂等。"""
        blueprint = getattr(ws, "world_blueprint", None) or {}
        if blueprint and not blueprint.get("legacy_adapter"):
            return None                              # typed world 冻结；L4 只能读取自然存在的偏好轨迹
        field, opts = self._axis(profile)
        if not field or len(opts) < 2:
            return None
        n = ws.n_sessions or 0
        if n < K_MIN:
            return None
        # 若世界蓝图已把重复选择声明为真实字段，L4 直接骑这条字段并确定性整形；
        # 只有 legacy/蓝图未声明时才创建带 TAG 的派生字段，避免同一偏好语义出现双流。
        declared_in_world = ws.has_field(field)
        cf = field if declared_in_world else self._choice_field(field)
        # ★老产物守卫(刀1审计):老 run 的世界可能带叠词字段「{cf}倾向」(旧 TAG 拼接产物)。新代码续跑老 run
        #   会静默换字段名 → L4 退化/双流并存。检测到即显式告警(不迁移,建议全新 run)。
        legacy = f"{cf}{self.CHOICE_FIELD_TAG}"
        if any(legacy in flds for flds in ws.entities.values()):
            return f"  ⚠L4 检测到老叠词字段「{legacy}」(旧版产物):新命名规则不兼容续跑,跳过注入——请用全新 run"
        if declared_in_world:
            targets = [e for e, flds in ws.entities.items() if cf in flds]
        else:
            targets = [e for e, flds in ws.entities.items() if len(flds) >= 2 and cf not in flds]  # 主体实体(避开单字段人员)
        if not targets:
            return None
        for i, ent in enumerate(targets):
            seq, _pref = _build_choice_seq(opts, n, seed=20260608 + i, pref_idx=i)   # ★pref 跨实体轮转,防 gold 全同退化
            prev = None
            ops = []
            for s, v in enumerate(seq):              # 每期都表态(复选)→ 散落证据,逼聚合
                ops.append(Op(s, ws.date_of_session(s), SET if prev is None else UPDATE, v, prev))
                prev = v
            ws.entities[ent][cf] = Timeline(ops)
        action = "整形" if declared_in_world else "注入"
        return f"  ★偏好{action}:{len(targets)} 个「{cf}」有偏选择流(众数=偏好,末位≠众数·抗recency;{len(opts)}选项×{n}期)"

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        profile = (wp or {}).get("domain_profile", {})
        field, opts = self._axis(profile)
        if not field:
            return []
        owner = self._typed_owner(ws, profile, field)
        if owner == "":
            return []
        cf = field if owner or ws.has_field(field) else self._choice_field(field)
        out = []
        for ent, flds in ws.entities.items():
            if owner and ws.entity_types.get(ent) != owner:
                continue
            tl = flds.get(cf)
            if not tl:
                continue
            vals = [v for (_s, _d, v) in tl.set_values()]
            mode, n_mode, n_runner, unique = _mode(vals)
            latest = vals[-1] if vals else None
            if not (unique and len(vals) >= K_MIN and (n_mode - n_runner) >= MARGIN and _norm(mode) != _norm(latest)):
                continue                              # 源头只产良定义(闸只兜底)
            ev = sorted({s for (s, _d, v) in tl.set_values() if _norm(v) == _norm(mode)})
            out.append({"line": self.id, "capability": "L4_preference", "entity": ent, "field": cf,
                        "gt": mode, "evidence_sessions": ev,
                        "aux": {"options": opts, "n_total": len(vals), "n_pref": n_mode,
                                "n_runner": n_runner, "latest": latest,
                                "ans_kind": field_kind(cf, mode, (wp or {}).get("domain_profile", {}))}})
            if len(out) >= target:
                break
        return out

    def gt(self, ws, o: dict):
        """护城河:从世界复算众数,与烘焙相等(过校验闸)。"""
        tl = ws.entities.get(o["entity"], {}).get(o["field"])
        vals = [v for (_s, _d, v) in tl.set_values()] if tl else []
        return _mode(vals)[0]

    def intent(self, o: dict) -> tuple[str, list]:
        ent, fld, aux, gt = o.get("entity", ""), o.get("field", ""), (o.get("aux") or {}), o.get("gt")
        q = interrogative(aux.get("ans_kind"))        # 坑#5:疑问词走 kind,不写死
        n = aux.get("n_total", "")
        s = (f"综合【{ent}】这 {n} 期「{fld}」的记录,它【一贯/整体】更倾向于哪一种?{q} "
             f"——看整体倾向、别只看最近一次(最近一次未必代表偏好);答案是最常出现的那一种。")
        hide = [str(gt)]                              # 只藏"偏好结论";每期具体值留在语料(边B证据)
        return s, hide

    def ground(self, order, evidence_docs, all_signal_text="") -> tuple:
        """边B:偏好众数 `pref` 须在 ≥K_MIN 个不同 session 的文档就近实体出现(散落证据齐,读者才可聚合)。"""
        from pipeline.grounding import attributed
        ent, pref = order.get("entity", ""), str(order.get("gt"))
        by_sess = {}
        for d in evidence_docs:
            by_sess.setdefault(d["session"], []).append(d["content"])
        hit = sum(1 for s, docs in by_sess.items() if attributed(pref, ent, docs))
        if hit >= K_MIN:
            return ("grounded", f"偏好「{pref}」在 {hit} 周就近「{ent}」(散落证据足,可聚合)")
        return ("drop", f"偏好「{pref}」仅 {hit} 周就近「{ent}」(<{K_MIN},证据不足以聚合)")

    def well_posed(self, order: dict, ws) -> tuple:
        """边A:偏好唯一且可从证据稳定推出。WP1唯一众数/WP2领先够/WP3抗recency/WP4样本足/WP5复算。
        源头 enumerate 已只产良定义 → 本闸≈防回归(新鲜 order 应≈0 触发)。"""
        ent, fld, gt = order.get("entity", ""), order.get("field", ""), order.get("gt")
        tl = ws.entities.get(ent, {}).get(fld)
        if not tl:
            return ("drop", f"well_posed:L4 实体「{ent}」无「{fld}」选择流")
        vals = [v for (_s, _d, v) in tl.set_values()]
        if len(vals) < K_MIN:                                              # WP4
            return ("drop", f"well_posed:L4 样本不足 {len(vals)}<{K_MIN}")
        mode, n_mode, n_runner, unique = _mode(vals)
        if not unique:                                                     # WP1
            return ("drop", "well_posed:L4 众数并列(偏好不唯一)")
        if (n_mode - n_runner) < MARGIN:                                   # WP2(★MARGIN 须真实数据定标)
            return ("drop", f"well_posed:L4 领先不足({n_mode}-{n_runner}<{MARGIN}),读者难稳定聚合")
        if _norm(mode) == _norm(vals[-1]):                                 # WP3 抗 recency
            return ("drop", "well_posed:L4 众数==最近一次(可被 recency/KU 直接答中,退化)")
        if _norm(gt) != _norm(mode):                                       # WP5 护城河
            return ("drop", f"well_posed:L4 gold≠复算众数(gold={gt} 世界={mode})")
        return ("well_posed", "")


# 自检:python -m pipeline.lines.L4_preference
if __name__ == "__main__":
    import sys
    line = PreferenceLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # 合成世界:3 个主体实体(各≥2字段)+ 偏好轴「周会形式」∈{线上,线下,混合}
    prof = {"field_schema": [{"name": "周会形式倾向", "kind": "category"}],
            "preference_axis": {"field": "周会形式", "options": ["线上", "线下", "混合"]}}
    ents = {}
    for nm in ("研发一部", "研发二部", "数据部"):
        ents[nm] = {"负责人": Timeline([Op(0, _date_of(0), SET, "张三", None)]),
                    "P0率": Timeline([Op(0, _date_of(0), SET, "1%", None), Op(2, _date_of(2), UPDATE, "2%", "1%")])}
    ws = WorldState(ents, n_sessions=8)

    ck("feasible:有 preference_axis + 够周数", line.feasible(ws, prof)[0] is True)
    note = line.prepare(ws, prof)
    ck("prepare 注入了选择流字段", isinstance(note, str) and "偏好注入" in note)
    ck("prepare 幂等(再调不叠加)", line.prepare(ws, prof) is None or all(
        len([f for f in flds if f.endswith("倾向")]) == 1 for flds in ws.entities.values() if any(f.endswith("倾向") for f in flds)))

    orders = line.enumerate(ws, wp={"domain_profile": prof})
    ck("枚举出≥1 道偏好题", len(orders) >= 1)
    o0 = orders[0]
    ck("gold = 干净类别值(裸 str,无 session/周/dict —— 消费者无可误读)",
       isinstance(o0["gt"], str) and not any(k in str(o0["gt"]) for k in ("session", "周", "week", "{")))
    ck("校验闸:gt() 复算 == 烘焙", all(line.gt(ws, o) == o["gt"] for o in orders))
    ck("★签名:众数 ≠ 最近一次(抗 recency)", all(_norm(o["gt"]) != _norm(o["aux"]["latest"]) for o in orders))
    ck("领先 ≥MARGIN", all((o["aux"]["n_pref"] - o["aux"]["n_runner"]) >= MARGIN for o in orders))
    # ★防退化(run110317 实锤):pref 跨实体轮转 → 多实体 gold 必须分散,不准全同(零区分度)
    golds = {_norm(o["gt"]) for o in orders}
    ck("★轮转防退化:多实体 gold 分散(3实体×3选项 → ≥2 个不同)", len(golds) >= min(2, len(orders)) and (len(golds) >= 2 if len(orders) >= 2 else True))

    intent, hide = line.intent(o0)
    ck("题面强制聚合(含'整体/一贯'+'别只看最近')", ("整体" in intent or "一贯" in intent) and "最近" in intent)
    ck("题面疑问词走 kind(category→是什么)", "是什么" in intent)
    ck("答案不进题面", o0["gt"] not in intent)

    # well_posed:正例过
    ck("well_posed:良定义偏好题 → 过", all(line.well_posed(o, ws)[0] == "well_posed" for o in orders))
    # well_posed:手构 ill-posed(并列众数)→ drop
    tie = WorldState({"X": {"a": Timeline([Op(0, _date_of(0), SET, "1", None)]),
                           "选倾向": Timeline([Op(s, _date_of(s), SET if s == 0 else UPDATE,
                                              ("线上" if s % 2 == 0 else "线下"), None) for s in range(8)])}}, n_sessions=8)
    tie_o = {"line": "L4_preference", "capability": "L4_preference", "entity": "X", "field": "选倾向",
             "gt": "线上", "aux": {}}
    ck("well_posed:并列众数 → drop", line.well_posed(tie_o, tie)[0] == "drop")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L4_preference self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
