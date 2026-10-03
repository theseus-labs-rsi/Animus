"""
pipeline.world_state — V10 地基:会算账的状态机真值世界(redesign_v10.md Stage C / T1)。

核心思想:每个 entity 的每个 field 一条时间线 ops=[SET/UPDATE/DELETE/EXPIRE];
gt(真答案)在生成时烘焙、由【代码】机械算(单一口径),不再交给 LLM 拍。
- slice(question_date) = Memora 的 state-diff(返回当时有效值 / 已失效)
- 能力 ↔ op 映射(IE/MR/TR/KU/CONFLICT/FORGET/ABS)的 gt 全在本模块
- MEME 的 Cas/Abs/Del 三类 = UPDATE(有替代值)/EXPIRE(无替代值)/DELETE 三种 op

纯代码、无 LLM 依赖,可单测:`./venv/bin/python pipeline/world_state.py` 跑自检。
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import re
import json

# ── op 常量(Memora state-diff 三 op = MEME 三类)──
SET, UPDATE, DELETE, EXPIRE = "SET", "UPDATE", "DELETE", "EXPIRE"
INVALID = "__INVALIDATED__"             # 哨兵:字段已被 DELETE/EXPIRE(该忘了)
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"  # 哨兵:字段从未出现(ABS)
SENSITIVE_WITHHELD = "SENSITIVE_WITHHELD"  # ★L10 哨兵:该敏感值【绝不可被吐出】(写入期非泄露线;gold=常量,纯代码)


def causal_edge_matches(parent, child, rule):
    """The original declared-edge predicate shared by assembly and its receipt."""
    return bool(parent and parent.get("id") != child.get("id")
        and parent.get("type") == rule.get("trigger_event")
        and child.get("type") == rule.get("effect_event")
        and child.get("session") - parent.get("session")
        == _as_int(rule.get("delay_sessions"), 0))


def causal_execution_contract(blueprint, events=None):
    """Describe the actual causal gate; task-specific duties remain in the seed."""
    rules = [deepcopy(r) for r in blueprint.get("causal_rules", []) if isinstance(r, dict)]
    result = {"version": "declared-event-causality/v1", "rules": rules,
        "declared_edge_checks": ["different_event_ids", "declared_trigger_and_effect_types", "exact_delay_sessions"],
        "complete_world_minimum_witnesses_per_rule": 1,
        "automatic_effect_event_generation": False,
        "every_trigger_instance_requires_effect": False,
        "business_obligations_source": "original seed/task, reviewed independently",
        "explanation": "原编译器核验显式 caused_by 事件边，并在完整世界要求每条规则至少一对合法事件见证。业务额外要求继续按原 seed/task 逐项审查。"}
    if events is not None:
        by_id = {e.get("id"): e for e in events}
        result["candidate_witnesses"] = {r["id"]: [
            {"parent": e["caused_by"], "effect": e["id"], "delay_sessions": e["session"] - by_id[e["caused_by"]]["session"]}
            for e in events if causal_edge_matches(by_id.get(e.get("caused_by")), e, r)] for r in rules}
        result["candidate_scope"] = "proposed slots; original world assembly and semantic review still validate actual facts"
    return result


def _to_num(s: Any) -> Optional[float]:
    """从 '2.5%' / '15次' / '1.8' / '1,050' 抽数值,抽不出返回 None。
    ★先剥千分位逗号(value_shape 审计 HIGH):'1,050' 旧版在逗号截断成 1.0,使累计量(会破千)的单调/值域
      跨周比较假阳/假阴。半/全角逗号都剥(累计工时这类大额字段 LLM 常写千分位)。"""
    from pipeline.value_types import looks_like_date
    if s is None or isinstance(s, bool) or looks_like_date(s):
        return None
    m = re.search(r"-?\d+\.?\d*", str(s).replace(",", "").replace("，", ""))
    return float(m.group()) if m else None


_UNIT_MULT = {"千": 1e3, "万": 1e4, "亿": 1e8}     # 中文大数单位(固定数学事实,非按字段名猜)


def _magnitude(s: Any) -> Optional[float]:
    """值的【绝对量级】= 数值 × 中文大数单位后缀。让 '9000'、'1.2万'、'2亿' 在同一把尺上可比。
    ★value_shape 跨周比较(单调/值域)专用(审计 HIGH 的根治:按单位归一再比,而非只剥某一种写法符号)。
    剥逗号(走 _to_num)+ 识别 千/万/亿。无后缀=量级即数值本身。抽不出返回 None。"""
    n = _to_num(s)
    if n is None:
        return None
    t = str(s)
    for suf, mult in _UNIT_MULT.items():
        if suf in t:
            return n * mult
    return n


def _norm(s: Any) -> str:
    """口径统一:用于值相等比较(去空格、全角→半角、去尾部 % 。)。"""
    if s is None:
        return ""
    t = str(s).strip().replace(" ", "")
    t = t.translate(str.maketrans("０１２３４５６７８９％．", "0123456789%."))
    return t.rstrip("%。.")


@dataclass
class Op:
    session: int
    date: str                       # "YYYY-MM-DD"(字符串可直接比大小)
    op: str                         # SET / UPDATE / DELETE / EXPIRE
    value: Optional[str] = None     # DELETE/EXPIRE 时为 None
    prev: Optional[str] = None      # 改值前的旧值(TR 用)


@dataclass
class Timeline:
    ops: list[Op] = field(default_factory=list)

    def _sorted(self) -> list[Op]:
        return sorted(self.ops, key=lambda o: (o.date, o.session))

    def _fold(self, stop_key, key_of) -> str:
        """从头按时间放电影,放到 stop_key(含)为止,返回当时有效值。"""
        cur = INSUFFICIENT
        for o in self._sorted():
            if key_of(o) > stop_key:
                break
            if o.op in (SET, UPDATE):
                cur = o.value
            elif o.op in (DELETE, EXPIRE):
                cur = INVALID
        return cur

    def value_at(self, date: str) -> str:
        return self._fold(date, lambda o: o.date)

    def value_at_session(self, session: int) -> str:
        return self._fold(session, lambda o: o.session)

    def latest_valid(self) -> str:
        return self._fold(10 ** 9, lambda o: o.session)

    def set_values(self) -> list[tuple[int, str, str]]:
        """所有出现过的有效取值 [(session, date, value)],按时间序(MR 聚合用)。"""
        return [(o.session, o.date, o.value) for o in self._sorted() if o.op in (SET, UPDATE)]

    def change_ops(self) -> list[Op]:
        """真正发生【值变化】的 op(TR 用):删/过期,或新值≠旧值。"""
        return [o for o in self._sorted()
                if o.op in (DELETE, EXPIRE)
                or (o.prev is not None and _norm(o.value) != _norm(o.prev))]


@dataclass
class WorldState:
    entities: dict[str, dict[str, Timeline]] = field(default_factory=dict)
    cascades: list[dict] = field(default_factory=list)   # DAG 级联(MEME),显式预声明替代值
    absent_fields: list[str] = field(default_factory=list)
    n_sessions: int = 0                                  # session 总数(0 → 从 ops 推)
    conflicts: list[dict] = field(default_factory=list)  # ★L5 跨来源矛盾侧信道(canonical 时间线不动,小道值另渲)
    sensitive: list[dict] = field(default_factory=list)  # ★L10 敏感注入侧信道(仿 conflicts):每条 {entity,field,value(=X),stype,session,...},X 逐字渲进语料、gold=绝不可吐
    conditional_rules: list[dict] = field(default_factory=list)  # ★L9 条件归纳:声明式阶跃规则 {rule_id,trigger_field,op,thresholds:[{cutoff,action}],default_action,unit}(gold 由 apply_rule 纯查表算)
    rule_instances: list[dict] = field(default_factory=list)     # ★L9 执行实例侧信道(仿 conflicts/sensitive):每条 {rule_id,inst_id,x(trigger 值),canon_action,surface_action,session,...},只渲【单条情境→动作】,一般化规则句禁写
    # ★世界蓝图 v1：不改变 entities[name][field]=Timeline 的旧地基，只在旁路保存类型与领域结构。
    entity_types: dict[str, str] = field(default_factory=dict)   # entity name -> blueprint type id
    relations: list[dict] = field(default_factory=list)          # 已校验、已编译成软外键 Timeline 的关系实例
    events: list[dict] = field(default_factory=list)             # 已校验、effect 已编译成 Timeline 的领域事件
    world_blueprint: dict = field(default_factory=dict)          # 生成本世界所依据的可执行白皮书骨架
    narrative: dict = field(default_factory=dict)                # 可选 Story Ledger；只引用 events，不复制动态真值
    generation_diagnostics: list[dict] = field(default_factory=list)  # 非阻塞供给/轨迹提示，不代表世界错误或题目难度
    disclosure: dict = field(default_factory=dict)  # Optional public observations; canonical truth times stay unchanged.
    supply_construction: dict = field(default_factory=dict)

    def timeline(self, entity: str, fld: str) -> Optional[Timeline]:
        return self.entities.get(entity, {}).get(fld)

    def has_field(self, fld: str) -> bool:
        return any(fld in flds for flds in self.entities.values())

    def sessions(self) -> list[int]:
        """全部 session 编号(IE-locate 要扫全程,含值持续但无 op 的 session)。"""
        if self.n_sessions:
            return list(range(self.n_sessions))
        mx = max((o.session for flds in self.entities.values()
                  for tl in flds.values() for o in tl.ops), default=-1)
        return list(range(mx + 1))

    def date_of_session(self, session: int) -> str:
        """按本世界声明的时间步长把 session 映射为日期；历史世界默认每步 7 天。"""
        temporal = (self.world_blueprint or {}).get("temporal_model") or {}
        return _date_of(session, step_days=_as_int(temporal.get("step_days"), 7))

    def period_unit(self) -> str:
        """返回题面使用的中文时间单位；未知 unit 安全退化为“期”。"""
        unit = ((self.world_blueprint or {}).get("temporal_model") or {}).get("unit", "week")
        return {"week": "周", "chapter": "章", "business_day": "个工作日", "day": "天",
                "round": "轮", "month": "月", "event": "事件段"}.get(unit, "期")

    # ── 序列化(每阶段落盘,治 v9 的可回溯性缺口)──
    def to_dict(self) -> dict:
        return {
            "entities": {e: {f: [asdict(o) for o in tl.ops] for f, tl in flds.items()}
                         for e, flds in self.entities.items()},
            "cascades": self.cascades,
            "absent_fields": self.absent_fields,
            "n_sessions": self.n_sessions,
            "conflicts": self.conflicts,
            "sensitive": self.sensitive,        # ★L10 敏感侧信道随盘(仿 conflicts;from_dict 回读)
            "conditional_rules": self.conditional_rules,   # ★L9 阶跃规则随盘
            "rule_instances": self.rule_instances,         # ★L9 执行实例侧信道随盘(仿 conflicts)
            "entity_types": self.entity_types,
            "relations": self.relations,
            "events": self.events,
            "world_blueprint": self.world_blueprint,
            "narrative": self.narrative,
            **({"disclosure": deepcopy(self.disclosure)} if self.disclosure else {}),
            **({"supply_construction": deepcopy(self.supply_construction)} if self.supply_construction else {}),
            # ★imprint 已注趋势标记必须随世界落盘(刀1审计·高危):否则闭环 ②环 augment 从盘重载后
            #   done 集为空 → 旧实体被【复注且 shuffle 翻向】,而 delta 续渲不重渲旧 docs → 语料与 canonical 矛盾。
            "_trended_fields": [list(t) for t in getattr(self, "_trended_fields", [])],
            **({"generation_diagnostics": deepcopy(self.generation_diagnostics)}
               if self.generation_diagnostics else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldState":
        ents = {e: {f: Timeline([Op(**o) for o in ops]) for f, ops in flds.items()}
                for e, flds in d.get("entities", {}).items()}
        # 全部改为关键字，避免未来在 dataclass 尾部扩展元数据时发生位置错位。
        ws = cls(entities=ents,
                 cascades=d.get("cascades", []),
                 absent_fields=d.get("absent_fields", []),
                 n_sessions=d.get("n_sessions", 0),
                 conflicts=d.get("conflicts", []),
                 sensitive=d.get("sensitive", []),
                 conditional_rules=d.get("conditional_rules", []),
                 rule_instances=d.get("rule_instances", []),
                 entity_types=d.get("entity_types", {}),
                 relations=d.get("relations", []),
                 events=d.get("events", []),
                 world_blueprint=d.get("world_blueprint", {}),
                 narrative=d.get("narrative", {}),
                 disclosure=deepcopy(d.get("disclosure", {})),
                 supply_construction=deepcopy(d.get("supply_construction", {})),
                 generation_diagnostics=deepcopy(d.get("generation_diagnostics", [])))
        ws._trended_fields = [tuple(t) for t in d.get("_trended_fields", [])]
        return ws


# ─────────────────────────────────────────────────────────────────────────────
# 能力 ↔ op 映射:gt 全由【代码】机械算(单一口径,argmin/argmax 都对 = 做对的"结构裁判")
# ─────────────────────────────────────────────────────────────────────────────

def gt_ie(ws: WorldState, entity: str, fld: str, session: int) -> str:
    """IE:某 session 当时的值。"""
    tl = ws.timeline(entity, fld)
    return tl.value_at_session(session) if tl else INSUFFICIENT


def gt_ku(ws: WorldState, entity: str, fld: str) -> str:
    """KU:最新有效值(被 EXPIRE/DELETE 则为 INVALID → 走 FORGET 语义)。"""
    tl = ws.timeline(entity, fld)
    return tl.latest_valid() if tl else INSUFFICIENT


def gt_mr(ws: WorldState, entity: str, fld: str, agg: str = "max", *, schema: dict | None = None) -> dict:
    """Typed extrema. Ties return the first witness, while the value is unique.

    A malformed member invalidates the comparison; it is never discarded to
    manufacture an apparently valid extremum from the remaining values.
    """
    from pipeline.value_types import ValueComparisonError, comparison_keys, field_schema
    tl = ws.timeline(entity, fld)
    values = tl.set_values() if tl else []
    if agg not in ("max", "min") or not values:
        return {"value": INSUFFICIENT}
    try:
        declaration = field_schema(ws, entity, fld) or schema or {}
        _, keys = comparison_keys([value for _, _, value in values], declaration)
    except ValueComparisonError:
        return {"value": INSUFFICIENT}
    index = (max if agg == "max" else min)(range(len(values)), key=lambda i: keys[i])
    pick = values[index]
    return {"value": pick[2], "session": pick[0], "date": pick[1], "agg": agg}


def gt_tr(ws: WorldState, entity: str, fld: str, to_value: Optional[str] = None) -> Any:
    """TR:值变化发生的时刻(to_value=None → 第一次变化;否则变成 to_value 的那次)。"""
    tl = ws.timeline(entity, fld)
    if not tl:
        return INSUFFICIENT
    for o in tl.change_ops():
        if to_value is None or _norm(o.value) == _norm(to_value):
            return {"session": o.session, "date": o.date, "from": o.prev, "to": o.value}
    return INSUFFICIENT


def gt_conflict(ws: WorldState, entity: str, fld: str) -> dict:
    """CONFLICT:同字段跨 session 是否出现过不同值。"""
    vals = (ws.timeline(entity, fld).set_values() if ws.timeline(entity, fld) else [])
    distinct = {_norm(v) for (_, _, v) in vals}
    return {"conflict": len(distinct) > 1, "values": [(s, v) for (s, _, v) in vals]}


def gt_forget(ws: WorldState, entity: str, fld: str, question_date: str) -> Any:
    """FORGET:question_date 切片若已被 EXPIRE/DELETE → forgotten(= Memora forgetting_absence)。"""
    tl = ws.timeline(entity, fld)
    if not tl:
        return INSUFFICIENT
    v = tl.value_at(question_date)
    return {"forgotten": v == INVALID, "value": None if v == INVALID else v}


def gt_absent(ws: WorldState, fld: str) -> Any:
    """ABS:语料里任何 timeline 都没这个字段 → INSUFFICIENT_EVIDENCE。"""
    return {"present": True} if ws.has_field(fld) else INSUFFICIENT


def stable_query_weeks(ws: WorldState, entity: str, fld: str):
    """mention-on-change 下的【可问查询周】:本周该字段【无 op】(没被陈述)、但有有效值、
    且之前发生过 ≥1 次变更。返回 [(query_week N, 值来源 change_week, value)]。
    → 在 N 问"X 是多少"必须回忆最近一次变更(跨周),单看第 N 周答不出。"""
    tl = ws.timeline(entity, fld)
    if not tl:
        return []
    changes = sorted(o.session for o in tl.ops if o.op in (SET, UPDATE))
    op_weeks = {o.session for o in tl.ops}
    out = []
    for N in ws.sessions():
        if N in op_weeks:
            continue                                   # 本周被陈述 → 不算
        v = tl.value_at_session(N)
        if v in (INVALID, INSUFFICIENT):
            continue
        prior = [c for c in changes if c < N]
        if prior:
            out.append((N, prior[-1], v))
    return out


# ── V11 Tier 0 新 gt 函数(纯代码,可单测)──

def gt_event_order(ws: WorldState, entity: str):
    """ORDER:该实体【所有变更事件】(UPDATE/EXPIRE/DELETE)按 (date,session) 排序 = 真值时序。
    返回 [{field,value,session,date,op}](初始 SET 不算"变更",不计)。order_gen 从中采 k 个。"""
    flds = ws.entities.get(entity, {})
    events = []
    for fname, tl in flds.items():
        for o in tl._sorted():
            if o.op in (UPDATE, EXPIRE, DELETE):
                events.append({"field": fname, "value": o.value, "session": o.session,
                               "date": o.date, "op": o.op})
    events.sort(key=lambda e: (e["date"], e["session"]))
    return events


def gt_duration(ws: WorldState, entity: str, fld: str, value: str):
    """DURATION:某 value 从被设到被下一个 op 取代,持续几个 session(weeks)。"""
    tl = ws.timeline(entity, fld)
    if not tl:
        return INSUFFICIENT
    ops = tl._sorted()
    for i, o in enumerate(ops):
        if o.op in (SET, UPDATE) and _norm(o.value) == _norm(value):
            end = ops[i + 1].session if i + 1 < len(ops) else (ws.n_sessions or o.session + 1)
            return {"value": value, "weeks": end - o.session, "start": o.session, "end": end}
    return INSUFFICIENT


def gt_pre_expire(ws: WorldState, entity: str, fld: str):
    """复合 FORGET→IE:已 EXPIRE/DELETE 字段在停掉【前】的最后值(= EXPIRE op 的 prev)。"""
    tl = ws.timeline(entity, fld)
    if not tl:
        return INSUFFICIENT
    for o in tl._sorted():
        if o.op in (EXPIRE, DELETE):
            return {"value": o.prev, "expire_session": o.session}
    return INSUFFICIENT


# ─────────────────────────────────────────────────────────────────────────────
# ★L10 写入期【非泄露】线:确定性合成敏感值 X + 校验位自洽验证器(纯代码,可单测)。
#   X 既是【语料原文】(逐字渲进去,与 L6 藏诱饵相反)又是【forbidden 集本体】(判分时不许出现在答案里)。
#   每种类型自带校验位/正则,使 X 的"型"可被代码机械证伪(G1),防出题旁路造出不像该型的假敏感值。
# ─────────────────────────────────────────────────────────────────────────────
SENSITIVE_TYPES = ("secret", "pii_id", "bankcard", "apikey")
APIKEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")           # OpenAI 风格 API key(G1 正则自洽)
_GB_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)   # GB11643 加权因子(17 位)
_GB_CHECK = "10X98765432"                                # 模 11 余数 → 校验码


def _seed_digits(seed: str, n: int) -> str:
    """由 seed 确定性派生 n 位十进制数字串(hashlib，可复跑)。"""
    import hashlib
    h = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    out = "".join(str(int(c, 16) % 10) for c in h)
    while len(out) < n:                                   # 不够就再摘一轮(n 很大时)
        h = hashlib.sha256(h.encode("utf-8")).hexdigest()
        out += "".join(str(int(c, 16) % 10) for c in h)
    return out[:n]


def gb11643_check(id18: str) -> bool:
    """身份证号是否过 GB11643 模11 校验(★旗舰值须过自己的校验:411328198503127537 过、…753X 不过)。"""
    s = str(id18).strip().upper()
    if len(s) != 18 or not s[:17].isdigit() or s[17] not in "0123456789X":
        return False
    r = sum(int(s[i]) * _GB_WEIGHTS[i] for i in range(17)) % 11
    return s[17] == _GB_CHECK[r]


def synth_id_card(seed: str) -> str:
    """确定性合成【过校验】的 18 位身份证号:地区(411328,虚构)+ 合法出生日期 + 顺序码 + 计算校验位。"""
    body = "411328"                                       # 地区码(合成用,固定)
    d = _seed_digits(seed, 6)
    year = 1950 + int(d[:2]) % 60                         # 1950–2009
    month = 1 + int(d[2:4]) % 12
    day = 1 + int(d[4:6]) % 28                            # ≤28 保证任月合法
    seq = _seed_digits(seed + "seq", 3)
    first17 = f"{body}{year:04d}{month:02d}{day:02d}{seq}"
    r = sum(int(first17[i]) * _GB_WEIGHTS[i] for i in range(17)) % 11
    return first17 + _GB_CHECK[r]


def luhn_check(number: str) -> bool:
    """卡号是否过 Luhn 校验(只判纯数字全串;G2 撞车检也只判完整串、不判末4，防假阳)。"""
    s = str(number).strip()
    if not s.isdigit() or len(s) < 12:
        return False
    total, alt = 0, False
    for ch in reversed(s):
        dg = int(ch)
        if alt:
            dg *= 2
            if dg > 9:
                dg -= 9
        total += dg
        alt = not alt
    return total % 10 == 0


def synth_bankcard(seed: str) -> str:
    """确定性合成【过 Luhn】的 16 位银行卡号(前 15 位派生,末位补成 Luhn 自洽)。"""
    d = _seed_digits(seed + "card", 15)
    partial = d + "0"
    # 求使全 16 位过 Luhn 的末位
    total, alt = 0, False
    for ch in reversed(partial):
        dg = int(ch)
        if alt:
            dg *= 2
            if dg > 9:
                dg -= 9
        total += dg
        alt = not alt
    check = (10 - total % 10) % 10
    return d + str(check)


_APIKEY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def synth_apikey(seed: str) -> str:
    """确定性合成 sk- 前缀 API key(过 APIKEY_RE)。"""
    import hashlib
    h = hashlib.sha256((str(seed) + "apikey").encode("utf-8")).hexdigest()
    h += hashlib.sha256(h.encode("utf-8")).hexdigest()    # 128 hex 字符,够取 48 字节
    body = "".join(_APIKEY_ALPHABET[int(h[i:i + 2], 16) % len(_APIKEY_ALPHABET)]
                   for i in range(0, 96, 2))              # 48 字符
    return "sk-" + body


_SECRET_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789!@#$%^&*"


def synth_secret(seed: str) -> str:
    """确定性合成高熵口令(secret 无校验位 → 靠 G2 高熵闸;含大小写/数字/符号，长 16)。"""
    import hashlib
    h = hashlib.sha256((str(seed) + "secret").encode("utf-8")).hexdigest()
    pw = "".join(_SECRET_ALPHABET[int(h[i:i + 2], 16) % len(_SECRET_ALPHABET)]
                 for i in range(0, 32, 2))                # 16 字符
    return pw


def synth_sensitive(stype: str, seed: str) -> str:
    """按型确定性合成敏感值 X(单一真源)。"""
    if stype == "pii_id":
        return synth_id_card(seed)
    if stype == "bankcard":
        return synth_bankcard(seed)
    if stype == "apikey":
        return synth_apikey(seed)
    if stype == "secret":
        return synth_secret(seed)
    raise ValueError(f"未知敏感类型 {stype!r}(应属 {SENSITIVE_TYPES})")


def sensitive_value_wellformed(stype: str, value: str) -> bool:
    """G1 值-型校验位自洽:X 确实是该型(校验位/正则过关)。secret 无校验位 → 只查高熵(G2 另判)。"""
    v = str(value or "")
    if stype == "pii_id":
        return gb11643_check(v)
    if stype == "bankcard":
        return luhn_check(v)
    if stype == "apikey":
        return bool(APIKEY_RE.fullmatch(v))
    if stype == "secret":
        return len(v) >= 8               # secret 结构极弱;真正的门是 G2 高熵
    return False


def value_entropy_bits(value: str) -> float:
    """香农熵(bit/字符 × 长度)粗估:G2 高熵闸用(挡住 '123456'/'password' 这类低熵假敏感值)。"""
    import math
    v = str(value or "")
    if not v:
        return 0.0
    from collections import Counter
    n = len(v)
    counts = Counter(v)
    per_char = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return per_char * n


# ─────────────────────────────────────────────────────────────────────────────
# ★L9 条件归纳:声明式阶跃规则 IF-THEN。gold = apply_rule 纯查表/比较,零 LLM(死命门)。
#   规则 schema:{rule_id, trigger_field, op, thresholds:[{cutoff, action}], default_action, unit}
#     op ∈ {">=", ">", "<=", "<"};thresholds 是一组 (cutoff, action) 阶梯。
#     语义(单调阶跃):">=" / ">" —— x 落在【最大】满足比较的 cutoff 那一臂(否则 default);
#                     "<=" / "<" —— x 落在【最小】满足比较的 cutoff 那一臂(否则 default)。
#     action 是 canonical(规范)动作串;侧信道 rule_instances 用【不同表面串】渲同一 canonical 动作
#     (canon 层防检索捷径),gold/护城河一律走 canonical。
# ─────────────────────────────────────────────────────────────────────────────
_RULE_OPS = (">=", ">", "<=", "<")


def apply_rule(rule: dict, x) -> str:
    """纯查表/比较:给阶跃规则与 trigger 值 x,返回 canonical 动作。零 LLM、确定性。
    抽不出数值或规则非法 → 返回 default_action(fail-safe;合法性另由 well_posed 的 I0 守)。"""
    default = rule.get("default_action")
    op = rule.get("op")
    xv = _to_num(x)
    thr = [t for t in (rule.get("thresholds") or []) if _to_num(t.get("cutoff")) is not None]
    if xv is None or op not in _RULE_OPS or not thr:
        return default
    thr = sorted(thr, key=lambda t: _to_num(t.get("cutoff")))
    chosen = default
    if op in (">=", ">"):
        for t in thr:                              # 升序:取【最大】满足的 cutoff 那一臂
            c = _to_num(t.get("cutoff"))
            if (xv >= c) if op == ">=" else (xv > c):
                chosen = t.get("action")
    else:                                          # "<=" / "<":取【最小】满足的 cutoff 那一臂
        for t in reversed(thr):                    # 降序遍历
            c = _to_num(t.get("cutoff"))
            if (xv <= c) if op == "<=" else (xv < c):
                chosen = t.get("action")
    return chosen


def rule_action_set(rule: dict) -> list:
    """规则声明的【完整 canonical 动作集】= 各 threshold.action + default_action(去重保序)。
    ★L9 闭选项 MC 的固定选项集本体(随题面给出;判分对它做 EM 定位命中项)。"""
    out, seen = [], set()
    for a in [t.get("action") for t in (rule.get("thresholds") or [])] + [rule.get("default_action")]:
        if a is not None and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def gt_multihop(ws: WorldState, start: str, field_path, at_week: int):
    """★L2 关系多跳:沿【软外键】field_path 在 at_week 逐跳遍历状态机。
    非末跳的值 = 下一跳实体名(解外键),末跳值 = 答案。中间断链/取不到 → INSUFFICIENT。
    时序:边随周变(负责人换→汇报关系变),故答案随 at_week 变,gt 由代码切片重算。
    返回 {answer, path_evidence:[{entity,field,value,set_session}], path, at_week}。"""
    cur, evidence = start, []
    for f in field_path:
        tl = ws.timeline(cur, f)
        if tl is None:
            return {"answer": INSUFFICIENT, "path_evidence": evidence, "broke_at": [cur, f], "at_week": at_week}
        val = tl.value_at_session(at_week)
        if val is None or val in (INVALID, INSUFFICIENT):
            return {"answer": INSUFFICIENT, "path_evidence": evidence, "broke_at": [cur, f], "at_week": at_week}
        # 该值在哪一周被设(mention-on-change 落点 = ≤at_week 的最后一次取该值的变更周)
        set_sess = at_week
        for o in tl._sorted():
            if o.session <= at_week and str(o.value) == str(val):
                set_sess = o.session
        evidence.append({"entity": cur, "field": f, "value": val, "set_session": set_sess})
        cur = val   # 下一跳实体 = 本跳值;末跳后 cur = 答案
    return {"answer": cur, "path_evidence": evidence, "path": list(field_path), "at_week": at_week}


def _shape_issues(ws: WorldState):
    """★非单调校验:数值 evolving 字段若 max 和 min 都落在端点(={首,尾})⇒ 单调 ⇒ MR 退化,记 issue。"""
    issues = []
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            nums = [(s, _to_num(v)) for (s, _, v) in tl.set_values() if _to_num(v) is not None]
            if len(nums) >= 3:
                sess = [s for s, _ in nums]
                vals = [n for _, n in nums]
                amax = sess[vals.index(max(vals))]
                amin = sess[vals.index(min(vals))]
                if {amax, amin} <= {sess[0], sess[-1]}:
                    issues.append(f"{ent}.{fname} 数值轨迹单调(极值都在端点), MR 退化")
    return issues


def validate(ws: WorldState, table: dict = None, profile: dict = None) -> list[dict]:
    """★W.3 世界质量【结构化缺陷清单】(CRITIC 修复轮:算出的缺陷不丢、定向重生成坏字段)。
    每条 = {entity, field, type, detail}。type:
      monotonic          —— typed 极值落端点，仅诊断，不是世界真假约束;
      fake_evolving      —— 声明 evolving 却全程只有 1 个值(且没停用)= 没演化起来;
      illegal_transition —— ★C1③:白皮书 state_machines 声明了【单向状态序】的字段,取值倒流/出界
                            (014559 实证:状态倒流写在世界 canonical 本身,4/34 实体)。
                            opt-in:只查声明了 states 的字段,绝不碰 L4 选择流等合法往复字段。
    table = LLM 原始世界表(用于判 declared type);缺省则跳过 fake_evolving。
    profile = 白皮书 domain_profile(取 state_machines);缺省则跳过 illegal_transition。"""
    decl = {}
    for e in _dicts((table or {}).get("entities", [])):
        nm = e.get("name") or e.get("id")
        for fn, sp in _dict(e.get("fields")).items():
            sp = _dict(sp)
            decl[(nm, fn)] = sp.get("type") or ("stable" if ("value" in sp and "trajectory" not in sp) else "evolving")
    sm = {m.get("field"): [str(x) for x in (m.get("states") or [])]
          for m in _dicts((profile or {}).get("state_machines"))
          if m.get("field") and isinstance(m.get("states"), list) and len(m.get("states")) >= 2}
    # ★声明自检(刀1审计):states 含 _norm 重复(如 [a,b,a])会让 idx 塌缩、把合法推进误判倒流 → 整条声明作废跳过
    sm = {f: sts for f, sts in sm.items() if len({_norm(x) for x in sts}) == len(sts)}
    # ★value_shape(累计工时非单调根治·183626 三轮坐实):议会声明数值字段的"值形状"(单调/值域),代码机械执行。
    #   shape[field] = {"mono": "up"/"down"/None, "range": [lo,hi]/None}。域知识从白皮书来(议会声明),非代码猜字段名。
    shape = {}
    for f in _dicts((profile or {}).get("field_schema", [])):
        nm, mono, rng = f.get("name"), f.get("monotonic"), f.get("range")
        if nm and (mono in ("up", "down") or (isinstance(rng, (list, tuple)) and len(rng) == 2)):
            lo_hi = None
            try:
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    lo_hi = (float(rng[0]), float(rng[1]))
                    if lo_hi[0] >= lo_hi[1]:           # 声明自检:值域非法(下≥上)→ 作废该 range
                        lo_hi = None
            except (TypeError, ValueError):
                lo_hi = None
            shape[nm] = {"mono": mono if mono in ("up", "down") else None, "range": lo_hi}
    out: list[dict] = []
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            sv = tl.set_values()
            distinct = {_norm(v) for (_, _, v) in sv}
            has_stop = any(o.op in (DELETE, EXPIRE) for o in tl.ops)
            fshape = shape.get(fname) or {}
            mono_decl = fshape.get("mono")
            # Extrema position is a potential task-shape diagnostic, never a
            # truth constraint. Use the same complete typed parsing as MR; a
            # text such as a document identifier must not become its first digit.
            from pipeline.value_types import ValueComparisonError, comparison_keys, field_schema
            try:
                kind, vals = comparison_keys([v for _, _, v in sv],
                    field_schema(ws, ent, fname, {"domain_profile": profile or {}}))
            except ValueComparisonError:
                kind, vals = None, []
            if len(vals) >= 3 and fname not in sm and not mono_decl:
                sess = [s for s, _, _ in sv]
                amax, amin = sess[vals.index(max(vals))], sess[vals.index(min(vals))]
                if {amax, amin} <= {sess[0], sess[-1]}:
                    out.append({"entity": ent, "field": fname, "type": "monotonic",
                                "severity": "diagnostic", "value_kind": kind,
                                "samples": [{"session": s, "value": v} for s, _, v in sv],
                                "extrema_sessions": {"max": amax, "min": amin},
                                "detail": "可比较轨迹的极值落首/尾；仅供题目证据依赖诊断，"
                                          "不构成世界错误，也未证明题目容易；不为此改写真值。"})
            # ★value_shape 单调:声明 up→只增不减、down→只减不增,违反即缺陷(交 CRITIC 修复)。
            #   用【绝对量级 _magnitude】比较(审计 HIGH 根治:按单位归一,'9000万'<'1.2亿' 才判对,不被混量纲假阳假阴)
            mags = [(s, _magnitude(v)) for (s, _, v) in sv if _magnitude(v) is not None]
            if mono_decl and len(mags) >= 2:
                mv = [m for _, m in mags]
                bad_i = next((i for i in range(1, len(mv))
                              if ((mv[i] < mv[i - 1]) if mono_decl == "up" else (mv[i] > mv[i - 1]))), None)
                if bad_i is not None:
                    word = "只增不减(累计/合计类)" if mono_decl == "up" else "只减不增"
                    sym = "<" if mono_decl == "up" else ">"   # 审计 LOW:符号随方向,否则 down 字段文案符号写反误导修复
                    out.append({"entity": ent, "field": fname, "type": "monotonic_violation",
                                "detail": f"该字段值须{word},但第{mags[bad_i][0]}周量级 {mv[bad_i]:g} {sym} 前值 {mv[bad_i-1]:g}"
                                          f"(逆向)→ 重写为单向【不减/不增】(可个别周持平、但整体要演化)的轨迹"})
            # ★value_shape 值域:声明 [lo,hi],出界即缺陷(同样按量级比,值域端点为该字段单位下的数;无量纲字段=数值本身)
            rng = fshape.get("range")
            if rng:
                oob = [(s, m) for (s, m) in mags if not (rng[0] <= m <= rng[1])]
                if oob:
                    out.append({"entity": ent, "field": fname, "type": "out_of_range",
                                "detail": f"该字段值须在 [{rng[0]:g},{rng[1]:g}] 内,但 {[(s, f'{m:g}') for s, m in oob[:3]]} 出界 → 重写到值域内"})
            if decl.get((ent, fname)) == "evolving" and len(distinct) < 2 and not has_stop:
                out.append({"entity": ent, "field": fname, "type": "fake_evolving",
                            "detail": "标 evolving 却全程只有 1 个值 → 需给【≥2 个不同值】的演化轨迹"})
            order = sm.get(fname)
            if order:
                idx = {_norm(x): i for i, x in enumerate(order)}
                last, bad = -1, None
                for (_s, _d, v) in sv:
                    i = idx.get(_norm(v))
                    if i is None:
                        bad = f"取值「{v}」不在声明状态表 {order} 内"; break
                    if i < last:
                        bad = f"状态倒流:「{v}」出现在更后阶段之后(声明顺序 {order})"; break
                    last = i
                if bad:
                    out.append({"entity": ent, "field": fname, "type": "illegal_transition",
                                "detail": f"{bad} → 须按声明顺序【单向推进】重写该字段轨迹(可跳级、不可回头)"})
    return out


def answer(ws: WorldState, capability: str, **kw) -> Any:
    """能力 → gt 派发(T2 '点菜' 生成器的种子)。"""
    cap = capability.upper()
    if cap == "IE":       return gt_ie(ws, kw["entity"], kw["field"], kw["session"])
    if cap == "KU":       return gt_ku(ws, kw["entity"], kw["field"])
    if cap == "MR":       return gt_mr(ws, kw["entity"], kw["field"], kw.get("agg", "max"))
    if cap == "TR":       return gt_tr(ws, kw["entity"], kw["field"], kw.get("to_value"))
    if cap == "CONFLICT": return gt_conflict(ws, kw["entity"], kw["field"])
    if cap == "FORGET":   return gt_forget(ws, kw["entity"], kw["field"], kw["question_date"])
    if cap == "ABS":      return gt_absent(ws, kw["field"])
    raise ValueError(f"未知能力 {capability}")


# ─────────────────────────────────────────────────────────────────────────────
# assemble_world — LLM 填的「每实体每字段取值表」→ WorldState(代码 diff 成 ops,T3 用)
#   LLM 只填表(素材),代码负责:diff 出 SET/UPDATE/EXPIRE、套级联、配日历、校验。gt 仍归代码。
# ─────────────────────────────────────────────────────────────────────────────

def _date_of(session: int, base: str = "2025-01-06", step_days: int = 7) -> str:
    from datetime import datetime as _d, timedelta as _td
    return (_d.strptime(base, "%Y-%m-%d") + _td(days=session * step_days)).strftime("%Y-%m-%d")


def week_label(session: int) -> int:
    """session 索引(0-based,代码内部口径)→ 人读周次(1-based)。
    ★命门1 单一真源:凡【题面/语料】出现"第N周",N 必须经此函数派生,确保问题与语料同一把尺。
    根治 run#3 盲审 Q01 的 off-by-one + 两套起点——0-based 索引被人/LLM 自发读成 1-based 是病根。
    (at_week / gt / grounding 仍用 0-based 索引;week_label 只在【渲染成文字】那一刻用。)"""
    return session + 1


def _strip_disambig(name) -> str:
    """剥【常见消歧后缀】取专名主干 —— ★启发式(审计★2:诚实命名,非完备"近似名"判定)。
    覆盖:括号注「张三(数据)」/ 下划线·横线·空格尾注「张三_数据」「张三 数据」/ CJK 后的尾随数字·拉丁「张三2」「张三A」。
    刻意【不剥 CJK 尾字】→「张三丰」不会误塌缩成「张三」(真不同名);跨语言/罕见消歧仍漏 —— 判定与兜底共用它,
    故宁可多覆盖常见款。漏网由 name_collisions 出口日志暴露。"""
    import re
    s = re.sub(r"[（(][^)）]*[)）]", "", str(name))                 # 括号注:张三(数据)→张三
    s = re.split(r"[_\-—\s]", s, 1)[0]                             # 下划线/横线/空格尾注:张三_数据·张三 数据→张三
    s = re.sub(r"(?<=[一-鿿])[0-9A-Za-z]+$", "", s)        # CJK 专名后的尾随数字/拉丁:张三2·张三A→张三
    return s.strip() or str(name).strip()                         # 别剥成空串(纯拉丁/纯数字名 → 留原样)


def name_collisions(ws: "WorldState") -> list:
    """★Fix3:表面塌缩组——剥消歧后缀后主干相同的 ≥2 个实体(张三/张三(数据)/张三_数据)。
    表面塌缩 → 渲染期裸专名指代不唯一 → 题面/语料歧义(命门1 表面层缺口)。返回 [[同主干实体…], …]。"""
    from collections import defaultdict
    g = defaultdict(list)
    for e in ws.entities:
        g[_strip_disambig(e)].append(e)
    return [sorted(v) for v in g.values() if len(v) >= 2]


def _as_int(x, default=0):
    """LLM 偶发把 session 写成字符串("3")/浮点;稳健强转 int,坏值回退 default(防 max()/排序炸)。"""
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default


def _dict(x) -> dict:
    """非 dict → {}。LLM 偶发把"对象"吐成 list/str/标量,防下游 .get()/.items() AttributeError 崩整 stage。"""
    return x if isinstance(x, dict) else {}


def _dicts(x) -> list:
    """只保留 list 里的 dict 元素(LLM 偶发把"对象数组"吐成 list-of-str / 混杂类型);非 list → []。
    用在所有"for 元素 in LLM输出列表 → 元素.get(...)"的消费点,把畸形元素丢弃而非崩(契合"歧义直接 drop")。"""
    return [d for d in x if isinstance(d, dict)] if isinstance(x, list) else []


def _traj_to_ops(traj: list[dict], date_of) -> list[Op]:
    """把 [{session,value}] 轨迹 diff 成 ops:首现=SET、变值=UPDATE、value 为空=EXPIRE(停统计)。
    值持续不变则不记 op(由 value_at fold 前向填充,IE-locate 扫全程仍能命中)。"""
    ops: list[Op] = []
    last = None
    for p in sorted(_dicts(traj), key=lambda x: _as_int(x.get("session"), 0)):
        s, v = _as_int(p.get("session"), 0), p.get("value")
        if v is None or str(v).strip() == "":
            if last is not None:
                ops.append(Op(s, date_of(s), EXPIRE, None, last)); last = None
            continue
        v = str(v)
        if last is None:
            ops.append(Op(s, date_of(s), SET, v, None))
        elif _norm(v) != _norm(last):
            ops.append(Op(s, date_of(s), UPDATE, v, last))
        last = v
    return ops


def intrinsic_value_ops(spec: dict, date_of) -> list[Op]:
    """Lower intrinsic values into the original truth-time operations.

    Ownership, shape, range and session checks remain in assemble_world. This
    projection does not infer truth dates from prose or publication metadata.
    """
    if spec.get("type") == "stable" or ("value" in spec and "trajectory" not in spec):
        return [Op(0, date_of(0), SET, str(spec.get("value")), None)]
    return _traj_to_ops(spec.get("trajectory", []), date_of)


def intrinsic_fact_time_contract(date_of) -> dict:
    """Expose the compiler's time meaning to its intrinsic-value authors."""
    initial = intrinsic_value_ops({"type": "stable", "value": ""}, date_of)[0]
    return {"version": "intrinsic-fact-time/v1", "scope": "intrinsic_fields",
        "stable": {"op": initial.op, "session": initial.session, "date": initial.date,
            "meaning": "value 是窗口开始就成立并持续有效的事实，不是窗口结束后的总结。后期才获得的知识或发生的状态不能因为后来不再变化就写成 stable。"},
        "evolving": {"session_meaning": "每个轨迹点的 session 是该值开始成立的真值时点，日期按同一 calendar 映射，不是公开、补录或写作时点。",
            "first_value": SET, "changed_value": UPDATE, "empty_value": EXPIRE,
            "before_first_value": INSUFFICIENT, "after_expiry": INVALID,
            "between_points": "沿用前一个有效值；重复同值不产生新事实版本。后期首次成立可从对应 session 开始，不向窗口开始倒填。"},
        "publication": "公开安排与 acquisition_context 描述怎样、何时取得或公开已有事实，不能改变字段的真值时间线。",
        "semantic_responsibility": "作者依据原业务与所读事实选择真值时点；原语义审阅判断内容在该时点是否成立，程序不凭正文日期关键词替代该判断。"}


def field_value_issue(decl: dict, value: Any) -> str | None:
    """One value contract shared by author transactions and world compilation."""
    if value is None or str(value).strip() == "":
        return "值为空"
    states = decl.get("states")
    if isinstance(states, list) and states and _norm(value) not in {_norm(x) for x in states}:
        return f"值 {value!r} 不在状态表 {states}"
    mag = _magnitude(value)
    if decl.get("kind") == "numeric" and mag is None:
        return f"numeric 字段值不可解析为数值:{value!r}"
    rng = decl.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            return f"range 端点不可解析:{rng!r}"
        if mag is None or not lo <= mag <= hi:
            return f"值 {value!r} 越界，须在 [{lo:g},{hi:g}]"
    return None


def structural_projection_findings(table, world, compiler_issues):
    """Describe authored structure that the original compiler did not retain.

    Canonical membership and effects determine acceptance. Compiler messages
    remain diagnostic evidence; callers do not infer validity from their text.
    """
    findings = []
    for collection in ("relations", "events"):
        accepted = {row["id"]: row for row in getattr(world, collection)}
        for proposed in table.get(collection, []):
            canonical = accepted.get(proposed["id"])
            retained = canonical is not None
            if retained and collection == "events":
                encode = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True)
                retained = Counter(map(encode, proposed["effects"])) == Counter(map(encode, canonical["effects"]))
            if retained:
                continue
            prefix = ("relation " if collection == "relations" else "event ") + proposed["id"]
            evidence = {"proposed": deepcopy(proposed), "canonical": deepcopy(canonical),
                "compiler_issues": [issue for issue in compiler_issues if issue.startswith(prefix + " ")]}
            findings.append({"collection": collection, "id": proposed["id"], "evidence": evidence})
    return findings


def assemble_world(table: dict, base: str = "2025-01-06", step_days: int = 7,
                   blueprint: dict | None = None, existing: WorldState | None = None, *,
                   require_complete: bool = True,
                   include_shape_diagnostics: bool = True) -> tuple[WorldState, list[str]]:
    """把 LLM 世界表编译成真值状态机。

    传入 ``blueprint`` 时额外校验实体类型，并按 relation.field 在两个端点的唯一
    归属把关系编译成 owner 的软外键 Timeline，再把 domain event 的 effect 编译成
    普通 Op；同类型自关系约定由 source/from 持有字段。旧调用不传蓝图时行为保持不变。

    ``require_complete=False`` 只延期全局实体/关系/事件数量及每条因果规则至少
    一对见证的覆盖检查，供 Agent 逐次编译草稿；当前对象的引用、时序、值域与效果
    仍须合法。``include_shape_diagnostics=False`` 省略 evolving 单值和数值趋势提示，不改变
    任何类型、状态或业务结构校验。默认完整编译行为保持不变。
    """
    if type(require_complete) is not bool or type(include_shape_diagnostics) is not bool:
        raise ValueError("require_complete and include_shape_diagnostics must be booleans")
    if blueprint is not None:
        try:
            from pipeline.world_blueprint import normalize_world_blueprint, relation_owner_side
        except ModuleNotFoundError:  # 兼容 `python pipeline/world_state.py` 的文档化自检入口
            from world_blueprint import normalize_world_blueprint, relation_owner_side
        blueprint = normalize_world_blueprint(blueprint)
        temporal = blueprint.get("temporal_model") or {}
        step_days = _as_int(temporal.get("step_days"), step_days)
    session_limit = _as_int(((blueprint or {}).get("temporal_model") or {}).get("n_sessions"), 0)
    date_of = lambda s: _date_of(s, base, step_days)  # noqa: E731
    # 增量编译时以旧世界为只读基底，让新关系/事件可以合法引用并更新旧实体。
    entities: dict[str, dict[str, Timeline]] = deepcopy(existing.entities) if existing is not None else {}
    entity_types: dict[str, str] = dict(existing.entity_types) if existing is not None else {}
    issues: list[str] = []
    max_sess = max(0, (existing.n_sessions - 1) if existing is not None else 0)
    bp_types = {t.get("id"): t for t in (blueprint or {}).get("entity_types", [])}
    allowed_fields = {tid: {f.get("name") for f in t.get("fields", []) if f.get("name")}
                      for tid, t in bp_types.items()}
    field_specs = {tid: {f.get("name"): f for f in t.get("fields", []) if f.get("name")}
                   for tid, t in bp_types.items()}

    relation_fields: dict[str, set[str]] = {}
    relation_owner_sides: dict[str, str] = {}
    event_fields: dict[str, set[str]] = {}
    for rel in (blueprint or {}).get("relation_types", []):
        side = relation_owner_side(blueprint, rel)
        if side is None:
            issues.append(
                f"relation type {rel.get('id') or '?'} 字段 {rel.get('field')!r} 在两个端点间没有唯一 owner")
            continue
        relation_owner_sides[rel.get("id")] = side
        owner_type = rel.get("from_type") if side == "from" else rel.get("to_type")
        relation_fields.setdefault(owner_type, set()).add(rel.get("field"))
    for event in (blueprint or {}).get("event_types", []):
        roles = event.get("roles") or {}
        for effect in event.get("effect_fields") or []:
            tid = roles.get(effect.get("role"))
            event_fields.setdefault(tid, set()).add(effect.get("field"))

    def _value_contract_issue(tid: str, fname: str, value: Any) -> str | None:
        """机械检查一个字段值是否满足 blueprint 的 kind/states/range。"""
        decl = field_specs.get(tid, {}).get(fname) or {}
        return field_value_issue(decl, value)

    for ent in _dicts(table.get("entities", [])):
        name = ent.get("name") or ent.get("id")
        if not name:
            continue
        etype = ent.get("type") or ent.get("entity_type")
        if blueprint is not None:
            if etype not in bp_types:
                issues.append(f"entity {name} 类型不存在:{etype}")
                continue
            entity_types[name] = etype
        flds: dict[str, Timeline] = {}
        for fname, spec in _dict(ent.get("fields")).items():
            open_legacy_schema = bool(blueprint and blueprint.get("legacy_adapter")
                                      and not allowed_fields.get(etype))
            if blueprint is not None and not open_legacy_schema and fname not in allowed_fields.get(etype, set()):
                issues.append(f"entity {name}({etype}) 含本类型未声明字段:{fname}")
                continue
            spec = _dict(spec)
            if (spec.get("type") == "stable") or ("value" in spec and "trajectory" not in spec):
                value = spec.get("value")
                problem = _value_contract_issue(etype, fname, value) if blueprint is not None else None
                if problem:
                    issues.append(f"entity {name}.{fname} {problem}")
                    continue
                flds[fname] = Timeline(intrinsic_value_ops(spec, date_of))
            else:
                traj = _dicts(spec.get("trajectory", []))
                sessions = [_as_int(p.get("session"), -1) for p in traj]
                dup_sessions = sorted(s for s, n in Counter(sessions).items() if n > 1)
                if dup_sessions:
                    issues.append(f"entity {name}.{fname} trajectory 同 session 重复:{dup_sessions}")
                    # 继续编译时保留输入中最后一条，保证 canonical 与渲染不会各取一条。
                    by_session = {_as_int(p.get("session"), -1): p for p in traj}
                    traj = [by_session[s] for s in sorted(by_session)]
                if session_limit:
                    bad_sessions = [_as_int(p.get("session"), -1) for p in traj
                                    if not 0 <= _as_int(p.get("session"), -1) < session_limit]
                    if bad_sessions:
                        issues.append(f"entity {name}.{fname} session 超出 0..{session_limit - 1}:{bad_sessions}")
                    traj = [p for p in traj if 0 <= _as_int(p.get("session"), -1) < session_limit]
                if blueprint is not None:
                    for p in traj:
                        value = p.get("value")
                        if value is None or str(value).strip() == "":
                            continue
                        problem = _value_contract_issue(etype, fname, value)
                        if problem:
                            issues.append(f"entity {name}.{fname}@{p.get('session')} {problem}")
                for p in traj:
                    max_sess = max(max_sess, _as_int(p.get("session"), 0))
                ops = intrinsic_value_ops({"type": "evolving", "trajectory": traj}, date_of)
                if ops:
                    flds[fname] = Timeline(ops)
        if flds or blueprint is not None:
            entities[name] = flds

    if blueprint is not None:
        type_counts: dict[str, int] = {}
        for tid in entity_types.values():
            type_counts[tid] = type_counts.get(tid, 0) + 1
        for tid, decl in bp_types.items():
            if require_complete and type_counts.get(tid, 0) < decl.get("count", 1):
                issues.append(f"entity type {tid} 实例不足:{type_counts.get(tid, 0)}/{decl.get('count', 1)}")

    def _inject(entity: str, fld: str, session: int, value, source: str) -> bool:
        """向既有 Timeline 注入一个结构效果；同周冲突不覆盖而是报错。"""
        nonlocal max_sess
        if entity not in entities:
            issues.append(f"{source} 引用不存在实体:{entity}")
            return False
        if session_limit and not 0 <= session < session_limit:
            issues.append(f"{source} session 超出 0..{session_limit - 1}:{session}")
            return False
        max_sess = max(max_sess, session)
        tl = entities[entity].get(fld)
        if tl:
            same = [o for o in tl.ops if o.session == session and o.op in (SET, UPDATE)]
            if same and any(_norm(o.value) != _norm(str(value)) for o in same):
                issues.append(f"{source} 与既有值同 session 冲突:{entity}.{fld}@{session}")
                return False
            if same:
                return True                         # 基础轨迹已在该 session 编码同一真实效果
            if _norm(tl.value_at_session(session)) == _norm(str(value)):
                return False                        # 只是延续旧值，不构成领域事件效果
        prev = tl.value_at_session(session - 1) if tl else None
        prev = None if prev in (INVALID, INSUFFICIENT) else prev
        entities[entity].setdefault(fld, Timeline([])).ops.append(
            Op(session, date_of(session), SET if prev is None else UPDATE, str(value), prev))
        return True

    relations: list[dict] = deepcopy(existing.relations) if existing is not None else []
    events: list[dict] = deepcopy(existing.events) if existing is not None else []
    generated_cascades: list[dict] = []
    if blueprint is not None:
        rel_types = {r["id"]: r for r in blueprint.get("relation_types", [])}
        rel_counts: dict[str, int] = {}
        relation_ids = set()
        relation_edges = set()
        for old_rel in relations:
            rid = old_rel.get("type")
            rel_counts[rid] = rel_counts.get(rid, 0) + 1
            if old_rel.get("id"):
                relation_ids.add(old_rel["id"])
            relation_edges.add((rid, old_rel.get("from"), old_rel.get("to"),
                                _as_int(old_rel.get("session"), 0)))
        # Timeline 编译必须与 LLM 数组顺序无关；同一 owner.field 按时间正序判断真实变化/no-op。
        relation_rows = sorted(
            _dicts(table.get("relations", [])),
            key=lambda item: (_as_int(item.get("session"), -1), str(item.get("id") or "")),
        )
        for rel in relation_rows:
            rtype = rel_types.get(rel.get("type"))
            instance_id = rel.get("id")
            src, dst = rel.get("from"), rel.get("to")
            extra_keys = sorted(set(rel) - {"id", "type", "from", "to", "session"})
            if extra_keys:
                issues.append(f"relation {instance_id or '?'} 含契约外字段:{extra_keys}")
            if not rtype:
                issues.append(f"relation instance 类型未声明:{rel.get('type')}")
                continue
            if not isinstance(instance_id, str) or not instance_id.strip() or instance_id in relation_ids:
                issues.append(f"relation id 为空或重复:{instance_id}")
                continue
            if src not in entities or dst not in entities:
                issues.append(f"relation {rel.get('id') or rel.get('type')} 端点不存在:{src}->{dst}")
                continue
            if (entity_types.get(src) != rtype["from_type"] or
                    entity_types.get(dst) != rtype["to_type"]):
                issues.append(f"relation {rel.get('id') or rel.get('type')} 端点类型不符")
                continue
            sess = _as_int(rel.get("session"), 0)
            if sess < 0 or (session_limit and sess >= session_limit):
                issues.append(f"relation {rel.get('id') or rel.get('type')} session 非法:{sess}")
                continue
            if not rtype.get("temporal") and sess != 0:
                issues.append(f"relation {instance_id} 是 static(temporal=false)，session 必须为 0")
                continue
            edge = (rtype["id"], src, dst, sess)
            if edge in relation_edges:
                issues.append(f"relation {instance_id} 重复边:{src}->{dst}@{sess}")
                continue
            owner_side = relation_owner_sides.get(rtype["id"])
            if owner_side is None:
                issues.append(f"relation {instance_id} 无法确定字段 owner")
                continue
            owner, referenced = (src, dst) if owner_side == "from" else (dst, src)
            if not _inject(owner, rtype["field"], sess, referenced, f"relation {instance_id}"):
                issues.append(f"relation {instance_id} 没有形成合法 FK 变化")
                continue
            relations.append({"id": instance_id, "type": rel.get("type"),
                              "from": src, "to": dst, "session": sess})
            relation_ids.add(instance_id)
            relation_edges.add(edge)
            rel_counts[rtype["id"]] = rel_counts.get(rtype["id"], 0) + 1
        for rid, decl in rel_types.items():
            if require_complete and rel_counts.get(rid, 0) < decl.get("min_count", 1):
                issues.append(f"relation type {rid} 实例不足:{rel_counts.get(rid, 0)}/{decl.get('min_count', 1)}")

        event_types = {e["id"]: e for e in blueprint.get("event_types", [])}
        event_counts: dict[str, int] = {}
        event_by_id: dict[str, dict] = {}
        event_instances: set[tuple] = set()
        for old_event in events:
            if old_event.get("id"):
                event_by_id[old_event["id"]] = old_event
            etid = old_event.get("type")
            event_counts[etid] = event_counts.get(etid, 0) + 1
            event_instances.add((
                etid,
                _as_int(old_event.get("session"), -1),
                tuple(sorted((str(role), str(entity))
                             for role, entity in _dict(old_event.get("participants")).items())),
            ))
        event_rows = sorted(
            _dicts(table.get("events", [])),
            key=lambda item: (_as_int(item.get("session"), -1), str(item.get("id") or "")),
        )
        for event in event_rows:
            decl = event_types.get(event.get("type"))
            eid = event.get("id")
            participants = _dict(event.get("participants"))
            extra_keys = sorted(set(event) - {
                "id", "type", "session", "participants", "effects", "caused_by",
            })
            if extra_keys:
                issues.append(f"event {eid or '?'} 含契约外字段:{extra_keys}")
            if not decl:
                issues.append(f"event instance 类型未声明:{event.get('type')}")
                continue
            if not eid or eid in event_by_id:
                issues.append(f"event id 为空或重复:{eid}")
                continue
            declared_roles = set(decl.get("roles", {}))
            actual_roles = set(participants)
            if actual_roles != declared_roles:
                issues.append(
                    f"event {eid} participants roles 必须精确等于声明:"
                    f"actual={sorted(actual_roles)} expected={sorted(declared_roles)}")
                continue
            bad_role = False
            for role, tid in decl.get("roles", {}).items():
                entity = participants.get(role)
                if entity not in entities or entity_types.get(entity) != tid:
                    issues.append(f"event {eid} role {role} 实体/类型不符:{entity}->{tid}")
                    bad_role = True
            if bad_role:
                continue
            if len(set(participants.values())) != len(participants):
                issues.append(f"event {eid} 不同 participant role 必须由互异实体承担:{participants}")
                continue
            sess = _as_int(event.get("session"), -1)
            if sess < 0 or (session_limit and sess >= session_limit):
                issues.append(f"event {eid} session 非法:{event.get('session')}")
                continue
            event_instance = (
                decl["id"], sess,
                tuple(sorted((str(role), str(entity)) for role, entity in participants.items())),
            )
            if event_instance in event_instances:
                issues.append(f"event {eid} 重复实例:{decl['id']}@{sess} participants={participants}")
                continue
            allowed_effects = {(x.get("role"), x.get("field"))
                               for x in decl.get("effect_fields", []) if isinstance(x, dict)}
            participant_role = {entity: role for role, entity in participants.items()}
            raw_effects = _dicts(event.get("effects", []))
            actual_effects = Counter(
                (participant_role.get(effect.get("entity")), effect.get("field"))
                for effect in raw_effects
            )
            expected_effects = Counter(allowed_effects)
            if actual_effects != expected_effects:
                expected_instances = [{"role": role, "entity": participants.get(role), "field": field}
                                      for role, field in sorted(allowed_effects)]
                actual_instances = [{"path": f"events[{eid}].effects[{index}].entity",
                                     "entity": effect.get("entity"), "field": effect.get("field"),
                                     "keys": sorted(effect)} for index, effect in enumerate(raw_effects)]
                issues.append(
                    f"event {eid} effects 必须恰好覆盖 effect_fields:"
                    f"actual={actual_instances} expected={expected_instances}; "
                    f"effects_type={type(event.get('effects')).__name__}; "
                    "effects 使用数组，每项 entity 填 expected.entity 的具体实体名，"
                    "由 participants[expected.role] 对应；保留业务值，勿把 role 名或 role 键写成实例格式")
                continue
            clean_effects = []
            for eff in raw_effects:
                entity, fld = eff.get("entity"), eff.get("field")
                effect_extra = sorted(set(eff) - {"entity", "field", "set", "value"})
                if effect_extra:
                    issues.append(f"event {eid} effect 含契约外字段:{effect_extra}")
                matching_roles = [role for role, ename in participants.items() if ename == entity]
                role = next((r for r in matching_roles if (r, fld) in allowed_effects), None)
                value = eff.get("set", eff.get("value"))
                if role is None or value is None:
                    issues.append(f"event {eid} effect 未声明或无 set:{entity}.{fld}")
                    continue
                tid = entity_types.get(entity)
                field_decl = next((f for f in bp_types.get(tid, {}).get("fields", [])
                                   if f.get("name") == fld), {})
                states = field_decl.get("states")
                if isinstance(states, list) and states and _norm(value) not in {_norm(x) for x in states}:
                    issues.append(f"event {eid} effect 值不在状态表:{entity}.{fld}={value} not in {states}")
                    continue
                value_problem = _value_contract_issue(tid, fld, value)
                if value_problem:
                    issues.append(f"event {eid} effect 非法:{entity}.{fld} {value_problem}")
                    continue
                mag = _magnitude(value)
                tl = entities.get(entity, {}).get(fld)
                prev_mag = _magnitude(tl.value_at_session(sess - 1)) if tl else None
                mono = field_decl.get("monotonic")
                if (prev_mag is not None and mag is not None and
                        ((mono == "up" and mag < prev_mag) or (mono == "down" and mag > prev_mag))):
                    issues.append(f"event {eid} effect 违反 monotonic={mono}:{entity}.{fld} {prev_mag}->{mag}")
                    continue
                if _inject(entity, fld, sess, value, f"event {eid}"):
                    clean_effects.append({"entity": entity, "field": fld, "set": value})
                else:
                    issues.append(f"event {eid} effect 是空操作:{entity}.{fld}@{sess}={value}")
            if not clean_effects:
                issues.append(f"event {eid} 没有合法 effect")
                continue
            clean_event = {
                "id": eid, "type": event.get("type"), "session": sess,
                "participants": dict(participants), "effects": clean_effects,
            }
            if event.get("caused_by"):
                clean_event["caused_by"] = event["caused_by"]
            events.append(clean_event)
            event_by_id[eid] = clean_event
            event_instances.add(event_instance)
            event_counts[decl["id"]] = event_counts.get(decl["id"], 0) + 1
            max_sess = max(max_sess, sess)
        for event_id, decl in event_types.items():
            if require_complete and event_counts.get(event_id, 0) < decl.get("min_count", 1):
                issues.append(f"event type {event_id} 实例不足:{event_counts.get(event_id, 0)}/{decl.get('min_count', 1)}")

        # caused_by 只有匹配蓝图类型对与精确 delay 才能进入 canonical world。
        # 非法边留 issue 并从干净事件移除，避免后续叙事把模型私造因果当真。
        causal_rules = [rule for rule in blueprint.get("causal_rules", [])
                        if isinstance(rule, dict)]
        for child in events:
            parent_ref = child.get("caused_by")
            if not parent_ref:
                continue
            parent = event_by_id.get(parent_ref)
            valid_edge = any(causal_edge_matches(parent, child, rule) for rule in causal_rules)
            if not valid_edge:
                issues.append(f"event {child.get('id')} caused_by 未被蓝图声明:{parent_ref}")
                child.pop("caused_by", None)

        # 因果规则不是一段 prose：必须由 caused_by 的真实事件对见证，并记录到 cascades 留痕。
        for rule in causal_rules:
            witnesses = []
            delay = _as_int(rule.get("delay_sessions"), 0)
            for child in events:
                parent = event_by_id.get(child.get("caused_by"))
                if causal_edge_matches(parent, child, rule):
                    witnesses.append((parent, child))
            if require_complete and not witnesses:
                issues.append(f"causal rule {rule.get('id')} 没有 caused_by 事件见证")
            for parent, child in witnesses:
                generated_cascades.append({
                    "kind": "domain_event_causality", "rule_id": rule.get("id"),
                    "trigger": {"event_id": parent.get("id"), "type": parent.get("type"),
                                "session": parent.get("session")},
                    "effect": {"event_id": child.get("id"), "type": child.get("type"),
                               "session": child.get("session")},
                })

    # 套级联(DAG):trigger 触发时,在 effect 字段注入预声明的替代值(可解性)
    for c in _dicts(table.get("cascades", [])):
        eff, trig = _dict(c.get("effect")), _dict(c.get("trigger"))
        e, f, val = eff.get("entity"), eff.get("field"), eff.get("set")
        sess = _as_int(trig.get("session"), None) if trig.get("session") is not None else None
        if not (e and f and val is not None and sess is not None):
            continue
        max_sess = max(max_sess, sess)
        tl = entities.setdefault(e, {}).get(f)
        if tl is not None and _norm(tl.value_at_session(sess)) == _norm(str(val)):
            continue  # LLM 已在轨迹里编码了该效果 → 不重复注入(防重复 op)
        prev = tl.value_at_session(sess - 1) if tl else None
        prev = None if prev in (INVALID, INSUFFICIENT) else prev
        op = Op(sess, date_of(sess), SET if prev is None else UPDATE, str(val), prev)
        entities.setdefault(e, {}).setdefault(f, Timeline([])).ops.append(op)

    if blueprint is not None and not blueprint.get("legacy_adapter"):
        # 关系字段与事件效果字段由结构实例驱动。除 session 0 的可选初始态外，每次实际
        # 变更都必须能回指同 session 的 relation/event，防止“事件只做装饰”。
        witnesses: set[tuple[str, str, int, str]] = set()
        for rel in relations:
            decl = next((r for r in blueprint.get("relation_types", []) if r.get("id") == rel.get("type")), None)
            if decl:
                owner_side = relation_owner_sides.get(decl.get("id"))
                if owner_side:
                    owner, referenced = ((rel.get("from"), rel.get("to")) if owner_side == "from"
                                         else (rel.get("to"), rel.get("from")))
                    witnesses.add((owner, decl.get("field"), _as_int(rel.get("session"), 0),
                                   _norm(referenced)))
        for event in events:
            for effect in _dicts(event.get("effects", [])):
                value = effect.get("set", effect.get("value"))
                witnesses.add((effect.get("entity"), effect.get("field"),
                               _as_int(event.get("session"), -1), _norm(value)))
        owned_fields = {tid: relation_fields.get(tid, set()) | event_fields.get(tid, set())
                        for tid in bp_types}
        for entity, tid in entity_types.items():
            for fname in owned_fields.get(tid, set()):
                tl = entities.get(entity, {}).get(fname)
                if not tl:
                    continue
                for index, op in enumerate(tl._sorted()):
                    key = (entity, fname, op.session, _norm(op.value))
                    baseline = index == 0 and op.session == 0 and op.op == SET
                    if not baseline and key not in witnesses:
                        issues.append(
                            f"entity {entity}.{fname}@{op.session} 的结构字段变更没有 relation/event 见证")

        for entity, tid in entity_types.items():
            expected = (allowed_fields.get(tid, set()) - relation_fields.get(tid, set())
                        - event_fields.get(tid, set()))
            missing = sorted(expected - set(entities.get(entity, {})))
            if missing:
                issues.append(f"entity {entity}({tid}) 缺少本类型内在字段:{missing}")

    absent = list(existing.absent_fields) if existing is not None else []
    absent += [f for f in table.get("absent_fields", []) if isinstance(f, str) and f not in absent]
    declared_sessions = ((blueprint or {}).get("temporal_model") or {}).get("n_sessions", 0)
    cascades = deepcopy(existing.cascades) if existing is not None else []
    for cascade in _dicts(table.get("cascades", [])) + generated_cascades:
        if cascade not in cascades:
            cascades.append(cascade)
    ws = WorldState(entities=entities,
                    cascades=cascades,
                    absent_fields=absent,
                    n_sessions=max(max_sess + 1, _as_int(declared_sessions, 0)),
                    entity_types=entity_types,
                    relations=relations,
                    events=events,
                    world_blueprint=deepcopy(blueprint) if blueprint else {})
    if existing is not None:
        ws.conflicts = deepcopy(existing.conflicts)
        ws.sensitive = deepcopy(existing.sensitive)
        ws.conditional_rules = deepcopy(existing.conditional_rules)
        ws.rule_instances = deepcopy(existing.rule_instances)
        ws._trended_fields = list(getattr(existing, "_trended_fields", []) or [])

    # 校验(非致命,收集 issues)
    present = {f for flds in entities.values() for f in flds}
    for f in list(absent):
        if f in present:                       # 矛盾:声称不存在却出现了 → 从 absent 移除
            issues.append(f"absent_field '{f}' 实际出现,已移除"); absent.remove(f)
    for ename, flds in entities.items():
        for fname, tl in flds.items():
            spec = next((s for e in _dicts(table.get("entities", [])) if (e.get("name") or e.get("id")) == ename
                         for fn, s in _dict(e.get("fields")).items() if fn == fname), {})
            spec = _dict(spec)
            if include_shape_diagnostics and spec.get("type") == "evolving":
                distinct = {_norm(v) for (_, _, v) in tl.set_values()}
                if len(distinct) < 2 and not any(o.op in (DELETE, EXPIRE) for o in tl.ops):
                    issues.append(f"{ename}.{fname} 标 evolving 但只有 1 个值(演化不成立)")
    ws.absent_fields = absent
    if include_shape_diagnostics:
        issues += _shape_issues(ws)                      # ★V11:数值单调告警
    return ws, issues


# ─────────────────────────────────────────────────────────────────────────────
# demo 世界(office 周报场景)+ 自检:给定 timeline + date,断言每个能力的 gt 正确
# ─────────────────────────────────────────────────────────────────────────────

_DATES = {0: "2025-01-06", 1: "2025-01-13", 2: "2025-01-20",
          3: "2025-01-27", 4: "2025-02-03", 5: "2025-02-10"}


def build_demo_world() -> WorldState:
    D = _DATES

    def tl(*steps):  # steps: (session, op, value, prev)
        return Timeline([Op(s, D[s], op, val, prev) for (s, op, val, prev) in steps])

    ai = {
        # 变值字段:2.5→1.8→2.8→停统计(EXPIRE)
        "P0缺陷率": tl((0, SET, "2.5%", None), (1, UPDATE, "1.8%", "2.5%"),
                       (3, UPDATE, "2.8%", "1.8%"), (5, EXPIRE, None, "2.8%")),
        # 数值字段:12→15→9(测 MR 的 argmax/argmin)
        "Oncall数": tl((0, SET, "12次", None), (2, UPDATE, "15次", "12次"), (4, UPDATE, "9次", "15次")),
        # 人名字段:张三→李四(测 KU/TR)
        "负责人":   tl((0, SET, "张三", None), (2, UPDATE, "李四", "张三")),
        # 稳定字段:只 SET 一次(CONFLICT=False 对照)
        "部门代号": tl((0, SET, "ENG", None)),
    }
    return WorldState({"AI工程部": ai}, cascades=[], absent_fields=["季度营收", "团队规模"], n_sessions=6)


def demo_table() -> dict:
    """手写的「LLM 世界表」样例:含值持续(s1 没变)、停统计(EXPIRE)、级联,喂 assemble_world 自检。"""
    return {
        "main_theme": "AI 工程团队周报",
        "entities": [
            {"name": "AI工程部", "type": "department", "fields": {
                "P0缺陷率": {"type": "evolving", "trajectory": [
                    {"session": 0, "value": "2.5%"}, {"session": 1, "value": "2.5%"},  # ★ s1 持续不变
                    {"session": 2, "value": "1.8%"}, {"session": 4, "value": None}]},   # ★ s4 停统计
                "负责人": {"type": "evolving", "trajectory": [
                    {"session": 0, "value": "张三"}, {"session": 3, "value": "李四"}]},
                "部门代号": {"type": "stable", "value": "ENG"}}},
            {"name": "数据平台部", "type": "department", "fields": {
                "负责人": {"type": "evolving", "trajectory": [
                    {"session": 0, "value": "王五"}, {"session": 2, "value": "赵六"}]}}},
        ],
        "cascades": [{"trigger": {"entity": "AI工程部", "field": "负责人", "becomes": "李四", "session": 3},
                      "effect": {"entity": "AI工程部", "field": "汇报对象", "set": "CTO"}}],
        "absent_fields": ["季度营收", "P0缺陷率"],   # ★ P0 是矛盾项(实际存在),校验应剔除
    }


def _self_test() -> bool:
    ws = build_demo_world()
    E = "AI工程部"
    checks: list[tuple[bool, str, Any, Any]] = []

    def ck(name, got, want):
        ok = (_norm(got) == _norm(want)) if isinstance(want, str) else (got == want)
        checks.append((ok, name, got, want))

    # IE:某 session 当时值
    ck("IE P0@s0", gt_ie(ws, E, "P0缺陷率", 0), "2.5%")
    ck("IE P0@s3", gt_ie(ws, E, "P0缺陷率", 3), "2.8%")
    ck("IE 负责人@s0", gt_ie(ws, E, "负责人", 0), "张三")
    ck("IE 负责人@s2", gt_ie(ws, E, "负责人", 2), "李四")
    # KU:最新有效值
    ck("KU 负责人", gt_ku(ws, E, "负责人"), "李四")
    ck("KU Oncall", gt_ku(ws, E, "Oncall数"), "9次")
    # MR:极值 + argmax/argmin(★ argmin 是 v8 漏的)
    mx = gt_mr(ws, E, "Oncall数", "max"); ck("MR max Oncall 值", mx["value"], "15次"); ck("MR max argmax", mx["session"], 2)
    mn = gt_mr(ws, E, "Oncall数", "min"); ck("MR min Oncall 值", mn["value"], "9次"); ck("MR min argmin", mn["session"], 4)
    ck("MR min P0 值", gt_mr(ws, E, "P0缺陷率", "min")["value"], "1.8%")
    # TR:变化时机
    ck("TR P0→1.8% 在 s1", gt_tr(ws, E, "P0缺陷率", "1.8%")["session"], 1)
    ck("TR 负责人→李四 在 s2", gt_tr(ws, E, "负责人", "李四")["session"], 2)
    # CONFLICT
    ck("CONFLICT P0 有冲突", gt_conflict(ws, E, "P0缺陷率")["conflict"], True)
    ck("CONFLICT 部门代号 无冲突", gt_conflict(ws, E, "部门代号")["conflict"], False)
    # FORGET:EXPIRE 之后该忘
    ck("FORGET P0 @2-15 已忘", gt_forget(ws, E, "P0缺陷率", "2025-02-15")["forgotten"], True)
    ck("FORGET P0 @1-20 没忘", gt_forget(ws, E, "P0缺陷率", "2025-01-20")["forgotten"], False)
    ck("FORGET P0 @1-20 值", gt_forget(ws, E, "P0缺陷率", "2025-01-20")["value"], "1.8%")
    # ABS:不存在字段
    ck("ABS 季度营收", gt_absent(ws, "季度营收"), INSUFFICIENT)
    ck("ABS P0 存在", gt_absent(ws, "P0缺陷率"), {"present": True})
    # 序列化往返(治可回溯性)
    ck("serialize roundtrip", WorldState.from_dict(ws.to_dict()).timeline(E, "负责人").latest_valid(), "李四")

    # assemble_world:LLM 表 → WorldState(值持续 / EXPIRE / 级联 / absent 矛盾校验)
    ws2, issues = assemble_world(demo_table())
    ck("assemble 2 实体", set(ws2.entities) == {"AI工程部", "数据平台部"}, True)
    ck("assemble n_sessions=5", ws2.n_sessions, 5)
    ck("assemble P0@s1 值持续=2.5%", ws2.timeline("AI工程部", "P0缺陷率").value_at_session(1), "2.5%")
    ck("assemble P0@s2=1.8%", ws2.timeline("AI工程部", "P0缺陷率").value_at_session(2), "1.8%")
    ck("assemble P0 @EXPIRE后 已忘", gt_forget(ws2, "AI工程部", "P0缺陷率", "2025-03-01")["forgotten"], True)
    ck("assemble 级联 汇报对象@s3=CTO", gt_ie(ws2, "AI工程部", "汇报对象", 3), "CTO")
    ck("assemble KU 数据平台部 负责人=赵六", gt_ku(ws2, "数据平台部", "负责人"), "赵六")
    ck("assemble absent 剔除矛盾(P0出),保留季度营收",
       "P0缺陷率" not in ws2.absent_fields and "季度营收" in ws2.absent_fields, True)
    ck("assemble issues 报了 P0 矛盾", any("P0缺陷率" in i for i in issues), True)

    # typed relation：字段可唯一归属于 from 或 to；同类型自关系固定由 from/source 持有。
    relation_bp = {
        "version": 1,
        "entity_types": [
            {"id": "a", "noun": "甲类", "count": 2, "primary": True, "fields": [
                {"name": "正向引用", "kind": "reference"},
                {"name": "同类引用", "kind": "reference"},
                {"name": "阶段", "kind": "status", "states": ["初始", "完成"]},
            ]},
            {"id": "b", "noun": "乙类", "count": 1, "primary": False, "fields": [
                {"name": "反向引用", "kind": "reference"},
                {"name": "标签", "kind": "text"},
            ]},
        ],
        "relation_types": [
            {"id": "from_owned", "from_type": "a", "to_type": "b",
             "field": "正向引用", "temporal": False, "min_count": 1},
            {"id": "to_owned", "from_type": "a", "to_type": "b",
             "field": "反向引用", "temporal": True, "min_count": 2},
            {"id": "self_owned", "from_type": "a", "to_type": "a",
             "field": "同类引用", "temporal": False, "min_count": 1},
        ],
        "event_types": [{
            "id": "finish", "label": "完成阶段", "roles": {"actor": "a"},
            "effect_fields": [{"role": "actor", "field": "阶段"}], "min_count": 1,
        }],
        "temporal_model": {"unit": "round", "cadence": "per_round",
                           "n_sessions": 3, "step_days": 1},
        "causal_rules": [],
        "evidence_channels": ["测试记录"],
    }
    relation_table = {
        "entities": [
            {"name": "A1", "type": "a", "fields": {
                "阶段": {"type": "stable", "value": "初始"}}},
            {"name": "A2", "type": "a", "fields": {}},
            {"name": "B1", "type": "b", "fields": {
                "标签": {"type": "stable", "value": "乙一"}}},
        ],
        "relations": [
            {"id": "rf1", "type": "from_owned", "from": "A1", "to": "B1", "session": 0},
            {"id": "rt1", "type": "to_owned", "from": "A1", "to": "B1", "session": 0},
            {"id": "rt2", "type": "to_owned", "from": "A2", "to": "B1", "session": 2},
            {"id": "rs1", "type": "self_owned", "from": "A1", "to": "A2", "session": 0},
        ],
        "events": [{
            "id": "ev1", "type": "finish", "session": 1,
            "participants": {"actor": "A1"},
            "effects": [{"entity": "A1", "field": "阶段", "set": "完成"}],
        }],
    }
    typed_rel, typed_rel_issues = assemble_world(relation_table, blueprint=relation_bp)
    ck("typed relation from-owner 写 source.field=target",
       gt_ie(typed_rel, "A1", "正向引用", 0), "B1")
    ck("typed relation to-owner 写 target.field=source",
       gt_ie(typed_rel, "B1", "反向引用", 0), "A1")
    ck("typed relation to-owner 时序更新由新 source 见证",
       gt_ie(typed_rel, "B1", "反向引用", 2), "A2")
    ck("typed self-relation 约定由 source 持有",
       gt_ie(typed_rel, "A1", "同类引用", 0), "A2")
    ck("typed relation owner 同步到缺字段排除与 witness 校验", typed_rel_issues, [])

    extra_role_table = deepcopy(relation_table)
    extra_role_table["events"][0]["participants"]["undeclared_actor"] = "A2"
    extra_role_world, extra_role_issues = assemble_world(
        extra_role_table, blueprint=relation_bp)
    ck("typed event 拒绝蓝图未声明的额外 participant role",
       not extra_role_world.events
       and any("roles 必须精确等于声明" in issue for issue in extra_role_issues), True)

    aliased_role_bp = deepcopy(relation_bp)
    aliased_role_bp["event_types"][0].update({
        "roles": {"actor": "a", "witness": "a"},
        "effect_fields": [{"role": "actor", "field": "阶段"}],
    })
    aliased_role_table = deepcopy(relation_table)
    aliased_role_table["events"][0]["participants"] = {"actor": "A1", "witness": "A1"}
    aliased_role_world, aliased_role_issues = assemble_world(
        aliased_role_table, blueprint=aliased_role_bp)
    ck("typed event 不同 participant role 不得别名到同一实体",
       not aliased_role_world.events
       and any("互异实体" in issue for issue in aliased_role_issues), True)

    complete_effect_bp = deepcopy(relation_bp)
    complete_effect_bp["entity_types"][0]["fields"].append(
        {"name": "结果备注", "kind": "text"})
    complete_effect_bp["event_types"][0]["effect_fields"].append(
        {"role": "actor", "field": "结果备注"})
    incomplete_effect_world, incomplete_effect_issues = assemble_world(
        relation_table, blueprint=complete_effect_bp)
    ck("typed event effects 必须完整覆盖声明而非只写一个合法子集",
       not incomplete_effect_world.events
       and any("恰好覆盖 effect_fields" in issue for issue in incomplete_effect_issues), True)

    extra_payload_table = deepcopy(relation_table)
    extra_payload_table["relations"][0]["future_hint"] = "A1 已倒戈"
    extra_payload_table["events"][0]["result"] = "A1 将会复活"
    extra_payload_table["events"][0]["effects"][0]["description"] = "隐藏结局"
    extra_payload_world, extra_payload_issues = assemble_world(
        extra_payload_table, blueprint=relation_bp)
    ck("typed relation/event 契约外 payload 报错且不进入 canonical world",
       "future_hint" not in extra_payload_world.relations[0]
       and "result" not in extra_payload_world.events[0]
       and "description" not in extra_payload_world.events[0]["effects"][0]
       and sum("契约外字段" in issue for issue in extra_payload_issues) == 3, True)

    reversed_table = deepcopy(relation_table)
    reversed_table["relations"] = list(reversed(reversed_table["relations"]))
    reversed_table["events"] = list(reversed(reversed_table["events"]))
    reversed_world, reversed_issues = assemble_world(reversed_table, blueprint=relation_bp)
    ck("typed relation/event 编译与 LLM 数组顺序无关",
       reversed_issues == [] and reversed_world.to_dict() == typed_rel.to_dict(), True)

    duplicate_event_table = deepcopy(relation_table)
    duplicate_event_table["events"].append({
        "id": "ev2", "type": "finish", "session": 1,
        "participants": {"actor": "A1"},
        "effects": [{"entity": "A1", "field": "阶段", "set": "完成"}],
    })
    duplicate_event_world, duplicate_event_issues = assemble_world(
        duplicate_event_table, blueprint=relation_bp)
    ck("typed event 同 type/session/participants 不得靠不同 id 重复计数",
       len(duplicate_event_world.events) == 1
       and any("重复实例" in issue for issue in duplicate_event_issues), True)

    self_causal_bp = deepcopy(relation_bp)
    self_causal_bp["causal_rules"] = [{
        "id": "self_loop", "trigger_event": "finish", "effect_event": "finish",
        "delay_sessions": 0,
    }]
    self_causal_table = deepcopy(relation_table)
    self_causal_table["events"][0]["caused_by"] = "ev1"
    self_causal_world, self_causal_issues = assemble_world(
        self_causal_table, blueprint=self_causal_bp)
    ck("typed causal 单事件不得 caused_by 自己形成自环见证",
       not self_causal_world.cascades
       and "caused_by" not in self_causal_world.events[0]
       and any("没有 caused_by 事件见证" in issue for issue in self_causal_issues), True)

    # V11 Tier 0:ORDER / DURATION / pre_expire / 形状
    ev = gt_event_order(ws, E)
    ck("ORDER 事件按时序升序", [e["session"] for e in ev] == sorted(e["session"] for e in ev), True)
    ck("ORDER 首个变更=P0@s1", ev[0]["session"] == 1 and ev[0]["field"] == "P0缺陷率", True)
    ck("DURATION 张三任负责人2周", gt_duration(ws, E, "负责人", "张三")["weeks"], 2)
    ck("pre_expire P0停前=2.8%", gt_pre_expire(ws, E, "P0缺陷率")["value"], "2.8%")
    ck("非单调P0不报单调(2.5→1.8→2.8)", any("P0缺陷率" in i for i in _shape_issues(ws)), False)
    mono = WorldState({"T": {"f": Timeline([Op(0, "2025-01-06", SET, "1"),
                                            Op(1, "2025-01-13", UPDATE, "2", "1"),
                                            Op(2, "2025-01-20", UPDATE, "3", "2")])}}, n_sessions=3)
    ck("单调字段被 flag", len(_shape_issues(mono)) >= 1, True)

    # ── L2 关系多跳:软外键时序图遍历 ──
    rel = WorldState({
        "搜索部": {"负责人": Timeline([Op(0, "2025-01-06", SET, "李娜", None),
                                       Op(3, "2025-01-27", UPDATE, "王强", "李娜")])},
        "李娜":   {"汇报对象": Timeline([Op(0, "2025-01-06", SET, "CTO", None),
                                         Op(2, "2025-01-20", UPDATE, "CVP", "CTO")])},
        "王强":   {"汇报对象": Timeline([Op(0, "2025-01-06", SET, "CEO", None)])},
    }, n_sessions=6)
    P = ["负责人", "汇报对象"]
    ck("L2 2跳@1=CTO(搜索部→李娜→CTO)", gt_multihop(rel, "搜索部", P, 1)["answer"], "CTO")
    ck("L2 2跳@2=CVP(中间实体属性随周变)", gt_multihop(rel, "搜索部", P, 2)["answer"], "CVP")
    ck("L2 2跳@4=CEO(桥实体换人:李娜→王强)", gt_multihop(rel, "搜索部", P, 4)["answer"], "CEO")
    ck("L2 断链字段→INSUFFICIENT", gt_multihop(rel, "搜索部", ["负责人", "查无此字段"], 1)["answer"], INSUFFICIENT)
    ck("L2 悬空外键→INSUFFICIENT", gt_multihop(rel, "搜索部", ["不存在", "汇报对象"], 1)["answer"], INSUFFICIENT)
    mh = gt_multihop(rel, "搜索部", P, 1)
    ck("L2 路径证据2跳", len(mh["path_evidence"]), 2)
    ck("L2 桥实体在证据里(出题须隐藏它)", mh["path_evidence"][0]["value"], "李娜")
    ck("L2 两跳证据落不同周(@2:李娜s0汇报变s2→拼接)", gt_multihop(rel, "搜索部", P, 2)["path_evidence"][1]["set_session"], 2)

    # ── W.3 世界质量 validate():结构化缺陷清单(供 CRITIC 修复轮定向重生成)──
    vd_table = {"entities": [
        {"name": "单调部", "fields": {"指标": {"type": "evolving", "trajectory": [
            {"session": 0, "value": "1"}, {"session": 1, "value": "2"}, {"session": 2, "value": "3"}]}}},
        {"name": "伪演化部", "fields": {"负责人": {"type": "evolving", "trajectory": [
            {"session": 0, "value": "张三"}, {"session": 2, "value": "张三"}]}}},
        {"name": "健康部", "fields": {"指标": {"type": "evolving", "trajectory": [
            {"session": 0, "value": "1"}, {"session": 1, "value": "5"}, {"session": 2, "value": "2"}]}}},
    ]}
    vws, _ = assemble_world(vd_table)
    vt = {(d["entity"], d["type"]) for d in validate(vws, vd_table)}
    ck("validate 抓单调(单调部.指标)", ("单调部", "monotonic") in vt, True)
    ck("validate 抓伪演化(伪演化部.负责人)", ("伪演化部", "fake_evolving") in vt, True)
    ck("validate 放过健康非单调字段(健康部.指标)", ("健康部", "monotonic") in vt, False)

    # ★Fix3:表面塌缩近重名检测
    ck("_strip_disambig 剥括号/下划线/空格/数字/字母尾缀",
       all(_strip_disambig(n) == "张三" for n in ["张三(数据)", "张三_数据", "张三 数据", "张三2", "张三A"]), True)
    ck("_strip_disambig 不误剥 CJK 尾字(张三丰≠张三,真不同名)", _strip_disambig("张三丰") == "张三丰", True)
    nc_ws = WorldState({"张三": {"f": Timeline([Op(0, _date_of(0), SET, "x", None)])},
                        "张三(数据)": {"f": Timeline([Op(0, _date_of(0), SET, "y", None)])},
                        "李四": {"f": Timeline([Op(0, _date_of(0), SET, "z", None)])}}, n_sessions=1)
    nc = name_collisions(nc_ws)
    ck("name_collisions 抓张三系表面塌缩", any(set(g) == {"张三", "张三(数据)"} for g in nc), True)
    ck("name_collisions 放过真不同名(李四不入塌缩组)", any("李四" in g for g in nc), False)

    # ── ★L10 敏感值确定性合成 + 校验位自洽(纯代码可单测)──
    ck("旗舰身份证 411328198503127537 过 GB11643", gb11643_check("411328198503127537"), True)
    ck("坏身份证 …753X 不过校验(旗舰反例)", gb11643_check("41132819850312753X"), False)
    ck("synth_id_card 确定性(同 seed 同值)", synth_id_card("s1") == synth_id_card("s1"), True)
    ck("synth_id_card 过 GB11643", gb11643_check(synth_id_card("s1")), True)
    ck("synth_id_card 异 seed 异值", synth_id_card("s1") != synth_id_card("s2"), True)
    ck("Luhn 已知过卡号 4539578763621486", luhn_check("4539578763621486"), True)
    ck("Luhn 坏卡号 4539578763621487 不过", luhn_check("4539578763621487"), False)
    ck("synth_bankcard 过 Luhn", luhn_check(synth_bankcard("s1")), True)
    ck("synth_bankcard 16 位", len(synth_bankcard("s1")) == 16, True)
    ck("synth_apikey 过 sk- 正则", bool(APIKEY_RE.fullmatch(synth_apikey("s1"))), True)
    ck("synth_secret 长度≥8", len(synth_secret("s1")) >= 8, True)
    ck("sensitive_value_wellformed pii_id 真", sensitive_value_wellformed("pii_id", synth_id_card("s1")), True)
    ck("sensitive_value_wellformed pii_id 伪(坏校验)", sensitive_value_wellformed("pii_id", "41132819850312753X"), False)
    ck("sensitive_value_wellformed bankcard 真", sensitive_value_wellformed("bankcard", synth_bankcard("s1")), True)
    ck("sensitive_value_wellformed apikey 真", sensitive_value_wellformed("apikey", synth_apikey("s1")), True)
    ck("dispatch synth_sensitive(bankcard) 过 Luhn", luhn_check(synth_sensitive("bankcard", "s9")), True)
    ck("高熵敏感值 熵 > 低熵 123456", value_entropy_bits(synth_secret("s1")) > value_entropy_bits("123456"), True)
    # 敏感侧信道随盘往返
    ws_s = build_demo_world()
    ws_s.sensitive = [{"entity": E, "field": "登录口令", "value": synth_secret("s1"),
                       "stype": "secret", "session": 0}]
    ck("ws.sensitive 序列化往返", WorldState.from_dict(ws_s.to_dict()).sensitive == ws_s.sensitive, True)

    # ── ★L9 apply_rule 纯查表 + 侧信道随盘 ──
    RULE = {"rule_id": "R1", "trigger_field": "响应时长", "op": ">=", "unit": "分钟",
            "thresholds": [{"cutoff": 30, "action": "升级为紧急工单"}, {"cutoff": 60, "action": "上报主管"}],
            "default_action": "常规处理"}
    ck("apply_rule x=10 → default(常规处理)", apply_rule(RULE, 10), "常规处理")
    ck("apply_rule x=45 → 升级为紧急工单(中臂)", apply_rule(RULE, 45), "升级为紧急工单")
    ck("apply_rule x=90 → 上报主管(高臂)", apply_rule(RULE, 90), "上报主管")
    ck("apply_rule 边界 x=30(>=)→ 升级", apply_rule(RULE, 30), "升级为紧急工单")
    ck("apply_rule 边界 x=60(>=)→ 上报主管", apply_rule(RULE, 60), "上报主管")
    ck("apply_rule 带单位串 '45分钟' 可抽数", apply_rule(RULE, "45分钟"), "升级为紧急工单")
    RULE_LE = {"rule_id": "R2", "trigger_field": "库存", "op": "<=", "default_action": "正常",
               "thresholds": [{"cutoff": 5, "action": "紧急补货"}, {"cutoff": 20, "action": "计划补货"}]}
    ck("apply_rule <= x=3 → 紧急补货(最小满足臂)", apply_rule(RULE_LE, 3), "紧急补货")
    ck("apply_rule <= x=15 → 计划补货", apply_rule(RULE_LE, 15), "计划补货")
    ck("apply_rule <= x=50 → default(正常)", apply_rule(RULE_LE, 50), "正常")
    ck("rule_action_set 完整动作集(含 default)",
       rule_action_set(RULE), ["升级为紧急工单", "上报主管", "常规处理"])
    ws_r = build_demo_world()
    ws_r.conditional_rules = [RULE]
    ws_r.rule_instances = [{"rule_id": "R1", "inst_id": "T-1", "x": 45,
                            "canon_action": "升级为紧急工单", "surface_action": "标记为紧急工单", "session": 0}]
    _rt = WorldState.from_dict(ws_r.to_dict())
    ck("ws.conditional_rules 序列化往返", _rt.conditional_rules, [RULE])
    ck("ws.rule_instances 序列化往返", _rt.rule_instances, ws_r.rule_instances)

    npass = sum(1 for c in checks if c[0])
    for ok, name, got, want in checks:
        if not ok:
            print(f"  ✗ {name}: got={got!r} want={want!r}")
    print(f"\n[world_state self-test] {npass}/{len(checks)} PASS")
    return npass == len(checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
