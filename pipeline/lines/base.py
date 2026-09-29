"""
pipeline.lines.base —— ProductionLine 抽象基类 + Order 契约(redesign §6 命门2 的代码落地)。

一条产线 = 一个 ProductionLine 子类,自带四件套(穿过 3 个 stage):
  prepare(ws, profile)   —— world stage:把本线需要的【世界基质】备进共享世界(默认 no-op)
  enumerate(ws, target, wp) —— orders stage:点菜,选 (entity,field,cap,aux) 并烘焙代码 gt → list[订单 dict]
  gt(ws, order)          —— ★强制抽象:给订单规格,用【代码】算答案(护城河 + 校验闸)
  intent(order)          —— questions stage:(提问意图, 须隐藏词表)供出题

加一条线 = 新增一个 lines/Lx.py 子类 + 在 lines/__init__.LINES 加一行,别处不动。
详见 docs/anchors/production_line_design.md。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ════════════════════════════════════════════════════════════════════════════
# ★疑问词单一真源(Fix2):答案【类型】决定疑问词,类型 = 字段 kind(命门1 扩展到表面)。
#   kind 先查世界 field_schema(单一真源),查不到按【值】兜底;person 只能来自 schema(名字串猜不出"人")。
#   各产线 intent 一律 interrogative(ans_kind) 派生,不许写死"是谁/是多少" → 治"管理跨度是谁"类类型错配整类。
# ════════════════════════════════════════════════════════════════════════════
_INTERROGATIVE = {"person": "是谁", "numeric": "是多少", "number": "是多少"}  # ★词表对齐 field_schema(kind="numeric");"number" 留作历史别名


def field_kind(field_name: str, sample_value: Any = None, profile: dict | None = None) -> str:
    """字段答案类型:person / number / 其它(text/status/category…)。先 schema 后值兜底。
    ★审计★3(诚实降级,非 bug):值兜底【出不了 person】(名字串猜不出"人")——schema 没标 kind 时
    人名字段降级判 text → 疑问词"是什么"(而非"是谁")。这是【安全降级】:绝不会错成"是多少"。
    person 的正确性外包给白皮书 field_schema 的 kind 质量;要硬保证须在白皮书校验阶段强制 person 字段标 kind。"""
    schema = {f.get("name"): f.get("kind") for f in (profile or {}).get("field_schema", [])}
    k = schema.get(field_name)
    if k:
        return k
    from pipeline.world_state import _to_num
    return "numeric" if _to_num(sample_value) is not None else "text"   # ★兜底用 'numeric'(与 schema 同词表),不再 'number'


def interrogative(kind: str | None) -> str:
    """kind → 疑问词。person=是谁 / number=是多少 / 其它=是什么(status/category 等)。"""
    return _INTERROGATIVE.get(kind or "", "是什么")


def sample_field_value(ws, entity: str, field_name: str):
    """从世界取该 (实体,字段) 的一个代表值(给 field_kind 的值兜底用)。"""
    tl = getattr(ws, "entities", {}).get(entity, {}).get(field_name)
    sv = tl.set_values() if tl else []
    return sv[-1][2] if sv else None


@dataclass
class Order:
    """枚举期的内部脚手架;enumerate 最终包成 line-tagged dict 交给下游(gt 已烘焙)。"""
    capability: str
    entity: str
    field: str
    gt: Any                                  # 代码算好的真答案(state-diff 烘焙)
    evidence_sessions: list[int] = field(default_factory=list)  # 必须读齐(≥2 = 跨文档)
    question_date: Optional[str] = None      # KU/FORGET 的切片时点
    aux: dict = field(default_factory=dict)  # 能力专属(MR.agg / L2.path,bridge …)


class ProductionLine(ABC):
    # ── 身份(= 旧 LineSpec 字段,并进类属性)──
    id: str = ""
    title: str = ""
    memory: str = ""
    gt_substrate: str = ""           # gt 基质描述(护城河,人读)
    implemented: bool = True

    # 显式 world_blueprint 冻结后，默认禁止 prepare 再改世界。只有不修改实体、
    # 关系、事件或 canonical 时间线，仅派生评测证据侧信道的产线，才可显式打开。
    typed_overlay_safe: bool = False

    # ── 依赖声明:本线产题所需的【世界基质特征】(可代码判定的结构化标签)──
    #   Skill-it 依赖图思想的【忠实落地】:我们是评测生成器、不做在线训练混合,
    #   所以这里不是"按学习速度动态重配权重",而是"每条产线依赖世界里的某种基质特征才能产题"。
    #   约定标签(由 feasible 据此轻量判定):
    #     "person_fields>=N"     —— profile.field_schema 里 kind==person 的字段数 ≥ N(造软外键链)
    #     "multi_event_timelines"—— 存在实体携带 ≥3 个变更事件(gt_event_order 才有料)
    #     "text_fields"          —— 存在文本类字段(person/status/category 或值全非数值;供注入矛盾)
    #   空 = 根线,只需基础世界(随时间演化的字段),恒可行。
    requires: list[str] = []

    # ── world stage:把本线需要的基质叠进【共享世界】(命门1)。默认无需额外基质。──
    def prepare(self, ws, profile: dict) -> Optional[str]:
        """返回一句日志(被编排打印)或 None。"""
        return None

    # ── 激活可行性闸:在 prepare 之前判断【当前世界/画像】能否喂饱本线的基质需求。──
    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        """返回 (能否产题, 一句人读原因)。基类默认无依赖(根线)→ 恒 True;
        有 requires 的子类按各自标签覆写轻量判定(数 person 字段 / 看事件数 / 看文本字段)。
        作用:把"基质不满足 → 产线静默产 0 单"显式化、可日志化(替代 enumerate 默默吐空)。"""
        return True, "无基质依赖(根线)"

    # ── orders stage:点菜 + 烘焙 gt → line-tagged 订单 dict 列表。──
    @abstractmethod
    def enumerate(self, ws, target: int, wp: Optional[dict] = None) -> list[dict]:
        ...

    # ── ★护城河:给订单规格,用代码重算答案(L1 按 capability 派发;L2 = gt_multihop)。──
    @abstractmethod
    def gt(self, ws, order: dict) -> Any:
        ...

    # ── questions stage:(提问意图, 须隐藏词表)。──
    @abstractmethod
    def intent(self, order: dict) -> tuple[str, list]:
        ...

    # ── ★命门3 接地闸第5件套(§G):gold 必须在【渲染语料】里逐字+就近归属可验,否则弃题。──
    def ground(self, order: dict, evidence_docs: list, all_signal_text: str = "") -> tuple:
        """默认:gold 单标量就近归属于实体(§G.5 默认行)。gt 非单标量的线(L1 多 cap / L2 / L3 / L5)覆写。
        返回 (status, reason),status ∈ {"grounded","drop"};无标量可抽 = fail-closed 弃(§G.12.4)。"""
        from pipeline.grounding import gold_scalar, attributed
        s = gold_scalar(order)
        if not s:
            return ("drop", "fail-closed:无法从 gt 抽取待验标量(该线应覆写 ground())")
        ent = order.get("entity", "")
        if attributed(s, ent, [d["content"] for d in evidence_docs]):
            return ("grounded", f"'{s}' 就近归属「{ent}」")
        return ("drop", f"'{s}' 未在证据文档就近归属「{ent}」")

    # ── ★边 A 闸(验证三角 A 边:题面↔答案【良定义】)。与 ground()(B 边)对称的第6件套。──
    def well_posed(self, order: dict, ws) -> tuple:
        """gold 是该订单问题意图在【世界】里【唯一、合法、可复算】的答案吗?是 → 留,否 → 弃。
        纯代码、零 LLM、绝不碰 corpus;跑在 orders→questions 支(出题前剔题)。
        返回 (status, reason),status ∈ {"well_posed","drop"}。
        ★默认 pass-through(opt-in 收紧,非 fail-closed):未实现的线不拦题。
        已建线(L1/L2/L3/L5)各自覆写,设计见 docs/anchors/edge_a/*.md(经对抗式 QA 实测背书)。"""
        return ("well_posed", "")

    def __repr__(self) -> str:
        return f"<Line {self.id}({self.title}){'' if self.implemented else ' [未建]'}>"
