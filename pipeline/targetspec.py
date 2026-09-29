"""
pipeline.targetspec —— 闭环 TargetSpec(closed_loop_targetspec_design.md §1/§3 的代码落地,v0)。

把"给 tokens、题数随缘"升成"给目标 floor、反推世界规模"。本模块只放【纯代码、可离线自检】的两件:
  · TargetSpec  —— 旋钮面板(min_questions / per_line_min / …);
  · invert_rate —— 反推率模型:per_line_min + 保守 survival → 世界参数(n_ent/n_sess/quota_L1/max_n_conflicts)。

★盲审硬化(见设计 §3/§9):率模型【只给起点、不求精度】—— 用【保守】survival(实测×0.8 量级)做除数 +
  slack 放大 + 输出【夹 clamp】防反推出几百实体;真正的纠偏来自 build_to_target 的【②实测环】(Slice 2)。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# ── 保守先验(run #3 实测 ×~0.8;盲审 C1:survival 方差近 2× 且偏乐观,必须保守)──
SURVIVAL_PRIOR = {"L1_timeline": 0.75, "L2_relational": 0.85, "L3_process": 0.45, "L5_conflict": 0.85}
# ── 粗率模型(run #3 标定 ×~0.8;盲审 C2:单点不可信,只当起点,②环纠偏)──
RATE_L2_PER_PERSON = 0.5      # 每人员实体 ~产 0.5 个 L2 两跳题
RATE_L3_PER_ENTITY = 0.4      # 每实体 ~产 0.4 个 L3 排序题(有 ≥3 跨周事件的比例)
N_ENT_CLAMP = (8, 80)         # 反推世界规模上下夹(防几百实体 / 防太小)
N_SESS_CLAMP = (6, 26)
DEFAULT_SLACK = 1.4           # over-provision 余量(盲审 C1:survival 不稳,slack 偏大)


@dataclass
class TargetSpec:
    """旋钮面板(能设的那一半);并入白皮书 §4 schema。"""
    min_questions: int = 80
    per_line_min: dict = field(default_factory=dict)        # {line_id: 下限};Σ ≤ min_questions
    per_line_max: dict = field(default_factory=dict)        # 可选上限(防某线爆)
    difficulty_dist: dict = field(default_factory=dict)     # v1:{easy,med,hard};v0 不用
    haystack_ratio: float = 9.0                             # filler_chars / other_body_chars
    time_span_weeks: int | None = None                      # None = 用默认/反推
    total_only: bool = False                               # weighted line targets remain soft; explicit floors still apply
    max_world_entities: int = 80                           # resource bound for each independent world

    @classmethod
    def from_dict(cls, d: dict) -> "TargetSpec":
        d = d or {}
        return cls(min_questions=int(d.get("min_questions", 80)),
                   per_line_min=dict(d.get("per_line_min", {})),
                   per_line_max=dict(d.get("per_line_max", {})),
                   difficulty_dist=dict(d.get("difficulty_dist", {})),
                   haystack_ratio=float(d.get("haystack_ratio", 9.0)),
                   time_span_weeks=d.get("time_span_weeks"),
                   total_only=bool(d.get("total_only", False)),
                   max_world_entities=int(d.get("max_world_entities", 80)))

    def derive_per_line_min(self, active_lines: list[dict]) -> dict:
        """缺省:由白皮书 active_lines.weight × min_questions 派生每线 floor(盲审:weight 终于被读)。
        已显式给 per_line_min 的线不覆盖。"""
        if not active_lines:
            return dict(self.per_line_min)
        wsum = sum(float(l.get("weight") or 0) for l in active_lines) or 1.0
        out = dict(self.per_line_min)
        for l in active_lines:
            lid = l.get("line")
            if lid and lid not in out:
                out[lid] = max(1, round(self.min_questions * float(l.get("weight") or 0) / wsum))
        return out


@dataclass
class WorldParams:
    """invert_rate 的产物:喂给 build_world(覆盖白皮书 entities.count / timeline.n_sessions)+ run_lines 配额。"""
    n_entities: int
    n_sessions: int
    quota_L1: int                       # L1 配额(取代写死 _PLAN(29))
    max_n_conflicts: int                # L5 注入条数(取代写死 3)
    target_orders: dict                 # {line_id: 该产多少 order}(已 over-provision)


def _clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))


def invert_rate(spec: TargetSpec, survival: dict | None = None,
                slack: float = DEFAULT_SLACK, base_min_entities: int = 12) -> WorldParams:
    """反推:per_line_min + 保守 survival → 满足供给的【最小】世界 + 各线 order 配额。
    target_orders_Lx = ceil(per_line_min_Lx / survival_Lx) · slack  (over-provision);世界规模由各线反推取 max,夹 clamp。"""
    sv = {**SURVIVAL_PRIOR, **(survival or {})}
    plm = spec.per_line_min or {}

    target_orders: dict = {}
    for lid, floor in plm.items():
        s = max(sv.get(lid, 0.6), 0.2)                      # survival 下限保护(别除爆)
        target_orders[lid] = max(int(floor), math.ceil(floor / s * slack))

    n_person = math.ceil(target_orders.get("L2_relational", 0) / RATE_L2_PER_PERSON) if target_orders.get("L2_relational") else 0
    n_ent_L3 = math.ceil(target_orders.get("L3_process", 0) / RATE_L3_PER_ENTITY) if target_orders.get("L3_process") else 0
    if not N_ENT_CLAMP[0] <= spec.max_world_entities <= N_ENT_CLAMP[1]:
        raise ValueError("max_world_entities must be between 8 and 80")
    n_entities = _clamp(max(base_min_entities, n_person, n_ent_L3), N_ENT_CLAMP[0], spec.max_world_entities)
    n_sessions = _clamp(spec.time_span_weeks or 10, *N_SESS_CLAMP)
    quota_L1 = int(target_orders.get("L1_timeline", 0))
    max_n_conflicts = int(target_orders.get("L5_conflict", 0))
    return WorldParams(n_entities=n_entities, n_sessions=n_sessions,
                       quota_L1=quota_L1, max_n_conflicts=max_n_conflicts, target_orders=target_orders)


# 自检:python -m pipeline.targetspec
if __name__ == "__main__":
    import sys
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    spec = TargetSpec(min_questions=80,
                      per_line_min={"L1_timeline": 30, "L2_relational": 15, "L3_process": 15, "L5_conflict": 8})
    wp = invert_rate(spec)
    ck("target_orders 全部 over-provision(≥ per_line_min)",
       all(wp.target_orders[k] >= spec.per_line_min[k] for k in spec.per_line_min))
    ck("L3 over-provision 体现保守 survival(.45 → target≈floor/0.45·1.4)",
       wp.target_orders["L3_process"] >= math.ceil(15 / 0.45))
    ck("n_entities 在 clamp 内", N_ENT_CLAMP[0] <= wp.n_entities <= N_ENT_CLAMP[1])
    ck("n_sessions 在 clamp 内", N_SESS_CLAMP[0] <= wp.n_sessions <= N_SESS_CLAMP[1])
    ck("quota_L1 = L1 target_orders", wp.quota_L1 == wp.target_orders["L1_timeline"])
    ck("max_n_conflicts = L5 target_orders", wp.max_n_conflicts == wp.target_orders["L5_conflict"])

    # clamp:超大 floor → n_entities 夹到上限,不反推出几百
    big = TargetSpec(min_questions=999, per_line_min={"L3_process": 999})
    ck("超大 floor → n_entities 夹上限(防几百实体)", invert_rate(big).n_entities == N_ENT_CLAMP[1])

    # 保守 survival 比实测更小(反推世界更大,防 floor 不达)
    ck("保守 survival < run#3 实测(L3 .45 < .57)", SURVIVAL_PRIOR["L3_process"] < 0.57)

    # derive_per_line_min:weight 终于被读
    al = [{"line": "L1_timeline", "weight": 0.5}, {"line": "L2_relational", "weight": 0.5}]
    dpl = TargetSpec(min_questions=40).derive_per_line_min(al)
    ck("派生 per_line_min(weight×min_questions)", dpl.get("L1_timeline") == 20 and dpl.get("L2_relational") == 20)

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[targetspec self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
