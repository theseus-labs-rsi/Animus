"""Compile delivery demand before accepting a world; verify actual capacity later.

Schema bounds reject impossible plans. The original order enumerators certify
the completed world's capacity. Neither an author's estimate nor a type count
can substitute for that second gate.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pipeline.blueprint_feasibility import BlueprintReviewRequested
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint

POLICY = "capacity_contract/v2"
VERSION = "supply-driven/v3"


def revision_implementation():
    folder = Path(__file__).parent
    return digest({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   for name in ("supply_capacity.py", "world_blueprint.py", "blueprint_feasibility.py",
                                "seed_pack.py", "seed_world.py", "instance_plan.py", "capability_contract.py")})
RULES = {
    "L1_timeline": "独立对象属性的可公开跨期变化；区分发生时间与公开时间。",
    "L2_relational": "独立关系链及公开时点；需要分散的关系证据，避免同一答案换起点。",
    "L3_process": "独立真实事件链，每链至少三个跨期事件；公开材料分别保留各事件。",
    "L4_preference": "明确偏好轴上的多次独立决策与实际取舍，保留适用条件。",
    "L5_conflict": "独立对象属性上的来源分歧，每项有双方来源、时点与裁决依据。",
    "L6_refusal": "真实观察缺口；避免集中使用错实体类型或统一越界时间模板。",
    "L7_consolidation": "独立对象数值轨迹至少四期，或独立自然比较组；真实变化经枚举器核验。",
    "L8_transition": "独立状态对象，末期有唯一合法后继；已终结对象不计当前态供给。",
}
AUTHOR_SYSTEM = """你负责交付目标驱动的世界容量规划。世界还未发布，读取原业务、seed与实际逐线缺口。
在授权上限内调整现有类型数量与事件/关系实例下限；保留exact基数、原字段语义、业务规则、种子必需结构与时间范围。
增加独立案例、事件链、数值轨迹和状态对象；同一事实改名、排列和重复不计新增供给。
只能输出已有类型的entity_counts、已有事件的event_min_counts和已有关系的relation_min_counts。
每个数量为非负整数，实体类型数量至少原种子下限，不能缩小当前类型数量。exact类型保持原值。
line_constructions逐项覆盖请求能力线，说明新增结构如何支撑要求的独立候选数量。
若现有schema或授权上限容纳不下需求，decision=unresolved并给出具体缺口，不提出越界数量。
严格JSON：{"decision":"revise|unresolved","reason":"依据","entity_counts":{},"event_min_counts":{},
"relation_min_counts":{},"line_constructions":[{"line":"原能力线id","construction":"独立结构及公开证据安排"}]}。"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def enabled(wp):
    return (wp.get("supply_plan") or {}).get("version") == 2


def requirements(allocation):
    from pipeline.capability_contract import requirements as capability_requirements
    return [capability_requirements(line,amount) for line,amount in allocation.items() if amount > 0]


def blueprint_capacity_report(blueprint, brief):
    bp = normalize_world_blueprint({"world_blueprint": deepcopy(blueprint)})
    allocation = brief["candidate_allocation"]
    types = bp["entity_types"]
    periods = bp["temporal_model"]["n_sessions"]
    total = sum(t["count"] for t in types)
    bounds = {
        "L5_conflict": sum(t["count"] * sum(f.get("kind") in {"person", "status", "category", "text", "reference"}
                            for f in t["fields"]) for t in types),
        "L7_consolidation": (sum(t["count"] * sum(f.get("kind") in {"numeric", "number"} for f in t["fields"])
                              for t in types) if periods >= 4 else 0)
                             + sum((t["count"] // 2) * sum(bool(f.get("name")) for f in t["fields"]) for t in types),
        "L8_transition": sum(t["count"] * sum(len(f.get("states") or []) >= 2 for f in t["fields"])
                              for t in types),
    }
    rows, issues = [], []
    for line, required in allocation.items():
        bound = bounds.get(line)
        row = {"line": line, "required_candidates": required, "schema_upper_bound": bound,
               "certificate_kind": "proven_upper_bound" if bound is not None else "unknown",
               "status": "impossible" if bound is not None and bound < required else "requires_actual_world_inventory"}
        rows.append(row)
        if row["status"] == "impossible":
            issues.append({"line": line, "code": "schema_capacity_below_reserve", **row})
    limits = brief.get("world_limits") or {}
    if limits.get("max_world_entities") is not None and total > limits["max_world_entities"]:
        issues.append({"code": "entity_limit", "actual": total, "limit": limits["max_world_entities"]})
    if limits.get("time_span_weeks") is not None and periods != limits["time_span_weeks"]:
        issues.append({"code": "period_contract", "actual": periods, "required": limits["time_span_weeks"]})
    return {"version": POLICY, "blueprint_hash": digest(bp), "rows": rows, "issues": issues,
            "status": "blocked" if issues else "awaiting_actual_world_capacity", "entity_count": total,
            "requirements": requirements(allocation), "evidence_basis": "schema_upper_bounds_not_realized_supply"}


class CapacityReviewRequested(BlueprintReviewRequested):
    """Route quantity deficits to capacity planning, separate from semantic repair."""


def revise_capacity(wp, tracer, feedback, audit, *, checkpoint=None):
    try:
        return _revise_capacity(wp,tracer,feedback,audit,checkpoint=checkpoint)
    finally:
        if checkpoint is not None:checkpoint(audit)


def _revise_capacity(wp, tracer, feedback, audit, *, checkpoint=None):
    """Revise an unpublished quantity plan through restricted, audited edits."""
    from pipeline.instance_plan import enabled, create
    def save():
        if checkpoint is not None:checkpoint(audit)
    if enabled(wp):
        from pipeline.capability_contract import requirements as capability_requirements
        current = deepcopy(wp)
        current["supply_plan"]["requirements"] = [capability_requirements(line,amount)
            for line,amount in current["supply_plan"]["candidate_allocation"].items() if amount]
        installed = current.get("business_instance_plan")
        proposal = ({"decision": "plan", "reason": installed.get("reason", ""),
            "units": deepcopy(installed["units"]), "cohorts": deepcopy(installed["cohorts"]),
            **({"execution_policy": installed["execution_policy"]} if installed.get("execution_policy") else {})} if installed else None)
        audit.update(before=deepcopy(wp),feedback=deepcopy(feedback),implementation=revision_implementation())
        result = create(current,tracer,feedback,audit,initial_proposal=proposal,repair_required=bool(proposal),checkpoint=checkpoint)
        audit.update(before=deepcopy(wp),feedback=deepcopy(feedback),implementation=revision_implementation())
        return result
    from pipeline.seed_pack import validate_seed_blueprint
    from pipeline.seed_world import seed_business_context
    from pipeline.blueprint_feasibility import assess
    bp = normalize_world_blueprint(wp)
    plan = wp["supply_plan"]
    payload = {"business": seed_business_context(wp), "blueprint": bp,
               "demand": requirements(plan["candidate_allocation"]),
               "world_limits": plan.get("world_limits", {}), "actual_deficits": feedback}
    audit.update(version=POLICY, implementation=revision_implementation(),
                 input_hash=digest(payload), before=deepcopy(wp), feedback=deepcopy(feedback))
    audit['author_call']={'status':'requested'};save()
    try:
        raw = tracer.chat_json("council.supply_capacity", [{"role": "system", "content": AUTHOR_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.2, max_tokens=8192, retries=3, strict_json=True)
    except BaseException as error:
        audit['author_call'].update(status='call_failed',error_type=type(error).__name__);save();raise
    audit["proposal"] = deepcopy(raw)
    audit['author_call']['status']='response_received';save()
    from execution_control import global_stop_exception
    stopped=global_stop_exception(raw)
    if stopped is not None:
        audit['author_call']['status']='provider_error';save();raise stopped
    if not isinstance(raw, dict) or raw.get("decision") != "revise" or not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
        audit["status"] = "unresolved"
        raise WorldBlueprintError("Supply capacity architect could not fit the requested independent structures: " + str(raw.get("reason") if isinstance(raw, dict) else raw))
    constructions = raw.get("line_constructions")
    desired = {r["line"] for r in requirements(plan["candidate_allocation"])}
    if (not isinstance(constructions, list) or len(constructions) != len(desired)
            or any(not isinstance(r, dict) or not isinstance(r.get("construction"), str) or not r["construction"].strip() for r in constructions)
            or {r.get("line") for r in constructions} != desired):
        raise WorldBlueprintError("Capacity revision needs one factual construction for every requested line")
    result = deepcopy(wp)
    revised = result["world_blueprint"]
    for group, argument, field in [("entity_types", "entity_counts", "count"),
                                   ("event_types", "event_min_counts", "min_count"),
                                   ("relation_types", "relation_min_counts", "min_count")]:
        edits = raw.get(argument)
        originals = {r["id"]: r for r in bp[group]}
        if not isinstance(edits, dict) or set(edits) - set(originals):
            raise WorldBlueprintError("Capacity revision contains unknown declarations")
        for row in revised[group]:
            amount = edits.get(row["id"], row.get(field, 0))
            if type(amount) is not int or amount < row.get(field, 0):
                raise WorldBlueprintError("Capacity revision may only increase declared instance counts")
            if group == "entity_types" and row.get("cardinality_policy") == "exact" and amount != row[field]:
                raise WorldBlueprintError("Capacity revision changed an exact identity count")
            row[field] = amount
    revised = normalize_world_blueprint(result)
    result["world_blueprint"] = revised
    report = blueprint_capacity_report(revised, plan)
    audit["capacity"] = report
    if report["issues"]:
        raise WorldBlueprintError("Capacity revision violates declared scale/capacity limits: " + json.dumps(report["issues"]))
    if result.get("seed_contract"):
        seed = validate_seed_blueprint(result, result["seed_contract"])
        if not seed["passed"]:
            raise WorldBlueprintError("Capacity revision violates original seed: " + str(seed["issues"]))
        result["seed_audit"] = seed
    if revised == bp:
        raise WorldBlueprintError("Capacity revision made no structural change")
    result.setdefault("shared_world_spec", {}).setdefault("entities", {})["count"] = report["entity_count"]
    result["supply_plan"]["capacity_blueprint"] = report
    result["supply_plan"]["line_constructions"] = deepcopy(constructions)
    result.setdefault("domain_profile", {}).update(supply_policy=POLICY,
        l5_max_conflicts=plan["candidate_allocation"].get("L5_conflict", 0))
    if result.get("seed_contract"):
        audit['business_review_call']={'status':'requested'};save()
        try:
            review = assess(result, tracer, feedback={"capacity_request": feedback, "proposal": raw})
        except BaseException as error:
            audit['business_review_call'].update(status='call_failed',error_type=type(error).__name__);save();raise
        audit['business_review_call'].update(status='response_received',response=deepcopy(review));save()
        audit["business_review"] = review
        if review["decision"] != "accept":
            raise WorldBlueprintError("Capacity revision business feasibility unresolved")
    audit.update(status="accepted", after=deepcopy(result), after_hash=digest(result))
    return result


def require_order_capacity(run):
    """Hard gate before allocating any corpus/render/review work."""
    wp = run.read("01_whitepaper.json")
    if not enabled(wp):
        return
    from collections import Counter
    observed = Counter(q["line"] for q in run.read("03_orders.json"))
    required = wp["supply_plan"]["candidate_allocation"]
    deficits = {line: amount - observed[line] for line, amount in required.items() if observed[line] < amount}
    receipt = {"version": POLICY, "world_hash": digest(run.read("02_world.json")),
               "orders_hash": digest(run.read("03_orders.json")), "required": required,
               "actual": dict(observed), "deficits": deficits, "passed": not deficits}
    run.write("03_capacity_gate.json", receipt)
    if deficits:
        raise CapacityReviewRequested({"reason": "Actual well-posed order supply is below the per-line reserve",
                                       "capacity_gate": receipt})
