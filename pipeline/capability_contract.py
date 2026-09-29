"""Versioned demand and certificates backed by the original production lines."""
from copy import deepcopy
import hashlib
import json

VERSION = "capability-demand/v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def requirements(line_id, quota, policy=None):
    if type(quota) is not int or quota < 0:
        raise ValueError("Capability quota must be a nonnegative integer")
    from pipeline.lines import line_for
    rows = {
        "L1_timeline": ("独立对象字段的跨期变化，明确发生和公开时间", ["public_temporal_change"]),
        "L2_relational": ("真实关系路径、有效期与分散公开证据", ["valid_relation_path"]),
        "L3_process": ("独立因果业务过程，至少三个跨期且唯一可定位的事件", ["distinct_locatable_episode"]),
        "L4_preference": ("明确偏好轴及独立决策过程", ["declared_preference_axis"]),
        "L5_conflict": ("独立属性主张的来源分歧、同一范围与裁决依据", ["distinct_source_claim"]),
        "L6_refusal": ("公开范围内的真实缺失；记录缺失子型与反证范围", ["public_absence_scope"]),
        "L7_consolidation": ("适用业务数值的跨期历史，或先验自然比较组", ["native_numeric_history", "natural_cohort"]),
        "L8_transition": ("独立状态对象，终期非终态且有唯一合法后继", ["nonterminal_unique_successor"]),
    }
    if line_id not in rows:
        raise ValueError("No executable capability contract: " + line_id)
    instruction, carriers = rows[line_id]
    result = {"version": VERSION, "line": line_id, "independent_candidates": quota,
        "construction_requirement": instruction, "carriers": carriers,
        "verification": "original enumerate / gt / well_posed",
        "distinctness": ["entity_roles", "field", "support_facts", "query_scope", "reasoning_operation"]}
    result.update(line_for(line_id).construction_spec())
    if line_id == "L7_consolidation":
        result["subtype_minimum"] = deepcopy((policy or {}).get("L7_subtype_minimum", {"native_trend": min(1, quota)}))
    return result


def family_key(order):
    """Ignore wording/IDs; retain the original factual scope and operation."""
    aux = deepcopy(order.get("aux") or {})
    for key in ("presentation_order", "display_order", "qid", "question", "intent"):
        aux.pop(key, None)
    for key in ("candidates", "events", "event_ids"):
        if isinstance(aux.get(key), list):
            aux[key] = sorted(aux[key], key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
    return digest({key: order.get(key) for key in
        ("line", "capability", "entity", "field", "question_date", "evidence_sessions")} | {"aux": aux})


def verify_order(line, world, order):
    status, reason = line.well_posed(order, world)
    recomputed = line.gt(world, order)
    passed = status == "well_posed" and recomputed == order.get("gt")
    return {"line": line.id, "family_key": family_key(order), "passed": passed,
        "well_posed": status, "reason": reason, "answer_matches": recomputed == order.get("gt"),
        "support": {key: deepcopy(order.get(key)) for key in
            ("entity", "field", "question_date", "evidence_sessions", "aux")}}


def actual_capacity(wp, world):
    """Enumerate an isolated world; preserve all original preparation and answers."""
    from pipeline.lines import LINES, prepare_lines, run_lines
    from pipeline.world_gen import preview_supply_structure
    shadow = deepcopy(world)
    preview_supply_structure(wp, shadow)
    prepare_lines(wp, shadow, lambda *_: None)
    stats = {}
    orders = run_lines(wp, shadow, lambda *_: None,
        question_budget=wp["supply_plan"]["candidate_budget"], stats=stats)
    lines = {line.id: line for line in LINES}
    # The production selector is a bounded sample. Certify the same bounded
    # original pool so an obligation is not rejected merely by sample order.
    limit = stats["candidate_limit_per_line"]
    pool = [order for row in stats["lines"] if row["feasible"] and row["allocated"]
            for order in lines[row["line"]].enumerate(shadow, limit, wp)]
    certificates, families, duplicates = [], set(), []
    for order in pool:
        certificate = verify_order(lines[order["line"]], shadow, order)
        key = (order["line"], certificate["family_key"])
        if key in families:
            duplicates.append(certificate)
        elif certificate["passed"]:
            families.add(key)
            certificates.append(certificate)
    return {"version": VERSION, "world_hash": digest(world.to_dict()),
        "certificates": certificates, "duplicates": duplicates,
        "original_statistics": stats, "orders_hash": digest(orders),
        "pool_limit_per_line": limit, "pool_hash": digest(pool)}
