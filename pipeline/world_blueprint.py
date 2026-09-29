"""场景世界蓝图：把白皮书里的“世界骨架”变成可校验、可执行的契约。

蓝图只描述领域本体：实体类型、软外键关系、领域事件与时间制度。记忆能力产线
不反向决定这些结构。历史白皮书没有 ``world_blueprint`` 时才走单类型兼容适配；
显式提供的新蓝图一旦非法就立即失败，避免悄悄退化回扁平字段表。
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from typing import Any


class WorldBlueprintError(ValueError):
    """白皮书世界骨架不满足可执行契约。"""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_number(value: Any) -> float | None:
    """把 JSON 数值或数字字符串规范成 float；领域单位串不在 blueprint range 中兜底。"""
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _looks_like_identity_field(name: str) -> bool:
    """判断字段是否只是实体身份/专名的重复表示，不应被事件改写。"""
    normalized = str(name or "").strip().lower().replace("_", "").replace("-", "")
    if normalized in {"name", "id", "identifier", "entityname", "entityid"}:
        return True
    return normalized.endswith(("名称", "姓名", "编号", "标识", "标识符", "唯一id", "唯一标识"))


def _legacy_blueprint(wp: dict) -> dict:
    """把历史白皮书投影成单类型蓝图；只用于读取没有显式蓝图的旧产物。"""
    profile = wp.get("domain_profile") or {}
    spec = wp.get("shared_world_spec") or {}
    timeline = spec.get("timeline") or {}
    return {
        "version": 1,
        "legacy_adapter": True,
        "entity_types": [{
            "id": "legacy_entity",
            "noun": profile.get("entity_noun") or "实体",
            "count": max(1, _as_int((spec.get("entities") or {}).get("count"), 10)),
            "primary": True,
            "fields": deepcopy(profile.get("field_schema") or []),
        }],
        "relation_types": [],
        "event_types": [],
        "temporal_model": {
            "unit": timeline.get("unit") or "week",
            "cadence": timeline.get("cadence") or "weekly",
            "n_sessions": max(2, _as_int(timeline.get("n_sessions"), 10)),
            "step_days": max(1, _as_int(timeline.get("step_days"), 7)),
        },
        "causal_rules": [],
        "evidence_channels": deepcopy(profile.get("doc_genres") or []),
        "invariants": [],
    }


def _canonical_blueprint(raw: dict) -> dict:
    """只做无损的形状归一，不替非法引用或缺失核心字段兜底。"""
    legacy = raw.get("legacy_adapter") is True
    entity_types = []
    for item in raw.get("entity_types") or []:
        if not isinstance(item, dict):
            entity_types.append(item)
            continue
        entity_types.append({
            **deepcopy(item),
            "id": str(item.get("id") or "").strip() if legacy else item.get("id"),
            "noun": str(item.get("noun") or "").strip() if legacy else item.get("noun"),
            "count": _as_int(item.get("count"), 0) if legacy else item.get("count"),
            "primary": bool(item.get("primary", False)) if legacy else item.get("primary", False),
            "fields": deepcopy(item.get("fields") or []),
        })

    relation_types = []
    for item in raw.get("relation_types") or []:
        if not isinstance(item, dict):
            relation_types.append(item)
            continue
        relation_types.append({
            **deepcopy(item),
            "id": str(item.get("id") or "").strip() if legacy else item.get("id"),
            "from_type": str(item.get("from_type") or "").strip() if legacy else item.get("from_type"),
            "to_type": str(item.get("to_type") or "").strip() if legacy else item.get("to_type"),
            "field": str(item.get("field") or "").strip() if legacy else item.get("field"),
            "temporal": bool(item.get("temporal", False)) if legacy else item.get("temporal"),
            "min_count": max(0, _as_int(item.get("min_count"), 1)) if legacy else item.get("min_count"),
        })

    event_types = []
    for item in raw.get("event_types") or []:
        if not isinstance(item, dict):
            event_types.append(item)
            continue
        event_types.append({
            **deepcopy(item),
            "id": str(item.get("id") or "").strip() if legacy else item.get("id"),
            "roles": deepcopy(item.get("roles") or {}),
            "effect_fields": deepcopy(item.get("effect_fields") or []),
            "min_count": max(0, _as_int(item.get("min_count"), 1)) if legacy else item.get("min_count"),
        })

    temporal = deepcopy(raw.get("temporal_model") or {})
    if legacy:
        temporal["step_days"] = _as_int(temporal.get("step_days"), 7)
        temporal["n_sessions"] = _as_int(temporal.get("n_sessions"), 10)

    return {
        **deepcopy(raw),
        "version": _as_int(raw.get("version"), 1) if legacy else raw.get("version", 1),
        "entity_types": entity_types,
        "relation_types": relation_types,
        "event_types": event_types,
        "temporal_model": temporal,
        "causal_rules": deepcopy(raw.get("causal_rules") or []),
        "evidence_channels": deepcopy(raw.get("evidence_channels") or []),
        "invariants": deepcopy(raw.get("invariants") or []),
    }


def relation_owner_side(blueprint: dict, relation: dict) -> str | None:
    """按关系两个端点的局部字段声明返回 ``from``/``to`` owner；非法或歧义返回 None。"""
    if not isinstance(blueprint, dict) or not isinstance(relation, dict):
        return None
    fields_by_type = {
        entity_type.get("id"): {
            field.get("name") for field in (entity_type.get("fields") or [])
            if isinstance(field, dict) and isinstance(field.get("name"), str)
        }
        for entity_type in (blueprint.get("entity_types") or [])
        if isinstance(entity_type, dict) and isinstance(entity_type.get("id"), str)
    }
    src, dst, field = relation.get("from_type"), relation.get("to_type"), relation.get("field")
    if not all(isinstance(value, str) for value in (src, dst, field)):
        return None
    source_owns = field in fields_by_type.get(src, set())
    if src == dst:
        return "from" if source_owns else None
    target_owns = field in fields_by_type.get(dst, set())
    if source_owns == target_owns:
        return None
    return "from" if source_owns else "to"


def event_role_requirements(blueprint: dict) -> dict[str, int]:
    """返回一次事件实例为保持角色互异，各实体类型至少需要的实例数。"""
    required: dict[str, int] = {}
    for event in blueprint.get("event_types") or []:
        if not isinstance(event, dict):
            continue
        multiplicity = Counter(
            type_id for type_id in (event.get("roles") or {}).values()
            if isinstance(type_id, str) and type_id
        )
        for type_id, count in multiplicity.items():
            required[type_id] = max(required.get(type_id, 0), count)
    return required


def repair_blueprint_candidate(value: dict) -> tuple[dict, list[str]]:
    """纯函数修复模型候选中可确定的 FK 归属与最小实例数。

    标量 FK 放在实体数更多的一端（并列稳定选 ``from``）；如果字段已经只在
    一个端点声明，则尊重该端点。函数不删除关系或事件，只删除未被任何关系
    使用的 ``reference`` 字段，把非法的 ``min_count`` 收敛到 1，并为同类型
    的多个事件角色提供足够的互异实体。声明为 exact 的类型不擅自扩容，留给
    蓝图硬校验要求模型重设计事件。
    """
    repaired = deepcopy(value)
    if not isinstance(repaired, dict):
        return repaired, []
    blueprint = repaired.get("world_blueprint", repaired)
    if not isinstance(blueprint, dict):
        return repaired, []

    raw_types = blueprint.get("entity_types")
    raw_relations = blueprint.get("relation_types")
    raw_events = blueprint.get("event_types")
    if not isinstance(raw_types, list):
        raw_types = []
    if not isinstance(raw_relations, list):
        raw_relations = []
    if not isinstance(raw_events, list):
        raw_events = []
    types = {
        item.get("id"): item for item in raw_types
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    repairs: list[str] = []

    def _fields(entity_type: dict) -> list:
        fields = entity_type.get("fields")
        return fields if isinstance(fields, list) else []

    def _has_field(entity_type: dict, name: str) -> bool:
        return any(isinstance(item, dict) and item.get("name") == name
                   for item in _fields(entity_type))

    def _entity_count(entity_type: dict) -> int:
        count = entity_type.get("count")
        return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0

    def _choose_side(relation: dict, source: dict, target: dict, field_name: str) -> str:
        if relation.get("from_type") == relation.get("to_type"):
            return "from"
        source_has = _has_field(source, field_name)
        target_has = _has_field(target, field_name)
        if source_has != target_has:
            return "from" if source_has else "to"
        if _entity_count(target) > _entity_count(source):
            return "to"
        return "from"

    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        relation_id = relation.get("id") or "?"
        minimum = relation.get("min_count")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            relation["min_count"] = 1
            repairs.append(f"relation {relation_id}.min_count→1")

        source_id, target_id, field_name = (
            relation.get("from_type"), relation.get("to_type"), relation.get("field"))
        source, target = types.get(source_id), types.get(target_id)
        if (source is None or target is None or not isinstance(field_name, str)
                or not field_name or field_name != field_name.strip()):
            continue
        owner_side = _choose_side(relation, source, target, field_name)
        owner = source if owner_side == "from" else target
        owner_id = source_id if owner_side == "from" else target_id
        other = target if owner_side == "from" else source
        other_id = target_id if owner_side == "from" else source_id

        owner_fields = _fields(owner)
        if not isinstance(owner.get("fields"), list):
            continue
        matches = [item for item in owner_fields
                   if isinstance(item, dict) and item.get("name") == field_name]
        if matches:
            kept = matches[0]
            if kept.get("kind") != "reference":
                kept["kind"] = "reference"
                repairs.append(f"relation {relation_id} owner {owner_id}.{field_name}.kind→reference")
            if len(matches) > 1:
                deduplicated = []
                kept_match = False
                for item in owner_fields:
                    is_match = isinstance(item, dict) and item.get("name") == field_name
                    if is_match and kept_match:
                        continue
                    kept_match = kept_match or is_match
                    deduplicated.append(item)
                owner["fields"] = deduplicated
                repairs.append(
                    f"relation {relation_id} owner {owner_id}.{field_name} 删除重复声明")
        else:
            owner_fields.append({"name": field_name, "kind": "reference"})
            repairs.append(f"relation {relation_id} 创建 owner {owner_id}.{field_name}")

        if source_id != target_id and isinstance(other.get("fields"), list):
            before = len(other["fields"])
            other["fields"] = [
                item for item in other["fields"]
                if not (isinstance(item, dict) and item.get("name") == field_name)
            ]
            if len(other["fields"]) != before:
                repairs.append(f"relation {relation_id} 删除非 owner {other_id}.{field_name}")
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        minimum = event.get("min_count")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            event["min_count"] = 1
            repairs.append(f"event {event.get('id') or '?'}.min_count→1")

    for type_id, required in event_role_requirements(blueprint).items():
        entity_type = types.get(type_id)
        if not entity_type or entity_type.get("cardinality_policy") == "exact":
            continue
        if _entity_count(entity_type) < required:
            entity_type["count"] = required
            repairs.append(f"entity {type_id}.count→{required}(event 角色互异容量)")

    bound_fields: set[tuple[str, str]] = set()
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        side = relation_owner_side(blueprint, relation)
        if side is None:
            continue
        owner_id = relation.get("from_type") if side == "from" else relation.get("to_type")
        bound_fields.add((owner_id, relation.get("field")))

    for type_id, entity_type in types.items():
        fields = entity_type.get("fields")
        if not isinstance(fields, list):
            continue
        kept = [
            item for item in fields
            if not (isinstance(item, dict) and item.get("kind") == "reference"
                    and (type_id, item.get("name")) not in bound_fields)
        ]
        removed = len(fields) - len(kept)
        if removed:
            entity_type["fields"] = kept
            repairs.append(f"entity {type_id} 删除 {removed} 个悬空 reference 字段")
    return repaired, repairs


def relation_capacity(blueprint: dict, relation: dict) -> int:
    """返回当前标量 FK 编译模型下，单一 relation type 可形成的最大有效实例数。"""
    side = relation_owner_side(blueprint, relation)
    if side is None:
        return 0
    type_counts = {
        entity_type.get("id"): entity_type.get("count")
        for entity_type in (blueprint.get("entity_types") or [])
        if isinstance(entity_type, dict)
        and isinstance(entity_type.get("count"), int)
        and not isinstance(entity_type.get("count"), bool)
    }
    owner_type = relation.get("from_type") if side == "from" else relation.get("to_type")
    referenced_type = relation.get("to_type") if side == "from" else relation.get("from_type")
    owner_count = max(0, type_counts.get(owner_type, 0))
    referenced_count = max(0, type_counts.get(referenced_type, 0))
    if not relation.get("temporal") or referenced_count < 2:
        return owner_count
    temporal = blueprint.get("temporal_model") or {}
    n_sessions = temporal.get("n_sessions")
    if isinstance(n_sessions, bool) or not isinstance(n_sessions, int):
        return 0
    return owner_count * max(0, n_sessions)


def normalize_world_blueprint(value: dict) -> dict:
    """返回规范蓝图。

    ``value`` 可为完整白皮书或裸蓝图。只有完整白皮书完全没有
    ``world_blueprint`` 键时才启用历史适配；显式蓝图非法会抛
    :class:`WorldBlueprintError`。
    """
    if not isinstance(value, dict):
        raise WorldBlueprintError("world blueprint 必须是 JSON object")
    if "world_blueprint" in value:
        raw = value.get("world_blueprint")
        if not isinstance(raw, dict):
            raise WorldBlueprintError("world_blueprint 必须是 JSON object")
    elif "entity_types" in value or "temporal_model" in value:
        raw = value
    elif "domain_profile" in value or "shared_world_spec" in value:
        raw = _legacy_blueprint(value)
    else:
        raise WorldBlueprintError("缺少 world_blueprint/entity_types")

    bp = _canonical_blueprint(raw)
    issues = validate_world_blueprint(bp)
    if issues:
        raise WorldBlueprintError("world_blueprint 非法:\n- " + "\n- ".join(issues))
    return bp


def validate_world_blueprint(bp: dict) -> list[str]:
    """机械校验蓝图引用闭包；返回全部问题，不调用 LLM。"""
    if not isinstance(bp, dict):
        return ["蓝图必须是 object"]
    issues: list[str] = []
    if isinstance(bp.get("version"), bool) or bp.get("version") != 1:
        issues.append("version 目前必须为 1")

    entity_types = bp.get("entity_types") or []
    if not isinstance(entity_types, list) or not entity_types:
        return issues + ["entity_types 至少声明一种实体"]
    if any(not isinstance(t, dict) for t in entity_types):
        return issues + ["entity_types 每项必须是 object"]
    if not bp.get("legacy_adapter") and len(entity_types) < 2:
        issues.append("显式 world_blueprint 至少需要 2 种 entity type，禁止退化成单类型字段表")

    type_ids = []
    for t in entity_types:
        tid = t.get("id")
        if not isinstance(tid, str) or not tid or tid != tid.strip():
            issues.append(f"entity type id 必须是无首尾空格的非空字符串:{tid!r}")
        else:
            type_ids.append(tid)
    for dup, count in Counter(type_ids).items():
        if not dup:
            issues.append("entity type id 不得为空")
        elif count > 1:
            issues.append(f"entity type id 重复:{dup}")
    type_map = {t.get("id"): t for t in entity_types
                if isinstance(t.get("id"), str) and t.get("id") and t.get("id") == t.get("id").strip()}
    primaries = [t.get("id") for t in entity_types if t.get("primary") is True]
    if any(not isinstance(t.get("primary", False), bool) for t in entity_types):
        issues.append("entity type primary 必须是 JSON boolean")
    if len(primaries) != 1:
        issues.append(f"必须且只能有一个 primary entity type，当前={primaries}")

    # 旧产线仍会按字段名查 kind/unit：跨类型同名可以存在，但核心值约束必须一致，
    # 否则同一个“状态/金额”会在全局投影里串型。
    global_fields: dict[str, tuple[str, tuple]] = {}
    fields_by_type: dict[str, set[str]] = {}
    field_specs_by_type: dict[str, dict[str, dict]] = {}
    relation_endpoints = {
        endpoint
        for relation in (bp.get("relation_types") or []) if isinstance(relation, dict)
        for endpoint in (relation.get("from_type"), relation.get("to_type"))
        if isinstance(endpoint, str)
    }
    for t in entity_types:
        tid = t.get("id")
        if not isinstance(t.get("noun"), str) or not t.get("noun") or t.get("noun") != t.get("noun").strip():
            issues.append(f"entity type {tid or '?'} 缺 noun")
        if isinstance(t.get("count"), bool) or not isinstance(t.get("count"), int) or t.get("count", 0) < 1:
            issues.append(f"entity type {tid or '?'} count 必须 >=1")
        cardinality = t.get("cardinality_policy")
        if cardinality not in (None, "exact"):
            issues.append(
                f"entity type {tid or '?'} cardinality_policy 只允许 exact；可增长类型直接省略")
        fields = t.get("fields") or []
        if not isinstance(fields, list):
            issues.append(f"entity type {tid or '?'} fields 必须是 list")
            fields = []
        elif not fields and not bp.get("legacy_adapter") and tid not in relation_endpoints:
            issues.append(
                f"entity type {tid or '?'} 零字段时必须是 relation endpoint，"
                "否则只是不可观察的孤立专名集合")
        names: list[str] = []
        for f in fields:
            if (not isinstance(f, dict) or not isinstance(f.get("name"), str)
                    or not f.get("name") or f.get("name") != f.get("name").strip()):
                issues.append(f"entity type {tid or '?'} 含无名字段")
                continue
            name = f["name"]
            names.append(name)
            kind = f.get("kind")
            if not isinstance(kind, str) or not kind or kind != kind.strip():
                issues.append(f"entity type {tid or '?'} 字段 {name} 缺 kind")
            foreign_keys = sorted(key for key in ("ref_type", "to_type", "target_type", "entity_type", "references")
                                  if key in f and f.get(key) not in (None, "", []))
            if foreign_keys:
                issues.append(
                    f"entity type {tid or '?'} 字段 {name} 含 schema 外目标键 {foreign_keys};"
                    "reference 目标只能由 relation_types 表达")
            mono = f.get("monotonic")
            if mono not in (None, "", "up", "down"):
                issues.append(f"entity type {tid or '?'} 字段 {name} monotonic 只能为 up/down")
            if mono in ("up", "down") and kind != "numeric":
                issues.append(f"entity type {tid or '?'} 字段 {name} 非 numeric 却声明 monotonic")
            rng = f.get("range")
            if rng is not None:
                numeric_bounds = (isinstance(rng, (list, tuple)) and len(rng) == 2
                                  and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng))
                if not numeric_bounds:
                    issues.append(f"entity type {tid or '?'} 字段 {name} range 必须是两个数值端点")
                elif rng[0] >= rng[1]:
                    issues.append(f"entity type {tid or '?'} 字段 {name} range 下界必须小于上界")
                elif kind != "numeric":
                    issues.append(f"entity type {tid or '?'} 字段 {name} 非 numeric 却声明 range")
            states = f.get("states")
            if states is not None and (not isinstance(states, list) or len(states) < 2
                                       or len({str(x).strip() for x in states}) != len(states)):
                issues.append(f"entity type {tid or '?'} 字段 {name} states 至少两个且不得重复")
            shape = (f.get("kind"), f.get("unit"), f.get("monotonic"),
                     tuple(f.get("range")) if isinstance(f.get("range"), (list, tuple)) else None,
                     tuple(f.get("states")) if isinstance(f.get("states"), list) else None)
            prior = global_fields.get(name)
            if prior is not None and prior[1] != shape:
                issues.append(f"跨类型同名字段约束冲突:{name} 在 {prior[0]}/{tid} 的 kind/unit/shape 不一致")
            global_fields[name] = (tid, shape)
        for dup, count in Counter(names).items():
            if count > 1:
                issues.append(f"entity type {tid or '?'} 字段重复:{dup}")
        if isinstance(tid, str):
            fields_by_type[tid] = set(names)
            field_specs_by_type[tid] = {
                field.get("name"): field for field in fields
                if isinstance(field, dict) and isinstance(field.get("name"), str)
            }

    def _objects(key: str) -> list[dict]:
        seq = bp.get(key) or []
        if not isinstance(seq, list):
            issues.append(f"{key} 必须是 array")
            return []
        bad = sum(not isinstance(x, dict) for x in seq)
        if bad:
            issues.append(f"{key} 含 {bad} 个非 object 项")
        return [x for x in seq if isinstance(x, dict)]

    relations = _objects("relation_types")
    events = _objects("event_types")
    rules = _objects("causal_rules")
    if not bp.get("legacy_adapter") and not relations:
        issues.append("显式 world_blueprint 至少需要 1 种 relation type")
    if not bp.get("legacy_adapter") and not events:
        issues.append("显式 world_blueprint 至少需要 1 种 domain event type")
    for key, seq in (("relation", relations), ("event", events), ("causal rule", rules)):
        ids = []
        for item in seq:
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id or item_id != item_id.strip():
                issues.append(f"{key} id 必须是无首尾空格的非空字符串:{item_id!r}")
            else:
                ids.append(item_id)
        for dup, count in Counter(ids).items():
            if not dup:
                issues.append(f"{key} id 不得为空")
            elif count > 1:
                issues.append(f"{key} id 重复:{dup}")

    relation_owned_fields: dict[tuple[str, str], str] = {}
    for rel in relations:
        rid, src, dst, fld = (rel.get("id"), rel.get("from_type"),
                              rel.get("to_type"), rel.get("field"))
        if not all(isinstance(x, str) and x == x.strip() for x in (src, dst, fld)):
            issues.append(f"relation {rid or '?'} from_type/to_type/field 必须是规范字符串")
        if not isinstance(src, str) or not isinstance(dst, str) or src not in type_map or dst not in type_map:
            issues.append(f"relation {rid or '?'} 类型引用不存在:{src}->{dst}")
        if isinstance(src, str) and isinstance(dst, str) and isinstance(fld, str):
            side = relation_owner_side(bp, rel)
            if side is None:
                source_owns = fld in fields_by_type.get(src, set())
                target_owns = fld in fields_by_type.get(dst, set())
                if src == dst:
                    issues.append(
                        f"relation {rid or '?'} 字段 {fld!r} 不属于自关系类型 {src}")
                else:
                    ownership = "同时属于两端" if source_owns and target_owns else "不属于任一端"
                    issues.append(
                        f"relation {rid or '?'} 字段 {fld!r} {ownership};"
                        f"必须恰好属于 from_type {src} 或 to_type {dst} 之一")
            else:
                owner_type = src if side == "from" else dst
                owner_spec = field_specs_by_type.get(owner_type, {}).get(fld) or {}
                if owner_spec.get("kind") != "reference":
                    issues.append(
                        f"relation {rid or '?'} owner 字段 {owner_type}.{fld} kind 必须是 reference，"
                        f"当前={owner_spec.get('kind')!r}")
                owner_key = (owner_type, fld)
                prior_relation = relation_owned_fields.get(owner_key)
                if prior_relation is not None:
                    issues.append(
                        f"relation {rid or '?'} 与 {prior_relation} 复用 owner 字段 "
                        f"{owner_type}.{fld};v1 每个 reference 字段只能承载一种关系")
                else:
                    relation_owned_fields[owner_key] = rid or "?"
        if not isinstance(rel.get("temporal"), bool):
            issues.append(f"relation {rid or '?'} temporal 必须是 JSON boolean")
        minimum = 0 if bp.get("legacy_adapter") else 1
        if (isinstance(rel.get("min_count"), bool) or not isinstance(rel.get("min_count"), int)
                or rel.get("min_count", -1) < minimum):
            issues.append(f"relation {rid or '?'} min_count 必须 >={minimum}")
        elif relation_owner_side(bp, rel) is not None:
            capacity = relation_capacity(bp, rel)
            if rel["min_count"] > capacity:
                issues.append(
                    f"relation {rid or '?'} min_count={rel['min_count']} 超过标量 FK 最大容量 {capacity}")

    for owner_type, specs in field_specs_by_type.items():
        for field_name, spec in specs.items():
            if spec.get("kind") == "reference" and (owner_type, field_name) not in relation_owned_fields:
                issues.append(
                    f"reference 字段 {owner_type}.{field_name} 未绑定 relation type;"
                    "v1 不允许悬空引用")

    event_ids = {e.get("id") for e in events if isinstance(e.get("id"), str) and e.get("id")}
    event_labels = [e.get("label") for e in events
                    if isinstance(e.get("label"), str) and e.get("label").strip()]
    for duplicate, count in Counter(event_labels).items():
        if count > 1:
            issues.append(f"event label 重复:{duplicate}")
    for event in events:
        eid, roles = event.get("id"), event.get("roles")
        label = event.get("label")
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            issues.append(f"event {eid or '?'} 缺少规范的人类可读 label")
        if not isinstance(roles, dict) or not roles:
            issues.append(f"event {eid or '?'} roles 至少一个")
            roles = {}
        for role, tid in roles.items():
            if (not isinstance(role, str) or not role or role != role.strip()
                    or not isinstance(tid, str) or tid != tid.strip() or tid not in type_map):
                issues.append(f"event {eid or '?'} role 引用不存在:{role}->{tid}")
        for tid, required in Counter(roles.values()).items():
            entity_type = type_map.get(tid) or {}
            count = entity_type.get("count")
            if (isinstance(count, int) and not isinstance(count, bool)
                    and count < required):
                issues.append(
                    f"event {eid or '?'} 的 {required} 个 {tid} 角色必须由互异实体承担，"
                    f"但该类型 count={count}")
        effects = event.get("effect_fields") or []
        if not isinstance(effects, list) or not effects:
            issues.append(f"event {eid or '?'} effect_fields 至少一个")
            effects = []
        for eff in effects:
            if not isinstance(eff, dict):
                issues.append(f"event {eid or '?'} effect 必须是 object")
                continue
            role, fld = eff.get("role"), eff.get("field")
            tid = roles.get(role)
            if role not in roles:
                issues.append(f"event {eid or '?'} effect role 不存在:{role}")
            elif not isinstance(tid, str) or not isinstance(fld, str) or fld not in fields_by_type.get(tid, set()):
                issues.append(f"event {eid or '?'} effect 字段 {fld!r} 不属于 role {role}({tid})")
            elif (tid, fld) in relation_owned_fields:
                issues.append(
                    f"event {eid or '?'} effect 不得写 relation-owned 字段 {tid}.{fld} "
                    f"(relation={relation_owned_fields[(tid, fld)]})")
            elif _looks_like_identity_field(fld):
                issues.append(
                    f"event {eid or '?'} effect 不得写实体身份字段 {tid}.{fld};"
                    "事件必须改变状态/数值/可变属性，不能把实体专名重复 SET 给名称字段")
        duplicate_effects = [pair for pair, count in Counter(
            (effect.get("role"), effect.get("field"))
            for effect in effects if isinstance(effect, dict)
        ).items() if count > 1]
        if duplicate_effects:
            issues.append(f"event {eid or '?'} effect_fields 重复:{duplicate_effects}")
        minimum = 0 if bp.get("legacy_adapter") else 1
        if (isinstance(event.get("min_count"), bool) or not isinstance(event.get("min_count"), int)
                or event.get("min_count", -1) < minimum):
            issues.append(f"event {eid or '?'} min_count 必须 >={minimum}")

    for rule in rules:
        rid = rule.get("id") or "?"
        trigger, effect = rule.get("trigger_event"), rule.get("effect_event")
        if (not isinstance(trigger, str) or not isinstance(effect, str)
                or trigger not in event_ids or effect not in event_ids):
            issues.append(f"causal rule {rid} 事件引用不存在")
        if (isinstance(rule.get("delay_sessions"), bool)
                or not isinstance(rule.get("delay_sessions"), int)
                or rule.get("delay_sessions", -1) < 0):
            issues.append(f"causal rule {rid} delay_sessions 必须 >=0")
        else:
            n_sessions = (bp.get("temporal_model") or {}).get("n_sessions")
            if isinstance(n_sessions, int) and not isinstance(n_sessions, bool) \
                    and rule["delay_sessions"] >= n_sessions:
                issues.append(
                    f"causal rule {rid} delay_sessions={rule['delay_sessions']} "
                    f"必须小于 n_sessions={n_sessions}")

    temporal = bp.get("temporal_model")
    if not isinstance(temporal, dict):
        issues.append("temporal_model 必须是 object")
    else:
        if not str(temporal.get("unit") or "").strip():
            issues.append("temporal_model.unit 不得为空")
        if not str(temporal.get("cadence") or "").strip():
            issues.append("temporal_model.cadence 不得为空")
        if (isinstance(temporal.get("n_sessions"), bool)
                or not isinstance(temporal.get("n_sessions"), int) or temporal.get("n_sessions", 0) < 2):
            issues.append("temporal_model.n_sessions 必须 >=2")
        if (isinstance(temporal.get("step_days"), bool)
                or not isinstance(temporal.get("step_days"), int) or temporal.get("step_days", 0) < 1):
            issues.append("temporal_model.step_days 必须 >=1")
    channels = bp.get("evidence_channels") or []
    if (not bp.get("legacy_adapter") and
            (not isinstance(channels, list) or not any(isinstance(x, str) and x.strip() for x in channels))):
        issues.append("显式 world_blueprint 至少需要 1 个 evidence channel")
    if isinstance(channels, list) and any(not isinstance(x, str) or not x.strip() for x in channels):
        issues.append("evidence_channels 每项必须是非空字符串")
    # v1 只能承诺真正执行的约束。自由文本 invariant 无法由实例编译器核验，
    # 与其把装饰性 prose 冒充契约，不如要求架构师改写成已有结构约束。
    invariants = bp.get("invariants")
    if not bp.get("legacy_adapter") and invariants not in (None, []):
        issues.append("v1 不接受不可执行的 prose invariants；请改写为字段/关系/事件/因果约束")
    return issues


def structure_signature(value: dict) -> tuple:
    """返回去除领域词汇后的结构签名，用来检查场景不是简单换皮。

    签名保留字段 kind/约束形状、关系拓扑度、事件角色/效果形状、因果图和时间制度，
    故不会因为 ``玩家`` 改名成 ``员工`` 就误判为新结构。
    """
    bp = normalize_world_blueprint(value)
    types = bp["entity_types"]
    type_map = {t["id"]: t for t in types}

    def _field_shape(field: dict) -> tuple:
        """只保留字段约束形状，不把领域字段名或状态词带进签名。"""
        rng = field.get("range")
        states = field.get("states")
        return (
            field.get("kind") or "unknown",
            bool(field.get("unit")),
            field.get("monotonic") or "",
            bool(isinstance(rng, (list, tuple)) and len(rng) == 2),
            len(states) if isinstance(states, list) else 0,
        )

    fields = {
        t["id"]: {f.get("name"): _field_shape(f) for f in (t.get("fields") or [])}
        for t in types
    }

    def _relation_field_shape(relation: dict) -> tuple[str, tuple]:
        """返回关系字段相对于有向边的持有侧与真实形状。"""
        src, dst, field = relation["from_type"], relation["to_type"], relation.get("field")
        side = relation_owner_side(bp, relation)
        if side == "from":
            return "from", fields.get(src, {}).get(field, ("unknown", False, "", False, 0))
        return "to", fields.get(dst, {}).get(field, ("unknown", False, "", False, 0))

    base = {
        t["id"]: (bool(t.get("primary")), tuple(sorted(fields[t["id"]].values())))
        for t in types
    }

    def _digest(shape: tuple) -> str:
        return hashlib.sha256(repr(shape).encode("utf-8")).hexdigest()

    # 用标签细化得到与声明顺序、type id 和 noun 无关的有向类型图指纹。
    # count / min_count / n_sessions 属于产量旋钮，不属于世界骨架，故刻意不进入签名。
    labels = {tid: _digest(shape) for tid, shape in base.items()}
    for _ in range(max(2, len(types) + 1)):
        relation_context: dict[str, list[tuple]] = {tid: [] for tid in type_map}
        for rel in bp["relation_types"]:
            src, dst = rel["from_type"], rel["to_type"]
            owner_side, fshape = _relation_field_shape(rel)
            relation_context[src].append(
                ("out", labels[dst], bool(rel.get("temporal")), owner_side, fshape))
            relation_context[dst].append(
                ("in", labels[src], bool(rel.get("temporal")), owner_side, fshape))

        event_context: dict[str, list[tuple]] = {tid: [] for tid in type_map}
        for event in bp["event_types"]:
            roles = event.get("roles") or {}
            effects_by_role: dict[str, list[tuple]] = {}
            for effect in event.get("effect_fields") or []:
                role = effect.get("role")
                tid = roles.get(role)
                effects_by_role.setdefault(role, []).append(
                    fields.get(tid, {}).get(effect.get("field"), ("unknown", False, "", False, 0)))
            role_shapes = tuple(sorted(
                (labels[tid], tuple(sorted(effects_by_role.get(role, []))))
                for role, tid in roles.items()
            ))
            event_shape = (role_shapes,)
            for tid in roles.values():
                event_context[tid].append(event_shape)

        new_labels = {
            tid: _digest((base[tid], tuple(sorted(relation_context[tid])),
                          tuple(sorted(event_context[tid]))))
            for tid in type_map
        }
        if new_labels == labels:
            break
        labels = new_labels

    type_shapes = tuple(sorted(labels.values()))
    relation_shapes = tuple(sorted(
        (labels[r["from_type"]], labels[r["to_type"]], bool(r.get("temporal")),
         *_relation_field_shape(r))
        for r in bp["relation_types"]
    ))

    event_shapes_by_id: dict[str, tuple] = {}
    for event in bp["event_types"]:
        roles = event.get("roles") or {}
        effects_by_role: dict[str, list[tuple]] = {}
        for effect in event.get("effect_fields") or []:
            role = effect.get("role")
            tid = roles.get(role)
            effects_by_role.setdefault(role, []).append(
                fields.get(tid, {}).get(effect.get("field"), ("unknown", False, "", False, 0)))
        event_shapes_by_id[event["id"]] = tuple(sorted(
            (labels[tid], tuple(sorted(effects_by_role.get(role, []))))
            for role, tid in roles.items()
        ))
    event_shapes = tuple(sorted(event_shapes_by_id.values()))
    causal_shapes = tuple(sorted(
        (event_shapes_by_id[r["trigger_event"]], event_shapes_by_id[r["effect_event"]],
         _as_int(r.get("delay_sessions"), 0))
        for r in bp["causal_rules"]
    ))
    tm = bp["temporal_model"]
    return (type_shapes, relation_shapes, event_shapes, causal_shapes,
            tm.get("unit"), tm.get("cadence"), len(bp.get("evidence_channels") or []))
