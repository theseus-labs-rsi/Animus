"""Check that a seed contract survives world instantiation and subsequent passes.

This checks the executable vocabulary already supported by WorldState: typed
entities, canonical field timelines, relations, event effects and causal edges.
It does not claim to execute financial formulae, source authority adjudication or
separate publication/observation clocks. Those need separate implementations.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

from pipeline.world_blueprint import WorldBlueprintError, relation_owner_side
from pipeline.world_state import SET, UPDATE, INVALID, INSUFFICIENT, WorldState, _magnitude, _norm


class SeedWorldError(WorldBlueprintError):
    """A generated or reloaded world no longer implements its seed contract."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("seed world contract failed:\n- " + "\n- ".join(report["issues"]))


def seed_protected_fields(wp: dict) -> set[tuple[str, str]]:
    """Fields whose values may not be rewritten to manufacture benchmark trends."""
    requirements = (wp.get("seed_contract") or {}).get("blueprint_requirements") or {}
    protected = {(item["id"], field["name"])
                 for item in requirements.get("entity_types", [])
                 for field in item.get("fields", [])}
    blueprint = wp.get("world_blueprint") or {}
    for relation in requirements.get("relation_types", []):
        declaration = next((r for r in blueprint.get("relation_types", [])
                            if r.get("id") == relation.get("id")), relation)
        side = relation_owner_side(blueprint, declaration)
        owner = declaration.get("from_type") if side == "from" else declaration.get("to_type")
        protected.add((owner, declaration.get("field")))
    for event in requirements.get("event_types", []):
        declaration = next((e for e in blueprint.get("event_types", [])
                            if e.get("id") == event.get("id")), event)
        for effect in declaration.get("effect_fields", []):
            protected.add(((declaration.get("roles") or {}).get(effect.get("role")),
                           effect.get("field")))
    return {(entity_type, field) for entity_type, field in protected if entity_type and field}


def seed_business_context(wp: dict) -> dict:
    """Author semantics and minimal source descriptors, excluding audit/grading data."""
    contract = wp.get("seed_contract") or {}
    if contract.get("schema_version") == 2:
        from pipeline.seed_pack import generation_context
        context = generation_context(contract)
        # Minimal descriptors make attached source IDs interpretable: a normative
        # rule and a factual record have different scope. The projection already
        # removed file paths, hashes, private audit and evaluator-only sources.
        return {key: deepcopy(context[key]) for key in (
            "schema_version", "seed_id", "family", "title", "description", "task",
            "sources", "mechanisms", "blueprint_requirements", "generation_contract", "context_policy") if key in context}
    task = contract.get("task") or {}
    return {
        "task": {key: task[key] for key in ("objective", "instructions") if key in task},
        "mechanisms": [
            {key: item[key] for key in ("id", "description", "required_entity_types",
                                       "required_relation_types", "required_event_types",
                                       "required_causal_rules") if key in item}
            for item in contract.get("mechanisms", [])
        ],
    }


def seed_world_prompt(wp: dict) -> str:
    """Pass the frozen task and mechanism context, without source/rubric records."""
    contract = wp.get("seed_contract")
    if not contract:
        return ""
    from pipeline.seed_pack import core_requirements
    requirements = (core_requirements(contract) if contract.get("schema_version") == 2
                    else contract.get("blueprint_requirements") or {})
    context = seed_business_context(wp)
    return ("\n【真实任务种子约束，必须落实到可执行世界】\n"
            "以下结构是不可删除的最低要求。必要实体字段必须有实际时间线；关系必须写入"
            "canonical 外键；事件必须改变声明的字段，并提供实际参与者。因果规则必须有"
            "caused_by 和精确时间差见证。不得仅在说明或 metadata 声称已覆盖。"
            "causal_rules.shared_roles 列出的角色必须在父子事件中指向同一个实体；"
            "event_types.relation_bindings 指定的 from_role/to_role 必须在事件发生的 session"
            "满足指定 relation 的当前 canonical 外键，历史已过期关系或未来关系不能替代。"
            "种子字段按业务机制变化，不为出题刻意制造峰谷或 null；不要虚构尚未支持的"
            "公式执行或多时间语义。\n"
            + json.dumps(requirements, ensure_ascii=False)
            + "\n【已冻结的任务与业务机制背景】\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n依据任务和已有字段事实安排必要的业务关系与事件；不要把最低数量当作上限，"
            "也不要为题型、难度或凑覆盖数量制造不必要的变化。")


def seed_entity_prompt(wp: dict, type_id: str) -> str:
    """Supply intrinsic authors with versioned, generator-safe seed semantics.

    V1 retains its short objective/mechanism context. V2 also keeps complete
    field meanings and cross-type definitions without raw sources or audit.
    """
    contract = wp.get("seed_contract")
    if not contract:
        return ""

    if contract.get("schema_version") == 2:
        context = seed_business_context(wp)
        # Definitions can refer across entity types (units, reporting periods,
        # transition prerequisites). Preserve the full semantic declaration;
        # the selected entity and original author ownership still bound output.
        return ("\n【种子业务定义：当前实体类型 " + type_id + "】\n"
                + json.dumps(context, ensure_ascii=False)
                + "\n只生成当前类型允许的内在字段。按字段所属对象、定义、单位、期间与状态含义生成合成值；"
                  "保留业务规则的适用条件和来源规定/合成安排区别。语义定义由你结合任务解释，"
                  "未实现的公式、转换条件或因果间隔不能假称已由程序执行。"
                  "关系与事件由后续结构作者生成；本阶段省略关系外键及事件驱动字段，"
                  "没有允许的内在字段时返回 fields={}。后台业务说明尚未成为公开材料。")

    def short(value, limit=320):
        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    mechanisms = [short(item.get("description")) for item in contract.get("mechanisms", [])
                  if type_id in item.get("required_entity_types", [])]
    context = {"任务目标": short((contract.get("task") or {}).get("objective")),
               "当前类型": type_id, "相关业务机制背景": mechanisms}
    return ("\n【种子实体背景：本阶段只生成当前类型的内在字段】\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n按字段清单生成符合业务含义的合成值，遵守类型、单位、状态和值域。"
            "字段可以自然演化或保持稳定，不为题型制造峰谷或强塞 null。"
            "关系和事件将在后续 structure 阶段单独生成；本阶段省略关系外键和事件驱动字段，"
            "不要生成关系/事件实例。若没有允许的内在字段，返回 fields={}。")


def causal_roles_match(rule: dict, parent: dict, child: dict) -> bool:
    """An explicit same-business-object binding; absent bindings are compatible."""
    parent_roles = parent.get("participants") or {}
    child_roles = child.get("participants") or {}
    return all(bool(parent_roles.get(role)) and parent_roles.get(role) == child_roles.get(role)
               for role in rule.get("shared_roles", []))


def seed_world_report(wp: dict, ws: WorldState, *, require_complete: bool = True) -> dict:
    """Check a full world or the facts already committed to an Agent draft.

    Partial validation defers minimum global coverage, never the validity of an
    existing entity, field, relation, effect, time or declared causal edge.
    Exact entity cardinalities remain upper bounds throughout drafting.
    """
    if type(require_complete) is not bool:
        raise ValueError("require_complete must be a boolean")
    contract = wp.get("seed_contract")
    if contract is None:
        return {"passed": True, "applied": False, "issues": [], "mechanism_coverage": []}
    from pipeline.seed_pack import core_requirements, validate_seed_blueprint

    issues: list[str] = []
    # Validate both the intended blueprint and the one used by a persisted world.
    for label, candidate in (("whitepaper", wp), ("world", ws.world_blueprint)):
        report = validate_seed_blueprint(candidate, contract)
        issues.extend(f"seed {label}: {issue}" for issue in report.get("issues", []))
    requirements = core_requirements(contract)
    blueprint = ws.world_blueprint or {}
    relation_declarations = {item["id"]: item for item in blueprint.get("relation_types", [])}
    event_declarations = {item["id"]: item for item in blueprint.get("event_types", [])}
    causal_declarations = {item["id"]: item for item in blueprint.get("causal_rules", [])}
    minimum_sessions = (requirements.get("temporal_model") or {}).get("min_sessions", 0)
    if ws.n_sessions < minimum_sessions:
        issues.append(f"seed temporal_model: n_sessions={ws.n_sessions} < {minimum_sessions}")

    actual_entities: dict[str, list[str]] = {}
    declared_entity_ids = {item["id"] for item in blueprint.get("entity_types", [])}
    for name in ws.entities:
        if not require_complete and ws.entity_types.get(name) not in declared_entity_ids:
            issues.append(f"seed entity {name}: undeclared actual type {ws.entity_types.get(name)}")
        actual_entities.setdefault(ws.entity_types.get(name), []).append(name)
    driven_fields = set()
    for declaration in relation_declarations.values():
        owner = declaration.get("from_type") if relation_owner_side(blueprint, declaration) == "from" \
            else declaration.get("to_type")
        driven_fields.add((owner, declaration.get("field")))
    for declaration in event_declarations.values():
        driven_fields.update(((declaration.get("roles") or {}).get(effect.get("role")), effect.get("field"))
                             for effect in declaration.get("effect_fields", []))
    for requirement in requirements.get("entity_types", []):
        type_id = requirement["id"]
        names = actual_entities.get(type_id, [])
        minimum = requirement.get("count", 1)
        if require_complete and len(names) < minimum:
            issues.append(f"seed entity {type_id}: actual count {len(names)} < {minimum}")
        if (requirement.get("cardinality_policy") == "exact" and len(names) != minimum
                and (require_complete or len(names) > minimum)):
            issues.append(f"seed entity {type_id}: actual count {len(names)} != exact {minimum}")
        for field in requirement.get("fields", []):
            field_name = field["name"]
            # A relation or event may involve a subset of entities. Require a
            # real timeline for each required field, not an empty schema entry.
            if require_complete and not any((tl := ws.timeline(name, field_name)) and
                       any(op.op in (SET, UPDATE) and op.value not in
                           (None, "", INSUFFICIENT, INVALID) for op in tl.ops)
                       for name in names):
                issues.append(f"seed field {type_id}.{field_name}: no canonical values")
            for name in names:
                timeline = ws.timeline(name, field_name)
                if not timeline or not timeline.ops:
                    if (type_id, field_name) not in driven_fields:
                        issues.append(f"seed field {name}.{field_name}: missing intrinsic timeline")
                    continue
                for op in timeline.ops:
                    if (type(op.session) is not int or not 0 <= op.session < ws.n_sessions
                            or op.date != ws.date_of_session(op.session)):
                        issues.append(f"seed field {name}.{field_name}: invalid canonical time")
                    if op.op not in (SET, UPDATE):
                        continue
                    numeric = _magnitude(op.value)
                    if (op.value in (None, "", INSUFFICIENT, INVALID)
                            or field.get("kind") == "numeric" and numeric is None
                            or field.get("states") and _norm(op.value) not in {_norm(v) for v in field["states"]}
                            or field.get("range") and (numeric is None
                                                       or not field["range"][0] <= numeric <= field["range"][1])):
                        issues.append(f"seed field {name}.{field_name}: value violates required field schema")

    def witness(entity, field, session, value) -> bool:
        if not isinstance(session, int) or isinstance(session, bool) or not 0 <= session < ws.n_sessions:
            return False
        timeline = ws.timeline(entity, field)
        if timeline is None or value is None:
            return False
        at_session = [op for op in timeline.ops if op.session == session]
        # Exactly one canonical operation must represent the effect; checking
        # latest value alone would accept metadata whose effects never happened.
        if len(at_session) != 1:
            return False
        op = at_session[0]
        return (op.op in (SET, UPDATE) and str(op.value) == str(value)
                and op.date == ws.date_of_session(session)
                and str(timeline.value_at_session(session - 1)) != str(value))

    # Validate declared structural rows, including extensions that legitimately
    # touch a seed-owned field; an extension must not become an unverified write.
    required_relation_ids = set(relation_declarations)
    required_event_ids = set(event_declarations)
    # Requirements may reference a causal rule without repeating its event
    # declarations; still validate the effects of both sides of the rule.
    for rule in requirements.get("causal_rules", []):
        declaration = causal_declarations.get(rule["id"], {})
        required_event_ids.update([declaration.get("trigger_event"), declaration.get("effect_event")])
    relation_ids = Counter(item.get("id") for item in ws.relations)
    relation_edges = Counter((item.get("type"), item.get("from"), item.get("to"), item.get("session"))
                             for item in ws.relations)
    valid_relations: dict[str, list[str]] = {}
    structural_fields: set[tuple[str, str]] = set(driven_fields) if not require_complete else set()
    structural_witnesses: set[tuple[str, str, int, str]] = set()
    for relation in ws.relations:
        type_id = relation.get("type")
        if type_id not in required_relation_ids:
            if not require_complete:
                issues.append(f"seed relation {relation.get('id')}: undeclared actual type {type_id}")
            continue
        declaration = relation_declarations.get(type_id, {})
        source, target, session = relation.get("from"), relation.get("to"), relation.get("session")
        owner_side = relation_owner_side(blueprint, declaration)
        owner, referred = (source, target) if owner_side == "from" else (target, source)
        structural_fields.add((declaration.get("from_type") if owner_side == "from"
                               else declaration.get("to_type"), declaration.get("field")))
        valid = (bool(relation.get("id")) and relation_ids[relation.get("id")] == 1
                 and relation_edges[(type_id, source, target, session)] == 1
                 and source in ws.entities and target in ws.entities
                 and ws.entity_types.get(source) == declaration.get("from_type")
                 and ws.entity_types.get(target) == declaration.get("to_type")
                 and owner_side is not None
                 and (declaration.get("temporal") or session == 0)
                 and witness(owner, declaration.get("field"), session, referred))
        if valid:
            valid_relations.setdefault(type_id, []).append(relation["id"])
            structural_witnesses.add((owner, declaration.get("field"), session, str(referred)))
        else:
            issues.append(f"seed relation {relation.get('id')}: missing/invalid canonical FK witness")
    for requirement in requirements.get("relation_types", []):
        count = len(valid_relations.get(requirement["id"], []))
        minimum = requirement.get("min_count", 1)
        if require_complete and count < minimum:
            issues.append(f"seed relation {requirement['id']}: valid count {count} < {minimum}")

    event_ids = Counter(item.get("id") for item in ws.events)
    event_keys = Counter((item.get("type"), item.get("session"),
                          tuple(sorted((item.get("participants") or {}).items()))) for item in ws.events)
    valid_events: dict[str, list[dict]] = {}
    for event in ws.events:
        type_id = event.get("type")
        if type_id not in required_event_ids:
            if not require_complete:
                issues.append(f"seed event {event.get('id')}: undeclared actual type {type_id}")
            continue
        declaration = event_declarations.get(type_id, {})
        participants = event.get("participants") or {}
        roles = declaration.get("roles") or {}
        structural_fields.update((roles.get(effect.get("role")), effect.get("field"))
                                 for effect in declaration.get("effect_fields", []))
        valid = (bool(event.get("id")) and event_ids[event.get("id")] == 1
                 and event_keys[(type_id, event.get("session"), tuple(sorted(participants.items())))] == 1
                 and set(participants) == set(roles)
                 and len(set(participants.values())) == len(participants)
                 and all(name in ws.entities and ws.entity_types.get(name) == roles[role]
                         for role, name in participants.items()))
        expected_effects = Counter((participants.get(effect.get("role")), effect.get("field"))
                                   for effect in declaration.get("effect_fields", []))
        effects = event.get("effects") or []
        actual_effects = Counter((effect.get("entity"), effect.get("field")) for effect in effects)
        valid = (valid and bool(effects) and actual_effects == expected_effects
                 and all(witness(effect.get("entity"), effect.get("field"), event.get("session"),
                                 effect.get("set", effect.get("value"))) for effect in effects))
        for binding in declaration.get("relation_bindings", []):
            relation_id = binding.get("relation")
            relation = relation_declarations.get(relation_id, {})
            source, target = (participants.get(binding.get("from_role")),
                              participants.get(binding.get("to_role")))
            side = relation_owner_side(blueprint, relation)
            owner, referred = (source, target) if side == "from" else (target, source)
            timeline = ws.timeline(owner, relation.get("field"))
            session = event.get("session")
            binding_valid = (side is not None and source in ws.entities and target in ws.entities
                             and type(session) is int and 0 <= session < ws.n_sessions
                             and timeline is not None and timeline.value_at_session(session) == referred
                             and any(row.get("id") in valid_relations.get(relation_id, [])
                                     and row.get("from") == source and row.get("to") == target
                                     and row["session"] <= session for row in ws.relations))
            if not binding_valid:
                valid = False
                issues.append(f"seed event {event.get('id')}: relation binding {relation_id} "
                              f"does not hold for {source}->{target}@{session}")
        if valid:
            valid_events.setdefault(type_id, []).append(event)
            structural_witnesses.update((effect.get("entity"), effect.get("field"), event["session"],
                                         str(effect.get("set", effect.get("value")))) for effect in effects)
        else:
            issues.append(f"seed event {event.get('id')}: participants/effects have no complete canonical witness")
    for requirement in requirements.get("event_types", []):
        count = len(valid_events.get(requirement["id"], []))
        minimum = requirement.get("min_count", 1)
        if require_complete and count < minimum:
            issues.append(f"seed event {requirement['id']}: valid count {count} < {minimum}")

    # Checking events -> ops alone misses a later overwrite inserted by an
    # enrichment pass. Every non-baseline op on these fields must also point
    # back to a structural witness.
    for entity, fields in ws.entities.items():
        for field, timeline in fields.items():
            if (ws.entity_types.get(entity), field) not in structural_fields:
                continue
            for index, op in enumerate(timeline._sorted()):
                baseline = index == 0 and op.session == 0 and op.op == SET
                if not baseline and (entity, field, op.session, str(op.value)) not in structural_witnesses:
                    issues.append(f"seed canonical {entity}.{field}@{op.session}: no relation/event witness")

    causal_witnesses: dict[str, list[dict]] = {}
    required_causal_ids = {item["id"] for item in requirements.get("causal_rules", [])}
    if not require_complete:
        # Full mode has the historical rule-witness coverage check below.
        # Prefix mode must validate asserted edges directly: postponing missing
        # coverage must not make an invalid parent/ref/delay appear acceptable.
        valid_event_ids = {event["id"] for rows in valid_events.values() for event in rows}
        event_by_id = {event.get("id"): event for event in ws.events}
        for child in ws.events:
            parent_ref = child.get("caused_by")
            if not parent_ref:
                continue
            parent = event_by_id.get(parent_ref) if isinstance(parent_ref, str) else None
            if not (parent and parent.get("id") in valid_event_ids and child.get("id") in valid_event_ids
                    and parent.get("id") != child.get("id")
                    and any(parent.get("type") == declaration.get("trigger_event")
                            and child.get("type") == declaration.get("effect_event")
                            and child["session"] - parent["session"] == declaration.get("delay_sessions")
                            and causal_roles_match(declaration, parent, child)
                            for declaration in causal_declarations.values())):
                issues.append(f"seed event {child.get('id')}: invalid asserted caused_by edge {parent_ref}")
    for rule_id, declaration in causal_declarations.items():
        if rule_id not in required_causal_ids and not declaration.get("shared_roles"):
            continue
        for parent in valid_events.get(declaration.get("trigger_event"), []):
            for child in valid_events.get(declaration.get("effect_event"), []):
                if (child.get("caused_by") == parent.get("id") and child.get("id") != parent.get("id")
                        and child["session"] - parent["session"] == declaration.get("delay_sessions")):
                    if not causal_roles_match(declaration, parent, child):
                        issues.append(f"seed causal rule {rule_id}: shared_roles mismatch "
                                      f"{parent['id']}->{child['id']} for {declaration['shared_roles']}")
                        continue
                    causal_witnesses.setdefault(rule_id, []).append(
                        {"trigger": parent["id"], "effect": child["id"]})
        if require_complete and not causal_witnesses.get(rule_id):
            issues.append(f"seed causal rule {rule_id}: no executable caused_by witness")

    coverage = []
    for mechanism in contract.get("mechanisms", []):
        coverage.append({
            "mechanism_id": mechanism.get("id"),
            "entities": {key: actual_entities.get(key, []) for key in mechanism.get("required_entity_types", [])},
            "relations": {key: valid_relations.get(key, []) for key in mechanism.get("required_relation_types", [])},
            "events": {key: [event["id"] for event in valid_events.get(key, [])]
                       for key in mechanism.get("required_event_types", [])},
            "causal_rules": {key: causal_witnesses.get(key, []) for key in mechanism.get("required_causal_rules", [])},
        })
    return {"passed": not issues, "applied": True, "seed_id": contract.get("seed_id"),
            **({"require_complete": False} if not require_complete else {}),
            "seed_digest": contract.get("digest"), "issues": issues,
            "mechanism_coverage": coverage,
            "validation_scope": "typed structure, canonical effects and causal edges; not formula execution or bitemporal semantics"}


def seed_world_issues(wp: dict, ws: WorldState, *, require_complete: bool = True) -> list[str]:
    return seed_world_report(wp, ws, require_complete=require_complete)["issues"]


def validate_seed_world(wp: dict, ws: WorldState) -> dict:
    report = seed_world_report(wp, ws)
    if not report["passed"]:
        raise SeedWorldError(report)
    return report
