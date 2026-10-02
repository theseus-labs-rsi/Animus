"""Execute one business layout: authors fill values, Python owns structure and dispatch.

Every submitted table goes through the original world compiler. The layout is
the only source of identities, relations, event roles and time. There is no
second planner and no model-authored slot mapping.
"""
from copy import deepcopy
from pathlib import Path
import json
import threading

import config
from pipeline.instance_plan import index, progress, fulfillment
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint, relation_owner_side

VERSION = "business-value-executor/v1"
VALUE_TRANSACTION_VERSION = "unit-value-transaction/v1"
DEFAULT_VALUE_REPAIR_ATTEMPTS = 6
SYSTEM = """你为已经安排好的业务单元创作具体内容。程序已固定对象、关系、事件角色、时间和因果；按输入中的槽位填写名称、内在字段、窗口初态及事件造成的新值。
填写 objects 和 event_values 两个对象。objects 的键为本批对象槽位，每项为 {"name":"唯一自然专名","fields":{"内在字段":{"type":"stable","value":"内容"}或{"type":"evolving","trajectory":[{"session":0,"value":"内容"}]}},"initial_values":{"允许初态的事件字段":"起始值"}}。event_values 的键为本批事件槽位，每项按角色填写字段值，例如 {"report":{"capital":"业务值"}}。收到审阅反馈时，增加 issue_responses 列表，逐项说明本批实际修改及理由，每项 {"issue_id":"原问题编号","disposition":"addressed/disputed/deferred","response":"具体事实与理由"}。
对象的关系字段由程序从关系表生成，事件字段由程序从 event_values 生成。fields 只填写 intrinsic_fields。initial_values 仅采用允许的 initial_fields；某字段在第0期已有事件写入时，其初态已由该事件承担。未知字段可保持缺失。事件必须产生真实变化，取值遵守字段范围、状态先后和实际业务；前后读数来自真实业务，不预选题目答案。当前依赖世界是已完成事实，引用其中的真实身份。
内在字段按 fact_time_contract 选择 stable 或 evolving；stable 从窗口开始即成立，后期首次成立的事实写对应 session 的轨迹，不用公开或补录说明代替真值时点。
response_schema 给出本批可填写的完整 JSON 结构；properties 为空的对象填写 {}。initial_values 和 event_values 的每个字段只填数字或字符串，例如数值 55，直接写 55；字段的 kind、type、unit 属于程序已有的声明，不复制到字段值中。fields 内采用 stable/value 或 evolving/trajectory 的原事实格式。
结合 business、业务目的、观测安排和原审查反馈创作；数值按各期观测形成有业务依据的历史，文档主张保留独立来源与真实未知。需要修订时按具体错误修正本批内容。对同一对象字段，一次性核对本批全部事件的时间顺序：每个 effect 都必须相对前值发生变化；单向状态按声明顺序严格推进。允许填写的 initial_values 是可选初态，不能把第一次事件的新值同时写成其初态。若固定事件/字段/时间使这些要求无法同时成立，返回 {"decision":"unresolved","reason":"具体冲突及相关槽位"}；程序会把诊断交回原世界设计者修订结构。
回复严格 JSON，objects/event_values 的键采用 task 的原槽位。具体值中引用已知人名或对象时采用依赖世界的真实名称；本批对象名称保持自然且互异。结构由程序装配。"""

# Compile the author interface from the same ownership projection used to
# assemble facts. Original world compilation owns value and business rules.
SCALAR = {"type": ["string", "number"]}
FACT_VALUE_SCHEMA = {"oneOf":[
    {"type":"object", "required":["type", "value"], "properties":{"type":{"const":"stable"}, "value":SCALAR}},
    {"type":"object", "required":["type", "trajectory"], "properties":{"type":{"const":"evolving"},
        "trajectory":{"type":"array", "items":{"type":"object", "required":["session", "value"],
            "properties":{"session":{"type":"integer"}, "value":SCALAR}}}}}]}


def author_contract(author_task):
    """The compiled layout supplies every response key, including empty maps."""
    def record(properties, required=None):
        return {"type":"object", "properties":properties,
            "required":list(properties) if required is None else required,
            "additionalProperties":False}
    objects = {}
    for obj in author_task["objects"]:
        objects[obj["slot"]] = record({
            "name":({"type":"string","const":author_task["known_identities"][obj["slot"]]}
                if obj["slot"] in author_task["known_identities"] else {"type":"string"}),
            "fields":record({f["name"]:deepcopy(FACT_VALUE_SCHEMA) for f in obj["intrinsic_fields"]}),
            "initial_values":record({f["name"]:deepcopy(SCALAR) for f in obj["initial_fields"]}, [])},
            ["name", "fields"])
    events = {event["slot"]:record({role:record({f["name"]:deepcopy(SCALAR) for f in fields})
        for role, fields in event["value_fields"].items()}) for event in author_task["events"]}
    return record({"objects":record(objects), "event_values":record(events),
        "issue_responses":{"type":"array", "items":{"type":"object"}}}, ["objects", "event_values"])


def response_contract(author_task):
    return {"oneOf":[author_contract(author_task), {
        "type":"object", "required":["decision", "reason"],
        "properties":{"decision":{"const":"unresolved"}, "reason":{"type":"string", "minLength":1}},
        "additionalProperties":False}]}


def _record(value, contract, path):
    """Explain existing key-set errors using the exact supplied contract."""
    expected = list(contract["properties"])
    missing = [key for key in contract["required"] if not isinstance(value, dict) or key not in value]
    unexpected = sorted(set(value) - set(expected)) if isinstance(value, dict) else []
    if not isinstance(value, dict) or missing or unexpected:
        detail = {"path":path, "expected_keys":expected, "missing_keys":missing,
            "unexpected_keys":unexpected, "received_type":type(value).__name__}
        raise WorldBlueprintError("Author response keys differ from the assigned contract: "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True))
    return value


def _path(parent, key):
    return parent + "/" + key.replace("~", "~0").replace("/", "~1")


def assembled_facts(wp, units, complete=False, identities=None):
    """Stage contiguous relation histories; the final commit includes all facts.

    A business unit can own both A@1 and A@4 while intervening B@2 belongs
    to a later unit. Store its complete facts, but expose A@4 to the original
    incremental compiler once all earlier declared transitions are present.
    """
    from pipeline.world_agent import _combine
    table = _combine(units)
    authored_entities = {row["name"] for row in table["entities"]}
    if identities and not complete:
        present={e["name"] for e in table["entities"]}
        table["entities"].extend({"name":identities[o["slot"]],"type":o["type"],"fields":{}}
            for unit in wp["business_instance_plan"]["units"] for o in unit["objects"] if identities[o["slot"]] not in present)
        present_relations={r["id"] for r in table["relations"]}
        table["relations"].extend({"id":r["slot"],"type":r["type"],"from":identities[r["from"]],
            "to":identities[r["to"]],"session":r["session"]} for u in wp["business_instance_plan"]["units"]
            for r in u["relations"] if r["slot"] not in present_relations)
    if complete:
        return table
    bp = wp["world_blueprint"]
    specs = {r["id"]:r for r in bp["relation_types"]}
    chains = {}
    for unit in wp["business_instance_plan"]["units"]:
        for row in unit["relations"]:
            spec = specs[row["type"]]
            if spec.get("temporal"):
                owner = row[relation_owner_side(bp, spec)]
                chains.setdefault((owner, spec["field"]), []).append(row)
    available = {r["id"] for r in table["relations"]}
    identities = {obj["slot"]:obj["slot"] for unit in wp["business_instance_plan"]["units"] for obj in unit["objects"]}
    for unit in units:
        identities.update(unit.get("instance_mapping", {}))
    staged, active, barriers = set(), set(), {}
    for chain in chains.values():
        staged.update(r["slot"] for r in chain)
        for row in sorted(chain, key=lambda r:(r["session"], r["slot"])):
            if row["slot"] not in available:
                owner = row[relation_owner_side(bp, specs[row["type"]])]
                actual = identities[owner]
                barriers[actual] = min(barriers.get(actual, row["session"]), row["session"])
                break
            active.add(row["slot"])
    table["relations"] = [r for r in table["relations"] if r["id"] not in staged or r["id"] in active]
    planned_events = {event["slot"]: event for unit in wp["business_instance_plan"]["units"] for event in unit["events"]}
    deferred = {event["id"] for event in table["events"] if event["id"] in planned_events
        and any(owner in event["participants"].values() and event["session"] >= session
            for owner, session in barriers.items())}
    # A later write cannot be judged against a placeholder or an incomplete
    # history. Stage it until its value owner's initial state and every earlier
    # planned writer are present; then use the original transition compiler.
    event_types = {e['id']: e for e in bp.get('event_types', [])}
    writes, predecessors = {}, {}
    for slot, event in planned_events.items():
        for effect in event_types.get(event.get('type'), {}).get('effect_fields', []):
            owner = event['participants'][effect['role']]
            writes.setdefault((owner, effect['field']), []).append(event)
    for chain in writes.values():
        earlier = set()
        for event in sorted(chain, key=lambda row:(row['session'], row['slot'])):
            predecessors.setdefault(event['slot'], set()).update(earlier)
            earlier.add(event['slot'])
    for event in table['events']:
        if event['id'] in planned_events and any(effect['entity'] not in authored_entities for effect in event.get('effects', [])):
            deferred.add(event['id'])
    available_events = {e['id'] for e in table['events']}
    while True:
        descendants = {event["id"] for event in table["events"] if event["id"] in planned_events
            and (event.get("caused_by") in deferred
                or predecessors.get(event['id'], set()) - (available_events - deferred))}
        if descendants <= deferred:
            break
        deferred.update(descendants)
    table["events"] = [event for event in table["events"] if event["id"] not in deferred]
    return table


def declared_context(wp, units, identities, blueprint, existing):
    """Expose shared names and declared links while their values are being authored.

    This view supports references between writers. It is never a world-quality
    certificate. Only the original complete compiler certifies the transaction.
    """
    from pipeline.world_state import assemble_world
    from pipeline.world_joint_plan import apply_initial_states
    table = assembled_facts(wp, units, identities=identities)
    table["entities"], _ = apply_initial_states(table, table, blueprint,
        existing.entities if existing is not None else ())
    table.pop("initial_states")
    world, _ = assemble_world(table, blueprint=blueprint, existing=existing,
        require_complete=False, include_shape_diagnostics=False)
    return world


def task(wp, unit, known):
    """Project field ownership from the original schema into one author request."""
    bp = wp["world_blueprint"]
    types = {t["id"]: t for t in bp["entity_types"]}
    etypes = {t["id"]: t for t in bp["event_types"]}
    from pipeline.world_agent import _context
    routes = _context(wp, bp)["field_write_routes"]
    # Initial states and event effects share one global session-0 namespace.
    # A different unit may write an object owned by this author, especially
    # with shared identities, so project permissions from the entire plan.
    zero_writes = {(e["participants"][effect["role"]], effect["field"])
        for planned_unit in wp["business_instance_plan"]["units"]
        for e in planned_unit["events"] if e["session"] == 0
        for effect in etypes[e["type"]]["effect_fields"]}
    objects = []
    for obj in unit["objects"]:
        owned = routes[obj["type"]]
        intrinsic = set(owned["intrinsic_fields"])
        initial = {name for name, paths in owned["structure_fields"].items()
            if all(p["via"] == "event" for p in paths) and (obj["slot"], name) not in zero_writes}
        objects.append({**deepcopy(obj),
            "intrinsic_fields": [deepcopy(f) for f in types[obj["type"]]["fields"] if f["name"] in intrinsic],
            "initial_fields": [deepcopy(f) for f in types[obj["type"]]["fields"] if f["name"] in initial]})
    events = []
    for event in unit["events"]:
        spec = etypes[event["type"]]
        values = {}
        for effect in spec["effect_fields"]:
            field = next(f for f in types[spec["roles"][effect["role"]]]["fields"] if f["name"] == effect["field"])
            values.setdefault(effect["role"], []).append(deepcopy(field))
        events.append({**deepcopy(event), "value_fields": values})
    return {"unit_id": unit["unit_id"], "business_purpose": unit["business_purpose"],
        "objects": objects, "relations": deepcopy(unit["relations"]), "events": events,
        "observations": deepcopy(unit["observations"]), "obligations": deepcopy(unit["obligations"]),
        "publications": deepcopy(unit["publications"]), "known_identities": deepcopy(known)}


def materialize(author_task, raw):
    """Lower values into the original fact tables; identities are allocated once."""
    contract = author_contract(author_task)
    _record(raw, contract, "")
    responses = raw.get("issue_responses", [])
    if (not isinstance(responses, list) or any(not isinstance(item, dict)
            or not isinstance(item.get("issue_id"), str) or not item["issue_id"].strip()
            or item.get("disposition") not in ("addressed", "disputed", "deferred")
            or not isinstance(item.get("response"), str) or not item["response"].strip() for item in responses)):
        raise WorldBlueprintError("Review feedback responses require an issue_id, disposition and factual reason")
    objects = _record(raw["objects"], contract["properties"]["objects"], "/objects")
    values = _record(raw["event_values"], contract["properties"]["event_values"], "/event_values")
    mapping = deepcopy(author_task["known_identities"])
    facts = {key: [] for key in ("entities", "relations", "events", "initial_states")}
    for obj in author_task["objects"]:
        shape = contract["properties"]["objects"]["properties"][obj["slot"]]
        path = _path("/objects", obj["slot"])
        item = _record(objects[obj["slot"]], shape, path)
        name, fields, initial = item.get("name"), item.get("fields"), item.get("initial_values", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(fields, dict) or not isinstance(initial, dict):
            raise WorldBlueprintError("Object name and field dictionaries are required")
        _record(fields, shape["properties"]["fields"], path + "/fields")
        _record(initial, shape["properties"]["initial_values"], path + "/initial_values")
        fixed = mapping.get(obj["slot"])
        if fixed is not None and name != fixed:
            raise WorldBlueprintError("Object must retain its allocated identity: " + obj["slot"])
        if name in mapping.values() and fixed != name:
            raise WorldBlueprintError("New object name collides with an allocated identity: " + name)
        mapping[obj["slot"]] = name
        facts["entities"].append({"name": name, "type": obj["type"], "fields": deepcopy(fields)})
        facts["initial_states"].extend({"entity": name, "field": field, "session": 0, "value": deepcopy(value)}
            for field, value in initial.items())
    for relation in author_task["relations"]:
        mapping[relation["slot"]] = relation["slot"]
        facts["relations"].append({"id": relation["slot"], "type": relation["type"],
            "from": mapping[relation["from"]], "to": mapping[relation["to"]], "session": relation["session"]})
    for event in author_task["events"]:
        shape = contract["properties"]["event_values"]["properties"][event["slot"]]
        path = _path("/event_values", event["slot"])
        supplied = _record(values[event["slot"]], shape, path)
        effects = []
        for role, fields in event["value_fields"].items():
            row = _record(supplied[role], shape["properties"][role], _path(path, role))
            effects.extend({"entity": mapping[event["participants"][role]], "field": f["name"],
                "set": deepcopy(row[f["name"]])} for f in fields)
        mapping[event["slot"]] = event["slot"]
        fact = {"id": event["slot"], "type": event["type"], "session": event["session"],
            "participants": {role: mapping[slot] for role, slot in event["participants"].items()}, "effects": effects}
        if event.get("caused_by"):
            fact["caused_by"] = event["caused_by"]
        facts["events"].append(fact)
    own = {o["slot"] for o in author_task["objects"]}
    new_mapping = {slot: actual for slot, actual in mapping.items() if slot not in author_task["known_identities"] or slot in own}
    return facts, new_mapping


def authored_value_issues(facts, blueprint, existing=None, entity_types=None):
    """Use original value rules before author facts enter the shared context."""
    from pipeline.world_joint_plan import apply_initial_states
    from pipeline.world_state import field_value_issue
    issues = []
    try:
        apply_initial_states(facts, facts, blueprint,
            existing.entities if existing is not None else ())
    except ValueError as error:
        issues.append(str(error))
    declarations = {t['id']:{f['name']:f for f in t['fields']} for t in blueprint['entity_types']}
    types = {**(entity_types or {}), **{e['name']:e['type'] for e in facts['entities']}}
    def check(name, field, value, origin):
        declaration = declarations[types[name]][field]
        problem = field_value_issue(declaration,value)
        if problem:
            issues.append(origin + ': ' + problem + '; declared_field=' + json.dumps(declaration,ensure_ascii=False))
    for entity in facts['entities']:
        for field, spec in entity['fields'].items():
            if spec.get('type') == 'stable' or ('value' in spec and 'trajectory' not in spec):
                check(entity['name'],field,spec.get('value'),'entity '+entity['name']+'.'+field)
            else:
                for row in spec.get('trajectory',[]):
                    check(entity['name'],field,row.get('value'),'entity '+entity['name']+'.'+field+'@'+str(row.get('session')))
    for event in facts['events']:
        for effect in event['effects']:
            check(effect['entity'],effect['field'],effect['set'],'event '+event['id']+' '+effect['entity']+'.'+effect['field'])
    return issues


def event_write_context(author_task, world):
    """Project shared histories into the value author's existing effect contract.

    This view adds no validation rule. The original compiler still decides
    whether each declared state write is a real change. In particular a value
    observed at an event is not interchangeable with an event-caused new value.
    """
    rows = []
    histories = world.to_dict()["entities"]
    for event in author_task["events"]:
        for role, fields in event["value_fields"].items():
            entity = author_task["known_identities"].get(event["participants"][role])
            if not entity:
                continue
            for field in fields:
                writes = [{"event_id":e["id"],"session":e["session"],"value":effect["set"]}
                    for e in world.events for effect in e.get("effects", [])
                    if effect["entity"] == entity and effect["field"] == field["name"]]
                rows.append({"event_id":event["slot"],"session":event["session"],
                    "role":role,"entity":entity,"field":field["name"],
                    "declared_field":deepcopy(field),
                    "known_history":deepcopy(histories.get(entity, {}).get(field["name"])),
                    "known_writes":sorted(writes, key=lambda row:(row["session"], row["event_id"])),
                    "meaning":"Required event-caused new value, not a readout of an unchanged value. Use the shared history and business evidence; do not remove a required field or silently drop its effect."})
    return rows


def review_value_groups(wp, state, feedback):
    """Route a review's declared intrinsic targets to their existing authors.

    The review supplies the semantic judgment. This only resolves identities
    and field ownership from the accepted layout; it does not interpret dates,
    prose, or the truth of a finding.
    """
    targets = feedback.get("repair_targets") if isinstance(feedback, dict) else None
    if (not isinstance(targets, dict) or not {"intrinsic", "structure"} <= set(targets)
            or set(targets) - {"intrinsic", "structure", "disclosure"}
            or not isinstance(targets["intrinsic"], list) or type(targets["structure"]) is not bool):
        raise WorldBlueprintError("Review feedback must identify the original intrinsic/structure repair targets")
    owners = {}
    accepted = {row["unit_id"]: row for row in state["units"]}
    for unit in wp["business_instance_plan"]["units"]:
        row = accepted.get(unit["unit_id"])
        if row is None:
            raise WorldBlueprintError("Review correction requires the accepted business units")
        for obj in task(wp, unit, row["instance_mapping"])["objects"]:
            name = row["instance_mapping"][obj["slot"]]
            owners[name] = (unit["unit_id"], {f["name"] for f in obj["intrinsic_fields"]})
    groups, seen = {}, set()
    for target in targets["intrinsic"]:
        if not isinstance(target, dict) or set(target) != {"entity", "fields"}:
            raise WorldBlueprintError("Invalid intrinsic review target")
        name, fields = target["entity"], target["fields"]
        if (not isinstance(name, str) or name not in owners or name in seen
                or not isinstance(fields, list) or not fields
                or any(not isinstance(f, str) or f not in owners[name][1] for f in fields)
                or len(set(fields)) != len(fields)):
            raise WorldBlueprintError("Review targets must belong to the original intrinsic value author")
        seen.add(name)
        unit_id = owners[name][0]
        groups.setdefault(unit_id, []).extend({"code":"world_semantic_review", "retry_group":"unit_values",
            "affected_ids":[name], "carrier":{"entities":[name], "field":field}}
            for field in fields)
    if not groups and not targets["structure"]:
        raise WorldBlueprintError("Review feedback has no business value or layout repair target")
    return groups


def merge_local_values(request, previous, reply, findings):
    """Apply only the declared repair scope, preserving the complete original reply elsewhere."""
    contract = author_contract(request)
    _record(reply, contract, "")
    for section in ("objects", "event_values"):
        _record(reply[section], contract["properties"][section], "/" + section)
    old = previous["author_output"]
    merged = deepcopy(old)
    names = {**request["known_identities"], **previous["instance_mapping"]}
    permitted = {(name, row["carrier"].get("field"))
                 for row in findings for name in row["carrier"]["entities"]}
    applied, ignored = [], []

    def allowed(slot, field):
        return (names.get(slot), field) in permitted or (names.get(slot), None) in permitted

    for slot, obj in reply["objects"].items():
        spec = contract["properties"]["objects"]["properties"][slot]
        _record(obj, spec, _path("/objects", slot))
        for section in ("fields", "initial_values"):
            _record(obj.get(section, {}), spec["properties"][section], _path(_path("/objects", slot), section))
        if obj.get("name") != old["objects"][slot].get("name"):
            ignored.append(f"/objects/{slot}/name")
        for section in ("fields", "initial_values"):
            before, after = old["objects"][slot].get(section, {}), obj.get(section, {})
            for field in sorted(set(before) | set(after)):
                if field in before and field in after and before[field] == after[field]:
                    continue
                path = f"/objects/{slot}/{section}/{field}"
                if not allowed(slot, field):
                    ignored.append(path)
                    continue
                target = merged["objects"][slot].setdefault(section, {})
                if field in after:
                    target[field] = deepcopy(after[field])
                else:
                    target.pop(field, None)
                applied.append(path)
    for event in request["events"]:
        slot = event["slot"]
        spec = contract["properties"]["event_values"]["properties"][slot]
        _record(reply["event_values"][slot], spec, _path("/event_values", slot))
        for role, values in reply["event_values"][slot].items():
            _record(values, spec["properties"][role], _path(_path("/event_values", slot), role))
            for field, value in values.items():
                if value == old["event_values"][slot][role][field]:
                    continue
                path = f"/event_values/{slot}/{role}/{field}"
                if allowed(event["participants"][role], field):
                    merged["event_values"][slot][role][field] = deepcopy(value)
                    applied.append(path)
                else:
                    ignored.append(path)
    merged["issue_responses"] = deepcopy(reply.get("issue_responses", []))
    return merged, {"version": "scoped-value-merge/v1", "applied_paths": applied,
                    "ignored_paths": ignored, "disposition": "preserve_unrelated_facts"}


def generate_world(wp, tracer, existing=None, checkpoint_path=None, feedback=None, log=print, resume_state=None,
                   repair_max_calls=None, allow_supply_shortfall=False):
    """Dependency-ready author tasks run concurrently; commits use the original compiler."""
    from pipeline.world_agent import _binding, _digest, _save, _compiled, _combine, _context, _unit_issues
    from pipeline.seed_world import validate_seed_world
    from pipeline.supply_capacity import CapacityReviewRequested
    from pipeline.instance_plan import compile_plan, PlanConflict
    plan = wp["business_instance_plan"]
    try:
        compile_plan(wp, {"decision": "plan", **{k: deepcopy(plan[k]) for k in ("reason", "units", "cohorts")},
                          **({"execution_policy":plan["execution_policy"]} if plan.get("execution_policy") else {})})
    except PlanConflict as error:
        raise CapacityReviewRequested({"phase": "business_layout", "findings": error.findings}) from error
    bp = normalize_world_blueprint(wp)
    binding = _binding(wp, existing)
    units, *_ = index(wp["business_instance_plan"])
    maximum = (wp.get("world_generation") or {}).get("max_steps", min(240, max(32, sum(t["count"] for t in bp["entity_types"]) * 3)))
    if type(maximum) is not int or not 1 <= maximum <= 600:
        raise WorldBlueprintError("world_generation.max_steps must be 1..600")
    repair_limit = (wp.get("world_generation") or {}).get("value_repair_attempts", min(maximum, DEFAULT_VALUE_REPAIR_ATTEMPTS))
    if type(repair_limit) is not int or not 1 <= repair_limit <= maximum:
        raise WorldBlueprintError("world_generation.value_repair_attempts must be positive and within max_steps")
    state = {"version": VERSION, "binding": binding, "units": [], "retired_units": [], "log": [],
        "status": "building", "steps": 0, "feedback_history": [], "issue_responses": [],
        "input_snapshot": {"whitepaper": deepcopy(wp), "blueprint": deepcopy(bp), "max_steps": maximum}}
    saved = json.loads(Path(checkpoint_path).read_text(encoding="utf-8")) if checkpoint_path and Path(checkpoint_path).exists() else resume_state
    if saved is not None:
        if (saved.get("version") != VERSION or saved.get("binding") != binding
                or saved.get("checkpoint_hash") != _digest({k: v for k, v in saved.items() if k != "checkpoint_hash"})
                or saved.get("input_snapshot") != state["input_snapshot"]):
            raise WorldBlueprintError("Business value checkpoint input/source binding mismatch")
        state = deepcopy(saved)
    state.setdefault("local_repairs", [])
    if repair_max_calls is not None and (type(repair_max_calls) is not int or repair_max_calls < 0):
        raise WorldBlueprintError("world repair max_calls must be a nonnegative integer")
    pending_review = next((row for row in state["feedback_history"] if row.get("status") == "pending"), None)
    if pending_review and feedback and pending_review["hash"] != _digest(feedback):
        raise WorldBlueprintError("An unfinished review transaction cannot be replaced by different feedback")
    if pending_review:
        if repair_max_calls is not None and pending_review.get("max_calls") != repair_max_calls:
            raise WorldBlueprintError("Saved review transaction call allowance cannot change on resume")
        feedback = deepcopy(pending_review["feedback"])
    elif feedback and (not state["feedback_history"] or state["feedback_history"][-1]["hash"] != _digest(feedback)):
        if any(row.get("version") == VALUE_TRANSACTION_VERSION and row["status"] == "running"
               for row in state["local_repairs"]):
            raise WorldBlueprintError("Finish the existing value transaction before accepting new review feedback")
        groups = review_value_groups(wp, state, feedback)
        pending_review = {"hash":_digest(feedback), "feedback":deepcopy(feedback),
                          "groups":groups, "status":"pending", "start_steps":state["steps"],
                          "max_calls":repair_max_calls}
        state["feedback_history"].append(pending_review)
        state["status"] = "building"
    lock = threading.RLock()
    halted = threading.Event()
    def save():
        _save(checkpoint_path, state)
    def request_design(code, reason, affected_ids, evidence):
        """All rejected business content returns to the existing design owner."""
        request = {"phase":"business_layout", "findings":[{
            "kind":"business", "code":code, "message":reason,
            "affected_ids":list(affected_ids), "evidence":deepcopy(evidence),
            "allowed_edit_scope":"uncommitted_plan_with_original_seed_and_limits"}]}
        with lock:
            if state.get("fulfillment"):
                request["instance_fulfillment"] = deepcopy(state["fulfillment"])
            state.setdefault("layout_requests", []).append(deepcopy(request))
            state.update(status="capacity_review_requested", layout_request=deepcopy(request))
            halted.set(); save()
        raise CapacityReviewRequested(request)
    if (state.get("status") == "capacity_review_requested" and state.get("layout_request")
            and not (allow_supply_shortfall and (state.get("candidate") or {}).get("compiler") == "passed")):
        raise CapacityReviewRequested(deepcopy(state["layout_request"]))
    from pipeline.construction import enabled as construction_enabled, resolve as resolve_construction
    executable = construction_enabled(wp)
    shared_identities = plan.get("execution_policy") == "shared-identities/v1"
    if shared_identities and not state.get("identity_registry"):
        catalog=[dict(o,business_purpose=u["business_purpose"]) for u in plan["units"] for o in u["objects"]]
        payload={"business":_context(wp,bp)["business"],"objects":catalog}
        for attempt in range(3):
            if state["steps"] >= maximum:
                request_design("identity_allocation_unresolved", "World author allowance exhausted before identity allocation",
                    units, {"compiler_issues":payload.get("feedback")})
            state["steps"] += 1;save()
            raw=tracer.chat_json("world.instance.identities",[{"role":"system","content":"为已经编译的对象目录分配自然、互异且符合原seed的名称。只输出JSON {identities:{原slot:名称}}，覆盖全部对象；本阶段不填写事实或字段值。"},
                {"role":"user","content":json.dumps(payload,ensure_ascii=False)}],model=config.STRUCTURE_MODEL,temperature=.3,max_tokens=32768,retries=3,response_format={"type":"json_object"})
            names=raw.get("identities") if isinstance(raw,dict) else None
            if isinstance(raw, dict) and "__error__" in raw:
                from execution_control import global_stop_exception
                error = global_stop_exception(raw) or RuntimeError("Identity allocation provider failure: " + str(raw["__error__"]))
                state.update(status="error", error=f"{type(error).__name__}: {error}")
                save(); raise error
            if (isinstance(names,dict) and set(names)=={o["slot"] for o in catalog}
                    and all(isinstance(n,str) and n.strip() for n in names.values()) and len(set(names.values()))==len(names)):
                state["identity_registry"]=deepcopy(names);save();break
            payload["feedback"]={"expected_slots":[o["slot"] for o in catalog],"previous_reply":raw,"error":"Return every original slot with one distinct nonempty name"}
        else:
            request_design("identity_allocation_unresolved", "Identity allocation remains structurally unresolved",
                units, {"author_response":raw, "compiler_issues":payload["feedback"]})
    def author(unit, world, local_feedback=None, previous=None, review_feedback=None, *, accepted_units=None):
        # Parallel result commits must not change another author's frozen input.
        base_units = deepcopy(state["units"] if accepted_units is None else accepted_units)
        semantic_repair = bool(previous and any(f.get("code") == "world_semantic_review" for f in local_feedback))
        known = (deepcopy(state["identity_registry"]) if shared_identities
            else progress(wp, {**state, "units":base_units})["slot_mapping"])
        if previous and not shared_identities:
            own = {o["slot"] for o in unit["objects"]}
            known = {slot: name for slot, name in known.items() if slot not in own}
        request = task(wp, unit, known)
        value_free = not request["objects"] and not request["events"]
        names = {actual for slot, actual in known.items() if slot in index(wp["business_instance_plan"])[1]}
        context = {"entities": [{"name": n, "type": world.entity_types[n], "canonical_fields": world.to_dict()["entities"][n]} for n in sorted(names)],
            "events": deepcopy(world.events), "relations": deepcopy(world.relations)}
        global_context = _context(wp, bp)
        payload = {"business": global_context["business"], "calendar": global_context["calendar"],
            "fact_time_contract": global_context["fact_time_contract"],
            "task": request, "dependency_world": context, "feedback": deepcopy(review_feedback if previous else feedback),
            "response_schema": response_contract(request),
            "event_write_context":event_write_context(request, world)}
        if executable:
            from pipeline.construction import author_requirements
            payload["construction_requirements"] = author_requirements(wp, unit['unit_id'])
        if previous:
            payload["local_repair"] = {"issues": deepcopy(local_feedback), "previous_response": previous["author_output"],
                "instruction": "Revise the affected values with business evidence. The program applies only the listed entity/field scope to the accepted original; names, relationships, roles, schedules and unrelated values are retained. Return issue_responses for disputed or deferred findings; these do not change the original checks."}
        cycle_hash = _digest(payload)
        rejected_projection = []
        effective_raw, merge_receipt = None, None
        def compile_values(raw):
            nonlocal rejected_projection, effective_raw, merge_receipt
            rejected_projection = []
            effective_raw, merge_receipt = None, None
            effective_raw, merge_receipt = (merge_local_values(request, previous, raw, local_feedback)
                                           if previous else (deepcopy(raw), None))
            raw = effective_raw
            facts, mapping = materialize(request, raw)
            if semantic_repair and not raw.get("issue_responses"):
                return facts, mapping, ["New review feedback requires explicit issue_responses before finish"]
            # Initial states are part of the author's transaction. Run the
            # original validator before a reply can enter the shared context;
            # its detailed findings use the existing bounded correction loop.
            value_issues = authored_value_issues(facts,bp,existing,world.entity_types)
            if value_issues:
                return facts,mapping,value_issues
            if previous and any(mapping.get(o["slot"]) != previous["instance_mapping"].get(o["slot"]) for o in unit["objects"]):
                return facts, mapping, ["Local value correction must preserve the allocated object identities"]
            validation_world=world
            if shared_identities:
                validation_world=deepcopy(world)
                accepted_names={e["name"] for row in base_units for e in row["raw"]["entities"]}
                for entity in facts["entities"]:
                    if entity["name"] not in accepted_names:validation_world.entities.pop(entity["name"],None)
                validation_world.relations=[r for r in validation_world.relations if r["id"] not in {x["id"] for x in facts["relations"]}]
            # A compiled unit may only read prior facts to declare observations,
            # obligations or publications. It has no authorable values, and an
            # empty fact table is correct for that unit.
            no_new_facts = value_free and not any(facts.values())
            issues = [] if previous or no_new_facts else _unit_issues(facts, context, validation_world, bp)
            if not issues:
                base = [row for row in base_units if not previous or row["unit_id"] != unit["unit_id"]]
                staged = assembled_facts(wp, base + [{"raw":facts, "instance_mapping":mapping}],
                    complete=bool(previous),identities=state.get("identity_registry"))
                _world, _applied, _initial, compiler_issues = _compiled(staged,
                    bp, existing, bool(previous), wp)
                if shared_identities and not semantic_repair:
                    from pipeline.world_state import structural_projection_findings
                    own_events = {event['id'] for event in facts['events']}
                    failures = [f for f in structural_projection_findings(staged, _world, compiler_issues)
                        if f['collection'] == 'events' and f['id'] in own_events]
                    rejected_projection = deepcopy(failures)
                    issues = [json.dumps(f, ensure_ascii=False) for f in failures]
                else:
                    issues = compiler_issues
                if not issues and previous:
                    for condition in local_feedback:
                        if condition.get('line'):
                            from pipeline.lines import line_for
                            issues.extend(line_for(condition["line"]).value_issues(_world, condition["carrier"]))
            return facts, mapping, issues
        def request_layout(raw, issues, reason):
            # Values cannot change fixed event declarations or their schedule.
            # Give the existing layout transaction the original failed facts;
            # it may repair the layout or dispute the finding under the same
            # compiler, business review and production revision allowance.
            request_design("fixed_layout_values_unresolved", reason, [unit["unit_id"]],
                {"task":request, "author_response":raw, "compiler_issues":issues,
                 "projection":rejected_projection})
        def accepted_row(raw, facts, mapping):
            raw = effective_raw if previous else raw
            row = {"unit_id":unit["unit_id"], "intent":unit["business_purpose"], "raw":facts,
                "raw_hash":_digest(facts), "context":context, "instance_units":[unit["unit_id"]],
                "instance_mapping":mapping, "issue_responses":deepcopy(raw.get("issue_responses", []))}
            row["author_output"] = deepcopy(raw)
            if merge_receipt is not None:
                row["scoped_merge"] = deepcopy(merge_receipt)
            return row
        if value_free and previous is None:
            # Relations are already fully specified by the compiled plan;
            # read-only units add no fact rows at all. Neither needs an LLM
            # value request, and downstream fulfillment still checks every
            # observation and capability obligation against the actual world.
            raw = {"objects": {}, "event_values": {}}
            try:
                facts, mapping, issues = compile_values(raw)
            except (WorldBlueprintError, TypeError, KeyError, ValueError) as error:
                issues = [str(error)]
            if issues:
                request_layout(raw, issues, "Compiled no-value business unit cannot be executed")
            row = accepted_row(raw, facts, mapping)
            row["execution"] = "deterministic_no_values/v1"
            return row
        # Returned facts remain available across an execution repair. Recompile
        # them against today's dependency world; old rejection records stay
        # intact, and external review feedback must match before reuse.
        last_failure = None
        raw, issues = None, ["No usable author reply was returned in the admitted repair slots"]
        for prior in reversed(deepcopy(state["log"])) if not previous else []:
            old_input, raw = prior.get("input", {}), prior.get("raw_output")
            if (prior.get("unit_id") != unit["unit_id"] or not isinstance(raw, dict) or "__error__" in raw
                    or any(old_input.get(key) != payload[key] for key in
                        ("task", "business", "calendar", "feedback", "response_schema"))
                    or old_input.get("construction_requirements") != payload.get("construction_requirements")):
                continue
            try:
                facts, mapping, issues = compile_values(raw)
            except (WorldBlueprintError, TypeError, KeyError, ValueError) as error:
                issues = [str(error)]
            if not issues:
                row = accepted_row(raw, facts, mapping)
                row["revalidated_from"] = {"attempt":prior["attempt"], "old_status":prior["status"],
                    "original_input_hash":_digest(old_input), "original_raw_hash":_digest(raw),
                    "old_issues":deepcopy(prior.get("issues")), "current_issues":[],
                    "validation_context_hash":_digest(context), "implementation":deepcopy(binding["implementation"])}
                return row
            # A persisted reply that fails today's compiler is an input to
            # correction, not a reason to buy the original request again.
            # Keep the old reply and findings; record this revalidation apart.
            finding = {"unit_id":unit["unit_id"], "original_input_hash":_digest(old_input),
                "original_raw_hash":_digest(raw), "issues":deepcopy(issues),
                "implementation":deepcopy(binding["implementation"])}
            with lock:
                recorded = state.setdefault("revalidation_findings", [])
                if finding not in recorded:
                    recorded.append(finding); save()
            payload["correction"] = {"original_response":deepcopy(raw), "compiler_issues":issues}
            last_failure = _digest({"response":raw, "issues":issues})
            if any(
                    old != prior and old.get("unit_id") == unit["unit_id"]
                    and old.get("raw_output") == raw and old.get("issues") == issues
                    and all(old.get("input", {}).get(key) == payload.get(key) for key in
                        ("task", "business", "calendar", "feedback", "response_schema", "construction_requirements"))
                    for old in state["log"]):
                request_layout(raw, issues, "The saved value corrections repeated the same compiler rejection; review the fixed layout before buying another value correction.")
            break
        # Both initial authoring and correction have the same bounded handoff.
        # Exhaustion preserves diagnostics and spends the shared design budget.
        for attempt in range(min(repair_limit, maximum)):
            with lock:
                admitted = next((row for row in reversed(state["log"]) if row.get("cycle_hash") == cycle_hash
                    and row.get("attempt") == attempt + 1), None)
                if previous and admitted is not None:
                    if admitted.get("input") != payload:
                        raise WorldBlueprintError("Saved local correction input differs from its original attempt")
                    # Admission consumes the original attempt even if its reply
                    # was lost. A resume may use the remaining slots, not buy
                    # the missing reply in the same slot again.
                    if "raw_output" not in admitted or (isinstance(admitted["raw_output"], dict) and "__error__" in admitted["raw_output"]):
                        continue
                entry = admitted if admitted is not None and admitted.get("input") == payload and "raw_output" in admitted \
                    and not (isinstance(admitted["raw_output"], dict) and "__error__" in admitted["raw_output"]) else None
                cached = entry is not None
                if not cached:
                    if halted.is_set():
                        if state.get("layout_request"):
                            raise CapacityReviewRequested(deepcopy(state["layout_request"]))
                        raise RuntimeError("Business author dispatch stopped after a concurrent execution failure")
                    if state["steps"] >= maximum:
                        request_layout(raw, issues, "Original world value allowance exhausted")
                    if (semantic_repair and pending_review and pending_review.get("max_calls") is not None
                            and state["steps"] - pending_review["start_steps"] >= pending_review["max_calls"]):
                        request_layout(raw, issues, "Original review correction call allowance exhausted")
                    state["steps"] += 1
                    entry = {"step": "world.instance.values", "unit_id": unit["unit_id"], "attempt": attempt + 1,
                        "cycle_hash": cycle_hash, "input": deepcopy(payload), "status": "running"}
                    state["log"].append(entry); save()
            try:
                if _binding(wp, existing) != binding:
                    raise WorldBlueprintError("Business value implementation/input drift")
                raw = deepcopy(entry["raw_output"]) if cached else tracer.chat_json("world.instance.values", [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    model=config.STRUCTURE_MODEL, temperature=.3, max_tokens=32768,
                    retries=3, strict_json=True, response_format={"type": "json_object"})
                with lock:
                    entry["raw_output"] = deepcopy(raw); entry["status"] = "returned"; save()
                if _binding(wp, existing) != binding:
                    raise WorldBlueprintError("Business value implementation/input drift after response")
                if isinstance(raw, dict) and "__error__" in raw:
                    from execution_control import global_stop_exception
                    stopped = global_stop_exception(raw)
                    if stopped is not None:
                        raise stopped
                    raise RuntimeError("Business value author execution failed: " + str(raw["__error__"]))
                if (isinstance(raw, dict) and set(raw) == {"decision", "reason"} and raw.get("decision") == "unresolved"
                        and isinstance(raw.get("reason"), str) and raw["reason"].strip()):
                    request_layout(raw, [raw["reason"]], "Value author reports a conflict in the fixed layout")
                try:
                    facts, mapping, issues = compile_values(raw)
                except (WorldBlueprintError, TypeError, KeyError, ValueError) as error:
                    issues = [str(error)]
                with lock:
                    entry["issues"] = deepcopy(issues); entry["status"] = "rejected" if issues else "ready"; save()
                    if previous and effective_raw is not None:
                        entry.update(applied_output=deepcopy(effective_raw), scoped_merge=deepcopy(merge_receipt)); save()
                if not issues:
                    return accepted_row(raw, facts, mapping)
                failure = _digest({"response": raw, "issues": issues})
                if failure == last_failure:
                    request_layout(raw, issues, "The value author repeated the same rejected correction; review the layout and actual supply with the original facts and compiler findings.")
                last_failure = failure
                payload["correction"] = {"original_response": raw, "compiler_issues": issues}
                if state["steps"] >= maximum:
                    request_layout(raw, issues, "The original value-stage allowance is exhausted; review the fixed layout without extending that allowance.")
            except CapacityReviewRequested:
                with lock:
                    entry["status"] = "design_review_requested"; save()
                raise
            except Exception as error:
                halted.set()
                with lock:
                    entry.update(status="execution_error", error=f"{type(error).__name__}: {error}"); save()
                raise
        request_layout(raw, issues, "The bounded value correction transaction remains unresolved; review the layout and actual supply while retaining accepted business units.")
    save()
    def repair_values(groups, world, review_hash=None):
        """One durable bounded value transaction for every source of feedback."""
        from pipeline.world_state import WorldState
        for unit_id, findings in groups.items():
            repairs = [r for r in state["local_repairs"] if r["unit_id"] == unit_id]
            if not review_hash and executable and not progress(wp, state)["pending_units"]:
                current, _applied, _initial, problems = _compiled(
                    assembled_facts(wp, state["units"], complete=True), bp, existing, True, wp)
                if not problems:
                    world = current
                    proof = fulfillment(wp, world, state)
                    state["fulfillment"] = deepcopy(proof)
                    current_findings = []
                    for finding in proof["findings"]:
                        if (finding.get("retry_group") != "unit_values"
                                or unit_id not in finding.get("responsible_units", [finding.get("unit_id")])):
                            continue
                        row = next(r for r in plan["construction"] if r["id"] in finding["affected_ids"])
                        current_findings.append({**finding, "line":row["line"]})
                    # Event compiler repairs still apply even if capacity is full.
                    current_findings.extend(f for f in findings if f.get("code") == "event_compilation")
                    if current_findings != findings:
                        state.setdefault("supply_rechecks", []).append({"unit_id":unit_id,
                            "previous_findings":deepcopy(findings), "current_findings":deepcopy(current_findings),
                            "actual_supply":deepcopy(proof["actual_supply"]), "world_hash":proof["world_hash"]})
                        for receipt in repairs:
                            if receipt.get("version") == VALUE_TRANSACTION_VERSION and receipt["status"] == "running":
                                receipt.update(status="superseded_by_inventory", current_findings=deepcopy(current_findings))
                        save()
                    findings = current_findings
                    if not findings:
                        continue
            if review_hash and any(r.get("review_hash") == review_hash and r["status"] == "completed" for r in repairs):
                continue
            old = next(r for r in state["units"] if r["unit_id"] == unit_id)
            receipt = next((r for r in repairs if r.get("version") == VALUE_TRANSACTION_VERSION and r["status"] == "running"), None)
            if receipt is None:
                if len(repairs) >= repair_limit:
                    raise CapacityReviewRequested({"phase":"business_layout", "unit_id":unit_id,
                        "reason":"Local value repair allowance exhausted; retain the compiled world and review actual supply",
                        "instance_fulfillment":deepcopy(state.get("fulfillment", {})), "findings":deepcopy(findings)})
                receipt = {"version":VALUE_TRANSACTION_VERSION, "unit_id":unit_id, "before_hash":_digest(old), "findings":deepcopy(findings),
                    "dependency_world":deepcopy(world.to_dict()), "feedback":deepcopy(feedback), "review_hash":review_hash,
                    "status":"running"}
                state["local_repairs"].append(receipt); save()
            if (receipt["before_hash"] != _digest(old) or receipt["findings"] != findings
                    or receipt.get("review_hash") != review_hash):
                raise WorldBlueprintError("Saved value transaction differs from its accepted values or findings")
            frozen_world = WorldState.from_dict(receipt["dependency_world"]) if "dependency_world" in receipt else world
            revised = author(units[unit_id], frozen_world, findings, old, receipt.get("feedback"))
            receipt.update(status="completed", after_hash=_digest(revised))
            state["retired_units"].append(deepcopy(old))
            state["units"] = [revised if r["unit_id"] == unit_id else r for r in state["units"]]
            state["issue_responses"].extend(deepcopy(revised["issue_responses"]))
            save()
    try:
        while True:
            if shared_identities:
                try:
                    world = declared_context(wp,state["units"],state["identity_registry"],bp,existing)
                except WorldBlueprintError as error:
                    request_design("partial_world_compilation", "Shared business context cannot be assembled", units,
                        {"compiler_issues":[str(error)]})
                issues = []
            else:
                try:
                    world, applied, initial, issues = _compiled(assembled_facts(wp, state["units"]), bp, existing, False, wp)
                except WorldBlueprintError as error:
                    request_design("partial_world_compilation", "Accepted business units cannot be assembled", units,
                        {"compiler_issues":[str(error)]})
            if issues:
                request_design("partial_world_compilation", "Accepted units conflict in the assembled world", units,
                    {"compiler_issues":issues, "accepted_units":[r["unit_id"] for r in state["units"]]})
            if pending_review and pending_review["feedback"]["repair_targets"]["structure"]:
                evidence = {"phase":"business_layout", "findings":[{
                    "kind":"business", "code":"world_review_structure", "message":"Original review requires a layout correction",
                    "affected_ids":list(pending_review["groups"]), "evidence":deepcopy(pending_review["feedback"]),
                    "allowed_edit_scope":"uncommitted_plan_with_original_seed_and_limits"}]}
                state["layout_request"] = evidence; save()
                raise CapacityReviewRequested(evidence)
            # Historical receipts remain evidence and consume their original
            # allowance. Only the new durable protocol has a resumable pending
            # transaction; an old interrupted receipt can precede a later
            # accepted correction without being current work.
            running = next((r for r in state["local_repairs"]
                if r.get("version") == VALUE_TRANSACTION_VERSION and r["status"] == "running"), None)
            if running and allow_supply_shortfall and not running.get("review_hash"):
                running.update(status="deferred_after_design_limit")
                save()
                continue
            if running:
                repair_values({running["unit_id"]:running["findings"]}, world, running.get("review_hash"))
                continue
            if pending_review:
                repair_values(pending_review["groups"], world, pending_review["hash"])
                pending_review["status"] = "completed"; save()
                pending_review = None
                continue
            ready = progress(wp, state)["ready_units"]
            if not progress(wp, state)["pending_units"]:
                complete = assembled_facts(wp, state["units"], complete=True)
                try:
                    world, applied, initial, issues = _compiled(complete, bp, existing, True, wp)
                except WorldBlueprintError as error:
                    request_design("complete_world_compilation", "Complete business world cannot be assembled", units,
                        {"compiler_issues":[str(error)]})
                if shared_identities and issues:
                    from pipeline.world_state import structural_projection_findings
                    _units, _objects, _events, _relations, owners = index(plan)
                    failures = structural_projection_findings(complete, world, issues)
                    groups = {}
                    for failed in failures:
                        if failed['collection'] != 'events':
                            continue
                        proposed = failed['evidence']['proposed']
                        canonical = failed['evidence']['canonical']
                        retained = (canonical or {}).get('effects', [])
                        for effect in proposed['effects']:
                            if effect not in retained:
                                groups.setdefault(owners[failed['id']], []).append({
                                    'code':'event_compilation', 'retry_group':'unit_values',
                                    'affected_ids':[failed['id']], 'evidence':deepcopy(failed['evidence']),
                                    'carrier':{'entities':[effect['entity']], 'field':effect['field']}})
                    if groups:
                        state.setdefault('compiler_findings', []).append(deepcopy(failures)); save()
                        repair_values(groups, world)
                        continue
                if issues:
                    request_design("complete_world_compilation", "Complete business world compilation failed", units,
                        {"compiler_issues":issues})
                try:
                    validate_seed_world(wp, world)
                except WorldBlueprintError as error:
                    request_design("seed_world_compilation", "Complete business world does not satisfy the seed", units,
                        {"compiler_issues":[str(error)], "seed_report":getattr(error, "report", None)})
                if executable:
                    world.supply_construction = resolve_construction(wp, progress(wp,state)["slot_mapping"])
                proof = fulfillment(wp, world, state)
                # Preserve the valid world independently from its supply verdict.
                state.update(fulfillment=deepcopy(proof), initial_states=initial,
                    candidate={"table":deepcopy(applied), "world_hash":proof["world_hash"],
                               "table_hash":_digest(applied), "compiler":"passed"})
                save()
                if not proof["passed"]:
                    if allow_supply_shortfall:
                        state.update(status="completed_with_warnings", result_hash=_digest(applied),
                            warning={"code":"source_supply_shortfall_after_design_limit",
                                     "release_eligible":False, "findings":deepcopy(proof["findings"])})
                        save()
                        return applied, deepcopy(state)
                    local = [f for f in proof["findings"] if f.get("retry_group") == "unit_values"]
                    if executable and local:
                        groups = {}
                        for f in local:
                            row = next(r for r in wp["business_instance_plan"]["construction"] if r["id"] in f["affected_ids"])
                            for unit_id in f.get('responsible_units', [f['unit_id']]):
                                groups.setdefault(unit_id, []).append({**f, "line":row["line"]})
                        repair_values(groups, world)
                        continue
                    state.update(status="capacity_review_requested", fulfillment=proof); save()
                    raise CapacityReviewRequested({"phase": "actual_instance_fulfillment", "findings": proof["findings"], "proof": proof})
                state.update(status="completed", initial_states=initial, result_hash=_digest(applied), fulfillment=proof); save()
                return applied, deepcopy(state)
            if not ready:
                request_design("dependency_schedule_unresolved", "Business dependency graph has no ready task", units,
                    {"progress":progress(wp, state)})
            workers = min(max(1, int(config.LLM_CONCURRENCY)), len(ready))
            batch_units = deepcopy(state["units"])
            def commit(row):
                with lock:
                    issues = []
                    if not shared_identities:
                        try:
                            _world, _applied, _initial, issues = _compiled(assembled_facts(wp, state["units"] + [row]), bp, existing, False, wp)
                        except WorldBlueprintError as error:
                            issues = [str(error)]
                    if issues:
                        request_design("concurrent_business_merge", "Concurrent business merge conflict", [row["unit_id"]],
                            {"compiler_issues":issues, "candidate_unit":row})
                    state["units"].append(row)
                    state["issue_responses"].extend(deepcopy(row.get("issue_responses", []))); save()
                log(f"  ✓ 业务单元 {row['unit_id']}: {len(row['raw']['entities'])} 实体 / {len(row['raw']['events'])} 事件")
            config.pmap(lambda uid: author(units[uid], world, accepted_units=batch_units), ready,
                workers=workers, on_result=commit)
    except CapacityReviewRequested as request:
        state.update(status="capacity_review_requested", layout_request=deepcopy(request.evidence))
        save(); raise
    except Exception as error:
        state.update(status="error", error=f"{type(error).__name__}: {error}"); save(); raise
