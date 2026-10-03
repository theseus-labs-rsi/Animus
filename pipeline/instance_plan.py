"""Executable business layout between capability demand and world authorship.

Plans allocate identities, roles and time, never numeric answers. Original
compilation, seed/business review and order enumeration remain the validators.
"""
from collections import Counter
from copy import deepcopy
import json

import config
from pipeline.capability_contract import digest, requirements
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint, relation_owner_side

VERSION = "business-instance-plan/v3"
PRODUCTION_VERSION = "supply-driven/v5"
FULFILLMENT_POLICY = "actual-independent-supply/v1"


class PlanConflict(WorldBlueprintError):
    def __init__(self, findings):
        self.findings = deepcopy(findings)
        super().__init__("Business instance plan conflicts: " + json.dumps(findings, ensure_ascii=False))


def finding(code, message, ids=(), kind="structure", evidence=None):
    return {"kind": kind, "code": code, "message": message, "affected_ids": list(ids),
        "evidence": deepcopy(evidence), "retry_group": "business_plan",
        "allowed_edit_scope": "uncommitted_plan_with_original_seed_and_limits"}


SYSTEM = """你规划同一领域世界里的具体业务实例，使每条能力线的独立候选储备可以实际实现。
读取原seed业务、完整蓝图、需求合同和真实失败反馈。输出完整可执行业务布局，保持原字段语义、exact身份、业务规则和实体/时间上限。
每个单元是一段独立但可共享机构的自然业务过程。给每个对象/关系/事件唯一slot；跨单元引用同一slot并声明depends_on。计划覆盖蓝图所有类型和事件/关系最低数量。数量由槽位导出。
安排真实业务过程、事件角色、因果父节点、零基session窗口、字段观测期和公开渠道。event-owned字段必须由已有事件写入；relation-owned字段须通过已有关系。标量绑定的拥有方只能绑定一个目标。一个不可逆状态对象的写入次数受其声明阶段数量约束。
每条要求线在全计划合计列出independent_candidates数量的obligations，每条需求只归属一个单元，id全局唯一；carrier.entities引用支持该任务的对象槽位，carrier.field为可选已有字段。L7可用native_trend或compare：趋势需适合业务的数值轨迹及足够的独立期，禁止选天然单调累计量；比较组在观察结果前按真实类型/稳定归属声明，改变组内对象数不会增加独立组数。observations说明观测机制和sessions，具体数值由事实作者创作。
L3以独立因果过程分配，保留唯一定位；L5安排独立来源主张及范围；L8保留合理的在办实例；L6保留真实未知与多样子型。一个业务单元可承担多条能力，但同一事实换措辞或排列不计新增。
禁止预填gt、answer、winner、trend_direction、value或固定数字轨迹。给出业务目的、约束和采样安排，事实作者根据业务生成值，原验证器随后核答案。
严格JSON：{"decision":"plan|unresolved","reason":"业务依据","units":[{"unit_id":"id","business_purpose":"业务过程","depends_on":[],"objects":[{"slot":"object-id","type":"已有类型"}],"relations":[{"slot":"relation-id","type":"已有关系","from":"对象slot","to":"对象slot","session":0}],"events":[{"slot":"event-id","type":"已有事件","participants":{"角色":"对象slot"},"session":1,"caused_by":"可选父事件slot"}],"observations":[{"entity":"对象slot","field":"已有字段","sessions":[0,1,2,3],"mechanism":"业务观测原因"}],"obligations":[{"id":"需求id","line":"规范line id","subtype":"可选","carrier":{"entities":["对象slot"],"field":"可选已有字段"}}],"publications":[{"fact_slot":"对象/关系/事件slot","channel":"已有公开渠道","session":1}]}],"cohorts":[{"id":"group-id","basis":"先验业务依据","members":["对象slot"]}]}。
若现有schema或授权范围无法实现，返回unresolved并列出具体约束冲突。"""

SYSTEM += """\nconstruction条件来自原出题器，须落实到对象、字段和时间安排。L3每项需求聚焦一个对象，该对象至少有三个不同期的真实变更，初始SET不算变更，值需可唯一定位。
L5每项需求指定真实对象的同一个已声明文本/类别/人名/状态字段；程序会从另一个对象该字段的真实取值构造低可信主张，须保留同字段不同取值的业务来源。不要把两个文档名当成冲突对象。
L7明确填写native_trend或compare；native_trend字段原生数值、至少四个观测期、首末净变化占摆幅至少30%、包含与全段方向相反的局部变化，具体读数由值作者创作。
compare按程序的真实类型和稳定reference字段分组。若同类型按某个稳定归属字段分成多个组，选其中一个完整自然组；不要仅用basis文字把不同组并在一起。比较字段需要至少两次变动数的领先幅度，领先对象由实际值决定。
每线储备使用不同对象/字段/事实范围，重复同一carrier并换id不能增加储备。跨线可共享已有过程；按依赖图并行独立业务单元。"""

REPAIR_SYSTEM = """你修订一份已保存的业务实例计划，解决程序或原业务审查提供的具体问题。
保留原seed、schema、角色、期数、实体上限和逐线配额。只输出对具体记录的变更；未列入变更的单元和记录由程序逐字保留。
每条能力合同的independent_candidates是全计划总储备。每条需求只归属一个业务单元，id全局唯一；跨单元使用事实时通过carrier和depends_on引用，不复制需求。
先核对已有记录，再针对findings修订相关关系、事件、观测或需求。可以增加确有业务依据的对象和记录，完整原编译器及业务审查会重新检验。
严格JSON：{"decision":"repair|unresolved","reason":"依据","changes":[{"unit_id":"已有单元id","collection":"objects|relations|events|observations|obligations|publications|depends_on","operation":"add|replace|remove","key":"记录键","value":{}}],"cohort_changes":[]}。
objects/relations/events用slot为key，obligations和cohorts用id；key统一定义记录身份，value可省略重复的slot/id，程序补入；若重复填写须相同。observations/publications的新增记录由程序从value推导自然键，add的key可省略。replace/remove分别用entity|field或fact_slot|channel|session定位原记录。add必须新记录，replace/remove必须存在唯一记录；replace保留原身份。depends_on只允许replace，value为完整依赖单元id数组，无key。
cohort_changes采用同样operation/key/value。changes和cohort_changes各最多64项，不能传入完整units或修改业务单元身份。
禁止预填答案和具体数值。若在原schema/授权范围内无法解决，返回unresolved和具体业务冲突。"""


REPAIR_SYSTEM += "\n定位原记录时优先使用 selector 对象：observations 用 {entity,field}，publications 用 {fact_slot,channel,session}。selector 必须唯一匹配原记录。公开时间可在所选原记录上修改，fact_slot 与 channel 保留，新的自然键不得与其他记录重复。edit_catalog 已列出可直接使用的 selector；add 的新记录从 value 推导身份。"


def _record_key(collection, row):
    if not isinstance(row, dict):
        return None
    if collection in ("objects", "relations", "events"):
        return row.get("slot")
    if collection in ("obligations", "cohorts"):
        return row.get("id")
    if collection == "observations":
        return str(row.get("entity")) + "|" + str(row.get("field"))
    if collection == "publications":
        return "|".join(str(row.get(k)) for k in ("fact_slot", "channel", "session"))
    return None


def _apply_repair(proposal, change_set):
    """Apply keyed edits atomically; all untouched original rows survive."""
    if (not isinstance(change_set,dict) or change_set.get("decision") != "repair"
            or set(change_set) - {"decision","reason","changes","cohort_changes"}):
        raise PlanConflict([finding("repair_shape", "Repair requires an explicit bounded change set")])
    result = deepcopy(proposal)
    units = result.get("units")
    if (not isinstance(units,list) or any(not isinstance(u,dict) for u in units)
            or len({u.get("unit_id") for u in units}) != len(units)):
        raise PlanConflict([finding("repair_ambiguous_unit", "Repair needs unique existing business units")])
    by_unit = {u["unit_id"]:u for u in units}
    changes, cohorts = change_set.get("changes",[]), change_set.get("cohort_changes",[])
    if any(not isinstance(rows,list) or len(rows)>64 for rows in (changes,cohorts)):
        raise PlanConflict([finding("repair_shape", "Each change array has at most 64 entries")])
    allowed = {"objects","relations","events","observations","obligations","publications","depends_on"}
    touched = set()
    for change in [dict(c,collection="cohorts") if isinstance(c,dict) else c for c in cohorts] + changes:
        if not isinstance(change,dict):
            raise PlanConflict([finding("repair_shape", "Every change must be an object")])
        collection, operation = change.get("collection"), change.get("operation")
        unit_id = change.get("unit_id")
        if collection == "cohorts":
            container = result
        elif collection in allowed and unit_id in by_unit:
            container = by_unit[unit_id]
        else:
            raise PlanConflict([finding("repair_target", "Unknown unit or forbidden collection", [str(unit_id),str(collection)])])
        key = change.get("key")
        selector = change.get('selector')
        identity_columns = {'observations': ('entity', 'field'), 'publications': ('fact_slot', 'channel', 'session')}
        if selector is None and collection in identity_columns and isinstance(key, str):
            # Older authors sometimes spell the field names in the natural key.
            # Resolve the explicit field/value pairs without guessing a target.
            parts = key.split('|')
            if len(parts) % 2 == 0 and len(set(parts[::2])) == len(parts)//2 and set(parts[::2]) <= set(identity_columns[collection]):
                selector = dict(zip(parts[::2], parts[1::2]))
        rows = container.setdefault(collection, [])
        if operation != 'add' and selector is not None:
            columns = identity_columns.get(collection, ('slot',) if collection in ('objects', 'relations', 'events') else ('id',))
            if not isinstance(selector, dict) or not selector or set(selector) - set(columns):
                raise PlanConflict([finding('repair_selector', 'Use the supplied identity fields to select one existing record', [str(unit_id)])])
            matches = [i for i, row in enumerate(rows) if all(str(row.get(k)) == str(v) for k, v in selector.items())]
            if len(matches) != 1:
                raise PlanConflict([finding('repair_target', 'Selector must identify exactly one existing record', [str(unit_id)], evidence={'collection': collection, 'selector': selector, 'matches': len(matches)})])
            key = _record_key(collection, rows[matches[0]])
        if operation == "add" and collection in ("observations","publications"):
            key = _record_key(collection,change.get("value"))
        identity = (unit_id,collection,key)
        if identity in touched:
            raise PlanConflict([finding("repair_duplicate_edit", "A target can be edited only once in a transaction", [str(identity)])])
        touched.add(identity)
        if collection == "depends_on":
            value = change.get("value")
            if operation != "replace" or not isinstance(value,list) or any(not isinstance(u,str) for u in value):
                raise PlanConflict([finding("repair_dependency", "Dependency repair requires a replacement id array")])
            container[collection] = deepcopy(value)
            continue
        if not isinstance(key,str) or not key:
            raise PlanConflict([finding("repair_target", "Record changes require a nonempty stable key")])
        matches = [i for i,row in enumerate(rows) if _record_key(collection,row) == key]
        if operation == "add":
            if matches:
                raise PlanConflict([finding("repair_add_exists", "Add cannot overwrite an existing record", [key])])
        elif operation not in ("replace","remove") or len(matches) != 1:
            raise PlanConflict([finding("repair_target", "Replace/remove needs one exact existing record", [key])])
        if operation == "remove":
            rows.pop(matches[0])
        else:
            value = deepcopy(change.get("value"))
            identity_field = "slot" if collection in ("objects","relations","events") else "id" if collection in ("obligations","cohorts") else None
            if isinstance(value,dict) and identity_field and identity_field not in value:
                value[identity_field] = key
            new_key = _record_key(collection,value)
            publication_move = (operation == 'replace' and collection == 'publications' and isinstance(value,dict)
                and all(value.get(k) == rows[matches[0]].get(k) for k in ('fact_slot','channel')))
            if new_key != key and not publication_move:
                raise PlanConflict([finding("repair_identity", "Replacement must retain the selected stable key", [key])])
            if publication_move and any(i != matches[0] and _record_key(collection,row) == new_key for i,row in enumerate(rows)):
                raise PlanConflict([finding('repair_duplicate_edit', 'Updated publication conflicts with another existing record', [new_key])])
            if operation == "add": rows.append(deepcopy(value))
            else: rows[matches[0]] = deepcopy(value)
    if digest(result) == digest(proposal):
        raise PlanConflict([finding("repair_no_change", "A rejected plan requires a concrete change before reassessment")])
    return result


def apply_repair(proposal, change_set):
    try:
        return _apply_repair(proposal,change_set)
    except (TypeError,KeyError,AttributeError) as error:
        raise PlanConflict([finding("repair_shape",str(error))]) from error


def enabled(wp):
    return (wp.get("supply_plan") or {}).get("instance_policy") in (VERSION, "business-instance-plan/v2")


def index(plan):
    units = {row["unit_id"]: row for row in plan["units"]}
    objects, events, relations, owners = {}, {}, {}, {}
    for unit_id, unit in units.items():
        for key, target in (("objects", objects), ("events", events), ("relations", relations)):
            for row in unit[key]:
                target[row["slot"]] = row
                owners[row["slot"]] = unit_id
    return units, objects, events, relations, owners


def execution_contract(wp):
    """Expose compiled identity and capability scope to the business reviewer.

    This is a projection of the accepted layout, not a second validator or a
    certificate for future content. Business requirements remain in the seed.
    """
    plan = wp.get('business_instance_plan')
    if not plan:
        return None
    units, objects, events, _relations, owners = index(plan)
    readers = {slot: set() for slot in objects}
    for uid, unit in units.items():
        refs = {r[k] for r in unit['relations'] for k in ('from', 'to')}
        refs.update(p for e in unit['events'] for p in e['participants'].values())
        refs.update(o['entity'] for o in unit['observations'])
        refs.update(e for row in unit['obligations'] for e in row['carrier']['entities'])
        for slot in refs:
            readers[slot].add(uid)
    bp = wp['world_blueprint']
    event_types = {e['id']: e for e in bp['event_types']}
    numeric_fields = {(t['id'], f['name']) for t in bp['entity_types']
                      for f in t['fields'] if f.get('kind') == 'numeric'}
    event_owned_numeric = {(event['roles'][effect['role']], effect['field'])
                           for event in bp['event_types'] for effect in event['effect_fields']
                           if (event['roles'][effect['role']], effect['field']) in numeric_fields}
    writes = {}
    for event in events.values():
        for effect in event_types[event['type']]['effect_fields']:
            slot, field = event['participants'][effect['role']], effect['field']
            if (objects[slot]['type'], field) in numeric_fields:
                writes.setdefault((slot, field), []).append({
                    'event': event['slot'], 'session': event['session'],
                    'writer_unit': owners[event['slot']]})
    obligations = [dict(deepcopy(row), unit_id=uid) for uid, unit in units.items()
                   for row in unit['obligations']]
    allocated = Counter(obj['type'] for obj in objects.values())
    return {'version': 'compiled-instance-execution/v2', 'plan_hash': plan['plan_hash'],
        'scope': 'Compiled slots and proposed schedule; actual values, world and question quality remain to be validated',
        'identity_semantics': 'Each slot has one definition. Identical slot references in different units share that object; units express writing ownership, not separate identity namespaces.',
        'declared_object_minima': {row['id']: row['count'] for row in bp['entity_types']},
        'allocated_object_counts': dict(allocated),
        'numeric_field_ownership': {
            'event_owned': [{'entity_type': kind, 'field': field}
                            for kind, field in sorted(event_owned_numeric)],
            'intrinsic_observable': [{'entity_type': kind, 'field': field}
                                     for kind, field in sorted(numeric_fields - event_owned_numeric)],
            'interpretation': 'Only event-owned fields require event effect writes for their changes. Intrinsic observable fields may receive genuine authored observations under the business rules; neither list certifies future values.'},
        'objects': [{'slot': slot, 'type': obj['type'], 'definition_unit': owners[slot],
                     'referencing_units': sorted(readers[slot])} for slot, obj in sorted(objects.items())],
        'capability_scope': {'allocation_unit': 'each explicit obligation and its carrier',
            'obligations': obligations, 'construction_conditions': deepcopy(plan.get('construction', [])),
            'business_requirements_source': 'Original seed/task; numerical question thresholds apply to their designated capability carriers'},
        'numeric_event_writes': [{'entity': slot, 'field': field, 'writes': rows}
                                for (slot, field), rows in sorted(writes.items())],
        'observations': [dict(deepcopy(o), writer_unit=uid) for uid, unit in units.items()
                         for o in unit['observations']]}


def _compile_plan(wp, proposal):
    """Check finite resources, bindings, temporal routes and demand together."""
    problems = []
    record_owners = {}
    def fail(code, message, ids=(), evidence=None):
        ids = list(ids)
        # A record-level failure must retain its writer even if later edits
        # removed the record's required fields. Never infer missing fact data.
        responsible = {owner for identity in ids if isinstance(identity, str)
                       for owner in record_owners.get(identity, ())}
        ids.extend(owner for owner in sorted(responsible) if owner not in ids)
        problems.append(finding(code, message, ids, evidence=evidence))
    if not isinstance(proposal, dict) or proposal.get("decision") != "plan":
        raise PlanConflict([finding("plan_shape", "Expected a complete plan decision and units")])
    raw_units = proposal.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise PlanConflict([finding("plan_shape", "units must be a nonempty array")])
    units, global_ids = [], set()
    forbidden = {"gt", "answer", "winner", "trend_direction", "numeric_values", "value"}
    def check_answers(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                fail("answer_in_plan", "Business layout must not predetermine answers", evidence=sorted(forbidden.intersection(value)))
            for child in value.values(): check_answers(child)
        elif isinstance(value, list):
            for child in value: check_answers(child)
    check_answers(proposal)
    for raw in raw_units:
        if not isinstance(raw, dict) or not isinstance(raw.get("unit_id"), str) or not raw["unit_id"]:
            fail("unit_shape", "Every unit needs a unique nonempty unit_id"); continue
        unit = deepcopy(raw)
        if not isinstance(unit.get("business_purpose"), str) or not unit["business_purpose"].strip():
            fail("unit_purpose", "Every unit needs a concrete business purpose", [unit["unit_id"]])
        for key in ("objects", "relations", "events", "observations", "obligations", "publications", "depends_on"):
            unit.setdefault(key, [])
            if not isinstance(unit[key], list):
                fail("unit_shape", key + " must be an array", [unit["unit_id"]]); unit[key] = []
        if not any(unit[key] for key in (
                "objects", "relations", "events", "observations", "obligations", "publications")):
            fail("empty_unit", "A business unit must add facts or declare an observation, obligation, or publication",
                 [unit["unit_id"]])
        record_fields = {
            "objects": {"slot": str, "type": str},
            "relations": {"slot": str, "type": str, "from": str, "to": str, "session": int},
            "events": {"slot": str, "type": str, "participants": dict, "session": int},
            "observations": {"entity": str, "field": str, "sessions": list},
            "obligations": {"id": str, "line": str, "carrier": dict},
            "publications": {"fact_slot": str, "channel": str, "session": int},
        }
        for collection, fields in record_fields.items():
            for position, row in enumerate(unit[collection]):
                ids = [unit["unit_id"]]
                if not isinstance(row, dict):
                    fail("record_shape", "Plan records must be objects", ids,
                         {"collection": collection, "record_index": position})
                    continue
                identity = row.get("slot", row.get("id"))
                if isinstance(identity, str) and identity:
                    ids.append(identity)
                    record_owners.setdefault(identity, set()).add(unit["unit_id"])
                missing = [key for key in fields if key not in row]
                if collection == "relations" and "session" in missing:
                    declaration = next((item for item in wp["world_blueprint"]["relation_types"]
                                        if item.get("id") == row.get("type")), None)
                    if declaration is not None and not declaration.get("temporal"):
                        # Static bindings already lower to session zero below;
                        # retain that original representation, without filling
                        # any authored event or temporal-relation fields.
                        missing.remove("session")
                invalid = {key: expected.__name__ for key, expected in fields.items()
                           if key in row and (type(row[key]) is not int if expected is int
                                              else not isinstance(row[key], expected))}
                if isinstance(row.get("participants"), dict) and any(
                        not isinstance(role, str) or not isinstance(ref, str)
                        for role, ref in row["participants"].items()):
                    invalid["participants"] = "object of role names to object slot strings"
                if isinstance(row.get("sessions"), list) and any(type(s) is not int for s in row["sessions"]):
                    invalid["sessions"] = "array of integer session indices"
                carrier = row.get("carrier")
                if isinstance(carrier, dict) and isinstance(carrier.get("entities"), list) and any(
                        not isinstance(ref, str) for ref in carrier["entities"]):
                    invalid["carrier.entities"] = "array of object slot strings"
                if missing or invalid:
                    fail("record_shape", "Plan record is incomplete or has invalid field types", ids,
                         {"collection": collection, "record_index": position,
                          "missing_fields": missing, "invalid_fields": invalid})
                    if collection in ("events", "relations") and ("session" in missing or "session" in invalid):
                        # Retain the established diagnostic consumed by callers;
                        # shape validation adds owner context, not a new meaning.
                        fail("session_out_of_range", "Sessions are zero-based integer indices within the frozen calendar", ids)
        if any(not isinstance(dependency, str) for dependency in unit["depends_on"]):
            fail("record_shape", "Dependencies must be an array of unit ID strings", [unit["unit_id"]],
                 {"collection": "depends_on", "invalid_fields": {"depends_on": "array of unit ID strings"}})
        for row in [unit] + unit["objects"] + unit["relations"] + unit["events"] + unit["obligations"]:
            if not isinstance(row, dict):
                continue  # The collection-specific finding above retains its unit.
            name = row.get("unit_id", row.get("slot", row.get("id")))
            if not isinstance(name, str) or not name or name in global_ids:
                fail("slot_identity", "All unit, fact and obligation identities must be distinct", [str(name)])
            else: global_ids.add(name)
        units.append(unit)
    if problems: raise PlanConflict(problems)
    plan = {"version": VERSION, "units": units, "cohorts": deepcopy(proposal.get("cohorts") or []),
        "reason": proposal.get("reason", ""), "requirements": deepcopy(wp["supply_plan"]["requirements"])}
    if proposal.get("execution_policy") == "shared-identities/v1":
        plan["execution_policy"] = proposal["execution_policy"]
    unit_map, objects, events, relations, owners = index(plan)
    bp = normalize_world_blueprint(wp)
    types = {t["id"]: t for t in bp["entity_types"]}
    etypes = {t["id"]: t for t in bp["event_types"]}
    rtypes = {t["id"]: t for t in bp["relation_types"]}
    periods = bp["temporal_model"]["n_sessions"]
    def session(row):
        value = row.get("session")
        if type(value) is not int or not 0 <= value < periods:
            fail("session_out_of_range", "Sessions are zero-based and within the frozen calendar", [row.get("slot", "?")])
            return None
        return value
    for slot, obj in objects.items():
        if obj.get("type") not in types:
            fail("unknown_entity_type", "Object type must exist in the original blueprint", [slot])
    counts = Counter(row.get("type") for row in objects.values())
    revised = deepcopy(wp)
    for row in revised["world_blueprint"]["entity_types"]:
        amount = counts[row["id"]]
        if amount < row["count"]:
            fail("missing_declared_objects", "Plan must allocate all declared object minima", [row["id"]], {"required":row["count"],"actual":amount})
        if row.get("cardinality_policy") == "exact" and amount != row["count"]:
            fail("exact_identity_count", "Original exact identities must retain their count", [row["id"]])
        row["count"] = max(amount, row["count"])
    limit = wp["supply_plan"].get("world_limits", {}).get("max_world_entities")
    if limit is not None and len(objects) > limit:
        fail("entity_cap", "Plan exceeds the original entity cap", evidence={"planned":len(objects),"cap":limit})
    writes, scalar_targets = {}, {}
    lowering = []
    for slot, rel in relations.items():
        spec = rtypes.get(rel.get("type"))
        if not spec:
            fail("unknown_relation", "Relation type is undeclared", [slot]); continue
        # Static bindings have no creation time in the original world schema.
        # Lower the layout into that representation once; publication time is
        # recorded independently in the public evidence obligations.
        if not spec.get("temporal") and rel.get("session") != 0:
            lowering.append({"slot": slot, "rule": "original_static_relation", "proposed_session": rel.get("session"), "canonical_session": 0})
            rel["session"] = 0
        sid = session(rel)
        for side in ("from", "to"):
            if objects.get(rel.get(side), {}).get("type") != spec[side+"_type"]:
                fail("relation_role_type", "Relation endpoint is missing or has the wrong type", [slot, str(rel.get(side))])
        owner_side = relation_owner_side(bp, spec)
        if owner_side:
            key = (rel[owner_side], spec["field"], sid if spec.get("temporal") else None)
            target = rel["to" if owner_side == "from" else "from"]
            field = next((f for f in types[spec[owner_side+"_type"]]["fields"] if f["name"] == spec["field"]), {})
            if field.get("kind") not in ("set", "list") and key in scalar_targets and scalar_targets[key] != target:
                fail("scalar_binding_conflict", "One scalar owner cannot bind multiple targets at this time", [slot], {"owner":key[0],"field":key[1]})
            scalar_targets[key] = target
    for slot, event in events.items():
        spec = etypes.get(event.get("type"))
        if not spec:
            fail("unknown_event", "Event type is undeclared", [slot]); continue
        sid = session(event)
        participants = event.get("participants")
        if not isinstance(participants, dict) or set(participants) != set(spec["roles"]):
            fail("event_roles", "Event must bind every declared role exactly", [slot]); continue
        for role, kind in spec["roles"].items():
            if objects.get(participants[role], {}).get("type") != kind:
                fail("event_role_type", "Event role is missing or has the wrong type", [slot,role])
        for effect in spec["effect_fields"]:
            key = (participants[effect["role"]], effect["field"], sid)
            if key in writes:
                fail("same_period_write", "Two events write the same object field in one period", [slot,writes[key]])
            writes[key] = slot
        parent = event.get("caused_by")
        if parent:
            prior = events.get(parent)
            if not prior:
                fail("causal_parent_missing", "Parent event slot is undeclared", [slot,str(parent)])
            else:
                applicable = [r for r in bp["causal_rules"] if r["trigger_event"] == prior.get("type") and r["effect_event"] == spec["id"]]
                if not applicable or sid is None or not isinstance(prior.get("session"), int) or not any(
                    sid-prior["session"] == r["delay_sessions"] and all(participants.get(role) == prior.get("participants", {}).get(role)
                    for role in r.get("shared_roles", [])) for r in applicable):
                    fail("causal_schedule", "Parent, exact delay and shared roles must satisfy the original causal rule", [slot,parent],
                        {"parent":deepcopy(prior),"effect":deepcopy(event),"applicable_rules":deepcopy(applicable),
                         "actual_delay_sessions":sid-prior["session"] if sid is not None and isinstance(prior.get("session"),int) else None})
    for obj_slot, obj in objects.items():
        for field in types.get(obj.get("type"), {}).get("fields", []):
            n = sum(key[:2] == (obj_slot, field["name"]) for key in writes)
            if field.get("states") and n > len(field["states"]):
                fail("state_write_capacity", "Event writes exceed the distinct one-way states available", [obj_slot,field["name"]])
    for group, instances in (("event_types", events), ("relation_types", relations)):
        actual = Counter(row.get("type") for row in instances.values())
        for row in revised["world_blueprint"][group]:
            if actual[row["id"]] < row.get("min_count",0):
                fail("missing_declared_instances", "Plan must cover original event/relation minima", [row["id"]], {"required":row.get("min_count",0),"actual":actual[row["id"]]})
            # The blueprint declares a minimum, while the executable plan owns
            # every exact fact slot. Repeated records of the same scalar link
            # do not raise its number of possible owners.
    # Invalid/unknown types already have structured findings. Counting must
    # never replace those findings with an unowned KeyError/plan_shape error.
    plan["instance_counts"]={"events":dict(Counter(e.get("type") for e in events.values())),
        "relations":dict(Counter(r.get("type") for r in relations.values()))}
    if not problems and wp['supply_plan'].get('instance_policy') == VERSION:
        from pipeline.construction import lower_observations
        plan,projection=lower_observations(plan,bp)
        units=plan['units']
        unit_map,objects,events,relations,owners=index(plan)
        if projection:plan['observation_lowering']=projection
    allocation = wp["supply_plan"]["candidate_allocation"]
    declared = Counter()
    for unit in units:
        refs = set()
        for rel in unit["relations"]: refs.update([rel["from"],rel["to"]])
        for event in unit["events"]:
            refs.update((event.get("participants") or {}).values())
            if event.get("caused_by"): refs.add(event["caused_by"])
        for observation in unit["observations"]:
            obj = objects.get(observation.get("entity"),{})
            field = next((f for f in types.get(obj.get("type"),{}).get("fields",[]) if f["name"] == observation.get("field")),None)
            sessions = observation.get("sessions")
            if not field or not isinstance(sessions,list) or not sessions or len(set(sessions)) != len(sessions) or any(type(s) is not int or not 0 <= s < periods for s in sessions):
                fail("observation_shape", "Observation needs a declared field and distinct valid periods", [unit["unit_id"]]); continue
            refs.add(observation["entity"])
            # Observations read the current state. The original world retains a
            # prior write until the next write; an observation is not a mutation.
            # Line-owned construction checks and the original value proof assess
            # how many real changes the requested question family needs.
        for obligation in unit["obligations"]:
            line = obligation.get("line")
            carrier = obligation.get("carrier")
            if line not in allocation or not isinstance(carrier,dict) or not isinstance(carrier.get("entities"),list) or not carrier["entities"] or any(ref not in objects for ref in carrier["entities"]):
                fail("obligation_carrier", "Every requested obligation needs declared supporting object slots", [obligation["id"]]); continue
            declared[line] += 1
            refs.update(carrier["entities"])
            if wp['supply_plan'].get('instance_policy') != VERSION and line == "L7_consolidation" and obligation.get("subtype") == "native_trend":
                from pipeline.lines.L7_consolidation import T_MIN
                matches = [o for u in units for o in u["observations"] if o.get("entity") in carrier["entities"] and o.get("field") == carrier.get("field")]
                field = next((f for f in types[objects[carrier["entities"][0]]["type"]]["fields"] if f["name"] == carrier.get("field")),{})
                if field.get("kind") != "numeric" or field.get("monotonic") in ("up","down") or not any(len(set(o.get("sessions",[]))) >= T_MIN for o in matches):
                    fail("native_trend_plan", "Native trend needs an applicable numeric carrier and original distinct-period requirement", [obligation["id"]])
        for publication in unit["publications"]:
            if (not isinstance(publication,dict) or publication.get("fact_slot") not in owners
                    or publication.get("channel") not in bp["evidence_channels"]
                    or type(publication.get("session")) is not int or not 0 <= publication["session"] < periods):
                fail("publication_shape", "Public obligation needs an existing fact, channel and valid period", [unit["unit_id"]]); continue
            ref = publication["fact_slot"]
            refs.add(ref)
            occurred = (events.get(ref) or relations.get(ref) or {}).get("session",0)
            if publication["session"] < occurred:
                fail("publication_before_fact", "Publication cannot precede the allocated fact", [ref])
        dependencies = set(unit["depends_on"])
        if plan.get("execution_policy") == "shared-identities/v1":
            refs = {e["caused_by"] for e in unit["events"] if e.get("caused_by")}
        required = {owners[ref] for ref in refs if ref in owners and owners[ref] != unit["unit_id"]}
        if required - dependencies or dependencies - set(unit_map) or unit["unit_id"] in dependencies:
            fail("unit_dependencies", "Declare every cross-unit dependency and no unknown/self dependency", [unit["unit_id"]], {"required":sorted(required),"declared":sorted(dependencies)})
    for line, amount in allocation.items():
        if declared[line] != amount:
            fail("demand_slots", "Obligation count must equal the frozen per-line reserve", [line], {"required":amount,"actual":declared[line]})
    for requirement in plan["requirements"]:
        for subtype,minimum in requirement.get("subtype_minimum",{}).items():
            actual = sum(o.get("line") == requirement["line"] and o.get("subtype") == subtype for u in units for o in u["obligations"])
            if actual < minimum:
                fail("subtype_slots", "Plan must cover the frozen capability subtypes", [requirement["line"],subtype], {"required":minimum,"actual":actual})
    remaining = set(unit_map); ready = set()
    while remaining:
        batch = {u for u in remaining if set(unit_map[u]["depends_on"]) <= ready}
        if not batch:
            fail("dependency_cycle", "Business units contain a dependency cycle", sorted(remaining)); break
        ready.update(batch); remaining -= batch
    for cohort in plan["cohorts"]:
        members = cohort.get("members") if isinstance(cohort,dict) else None
        if not isinstance(members,list) or len(set(members)) < 2 or any(m not in objects for m in members) or not cohort.get("basis") or not isinstance(cohort.get("id"),str) or not cohort["id"]:
            fail("cohort_shape", "Comparison groups need distinct declared members and a prior business basis")
    if problems: raise PlanConflict(problems)
    # Project symbolic identities through the original relation assembler.
    # Intrinsic values belong to the author phase; only relation timelines
    # participate here. Later units can overwrite earlier scalar relations.
    from pipeline.world_state import assemble_world, structural_projection_findings
    from pipeline.seed_world import event_relation_binding_findings
    relation_table = {
        "entities": [{"name": slot, "type": row["type"], "fields": {}} for slot, row in objects.items()],
        "relations": [{"id": slot, **{k: row[k] for k in ("type", "from", "to", "session")}}
                      for slot, row in relations.items()], "events": []}
    projected, value_issues = assemble_world(relation_table,
        blueprint=bp, require_complete=False, include_shape_diagnostics=False)
    for rejected in structural_projection_findings(relation_table, projected, value_issues):
        row = rejected["evidence"]["proposed"]
        fail("relation_compilation", "The original compiler did not retain this planned relationship",
            [row["id"]], dict(rejected["evidence"], responsible_unit=owners[row["id"]]))
    valid_relations = {}
    for row in projected.relations:
        valid_relations.setdefault(row["type"], []).append(row["id"])
    for slot, event in events.items():
        for problem in event_relation_binding_findings(projected, bp, dict(event, id=slot), valid_relations):
            fail("relation_schedule", problem["message"], [slot, problem["relation"]], problem)
    # Relation projection and line construction read independent parts of the
    # already shape-checked plan. Report both sets of findings in one revision
    # so repairing a relationship does not reveal a previously hidden line
    # deficit only after the next paid author call.
    relation_issues = deepcopy(problems)
    revised["world_blueprint"] = normalize_world_blueprint(revised)
    if wp["supply_plan"].get("instance_policy") == VERSION:
        from pipeline.construction import VERSION as CONSTRUCTION_VERSION, compile_rows
        rows, construction_issues = compile_rows(plan, revised["world_blueprint"])
        from pipeline.lines.L7_consolidation import ConsolidationLine
        cohort_issues = []
        if not relation_issues:
            natural = ConsolidationLine._natural_cohorts(projected)
            compared={frozenset(o['carrier']['entities']) for u in units for o in u['obligations']
                if o['line']=='L7_consolidation' and o.get('subtype') in ('compare','comparison')}
            plan['inactive_cohort_ids']=[c['id'] for c in plan['cohorts'] if frozenset(c['members']) not in compared]
            for cohort in plan["cohorts"]:
                if cohort['id'] in plan['inactive_cohort_ids']:continue
                if frozenset(cohort["members"]) not in {frozenset(c["members"]) for c in natural}:
                    cohort_issues.append(finding("cohort_layout", "Use the actual natural partition before authoring values", [cohort["id"]],
                        evidence={"declared":cohort["members"], "natural":natural}))
        if relation_issues or construction_issues or cohort_issues:
            raise PlanConflict([*relation_issues, *construction_issues, *cohort_issues])
        plan.update(construction_version=CONSTRUCTION_VERSION, construction=rows)
    elif relation_issues:
        raise PlanConflict(relation_issues)
    from pipeline.supply_capacity import blueprint_capacity_report
    capacity = blueprint_capacity_report(revised["world_blueprint"],revised["supply_plan"])
    if capacity["issues"]:
        raise PlanConflict([finding("capacity_bound","Compiled layout remains below a necessary capacity bound",evidence=capacity["issues"])])
    from pipeline.seed_pack import validate_seed_blueprint
    if revised.get("seed_contract"):
        seed = validate_seed_blueprint(revised,revised["seed_contract"])
        if not seed["passed"]: raise PlanConflict([finding("seed_constraint", "Plan violates the original seed", evidence=seed["issues"])])
        revised["seed_audit"] = seed
    revised.setdefault("shared_world_spec",{}).setdefault("entities",{})["count"] = sum(t["count"] for t in revised["world_blueprint"]["entity_types"])
    plan["lowering"] = lowering
    plan["blueprint_hash"] = digest(revised["world_blueprint"])
    plan["plan_hash"] = digest(plan)
    revised["business_instance_plan"] = plan
    return revised


def compile_plan(wp, proposal):
    try:
        return _compile_plan(wp, proposal)
    except PlanConflict:
        raise
    except WorldBlueprintError as error:
        raise PlanConflict([finding("blueprint_compilation", str(error))]) from error
    except (TypeError, KeyError, AttributeError) as error:
        raise PlanConflict([finding("plan_shape", str(error))]) from error


def active_mapping(state):
    result = deepcopy(state.get("identity_registry",{}))
    for unit in state["units"]:
        for slot, actual in unit.get("instance_mapping", {}).items():
            if slot in result and result[slot] != actual:
                raise PlanConflict([finding("mapping_collision", "Accepted slot mapping changed", [slot])])
            result[slot] = actual
    return result


def publication_obligations(wp,state):
    mappings = active_mapping(state)
    return {"plan_hash":wp["business_instance_plan"]["plan_hash"],"rows":[
        {**deepcopy(row),"actual_fact_id":mappings.get(row["fact_slot"]),"unit_id":unit["unit_id"]}
        for unit in wp["business_instance_plan"]["units"] for row in unit["publications"]]}


def progress(wp, state):
    unit_map, _, _, _, _ = index(wp["business_instance_plan"])
    done = {unit for accepted in state["units"] for unit in accepted.get("instance_units", [])}
    return {"completed_units": sorted(done), "ready_units": [unit for unit, row in unit_map.items()
        if unit not in done and set(row["depends_on"]) <= done],
        "pending_units": [unit for unit in unit_map if unit not in done], "slot_mapping": active_mapping(state)}


def fulfillment(wp, world, state):
    """Certify actual independent supply; keep the proposed carriers as a report.

    The original complete world compiler and seed gate run before this entry.
    Planned carriers explain the author's intent, but valid alternatives from
    the same original line may realize demand without rewriting that intent.
    """
    from pipeline.capability_contract import actual_capacity
    plan = wp["business_instance_plan"]
    status = progress(wp,state)
    mappings = status["slot_mapping"]
    from pipeline.construction import enabled as construction_enabled, resolve
    if construction_enabled(wp):
        world = deepcopy(world)
        world.supply_construction = resolve(wp, mappings)
    capacity = actual_capacity(wp,world)
    problems = []
    if status["pending_units"]:
        problems.append(finding("unrealized_units","Planned business units remain unfinished",status["pending_units"]))
    obligations = [row for unit in plan["units"] for row in unit["obligations"]]
    certificates = capacity["certificates"]
    from pipeline.lines.L7_consolidation import ConsolidationLine
    natural = {frozenset(row["members"]) for row in ConsolidationLine._natural_cohorts(world)}
    declared_cohorts = {frozenset(mappings.get(ref) for ref in row["members"]) for row in plan["cohorts"]}
    for row in plan["cohorts"]:
        if frozenset(mappings.get(ref) for ref in row["members"]) not in natural:
            problems.append(finding("cohort_unrealized","The prior comparison group differs from the original natural partition",[row["id"]]))
    candidates = {}
    for obligation in obligations:
        carrier = obligation["carrier"]
        entities = {mappings.get(ref) for ref in carrier["entities"]}
        choices = []
        for i,certificate in enumerate(certificates):
            support = certificate["support"]
            from pipeline.lines import line_for
            if certificate["line"] != obligation["line"] or not line_for(obligation["line"]).matches_carrier(carrier, entities, support):
                continue
            if obligation.get("subtype") == "native_trend" and (support.get("aux") or {}).get("sub") != "S1_trend":
                continue
            if obligation.get("subtype") == "compare" and (support.get("aux") or {}).get("sub") != "S2_compare":
                continue
            if obligation.get("subtype") == "compare" and frozenset(((support.get("aux") or {}).get("supply_cohort") or {}).get("members",[])) not in declared_cohorts:
                continue
            choices.append(i)
        candidates[obligation["id"]] = choices
    assigned = {}
    def allocate(slot,visited):
        for i in candidates[slot]:
            if i in visited: continue
            visited.add(i)
            if i not in assigned or allocate(assigned[i],visited):
                assigned[i] = slot
                return True
        return False
    for obligation in sorted(obligations,key=lambda row:len(candidates[row["id"]])):
        if not allocate(obligation["id"],set()):
            problems.append(finding("obligation_unrealized", "No distinct original validated order realizes this obligation",[obligation["id"]],evidence=obligation))
    for unit in plan["units"]:
        for observation in unit["observations"]:
            entity = mappings.get(observation["entity"])
            timeline = world.entities.get(entity,{}).get(observation["field"])
            from pipeline.world_state import INSUFFICIENT, INVALID
            periods = {s for s in observation['sessions'] if timeline and
                       timeline.value_at_session(s) not in (INSUFFICIENT, INVALID, None)}
            if not set(observation["sessions"]) <= periods:
                problems.append(finding("observation_unrealized", "Actual observations do not cover the planned periods",[observation["entity"],observation["field"]],evidence={"actual":sorted(periods),"required":observation["sessions"]}))
    if construction_enabled(wp):
        from pipeline.construction import value_findings
        problems.extend(value_findings(wp,world,mappings))
    planned = {"passed":not problems,"findings":problems,
        "assignments":{slot:certificates[i]["family_key"] for i,slot in assigned.items()}}
    hard = [item for item in problems if item["code"] == "unrealized_units"]
    allocation = wp["supply_plan"]["candidate_allocation"]
    witnesses = {}
    deficits = set()
    bounded = {row["line"]: row["pool_at_limit"] for row in
               capacity["original_statistics"]["lines"]}
    for line, amount in allocation.items():
        available = {c["family_key"]:c for c in certificates if c["line"] == line}
        witnesses[line] = {"required":amount,"available":len(available),
            "family_keys":list(available)[:amount],"passed":len(available) >= amount,
            "pool_at_limit":bool(bounded.get(line))}
        if len(available) < amount:
            deficits.add(line)
            inconclusive = bool(bounded.get(line))
            hard.append(finding("actual_supply_search_inconclusive" if inconclusive else "actual_supply_shortfall",
                "The bounded original pool cannot establish a shortage" if inconclusive else
                "The original validated independent supply is below this line's reserve",
                [line], evidence={"required":amount,"actual":len(available),"substitution_from_other_lines":False}))
    l7 = "L7_consolidation"
    minimum = (max([1] + [row.get("subtype_minimum", {}).get("native_trend", 0)
        for row in wp["supply_plan"].get("requirements", []) if row["line"] == l7])
        if allocation.get(l7) else 0)
    native = {c["family_key"] for c in certificates if c["line"] == l7
        and (c["support"].get("aux") or {}).get("sub") == "S1_trend"}
    if len(native) < minimum:
        deficits.add(l7)
        inconclusive = bool(bounded.get(l7))
        hard.append(finding("actual_native_trend_search_inconclusive" if inconclusive else "actual_native_trend_shortfall",
            "The bounded original pool cannot establish native trend shortage" if inconclusive else
            "Original native trends cannot be replaced by comparison questions",
            [l7], evidence={"required":minimum,"actual":len(native)}))
    # Existing value transactions remain available when actual supply is short.
    # An unsuccessful named carrier alone does not trigger more production.
    for item in problems:
        if item.get("retry_group") == "unit_values":
            row = next((r for r in plan.get("construction", []) if r["id"] in item["affected_ids"]), None)
            if row and row["line"] in deficits:
                hard.append(item)
    return {"version":VERSION,"verification_policy":FULFILLMENT_POLICY,
        "plan_hash":plan["plan_hash"],"world_hash":capacity["world_hash"],
        "passed":not hard,"findings":hard,"assignments":planned["assignments"],
        "planned_fulfillment":planned,"actual_supply":witnesses,
        "native_trend":{"required":minimum,"actual":len(native),"family_keys":sorted(native)},
        "capacity":capacity,"progress":status}


def create(wp, tracer, feedback, audit, *, max_attempts=3, initial_proposal=None, repair_required=False, resource_resume=None, mechanical_attempts=3, initial_business_revision=None, checkpoint=None, max_joint_batches=None):
    """Persist the same bounded transaction on normal return and every failure."""
    try:
        return _create(wp,tracer,feedback,audit,max_attempts=max_attempts,initial_proposal=initial_proposal,
            repair_required=repair_required,resource_resume=resource_resume,mechanical_attempts=mechanical_attempts,
            initial_business_revision=initial_business_revision,checkpoint=checkpoint,
            max_joint_batches=max_joint_batches)
    finally:
        if checkpoint is not None:checkpoint(audit)


def _create(wp, tracer, feedback, audit, *, max_attempts=3, initial_proposal=None, repair_required=False, resource_resume=None, mechanical_attempts=3, initial_business_revision=None, checkpoint=None, max_joint_batches=None):
    """One bounded proposal transaction; mechanical and business repair share it."""
    from pipeline.blueprint_feasibility import assess
    if not 1 <= mechanical_attempts <= 3:
        raise ValueError('Mechanical attempt allowance must be within the original bounded limit')
    if audit.get('attempts') or audit.get('transactions') or audit.get('business_revisions'):
        raise PlanConflict([finding('plan_recovery_required','Existing plan attempts require explicit recovery; author allowances cannot restart',kind='design')])
    def save():
        if checkpoint is not None:checkpoint(audit)
    def reviewed(revised, target, review_feedback):
        call={'status':'requested'};target['business_review_call']=call;save()
        try:
            result=assess(revised,tracer,feedback=review_feedback)
        except BaseException as error:
            call.update(status='call_failed',error_type=type(error).__name__);save();raise
        call.update(status='response_received',response=deepcopy(result));save()
        return result
    audit.update(status="planning", before_hash=digest(wp), attempts=[]);save()
    current_feedback = deepcopy(feedback)
    proposal = deepcopy(initial_proposal)
    if proposal is not None:
        audit["recovery_proposal_hash"] = digest(proposal)
        if repair_required and any(f.get('kind')=='business' for f in feedback.get('findings',[])):
            from pipeline.layout_revision import revise
            return revise(wp,proposal,feedback,tracer,audit,max_attempts=max_attempts,initial_revision=initial_business_revision,checkpoint=checkpoint)
        try:
            revised = compile_plan(wp,proposal)
        except PlanConflict as error:
            if max_attempts == 0:
                audit.update(status="restored_structure_failed",findings=deepcopy(error.findings))
                raise
            current_feedback = {"original":feedback,"rejected_plan":proposal,"findings":error.findings}
        else:
            if repair_required:
                current_feedback = {"original":feedback,"rejected_plan":proposal,
                    "findings":deepcopy(feedback.get("findings", []))}
            else:
                review = reviewed(revised,audit,{"business_instance_plan":revised["business_instance_plan"],
                    "original_request":feedback}) if revised.get("seed_contract") else {"decision":"accept"}
                audit["restored_business_review"] = deepcopy(review)
                if review["decision"] == "unresolved":
                    audit["status"] = "design_unresolved"
                    raise PlanConflict([finding("business_unresolved","Business review unresolved",kind="design",evidence=review)])
                if review["decision"] == "accept":
                    audit.update(status="accepted",after_hash=digest(revised),after=deepcopy(revised))
                    return revised
                current_feedback = {"original":feedback,"rejected_plan":proposal,
                    "findings":[finding("business_repair","Original business review requests revision",kind="business",evidence=review)]}
            if max_attempts == 0:
                audit["status"] = "business_repair_required"
                raise PlanConflict(current_feedback["findings"])
    for number in range(1,max_attempts+1):
        if proposal is not None and any(f.get('kind')=='business' for f in current_feedback.get('findings',[])):
            from pipeline.layout_revision import revise
            return revise(wp,proposal,current_feedback,tracer,audit,max_attempts=max_attempts-number+1,checkpoint=checkpoint)
        payload = {"business":wp.get("seed_contract"), "scenario":wp.get("scenario"),
            "blueprint":wp["world_blueprint"], "requirements":wp["supply_plan"]["requirements"],
            "limits":wp["supply_plan"]["world_limits"], "feedback":current_feedback}
        repairing = proposal is not None
        scoped_candidate = None
        scoped=repairing and len(proposal.get("units", [])) >= 3
        attempt = {"attempt":number,"request_hash":digest(payload),'status':'requested',
            'request_step':'scoped_unit_transaction' if scoped else 'council.instance_plan_repair' if repairing else 'council.instance_plan'}
        if repairing:attempt['before_proposal_hash']=digest(proposal)
        audit['attempts'].append(attempt);save()
        if scoped:
            from pipeline.plan_transactions import repair
            # One bounded compiler transaction owns author attempts. The legacy
            # outer mechanical loop must not reset those counters on failure.
            transaction={"mechanical_attempt":1};audit.setdefault('transactions',[]).append(transaction)
            save()
            saved_resource=resource_resume;resource_resume=None
            try:
                scoped_candidate=repair(wp,proposal,current_feedback,tracer,transaction,
                    resource_resume=saved_resource,checkpoint=lambda _transaction:save(),
                    max_joint_batches=max_joint_batches)
            except PlanConflict as error:
                transaction['findings']=deepcopy(error.findings)
                audit.update(status='mechanical_recovery_required',findings=deepcopy(error.findings))
                attempt.update(status='transaction_rejected',findings=deepcopy(error.findings));save()
                raise
            except BaseException as error:
                attempt.update(status='transaction_failed',error_type=type(error).__name__);save();raise
            raw = {"decision":"repair", "transaction":transaction}
        else:
            try:
                raw = tracer.chat_json("council.instance_plan_repair" if repairing else "council.instance_plan", [{"role":"system","content":REPAIR_SYSTEM if repairing else SYSTEM},
                    {"role":"user","content":json.dumps(payload,ensure_ascii=False)}], model=config.STRUCTURE_MODEL,
                    temperature=.3,max_tokens=32768,retries=3,response_format={"type":"json_object"})
            except BaseException as error:
                attempt.update(status='call_failed',error_type=type(error).__name__);save();raise
        attempt.update(status='response_received',response=deepcopy(raw));save()
        if repairing: attempt.update(change_set=deepcopy(raw),before_proposal_hash=digest(proposal))
        if isinstance(raw,dict) and "__error__" in raw:
            audit["status"] = "provider_error"
            attempt['status']='provider_error';save()
            from execution_control import global_stop_exception
            stopped=global_stop_exception(raw)
            if stopped is not None:raise stopped
            raise RuntimeError("Instance plan provider failure: " + str(raw["__error__"]))
        if isinstance(raw,dict) and raw.get("decision") == "unresolved":
            attempt["findings"] = [finding("plan_unresolved",str(raw.get("reason")),kind="design")]
            audit["status"] = "design_unresolved"
            attempt['status']='unresolved';save()
            raise PlanConflict(attempt["findings"])
        if repairing:
            try:
                candidate = scoped_candidate if scoped_candidate is not None else apply_repair(proposal,raw)
            except PlanConflict as error:
                attempt.update(proposal=deepcopy(proposal),findings=error.findings)
                current_feedback = {**current_feedback,"rejected_change_set":raw,"repair_error":error.findings}
                attempt['status']='repair_rejected';save()
                continue
            proposal = candidate
        else:
            proposal = deepcopy(raw)
        attempt["proposal"] = deepcopy(proposal)
        save()
        try:
            revised = compile_plan(wp,proposal)
        except PlanConflict as error:
            attempt["findings"] = error.findings
            current_feedback = {"original":feedback,"rejected_plan":proposal,"findings":error.findings}
            attempt['status']='compiler_rejected';save()
            continue
        except WorldBlueprintError as error:
            attempt["findings"] = [finding("blueprint_compilation",str(error))]
            current_feedback = {"original":feedback,"rejected_plan":proposal,"findings":attempt["findings"]}
            attempt['status']='compiler_rejected';save()
            continue
        review = reviewed(revised,attempt,{"business_instance_plan":revised["business_instance_plan"],"original_request":feedback}) if revised.get("seed_contract") else {"decision":"accept"}
        attempt["business_review"] = deepcopy(review)
        if review["decision"] == "unresolved":
            audit["status"] = "design_unresolved"
            raise PlanConflict([finding("business_unresolved","Business review unresolved",kind="design",evidence=review)])
        if review["decision"] != "accept":
            attempt["findings"] = [finding("business_repair","Original business review requests revision",kind="business",evidence=review)]
            current_feedback = {"original":feedback,"rejected_plan":proposal,"findings":attempt["findings"]}
            continue
        audit.update(status="accepted",after_hash=digest(revised),after=deepcopy(revised))
        return revised
    audit["status"] = "attempts_exhausted"
    raise PlanConflict([finding("plan_attempts_exhausted","Business plan repair allowance exhausted",evidence=audit["attempts"][-1].get("findings"))])
