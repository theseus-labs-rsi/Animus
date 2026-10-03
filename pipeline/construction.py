"""Compile and route line-owned construction conditions; keep facts in the world."""
from copy import deepcopy

VERSION = "line-construction/v1"


def enabled(wp):
    return (wp.get("business_instance_plan") or {}).get("construction_version") == VERSION


def lower_observations(proposal, blueprint):
    """Compile sampling from line contracts without creating values or events.

    Event-owned fields use only their existing planned write periods.
    The returned provenance makes every derived schedule visible to business
    review and later actual-history validation.
    """
    from pipeline.instance_plan import index
    from pipeline.world_agent import _context
    from pipeline.lines import line_for
    plan=deepcopy(proposal)
    units,objects,events,_,owners=index(plan)
    ownership=_context({'world_blueprint':blueprint},blueprint)['field_write_routes']
    provenance=[]
    for unit in plan['units']:
        for obligation in unit['obligations']:
            carrier={**obligation['carrier'],'subtype':obligation.get('subtype')}
            for observation in line_for(obligation['line']).planned_observations(carrier,blueprint,objects,events,ownership):
                minimum=observation.pop('minimum_points')
                existing=[o for u in units.values() for o in u['observations']
                    if (o.get('entity'),o.get('field'))==(observation['entity'],observation['field'])]
                actual={s for o in existing for s in o.get('sessions',[])}
                if len(actual)>=minimum:continue
                owner=units[owners[observation['entity']]]
                own=next((o for o in owner['observations'] if (o.get('entity'),o.get('field'))==(observation['entity'],observation['field'])),None)
                if own is None:owner['observations'].append(deepcopy(observation))
                else:own['sessions']=sorted(set(own['sessions'])|set(observation['sessions']))
                provenance.append({'obligation_id':obligation['id'],'line':obligation['line'],
                    'writer_unit':owner['unit_id'],'entity':observation['entity'],'field':observation['field'],
                    'original_observation_periods':sorted(actual),'compiled_observation_periods':observation['sessions'],
                    'construction_version':VERSION,'facts_created':False})
    return plan,provenance


def compile_rows(plan, blueprint):
    from pipeline.instance_plan import index
    from pipeline.lines import line_for
    _units, objects, events, _relations, _owners = index(plan)
    observations = [o for u in plan["units"] for o in u["observations"]]
    rows, issues, seen = [], [], set()
    for unit in plan["units"]:
        for obligation in unit["obligations"]:
            carrier = deepcopy(obligation["carrier"])
            carrier["subtype"] = obligation.get("subtype")
            if carrier["subtype"] == "comparison":
                carrier["subtype"] = obligation["subtype"] = "compare"
            line = line_for(obligation["line"])
            local = line.carrier_field_issues(carrier, blueprint, objects)
            local.extend(line.construction_issues(carrier, blueprint, objects, events, observations))
            key = (line.id, tuple(sorted(carrier["entities"])), carrier.get("field"), carrier["subtype"])
            if key in seen:
                local.append({"code": "duplicate_construction", "message": "Allocate distinct factual scopes; repeated demand ids do not create new candidates"})
            seen.add(key)
            issues.extend({"kind": "structure", "code": p["code"], "message": p["message"],
                "affected_ids": [obligation["id"]], "evidence": p, "retry_group": "business_plan",
                "allowed_edit_scope": "uncommitted_plan_with_original_seed_and_limits"} for p in local)
            rows.append({"id": obligation["id"], "line": line.id, "unit_id": unit["unit_id"],
                         "carrier": carrier, "conditions": line.construction_spec()})
    return rows, issues


def resolve(wp, mappings):
    rows = []
    for row in wp["business_instance_plan"].get("construction", []):
        if all(e in mappings for e in row["carrier"]["entities"]):
            item = deepcopy(row)
            item["carrier"]["entities"] = [mappings[e] for e in row["carrier"]["entities"]]
            rows.append(item)
    return {"version": VERSION, "rows": rows}


def writer_units(wp, row):
    """Route field requirements to the tasks that actually supply their values."""
    entities, field = set(row['carrier']['entities']), row['carrier'].get('field')
    event_types = {e['id']:e for e in wp['world_blueprint']['event_types']}
    result = []
    for unit in wp['business_instance_plan']['units']:
        owns = any(o['slot'] in entities for o in unit['objects'])
        writes = any(e['participants'][effect['role']] in entities and
                     (not field or effect['field'] == field)
                     for e in unit['events'] for effect in event_types[e['type']]['effect_fields'])
        if owns or writes:
            result.append(unit['unit_id'])
    return result or [row['unit_id']]


def author_requirements(wp, unit_id):
    result = []
    for row in wp['business_instance_plan'].get('construction', []):
        if unit_id in writer_units(wp, row):
            item = deepcopy(row)
            item['planned_observations'] = [deepcopy(o) for u in wp['business_instance_plan']['units'] for o in u['observations']
                if o['entity'] in row['carrier']['entities'] and (not row['carrier'].get('field') or o['field'] == row['carrier']['field'])]
            result.append(item)
    return result


def value_findings(wp, world, mappings):
    from pipeline.lines import line_for
    findings = []
    for row in resolve(wp, mappings)["rows"]:
        for issue in line_for(row["line"]).value_issues(world, row["carrier"]):
            original = next(r for r in wp['business_instance_plan']['construction'] if r['id'] == row['id'])
            writers = writer_units(wp, original)
            findings.append({**issue, "kind": "value", "affected_ids": [row["id"]],
                "unit_id": writers[0], "responsible_units": writers, "retry_group": "unit_values", "carrier": row["carrier"]})
    return findings


def compact_feedback(evidence):
    """Send failure ownership and relevant proof, not the whole candidate pool."""
    result = deepcopy({k: v for k, v in evidence.items() if k not in ("proof", "instance_fulfillment")})
    proof = evidence.get("proof") or evidence.get("instance_fulfillment") or {}
    if proof:
        # A local compiler rejection and the overall shortage are different
        # evidence. Keep both so the designer sees why the value author stopped.
        findings = result.setdefault("findings", [])
        findings.extend(deepcopy(row) for row in proof.get("findings", []) if row not in findings)
        result["world_hash"] = proof.get("world_hash")
        result["capacity_by_line"] = deepcopy(proof.get("capacity", {}).get("original_statistics", {}).get("lines", []))
    return result
