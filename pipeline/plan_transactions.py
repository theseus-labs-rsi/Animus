"""Separate resource allocation from bounded, parallel fact-plan edits.

Every model result is an uncommitted proposal. A complete merged transaction
still passes the original joint compiler and business review in instance_plan.
"""
from collections import Counter
from copy import deepcopy
import json
from threading import RLock

import config

RETRY_POLICY_V2 = 'separate-author-protocol/v2'


def build_retry_policy_migration(audit, pairs):
    """Bind an explicit, proven historical encoding correction; never erase rows."""
    from pipeline.capability_contract import digest
    from pipeline.instance_plan import PlanConflict, finding
    def reject(message):raise PlanConflict([finding('retry_policy_migration',message)])
    if not isinstance(pairs,list):reject('Historical protocol pairs must be an explicit list')
    bindings=[];used=set()
    for pair in pairs:
        if not isinstance(pair,dict) or set(pair)!={'unit_id','first_attempt','second_attempt'}:
            reject('Historical protocol pair requires exact unit and attempt indices')
        uid=pair['unit_id']; first=pair['first_attempt'];second=pair['second_attempt']
        if not isinstance(uid,str) or type(first) is not int or type(second) is not int or first<1 or second!=first+1:
            reject('Historical protocol pair must be consecutive original requests')
        rows=audit.get('edit_attempts',{}).get(uid,[])
        if second>len(rows) or any((uid,n) in used for n in (first,second)):reject('Historical protocol pairs overlap or exceed saved history')
        left,right=rows[first-1],rows[second-1]
        batch=left.get('batch_iteration')
        drafts=audit.get('drafts',[])
        if type(batch) is not int or batch<2 or right.get('batch_iteration')!=batch or len(drafts)<batch-1:
            reject('Historical correction must bind the same saved batch and its prior candidate')
        batch_base=drafts[batch-2]['candidate']
        units=[u for u in batch_base['units'] if u['unit_id']==uid]
        if len(units)!=1:reject('Historical selector has no unique original unit')
        if (left.get('status')!='locally_rejected' or right.get('status') not in ('locally_compiled','joint_rejected')
                or {f.get('code') for f in left.get('findings',[])}!={'repair_shape'}):
            reject('Historical pair must bind a shape rejection and its complete response')
        a=deepcopy(left.get('proposal'));b=deepcopy(right.get('proposal'))
        if not isinstance(a,dict) or not isinstance(b,dict) or a.get('decision')!='repair' or b.get('decision')!='repair':
            reject('Historical pair requires complete original repair responses')
        a.pop('reason',None);b.pop('reason',None);fixed=0
        if not isinstance(a.get('changes'),list):reject('Historical correction has no edit array')
        for c in a['changes']:
            key=c.get('key') if isinstance(c,dict) else None
            if isinstance(key,dict):
                if (c.get('collection')!='observations' or c.get('operation') not in ('replace','remove')
                        or set(key)!={'entity','field'} or not all(isinstance(v,str) and v for v in key.values())
                        or c.get('unit_id')!=uid or 'selector' in c):reject('Historical correction exceeds selector encoding')
                originals=[o for o in units[0]['observations'] if o.get('entity')==key['entity'] and o.get('field')==key['field']]
                if len(originals)!=1:reject('Historical selector is not unique in the original edit catalog')
                original_selector_sha=digest(originals[0])
                # This is a proof projection only; neither response nor candidate is edited.
                c['key']=key['entity']+'|'+key['field'];fixed+=1
        if fixed!=1 or a!=b:reject('Historical correction changes business content or is not exactly one selector encoding')
        used.update((uid,n) for n in (first,second))
        bindings.append(dict(pair,first_row_sha256=digest(left),second_row_sha256=digest(right),
            first_proposal_sha256=digest(left['proposal']),second_proposal_sha256=digest(right['proposal']),
            batch_iteration=batch,batch_base_sha256=digest(batch_base),selector_original_record_sha256=original_selector_sha,
            equivalence='one_observation_selector_encoding_only'))
    return {'version':RETRY_POLICY_V2,'max_semantic_attempts':3,'max_protocol_attempts':3,'max_json_attempts':3,
        'legacy_attempt_prefix':{uid:[digest(row) for row in rows] for uid,rows in audit.get('edit_attempts',{}).items()},
        'legacy_draft_prefix':[digest(d) for d in audit.get('drafts',[])],
        'historical_protocol_pairs':bindings}


def retry_policy_counts(audit, policy=None):
    """Conservative semantic counts derived from immutable original request rows."""
    from pipeline.capability_contract import digest
    from pipeline.instance_plan import PlanConflict, finding
    rows_by_uid=audit.get('edit_attempts',{})
    if policy is None:return {uid:len(rows) for uid,rows in rows_by_uid.items()}
    def reject(message):raise PlanConflict([finding('retry_policy_migration',message)])
    if (not isinstance(policy,dict) or policy.get('version')!=RETRY_POLICY_V2
            or any(policy.get(k)!=3 for k in ('max_semantic_attempts','max_protocol_attempts','max_json_attempts'))):
        reject('Retry policy must retain three semantic, protocol and JSON attempts')
    prefixes=policy.get('legacy_attempt_prefix')
    if not isinstance(prefixes,dict):reject('Retry policy lacks frozen original attempt prefixes')
    draft_hashes=policy.get('legacy_draft_prefix')
    if not isinstance(draft_hashes,list) or [digest(d) for d in audit.get('drafts',[])[:len(draft_hashes)]]!=draft_hashes:
        reject('Original draft history changed')
    legacy={'edit_attempts':{},'drafts':deepcopy(audit.get('drafts',[])[:len(draft_hashes)])}
    for uid,hashes in prefixes.items():
        rows=rows_by_uid.get(uid,[])
        if not isinstance(hashes,list) or len(rows)<len(hashes) or [digest(r) for r in rows[:len(hashes)]]!=hashes:
            reject('Original attempt history changed for '+str(uid))
        legacy['edit_attempts'][uid]=deepcopy(rows[:len(hashes)])
    bindings=policy.get('historical_protocol_pairs')
    if not isinstance(bindings,list):reject('Historical protocol mapping is missing')
    expected=build_retry_policy_migration(legacy,[{k:p[k] for k in ('unit_id','first_attempt','second_attempt')}
        for p in bindings if isinstance(p,dict) and all(k in p for k in ('unit_id','first_attempt','second_attempt'))])
    if expected!=policy:reject('Retry policy evidence or limits differ from the bound migration')
    counts={uid:len(rows) for uid,rows in rows_by_uid.items()}
    for binding in bindings:counts[binding['unit_id']]-=1
    for uid,rows in rows_by_uid.items():
        prefix=len(prefixes.get(uid,[]));discount=sum(p['unit_id']==uid for p in bindings)
        for n,row in enumerate(rows[prefix:],prefix+1):
            if (row.get('retry_policy')!=RETRY_POLICY_V2 or row.get('semantic_attempt')!=n-discount
                    or row.get('attempt')!=n or len(row.get('protocol_attempts',[]))>3):
                reject('New author attempt is not bound to the same semantic allowance')
        if counts[uid]>3:reject('Original semantic author allowance was exceeded')
    return counts


def _protocol_errors(wp, raw):
    """Only wire/record shape belongs here; identity, scope and truth stay semantic."""
    errors=[]
    def text(v):return isinstance(v,str) and bool(v.strip())
    def string_list(v):return isinstance(v,list) and all(text(x) for x in v)
    def bad(path,message):errors.append({'path':path,'message':message})
    if not isinstance(raw,dict):return [{'path':'$','message':'Expected a JSON object'}]
    if not isinstance(raw.get('decision'),str) or raw['decision'] not in ('repair','unresolved'):
        bad('decision','Expected repair or unresolved')
    if raw.get('decision')=='unresolved':
        if not text(raw.get('reason')):bad('reason','Unresolved requires its concrete reason')
        return errors
    if set(raw)-{'decision','reason','changes','cohort_changes'}:bad('$','Unknown response keys')
    for name in ('changes','cohort_changes'):
        if not isinstance(raw.get(name,[]),list):bad(name,'Expected an edit array')
    schemas={
        'objects':({'type'},{'slot','type'}),'relations':({'type','from','to'},{'slot','type','from','to','session'}),
        'events':({'type','participants','session'},{'slot','type','participants','session','caused_by'}),
        'observations':({'entity','field','sessions'},{'entity','field','sessions','mechanism'}),
        'obligations':({'line','carrier'},{'id','line','carrier','subtype'}),
        'publications':({'fact_slot','channel','session'},{'fact_slot','channel','session'}),
        'cohorts':({'basis','members'},{'id','basis','members'})}
    temporal={r['id'] for r in wp['world_blueprint']['relation_types'] if r.get('temporal')}
    for group in ('changes','cohort_changes'):
        rows=raw.get(group,[])
        if not isinstance(rows,list):continue
        for i,c in enumerate(rows):
            path=f'{group}[{i}]'
            if not isinstance(c,dict):bad(path,'Expected an edit object');continue
            collection='cohorts' if group=='cohort_changes' else c.get('collection')
            if not text(collection) or collection not in set(schemas)|{'depends_on'}:
                bad(path+'.collection','Expected a declared edit collection');continue
            if group=='changes' and not text(c.get('unit_id')):bad(path+'.unit_id','Expected a unit string')
            op=c.get('operation')
            if not isinstance(op,str) or op not in ('add','replace','remove'):bad(path+'.operation','Expected add, replace or remove');continue
            if 'key' in c and not text(c['key']):bad(path+'.key','key must be a string from edit_catalog; use selector for an object selector')
            if 'selector' in c and (not isinstance(c['selector'],dict) or not c['selector'] or not all(text(k) and (text(v) or type(v) is int) for k,v in c['selector'].items())):
                bad(path+'.selector','Expected a nonempty scalar selector object from edit_catalog')
            if collection=='depends_on':
                if not string_list(c.get('value')):bad(path+'.value','Expected an array of unit strings')
                continue
            inferred_add=op=='add' and collection in ('observations','publications')
            if not inferred_add and 'key' not in c and 'selector' not in c:bad(path,'An edit requires key or selector from edit_catalog')
            if op=='remove':continue
            value=c.get('value')
            if not isinstance(value,dict):bad(path+'.value','Expected a complete record object');continue
            required,allowed=schemas[collection]
            if collection=='relations' and text(value.get('type')) and value['type'] in temporal:required=required|{'session'}
            if required-set(value):bad(path+'.value','Missing required fields: '+', '.join(sorted(required-set(value))))
            if set(value)-allowed:bad(path+'.value','Unsupported record fields: '+', '.join(sorted(set(value)-allowed)))
            for k in ('slot','id','type','from','to','caused_by','entity','field','mechanism','line','subtype','fact_slot','channel','basis'):
                if k in value and not text(value[k]):bad(path+'.value.'+k,'Expected a nonempty string')
            if 'session' in value and type(value['session']) is not int:bad(path+'.value.session','Expected an integer')
            if 'sessions' in value and (not isinstance(value['sessions'],list) or not all(type(v) is int for v in value['sessions'])):bad(path+'.value.sessions','Expected integer periods')
            if 'participants' in value and (not isinstance(value['participants'],dict) or not all(text(k) and text(v) for k,v in value['participants'].items())):bad(path+'.value.participants','Expected role-to-slot strings')
            if 'members' in value and not string_list(value['members']):bad(path+'.value.members','Expected object slot strings')
            if 'carrier' in value:
                carrier=value['carrier']
                if not isinstance(carrier,dict) or set(carrier)-{'entities','field'} or not string_list(carrier.get('entities')):bad(path+'.value.carrier','Expected entities string array and optional field')
                elif 'field' in carrier and not text(carrier['field']):bad(path+'.value.carrier.field','Expected a nonempty string')
    return errors


def _state_capacity_context(wp, candidate, findings):
    """Expand original compiler evidence only; propose no new event or value."""
    bp=wp['world_blueprint'];types={t['id']:t for t in bp['entity_types']};specs={s['id']:s for s in bp['event_types']}
    objects={o['slot']:o for u in candidate['units'] for o in u['objects']};contexts=[]
    for f in findings:
        if f.get('code')!='state_write_capacity':continue
        ids=f.get('affected_ids',[])
        if len(ids)<2 or ids[0] not in objects:continue
        entity,field=ids[:2]
        declaration=next((s for s in types.get(objects[entity]['type'],{}).get('fields',[]) if s.get('name')==field),None)
        if not declaration or not isinstance(declaration.get('states'),list):continue
        writers=[]
        for unit in candidate['units']:
            for event in unit['events']:
                effects=specs.get(event['type'],{}).get('effect_fields',[])
                affected=[{'entity':event['participants'].get(e['role']),'field':e['field'],'role':e['role']} for e in effects]
                if any(e['entity']==entity and e['field']==field for e in affected):
                    writers.append({'event_slot':event['slot'],'event_type':event['type'],'session':event['session'],
                        'responsible_unit':unit['unit_id'],'participants':deepcopy(event['participants']),'all_effect_fields':affected})
        contexts.append({'finding_code':f['code'],'entity':entity,'field':field,'declared_states':deepcopy(declaration['states']),
            'original_field_declaration':deepcopy(declaration),
            'distinct_state_count':len(declaration['states']),'writer_count':len(writers),'writers':writers,
            'instruction':'These are existing authored writers and original declared states. The state count is only a necessary mechanical bound: event business meaning and the original one-way transitions must still hold. Repeating the same state event to extend another field trajectory is not a valid repair. Correct the actual conflict under the original business constraints; this context supplies no replacement facts.'})
    return contexts


RESOURCE_SYSTEM = """你为已保存的业务计划整理共享对象。原计划尚未提交，领域和所有事实槽位保留。
根据对象的类型、业务用途、引用和写入时段，识别可由同一真实对象承担的槽位。返回 aliases：旧槽位 -> 保留槽位。
只能合并同类型对象。保留槽位必须存在且不能同时被合并。程序会重接所有引用，保留全部关系、事件和出题需求，再运行原联合编译与业务审查。
每类保留对象不少于原最低数量；exact数量必须相等；全体对象不超过实体上限。避免把同一期需要互异身份的角色合并。
仅输出JSON {"decision":"repair|unresolved","reason":"简短业务依据","aliases":{"old":"retained"}}。无法在约束内实现时说明具体冲突。"""


def apply_aliases(proposal, aliases, wp, *, preserve_existing_deficits=False):
    from pipeline.instance_plan import PlanConflict, finding, index
    units, objects, _events, _relations, _owners = index(proposal)
    if not isinstance(aliases, dict):
        raise PlanConflict([finding("resource_shape", "Shared resources require an explicit alias map")])
    for old, new in aliases.items():
        if (old not in objects or new not in objects or old == new or new in aliases
                or objects[old]["type"] != objects[new]["type"]):
            raise PlanConflict([finding("resource_identity", "Aliases need distinct existing, same-type, retained identities", [old, new])])
    result = deepcopy(proposal)
    def ref(value): return aliases.get(value, value)
    for unit in result["units"]:
        unit["objects"] = [o for o in unit["objects"] if o["slot"] not in aliases]
        for row in unit["relations"]:
            row["from"], row["to"] = ref(row["from"]), ref(row["to"])
        for row in unit["events"]:
            row["participants"] = {role:ref(slot) for role,slot in row["participants"].items()}
        observations = {}
        for row in unit["observations"]:
            row["entity"] = ref(row["entity"])
            key = (row["entity"],row["field"])
            if key in observations:
                observations[key]["sessions"] = sorted(set(observations[key]["sessions"] + row["sessions"]))
            else: observations[key] = row
        unit["observations"] = list(observations.values())
        for row in unit["obligations"]:
            row["carrier"]["entities"] = list(dict.fromkeys(ref(e) for e in row["carrier"]["entities"]))
        for row in unit["publications"]: row["fact_slot"] = ref(row["fact_slot"])
    for row in result.get("cohorts", []):
        row["members"] = list(dict.fromkeys(ref(e) for e in row["members"]))
    counts = Counter(o["type"] for u in result["units"] for o in u["objects"])
    before = Counter(o["type"] for u in proposal["units"] for o in u["objects"])
    for row in wp["world_blueprint"]["entity_types"]:
        actual, minimum = counts[row["id"]], row["count"]
        floor = min(before[row["id"]], minimum) if preserve_existing_deficits else minimum
        if actual < floor or (row.get("cardinality_policy") == "exact" and actual > minimum):
            raise PlanConflict([finding("resource_count", "Shared objects must retain original type cardinality", [row["id"]], evidence={"required":minimum,"actual":actual})])
    if sum(counts.values()) > wp["supply_plan"]["world_limits"]["max_world_entities"]:
        raise PlanConflict([finding("resource_cap", "Shared resources still exceed the original entity cap",evidence={
            "actual_object_count":sum(counts.values()),"entity_cap":wp["supply_plan"]["world_limits"]["max_world_entities"],
            "actual_counts_by_type":dict(counts)})])
    return result


def derived_dependencies(proposal):
    """Identity allocation precedes facts; only causal parents order writers."""
    from pipeline.instance_plan import index
    _units, _objects, _events, _relations, owners = index(proposal)
    result = deepcopy(proposal)
    result["execution_policy"] = "shared-identities/v1"
    for unit in result["units"]:
        refs = set()
        for row in unit["events"]:
            if row.get("caused_by"): refs.add(row["caused_by"])
        unit["depends_on"] = sorted({owners[r] for r in refs if r in owners and owners[r] != unit["unit_id"]})
    return result


def validate_unit_patch(wp, base, raw, uid, cohort_owners, available_objects):
    """Validate complete authored records before committing any part of a unit.

    The keyed editor may supply the repeated slot/id only. Missing business
    fields are never inherited from the replaced row or invented by this gate.
    The original joint compiler still owns cross-record semantic acceptance.
    """
    from pipeline.instance_plan import PlanConflict, finding, apply_repair, index, _record_key
    from pipeline.capability_contract import digest
    try:
        candidate = apply_repair(base, raw)
    except PlanConflict as error:
        raise PlanConflict([dict(f, affected_ids=list(dict.fromkeys([uid]+f.get('affected_ids', []))),
            evidence={'responsible_unit':uid, 'original_finding':deepcopy(f)}) for f in error.findings]) from error
    changes=raw.get('changes',[])
    cohorts=raw.get('cohort_changes',[])
    if (any(c.get('unit_id') != uid for c in changes)
            or any(cohort_owners.get(c.get('key')) != uid for c in cohorts)):
        raise PlanConflict([finding('repair_scope','Unit edits must retain their assigned scope',[uid])])
    schemas={
        'objects':({'type'}, {'slot','type'}),
        'relations':({'type','from','to'}, {'slot','type','from','to','session'}),
        'events':({'type','participants','session'}, {'slot','type','participants','session','caused_by'}),
        'observations':({'entity','field','sessions'}, {'entity','field','sessions','mechanism'}),
        'obligations':({'line','carrier'}, {'id','line','carrier','subtype'}),
        'publications':({'fact_slot','channel','session'}, {'fact_slot','channel','session'}),
        'cohorts':({'basis','members'}, {'id','basis','members'}),
    }
    bp=wp['world_blueprint']; periods=bp['temporal_model']['n_sessions']
    declarations={c:{s['id']:s for s in bp[g]} for c,g in
                  [('objects','entity_types'),('relations','relation_types'),('events','event_types')]}
    units,objects,events,relations,owners=index(candidate)
    problems=[]
    def text(value):return isinstance(value,str) and bool(value.strip())
    def strings(value, minimum=0):return isinstance(value,list) and len(value)>=minimum and all(text(v) for v in value)
    def session(value):return type(value) is int and 0<=value<periods
    for change in changes+[dict(c,collection='cohorts') for c in cohorts]:
        collection=change['collection']; op=change['operation']; value=change.get('value')
        key=change.get('key')
        if key is None:key=change.get('selector') or (_record_key(collection,value) if isinstance(value,dict) else None)
        issues=[]; declared=None
        if op=='remove':
            # A local deletion cannot transfer responsibility to readers whose
            # references were valid before this author removed the shared fact.
            removed=str(key)
            users=[]
            for other in candidate['units']:
                if other['unit_id']==uid:continue
                refs={r.get(s) for r in other['relations'] for s in ('from','to')}
                refs.update(v for e in other['events'] for v in e.get('participants',{}).values())
                refs.update(e.get('caused_by') for e in other['events'])
                refs.update(o.get('entity') for o in other['observations'])
                refs.update(v for o in other['obligations'] for v in o.get('carrier',{}).get('entities',[]))
                refs.update(p.get('fact_slot') for p in other['publications'])
                if removed in refs:users.append(other['unit_id'])
            if users:problems.append(finding('repair_shared_delete','Author cannot delete a fact still used by another unit',
                [uid,removed],evidence={'responsible_unit':uid,'collection':collection,'original_key':key,'referencing_units':users}))
            continue
        if collection=='depends_on':
            if (not strings(value) or len(set(value))!=len(value)
                    or any(u not in units or u==uid for u in value)):
                issues.append('depends_on must contain distinct existing other unit ids')
        elif not isinstance(value,dict):
            issues.append('add/replace value must be a complete record object')
        else:
            required,allowed=schemas[collection]
            declared=declarations.get(collection,{}).get(value.get('type')) if text(value.get('type')) else None
            if collection=='relations' and declared and declared.get('temporal'):required=required|{'session'}
            missing=sorted(required-set(value)); unknown=sorted(set(value)-allowed)
            if missing:issues.append('missing required fields: '+', '.join(missing))
            if unknown:issues.append('unsupported fields would not become planned facts: '+', '.join(unknown))
            for name in ('slot','id'):
                if name in value and not text(value[name]):issues.append(name+' must be a nonempty string')
            if collection in declarations and declared is None:issues.append('type must name an existing '+collection+' declaration')
            if collection=='relations':
                for side in ('from','to'):
                    if not text(value.get(side)):issues.append(side+' must be an object slot')
                    elif declared and objects.get(value[side],{}).get('type')!=declared[side+'_type']:
                        issues.append(side+' must bind an existing '+declared[side+'_type']+' object')
                # Static relations retain the original compiler's canonical
                # session-zero lowering. A supplied session is still typed.
                if 'session' in value and not session(value['session']):issues.append('session must be an integer in the frozen calendar')
            elif collection=='events':
                if not session(value.get('session')):issues.append('session must be an integer in the frozen calendar')
                participants=value.get('participants')
                if not isinstance(participants,dict):issues.append('participants must be a role-to-slot object')
                elif declared:
                    if set(participants)!=set(declared['roles']):issues.append('participants must bind every declared role exactly')
                    for role,kind in declared['roles'].items():
                        if not text(participants.get(role)) or objects.get(participants.get(role),{}).get('type')!=kind:
                            issues.append('role '+role+' must bind an existing '+kind+' object')
                if 'caused_by' in value and not text(value['caused_by']):issues.append('caused_by must be a nonempty event slot')
            elif collection=='observations':
                for field in ('entity','field'):
                    if not text(value.get(field)):issues.append(field+' must be a nonempty string')
                slots=value.get('sessions')
                if (not isinstance(slots,list) or not slots or not all(session(s) for s in slots)
                        or len(set(slots))!=len(slots)):issues.append('sessions must be distinct valid integer periods')
                if 'mechanism' in value and not text(value['mechanism']):issues.append('mechanism must be nonempty text')
            elif collection=='obligations':
                if not text(value.get('line')) or value['line'] not in wp['supply_plan']['candidate_allocation']:issues.append('line must be an allocated capability')
                carrier=value.get('carrier')
                if not isinstance(carrier,dict) or set(carrier)-{'entities','field'}:issues.append('carrier requires only entities and optional field')
                if not isinstance(carrier,dict) or not strings(carrier.get('entities'),1):issues.append('carrier.entities must be a nonempty slot array')
                if isinstance(carrier,dict) and 'field' in carrier and not text(carrier['field']):issues.append('carrier.field must be a nonempty declared field name')
                if 'subtype' in value and not text(value['subtype']):issues.append('subtype must be nonempty text')
            elif collection=='publications':
                if not text(value.get('fact_slot')):issues.append('fact_slot must be a nonempty slot')
                if not text(value.get('channel')) or value['channel'] not in bp['evidence_channels']:issues.append('channel must be an existing evidence channel')
                if not session(value.get('session')):issues.append('session must be an integer in the frozen calendar')
            elif collection=='cohorts':
                if not text(value.get('basis')):issues.append('basis must be nonempty prior business text')
                members=value.get('members')
                if not strings(members,2) or len(set(members))!=len(members):issues.append('members must be distinct object slots, at least two')
        if issues:
            old_unit=next(u for u in base['units'] if u['unit_id']==uid)
            originals=base.get('cohorts',[]) if collection=='cohorts' else old_unit.get(collection,[])
            original=next((r for r in originals if isinstance(r,dict) and _record_key(collection,r)==key),None)
            original_schema=declarations.get(collection,{}).get(original.get('type')) if original else None
            problems.append(finding('repair_record_shape','Authored '+collection+' '+op+' is not a complete typed record',
                [uid,str(key)],evidence={'responsible_unit':uid,'collection':collection,'operation':op,'original_key':deepcopy(key),
                    'issues':issues,'required_fields':sorted(schemas[collection][0]) if collection in schemas else [],
                    'allowed_fields':sorted(schemas[collection][1]) if collection in schemas else [],
                    'declared_schema':deepcopy(declared),'original_record':deepcopy(original),
                    'original_declared_schema':deepcopy(original_schema),'replacement_semantics':'complete_record_not_merge'}))
    identities={}
    for unit in candidate['units']:
        for collection,rows,keyfield in [('units',[unit],'unit_id')]+[(c,unit[c],'id' if c=='obligations' else 'slot') for c in ('objects','relations','events','obligations')]:
            for row in rows:
                key=row.get(keyfield)
                if key in identities:problems.append(finding('repair_identity_collision','Authored identity must remain globally unique',
                    [uid,str(key)],evidence={'responsible_unit':uid,'original_key':key,'first_owner':identities[key],'other_owner':unit['unit_id']}))
                identities[key]=unit['unit_id']
    for change in changes:
        if change['operation']=='add' and change['collection'] in ('objects','relations','events','obligations'):
            key=change['key']
            namespace=max((name for name in units if key.startswith(name)),key=len,default=None)
            if namespace!=uid:problems.append(finding('repair_identity_namespace','New authored identities must belong to the responsible unit id namespace',
                [uid,key],evidence={'responsible_unit':uid,'original_key':key}))
    if problems:raise PlanConflict(problems)
    original=next(u for u in base['units'] if u['unit_id']==uid)
    if len(units[uid]['objects'])-len(original['objects'])>available_objects:
        raise PlanConflict([finding('resource_capacity','Unit exceeded its assigned new object capacity',[uid])])
    if digest(derived_dependencies(candidate))==digest(derived_dependencies(base)):
        raise PlanConflict([finding('repair_no_change','The change disappears under the declared dependency derivation',[uid])])
    return candidate


def _reply(tracer, step, system, payload, attempt=None, update=None, *, defer_decision=False):
    from pipeline.instance_plan import PlanConflict, finding
    try:
        from execution_control import check_stop
        check_stop()
        if callable(getattr(tracer,'check_global_stop',None)):tracer.check_global_stop()
        raw = tracer.chat_json(step,[{"role":"system","content":system},
            {"role":"user","content":json.dumps(payload,ensure_ascii=False)}],
            model=config.STRUCTURE_MODEL, temperature=.3, max_tokens=32768, retries=3,
            response_format={"type":"json_object"})
    except BaseException as error:
        if update is not None:update(status="call_failed", error_type=type(error).__name__)
        elif attempt is not None:attempt.update(status="call_failed", error_type=type(error).__name__)
        raise
    if update is not None:update(proposal=deepcopy(raw), status="response_received")
    elif attempt is not None:attempt.update(proposal=deepcopy(raw), status="response_received")
    if isinstance(raw,dict) and "__error__" in raw:
        if update is not None:update(status="provider_error")
        elif attempt is not None:attempt["status"]="provider_error"
        from execution_control import global_stop_exception
        stopped=global_stop_exception(raw)
        if stopped is not None:raise stopped
        raise RuntimeError("Plan transaction provider failure: " + str(raw["__error__"]))
    if not defer_decision and isinstance(raw,dict) and raw.get("decision") == "unresolved":
        if update is not None:update(status="unresolved")
        elif attempt is not None:attempt["status"]="unresolved"
        raise PlanConflict([finding("plan_unresolved", str(raw.get("reason")), kind="design")])
    return raw


def _route_findings(wp, working, findings):
    """Assign global deficits once, then route local facts to their real owners."""
    from pipeline.instance_plan import PlanConflict, finding, index
    units, objects, events, relations, owners = index(working)
    owners.update({o['id']:uid for uid,u in units.items() for o in u['obligations']})
    cohort_owners = {c['id']:min((owners[e] for e in c['members'] if e in owners),default=min(units))
                     for c in working.get('cohorts',[])}
    owners.update(cohort_owners)
    assigned = {uid:Counter(o['line'] for o in u['obligations']) for uid,u in units.items()}
    for line,total in wp['supply_plan']['candidate_allocation'].items():
        difference=total-sum(c[line] for c in assigned.values())
        eligible=sorted(uid for uid in units if assigned[uid][line]) or sorted(units)
        for i in range(max(0,difference)):assigned[eligible[i%len(eligible)]][line]+=1
        for _ in range(max(0,-difference)):
            uid=max(eligible,key=lambda k:assigned[k][line]);assigned[uid][line]-=1
    routed={uid:[] for uid in units}
    additions={uid:{'objects':{},'events':{},'relations':{}} for uid in units}
    subtype_totals={uid:{} for uid in units}
    routes=[]; handled=set()
    def choose(collection,kind):
        specs=wp['world_blueprint']
        required=set()
        if collection=='events':required=set(next(e for e in specs['event_types'] if e['id']==kind)['roles'].values())
        elif collection=='relations':
            spec=next(r for r in specs['relation_types'] if r['id']==kind)
            required={spec['from_type'],spec['to_type']}
        def priority(uid):
            unit=units[uid]
            count=sum(r.get('type')==kind for r in unit[collection])
            kinds={r.get('type') for r in unit['objects']}
            return (-count,-len(kinds & required),uid)
        return min(units,key=priority)
    for item in findings:
        code=item.get('code'); ids=item.get('affected_ids') or []; targets=set()
        if code in ('missing_declared_objects','exact_identity_count','missing_declared_instances'):
            groups=[('objects','entity_types')] if code!='missing_declared_instances' else [('events','event_types'),('relations','relation_types')]
            for collection,group in groups:
                for spec in wp['world_blueprint'][group]:
                    if spec['id'] not in ids:continue
                    actual=sum(r.get('type')==spec['id'] for u in units.values() for r in u[collection])
                    minimum=spec.get('count',0) if collection=='objects' else spec.get('min_count',0)
                    deficit=minimum-actual
                    if deficit<=0:continue
                    uid=choose(collection,spec['id']);targets.add(uid)
                    identity=(collection,spec['id'])
                    if identity not in handled:
                        additions[uid][collection][spec['id']]=deficit;handled.add(identity)
        elif code=='demand_slots':
            targets={uid for uid,u in units.items() if any(assigned[uid][line]!=sum(o['line']==line for o in u['obligations']) for line in ids)}
        elif code=='subtype_slots' and len(ids)==2:
            line,subtype=ids
            totals={uid:sum(o.get('line')==line and o.get('subtype')==subtype for o in u['obligations']) for uid,u in units.items()}
            minimum=(item.get('evidence') or {}).get('required',0)
            missing=minimum-sum(totals.values())
            for uid in sorted(units,key=lambda u:(-totals[u],-assigned[u][line],u)):
                extra=min(missing,max(0,assigned[uid][line]-totals[uid]))
                if extra>0:
                    totals[uid]+=extra;missing-=extra;targets.add(uid)
                    subtype_totals[uid].setdefault(line,{})[subtype]=totals[uid]
            if missing>0:targets=set()
        else:
            for uid,unit in units.items():
                mentioned={uid}
                def mention(values):mentioned.update(v for v in values if isinstance(v,str))
                for row in unit['relations']:mention((row.get('slot'),row.get('from'),row.get('to')))
                for row in unit['events']:
                    participants=row.get('participants')
                    mention((row.get('slot'),))
                    if isinstance(participants,dict):mention(participants.values())
                mention(o.get('entity') for o in unit['observations'] if isinstance(o,dict))
                for obligation in unit['obligations']:
                    carrier=obligation.get('carrier')
                    if isinstance(carrier,dict) and isinstance(carrier.get('entities'),list):mention(carrier['entities'])
                mention(p.get('fact_slot') for p in unit['publications'] if isinstance(p,dict))
                if any(x in mentioned or owners.get(x)==uid for x in ids):targets.add(uid)
            if code=='relation_schedule':
                evidence=item.get('evidence') or {}
                for slot,row in relations.items():
                    if row.get('type')==evidence.get('relation') and row.get('from')==evidence.get('source') and row.get('to')==evidence.get('target'):
                        targets.add(owners[slot])
        # These defects require a complete plan/declaration author, not a record editor.
        if code in {'plan_shape','unit_shape','unit_purpose','slot_shape','slot_identity','answer_in_plan','blueprint_compilation','seed_constraint','capacity_bound'}:
            targets=set()
        if not targets:
            raise PlanConflict([finding('repair_routing','Compiler finding has no legal local repair owner',ids,
                evidence={'unrouted_finding':deepcopy(item),'author_calls_started':False})])
        for uid in sorted(targets):routed[uid].append(deepcopy(item))
        routes.append({'finding':deepcopy(item),'responsible_units':sorted(targets)})
    available=wp['supply_plan']['world_limits']['max_world_entities']-len(objects)
    capacity={uid:sum(additions[uid]['objects'].values()) for uid in units}
    if sum(capacity.values())>available:
        raise PlanConflict([finding('resource_capacity','Object minima exceed the available original entity capacity',
            evidence={'required_additions':sum(capacity.values()),'available':available})])
    jobs=sorted(uid for uid in units if routed[uid])
    if not jobs:raise PlanConflict([finding('repair_routing','No compiler finding produced a legal repair job')])
    # Reserve declared minima first, then divide the remaining finite slots.
    # Every slot is assigned once; unrelated authors cannot exhaust one another
    # merely because their ids sort after the first job.
    for position in range(available-sum(capacity.values())):
        capacity[jobs[position%len(jobs)]]+=1
    return units,objects,events,relations,cohort_owners,assigned,routed,additions,subtype_totals,capacity,routes


def _restore_batch(wp, proposal, audit, retry_policy=None):
    """Reconstruct a recorded batch without changing any historical response.

    Every replayed response must have its original consumed attempt. Full
    compiler findings are collected on shape-valid responses before any new
    request; a response being replayable is not a semantic acceptance claim.
    """
    from pipeline.instance_plan import PlanConflict, finding, apply_repair, compile_plan
    from pipeline.capability_contract import digest
    def stop(message):raise PlanConflict([finding('repair_checkpoint',message)])
    counts=retry_policy_counts(audit,retry_policy)
    for uid,attempts in audit.get('edit_attempts',{}).items():
        if counts[uid]>3:stop('Original local attempt limit was exceeded')
        for number,row in enumerate(attempts,1):
            if row.get('attempt')!=number or not isinstance(row.get('proposal'),dict):stop('An original request lacks a complete recorded response')
            if row.get('status') not in ('response_received','locally_rejected','locally_compiled','joint_rejected') or '__error__' in row['proposal']:
                stop('Unsettled or failed original requests cannot be replayed')
            if row.get('retry_policy')==RETRY_POLICY_V2:
                protocol=row.get('protocol_attempts',[])
                if (not protocol or [p.get('protocol_attempt') for p in protocol]!=list(range(1,len(protocol)+1))
                        or protocol[-1].get('status')!='protocol_valid' or protocol[-1].get('proposal')!=row['proposal']
                        or any(p.get('status')!='protocol_rejected' or not _protocol_errors(wp,p.get('proposal')) for p in protocol[:-1])):
                    stop('Saved protocol attempts are incomplete or not bound to their final response')
    active=audit.get('active_batch')
    in_progress=bool(active and active.get('status')=='editing')
    if in_progress:
        base=deepcopy(active['base']); iteration=active['iteration']; findings=deepcopy(active['findings'])
        replies=deepcopy(active.get('unit_proposals',{}))
        # A response may have reached disk immediately before its callback.
        for uid,rows in audit.get('edit_attempts',{}).items():
            last=rows[-1]
            if last.get('batch_iteration')==iteration:replies[uid]=deepcopy(last['proposal'])
    else:
        drafts=audit.get('drafts',[])
        if not drafts:stop('No saved repair batch can be reconstructed')
        latest=drafts[-1]; iteration=latest['iteration']
        if [d.get('iteration') for d in drafts]!=list(range(1,len(drafts)+1)):
            stop('Recorded joint draft history is not contiguous')
        base=derived_dependencies(deepcopy(proposal)); findings=None
        if audit.get('resource_proposal'):
            base=derived_dependencies(apply_aliases(base,audit['resource_proposal']['aliases'],wp,preserve_existing_deficits=True))
        # Explicit revalidation can rebuild a later batch from an earlier
        # healthy base. Keep defective historical drafts verbatim, and verify
        # their original replies without treating them as accepted truth.
        known_bases={digest(base):deepcopy(base)}
        for draft in drafts:
            rebases=[r for r in audit.get('checkpoint_revalidations',[]) if r.get('iteration')==draft['iteration']]
            if rebases:
                base_hash=rebases[-1].get('base_sha256')
                if base_hash not in known_bases:stop('Checkpoint revalidation base is not an original recorded candidate')
                base=deepcopy(known_bases[base_hash])
            raw_base=deepcopy(base)
            for uid,raw in draft['unit_proposals'].items():
                if not any(a.get('proposal')==raw for a in audit.get('edit_attempts',{}).get(uid,[])):
                    stop('Saved batch response is not bound to an original consumed attempt')
                raw_base=apply_repair(raw_base,raw)
            raw_base=derived_dependencies(raw_base)
            if digest(raw_base)!=digest(draft['candidate']):stop('Recorded batch candidate does not match its original responses')
            known_bases[digest(raw_base)]=deepcopy(raw_base)
            if draft is latest:break
            base=raw_base
        replies=deepcopy(latest['unit_proposals'])
        route=next((r for r in reversed(audit.get('routing',[])) if r.get('iteration')==iteration),None)
        if route is None:stop('Recorded batch has no original responsibility routing')
        findings=[deepcopy(r['finding']) for r in route['routes']]
    if type(iteration) is not int or iteration<1:stop('Recorded mechanical iteration is invalid')
    if in_progress and iteration!=len(audit.get('drafts',[]))+1:stop('Active batch does not follow recorded joint drafts')
    # Draft numbers describe append-only history, not a second author quota.
    # The original per-author three requests remain the decreasing bound.
    next_iteration=iteration if in_progress else iteration+1
    routing=_route_findings(wp,base,findings)
    pending={}; projection=deepcopy(base); replayed=[]
    touched={}
    for uid,raw in replies.items():
        if uid not in routing[6] or not routing[6][uid]:stop('Saved response is outside its original responsibility routing')
        if not any(a.get('proposal')==raw for a in audit.get('edit_attempts',{}).get(uid,[])):
            stop('Saved response is not bound to an original consumed attempt')
        try:validate_unit_patch(wp,base,raw,uid,routing[4],routing[9][uid])
        except PlanConflict as error:pending[uid]=deepcopy(error.findings)
        else:
            projection=apply_repair(projection,raw);replayed.append(uid)
            for c in raw.get('changes',[]):
                if isinstance(c.get('key'),str):touched.setdefault(c['key'],set()).add(uid)
    local_invalid=bool(pending)
    joint=[]
    try:compile_plan(wp,derived_dependencies(projection))
    except PlanConflict as error:
        joint=deepcopy(error.findings)
        for f in joint:
            responsible=set().union(*(touched.get(k,set()) for k in f.get('affected_ids',[])))
            for uid in responsible:pending.setdefault(uid,[]).append(deepcopy(f))
    if not local_invalid:
        # All saved edits are complete records. Continue from the actual joint
        # candidate and route its current findings, including deeper compiler
        # gates that were hidden by the previous batch's defects. This avoids
        # repeating already successful authors merely to replay their replies.
        base=derived_dependencies(projection);findings=joint;replies={};pending={}
        if joint:
            current_route=_route_findings(wp,base,joint)
            pending={uid:deepcopy(rows) for uid,rows in current_route[6].items() if rows}
    for uid in pending:
        if counts.get(uid,0)>=3:
            stop('Original local author allowance exhausted for '+uid)
    return base,next_iteration-1,findings,replies,pending,{
        'version':'typed-local-replay/v1','original_iteration':iteration,'iteration':next_iteration,'base_sha256':digest(base),
        'joint_drafts_consumed':len(audit.get('drafts',[])),
        'scheduler_version':'original-author-slots/v1','max_attempts_per_author':3,
        'replayed_units':replayed,'local_retry_findings':deepcopy(pending),
        'joint_projection_findings':joint,'historical_attempts_preserved':True}


def repair(wp, proposal, feedback, tracer, audit, resource_resume=None, checkpoint=None, resume_checkpoint=False, retry_policy=None, max_joint_batches=None):
    from pipeline.instance_plan import REPAIR_SYSTEM, PlanConflict, finding, compile_plan, apply_repair, index
    working = deepcopy(proposal)
    try:compile_plan(wp,working)
    except PlanConflict as error:
        unsupported=[f for f in error.findings if f.get('code') in {'plan_shape','unit_shape','unit_purpose','slot_shape','slot_identity','answer_in_plan','cohort_shape'}]
        if unsupported:raise PlanConflict([finding('repair_routing','Plan shape requires its complete plan author',evidence={'unrouted_findings':unsupported,'author_calls_started':False})])
    for unit in working['units']:
        for key in ('objects','events','relations','observations','obligations','publications','depends_on'):unit.setdefault(key,[])
    working.setdefault('cohorts',[])
    lock=RLock()
    def save():
        with lock:
            if checkpoint is not None:checkpoint(audit)
    def record(entry, **values):
        with lock:entry.update(values);save()
    if audit.get('retry_policy_migrations') and retry_policy!=audit['retry_policy_migrations'][-1]['policy']:
        raise PlanConflict([finding('retry_policy_migration','Saved policy requires the same explicit bound policy on reentry')])
    retry_policy_counts(audit,retry_policy)
    prior_attempts=list(audit.get('resource_attempts',[]))+[a for rows in audit.get('edit_attempts',{}).values() for a in rows]
    preserved_attempt_counts={uid:len(rows) for uid,rows in audit.get('edit_attempts',{}).items()}
    if audit.get('drafts') and not resume_checkpoint:
        raise PlanConflict([finding('repair_checkpoint','Saved repair history requires explicit checkpoint continuation')])
    for attempt in prior_attempts:
        raw=attempt.get('proposal')
        if attempt.get('status')=='unresolved' or (isinstance(raw,dict) and raw.get('decision')=='unresolved'):
            raise PlanConflict([finding('plan_unresolved','Original author already reported this transaction unresolved',
                kind='design',evidence=deepcopy(attempt))])
    resume=None
    if resume_checkpoint:
        resume=_restore_batch(wp,working,audit,retry_policy=retry_policy)
        working=deepcopy(resume[0])
        audit.setdefault('checkpoint_revalidations',[]).append(deepcopy(resume[5]));save()
    if retry_policy is not None and not audit.get('retry_policy_migrations'):
        audit.setdefault('retry_policy_migrations',[]).append({'policy':deepcopy(retry_policy),
            'derived_semantic_counts':retry_policy_counts(audit,retry_policy),'original_history_preserved':True})
        save()
    original_objects = [dict(o, unit_id=u["unit_id"], business_purpose=u["business_purpose"])
                        for u in working["units"] for o in u["objects"]]
    types = wp["world_blueprint"]["entity_types"]
    counts = Counter(o["type"] for o in original_objects)
    resource_conflict = len(original_objects) > wp["supply_plan"]["world_limits"]["max_world_entities"] or any(
        t.get("cardinality_policy") == "exact" and counts[t["id"]] > t["count"] for t in types)
    if resource_conflict and resume is None:
        payload = {"business":wp.get("seed_contract"),"objects":original_objects,
                   "cardinalities":[{k:t.get(k) for k in ("id","count","cardinality_policy")} for t in types],
                   "entity_cap":wp["supply_plan"]["world_limits"]["max_world_entities"],"cohorts":working.get("cohorts",[]),
                   "actual_object_count":len(original_objects),"actual_counts_by_type":dict(counts),
                   "event_roles":[{k:e.get(k) for k in ("slot","type","participants","session")} for u in working["units"] for e in u["events"]]}
        attempts=audit.setdefault("resource_attempts",[])
        for attempt in range(len(attempts),3):
            reused=attempt==0 and resource_resume is not None
            row={"attempt":attempt+1,"status":"requested","revalidated_from_parent":reused};attempts.append(row)
            save()
            raw=deepcopy(resource_resume) if reused else _reply(tracer,"council.instance_resources",RESOURCE_SYSTEM,payload,row,lambda **v:record(row,**v))
            audit["resource_proposal"] = deepcopy(raw)
            row['proposal']=deepcopy(raw)
            save()
            try:
                working=apply_aliases(working,raw.get("aliases") if isinstance(raw,dict) else None,wp,preserve_existing_deficits=True)
            except PlanConflict as error:
                row["findings"]=deepcopy(error.findings)
                save()
                payload["mechanical_feedback"]={"previous_response":raw,"compiler_findings":error.findings,
                    "actual_object_count":len(original_objects),"entity_cap":payload["entity_cap"],
                    "instruction":"Count the original supplied catalog, repair only the alias transaction; retain business constraints and distinct required roles. If infeasible, report unresolved with the conflicting roles."}
                if attempt==2:raise
            else:
                record(row,status='locally_compiled')
                break
    working = derived_dependencies(working)
    if resume is not None:findings=deepcopy(resume[2])
    else:
        try:compile_plan(wp, working)
        except PlanConflict as error:findings = error.findings
        else:
            findings = deepcopy(feedback.get("findings", []))
            if not any(f.get("kind") == "business" for f in findings):return working
    # One atomic transaction routes joint compiler feedback within each author's
    # original three-attempt allowance. Successful parallel edits survive.
    iteration=resume[1] if resume else 0
    while True:
        if not findings:return working
        if max_joint_batches is not None and iteration >= max_joint_batches:
            raise PlanConflict([finding('repair_joint_batch_limit',
                'The external experiment allowance for complete joint candidate revisions is exhausted',
                evidence={'max_joint_batches':max_joint_batches,'completed_joint_batches':iteration,
                          'compiler_findings':deepcopy(findings)})])
        (units,objects,events,relations,cohort_owners,assigned,routed,additions,
         subtype_totals,capacity,routes)=_route_findings(wp,working,findings)
        replay=resume[3] if resume is not None and iteration==resume[1] else {}
        retry_findings=resume[4] if resume is not None and iteration==resume[1] else {}
        requested_units=[uid for uid,rows in routed.items() if rows and (uid not in replay or uid in retry_findings)]
        semantic_counts=retry_policy_counts(audit,retry_policy)
        exhausted=[uid for uid in requested_units if semantic_counts.get(uid,0)>=3]
        if exhausted:
            raise PlanConflict([finding('repair_author_exhausted','Original local author allowance exhausted',exhausted,
                evidence={'max_attempts_per_author':3,'compiler_findings':deepcopy(findings),'author_calls_started':False})])
        if not requested_units:
            raise PlanConflict([finding('repair_no_progress','No unused responsible author slot can advance this batch',
                evidence={'compiler_findings':deepcopy(findings),'author_calls_started':False})])
        attempts_before=sum(len(rows) for rows in audit.get('edit_attempts',{}).values())
        audit.setdefault('routing',[]).append({'iteration':iteration+1,'routes':routes,
            'assigned_new_object_slots':deepcopy(capacity),'assigned_instance_additions':deepcopy(additions)})
        save()
        audit['active_batch']={'iteration':iteration+1,'base':deepcopy(working),'findings':deepcopy(findings),
                               'unit_proposals':deepcopy(replay),'status':'editing'};save()
        jobs = []
        for uid,unit in units.items():
            local = routed[uid]
            if not local: continue
            previous=audit.get('edit_attempts',{}).get(uid,[])
            if len(previous)>preserved_attempt_counts.get(uid,0) and not replay:
                previous[-1]['joint_findings']=deepcopy(local)
                if previous[-1].get('status')=='locally_compiled':previous[-1]['status']='joint_rejected'
                save()
            jobs.append((uid, {"business":wp.get("seed_contract"),"blueprint":wp["world_blueprint"],
                "current_candidate":working,
                "requirements":wp["supply_plan"]["requirements"],"unit":unit,
                "assigned_line_totals":dict(assigned[uid]),"shared_objects":list(objects.values()),
                "assigned_instance_additions":additions[uid],"assigned_subtype_minimum":subtype_totals[uid],
                "entity_cap":wp["supply_plan"]["world_limits"]["max_world_entities"],"allocated_object_count":len(objects),
                "available_new_object_slots":capacity[uid],
                "shared_relations":list(relations.values()),"shared_events":list(events.values()),
                "shared_obligations":[o for other in units.values() for o in other['obligations']],
                "cohorts":working.get("cohorts",[]),"allowed_cohort_ids":[cid for cid,owner in cohort_owners.items() if owner == uid],"findings":retry_findings.get(uid,local)}))
        system = REPAIR_SYSTEM + "\n本请求只修改unit中的记录，其他单元保持。assigned_line_totals为本单元的逐线总需求数；assigned_instance_additions是全局缺额中仅分派给本单元的补充数量，不要替其他作者重复补齐；assigned_subtype_minimum为本单元应覆盖的子型最低数量，仍须保留逐线总额。新对象、事件、关系、需求的键以本unit_id为前缀；观察和公开安排的新记录无需自行造键。跨单元对象可引用shared_objects；跨单元事件不可编辑。available_new_object_slots是专门分派给本单元的新增对象容量，为0时复用已有对象。依赖关系由真实引用自动推导。观察可以读取以前事件写入后延续的值，数值轨迹及过程真实变更仍须满足construction。"
        system += '\nreplace是完整记录替换，不是字段合并。必须输出所选记录的全部必需字段（事件包括type、participants、session）；只有重复slot/id可省略。value仅允许原计划schema列出的字段，真实属性值留给后续事实作者。已给出的原记录和schema用于作者核对，程序不会补事实。'
        def run(job):
            uid,payload=job
            from pipeline.instance_plan import _record_key
            columns={'objects':('slot',),'relations':('slot',),'events':('slot',),'obligations':('id',),
                     'observations':('entity','field'),'publications':('fact_slot','channel','session')}
            payload['edit_catalog']={c:[{'key':_record_key(c,r),'selector':{k:r[k] for k in keys}} for r in payload['unit'][c]] for c,keys in columns.items()}
            if retry_policy is not None:
                payload['mechanical_capacity_context']=_state_capacity_context(wp,working,payload['findings'])
                payload['retry_policy']={'version':RETRY_POLICY_V2,'semantic_attempt':semantic_counts.get(uid,0)+1,
                    'max_semantic_attempts':3,'max_protocol_attempts':3,'max_json_attempts':3,
                    'historical_protocol_pairs':[deepcopy(p) for p in retry_policy['historical_protocol_pairs'] if p['unit_id']==uid]}
            seen=set()
            with lock:local_audit=audit.setdefault('edit_attempts',{}).setdefault(uid,[])
            if uid in replay:
                if uid not in retry_findings:return uid,deepcopy(replay[uid])
                payload['edit_feedback']={'previous_response':deepcopy(replay[uid]),'compiler_findings':deepcopy(retry_findings[uid]),
                    'instruction':'The saved response was revalidated, not committed. Return the complete corrected change set against the supplied original unit. Retain justified changes, provide all record fields, and correct these exact findings. Original consumed attempts count toward the same limit of three.'}
            elif local_audit and 'proposal' in local_audit[-1]:
                payload['edit_feedback']={'previous_response':local_audit[-1]['proposal'],'compiler_findings':deepcopy(payload['findings']),
                    'instruction':'The supplied unit includes all previous accepted edits. Correct only the remaining joint compiler findings; keep the assigned demand, business and limits.'}
            for local_attempt in range(semantic_counts.get(uid,0),3):
                payload.pop('protocol_feedback',None)
                if retry_policy is not None:payload['retry_policy']['semantic_attempt']=local_attempt+1
                entry={'attempt':len(local_audit)+1,'status':'requested','batch_iteration':iteration+1}
                if retry_policy is not None:entry.update(retry_policy=RETRY_POLICY_V2,semantic_attempt=local_attempt+1,protocol_attempts=[])
                with lock:local_audit.append(entry);save()
                if retry_policy is None:
                    raw=_reply(tracer,"council.instance_plan_unit_repair",system,payload,entry,lambda **v:record(entry,**v))
                else:
                    for protocol_attempt in range(1,4):
                        protocol={'protocol_attempt':protocol_attempt,'status':'requested'}
                        with lock:entry['protocol_attempts'].append(protocol);save()
                        try:
                            raw=_reply(tracer,"council.instance_plan_unit_repair",system,payload,protocol,
                                lambda **v:record(protocol,**v),defer_decision=True)
                        except BaseException as error:
                            record(entry,status=protocol.get('status','call_failed'),error_type=type(error).__name__)
                            raise
                        errors=_protocol_errors(wp,raw)
                        if not errors:
                            record(protocol,status='protocol_valid')
                            record(entry,status='response_received',proposal=deepcopy(raw))
                            if raw['decision']=='unresolved':
                                record(entry,status='unresolved')
                                raise PlanConflict([finding('plan_unresolved',raw['reason'],kind='design')])
                            break
                        record(protocol,status='protocol_rejected',protocol_errors=deepcopy(errors))
                        if protocol_attempt==3:
                            record(entry,status='protocol_exhausted',proposal=deepcopy(raw))
                            raise PlanConflict([finding('repair_protocol_exhausted','Author protocol correction allowance exhausted',[uid],
                                evidence={'protocol_errors':errors,'semantic_attempt':local_attempt+1,'max_protocol_attempts':3})])
                        payload['protocol_feedback']={'previous_response':deepcopy(raw),'errors':deepcopy(errors),
                            'instruction':'Correct only these JSON/edit protocol errors in the complete response using edit_catalog. Keep the original business constraints and findings. No candidate is committed until protocol and the original compiler both pass.'}
                try:
                    validate_unit_patch(wp,working,raw,uid,cohort_owners,payload['available_new_object_slots'])
                except PlanConflict as error:
                    record(entry,findings=deepcopy(error.findings),status='locally_rejected')
                    signature=json.dumps([raw,error.findings],ensure_ascii=False,sort_keys=True)
                    if signature in seen or local_attempt==2:raise
                    seen.add(signature)
                    payload['edit_feedback']={'previous_response':raw,'compiler_findings':error.findings,
                        'instruction':'Return the complete corrected change set against the supplied original unit. Retain successful changes from previous_response; correct the exact remaining compiler findings using the shared facts and edit_catalog. Original assigned demand, business and limits remain.'}
                    continue
                record(entry,status='locally_compiled')
                return uid,raw
            raise PlanConflict(payload['findings'])
        if audit.get('unit_proposals'):
            audit.setdefault('prior_unit_proposals',[]).append(deepcopy(audit['unit_proposals']))
        audit["unit_proposals"]={};save()
        def on_result(result):
            uid,raw=result
            with lock:
                audit['unit_proposals'][uid]=deepcopy(raw)
                audit['active_batch']['unit_proposals'][uid]=deepcopy(raw);save()
        config.pmap(run, jobs, workers=min(16,len(jobs) or 1),on_result=on_result)
        attempts_after=sum(len(rows) for rows in audit.get('edit_attempts',{}).values())
        if attempts_after<=attempts_before:
            raise PlanConflict([finding('repair_no_progress','A joint draft requires a newly consumed original author slot')])
        # Each bounded unit proposal is applied to the same immutable transaction base.
        for uid,raw in audit["unit_proposals"].items():
            if raw.get("changes") or raw.get("cohort_changes"):
                working=apply_repair(working,raw)
        working=derived_dependencies(working)
        draft={'iteration':iteration+1,'candidate':deepcopy(working),'unit_proposals':deepcopy(audit['unit_proposals']),
               'author_slots_before':attempts_before,'author_slots_after':attempts_after}
        audit.setdefault('drafts',[]).append(draft)
        save()
        try:
            compile_plan(wp,working)
        except PlanConflict as error:
            findings=deepcopy(error.findings);draft['findings']=deepcopy(findings)
            audit['active_batch']['status']='joint_rejected';save()
            iteration+=1
        else:
            draft['status']='jointly_compiled';audit['active_batch']['status']='jointly_compiled'
            save()
            return working
