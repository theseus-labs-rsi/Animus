"""Atomic business revisions of schema meaning and its dependent instance plan.

Semantic aliases are authored. The existing seed and joint compiler validate
the revised layout, and the original business reviewer judges its meaning.
"""
from copy import deepcopy
import json

import config
from pipeline.capability_contract import digest


SYSTEM = """你是同一业务世界的架构师，修订尚未生成事实的业务布局。
business和原seed是依据，review为可错的意见。读取compiled_execution中的共同身份、字段写入和逐线考查范围，逐项区分真实业务问题与误读。相同slot跨单元引用同一对象，别名不用于掩盖不存在的对象；某条能力条件适用于其carrier，原seed明确的额外业务要求照常保留。
保留正确对象、关系、事件和各线候选配额。对于确属同义的字段，可用field_aliases合并到已有规范字段；程序会统一所有对应字段声明、事件写入、观察和carrier字段，并合并重复观察。不得删除seed必需字段、放宽states/range/monotonic或修改seed/实体/时间/配额上限。
合并可能让两项需求落到同一事实，须通过changes改配其中一项到自然、独立的已有业务载体。只有通过原construction条件的载体才可承担该线。输入列出了字段与计划事件；已有intrinsic numeric字段可由事实作者提供合法观测，event-owned字段必须具有足够的真实写入期。具体数值及答案由后续事实作者生成。
changes与cohort_changes沿用按unit_id、collection、key或selector的添加/替换/删除；选择器基于field_aliases应用后的计划。保留成功修改；格式或编译失败时针对具体反馈返回完整修正事务。公开安排和原业务机制保持。
若原任务的必需过程缺少可表达的声明，可用blueprint_additions添加有业务依据的新实体类型、关系类型、事件类型或因果规则，并在同一changes事务中安排对应事实。新增声明沿用输入blueprint的完整原schema；不能覆盖已有id、修改或删除原声明，不能放宽原状态与因果约束。新增事实仍须通过原seed、联合编译与原业务复核。数值轨迹可以由真实观测事件承载，不能为制造轨迹重复执行不可逆生命周期。
严格JSON：{"decision":"repair|unresolved","reason":"业务依据与对原意见的逐项处理","field_aliases":[{"entity_type":"已有类型","source_field":"已有同义字段","target_field":"保留的已有字段","reason":"二者确实表达同一含义的依据"}],"blueprint_additions":{"entity_types":[],"relation_types":[],"event_types":[],"causal_rules":[]},"changes":[],"cohort_changes":[]}。
禁止添加gt、answer、winner或固定数值轨迹；无法在原seed和限额内实现时返回unresolved及具体原因。"""


SYSTEM+='\n同义字段统一采用原seed的规范名称与完整声明；程序会同步投影本事务中的字段引用。两个名称均为seed必需字段时，保留两个独立声明并报告具体冲突。'


def canonical_aliases(wp, aliases):
    """Select vocabulary from the authoritative seed for authored synonyms."""
    from pipeline.seed_pack import core_requirements
    from pipeline.instance_plan import PlanConflict, finding
    required={(t['id'],f['name']) for t in core_requirements(wp['seed_contract'])['entity_types']
              for f in t.get('fields',[])} if wp.get('seed_contract') else set()
    result=[]
    for alias in aliases:
        row=deepcopy(alias);kind,source,target=(row[k] for k in ('entity_type','source_field','target_field'))
        if (kind,source) in required and (kind,target) in required:
            raise PlanConflict([finding('business_revision_alias','Both fields are required by the original seed; preserve their separate declarations')])
        if (kind,source) in required:row.update(source_field=target,target_field=source)
        result.append(row)
    return result


def _record_shapes(wp, proposal, edits=None):
    """Guard structural reads without running semantic compilation early.

    Alias and declaration edits are still one private transaction. No fields
    are inherited into an author's incomplete replacement by this gate.
    """
    from pipeline.instance_plan import PlanConflict, finding
    specs={
        'objects':({'slot':str,'type':str},{'slot','type'}),
        'relations':({'slot':str,'type':str,'from':str,'to':str},{'slot','type','from','to','session'}),
        'events':({'slot':str,'type':str,'participants':dict,'session':int},{'slot','type','participants','session','caused_by'}),
        'observations':({'entity':str,'field':str,'sessions':list},{'entity','field','sessions','mechanism'}),
        'obligations':({'id':str,'line':str,'carrier':dict},{'id','line','carrier','subtype'}),
        'publications':({'fact_slot':str,'channel':str,'session':int},{'fact_slot','channel','session'}),
    }
    problems=[]
    def text(value):return isinstance(value,str) and bool(value.strip())
    def fail(uid, collection, row, issues, position=None):
        key=row.get('slot',row.get('id')) if isinstance(row,dict) else None
        problems.append(finding('business_revision_record_shape','Business revision requires complete typed records',
            [uid]+([key] if text(key) else []),evidence={'responsible_unit':uid,'collection':collection,
                'record_index':position,'record_id':key,'issues':issues,'replacement_semantics':'complete_record_not_merge'}))
    types={row['id'] for row in wp['world_blueprint']['entity_types']}
    relations={row['id']:row for row in wp['world_blueprint']['relation_types']}
    objects={}
    for unit in proposal['units']:
        uid=unit['unit_id']
        for collection,(required,_allowed) in specs.items():
            rows=unit.get(collection,[])
            if not isinstance(rows,list):fail(uid,collection,None,['collection must be an array']);continue
            for position,row in enumerate(rows):
                if not isinstance(row,dict):fail(uid,collection,row,['record must be an object'],position);continue
                issues=[]
                for field,kind in required.items():
                    value=row.get(field)
                    valid=(text(value) if kind is str else type(value) is int if kind is int else isinstance(value,kind))
                    if not valid:issues.append(field+' must be '+kind.__name__)
                if collection=='objects' and text(row.get('type')) and row['type'] not in types:
                    issues.append('type must be a declared object type')
                if collection=='relations':
                    temporal=relations.get(row.get('type'),{}).get('temporal') if text(row.get('type')) else False
                    if ('session' in row or temporal) and type(row.get('session')) is not int:issues.append('session must be int')
                if collection=='events':
                    participants=row.get('participants')
                    if isinstance(participants,dict) and any(not text(k) or not text(v) for k,v in participants.items()):
                        issues.append('participants must map role names to object slots')
                    if 'caused_by' in row and not text(row['caused_by']):issues.append('caused_by must be str')
                if collection=='observations':
                    if isinstance(row.get('sessions'),list) and any(type(s) is not int for s in row['sessions']):issues.append('sessions must contain int')
                    if 'mechanism' in row and not isinstance(row['mechanism'],str):issues.append('mechanism must be str')
                if collection=='obligations' and isinstance(row.get('carrier'),dict):
                    carrier=row['carrier']
                    if not isinstance(carrier.get('entities'),list) or any(not text(e) for e in carrier.get('entities',[])):
                        issues.append('carrier.entities must be an array of object slots')
                    if carrier.get('field') is not None and not isinstance(carrier['field'],str):issues.append('carrier.field must be str when supplied')
                if issues:fail(uid,collection,row,issues,position)
                if collection=='objects' and text(row.get('slot')):objects[row['slot']]=row
    if problems:raise PlanConflict(problems)
    for unit in proposal['units']:
        for collection in ('observations','obligations'):
            for position,row in enumerate(unit.get(collection,[])):
                refs=[row['entity']] if collection=='observations' else row['carrier']['entities']
                missing=[ref for ref in refs if ref not in objects]
                if missing:fail(unit['unit_id'],collection,row,['projection references missing objects: '+', '.join(missing)],position)
    for change in (edits or {}).get('changes',[]):
        collection=change['collection'];value=change.get('value')
        if change['operation']!='remove' and collection in specs and isinstance(value,dict):
            unknown=sorted(set(value)-specs[collection][1])
            if unknown:fail(change['unit_id'],collection,{**value,('id' if collection=='obligations' else 'slot'):change.get('key')},
                            ['unsupported fields would not become planned facts: '+', '.join(unknown)])
    if problems:raise PlanConflict(problems)


def apply(wp, proposal, revision):
    from pipeline.instance_plan import PlanConflict, finding, index, apply_repair, compile_plan
    from pipeline.world_blueprint import relation_owner_side
    if not isinstance(revision, dict) or revision.get('decision') != 'repair':
        raise PlanConflict([finding('business_revision_shape','Expected an authored repair transaction')])
    aliases=revision.get('field_aliases',[])
    if not isinstance(aliases,list) or len(aliases)>32:
        raise PlanConflict([finding('business_revision_shape','field_aliases must be a bounded array')])
    updated, plan=deepcopy(wp),deepcopy(proposal)
    bp=updated['world_blueprint']
    additions=revision.get('blueprint_additions',{})
    groups={'entity_types','relation_types','event_types','causal_rules'}
    if not isinstance(additions,dict) or set(additions)-groups:
        raise PlanConflict([finding('business_revision_shape','Blueprint additions use the original declaration collections')])
    for group,rows in additions.items():
        if not isinstance(rows,list) or len(rows)>32:
            raise PlanConflict([finding('business_revision_shape','Declaration additions must be bounded arrays')])
        existing={row['id'] for row in bp[group]}
        for row in rows:
            key=row.get('id') if isinstance(row,dict) else None
            if not isinstance(key,str) or not key or key in existing:
                raise PlanConflict([finding('business_revision_declaration','New declarations need a unique id; original declarations remain immutable',[key] if isinstance(key,str) else [])])
            existing.add(key)
            bp[group].append(deepcopy(row))
    # Validate new declarations and their references with the original schema.
    # All edits remain private until the dependent plan passes its joint compiler.
    from pipeline.world_blueprint import normalize_world_blueprint,WorldBlueprintError
    try:
        bp=normalize_world_blueprint(updated)
    except WorldBlueprintError as error:
        raise PlanConflict([finding('blueprint_compilation',str(error))]) from error
    updated['world_blueprint']=bp
    old_bp=deepcopy(bp)
    kinds={t['id']:t for t in bp['entity_types']}
    mapping={}
    for alias in aliases:
        if not isinstance(alias,dict) or set(alias)-{'entity_type','source_field','target_field','reason'}:
            raise PlanConflict([finding('business_revision_shape','Field aliases contain only type, source, target and reason')])
        kind,source,target=alias.get('entity_type'),alias.get('source_field'),alias.get('target_field')
        if any(not isinstance(value,str) or not value for value in (kind,source,target)):
            raise PlanConflict([finding('business_revision_alias','Type and field identifiers must be declared nonempty strings')])
        fields={f['name']:f for f in kinds.get(kind,{}).get('fields',[])}
        if (source not in fields or target not in fields or source==target or (kind,source) in mapping
                or not isinstance(alias.get('reason'),str) or not alias['reason'].strip()):
            raise PlanConflict([finding('business_revision_alias','Alias needs two distinct declared fields and its business reason')])
        mapping[kind,source]=target
    resolved=canonical_aliases(wp,aliases)
    mapping={(r['entity_type'],r['source_field']):r['target_field'] for r in resolved}
    if any((kind,target) in mapping for (kind,_source),target in mapping.items()):
        raise PlanConflict([finding('business_revision_alias','Use direct aliases to the final canonical field')])
    _record_shapes(updated,plan)
    units,objects,_events,_relations,owners=index(plan)
    def rename(kind,field):return mapping.get((kind,field),field)
    for kind in bp['entity_types']:
        kind['fields']=[f for f in kind['fields'] if (kind['id'],f['name']) not in mapping]
    for row in bp['relation_types']:
        old=next(r for r in old_bp['relation_types'] if r['id']==row['id'])
        side=relation_owner_side(old_bp,old)
        row['field']=rename(old['from_type'] if side=='from' else old['to_type'],row['field'])
    for event in bp['event_types']:
        effects=[]
        for effect in event['effect_fields']:
            field=rename(event['roles'][effect['role']],effect['field'])
            rewritten=dict(effect,field=field)
            if rewritten not in effects:effects.append(rewritten)
        event['effect_fields']=effects
    observations={}
    for unit in units.values():
        for row in unit['observations']:
            field=rename(objects[row['entity']]['type'],row['field'])
            key=row['entity'],field
            merged=observations.setdefault(key,{'entity':row['entity'],'field':field,'sessions':[], 'mechanism':[]})
            merged['sessions']=sorted(set(merged['sessions'])|set(row['sessions']))
            if row.get('mechanism') and row['mechanism'] not in merged['mechanism']:merged['mechanism'].append(row['mechanism'])
        unit['observations']=[]
        for row in unit['obligations']:
            carrier=row['carrier'];field=carrier.get('field')
            replacements={rename(objects[e]['type'],field) for e in carrier['entities']}
            changed=replacements-{field}
            if len(changed)>1:
                raise PlanConflict([finding('business_revision_alias','Heterogeneous carrier requires an explicit authored field choice',[row['id']])])
            if changed:carrier['field']=changed.pop()
    for (entity,_field),row in observations.items():
        if row['mechanism']:row['mechanism']='；'.join(row['mechanism'])
        else:row.pop('mechanism')
        units[owners[entity]]['observations'].append(row)
    if revision.get('changes') or revision.get('cohort_changes'):
        edits={k:deepcopy(revision[k]) for k in ('decision','reason','changes','cohort_changes') if k in revision}
        plan=apply_repair(plan,edits)
        _record_shapes(updated,plan,edits)
        # Resolve authored field references to the canonical seed vocabulary.
        new_units,new_objects,*_=index(plan)
        for unit in new_units.values():
            for row in unit['observations']:
                if row['entity'] in new_objects:row['field']=rename(new_objects[row['entity']]['type'],row['field'])
            for row in unit['obligations']:
                carrier=row['carrier'];field=carrier.get('field')
                choices={rename(new_objects[e]['type'],field) for e in carrier['entities'] if e in new_objects}
                if field is not None and len(choices)==1:carrier['field']=choices.pop()
    if digest(updated)==digest(wp) and digest(plan)==digest(proposal):
        raise PlanConflict([finding('business_revision_no_change','Business revision must make a concrete change')])
    projected={}
    for kind in bp['entity_types']:
        for field in kind['fields']:projected.setdefault(field['name'],deepcopy(field))
    profile=updated.setdefault('domain_profile',{})
    profile['field_schema']=list(projected.values())
    profile['state_machines']=[{'field':f['name'],'states':f['states']} for f in projected.values() if f.get('states')]
    revised=compile_plan(updated,plan)
    revised['business_layout_projection']={'authored_field_aliases':deepcopy(aliases),'canonical_field_aliases':resolved}
    if any(additions.values()):revised['business_layout_projection']['blueprint_additions']=deepcopy(additions)
    return revised,plan


def _business_protocol_errors(wp, raw):
    """Separate response shape from the original business/compile decision."""
    from pipeline.plan_transactions import _protocol_errors
    if not isinstance(raw,dict):return [{'path':'$','message':'Expected a business revision JSON object'}]
    if raw.get('decision')=='unresolved':return _protocol_errors(wp,raw)
    errors=[]
    def text(v):return isinstance(v,str) and bool(v.strip())
    def bad(path,message):errors.append({'path':path,'message':message})
    allowed={'decision','reason','field_aliases','blueprint_additions','changes','cohort_changes'}
    if set(raw)-allowed:bad('$','Unknown business revision keys')
    aliases=raw.get('field_aliases',[])
    if not isinstance(aliases,list) or len(aliases)>32:bad('field_aliases','Expected at most 32 alias objects')
    else:
        for i,row in enumerate(aliases):
            path=f'field_aliases[{i}]'
            if not isinstance(row,dict):bad(path,'Expected an alias object');continue
            if set(row)-{'entity_type','source_field','target_field','reason'}:bad(path,'Unknown alias fields')
            for field in ('entity_type','source_field','target_field','reason'):
                if not text(row.get(field)):bad(path+'.'+field,'Expected a nonempty string')
    additions=raw.get('blueprint_additions',{})
    groups={'entity_types','relation_types','event_types','causal_rules'}
    extended=deepcopy(wp)
    if not isinstance(additions,dict) or set(additions)-groups:bad('blueprint_additions','Expected only original declaration collections')
    else:
        for group,rows in additions.items():
            if not isinstance(rows,list) or len(rows)>32:bad('blueprint_additions.'+group,'Expected at most 32 declaration objects');continue
            for i,row in enumerate(rows):
                path=f'blueprint_additions.{group}[{i}]'
                if not isinstance(row,dict):bad(path,'Expected a declaration object');continue
                if not text(row.get('id')):bad(path+'.id','Expected a nonempty declaration id string')
                # These fields are read structurally by the unchanged schema
                # compiler. Values/references and feasibility remain its job.
                shapes={'entity_types':{'fields':list},'relation_types':{},
                        'event_types':{'roles':dict,'effect_fields':list},'causal_rules':{'shared_roles':list}}
                for field,kind in shapes[group].items():
                    if field in row and not isinstance(row[field],kind):bad(path+'.'+field,'Expected '+kind.__name__)
                for field in ('count','min_count','delay_sessions'):
                    if field in row and type(row[field]) is not int:bad(path+'.'+field,'Expected an integer')
                for field in ('from_type','to_type','field','trigger_event','effect_event','noun','label'):
                    if field in row and not text(row[field]):bad(path+'.'+field,'Expected a nonempty string')
                if isinstance(row.get('roles'),dict) and not all(text(k) and text(v) for k,v in row['roles'].items()):bad(path+'.roles','Expected role-to-type strings')
                if isinstance(row.get('effect_fields'),list):
                    for j,effect in enumerate(row['effect_fields']):
                        if not isinstance(effect,dict) or not text(effect.get('role')) or not text(effect.get('field')):bad(path+f'.effect_fields[{j}]','Expected role and field strings')
                if isinstance(row.get('shared_roles'),list) and not all(text(v) for v in row['shared_roles']):bad(path+'.shared_roles','Expected role strings')
                if group=='relation_types' and text(row.get('id')):extended['world_blueprint'][group].append(deepcopy(row))
    edits={k:deepcopy(raw[k]) for k in ('decision','reason','changes','cohort_changes') if k in raw}
    # The business projection already accepts an unspecified carrier field and
    # an empty optional observation mechanism. Preserve that wire contract;
    # this validation-only view never rewrites the saved author response.
    for change in edits.get('changes',[]) if isinstance(edits.get('changes',[]),list) else []:
        value=change.get('value') if isinstance(change,dict) else None
        if not isinstance(value,dict):continue
        if change.get('collection')=='obligations' and isinstance(value.get('carrier'),dict):
            if value['carrier'].get('field',False) is None:value['carrier'].pop('field')
        if change.get('collection')=='observations' and value.get('mechanism',False)=='':value.pop('mechanism')
    # Delegate complete record/key/selector typing to the existing patch wire
    # contract. No records are applied and no compiler gate is relaxed here.
    errors.extend(_protocol_errors(extended,edits))
    return errors


def _business_checkpoint(wp, proposal, audit, diagnostic, checkpoint, max_attempts):
    """Validate cumulative slots; paid authority/admission belongs to recovery.

    The explicit recovery receipt binds these exact rows after checking their
    physical trace and the user's additional allowance. This gate never clears
    or edits a historical author/reviewer response, and never retries a slot
    whose outcome is uncertain.
    """
    from pipeline.instance_plan import PlanConflict, finding
    def require(ok,message):
        if not ok:raise PlanConflict([finding('business_checkpoint',message,kind='design')])
    rows=audit.get('business_revisions')
    require(isinstance(rows,list) and 1<=len(rows)<3,'Explicit business recovery needs one or two completed original slots')
    expected={'version':'business-revision-checkpoint/v1','candidate_sha256':digest(proposal),
        'diagnostic_sha256':digest(diagnostic),'business_revisions_sha256':digest(rows),
        'completed_business_attempts':len(rows),'max_business_attempts':3,'max_additional_physical_requests':36}
    require(isinstance(checkpoint,dict) and checkpoint==expected,'Business checkpoint binding or approved bounds changed')
    require(type(max_attempts) is int and max_attempts==3,'Explicit business recovery retains total semantic limit three')
    require(audit.get('uncompiled_diagnostic')==diagnostic and audit.get('uncompiled_candidate_sha256')==digest(proposal),
        'Recovery must retain the original uncompiled diagnostic and candidate')
    require(audit.get('status') not in ('accepted','design_unresolved'),'Terminal accepted or unresolved business decisions cannot be reopened')
    for number,row in enumerate(rows,1):
        require(isinstance(row,dict) and type(row.get('attempt')) is int and row['attempt']==number,
            'Business author slots must remain contiguous and monotonic')
        raw=row.get('proposal');protocol=row.get('protocol_attempts')
        require(isinstance(raw,dict) and raw.get('decision')=='repair' and not _business_protocol_errors(wp,raw),
            'Completed business slot lacks a protocol-valid substantive repair')
        require(isinstance(protocol,list) and 1<=len(protocol)<=3,'Business protocol history is absent or exceeds its original bound')
        for n,item in enumerate(protocol,1):
            require(isinstance(item,dict) and type(item.get('protocol_attempt')) is int and item['protocol_attempt']==n,
                'Business protocol slots must remain contiguous')
            require('proposal' in item,'Pending or uncertain business protocol calls cannot be replayed')
            errors=_business_protocol_errors(wp,item['proposal'])
            if n<len(protocol):
                require(item.get('status')=='protocol_rejected' and bool(errors) and item.get('protocol_errors')==errors,
                    'Earlier protocol response was valid or its rejection evidence changed')
            else:
                require(item.get('status')=='protocol_valid' and not errors and item['proposal']==raw,
                    'Last protocol response must be the saved original substantive proposal')
        require(isinstance(row.get('findings'),list) and bool(row['findings']),'Rejected business slot needs its full feedback')
        review=row.get('business_review');call=row.get('business_review_call')
        if row.get('status')=='compiler_rejected':
            require(not review and not call,'Compiler-rejected slot cannot have a later review call')
        else:
            require(row.get('status') in ('response_received','business_review_rejected') and isinstance(review,dict)
                and review.get('decision')=='repair' and isinstance(call,dict) and call.get('status')=='response_received'
                and call.get('response')==review,'Only a fully returned negative review permits the next semantic slot')
    return len(rows)


def revise(wp, proposal, feedback, tracer, audit, max_attempts=3, initial_revision=None, checkpoint=None, uncompiled_diagnostic=None, business_resume_checkpoint=None):
    try:
        return _revise(wp,proposal,feedback,tracer,audit,max_attempts=max_attempts,initial_revision=initial_revision,
            checkpoint=checkpoint,uncompiled_diagnostic=uncompiled_diagnostic,business_resume_checkpoint=business_resume_checkpoint)
    finally:
        if checkpoint is not None:checkpoint(audit)


def _revise(wp, proposal, feedback, tracer, audit, max_attempts=3, initial_revision=None, checkpoint=None, uncompiled_diagnostic=None, business_resume_checkpoint=None):
    from pipeline.instance_plan import PlanConflict, finding, execution_contract, compile_plan, REPAIR_SYSTEM
    from pipeline.blueprint_feasibility import assess
    if not 1<=max_attempts<=3:raise ValueError('Business revision allowance must be 1 to 3')
    resume=business_resume_checkpoint is not None
    if audit.get('business_revisions') and not resume:
        raise PlanConflict([finding('business_recovery_required','Existing business revisions require explicit recovery; attempts cannot restart',kind='design')])
    if resume and (uncompiled_diagnostic is None or initial_revision is not None):
        raise PlanConflict([finding('business_checkpoint','Business recovery requires the original independent diagnostic and no reused author call',kind='design')])
    consumed=_business_checkpoint(wp,proposal,audit,uncompiled_diagnostic,business_resume_checkpoint,max_attempts) if resume else 0
    def save():
        if checkpoint is not None:checkpoint(audit)
    payload={'business':wp.get('seed_contract'),'blueprint':wp['world_blueprint'],
        'proposal':deepcopy(proposal),
        'requirements':wp['supply_plan']['requirements'],'limits':wp['supply_plan']['world_limits'],
        'feedback':deepcopy(feedback)}
    system=SYSTEM
    if uncompiled_diagnostic is None:
        payload['compiled_execution']=execution_contract(compile_plan(wp,proposal))
    else:
        if type(max_attempts) is not int or max_attempts!=(3 if resume else 1) or initial_revision is not None:
            raise PlanConflict([finding('business_diagnostic_scope','Uncompiled escalation consumes exactly the one original business revision; no reused unbound author reply',kind='design')])
        from pipeline.blueprint_feasibility import validate_uncompiled_diagnostic
        diagnostic_input=validate_uncompiled_diagnostic(wp,proposal,uncompiled_diagnostic)
        if uncompiled_diagnostic.get('decision')!='repair':
            raise PlanConflict([finding('business_diagnostic_decision','Only a bound independent repair diagnosis may invoke the business author',kind='design')])
        payload=deepcopy(diagnostic_input)
        payload.update(input_mode='uncompiled_candidate_diagnosed',independent_diagnostic={
            key:deepcopy(uncompiled_diagnostic[key]) for key in
            ('version','phase','candidate_sha256','review_input_hash','decision','responsibility_targets')})
        payload['independent_diagnostic'].update(review=deepcopy(uncompiled_diagnostic['review']['review']),
            protocol=deepcopy(uncompiled_diagnostic['review']['protocol']))
        system=SYSTEM.replace('读取compiled_execution中的共同身份、字段写入和逐线考查范围',
            '读取uncompiled_candidate、blueprint和requirements中的共同身份、字段写入和逐线考查范围')
        system+='\n本次候选尚未通过联合编译，input_mode明确标为uncompiled_candidate_diagnosed；没有compiled_execution或已通过声明。independent_diagnostic是原独立审阅者对完整失败证据的真实repair意见，不代表通过。局部作者已用次数不重置。输出仍须通过原完整联合编译与独立业务复核后才提交。'
        if resume:
            system+='\n本次为用户明确追加后的累计业务修订：总语义上限3，原B1及其失败不删除、不重发。原诊断中的remaining_business_attempts=1是历史绑定，当前剩余次数以authorized_business_continuation为准。完整读取先前业务回复和所有compiler/reviewer负面意见；对同一未提交原候选返回完整修正事务，不能仅重复上次失败方案。协议纠正不增加语义槽，内容或能力错误只能使用下一个已授权语义槽。'
            payload['authorized_business_continuation']={'version':'business-cumulative-semantic/v2',
                'total_semantic_limit':3,'consumed_semantic_attempts':consumed,'semantic_attempt':consumed+1,
                'max_additional_physical_requests':36,'max_protocol_attempts':3,'max_json_attempts_per_protocol':3,
                'review_required_after_compilation':True}
            audit.setdefault('business_resume_history',[]).append({'checkpoint':deepcopy(business_resume_checkpoint),
                'prior_status':audit.get('status'),'start_next_attempt':consumed+1})
            audit['status']='business_repair_running'
        else:
            system+='\n本次只有原唯一一次业务修订。'
            audit.update(input_mode='uncompiled_candidate_diagnosed',uncompiled_diagnostic=deepcopy(uncompiled_diagnostic),
                uncompiled_candidate_sha256=digest(proposal),original_business_revision_allowance=1,
                author_protocol={'version':'uncompiled-business-author-protocol/v1','max_business_attempts':1,
                    'max_protocol_attempts':3,'max_json_attempts_per_protocol':3,'max_physical_author_requests':9})
        save()
    attempts=audit.setdefault('business_revisions',[]);seen=set()
    if resume:
        for prior in attempts:seen.add(digest({'proposal':prior['proposal'],'findings':prior['findings']}))
        payload['revision_feedback']={'previous_response':deepcopy(attempts[-1]['proposal']),
            'findings':deepcopy(attempts[-1]['findings']),
            'instruction':'Return the complete corrected transaction against the unchanged original blueprint and candidate; address all prior compiler and reviewer negatives. Observation schedules are not new state writes.'}
    for number in range(consumed,max_attempts):
        if resume:
            payload['authorized_business_continuation'].update(semantic_attempt=number+1,consumed_semantic_attempts=number)
            payload['business_failure_history']=deepcopy(attempts)
        reused=number==0 and initial_revision is not None
        entry={'attempt':number+1,'status':'revalidating' if reused else 'requested','revalidated_from_parent':reused};attempts.append(entry);save()
        request=deepcopy(payload)
        if uncompiled_diagnostic is not None:entry['protocol_attempts']=[];save()
        for protocol_number in range(1,4 if uncompiled_diagnostic is not None else 2):
            protocol=None
            if uncompiled_diagnostic is not None:
                protocol={'protocol_attempt':protocol_number,'status':'requested'}
                entry['protocol_attempts'].append(protocol);save()
            try:
                from execution_control import check_stop
                check_stop()
                if callable(getattr(tracer,'check_global_stop',None)):tracer.check_global_stop()
                raw=deepcopy(initial_revision) if reused else tracer.chat_json('council.instance_plan_business_repair',[
                    {'role':'system','content':system+'\n原计划记录编辑规则：\n'+REPAIR_SYSTEM},
                    {'role':'user','content':json.dumps(request,ensure_ascii=False)}],
                    model=config.STRUCTURE_MODEL,temperature=.2,max_tokens=32768,retries=3,response_format={'type':'json_object'})
            except BaseException as error:
                if protocol is not None:protocol.update(status='call_failed',error_type=type(error).__name__)
                entry.update(status='call_failed',error_type=type(error).__name__);save();raise
            if protocol is None:break
            protocol.update(status='response_received',proposal=deepcopy(raw));save()
            if isinstance(raw,dict) and '__error__' in raw:
                protocol['status']='provider_error';save();break
            errors=_business_protocol_errors(wp,raw)
            if not errors:protocol['status']='protocol_valid';save();break
            protocol.update(status='protocol_rejected',protocol_errors=deepcopy(errors));save()
            if protocol_number==3:
                entry.update(status='protocol_exhausted',proposal=deepcopy(raw),findings=[finding('business_revision_protocol_exhausted',
                    'Original business author protocol allowance exhausted',evidence={'errors':errors,'max_protocol_attempts':3,'max_json_attempts':3})]);save()
                audit.update(status='business_repair_required',findings=deepcopy(entry['findings']));save()
                raise PlanConflict(entry['findings'])
            request=deepcopy(payload)
            request['protocol_feedback']={'previous_response':deepcopy(raw),'errors':deepcopy(errors),
                'instruction':'Correct only the response protocol and complete record types against the unchanged original candidate, diagnostic and schema. Do not invent missing business values in code. This semantic revision slot is not renewed; the first protocol-valid response must pass the original compiler and final independent reviewer.'}
        entry.update(proposal=deepcopy(raw),status='response_received');save()
        if not isinstance(raw,dict) or '__error__' in raw:
            entry['status']='provider_error';save()
            from execution_control import global_stop_exception
            stopped=global_stop_exception(raw)
            if stopped is not None:raise stopped
            raise RuntimeError('Business revision provider failure: '+str(raw))
        if raw.get('decision')=='unresolved':
            audit['status']='design_unresolved'
            entry['status']='unresolved';save()
            raise PlanConflict([finding('business_unresolved',str(raw.get('reason')),kind='design')])
        try:
            revised,_candidate=apply(wp,proposal,raw)
            if resume:
                candidate_hash=digest({'blueprint':revised['world_blueprint'],'candidate':_candidate})
                if any(old.get('compiled_candidate_sha256')==candidate_hash and
                       (old.get('business_review') or {}).get('decision')=='repair' for old in attempts[:-1]):
                    raise PlanConflict([finding('business_revision_no_progress',
                        'A previously rejected compiled candidate requires substantive author changes before another independent review',
                        kind='business',evidence={'compiled_candidate_sha256':candidate_hash})])
                entry['compiled_candidate_sha256']=candidate_hash;save()
        except PlanConflict as error:
            entry.update(findings=deepcopy(error.findings),status='compiler_rejected');save()
        else:
            entry['business_review_call']={'status':'requested'};save()
            try:
                review=assess(revised,tracer,feedback={'business_instance_plan':revised['business_instance_plan'],
                    'original_request':feedback,'author_revision':raw,'compiled_projection':revised['business_layout_projection']})
            except BaseException as error:
                entry['business_review_call'].update(status='call_failed',error_type=type(error).__name__);save();raise
            entry['business_review_call'].update(status='response_received',response=deepcopy(review));save()
            entry['business_review']=deepcopy(review)
            if review['decision']=='accept':
                revised.setdefault('blueprint_revisions',[]).append({'before_hash':digest(wp['world_blueprint']),
                    'after_hash':digest(revised['world_blueprint']),'author':deepcopy(raw),'review':deepcopy(review)})
                audit.update(status='accepted',after_hash=digest(revised),after=deepcopy(revised))
                return revised
            if review['decision']=='unresolved':
                audit['status']='design_unresolved'
                raise PlanConflict([finding('business_unresolved','Original business review remains unresolved',kind='design',evidence=review)])
            entry['findings']=[finding('business_repair','Original business review requests revision',kind='business',evidence=review)]
            if resume:entry['status']='business_review_rejected';save()
        signature=digest({'proposal':raw,'findings':entry['findings']})
        if signature in seen and not resume:break
        seen.add(signature)
        payload['revision_feedback']={'previous_response':raw,'findings':entry['findings'],
            'instruction':'Return the complete corrected transaction against the unchanged original blueprint and proposal; retain successful changes.'}
    audit.update(status='business_repair_required',findings=deepcopy(attempts[-1]['findings']))
    raise PlanConflict(attempts[-1]['findings'])
