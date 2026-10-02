"""Business feasibility review inside the original whitepaper/world loop."""
from copy import deepcopy
import hashlib
import json

import config
from pipeline.seed_pack import validate_seed_blueprint
from pipeline.seed_world import seed_business_context
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint

VERSION = "blueprint-feasibility/v1"
REVIEW_PROTOCOL_VERSION = "blueprint-review-protocol/v1"
CONSTRAINT_PROTOCOL_VERSION = "blueprint-constraint-protocol/v1"
REVIEW_PROTOCOL_ATTEMPTS = 3
JSON_ATTEMPTS_PER_PROTOCOL = 3
SYSTEM = """你复核原 pipeline 的候选白皮书能否表达原任务，不生成世界、题目或评分。
逐项读 business 中的必需机制，构想一个具体而简短的完整业务过程，检查 blueprint 的字段、关系、事件、状态与因果约束是否允许该过程成立。
特别核对会反复确认、重开、恢复或修订的过程。fields.states 在当前编译器中表示不可逆的单向生命周期，不能用于可循环运行状态；后者用 kind=status 并省略 states，由后续世界语义审阅判断实际变化。
不要因为存在字段、事件名称或数量就判机制可实现。核对真实前态、依赖变更、触发、后续响应能否由同一对象上的事件表达；精确因果延迟、字段所有权和同一期同字段唯一写入也须相容。instance_execution.numeric_field_ownership 明确区分事件拥有字段与可由真实业务观测赋值的原生字段；numeric_event_writes 没列出原生字段，不代表其不能变化。只有原任务确实要求某字段由具名事件写入时，才可据缺少事件写入判冲突。
instance_execution.declared_object_minima 是每类型不可降低的计划下限。不同 slot 是不同实体，不得仅凭名称相似断定重复身份。原任务要求若干核心对象保持独立，若未说恰好、仅有或最多该数量，不得推断整个世界只能有该数量。任何合并建议须先核算不违反声明下限；若确有原任务总数上限与声明下限冲突，应明确指向上游 schema 而非建议删掉计划对象。
business/任务是依据；普通行业习惯、可选示例和能力陷阱不能升级为额外必需规则。允许合法未知、不同等价流程和未触发实例，只要求每项必需机制至少有一种合法完整见证。
feedback 是下游可错的失败证据，需独立判断。不要为了迁就现有程序错误删掉业务要求，也不要把合成过程说成已经生成或已经通过世界验收。
输出严格 JSON：{"decision":"accept|repair|unresolved","reason":"整体依据",
"mechanism_checks":[{"mechanism_id":"逐一复制全部机制id","walkthrough":"按具体对象简述可行过程或阻塞所在","status":"feasible|blocked|unresolved"}],
"issues":[{"finding":"具体冲突字段/约束及原任务依据","suggestion":"最小修订建议或所缺信息"}]}。
accept 需要全部机制 feasible 且 issues=[]；repair 需要至少一个具体 issue；无法判断或原任务本身冲突用 unresolved。"""
REPAIR_SYSTEM = """你是原白皮书架构师，处理世界作者报告的约束矛盾。只提出有业务依据的最小字段约束修订。
可编辑字段的可选 states/range/monotonic。states 代表单向不可逆生命周期；可循环状态可删除 states，但必须保留字段及业务复核义务。不得改 seed、字段名/类型/单位、实体数量、时间安排、事件/关系定义或因果要求。
不能仅因生成器多次失败就放宽约束。结合 business、blueprint、feedback 判断是原声明过强/错误，还是生成器应修自己的事件。若当前范围没有有据修法，edits=[]并说明未决原因。
严格 JSON：{"reason":"整体依据","edits":[{"entity_type":"原类型id","field":"原字段名","remove":["states"],"set":{},"reason":"原任务证据与修改含义"}]}。
remove/set 仅包含确实需改变的 states/range/monotonic 键，不能同一键同时 remove 和 set。"""


class BlueprintReviewRequested(WorldBlueprintError):
    def __init__(self, evidence):
        self.evidence = deepcopy(evidence)
        super().__init__("world author requested upstream blueprint review: " + str(evidence.get("reason")))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _call(tracer, step, system, payload, *, allow_invalid_shape=False):
    from execution_control import check_stop, global_stop_exception
    check_stop()
    if callable(getattr(tracer, 'check_global_stop', None)):tracer.check_global_stop()
    result = tracer.chat_json(step, [{"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=config.STRUCTURE_MODEL, temperature=0.2, max_tokens=8192,
        retries=JSON_ATTEMPTS_PER_PROTOCOL, strict_json=True, response_format={"type": "json_object"})
    stopped=global_stop_exception(result)
    if stopped is not None:raise stopped
    if (isinstance(result,dict) and '__error__' in result) or (not allow_invalid_shape and not isinstance(result,dict)):
        error=WorldBlueprintError("blueprint review execution failed: " + str(result))
        error.raw=deepcopy(result)
        raise error
    return result


PLAN_DIAGNOSTIC_VERSION = "uncompiled-plan-diagnostic/v1"
PLAN_DIAGNOSTIC_PHASE = "uncompiled_plan_escalation"
PLAN_DIAGNOSTIC_SYSTEM = SYSTEM + """\n本轮是原独立业务审阅者对未提交计划的责任诊断。mechanical_status=rejected_uncompiled：候选没有通过原编译器，绝不能假设其可执行，也没有 instance_execution。
完整原 seed、业务、schema、逐线 construction 需求、上限、候选和全部历史均在输入中。旧意见可错，请独立区分候选作者错误、载体/上层分配错误、schema表达能力不足和原任务矛盾；不得删需求、降低 quality 或把观察期等同真实写入期。
原业务架构修订只剩 remaining_business_attempts 次。repair 仅说明存在有原任务依据的上层修订方向，不代表任何事实、编译或业务验收已通过。无法给出有据方向则 unresolved；若你认为候选业务上可接受可返回 accept，但机械拒绝仍保留，程序不会因此进入作者或安装候选。
在原输出 schema 上增加 responsibility_targets 数组。repair 必须逐一覆盖 findings 的索引，每个目标只能选择 responsibility_catalog 已列原责任记录：
{"unit_id":"原unit", "collection":"原collection", "key":"原记录键", "finding_indices":[0], "reason":"这条原记录对该失败承担何种责任及原任务依据"}。
目标是责任依据，不是要求事先编造修改或真值。若更换载体才有可行方向，指向原失败 obligation 并说明，由原架构师在原 seed/限额内形成真实事务。不能伪造新 unit、obligation 或事实目标，不能把其他无关已有记录当责任目标。accept/unresolved 使用 responsibility_targets=[]。"""


class PlanDiagnosticBindingError(WorldBlueprintError):
    """A semantic target/binding is outside the frozen rejected transaction."""
    def __init__(self, message, *, review=None, protocol=None):
        self.review = deepcopy(review)
        self.protocol = deepcopy(protocol)
        super().__init__(message)


def _diagnostic_catalog(candidate, findings):
    """Resolve only explicit IDs/owners and their factual references, not prose.

    This is an identity projection of a rejected candidate, never compilation.
    The bound record content remains in the candidate and its record digest.
    """
    from pipeline.instance_plan import _record_key
    units = candidate.get('units') if isinstance(candidate, dict) else None
    if (not isinstance(units, list) or not units or not isinstance(findings, list)
            or not findings or any(not isinstance(row, dict) for row in findings)):
        raise PlanDiagnosticBindingError('Diagnostic requires original candidate units and nonempty compiler findings')
    records, unit_ids, identities = [], set(), set()
    for unit in units:
        uid = unit.get('unit_id') if isinstance(unit, dict) else None
        if not isinstance(uid, str) or not uid or uid in unit_ids:
            raise PlanDiagnosticBindingError('Diagnostic candidate has ambiguous unit identities')
        unit_ids.add(uid)
        for collection in ('objects', 'events', 'relations', 'observations', 'obligations', 'publications'):
            rows = unit.get(collection, [])
            if not isinstance(rows, list):
                raise PlanDiagnosticBindingError('Diagnostic candidate collection must be an array')
            for row in rows:
                key = _record_key(collection, row)
                identity = (uid, collection, key) if isinstance(key, str) and key else None
                if identity is None or identity in identities:
                    raise PlanDiagnosticBindingError('Diagnostic candidate record identity is missing or ambiguous')
                identities.add(identity)
                records.append((identity, row))
    def mentioned(value):
        result = set()
        if isinstance(value, dict):
            nested = (value.get('evidence') or {}).get('compiler_findings') if isinstance(value.get('evidence'), dict) else None
            if isinstance(nested, list) and nested:
                return mentioned(nested)  # The exhausted-owner wrapper is not a broader repair mandate.
            for key in ('affected_ids', 'responsible_unit', 'unit_id', 'record_id'):
                ids = value.get(key, [])
                if isinstance(ids, str): result.add(ids)
                elif isinstance(ids, list): result.update(v for v in ids if isinstance(v, str))
            # Structured evidence may contain the underlying compiler findings.
            for child in value.values():
                if isinstance(child, (dict, list)): result.update(mentioned(child))
        elif isinstance(value, list):
            for child in value: result.update(mentioned(child))
        return result
    def references(collection, row):
        values = []
        if collection == 'obligations': values = (row.get('carrier') or {}).get('entities', [])
        elif collection == 'observations': values = [row.get('entity')]
        elif collection == 'publications': values = [row.get('fact_slot')]
        elif collection == 'relations': values = [row.get('from'), row.get('to')]
        elif collection == 'events': values = list((row.get('participants') or {}).values())
        return {v for v in values if isinstance(v, str)}
    selected = {}
    for number, evidence in enumerate(findings):
        ids = mentioned(evidence)
        direct = [(identity, row) for identity, row in records
                  if identity[0] in ids or identity[2] in ids]
        refs = set().union(*(references(identity[1], row) for identity, row in direct)) if direct else set()
        for identity, row in records:
            if (identity, row) in direct or (identity[1] == 'objects' and identity[2] in refs):
                entry = selected.setdefault(identity, {'unit_id': identity[0], 'collection': identity[1],
                    'key': identity[2], 'record_sha256': digest(row), 'finding_indices': []})
                entry['finding_indices'].append(number)
    return [selected[key] for key in sorted(selected)]


def _diagnostic_correction(payload, raw, errors, number):
    request = deepcopy(payload)
    request['protocol_feedback'] = {'previous_response': deepcopy(raw), 'errors': deepcopy(errors),
        'attempt': number, 'max_attempts': REVIEW_PROTOCOL_ATTEMPTS,
        'instruction': 'Correct only output protocol against the unchanged rejected candidate and original evidence. '
            'Preserve substantive negative findings; unresolved is valid. Do not invent responsibility identities '
            'or change your semantic opinion merely to obtain repair/accept.'}
    return request


def _diagnostic_payload(wp, candidate, findings, history, source_binding,
                        retry_policy_binding, remaining_business_attempts):
    if type(remaining_business_attempts) is not int or remaining_business_attempts != 1:
        raise PlanDiagnosticBindingError('Diagnostic preserves the one remaining original business revision')
    if any(not isinstance(value, dict) or not value for value in
           (history, source_binding, retry_policy_binding, wp.get('seed_contract'))):
        raise PlanDiagnosticBindingError('Diagnostic requires complete original seed, history, source and policy bindings')
    supply = wp.get('supply_plan') or {}
    requirements, limits = supply.get('requirements'), supply.get('world_limits')
    if not isinstance(requirements, list) or not requirements or not isinstance(limits, dict) or not limits:
        raise PlanDiagnosticBindingError('Diagnostic requires original supply requirements and world limits')
    return deepcopy({'phase': PLAN_DIAGNOSTIC_PHASE, 'mechanical_status': 'rejected_uncompiled',
        'seed_contract': wp['seed_contract'], 'business': seed_business_context(wp),
        'blueprint': normalize_world_blueprint(wp), 'supply_plan': supply,
        'delivery_target': wp.get('delivery_target'), 'requirements': requirements, 'limits': limits,
        'uncompiled_candidate': candidate, 'findings': findings, 'history': history,
        'remaining_business_attempts': remaining_business_attempts,
        'source_binding': source_binding, 'retry_policy_binding': retry_policy_binding,
        'responsibility_catalog': _diagnostic_catalog(candidate, findings)})


def _diagnostic_target_errors(raw):
    if not isinstance(raw, dict): return []  # Original reviewer validator diagnoses the root.
    targets = raw.get('responsibility_targets')
    if not isinstance(targets, list):
        return [{'path': 'responsibility_targets', 'message': 'Expected an array of original responsibility records'}]
    errors = []
    for number, target in enumerate(targets):
        path = 'responsibility_targets[' + str(number) + ']'
        if not isinstance(target, dict):
            errors.append({'path': path, 'message': 'Expected a responsibility target object'}); continue
        if set(target) - {'unit_id', 'collection', 'key', 'finding_indices', 'reason'}:
            errors.append({'path': path, 'message': 'Use only the documented responsibility target fields'})
        for key in ('unit_id', 'collection', 'key', 'reason'):
            if not isinstance(target.get(key), str) or not target[key].strip():
                errors.append({'path': path + '.' + key, 'message': 'Expected a nonempty string'})
        ids = target.get('finding_indices')
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            errors.append({'path': path + '.finding_indices', 'message': 'Expected a nonempty array of integer finding indices'})
    if raw.get('decision') == 'repair' and not targets:
        errors.append({'path': 'responsibility_targets', 'message': 'repair requires original responsible records; use unresolved when no supported repair can be identified'})
    return errors


def _validate_diagnostic_semantics(raw, payload):
    """Wrong identities/scopes stop once; they are not invitations to re-review."""
    if not isinstance(raw, dict): return
    allowed_mechanisms = {row['id'] for row in payload['business'].get('mechanisms', [])}
    checks = raw.get('mechanism_checks')
    if isinstance(checks, list):
        for row in checks:
            if isinstance(row, dict) and isinstance(row.get('mechanism_id'), str) and row['mechanism_id'] not in allowed_mechanisms:
                raise PlanDiagnosticBindingError('Diagnostic cites an unknown original mechanism')
    catalog = {(row['unit_id'], row['collection'], row['key']): row for row in payload['responsibility_catalog']}
    targets = raw.get('responsibility_targets')
    if not isinstance(targets, list): return
    covered, seen = set(), set()
    for target in targets:
        if not isinstance(target, dict): continue
        values = tuple(target.get(k) for k in ('unit_id', 'collection', 'key'))
        if not all(isinstance(v, str) and v for v in values): continue
        if values not in catalog:
            raise PlanDiagnosticBindingError('Diagnostic target is unknown or outside the current compiler findings: ' + repr(values))
        if values in seen: raise PlanDiagnosticBindingError('Diagnostic repeats a responsibility target')
        seen.add(values)
        ids = target.get('finding_indices')
        if isinstance(ids, list) and ids and all(type(i) is int for i in ids):
            if len(set(ids)) != len(ids) or set(ids) - set(catalog[values]['finding_indices']):
                raise PlanDiagnosticBindingError('Diagnostic target is bound to the wrong compiler finding')
            covered.update(ids)
    if not _diagnostic_target_errors(raw):
        if raw.get('decision') == 'repair' and covered != set(range(len(payload['findings']))):
            raise PlanDiagnosticBindingError('Diagnostic repair does not account for every current compiler finding')
        if raw.get('decision') in ('accept', 'unresolved') and targets:
            raise PlanDiagnosticBindingError('Only a repair opinion may request an original responsibility transaction')


def diagnose_uncompiled_plan(wp, candidate, findings, tracer, *, history,
                            source_binding, retry_policy_binding, remaining_business_attempts=1):
    """Ask the original reviewer once semantically, with bounded wire correction.

    No candidate is installed or compiled here. Physical request evidence and
    cumulative admission stay with the supplied original run Tracer.
    """
    payload = _diagnostic_payload(wp, candidate, findings, history, source_binding,
                                  retry_policy_binding, remaining_business_attempts)
    expected = [row['id'] for row in payload['business'].get('mechanisms', [])]
    protocol = {'version': REVIEW_PROTOCOL_VERSION, 'phase': PLAN_DIAGNOSTIC_PHASE,
        'max_protocol_attempts': REVIEW_PROTOCOL_ATTEMPTS,
        'max_json_attempts_per_protocol': JSON_ATTEMPTS_PER_PROTOCOL,
        'max_physical_requests': REVIEW_PROTOCOL_ATTEMPTS * JSON_ATTEMPTS_PER_PROTOCOL, 'attempts': []}
    request = payload
    for number in range(1, REVIEW_PROTOCOL_ATTEMPTS + 1):
        raw = _call(tracer, 'council.blueprint_feasibility', PLAN_DIAGNOSTIC_SYSTEM, request, allow_invalid_shape=True)
        errors = _review_errors(raw, expected) + _diagnostic_target_errors(raw)
        entry = {'attempt': number, 'input_hash': digest(request), 'review': deepcopy(raw), 'errors': deepcopy(errors)}
        protocol['attempts'].append(entry)
        try: _validate_diagnostic_semantics(raw, payload)
        except PlanDiagnosticBindingError as error:
            entry['semantic_rejection'] = str(error)
            raise PlanDiagnosticBindingError(str(error), review=raw, protocol=protocol) from error
        if not errors: break
        if number == REVIEW_PROTOCOL_ATTEMPTS: raise BlueprintReviewProtocolError(protocol)
        request = _diagnostic_correction(payload, raw, errors, number + 1)
    reviewed = {'version': VERSION, 'input_hash': digest(payload), 'input': payload,
                'decision': raw['decision'], 'review': deepcopy(raw), 'protocol': protocol,
                'scope': 'Original reviewer diagnosis of a mechanically rejected uncommitted candidate; no compilation or acceptance'}
    return {'version': PLAN_DIAGNOSTIC_VERSION, 'phase': PLAN_DIAGNOSTIC_PHASE,
        'candidate_sha256': digest(candidate), 'blueprint_sha256': digest(payload['blueprint']),
        'requirements_sha256': digest(payload['requirements']), 'limits_sha256': digest(payload['limits']),
        'seed_contract_sha256': digest(payload['seed_contract']), 'review_input_hash': digest(payload),
        'decision': raw['decision'], 'responsibility_targets': deepcopy(raw['responsibility_targets']),
        'author_allowed': raw['decision'] == 'repair', 'review': reviewed}


def validate_uncompiled_diagnostic(wp, candidate, receipt):
    """Recheck frozen input and original targets; physical provenance is external.

    Return the bound input for the original layout author's payload. This does
    not grant permission to act on accept/unresolved or to change repair limits.
    """
    try:
        reviewed = receipt['review']; payload = reviewed['input']; raw = reviewed['review']
        expected_payload = _diagnostic_payload(wp, candidate, payload['findings'], payload['history'],
            payload['source_binding'], payload['retry_policy_binding'], payload['remaining_business_attempts'])
        expected = {'version': PLAN_DIAGNOSTIC_VERSION, 'phase': PLAN_DIAGNOSTIC_PHASE,
            'candidate_sha256': digest(candidate), 'blueprint_sha256': digest(expected_payload['blueprint']),
            'requirements_sha256': digest(expected_payload['requirements']), 'limits_sha256': digest(expected_payload['limits']),
            'seed_contract_sha256': digest(expected_payload['seed_contract']), 'review_input_hash': digest(expected_payload),
            'decision': raw['decision'], 'responsibility_targets': raw['responsibility_targets'],
            'author_allowed': raw['decision'] == 'repair'}
        if (payload != expected_payload or any(receipt.get(k) != v for k, v in expected.items())
                or type(receipt.get('author_allowed')) is not bool):
            raise PlanDiagnosticBindingError('Diagnostic input/identity hashes differ from the original rejected transaction')
        if reviewed['version'] != VERSION or reviewed['input_hash'] != digest(payload) or reviewed['decision'] != raw['decision']:
            raise PlanDiagnosticBindingError('Diagnostic review envelope does not match its original input and opinion')
        ids = [row['id'] for row in payload['business'].get('mechanisms', [])]
        if _review_errors(raw, ids) or _diagnostic_target_errors(raw):
            raise PlanDiagnosticBindingError('Diagnostic final review is not protocol-valid')
        _validate_diagnostic_semantics(raw, payload)
        protocol = reviewed['protocol']; attempts = protocol['attempts']
        if (protocol.get('phase') != PLAN_DIAGNOSTIC_PHASE or protocol.get('version') != REVIEW_PROTOCOL_VERSION
                or protocol.get('max_protocol_attempts') != REVIEW_PROTOCOL_ATTEMPTS
                or protocol.get('max_json_attempts_per_protocol') != JSON_ATTEMPTS_PER_PROTOCOL
                or protocol.get('max_physical_requests') != REVIEW_PROTOCOL_ATTEMPTS * JSON_ATTEMPTS_PER_PROTOCOL
                or not 1 <= len(attempts) <= REVIEW_PROTOCOL_ATTEMPTS):
            raise PlanDiagnosticBindingError('Diagnostic protocol bounds were changed')
        request = payload
        for number, entry in enumerate(attempts, 1):
            errors = _review_errors(entry['review'], ids) + _diagnostic_target_errors(entry['review'])
            _validate_diagnostic_semantics(entry['review'], payload)
            if (entry['attempt'] != number or entry['errors'] != errors or ('semantic_rejection' in entry)
                    or entry['input_hash'] != digest(request)):
                raise PlanDiagnosticBindingError('Diagnostic protocol history was changed')
            if number < len(attempts) and not errors:
                raise PlanDiagnosticBindingError('Diagnostic retried a valid substantive opinion')
            request = _diagnostic_correction(payload, entry['review'], errors, number + 1)
        if attempts[-1]['review'] != raw or attempts[-1]['errors']:
            raise PlanDiagnosticBindingError('Diagnostic final opinion is not the last valid response')
        return deepcopy(payload)
    except PlanDiagnosticBindingError: raise
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise PlanDiagnosticBindingError('Malformed diagnostic receipt: ' + str(error)) from error


class BlueprintReviewProtocolError(WorldBlueprintError):
    """The reviewer exhausted formatting attempts, not semantic author repairs."""
    def __init__(self, protocol):
        self.protocol=deepcopy(protocol)
        super().__init__('blueprint feasibility review protocol exhausted: '+
                         json.dumps(protocol['attempts'][-1]['errors'],ensure_ascii=False))


def _review_errors(raw, expected):
    """Validate parsed JSON without using untrusted values as set members."""
    errors=[]
    def reject(path, message):errors.append({'path':path,'message':message})
    def text(value):return isinstance(value,str) and bool(value.strip())
    if not isinstance(raw,dict):
        return [{'path':'$','message':'Review must be a JSON object'}]
    decision=raw.get('decision');checks=raw.get('mechanism_checks');issues=raw.get('issues')
    if not isinstance(decision,str) or decision not in {'accept','repair','unresolved'}:
        reject('decision','Expected a string: accept, repair, or unresolved')
    if not text(raw.get('reason')):reject('reason','Expected a nonempty explanation string')
    if not isinstance(checks,list):reject('mechanism_checks','Expected an array with every original mechanism exactly once')
    else:
        ids=[]
        for i,row in enumerate(checks):
            path='mechanism_checks['+str(i)+']'
            if not isinstance(row,dict):reject(path,'Expected an object');continue
            if not text(row.get('mechanism_id')):reject(path+'.mechanism_id','Expected an original mechanism id string')
            else:ids.append(row['mechanism_id'])
            status=row.get('status')
            if not isinstance(status,str) or status not in {'feasible','blocked','unresolved'}:
                reject(path+'.status','Expected a string: feasible, blocked, or unresolved')
            if not text(row.get('walkthrough')):reject(path+'.walkthrough','Expected a nonempty concrete walkthrough string')
        if sorted(ids)!=sorted(expected):
            reject('mechanism_checks.mechanism_id','Cover each original id exactly once: '+json.dumps(expected,ensure_ascii=False))
    if not isinstance(issues,list):reject('issues','Expected an array, including [] when there are no issues')
    else:
        for i,row in enumerate(issues):
            path='issues['+str(i)+']'
            if not isinstance(row,dict):reject(path,'Expected an object');continue
            if not text(row.get('finding')):reject(path+'.finding','Expected a nonempty specific conflict string')
            if not isinstance(row.get('suggestion'),str):reject(path+'.suggestion','Expected a suggestion string')
    if decision=='accept':
        if isinstance(issues,list) and issues:reject('decision','accept requires issues=[]; retain a negative verdict when concerns remain')
        if isinstance(checks,list) and any(not isinstance(c,dict) or c.get('status')!='feasible' for c in checks):
            reject('decision','accept requires every mechanism feasible; do not change a substantive negative opinion to satisfy the format')
    elif decision=='repair' and isinstance(issues,list) and not issues:
        reject('issues','repair requires at least one specific issue')
    return errors


def assess(wp, tracer, feedback=None):
    from pipeline.world_state import causal_execution_contract
    blueprint = normalize_world_blueprint(wp)
    instance_plan = wp.get("business_instance_plan")
    events = ([dict(e, id=e["slot"]) for u in instance_plan["units"] for e in u["events"]]
              if instance_plan else None)
    payload = {"business": seed_business_context(wp), "blueprint": blueprint,
               "mechanical_execution": causal_execution_contract(blueprint, events),
               "feedback": feedback}
    if instance_plan:
        from pipeline.instance_plan import execution_contract
        payload['instance_execution'] = execution_contract(wp)
    expected = [m["id"] for m in payload["business"].get("mechanisms", [])]
    protocol={'version':REVIEW_PROTOCOL_VERSION,'max_protocol_attempts':REVIEW_PROTOCOL_ATTEMPTS,
              'max_json_attempts_per_protocol':JSON_ATTEMPTS_PER_PROTOCOL,
              'max_physical_requests':REVIEW_PROTOCOL_ATTEMPTS*JSON_ATTEMPTS_PER_PROTOCOL,'attempts':[]}
    request=payload
    for attempt in range(1,REVIEW_PROTOCOL_ATTEMPTS+1):
        # A valid repair/unresolved is a final reviewer opinion. Only malformed
        # parsed replies enter this loop; transport/parse exhaustion and global
        # stops propagate directly. Every physical request retains admission.
        raw=_call(tracer,'council.blueprint_feasibility',SYSTEM,request,allow_invalid_shape=True)
        errors=_review_errors(raw,expected)
        protocol['attempts'].append({'attempt':attempt,'review':deepcopy(raw),'errors':deepcopy(errors)})
        if not errors:break
        if attempt==REVIEW_PROTOCOL_ATTEMPTS:raise BlueprintReviewProtocolError(protocol)
        request=deepcopy(payload)
        request['protocol_feedback']={'previous_response':deepcopy(raw),'errors':deepcopy(errors),
            'attempt':attempt+1,'max_attempts':REVIEW_PROTOCOL_ATTEMPTS,
            'instruction':'Correct only the response protocol against the unchanged original input. Return the complete review. Preserve substantive repair/unresolved findings; a negative opinion is valid and must not be changed merely to obtain accept.'}
    return {"version": VERSION, "input_hash": digest(payload), "input": payload,
            "decision": raw["decision"], "review": raw,"protocol":protocol,
            "scope": "LLM assessment of expressibility; actual world still needs all original gates"}


class BlueprintConstraintProtocolError(WorldBlueprintError):
    """The original constraint author exhausted protocol corrections."""
    def __init__(self, protocol):
        self.protocol = deepcopy(protocol)
        super().__init__("blueprint constraint author protocol exhausted: " +
                         json.dumps(protocol["attempts"][-1]["errors"], ensure_ascii=False))


def _constraint_errors(raw):
    """Check wire types before hashing selectors; do not decide business scope."""
    errors = []
    def reject(path, message): errors.append({"path": path, "message": message})
    def text(value): return isinstance(value, str) and bool(value.strip())
    if not isinstance(raw, dict):
        return [{"path": "$", "message": "Constraint proposal must be a JSON object"}]
    if not text(raw.get("reason")): reject("reason", "Expected a nonempty explanation string")
    edits = raw.get("edits")
    if not isinstance(edits, list):
        reject("edits", "Expected an array; [] is a valid unsupported/unresolved reply")
        return errors
    for index, edit in enumerate(edits):
        path = "edits[" + str(index) + "]"
        if not isinstance(edit, dict):
            reject(path, "Expected an object"); continue
        for key in ("entity_type", "field", "reason"):
            if not text(edit.get(key)): reject(path + "." + key, "Expected a nonempty string")
        remove, values = edit.get("remove", []), edit.get("set", {})
        if not isinstance(remove, list) or any(not isinstance(key, str) for key in remove):
            reject(path + ".remove", "Expected an array of constraint-name strings")
        if not isinstance(values, dict): reject(path + ".set", "Expected an object")
    return errors


def _constraint_proposal(tracer, payload, audit):
    protocol = {"version": CONSTRAINT_PROTOCOL_VERSION,
                "max_protocol_attempts": REVIEW_PROTOCOL_ATTEMPTS,
                "max_json_attempts_per_protocol": JSON_ATTEMPTS_PER_PROTOCOL,
                "max_physical_requests": REVIEW_PROTOCOL_ATTEMPTS * JSON_ATTEMPTS_PER_PROTOCOL,
                "input_hash": digest(payload), "attempts": []}
    audit["author_protocol"] = protocol
    request = payload
    for number in range(1, REVIEW_PROTOCOL_ATTEMPTS + 1):
        entry = {"attempt": number, "input_hash": digest(request), "status": "requested",
                 "response": None, "errors": []}
        protocol["attempts"].append(entry)
        try:
            raw = _call(tracer, "council.blueprint_constraint_repair", REPAIR_SYSTEM,
                        request, allow_invalid_shape=True)
        except Exception as error:
            entry.update(status="execution_failed", error_type=type(error).__name__,
                         error=str(error))
            raise
        audit["author"] = deepcopy(raw)
        errors = _constraint_errors(raw)
        entry.update(response=deepcopy(raw), errors=deepcopy(errors),
                     status="invalid_protocol" if errors else "protocol_valid")
        if not errors:
            return raw
        if number == REVIEW_PROTOCOL_ATTEMPTS:
            raise BlueprintConstraintProtocolError(protocol)
        request = deepcopy(payload)
        request["protocol_feedback"] = {"previous_response": deepcopy(raw), "errors": deepcopy(errors),
            "attempt": number + 1, "max_attempts": REVIEW_PROTOCOL_ATTEMPTS,
            "instruction": "Correct only output protocol against the unchanged business, blueprint and feedback. "
                "Return a complete proposal. Preserve substantive uncertainty: edits=[] with a reason is valid. "
                "Do not invent or relax business constraints just to return nonempty edits."}


def revise_constraints(wp, tracer, feedback, audit):
    """LLM authors edits; exact seed constraints and execution scale stay fixed."""
    audit.update(version=VERSION, before=deepcopy(wp), feedback=deepcopy(feedback), status="proposed")
    payload = {"business": seed_business_context(wp), "blueprint": normalize_world_blueprint(wp),
               "feedback": feedback}
    raw = _constraint_proposal(tracer, payload, audit)
    edits = raw.get("edits")
    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip() or not isinstance(edits, list) or not edits:
        raise WorldBlueprintError("Upstream architect has no supported constraint repair")
    result = deepcopy(wp)
    fields = {(t["id"], f["name"]): f for t in result["world_blueprint"]["entity_types"] for f in t["fields"]}
    allowed, touched = {"states", "range", "monotonic"}, set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise WorldBlueprintError("Constraint edit must be an object")
        key = (edit.get("entity_type"), edit.get("field"))
        remove, values = edit.get("remove", []), edit.get("set", {})
        if (key not in fields or key in touched or not isinstance(remove, list)
                or any(not isinstance(k, str) for k in remove) or not isinstance(values, dict)
                or not (set(remove) | set(values))
                or (set(remove) | set(values)) - allowed or set(remove) & set(values)
                or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()):
            raise WorldBlueprintError("Unsupported, duplicate or empty constraint edit")
        touched.add(key)
        for name in remove:
            if name not in fields[key]:
                raise WorldBlueprintError("Cannot remove an absent field constraint")
            del fields[key][name]
        fields[key].update(deepcopy(values))
    normalize_world_blueprint(result)
    contract = result.get("seed_contract")
    if contract:
        seed_audit = validate_seed_blueprint(result, contract)
        if not seed_audit["passed"]:
            raise WorldBlueprintError("Constraint revision violates seed: " + "; ".join(seed_audit["issues"]))
        result["seed_audit"] = seed_audit
    if result["world_blueprint"] == wp["world_blueprint"]:
        raise WorldBlueprintError("Constraint revision made no change")
    # Refresh the original compatibility projection from its one authoritative blueprint.
    projected = {}
    for kind in result["world_blueprint"]["entity_types"]:
        for field in kind["fields"]:
            projected.setdefault(field["name"], deepcopy(field))
    profile = result.setdefault("domain_profile", {})
    profile["field_schema"] = list(projected.values())
    profile["state_machines"] = [{"field": f["name"], "states": f["states"]}
                                 for f in projected.values() if f.get("states")]
    audit["candidate"] = deepcopy(result)
    review = assess(result, tracer, feedback={"request": feedback, "author_changes": raw})
    audit["review"] = review
    if review["decision"] != "accept":
        raise WorldBlueprintError("Revised blueprint remains unresolved; original contract retained")
    result.setdefault("blueprint_revisions", []).append({"before_hash": digest(wp["world_blueprint"]),
        "after_hash": digest(result["world_blueprint"]), "author": deepcopy(raw), "review": review})
    audit.update(status="accepted", after=deepcopy(result))
    return result
