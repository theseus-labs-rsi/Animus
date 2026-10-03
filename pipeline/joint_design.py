"""One bounded, unpublished schema and instance-plan design transaction.

This is the v6 entry. Saved v4/v5 papers and their recovery ledgers keep using
their original controller. Every model response is a candidate until the full
compiler and independent business reviewer accept the same draft.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re

import config
from pipeline.capability_contract import digest
from pipeline.instance_plan import PlanConflict, SYSTEM as PLAN_SYSTEM, compile_plan, finding

VERSION = "joint-supply-design/v2"
PRODUCTION_VERSION = "supply-driven/v6"
MAX_REVISIONS = 32
MAX_INITIAL_SCHEMA_ATTEMPTS = 3
MAX_REJECTED_PATCH_STREAK = 2


REVISION_SYSTEM = """你在同一个尚未定稿的世界设计包里修订 schema 与具体业务实例计划。
原 seed 的身份、业务机制、未知边界、逐线独立候选数量、实体和时间上限是硬约束。错误清单来自完整编译器或业务审阅，审阅建议可以质疑，但不得把未经完整编译的建议称为可执行修法。每类 schema 的 entity_types.count 是计划对象下限；审阅建议若要求合并对象导致低于该下限，不能以 patch 删除对象，须先判断审阅是否误读原 seed 的核心对象数量。若原 seed 确实规定总数上限且 schema.count 错了，选择 replace_draft 修正 schema；若没有总数上限，不因名称相似而合并不同 slot。数值字段只在其类型/字段出现在事件 effect_fields 时才是 event-owned；其余可由业务上真实的公报或观测赋值，不能仅因 numeric_event_writes 为空而添加虚构事件。同一事务应覆盖 open_design_findings 中所有仍然成立的问题；revision_transaction 反馈只说明上一次编辑格式错误，不能取代原设计问题。若需要跨单元重接很多身份，优先考虑 replace_plan 或 replace_draft，而不是几十条易漏引用的 patch。cohort_layout 的 members 必须恰好等于反馈 natural 列表中的一个真实分组，不可随意抽取或合并；修已有组时用 cohort_changes 的 replace、原 id 为 key、完整新记录为 value。观测不是 event-owned 字段的真实写入，初态不是一次变化；不可逆生命周期不能重复执行来凑趋势。业务对象、关系、事件和比较组必须有自然目的，禁止预填题目答案或数值。
只从以下三种操作选一种：
1. patch：对当前未定稿包做一个原子事务。revision 必须采用原业务修订的 {"decision":"repair","reason":"...","field_aliases":[],"blueprint_additions":{},"changes":[],"cohort_changes":[]}；changes 的每项按 unit_id、collection、operation(add|replace|remove)、key/selector、完整 value 定位。此操作可新增声明，不可删除或改写旧声明。
2. replace_plan：当前 schema 正确，但计划需要整体重排；下一请求将请计划作者基于同一 schema 重写完整计划。不要在本响应里输出计划。
3. replace_draft：当前 schema 本身妨碍业务，下一请求将请世界架构师重写完整 schema，再由计划作者重写计划。说明需改变的结构与必须保留的 seed 机制。
如果原约束内确实无法完成，返回 unresolved 并说明冲突。
严格返回 JSON：{"decision":"patch|replace_plan|replace_draft|unresolved","reason":"具体业务及编译依据","revision":{}}。仅 patch 使用 revision。replace 定位原记录时，events/relations/objects 用 selector:{"slot":"原slot"}，obligations 用 selector:{"id":"原id"}；value 必须是该记录修改后的完整对象。cohort_changes 新增组写成 {"operation":"add","key":"组id","value":{"id":"组id","basis":"先验依据","members":["对象slot"]}}，不得把记录包在 cohort 字段里。对象记录的稳定身份是 slot，不要给 value 增加 key 字段。不要把 selector 写成 events[slot=x].session 之类的路径，也不要把 value 写成单个数。先按反馈里的 affected_record_owners 找到记录所属 unit_id。若反馈标明 revision_transaction，错误属于刚提交的编辑事务、原候选尚未改变，应修正事务格式，不要臆测原业务对象缺字段。不得声明通过，程序会对新候选重新运行完整检查。"""


def _save(checkpoint, audit):
    checkpoint(deepcopy(audit))


def _call(tracer, step, system, payload, audit, checkpoint, *, max_tokens=32768):
    entry = {"step": step, "request_hash": digest(payload), "status": "requested"}
    audit["calls"].append(entry)
    _save(checkpoint, audit)
    try:
        # Inherit the runner's engineering retry budget; direct calls retain
        # config.chat_json's finite default. This author has no stricter policy.
        raw = tracer.chat_json(step, [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            model=config.STRUCTURE_MODEL, temperature=.2, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            retry_clean_header_deadline=True)
    except BaseException as error:
        entry.update(status="call_failed", error_type=type(error).__name__, error=str(error))
        _save(checkpoint, audit)
        raise
    entry.update(status="response_received", response=deepcopy(raw))
    _save(checkpoint, audit)
    if not isinstance(raw, dict) or "__error__" in raw:
        entry["status"] = "provider_error"
        _save(checkpoint, audit)
        from execution_control import ExecutionStopped, global_stop_exception
        stopped = global_stop_exception(raw) if isinstance(raw, dict) else None
        if stopped is not None:
            raise stopped
        metadata = raw.get("__error_metadata__") if isinstance(raw, dict) else None
        kind = metadata.get("kind") if isinstance(metadata, dict) else None
        raise ExecutionStopped(f"{step} returned no usable JSON object: {str(raw)[:500]}",
                               kind=kind or "invalid_json_response", raw=raw)
    return raw


def _plan_payload(wp, feedback):
    return {"business": wp.get("seed_contract"), "scenario": wp.get("scenario"),
            "blueprint": wp["world_blueprint"],
            "requirements": wp["supply_plan"]["requirements"],
            "limits": wp["supply_plan"]["world_limits"], "feedback": feedback}


def _findings(error):
    if isinstance(error, PlanConflict):
        return deepcopy(error.findings)
    return [finding("joint_draft_error", f"{type(error).__name__}: {error}")]


def _review(revised, tracer, audit, checkpoint):
    if not revised.get("seed_contract"):
        return {"decision": "accept", "scope": "No seed business review required"}
    from pipeline.blueprint_feasibility import assess
    entry = {"status": "requested", "candidate_hash": digest(revised)}
    audit["reviews"].append(entry)
    _save(checkpoint, audit)
    try:
        opinion = assess(revised, tracer, feedback={
            "phase": "joint_schema_and_instance_plan", "requirements": revised["supply_plan"]["requirements"]})
    except BaseException as error:
        entry.update(status="call_failed", error_type=type(error).__name__, error=str(error))
        _save(checkpoint, audit)
        raise
    entry.update(status="response_received", response=deepcopy(opinion))
    _save(checkpoint, audit)
    return opinion


def _candidate(wp, proposal, tracer, audit, checkpoint):
    key = digest({"schema": wp["world_blueprint"], "proposal": proposal})
    # A plan author's capacity diagnosis is input to the world architect.
    # It has no executable units, so neither compiler nor reviewer ran.
    if proposal.get("decision") == "unresolved":
        findings = [finding("plan_unresolved", str(proposal.get("reason", "")),
                            kind="design", evidence=proposal)]
        audit["checks"].append({
            "candidate_hash": key, "schema_hash": digest(wp["world_blueprint"]),
            "plan_hash": digest(proposal), "compiler": "not_run",
            "business_review": "not_run", "findings": deepcopy(findings)})
        audit.update(draft=deepcopy(wp), proposal=deepcopy(proposal))
        _save(checkpoint, audit)
        return None, findings
    if key in audit["candidate_hashes"]:
        findings = [finding("joint_cycle", "A complete design candidate was already checked; redesign its structure",
                            kind="design", evidence={"candidate_hash": key})]
        audit.update(draft=deepcopy(wp), proposal=deepcopy(proposal))
        _save(checkpoint, audit)
        return None, findings
    audit["candidate_hashes"].append(key)
    row = {"candidate_hash": key, "schema_hash": digest(wp["world_blueprint"]),
           "plan_hash": digest(proposal), "compiler": "requested", "business_review": "blocked_by_compiler"}
    audit["checks"].append(row)
    audit.update(draft=deepcopy(wp), proposal=deepcopy(proposal))
    _save(checkpoint, audit)
    try:
        revised = compile_plan(wp, proposal)
    except PlanConflict as error:
        row.update(compiler="rejected", findings=deepcopy(error.findings))
        _save(checkpoint, audit)
        return None, deepcopy(error.findings)
    row["compiler"] = "passed"
    row["compiled_hash"] = digest(revised)
    _save(checkpoint, audit)
    try:
        opinion = _review(revised, tracer, audit, checkpoint)
    except Exception as error:
        from pipeline.blueprint_feasibility import BlueprintReviewProtocolError
        if not isinstance(error, BlueprintReviewProtocolError):
            raise
        row.update(business_review="protocol_exhausted",
                   review_protocol=deepcopy(error.protocol))
        _save(checkpoint, audit)
        return None, [finding("business_review_protocol_exhausted",
            "Independent business reviewer exhausted its bounded response protocol",
            kind="design", evidence=deepcopy(error.protocol))]
    row["business_review"] = opinion["decision"]
    row["review_hash"] = digest(opinion)
    _save(checkpoint, audit)
    if opinion["decision"] == "accept":
        return revised, []
    if opinion["decision"] == "unresolved":
        return None, [finding("business_unresolved", "Independent business review is unresolved",
                              kind="design", evidence={"input_hash": opinion["input_hash"],
                                                       "review": opinion["review"]})]
    return None, [finding("business_repair", "Independent review requests a business revision",
                          kind="business", evidence={"input_hash": opinion["input_hash"],
                                                       "review": opinion["review"]})]


def _feedback(wp, proposal, findings, audit):
    checked = audit["checks"][-1] if audit["checks"] else {}
    owners = {}
    if isinstance(proposal, dict):
        for unit in proposal.get("units", []):
            if not isinstance(unit, dict):
                continue
            for collection in ("objects", "relations", "events", "obligations"):
                for row in unit.get(collection, []):
                    if isinstance(row, dict):
                        key = row.get("id" if collection == "obligations" else "slot")
                        if isinstance(key, str):
                            owners.setdefault(key, []).append({"unit_id": unit.get("unit_id"),
                                                                 "collection": collection})
    affected = sorted({identity for issue in findings for identity in issue.get("affected_ids", [])
                       if isinstance(identity, str) and identity in owners})
    objects = {row.get("slot"): row.get("type") for unit in
               (proposal.get("units", []) if isinstance(proposal, dict) else [])
               if isinstance(unit, dict) for row in unit.get("objects", [])
               if isinstance(row, dict) and isinstance(row.get("slot"), str)
               and isinstance(row.get("type"), str)}
    result = {"seed": wp.get("seed_contract"),
            "frozen_budget": wp["supply_plan"]["requirements"],
            "limits": wp["supply_plan"]["world_limits"],
            "schema": wp["world_blueprint"], "proposal": proposal,
            "declared_object_minima": {row["id"]: row["count"]
                                       for row in wp["world_blueprint"]["entity_types"]},
            "allocated_object_counts": dict(Counter(objects.values())),
            "current_findings": findings,
            "open_design_findings": audit.get("open_design_findings", findings),
            "affected_record_owners": {key: owners[key] for key in affected},
            "check_coverage": {key: checked.get(key) for key in
                               ("candidate_hash", "compiler", "business_review")},
            "previous_actions": [{"decision": row.get("decision"), "reason": row.get("reason"),
                                  "before_hash": row.get("before_hash"), "after_hash": row.get("after_hash"),
                                  "status": row.get("status"),
                                  "rejected_codes": [issue.get("code") for issue in row.get("findings", [])]}
                                 for row in audit["actions"][-MAX_REVISIONS:]],
            "revision_allowance": {"total": MAX_REVISIONS,
                                   "used": audit["used_revisions"],
                                   "remaining_after_this_action": MAX_REVISIONS - audit["used_revisions"] - 1},
            "warning": "Unexecuted checks are not passes; all proposals require complete joint recompilation."}
    if proposal.get("decision") == "unresolved":
        result["schema_redesign_instruction"] = (
            "Review the plan author's constraint diagnosis against the construction requirements and prior compiler evidence. "
            "Its capacity estimates are author claims. Revise the world schema where needed while preserving the seed, "
            "frozen candidate requirements, construction rules and entity/time limits.")
        result["last_executed_check"] = next((deepcopy(row) for row in reversed(audit["checks"])
                                               if row.get("compiler") in {"passed", "rejected"}), None)
    last = audit["actions"][-1] if audit["actions"] else None
    if last and last.get("status") == "rejected":
        result["feedback_origin"] = "revision_transaction: the previous edit transaction was rejected; the current business plan is unchanged"
        response = next((row["response"] for row in reversed(audit.get("calls", []))
                         if row.get("step") == "council.joint_design_revision"
                         and isinstance(row.get("response"), dict)
                         and isinstance(row["response"].get("revision"), dict)), None)
        if isinstance(response, dict) and isinstance(response.get("revision"), dict):
            result["last_rejected_revision"] = response["revision"]
    return result


def _rejected_patch_streak(audit):
    count = 0
    for row in reversed(audit.get("actions", [])):
        if row.get("decision") != "patch" or row.get("status") != "rejected":
            break
        count += 1
    return count


def normalize_record_field_edits(proposal, revision):
    """Normalize two unambiguous edit shorthands without changing intent.

    A model may name ``events[slot=x].session`` instead of copying the full
    event, or put a complete added cohort under ``cohort``. Ambiguous edits
    and all other partial edits still fail before the full compiler runs.
    """
    if not isinstance(revision, dict):
        return revision, []
    result = deepcopy(revision)
    normalized = []
    for change in result.get("cohort_changes", []) if isinstance(result.get("cohort_changes", []), list) else []:
        if (isinstance(change, dict) and change.get("operation") == "add"
                and "key" not in change and "selector" not in change
                and "value" not in change and isinstance(change.get("cohort"), dict)):
            cohort = change["cohort"]
            identity = cohort.get("id")
            if isinstance(identity, str) and identity.strip():
                change["key"] = identity
                change["value"] = change.pop("cohort")
                normalized.append({"collection": "cohorts", "operation": "add",
                                   "key": identity, "record_hash": digest(cohort)})
    for change in result.get("changes", []) if isinstance(result.get("changes", []), list) else []:
        if not isinstance(change, dict) or change.get("operation") != "replace":
            continue
        raw = change.get("selector")
        if not isinstance(raw, str):
            continue
        match = re.fullmatch(r"([a-z_]+)\[([a-z_]+)=([^\[\]]+)\]\.([a-z_]+)", raw)
        collection = change.get("collection")
        if (match is None or match.group(1) != collection or match.group(4) != "session"
                or collection not in ("events", "relations", "publications")
                or match.group(2) != ("fact_slot" if collection == "publications" else "slot")
                or type(change.get("value")) is not int):
            raise PlanConflict([finding("joint_partial_edit_shape",
                                        "A partial edit needs one exact session field and integer value")])
        units = [unit for unit in proposal["units"] if unit["unit_id"] == change.get("unit_id")]
        if len(units) != 1:
            raise PlanConflict([finding("joint_partial_edit_target", "Partial edit unit must be unique")])
        key, identity, field = match.group(2), match.group(3), match.group(4)
        rows = [row for row in units[0].get(collection, []) if str(row.get(key)) == identity]
        if len(rows) != 1:
            raise PlanConflict([finding("joint_partial_edit_target",
                                        "Partial edit must select exactly one existing record")])
        full = deepcopy(rows[0])
        full[field] = change["value"]
        change["selector"] = {key: identity}
        change["value"] = full
        normalized.append({"unit_id": change["unit_id"], "collection": collection,
                           "selector": raw, "record_hash_before": digest(rows[0]),
                           "record_hash_after": digest(full)})
    return result, normalized


def _replacement(run, wp, feedback, pack, delivery_brief):
    from pipeline.central_office import central_office
    from pipeline.factory import validate_seed_identity
    from pipeline.supply import attach_plan
    sc = run.read("00_input.json")
    cfg = run.manifest["config"]
    replacement = central_office(sc["description"], sc["few_shot"], run.tracer, run.log,
        seed_pack=pack, delivery_brief=delivery_brief, max_world_attempts=1,
        revision_of=wp, design_feedback=feedback, defer_business_review=True)
    if pack is not None:
        validate_seed_identity(run, replacement)
    for field in ("world_generation", "generation_contract", "quality_contract"):
        replacement[field] = deepcopy(wp[field])
    attach_plan(replacement, cfg["delivery_target"], cfg.get("question_budget"),
                cfg.get("single_pass_world_limits"), cfg.get("delivery_survival_rates"),
                capacity_driven=True, instance_driven=True)
    return replacement


def _degrade_after_limit(audit, findings, checkpoint, *,
                         status="exploratory_after_design_limit"):
    """Publish the last checked schema for an exploratory run without an invalid plan.

    A failed instance plan cannot be executed by the world writer. The schema,
    seed and supply targets remain available to the ordinary world agent; the
    failed proposal and every finding stay in the unpublished design ledger.
    """
    candidate = deepcopy(audit["draft"])
    candidate.pop("business_instance_plan", None)
    candidate["supply_plan"].pop("instance_policy", None)
    receipt = {"version": VERSION, "status": status,
               "release_eligible": False, "fallback": "blueprint_only",
               "whitepaper_hash": digest(candidate),
               "supply_plan_hash": digest(candidate["supply_plan"]),
               "schema_hash": digest(candidate["world_blueprint"]),
               "unexecuted_proposal_hash": digest(audit.get("proposal")),
               "used_revisions": audit["used_revisions"],
               "unresolved_findings": deepcopy(findings),
               "checks": {"joint_compiler": "unresolved", "business_review": "unresolved",
                          "actual_source_supply": "blocked_by_world",
                          "public_material": "blocked_by_world"}}
    audit.update(status="degraded", accepted=deepcopy(candidate),
                 receipt=deepcopy(receipt), findings=deepcopy(findings))
    _save(checkpoint, audit)
    return candidate, receipt


def exploratory_after_limit(run, wp):
    """Return whether this frozen paper has a bound exploratory design receipt."""
    if not run.has("01_joint_design_receipt.json"):
        return False
    receipt = run.read("01_joint_design_receipt.json")
    if receipt.get("status") not in {"exploratory_after_design_limit",
                                     "exploratory_after_review_limit"}:
        return False
    if (receipt.get("whitepaper_hash") != digest(wp)
            or receipt.get("supply_plan_hash") != digest(wp["supply_plan"])
            or receipt.get("schema_hash") != digest(wp["world_blueprint"])
            or receipt.get("release_eligible") is not False
            or receipt.get("fallback") != "blueprint_only"
            or wp.get("business_instance_plan")
            or wp["supply_plan"].get("instance_policy")):
        raise ValueError("Exploratory design receipt is stale or changed")
    return True


def accept(run, wp, pack, delivery_brief, audit, checkpoint, *, source_findings=None,
           initial_proposal=None, allow_degraded=False, resume_unpublished=False):
    """Return an accepted package or an explicitly degraded exploratory paper."""
    if audit.get("version") != VERSION:
        raise ValueError("Unknown joint design ledger version")
    if resume_unpublished:
        if (source_findings is not None or initial_proposal is not None
                or audit.get("status") != "designing"
                or not isinstance(audit.get("draft"), dict)
                or not isinstance(audit.get("proposal"), dict)
                or not isinstance(audit.get("used_revisions"), int)
                or not 0 <= audit["used_revisions"] < MAX_REVISIONS
                or not audit.get("actions") or audit["actions"][-1].get("status") != "rejected"
                or not isinstance(audit["actions"][-1].get("findings"), list)
                or not audit["actions"][-1]["findings"]
                or not audit.get("calls") or audit["calls"][-1].get("step") != "council.joint_design_revision"
                or audit["calls"][-1].get("status") != "provider_error"):
            raise ValueError("Joint design transport continuation requires a rejected edit and a saved failed revision call")
        wp = deepcopy(audit["draft"])
        proposal = deepcopy(audit["proposal"])
        pending_findings = deepcopy(audit["actions"][-1]["findings"])
        if audit["actions"][-1].get("before_hash") != digest({"schema": wp["world_blueprint"], "proposal": proposal}):
            raise ValueError("Joint design saved candidate differs from its last rejected edit")
        if digest(_feedback(wp, proposal, pending_findings, audit)) != audit["calls"][-1].get("request_hash"):
            raise ValueError("Joint design resumed request differs from the failed request")
    elif source_findings is None:
        if audit.get("status") != "initial_draft_ready":
            raise ValueError("Joint design requires its saved unpublished initial draft")
        world_attempts = wp["_council_views"].get("world_author_attempts", 1)
        audit.update(status="designing", draft=deepcopy(wp), proposal=None,
                     used_revisions=world_attempts - 1,
                     initial_schema_attempts=world_attempts,
                     candidate_hashes=[], checks=[], calls=[], reviews=[], actions=[],
                     open_design_findings=[])
        _save(checkpoint, audit)
        proposal = (deepcopy(initial_proposal) if initial_proposal is not None else
                    _call(run.tracer, "council.instance_plan", PLAN_SYSTEM,
                          _plan_payload(wp, {"phase": "joint_initial_business_layout",
                                             "seed_mechanisms_must_be_witnessed": True}),
                          audit, checkpoint,
                          max_tokens=65536 if (wp.get("delivery_target") or {}).get("version") == 2
                          else 32768))
        if initial_proposal is not None:
            audit["fixture_initial_proposal_hash"] = digest(initial_proposal)
            _save(checkpoint, audit)
        pending_findings = None
    else:
        if (audit.get("status") != "accepted" or not isinstance(source_findings, list)
                or not source_findings):
            raise ValueError("Source revision requires an accepted joint draft and findings")
        if audit["receipt"]["whitepaper_hash"] != digest(run.read("01_whitepaper.json")):
            raise ValueError("Source revision is not bound to the current accepted whitepaper")
        wp = deepcopy(audit["draft"])
        proposal = deepcopy(audit["proposal"])
        audit.update(status="revising_after_world_probe")
        audit["open_design_findings"] = deepcopy(source_findings)
        audit.setdefault("source_findings", []).append(deepcopy(source_findings))
        _save(checkpoint, audit)
        pending_findings = deepcopy(source_findings)
    while True:
        if pending_findings is None:
            revised, findings = _candidate(wp, proposal, run.tracer, audit, checkpoint)
            if revised is None:
                audit["open_design_findings"] = deepcopy(findings)
                _save(checkpoint, audit)
        else:
            revised, findings = None, pending_findings
            pending_findings = None
        if (allow_degraded and any(item.get("code") == "business_review_protocol_exhausted"
                                   for item in findings)):
            return _degrade_after_limit(audit, findings, checkpoint,
                                        status="exploratory_after_review_limit")
        if revised is not None:
            receipt = {"version": VERSION, "status": "ready_for_world_probe",
                       "whitepaper_hash": digest(revised),
                       "supply_plan_hash": digest(revised["supply_plan"]),
                       "business_instance_plan_hash": digest(revised["business_instance_plan"]),
                       "schema_hash": digest(revised["world_blueprint"]),
                       "used_revisions": audit["used_revisions"],
                       "checks": {"joint_compiler": "passed", "business_review": "passed",
                                  "actual_source_supply": "blocked_by_world",
                                  "public_material": "blocked_by_world"}}
            audit.update(status="accepted", accepted=deepcopy(revised), receipt=receipt)
            _save(checkpoint, audit)
            return revised, receipt
        if audit["used_revisions"] >= MAX_REVISIONS:
            if allow_degraded:
                return _degrade_after_limit(audit, findings, checkpoint)
            audit.update(status="revisions_exhausted", findings=findings)
            _save(checkpoint, audit)
            raise PlanConflict([finding("joint_revisions_exhausted",
                                        "The shared design revision allowance is exhausted",
                                        kind="design", evidence=findings)])
        feedback = _feedback(wp, proposal, findings, audit)
        if proposal.get("decision") == "unresolved" or any(f.get("code") == "joint_cycle" for f in findings):
            action = {
                "decision": "replace_draft",
                "reason": "The plan is unresolved or repeats a rejected design. Reconsider the world structure using the original diagnosis, preserving seed mechanisms, candidate requirements and world limits; then author and check a new plan.",
                "source": "controller_plan_unresolved" if proposal.get("decision") == "unresolved" else "controller_repeated_candidate"}
        elif _rejected_patch_streak(audit) >= MAX_REJECTED_PATCH_STREAK:
            action = {"decision": "replace_plan",
                      "reason": "The controller selected a complete plan rewrite after two rejected atomic patches; preserve the original schema, seed and limits.",
                      "source": "controller_rejected_patch_fallback"}
        else:
            action = _call(run.tracer, "council.joint_design_revision", REVISION_SYSTEM,
                           feedback, audit, checkpoint, max_tokens=32768)
        if action.get("decision") == "unresolved":
            # A legitimate inability report is feedback for the world designer.
            # It consumes the same shared revision slot as any other redesign.
            feedback["revision_author_diagnosis"] = deepcopy(action)
            action = {"decision":"replace_draft", "reason":str(action.get("reason")),
                      "source":"controller_revision_unresolved"}
        audit["used_revisions"] += 1
        action_row = {"decision": action.get("decision"), "reason": action.get("reason"),
                      "before_hash": digest({"schema": wp["world_blueprint"], "proposal": proposal}),
                      "status": "applying"}
        if action.get("source"):
            action_row["source"] = action["source"]
        audit["actions"].append(action_row)
        _save(checkpoint, audit)
        try:
            if action.get("decision") == "patch":
                from pipeline.layout_revision import apply, _business_protocol_errors
                revision = action.get("revision")
                revision, normalized = normalize_record_field_edits(proposal, revision)
                if normalized:
                    action_row["mechanical_normalization"] = normalized
                    _save(checkpoint, audit)
                wire_errors = _business_protocol_errors(wp, revision)
                if wire_errors:
                    raise PlanConflict([finding("joint_revision_wire",
                                                "The proposed edit transaction has invalid fields; the current business plan is unchanged",
                                                kind="design", evidence={"errors": wire_errors})])
                wp_after, proposal_after = apply(wp, proposal, revision)
                wp = deepcopy(wp_after)
                wp.pop("business_instance_plan", None)
                proposal = proposal_after
            elif action.get("decision") == "replace_plan":
                proposal = _call(run.tracer, "council.instance_plan_replacement", PLAN_SYSTEM,
                                 _plan_payload(wp, feedback), audit, checkpoint,
                                 max_tokens=65536 if (wp.get("delivery_target") or {}).get("version") == 2
                                 else 32768)
            elif action.get("decision") == "replace_draft":
                wp_after = _replacement(run, wp, feedback, pack, delivery_brief)
                proposal_after = _call(
                    run.tracer, "council.instance_plan_replacement", PLAN_SYSTEM,
                    _plan_payload(wp_after, feedback), audit, checkpoint,
                    max_tokens=65536 if (wp_after.get("delivery_target") or {}).get("version") == 2
                    else 32768)
                wp, proposal = wp_after, proposal_after
            else:
                raise PlanConflict([finding("joint_revision_shape",
                                            "Choose patch, replace_plan, replace_draft or unresolved")])
        except Exception as error:
            from execution_control import global_failure
            from pipeline.central_office import CouncilExecutionError
            if isinstance(error, (config.ChatJSONError, CouncilExecutionError)) or global_failure(error):
                action_row.update(status="call_failed", error_type=type(error).__name__,
                                  error=str(error))
                if isinstance(error, CouncilExecutionError):
                    action_row["call_failure"] = deepcopy(error.result)
                _save(checkpoint, audit)
                raise
            rejected = _findings(error)
            action_row.update(status="rejected", findings=deepcopy(rejected))
            _save(checkpoint, audit)
            if audit["used_revisions"] >= MAX_REVISIONS:
                if allow_degraded:
                    return _degrade_after_limit(audit, rejected, checkpoint)
                audit["status"] = "revisions_exhausted"
                _save(checkpoint, audit)
                raise PlanConflict(rejected) from error
            # A malformed edit has not changed the current candidate. Ask the
            # remaining shared revision against the original full draft.
            pending_findings = rejected
            continue
        action_row.update(status="applied", after_hash=digest({"schema": wp["world_blueprint"],
                                                                "proposal": proposal}))
        _save(checkpoint, audit)


def revise_supply(run, evidence, audit, checkpoint):
    """Spend only remaining joint revisions on world or materialized deficits."""
    from pipeline.construction import compact_feedback
    from pipeline.factory import validate_seed_input
    sc = run.read("00_input.json")
    pack = validate_seed_input(run, sc)
    feedback = compact_feedback(evidence)
    blueprint_review = evidence.get("phase") == "world_blueprint_review"
    layout_failure = evidence.get("phase") == "business_layout"
    problems = [finding("world_blueprint_review" if blueprint_review else
                         "world_layout_unresolved" if layout_failure else "source_supply_shortfall",
                         "The world author requested an upstream blueprint correction" if blueprint_review else
                         "Fixed business declarations prevent the author from producing compiler-valid values; revise the event, field or schedule constraints using the preserved compiler findings" if layout_failure else
                         "The original materialized-world candidate gate found a line deficit",
                        kind="design", evidence=feedback)]
    return accept(run, audit["draft"], pack, audit["draft"]["supply_plan"],
                  audit, checkpoint, source_findings=problems, allow_degraded=True)
