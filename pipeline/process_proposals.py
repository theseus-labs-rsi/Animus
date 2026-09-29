"""One bounded author proposal in the original orders stage.

This module verifies frozen-world references, not business conclusions. Natural
references remain proposals for the existing public semantic review.
"""
from copy import deepcopy
import hashlib
import json

VERSION = "original-process-proposal/v2"
CAPABILITY = "L3_process_trace"
STEP = "orders.process_propose"
CAPABILITY_PURPOSE = """理解实际发生的业务变化，以及这些变化与相关对象、处理步骤或后续结果之间的联系。
仅把多个字段查询并列，或只按日期排列记录，本身不等于实现这一能力目的；应按实际任务含义判断，不按字段数量、跳数、关键词或固定问句判定。
解题可以需要理解这些联系而最终只给一个简短结论；不强制答题者报告中间步骤、长理由或额外解释。
只依据冻结世界中的实际记录提出问题，不从单次发生或内部因果标签推出普遍必然、唯一原因或反事实。"""
SYSTEM = """你是原流水线的业务过程问题作者。只依据给定冻结世界与任务，提出值得让公开正文读者回答的业务过程问题。
阅读实际对象、参与者、关系、事件 effects 与时点；事件名称不是业务正确性的证明，发生先后也不自动等于因果。
内部世界是作者的事实依据，不是未来答题者可见的材料。问题的关键前提与理由必须能够由实际公开正文支持，后续会独立审阅；不要假定正文已经表达了内部关系。
自然意图面向公开读者，使用对象的实际名称与公开时间表达，不把 evt-X、session=2 等内部定位号当题面条件；witness_refs 中仍须使用原内部 ID 定位。
不要补造规则、事实、唯一原因或反事实。已发生的实例不自动证明普遍规则、必然下一步或未发生的结果。
可提出比上限少的问题，包括零题；不要为凑数写排序换皮题。自由说明有待证实的前提，不强制回答长度或固定业务模板。
每条给出完整自然意图、真实引用和待审参考。witness_refs 只填原事件/关系 ID 和实际字段定位，不能提交改写的世界事实。
输出 JSON：{"proposals":[{"id":"p1","entity":"原实体全名","at_session":0,"intent":"自然任务与范围","witness_refs":{"event_ids":["原事件ID"],"relation_ids":[],"facts":[{"entity":"原实体","field":"原字段","session":0}]},"reference_proposal":{"answer":"待审答案，也可为JSON值","rationale":"理由"},"unresolved_preconditions":"尚需公开正文说明的前提；没有则空串"}],"reason":"提案说明或为何无合适问题"}。
数组示例不是要求存在该事件或该数量；只用本轮真实引用。任何参考都只是作者意见，不是机械认证的正确答案。"""
SYSTEM += "\n本能力目的（不是答案模板）：\n" + CAPABILITY_PURPOSE


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a string")
    return value


def _session(value, ws):
    if type(value) is not int or not 0 <= value < ws.n_sessions:
        raise ValueError("Session outside frozen world")
    return value


def _typed(wp, ws):
    from eval.answer_task_review import POLICY_VERSION
    quality = wp.get("quality_contract") or {}
    if (quality.get("scoring_policy") != POLICY_VERSION
            or quality.get("public_semantic_review") is not True
            or quality.get("isolated_reference_audit") is not True):
        raise ValueError("Process proposals require policy A and isolated public semantic review")
    if not ws.world_blueprint or ws.world_blueprint.get("legacy_adapter"):
        raise ValueError("Process proposals require a frozen typed world")


def _author_world(world):
    """Keep all business facts/plan semantics, omit only internal audit envelopes.

    This is still a private author input, never a public solver view. The full
    unprojected world remains the binding identity below. Historical prompts
    and receipts are redundant evidence of how the plan was made, not another
    source of business facts to feed back into question authorship.
    """
    projected = deepcopy(world)
    plan = projected.get("disclosure")
    if isinstance(plan, dict):
        for key in ("author_input", "source_inputs", "binding", "call", "parts", "batch_binding", "transcript"):
            plan.pop(key, None)
    return projected


def prepare_process_proposals(wp, ws, *, target, model, max_tokens=8192):
    _typed(wp, ws)
    if type(target) is not int or target < 0:
        raise ValueError("target must be a nonnegative integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    _text(model, "model")
    world = deepcopy(ws.to_dict())
    paper = deepcopy(wp)
    if (wp.get("seed_contract") or {}).get("schema_version") == 2:
        from pipeline.seed_world import seed_business_context
        paper = {key: deepcopy(wp[key]) for key in ("domain_profile", "world_blueprint",
            "shared_world_spec", "quality_contract", "traps", "capability_targets", "active_lines") if key in wp}
        paper["seed_business_context"] = seed_business_context(wp)
    payload = {"target_maximum": target, "whitepaper": paper, "world": _author_world(world),
               "scope": "Author-only frozen facts; no public-answerability certification"}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
    binding = {"version": VERSION, "wp_hash": _hash(wp), "world_hash": _hash(world),
               "target": target, "model": model, "max_tokens": max_tokens,
               "messages_hash": _hash(messages)}
    return {"binding": binding, "messages": messages}


def rebuild_witness(ws, entity, at_session, refs):
    """Resolve only declared locators, preserving canonical objects verbatim."""
    _session(at_session, ws)
    if entity not in ws.entities:
        raise ValueError("Focus entity missing from frozen world")
    if not isinstance(refs, dict) or set(refs) != {"event_ids", "relation_ids", "facts"}:
        raise ValueError("Invalid witness reference structure")
    events_by_id = {x["id"]: x for x in ws.events}
    relations_by_id = {x["id"]: x for x in ws.relations}
    if len(events_by_id) != len(ws.events) or len(relations_by_id) != len(ws.relations):
        raise ValueError("Duplicate canonical identity")
    selected = {}
    for key, lookup in (("event_ids", events_by_id), ("relation_ids", relations_by_id)):
        ids = refs[key]
        if (not isinstance(ids, list) or any(not isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids) or (key == "event_ids" and not ids)):
            raise ValueError("Witness IDs must be unique; at least one actual event required")
        if any(x not in lookup for x in ids):
            raise ValueError("Witness ID not in frozen world")
        if any(lookup[x]["session"] > at_session for x in ids):
            raise ValueError("Witness event/relation is later than the requested scope")
        # Input order is not a proposed causal sequence; retain canonical order.
        selected[key] = [deepcopy(item) for item in lookup.values() if item["id"] in ids]
    if not isinstance(refs["facts"], list):
        raise ValueError("facts must be a list")
    facts, seen = [], set()
    for ref in refs["facts"]:
        if not isinstance(ref, dict) or set(ref) != {"entity", "field", "session"}:
            raise ValueError("Invalid field locator")
        name, field, session = ref["entity"], ref["field"], _session(ref["session"], ws)
        if not isinstance(name, str) or not isinstance(field, str):
            raise ValueError("Field locator names must be strings")
        key = (name, field, session)
        if key in seen or session > at_session:
            raise ValueError("Duplicate or future field locator")
        seen.add(key)
        timeline = ws.entities.get(name, {}).get(field)
        if timeline is None:
            raise ValueError("Field locator missing from frozen world")
        history = [{"session": op.session, "date": op.date, "op": op.op, "value": op.value}
                   for op in timeline._sorted() if op.session <= session]
        facts.append({**deepcopy(ref), "known": bool(history),
                      "value": timeline.value_at_session(session), "history": history})
    facts.sort(key=lambda row: (row["entity"], row["field"], row["session"]))
    events = selected["event_ids"]
    event_ids = {event["id"] for event in events}
    edges = [{"parent_id": event["caused_by"], "child_id": event["id"]}
             for event in events if event.get("caused_by") in event_ids]
    return {"events": events, "relations": selected["relation_ids"],
            "facts": facts, "causal_edges": edges}


def _proposal_order(proposal, ws):
    required = {"id", "entity", "at_session", "intent", "witness_refs",
                "reference_proposal", "unresolved_preconditions"}
    if not isinstance(proposal, dict) or not required.issubset(proposal):
        raise ValueError("Process proposal is missing required fields")
    for key in ("id", "entity", "intent"):
        _text(proposal[key], key)
    _text(proposal["unresolved_preconditions"], "unresolved_preconditions", empty=True)
    ref = proposal["reference_proposal"]
    if not isinstance(ref, dict) or not {"answer", "rationale"}.issubset(ref):
        raise ValueError("Explicit answer and rationale required")
    _text(ref["rationale"], "rationale", empty=True)
    _hash(ref)  # JSON values including null/empty containers are preserved.
    witness = rebuild_witness(ws, proposal["entity"], proposal["at_session"], proposal["witness_refs"])
    normalized = {key: deepcopy(proposal[key]) for key in sorted(required)}
    process = {"version": VERSION, "proposal_id": proposal["id"],
               "world_hash": _hash(ws.to_dict()), "proposal_hash": _hash(normalized),
               **{key: deepcopy(proposal[key]) for key in
                  ("intent", "at_session", "witness_refs", "unresolved_preconditions")}}
    sessions = {item["session"] for key in ("events", "relations", "facts") for item in witness[key]}
    sessions.update(op["session"] for fact in witness["facts"] for op in fact["history"])
    return {"line": "L3_process", "capability": CAPABILITY, "entity": proposal["entity"],
            "field": "", "gt": witness, "evidence_sessions": sorted(sessions),
            "reference_proposal": deepcopy(ref),
            "aux": {"scorer": "semantic", "gt_scope": "canonical_witness_not_semantic_proof",
                    "time_unit": ws.period_unit(), "process": process}}


def validate_process_order(order, ws):
    if order.get("line") != "L3_process" or order.get("capability") != CAPABILITY:
        raise ValueError("Not a typed process order")
    aux = order.get("aux") or {}
    process = aux.get("process") or {}
    if process.get("version") != VERSION or process.get("world_hash") != _hash(ws.to_dict()):
        raise ValueError("Process order belongs to a different frozen world/version")
    proposal = {"id": process.get("proposal_id"), "entity": order.get("entity"),
                "reference_proposal": deepcopy(order.get("reference_proposal")),
                **{key: deepcopy(process.get(key)) for key in
                   ("intent", "at_session", "witness_refs", "unresolved_preconditions")}}
    rebuilt = _proposal_order(proposal, ws)
    for key in ("entity", "field", "gt", "evidence_sessions", "reference_proposal", "aux"):
        if rebuilt[key] != order.get(key):
            raise ValueError(f"Process order differs from its recorded proposal/witness: {key}")
    return deepcopy(rebuilt["gt"])


def _derive(raw, ws, target):
    _hash(raw)  # Refuse non-JSON/nonfinite output; never silently serialize it.
    if (not isinstance(raw, dict) or not isinstance(raw.get("proposals"), list)
            or not isinstance(raw.get("reason"), str)):
        raise ValueError("Expected proposals array and reason string")
    if len(raw["proposals"]) > target:
        raise ValueError("Proposal count exceeds the frozen maximum; no silent truncation")
    ids = [item.get("id") for item in raw["proposals"] if isinstance(item, dict)]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Proposal IDs must be unique strings")
    items, orders, identities = [], [], set()
    for index, proposal in enumerate(raw["proposals"]):
        row = {"index": index, "proposal": deepcopy(proposal), "status": "rejected", "reason": None}
        try:
            order = _proposal_order(proposal, ws)
            identity = _hash({key: proposal[key] for key in proposal
                              if key in {"entity", "at_session", "intent", "witness_refs",
                                         "reference_proposal", "unresolved_preconditions"}})
            if identity in identities:
                raise ValueError("Exact duplicate proposal content under a different ID")
            identities.add(identity)
            orders.append(order)
            row.update(status="structurally_bound", order=deepcopy(order))
        except (ValueError, TypeError, KeyError) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        items.append(row)
    return items, orders


def validate_process_report(report, wp, ws):
    if isinstance(report, dict) and report.get("strategy") == "process-batches/v1":
        from pipeline.process_batches import validate
        return validate(report, wp, ws)
    if (not isinstance(report, dict) or report.get("version") != VERSION
            or report.get("status") != "completed"):
        raise ValueError("Process proposal execution is not complete")
    binding = report.get("binding") or {}
    prepared = prepare_process_proposals(wp, ws, target=binding.get("target"),
        model=binding.get("model"), max_tokens=binding.get("max_tokens"))
    if report.get("binding") != prepared["binding"] or report.get("messages") != prepared["messages"]:
        raise ValueError("Process proposal binding/messages changed")
    if report.get("calls_used") != (1 if binding["target"] else 0):
        raise ValueError("Invalid proposal call denominator")
    items, orders = _derive(report.get("raw_output"), ws, binding["target"])
    if report.get("items") != items or report.get("orders") != orders:
        raise ValueError("Process report differs from preserved raw proposals")
    return deepcopy(orders)


def propose_process_orders(wp, ws, *, target, chat_json, model, max_calls=1,
                           max_tokens=8192, max_input_chars=200000, record=None, checkpoint_path=None):
    prepared = prepare_process_proposals(wp, ws, target=target, model=model, max_tokens=max_tokens)
    if checkpoint_path is not None and target > 0 and (target > 8 or
            sum(len(m["content"]) for m in prepared["messages"]) > max_input_chars):
        from pipeline.process_batches import propose
        return propose(wp, ws, target=target, chat_json=chat_json, model=model,
                       max_tokens=max_tokens, checkpoint_path=checkpoint_path)
    if type(max_calls) is not int or max_calls not in (0, 1):
        raise ValueError("One-attempt proposal budget must be zero or one")
    if type(max_input_chars) is not int or max_input_chars <= 0:
        raise ValueError("max_input_chars must be positive")
    report = {"version": VERSION, "status": "not_started", **deepcopy(prepared),
              "calls_used": 0, "raw_output": None, "items": [], "orders": [], "error": None}
    try:
        if sum(len(m["content"]) for m in prepared["messages"]) > max_input_chars:
            raise ValueError("Full process proposal input exceeds budget; no truncation")
        if target == 0:
            raw = {"proposals": [], "reason": "Frozen proposal budget is zero"}
        else:
            if not max_calls:
                raise ValueError("Proposal call budget exhausted")
            if record:
                record({"event": "started", "step": STEP, "binding": deepcopy(prepared["binding"])})
            report["calls_used"] = 1
            raw = chat_json(STEP, deepcopy(prepared["messages"]), model=model, temperature=0.3,
                            top_p=1, max_tokens=max_tokens, retries=1, strict_json=True,
                            response_format={"type": "json_object"})
            report["raw_output"] = deepcopy(raw)
            if record:
                record({"event": "finished", "step": STEP, "raw_output": deepcopy(raw)})
        report["raw_output"] = deepcopy(raw)
        if isinstance(raw, dict) and "__error__" in raw:
            raise RuntimeError("Provider returned an execution error")
        if prepared != prepare_process_proposals(wp, ws, target=target, model=model, max_tokens=max_tokens):
            raise ValueError("Frozen world/whitepaper changed during proposal execution")
        report["items"], report["orders"] = _derive(raw, ws, target)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", orders=[])
    return report
