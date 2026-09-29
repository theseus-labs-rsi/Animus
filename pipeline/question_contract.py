"""Versioned question semantics shared by rendering, publication and scoring.

The first contract version deliberately restricts L1/L3/L5/L6 to deterministic
templates. Arbitrary paraphrases cannot be proved equivalent by keyword checks.
Other lines retain protected-slot checks and expose that weaker validation mode.
No question is repaired by modifying its gold or its world.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import unicodedata

CONTRACT_VERSION = 2
_STRICT_LINES = {"L1_timeline", "L3_process", "L5_conflict", "L6_refusal"}
_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _normal(text) -> str:
    # Normalize presentation only. Minus signs, decimal points, date separators
    # and punctuation inside identifiers must not change value semantics.
    text = unicodedata.normalize("NFKC", str(text or ""))
    removable = set("，,。？！!?：:；;【】「」『』“”‘’")
    return "".join(c for c in text if not c.isspace() and c not in removable)


def _digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _schema(order, wp) -> dict:
    aux = order.get("aux") or {}
    relational = order.get("line") == "L2_relational"
    field = (order.get("answer_field") or (aux.get("path") or [None])[-1]) if relational else order.get("field")
    entity_type = order.get("answer_entity_type") if relational else order.get("entity_type")
    declarations = [f for t in (wp or {}).get("world_blueprint", {}).get("entity_types", [])
                    if entity_type is None or t.get("id") == entity_type
                    for f in t.get("fields", []) if f.get("name") == field]
    if not declarations:
        declarations = [f for f in (wp or {}).get("domain_profile", {}).get("field_schema", []) if f.get("name") == field]
    if declarations:
        if any(declaration != declarations[0] for declaration in declarations[1:]):
            return {"name": field, "kind": "unresolved", "schema_error": "ambiguous_entity_type"}
        return deepcopy(declarations[0])
    schema = deepcopy(aux["value_schema"]) if isinstance(aux.get("value_schema"), dict) and aux["value_schema"].get("kind") else {"kind": aux.get("ans_kind", "text")}
    if "allowed_aliases" in order:
        schema["allowed_aliases"] = deepcopy(order["allowed_aliases"])
    return schema


def bind_question_world(order: dict, ws) -> dict:
    """Bind schema identity to the world; also usable as a release-time oracle."""
    result = deepcopy(order)
    result["entity_type"] = ws.entity_types.get(order.get("entity"))
    if order.get("line") == "L2_relational":
        from pipeline.world_state import gt_multihop
        aux = order.get("aux") or {}
        evidence = gt_multihop(ws, order.get("entity"), aux.get("path") or [], aux.get("at_week")).get("path_evidence", [])
        terminal = evidence[-1] if evidence else {}
        result["answer_entity_type"] = ws.entity_types.get(terminal.get("entity"))
        result["answer_field"] = terminal.get("field")
    return result


def _display_events(order: dict) -> list[dict]:
    """Stable event-card presentation, independent of the correct chronology."""
    events = list(order.get("gt") or [])
    shown = sorted(events, key=lambda event: _digest({"entity": order.get("entity"),
        "card": {key: event.get(key) for key in ("field", "value", "op")}}))
    if len(shown) > 1 and shown == events:
        shown = shown[1:] + shown[:1]
    return shown


def render_intent(order: dict, contract: dict | None = None) -> tuple[str, list]:
    """Use the frozen compiled display; historical v1 retains its original text."""
    from pipeline.lines import line_for
    contract = contract or build_question_contract(order)
    line = line_for(order.get("line", ""))
    _intent, hide = line.intent(order) if line else ("", [])
    return contract.get("canonical_question", _intent), hide


def build_question_contract(order: dict, wp: dict | None = None, *, contract_version: int = CONTRACT_VERSION) -> dict:
    from pipeline.lines import line_for
    from eval.answer_task_review import POLICY_VERSION

    if contract_version not in (1, CONTRACT_VERSION):
        raise ValueError("Unknown question contract version")
    scoring_policy = ((wp or {}).get("quality_contract") or {}).get("scoring_policy")
    if scoring_policy not in (None, POLICY_VERSION):
        raise ValueError("Unknown question scoring policy")

    line_id, cap = order.get("line", ""), order.get("capability", "")
    aux, gold = order.get("aux") or {}, order.get("gt")
    process_trace = cap == "L3_process_trace"
    if process_trace:
        if scoring_policy != POLICY_VERSION:
            raise ValueError("Process references require the task-support semantic policy")
        reference = order.get("reference_proposal")
        if not isinstance(reference, dict) or "answer" not in reference:
            raise ValueError("Process question is missing its proposed natural reference")
        # The executable witness proves only the selected world records. The
        # natural answer remains an independent, publicly reviewed proposal.
        gold = deepcopy(reference["answer"])
    line = line_for(line_id)
    intent, hide = line.intent(order) if line else ("", [])
    schema = {"name": order.get("field", ""), **_schema(order, wp)}
    if cap == "DURATION":
        schema = {"name": order.get("field", ""), "kind": "duration", "unit": aux.get("time_unit") or "周"}
    value_kind = schema.get("kind") or "text"
    answer_kind, abstention_kind = "value", None
    if process_trace:
        answer_kind = "structured"
    elif cap == "L3_order":
        answer_kind = "order"
    elif cap == "FORGET":
        answer_kind, abstention_kind = "abstention", "forgotten"
    elif cap == "ABS" or line_id == "L6_refusal":
        answer_kind = "abstention"
        abstention_kind = {"T2_window": "out_of_scope", "T3_premise": "forgotten"}.get(aux.get("refusal_type"), "never_known")
    elif cap == "TR":
        answer_kind = "time"
    elif value_kind in ("status", "category") or line_id in ("L4_preference", "L7_consolidation", "L9_induction") or cap == "L8_next":
        answer_kind = "enum"
    elif isinstance(gold, list):
        answer_kind = "set"
    elif isinstance(gold, dict) and cap not in ("IE", "MR", "DURATION"):
        answer_kind = "structured"
    scalar = gold.get("value") if isinstance(gold, dict) else gold
    aliases = schema.get("allowed_aliases", [])
    if isinstance(aliases, dict):
        aliases = aliases.get(str(scalar), [])
    aliases = [a for a in aliases if isinstance(a, str) and a.strip()] if isinstance(aliases, list) else []

    unit = aux.get("time_unit") or "周"
    query_time = None
    if process_trace:
        session = (aux.get("process") or {}).get("at_session")
        query_time = {"kind": "period", "ordinal": session + 1 if type(session) is int else None,
                      "unit": unit, "index_base": 1}
    elif line_id == "L6_refusal" and aux.get("refusal_type") == "T2_window":
        period = (aux.get("probe") or {}).get("at_week")
        query_time = {"kind": "period", "ordinal": period, "unit": unit, "index_base": 1}
    elif line_id == "L5_conflict" or cap == "IE" or line_id == "L2_relational":
        session = aux.get("session") if line_id == "L5_conflict" else aux.get("at_week")
        query_time = {"kind": "period", "ordinal": session + 1 if isinstance(session, int) and not isinstance(session, bool) else None,
                      "unit": unit, "index_base": 1}
    elif cap in ("KU", "FORGET") or aux.get("refusal_type") == "T3_premise":
        query_time = {"kind": "latest"}

    event_refs = []
    if cap == "L3_order" and isinstance(gold, list):
        for event in gold:
            if not isinstance(event, dict):
                continue
            identity = {k: event.get(k) for k in ("field", "value", "op", "session", "date")}
            event_refs.append({"id": "event_" + _digest(identity)[:20], **identity})
        if contract_version >= 2:
            shown = _display_events(order)
            cards = "、".join((f"「{event.get('field', '')}」变为 {event.get('value')}"
                if event.get("value") else f"「{event.get('field', '')}」停止统计") for event in shown)
            intent = f"请把【{order.get('entity', '')}】发生的下列事件按发生时间先后排序：{cards}。"

    required = []
    entity = order.get("entity") or ""
    if line_id == "L9_induction":
        required.append({"kind": "literal", "text": aux.get("subject_noun", "工单")})
        required.append({"kind": "literal", "text": str(aux.get("x_star")) + str(aux.get("unit", ""))})
        required.extend({"kind": "literal", "text": option} for option in aux.get("options", []))
    elif entity:
        required.append({"kind": "entity", "text": entity})
    if event_refs:
        for event in event_refs:
            required.append({"kind": "event", "event_id": event["id"], "field": event["field"], "value": event["value"], "op": event["op"]})
    elif line_id == "L2_relational":
        required.extend({"kind": "field", "text": field} for field in aux.get("path", []) if isinstance(field, str))
    elif order.get("field"):
        required.append({"kind": "field", "text": order["field"]})
    if query_time:
        required.append({"kind": "query_time", **query_time, "constraint_type": "query_time"})
    if cap == "DURATION":
        required.append({"kind": "literal", "text": str(aux.get("value"))})

    # Only an exact final answer is forbidden here. Event values, duration
    # targets and multiple-choice options are intentionally visible inputs.
    forbidden = []
    if line_id == "L5_conflict":
        forbidden = [aux.get("authoritative_value"), aux.get("rumor_value")]
    elif line_id in ("L1_timeline", "L2_relational") and cap not in ("TR", "DURATION", "FORGET", "ABS"):
        forbidden = [scalar] if isinstance(scalar, (str, int, float)) else []
    forbidden = [str(value) for value in forbidden if value is not None and str(value) and str(value) != _INSUFFICIENT]

    semantic = {"version": contract_version, "line": line_id, "capability": cap,
                "entity": entity, "entity_type": order.get("entity_type"),
                "answer_entity_type": order.get("answer_entity_type"), "answer_field": order.get("answer_field"),
                "field": order.get("field", ""), "gold": gold,
                "answer_kind": answer_kind, "value_kind": value_kind, "value_schema": schema,
                "allowed_aliases": aliases, "abstention_kind": abstention_kind,
                "query_time": query_time, "event_refs": event_refs,
                "parameters": {key: deepcopy(aux[key]) for key in (
                    "path", "agg", "sub", "value", "rule_id", "options", "refusal_type", "probe",
                    "x_star", "trigger_field", "unit", "subject_noun", "candidates",
                    "duration_end_op", "duration_end_value",
                    "rule", "authoritative_source", "rumor_source", "authoritative_value", "rumor_value") if key in aux}}
    if process_trace:
        from pipeline.process_proposals import CAPABILITY_PURPOSE
        semantic["capability_purpose"] = CAPABILITY_PURPOSE
        semantic.update(reference_proposal=deepcopy(order["reference_proposal"]),
                        canonical_witness=deepcopy(order["gt"]),
                        reference_authority="proposed_not_mechanically_proven")
        semantic["parameters"]["process"] = deepcopy(aux.get("process"))
    if contract_version >= 2:
        semantic["answer_leakage_policy"] = {
            "scope": "question_text_only",
            "excluded_from_question_text": list(dict.fromkeys(forbidden + [str(value) for value in hide if value is not None])),
            "reference_answer_scope": "gold_remains_the_required_answer",
            "visible_inputs": "required_constraints_and_event_cards",
            "legacy_field_scope": "hidden_values_and_forbidden_answer_values_apply_only_to_question_text",
        }
        semantic["time_coordinates"] = {
            "session_index_base": 0, "display_period_index_base": 1,
            "conversion": "display_period=session_index+1",
            "refusal_probe_period_index_base": 1,
            "refusal_probe_scope": "aux.probe.at_week_for_T2_window_is_already_a_display_period",
            "gold_time_format": "preserve_original_gold_fields",
        }
        if event_refs:
            ids = {_digest({key: ref.get(key) for key in ("field", "value", "op", "session", "date")}): ref["id"] for ref in event_refs}
            semantic["event_display"] = {"version": "event-card-order/v1", "policy": "stable_hash_with_non_chronological_guard",
                "event_ids": [ids[_digest({key: event.get(key) for key in ("field", "value", "op", "session", "date")})]
                              for event in _display_events(order)],
                "preserve_display_order_in_question": True}
    contract = {**semantic, "scoring_scope": "primary_answer",
                "required_constraints": required,
                "render_policy": "canonical_template" if line_id in _STRICT_LINES or getattr(line, "deterministic_phrasing", False) else "protected_slots",
                "canonical_question": intent, "forbidden_answer_values": forbidden,
                "hidden_values": [str(value) for value in hide if value is not None],
                "contract_id": "qc_" + _digest(semantic)}
    if scoring_policy:
        contract.update(scoring_scope="task_with_supporting_reasons", scoring_policy=scoring_policy)
        # Keep task metadata, while semantic equivalence is reviewed by an LLM.
        # A fixed-choice renderer may still deliberately emit its exact options.
        contract["render_policy"] = "semantic_review"
        contract["contract_id"] = "qc_" + _digest({**semantic, "scoring_policy": scoring_policy})
    return contract


def attach_question_contract(order: dict, wp: dict | None = None) -> dict:
    result = deepcopy(order)
    result["question_contract"] = build_question_contract(order, wp)
    result["qid"] = order.get("qid") or "q_" + result["question_contract"]["contract_id"][3:27]
    return result


def validate_question(question, contract: dict | None = None) -> list[dict]:
    """Validate a question dict or (text, contract); an empty list means pass.

    Canonical templates are not exempt: missing slots and malformed metadata
    invalidate the fallback as well. This is a bounded template guarantee, not
    a claim that arbitrary natural-language equivalence is decidable.
    """
    order = question if isinstance(question, dict) else None
    if order is not None:
        contract = contract or question.get("question_contract")
        text = question.get("question", "")
    else:
        text = question
    issues = []
    def issue(code, message):
        issues.append({"code": code, "message": message})
    if not isinstance(contract, dict) or contract.get("version") not in (1, CONTRACT_VERSION):
        return [{"code": "missing_contract", "message": "A supported question contract is required"}]
    if (contract.get("value_schema") or {}).get("schema_error"):
        issue("invalid_schema", "Entity-specific field declaration is unresolved")
    if order is not None:
        # The declaration is frozen because this API need not have the original
        # whitepaper. Publication must additionally bind it against that source.
        declaration = contract.get("value_schema") or {}
        policy = contract.get("scoring_policy")
        rebuilt = build_question_contract(order, {
            "domain_profile": {"field_schema": [declaration]},
            "quality_contract": {"scoring_policy": policy}}, contract_version=contract["version"])
        if contract != rebuilt:
            issue("contract_mismatch", "Question metadata no longer matches its frozen semantics")
        if contract.get("canonical_question") != rebuilt.get("canonical_question"):
            issue("template_mismatch", "Canonical template differs from the executable renderer")
    if not isinstance(text, str) or not text.strip():
        return [{"code": "empty_question", "message": "Question text is empty"}]
    normalized = _normal(text)
    canonical = contract.get("canonical_question")
    if not isinstance(canonical, str) or not canonical.strip():
        issue("invalid_template", "No deterministic question template")
    if contract.get("event_display") and normalized != _normal(canonical):
        issue("event_display_changed", "Compiled event cards must retain their frozen presentation order")
    if contract.get("render_policy") == "semantic_review":
        # No substring/template test can decide whether a paraphrase preserves
        # task meaning. Release separately checks the version-bound LLM review.
        return issues
    if contract.get("render_policy") == "canonical_template" and normalized != _normal(canonical):
        issue("unverified_paraphrase", "Paraphrase is outside the verified template language")
    for constraint in contract.get("required_constraints", []):
        kind = constraint.get("constraint_type") or constraint.get("kind")
        if kind in ("entity", "field", "literal"):
            value = _normal(constraint.get("text"))
            if value and value not in normalized:
                issue("missing_" + kind, f"Required {kind} is absent: {constraint.get('text')}")
        elif kind == "event":
            field = constraint.get("field")
            value = constraint.get("value")
            if not field or _normal(field) not in normalized:
                issue("missing_event", f"Event field is absent: {field}")
            elif value not in (None, "") and _normal(f"{field}变为{value}") not in normalized:
                issue("missing_event", f"Target event is not identified: {field}={value}")
            elif value in (None, "") and _normal(f"{field}停止统计") not in normalized:
                issue("missing_event", f"Stop event is not identified: {field}")
        elif kind == "query_time":
            if constraint.get("kind") == "period":
                ordinal = constraint.get("ordinal")
                if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
                    issue("invalid_time", "Query period must be a positive one-based integer")
                elif _normal(f"第{ordinal}{constraint.get('unit')}") not in normalized:
                    issue("missing_time", "Required query period is absent")
            elif not any(word in normalized for word in ("现在", "最新")):
                issue("missing_time", "Latest/current-time qualification is absent")
    if contract.get("answer_kind") == "order" and len(contract.get("event_refs", [])) < 3:
        issue("invalid_events", "At least three explicit event identities are required")
    if contract.get("line") == "L6_refusal" and contract.get("abstention_kind") not in ("never_known", "forgotten", "out_of_scope"):
        issue("invalid_abstention", "Unknown abstention semantics")
    # Ignore values that only occur inside required input slots (an entity
    # named Report2023, a query period, or an event card). The canonical text
    # itself receives this check; fallback is not a blanket exemption.
    leak_text = normalized
    for constraint in contract.get("required_constraints", []):
        if constraint.get("kind") in ("entity", "field"):
            leak_text = leak_text.replace(_normal(constraint.get("text")), "")
        if constraint.get("constraint_type") == "query_time" and constraint.get("kind") == "period":
            leak_text = leak_text.replace(_normal(f"第{constraint.get('ordinal')}{constraint.get('unit')}"), "")
    for value in contract.get("forbidden_answer_values", []):
        needle = _normal(value)
        found = bool(needle and needle in leak_text)
        if needle.isdigit():
            found = bool(re.search(r"(?<!\d)" + re.escape(needle) + r"(?!\d)", leak_text))
        if found:
            issue("answer_leak", "Final answer appears in the question")
    return issues
