"""One original-world author plan, plus structure-owned initial-state checks.

The plan coordinates authors; it is never compiled as canonical evidence.
Validation checks bindings/ownership, not business correctness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pipeline.seed_world import seed_business_context
from pipeline.world_blueprint import WorldBlueprintError, relation_owner_side
from pipeline.value_types import parse_date, parse_number

VERSION = "original-world-joint-plan/v2"
SYSTEM = """你是原世界生成阶段的作者。先为冻结任务与蓝图安排一个一致、规模有限的业务小故事，供后续字段作者和关系事件作者共同实现。此计划是可质疑的创作提案，不是已发生的 canonical 事实，不是审阅通过或覆盖凭据。
输入给出完整任务说明、业务机制、实际 session 日历、冻结蓝图和已有世界。不得改任务、类型、字段、数量或已有事实，不读取或编造问题、金标、rubric 或原始来源资料。
先联合考虑对象各自的身份和用途、时间/期间含义、窗口开始已存在的状态、需要体现的业务变化及其前后依赖，再分配内在字段。说明哪些事实应保持稳定，哪些变化有任务依据；静态对象或合法未触发条件无需强行变化，不把可选机制变成每个实体的义务。
若任务确实要求展示某机制，安排能体现它的具体前后过程，不能只用事件名称、因果边或字段数量代替。首次建立与改变既有状态应区分；日期字段与实际日历的关系应明确，不凭领域名称猜历法。计划不必把所有条件都触发。
plan 用简洁自然语言和角色分组协调共同故事、必要起始状态与跨期变化；不要为每个实体/字段/每期展开表格，也不预先编排一份固定专名清单。专名由后续类型作者生成，后续作者将看到前批实际对象和事实。已有实体不能重命名或重建。若窗口或蓝图不足以实现任务，在 limitations 如实说明，不偷偷扩容。
只输出一个完整 JSON：{"plan":"联合故事、角色用途、时间含义、起始事实、必要变化与合法不变项","limitations":"局限或无"}。"""

BASELINE_INSTRUCTION = """
【结构作者独占的窗口初态】
除完整 relations/events，可返回 initial_states 数组；缺省或省略表示保留当前候选已存初态（首次为空）。每项严格为 {"entity":"本次新增实体","field":"该实体类型的 event-owned 非 relation 字段","session":0,"value":"符合字段声明的起始值"}。
只在业务上需要窗口开始已有状态时填写，不把初态当成新事件或机制见证。不得为已有世界实体写初态，不得写内在字段/关系字段；每个实体字段只能一项；初态和 event effect 不能占用相同 session-0 槽，即使值相同也禁止。后续事件仍须相对初态造成真实变化。返回 [] 可清除本次候选初态；原已有世界不受影响。
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def implementation_hashes():
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("world_joint_plan.py", "world_gen.py", "seed_world.py", "seed_pack.py", "seed_v2.py",
                         "world_blueprint.py", "world_state.py", "value_types.py", "prompts.py")}


def make_context(wp, blueprint, calendar, existing=None):
    old = existing.to_dict() if existing is not None else {}
    return {**seed_business_context(wp), "calendar": deepcopy(calendar),
            "blueprint": deepcopy(blueprint),
            "existing_world": {key: deepcopy(old[key]) for key in
                               ("entities", "entity_types", "relations", "events",
                                "cascades", "absent_fields", "n_sessions") if key in old}}


RESPONSE_SCHEMA = {
    "type": "object", "required": ["plan", "limitations"],
    "additionalProperties": False,
    "properties": {key: {"type": "string", "minLength": 1}
                   for key in ("plan", "limitations")},
}


def _shape_issues(raw):
    """Only returned JSON shape; no business or semantic judgement."""
    if not isinstance(raw, dict):
        return {"expected_type": "object", "actual_type": type(raw).__name__}
    issues = {}
    missing = sorted({"plan", "limitations"} - set(raw))
    extra = sorted(set(raw) - {"plan", "limitations"})
    invalid = [key for key in ("plan", "limitations") if key in raw
               and (not isinstance(raw[key], str) or not raw[key].strip())]
    if missing:
        issues["missing_keys"] = missing
    if extra:
        issues["unexpected_keys"] = extra
    if invalid:
        issues["nonempty_string_required"] = invalid
    return issues


def _check_raw(raw, context):
    issues = _shape_issues(raw)
    if issues:
        raise WorldBlueprintError("joint plan shape: " + json.dumps(issues, ensure_ascii=False))


def _messages(context):
    return [{"role": "system", "content": SYSTEM + "\n输出结构（字符串不得为空白）："
             + json.dumps(RESPONSE_SCHEMA, ensure_ascii=False)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]


def _repair_messages(context, raw, issues):
    return _messages(context) + [
        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
        {"role": "user", "content": json.dumps({
            "format_repair": "上次返回未满足结构合同。仅修正所列字段/类型问题，保持原任务和业务安排；"
                             "重发完整 JSON，不添加解释或新业务要求。此次为唯一一次结构纠错。",
            "issues": issues, "response_schema": RESPONSE_SCHEMA}, ensure_ascii=False)}]


def author_plan(wp, context, tracer, *, model, max_tokens):
    """One author call, at most one correction of a returned JSON shape error."""
    params = {"model": model, "temperature": 0.4, "max_tokens": max_tokens,
              "retries": 1, "strict_json": True, "response_format": {"type": "json_object"}}
    result = {"version": VERSION, "status": "error", "input": deepcopy(context),
              "messages": _messages(context), "parameters": params, "raw_output": None,
              "attempts": []}
    binding = {"wp_hash": digest(wp), "input_hash": digest(context),
               "parameters_hash": digest(params),
               "implementation_hashes": implementation_hashes()}
    for index in range(2):
        attempt = {"step": "world.joint_plan", "attempt": index + 1,
                   "messages": deepcopy(result["messages"]), "parameters": deepcopy(params),
                   "raw_output": None, "status": "execution_error"}
        result["attempts"].append(attempt)
        result["raw_output"] = None
        try:
            if implementation_hashes() != binding["implementation_hashes"]:
                raise WorldBlueprintError("joint plan implementation drift before dispatch")
            raw = tracer.chat_json("world.joint_plan", deepcopy(attempt["messages"]), **params)
            attempt["raw_output"] = deepcopy(raw)
            result["raw_output"] = deepcopy(raw)
            # Tracer converts provider/parser exceptions to this reserved sentinel.
            if isinstance(raw, dict) and "__error__" in raw:
                raise RuntimeError("joint plan execution failed: " + str(raw["__error__"]))
            if implementation_hashes() != binding["implementation_hashes"]:
                raise WorldBlueprintError("joint plan implementation drift after response")
        except Exception as error:
            result["error"] = attempt["error"] = f"{type(error).__name__}: {error}"
            break
        issues = _shape_issues(raw)
        attempt["issues"] = deepcopy(issues)
        if not issues:
            attempt["status"] = "ready"
            result["status"] = "ready"
            break
        attempt["status"] = "invalid_shape"
        if index == 1:
            result["error"] = "joint plan shape: " + json.dumps(issues, ensure_ascii=False)
            break
        result["messages"] = _repair_messages(context, raw, issues)
    result["binding"] = {**binding, "messages_hash": digest(result["messages"]),
                         "raw_hash": digest(result["raw_output"]),
                         "attempts_hash": digest(result["attempts"])}
    result["plan_hash"] = digest(result)
    return result


def validate_plan(plan, wp, context):
    """Recompute original inputs, full raw and current implementation identity."""
    try:
        if not isinstance(plan, dict) or plan.get("version") != VERSION or plan.get("status") != "ready":
            raise WorldBlueprintError("no valid joint author plan")
        if plan.get("plan_hash") != digest({k: v for k, v in plan.items() if k != "plan_hash"}):
            raise WorldBlueprintError("joint plan envelope hash mismatch")
        params = plan["parameters"]
        if (not isinstance(params, dict) or params.get("retries") != 1
                or params.get("strict_json") is not True or params.get("temperature") != 0.4
                or params.get("response_format") != {"type": "json_object"}):
            raise WorldBlueprintError("joint plan call contract mismatch")
        attempts = plan["attempts"]
        if not isinstance(attempts, list) or len(attempts) not in (1, 2):
            raise WorldBlueprintError("joint plan attempt budget mismatch")
        messages = _messages(context)
        for index, attempt in enumerate(attempts):
            raw = attempt["raw_output"]
            if isinstance(raw, dict) and "__error__" in raw:
                raise WorldBlueprintError("execution failure cannot become a shape correction")
            issues = _shape_issues(raw)
            expected_status = "invalid_shape" if index < len(attempts) - 1 else "ready"
            if (attempt.get("step") != "world.joint_plan" or attempt.get("attempt") != index + 1
                    or attempt.get("messages") != messages or attempt.get("parameters") != params
                    or attempt.get("status") != expected_status or attempt.get("issues") != issues
                    or bool(issues) != (expected_status == "invalid_shape")):
                raise WorldBlueprintError("joint plan attempt/format-repair binding mismatch")
            if issues:
                messages = _repair_messages(context, raw, issues)
        if plan["raw_output"] != attempts[-1]["raw_output"]:
            raise WorldBlueprintError("joint plan final raw mismatch")
        expected = {"wp_hash": digest(wp), "input_hash": digest(context),
                    "messages_hash": digest(messages), "parameters_hash": digest(params),
                    "implementation_hashes": implementation_hashes(),
                    "raw_hash": digest(plan["raw_output"]), "attempts_hash": digest(attempts)}
        if (plan.get("binding") != expected or plan.get("input") != context
                or plan.get("messages") != messages):
            raise WorldBlueprintError("joint plan input/source binding mismatch")
        _check_raw(plan["raw_output"], context)
        return []
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"{type(error).__name__}: {error}"]


def shared_prompt(plan):
    return ("\n【共同作者上下文：冻结任务、日历与同一联合提案】\n"
            + json.dumps({"context": plan["input"], "plan_hash": plan["plan_hash"],
                          "proposal": plan["raw_output"]}, ensure_ascii=False)
            + "\n仅实现本角色拥有的字段/结构；提案中的说明不能代替实际 world 事实。"
              "原任务优先；不得为了凑机制强迫合法稳定对象变化。")


def apply_initial_states(merged, structure, blueprint, existing_names=(), previous=()):
    """Return a fresh entity list and validated baseline rows, never edit inputs."""
    rows = structure.get("initial_states", list(previous))
    if not isinstance(rows, list):
        raise WorldBlueprintError("structure initial_states must be a list")
    entities = deepcopy(merged["entities"])
    by_name = {e["name"]: e for e in entities}
    field_specs = {t["id"]: {f["name"]: f for f in t.get("fields", [])}
                   for t in blueprint["entity_types"]}
    relation_fields, event_fields = {}, {}
    for rel in blueprint.get("relation_types", []):
        owner = rel["from_type"] if relation_owner_side(blueprint, rel) == "from" else rel["to_type"]
        relation_fields.setdefault(owner, set()).add(rel["field"])
    for event in blueprint.get("event_types", []):
        for effect in event.get("effect_fields", []):
            event_fields.setdefault(event["roles"][effect["role"]], set()).add(effect["field"])
    # Remove only baseline fields actually owned by the previous structure proposal.
    for row in previous:
        entity = by_name.get(row["entity"])
        expected = {"type": "stable", "value": row["value"]}
        if entity is None or entity.get("fields", {}).get(row["field"]) != expected:
            raise WorldBlueprintError("saved initial-state field binding mismatch")
        del entity["fields"][row["field"]]
    occupied = set()
    for row in rows:
        try:
            if (not isinstance(row, dict) or set(row) != {"entity", "field", "session", "value"}
                    or not isinstance(row["entity"], str) or not isinstance(row["field"], str)
                    or type(row["session"]) is not int or row["session"] != 0):
                raise WorldBlueprintError("initial state requires entity/field/value and integer session 0")
            name, field, value = row["entity"], row["field"], row["value"]
            if name in existing_names or name not in by_name:
                raise WorldBlueprintError("initial state may only address this build's new entities")
            entity = by_name[name]
            tid = entity["type"]
            if field not in event_fields.get(tid, set()) - relation_fields.get(tid, set()):
                raise WorldBlueprintError("initial state must be event-owned and not relation-owned")
            if (name, field) in occupied or field in entity.get("fields", {}):
                raise WorldBlueprintError("initial state duplicate or already-written field")
            occupied.add((name, field))
            decl = field_specs[tid][field]
            if (isinstance(value, (dict, list, bool)) or value is None
                    or not isinstance(value, (str, int, float)) or not str(value).strip()):
                raise WorldBlueprintError("initial state value must be a nonempty scalar of the declared kind")
            if decl.get("kind") == "numeric":
                number, _ = parse_number(value, decl.get("unit"))
                if decl.get("range") and not decl["range"][0] <= number <= decl["range"][1]:
                    raise WorldBlueprintError("initial state numeric value outside declared range")
            elif not isinstance(value, str):
                raise WorldBlueprintError("initial state nonnumeric value must be a string")
            if decl.get("kind") == "date":
                parse_date(value)
            if decl.get("states") and value not in decl["states"]:
                raise WorldBlueprintError("initial state value outside declared states")
            entity.setdefault("fields", {})[field] = {"type": "stable", "value": value}
        except ValueError as error:
            name = row.get("entity") if isinstance(row, dict) else None
            field = row.get("field") if isinstance(row, dict) else None
            declaration = (field_specs.get(by_name.get(name, {}).get("type"), {}).get(field, {})
                           if isinstance(name, str) and isinstance(field, str) else {})
            detail = {"entity": name, "field": field, "session": row.get("session") if isinstance(row, dict) else None,
                      "declared_field": declaration, "received_value": row.get("value") if isinstance(row, dict) else row}
            raise type(error)(str(error) + "; initial_state=" + json.dumps(detail, ensure_ascii=False)) from error
    for event in structure.get("events", []):
        if isinstance(event, dict) and event.get("session") == 0:
            for effect in event.get("effects", []):
                if isinstance(effect, dict) and (effect.get("entity"), effect.get("field")) in occupied:
                    raise WorldBlueprintError("initial state and event share a session-0 effect slot")
    return entities, deepcopy(rows)
