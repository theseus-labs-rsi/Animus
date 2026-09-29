"""Agent-directed work units inside the original world stage.

The planner chooses business boundaries and reads; the compiler only checks the
returned facts. Accepted work is an ordered log. Revising a unit rolls back its
suffix, which keeps dependency invalidation explicit without a second graph.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import config
from pipeline.seed_world import seed_business_context, seed_protected_fields, seed_world_issues, validate_seed_world
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint, relation_owner_side
from pipeline.world_joint_plan import apply_initial_states, BASELINE_INSTRUCTION
from pipeline.world_state import WorldState, assemble_world, validate
from pipeline.blueprint_feasibility import BlueprintReviewRequested

VERSION = "original-world-agent/v3"
TABLE_KEYS = ("entities", "relations", "events", "initial_states")
MAX_CONTEXT_CHARS = 120_000
FORMAT_ATTEMPTS = 3
PLANNER_SYSTEM = """你管理原 pipeline 的世界生成。任务边界、共享对象、先后依赖由你按业务判断，计划可随实际结果修订。每次只输出一个动作 JSON。
目标是逐步建立同一个世界。优先让完整业务过程的对象、关系、初态、事件一起生成；不要按固定实体数、类型或时间窗口机械切割。先安排整体布局，再细化下一项工作；可合并尚未完成的任务。合法稳定、未知、未触发情形可以保留。
所有原任务/蓝图/日历保持不变；数量为全局最低要求，每个工作单元无需满足全部最低要求。机制需要真实对象上的前后过程，文字计划和数量本身不能证明业务机制成立。
可用动作：
1. {"action":"inspect","reason":"为何需要这些事实","read_entities":["已有专名"],"read_events":["已有事件id"],"read_units":["可选，撤回/已接受单元id"]}：按引用读取实际完整字段历史及有关结构，下轮看到结果。
2. {"action":"write","unit_id":"本次唯一id","intent":"完整业务过程及边界、需要新增的角色/对象、落实的机制与先后过程","plan":"简洁的剩余工作安排","read_entities":["本任务会引用的已有实体"],"read_events":["依赖的已有事件id"]}：委托作者补充一段实际世界。旧实体须读取，作者只新建实体，旧对象的变化通过新关系/事件表达。不要重复输出已接受的事实。
3. {"action":"revise","unit_id":"需要修改的已接受单元id","reason":"原事实哪里有问题，如何重新组织"}：撤回该单元和它后面全部单元；撤回事实将不再可引用。随后用 write 重新安排，可合并/拆分。更早的单元保持不变；撤回原稿仍可 inspect，通过 read_units 请求。
4. {"action":"finish","reason":"全局事实已完整，说明机制怎样落实","issue_responses":[{"issue_id":"如有外部审阅问题，逐项填写","disposition":"addressed|disputed|deferred","response":"结合原任务与真实事实给出回应依据"}]}：交完整编译和种子验收；失败会返回问题，继续补写/修订。之后仍需原系统世界语义审阅和公开安排闸。
5. {"action":"report_unresolved","reason":"具体无法完成的要求或上下文问题"}：如实停止。
6. {"action":"request_blueprint_review","reason":"原任务要求与哪项蓝图约束冲突，引用失败证据","suggestion":"建议上游复核哪些字段约束"}：请求原白皮书作者复核。你不能改蓝图；上游可能拒绝修改。先区分自身事实错误与约束矛盾，不能仅因生成失败就要求删业务要求。
同类错误反复出现时，结合 prior_failures 重新判断原因；可 inspect 读全旧事实，revise 更早的错误单元，或请求上游约束复核。循环确认/重开若受到 states 单向顺序阻碍，应提交矛盾证据。不要通过更换 unit_id 反复提交相同错误。occupied_ids 只列全局已用编号，实际业务事实需按引用读取。
执行输出不足或上下文过大时缩小工作单元，保留已接受成果。外部审阅意见可以质疑，须基于原任务和事实逐项回应；可修改，也可给出依据反驳并保持事实，后续原语义闸将独立重审。新审阅反馈不得无回应直接 finish。
只输出该动作所需字段，禁止编造问题、答案、评分或另建世界。"""

AUTHOR_SYSTEM = """你是原世界阶段的一位业务作者。完成 planner 指定的一个业务过程。输入含冻结业务要求/完整蓝图/日历，以及明确读取的真实世界上下文。只写本单元增量。
把同一业务过程的内在字段、关系、初态、事件和因果共同考虑。保持公司/报告期/版本/来源/人物身份一致；业务含义由你理解，程序只检查类型与引用。若缺少信息，可返回 {"needs_context":"具体缺什么，建议读取哪些已有对象"}，不得猜测未读的已有事实。
严格 JSON：{"entities":[{"name":"唯一专名","type":"蓝图类型id","purpose":"可选作者用途说明","fields":{"内在字段":{"type":"stable","value":"值"},"变化字段":{"type":"evolving","trajectory":[{"session":0,"value":"值"},{"session":2,"value":"另一值"}]}}}],"relations":[{"id":"唯一id","type":"关系类型id","from":"实体名","to":"实体名","session":0}],"events":[{"id":"唯一id","type":"事件类型id","session":1,"participants":{"角色":"实体名"},"effects":[{"entity":"角色所指实体","field":"声明effect字段","set":"新值"}],"caused_by":"可省的父事件id"}],"initial_states":[]}。
entities 只创建新对象，必须包含本类型全部内在字段。relation-owned/event-owned 字段仅通过关系、事件与合法 initial_states 生成，禁止在 entities.fields 暗写结构字段。静态关系 session=0，时变关系按业务安排，禁止重复边凑数量。事件角色准确且互异，effects 恰好覆盖声明，每个效果造成真实变化；因果 caused_by 引用真实父事件并遵守声明精确时间差。父事件可以是本次新增，也可以是明确读取的已有事件。
对已有实体仅引用上下文里完整读取的实体，不改写其既有字段；所有旧事件/关系保留，新增实例id唯一。同一字段同一期不得双写冲突。数量目标在整个世界验收，本单位只完成指定业务范围，禁止填满全世界。不要输出 cascades 或额外事实字段。
""" + BASELINE_INSTRUCTION + """
回复只能包含一个完整 JSON 对象，不得在对象前后添加说明或代码围栏。
业务留痕通过蓝图声明的事实字段、关系和事件表达；若任务要求的留痕无法用当前蓝图表达，返回 needs_context 说明缺口，交由 planner 重排任务。
"""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _binding(wp, existing):
    directory = Path(__file__).parent
    return {"version": VERSION, "wp_hash": _digest(wp),
            "existing_hash": _digest(existing.to_dict() if existing is not None else None),
            "model": config.STRUCTURE_MODEL,
            "implementation": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in ("world_agent.py", "world_gen.py", "world_state.py", "world_blueprint.py",
                                            "world_joint_plan.py", "seed_world.py", "value_types.py", "blueprint_feasibility.py")}}


def _save(path, state):
    state["checkpoint_hash"] = _digest({k: v for k, v in state.items() if k != "checkpoint_hash"})
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _empty():
    return {key: [] for key in TABLE_KEYS}


def _combine(units):
    table = _empty()
    for unit in units:
        for key in TABLE_KEYS:
            table[key].extend(deepcopy(unit["raw"][key]))
    return table


def _compiled(table, blueprint, existing, complete, wp=None):
    applied = deepcopy(table)
    applied["entities"], initial = apply_initial_states(
        table, table, blueprint, existing.entities if existing is not None else ())
    applied.pop("initial_states")
    applied.update(cascades=[], absent_fields=[])
    world, issues = assemble_world(applied, blueprint=blueprint, existing=existing,
                                   require_complete=complete, include_shape_diagnostics=False)
    # Preserve the original declared state/range/monotonic checks. Numeric trend
    # diagnostics are explicitly advisory in the original world gate.
    defects = []
    for kind in blueprint["entity_types"]:
        profile = {"field_schema": kind["fields"],
                   "state_machines": [{"field": field["name"], "states": field["states"]}
                                      for field in kind["fields"] if field.get("states")]}
        typed_world = deepcopy(world)
        typed_world.entities = {name: fields for name, fields in world.entities.items()
                                if world.entity_types.get(name) == kind["id"]}
        from pipeline.world_gen import _blocking_world_defects
        defects.extend(_blocking_world_defects(validate(typed_world, applied, profile), False,
                      protected_fields=seed_protected_fields(wp or {}), entity_types=world.entity_types))
    issues += [f"{item.get('entity')}.{item.get('field')} [{item.get('type')}]: {item.get('detail')}"
               for item in defects]
    if wp is not None:
        issues.extend(seed_world_issues(wp, world, require_complete=complete))
    return world, applied, initial, issues


def _context(wp, blueprint):
    calendar = WorldState(n_sessions=blueprint["temporal_model"]["n_sessions"], world_blueprint=blueprint)
    ownership = {}
    for kind in blueprint["entity_types"]:
        routes = {f["name"]: [] for f in kind["fields"]}
        for relation in blueprint["relation_types"]:
            side = relation_owner_side(blueprint, relation)
            if relation[side + "_type"] == kind["id"]:
                routes[relation["field"]].append({"via": "relation", "type": relation["id"]})
        for event in blueprint["event_types"]:
            for effect in event["effect_fields"]:
                if event["roles"][effect["role"]] == kind["id"]:
                    routes[effect["field"]].append({"via": "event", "type": event["id"], "role": effect["role"]})
        ownership[kind["id"]] = {"intrinsic_fields": [name for name, route in routes.items() if not route],
                                "structure_fields": {name: route for name, route in routes.items() if route}}
    return {"business": seed_business_context(wp), "blueprint": blueprint, "field_write_routes": ownership,
            "calendar": [{"session": session, "date": calendar.date_of_session(session)}
                         for session in calendar.sessions()],
            "world_brief": {key: deepcopy(wp[key]) for key in ("scenario", "shared_world_spec") if key in wp}}


def _index(world, blueprint):
    entity_counts = Counter(world.entity_types.values())
    relation_counts = Counter(item["type"] for item in world.relations)
    event_counts = Counter(item["type"] for item in world.events)
    return {"entities": [{"name": name, "type": kind} for name, kind in world.entity_types.items()],
            "relations": [{key: item[key] for key in ("id", "type", "from", "to", "session") if key in item}
                          for item in world.relations],
            "events": [{key: item[key] for key in ("id", "type", "session", "participants")}
                       for item in world.events],
            "remaining_minima": {
                "entities": {item["id"]: max(0, item["count"] - entity_counts[item["id"]])
                             for item in blueprint["entity_types"]},
                "relations": {item["id"]: max(0, item.get("min_count", 0) - relation_counts[item["id"]])
                              for item in blueprint["relation_types"]},
                "events": {item["id"]: max(0, item.get("min_count", 0) - event_counts[item["id"]])
                           for item in blueprint["event_types"]}},
            "coverage_note": "这里只统计数量；业务机制与因果见证由事实验收和原语义闸继续核对。"}


def _names(action, key):
    values = action.get(key, [])
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        raise WorldBlueprintError(f"{key} must be a list of nonempty strings")
    return set(values)


def _read(action, world, state):
    names, event_ids = _names(action, "read_entities"), _names(action, "read_events")
    events_by_id = {item["id"]: item for item in world.events}
    missing = (names - set(world.entities)) | (event_ids - set(events_by_id))
    if missing:
        raise WorldBlueprintError("unknown read references: " + ", ".join(sorted(missing)))
    for event_id in event_ids:
        names.update(events_by_id[event_id]["participants"].values())
    # Full histories and adjacent structures of requested entities expose prior
    # effects and shared links. Neighbor identities are visible but not declared
    # fully read; authors must explicitly request their fields before use.
    data = world.to_dict()
    events = [item for item in world.events if item["id"] in event_ids
              or names.intersection(item["participants"].values())]
    relations = [item for item in world.relations if item["from"] in names or item["to"] in names]
    available_units = {item["unit_id"]: item for item in state["retired_units"] + state["units"]}
    unit_ids = _names(action, "read_units")
    if unit_ids - set(available_units):
        raise WorldBlueprintError("unknown read_units")
    return {"entities": [{"name": name, "type": world.entity_types[name],
                           "canonical_fields": data["entities"][name]} for name in sorted(names)],
            "relations": deepcopy(relations), "events": deepcopy(events),
            "prior_unit_proposals": [deepcopy(available_units[unit_id]["raw"]) for unit_id in sorted(unit_ids)]}


def _unit_issues(raw, context, world, blueprint):
    if not isinstance(raw, dict) or set(raw) != set(TABLE_KEYS):
        return ["unit must contain exactly entities/relations/events/initial_states arrays"]
    if any(not isinstance(raw[key], list) or any(not isinstance(row, dict) for row in raw[key])
           for key in TABLE_KEYS):
        return ["all unit arrays must contain objects"]
    if not any(raw.values()):
        return ["empty write has no facts; inspect, finish or report_unresolved instead"]
    issues, names = [], []
    owned = {}
    for relation in blueprint["relation_types"]:
        side = relation_owner_side(blueprint, relation)
        tid = relation["from_type"] if side == "from" else relation["to_type"]
        owned.setdefault(tid, set()).add(relation["field"])
    for event in blueprint["event_types"]:
        for effect in event["effect_fields"]:
            owned.setdefault(event["roles"][effect["role"]], set()).add(effect["field"])
    for entity in raw["entities"]:
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip() or not isinstance(entity.get("fields"), dict):
            return ["entity needs nonempty name and fields object"]
        if set(entity) - {"name", "type", "fields", "purpose"}:
            issues.append(f"entity {name} contains undeclared payload keys")
        names.append(name)
        if name in world.entities:
            issues.append(f"cannot recreate accepted/existing entity {name}; revise its owning unit")
        if owned.get(entity.get("type"), set()).intersection(entity["fields"]):
            wrong = sorted(owned[entity.get("type")].intersection(entity["fields"]))
            issues.append(f"entity {name} writes structure-owned fields {wrong}; use field_write_routes and relations/events/initial_states")
        for field, value in entity["fields"].items():
            if not isinstance(value, dict) or value.get("type") not in ("stable", "evolving"):
                issues.append(f"entity {name}.{field} needs stable/evolving specification")
    if len(names) != len(set(names)):
        issues.append("duplicate new entity identities")
    readable = {item["name"] for item in context["entities"]}
    allowed = readable | set(names)
    known_events = {item["id"] for item in world.events}
    read_events = {item["id"] for item in context["events"]}
    new_event_ids = {item.get("id") for item in raw["events"] if isinstance(item.get("id"), str)}
    known_relations = {item["id"] for item in world.relations}
    for key, known in (("events", known_events), ("relations", known_relations)):
        identifiers = [item.get("id") for item in raw[key]]
        if any(not isinstance(value, str) or not value for value in identifiers):
            issues.append(f"{key} require nonempty ids")
        elif len(identifiers) != len(set(identifiers)) or set(identifiers).intersection(known):
            duplicates = sorted({value for value, count in Counter(identifiers).items() if count > 1}
                                | set(identifiers).intersection(known))
            existing_rows = [item for item in getattr(world, key) if item["id"] in duplicates]
            issues.append(f"{key} duplicate accepted or new ids {duplicates}; existing={json.dumps(existing_rows, ensure_ascii=False)}")
    for relation in raw["relations"]:
        if any(not isinstance(relation.get(key), str) or relation[key] not in allowed for key in ("from", "to")):
            issues.append("relation references an unread or unknown entity")
    for event in raw["events"]:
        participants = event.get("participants")
        if not isinstance(participants, dict) or any(not isinstance(name, str) or name not in allowed for name in participants.values()):
            issues.append("event references an unread or unknown participant")
        if event.get("caused_by") and event["caused_by"] not in read_events | new_event_ids:
            issues.append("caused_by references an unread or unknown event")
    # New initial states must belong to entities created in this unit. This
    # prevents a later unit changing the beginning of an accepted history.
    if any(item.get("entity") not in set(names) for item in raw["initial_states"]):
        issues.append("initial_states may only address entities created in this unit")
    return issues


def generate_world(wp, tracer, existing=None, checkpoint_path=None, feedback=None, log=print, resume_state=None):
    """Return (compiler-ready raw table, audited metadata); never publish a world."""
    blueprint = normalize_world_blueprint(wp)
    binding = _binding(wp, existing)
    policy = wp.get("world_generation") or {}
    max_steps = policy.get("max_steps", min(240, max(32, sum(t["count"] for t in blueprint["entity_types"]) * 3)))
    if type(max_steps) is not int or not 1 <= max_steps <= 600:
        raise WorldBlueprintError("world_generation.max_steps must be 1..600")
    state = {"version": VERSION, "binding": binding, "units": [], "retired_units": [],
             "log": [], "status": "building", "steps": 0, "feedback_history": [],
             "observation": None, "plan": "", "feedback_revision_required": False, "issue_responses": [],
             "truncated_work": []}
    recovered = (json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
                 if checkpoint_path is not None and Path(checkpoint_path).exists() else resume_state)
    if recovered is not None:
        state = deepcopy(recovered)
        if (state.get("checkpoint_hash") != _digest({k: v for k, v in state.items() if k != "checkpoint_hash"})
                or state.get("binding") != binding):
            raise WorldBlueprintError("world agent checkpoint input/source binding mismatch")
    state.setdefault("prior_failures", [])
    if feedback and (not state["feedback_history"] or state["feedback_history"][-1]["hash"] != _digest(feedback)):
        state["feedback_history"].append({"hash": _digest(feedback), "feedback": deepcopy(feedback)})
        state["feedback_revision_required"] = bool(state["units"])
        state["status"] = "building"
        state["observation"] = {"review_feedback": deepcopy(feedback), "revision_required": bool(state["units"])}
    frozen = _context(wp, blueprint)

    def save():
        _save(checkpoint_path, state)

    def call(step, payload, system, max_tokens):
        if len(json.dumps(payload, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
            raise WorldBlueprintError("requested context too large; inspect fewer objects or split the business work unit")
        attempt = {"step": step, "input": deepcopy(payload), "status": "running"}
        state["log"].append(attempt)
        save()
        try:
            if _binding(wp, existing) != binding:
                raise WorldBlueprintError("world agent implementation/input changed during execution")
            raw = tracer.chat_json(step, [{"role": "system", "content": system},
                                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                   model=config.STRUCTURE_MODEL, temperature=0.3,
                                   max_tokens=max_tokens, retries=FORMAT_ATTEMPTS, strict_json=True,
                                   response_format={"type": "json_object"})
            if _binding(wp, existing) != binding:
                raise WorldBlueprintError("world agent implementation/input changed while awaiting response")
            attempt["raw_output"] = deepcopy(raw)
            if isinstance(raw, dict) and "__error__" in raw:
                error_metadata = raw.get("__error_metadata__") or {}
                if (step == "world.agent.write" and error_metadata.get("kind") == "output_truncated"
                        and error_metadata.get("response_received") is True):
                    attempt["status"] = "output_truncated_replan_required"
                    save()
                    return raw
                raise RuntimeError("world agent execution error: " + str(raw["__error__"]))
            attempt["status"] = "returned"
            return raw
        except Exception as error:
            attempt.update(status="execution_error", error=f"{type(error).__name__}: {error}")
            state.update(status="execution_error", observation={"execution_error": attempt["error"],
                         "instruction": "The accepted units are retained; retry only through explicit resume."})
            save()
            raise

    try:
        while True:
            world, applied, initial, integrity = _compiled(_combine(state["units"]), blueprint, existing, False, wp)
            if integrity:
                raise WorldBlueprintError("accepted checkpoint has invalid prefix: " + "; ".join(integrity))
            if state["status"] == "completed":
                _, applied, initial, issues = _compiled(_combine(state["units"]), blueprint, existing, True, wp)
                if issues:
                    raise WorldBlueprintError("completed checkpoint failed full compilation: " + "; ".join(issues))
                validate_seed_world(wp, world)
                return applied, deepcopy(state)
            if state["steps"] >= max_steps:
                raise WorldBlueprintError(f"world agent planning step budget exhausted ({max_steps})")
            state["steps"] += 1
            state["status"] = "building"
            payload = {**frozen, "current": _index(world, blueprint), "plan": state["plan"],
                       "accepted_units": [{"unit_id": unit["unit_id"], "intent": unit["intent"],
                                           "entities": [row["name"] for row in unit["raw"]["entities"]]}
                                          for unit in state["units"]],
                       "observation": state["observation"],
                       "prior_failures": state["prior_failures"][-5:],
                       "remaining_steps": max_steps - state["steps"],
                       "review_feedback": state["feedback_history"][-1] if state["feedback_history"] else None,
                       "feedback_revision_required": state["feedback_revision_required"]}
            action = call("world.agent.plan", payload, PLANNER_SYSTEM, 8192)
            try:
                if not isinstance(action, dict):
                    raise WorldBlueprintError("planner must return one action object")
                kind = action.get("action")
                if kind == "inspect":
                    inspection = _read(action, world, state)
                    if len(json.dumps({**frozen, "inspection": inspection}, ensure_ascii=False)) > MAX_CONTEXT_CHARS * 0.8:
                        raise WorldBlueprintError("inspection too large; request fewer objects or units")
                    state["observation"] = {"inspection": inspection}
                elif kind == "revise":
                    index = next((i for i, unit in enumerate(state["units"]) if unit["unit_id"] == action.get("unit_id")), None)
                    if index is None or not isinstance(action.get("reason"), str) or not action["reason"].strip():
                        raise WorldBlueprintError("revise requires accepted unit_id and reason")
                    suffix = state["units"][index:]
                    state["retired_units"].extend(deepcopy(suffix))
                    del state["units"][index:]
                    state["observation"] = {"retired_units": [item["unit_id"] for item in suffix], "reason": action["reason"]}
                elif kind == "write":
                    unit_id, intent = action.get("unit_id"), action.get("intent")
                    if (not isinstance(unit_id, str) or not unit_id or not isinstance(intent, str) or not intent.strip()
                            or unit_id in {unit["unit_id"] for unit in state["units"]}):
                        raise WorldBlueprintError("write requires unique unit_id and business intent")
                    context = _read(action, world, state)
                    work_hash = _digest({key: action.get(key) for key in ("intent", "read_entities", "read_events", "read_units")})
                    if work_hash in state["truncated_work"]:
                        raise WorldBlueprintError("identical truncated work cannot be retried; change the business boundary/intent or read scope")
                    author_input = {**frozen, "task": action, "read_context": context,
                                    "occupied_names": list(world.entities),
                                    "occupied_ids": {"relations": [r["id"] for r in world.relations],
                                                     "events": [e["id"] for e in world.events]},
                                    "feedback": state["observation"]}
                    if len(json.dumps(author_input, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
                        raise WorldBlueprintError("write context exceeds limit; choose narrower reads or split the task")
                    raw = call("world.agent.write", author_input, AUTHOR_SYSTEM, 16384)
                    if isinstance(raw, dict) and "__error__" in raw:
                        state["truncated_work"].append(work_hash)
                        state["observation"] = {"unit_id": unit_id, "output_truncated": raw["__error_metadata__"],
                                                "instruction": "Split/replan this work unit before another write; accepted facts are retained."}
                        save()
                        continue
                    if isinstance(raw, dict) and set(raw) == {"needs_context"}:
                        state["observation"] = {"unit_id": unit_id, "author_needs_context": raw["needs_context"]}
                        save()
                        continue
                    try:
                        issues = _unit_issues(raw, context, world, blueprint)
                    except (TypeError, KeyError) as error:
                        issues = [f"invalid unit shape: {type(error).__name__}: {error}"]
                    if not issues:
                        proposed = state["units"] + [{"raw": raw}]
                        try:
                            _, _, _, issues = _compiled(_combine(proposed), blueprint, existing, False, wp)
                        except (ValueError, TypeError, KeyError) as error:
                            issues = [f"{type(error).__name__}: {error}"]
                    if issues:
                        state["observation"] = {"rejected_unit": unit_id, "issues": issues,
                                                "instruction": "Prior proposal was rejected atomically. Replan this unit; accepted facts are unchanged."}
                        state["prior_failures"].append({"unit_id": unit_id, "intent": intent, "issues": issues})
                        encoded = json.dumps(raw, ensure_ascii=False)
                        state["observation"]["rejected_proposal"] = (deepcopy(raw) if len(encoded) <= 30000 else
                            {"preview": encoded[:30000], "truncated": True,
                             "instruction": "Failed proposal is large; inspect relevant existing facts and plan a smaller business unit."})
                    else:
                        state["units"].append({"unit_id": unit_id, "intent": intent, "action": deepcopy(action),
                                               "context": context, "raw": deepcopy(raw), "raw_hash": _digest(raw)})
                        if isinstance(action.get("plan"), str):
                            state["plan"] = action["plan"]
                        state["observation"] = {"accepted_unit": unit_id, "counts": {key: len(raw[key]) for key in TABLE_KEYS}}
                        log(f"  ✓ Agent 世界单元 {unit_id}: {len(raw['entities'])} 实体 / {len(raw['events'])} 事件")
                elif kind == "finish":
                    if state["feedback_revision_required"]:
                        responses = action.get("issue_responses")
                        if (not isinstance(responses, list) or not responses
                                or any(not isinstance(item, dict)
                                       or not isinstance(item.get("issue_id"), str) or not item["issue_id"].strip()
                                       or item.get("disposition") not in ("addressed", "disputed", "deferred")
                                       or not isinstance(item.get("response"), str) or not item["response"].strip()
                                       for item in responses)):
                            raise WorldBlueprintError("new review feedback requires explicit issue_responses with factual reasons before finish")
                        state["issue_responses"] = deepcopy(responses)
                    world, applied, initial, issues = _compiled(_combine(state["units"]), blueprint, existing, True, wp)
                    if not issues:
                        try:
                            validate_seed_world(wp, world)
                        except ValueError as error:
                            issues = [str(error)]
                    if issues:
                        state["observation"] = {"finish_rejected": issues}
                    else:
                        state.update(status="completed", initial_states=initial, result_hash=_digest(applied), feedback_revision_required=False)
                        save()
                        return applied, deepcopy(state)
                elif kind == "report_unresolved":
                    state.update(status="unresolved", observation={"unresolved": action.get("reason")})
                    save()
                    raise WorldBlueprintError("world agent unresolved: " + str(action.get("reason")))
                elif kind == "request_blueprint_review":
                    if not isinstance(action.get("reason"), str) or not action["reason"].strip():
                        raise WorldBlueprintError("blueprint review request requires concrete reason")
                    evidence = {"reason": action["reason"], "suggestion": action.get("suggestion"),
                                "last_observation": deepcopy(state["observation"]),
                                "prior_failures": deepcopy(state["prior_failures"][-5:])}
                    state.update(status="blueprint_review_requested", upstream_request=evidence)
                    save()
                    raise BlueprintReviewRequested(evidence)
                else:
                    raise WorldBlueprintError("unknown planner action; use inspect/write/revise/finish/report_unresolved")
            except BlueprintReviewRequested:
                raise
            except WorldBlueprintError as error:
                if state["status"] == "unresolved":
                    raise
                state["observation"] = {"action_rejected": str(error)}
            save()
    except Exception as error:
        if state["status"] not in ("execution_error", "unresolved", "blueprint_review_requested"):
            state.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise
