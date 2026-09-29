"""
pipeline.world_gen —— §5 共享世界生成(从 run_factory_v2 拆出,行为不变)。
build_world:并发批次让 LLM 填世界表 → assemble_world 成状态机 → W.3 CRITIC 修复轮(validate 缺陷定向重生成)。
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import random
import re
import threading
import config
from pipeline.world_state import (assemble_world, validate, WorldState, _strip_disambig, name_collisions,
                                   Op, SET, UPDATE, EXPIRE, DELETE, INSUFFICIENT, INVALID,
                                   _to_num, _as_int)
from pipeline.world_blueprint import relation_owner_side
from pipeline.prompts import render
from pipeline.story import (connected_story_entities, disconnected_story_events,
                            narrative_world_issues)
from pipeline.seed_world import (causal_roles_match, seed_entity_prompt, seed_protected_fields, seed_world_issues,
                                 seed_world_prompt, validate_seed_world)

# 两类世界调用需要生成较长的严格 JSON。真实失败样本表明 8192 token 会被
# deep reasoning 全部耗尽而正文为空；16384 的同 prompt 对照返回完整 JSON。
WORLD_JSON_MAX_TOKENS = 16_384
WORLD_DRAFT_VERSION = "original-world-draft/v2"


def _world_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class _RepairTracer:
    """One shared attempt budget for the original authors, including repairs."""
    def __init__(self, tracer, state, max_calls):
        self.tracer, self.state, self.max_calls = tracer, state, max_calls
        self.lock = threading.Lock()
        self.fatal = None
        self.entries_by_output_id = {}

    def chat_json(self, step, messages, **kwargs):
        from pipeline.world_blueprint import WorldBlueprintError
        with self.lock:
            if self.fatal:
                raise WorldBlueprintError("world repair stopped after author execution failure")
            calls = self.state["repair_log"]
            if len(calls) >= self.max_calls:
                raise WorldBlueprintError("world repair author call budget exhausted")
            entry = {"step": step, "status": "running", "raw_output": None,
                     "issue_responses": [], "actual_changes": []}
            calls.append(entry)
        try:
            # The optional outer repair budget counts actual author attempts;
            # legacy author defaults must not hide extra transport/JSON retries.
            kwargs = {**kwargs, "retries": 1, "strict_json": True}
            output = self.tracer.chat_json(step, messages, **kwargs)
            entry["raw_output"] = deepcopy(output)
            if not isinstance(output, dict) or "__error__" in output:
                raise WorldBlueprintError(f"{step} author execution failed: {output}")
            entry["issue_responses"] = deepcopy(output.get("issue_responses", []))
            entry["status"] = "returned"
            with self.lock:
                self.entries_by_output_id[id(output)] = entry
            return output
        except Exception as error:
            with self.lock:
                entry["status"] = "error"
                entry["error"] = f"{type(error).__name__}: {error}"
                self.fatal = entry["error"]
            raise

    def record_changes(self, output, before, after, **metadata):
        with self.lock:
            entry = self.entries_by_output_id[id(output)]
            entry.update(metadata)
            entry["actual_changes"] = _repair_changes(before, after)


def _repair_changes(before, after):
    """Record actual author changes without interpreting their business meaning."""
    changes = []
    old_entities = {e["name"]: e.get("fields", {}) for e in before.get("entities", [])}
    for entity in after.get("entities", []):
        old = old_entities.get(entity["name"], {})
        for field in sorted(set(old) | set(entity.get("fields", {}))):
            if old.get(field) != entity.get("fields", {}).get(field):
                changes.append({"entity": entity["name"], "field": field,
                                "before": deepcopy(old.get(field)),
                                "after": deepcopy(entity.get("fields", {}).get(field))})
    for key in ("relations", "events"):
        if before.get(key) != after.get(key):
            changes.append({"structure": key, "before": deepcopy(before.get(key)),
                            "after": deepcopy(after.get(key))})
    return changes


def build_world(wp, tracer, log=print, existing=None, narrative: bool = False, *,
                draft_out: dict | None = None, repair_input: dict | None = None,
                checkpoint_path=None) -> WorldState:
    """Original builder, optionally retaining a bound draft or repairing it once.

    Repair input: {draft, feedback, targets: {intrinsic: [{entity, fields}],
    structure: bool}, max_calls: 4}. Feedback remains fallible author input.
    ``draft_out`` is updated even on failure; the return type stays WorldState.
    """
    from pipeline.world_blueprint import WorldBlueprintError
    if draft_out is not None and not isinstance(draft_out, dict):
        raise TypeError("draft_out must be a dictionary")
    if (wp.get("world_generation") or {}).get("strategy") == "agentic":
        if narrative:
            raise WorldBlueprintError("Agentic business world generation does not support game Story Ledger")
        return _build_agentic_world(wp, tracer, log, existing, draft_out, repair_input, checkpoint_path)
    state = None
    if draft_out is not None or repair_input is not None:
        state = {"version": WORLD_DRAFT_VERSION, "wp_hash": _world_digest(wp),
                 "existing_base": deepcopy(existing.to_dict()) if existing is not None else None,
                 "narrative": narrative, "merged": None, "raw_merged": None,
                 "joint_plan": None, "initial_states": [],
                 "repair_log": [], "status": "building"}
    try:
        if repair_input is not None:
            if not isinstance(repair_input, dict):
                raise WorldBlueprintError("repair_input must be an object")
            draft = deepcopy(repair_input.get("draft"))
            if not isinstance(draft, dict) or draft.get("version") != WORLD_DRAFT_VERSION:
                raise WorldBlueprintError("unsupported world draft")
            if draft.get("draft_hash") != _world_digest({k: v for k, v in draft.items() if k != "draft_hash"}):
                raise WorldBlueprintError("world draft binding mismatch")
            if draft.get("wp_hash") != _world_digest(wp) or draft.get("narrative") != narrative:
                raise WorldBlueprintError("world draft whitepaper/mode mismatch")
            if draft.get("status") != "completed" or not isinstance(draft.get("merged"), dict):
                raise WorldBlueprintError("world draft has no completed candidate")
            if draft.get("candidate_world_hash") != _world_digest(draft.get("candidate_world")):
                raise WorldBlueprintError("world draft candidate binding mismatch")
            if existing is not None and existing.to_dict() != draft.get("existing_base"):
                raise WorldBlueprintError("world repair existing baseline mismatch")
            if draft.get("mode") == "existing_noop":
                raise WorldBlueprintError("world repair cannot alter a published existing-only baseline")
            targets = repair_input.get("targets")
            if (not isinstance(targets, dict) or set(targets) != {"intrinsic", "structure"}
                    or not isinstance(targets["intrinsic"], list) or type(targets["structure"]) is not bool):
                raise WorldBlueprintError("world repair targets must identify intrinsic fields and structure")
            if not targets["intrinsic"] and not targets["structure"]:
                raise WorldBlueprintError("world repair has no author targets")
            max_calls = repair_input.get("max_calls", 4)
            if type(max_calls) is not int or max_calls < 0:
                raise WorldBlueprintError("world repair max_calls must be a nonnegative integer")
            if not isinstance(repair_input.get("feedback"), (dict, list, str)) or not repair_input["feedback"]:
                raise WorldBlueprintError("world repair requires the original review feedback")
            state.update({"existing_base": deepcopy(draft["existing_base"]),
                          "merged": deepcopy(draft["merged"]),
                          "raw_merged": deepcopy(draft.get("raw_merged")),
                          "joint_plan": deepcopy(draft.get("joint_plan")),
                          "initial_states": deepcopy(draft.get("initial_states", [])),
                          "parent_draft_hash": draft["draft_hash"],
                          "repair_targets": deepcopy(targets),
                          "feedback": deepcopy(repair_input["feedback"]),
                          "max_calls": max_calls, "source_draft": draft})
            existing = WorldState.from_dict(deepcopy(draft["existing_base"])) if draft["existing_base"] is not None else None
            tracer = _RepairTracer(tracer, state, max_calls)
        world = _build_world(wp, tracer, log, existing, narrative, _draft_state=state)
        if state is not None:
            state.update({"status": "completed", "candidate_world": deepcopy(world.to_dict()),
                          "candidate_world_hash": _world_digest(world.to_dict())})
        return world
    except Exception as error:
        if state is not None:
            state.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if state is not None and draft_out is not None:
            state.pop("source_draft", None)
            saved = deepcopy(state)
            saved["draft_hash"] = _world_digest(saved)
            draft_out.clear()
            draft_out.update(saved)


def _build_agentic_world(wp, tracer, log, existing, draft_out, repair_input, checkpoint_path):
    """Use the original compiler, finalization and factory gates after agent authoring."""
    from pipeline.world_agent import generate_world
    from pipeline.world_blueprint import normalize_world_blueprint, WorldBlueprintError
    from pipeline.seed_pack import validate_seed_blueprint
    blueprint = normalize_world_blueprint(wp)
    if wp.get("seed_contract") is not None:
        audit = validate_seed_blueprint(wp, wp["seed_contract"])
        if not audit["passed"]:
            raise WorldBlueprintError("Seed blueprint is invalid: " + "; ".join(audit["issues"]))
    baseline = deepcopy(existing.to_dict()) if existing is not None else None
    state = {"version": WORLD_DRAFT_VERSION, "generation_strategy": "agentic",
             "wp_hash": _world_digest(wp), "existing_base": baseline, "narrative": False,
             "merged": None, "raw_merged": None, "joint_plan": None, "initial_states": [],
             "repair_log": [], "status": "building"}
    feedback, resume = None, None
    try:
        if repair_input is not None:
            if not isinstance(repair_input, dict):
                raise WorldBlueprintError("Agent repair input must be an object")
            draft = repair_input.get("draft")
            if (not isinstance(draft, dict) or draft.get("generation_strategy") != "agentic"
                    or draft.get("version") != WORLD_DRAFT_VERSION or draft.get("status") != "completed"
                    or draft.get("draft_hash") != _world_digest({k: v for k, v in draft.items() if k != "draft_hash"})
                    or draft.get("wp_hash") != _world_digest(wp) or draft.get("existing_base") != baseline
                    or draft.get("candidate_world_hash") != _world_digest(draft.get("candidate_world"))):
                raise WorldBlueprintError("Agent repair draft is stale or changed")
            feedback = repair_input.get("feedback")
            if not isinstance(feedback, dict) or not feedback:
                raise WorldBlueprintError("Agent repair requires the original review feedback")
            # Saved opinions remain intact in the factory artifact. Repeated full
            # world/prompt envelopes do not belong in the planner's next context.
            feedback = {k: deepcopy(v) for k, v in feedback.items()
                        if k not in {"messages", "input_snapshot", "binding", "previous"}}
            resume = deepcopy(draft.get("agent"))
            state["parent_draft_hash"] = draft["draft_hash"]
        table, agent = generate_world(wp, tracer, existing=existing,
            checkpoint_path=checkpoint_path, feedback=feedback, resume_state=resume, log=log)
        state.update(merged=deepcopy(table), raw_merged=deepcopy(table), agent=deepcopy(agent),
                     initial_states=deepcopy(agent.get("initial_states", [])))
        if feedback is not None:
            state["repair_log"] = [{"issue_responses": deepcopy(agent.get("issue_responses", [])),
                "accepted_units": [{"unit_id": row["unit_id"], "intent": row["intent"]}
                                   for row in agent.get("units", [])],
                "retired_units": [row["unit_id"] for row in agent.get("retired_units", [])]}]
        world, issues = assemble_world(table, blueprint=blueprint, existing=existing,
                                      include_shape_diagnostics=False)
        if issues:
            raise WorldBlueprintError("Agent world has unresolved compiler errors: " + "; ".join(issues))
        profile = deepcopy(wp.get("domain_profile") or {})
        fields = {}
        for item in blueprint["entity_types"]:
            for field in item.get("fields", []):
                fields.setdefault(field["name"], deepcopy(field))
        profile["field_schema"] = list(fields.values())
        profile["entity_noun"] = next(t["noun"] for t in blueprint["entity_types"] if t.get("primary"))
        profile["state_machines"] = [{"field": f["name"], "states": f["states"]}
                                     for f in fields.values() if f.get("states")]
        protected = seed_protected_fields(wp)
        profile["seed_protected_fields"] = sorted(protected)
        blocking = []
        for kind in blueprint["entity_types"]:
            typed_world = deepcopy(world)
            typed_world.entities = {name: rows for name, rows in world.entities.items()
                                    if world.entity_types.get(name) == kind["id"]}
            typed_profile = {"field_schema": kind["fields"], "state_machines": [
                {"field": field["name"], "states": field["states"]}
                for field in kind["fields"] if field.get("states")]}
            blocking.extend(_blocking_world_defects(validate(typed_world, table, typed_profile), False,
                             protected_fields=protected, entity_types=world.entity_types))
        if blocking:
            raise WorldBlueprintError("Agent world has unresolved truth defects: "
                                      + json.dumps(blocking, ensure_ascii=False))
        world = _finish_world(wp, world, table, profile, blueprint, False, protected, log)
        state.update(status="completed", candidate_world=deepcopy(world.to_dict()),
                     candidate_world_hash=_world_digest(world.to_dict()))
        return world
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if draft_out is not None:
            saved = deepcopy(state)
            saved["draft_hash"] = _world_digest(saved)
            draft_out.clear()
            draft_out.update(saved)

_BARE_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_FIELD_KIND_SUFFIX = re.compile(
    r"\s*[\(\uff08]\s*(?:text|status|category|numeric|number|person|reference|string)\s*[\)\uff09]\s*$",
    re.IGNORECASE,
)

_NARRATIVE_ADVISORY_DEFECTS = frozenset({"monotonic", "fake_evolving"})


def _blocking_world_defects(defects: list[dict], narrative: bool, *,
                            protected_fields: set[tuple[str, str]] | None = None,
                            entity_types: dict[str, str] | None = None) -> list[dict]:
    """Separate declared truth constraints from optional task-shape diagnostics."""
    protected_fields = protected_fields or set()
    entity_types = entity_types or {}
    return [item for item in defects
            if item.get("type") != "monotonic"
            and not (item.get("type") in _NARRATIVE_ADVISORY_DEFECTS and
                    (narrative or (entity_types.get(item.get("entity")), item.get("field"))
                     in protected_fields))]


def _partition_structure_driven_defects(
        defects: list[dict], entity_types: dict[str, str],
        relation_fields: dict[str, set[str]],
        event_fields: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    """把结构单一真源字段的缺陷与普通字段缺陷分开。

    关系/事件驱动字段只能通过重写 relation/event 修复；若交给普通字段 critic，
    它会制造没有结构见证的平行轨迹。
    """
    structural: list[dict] = []
    intrinsic: list[dict] = []
    for defect in defects:
        tid = entity_types.get(defect.get("entity"))
        owned = relation_fields.get(tid, set()) | event_fields.get(tid, set())
        (structural if defect.get("field") in owned else intrinsic).append(defect)
    return structural, intrinsic


def _canonicalize_generated_field_names(fields: dict, allowed: set[str],
                                         field_schema: list[dict] | None = None) -> dict:
    """将模型偶发输出的“字段名(kind)”归一为蓝图字段名。

    只在剥掉已知 kind 后与 allowed 中某一字段精确相等时生效；
    若模型直接把唯一的 kind 当成字段键，也只在该 kind 在当前类型中
    唯一对应一个蓝图字段时归一；未知或多义字段仍留给 off-schema 门禁删除。
    """
    kind_names: dict[str, list[str]] = {}
    for item in field_schema or []:
        name = item.get("name")
        kind = str(item.get("kind") or "").strip().lower()
        if name in allowed and kind:
            kind_names.setdefault(kind, []).append(name)
    unique_kind_name = {
        kind: names[0] for kind, names in kind_names.items() if len(names) == 1
    }
    out = {}
    for raw_name, spec in fields.items():
        name = raw_name
        if isinstance(raw_name, str) and raw_name not in allowed:
            candidate = _FIELD_KIND_SUFFIX.sub("", raw_name)
            if candidate in allowed:
                name = candidate
            elif raw_name.strip().lower() in unique_kind_name:
                name = unique_kind_name[raw_name.strip().lower()]
        # 模型同时给标准键和带注解键时，优先保留标准键。
        if name not in out or raw_name in allowed:
            out[name] = spec
    return out


def _intrinsic_proposal_issues(entity: dict, blueprint: dict) -> list[str]:
    """Reuse the canonical compiler before counting a typed entity as generated.

    Only this entity's field issues belong to its author. Missing other entity
    counts, relations and events belong to later original stages and are ignored.
    This does not decide business meaning or create any replacement value.
    """
    _, issues = assemble_world({"entities": [entity]}, blueprint=blueprint)
    name = entity["name"]
    prefixes = (f"entity {name}.", f"entity {name}(")
    return [issue for issue in issues if issue.startswith(prefixes)]


def _filter_incremental_structure(structure: dict, existing: WorldState,
                                  blueprint: dict) -> tuple[list[dict], list[dict]]:
    """过滤增量骨架中与旧世界重复或同期冲突的候选。

    保留引用旧实体的真新关系/事件，但不让模型重放 rel-1/evt-1，
    也不让它在旧 canonical timeline 已有操作的同一 session 再写一次。
    """
    old_rel_ids = {item.get("id") for item in existing.relations if item.get("id")}
    old_rel_edges = {(item.get("type"), item.get("from"), item.get("to"),
                      _as_int(item.get("session"), 0)) for item in existing.relations}
    rel_decls = {item.get("id"): item for item in blueprint.get("relation_types", [])}
    relations = []
    for item in (structure.get("relations") or []):
        if not isinstance(item, dict) or item.get("id") in old_rel_ids:
            continue
        edge = (item.get("type"), item.get("from"), item.get("to"),
                _as_int(item.get("session"), 0))
        if edge in old_rel_edges:
            continue
        decl = rel_decls.get(item.get("type")) or {}
        side = relation_owner_side(blueprint, decl) if decl else None
        owner = item.get("from") if side != "to" else item.get("to")
        timeline = existing.timeline(owner, decl.get("field")) if owner and decl.get("field") else None
        if timeline and any(op.session == edge[3] for op in timeline.ops):
            continue
        relations.append(item)

    old_event_ids = {item.get("id") for item in existing.events if item.get("id")}
    old_event_keys = {(item.get("type"), _as_int(item.get("session"), -1),
                       json.dumps(item.get("participants") or {}, ensure_ascii=False, sort_keys=True))
                      for item in existing.events}
    events = []
    for item in (structure.get("events") or []):
        if not isinstance(item, dict) or item.get("id") in old_event_ids:
            continue
        session = _as_int(item.get("session"), -1)
        key = (item.get("type"), session,
               json.dumps(item.get("participants") or {}, ensure_ascii=False, sort_keys=True))
        if key in old_event_keys:
            continue
        clashes = False
        for effect in item.get("effects") or []:
            entity, field = effect.get("entity"), effect.get("field")
            timeline = existing.timeline(entity, field) if entity and field else None
            if timeline and any(op.session == session for op in timeline.ops):
                clashes = True
                break
        if not clashes:
            events.append(item)
    return relations, events


def _check_repair_baseline(structure, existing, blueprint):
    """Reject attempts to rewrite published operations before delta filtering."""
    from pipeline.world_blueprint import WorldBlueprintError
    for key in ("relations", "events"):
        old = {item.get("id"): item for item in getattr(existing, key)}
        for item in structure.get(key, []):
            if not isinstance(item, dict):
                raise WorldBlueprintError("world structure repair contains a non-object instance")
            if item.get("id") in old and item != old[item["id"]]:
                raise WorldBlueprintError("world structure repair changed a published instance")
            if item.get("id") in old:
                continue
            slots = []
            if key == "relations":
                declaration = next((d for d in blueprint.get("relation_types", [])
                                    if d["id"] == item.get("type")), None)
                if declaration:
                    owner = item.get("from") if relation_owner_side(blueprint, declaration) != "to" else item.get("to")
                    slots.append((owner, declaration["field"]))
            else:
                slots.extend((e.get("entity"), e.get("field")) for e in item.get("effects", [])
                             if isinstance(e, dict))
            for entity, field in slots:
                timeline = existing.timeline(entity, field)
                if timeline and any(op.session == item.get("session") for op in timeline.ops):
                    raise WorldBlueprintError("world structure repair rewrote a published operation slot")


def _wire_declared_causality(structure: dict, blueprint: dict) -> int:
    """已有事件满足类型与时差时校正 caused_by，不创建事件或改写 session。"""
    events = [event for event in structure.get("events", []) if isinstance(event, dict)]
    rules = [rule for rule in blueprint.get("causal_rules", []) if isinstance(rule, dict)]
    wired = 0
    for rule in rules:
        trigger_type = rule.get("trigger_event")
        effect_type = rule.get("effect_event")
        delay = _as_int(rule.get("delay_sessions"), -1)
        parents = sorted((event for event in events
                          if event.get("type") == trigger_type and event.get("id")),
                         key=lambda event: (_as_int(event.get("session"), -1), str(event.get("id"))))
        children = sorted((event for event in events
                           if event.get("type") == effect_type and event.get("id")),
                          key=lambda event: (_as_int(event.get("session"), -1), str(event.get("id"))))
        if any(child.get("caused_by") == parent.get("id")
               and child.get("id") != parent.get("id")
               and causal_roles_match(rule, parent, child)
               and _as_int(child.get("session"), -1) - _as_int(parent.get("session"), -1) == delay
               for child in children for parent in parents):
            continue
        pair = next(((parent, child) for child in children
                     for parent in parents
                     if child.get("id") != parent.get("id")
                     and causal_roles_match(rule, parent, child)
                     and _as_int(child.get("session"), -1)
                     - _as_int(parent.get("session"), -1) == delay), None)
        if pair:
            parent, child = pair
            child["caused_by"] = parent["id"]
            wired += 1
    return wired


def _connect_narrative_events(structure: dict, trial: WorldState, blueprint: dict) -> int:
    """把断开事件的一个同类型 participant 换成主角分量内实体；无可用候选则不动。"""
    disconnected = set(disconnected_story_events(trial))
    connected = connected_story_entities(trial)
    if not disconnected or not connected:
        return 0
    declarations = {
        item.get("id"): item for item in blueprint.get("event_types", [])
        if isinstance(item, dict) and item.get("id")
    }
    candidates_by_type: dict[str, list[str]] = {}
    for entity in sorted(connected):
        candidates_by_type.setdefault(trial.entity_types.get(entity), []).append(entity)

    for event in structure.get("events", []):
        if not isinstance(event, dict) or event.get("id") not in disconnected:
            continue
        declaration = declarations.get(event.get("type")) or {}
        roles = declaration.get("roles") or {}
        participants = event.get("participants") or {}
        for role, old_entity in list(participants.items()):
            choices = [name for name in candidates_by_type.get(roles.get(role), [])
                       if name != old_entity]
            if not choices:
                continue
            replacement = choices[0]
            participants[role] = replacement
            for effect in event.get("effects") or []:
                if isinstance(effect, dict) and effect.get("entity") == old_entity:
                    effect["entity"] = replacement
            return 1
    return 0


def _affix_units(ws: WorldState, profile: dict | None, log=print) -> int:
    """★C1② 单位真源化:schema 声明了 unit 的字段,canonical 裸数值统一补上单位(代码定,不靠 LLM 服从)。
    根(014559 实证):世界 1000+ 数值 op 全裸数、渲染 LLM 31% 表述自发加'万元'、36/43 实体跨周单位忽有忽无
    ——量纲是 gold 的一部分却没有真源。补法 = 真源迁移:单位进 canonical,渲染核验子串尺对'加单位'的
    单向失明自动闭合('200万元' 必须逐字在场),零新增规则。幂等(已带单位的值非裸数,二次调用 no-op)。
    Op.prev 同步补(TR gt 的 {from,to} 也是人类口径)。非裸数值(LLM 自带别的写法)不动,只计数提示。"""
    units = {f.get("name"): str(f.get("unit")).strip()
             for f in (profile or {}).get("field_schema", []) if f.get("unit")}
    # ★unit 声明自检(刀1审计):unit 本身是纯数字(LLM 错填如 '0')会使 vs+u 仍匹配 _BARE_NUM → 每轮 ×10 跑飞
    bad_units = {f: u for f, u in units.items() if not u or _BARE_NUM.fullmatch(u)}
    if bad_units:
        log(f"  ⚠单位声明非法(纯数字/空),跳过:{bad_units}")
        units = {f: u for f, u in units.items() if f not in bad_units}
    if not units:
        return 0
    n, odd = 0, 0
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            u = units.get(fname)
            if not u:
                continue
            for o in tl.ops:
                for attr in ("value", "prev"):
                    v = getattr(o, attr)
                    if v is None:
                        continue
                    vs = str(v).strip()
                    if _BARE_NUM.fullmatch(vs):
                        setattr(o, attr, vs + u); n += 1
                    elif not vs.endswith(u) and _to_num(vs) is not None:
                        odd += 1                      # 非裸且后缀≠声明单位(如'1.2亿'):不动,只计数告警(混量纲闸未建,记 backlog)
    if n or odd:
        log(f"  ★单位真源化:{n} 个裸数值补上 schema 单位(单一真源){f';⚠{odd} 个异型写法未动(混量纲风险,人工抽查)' if odd else ''}")
    return n

# ── ★结构后处理(B:L7 没料的根治)──────────────────────────────────────────────
# LLM 吐的数值轨迹是【无趋势噪声 + 样样每周变】(实测:血压=[130,145,120,135…]、变动数全顶满)。
# L7-S1(趋势)要的"单调走向"世界里根本没有 → 产 0 单。趋势这种 type-critical 结构【不能靠 prompt 让 LLM 造】
# (它给不出带精确 margin+末段反转的可证伪趋势,踩坑账#6),必须【确定性代码】构造。
# 放 world_gen(而非 L7.prepare):趋势是【世界属性】、在世界构造期一次性定,早于任何线 enumerate/渲染
# → 单一真源不破、不可能矛盾;L7 退回"骑世界";整个 benchmark 也获得真实结构(不止 L7)。
_TREND_MIN_PTS = 5        # 需 ≥5 个点:三形状都要"首末净方向 + 中段≥1 局部反向"成形(点太少摆不开)


def _trend_values(n: int, lo: float, hi: float, up: bool, int_like: bool, decimals: int, shape: str = "end_reversal"):
    """造 n 个值,整体【首末净方向】= up(升)/down(降),但中段形状可变(174925 D:11 题全'单调+末周反跳'
    → 可被"永远看首末"识破)。三种形状,全部满足协议(趋势=首末净方向)且【看首末对、看局部错】:
      end_reversal : 前段单调 + 末步反向(看最近一段会判反)
      mid_dip      : 整体上行但中段一个深谷 / 整体下行但中段一个高峰(看中段会判反)
      late_surge   : 大部分平缓、末段才显著净移(看前段会判"没趋势/相反")
    净方向恒由首末锚定:首=lo·末=hi(up)/首=hi·末=lo(down),保证首末净方向唯一可裁。
    返回字符串列表;四舍五入后破坏【首末净方向严格成立 且 至少一处局部反向】→ None(跳过,不注坏趋势)。"""
    if n < 3 or hi <= lo:
        return None
    span = hi - lo
    first, last = (lo, hi) if up else (hi, lo)                  # 首末锚死净方向
    mid = n - 2                                                 # 中间点数
    if shape == "mid_dip":                                      # 中段一个反向极值(谷/峰)
        ext = hi if not up else lo                              # up→中段探到 lo(谷);down→探到 hi(峰)
        k = mid // 2 + 1
        raw = [first] + [ext if i == k else first + (last - first) * i / (n - 1) for i in range(1, n - 1)] + [last]
    elif shape == "late_surge":                                 # 前段微动、末段净移(看前段判不出/判反)
        plateau = first + (last - first) * 0.12
        raw = [first] + [plateau + (first - plateau) * 0.3 * (1 if i % 2 else -1) for i in range(1, n - 1)] + [last]
    else:                                                       # end_reversal(默认)
        step = span / (n - 2)
        raw = [(lo + i * step) if up else (hi - i * step) for i in range(n - 1)]
        raw.append(raw[-1] - step if up else raw[-1] + step)
    fmt = (lambda x: str(int(round(x)))) if int_like else (lambda x: f"{x:.{decimals}f}")
    out = [fmt(x) for x in raw]
    nums = [float(x) for x in out]
    from pipeline.lines.L7_consolidation import NET_FRAC                        # ★单一真源(审计:勿抄字面量 0.30,防漂移)
    net = (nums[-1] - nums[0]) if up else (nums[0] - nums[-1])
    net_ok = net >= NET_FRAC * (max(nums) - min(nums)) > 0                      # 首末净方向清晰(与 L7 _trend_label 同口径同常量)
    has_local_rev = any((nums[i] < nums[i - 1]) if up else (nums[i] > nums[i - 1]) for i in range(1, n))  # ≥1 处局部反向(挫"看局部"捷径)
    distinct = len({round(x, 6) for x in nums}) >= 3                           # 非恒定(≥3 个不同值;late_surge 平台期本就有重复,不强求 n-1)
    return out if (net_ok and has_local_rev and distinct) else None


def imprint_structure(ws: WorldState, log=print, profile: dict | None = None, seed: int = 20260608) -> int:
    """给一小撮【纯数值、≥5点、未注过】的字段注入真趋势(三形状轮转,首末净方向清晰、值域/格式不变)。
    幂等(已注的实体/字段跳过,augment 只注新);只动数值字段 → 与 L4(category)/L5(text)/L2(person)基质不相交,不撞。
    停统操作原样保留，因此同一历史序列仍可支持 FORGET/L6。返回注入字段数。"""
    rng = random.Random(seed)
    done = set(getattr(ws, "_trended_fields", []) or [])
    ws._trended_fields = list(done)
    # ★value_shape:声明了【单调形状】的字段(累计/合计等)绝不注人造趋势——它们本就单向单调(world-gen+validate 保证),
    #   安 end_reversal/mid_dip 会直接违反单调语义(183626:imprint 给"累计工时"安"下降"→逻辑不可能 gold)。L7 也不问它们的"趋势"。
    shaped = {f.get("name") for f in (profile or {}).get("field_schema", []) if f.get("monotonic") in ("up", "down")}
    blueprint = getattr(ws, "world_blueprint", None) or {}
    typed = bool(blueprint and not blueprint.get("legacy_adapter"))
    numeric_fields: set[tuple[str, str]] = set()
    structure_fields: set[tuple[str, str]] = set()
    protected_fields = {tuple(pair) for pair in (profile or {}).get("seed_protected_fields", [])}
    if typed:
        for entity_type in blueprint.get("entity_types") or []:
            tid = entity_type.get("id")
            for field in entity_type.get("fields") or []:
                if field.get("kind") == "numeric":
                    numeric_fields.add((tid, field.get("name")))
        for relation in blueprint.get("relation_types") or []:
            side = relation_owner_side(blueprint, relation)
            owner = relation.get("from_type") if side == "from" else relation.get("to_type")
            structure_fields.add((owner, relation.get("field")))
        for event in blueprint.get("event_types") or []:
            roles = event.get("roles") or {}
            for effect in event.get("effect_fields") or []:
                structure_fields.add((roles.get(effect.get("role")), effect.get("field")))
    cands = []
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            if (ent, fname) in done or fname in shaped:
                continue
            if typed:
                key = (ws.entity_types.get(ent), fname)
                if key not in numeric_fields or key in structure_fields or key in protected_fields:
                    continue
            ops = tl._sorted()
            vals = [o.value for o in ops if o.op in (SET, UPDATE) and o.value]
            nums = [_to_num(v) for v in vals]
            if len(vals) < _TREND_MIN_PTS or any(x is None for x in nums):
                continue
            cands.append((ent, fname, tl, vals, nums))
    rng.shuffle(cands)
    cap = int((profile or {}).get("l7_max_trends", 24))             # 注入上限(够喂 L7 配额、又不淹没全世界震荡多样性)
    n_done = 0
    for i, (ent, fname, tl, vals, nums) in enumerate(cands):
        if n_done >= cap:
            break
        # ★后缀保真(C1② 配套):'450万元'/'2.5%' 重写后必须回贴后缀——旧实现 fmt 输出裸数,
        #   会把单位/百分号整条剥掉(二次剥单位源)。各值后缀不一致 → 字段口径已乱,跳过不注。
        sufs = {re.sub(r"^-?\d+(?:\.\d+)?", "", str(v)).strip() for v in vals}
        if len(sufs) != 1:
            continue
        suf = sufs.pop()
        lo, hi = min(nums), max(nums)
        int_like = all(float(x) == int(x) for x in nums)
        decimals = max((len(m.group(1)) for v in vals if (m := re.search(r"\.(\d+)", str(v)))), default=1)
        pts = [(o.session, o.date) for o in tl._sorted() if o.op in (SET, UPDATE)]
        shape = ("end_reversal", "mid_dip", "late_surge")[n_done % 3]   # ★形状轮转(174925:11题全单调+末反跳可被识破)
        new_vals = _trend_values(len(pts), lo, hi, up=(n_done % 2 == 0), int_like=int_like, decimals=decimals, shape=shape)
        if new_vals is None:                                        # 值域太窄,造不出严格趋势 → 跳过
            continue
        new_ops, prev = [], None
        for (s, d), v in zip(pts, new_vals):
            nv = v + suf
            new_ops.append(Op(s, d, SET if prev is None else UPDATE, nv, prev)); prev = nv
        # 只替换有效值轨迹；EXPIRE/DELETE 是独立的可观测事实，必须原样保留给 FORGET/L6。
        tl.ops = new_ops + [o for o in tl._sorted() if o.op not in (SET, UPDATE)]
        ws._trended_fields.append((ent, fname)); n_done += 1
    if n_done:
        log(f"  ★结构后处理:{n_done} 个数值字段注入真趋势(三形状轮转 end_reversal/mid_dip/late_surge,首末净方向清晰、值域/格式不变)→ 喂 L7-S1")
    return n_done


def _field_desc(f: dict) -> str:
    """字段描述串(给 world prompt):名(kind + value_shape 约束)。约束随白皮书声明走,代码只转述、不猜。"""
    ann = [f.get("kind") or "?"]
    if f.get("unit"):
        ann.append(f"单位{f['unit']}")
    if f.get("monotonic") == "up":
        ann.append("累计只增不减")
    elif f.get("monotonic") == "down":
        ann.append("只减不增")
    rng = f.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        ann.append(f"值域{rng[0]}-{rng[1]}")
    return f"{f.get('name')}({'·'.join(str(a) for a in ann)})"


def _natural_identity_mode(wp: dict, blueprint: dict) -> bool:
    """Existing typed/seed/quality modes allow distinct full object identities."""
    quality = wp.get("quality_contract") or {}
    return (not blueprint.get("legacy_adapter", False) or bool(wp.get("seed_contract"))
            or any(quality.get(key) for key in ("scoring_policy", "world_semantic_review",
                                                "corpus_review", "public_semantic_review")))


def _world_system(profile: dict, type_id: str, time_unit: str, open_schema: bool = False,
                  narrative: bool = False, typed: bool = False, seeded: bool = False,
                  natural_identity: bool = False) -> str:
    noun = profile.get("entity_noun", "实体")
    fields = profile.get("field_schema", [])
    fdesc = "、".join(_field_desc(f) for f in fields) or ("若干随时间演化字段" if open_schema else "无内在字段")
    stopped = profile.get("stopped_phrase", "停止/失效")
    natural_identity = natural_identity or typed or seeded
    world_scope = f"只为蓝图实体类型【{type_id} / {noun}】设计一批随时间({time_unit})演化的真值表。"
    identity_policy = ('★所有专名(实体名 + 人名类字段值)**表面互不近似**:禁止'
                       '"张三/张三(数据)/张三_数据"这类共享主干的近重名'
                       '(下游机械校验表面塌缩,近重名整条作废)。')
    if natural_identity:
        world_scope = (f"只为蓝图实体类型【{type_id} / {noun}】填写给定观察窗口({time_unit})内的事实。"
                       "按对象与字段的业务含义判断稳定或变化；时间推进本身不要求改变取值。"
                       "区分对象的固定内容、后续修订和业务状态，不把不同对象或版本混成一个对象的逐期变化。")
        identity_policy = ("实体的完整名称必须唯一并能区分实际对象；合法共享主体、名称主干、年份或版本系列均允许。"
                           "保留有业务意义的区分信息，不用不同名称伪造同一对象，也不为避免相似而虚构不同主体。"
                           "使用完整名称和已声明关系表达身份；实际指代不清应说明，不能只因主干相同判错。")
    if narrative or seeded:
        numeric_policy = (
            ("- numeric 字段:按领域规律与事件节奏自然变化或保持稳定；不得为题型刻意制造峰谷。"
             if natural_identity else
             "- numeric 字段:按领域规律与事件节奏自然变化；不得为题型刻意制造峰谷。")
            + "若清单声明累计只增/只减/值域，必须遵守；数值写纯阿拉伯数字、不加千分位逗号，单位按字段清单。")
        coverage_policy = (
            "- 本阶段只填写清单中的内在字段，可以自然演化或保持稳定，不强制停用/null；"
            "关系和事件驱动字段留给后续阶段，本阶段不得填写；无内在字段时输出 fields={}。"
            if seeded else
            "- 非结构驱动字段可按剧情需要演化或保持稳定，不强制停用/null；"
            "domain event 驱动字段仍只给可选初态。")
    else:
        numeric_policy = (
            "- numeric 字段:按领域规律自然变化或保持稳定，不为题型刻意制造中间峰谷；"
            "若字段标了累计只增则逐期不减、标了只减则逐期不增、标了值域则全程不出界。"
            "所有数值写纯阿拉伯数字、不加千分位逗号，单位按字段清单。")
        coverage_policy = (
            "- 非结构驱动字段按领域自然生命周期演化或保持稳定，不为题型强制停用/null；"
            "若全部字段由 domain event 驱动，可直接输出空 fields。"
            if typed or natural_identity else
            f"- 若存在非结构驱动字段，至少 1 个字段末尾 null(={stopped})；"
            "若全部字段由 domain event 驱动，可直接输出空 fields。")
    return render("world.system", noun=noun, type_id=type_id, time_unit=time_unit,
                  fdesc=fdesc, stopped=stopped, numeric_policy=numeric_policy,
                  coverage_policy=coverage_policy, world_scope=world_scope,
                  identity_policy=identity_policy)


def _finish_world(wp, ws, merged, profile, blueprint, narrative, protected_fields, log):
    """Original deterministic finalization, also used for zero-call draft replay."""
    typed_contract = not blueprint.get("legacy_adapter", False)
    if typed_contract:
        active_l7 = any(str(item.get("line") or "").strip().lower().startswith("l7")
                        and float(item.get("weight") or 0) > 0
                        for item in wp.get("active_lines", []) if isinstance(item, dict))
        planted = imprint_structure(ws, log, profile) if active_l7 else 0
        log("  ✓ typed world 冻结:跳过 legacy 单位补写，关系/事件字段保持同一真源"
            + (f"；L7 对 {planted} 个内在数值字段完成确定性基质整形" if active_l7 else ""))
    else:
        _affix_units(ws, profile, log)
        imprint_structure(ws, log, profile)
    validate_seed_world(wp, ws)
    final_checks = validate(ws, merged, profile)
    blocking = _blocking_world_defects(final_checks, narrative,
                                      protected_fields=protected_fields, entity_types=ws.entity_types)
    ws.generation_diagnostics = [{**item, "severity": "diagnostic",
        "scope": "generation_advisory_not_world_error_or_measured_question_difficulty"}
        for item in final_checks if item not in blocking]
    if _natural_identity_mode(wp, blueprint):
        ws.generation_diagnostics.extend({
            "type": "similar_name_stem", "entities": names, "severity": "diagnostic",
            "scope": "surface_similarity_not_identity_error_or_semantic_acceptance",
            "detail": "完整名称不同但启发式主干相同；保留原身份，依据实际对象、关系与公开表达判断是否清楚。"
        } for names in sorted(name_collisions(ws)))
    return ws


def _build_world(wp, tracer, log=print, existing=None, narrative: bool = False,
                 *, _draft_state=None) -> WorldState:
    """按白皮书蓝图生成 typed world，并将关系/事件编译进既有 Timeline 地基。

    历史白皮书会先适配成单类型蓝图；显式蓝图不合法、类型数量不足或声明结构没有
    合法实例时 fail-loud，不允许静默退化回扁平世界。``existing`` 只增长缺口实体。
    ``narrative=True`` 时等 canonical world 冻结后，再独立调用至多 4 次
    ``world.story`` 生成轻量 Story Ledger；不新增 Agent 或 Stage。
    """
    from pipeline.world_blueprint import normalize_world_blueprint, WorldBlueprintError

    blueprint = normalize_world_blueprint(wp)
    repairing = _draft_state is not None and "source_draft" in _draft_state
    if _draft_state is not None:
        _draft_state["blueprint_hash"] = _world_digest(blueprint)
        if repairing and _draft_state["source_draft"].get("blueprint_hash") != _world_digest(blueprint):
            raise WorldBlueprintError("world draft blueprint binding mismatch")
    if wp.get("seed_contract") is not None:
        from pipeline.seed_pack import validate_seed_blueprint
        seed_blueprint_report = validate_seed_blueprint(wp, wp["seed_contract"])
        if not seed_blueprint_report["passed"]:
            raise WorldBlueprintError("种子蓝图约束未满足:\n- "
                                      + "\n- ".join(seed_blueprint_report["issues"]))
    seed_hint = seed_world_prompt(wp)
    protected_fields = seed_protected_fields(wp)
    typed_contract = not blueprint.get("legacy_adapter", False)
    if narrative and not typed_contract:
        raise WorldBlueprintError("game Story Ledger 只支持显式 typed world_blueprint")
    if narrative and existing is not None:
        raise WorldBlueprintError(
            "game Story Ledger 不允许增量缝合旧/新 canon；请全量重建 world 并重渲下游产物")
    temporal = blueprint["temporal_model"]
    n_sessions = int(temporal["n_sessions"])
    time_unit = temporal["unit"]
    type_specs = blueprint["entity_types"]
    primary = next(t for t in type_specs if t.get("primary"))

    # 下游旧产线继续消费 domain_profile，但字段集合只能由 blueprint 投影，不能成为第二真源。
    profile = dict(wp.get("domain_profile") or {})
    projected_fields = {}
    for t in type_specs:
        for f in t.get("fields") or []:
            projected_fields.setdefault(f.get("name"), dict(f))
    profile["entity_noun"] = primary["noun"]
    profile["field_schema"] = list(projected_fields.values())
    if seed_hint:
        profile["seed_protected_fields"] = sorted(protected_fields)
    derived_sms = [{"field": f["name"], "states": f["states"]}
                   for t in type_specs for f in t.get("fields", [])
                   if f.get("name") and isinstance(f.get("states"), list) and f.get("states")]
    if typed_contract:
        profile["state_machines"] = derived_sms
    elif derived_sms:
        old_sms = list(profile.get("state_machines") or [])
        seen_sms = {m.get("field") for m in old_sms}
        profile["state_machines"] = old_sms + [m for m in derived_sms if m["field"] not in seen_sms]

    spec = wp.get("shared_world_spec", {})
    noun = primary["noun"]
    # ★W.3:让 build_world 真正消费白皮书的 change_density / traps(此前全程无视)
    cd = (spec.get("timeline", {}) or {}).get("change_density", "")
    # Legacy keeps its surface-name heuristic. Modern authoring keeps distinct
    # full identities and records stem similarity as a fallible diagnostic.
    traps = [t.get("trap") for t in (wp.get("traps") or []) if t.get("trap")][:3]
    base_extra = ""
    if not narrative:
        base_extra = (f"★变更密度:evolving 字段尽量按「{cd}」铺满全程。" if cd else "")
        base_extra += (f"★陷阱布局:本场景需自然埋入这些坑——{traps}(如可矛盾的多源字段、易混字段)。" if traps else "")
    merged = (deepcopy(_draft_state["merged"]) if repairing else
              {"entities": [], "relations": [], "events": [], "cascades": [], "absent_fields": []})
    if _draft_state is not None:
        _draft_state["merged"] = merged
    base_ents = existing.entities if existing is not None else {}    # ★增量:在既有世界上只长新实体
    seen = set(base_ents)                             # 新实体名避开既有
    seen_base = {_strip_disambig(e) for e in base_ents}
    natural_identity = _natural_identity_mode(wp, blueprint)
    batch = 8

    existing_types = getattr(existing, "entity_types", {}) if existing is not None else {}
    if typed_contract and base_ents and not existing_types:
        raise WorldBlueprintError("typed blueprint 不能在缺少 entity_types 的旧世界上增量生成；请全量重建")
    if existing is not None and blueprint.get("legacy_adapter") and not existing_types:
        existing.entity_types.update({name: "legacy_entity" for name in base_ents})
        existing_types = existing.entity_types
    missing_total = sum(max(0, t["count"] - sum(1 for x in existing_types.values() if x == t["id"]))
                        for t in type_specs)
    if existing is not None and missing_total == 0 and not repairing:
        # 闭环已有实体数已达目标时保持真正 no-op：不调用 LLM、不重建旧 timeline，只补元数据。
        existing.n_sessions = max(existing.n_sessions or 0, n_sessions)
        existing.world_blueprint = blueprint
        coll = name_collisions(existing)
        log(f"  ✓ 基础世界:{len(existing.entities)} 实体 / {existing.n_sessions} {time_unit} / 增量无需新增"
            + (f" / ⚠表面塌缩近重名 {coll}" if coll else ""))
        if not typed_contract:
            _affix_units(existing, profile, log)
            imprint_structure(existing, log, profile)
        validate_seed_world(wp, existing)
        if _draft_state is not None:
            _draft_state["mode"] = "existing_noop"
            _draft_state["raw_merged"] = deepcopy(merged)
        return existing
    relation_fields = {}
    for rel in blueprint.get("relation_types", []):
        side = relation_owner_side(blueprint, rel)
        owner = rel["from_type"] if side == "from" else rel["to_type"]
        relation_fields.setdefault(owner, set()).add(rel["field"])
    event_fields = {}
    for event in blueprint.get("event_types", []):
        roles = event.get("roles") or {}
        for effect in event.get("effect_fields", []):
            tid = roles.get(effect.get("role"))
            event_fields.setdefault(tid, set()).add(effect.get("field"))

    # A single shared author proposal precedes independent field decisions.
    # It is retained in the original draft, never used as canonical evidence.
    joint_plan = None
    joint_text = ""
    joint_enabled = bool(seed_hint) and not narrative
    initial_states = deepcopy((_draft_state or {}).get("initial_states", []))
    if joint_enabled:
        from pipeline import world_joint_plan
        plan_calendar = WorldState(n_sessions=n_sessions, world_blueprint=blueprint)
        joint_context = world_joint_plan.make_context(
            wp, blueprint, [{"session": s, "date": plan_calendar.date_of_session(s)}
                            for s in range(n_sessions)], existing)
        if repairing:
            joint_plan = deepcopy(_draft_state["joint_plan"])
        else:
            joint_plan = world_joint_plan.author_plan(
                wp, joint_context, tracer, model=config.STRUCTURE_MODEL,
                max_tokens=WORLD_JSON_MAX_TOKENS)
            if _draft_state is not None:
                _draft_state["joint_plan"] = deepcopy(joint_plan)
        errors = world_joint_plan.validate_plan(joint_plan, wp, joint_context)
        if errors:
            raise WorldBlueprintError("joint world author plan failed: " + "; ".join(errors)
                                      + "; " + str((joint_plan or {}).get("error", "")))
        joint_text = world_joint_plan.shared_prompt(joint_plan)

    def check_joint_plan():
        if joint_enabled:
            errors = world_joint_plan.validate_plan(joint_plan, wp, joint_context)
            if errors:
                raise WorldBlueprintError("joint world author plan drift: " + "; ".join(errors))

    # Entity authors need the shared types and actual earlier objects even when
    # no seed/planning call is selected. This is read-only author context, not
    # permission to write another type's fields or rename existing identities.
    author_declarations = ({key: deepcopy(blueprint.get(key, []))
                            for key in ("entity_types", "relation_types")}
                           if natural_identity else None)
    published_objects = []
    if natural_identity and existing is not None:
        published_objects = [{"name": name, "type": existing_types.get(name),
                              "timelines": deepcopy(fields)}
                             for name, fields in existing.to_dict()["entities"].items()]

    repair_fields = {}
    if repairing:
        # Recompile the saved author proposal; never infer an editable proposal
        # backwards from the final Timeline. Deterministic original finalizers
        # must reproduce exactly the candidate that the caller reviewed.
        prior = _draft_state["source_draft"]
        check, issues = assemble_world(merged, blueprint=blueprint, existing=existing)
        if any(issue.startswith(("entity ", "relation", "event", "causal rule")) for issue in issues):
            raise WorldBlueprintError("world draft cannot recompile under its original blueprint")
        check.n_sessions = max(check.n_sessions, n_sessions,
                               existing.n_sessions if existing is not None else 0)
        if narrative:
            from pipeline.story import compile_story_ledger
            check.narrative = compile_story_ledger(check, prior["candidate_world"].get("narrative"))
        _finish_world(wp, check, merged, profile, blueprint, narrative, protected_fields, lambda *_: None)
        if check.to_dict() != prior["candidate_world"]:
            raise WorldBlueprintError("world draft does not reproduce the reviewed candidate")
        entities_by_name = {e["name"]: e for e in merged["entities"]}
        for target in _draft_state["repair_targets"]["intrinsic"]:
            if (not isinstance(target, dict) or set(target) != {"entity", "fields"}
                    or not isinstance(target["entity"], str) or not isinstance(target["fields"], list)
                    or not target["fields"] or any(not isinstance(f, str) for f in target["fields"])
                    or len(set(target["fields"])) != len(target["fields"])):
                raise WorldBlueprintError("invalid intrinsic world repair target")
            name = target["entity"]
            if name in base_ents or name not in entities_by_name or name in repair_fields:
                raise WorldBlueprintError("intrinsic repair cannot change existing/unknown/duplicate entities")
            entity = entities_by_name[name]
            owned = relation_fields.get(entity["type"], set()) | event_fields.get(entity["type"], set())
            if set(target["fields"]) & owned or not set(target["fields"]).issubset(entity.get("fields", {})):
                raise WorldBlueprintError("intrinsic repair may only target existing intrinsic proposal fields")
            repair_fields[name] = set(target["fields"])
        for name, fields in repair_fields.items():
            entity = entities_by_name[name]
            before = deepcopy(merged)
            check_joint_plan()
            output = tracer.chat_json("world.repair", [
                {"role": "system", "content": (
                    "你是原世界的内在字段作者，正在根据可质疑的业务审阅意见定向返修。"
                    "原任务和冻结蓝图是依据，审阅意见不是新增规则。只可返回授权实体的授权字段；"
                    "不改实体身份、其他字段、关系或事件，不降低种子要求，不为难度制造变化。"
                    "可以基于原任务与既有事实反驳误审或保留未决；解释不是新增世界事实。"
                    "未修改的字段可省略。严格JSON：{\"fields\":{},\"issue_responses\":"
                    "[{\"issue_id\":\"原意见id\",\"disposition\":\"addressed|disputed|deferred\","
                    "\"response\":\"基于原任务/事实的解释\"}]}")},
                {"role": "user", "content": json.dumps({
                    "entity": name, "allowed_fields": sorted(fields),
                    "world_blueprint": blueprint, "original_proposal": merged,
                    "reviewed_world": prior["candidate_world"],
                    "fallible_review_feedback": _draft_state["feedback"]}, ensure_ascii=False)
                 + seed_hint + joint_text}],
                temperature=0.2, max_tokens=WORLD_JSON_MAX_TOKENS,
                model=config.STRUCTURE_MODEL, retries=1, strict_json=True)
            check_joint_plan()
            new_fields = output.get("fields")
            if not isinstance(new_fields, dict) or not set(new_fields).issubset(fields):
                raise WorldBlueprintError("semantic intrinsic repair changed fields outside its targets")
            if any(k in output for k in ("entities", "relations", "events", "world_blueprint", "seed_contract")):
                raise WorldBlueprintError("semantic intrinsic repair attempted to change structure/identity")
            entity["fields"].update(deepcopy(new_fields))
            tracer.record_changes(output, before, merged, entity=name,
                                  allowed_fields=sorted(fields))

    # 每个 entity type 独立生成，只给本类型字段；类型不是 prompt 装饰，而是字段白名单的索引。
    for t in type_specs:
        if repairing:
            continue  # The reviewed entities remain the candidate; no hidden regeneration.
        tid, type_noun = t["id"], t["noun"]
        base_count = (sum(1 for x in existing_types.values() if x == tid)
                      if existing_types else (len(base_ents) if blueprint.get("legacy_adapter") else 0))
        want_total = t["count"]
        produced: list[dict] = []
        intrinsic = [f for f in t.get("fields", []) if f.get("name") not in relation_fields.get(tid, set())]
        # Seeded batches fill only intrinsic fields. Advertising event fields
        # in the system schema while forbidding them below made the model plan
        # the entire business workflow in an entity-only response.
        generation_fields = ([f for f in intrinsic if f.get("name") not in event_fields.get(tid, set())]
                             if seed_hint else intrinsic)
        type_profile = {**profile, "entity_noun": type_noun, "field_schema": generation_fields}
        type_field_names = {f.get("name") for f in (generation_fields if seed_hint else t.get("fields", []))}
        sm_decl = [f"「{m['field']}」只能取 {m['states']} 且按此序单向推进(可跳级、不可回头)"
                   for m in (profile.get("state_machines") or [])
                   if m.get("field") in type_field_names and m.get("states")]
        type_extra = (seed_entity_prompt(wp, tid) if seed_hint else base_extra) + (f"★状态机字段(C1③ 源头约束,validate 还会机械校验):{';'.join(sm_decl)}。"
                                   if sm_decl else "")
        owned = sorted(event_fields.get(tid, set()) - relation_fields.get(tid, set()))
        if owned:
            type_extra += (f"★这些字段由 domain event 驱动:{owned}。entity batch 必须省略它们；"
                           "代码也会删除误写值；只能由后续 event effect 单一真源写入。")
            if set(owned) == {f.get("name") for f in intrinsic if f.get("name")}:
                type_extra += ("★本类型全部字段都由 event 驱动：请直接输出 fields={}；"
                               "不要为了满足末尾 null 要求擅自生成任何字段轨迹。")
        open_schema = bool(blueprint.get("legacy_adapter") and not intrinsic)
        sysp = _world_system(type_profile, tid, time_unit, open_schema=open_schema,
                             narrative=narrative, typed=typed_contract, seeded=bool(seed_hint),
                             natural_identity=natural_identity)
        trajectory_request = (
            "非结构驱动字段按对象与字段含义保持稳定或自然变化，不为题型造峰谷或强塞 null"
            if natural_identity else
            "非结构驱动字段只按领域的自然节奏变化，不为题型造峰谷或强塞 null"
            if narrative or seed_hint or typed_contract else
            "非结构驱动字段按领域自然演化；不为题型强制中间峰谷，停用按已声明约定处理")

        rejected_intrinsic: list[dict] = []
        for rnd in range(4):
            need = want_total - base_count - len(produced)
            if need <= 0:
                break
            wants = [min(batch, max(0, need - i * batch)) for i in range((need + batch - 1) // batch)]
            actual_objects = [{key: deepcopy(e[key]) for key in ("name", "type", "fields", "purpose")
                               if key in e} for e in merged["entities"] + produced]

            author_context = ""
            if natural_identity:
                author_context = (
                    "\n【冻结白皮书的类型与关系声明；仅供跨对象理解，不扩大本批字段权限】"
                    + json.dumps(author_declarations, ensure_ascii=False)
                    + "\n【已发布对象及 canonical 字段时间线；只读，不得修改】"
                    + json.dumps(published_objects, ensure_ascii=False)
                    + "\n【此前已生成的实际对象与内在事实；purpose 仅为作者用途说明】"
                    + json.dumps(actual_objects, ensure_ascii=False)
                    + "\n上述本轮对象仍是待编译候选，不是已审通过的事实；"
                      "本轮尚无记录的字段与关系不能由名称或用途说明补成真值。"
                      "可以在新对象的标题或描述中准确引用已有对象的完整名称，"
                      "引用已有对象不等于新建同名实体。不要为躲避重名而另造被引用的人或对象。"
                      "仍只返回本批类型与授权内在字段；不改已有对象，不代写结构作者负责的关系或事件。")
                if joint_enabled:
                    author_context += (
                        "\n依据共同计划和已有实际对象分配本批用途，不引用尚不存在的计划别名。"
                        "可在每个 entity 附 purpose 简述其用途；它不新增 canonical 事实。")

            intrinsic_feedback = (
                "\n【上轮本类型未接受的内在字段候选及原编译错误】\n"
                + json.dumps(rejected_intrinsic, ensure_ascii=False)
                + "\n这些对象尚未计入已生成数量。只在本批类型原字段权限内修正或重新提出，"
                  "返回本轮所需数量；保留已接受对象，不补关系或事件，不改冻结声明。"
                if rejected_intrinsic else "")
            rejected_intrinsic = []

            def _world_batch(want):
                check_joint_plan()
                used_names = json.dumps(sorted(seen), ensure_ascii=False)
                return tracer.chat_json("world.batch",
                    [{"role": "system", "content": sysp},
                     {"role": "user", "content": render("world.user", want=want, noun=type_noun,
                                                          type_id=tid, smax=n_sessions - 1,
                                                          trajectory_request=trajectory_request,
                                                          extra=type_extra)
                      + joint_text
                      + author_context
                      + intrinsic_feedback
                      + ((f"\n【全世界已占用实体完整名称；不得再次新建同名实体或换类型冒用】{used_names}"
                          "\n必须返回足量、与本类型 noun 相称且完整名称不同的新实体；"
                          "标题或描述可按实际语义引用已有对象。") if natural_identity else
                         (f"\n【全世界已占用专名，禁止复用或换类型冒用】{used_names}"
                          "\n必须返回足量、与本类型 noun 相称的新专名。"))}],
                    temperature=0.7, max_tokens=WORLD_JSON_MAX_TOKENS,
                    strict_json=True, response_format={"type": "json_object"})

            for out in config.pmap(_world_batch, wants, workers=len(wants)):
                check_joint_plan()
                if not isinstance(out, dict) or "__error__" in out:
                    detail = (out.get("__error__", "非 JSON object")
                              if isinstance(out, dict) else "非 JSON object")
                    raise WorldBlueprintError(
                        f"world.batch[{tid}] 调用失败:{detail}")
                for raw in (out.get("entities", []) if isinstance(out, dict) else []):
                    if not isinstance(raw, dict):
                        continue
                    e = dict(raw)
                    if typed_contract and e.get("type") != tid:
                        continue
                    e["type"] = tid                 # legacy 输出常写 noun；入真源后一律用稳定 type id
                    nm = e.get("name")
                    if not nm or nm in seen or not isinstance(e.get("fields", {}), dict):
                        continue
                    base = _strip_disambig(nm)
                    if not natural_identity and base in seen_base:
                        continue
                    allowed = {f.get("name") for f in intrinsic if f.get("name")}
                    fields = _canonicalize_generated_field_names(
                        dict(e.get("fields") or {}), allowed, intrinsic)
                    off_schema = [] if open_schema else [fn for fn in fields if fn not in allowed]
                    for fn in off_schema:
                        fields.pop(fn, None)
                    if typed_contract:
                        # 事件字段连初态也只由 structure 生成。保留 entity batch 的 session-0
                        # 基线会与 session-0 event 双写，形成两个真源。
                        for fname in event_fields.get(tid, set()) - relation_fields.get(tid, set()):
                            fields.pop(fname, None)
                    e["fields"] = fields
                    if typed_contract:
                        issues = _intrinsic_proposal_issues(e, blueprint)
                        if issues:
                            rejection = {"proposal": deepcopy(e), "issues": issues}
                            rejected_intrinsic.append(rejection)
                            if _draft_state is not None:
                                _draft_state.setdefault("intrinsic_rejections", []).append(
                                    {"type_id": tid, "round": rnd + 1, **deepcopy(rejection)})
                            continue  # Retry this original type author within the same four rounds.
                    seen.add(nm); seen_base.add(base); produced.append(e)
                    if base_count + len(produced) >= want_total:
                        break
            log(f"  世界·{tid} round{rnd+1}:累计 {base_count + len(produced)}/{want_total} {type_noun}"
                f"{'(增量)' if existing is not None else ''}(并发 {len(wants)} 批)")
        if base_count + len(produced) < want_total:
            raise WorldBlueprintError(
                f"entity type {tid} 实例不足:{base_count + len(produced)}/{want_total}；4 轮后仍未满足蓝图")
        merged["entities"].extend(produced[:max(0, want_total - base_count)])

    # 关系/事件在所有 typed entities 生成后统一实例化；此处只负责 canonical world。
    structural_markers = ("entity ", "relation", "event", "causal rule")
    structure: dict = {}
    if (typed_contract and (blueprint.get("relation_types") or blueprint.get("event_types"))
            and (not repairing or _draft_state["repair_targets"]["structure"])):
        def _initial_state(name, tid, raw_fields=None):
            out = {}
            if raw_fields is None and existing is not None:
                for fname in event_fields.get(tid, set()):
                    tl = existing.entities.get(name, {}).get(fname)
                    if tl and tl.value_at_session(0) not in (None, INSUFFICIENT, INVALID):
                        out[fname] = tl.value_at_session(0)
                return out
            for fname in event_fields.get(tid, set()):
                spec = (raw_fields or {}).get(fname) or {}
                if spec.get("type") == "stable" or ("value" in spec and "trajectory" not in spec):
                    if spec.get("value") not in (None, ""):
                        out[fname] = spec.get("value")
                else:
                    first = next((p for p in spec.get("trajectory", [])
                                  if isinstance(p, dict) and _as_int(p.get("session"), -1) == 0
                                  and p.get("value") not in (None, "")), None)
                    if first:
                        out[fname] = first.get("value")
            return out

        # The structure author must see the already generated facts, not just
        # occupied names. Keep raw intrinsic specifications and existing
        # canonical operation histories distinct; neither is a new truth source.
        existing_data = existing.to_dict() if existing is not None else {}
        catalog = [{"name": e, "type": existing_types.get(e),
                    "initial_state": _initial_state(e, existing_types.get(e)),
                    "canonical_fields": existing_data["entities"][e]} for e in base_ents]
        catalog += [{"name": e.get("name"), "type": e.get("type"),
                     "initial_state": _initial_state(e.get("name"), e.get("type"), e.get("fields") or {}),
                     "intrinsic_fields": {f: deepcopy(value) for f, value in (e.get("fields") or {}).items()
                                          if not joint_enabled or f not in event_fields.get(e.get("type"), set())},
                     **({"purpose": deepcopy(e["purpose"])} if "purpose" in e else {})}
                    for e in merged["entities"]]
        existing_canonical = {key: deepcopy(existing_data.get(key, []))
                              for key in ("relations", "events", "cascades", "absent_fields")}
        calendar = WorldState(n_sessions=n_sessions, world_blueprint=blueprint)
        session_dates = [{"session": session, "date": calendar.date_of_session(session)}
                         for session in range(n_sessions)]
        review_hint = ("\n【本次定向返修：原候选与可质疑审阅意见】\n"
                       + json.dumps({"original_proposal": merged,
                                     "reviewed_world": _draft_state["source_draft"]["candidate_world"],
                                     "fallible_review_feedback": _draft_state["feedback"]}, ensure_ascii=False)
                       + ("\n只修改本次未发布候选的relations/events以及结构作者拥有的initial_states；"
                          "初态仍仅限新实体的event-owned非relation字段，不改实体身份、其他内在字段、蓝图或种子。"
                          if joint_enabled else
                          "\n只修改本次候选的relations/events，不改实体、内在字段、蓝图或种子。")
                       +
                       "审阅意见不是新增业务规则；可依据原任务/已有事实反驳并保持原结构，或明确未决。"
                       "不要为了题型或难度改造真值。返回完整候选结构，可附issue_responses列表，"
                       "各项含issue_id、disposition(addressed|disputed|deferred)、response；解释不能补写世界事实。"
                       if repairing else "")
        hint = ""
        for attempt in range(3):
            before_structure = deepcopy(merged) if repairing else None
            check_joint_plan()
            structure = tracer.chat_json("world.structure",
                [{"role": "system", "content": render("world.structure", smax=n_sessions - 1)
                  + (world_joint_plan.BASELINE_INSTRUCTION if joint_enabled else "")},
                 {"role": "user", "content": render(
                     "world.structure_user", time_unit=time_unit, cadence=temporal["cadence"],
                     smax=n_sessions - 1, blueprint=json.dumps(blueprint, ensure_ascii=False),
                     entities=json.dumps(catalog, ensure_ascii=False),
                     session_dates=json.dumps(session_dates, ensure_ascii=False),
                     existing_canonical=json.dumps(existing_canonical, ensure_ascii=False))
                 + (("\n【game 剧情节奏】事件至少铺到 3 个 session，任何一个 session 不得堆入过半事件；"
                     "按前置行动→结果组织，禁止把击败、完成、解锁等收束事件全塞进开场。"
                     "所有事件必须经共享参与者、显式关系或 caused_by 连到唯一主角。")
                    if narrative else "")
                 + seed_hint + joint_text + review_hint + hint}],
                temperature=0.4 if attempt == 0 else 0.2,
                max_tokens=WORLD_JSON_MAX_TOKENS,
                model=config.STRUCTURE_MODEL, retries=1, strict_json=True)
            check_joint_plan()
            if not isinstance(structure, dict) or "__error__" in structure:
                detail = structure.get("__error__", "非 JSON object") if isinstance(structure, dict) else "非 JSON object"
                raise WorldBlueprintError(f"world.structure 调用失败:{detail}")
            attempt_record = {"attempt": attempt + 1, "raw_output": deepcopy(structure),
                              "status": "returned"}
            if _draft_state is not None:
                _draft_state.setdefault("structure_attempts", []).append(attempt_record)
            if isinstance(structure, dict):
                if repairing:
                    if any(k in structure for k in ("entities", "world_blueprint", "seed_contract", "cascades", "absent_fields")):
                        raise WorldBlueprintError("world structure repair attempted to change non-structural inputs")
                    if not all(isinstance(structure.get(key), list)
                               and all(isinstance(item, dict) for item in structure[key])
                               for key in ("relations", "events")):
                        raise WorldBlueprintError("world structure repair needs complete relations/events arrays")
                    if existing is not None:
                        _check_repair_baseline(structure, existing, blueprint)
                _wire_declared_causality(structure, blueprint)
                if joint_enabled:
                    from pipeline.value_types import ValueComparisonError
                    try:
                        new_entities, new_initial_states = world_joint_plan.apply_initial_states(
                            merged, structure, blueprint, base_ents, initial_states)
                    except (WorldBlueprintError, ValueComparisonError) as error:
                        # Author-format/ownership errors use the same bounded structure
                        # feedback loop. The helper has not mutated the prior candidate.
                        attempt_record.update(status="invalid_initial_state", issues=[str(error)])
                        if attempt == 2:
                            raise WorldBlueprintError(
                                "world structure initial-state validation exhausted: " + str(error)) from error
                        hint = ("\n【上轮初态机械校验失败，必须基于该候选修正】\n- "
                                + str(error)
                                + "\n初态与事件不得占同一 session-0 槽，不得略过校验或修改已发布世界。"
                                  "修正本候选初态或合法事件安排；返回完整 relations/events/initial_states。"
                                + "\n【上轮未接受的完整候选 JSON】\n"
                                + json.dumps(structure, ensure_ascii=False))
                        log(f"  ⟳ 世界骨架初态修复轮{attempt+1}:{error}")
                        continue
                    merged["entities"] = new_entities
                    initial_states = new_initial_states
                    if _draft_state is not None:
                        _draft_state["initial_states"] = deepcopy(initial_states)
                relations = [x for x in structure.get("relations", []) if isinstance(x, dict)]
                events = [x for x in structure.get("events", []) if isinstance(x, dict)]
                if existing is not None:
                    relations, events = _filter_incremental_structure(structure, existing, blueprint)
                merged["relations"], merged["events"] = relations, events
                if _draft_state is not None:
                    if _draft_state["raw_merged"] is None:
                        _draft_state["raw_merged"] = deepcopy(merged)
                    if repairing:
                        tracer.record_changes(structure, before_structure, merged,
                                              allowed_structure=["relations", "events"]
                                              + (["initial_states"] if joint_enabled else []))
            trial, trial_issues = assemble_world(merged, blueprint=blueprint, existing=existing)
            if narrative and isinstance(structure, dict):
                connected = 0
                for _ in range(len(trial.events)):
                    if not _connect_narrative_events(structure, trial, blueprint):
                        break
                    connected += 1
                    merged["events"] = [x for x in structure.get("events", []) if isinstance(x, dict)]
                    trial, trial_issues = assemble_world(
                        merged, blueprint=blueprint, existing=existing)
                if connected:
                    log(f"  ✓ 代码连接 {connected} 个断开剧情事件到主角分量")
                if repairing:
                    tracer.record_changes(structure, before_structure, merged,
                                          allowed_structure=["relations", "events"]
                                          + (["initial_states"] if joint_enabled else []))
            structural_issues = [x for x in trial_issues if x.startswith(structural_markers)]
            structural_issues.extend(seed_world_issues(wp, trial))
            trial_defects = _blocking_world_defects(
                validate(trial, merged, profile), narrative,
                protected_fields=protected_fields, entity_types=trial.entity_types)
            driven_defects, _ = _partition_structure_driven_defects(
                trial_defects, trial.entity_types, relation_fields, event_fields)
            structural_issues.extend(
                "structure-driven field "
                f"{item.get('entity')}.{item.get('field')}[{item.get('type')}]:"
                f"{item.get('detail')}；必须只改 relations/events，不得另写实体轨迹"
                for item in driven_defects
            )
            if narrative:
                structural_issues.extend(
                    f"story world 不可叙事:{issue}" for issue in narrative_world_issues(trial))
            if not structural_issues:
                break
            log(f"  ⟳ 世界骨架实例修复轮{attempt+1}:{len(structural_issues)} 个契约违例")
            hint = ("\n【上轮机械校验失败，必须基于上轮候选逐项修正】\n- "
                    + "\n- ".join(structural_issues)
                    + "\n【上轮候选 JSON】\n"
                    + json.dumps(structure, ensure_ascii=False))

    if _draft_state is not None and _draft_state["raw_merged"] is None:
        _draft_state["raw_merged"] = deepcopy(merged)
    ws, compile_issues = assemble_world(merged, blueprint=blueprint, existing=existing)
    structural_issues = [x for x in compile_issues if x.startswith(structural_markers)]
    final_defects = _blocking_world_defects(validate(ws, merged, profile), narrative,
                                           protected_fields=protected_fields, entity_types=ws.entity_types)
    driven_defects, _ = _partition_structure_driven_defects(
        final_defects, ws.entity_types, relation_fields, event_fields)
    structural_issues.extend(
        "structure-driven field "
        f"{item.get('entity')}.{item.get('field')}[{item.get('type')}]:"
        f"{item.get('detail')}；必须只改 relations/events，不得另写实体轨迹"
        for item in driven_defects
    )
    if narrative:
        structural_issues.extend(
            f"story world 不可叙事:{issue}" for issue in narrative_world_issues(ws))
    structural_issues.extend(seed_world_issues(wp, ws))
    if structural_issues:
        raise WorldBlueprintError("世界实例未满足 blueprint:\n- " + "\n- ".join(structural_issues))
    # ★W.3 CRITIC 修复轮:assemble 算出的缺陷不再"只 log 就扔"——定向重生成坏字段(复用并行骨架:发散批次→收敛修复)
    ent_idx = {e.get("name"): e for e in merged["entities"]}
    for rep in range(3):
        all_defects = validate(ws, merged, profile)
        defects = _blocking_world_defects(all_defects, narrative,
                                          protected_fields=protected_fields, entity_types=ws.entity_types)
        escaped_structural, defects = _partition_structure_driven_defects(
            defects, ws.entity_types, relation_fields, event_fields)
        if escaped_structural:
            details = [
                f"{item.get('entity')}.{item.get('field')}[{item.get('type')}]:{item.get('detail')}"
                for item in escaped_structural
            ]
            raise WorldBlueprintError(
                "结构驱动字段缺陷未在 world.structure 阶段收敛:\n- "
                + "\n- ".join(details))
        if not defects:
            break
        by_ent: dict = {}
        for d in defects:
            by_ent.setdefault(d["entity"], []).append(d)
        if repairing and any(ent not in repair_fields or
                             not {d["field"] for d in ds}.issubset(repair_fields[ent])
                             for ent, ds in by_ent.items()):
            raise WorldBlueprintError("mechanical repair would exceed the reviewed intrinsic targets")
        log(f"  ⟳ 世界修复轮{rep+1}:{len(defects)} 个字段缺陷({len(by_ent)} 实体)→ 定向重生成坏字段")

        def _repair(item):
            check_joint_plan()
            ent, ds = item
            cur = (ent_idx.get(ent) or {}).get("fields", {})
            tid = (ent_idx.get(ent) or {}).get("type") or getattr(ws, "entity_types", {}).get(ent)
            repair_noun = next((t["noun"] for t in type_specs if t["id"] == tid), noun)
            lines = "\n".join(
                f"  字段「{d['field']}」缺陷[{d['type']}]:{d['detail']};当前={json.dumps(cur.get(d['field'], {}), ensure_ascii=False)}"
                for d in ds)
            return ent, tracer.chat_json("world.repair",
                [{"role": "system", "content": render("world.repair", noun=repair_noun, smax=n_sessions - 1)},
                 {"role": "user", "content": render("world.repair_user", noun=repair_noun, ent=ent, defects=lines, smax=n_sessions - 1)
                  + seed_entity_prompt(wp, tid) + joint_text}],
                temperature=0.8, max_tokens=4096)

        for ent, out in config.pmap(_repair, list(by_ent.items()), workers=min(8, len(by_ent))):
            check_joint_plan()
            e = ent_idx.get(ent)
            newf = (out.get("fields") if isinstance(out, dict) else None) or {}
            if repairing and not set(newf).issubset(repair_fields[ent]):
                raise WorldBlueprintError("mechanical author changed fields outside repair targets")
            if e and newf:                                # 只覆盖被点名的坏字段,不新增/不动其它字段
                before_fields = deepcopy(merged) if repairing else None
                e["fields"].update({k: v for k, v in newf.items() if k in e.get("fields", {})})
                if repairing:
                    tracer.record_changes(out, before_fields, merged, entity=ent,
                                          allowed_fields=sorted(repair_fields[ent]))
        ws, compile_issues = assemble_world(merged, blueprint=blueprint, existing=existing)
        structural_issues = [x for x in compile_issues if x.startswith(structural_markers)]
        if structural_issues:
            # 字段修复模型仍可能写出状态表外的值；这仍是下一轮的字段缺陷，
            # 不应在第一次失败时绕过既有三轮修复预算。
            log(f"  ↻ 字段修复仍有 {len(structural_issues)} 个 schema 值违例，交给下一轮")
    if structural_issues:
        raise WorldBlueprintError(
            "字段修复后世界结构仍不满足 blueprint:\n- " + "\n- ".join(structural_issues))
    ws.n_sessions = max(ws.n_sessions or 0, n_sessions, (existing.n_sessions if existing is not None else 0))
    validate_seed_world(wp, ws)
    if existing is not None:                          # ★增量 augment:只把【新实体】并入既有世界,旧实体/旧 docs 全不动
        # 保持调用方持有的对象身份不变，但用“旧世界 + delta 编译”的完整结果原子替换其状态。
        for attr in ("entities", "cascades", "absent_fields", "n_sessions", "conflicts", "sensitive",
                     "conditional_rules", "rule_instances", "entity_types", "relations", "events",
                     "world_blueprint"):
            setattr(existing, attr, getattr(ws, attr))
        existing._trended_fields = list(getattr(ws, "_trended_fields", []) or [])
        ws = existing
    rem = validate(ws, merged, profile)               # ★validate 在【注趋势前】跑(对 merged 一致,不误报);imprint 产出本就良构,无需复验
    blocking_rem = _blocking_world_defects(rem, narrative,
                                          protected_fields=protected_fields, entity_types=ws.entity_types)
    if len(rem) > len(blocking_rem):
        log(f"  · 保留 {len(rem) - len(blocking_rem)} 个非阻塞轨迹提示，不为题型强扭世界")
    if typed_contract and blocking_rem:
        details = [f"{d.get('entity')}.{d.get('field')}[{d.get('type')}]:{d.get('detail')}"
                   for d in blocking_rem]
        raise WorldBlueprintError("typed world 修复轮耗尽后仍有真值缺陷:\n- " + "\n- ".join(details))
    if narrative:
        # 世界已通过结构编译、字段修复和最终 validate；坏剧情只重试剧情，不回滚世界。
        from pipeline.story import (StoryLedgerError, compile_story_ledger,
                                    review_narrative_supportedness)

        story_entities = [
            {"name": name, "type": ws.entity_types.get(name)} for name in sorted(ws.entities)
        ]
        event_labels = {item.get("id"): item.get("label")
                        for item in blueprint.get("event_types", [])}
        story_canon = {
            "entities": story_entities,
            "relations": ws.relations,
            "events": [{**event, "label": event_labels.get(event.get("type"), event.get("type"))}
                       for event in ws.events],
        }
        story_hint = ""
        story_issues: list[str] = []
        for attempt in range(4):
            proposal = tracer.chat_json(
                "world.story",
                [{"role": "system", "content": render("world.story")},
                 {"role": "user", "content": render(
                     "world.story_user",
                     blueprint=json.dumps(blueprint, ensure_ascii=False),
                     entities=json.dumps(story_entities, ensure_ascii=False),
                     events=json.dumps(ws.events, ensure_ascii=False),
                     hint=story_hint)}],
                temperature=0.6 if attempt == 0 else 0.2, max_tokens=8192)
            try:
                ledger = compile_story_ledger(ws, proposal)
            except StoryLedgerError as exc:
                story_issues = [f"[{x.code}] {x.path}: {x.message}" for x in exc.issues]
                log(f"  ⟳ Story Ledger 修复轮{attempt + 1}:{len(story_issues)} 个契约违例")
                reason = "机械校验失败"
            else:
                story_issues = review_narrative_supportedness(
                    tracer, canon=story_canon, candidate=ledger, scope="Story Ledger")
                if not story_issues:
                    ws.narrative = ledger
                    break
                log(f"  ⟳ Story Ledger 语义修复轮{attempt + 1}:{len(story_issues)} 个无依据断言")
                reason = "语义审查失败"
            story_hint = (f"\n【上轮 Story Ledger {reason}，只修剧情编排，不改世界；"
                          "只改被点名句子，保留其余已通过内容】\n- "
                          + "\n- ".join(story_issues)
                          + "\n【上轮候选 JSON】\n"
                          + json.dumps(proposal, ensure_ascii=False))
        else:
            raise WorldBlueprintError("game Story Ledger 四轮修复后仍不合法:\n- "
                                      + "\n- ".join(story_issues))
    coll = name_collisions(ws)                        # ★Fix3:表面塌缩兜底检测(收集期已按主干去重,这里抓漏网)
    log(f"  ✓ 基础世界:{len(ws.entities)} 实体 / {ws.n_sessions} {time_unit} / 残留真值缺陷 {len(blocking_rem)}"
        f" / 非阻断生成提示 {len(rem) - len(blocking_rem)}"
        + (f" / ⚠表面塌缩近重名 {coll}" if coll else ""))  # 产线基质由 stage_world 的 line.prepare() 叠加
    # ★声明-世界对齐自检(刀1审计:声明字段在世界中无命中时静默 no-op,漂移不可观测)
    all_fields = {f for flds in ws.entities.values() for f in flds}
    ghost = [m.get("field") for m in (profile.get("state_machines") or []) if m.get("field") and m["field"] not in all_fields]
    ghost += [f.get("name") for f in profile.get("field_schema", []) if f.get("unit") and f.get("name") not in all_fields]
    if ghost:
        log(f"  ⚠声明字段未在世界命中(states/unit 约束将空转,检查议会命名一致性):{sorted(set(ghost))}")
    return _finish_world(wp, ws, merged, profile, blueprint, narrative, protected_fields, log)
