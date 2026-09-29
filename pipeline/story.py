"""游戏场景的最小 Story Ledger 契约。

Ledger 只回答“哪些 canonical event 属于哪一幕、这一幕承担什么叙事作用”。
参与者、状态变化、因果与物品来源继续只存在于 ``WorldState``，这里不复制也不复核。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable

from pipeline.world_state import WorldState


LEDGER_VERSION = 1
DRAMATIC_FUNCTIONS = frozenset({
    "setup",
    "inciting_incident",
    "rising_action",
    "reversal",
    "crisis",
    "climax",
    "resolution",
})
_DRAMATIC_PHASE = {
    "setup": 0,
    "inciting_incident": 1,
    "rising_action": 2,
    "reversal": 3,
    "crisis": 4,
    "climax": 5,
    "resolution": 6,
}
_LEDGER_KEYS = frozenset({
    "version", "scenario", "premise", "goal", "stakes",
    "protagonist_ref", "scenes",
})
_SCENE_KEYS = frozenset({
    "scene_id", "session", "order", "event_refs", "dramatic_function",
})


@dataclass(frozen=True)
class StoryIssue:
    """一个可定位、可序列化的 Story Ledger 契约错误。"""

    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        """返回适合写入审计报告的 JSON 字典。"""
        return asdict(self)


class StoryLedgerError(ValueError):
    """Story Ledger 无法与 canonical WorldState 对齐。"""

    def __init__(self, issues: Iterable[StoryIssue]):
        self.issues = list(issues)
        detail = "\n- ".join(f"[{x.code}] {x.path}: {x.message}" for x in self.issues)
        super().__init__("Story Ledger 非法" + (f":\n- {detail}" if detail else ""))


def _issue(issues: list[StoryIssue], code: str, path: str, message: str) -> None:
    issues.append(StoryIssue(code=code, path=path, message=message))


def _strict_int(value: Any) -> bool:
    """仅接受 JSON integer；bool 虽是 int 子类，也必须拒绝。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_declared_causal_edge(ws: WorldState, parent: dict | None, child: dict | None) -> bool:
    """判断事件边是否被蓝图以相同类型和时间差明确声明。"""
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return False
    parent_session, child_session = parent.get("session"), child.get("session")
    if not _strict_int(parent_session) or not _strict_int(child_session):
        return False
    delay = child_session - parent_session
    return any(
        isinstance(rule, dict)
        and rule.get("trigger_event") == parent.get("type")
        and rule.get("effect_event") == child.get("type")
        and rule.get("delay_sessions") == delay
        for rule in ((ws.world_blueprint or {}).get("causal_rules") or [])
    )


def _event_index(ws: WorldState, issues: list[StoryIssue]) -> dict[str, dict]:
    """建立 canonical event 索引；坏 ID 与重复 ID 都显式报错。"""
    out: dict[str, dict] = {}
    for index, event in enumerate(ws.events):
        path = f"WorldState.events[{index}]"
        if not isinstance(event, dict):
            _issue(issues, "world_event_invalid", path, "canonical event 必须是 object")
            continue
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            _issue(issues, "world_event_id_invalid", f"{path}.id", "canonical event id 必须是非空字符串")
            continue
        if event_id in out:
            _issue(issues, "world_event_id_duplicate", f"{path}.id", f"canonical event id 重复:{event_id}")
            continue
        out[event_id] = event
    return out


def _story_reachability(ws: WorldState) -> tuple[set[tuple[str, str]], set[str]]:
    """返回从唯一主角可达的图节点，以及 canonical event ID 集合。"""
    blueprint = ws.world_blueprint or {}
    primary_types = {
        item.get("id") for item in (blueprint.get("entity_types") or [])
        if isinstance(item, dict) and item.get("primary") is True and item.get("id")
    }
    roots = [name for name, type_id in ws.entity_types.items() if type_id in primary_types]
    if len(roots) != 1:
        return set(), {
            str(event.get("id")) for event in ws.events
            if isinstance(event, dict) and event.get("id")
        }

    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}

    def connect(left: tuple[str, str], right: tuple[str, str]) -> None:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)

    for relation in ws.relations:
        if isinstance(relation, dict) and relation.get("from") and relation.get("to"):
            connect(("entity", str(relation["from"])), ("entity", str(relation["to"])))
    event_by_id = {
        str(event["id"]): event for event in ws.events
        if isinstance(event, dict) and event.get("id")
    }
    event_ids = set(event_by_id)
    for event_id, event in event_by_id.items():
        if not isinstance(event, dict) or not event.get("id"):
            continue
        event_node = ("event", event_id)
        graph.setdefault(event_node, set())
        for entity in (event.get("participants") or {}).values():
            if entity:
                connect(event_node, ("entity", str(entity)))
        parent = event.get("caused_by")
        parent_event = event_by_id.get(str(parent)) if parent else None
        if _is_declared_causal_edge(ws, parent_event, event):
            connect(event_node, ("event", str(parent)))

    seen = {("entity", roots[0])}
    frontier = list(seen)
    while frontier:
        node = frontier.pop()
        for neighbor in graph.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen, event_ids


def connected_story_entities(ws: WorldState) -> set[str]:
    """返回经关系、事件参与或显式因果连到唯一主角的实体专名。"""
    seen, _ = _story_reachability(ws)
    return {name for kind, name in seen if kind == "entity"}


def disconnected_story_events(ws: WorldState) -> list[str]:
    """返回无法经事件参与、显式因果或世界关系连到唯一主角的事件 ID。"""
    seen, event_ids = _story_reachability(ws)
    return sorted(event_id for event_id in event_ids if ("event", event_id) not in seen)


def narrative_world_issues(ws: WorldState) -> list[str]:
    """检查剧情世界最小可叙事性：主线连通，且事件没有挤成一个数据库快照。"""
    issues = []
    disconnected = disconnected_story_events(ws)
    if disconnected:
        issues.append(f"事件无法经参与者/关系/因果连到唯一主角:{disconnected}")
    valid_events = [event for event in ws.events
                    if isinstance(event, dict) and event.get("id")
                    and isinstance(event.get("session"), int)
                    and not isinstance(event.get("session"), bool)]
    sessions = [event["session"] for event in valid_events]

    # 只有蓝图声明的 delay=0 因果边才是不可拆组件；模型随意写 caused_by
    # 不能借此绕过跨 session 的剧情节奏门禁。
    parent = {str(event["id"]): str(event["id"]) for event in valid_events}

    def _find(event_id: str) -> str:
        while parent[event_id] != event_id:
            parent[event_id] = parent[parent[event_id]]
            event_id = parent[event_id]
        return event_id

    def _union(left: str, right: str) -> None:
        left_root, right_root = _find(left), _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_id = {str(event["id"]): event for event in valid_events}
    for event_id, event in by_id.items():
        caused_by = str(event.get("caused_by") or "")
        prior = by_id.get(caused_by)
        if (prior and prior["session"] == event["session"]
                and _is_declared_causal_edge(ws, prior, event)):
            _union(event_id, caused_by)
    component_sizes: dict[str, int] = {}
    for event_id in parent:
        root = _find(event_id)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    required_span = min(3, ws.n_sessions, len(component_sizes))
    if required_span > 1 and len(set(sessions)) < required_span:
        issues.append(f"{len(sessions)} 个主线事件只覆盖 {len(set(sessions))} 个 session，至少需 {required_span} 个")
    if len(sessions) >= 4:
        peak = max((sessions.count(session) for session in set(sessions)), default=0)
        limit = max((len(sessions) + 1) // 2, max(component_sizes.values(), default=1))
        if peak > limit:
            issues.append(f"单一 session 堆叠 {peak}/{len(sessions)} 个主线事件，上限 {limit}")
    return issues


def review_narrative_supportedness(
        tracer, *, canon: dict, candidate: Any, scope: str) -> list[str]:
    """用一个只读语义闸找出 canonical 之外的具体断言；异常或坏格式一律不放行。"""
    from pipeline.prompts import render

    try:
        out = tracer.chat_json(
            "narrative.review",
            [{"role": "system", "content": render("narrative.review")},
             {"role": "user", "content": render(
                 "narrative.review_user", scope=scope,
                 canon=json.dumps(canon, ensure_ascii=False),
                 candidate=json.dumps(candidate, ensure_ascii=False))}],
            temperature=0.0, max_tokens=2048)
    except Exception as exc:
        return [f"语义审查调用失败:{type(exc).__name__}"]
    claims = out.get("unsupported_claims") if isinstance(out, dict) else None
    if not isinstance(claims, list):
        return ["语义审查返回格式非法"]
    return [str(item).strip() for item in claims if str(item).strip()]


def validate_story_ledger(
        ws: WorldState, ledger: dict, *, require_all_events: bool = True) -> list[StoryIssue]:
    """验证轻量 Ledger 的形状、唯一主角，以及 scene→event 引用闭包。"""
    issues: list[StoryIssue] = []
    if not isinstance(ledger, dict):
        return [StoryIssue("ledger_invalid", "$", "ledger 必须是 JSON object")]

    extras = sorted(set(ledger) - _LEDGER_KEYS)
    if extras:
        _issue(issues, "ledger_keys_invalid", "$", f"Ledger 含契约外字段:{extras}")
    if ledger.get("version") != LEDGER_VERSION:
        _issue(issues, "version_invalid", "version", f"只支持 version={LEDGER_VERSION}")
    if ledger.get("scenario") != "game":
        _issue(issues, "scenario_invalid", "scenario", "当前 Story Ledger 只支持 game")
    for key in ("premise", "goal", "stakes"):
        value = ledger.get(key)
        if not isinstance(value, str) or not value.strip():
            _issue(issues, "story_core_missing", key, f"{key} 必须是非空字符串")

    blueprint = ws.world_blueprint or {}
    primary_types = {
        item.get("id") for item in (blueprint.get("entity_types") or [])
        if isinstance(item, dict) and item.get("primary") is True and item.get("id")
    }
    primary_entities = sorted(
        name for name, type_id in ws.entity_types.items() if type_id in primary_types)
    protagonist = ledger.get("protagonist_ref")
    if len(primary_types) != 1:
        _issue(issues, "primary_type_not_unique", "WorldState.world_blueprint.entity_types",
               f"primary type 必须恰有一个，实际 {len(primary_types)}")
    if len(primary_entities) != 1:
        _issue(issues, "protagonist_not_unique", "protagonist_ref",
               f"primary type 实例必须恰有一个，实际 {primary_entities}")
    if not isinstance(protagonist, str) or not protagonist.strip():
        _issue(issues, "protagonist_invalid", "protagonist_ref", "protagonist_ref 必须是非空字符串")
    elif protagonist not in ws.entities:
        _issue(issues, "protagonist_unknown", "protagonist_ref", f"实体不存在:{protagonist}")
    elif primary_entities != [protagonist]:
        _issue(issues, "protagonist_not_primary", "protagonist_ref",
               f"主角必须是唯一 primary 实例:{primary_entities}")

    event_by_id = _event_index(ws, issues)
    for world_issue in narrative_world_issues(ws):
        _issue(issues, "narrative_world_invalid", "WorldState.events", world_issue)
    scenes = ledger.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        _issue(issues, "scenes_invalid", "scenes", "scenes 必须是非空数组")
        scenes = []

    seen_scene_ids: set[str] = set()
    seen_ranks: dict[tuple[int, int], str] = {}
    event_owners: dict[str, str] = {}
    ranked_phases: list[tuple[tuple[int, int], int, str]] = []
    for index, scene in enumerate(scenes):
        path = f"scenes[{index}]"
        if not isinstance(scene, dict):
            _issue(issues, "scene_invalid", path, "scene 必须是 object")
            continue
        extra_scene_keys = sorted(set(scene) - _SCENE_KEYS)
        if extra_scene_keys:
            _issue(issues, "scene_keys_invalid", path, f"scene 含契约外字段:{extra_scene_keys}")

        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            _issue(issues, "scene_id_invalid", f"{path}.scene_id", "scene_id 必须是非空字符串")
            scene_id = f"<invalid-{index}>"
        elif scene_id in seen_scene_ids:
            _issue(issues, "scene_id_duplicate", f"{path}.scene_id", f"scene_id 重复:{scene_id}")
        else:
            seen_scene_ids.add(scene_id)

        session = scene.get("session")
        order = scene.get("order")
        if not _strict_int(session):
            _issue(issues, "scene_session_invalid", f"{path}.session", "session 必须是整数且不能是 bool")
        elif session < 0 or session >= ws.n_sessions:
            _issue(issues, "scene_session_invalid", f"{path}.session",
                   f"session 必须在 0..{max(0, ws.n_sessions - 1)}")
        if not _strict_int(order):
            _issue(issues, "scene_order_invalid", f"{path}.order", "order 必须是整数且不能是 bool")
        elif order < 0:
            _issue(issues, "scene_order_invalid", f"{path}.order", "order 不能为负数")
        if _strict_int(session) and _strict_int(order):
            rank = (session, order)
            if rank in seen_ranks:
                _issue(issues, "scene_rank_duplicate", path,
                       f"与 {seen_ranks[rank]} 使用相同 rank={rank}")
            else:
                seen_ranks[rank] = scene_id

        dramatic_function = scene.get("dramatic_function")
        if dramatic_function not in DRAMATIC_FUNCTIONS:
            _issue(issues, "dramatic_function_invalid", f"{path}.dramatic_function",
                   f"只允许:{sorted(DRAMATIC_FUNCTIONS)}")
        elif _strict_int(session) and _strict_int(order):
            ranked_phases.append(((session, order), _DRAMATIC_PHASE[dramatic_function], scene_id))

        refs = scene.get("event_refs")
        if not isinstance(refs, list) or not refs:
            _issue(issues, "scene_without_event", f"{path}.event_refs",
                   "每个 scene 必须引用至少一个 canonical event")
            refs = []
        local_refs: set[str] = set()
        for ref_index, event_ref in enumerate(refs):
            ref_path = f"{path}.event_refs[{ref_index}]"
            if not isinstance(event_ref, str) or not event_ref.strip():
                _issue(issues, "event_ref_invalid", ref_path, "event ref 必须是非空字符串")
                continue
            if event_ref in local_refs:
                _issue(issues, "scene_event_ref_duplicate", ref_path,
                       f"同一 scene 重复引用 event:{event_ref}")
                continue
            local_refs.add(event_ref)
            event = event_by_id.get(event_ref)
            if event is None:
                _issue(issues, "event_ref_unknown", ref_path, f"canonical event 不存在:{event_ref}")
                continue
            if event_ref in event_owners:
                _issue(issues, "event_ref_reused", ref_path,
                       f"event 已属于 scene {event_owners[event_ref]}")
            else:
                event_owners[event_ref] = scene_id
            if _strict_int(session) and event.get("session") != session:
                _issue(issues, "event_scene_session_mismatch", ref_path,
                       f"event {event_ref}@{event.get('session')} 被放入 scene@{session}")

    if require_all_events:
        uncovered = sorted(set(event_by_id) - set(event_owners))
        if uncovered:
            _issue(issues, "canonical_events_uncovered", "scenes.event_refs",
                   f"以下 canonical event 未被 scene 覆盖:{uncovered}")
    ranked_phases.sort()
    for previous, current in zip(ranked_phases, ranked_phases[1:]):
        if current[1] < previous[1]:
            _issue(issues, "dramatic_phase_regression", "scenes",
                   f"叙事阶段倒退:{previous[2]} -> {current[2]}")
            break
    if len(ranked_phases) >= 3:
        if ranked_phases[0][1] > _DRAMATIC_PHASE["rising_action"]:
            _issue(issues, "dramatic_arc_starts_too_late", "scenes[0].dramatic_function",
                   "三幕以上剧情必须从 setup/inciting_incident/rising_action 起步")
        if ranked_phases[-1][1] != _DRAMATIC_PHASE["resolution"]:
            _issue(issues, "dramatic_arc_has_no_resolution", "scenes[-1].dramatic_function",
                   "三幕以上剧情必须以 resolution 收束已声明目标")
    return issues


def compile_story_ledger(
        ws: WorldState, proposal: dict, *, require_all_events: bool = True) -> dict:
    """将模型提案收口成固定轻量 schema；不补默认值，也不复制动态真值。"""
    if not isinstance(proposal, dict):
        raise StoryLedgerError([StoryIssue("proposal_invalid", "$", "story proposal 必须是 object")])

    raw_scenes = proposal.get("scenes")
    if not isinstance(raw_scenes, list):
        raw_scenes = []
    scenes = []
    for raw in raw_scenes:
        if not isinstance(raw, dict):
            scenes.append(raw)
            continue
        scenes.append({key: deepcopy(raw.get(key)) for key in (
            "scene_id", "session", "order", "event_refs", "dramatic_function")})
    ledger = {
        "version": LEDGER_VERSION,
        "scenario": "game",
        "premise": deepcopy(proposal.get("premise")),
        "goal": deepcopy(proposal.get("goal")),
        "stakes": deepcopy(proposal.get("stakes")),
        "protagonist_ref": deepcopy(proposal.get("protagonist_ref")),
        "scenes": scenes,
    }
    issues = validate_story_ledger(ws, ledger, require_all_events=require_all_events)
    if issues:
        raise StoryLedgerError(issues)
    return ledger


def replay_story_ledger(ws: WorldState, ledger: dict) -> list[dict]:
    """校验后按 ``(session, order)`` 返回 scene 浅副本，不派生事件动态真值。"""
    issues = validate_story_ledger(ws, ledger)
    if issues:
        raise StoryLedgerError(issues)
    ordered = sorted(ledger["scenes"], key=lambda scene: (
        scene["session"], scene["order"], scene["scene_id"]))
    return [{**scene, "event_refs": list(scene["event_refs"])} for scene in ordered]
