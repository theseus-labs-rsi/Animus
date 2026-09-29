"""game-only 轻量 Story Ledger 的离线自检。

跑法：``./venv/bin/python tests/story_selftest.py``。
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.story import (StoryLedgerError, compile_story_ledger,
                            disconnected_story_events, replay_story_ledger,
                            narrative_world_issues, validate_story_ledger)
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_gen import (_blocking_world_defects, _connect_narrative_events,
                                build_world)
from pipeline.world_state import assemble_world


checks: list[tuple[bool, str]] = []


def ck(name: str, condition) -> None:
    checks.append((bool(condition), name))


def issue_codes(ws, ledger: dict, **kwargs) -> set[str]:
    """返回问题 code 集，避免测试绑定具体提示文案。"""
    return {issue.code for issue in validate_story_ledger(ws, ledger, **kwargs)}


def make_world():
    """构造一个最小、合法的 typed game world。"""
    blueprint = {
        "version": 1,
        "entity_types": [
            {
                "id": "player", "noun": "玩家角色", "count": 1, "primary": True,
                "fields": [{"name": "任务状态", "kind": "status",
                            "states": ["待命", "追猎中", "完成"]}],
            },
            {
                "id": "boss", "noun": "首领", "count": 1, "primary": False,
                "fields": [
                    {"name": "存活状态", "kind": "status", "states": ["存活", "击败"]},
                    {"name": "掉落表引用", "kind": "reference"},
                ],
            },
            {
                "id": "equipment", "noun": "装备", "count": 1, "primary": False,
                "fields": [{"name": "装备状态", "kind": "status",
                            "states": ["未掉落", "已掉落", "已装备"]}],
            },
        ],
        "relation_types": [
            {"id": "potential_drop", "from_type": "boss", "to_type": "equipment",
             "field": "掉落表引用", "temporal": False, "min_count": 1},
        ],
        "event_types": [
            {"id": "hunt", "label": "开始追猎", "roles": {"actor": "player"},
             "effect_fields": [{"role": "actor", "field": "任务状态"}], "min_count": 1},
            {"id": "defeat_boss", "label": "击败首领",
             "roles": {"actor": "player", "target": "boss"},
             "effect_fields": [{"role": "target", "field": "存活状态"}], "min_count": 1},
            {"id": "drop_equipment", "label": "掉落装备",
             "roles": {"source": "boss", "item": "equipment"},
             "effect_fields": [{"role": "item", "field": "装备状态"}], "min_count": 1},
            {"id": "equip_item", "label": "装备物品",
             "roles": {"actor": "player", "item": "equipment"},
             "effect_fields": [{"role": "item", "field": "装备状态"}], "min_count": 1},
        ],
        "causal_rules": [
            {"id": "defeat_causes_drop", "trigger_event": "defeat_boss",
             "effect_event": "drop_equipment", "delay_sessions": 0},
        ],
        "temporal_model": {"unit": "chapter", "cadence": "event-driven",
                           "n_sessions": 4, "step_days": 1},
        "evidence_channels": ["剧情日志", "战斗记录"],
    }
    table = {
        "entities": [
            {"name": "艾尔文", "type": "player",
             "fields": {"任务状态": {"type": "stable", "value": "待命"}}},
            {"name": "霜狼", "type": "boss",
             "fields": {"存活状态": {"type": "stable", "value": "存活"}}},
            {"name": "霜牙", "type": "equipment",
             "fields": {"装备状态": {"type": "stable", "value": "未掉落"}}},
        ],
        "relations": [
            {"id": "rel-loot", "type": "potential_drop", "from": "霜狼", "to": "霜牙",
             "session": 0},
        ],
        "events": [
            {"id": "evt-hunt", "type": "hunt", "session": 1,
             "participants": {"actor": "艾尔文"},
             "effects": [{"entity": "艾尔文", "field": "任务状态", "set": "追猎中"}]},
            {"id": "evt-defeat", "type": "defeat_boss", "session": 2,
             "participants": {"actor": "艾尔文", "target": "霜狼"},
             "effects": [{"entity": "霜狼", "field": "存活状态", "set": "击败"}]},
            {"id": "evt-drop", "type": "drop_equipment", "session": 2,
             "participants": {"source": "霜狼", "item": "霜牙"},
             "effects": [{"entity": "霜牙", "field": "装备状态", "set": "已掉落"}],
             "caused_by": "evt-defeat"},
            {"id": "evt-equip", "type": "equip_item", "session": 3,
             "participants": {"actor": "艾尔文", "item": "霜牙"},
             "effects": [{"entity": "霜牙", "field": "装备状态", "set": "已装备"}]},
        ],
    }
    ws, issues = assemble_world(table, blueprint=blueprint)
    if issues:
        raise AssertionError(f"测试 fixture 必须合法，实际 issues={issues}")
    return ws


def make_proposal() -> dict:
    """返回覆盖全部 canonical event 的有效剧情提案。"""
    return {
        "premise": "北境通道被霜狼封锁。",
        "goal": "艾尔文必须打通北境通道。",
        "stakes": "失败会让山脚聚落断粮。",
        "protagonist_ref": "艾尔文",
        "scenes": [
            {"scene_id": "scene-hunt", "session": 1, "order": 0,
             "event_refs": ["evt-hunt"], "summary": "追猎开始。",
             "dramatic_function": "inciting_incident"},
            {"scene_id": "scene-defeat", "session": 2, "order": 0,
             "event_refs": ["evt-defeat"], "summary": "主角迎战守关首领。",
             "dramatic_function": "climax"},
            {"scene_id": "scene-drop", "session": 2, "order": 1,
             "event_refs": ["evt-drop"], "summary": "胜利打开新的可能。",
             "dramatic_function": "resolution"},
            {"scene_id": "scene-equip", "session": 3, "order": 0,
             "event_refs": ["evt-equip"], "summary": "主角带着战果继续前进。",
             "dramatic_function": "resolution"},
        ],
    }


ws = make_world()
proposal_with_old_fields = make_proposal()
proposal_with_old_fields["key_item_origins"] = [{"item_ref": "霜牙", "event_ref": "evt-drop"}]
proposal_with_old_fields["scenes"][0]["preconditions"] = [{"entity": "艾尔文"}]
proposal_with_old_fields["scenes"][1]["caused_by_scene_refs"] = ["scene-hunt"]
ledger = compile_story_ledger(ws, proposal_with_old_fields)
ck("编译结果只保留轻量 Ledger 字段",
   set(ledger) == {"version", "scenario", "premise", "goal", "stakes",
                   "protagonist_ref", "scenes"})
ck("编译结果只保留轻量 scene 字段",
   all(set(scene) == {"scene_id", "session", "order", "event_refs",
                      "dramatic_function"} for scene in ledger["scenes"]))
ck("有效 Ledger 零问题", validate_story_ledger(ws, ledger) == [])

unsorted = deepcopy(ledger)
unsorted["scenes"].reverse()
replayed = replay_story_ledger(ws, unsorted)
ck("replay 只按 rank 返回浅 scene，不派生动态真值",
   [scene["scene_id"] for scene in replayed]
   == ["scene-hunt", "scene-defeat", "scene-drop", "scene-equip"]
   and all("events" not in scene and "effects" not in scene for scene in replayed))

missing_core = deepcopy(ledger)
missing_core["goal"] = " "
ck("故事核心必须非空", "story_core_missing" in issue_codes(ws, missing_core))

multi_player = deepcopy(ws)
multi_player.entities["莉安娜"] = deepcopy(multi_player.entities["艾尔文"])
multi_player.entity_types["莉安娜"] = "player"
ck("主角必须是唯一 primary 实例",
   "protagonist_not_unique" in issue_codes(multi_player, ledger))

wrong_protagonist = deepcopy(ledger)
wrong_protagonist["protagonist_ref"] = "霜狼"
ck("Ledger 主角必须指向唯一 primary 实例",
   "protagonist_not_primary" in issue_codes(ws, wrong_protagonist))

for bad_value, expected_code in (("1", "scene_session_invalid"),
                                 (True, "scene_session_invalid"),
                                 (False, "scene_order_invalid")):
    bad_rank = deepcopy(ledger)
    key = "order" if bad_value is False else "session"
    bad_rank["scenes"][0][key] = bad_value
    ck(f"rank 严格拒绝 {bad_value!r}", expected_code in issue_codes(ws, bad_rank))

missing_order = deepcopy(make_proposal())
del missing_order["scenes"][0]["order"]
try:
    compile_story_ledger(ws, missing_order)
except StoryLedgerError as exc:
    missing_order_codes = {issue.code for issue in exc.issues}
else:
    missing_order_codes = set()
ck("compile 不为缺失 order 静默补默认值", "scene_order_invalid" in missing_order_codes)

duplicate_rank = deepcopy(ledger)
duplicate_rank["scenes"][2]["order"] = 0
ck("scene rank 不可重复", "scene_rank_duplicate" in issue_codes(ws, duplicate_rank))

bad_summary = deepcopy(ledger)
bad_summary["scenes"][0]["summary"] = "模型自由补写的事实"
ck("scene 不接受无消费方的自由摘要", "scene_keys_invalid" in issue_codes(ws, bad_summary))

bad_function = deepcopy(ledger)
bad_function["scenes"][0]["dramatic_function"] = "boss_fight"
ck("dramatic_function 限定小枚举",
   "dramatic_function_invalid" in issue_codes(ws, bad_function))

regressing_arc = deepcopy(ledger)
regressing_arc["scenes"][0]["dramatic_function"] = "climax"
regressing_arc["scenes"][1]["dramatic_function"] = "rising_action"
ck("dramatic_function 不允许随剧情推进倒退",
   "dramatic_phase_regression" in issue_codes(ws, regressing_arc))

late_arc = deepcopy(ledger)
late_arc["scenes"][0]["dramatic_function"] = "climax"
late_arc["scenes"][1]["dramatic_function"] = "climax"
ck("三幕剧情不能从高潮起步",
   "dramatic_arc_starts_too_late" in issue_codes(ws, late_arc))

open_ended_arc = deepcopy(ledger)
open_ended_arc["scenes"][-1]["dramatic_function"] = "climax"
ck("三幕剧情必须以 resolution 闭合目标",
   "dramatic_arc_has_no_resolution" in issue_codes(ws, open_ended_arc))

unknown_event = deepcopy(ledger)
unknown_event["scenes"][0]["event_refs"] = ["evt-ghost"]
unknown_codes = issue_codes(ws, unknown_event)
ck("未知 event 被拒绝", "event_ref_unknown" in unknown_codes)
ck("canonical event 缺失覆盖被拒绝", "canonical_events_uncovered" in unknown_codes)

duplicate_event = deepcopy(ledger)
duplicate_event["scenes"][1]["event_refs"].append("evt-hunt")
ck("canonical event 不可重复覆盖", "event_ref_reused" in issue_codes(ws, duplicate_event))

empty_scene = deepcopy(ledger)
empty_scene["scenes"][0]["event_refs"] = []
ck("scene 必须至少引用一个 event", "scene_without_event" in issue_codes(ws, empty_scene))

wrong_session = deepcopy(ledger)
wrong_session["scenes"][0]["session"] = 0
ck("scene 与 event session 必须一致",
   "event_scene_session_mismatch" in issue_codes(ws, wrong_session))

extra_key = deepcopy(ledger)
extra_key["scenes"][0]["effects"] = []
ck("持久化 Ledger 拒绝契约外字段", "scene_keys_invalid" in issue_codes(ws, extra_key))

disconnected_world = deepcopy(ws)
disconnected_world.entities["孤狼"] = deepcopy(disconnected_world.entities["霜狼"])
disconnected_world.entity_types["孤狼"] = "boss"
disconnected_world.events.append({
    "id": "evt-rogue", "type": "defeat_boss", "session": 3,
    "participants": {"target": "孤狼"}, "effects": [],
})
ck("主线事件必须经参与者、关系或因果连到唯一主角",
   disconnected_story_events(disconnected_world) == ["evt-rogue"])
disconnected_world.events[-1]["caused_by"] = "evt-hunt"
ck("未声明的 caused_by 不能把孤立事件伪装成主角可达",
   disconnected_story_events(disconnected_world) == ["evt-rogue"])

repair_world = deepcopy(ws)
repair_world.entities["孤狼"] = deepcopy(repair_world.entities["霜狼"])
repair_world.entity_types["孤狼"] = "boss"
repair_world.entities["孤刃"] = deepcopy(repair_world.entities["霜牙"])
repair_world.entity_types["孤刃"] = "equipment"
repair_event = {
    "id": "evt-rogue", "type": "drop_equipment", "session": 3,
    "participants": {"item": "孤刃", "source": "孤狼"},
    "effects": [{"entity": "孤刃", "field": "装备状态", "set": "已掉落"}],
}
repair_world.events.append(deepcopy(repair_event))
repair_structure = {"events": [repair_event]}
ck("断开事件可确定地复用主角分量内同类型实体，并同步 effect owner",
   _connect_narrative_events(repair_structure, repair_world,
                             repair_world.world_blueprint) == 1
   and repair_event["participants"]["item"] == "霜牙"
   and repair_event["effects"][0]["entity"] == "霜牙")

crowded_world = deepcopy(ws)
for event in crowded_world.events:
    event["session"] = 1
ck("主线事件不能全部挤成同一 session 的状态快照",
   any("至少需" in issue for issue in narrative_world_issues(crowded_world)))

zero_delay_world = deepcopy(ws)
for event in zero_delay_world.events:
    event["session"] = {
        "hunt": 0, "defeat_boss": 1, "drop_equipment": 1, "equip_item": 2,
    }[event["type"]]
ck("蓝图声明的同 session 零延迟因果对按一个不可拆组件计数",
   narrative_world_issues(zero_delay_world) == [])

illicit_chain_world = deepcopy(ws)
for index, event in enumerate(illicit_chain_world.events):
    event["session"] = 1
    if index:
        event["caused_by"] = illicit_chain_world.events[index - 1]["id"]
    else:
        event.pop("caused_by", None)
ck("未由蓝图声明的 caused_by 不能绕过跨 session 节奏门禁",
   any("至少需" in issue for issue in narrative_world_issues(illicit_chain_world)))

sample_defects = [
    {"type": "monotonic"},
    {"type": "fake_evolving"},
    {"type": "monotonic_violation"},
    {"type": "out_of_range"},
    {"type": "illegal_transition"},
]
ck("游戏叙事只放行两类题型整形缺陷",
   [item["type"] for item in _blocking_world_defects(sample_defects, True)] ==
   ["monotonic_violation", "out_of_range", "illegal_transition"])
ck("非叙事世界也不为端点极值强改真值，声明约束仍阻塞",
   _blocking_world_defects(sample_defects, False) == sample_defects[1:])


class _BuildTracer:
    """为 build_world 返回固定 world/story，并记录解耦后的调用。"""

    def __init__(self, bad_story_rounds: int = 0, always_bad: bool = False,
                 bad_review_rounds: int = 0) -> None:
        self.bad_story_rounds = bad_story_rounds
        self.always_bad = always_bad
        self.bad_review_rounds = bad_review_rounds
        self.calls: list[tuple[str, str]] = []
        self.story_calls = 0
        self.review_calls = 0

    def chat_json(self, tag, messages, **_kwargs):
        text = "\n".join(str(message.get("content", "")) for message in messages)
        self.calls.append((tag, text))
        if tag == "world.batch":
            if "type id=player" in text:
                entity = {"name": "艾尔文", "type": "player", "fields": {
                    "任务状态": {"type": "stable", "value": "待命"}}}
            elif "type id=boss" in text:
                entity = {"name": "霜狼", "type": "boss", "fields": {
                    "存活状态": {"type": "stable", "value": "存活"}}}
            elif "type id=equipment" in text:
                entity = {"name": "霜牙", "type": "equipment", "fields": {
                    "装备状态": {"type": "stable", "value": "未掉落"}}}
            else:
                raise AssertionError(f"未识别 entity type prompt:{text[:120]}")
            return {"entities": [entity]}
        if tag == "world.structure":
            return {"relations": deepcopy(ws.relations), "events": deepcopy(ws.events)}
        if tag == "world.story":
            self.story_calls += 1
            proposal = make_proposal()
            if self.always_bad or self.story_calls <= self.bad_story_rounds:
                proposal["scenes"][0]["event_refs"] = ["evt-ghost"]
            return proposal
        if tag == "narrative.review":
            self.review_calls += 1
            return {"unsupported_claims": (
                ["虚构了不存在的复活"] if self.review_calls <= self.bad_review_rounds else [])}
        if tag == "world.repair":
            return {"fields": {}}
        raise AssertionError(f"未预期 tracer 调用:{tag}")


build_wp = {
    "world_blueprint": deepcopy(ws.world_blueprint),
    "shared_world_spec": {"entities": {"count": 3}, "timeline": {"n_sessions": 4}},
    "domain_profile": {},
}

plain_tracer = _BuildTracer()
plain_world = build_world(deepcopy(build_wp), plain_tracer, log=lambda *_args: None)
ck("非 narrative 调用数与行为不变",
   [tag for tag, _ in plain_tracer.calls] ==
   ["world.batch", "world.batch", "world.batch", "world.structure"]
   and plain_world.narrative == {})
ck("world.structure prompt 不再夹带 Story",
   all("剧情编排器" not in text for tag, text in plain_tracer.calls
       if tag == "world.structure"))

story_tracer = _BuildTracer()
story_world = build_world(deepcopy(build_wp), story_tracer, log=lambda *_args: None, narrative=True)
ck("narrative 在冻结 world 后只增加一次独立 story 调用",
   [tag for tag, _ in story_tracer.calls] ==
   ["world.batch", "world.batch", "world.batch", "world.structure", "world.story",
    "narrative.review"])
ck("Story prompt 只进入 world.story",
   all(("剧情编排器" in text) == (tag == "world.story")
       for tag, text in story_tracer.calls))
ck("独立调用落盘合法轻 Ledger",
   validate_story_ledger(story_world, story_world.narrative) == [])

semantic_retry_tracer = _BuildTracer(bad_review_rounds=1)
semantic_world = build_world(deepcopy(build_wp), semantic_retry_tracer,
                             log=lambda *_args: None, narrative=True)
semantic_tags = [tag for tag, _ in semantic_retry_tracer.calls]
ck("Story 无依据断言只重试 Story 与只读 reviewer",
   semantic_tags.count("world.structure") == 1
   and semantic_tags.count("world.story") == 2
   and semantic_tags.count("narrative.review") == 2
   and validate_story_ledger(semantic_world, semantic_world.narrative) == [])

retry_tracer = _BuildTracer(bad_story_rounds=2)
retried_world = build_world(deepcopy(build_wp), retry_tracer,
                            log=lambda *_args: None, narrative=True)
retry_tags = [tag for tag, _ in retry_tracer.calls]
ck("坏 Story 只重试 Story，不重跑合法 structure",
   retry_tags.count("world.structure") == 1 and retry_tags.count("world.story") == 3
   and retry_tags.count("narrative.review") == 1
   and validate_story_ledger(retried_world, retried_world.narrative) == [])
ck("Story 重试携带上轮错误反馈",
   "上轮 Story Ledger 机械校验失败" in
   [text for tag, text in retry_tracer.calls if tag == "world.story"][1])

failed_tracer = _BuildTracer(always_bad=True)
try:
    build_world(deepcopy(build_wp), failed_tracer, log=lambda *_args: None, narrative=True)
except WorldBlueprintError:
    failed_closed = True
else:
    failed_closed = False
failed_tags = [tag for tag, _ in failed_tracer.calls]
ck("Story 四轮耗尽后 fail-closed，仍不重跑 structure",
   failed_closed and failed_tags.count("world.structure") == 1
   and failed_tags.count("world.story") == 4
   and failed_tags.count("narrative.review") == 0)


passed = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"\nStory Ledger: {passed}/{len(checks)} PASS")
if passed != len(checks):
    raise SystemExit(1)
