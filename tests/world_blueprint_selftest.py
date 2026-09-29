"""WorldBlueprint v1 离线自检。

锁住四件事：
1. Game / Office 的白皮书在结构签名上确实不同，而不只是换名词；
2. 旧白皮书可以规范化为单类型 blueprint；显式但非法的 blueprint 必须 fail-closed；
3. world_gen 真正消费实体类型、关系、事件与时间节律，并把关系/事件编译进真值世界；
4. 类型与事件元数据可以随 WorldState 落盘、回读。

运行：./venv/bin/python tests/world_blueprint_selftest.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.world_blueprint import (
    WorldBlueprintError,
    normalize_world_blueprint,
    relation_capacity,
    relation_owner_side,
    repair_blueprint_candidate,
    structure_signature,
    validate_world_blueprint,
)
from pipeline.central_office import (_assemble_whitepaper, _map_issues, _observed_blueprint_issues,
                                     _remove_inferred_identity_fields,
                                     _repair_candidate_blueprint,
                                     _separate_observed_relation_fields, central_office)
from pipeline.lines import prepare_lines
from pipeline.lines.L2_relational import RelationalLine
from pipeline.lines.L4_preference import PreferenceLine
from pipeline.prompts import render as render_prompt
from pipeline.render import (_corpus_system, _filler_system,
                             _missing_event_narratives, _session_facts,
                             _tracked_blocklist)
from pipeline.world_gen import (WORLD_JSON_MAX_TOKENS,
                                _canonicalize_generated_field_names,
                                _partition_structure_driven_defects,
                                _wire_declared_causality, build_world,
                                imprint_structure)
from pipeline.world_state import (WorldState, Timeline, Op, SET, UPDATE, EXPIRE,
                                  assemble_world, _date_of)


checks: list[tuple[bool, str]] = []


def ck(name: str, cond) -> None:
    checks.append((bool(cond), name))


ck("复杂世界 JSON 调用预算覆盖真实 reasoning 截断阈值",
   WORLD_JSON_MAX_TOKENS == 16_384)


def _game_whitepaper() -> dict:
    return {
        "domain_profile": {"entity_noun": "游戏对象", "field_schema": []},
        "shared_world_spec": {"entities": {"count": 10}, "timeline": {"n_sessions": 8}},
        "world_blueprint": {
            "entity_types": [
                {"id": "player", "noun": "玩家", "count": 1, "primary": True,
                 "fields": [{"name": "level", "kind": "numeric", "monotonic": "up"},
                            {"name": "equipped_item", "kind": "reference"},
                            {"name": "active_quest", "kind": "reference"}]},
                {"id": "boss", "noun": "Boss", "count": 2,
                 "fields": [{"name": "defeat_status", "kind": "status"},
                            {"name": "drop_item", "kind": "reference"}]},
                {"id": "equipment", "noun": "装备", "count": 4,
                 "fields": [{"name": "enhance_level", "kind": "numeric"},
                            {"name": "acquisition_status", "kind": "status",
                             "states": ["unowned", "owned"]}]},
                {"id": "quest", "noun": "任务", "count": 3,
                 "fields": [{"name": "quest_status", "kind": "status"},
                            {"name": "target_boss", "kind": "reference"}]},
            ],
            "relation_types": [
                {"id": "equips", "from_type": "player", "to_type": "equipment",
                 "field": "equipped_item", "temporal": True, "min_count": 1},
                {"id": "attempts", "from_type": "player", "to_type": "quest",
                 "field": "active_quest", "temporal": True, "min_count": 1},
                {"id": "targets", "from_type": "quest", "to_type": "boss",
                 "field": "target_boss", "temporal": False, "min_count": 1},
                {"id": "drops", "from_type": "boss", "to_type": "equipment",
                 "field": "drop_item", "temporal": False, "min_count": 1},
            ],
            "event_types": [
                {"id": "defeat_boss", "label": "击败首领", "roles": {"actor": "player", "target": "boss"},
                 "effect_fields": [{"role": "target", "field": "defeat_status"}], "min_count": 1},
                {"id": "acquire_item", "label": "获得装备", "roles": {"owner": "player", "item": "equipment"},
                 "effect_fields": [{"role": "item", "field": "acquisition_status"}], "min_count": 1},
                {"id": "complete_quest", "label": "完成任务", "roles": {"actor": "player", "quest": "quest"},
                 "effect_fields": [{"role": "quest", "field": "quest_status"}], "min_count": 1},
            ],
            "causal_rules": [
                {"id": "loot_after_defeat", "trigger_event": "defeat_boss",
                 "effect_event": "acquire_item", "delay_sessions": 0},
                {"id": "quest_after_loot", "trigger_event": "acquire_item",
                 "effect_event": "complete_quest", "delay_sessions": 1},
            ],
            "temporal_model": {"unit": "chapter", "n_sessions": 8, "cadence": "per_chapter", "step_days": 7},
            "evidence_channels": ["任务日志", "战利品记录"],
        },
    }


def _office_whitepaper() -> dict:
    return {
        "domain_profile": {"entity_noun": "办公对象", "field_schema": []},
        "shared_world_spec": {"entities": {"count": 15}, "timeline": {"n_sessions": 12}},
        "world_blueprint": {
            "entity_types": [
                {"id": "employee", "noun": "员工", "count": 5, "primary": True,
                 "fields": [{"name": "employment_status", "kind": "status"},
                            {"name": "employee_department", "kind": "reference"},
                            {"name": "transfer_status", "kind": "status",
                             "states": ["stable", "transferred"]}]},
                {"id": "department", "noun": "部门", "count": 2,
                 "fields": [{"name": "budget_owner", "kind": "person"},
                            {"name": "budget", "kind": "reference"}]},
                {"id": "project", "noun": "项目", "count": 3,
                 "fields": [{"name": "project_status", "kind": "status"},
                            {"name": "owner", "kind": "reference"},
                            {"name": "milestone", "kind": "reference"},
                            {"name": "project_department", "kind": "reference"}]},
                {"id": "milestone", "noun": "里程碑", "count": 3,
                 "fields": [{"name": "milestone_status", "kind": "status"}]},
                {"id": "budget", "noun": "预算", "count": 2,
                 "fields": [{"name": "approved_amount", "kind": "numeric", "unit": "万元"}]},
            ],
            "relation_types": [
                {"id": "member_of", "from_type": "employee", "to_type": "department",
                 "field": "employee_department", "temporal": True, "min_count": 1},
                {"id": "owns", "from_type": "employee", "to_type": "project",
                 "field": "owner", "temporal": True, "min_count": 1},
                {"id": "has_milestone", "from_type": "project", "to_type": "milestone",
                 "field": "milestone", "temporal": False, "min_count": 1},
                {"id": "controls_budget", "from_type": "department", "to_type": "budget",
                 "field": "budget", "temporal": True, "min_count": 1},
                {"id": "belongs_to", "from_type": "project", "to_type": "department",
                 "field": "project_department", "temporal": True, "min_count": 1},
            ],
            "event_types": [
                {"id": "employee_transfer", "label": "员工调动", "roles": {"employee": "employee", "to": "department"},
                 "effect_fields": [{"role": "employee", "field": "transfer_status"}], "min_count": 1},
                {"id": "budget_revision", "label": "预算修订", "roles": {"budget": "budget"},
                 "effect_fields": [{"role": "budget", "field": "approved_amount"}], "min_count": 1},
                {"id": "milestone_complete", "label": "里程碑完成", "roles": {"milestone": "milestone"},
                 "effect_fields": [{"role": "milestone", "field": "milestone_status"}], "min_count": 1},
                {"id": "project_delay", "label": "项目延期", "roles": {"project": "project"},
                 "effect_fields": [{"role": "project", "field": "project_status"}], "min_count": 1},
            ],
            "causal_rules": [
                {"id": "budget_causes_delay", "trigger_event": "budget_revision",
                 "effect_event": "project_delay", "delay_sessions": 1},
            ],
            "temporal_model": {"unit": "business_day", "n_sessions": 12, "cadence": "event_driven", "step_days": 1},
            "evidence_channels": ["人事通知", "预算审批单", "项目周报"],
        },
    }


def _rename_vocabulary(wp: dict) -> dict:
    """一致改名所有领域词；结构签名必须保持不变。"""
    out = deepcopy(wp)
    bp = out["world_blueprint"]
    type_ids = {t["id"]: f"type_{i}" for i, t in enumerate(bp["entity_types"])}
    event_ids = {e["id"]: f"event_{i}" for i, e in enumerate(bp["event_types"])}
    field_ids: dict[tuple[str, str], str] = {}
    for i, t in enumerate(bp["entity_types"]):
        old_type = t["id"]
        t["id"], t["noun"] = type_ids[old_type], f"noun_{i}"
        for j, field in enumerate(t["fields"]):
            old_field = field["name"]
            field["name"] = f"field_{i}_{j}"
            field_ids[(old_type, old_field)] = field["name"]
    for i, rel in enumerate(bp["relation_types"]):
        old_source, old_target = rel["from_type"], rel["to_type"]
        owner = old_source if (old_source, rel["field"]) in field_ids else old_target
        rel["id"] = f"relation_{i}"
        rel["from_type"], rel["to_type"] = type_ids[old_source], type_ids[old_target]
        rel["field"] = field_ids[(owner, rel["field"])]
    for event in bp["event_types"]:
        old_event = event["id"]
        old_roles = dict(event["roles"])
        event["id"] = event_ids[old_event]
        event["roles"] = {role: type_ids[tid] for role, tid in old_roles.items()}
        for effect in event["effect_fields"]:
            owner = old_roles[effect["role"]]
            effect["field"] = field_ids[(owner, effect["field"])]
    for i, rule in enumerate(bp["causal_rules"]):
        rule["id"] = f"causal_{i}"
        rule["trigger_event"] = event_ids[rule["trigger_event"]]
        rule["effect_event"] = event_ids[rule["effect_event"]]
    return out


def _views_with_blueprint(wp: dict) -> dict:
    """给代码综合器一个最小但完整的七视角结果；不调用网络。"""
    return {
        "observe": {"main_entity_noun": "不应成为真源", "observed_media": ["样例记录"],
                    "observed_entities": ["扁平旧主体"], "observed_fields": []},
        "skeptic": {},
        "map": {"per_line": [{"line": "L1_timeline", "applicable": True,
                                "instantiation": "按蓝图时间演化", "weight_hint": 0.5}]},
        "medium": {"recommended_mix": ["领域记录"]},
        "style": {"style_spec": {"tone": "客观"}},
        "traps": {"traps": []},
        "world": {"world_blueprint": deepcopy(wp["world_blueprint"])},
    }


# ════════ ① 白皮书结构差异：不是名词换皮 ════════
game = normalize_world_blueprint(_game_whitepaper())
office = normalize_world_blueprint(_office_whitepaper())
ck("① Game blueprint 合法", validate_world_blueprint(game) == [])
ck("① Office blueprint 合法", validate_world_blueprint(office) == [])
ck("① 一致改名所有领域词不改变结构签名",
   structure_signature(game) == structure_signature(_rename_vocabulary(_game_whitepaper())))
reordered_game = _game_whitepaper()
for key in ("entity_types", "relation_types", "event_types", "causal_rules"):
    reordered_game["world_blueprint"][key] = list(reversed(reordered_game["world_blueprint"][key]))
ck("① 声明数组顺序不改变结构签名",
   structure_signature(game) == structure_signature(reordered_game))
scaled_game = _game_whitepaper()
for index, entity_type in enumerate(scaled_game["world_blueprint"]["entity_types"]):
    entity_type["count"] += 10 + index
scaled_game["world_blueprint"]["temporal_model"]["n_sessions"] = 80
for declaration in (scaled_game["world_blueprint"]["relation_types"]
                    + scaled_game["world_blueprint"]["event_types"]):
    declaration["min_count"] += 7
ck("① 规模旋钮不改变世界骨架签名",
   structure_signature(game) == structure_signature(scaled_game))
ck("① Game / Office 核心本体·拓扑·动力学签名不同",
   structure_signature(game)[:4] != structure_signature(office)[:4])
ck("① 实体类型集合不同", {x["id"] for x in game["entity_types"]}.isdisjoint(
    {x["id"] for x in office["entity_types"]}))
ck("① 事件类型集合不同", {x["id"] for x in game["event_types"]}.isdisjoint(
    {x["id"] for x in office["event_types"]}))
ck("① 时间骨架不同", game["temporal_model"]["unit"] != office["temporal_model"]["unit"]
   and game["temporal_model"]["cadence"] != office["temporal_model"]["cadence"])

game_draft = _assemble_whitepaper(_views_with_blueprint(_game_whitepaper()), "游戏")
office_draft = _assemble_whitepaper(_views_with_blueprint(_office_whitepaper()), "办公")
ck("① 白皮书综合器保留 Game / Office 结构差异",
   structure_signature(game_draft) != structure_signature(office_draft))
ck("① blueprint primary 决定主体，不再由扁平 observe 决定",
   game_draft["domain_profile"]["entity_noun"] == "玩家"
   and office_draft["domain_profile"]["entity_noun"] == "员工")
ck("① blueprint 决定实体总数与时间制度",
   game_draft["shared_world_spec"]["entities"]["count"] == 10
   and game_draft["shared_world_spec"]["timeline"]["n_sessions"] == 8
   and game_draft["medium"]["cadence"] == "per_chapter")
missing_field_views = _views_with_blueprint(_game_whitepaper())
missing_field_views["observe"]["observed_fields"] = [{"name": "few-shot明确字段", "kind": "status"}]
try:
    _assemble_whitepaper(missing_field_views, "游戏")
    missing_field_rejected = False
except ValueError:
    missing_field_rejected = True
ck("① blueprint 漏掉 few-shot 已观测字段时白皮书 fail-closed", missing_field_rejected)
wrong_shape_views = _views_with_blueprint(_game_whitepaper())
wrong_shape_views["observe"]["observed_fields"] = [{"name": "level", "kind": "numeric", "unit": "级"}]
try:
    _assemble_whitepaper(wrong_shape_views, "游戏")
    wrong_shape_rejected = False
except ValueError:
    wrong_shape_rejected = True
ck("① blueprint 不得只同名覆盖、却改写 few-shot 的 kind/unit/range", wrong_shape_rejected)

sentinel_observed = {
    "observed_fields": [
        {"name": "level", "kind": "numeric", "unit": "可省", "monotonic": "不适用",
         "range": [None, ""]},
        {"name": "无", "kind": "未知"},
        None,
    ],
}
ck("① observe 可选占位不会伪装成 blueprint 硬冲突",
   _observed_blueprint_issues(game, sentinel_observed) == [])
ck("① observe 半空 range 仍 fail-closed",
   bool(_observed_blueprint_issues(game, {
       "observed_fields": [{"name": "level", "kind": "numeric", "range": [None, 100]}],
   })))
ck("① observe 畸形字段行返回明确 schema 问题而不崩溃",
   _observed_blueprint_issues(game, {"observed_fields": [42]})
   == ["observe.observed_fields[0] 必须是 object"])
blank_evidence = _game_whitepaper()["world_blueprint"]
blank_evidence["evidence_channels"] = ["", "无"]
repaired_evidence, evidence_repairs = _repair_candidate_blueprint(
    {"world_blueprint": blank_evidence}, {}, ["任务日志", "战斗记录"])
ck("① evidence 由 medium 单一真源投影，不为同一空字段反复调用架构师",
   repaired_evidence["world_blueprint"]["evidence_channels"] == ["任务日志", "战斗记录"]
   and any("单一真源投影" in item for item in evidence_repairs))
clean_media_views = _views_with_blueprint(_game_whitepaper())
clean_media_views["observe"].update({
    "observed_media": ["", "无", "任务日志"],
    "stopped_phrase_seen": "不适用",
})
clean_media_draft = _assemble_whitepaper(clean_media_views, "游戏")
ck("① observe 空媒介不污染 evidence，真实媒介保留",
   "任务日志" in clean_media_draft["world_blueprint"]["evidence_channels"]
   and "" not in clean_media_draft["world_blueprint"]["evidence_channels"]
   and "无" not in clean_media_draft["world_blueprint"]["evidence_channels"])
ck("① observe 停用措辞占位回退默认值",
   clean_media_draft["domain_profile"]["stopped_phrase"] == "停止/失效")


# ════════ ② normalize + fail-closed ════════
legacy_wp = {
    "domain_profile": {"entity_noun": "案件", "field_schema": [{"name": "状态", "kind": "status"}]},
    "shared_world_spec": {"entities": {"count": 7}, "timeline": {"n_sessions": 9}},
    "medium": {"cadence": "weekly"},
}
legacy = normalize_world_blueprint(legacy_wp)
ck("② 旧白皮书规范化为单一实体类型", len(legacy["entity_types"]) == 1)
ck("② legacy 保留实体数和字段", legacy["entity_types"][0]["count"] == 7
   and legacy["entity_types"][0]["fields"][0]["name"] == "状态")
ck("② legacy 保留 session 与 cadence", legacy["temporal_model"]["n_sessions"] == 9
   and legacy["temporal_model"]["cadence"] == "weekly")
ck("② legacy 补齐可选结构", legacy["relation_types"] == [] and legacy["event_types"] == []
   and legacy["causal_rules"] == [])


def _must_reject(name: str, blueprint: dict) -> None:
    rejected = False
    try:
        normalize_world_blueprint({"world_blueprint": blueprint})
    except (TypeError, ValueError):
        rejected = True
    ck(name, rejected)


_must_reject("② 显式空 blueprint 不准静默降级 legacy", {})
_must_reject("② 重复 entity type id fail-closed", {
    "entity_types": [
        {"id": "x", "noun": "甲", "count": 1, "primary": True, "fields": []},
        {"id": "x", "noun": "乙", "count": 1, "fields": []},
    ],
    "relation_types": [], "event_types": [], "causal_rules": [],
    "temporal_model": {"unit": "week", "n_sessions": 4, "cadence": "weekly", "step_days": 7},
})
_must_reject("② 关系端点引用不存在类型 fail-closed", {
    "entity_types": [{"id": "x", "noun": "甲", "count": 1, "primary": True, "fields": []}],
    "relation_types": [{"id": "bad", "from_type": "x", "to_type": "ghost",
                        "field": "link", "temporal": True, "min_count": 1}],
    "event_types": [], "causal_rules": [],
    "temporal_model": {"unit": "week", "n_sessions": 4, "cadence": "weekly"},
})

same_field_bp = {
    "entity_types": [
        {"id": "a", "noun": "甲", "count": 1, "primary": True,
         "fields": [{"name": "状态", "kind": "status"}, {"name": "关联乙", "kind": "reference"}]},
        {"id": "b", "noun": "乙", "count": 1,
         "fields": [{"name": "状态", "kind": "status"}]},
    ],
    "relation_types": [{"id": "links", "from_type": "a", "to_type": "b",
                         "field": "关联乙", "temporal": False, "min_count": 1}],
    "event_types": [{"id": "changes", "label": "状态变更", "roles": {"subject": "b"},
                     "effect_fields": [{"role": "subject", "field": "状态"}], "min_count": 1}],
    "causal_rules": [], "evidence_channels": ["记录"],
    "temporal_model": {"unit": "week", "n_sessions": 4, "cadence": "weekly", "step_days": 7},
}
ck("② 跨类型同名字段约束一致则允许", validate_world_blueprint(
    normalize_world_blueprint(same_field_bp)) == [])

reference_only_bp = deepcopy(same_field_bp)
reference_only_bp["entity_types"][1]["fields"] = []
reference_only_bp["event_types"][0] = {
    "id": "changes", "label": "状态变更", "roles": {"subject": "a"},
    "effect_fields": [{"role": "subject", "field": "状态"}], "min_count": 1,
}
ck("② 只作为关系目标的类型允许零领域字段",
   validate_world_blueprint(normalize_world_blueprint(reference_only_bp)) == [])

orphan_zero_field_bp = deepcopy(reference_only_bp)
orphan_zero_field_bp["relation_types"] = []
ck("② 零字段且不参与关系的孤立类型仍 fail-closed",
   any("零字段时必须是 relation endpoint" in issue
       for issue in validate_world_blueprint(orphan_zero_field_bp)))

identity_wrapper = {"world_blueprint": deepcopy(same_field_bp)}
identity_wrapper["world_blueprint"]["entity_types"][1]["fields"].append(
    {"name": "名称", "kind": "text"})
identity_repairs = _remove_inferred_identity_fields(identity_wrapper, {"observed_fields": []})
ck("② 未观察且未引用的名称字段由 canonical name 去重",
   bool(identity_repairs)
   and not any(field.get("name") == "名称"
               for field in identity_wrapper["world_blueprint"]["entity_types"][1]["fields"]))

observed_identity_wrapper = {"world_blueprint": deepcopy(same_field_bp)}
observed_identity_wrapper["world_blueprint"]["entity_types"][1]["fields"].append(
    {"name": "名称", "kind": "text"})
observed_identity_repairs = _remove_inferred_identity_fields(
    observed_identity_wrapper, {"observed_fields": [{"name": "名称", "kind": "text"}]})
ck("② few-shot 明示的身份字段仍是硬事实",
   not observed_identity_repairs
   and any(field.get("name") == "名称"
           for field in observed_identity_wrapper["world_blueprint"]["entity_types"][1]["fields"]))

reverse_owner_bp = deepcopy(same_field_bp)
reverse_owner_bp["entity_types"][1]["fields"].append(
    {"name": "关联甲", "kind": "reference", "unit": "entity_id"})
reverse_owner_bp["relation_types"][0]["field"] = "关联甲"
reverse_owner_bp["entity_types"][0]["fields"] = [
    field for field in reverse_owner_bp["entity_types"][0]["fields"]
    if field["name"] != "关联乙"
]
ck("② relation 字段只属于 to_type 时合法",
   validate_world_blueprint(normalize_world_blueprint(reverse_owner_bp)) == [])
ck("② relation owner 只在两个端点局部推断",
   relation_owner_side(reverse_owner_bp, reverse_owner_bp["relation_types"][0]) == "to")

both_endpoints_own_bp = deepcopy(same_field_bp)
both_endpoints_own_bp["entity_types"][1]["fields"].append(
    {"name": "关联乙", "kind": "reference"})
_must_reject("② relation 字段同时属于异类型两端 fail-closed", both_endpoints_own_bp)

neither_endpoint_owns_bp = deepcopy(same_field_bp)
neither_endpoint_owns_bp["relation_types"][0]["field"] = "未声明关联"
_must_reject("② relation 字段不属于异类型任一端 fail-closed", neither_endpoint_owns_bp)

non_reference_owner_bp = deepcopy(same_field_bp)
non_reference_owner_bp["entity_types"][0]["fields"][1]["kind"] = "category"
_must_reject("② relation owner 字段必须为 reference", non_reference_owner_bp)

schema_foreign_key_bp = deepcopy(same_field_bp)
schema_foreign_key_bp["entity_types"][0]["fields"][1]["target_type"] = "b"
_must_reject("② reference 目标不得用 schema 外键重复表达", schema_foreign_key_bp)

dangling_reference_bp = deepcopy(same_field_bp)
dangling_reference_bp["entity_types"][1]["fields"].append(
    {"name": "悬空引用", "kind": "reference"})
_must_reject("② reference 字段必须绑定且只绑定一个 relation", dangling_reference_bp)

# 模型修理 relation.field 时容易在“零端/两端/悬空”之间振荡；这些纯机械问题
# 在进入 validator 前一次性、确定性收敛，不消费新的 LLM 重试。
none_owner_bp = deepcopy(same_field_bp)
none_owner_bp["entity_types"][0]["count"] = 1
none_owner_bp["entity_types"][1]["count"] = 4
none_owner_bp["entity_types"][0]["fields"] = [
    field for field in none_owner_bp["entity_types"][0]["fields"]
    if field["name"] != "关联乙"
]
none_owner_bp["entity_types"][0]["fields"].append(
    {"name": "无关引用", "kind": "reference"})
none_owner_bp["relation_types"][0]["min_count"] = 0
none_owner_bp["event_types"][0]["min_count"] = "invalid"
none_owner_before = deepcopy(none_owner_bp)
none_owner_fixed, none_owner_repairs = repair_blueprint_candidate(none_owner_bp)
ck("② FK 归一是纯函数且零端时创建在实体数更多的一端",
   none_owner_bp == none_owner_before
   and relation_owner_side(none_owner_fixed, none_owner_fixed["relation_types"][0]) == "to"
   and any(field.get("name") == "关联乙" and field.get("kind") == "reference"
           for field in none_owner_fixed["entity_types"][1]["fields"])
   and bool(none_owner_repairs))
ck("② FK 归一删除所有未绑定 reference 并把非法 min_count 修为一",
   not any(field.get("name") == "无关引用"
           for field in none_owner_fixed["entity_types"][0]["fields"])
   and none_owner_fixed["relation_types"][0]["min_count"] == 1
   and none_owner_fixed["event_types"][0]["min_count"] == 1
   and len(none_owner_fixed["relation_types"]) == len(none_owner_bp["relation_types"])
   and len(none_owner_fixed["event_types"]) == len(none_owner_bp["event_types"])
   and validate_world_blueprint(normalize_world_blueprint(none_owner_fixed)) == [])

# game__20260906-060616 第六轮的真实失败形态：四条 FK 同时出现在两端，
# player 还残留两个与自身无关的 reference。数量与关系方向保持该 Run 原样。
smoke_fk_bp = {
    "version": 1,
    "entity_types": [
        {"id": "player_character", "noun": "玩家角色", "count": 1, "primary": True,
         "fields": [{"name": "等级", "kind": "numeric"},
                    {"name": "控制阵营", "kind": "reference"},
                    {"name": "拥有者", "kind": "reference"}]},
        {"id": "region", "noun": "地区", "count": 10,
         "fields": [{"name": "解锁状态", "kind": "status"},
                    {"name": "目标地区", "kind": "reference"}]},
        {"id": "quest", "noun": "任务", "count": 20,
         "fields": [{"name": "任务状态", "kind": "status"},
                    {"name": "目标地区", "kind": "reference"},
                    {"name": "目标首领", "kind": "reference"}]},
        {"id": "faction", "noun": "阵营", "count": 5,
         "fields": [{"name": "阵营名称", "kind": "text"},
                    {"name": "控制阵营", "kind": "reference"}]},
        {"id": "boss", "noun": "首领", "count": 8,
         "fields": [{"name": "存活状态", "kind": "status"},
                    {"name": "目标首领", "kind": "reference"},
                    {"name": "掉落来源", "kind": "reference"}]},
        {"id": "equipment", "noun": "装备", "count": 30,
         "fields": [{"name": "流转状态", "kind": "status"},
                    {"name": "掉落来源", "kind": "reference"},
                    {"name": "拥有者", "kind": "reference"}]},
    ],
    "relation_types": [
        {"id": "faction_controls_region", "from_type": "faction", "to_type": "region",
         "field": "控制阵营", "temporal": True, "min_count": 0},
        {"id": "quest_targets_region", "from_type": "quest", "to_type": "region",
         "field": "目标地区", "temporal": False, "min_count": 1},
        {"id": "quest_targets_boss", "from_type": "quest", "to_type": "boss",
         "field": "目标首领", "temporal": False, "min_count": 1},
        {"id": "boss_drops_equipment", "from_type": "boss", "to_type": "equipment",
         "field": "掉落来源", "temporal": True, "min_count": 1},
        {"id": "player_owns_equipment", "from_type": "player_character", "to_type": "equipment",
         "field": "拥有者", "temporal": True, "min_count": 1},
    ],
    "event_types": [{
        "id": "explore_region", "label": "探索地区",
        "roles": {"actor": "player_character", "target": "region"},
        "effect_fields": [{"role": "target", "field": "解锁状态"}], "min_count": 0,
    }],
    "causal_rules": [], "evidence_channels": ["剧情日志"],
    "temporal_model": {"unit": "chapter", "cadence": "per_encounter",
                       "n_sessions": 10, "step_days": 1},
}
smoke_fk_fixed, _ = repair_blueprint_candidate(smoke_fk_bp)
smoke_owner_sides = [
    relation_owner_side(smoke_fk_fixed, relation)
    for relation in smoke_fk_fixed["relation_types"]
]
ck("② 真实 smoke 的双端 FK 按多端收敛、单端 FK 保持原位",
   smoke_owner_sides == ["from", "from", "from", "to", "to"]
   and validate_world_blueprint(smoke_fk_fixed) == [])
ck("② 真实 smoke 的跨类型悬空 reference 被清除且声明不丢失",
   [field["name"] for field in smoke_fk_fixed["entity_types"][0]["fields"]] == ["等级"]
   and len(smoke_fk_fixed["relation_types"]) == 5
   and len(smoke_fk_fixed["event_types"]) == 1
   and all(item["min_count"] >= 1 for item in
           smoke_fk_fixed["relation_types"] + smoke_fk_fixed["event_types"]))

idempotent_fixed, idempotent_repairs = repair_blueprint_candidate(none_owner_fixed)
ck("② FK/min_count 归一幂等",
   idempotent_fixed == none_owner_fixed and idempotent_repairs == [])

multi_role_bp = deepcopy(same_field_bp)
multi_role_bp["event_types"][0].update({
    "roles": {"old_subject": "b", "new_subject": "b"},
    "effect_fields": [{"role": "new_subject", "field": "状态"}],
})
multi_role_fixed, multi_role_repairs = repair_blueprint_candidate(multi_role_bp)
ck("② 同类型多事件角色确定性扩容到互异实体容量",
   next(item for item in multi_role_fixed["entity_types"] if item["id"] == "b")["count"] == 2
   and any("event 角色互异容量" in item for item in multi_role_repairs)
   and validate_world_blueprint(normalize_world_blueprint(multi_role_fixed)) == [])

reused_reference_bp = deepcopy(same_field_bp)
reused_reference_bp["relation_types"].append({
    "id": "links_again", "from_type": "a", "to_type": "b",
    "field": "关联乙", "temporal": True, "min_count": 1,
})
_must_reject("② 多个 relation type 不得复用同一 owner 字段", reused_reference_bp)

event_writes_relation_bp = deepcopy(same_field_bp)
event_writes_relation_bp["event_types"][0].update({
    "roles": {"subject": "a"},
    "effect_fields": [{"role": "subject", "field": "关联乙"}],
})
_must_reject("② event effect 不得写 relation-owned reference 字段", event_writes_relation_bp)

event_writes_identity_bp = deepcopy(same_field_bp)
event_writes_identity_bp["entity_types"][1]["fields"].append(
    {"name": "对象名称", "kind": "category"})
event_writes_identity_bp["event_types"][0]["effect_fields"] = [
    {"role": "subject", "field": "对象名称"}]
_must_reject("② event effect 不得把实体专名重复写入名称字段", event_writes_identity_bp)

observed_relation_bp = deepcopy(same_field_bp)
observed_relation_bp["entity_types"][0]["fields"] = [
    field for field in observed_relation_bp["entity_types"][0]["fields"]
    if field["name"] != "关联乙"
] + [{"name": "所属阵营", "kind": "reference"}]
observed_relation_bp["entity_types"].append({
    "id": "c", "noun": "丙", "count": 1, "primary": False,
    "fields": [{"name": "所属阵营", "kind": "category"}],
})
observed_relation_bp["relation_types"][0]["field"] = "所属阵营"
observed_relation_wrapper = {"world_blueprint": observed_relation_bp}
observed_relation_repairs = _separate_observed_relation_fields(
    observed_relation_wrapper,
    {"observed_fields": [{"name": "所属阵营", "kind": "category"}]},
)
observed_relation_fixed = observed_relation_wrapper["world_blueprint"]
ck("② observed 显示字段与同名 relation FK 确定性拆分",
   bool(observed_relation_repairs)
   and next(field for field in observed_relation_fixed["entity_types"][0]["fields"]
            if field["name"] == "所属阵营")["kind"] == "category"
   and observed_relation_fixed["relation_types"][0]["field"] == "所属阵营引用"
   and validate_world_blueprint(normalize_world_blueprint(observed_relation_fixed)) == [])

observed_both_bp = deepcopy(same_field_bp)
observed_both_bp["entity_types"][0]["fields"] = [
    field for field in observed_both_bp["entity_types"][0]["fields"]
    if field["name"] != "关联乙"
] + [{"name": "所属阵营", "kind": "reference"}]
observed_both_bp["entity_types"][1]["fields"].append(
    {"name": "所属阵营", "kind": "reference"})
observed_both_bp["relation_types"][0]["field"] = "所属阵营"
observed_both_wrapper = {"world_blueprint": observed_both_bp}
observed_both_before = deepcopy(observed_both_wrapper)
observed_both_fixed, _ = _repair_candidate_blueprint(
    observed_both_wrapper,
    {"observed_fields": [{"name": "所属阵营", "kind": "category"}]},
)
ck("② observed 拆分后再归一，双端冲突不会删除展示硬事实",
   observed_both_wrapper == observed_both_before
   and observed_both_fixed["world_blueprint"]["relation_types"][0]["field"] == "所属阵营引用"
   and any(field.get("name") == "所属阵营" and field.get("kind") == "category"
           for entity_type in observed_both_fixed["world_blueprint"]["entity_types"]
           for field in entity_type["fields"])
   and validate_world_blueprint(normalize_world_blueprint(observed_both_fixed)) == [])

static_over_capacity_bp = deepcopy(reverse_owner_bp)
static_over_capacity_bp["relation_types"][0]["min_count"] = 2
_must_reject("② static 标量 FK min_count 不得超过 owner 数量", static_over_capacity_bp)
temporal_singleton_bp = deepcopy(static_over_capacity_bp)
temporal_singleton_bp["relation_types"][0]["temporal"] = True
_must_reject("② temporal FK 只有一个可引用实体时不能用重复 no-op 凑 min_count", temporal_singleton_bp)
temporal_alternating_bp = deepcopy(temporal_singleton_bp)
temporal_alternating_bp["entity_types"][0]["count"] = 2
ck("② temporal FK 有两个可引用实体时可跨 session 形成变化",
   relation_capacity(temporal_alternating_bp, temporal_alternating_bp["relation_types"][0]) == 4
   and validate_world_blueprint(normalize_world_blueprint(temporal_alternating_bp)) == [])

causal_out_of_window_bp = deepcopy(same_field_bp)
causal_out_of_window_bp["causal_rules"] = [{
    "id": "too_late", "trigger_event": "changes", "effect_event": "changes",
    "delay_sessions": 4,
}]
_must_reject("② causal delay 必须落在时间窗内", causal_out_of_window_bp)

self_relation_bp = deepcopy(same_field_bp)
self_relation_bp["relation_types"][0].update({"from_type": "a", "to_type": "a"})
ck("② 自关系字段按 source 侧持有",
   validate_world_blueprint(normalize_world_blueprint(self_relation_bp)) == [])

source_signature_bp = deepcopy(same_field_bp)
source_relation_shape = structure_signature(source_signature_bp)[1][0]
target_relation_shape = structure_signature(reverse_owner_bp)[1][0]
ck("② 结构签名编码 relation owner_side 与实际 owner 字段形状",
   source_relation_shape[3] == "from"
   and source_relation_shape[4][1] is False
   and target_relation_shape[3] == "to"
   and target_relation_shape[4][1] is True
   and structure_signature(source_signature_bp) != structure_signature(reverse_owner_bp))

conflicting_field_bp = deepcopy(same_field_bp)
conflicting_field_bp["entity_types"][1]["fields"][0]["kind"] = "numeric"
_must_reject("② 跨类型同名字段约束冲突 fail-closed", conflicting_field_bp)
string_boolean_bp = deepcopy(same_field_bp)
string_boolean_bp["entity_types"][0]["primary"] = "false"
string_boolean_bp["relation_types"][0]["temporal"] = "false"
_must_reject("② 显式 blueprint 不把字符串 false 强转为 true", string_boolean_bp)
bad_cardinality_bp = deepcopy(same_field_bp)
bad_cardinality_bp["entity_types"][0]["cardinality_policy"] = "fixed"
_must_reject("② cardinality_policy 只接受 exact（可增长时省略）", bad_cardinality_bp)
negative_count_bp = deepcopy(same_field_bp)
negative_count_bp["relation_types"][0]["min_count"] = -3
_must_reject("② 显式 blueprint 不把负 min_count 钳成合法值", negative_count_bp)
decorative_structure_bp = deepcopy(same_field_bp)
decorative_structure_bp["relation_types"][0]["min_count"] = 0
decorative_structure_bp["event_types"][0]["min_count"] = 0
_must_reject("② 关系与事件不能只声明类型却全部零实例", decorative_structure_bp)
string_range_bp = deepcopy(same_field_bp)
for entity_type in string_range_bp["entity_types"]:
    entity_type["fields"][0] = {"name": "状态", "kind": "numeric", "range": ["0", "10"]}
_must_reject("② range 端点必须是真 JSON 数字，不宽容修复数字字符串", string_range_bp)
prose_invariant_bp = deepcopy(same_field_bp)
prose_invariant_bp["invariants"] = ["状态变化必须合理"]
_must_reject("② 不可执行 prose invariant 不得伪装成已执行契约", prose_invariant_bp)


# ════════ ③ world_gen 真消费 typed entities / relations / events / cadence ════════
build_wp = {
    "domain_profile": {"entity_noun": "游戏对象", "field_schema": []},
    "shared_world_spec": {"entities": {"count": 99}, "timeline": {"n_sessions": 99}},
    "world_blueprint": {
        "entity_types": [
            {"id": "player", "noun": "玩家", "count": 1, "primary": True,
             "fields": [{"name": "level", "kind": "numeric"},
                        {"name": "equipped_item", "kind": "reference"},
                        {"name": "loot_state", "kind": "status"}]},
            {"id": "boss", "noun": "Boss", "count": 1,
             "fields": [{"name": "defeat_status", "kind": "status"}]},
            {"id": "equipment", "noun": "装备", "count": 1,
             "fields": [{"name": "enhance_level", "kind": "numeric"}]},
        ],
        "relation_types": [
            {"id": "equips", "from_type": "player", "to_type": "equipment",
             "field": "equipped_item", "temporal": True, "min_count": 1},
        ],
        "event_types": [
            {"id": "defeat_boss", "label": "击败首领", "roles": {"actor": "player", "target": "boss"},
             "effect_fields": [{"role": "target", "field": "defeat_status"}], "min_count": 1},
            {"id": "acquire_loot", "label": "获得战利品", "roles": {"owner": "player"},
             "effect_fields": [{"role": "owner", "field": "loot_state"}], "min_count": 1},
        ],
        "causal_rules": [
            {"id": "defeat_yields_loot", "trigger_event": "defeat_boss",
             "effect_event": "acquire_loot", "delay_sessions": 1},
        ],
        "temporal_model": {"unit": "chapter", "n_sessions": 4, "cadence": "per_chapter", "step_days": 3},
        "evidence_channels": ["战斗日志"],
    },
}


def _typed_table() -> dict:
    """给 assemble/build 两条路径共用的最小类型化实例表。"""
    return {
        "entities": [
            {"name": "旅者", "type": "player", "fields": {
                "level": {"type": "stable", "value": "9"},
            }},
            {"name": "熔岩巨兽", "type": "boss", "fields": {
                "defeat_status": {"type": "stable", "value": "alive"},
            }},
            {"name": "星铁剑", "type": "equipment", "fields": {
                "enhance_level": {"type": "stable", "value": "2"},
            }},
        ],
        "relations": [
            {"id": "rel-1", "type": "equips", "from": "旅者", "to": "星铁剑", "session": 1},
        ],
        "events": [
            {"id": "evt-1", "type": "defeat_boss", "session": 2,
             "participants": {"actor": "旅者", "target": "熔岩巨兽"},
             "effects": [{"entity": "熔岩巨兽", "field": "defeat_status", "set": "defeated"}]},
            {"id": "evt-2", "type": "acquire_loot", "session": 3, "caused_by": "evt-1",
             "participants": {"owner": "旅者"},
             "effects": [{"entity": "旅者", "field": "loot_state", "set": "acquired"}]},
        ],
        "cascades": [],
        "absent_fields": [],
    }


# 先直接锁住编译器；这样即使 build_world 的 prompt 路由变了，也不会把核心语义测成假绿。
build_bp = normalize_world_blueprint(build_wp)
causal_structure = {"events": [
    {"id": "cause", "type": "defeat_boss", "session": 1},
    {"id": "effect", "type": "acquire_loot", "session": 2},
]}
ck("③ 声明因果在已有事件满足时差时由代码补 caused_by",
   _wire_declared_causality(causal_structure, build_bp) == 1
   and causal_structure["events"][1].get("caused_by") == "cause")
wrong_parent_structure = {"events": [
    {"id": "cause", "type": "defeat_boss", "session": 1},
    {"id": "effect", "type": "acquire_loot", "session": 2, "caused_by": "ghost"},
]}
ck("③ 声明因果由代码校正错误 caused_by，不再交给模型重写",
   _wire_declared_causality(wrong_parent_structure, build_bp) == 1
   and wrong_parent_structure["events"][1].get("caused_by") == "cause")
wrong_delay_structure = {"events": [
    {"id": "cause", "type": "defeat_boss", "session": 0},
    {"id": "effect", "type": "acquire_loot", "session": 3},
]}
ck("③ 因果编译不改 session、不凭空连接错误时差",
   _wire_declared_causality(wrong_delay_structure, build_bp) == 0
   and not wrong_delay_structure["events"][1].get("caused_by"))
compiled, compile_issues = assemble_world(_typed_table(), blueprint=build_bp)
ck("③ assemble 类型化实例零缺陷", compile_issues == [])
ck("③ assemble 保存 entity type", compiled.entity_types == {
    "旅者": "player", "熔岩巨兽": "boss", "星铁剑": "equipment"})
ck("③ assemble 保存 relation / event 实例", len(compiled.relations) == 1 and len(compiled.events) == 2)
ck("③ assemble relation 编译进 source timeline", compiled.timeline("旅者", "equipped_item") is not None
   and compiled.timeline("旅者", "equipped_item").value_at_session(1) == "星铁剑")
ck("③ assemble event effect 编译进 target timeline", compiled.timeline("熔岩巨兽", "defeat_status") is not None
   and compiled.timeline("熔岩巨兽", "defeat_status").value_at_session(2) == "defeated")
ck("③ caused_by 事件对编译为可追溯 cascade", any(
    c.get("kind") == "domain_event_causality" and c.get("rule_id") == "defeat_yields_loot"
    for c in compiled.cascades))
ck("③ assemble 使用 blueprint n_sessions", compiled.n_sessions == 4)
ck("③ 非 weekly 时间制度按 step_days 生成 canonical 日期",
   compiled.timeline("熔岩巨兽", "defeat_status").ops[-1].date == _date_of(2, step_days=3))

unwitnessed_table = _typed_table()
unwitnessed_table["entities"][1]["fields"]["defeat_status"] = {
    "type": "evolving", "trajectory": [
        {"session": 0, "value": "alive"}, {"session": 1, "value": "wounded"},
    ]}
_unwitnessed_ws, unwitnessed_issues = assemble_world(unwitnessed_table, blueprint=build_bp)
ck("③ event-owned 字段的独立变化必须有领域事件见证",
   any("没有 relation/event 见证" in issue for issue in unwitnessed_issues))

numeric_bp = deepcopy(build_bp)
boss_field = next(t for t in numeric_bp["entity_types"] if t["id"] == "boss")["fields"][0]
boss_field.update({"kind": "numeric", "range": [0, 10]})
numeric_table = _typed_table()
numeric_table["entities"][1]["fields"]["defeat_status"] = {"type": "stable", "value": "1"}
numeric_table["events"][0]["effects"][0]["set"] = "banana"
_numeric_ws, numeric_issues = assemble_world(numeric_table, blueprint=numeric_bp)
ck("③ numeric event effect 不可解析时 fail-closed 而非绕过 range",
   any("numeric 字段值不可解析" in issue for issue in numeric_issues))

duplicate_session_table = _typed_table()
duplicate_session_table["entities"][0]["fields"]["level"] = {
    "type": "evolving", "trajectory": [
        {"session": 0, "value": "8"}, {"session": 0, "value": "9"},
    ]}
_duplicate_ws, duplicate_issues = assemble_world(duplicate_session_table, blueprint=build_bp)
ck("③ 同字段同 session 重复值明确拒绝",
   any("trajectory 同 session 重复" in issue for issue in duplicate_issues))

static_bp = deepcopy(build_bp)
static_bp["relation_types"][0]["temporal"] = False
static_table = _typed_table()
static_table["relations"][0]["session"] = 2
_static_ws, static_issues = assemble_world(static_table, blueprint=static_bp)
ck("③ temporal=false 静态关系只能从 session 0 成立",
   any("static(temporal=false)" in issue for issue in static_issues))


class _BlueprintTracer:
    """返回一个最小类型化游戏世界，并记录 world_gen 实际收到的 prompt。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[tuple[str, dict]] = []

    def chat_json(self, tag, messages, **_kw):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        self.calls.append((tag, text))
        self.kwargs.append((tag, dict(_kw)))
        if tag.startswith("world.repair"):
            return {"fields": {}}
        return _typed_table()


tracer = _BlueprintTracer()
ws = build_world(build_wp, tracer, log=lambda *_a: None)
prompt_text = "\n".join(text for tag, text in tracer.calls if tag.startswith("world."))
ck("③ prompt 消费全部 entity type", all(x in prompt_text for x in ("player", "boss", "equipment")))
ck("③ prompt 消费 relation type", "equips" in prompt_text)
ck("③ prompt 消费 event type", "defeat_boss" in prompt_text and "acquire_loot" in prompt_text)
structure_text = next(text for tag, text in tracer.calls if tag == "world.structure")
structure_entities = json.JSONDecoder().raw_decode(
    structure_text.split("【typed entities】", 1)[1].lstrip())[0]
structure_boss = next(entity for entity in structure_entities if entity["name"] == "熔岩巨兽")
ck("③ event-owned 字段不从 entity batch 泄入 structure 初态",
   structure_boss["initial_state"] == {}
   and "defeat_status" not in structure_boss["intrinsic_fields"])
structure_kwargs = next(kwargs for tag, kwargs in tracer.kwargs if tag == "world.structure")
ck("③ structure 固定模型、严格 JSON 且无内部重试",
   structure_kwargs.get("model") == config.STRUCTURE_MODEL
   and structure_kwargs.get("retries") == 1
   and structure_kwargs.get("strict_json") is True)
ck("③ prompt 消费 temporal cadence", "per_chapter" in prompt_text and "chapter" in prompt_text)
ck("③ blueprint 的 session 数覆盖旧 shared_world_spec", ws.n_sessions == 4)
ck("③ 每种声明类型达到 count", getattr(ws, "entity_types", {}) == {
    "旅者": "player", "熔岩巨兽": "boss", "星铁剑": "equipment"})
ck("③ relation 编译进 source timeline", ws.timeline("旅者", "equipped_item") is not None
   and ws.timeline("旅者", "equipped_item").value_at_session(1) == "星铁剑")
ck("③ event effect 编译进 target timeline", ws.timeline("熔岩巨兽", "defeat_status") is not None
   and ws.timeline("熔岩巨兽", "defeat_status").value_at_session(2) == "defeated")
ck("③ event 实例留存在世界真源", any(e.get("id") == "evt-1" for e in getattr(ws, "events", [])))


class _BatchErrorTracer:
    """模拟内层三次重试已经耗尽后的显式调用失败。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, tag, _messages, **_kw):
        self.calls += 1
        return {"__error__": "inner retries exhausted"}


batch_error_tracer = _BatchErrorTracer()
try:
    build_world(build_wp, batch_error_tracer, log=lambda *_args: None)
except WorldBlueprintError as exc:
    batch_error_closed = "world.batch[player] 调用失败" in str(exc)
else:
    batch_error_closed = False
ck("③ world.batch 内层失败后立即 fail-closed，不进入四轮外层重复",
   batch_error_closed and batch_error_tracer.calls == 1)


class _InvalidRepairTracer(_BlueprintTracer):
    """先制造可修缺陷，再让 repair 返回不可编译值。"""

    def chat_json(self, tag, messages, **_kw):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        self.calls.append((tag, text))
        if tag == "world.repair":
            return {"fields": {"level": {"type": "stable", "value": "banana"}}}
        table = _typed_table()
        table["entities"][0]["fields"]["level"] = {
            "type": "evolving",
            "trajectory": [{"session": 0, "value": "9"}, {"session": 1, "value": "8"}],
        }
        return table


invalid_repair_wp = deepcopy(build_wp)
next(field for field in invalid_repair_wp["world_blueprint"]["entity_types"][0]["fields"]
     if field["name"] == "level")["monotonic"] = "up"
try:
    build_world(invalid_repair_wp, _InvalidRepairTracer(), log=lambda *_args: None)
except WorldBlueprintError:
    invalid_repair_failed_closed = True
else:
    invalid_repair_failed_closed = False
ck("③ repair 写出不可编译字段后 typed world 必须 fail-closed",
   invalid_repair_failed_closed)

frozen_wp = deepcopy(build_wp)
frozen_wp["active_lines"] = [
    {"line": "L4_preference"}, {"line": "L9_induction"}, {"line": "L10_admission"},
]
frozen_wp["domain_profile"]["preference_axis"] = {
    "entity_type": "player", "field": "loot_state", "options": ["acquired", "unclaimed"]}
before_prepare = json.dumps(ws.to_dict(), ensure_ascii=False, sort_keys=True)
prepare_lines(frozen_wp, ws, log=lambda *_args: None)
PreferenceLine().prepare(ws, frozen_wp["domain_profile"])
after_prepare = json.dumps(ws.to_dict(), ensure_ascii=False, sort_keys=True)
ck("③ typed world 在能力 prepare 阶段 canonical core 完全冻结", before_prepare == after_prepare)
ck("③ typed world 不自动注入固定工单/敏感/工位字段",
   not ws.rule_instances and not ws.sensitive and all("工位编号" not in fields for fields in ws.entities.values()))

typed_l2_orders = RelationalLine().enumerate(compiled, wp=build_wp)
typed_l2_intents = [RelationalLine().intent(order)[0] for order in typed_l2_orders]
ck("③ typed L2 题面使用领域时间单位且不把任意中间对象叫成人",
   bool(typed_l2_intents) and all("章" in text and "那个人" not in text for text in typed_l2_intents))

roundtrip = WorldState.from_dict(ws.to_dict())
ck("④ entity_types 序列化往返", getattr(roundtrip, "entity_types", {}) == getattr(ws, "entity_types", {}))
ck("④ events 序列化往返", getattr(roundtrip, "events", []) == getattr(ws, "events", []))
ck("④ relations / blueprint / causal cascades 序列化往返",
   roundtrip.relations == ws.relations and roundtrip.world_blueprint == ws.world_blueprint
   and roundtrip.cascades == ws.cascades
   and any(c.get("kind") == "domain_event_causality" for c in roundtrip.cascades))
rt_rel = roundtrip.timeline("旅者", "equipped_item")
rt_event = roundtrip.timeline("熔岩巨兽", "defeat_status")
ck("④ 关系/事件编译后的 timeline 往返", rt_rel is not None and rt_event is not None
   and rt_rel.value_at_session(1) == "星铁剑" and rt_event.value_at_session(2) == "defeated")


# ═══════ ⑤ typed augment 合并新旧骨架实例 ═══════
augment_base_table = {
    "entities": [
        {"name": "P1", "type": "player", "fields": {
            "level": {"type": "stable", "value": "7"},
        }},
        {"name": "B1", "type": "boss", "fields": {
            "defeat_status": {"type": "stable", "value": "alive"},
        }},
        {"name": "G1", "type": "equipment", "fields": {
            "enhance_level": {"type": "stable", "value": "2"},
        }},
    ],
    "relations": [
        {"id": "rel-old", "type": "equips", "from": "P1", "to": "G1", "session": 1},
    ],
    "events": [
        {"id": "evt-old-defeat", "type": "defeat_boss", "session": 1,
         "participants": {"actor": "P1", "target": "B1"},
         "effects": [{"entity": "B1", "field": "defeat_status", "set": "defeated-old"}]},
        {"id": "evt-old-loot", "type": "acquire_loot", "session": 2,
         "caused_by": "evt-old-defeat", "participants": {"owner": "P1"},
         "effects": [{"entity": "P1", "field": "loot_state", "set": "acquired-old"}]},
    ],
    "cascades": [],
    "absent_fields": [],
}
augment_existing, augment_base_issues = assemble_world(augment_base_table, blueprint=build_bp)

augment_wp = deepcopy(build_wp)
next(t for t in augment_wp["world_blueprint"]["entity_types"]
     if t["id"] == "player")["count"] = 2


class _AugmentTracer:
    """只生成缺口 P2；新关系和新事件故意跨越 delta，引用旧世界 G1/B1。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, tag, messages, **_kw):
        text = "\n".join(str(message.get("content", "")) for message in messages)
        self.calls.append((tag, text))
        if tag == "world.batch":
            return {"entities": [{
                "name": "P2", "type": "player", "fields": {
                    "level": {"type": "stable", "value": "11"},
                    "loot_state": {"type": "stable", "value": "unclaimed"},
                },
            }]}
        if tag == "world.structure":
            return {
                "relations": [
                    {"id": "rel-new", "type": "equips", "from": "P2", "to": "G1", "session": 2},
                ],
                "events": [
                    {"id": "evt-new-defeat", "type": "defeat_boss", "session": 2,
                     "participants": {"actor": "P2", "target": "B1"},
                     "effects": [{"entity": "B1", "field": "defeat_status", "set": "defeated-new"}]},
                    {"id": "evt-new-loot", "type": "acquire_loot", "session": 3,
                     "caused_by": "evt-new-defeat", "participants": {"owner": "P2"},
                     "effects": [{"entity": "P2", "field": "loot_state", "set": "acquired-new"}]},
                ],
            }
        if tag.startswith("world.repair"):
            return {"fields": {}}
        raise AssertionError(f"未预期的 tracer 调用:{tag}")


augment_tracer = _AugmentTracer()
augment_result = build_world(
    augment_wp, augment_tracer, existing=augment_existing, log=lambda *_args: None)
augment_relation_ids = {relation.get("id") for relation in augment_result.relations}
augment_event_ids = {event.get("id") for event in augment_result.events}
ck("⑤ typed augment 基础世界零缺陷", augment_base_issues == [])
ck("⑤ typed augment 保持 existing 对象身份", augment_result is augment_existing)
ck("⑤ typed augment 只补 primary count 缺口",
   augment_result.entity_types == {"P1": "player", "P2": "player", "B1": "boss", "G1": "equipment"}
   and [tag for tag, _text in augment_tracer.calls].count("world.batch") == 1)
ck("⑤ typed augment 合并新旧 relation",
   augment_relation_ids == {"rel-old", "rel-new"}
   and augment_result.timeline("P1", "equipped_item").value_at_session(1) == "G1"
   and augment_result.timeline("P2", "equipped_item").value_at_session(2) == "G1")
ck("⑤ typed augment 合并新旧 event",
   augment_event_ids == {"evt-old-defeat", "evt-old-loot", "evt-new-defeat", "evt-new-loot"}
   and augment_result.timeline("B1", "defeat_status").value_at_session(1) == "defeated-old"
   and augment_result.timeline("B1", "defeat_status").value_at_session(2) == "defeated-new"
   and augment_result.timeline("P2", "loot_state").value_at_session(3) == "acquired-new")
ck("⑤ typed augment 结构 prompt 可引用旧 G1/B1",
   any('"name": "G1"' in text and '"name": "B1"' in text
       for tag, text in augment_tracer.calls if tag == "world.structure"))


# ═══════ ⑥ render signal 不丢类型图例与当期领域事件 ═══════
render_system = _corpus_system(build_wp["domain_profile"], compiled.world_blueprint)
render_facts = _session_facts(compiled, 2)
render_events = [event for event in compiled.events if event.get("session") == 2]
render_user = render_prompt(
    "corpus.user",
    s="3",
    time_unit=compiled.world_blueprint["temporal_model"]["unit"],
    date="2025-01-15",
    facts=json.dumps(render_facts, ensure_ascii=False),
    events=json.dumps(render_events, ensure_ascii=False),
    hint="",
)
render_signal_prompt = f"{render_system}\n{render_user}"
ck("⑥ signal system 携带全部 entity_type 图例", all(
    legend in render_system for legend in ("player=玩家", "boss=Boss", "equipment=装备")))
ck("⑥ signal fact 携带实体类型而非只有扁平字段",
   any(fact.get("entity") == "熔岩巨兽" and fact.get("entity_type") == "boss"
       for fact in render_facts)
   and '"entity_type": "boss"' in render_signal_prompt)
ck("⑥ signal user 只携带当期 domain event",
   '"id": "evt-1"' in render_user and '"type": "defeat_boss"' in render_user
   and '"id": "evt-2"' not in render_user)
event_for_gate = [{**render_events[0], "label": "击败首领"}]
ck("⑥ 只写 effect 字段值、漏掉参与者关系时事件忠实闸拒绝",
   bool(_missing_event_narratives(event_for_gate, ["熔岩巨兽的 defeat_status 变成 defeated。"])))
ck("⑥ event 动作可自然改写，但 participants 与 effect 三元组必须同篇",
   _missing_event_narratives(
       event_for_gate,
       ["旅者在交锋中战胜了对手，熔岩巨兽的 defeat_status 变成 defeated。"]
   ) == [])
duplicate_label = deepcopy(build_wp["world_blueprint"])
duplicate_label["event_types"][1]["label"] = duplicate_label["event_types"][0]["label"]
ck("⑥ event type 的人类可读 label 必须唯一，避免 provenance 歧义",
   any("event label 重复" in issue for issue in validate_world_blueprint(duplicate_label)))
game_filler_system = _filler_system(
    {"entity_noun": "游戏对象", "doc_genres": ["任务日志", "战利品记录"]},
    compiled.world_blueprint,
)
ck("⑥ filler prompt 携带冻结世界的类型、事件和证据渠道",
   all(marker in game_filler_system for marker in ("player", "击败首领", "任务日志")))
ck("⑥ filler 被约束为同世界旁支且禁止非办公场景套 OA 模板",
   "同一领域、同一叙事世界" in game_filler_system
   and "不得出现公司员工、OA、办公区、食堂培训" in game_filler_system)
blocked_terms = _tracked_blocklist(compiled, {"field_schema": []})
ck("⑥ filler 只禁被追踪专名，不禁同领域通用字段词",
   set(compiled.entities) <= blocked_terms
   and not ({field for fields in compiled.entities.values() for field in fields}
            & blocked_terms))


# ═══════ ⑦ central_office 先冻结世界，再映射能力 ═══════
architect_blueprint = deepcopy(_game_whitepaper()["world_blueprint"])
architect_blueprint["evidence_channels"] = ["architect-draft-channel"]
critic_rewrite = deepcopy(_office_whitepaper()["world_blueprint"])
critic_rewrite["evidence_channels"] = ["critic-illegal-rewrite"]


class _WorldFirstTracer:
    """离线议会桩：架构师先冻结骨架，critic 再恶意改写，验证硬门。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[tuple[str, dict]] = []

    def chat_json(self, tag, messages, **_kw):
        text = "\n".join(str(message.get("content", "")) for message in messages)
        self.calls.append((tag, text))
        self.kwargs.append((tag, dict(_kw)))
        if tag == "council.observe":
            return {"observed_media": [], "observed_entities": [], "observed_fields": []}
        if tag == "council.skeptic":
            return {"latent_entities": []}
        if tag == "council.medium":
            return {"recommended_mix": [], "common_media": [], "unconventional_media": []}
        if tag == "council.style":
            return {"style_spec": {"tone": "客观"}}
        if tag == "council.traps":
            return {"traps": []}
        if tag == "council.world":
            return {"world_blueprint": deepcopy(architect_blueprint)}
        if tag == "council.map":
            return {"per_line": [
                {"line": line_id, "applicable": line_id == "L1_timeline",
                 "instantiation": "只读已冻结的事件时序" if line_id == "L1_timeline" else "",
                 "gt_feasible": line_id == "L1_timeline",
                 "weight_hint": 0.5 if line_id == "L1_timeline" else 0}
                for line_id in (
                    "L1_timeline", "L2_relational", "L3_process", "L4_preference",
                    "L5_conflict", "L6_refusal", "L7_consolidation")
            ]}
        raise AssertionError(f"未预期的 council tracer 调用:{tag}")


world_first_tracer = _WorldFirstTracer()
world_first_wp = central_office(
    "测试 world-first 议会", [], world_first_tracer, log=lambda *_args: None)
world_first_tags = [tag for tag, _text in world_first_tracer.calls]
world_map_prompt = next(text for tag, text in world_first_tracer.calls if tag == "council.map")
expected_frozen_blueprint = normalize_world_blueprint(
    {"world_blueprint": architect_blueprint})
ck("⑦ world 严格先于 map",
   world_first_tags.index("council.world") < world_first_tags.index("council.map"))
ck("⑦ 已删无执行效果的 world_review 调用",
   "council.world_review" not in world_first_tags)
traps_kwargs = next(kwargs for tag, kwargs in world_first_tracer.kwargs
                    if tag == "council.traps")
ck("⑦ 可选 traps 视角只调用一次且严格 JSON",
   traps_kwargs.get("retries") == 1 and traps_kwargs.get("strict_json") is True)
ck("⑦ 已删可选全文 critic 重写，通过契约的草案直接定稿",
   "council.critique" not in world_first_tags)
ck("⑦ map prompt 收到架构师已冻结 blueprint，反方无权改写",
   "architect-draft-channel" in world_map_prompt
   and '"type": "defeat_boss"' not in world_map_prompt
   and '"id": "defeat_boss"' in world_map_prompt)
ck("⑦ 定稿保留已冻结 blueprint",
   structure_signature(world_first_wp["world_blueprint"])
   == structure_signature(expected_frozen_blueprint)
   and "architect-draft-channel" in world_first_wp["world_blueprint"]["evidence_channels"]
   and "critic-illegal-rewrite" not in world_first_wp["world_blueprint"]["evidence_channels"]
   and world_first_wp["domain_profile"]["entity_noun"] == "玩家")
ck("⑦ line_mapping 的不适用零权重线不进 active_lines",
   [item["line"] for item in world_first_wp["active_lines"]] == ["L1_timeline"])
world_architect_prompt = render_prompt("council.world")
world_repair_prompt = render_prompt("council.world_repair")
narrative_review_prompt = render_prompt("narrative.review")
ck("⑦ states 只表达不可逆生命周期，可循环运行状态必须省略",
   "不可逆" in world_architect_prompt and "必须省略 states" in world_architect_prompt
   and "可循环运行状态必须删除 states" in world_repair_prompt)
ck("⑦ 叙事审阅承认 CANON 文档日期与纯载体动作",
   "period_label/document_date" in narrative_review_prompt and "文档载体动词" in narrative_review_prompt)
complete_map = _WorldFirstTracer().chat_json("council.map", [])
ck("⑦ map 完整性契约接受 L1–L7 各一次", _map_issues(complete_map) == [])
ck("⑦ map 完整性契约拒绝缺 L6/L7",
   any("缺失产线" in issue for issue in _map_issues(
       {"per_line": complete_map["per_line"][:5]})))
# 模型常把 schema 的 kind 注解误抄进 JSON key；只允许受蓝图白名单约束的安全归一。
decorated_fields = _canonicalize_generated_field_names(
    {"当前任务(text)": {"type": "stable", "value": "调查"},
     "任务状态（status）": {"type": "stable", "value": "进行中"},
     "未知字段(text)": {"type": "stable", "value": "x"}},
    {"当前任务", "任务状态"},
)
ck("⑷ 已知 kind 后缀归一为蓝图字段",
   set(decorated_fields) >= {"当前任务", "任务状态"})
ck("⑷ 未知字段不被冒充成合法字段", "未知字段" not in decorated_fields)

# 模型也会直接把唯一的 kind 当作字段键；唯一映射可安全归一，多义映射必须拒绝。
kind_alias_fields = _canonicalize_generated_field_names(
    {"person": {"type": "stable", "value": "白泽"},
     "status": {"type": "stable", "value": "空闲"},
     "text": {"type": "stable", "value": "不可猜"}},
    {"执行体名", "可用状态", "备注一", "备注二"},
    [{"name": "执行体名", "kind": "person"},
     {"name": "可用状态", "kind": "status"},
     {"name": "备注一", "kind": "text"},
     {"name": "备注二", "kind": "text"}],
)
ck("⑷ 唯一 kind 键归一为蓝图字段",
   set(kind_alias_fields) >= {"执行体名", "可用状态"})
ck("⑷ 多义 kind 键不猜测字段", "text" in kind_alias_fields)

structural_defects, intrinsic_defects = _partition_structure_driven_defects(
    [{"entity": "调用一", "field": "调用状态", "type": "illegal_transition"},
     {"entity": "调用一", "field": "备注", "type": "fake_evolving"}],
    {"调用一": "tool_call"},
    {"tool_call": {"负责执行体"}},
    {"tool_call": {"调用状态"}},
)
ck("⑷ 结构驱动字段缺陷只回到 structure 修复",
   [item["field"] for item in structural_defects] == ["调用状态"])
ck("⑷ 普通字段缺陷仍交字段 critic",
   [item["field"] for item in intrinsic_defects] == ["备注"])

def _numeric_timeline(values: list[str]) -> Timeline:
    """构造可重现的数值时间线。"""
    return Timeline([
        Op(session=i, date=f"2025-01-{i + 1:02d}", op=SET if i == 0 else UPDATE,
           value=value, prev=None if i == 0 else values[i - 1])
        for i, value in enumerate(values)
    ])


typed_trend_blueprint = {
    "entity_types": [{
        "id": "report", "noun": "报告", "count": 4, "primary": True,
        "fields": [{"name": "置信度", "kind": "numeric"},
                   {"name": "事件分数", "kind": "numeric"}],
    }],
    "relation_types": [],
    "event_types": [{
        "id": "score", "roles": {"report": "report"},
        "effect_fields": [{"role": "report", "field": "事件分数"}], "min_count": 1,
    }],
}
typed_trend_world = WorldState(
    entities={f"报告{i}": {
        "置信度": _numeric_timeline(["0.2", "0.5", "0.3", "0.7", "0.4", "0.6"]),
        "事件分数": _numeric_timeline(["1", "4", "2", "5", "3", "6"]),
    } for i in range(4)},
    n_sessions=6,
    entity_types={f"报告{i}": "report" for i in range(4)},
    world_blueprint=typed_trend_blueprint,
)
typed_trend_world.entities["报告0"]["置信度"].ops.append(
    Op(session=6, date="2025-01-07", op=EXPIRE, value=None, prev="0.6"))
event_before = [op.value for op in typed_trend_world.entities["报告0"]["事件分数"].ops]
typed_planted = imprint_structure(
    typed_trend_world, log=lambda *_args: None,
    profile={"field_schema": [{"name": "置信度", "kind": "numeric"}],
             "l7_max_trends": 4}, seed=7)
ck("⑷ typed L7 只在内在 numeric 字段种足 4 条基质",
   typed_planted == 4 and len(typed_trend_world._trended_fields) == 4
   and {field for _entity, field in typed_trend_world._trended_fields} == {"置信度"})
ck("⑷ typed L7 不改事件单一真源字段",
   [op.value for op in typed_trend_world.entities["报告0"]["事件分数"].ops]
   == event_before)
ck("⑷ typed L7 保留停统操作供 L6 复用",
   typed_trend_world.entities["报告0"]["置信度"].ops[-1].op == EXPIRE)

npass = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    if not ok:
        print(f"  ✗ {name}")
print(f"[world_blueprint self-test] {npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
