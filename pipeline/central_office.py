"""
pipeline.central_office —— 中央办公室【并行议会】版:场景+few-shot → 白皮书。

redesign_factory_v2.md §3 的升级:从"单轮提案+批判"→ 7 个【议会视角】并行 → 综合 → 批判。
视角(都不叫 L、不叫宪法,避免与产线 L1–L7 / 元架构宪法撞名):
  观测  observe  : few-shot 字面给了啥(只抽取、不推断)
  怀疑  skeptic  : 假定 few-shot 系统性缺失,阐发缺了啥(带合理性缰绳 + observed/inferred 标注)
  映射  map      : 对照宪法 L1–L7,本场景每条该不该激活、怎么落地
  媒介  medium   : 本场景还能是什么形式(先穷尽常见、再补反常但合理 —— 大而全,非猎奇)
  文风  style    : 抽 few-shot 风格 DNA + 原文当渲染范例
  陷阱  traps    : 本场景天然在哪坑记忆系统(接 M1–M6,设计区分度)
  世界  world    : 先冻结类型/关系/事件/时间制度,定义场景不可换皮的“骨架”
→ 综合成白皮书草案 → 批判(覆盖/自洽/gt可行/区分度)→ 定稿。

白皮书 schema 与下游(build_world/run_lines/render)兼容:必出 world_blueprint / domain_profile / medium /
active_lines / shared_world_spec / capability_targets;另附 style_spec / traps / line_mapping /
observed_vs_inferred(留痕,后续渲染/诊断用)。
"""
from __future__ import annotations
from copy import deepcopy
import json
import config
from pipeline.lines import implemented_ids, line_for, taxonomy_prose
from pipeline.prompts import render          # ★议会 prompts 收编进注册表(council.*)
from pipeline.seed_pack import (
    attach_seed_contract,
    core_requirements,
    seed_context,
    project_required_blueprint,
    validate_seed_blueprint,
    validate_seed_pack,
)
from pipeline.world_blueprint import (
    WorldBlueprintError,
    _looks_like_identity_field,
    normalize_world_blueprint,
    repair_blueprint_candidate,
)

# ── 7 个议会视角的 system prompt ──────────────────────────────────────────
OBSERVE_SYS = render("council.observe")
SKEPTIC_SYS = render("council.skeptic")
MAP_SYS = render("council.map", taxonomy=taxonomy_prose())   # ★map 内联宪法 taxonomy(调用侧算好传入)
MEDIUM_SYS = render("council.medium")
STYLE_SYS = render("council.style")
TRAPS_SYS = render("council.traps")
WORLD_SYS = render("council.world")
WORLD_REPAIR_SYS = render("council.world_repair")
WORLD_BUSINESS_REPAIR_SYS = WORLD_SYS + """
【业务可执行性修订】
根据本轮业务复核的具体缺口修订候选蓝图。补齐事件的角色、可写效果和必要业务结构，让任务要求的过程与跨期轨迹能够实际发生。先逐项核对复核意见，再输出完整 world_blueprint。
seed 的必需结构保留原 id、label、角色、效果、约束和因果引用；新增角色、效果或事件按领域含义扩展，已有 label 原文保留。observe 冻结字段、实体与期数上限以及逐线容量要求继续适用。已通过的结构保留，修订集中在本轮指出的缺口及其必要依赖。
"""


def _world_revision_messages(error, candidate, observed_contract, seed_prompt, *,
                             business=False, preserve_business_additions=False):
    """Route executable-business feedback to the architect before schema repair."""
    if business:
        system = WORLD_BUSINESS_REPAIR_SYS
        user = ("【业务复核意见】\n" + error + "\n【待修候选】\n"
                + json.dumps(candidate, ensure_ascii=False)
                + "\n逐项修订业务缺口，输出完整 JSON。")
    else:
        system = WORLD_REPAIR_SYS
        if preserve_business_additions:
            system += ("\n本候选包含上一轮业务修订新增的结构。机械修理仅改错误清单指出的字段及必要引用；"
                       "保留其余合法角色、效果和事件。label 错误仅恢复 label，保留已合法的新增 effect_fields。")
        user = render("council.world_repair_user", errors=error,
                      candidate=json.dumps(candidate, ensure_ascii=False))
    user += f"\n【每轮都必须完整保留的 observe 冻结清单】\n{observed_contract}" + seed_prompt
    return system, user

# ── 综合 + 批判 ──────────────────────────────────────────────────────────
CRITIC_SYS = render("council.critic", taxonomy=taxonomy_prose())   # ★critic 也内联 canonical 菜单(014559:critic 重写丢 L7+自编名,根在它没拿到菜单)
# (SYNTH_SYS 已删:综合早已改为代码确定性装配 _assemble_whitepaper,该 prompt 不再被调用)

_PERSPECTIVES = [
    ("observe", OBSERVE_SYS, "据实抽取 few-shot 字面信息。"),
    ("skeptic", SKEPTIC_SYS, "阐发本场景 few-shot 没展示但真实存在的媒介/实体/关系(带缰绳)。"),
    ("map", MAP_SYS, "对照 L1–L7,逐条判断本场景该不该激活、怎么落地。"),
    ("medium", MEDIUM_SYS, "发散本场景所有可能形式(常见穷尽 + 反常但合理),大而全。"),
    ("style", STYLE_SYS, "抽 few-shot 风格 DNA。"),
    ("traps", TRAPS_SYS, "设计本场景坑记忆系统的陷阱(接区分度)。"),
    ("world", WORLD_SYS, "定义本场景不可换皮的实体类型、关系、领域事件、因果与时间制度。"),
]

# 中央办公室只负责当前已实现的 L1–L7 能力映射；L8+由世界基质自动激活。
_MAP_LINE_IDS = tuple(
    line_id for line_id in implemented_ids()
    if 1 <= int(line_id.split("_", 1)[0][1:]) <= 7
)


def _map_issues(candidate: dict, expected_lines=None) -> list[str]:
    """检查能力映射是否对 L1–L7 完整、唯一且可执行。"""
    expected_lines = tuple(expected_lines or _MAP_LINE_IDS)
    if not isinstance(candidate, dict) or "__error__" in candidate:
        return [f"map 调用失败:{(candidate or {}).get('__error__', '非 JSON object') if isinstance(candidate, dict) else '非 JSON object'}"]
    per_line = candidate.get("per_line")
    if not isinstance(per_line, list):
        return ["per_line 必须是 array"]
    issues: list[str] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(per_line):
        if not isinstance(item, dict):
            issues.append(f"per_line[{index}] 必须是 object")
            continue
        raw_id = item.get("line")
        line = line_for(raw_id) if isinstance(raw_id, str) else None
        line_id = line.id if line else None
        if line_id not in expected_lines or raw_id != line_id:
            issues.append(f"per_line[{index}].line 必须是 L1–L7 canonical id，实际={raw_id!r}")
            continue
        seen[line_id] = seen.get(line_id, 0) + 1
        if type(item.get("applicable")) is not bool:
            issues.append(f"{line_id}.applicable 必须是 boolean")
        if item.get("applicable"):
            if item.get("gt_feasible") is not True:
                issues.append(f"{line_id} 适用时 gt_feasible 必须为 true")
            if not isinstance(item.get("instantiation"), str) or not item["instantiation"].strip():
                issues.append(f"{line_id} 适用时 instantiation 不得为空")
            weight = item.get("weight_hint")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not 0 < float(weight) <= 1:
                issues.append(f"{line_id} 适用时 weight_hint 必须在 (0,1]")
    missing = [line_id for line_id in expected_lines if seen.get(line_id, 0) == 0]
    duplicate = [line_id for line_id in expected_lines if seen.get(line_id, 0) > 1]
    if missing:
        issues.append(f"缺失产线:{missing}")
    if duplicate:
        issues.append(f"重复产线:{duplicate}")
    return issues


def _g(d, *keys, default=None):
    cur = d if isinstance(d, dict) else {}
    for k in keys:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if cur not in ({}, None) else default


_OBSERVE_UNSPECIFIED = {
    "", "-", "[]", "无", "可省", "不适用", "未注明", "未提供", "未观测", "未知",
    "null", "none", "nil", "n/a", "unknown",
}


def _observe_unspecified(value) -> bool:
    """判断 observe 输出是否只是“未知/可省”占位，而非可冻结的字面事实。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _OBSERVE_UNSPECIFIED
    if isinstance(value, (list, tuple)):
        return not value or all(_observe_unspecified(item) for item in value)
    if isinstance(value, dict):
        return not value or all(_observe_unspecified(item) for item in value.values())
    return False


def _observed_strings(value) -> list[str]:
    """清洗 observe 的字符串数组，丢弃空值/占位符/畸形元素并保持顺序去重。"""
    if not isinstance(value, list):
        return []
    clean = [item.strip() for item in value
             if isinstance(item, str) and not _observe_unspecified(item)]
    return list(dict.fromkeys(clean))


def _normalize_haystack_plan(medium: dict) -> dict:
    """Turn the medium agent's bounded proposal into an executable corpus plan."""
    medium = medium if isinstance(medium, dict) else {}
    raw = medium.get("haystack_plan")
    raw = raw if isinstance(raw, dict) else {}
    requested = raw.get("filler_documents_per_session")
    documents = requested if type(requested) is int else 8
    documents = max(8, min(12, documents))

    def strings(value, limit):
        clean = [item.strip() for item in value
                 if isinstance(item, str) and item.strip()] if isinstance(value, list) else []
        return list(dict.fromkeys(clean))[:limit]

    genres = strings(raw.get("filler_genres"), 12)
    topics = strings(raw.get("filler_topics"), 20)
    why = raw.get("why")
    return {
        "version": "domain-native-haystack/v1",
        "filler_documents_per_session": documents,
        "filler_genres": genres,
        "filler_topics": topics,
        "why": why.strip()[:1000] if isinstance(why, str) and why.strip() else "",
    }


def _observed_blueprint_issues(blueprint: dict, observed: dict) -> list[str]:
    """检查蓝图是否完整且原样承接 few-shot 的字面字段硬事实。"""
    fields_by_name = {}
    for entity_type in blueprint.get("entity_types", []):
        for field in entity_type.get("fields") or []:
            fields_by_name.setdefault(field.get("name"), field)
    issues = []
    raw_fields = observed.get("observed_fields") if isinstance(observed, dict) else None
    if _observe_unspecified(raw_fields):
        raw_fields = []
    elif not isinstance(raw_fields, list):
        return ["observe.observed_fields 必须是 array"]
    for index, field in enumerate(raw_fields):
        if _observe_unspecified(field):
            continue
        if not isinstance(field, dict):
            issues.append(f"observe.observed_fields[{index}] 必须是 object")
            continue
        name = field.get("name")
        if _observe_unspecified(name):
            continue
        if not isinstance(name, str):
            issues.append(f"observe.observed_fields[{index}].name 必须是字符串")
            continue
        declared = fields_by_name.get(name)
        if declared is None:
            issues.append(f"未覆盖 few-shot 已观测字段:{name}")
            continue
        for attr in ("kind", "unit", "monotonic", "range"):
            value = field.get(attr)
            # observe 模型常把可选值输出成占位符；未知不是硬约束，半空值仍须 fail-closed。
            if not _observe_unspecified(value) and value != declared.get(attr):
                issues.append(
                    f"改写 few-shot 硬事实:{name}.{attr} observed={value!r},blueprint={declared.get(attr)!r}")
    return issues


def _separate_observed_relation_fields(candidate: dict, observed: dict, *,
                                      seed_reference_fields: set[str] | None = None) -> list[str]:
    """把 few-shot 展示字段与同名关系 FK 确定性拆成两条字段。

    LLM 容易在 ``category/person`` 硬事实与 ``reference`` 关系字段之间振荡。
    这里不改变关系端点或领域语义，只保留观察字段原形，并给使用它的关系创建
    独立 reference 字段；未绑定关系的误改 reference 则恢复观察 kind。
    返回修复说明用于 Run 留痕。
    """
    if not isinstance(candidate, dict):
        return []
    blueprint = candidate.get("world_blueprint", candidate)
    if not isinstance(blueprint, dict):
        return []
    raw_observed = observed.get("observed_fields") if isinstance(observed, dict) else None
    if not isinstance(raw_observed, list):
        return []
    observed_specs = {
        item.get("name"): item for item in raw_observed
        if isinstance(item, dict) and isinstance(item.get("name"), str)
        and item.get("name") and not _observe_unspecified(item.get("kind"))
    }
    if not observed_specs:
        return []

    types = {item.get("id"): item for item in blueprint.get("entity_types", [])
             if isinstance(item, dict) and isinstance(item.get("id"), str)}
    relations = [item for item in blueprint.get("relation_types", []) if isinstance(item, dict)]
    repairs: list[str] = []

    def _field(entity_type: dict, name: str):
        return next((item for item in entity_type.get("fields", [])
                     if isinstance(item, dict) and item.get("name") == name), None)

    def _fresh_reference_name(owner: dict, base: str) -> str:
        names = {item.get("name") for item in owner.get("fields", []) if isinstance(item, dict)}
        stem = f"{base}引用"
        if stem not in names:
            return stem
        index = 2
        while f"{stem}{index}" in names:
            index += 1
        return f"{stem}{index}"

    # 先处理确实把 observed 显示字段拿去当 relation.field 的关系。
    for relation in relations:
        old_name = relation.get("field")
        observed_spec = observed_specs.get(old_name)
        if not observed_spec:
            continue
        if old_name in (seed_reference_fields or set()) and observed_spec.get("kind") == "reference":
            # A reviewed seed explicitly owns this canonical FK. Observe's
            # textual display of its value does not create a second field.
            continue
        endpoints = [types.get(relation.get("from_type")), types.get(relation.get("to_type"))]
        owners = [entity_type for entity_type in endpoints
                  if entity_type is not None and _field(entity_type, old_name) is not None]
        if len(owners) != 1:
            continue
        owner = owners[0]
        owner_field = _field(owner, old_name)
        new_name = _fresh_reference_name(owner, old_name)
        owner["fields"].append({"name": new_name, "kind": "reference"})
        relation["field"] = new_name
        # 同名显示字段仍按 observe 硬事实保留，不能被关系改型。
        owner_field.update({key: deepcopy(value) for key, value in observed_spec.items()
                            if key in ("kind", "unit", "monotonic", "range")
                            and not _observe_unspecified(value)})
        repairs.append(f"{owner.get('id')}.{old_name}→显示字段 + {new_name}(relation={relation.get('id')})")

    # 再恢复没有关系绑定、却被模型误改成 reference 的 observed 同名字段。
    bound = {(entity_type.get("id"), relation.get("field"))
             for relation in relations for entity_type in types.values()
             if entity_type.get("id") in (relation.get("from_type"), relation.get("to_type"))
             and _field(entity_type, relation.get("field")) is not None}
    for type_id, entity_type in types.items():
        for field in entity_type.get("fields", []):
            spec = observed_specs.get(field.get("name")) if isinstance(field, dict) else None
            if not spec or (type_id, field.get("name")) in bound:
                continue
            for key in ("kind", "unit", "monotonic", "range"):
                value = spec.get(key)
                if not _observe_unspecified(value) and field.get(key) != value:
                    field[key] = deepcopy(value)
                    repairs.append(f"{type_id}.{field.get('name')}.{key} 恢复 observe={value!r}")
    return repairs


def _remove_inferred_identity_fields(candidate: dict, observed: dict, *,
                                     seed_required_fields: set[tuple[str, str]] | None = None) -> list[str]:
    """删除模型臆造的重复身份字段，保留 few-shot 硬事实与结构引用。

    每个 typed entity 已有 canonical ``name``。若架构师又给 Boss、Equipment
    等类型增加“名称”，实体生成模型就必须把同一事实再写进 ``fields``，实测会
    造成大量合法实体因缺这个冗余键而被丢弃。只有 few-shot 字面观察到的身份字段，
    或被关系/事件契约引用、被种子明确要求的字段，不能在这里机械删除。
    """
    if not isinstance(candidate, dict):
        return []
    blueprint = candidate.get("world_blueprint", candidate)
    if not isinstance(blueprint, dict):
        return []
    observed_names = {
        item.get("name") for item in (observed.get("observed_fields") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(observed, dict) else set()
    referenced = {
        relation.get("field") for relation in (blueprint.get("relation_types") or [])
        if isinstance(relation, dict)
    }
    referenced.update(
        effect.get("field")
        for event in (blueprint.get("event_types") or []) if isinstance(event, dict)
        for effect in (event.get("effect_fields") or []) if isinstance(effect, dict)
    )
    repairs: list[str] = []
    for entity_type in blueprint.get("entity_types") or []:
        if not isinstance(entity_type, dict) or not isinstance(entity_type.get("fields"), list):
            continue
        kept = []
        for field in entity_type["fields"]:
            name = field.get("name") if isinstance(field, dict) else None
            if (_looks_like_identity_field(name)
                    and name not in observed_names and name not in referenced
                    and (entity_type.get("id"), name) not in (seed_required_fields or set())):
                repairs.append(f"{entity_type.get('id')}.{name} 删除(与 canonical name 重复)")
                continue
            kept.append(field)
        entity_type["fields"] = kept
    return repairs


def _repair_candidate_blueprint(candidate: dict, observed: dict,
                                evidence_hints: list[str] | None = None, *,
                                seed_reference_fields: set[str] | None = None,
                                seed_required_fields: set[tuple[str, str]] | None = None) -> tuple[dict, list[str]]:
    """机械归一模型候选，并在展示字段拆分后再确认一次引用闭包。

    evidence 生态由专门的 medium 视角所有；world 只在自己给出有效渠道时保留，
    空值则投影 medium/observe 已产出的渠道，避免同一事实要求架构师重复生成。
    """
    repaired, repairs = repair_blueprint_candidate(candidate)
    repairs.extend(_separate_observed_relation_fields(
        repaired, observed, seed_reference_fields=seed_reference_fields))
    repairs.extend(_remove_inferred_identity_fields(
        repaired, observed, seed_required_fields=seed_required_fields))
    blueprint = repaired.get("world_blueprint", repaired) if isinstance(repaired, dict) else {}
    if isinstance(blueprint, dict):
        current = [item.strip() for item in (blueprint.get("evidence_channels") or [])
                   if isinstance(item, str) and item.strip()
                   and not _observe_unspecified(item)]
        hints = [item.strip() for item in (evidence_hints or [])
                 if isinstance(item, str) and item.strip()
                 and not _observe_unspecified(item)]
        if current:
            blueprint["evidence_channels"] = list(dict.fromkeys(current))
        elif hints:
            blueprint["evidence_channels"] = list(dict.fromkeys(hints))[:6]
            repairs.append("evidence_channels 从 medium/observe 单一真源投影")
    repaired, final_repairs = repair_blueprint_candidate(repaired)
    repairs.extend(final_repairs)
    return repaired, repairs


def _seed_observation_contract(observed: dict, pack: dict) -> tuple[dict, list[dict], set[str]]:
    """Reviewed seed schema overrides model-inferred kinds, without editing samples.

    Only attributes explicitly declared in the seed are authoritative. Literal
    values and other observe output keep their existing interpretation. Same-
    named fields cannot disagree across types because observe is untyped.
    """
    attributes = ("kind", "unit", "monotonic", "range")
    schemas: dict[str, dict] = {}
    reference_fields: set[str] = set()
    for entity in core_requirements(pack)["entity_types"]:
        for field in entity["fields"]:
            schema = {key: deepcopy(field[key]) for key in attributes if key in field}
            name = field["name"]
            if name in schemas and schemas[name] != schema:
                raise WorldBlueprintError(f"种子同名字段 schema 有歧义，无法对齐 observe:{name}")
            schemas[name] = schema
            if schema.get("kind") == "reference":
                reference_fields.add(name)
    effective = deepcopy(observed)
    overrides = []
    fields = effective.get("observed_fields") if isinstance(effective, dict) else None
    if isinstance(fields, list):
        for field in fields:
            if not isinstance(field, dict) or not isinstance(field.get("name"), str):
                continue
            for key, required in schemas.get(field["name"], {}).items():
                if field.get(key) != required:
                    overrides.append({"field": field["name"], "attribute": key,
                                      "observed": deepcopy(field.get(key)), "required": deepcopy(required),
                                      "reason": "curated_seed_schema_over_model_inference"})
                    field[key] = deepcopy(required)
    return effective, overrides, reference_fields


def _assemble_whitepaper(views: dict, desc: str) -> dict:
    """★综合 = 代码确定性装配；世界视角是必须通过机械校验的硬门。"""
    obs = views.get("observe") or {}
    skp = views.get("skeptic") or {}
    mp = views.get("map") or {}
    med = views.get("medium") or {}
    sty = views.get("style") or {}
    trp = views.get("traps") or {}
    # 新白皮书不允许 world 视角失败后悄悄退化成 legacy 单类型；历史适配只在读取旧产物时启用。
    world_view = views.get("world") or {}
    blueprint = normalize_world_blueprint(world_view)
    type_specs = blueprint["entity_types"]
    primary = next(t for t in type_specs if t.get("primary"))

    obs_ents = _observed_strings(obs.get("observed_entities"))
    observed_media = _observed_strings(obs.get("observed_media"))
    # blueprint 是字段/主体单一真源；observe 仅留 provenance，不再决定世界拓扑。
    entity_noun = primary["noun"]
    fields_by_name = {}
    for t in type_specs:
        for f in t.get("fields") or []:
            fields_by_name.setdefault(f.get("name"), deepcopy(f))
    observed_conflicts = _observed_blueprint_issues(blueprint, obs)
    if observed_conflicts:
        raise WorldBlueprintError(
            "world_blueprint 未承接 few-shot 硬事实:\n- " + "\n- ".join(observed_conflicts))
    fields = list(fields_by_name.values())

    # L4 只能映射到世界本来就存在的字段，不能在蓝图之后另造一条“偏好时间线”。
    preference_axis = deepcopy(mp.get("preference_axis")) if isinstance(mp.get("preference_axis"), dict) else None
    if preference_axis:
        axis_field = str(preference_axis.get("field") or "").strip()
        owners = [t["id"] for t in type_specs
                  if any(f.get("name") == axis_field for f in (t.get("fields") or []))]
        requested_owner = preference_axis.get("entity_type")
        owner = requested_owner if requested_owner in owners else (owners[0] if len(owners) == 1 else None)
        if not owner:
            preference_axis = None
        else:
            preference_axis["entity_type"] = owner
    blueprint["evidence_channels"] = list(dict.fromkeys(
        [x for x in blueprint.get("evidence_channels", []) if isinstance(x, str)]
        + observed_media))
    evidence = [x for x in blueprint.get("evidence_channels", []) if isinstance(x, str)]
    genres = list(dict.fromkeys(observed_media +
                                (med.get("recommended_mix") or []) + evidence)) or ["记录"]
    observed_stopped = obs.get("stopped_phrase_seen")
    stopped = (observed_stopped.strip()
               if isinstance(observed_stopped, str) and not _observe_unspecified(observed_stopped)
               else "停止/失效")

    # 旧 schema 的关系镜像从可执行蓝图派生，不能再由 skeptic 的 prose 充当死元数据。
    relations = [{"type": r["id"], "from": r["from_type"], "to": r["to_type"],
                  "field": r["field"], "temporal": r.get("temporal", False)}
                 for r in blueprint.get("relation_types", [])]

    # 激活产线:映射 applicable + 陷阱 boost(按 L<n> 前缀匹配)
    boost = {(t.get("boost_line") or "")[:2] for t in (trp.get("traps") or [])}
    active = []
    for pl in (mp.get("per_line") or []):
        if pl.get("applicable") and pl.get("line"):
            if str(pl.get("line")).lower().startswith("l4") and not preference_axis:
                continue
            w = float(pl.get("weight_hint") or 0.3)
            if pl["line"][:2] in boost:
                w = min(1.0, w + 0.1)
            active.append({"line": pl["line"], "weight": round(w, 2), "why": pl.get("instantiation", "")})
    if not active:
        raise WorldBlueprintError("map 完整审议后没有任何可执行产线")

    temporal = blueprint["temporal_model"]
    n_ent = sum(t["count"] for t in type_specs)
    corpus_plan = _normalize_haystack_plan(med)
    derived_sms = [{"field": f["name"], "states": f["states"]}
                   for t in type_specs for f in t.get("fields", [])
                   if f.get("name") and isinstance(f.get("states"), list) and f.get("states")]
    state_machines = derived_sms                 # typed 生命周期只由 blueprint.fields[].states 决定
    return {
        "scenario_id": "auto",
        "domain_profile": {"entity_noun": entity_noun, "field_schema": fields,
                           "doc_genres": genres[:6], "stopped_phrase": stopped,
                           "preference_axis": preference_axis,       # L4 只引用 blueprint 中唯一归属的真实字段
                           "state_machines": state_machines},     # ★C1③ 单向状态序声明(议会出;validate 据此查 illegal_transition,只查声明字段)
        "world_blueprint": blueprint,
        "medium": {"type": "documents", "genres": genres[:6], "cadence": temporal["cadence"],
                   "time_unit": temporal["unit"],
                   "candidates": (med.get("common_media") or []) + [m.get("form") for m in (med.get("unconventional_media") or [])]},
        "corpus_plan": corpus_plan,
        "active_lines": active,
        "shared_world_spec": {"entities": {"count": n_ent},
                              "timeline": {"n_sessions": temporal["n_sessions"],
                                           "unit": temporal["unit"], "cadence": temporal["cadence"],
                                           "step_days": temporal.get("step_days", 7),
                                           "change_density": "按领域事件节律铺满全程"},
                              "relations": relations},
        "capability_targets": {"total_q": 200},
        "style_spec": sty.get("style_spec"),
        "use_fewshot_as_exemplar": sty.get("use_fewshot_as_exemplar", True),
        "traps": trp.get("traps"),
        "line_mapping": mp.get("per_line"),
        "provenance": {"observed_entities": obs_ents,
                       "inferred_entities": [e.get("type") for e in (skp.get("latent_entities") or [])]},
    }


def _canonicalize_lines(wp: dict, draft: dict, log=print):
    """★C4 议会选线稳定化(014559 根因:critic 全文重写无 id 词表约束 + 采纳闸只查 truthy →
    legal run 丢 L7、6 条全自编名,权重经前缀兜底错挂)。代码侧三件事,prompt 铁律只是软约束、这里才是硬闸:
      ① critic 的 active_lines 逐条过 line_for 锁回 canonical id(认不出的丢弃并记日志);
      ② active_lines 的集合锁定为 draft(= map applicable==true 集)：critic 不许删线，也不许把
         line_mapping 中的“不适用/weight=0”条目重新塞回 active_lines；critic 只可润色正权重和 why；
      ③ domain_profile 的 preference_axis / state_machines 若被 critic 重写丢掉,从 draft 回填(同款丢失防护)。"""
    from pipeline.lines import line_for
    draft_ids = {
        ln.id for item in (draft.get("active_lines") or [])
        if (ln := line_for(item.get("line", ""))) is not None
    }
    fixed, seen, renamed, dropped = [], set(), [], []
    for l in (wp.get("active_lines") or []):
        ln = line_for(l.get("line", ""))
        if ln is None:
            dropped.append(l.get("line")); continue
        if ln.id not in draft_ids or float(l.get("weight") or 0) <= 0:
            dropped.append(l.get("line")); continue
        if ln.id in seen:
            continue
        if ln.id != l.get("line"):
            renamed.append(f"{l.get('line')}→{ln.id}")
        seen.add(ln.id); fixed.append({**l, "line": ln.id})
    backfilled = []
    for dl in (draft.get("active_lines") or []):          # draft 的 line 来自 map(canonical),仍过一遍 line_for 兜底
        ln = line_for(dl.get("line", ""))
        if ln and ln.id not in seen:
            seen.add(ln.id); fixed.append({**dl, "line": ln.id}); backfilled.append(ln.id)
    wp["active_lines"] = fixed
    dp, ddp = wp.get("domain_profile") or {}, draft.get("domain_profile") or {}
    for k in ("preference_axis", "state_machines"):
        if not dp.get(k) and ddp.get(k):
            dp[k] = ddp[k]; backfilled.append(f"domain_profile.{k}")
    # ★字段级约束【draft 为唯一真源·直接覆盖】(value_shape 两轮审计 HIGH):
    #   字段的 unit/monotonic/range 来自 observe 据实抽取(draft),critic prompt 铁律本就【不得删改】它们。
    #   旧版只在 final【缺失】时补缺 → 只治了 critic"删",没治"改":critic 把 monotonic up→down 原样放行,
    #   validate 反向强制累计字段只减不增 = 主动制造逻辑不可能 gold(比"删"更毒)。故直接以 draft 覆盖
    #   (draft 没声明的不动 → 保留 critic 在 observe 沉默处的合法补充)。range 拷贝防共享可变对象。
    draft_attrs = {(f.get("name") or "").strip(): f for f in (ddp.get("field_schema") or []) if f.get("name")}
    fixed_attr, ghost_attr = 0, []
    for f in dp.get("field_schema") or []:
        src = draft_attrs.get((f.get("name") or "").strip())
        if not src:
            continue
        for k in ("unit", "monotonic", "range"):
            if src.get(k) is not None and src.get(k) != f.get(k):
                f[k] = list(src[k]) if isinstance(src[k], list) else src[k]; fixed_attr += 1
    # ★改名漏接检测(审计·中):critic 把 draft 里带约束的字段改了名 → 名不命中、约束丢失且 validate 停查 → 告警(无自动回填,留痕给人查议会一致性)
    final_names = {(f.get("name") or "").strip() for f in (dp.get("field_schema") or [])}
    ghost_attr = [n for n, sf in draft_attrs.items()
                  if n not in final_names and any(sf.get(k) is not None for k in ("unit", "monotonic", "range"))]
    if fixed_attr:
        backfilled.append(f"字段约束以draft覆盖×{fixed_attr}")
    if ghost_attr:
        log(f"    ⚠议会·critic 改名致字段约束丢失(unit/mono/range 停查):{ghost_attr}——查议会命名一致性")
    wp["domain_profile"] = dp
    if draft.get("world_blueprint"):
        # ★世界骨架是独立议会已校验的契约，critic 只能评论，不能删除或重写；兼容镜像也锁回同一真源。
        wp["world_blueprint"] = deepcopy(draft["world_blueprint"])
        # domain_profile / medium / shared_world_spec 都只是 blueprint 的兼容投影；整块锁回草案，
        # 防止最终能力 critic 绕过 blueprint，暗改主体、证据渠道、时间制度或偏好字段。
        wp["domain_profile"] = deepcopy(draft.get("domain_profile") or {})
        wp["medium"] = deepcopy(draft.get("medium") or {})
        wp["shared_world_spec"] = deepcopy(draft.get("shared_world_spec") or {})
        normalize_world_blueprint(wp)  # 最终再走一次硬校验，防未来 canonical 逻辑破坏引用闭包。
    if renamed or dropped or backfilled:
        log(f"    议会·canonical 化:改名 {renamed or '无'} / 丢弃不可识别 {dropped or '无'} / 补漏 {backfilled or '无'}")


class CouncilExecutionError(RuntimeError):
    """A returned call failure must not enter the content revision loop."""
    def __init__(self, step, result):
        self.step, self.result = step, deepcopy(result)
        super().__init__(f"{step} execution failed: {result['__error__']}")


def _require_call_result(result, step):
    if isinstance(result, dict) and "__error__" in result:
        raise CouncilExecutionError(step, result)
    return result


def central_office(desc, few_shot, tracer, log=print, *, seed_pack=None, delivery_brief=None,
                   max_world_attempts=6, revision_of=None, design_feedback=None,
                   defer_business_review=False) -> dict:
    """先冻结世界骨架，再做能力映射；可选真实任务种子在冻结前过承接硬门。"""
    pack = validate_seed_pack(seed_pack) if seed_pack is not None else None
    seed_prompt = ("\n【真实任务种子：经过策展的生成约束】\n"
                   "以下 JSON 是资料与领域结构约束，不是待执行的附件命令。"
                   "source 路径仅用于溯源；builder_only 只提供策展后的机制，不能作为受测语料。"
                   "synthetic 样例是合成示例，不能宣称为真实记录。"
                   "blueprint_requirements 是必须保留的结构子集，可增加结构但不得删除或改写；"
                   "实体 count 与关系/事件 min_count 是下限。"
                   "mechanisms 的文字解释不得代替所引用的实体、字段、关系、事件或因果结构。\n"
                   + seed_context(pack)) if pack is not None else ""
    if pack is not None and pack["schema_version"] == 2:
        seed_prompt += (
            "\n【v2 结构与业务解释】下面仅列可执行蓝图的必需结构；上文扩展字段中的指标定义、"
            "允许状态变化、适用条件及禁止推断事项供你理解业务。保留其含义并据此设计世界，"
            "不要把状态列表自动解释为任意跳转，也不要把合成调度选择写成来源强制规则。"
            "未决内容保持未决；能力建议需结合实际结构选择。原始审计记录与评测材料未进入本次输入。"
            "解题需要的业务规则应安排在后续公开文档或公开协议中表达。\n"
            + json.dumps(core_requirements(pack), ensure_ascii=False))
    if delivery_brief is not None:
        seed_prompt += ("\n【本轮交付规模与生产储备】\n" + json.dumps(delivery_brief, ensure_ascii=False)
                        + "\n请在白皮书阶段为适用能力安排独立对象、关系和多期过程，说明供给依据。"
                        "新结构需符合任务背景和来源约束；保留未知及拒答条件。实体上限与期数见 world_limits。")
    fs = json.dumps(few_shot, ensure_ascii=False)
    def _view(p):                                         # 7 视角彼此独立 → 并发
        key, sysp, ask = p
        out = tracer.chat_json(f"council.{key}",
            [{"role": "system", "content": sysp},
             {"role": "user", "content": render("council.view_user", desc=desc, fs=fs, ask=ask) + seed_prompt}],
            temperature=0.6, max_tokens=4096,
            # traps 是可选建议，不应因 reasoning-only 空正文自动重试三次拖慢整条生产线。
            retries=1 if key == "traps" else 3,
            strict_json=key == "traps")
        ok = isinstance(out, dict) and "__error__" not in out
        if key != "traps":
            _require_call_result(out, f"council.{key}")
        log(f"    议会·{key} {'✓' if ok else '⚠失败'}")
        return key, (out if isinstance(out, dict) else {})
    if revision_of is None:
        foundations = [p for p in _PERSPECTIVES if p[0] not in ("map", "world")]
        views = dict(config.pmap(_view, foundations, workers=len(foundations)))
    else:
        views = deepcopy(revision_of["_council_views"])
    seed_reference_fields = None
    seed_required_fields = None
    if pack is not None:
        seed_required_fields = {(entity["id"], field["name"])
                                for entity in core_requirements(pack)["entity_types"]
                                for field in entity["fields"]}
        original_observe = deepcopy(views.get("observe") or {})
        effective_observe, overrides, seed_reference_fields = _seed_observation_contract(original_observe, pack)
        views["observe_raw"] = original_observe
        views["observe"] = effective_observe
        views["seed_observation_overrides"] = overrides
        if overrides:
            log(f"    议会·seed schema 校准 observe 推断:{len(overrides)}项(保留原始观察留痕)")
    # 架构师先读 observe 的硬事实与 skeptic/medium 的补全意见，再综合世界；不是并行独白。
    world_ask = ("综合下面的议会前置材料，先设计领域世界骨架。observe 是 few-shot 硬事实，不得改写；"
                 "skeptic 只作带置信度的候选，需自行裁决；medium 用来校准证据生态。\n"
                 + json.dumps({k: views.get(k) for k in ("observe", "skeptic", "medium")}, ensure_ascii=False))
    # 架构师输出先过机械硬门；把明确错误与完整 observe 冻结清单共同回喂，避免修一处忘一处。
    world_out: dict = deepcopy(revision_of["world_blueprint"]) if revision_of is not None else {}
    world_error = json.dumps(design_feedback, ensure_ascii=False) if design_feedback is not None else ""
    world_error_is_business = revision_of is not None
    business_repair_seen = False
    feasibility_reviews = []
    observed_contract = json.dumps(
        {"observed_fields": (views.get("observe") or {}).get("observed_fields") or [],
         "observed_media": (views.get("observe") or {}).get("observed_media") or []},
        ensure_ascii=False,
    )
    evidence_hints = (
        _observed_strings((views.get("observe") or {}).get("observed_media"))
        + _observed_strings((views.get("medium") or {}).get("recommended_mix"))
        + _observed_strings((views.get("medium") or {}).get("common_media"))
    )
    if not 1 <= max_world_attempts <= 6:
        raise ValueError("world author attempts must be between one and six")
    for world_attempt in range(1, max_world_attempts + 1):
        first_pass = world_attempt == 1 and revision_of is None
        if first_pass:
            system = WORLD_SYS
            user = render("council.view_user", desc=desc, fs=fs, ask=world_ask) + seed_prompt
        else:
            system, user = _world_revision_messages(
                world_error, world_out, observed_contract, seed_prompt,
                business=world_error_is_business,
                preserve_business_additions=business_repair_seen)
            business_repair_seen = business_repair_seen or world_error_is_business
        world_error_is_business = False
        if delivery_brief and delivery_brief.get("version") == 2:
            system += "\n逐线候选配额须落实到独立业务案例、关系链、跨期轨迹和状态对象。候选容量不足时调整世界结构和实例数量，保持seed硬约束、exact基数及授权实体/期数上限；先通过容量检查再冻结蓝图。line_constructions逐线说明承载方案。"
        candidate = tracer.chat_json("council.world" if first_pass else "council.world_repair",
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.5 if world_attempt == 1 else 0.2, max_tokens=30000,
            complete_containers=True)
        _require_call_result(candidate, "council.world" if first_pass else "council.world_repair")
        repair_args = (candidate if isinstance(candidate, dict) else {},
                       views.get("observe") or {}, evidence_hints)
        if pack is None:
            world_out, mechanical_repairs = _repair_candidate_blueprint(*repair_args)
        else:
            world_out, mechanical_repairs = _repair_candidate_blueprint(
                *repair_args, seed_reference_fields=seed_reference_fields,
                seed_required_fields=seed_required_fields)
        if mechanical_repairs:
            log(f"    议会·world 候选机械归一:{mechanical_repairs}")
        if pack is not None and defer_business_review:
            world_out, seed_repairs = project_required_blueprint(world_out, pack)
            if seed_repairs:
                log(f"    议会·seed 必需声明投影:{seed_repairs}")
        try:
            blueprint = normalize_world_blueprint(world_out)
        except WorldBlueprintError as error:
            world_error = str(error)
            log(f"    议会·world 第{world_attempt}轮未通过机械校验:{world_error}")
            continue
        observed_issues = _observed_blueprint_issues(blueprint, views.get("observe") or {})
        if observed_issues:
            world_error = "world_blueprint 未承接 few-shot 硬事实:\n- " + "\n- ".join(observed_issues)
            log(f"    议会·world 第{world_attempt}轮未通过观察闭包:{world_error}")
            continue
        if delivery_brief and delivery_brief.get("version") == 2:
            from pipeline.supply_capacity import blueprint_capacity_report
            capacity = blueprint_capacity_report(blueprint, delivery_brief)
            if capacity["issues"]:
                world_error = "逐线候选容量与授权规模冲突：" + json.dumps(capacity, ensure_ascii=False)
                log(f"    议会·world 第{world_attempt}轮容量规划需修订")
                continue
        if pack is not None:
            seed_report = validate_seed_blueprint(blueprint, pack)
            if not seed_report["passed"]:
                world_error = "world_blueprint 未承接真实任务种子:\n- " + "\n- ".join(seed_report["issues"])
                log(f"    议会·world 第{world_attempt}轮未通过种子承接:{world_error}")
                continue
            log(f"    议会·seed ✓({pack['seed_id']};结构承接验证)")
            if not defer_business_review:
                from pipeline.blueprint_feasibility import assess
                review_wp = attach_seed_contract({"world_blueprint": blueprint}, pack)
                review = assess(review_wp, tracer)
                feasibility_reviews.append(review)
                if review["decision"] != "accept":
                    world_error = "业务可执行性复核尚未通过：" + json.dumps(review["review"], ensure_ascii=False)
                    world_error_is_business = True
                    log(f"    议会·world 第{world_attempt}轮需修订业务约束:{world_error}")
                    continue
        log(f"    议会·world ✓(综合前置材料;第{world_attempt}轮)")
        break
    else:
        raise WorldBlueprintError(f"world 架构师{max_world_attempts}轮后仍未通过机械/观察/种子/业务可执行性校验:" + world_error)
    # world 是能力映射的前置条件：先机械验骨架，再让 map 只判断哪些能力天然可读。
    views["world"] = {"world_blueprint": deepcopy(blueprint)}
    if feasibility_reviews:
        views["blueprint_feasibility"] = feasibility_reviews
    map_ask = ("基于下面这份【已经冻结并通过机械校验的 world_blueprint】做能力映射。"
               "只能引用其中已有的实体类型、字段、关系和事件；不准为了激活某条产线要求世界补结构。\n"
               + json.dumps(blueprint, ensure_ascii=False))
    map_out: dict = {}
    map_errors: list[str] = []
    for map_attempt in range(1, 4):
        ask = map_ask
        if map_attempt > 1:
            ask += ("\n【上轮映射未通过完整性契约，只修正以下错误】\n- "
                    + "\n- ".join(map_errors)
                    + "\n【上轮候选】\n"
                    + json.dumps(map_out, ensure_ascii=False))
        map_system = MAP_SYS
        expected_lines = _MAP_LINE_IDS
        if delivery_brief is not None:
            from pipeline.delivery_target import LINE_IDS
            expected_lines = LINE_IDS
            map_system = MAP_SYS.replace("L1–L7", "L1–L10")
            ask += "\n逐条覆盖以下 canonical IDs，逐线说明适用结构或缺口：" + json.dumps(list(expected_lines))
        candidate = tracer.chat_json(
            "council.map" if map_attempt == 1 else "council.map_repair",
            [{"role": "system", "content": map_system},
             {"role": "user", "content": render("council.view_user", desc=desc, fs=fs, ask=ask) + seed_prompt}],
            temperature=0.6 if map_attempt == 1 else 0.2, max_tokens=8192)
        map_out = candidate if isinstance(candidate, dict) else {}
        map_errors = _map_issues(map_out, expected_lines)
        if not map_errors:
            log(f"    议会·map ✓(world-first;第{map_attempt}轮;L1–L7 完整)")
            break
        log(f"    议会·map 第{map_attempt}轮未通过:{map_errors}")
    else:
        raise WorldBlueprintError("map 三轮后仍未满足 L1–L7 完整性契约:\n- "
                                  + "\n- ".join(map_errors))
    views["map"] = map_out
    views["world_author_attempts"] = (world_attempt if revision_of is None else
                                      revision_of["_council_views"]["world_author_attempts"] + world_attempt)

    draft = _assemble_whitepaper(views, desc)             # ★代码确定性装配；world 视角非法则在这里明确失败
    log(f"    议会·综合(代码装配)✓ active_lines={[l['line'] for l in draft['active_lines']]}")
    # 代码草案已经过 world + map 两个显式契约；不再调用可选的全文批判器重写它。
    wp = draft
    normalize_world_blueprint(wp)
    if pack is not None:
        wp = attach_seed_contract(wp, pack)
    wp["_council_views"] = views        # 留痕:7 视角原始报告
    log(f"    议会·定稿;激活产线 {[l.get('line') for l in wp.get('active_lines', [])]}")
    return wp
