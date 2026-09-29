"""
pipeline.closed_loop —— §S 闭环旋钮 driver(从 run_factory_v2 拆出,行为不变)。
反推世界规模/配额 → ①供给环 → 渲 → 接地 → ②实测纠偏。4 个纯决策函数 + build_to_target 循环控制。
★build_to_target 内部【惰性 import】factory 的 stage(断开 factory↔closed_loop 导入环)。
"""
from __future__ import annotations
import dataclasses, hashlib, math, time
from copy import deepcopy
from pipeline.world_state import WorldState
from pipeline.world_blueprint import (WorldBlueprintError, event_role_requirements,
                                      relation_capacity, relation_owner_side)
from pipeline.lines import line_for
from pipeline.run import Run, _finalize_run, _run_stage, _run_stage_locked, _update_run_metadata
from pipeline.targetspec import TargetSpec, invert_rate, WorldParams, DEFAULT_SLACK


def _relation_key(relation: dict) -> tuple:
    """关系实例的稳定增量键；新契约要求 id 唯一，旧产物则退回结构键。"""
    return ("id", relation.get("id")) if relation.get("id") else (
        "edge", relation.get("type"), relation.get("from"), relation.get("to"), relation.get("session", 0))


def _event_key(event: dict) -> tuple:
    """领域事件实例的稳定增量键。"""
    return ("id", event.get("id")) if event.get("id") else (
        "event", event.get("type"), event.get("session"), repr(event.get("participants")))


def _seed_type_floors(wp: dict) -> dict[str, int]:
    requirements = (wp.get("seed_contract") or {}).get("blueprint_requirements") or {}
    return {item["id"]: item.get("count", 1)
            for item in requirements.get("entity_types", [])}


def _render_delta_scope(ws: WorldState, previous_entities: set,
                        previous_relations: set, previous_events: set) -> tuple[list[str], list[list]]:
    """计算增量语料范围：新实体全程，结构事件触碰的旧实体只补对应 session。"""
    new_entities = sorted(set(ws.entities) - previous_entities)
    new_set = set(new_entities)
    pairs: set[tuple[str, int]] = set()
    blueprint = ws.world_blueprint or {}
    relation_types = {r.get("id"): r for r in blueprint.get("relation_types", [])
                      if isinstance(r, dict) and r.get("id")}
    for relation in ws.relations:
        if _relation_key(relation) in previous_relations:
            continue
        owner = relation.get("from")
        declaration = relation_types.get(relation.get("type")) or {}
        if relation_owner_side(blueprint, declaration) == "to":
            owner = relation.get("to")
        # 历史世界、缺声明或字段归属不唯一时保持 source 回退。
        if owner and owner not in new_set:
            pairs.add((owner, int(relation.get("session", 0))))
    for event in ws.events:
        if _event_key(event) in previous_events:
            continue
        session = int(event.get("session", 0))
        for effect in event.get("effects") or []:
            entity = effect.get("entity")
            if entity and entity not in new_set:
                pairs.add((entity, session))
    return new_entities, [[entity, session] for entity, session in sorted(pairs)]


def _scale_world_contract(wp: dict, n_entities: int, n_sessions: int,
                          narrative: bool = False) -> None:
    """同步闭环规模旋钮到 legacy 镜像与显式 world_blueprint。

    多类型世界按白皮书原有人口比例缩放到闭环反推的目标规模；只有显式声明
    ``cardinality_policy=exact`` 的唯一角色保持原 count。
    relation/event 最小实例数同比缩放；没有显式基数约束的 primary 仍可正常扩容。
    没有 blueprint 的历史白皮书仍只改旧字段。
    """
    sw = wp.setdefault("shared_world_spec", {})
    bp = wp.get("world_blueprint")
    actual_entities, actual_sessions = n_entities, n_sessions
    if isinstance(bp, dict) and bp.get("entity_types"):
        types = bp["entity_types"]
        old_total = sum(int(t.get("count", 0)) for t in types)
        type_ids = [str(t.get("id") or "") for t in types]
        baseline = sw.get("_typed_scale_baseline")
        baseline_valid = (
            isinstance(baseline, dict)
            and isinstance(baseline.get("entity_counts"), dict)
            and set(baseline["entity_counts"]) == set(type_ids)
            and int(baseline.get("entity_total", 0)) > 0
        )
        if not baseline_valid:
            baseline = {
                "entity_total": old_total,
                "entity_counts": {str(t.get("id") or ""): int(t.get("count", 0)) for t in types},
                "relation_min_counts": {
                    str(item.get("id") or ""): int(item.get("min_count", 0))
                    for item in bp.get("relation_types") or []
                },
                "event_min_counts": {
                    str(item.get("id") or ""): int(item.get("min_count", 0))
                    for item in bp.get("event_types") or []
                },
            }
            sw["_typed_scale_baseline"] = baseline
        base_counts = [int(baseline["entity_counts"][tid]) for tid in type_ids]
        fixed = [i for i, item in enumerate(types)
                 if item.get("cardinality_policy") == "exact"]
        scalable = [i for i in range(len(types)) if i not in fixed]
        fixed_total = sum(base_counts[i] for i in fixed)
        desired = max(int(n_entities), fixed_total + len(scalable))
        if not fixed:
            # 未声明基数约束时保持旧行为，避免 office 等人口型 primary 被意外冻结。
            factor = desired / int(baseline["entity_total"])
            raw = [count * factor for count in base_counts]
            counts = [max(1, math.floor(value)) for value in raw]
            remainder = desired - sum(counts)
            order = sorted(range(len(types)), key=lambda i: (raw[i] - math.floor(raw[i]),
                                                              bool(types[i].get("primary"))), reverse=True)
            for i in order[:max(0, remainder)]:
                counts[i] += 1
        else:
            counts = list(base_counts)
            scalable_target = desired - fixed_total
            scalable_base = sum(base_counts[i] for i in scalable)
            free_slots = scalable_target - len(scalable)
            raw = [(base_counts[i] * free_slots / scalable_base) if scalable_base > 0
                   else (free_slots / len(scalable)) for i in scalable]
            scaled = [1 + math.floor(value) for value in raw]
            remainder = scalable_target - sum(scaled)
            order = sorted(range(len(scalable)),
                           key=lambda pos: raw[pos] - math.floor(raw[pos]), reverse=True)
            for pos in order[:max(0, remainder)]:
                scaled[pos] += 1
            for i, count in zip(scalable, scaled):
                counts[i] = count
        seed_floors = _seed_type_floors(wp)
        for t, count in zip(types, counts):
            t["count"] = max(count, seed_floors.get(t.get("id"), 0))
        # Rounding by proportions and then applying seed floors can overfill a
        # feasible target. Reclaim only scalable surplus; preserve every type,
        # seed minimum, exact cardinality and distinct-event-role requirement.
        role_floors = event_role_requirements(bp)
        floors = {i: max(1, seed_floors.get(types[i].get("id"), 0),
                         role_floors.get(types[i].get("id"), 0)) for i in scalable}
        excess = sum(int(t["count"]) for t in types) - desired
        donors = sorted(scalable, key=lambda i: (int(types[i]["count"]) - floors[i],
                                                 int(types[i]["count"])), reverse=True)
        for i in donors:
            take = min(max(0, excess), max(0, int(types[i]["count"]) - floors[i]))
            types[i]["count"] -= take
            excess -= take
        actual_entities = sum(int(t.get("count", 0)) for t in types)
        temporal = bp.setdefault("temporal_model", {})
        # 时间跨度与实体规模一样服从本轮旋钮；build_to_target 后续轮只会放大参数，
        # 因而不会把已经生成的世界反向缩短。
        seed_requirements = (wp.get("seed_contract") or {}).get("blueprint_requirements") or {}
        seed_temporal = seed_requirements.get("temporal_model") or {}
        temporal["n_sessions"] = max(2, int(n_sessions), seed_temporal.get("min_sessions", 0))
        actual_sessions = temporal["n_sessions"]
        density_factor = actual_entities / int(baseline["entity_total"])
        for key, declarations in (("relation_min_counts", bp.get("relation_types") or []),
                                  ("event_min_counts", bp.get("event_types") or [])):
            base_mins = baseline.get(key) or {}
            for declaration in declarations:
                declaration_id = str(declaration.get("id") or "")
                base_min = int(base_mins.get(declaration_id, declaration.get("min_count", 0)))
                if base_min > 0:
                    # 剧情事件是 canonical spine，不随为了补 benchmark 题而扩的外围实体膨胀。
                    scaled_min = base_min if narrative and key == "event_min_counts" \
                        else math.ceil(base_min * density_factor)
                    seed_key = "relation_types" if key == "relation_min_counts" else "event_types"
                    seed_min = next((item.get("min_count", 1)
                                     for item in seed_requirements.get(seed_key, [])
                                     if item["id"] == declaration_id), 1)
                    declaration["min_count"] = max(1, scaled_min, seed_min)
        # relation 是标量 FK；人口比例取整后，owner 侧可能没长大，不能让同比 min_count 超过容量。
        for relation in bp.get("relation_types") or []:
            if relation_owner_side(bp, relation) is not None:
                relation["min_count"] = min(int(relation.get("min_count", 0)),
                                             relation_capacity(bp, relation))
        # event 实例唯一键是 type/session/participants；最小数不得超过角色组合容量。
        counts_by_type = {item.get("id"): int(item.get("count", 0)) for item in types}
        for event in bp.get("event_types") or []:
            capacity = actual_sessions
            for type_id in (event.get("roles") or {}).values():
                capacity *= counts_by_type.get(type_id, 0)
            event["min_count"] = min(int(event.get("min_count", 0)), capacity)
        actual_sessions = temporal["n_sessions"]
    sw.setdefault("entities", {})["count"] = actual_entities
    sw.setdefault("timeline", {})["n_sessions"] = actual_sessions


def _ensure_l7_capacity(wp: dict, floor: int) -> dict:
    """在总实体数不变时，为 L7 分配足够的内在非单调数值字段实例。"""
    bp = wp.get("world_blueprint") or {}
    types = bp.get("entity_types") or []
    if floor <= 0 or not types:
        return {"before": 0, "after": 0, "moved": {}}

    structural: set[tuple[str, str]] = set()
    if wp.get("seed_contract"):
        from pipeline.seed_world import seed_protected_fields
        structural.update(seed_protected_fields(wp))
    for relation in bp.get("relation_types") or []:
        side = relation_owner_side(bp, relation)
        owner = relation.get("from_type") if side == "from" else relation.get("to_type")
        if owner and relation.get("field"):
            structural.add((owner, relation["field"]))
    for event in bp.get("event_types") or []:
        roles = event.get("roles") or {}
        for effect in event.get("effect_fields") or []:
            owner = roles.get(effect.get("role"))
            if owner and effect.get("field"):
                structural.add((owner, effect["field"]))

    widths = {}
    role_floors = event_role_requirements(bp)
    for type_id, seed_floor in _seed_type_floors(wp).items():
        role_floors[type_id] = max(role_floors.get(type_id, 0), seed_floor)
    for index, entity_type in enumerate(types):
        if entity_type.get("cardinality_policy") == "exact":
            continue
        type_id = entity_type.get("id")
        widths[index] = sum(
            field.get("kind") == "numeric"
            and field.get("monotonic") not in ("up", "down")
            and (type_id, field.get("name")) not in structural
            for field in entity_type.get("fields") or []
        )
    recipients = [index for index, width in widths.items() if width > 0]
    before = sum(int(types[index].get("count", 0)) * width for index, width in widths.items())
    capacity = before
    moved: dict[str, int] = {}
    while capacity < floor and recipients:
        recipient = max(recipients, key=lambda index: (widths[index], -int(types[index].get("count", 0))))
        donors = [
            index for index, entity_type in enumerate(types)
            if index != recipient
            and entity_type.get("cardinality_policy") != "exact"
            and int(entity_type.get("count", 0)) > max(1, role_floors.get(entity_type.get("id"), 0))
            and widths.get(index, 0) < widths[recipient]
        ]
        if not donors:
            break
        donor = max(donors, key=lambda index: (
            widths[recipient] - widths.get(index, 0),
            int(types[index].get("count", 0)),
            not bool(types[index].get("primary")),
        ))
        types[recipient]["count"] = int(types[recipient].get("count", 0)) + 1
        types[donor]["count"] = int(types[donor].get("count", 0)) - 1
        capacity += widths[recipient] - widths.get(donor, 0)
        moved[str(types[recipient].get("id"))] = moved.get(str(types[recipient].get("id")), 0) + 1
        moved[str(types[donor].get("id"))] = moved.get(str(types[donor].get("id")), 0) - 1

    # 类型人口重分配后重新夹住结构声明容量。
    for relation in bp.get("relation_types") or []:
        if relation_owner_side(bp, relation) is not None:
            relation["min_count"] = min(int(relation.get("min_count", 0)),
                                         relation_capacity(bp, relation))
    counts = {item.get("id"): int(item.get("count", 0)) for item in types}
    n_sessions = int((bp.get("temporal_model") or {}).get("n_sessions", 1))
    for event in bp.get("event_types") or []:
        event_capacity = n_sessions
        for type_id in (event.get("roles") or {}).values():
            event_capacity *= counts.get(type_id, 0)
        event["min_count"] = min(int(event.get("min_count", 0)), event_capacity)
    wp.setdefault("domain_profile", {})["l7_max_trends"] = max(
        int(wp.get("domain_profile", {}).get("l7_max_trends", 0)), int(floor))
    return {"before": before, "after": capacity, "moved": moved}


def _ensure_event_role_capacity(wp: dict) -> dict[str, int]:
    """总实体数不变地保证同一事件的不同角色可由互异实体承担。"""
    bp = wp.get("world_blueprint") or {}
    types = bp.get("entity_types") or []
    required = event_role_requirements(bp)
    for type_id, seed_floor in _seed_type_floors(wp).items():
        required[type_id] = max(required.get(type_id, 0), seed_floor)
    moved: dict[str, int] = {}
    for recipient in types:
        type_id = recipient.get("id")
        floor = required.get(type_id, 0)
        while int(recipient.get("count", 0)) < floor:
            if recipient.get("cardinality_policy") == "exact":
                raise WorldBlueprintError(
                    f"event 需要 {floor} 个互异 {type_id}，但 exact count={recipient.get('count')}")
            donors = [
                item for item in types
                if item is not recipient
                and item.get("cardinality_policy") != "exact"
                and int(item.get("count", 0))
                    > max(1, required.get(item.get("id"), 0))
            ]
            if not donors:
                raise WorldBlueprintError(
                    f"总实体数内无法为 event 分配 {floor} 个互异 {type_id}")
            donor = max(donors, key=lambda item: int(item.get("count", 0)))
            recipient["count"] = int(recipient.get("count", 0)) + 1
            donor["count"] = int(donor.get("count", 0)) - 1
            moved[str(type_id)] = moved.get(str(type_id), 0) + 1
            moved[str(donor.get("id"))] = moved.get(str(donor.get("id")), 0) - 1
    return moved


def _orders_by_line(orders) -> dict:
    """{line_id: 该线产了多少 order}。"""
    out: dict = {}
    for o in orders:
        out[o.get("line", "?")] = out.get(o.get("line", "?"), 0) + 1
    return out


def _order_shortfall(produced: dict, required_orders: dict) -> dict:
    """计算所有显式产线硬下限的订单缺口。"""
    return {
        lid: floor - produced.get(lid, 0)
        for lid, floor in required_orders.items()
        if produced.get(lid, 0) < floor
    }


def _order_deficit(produced: dict, required_orders: dict, feasible: set) -> dict:
    """①环判据：只在可行产线没有达到硬下限时扩世界。

    ``target_orders`` 含 survival 与 slack，是为了提高一次接地成功率的软生产量；
    它不能升级成重建契约。这里直接比较调用者的 ``per_line_min``，不设容差，
    也不让不可行线触发无意义扩容。
    """
    return {lid: amount for lid, amount in _order_shortfall(produced, required_orders).items()
            if lid in feasible}


def _grow_for_supply(params: WorldParams, deficit: dict, grow_sessions: bool = True) -> WorldParams:
    """①环成长:【实质】供不上才长世界,**按缺口比例温和补**实体(不再 ×1.4 一刀切)。夹 clamp。无赤字原样返回。
    粗率 ~0.4 单/实体(同 invert_rate 的 RATE_L3);缺口大才顺带加几周。"""
    from pipeline.targetspec import N_ENT_CLAMP, N_SESS_CLAMP
    if not deficit:
        return params
    short = sum(deficit.values())                                   # 实质短缺总单数
    add_ent = max(3, math.ceil(short / 0.4))                        # 按缺口比例补实体(0.4 单/实体粗率)
    n_ent = min(N_ENT_CLAMP[1], params.n_entities + add_ent)
    n_sess = (min(N_SESS_CLAMP[1], params.n_sessions + 2)
              if grow_sessions else params.n_sessions)             # 增量纠偏轮锁住旧语料的时间轴
    return dataclasses.replace(params, n_entities=n_ent, n_sessions=n_sess)


def _floor_status(by_line: dict, overall: dict, spec: TargetSpec, feasible: set):
    """②环判据。返回 (met, per_line_final{line:接地数}, growable[低于floor的可行线], permanent[有floor但不可行的线])。
    可行线低于 floor → 放大重渲也许能救(growable);不可行线 → 永久达不到(permanent,别空转)。
    met = 各线 floor 全满足(growable+permanent 皆空) 且 总数 ≥ min_questions。"""
    per_line_final = {lid: by_line.get(lid, {}).get("grounded", 0) for lid in spec.per_line_min}
    growable, permanent = [], []
    for lid, floor in spec.per_line_min.items():
        got = by_line.get(lid, {}).get("grounded", 0)
        if got >= floor:
            continue
        (growable if lid in feasible else permanent).append(f"{lid}:{got}/{floor}")
    total_ok = overall.get("grounded", 0) >= spec.min_questions
    met = (not growable) and (not permanent) and total_ok
    return met, per_line_final, growable, permanent


def _resolve_per_line_contract(active: list[str], requested: dict) -> dict:
    """规范化显式产线下限；未实现或未激活时直接报错，不得静默丢弃。"""
    resolved: dict[str, int] = {}
    unknown: list[str] = []
    inactive: list[str] = []
    for raw_id, floor in requested.items():
        line = line_for(raw_id)
        if line is None:
            unknown.append(str(raw_id))
            continue
        if line.id not in active:
            inactive.append(line.id)
            continue
        resolved[line.id] = int(floor)
    if unknown or inactive:
        parts = []
        if unknown:
            parts.append(f"未实现/不可识别={sorted(set(unknown))}")
        if inactive:
            parts.append(f"白皮书未激活={sorted(set(inactive))}")
        raise WorldBlueprintError("显式 per_line_min 与本轮执行计划冲突:"
                                  + "；".join(parts))
    return resolved


def _run_world_attempt(run: Run, wp: dict, stage_world, artifacts: dict, corpus_checkpoint: str):
    """Keep the published contract/world/opinion together through stage marking.

    The original stage may commit successfully before its final manifest mark
    fails. Hold the ordinary Run lock across that window and roll back only the
    published bundle and algorithm metadata; attempted drafts, opinions and
    provider traces remain available for audit.
    """
    from pipeline.world_semantics import REVIEW_ARTIFACT, WARNING_ARTIFACT
    with run.stage_write_lock("world"):
        run._reload_manifest_for_stage()
        names = (artifacts["whitepaper"], "01_seed_audit.json", artifacts["world"], REVIEW_ARTIFACT,
                 WARNING_ARTIFACT,
                 "02_world_repair_failure.json", "02_seed_audit.json", corpus_checkpoint)
        previous = {name: (run.dir / name).read_bytes() if (run.dir / name).is_file() else None
                    for name in names}
        prior_algo = deepcopy(run.manifest.get("algo", {}))
        started = time.time()
        try:
            run.write(artifacts["whitepaper"], wp)
            _run_stage_locked(run, "world", stage_world, artifacts["world"])
            # An explicit, audited upstream constraint revision may have been
            # accepted while constructing this unpublished world. Keep later
            # supply/grounding rounds on the actually published definition.
            actual_wp = run.read(artifacts["whitepaper"])
            wp.clear()
            wp.update(actual_wp)
        except BaseException as exc:
            for name, content in previous.items():
                path = run.dir / name
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    restore = path.with_name("." + path.name + ".closed-loop-restore")
                    try:
                        restore.write_bytes(content)
                        restore.replace(path)
                    finally:
                        restore.unlink(missing_ok=True)
            run.manifest["algo"] = prior_algo
            # mark() may have changed the in-memory or on-disk stage to done
            # before raising. Retain this attempt's failure, not old success.
            state = run.manifest.setdefault("stages", {}).setdefault("world", {})
            for key in ("ts", "finished_ts"):
                state.pop(key, None)
            run.fail_stage("world", exc, time.time() - started)
            raise


def build_to_target(run: Run, spec: TargetSpec, max_rounds: int = 2, order_subrounds: int = 2):
    """§S 闭环旋钮 v0。前置:input + whitepaper 已由 drive 跑完(本函数从 01_whitepaper.json 起,自管 world→grounding)。
    ★只管【循环控制】:反推规模/配额 → patch 白皮书+config → 直调【现有 stage 函数】(经 _run_stage)→ 读回产物判 floor。
      stage 体零复制(评审:避免第二条编排路径漂移);quota 经 run.config 喂给 stage_orders。
    一轮 = ①订单供给环(stage_world+stage_orders 多子轮长世界) → 出题 → 整轮重渲 → 接地 → ② floor 校验/纠偏。
    达标 MET 才发布；耗尽 max_rounds 或硬下限不可行时保留审计产物并明确失败。"""
    from pipeline.factory import (stage_world, stage_disclosure, stage_orders, stage_well_posed,
                                  run_generation_tail,
                                  ART, CORPUS_CKPT)
    wp = run.read(ART["whitepaper"])
    canon_lines = [{**l, "line": line_for(l.get("line", "")).id}                  # ★白皮书线 id 变体 → 规范 id(与 by_line / 先验键对齐)
                   for l in wp.get("active_lines", [])
                   if line_for(l.get("line", "")) and float(l.get("weight") or 0) > 0]
    active = list(dict.fromkeys(l["line"] for l in canon_lines))                  # 已建且去重的激活线
    explicit_per_line = bool(spec.per_line_min)
    if not explicit_per_line and not spec.total_only:                              # 默认保留逐线硬下限；total_only 只强制总量和显式下限
        spec.per_line_min = spec.derive_per_line_min(canon_lines)
    spec.per_line_min = _resolve_per_line_contract(active, spec.per_line_min)
    s = sum(spec.per_line_min.values())                                           # ★总下限盈余无处落 → 按比例摊进各线 floor
    if 0 < s < spec.min_questions and not spec.total_only:                         #   (使「逐线 floor 全达 ⟹ 总数达标」,消掉"总数单独差"的歧义态)
        scale = spec.min_questions / s
        spec.per_line_min = {k: math.ceil(v * scale) for k, v in spec.per_line_min.items()}
    production_spec = (dataclasses.replace(spec, per_line_min=spec.derive_per_line_min(canon_lines))
                       if spec.total_only else spec)
    params = invert_rate(production_spec)
    profile = wp.get("domain_profile", {})
    run.log(f"╔═ 闭环旋钮 build_to_target:min_q={spec.min_questions} per_line_min={spec.per_line_min}")
    run.log(f"║  invert_rate(保守先验,slack={DEFAULT_SLACK}) → n_ent={params.n_entities} n_sess={params.n_sessions} "
            f"target_orders={params.target_orders}")

    last = {"kept": [], "report": {}, "growable": [], "permanent": []}
    met = False
    run_cfg = run.manifest["config"]
    run_cfg["haystack_ratio"] = spec.haystack_ratio
    prev_entities = None                                                         # round1 已渲世界快照，作 ②环 delta 基线
    prev_relations: set = set()
    prev_events: set = set()
    narrative_mode = run.scenario == "game"
    production_failure = None
    try:
        for rnd in range(1, max_rounds + 1):
            run.log(f"╠═ 第 {rnd}/{max_rounds} 轮 ══════════════════════════════")
            _scale_world_contract(wp, params.n_entities, params.n_sessions,
                                  narrative=narrative_mode)                           # patch 规模旋钮；显式蓝图同步更新 primary/time
            if sum(int(t.get("count", 0)) for t in (wp.get("world_blueprint") or {}).get("entity_types", [])) > spec.max_world_entities:
                allocated = sum(int(t.get("count", 0)) for t in wp["world_blueprint"]["entity_types"])
                raise WorldBlueprintError(
                    f"World allocation has {allocated} entities after seed/type constraints; "
                    f"max_world_entities={spec.max_world_entities}. Check required type/cardinality minima.")
            event_role_moves = _ensure_event_role_capacity(wp)
            if event_role_moves:
                run.log(f"║  事件角色互异容量：总实体不变，类型重分配 {event_role_moves}")
            l7_capacity = _ensure_l7_capacity(
                wp, int(spec.per_line_min.get("L7_consolidation", 0)))
            if l7_capacity["moved"]:
                run.log(f"║  L7 基质容量 {l7_capacity['before']}→{l7_capacity['after']}，"
                        f"总实体不变，类型重分配 {l7_capacity['moved']}")
            sw = wp["shared_world_spec"]
            wp.setdefault("domain_profile", {})["l5_max_conflicts"] = params.max_n_conflicts
            run_cfg["quotas"] = dict(params.target_orders)                            # ★经 config 喂配额给 stage_orders(不内联 run_lines)
            # 普通场景续长实体；game 的 canon 很小，整轮重建比拼接新旧剧情更可靠。
            run_cfg["augment"] = (rnd > 1 and not narrative_mode)

            # ── ① 订单供给环:直调 stage_world + stage_orders(便宜,绝不渲);供不上 → 长世界重跑 ──
            feasible: set = set()
            for sub in range(1, order_subrounds + 1):
                _run_world_attempt(run, wp, stage_world, ART, CORPUS_CKPT)
                _run_stage(run, "disclosure", stage_disclosure, ART["disclosure"])
                _run_stage(run, "orders", stage_orders, ART["orders"])
                _run_stage(run, "well_posed", stage_well_posed, "03_well_posed_report.json")  # ★边A闸:赤字按【良定义后】供给算
                ws = WorldState.from_dict(run.read(ART["world"]))
                feasible = {ln.id for a in active if (ln := line_for(a)) and ln.feasible(ws, profile)[0]}
                produced = _orders_by_line(run.read(ART["orders"]))   # 此时 03_orders 已是过闸(良定义)子集
                deficit = _order_deficit(produced, spec.per_line_min, feasible)
                if spec.total_only and sum(produced.values()) < spec.min_questions:
                    deficit["total"] = spec.min_questions - sum(produced.values())
                run.log(f"║  ①供给子轮{sub}:实产 {produced} / 硬下限 {spec.per_line_min} "
                        f"(软配额 {params.target_orders}) → 硬赤字 {deficit or '无'}")
                if not deficit or sub == order_subrounds:
                    break
                params = _grow_for_supply(params, deficit, grow_sessions=(rnd == 1)) # 第2轮+只长实体，不破坏旧语料时间轴
                params = dataclasses.replace(params, n_entities=min(params.n_entities, spec.max_world_entities))
                _scale_world_contract(wp, params.n_entities, params.n_sessions,
                                      narrative=narrative_mode)
                event_role_moves = _ensure_event_role_capacity(wp)
                l7_capacity = _ensure_l7_capacity(
                    wp, int(spec.per_line_min.get("L7_consolidation", 0)))
                run.log(f"║  ↑供给不足 → 长世界 n_ent={params.n_entities} n_sess={params.n_sessions} 重建"
                        + (f"；事件角色互异类型重分配 {event_role_moves}" if event_role_moves else "")
                        + (f"；L7 基质容量 {l7_capacity['before']}→{l7_capacity['after']}，"
                           f"类型重分配 {l7_capacity['moved']}" if l7_capacity["moved"] else ""))

            # 订单数是问题数和接地题数的严格上界。供给子轮耗尽后仍缺硬下限时，
            # questions/corpus 不可能补回，必须在最贵的渲染前失败。
            order_shortfall = _order_shortfall(produced, spec.per_line_min)
            if spec.total_only and sum(produced.values()) < spec.min_questions:
                order_shortfall["total"] = spec.min_questions - sum(produced.values())
            if order_shortfall:
                _update_run_metadata(run, algo={
                    "targetspec": {"min_questions": spec.min_questions,
                                   "per_line_min": spec.per_line_min,
                                   "total_only": spec.total_only, "max_world_entities": spec.max_world_entities,
                                   "haystack_ratio": spec.haystack_ratio},
                    "orders_by_line": produced,
                    "met_status": f"UNMET_ORDER_SUPPLY: {order_shortfall}",
                })
                _update_run_metadata(
                    run, config_remove=("augment", "render_only", "render_only_pairs"))
                run.log(f"║  ⚠ 订单硬下限未满足:{order_shortfall}；继续让现有订单走完出题、语料、接地和质量阶段")

            # ── 出题 → 渲染(round1 全量;round2+ ★增量 delta:只渲新实体、旧 docs 原样保留,§10.1)→ 接地 ──
            ws = WorldState.from_dict(run.read(ART["world"]))
            if rnd == 1 or narrative_mode:
                run_cfg.pop("render_only", None)
                run_cfg.pop("render_only_pairs", None)
                prev_entities = set(ws.entities)
                prev_relations = {_relation_key(r) for r in ws.relations}
                prev_events = {_event_key(e) for e in ws.events}
                # 全量重渲只清中断点；旧正式 05 保留到 stage_corpus 成功后原子替换。
                (run.dir / CORPUS_CKPT).unlink(missing_ok=True)
                run_generation_tail(run)
            else:
                new_ents, touched_pairs = _render_delta_scope(
                    ws, prev_entities, prev_relations, prev_events)
                run_cfg["render_only"] = new_ents
                run_cfg["render_only_pairs"] = touched_pairs
                run.log(f"║  增量续渲:+{len(new_ents)} 新实体全程 + {len(touched_pairs)} 个旧实体·结构变化期")
                try:
                    run_generation_tail(run)
                finally:
                    _update_run_metadata(
                        run, config_remove=("render_only", "render_only_pairs"))
                prev_entities = set(ws.entities)
                prev_relations = {_relation_key(r) for r in ws.relations}
                prev_events = {_event_key(e) for e in ws.events}

            # ── ② floor 校验(读回 stage 产物判定;driver 只做循环决策)──
            report = run.read("06_grounding_report.json")
            met, per_line_final, growable, permanent = _floor_status(report["by_line"], report["overall"], spec, feasible)
            if spec.total_only:
                per_line_final.update({lid: row.get("grounded", 0) for lid, row in report["by_line"].items()})
            last = {"kept": run.read(ART["grounding"]), "report": report, "growable": growable, "permanent": permanent}
            o = report["overall"]
            run.log(f"║  ②接地后:总 {o['grounded']}/{spec.min_questions}  逐线 {per_line_final}")
            run.log(f"║  达标={met}  待长(可行未达){growable or '无'}  永久(不可行){permanent or '无'}")
            _update_run_metadata(run, algo={
                "targetspec": {"min_questions": spec.min_questions,
                               "per_line_min": spec.per_line_min,
                               "total_only": spec.total_only, "max_world_entities": spec.max_world_entities,
                               "haystack_ratio": spec.haystack_ratio},
                "per_line_final": per_line_final,
                "met_status": "MET" if met else f"round{rnd}_unmet",
            })
            if met:
                run.log(f"╚═ ✓ 旋钮达标(MET):总 {o['grounded']}≥{spec.min_questions},各线 floor 均满足。")
                break
            # An unresolved or rejected semantic candidate is not evidence that
            # the world needs more entities. Keep this bounded review's complete
            # artifacts; changing its world would not repair its reviewed meaning.
            if ((wp.get("quality_contract") or {}).get("public_semantic_review") is True
                    and (report.get("n_pending", 0) > 0 or report.get("n_dropped", 0) > 0)):
                _update_run_metadata(run, algo={
                    "met_status": "UNMET_QUALITY_REVIEW",
                    "quality_review_shortfall": {
                        "round": rnd,
                        "n_pending": report.get("n_pending", 0),
                        "n_rejected": report.get("n_dropped", 0),
                        "pending_qids": [item["qid"] for item in report.get("pending", [])],
                        "rejected_qids": [item["qid"] for item in report.get("drops", [])],
                        "grounding_report": "06_grounding_report.json",
                        "semantic_review": "06_semantic_review.json",
                        "decision_basis": "recorded_review_status_not_entity_shortage",
                    },
                })
                _update_run_metadata(
                    run, config_remove=("augment", "render_only", "render_only_pairs"))
                run.log("║  ⚠ 部分题目审阅未决或被拒绝；保留完整意见，继续剩余供给轮，最终由质量报告收口")
            if permanent:                                                            # floor 落在【不可行线】→ 永远达不到,长世界也救不了,别空转
                run.log(f"╚═ ⚠ floor 落在不可行线{permanent}(基质供不出)→ 无解,停止空转并判失败。")
                break
            if rnd < max_rounds:                                                     # ② 用实测 survival 放大(只增不减)
                measured = {lid: v["survival"] for lid, v in report["by_line"].items() if v.get("survival") is not None}
                params = invert_rate(production_spec, survival=measured, slack=DEFAULT_SLACK * 1.3)
                params = dataclasses.replace(params, n_entities=max(params.n_entities, len(ws.entities)),
                                             n_sessions=ws.n_sessions)               # ★augment 只长实体不长周(周变了旧 docs 就失效,失去增量意义)
                run.log(f"║  ②实测 survival={measured} → 重算 target_orders={params.target_orders} "
                        f"n_ent={params.n_entities} n_sess={params.n_sessions}(下一轮增量续渲:只长新实体)")
    except Exception as exc:
        # A later supply/review round may fail after a complete earlier set
        # already exists.  Stop scheduling provider work, retain those exact
        # artifacts, and still execute the read-only result summary.
        required = (ART["world"], ART["questions"], ART["corpus"], ART["grounding"])
        if not all(run.has(name) for name in required):
            raise
        production_failure = exc
        binding = {
            name: hashlib.sha256((run.dir / name).read_bytes()).hexdigest()
            for name in required
        }
        run.write("06_production_warning.json", {
            "version": "production-execution-warning/v1",
            "status": "warning",
            "release_eligible": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "binding": binding,
        })
        _update_run_metadata(run, algo={
            "met_status": "UNMET_EXECUTION",
            "production_failure": {
                "error_type": type(exc).__name__,
                "artifact": "06_production_warning.json",
            },
        })
        run.log(f"║  ⚠ 后续生产轮执行失败:{type(exc).__name__}: {str(exc)[:160]}；"
                "停止新增调用，保留现有候选并继续生成最终结果汇总")


    if not met:
        unmet = ([f"执行失败:{type(production_failure).__name__}"] if production_failure
                 else (last["growable"] + last["permanent"]) or ["总数未达 min_questions"])
        _update_run_metadata(run, algo={"met_status": ("UNMET_EXECUTION" if production_failure
                                                        else f"UNMET: {unmet}")})
        _update_run_metadata(
            run, config_remove=("augment", "render_only", "render_only_pairs"))
        run.log(f"╚═ ⚠ 生产轮次已走完，合格题目标仍有缺口:{unmet}；继续生成最终结果汇总。")
    _update_run_metadata(
        run, config_remove=("augment", "render_only", "render_only_pairs"))       # ★清增量信号,免泄漏到后续 --only 重跑
    if production_failure is None:
        (run.dir / "06_production_warning.json").unlink(missing_ok=True)
    from pipeline.factory import stage_quality
    _run_stage(run, "quality", stage_quality, ART["quality"])
    _finalize_run(run)
    return last["kept"], "MET" if met else "COMPLETED_UNMET"
