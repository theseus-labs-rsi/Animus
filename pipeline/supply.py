"""Early quantity planning and bounded feedback inside the existing stages."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from pipeline.delivery_target import DeliveryTarget, LINE_IDS, allocate_line_targets, plan_supply


def author_brief(target, candidate_budget=None, world_limits=None, survival_rates=None):
    target = DeliveryTarget.from_dict(target)
    lines = list(target.requested_lines) or list(LINE_IDS[:8])
    allocation = allocate_line_targets(target, lines)
    evidence_plan = None
    if survival_rates is not None:
        evidence_plan = plan_supply(target, [{"line": line, "applicable": True,
            "implemented": True, "reason": "Pre-world planning assumption; whitepaper confirms applicability"}
            for line in lines], survival_rates)
        if evidence_plan["candidate_reserve"] is None:
            raise ValueError("Every requested line needs a positive sourced retention rate")
    minimum = evidence_plan["candidate_reserve"] if evidence_plan else target.final_questions
    budget = (candidate_budget if candidate_budget is not None else
              minimum if evidence_plan else target.final_questions * 3)
    if type(budget) is not int or budget < target.final_questions:
        raise ValueError("Candidate budget must cover the final delivery target")
    if budget < minimum:
        raise ValueError("Candidate budget is below the sourced line reserves")
    # This is an explicit production reserve, not a measured survival forecast.
    reserved = ({row["line"]: row["candidate_reserve"] for row in evidence_plan["lines"]}
                if evidence_plan else {line: 0 for line in lines})
    extra = budget - sum(reserved.values())
    shares = {line: reserved[line] + extra * amount // target.final_questions
              for line, amount in allocation.items()}
    for line in sorted(lines, key=lambda lid: (-(extra * allocation[lid] % target.final_questions), LINE_IDS.index(lid)))[:budget-sum(shares.values())]:
        shares[line] += 1
    return {"version": 1, "final_allocation": allocation, "candidate_allocation": shares,
            "candidate_budget": budget, "reserve_basis": "sourced_retention_rates" if evidence_plan else
                "explicit_candidate_budget" if candidate_budget is not None else
                "planning_assumption_three_candidates_per_final_question",
            "retention_evidence": evidence_plan,
            "count_stage": target.count_stage, "corpus_tokens": target.corpus_tokens,
            "tokenizer": target.tokenizer, "world_limits": deepcopy(world_limits or {}),
            "instruction": "按任务背景规划足够多的独立对象、真实关系和跨期业务过程。逐线说明适用结构及容量；保留领域约束和真实未知。数量不足逐线记录。"}


def attach_plan(wp, target, candidate_budget=None, world_limits=None, survival_rates=None, *, capacity_driven=False, instance_driven=False):
    brief = author_brief(target, candidate_budget, world_limits, survival_rates)
    mapping = {row["line"]: row for row in wp.get("line_mapping", [])}
    support = []
    typed = not (wp.get("world_blueprint") or {}).get("legacy_adapter")
    for line in brief["final_allocation"]:
        row = mapping.get(line, {})
        implemented = not (typed and line in ("L9_induction", "L10_admission"))
        applicable = row.get("applicable") is True
        if line == "L4_preference" and not wp.get("domain_profile", {}).get("preference_axis"):
            applicable = False
        support.append({"line": line, "applicable": applicable, "implemented": implemented,
                        "reason": "Typed-world authoring support is pending" if not implemented else
                        str(row.get("instantiation") or row.get("reason") or "No applicable structure declared by the world architect")})
    wp["delivery_target"] = deepcopy(target)
    wp["supply_plan"] = {**brief, "applicability": support}
    if capacity_driven:
        from pipeline.supply_capacity import POLICY, blueprint_capacity_report
        unsupported = [row for row in support if not (row["applicable"] and row["implemented"])]
        if unsupported:
            raise ValueError("Requested production lines lack a usable domain carrier: " + str(unsupported))
        wp["supply_plan"]["version"] = 2
        if instance_driven:
            from pipeline.instance_plan import VERSION
            from pipeline.capability_contract import requirements
            wp["supply_plan"]["instance_policy"] = VERSION
            wp["supply_plan"]["requirements"] = [requirements(line,amount)
                for line,amount in brief["candidate_allocation"].items() if amount]
        wp["supply_plan"]["capacity_blueprint"] = blueprint_capacity_report(wp["world_blueprint"], brief)
        issues = wp["supply_plan"]["capacity_blueprint"]["issues"]
        blocking = [row for row in issues if not instance_driven or row["code"] != "schema_capacity_below_reserve"]
        if blocking:
            raise ValueError("Whitepaper cannot fit the production reserve: " + str(blocking))
        profile = wp.setdefault("domain_profile", {})
        profile["supply_policy"] = POLICY
        profile["l5_max_conflicts"] = brief["candidate_allocation"].get("L5_conflict", 0)
    return wp["supply_plan"]


def inventory(wp, world):
    """Use original enumerators and well-posed checks on an isolated world copy."""
    from pipeline.lines import prepare_lines, run_lines
    from pipeline.world_gen import preview_supply_structure
    shadow = deepcopy(world)
    preview_supply_structure(wp, shadow)
    prepare_lines(wp, shadow, lambda *args: None)
    stats = {}
    orders = run_lines(wp, shadow, lambda *args: None,
                       question_budget=wp["supply_plan"]["candidate_budget"], stats=stats)
    return {"world_hash": hashlib.sha256(json.dumps(world.to_dict(), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "selected": len(orders), "shortfall": stats["shortfall"], "lines": stats["lines"],
            "scope": "structural well-posed supply before wording and public-evidence review"}


def finish_feedback(wp, world, state):
    """Request a bounded pre-publication supplement; retain a short world at cap."""
    if not wp.get("delivery_target"):
        return None
    from pipeline.instance_plan import enabled, fulfillment
    if enabled(wp):
        receipt = fulfillment(wp,world,state)
        state["instance_fulfillment"] = receipt
        if not receipt["passed"]:
            current_hash = receipt["world_hash"]
            old = state.get("last_instance_deficit_hash")
            rounds = state.get("supply_supplement_rounds",0)
            maximum = DeliveryTarget.from_dict(wp["delivery_target"]).max_supply_rounds
            if old == current_hash or rounds >= maximum:
                from pipeline.supply_capacity import CapacityReviewRequested
                raise CapacityReviewRequested({"reason":"Compiled instance obligations remain unrealized", "instance_fulfillment":receipt})
            state["last_instance_deficit_hash"] = current_hash
            state["supply_supplement_rounds"] = rounds+1
            compact = {key:value for key,value in receipt.items() if key != "capacity"}
            compact["capacity"] = {key:value for key,value in receipt["capacity"].items() if key not in ("certificates","duplicates")}
            return {"instance_deficit":compact,"instruction":"按具体未兑现业务槽位和原枚举结果修订。保留所有真实反馈；需要变更计划时请求上游容量复核。"}
    current = inventory(wp, world)
    history = state.setdefault("supply_observations", [])
    history.append(current)
    if current["shortfall"] <= 0:
        return None
    rounds = state.get("supply_supplement_rounds", 0)
    maximum = DeliveryTarget.from_dict(wp["delivery_target"]).max_supply_rounds
    supported = {row["line"] for row in wp["supply_plan"]["applicability"] if row["applicable"] and row["implemented"]}
    gaps = [row for row in current["lines"] if row["line"] in supported and row["shortfall"]]
    if (wp.get("supply_plan") or {}).get("version") == 2:
        from pipeline.supply_capacity import CapacityReviewRequested, requirements
        unchanged = len(history) > 1 and history[-2]["world_hash"] == current["world_hash"]
        if not gaps or rounds >= maximum or unchanged:
            evidence = {"reason": "Actual per-line capacity remains below the frozen candidate reserve",
                        "inventory": current, "unchanged_world": unchanged,
                        "requirements": requirements(wp["supply_plan"]["candidate_allocation"])}
            state.update(status="blueprint_review_requested", upstream_request=evidence)
            raise CapacityReviewRequested(evidence)
        state["supply_supplement_rounds"] = rounds + 1
        return {"supply_shortfall": gaps, "round": rounds + 1, "maximum_rounds": maximum,
                "requirements": requirements({row["line"]: row["shortfall"] for row in gaps}),
                "instruction": "补充独立业务对象和过程，满足各线真实枚举缺口。需要改变蓝图数量时请求上游容量规划。再次finish会重新核验实际供给；相同世界的重复finish直接返回上游。"}
    if not gaps or rounds >= maximum:
        state["supply_warning"] = {"code": "candidate_supply_target_unmet", **current}
        return None
    state["supply_supplement_rounds"] = rounds + 1
    return {"supply_shortfall": gaps, "round": rounds + 1, "maximum_rounds": maximum,
            "instruction": "依据这些真实枚举缺口补充独立业务事实，使用现有蓝图、实体规模和时间范围。可 inspect 读旧事实，再 write 或 revise。避免同一组事实换名称或排列扩题。若蓝图容量已用尽，finish 并说明缺口；原有效世界和逐线缺口都会保存。"}


def balanced_selection_subset(selected, filter_report, target, *, strict_allocation=False):
    """Keep up to the final primary goal, spreading available scored items by line."""
    from pipeline.delivery_target import is_auxiliary_witness

    target = DeliveryTarget.from_dict(target)
    allocation = allocate_line_targets(target, list(target.requested_lines))
    decisions = {item["qid"]: item for item in filter_report["items"]}
    pools = {line: [] for line in allocation}
    witnesses = []
    for question in selected:
        qid = question["qid"]
        decision = decisions.get(qid)
        if decision is None or decision.get("disposition") != "kept_not_all_correct":
            raise ValueError(f"A final delivery item lacks complete four-athlete scores: {qid}")
        if is_auxiliary_witness(question):
            witnesses.append(question)
        elif question["line"] in pools:
            pools[question["line"]].append(question)
    chosen = {line: pool[:allocation[line]] for line, pool in pools.items()}
    remaining = max(0, target.final_questions - sum(map(len, chosen.values())))
    while remaining and not strict_allocation:
        available = [line for line in chosen if len(chosen[line]) < len(pools[line])
                     and len(chosen[line]) < target.per_line_max.get(line, target.final_questions)]
        if not available:
            break
        line = min(available, key=lambda item: (len(chosen[item]), list(allocation).index(item)))
        chosen[line].append(pools[line][len(chosen[line])])
        remaining -= 1
    kept_ids = {item["qid"] for items in chosen.values() for item in items}
    kept_ids.update(item["qid"] for item in witnesses)
    subset = [question for question in selected if question["qid"] in kept_ids]
    return subset, {"version": 1, "target": target.to_dict(),
                    "ideal_allocation": allocation,
                    "selected_by_line": {line: len(items) for line, items in chosen.items()},
                    "qids": [question["qid"] for question in subset],
                    "surplus_qids": [question["qid"] for question in selected if question["qid"] not in kept_ids]}


def write_delivery_report(run):
    target_dict = run.manifest.get("config", {}).get("delivery_target")
    if not target_dict:
        return None
    from pipeline.delivery_target import quality_rows_from_partition, summarize_delivery
    wp = run.read("01_whitepaper.json")
    read = lambda name: run.read(name) if run.has(name) else None
    candidates = read("04_questions.json")
    from pipeline.quality import quality_snapshot
    release = quality_snapshot(run.dir) if run.has("07_release.json") else None
    partition = ((release or {}).get("checks", {}).get("partition") or {}).get("qids")
    quality = quality_rows_from_partition(candidates, partition) if candidates is not None and partition is not None else None
    selection = None
    if run.has("09_selection.json"):
        from pipeline.calibration import selection_is_current
        if selection_is_current(run):
            selection = run.read("09_selection.json")
    report = summarize_delivery(DeliveryTarget.from_dict(target_dict), wp["supply_plan"]["applicability"],
        raw_orders=read("03_raw_orders.json"), effective_orders=read("03_orders.json"), candidates=candidates,
        quality_rows=quality, selection_report=selection,
        selection_candidates=read("06_grounded_questions.json") if selection is not None else None)
    token_scale = read("05_corpus_token_scale.json")
    report["corpus_target"]["measurement"] = token_scale
    from pipeline.factory import _corpus_is_current
    corpus_current = bool(token_scale and _corpus_is_current(run))
    report["corpus_target"]["current"] = corpus_current
    report["quality_eligible"] = bool(release and release.get("eligible"))
    report["delivery_target_met"] = bool(report["question_target_met"] and report["quality_eligible"]
                                         and corpus_current and token_scale.get("target_met"))
    if wp["supply_plan"].get("version") == 2:
        exact = all((row["primary"]["final_selected"] or 0) >= row["final_target"]
                    for row in report["by_line"] if row["final_target"])
        report["strict_line_allocation_met"] = exact
        report["delivery_target_met"] = report["delivery_target_met"] and exact
    run.write("10_delivery_target.json", report)
    return report
