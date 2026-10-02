"""A delivery-owned controller with early capacity gates and per-line refill.

One Run retains its tracer, physical-call budget and calibration budget across
rounds. Every replaced artifact is archived before a new world is constructed.
Original semantic verdicts stay in those archives. A review execution failure
requires diagnosis; it never becomes an automatic quantity-loss retry.
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from pipeline.run import drive
from pipeline.supply_capacity import VERSION, digest, revise_capacity

STATE = "11_production.json"


def save_state(run, state):
    expected = state.get("state_revision", 0)
    previous = run.read(STATE) if run.has(STATE) else None
    actual = previous.get("state_revision", 0) if previous else 0
    if actual != expected:
        raise ValueError("Production state changed outside its controller lease")
    saved = deepcopy(state)
    stable = lambda value: {k:v for k,v in value.items() if k not in ("generation_budget", "state_revision")}
    saved["state_revision"] = expected + (1 if previous is None or stable(previous) != stable(saved) else 0)
    if previous == saved:
        return
    run.write(STATE, saved)
    state.clear(); state.update(saved)


def ensure_instance_plan(run):
    from pipeline.instance_plan import PRODUCTION_VERSION, VERSION as PLAN_VERSION, create
    cfg = run.manifest["config"]
    if (cfg.get("production_control") or {}).get("version") not in (PRODUCTION_VERSION, "supply-driven/v4", "supply-driven/v6"):
        return
    wp = run.read("01_whitepaper.json")
    if cfg["production_control"]["version"] == "supply-driven/v6":
        from pipeline.joint_design import exploratory_after_limit
        if exploratory_after_limit(run, wp):
            return
    if wp.get("business_instance_plan"):
        plan = wp["business_instance_plan"]
        if plan["plan_hash"] != digest({k:v for k,v in plan.items() if k != "plan_hash"}) or plan["blueprint_hash"] != digest(wp["world_blueprint"]):
            raise ValueError("Frozen business instance plan changed")
        return
    from pipeline.capability_contract import requirements
    wp["supply_plan"]["instance_policy"] = PLAN_VERSION
    wp["supply_plan"]["requirements"] = [requirements(line,amount)
        for line,amount in wp["supply_plan"]["candidate_allocation"].items() if amount]
    name = "01_instance_plan_audit.json"
    audit = run.read(name) if run.has(name) else {}
    if audit.get("status") == "accepted" and audit.get("before_hash") == digest(wp) and audit.get("after_hash") == digest(audit.get("after")):
        revised = deepcopy(audit["after"])
    else:
        if run.has(name):
            from pipeline.instance_plan import PlanConflict, finding
            raise PlanConflict([finding('plan_recovery_required',
                'Saved initial plan audit requires explicit recovery; ordinary entry cannot replace prior attempts',kind='design')])
        try:
            revised = create(wp,run.tracer,{"phase":"initial_business_layout"},audit,
                checkpoint=lambda current:run.write(name,current))
        finally:
            run.write(name,audit)
    run.write("01_whitepaper.json",revised)
    run.write("01_supply_plan.json",revised["supply_plan"])
    run.write("01_instance_plan.json",revised["business_instance_plan"])


def frozen_identity(run):
    cfg = run.manifest["config"]
    return digest({"config": {key: cfg.get(key) for key in ("delivery_target", "question_budget",
        "single_pass_world_limits", "calibration", "production_control", "delivery_survival_rates", "acceptance_scope")},
        "models": {key: run.manifest.get("env", {}).get(key) for key in ("model", "discriminator_model",
            "structure_model", "reviewer_model", "judge_model", "reasoning_effort", "base_url")}})


def observe_budget(run, state):
    """Bind an installed physical-call wrapper; diagnose a plain unbounded CLI."""
    name = "experiment_profile.json"
    if not run.has(name):
        current = {"bounded": False, "diagnostic": "generation_budget_wrapper_not_installed"}
    else:
        ledger = run.read(name)
        keys = ("model", "max_calls", "max_cny")
        bounded = all(key in ledger for key in (*keys, "admitted_calls", "budget_consumed_cny"))
        current = {"bounded": bounded, "path": name, "ledger_hash": digest(ledger),
                   "cap_identity": digest({key: ledger.get(key) for key in keys}),
                   "admitted_calls": ledger.get("admitted_calls"),
                   "reported_estimate_cny": ledger.get("reported_usage_estimate_cny"),
                   "occupied_estimate_cny": ledger.get("budget_consumed_cny"),
                   "unsettled_reservations": ledger.get("unsettled_reservation_calls")}
        if not bounded:
            current["diagnostic"] = "generation_budget_wrapper_not_recognized"
    old = state.get("generation_budget")
    if old and old.get("bounded"):
        if not current.get("bounded") or old["cap_identity"] != current["cap_identity"]:
            # The wrapper validates an explicit cumulative cap authorization
            # against the original stopped ledger and trace before reopening.
            # Carry that transition into the controller without resetting its
            # physical-call history or any production/author allowance.
            proof = ((ledger.get("resume_history") or [{}])[-1].get("transport_recovery") or {}) if current.get("bounded") else {}
            prior_cap = {"model": proof.get("model"), "max_calls": proof.get("max_calls"), "max_cny": proof.get("prior_max_cny")}
            if not (proof.get("reason") == "explicit_cumulative_budget_resume"
                    and digest(prior_cap) == old["cap_identity"]
                    and proof.get("model") == ledger["model"]
                    and proof.get("max_calls") == ledger["max_calls"]
                    and proof.get("new_max_cny") == ledger["max_cny"]
                    and type(proof.get("prior_max_cny")) in (int, float)
                    and ledger["max_cny"] > proof["prior_max_cny"]
                    and type(proof.get("prior_admitted_calls")) is int
                    and old["admitted_calls"] <= proof["prior_admitted_calls"] <= current["admitted_calls"]
                    and type(proof.get("prior_budget_consumed_cny")) in (int, float)
                    and proof["prior_budget_consumed_cny"] <= current["occupied_estimate_cny"]
                    and all(isinstance(proof.get(k), str) and len(proof[k]) == 64
                            for k in ("authorization_sha256", "trace_sha256"))):
                raise ValueError("Generation physical-call budget changed or disappeared")
            current["authorized_cap_transition"] = {"previous_cap_identity": old["cap_identity"],
                "authorization_sha256": proof["authorization_sha256"], "new_max_cny": ledger["max_cny"]}
        if current["admitted_calls"] < old["admitted_calls"]:
            raise ValueError("Generation physical-call ledger was reset")
    state["generation_budget"] = current


def outcome(run, *, selected=False):
    """Read the registered quality/selection receipts, preserving all denominators."""
    from pipeline.supply import write_delivery_report
    report = write_delivery_report(run)
    plan = run.read("01_whitepaper.json")["supply_plan"]
    field = "final_selected" if selected else "released"
    counts = {row["line"]: row["primary"][field] for row in report["by_line"]}
    deficits = {line: goal - (counts.get(line) or 0) for line, goal in plan["final_allocation"].items()
                if (counts.get(line) or 0) < goal}
    # Count native trends only after their original qids survive the current
    # quality partition. A world-level S1 certificate cannot certify a released
    # subset that retained only comparisons (or a renamed planted trend).
    l7 = "L7_consolidation"
    declared_native = [row.get("subtype_minimum", {}).get("native_trend", 0)
                       for row in plan.get("requirements", []) if row.get("line") == l7]
    native_minimum = (max([int(bool(plan.get("instance_policy"))), *declared_native])
                      if plan["final_allocation"].get(l7) else 0)
    native = {"required": native_minimum, "count_stage": "selection_complete" if selected else "generation_quality",
              "actual": 0, "qids": [], "family_keys": [], "source_mismatches": []}
    if native_minimum:
        from pipeline.capability_contract import family_key
        from pipeline.quality import quality_snapshot
        release = quality_snapshot(run.dir)
        partition = ((release.get("checks") or {}).get("partition") or {}).get("qids") or {}
        kept = set(partition.get("released", [])) if release.get("eligible") else set()
        if selected:
            kept &= ({row["qid"] for row in run.read("09_selected_questions.json")}
                     if report["selection_complete"] and run.has("09_selected_questions.json") else set())
        candidates = {row["qid"]: row for row in run.read("04_questions.json")} if run.has("04_questions.json") else {}
        orders = {row["qid"]: row for row in run.read("03_orders.json")} if run.has("03_orders.json") else {}
        families = set()
        for qid in sorted(kept):
            question, order = candidates.get(qid), orders.get(qid)
            if not question or question.get("line") != l7:
                continue
            aux = (order or {}).get("aux") or {}
            if not order or family_key(question) != family_key(order):
                native["source_mismatches"].append(qid)
                continue
            if aux.get("sub") == "S1_trend" and aux.get("supply_origin") == "native_numeric_trajectory/v1":
                native["qids"].append(qid)
                families.add(family_key(order))
        native.update(actual=len(families), family_keys=sorted(families))
        shortfall = native_minimum - native["actual"]
        if shortfall > 0:
            deficits[l7] = max(deficits.get(l7, 0), shortfall)
    native["passed"] = native["actual"] >= native_minimum and not native["source_mismatches"]

    execution_errors = []
    if run.has("06_semantic_review.json"):
        semantic = run.read("06_semantic_review.json")
        for row in semantic.get("items", []) if isinstance(semantic, dict) else []:
            stages = row.get("stage_execution") or {}
            if any(isinstance(value, dict) and value.get("status") == "model_error" for value in stages.values()):
                execution_errors.append(row.get("source_qid", row.get("qid")))
    return {"count_stage": "selection_complete" if selected else "generation_quality",
            "counts": counts, "deficits": deficits, "report": report, "native_trend": native,
            "review_execution_errors": execution_errors,
            "passed": bool(not deficits and native["passed"] and report["quality_eligible"]
                 and not (run.has("03_capacity_gate.json") and
                          run.read("03_capacity_gate.json").get("passed") is False)
                 and not (run.has("02_source_supply_gate.json") and
                          run.read("02_source_supply_gate.json").get("release_eligible") is False)
                and report["corpus_target"]["current"]
                and (report["corpus_target"].get("measurement") or {}).get("target_met")
                and (not selected or report["selection_complete"]))}


def next_reserve(plan, result):
    """Use observed survival for each deficient line; retain the other reserves."""
    allocation = deepcopy(plan["candidate_allocation"])
    rows = {row["line"]: row["primary"] for row in result["report"]["by_line"]}
    for line, missing in result["deficits"].items():
        observed = rows[line]["effective_orders"] or 0
        retained = result["counts"].get(line) or 0
        if retained == 0:
            raise ValueError("Zero observed survival requires a line-design diagnosis: " + line)
        allocation[line] = max(allocation[line] + missing,
            math.ceil(observed * plan["final_allocation"][line] / retained))
    return allocation


def archive_round(run, number):
    """Write-once snapshots. A receipt verifies every byte before replacement."""
    folder = run.dir / "production" / f"round-{number:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in run.dir.iterdir()
                   if path.is_file() and path.name[:2].isdigit() and path.name.endswith(".json")
                   and path.name != STATE)
    entries = {}
    for path in files:
        body = path.read_bytes()
        target = folder / path.name
        if target.exists() and target.read_bytes() != body:
            raise ValueError("Production archive collision: " + path.name)
        if not target.exists():
            target.write_bytes(body)
        import hashlib
        entries[path.name] = hashlib.sha256(body).hexdigest()
    receipt = {"version": VERSION, "files": entries,
               "stages": deepcopy(run.manifest["stages"]), "tracer_calls": run.tracer.n}
    path = folder / "receipt.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Production archive receipt changed")
    if not path.exists():
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def retire_checkpoint_reuse(manifest, receipt, round_number, new_whitepaper_sha256):
    """Move a former world's resume binding into its archived round's history."""
    derived = manifest.get("derived_from") or {}
    reused = derived.get("world_construction_checkpoints") or []
    if not reused or receipt["files"]["01_whitepaper.json"] == new_whitepaper_sha256:
        return None
    if any(name not in receipt["files"] for name in reused):
        raise ValueError("Reused construction checkpoint is missing from the archived round")
    entry = {"archived_round": round_number,
        "files": {name: receipt["files"][name] for name in reused},
        "old_whitepaper_sha256": receipt["files"]["01_whitepaper.json"],
        "new_whitepaper_sha256": new_whitepaper_sha256}
    history = derived.setdefault("world_construction_checkpoint_history", [])
    if entry not in history:
        history.append(entry)
    derived.pop("world_construction_checkpoints")
    return entry


def install_revision(run, state, stages):
    """Replay an interrupted installation from its saved plan, without LLM calls."""
    receipt = state["replacement_receipt"]
    folder = run.dir / "production" / f"round-{state['round']:03d}"
    import hashlib
    for name, expected in receipt["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Production archive integrity failed: " + name)
    for name in receipt["files"]:
        if 2 <= int(name[:2]) <= 10:
            (run.dir / name).unlink(missing_ok=True)
    run.write("01_whitepaper.json", state["next_whitepaper"])
    run.write("01_supply_plan.json", state["next_whitepaper"]["supply_plan"])
    if state["next_whitepaper"].get("business_instance_plan"):
        run.write("01_instance_plan.json",state["next_whitepaper"]["business_instance_plan"])
    else:
        (run.dir / "01_instance_plan.json").unlink(missing_ok=True)
    if state.get("next_joint_design_receipt"):
        run.write("01_joint_design_receipt.json", state["next_joint_design_receipt"])
    if "seed_audit" in state["next_whitepaper"]:
        run.write("01_seed_audit.json", state["next_whitepaper"]["seed_audit"])
    retire_checkpoint_reuse(run.manifest, receipt, state["round"],
        hashlib.sha256((run.dir / "01_whitepaper.json").read_bytes()).hexdigest())
    for stage in stages:
        if stage.name not in ("input", "whitepaper"):
            run.manifest["stages"].pop(stage.name, None)
    run._save_manifest()
    committed = deepcopy(state)
    committed.update(round=state["round"] + 1, status="running")
    committed.pop("replacement_receipt", None)
    committed.pop("next_whitepaper", None)
    committed.pop("next_joint_design_receipt", None)
    save_state(run, committed)
    state.clear()
    state.update(committed)


def prepare_revision(run, state, stages, allocation, feedback):
    """Hold the original Run lock throughout plan/archival/installation."""
    with run.stage_write_lock("production_capacity_revision"):
        run._reload_manifest_for_stage()
        if frozen_identity(run) != state["identity"]:
            raise ValueError("Production configuration changed during the supply round")
        wp = run.read("01_whitepaper.json")
        wp["supply_plan"].update(candidate_allocation=allocation, candidate_budget=sum(allocation.values()))
        from pipeline.supply_capacity import revision_implementation
        name = f"11_revision_{state['round']:03d}.json"
        audit = run.read(name) if run.has(name) else {}
        if (audit.get("status") == "accepted" and audit.get("before") == wp
                and audit.get("feedback") == feedback
                and audit.get("after_hash") == digest(audit.get("after"))
                and audit.get("implementation") == revision_implementation()):
            revised = deepcopy(audit["after"])
        else:
            if run.has(name):
                raise ValueError('Saved capacity revision requires explicit recovery; ordinary entry cannot replace prior attempts')
            audit = {}
            failures = []
            request_feedback = feedback
            instance_driven = bool(wp["supply_plan"].get("instance_policy"))
            for attempt in range(1 if instance_driven else 3):
                audit = {}
                def checkpoint(current):
                    current['compiler_recovery_attempts']=deepcopy(failures)
                    run.write(name,current)
                try:
                    revised = revise_capacity(wp, run.tracer, request_feedback, audit,checkpoint=checkpoint)
                    audit["compiler_recovery_attempts"] = deepcopy(failures)
                    break
                except Exception as error:
                    # A rejected quantity proposal has not changed the world.
                    # Route only the compiler's concrete structural rejection
                    # back to its author; semantic refusals and global stops
                    # retain their original meaning.
                    from pipeline.world_blueprint import WorldBlueprintError
                    mechanical = (isinstance(error, WorldBlueprintError)
                        and str(error).startswith("world_blueprint 非法:")
                        and isinstance(audit.get("proposal"), dict))
                    failures.append({"audit": deepcopy(audit), "error": str(error)})
                    audit["compiler_recovery_attempts"] = deepcopy(failures)
                    if not mechanical or attempt == 2:
                        raise
                    request_feedback = {"original_capacity_deficit": deepcopy(feedback),
                        "rejected_quantity_proposal": deepcopy(audit["proposal"]),
                        "compiler_error": str(error),
                        "instruction": "原世界和白皮书保持冻结。修正未安装的数量提案，满足原schema、标量FK容量、exact基数及原实体上限；保持原逐线候选要求。"}
                finally:
                    run.write(name, audit)
        state.update(status="installing_revision", next_whitepaper=revised,
                     replacement_receipt=archive_round(run, state["round"]))
        save_state(run, state)
        install_revision(run, state, stages)


def prepare_joint_revision(run, state, stages, evidence):
    """Archive the old world before spending any remaining v6 design revision."""
    from pipeline.joint_design import revise_supply, MAX_REVISIONS
    with run.stage_write_lock("production_joint_revision"):
        run._reload_manifest_for_stage()
        if frozen_identity(run) != state["identity"]:
            raise ValueError("Production configuration changed during the joint supply round")
        receipt = archive_round(run, state["round"])
        current_wp = run.read("01_whitepaper.json")
        current_receipt = run.read("01_joint_design_receipt.json")
        name = "01_joint_design_audit.json"
        audit = run.read(name)
        revised, design_receipt = revise_supply(run, evidence, audit,
            lambda current: run.write(name, current))
        if (audit["used_revisions"] >= MAX_REVISIONS
                and design_receipt.get("fallback") == "blueprint_only"):
            # A failed redesign must not discard a complete compiled candidate.
            # Keep the original paper/checkpoint binding; the world stage will
            # recompile that checkpoint with the exhausted-limit policy.
            from pipeline.world_agent import _binding, _digest
            for path in sorted(run.dir.glob("02_world_agent_*.json")):
                saved = run.read(path.name)
                candidate = saved.get("candidate") or {}
                if (saved.get("binding") != _binding(current_wp, None)
                        or saved.get("checkpoint_hash") != _digest({k:v for k,v in saved.items() if k != "checkpoint_hash"})
                        or candidate.get("compiler") != "passed"
                        or candidate.get("table_hash") != _digest(candidate.get("table"))):
                    continue
                current_receipt.update(used_revisions=audit["used_revisions"], release_eligible=False,
                    fallback="retained_compiled_world", checkpoint=path.name,
                    unresolved_findings=deepcopy(design_receipt["unresolved_findings"]))
                audit.update(status="degraded_with_compiled_world", rejected_design_receipt=deepcopy(design_receipt),
                    accepted=deepcopy(current_wp), receipt=deepcopy(current_receipt))
                run.write(name, audit)
                run.write("01_joint_design_receipt.json", current_receipt)
                state["history"].append({"operation":"retain_compiled_world_after_design_limit",
                    "checkpoint":path.name, "checkpoint_hash":saved["checkpoint_hash"],
                    "world_hash":candidate["world_hash"], "release_eligible":False})
                save_state(run, state)
                return
        state.update(status="installing_revision", next_whitepaper=revised,
                     next_joint_design_receipt=design_receipt,
                     replacement_receipt=receipt)
        save_state(run, state)
        install_revision(run, state, stages)


def produce(run, *, to_stage=None, force=False):
    from pipeline.cumulative_entry_guard import check_before_production
    check_before_production(run)
    with run.stage_write_lock("production_controller", lock_name=".production.lock"):
        run._reload_manifest_for_stage()
        return _produce(run,to_stage=to_stage,force=force)


def _produce(run, *, to_stage=None, force=False):
    from pipeline.cumulative_entry_guard import check_before_production
    check_before_production(run)
    from pipeline.factory import STAGES, generation_stages
    from pipeline.calibration import production_calibration_support
    cfg = run.manifest["config"]
    if cfg.get("acceptance_scope"):
        from pipeline.delivery_target import AcceptanceScope
        AcceptanceScope.from_dict(cfg["acceptance_scope"])
    if cfg.get("calibration"):
        production_calibration_support(cfg["calibration"], cfg["production_control"])
    identity = frozen_identity(run)
    state = run.read(STATE) if run.has(STATE) else {"version": cfg["production_control"]["version"], "identity": identity,
        "round": 1, "status": "running", "history": []}
    local_controller = cfg["production_control"]["version"] in ("supply-driven/v5", "supply-driven/v6")
    if local_controller:
        state.setdefault("supply_round", 1)
        state.setdefault("layout_revisions", 0)
    if state.get("identity") != identity:
        raise ValueError("Production target/configuration changed; use a separately frozen run")
    observe_budget(run, state)
    stages = generation_stages(run)
    if state["status"] == "installing_revision":
        with run.stage_write_lock("production_install_revision"):
            run._reload_manifest_for_stage()
            install_revision(run, state, stages)
    elif state["status"] == "plan_recovery_required":
        from pipeline.initial_plan_recovery import recover_if_permitted, ARTIFACT
        if not recover_if_permitted(run):
            raise ValueError("Production requires recorded diagnosis before continuation: " + state["status"])
        recovery = run.read(ARTIFACT)
        state["history"].append({"operation": "explicit_initial_plan_recovery",
            "previous_status": state["status"], "original_findings": deepcopy(state.get("findings", [])),
            "request_sha256": recovery["request_sha256"], "after_hash": recovery["after_hash"]})
        state["status"] = "running"
        state.pop("findings", None)
    elif state["status"] in ("capacity_unresolved", "review_recovery_required", "design_diagnosis_required",
                             "round_limit", "search_inconclusive", "joint_design_limit"):
        raise ValueError("Production requires recorded diagnosis before continuation: " + state["status"])
    save_state(run, state)
    names = [stage.name for stage in stages]
    boundary = to_stage or names[-1]
    if boundary not in names:
        raise ValueError("Unknown production stop stage: " + boundary)
    try:
        drive(run, STAGES, to_stage=boundary if boundary in ("input", "whitepaper") else "whitepaper", force=force)
        if boundary in ("input", "whitepaper"):
            return state
        ensure_instance_plan(run)
        while True:
            observe_budget(run, state)
            from pipeline.supply_capacity import CapacityReviewRequested
            try:
                drive(run, STAGES, "world", min(boundary, "well_posed", key=lambda name: names.index(name)))
                if names.index(boundary) <= names.index("well_posed"):
                    state["status"] = "paused_at_stage"
                    save_state(run, state)
                    return state
                drive(run, STAGES, "corpus", "quality" if names.index(boundary) >= names.index("quality") else boundary)
            except CapacityReviewRequested as request:
                state["history"].append({"round": state["round"], "capacity_deficit": request.evidence})
                if cfg["production_control"]["version"] == "supply-driven/v6":
                    from pipeline.joint_design import MAX_REVISIONS
                    proof = request.evidence.get("instance_fulfillment") or {}
                    codes = {row.get("code") for row in proof.get("findings", [])}
                    if any(code and code.endswith("search_inconclusive") for code in codes):
                        state.update(status="search_inconclusive", result=request.evidence)
                        save_state(run, state)
                        return state
                    audit = run.read("01_joint_design_audit.json")
                    if audit["used_revisions"] >= MAX_REVISIONS:
                        # The world/order stages already permit a bound shortage
                        # at this limit. A remaining exception means there is no
                        # executable candidate; never retry the same stage forever.
                        state.update(status="joint_design_limit", result=request.evidence)
                        save_state(run, state)
                        raise
                    prepare_joint_revision(run, state, stages, request.evidence)
                    continue
                if ((state["layout_revisions"] >= 3) if local_controller else
                    state["round"] >= cfg["delivery_target"]["max_supply_rounds"]):
                    state.update(status="round_limit", result=request.evidence)
                    save_state(run, state)
                    return state
                from pipeline.construction import compact_feedback
                if local_controller:
                    state["layout_revisions"] += 1
                prepare_revision(run, state, stages,
                    run.read("01_whitepaper.json")["supply_plan"]["candidate_allocation"], compact_feedback(request.evidence))
                continue
            if names.index(boundary) < names.index("quality"):
                state["status"] = "paused_at_stage"
                save_state(run, state)
                return state
            result = outcome(run)
            observe_budget(run, state)
            if any(run.has(name) and run.read(name).get("release_eligible") is False for name in
                   ("01_joint_design_receipt.json", "02_source_supply_gate.json", "03_capacity_gate.json")):
                state.update(status="generation_completed_with_warnings", result=result, release_eligible=False)
                save_state(run, state)
                return state
            if result["review_execution_errors"] and not result["passed"]:
                state.update(status="review_recovery_required", result=result)
                save_state(run, state)
                return state
            if not result["report"]["quality_eligible"] or not result["report"]["corpus_target"]["current"]:
                state.update(status="review_recovery_required", result=result)
                save_state(run, state)
                return state
            if not (result["report"]["corpus_target"].get("measurement") or {}).get("target_met"):
                state.update(status="design_diagnosis_required", reason="Corpus token contract unmet", result=result)
                save_state(run, state)
                return state
            if result["passed"] and cfg.get("calibration") and boundary in ("calibration", "selection"):
                drive(run, stages, "calibration", boundary)
                if boundary == "calibration":
                    state.update(status="awaiting_selection", result=result)
                    save_state(run, state)
                    return state
                result = outcome(run, selected=True)
                observe_budget(run, state)
                if not result["report"]["selection_complete"]:
                    state.update(status="awaiting_complete_scores", result=result)
                    save_state(run, state)
                    return state
            if result["passed"]:
                state.update(status="delivered" if result["count_stage"] == "selection_complete" else
                             "generation_ready_awaiting_selection", result=result)
                save_state(run, state)
                return state
            state["history"].append({"round": state["round"], "result": result,
                                     "world_hash": digest(run.read("02_world.json"))})
            plan = run.read("01_whitepaper.json")["supply_plan"]
            maximum = cfg["delivery_target"]["max_supply_rounds"]
            if (state["supply_round"] if local_controller else state["round"]) >= maximum:
                state.update(status="round_limit", result=result)
                save_state(run, state)
                return state
            try:
                allocation = next_reserve(plan, result)
            except ValueError as error:
                state.update(status="design_diagnosis_required", result=result, reason=str(error))
                save_state(run, state)
                return state
            if local_controller:
                state["supply_round"] += 1
            prepare_revision(run, state, stages, allocation, {"stage": result["count_stage"],
                "per_line_losses": result, "next_allocation": allocation})
    except Exception as error:
        if state["status"] == "installing_revision":
            state["last_error"] = f"{type(error).__name__}: {error}"
            save_state(run, state)
            raise
        from pipeline.instance_plan import PlanConflict
        from pipeline.render import MaterialRejected
        from pipeline.supply_capacity import CapacityReviewRequested
        from pipeline.world_blueprint import WorldBlueprintError
        if isinstance(error,PlanConflict):
            state.update(status="design_diagnosis_required" if any(f["kind"] == "design" for f in error.findings) else "plan_recovery_required",findings=error.findings)
            save_state(run,state)
            raise
        if isinstance(error, MaterialRejected):
            state.update(status="review_recovery_required", material_feedback=deepcopy(error.report))
        else:
            state["status"] = ("capacity_unresolved" if isinstance(error,
                (CapacityReviewRequested, WorldBlueprintError)) else "execution_error")
        state.update(
                     error=f"{type(error).__name__}: {error}")
        save_state(run, state)
        raise
