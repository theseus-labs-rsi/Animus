"""Read-only diagnosis of a claimed cumulative child after reviewer interruption.

The reviewer can be charged before its physical receipt and gate are saved.
This audit replays the saved opinion against its inputs and the trace without
writing a missing receipt, reopening the execution claim, or calling a model.
"""
from __future__ import annotations

from pipeline.post_corpus_world_review import (
    AUTHORIZATION, BOUNDARY, EVIDENCE, GATE, LINEAGE, MIGRATION, PHYSICAL,
    REVIEW, SOURCE_EPOCH, STAGE_INVENTORY, _read_bound,
    _verify_migration_layers, expected_gate,
)
from pipeline.world_review_physical import (
    _prefix_and_tail, audit_new_review, digest, sha_file,
)


VERSION = "cumulative-review-interruption-audit/v1"


def _diagnostic(status, **details):
    return {"version": VERSION, "status": status, **details,
            "provider_calls_by_audit": 0, "writes_by_audit": 0,
            "execution_claim_reopened": False, "paid_resume_authorization": False,
            "production_ready": False}


def audit_saved_review(run):
    """Classify saved review evidence; never make an incomplete child runnable."""
    from pipeline import world_semantics
    from pipeline.world_state import WorldState

    cfg = (run.manifest.get("config") or {}).get("cumulative_recovery") or {}
    if cfg.get("version") != "explicit-cumulative-source-migration/v1":
        raise ValueError("No explicit cumulative migration control is installed")
    bindings = ((LINEAGE, "lineage_sha256"), (MIGRATION, "migration_sha256"),
                (EVIDENCE, "evidence_sha256"),
                (AUTHORIZATION, "authorization_sha256"),
                (BOUNDARY, "first_request_boundary_sha256"),
                (STAGE_INVENTORY, "stage_inventory_sha256"),
                (SOURCE_EPOCH, "source_epoch_sha256"))
    for _, key in bindings:
        if not isinstance(cfg.get(key), str) or len(cfg[key]) != 64:
            raise ValueError("Cumulative recovery control lacks " + key)
    data = {name: _read_bound(run, name, cfg[key]) for name, key in bindings}
    lineage, migration, boundary = data[LINEAGE], data[MIGRATION], data[BOUNDARY]
    _verify_migration_layers(run, lineage, migration, data[STAGE_INVENTORY],
        data[SOURCE_EPOCH], cfg["lineage_sha256"],
        cfg["stage_inventory_sha256"], cfg["source_epoch_sha256"])
    authorization = data[AUTHORIZATION]
    if (lineage.get("status") != "historical_lineage_verified"
            or migration.get("status") != "complete"
            or migration.get("complete_source_migration") is not True
            or migration.get("parent_lineage_sha256") != cfg["lineage_sha256"]
            or migration.get("first_new_operation") != "post_corpus_world.semantic_review"
            or authorization.get("policy") != "explicit-cumulative-cap-authorization/v1"
            or authorization.get("user_authorization") is not True
            or authorization.get("migration_sha256") != cfg["migration_sha256"]
            or authorization.get("first_request_boundary_sha256")
               != cfg["first_request_boundary_sha256"]
            or authorization.get("parent_profile_sha256")
               != lineage.get("budget", {}).get("profile_sha256")
            or authorization.get("parent_trace_sha256")
               != lineage.get("budget", {}).get("trace_sha256")
            or authorization.get("allowed_first_step") != "world.semantic_review"
            or boundary.get("first_step") != "world.semantic_review"
            or boundary.get("model") != lineage.get("budget", {}).get("model")
            or boundary.get("derived_manifest_sha256")
               != lineage.get("historical_copy", {}).get("reconciled_manifest_sha256")
            or boundary.get("ledger_sha256")
               != lineage.get("budget", {}).get("profile_sha256")
            or boundary.get("trace_sha256") != migration.get("historical_trace_sha256")
            or boundary.get("prompt_archive_sha256") != migration.get("historical_prompt_sha256")
            or boundary.get("evidence_hash") != data[EVIDENCE].get("envelope_hash")
            or run.manifest.get("run_id") != lineage.get("identity", {}).get("run_id")
            or run.read("11_production.json").get("identity")
               != lineage.get("identity", {}).get("production_identity")
            or sha_file(run.dir / "02_world_review.json")
               != lineage.get("historical_world_review_sha256")
            or sha_file(run.dir / "02_world_review_attempts.json")
               != lineage.get("historical_world_review_attempts_sha256")
            or sha_file(run.dir / "11_production.json")
               != lineage.get("historical_production_sha256")
            or sha_file(run.dir / "experiment_source_hashes.json")
               != lineage.get("initial_source_map_sha256")
            or sha_file(run.dir / "experiment_source_hashes_at_end.json")
               != lineage.get("ending_source_map_sha256")):
        raise ValueError("Claimed child lost its cumulative history or authority bindings")

    wp, task = run.read("01_whitepaper.json"), run.read("00_input.json")
    ws = WorldState.from_dict(run.read("02_world.json"))
    historical = run.read("02_world_review.json")
    if historical.get("status") != "passed":
        raise ValueError("Historical passed world opinion changed")
    world_semantics._validate_downstream_evidence(data[EVIDENCE], wp, ws, task,
                                                  historical)
    trace, prompts = run.dir / "llm_attempts.jsonl", run.dir / "prompts.jsonl"
    trace_tail = [row for old, row in _prefix_and_tail(trace,
        migration["historical_trace_bytes"], migration["historical_trace_sha256"])
        if not old]
    prompt_tail = [row for old, row in _prefix_and_tail(prompts,
        migration["historical_prompt_bytes"], migration["historical_prompt_sha256"])
        if not old]
    if not run.has(REVIEW):
        if run.has(PHYSICAL) or run.has(GATE):
            raise ValueError("Physical receipt or gate exists without a saved opinion")
        return _diagnostic("unresolved_activity_without_saved_opinion"
            if trace_tail or prompt_tail else "no_new_reviewer_activity",
            new_trace_events=len(trace_tail), new_prompt_rows=len(prompt_tail))

    review = run.read(REVIEW)
    if review.get("status") == "error":
        raise ValueError("Saved reviewer result contains no opinion")
    issues = world_semantics.validate_review(review, wp, ws, task_input=task)
    allowed = [] if review.get("status") == "passed" else [
        {"code": "world_review_not_passed", "status": review.get("status")}]
    if issues != allowed:
        raise ValueError("Saved reviewer opinion cannot replay against its world")
    first = next((row for row in trace_tail if row.get("event") == "request"), None)
    if (first is None or first.get("step") != boundary["first_step"]
            or digest(first.get("messages")) != boundary.get("messages_hash")
            or digest(first.get("parameters")) != boundary.get("params_hash")
            or first.get("model") != boundary.get("model")):
        raise ValueError("Saved review differs from the first authorized request")
    expected_physical = audit_new_review(review, trace, prompts,
        original_trace_bytes=migration["historical_trace_bytes"],
        original_trace_sha256=migration["historical_trace_sha256"],
        original_prompt_bytes=migration["historical_prompt_bytes"],
        original_prompt_sha256=migration["historical_prompt_sha256"])
    if run.has(PHYSICAL) and run.read(PHYSICAL) != expected_physical:
        raise ValueError("Saved physical receipt differs from the reviewer trace")
    if run.has(GATE):
        if not run.has(PHYSICAL):
            raise ValueError("World review gate exists without its physical receipt")
        if run.read(GATE) != expected_gate(run, lineage, review, expected_physical):
            raise ValueError("Saved world review gate differs from the proven opinion")
        status = "saved_gate_verified"
    elif run.has(PHYSICAL):
        status = "gate_missing_after_verified_opinion"
    else:
        status = "physical_and_gate_missing_after_verified_opinion"
    return _diagnostic(status, new_review_sha256=sha_file(run.dir / REVIEW),
        new_physical_requests=expected_physical["new_physical_requests"],
        new_trace_sha256=expected_physical["new_trace_sha256"],
        new_prompt_sha256=expected_physical["new_prompt_sha256"],
        saved_physical_receipt=run.has(PHYSICAL), saved_gate=run.has(GATE))
