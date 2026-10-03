"""Read-only admission plan for the original value authors after a new review.

This binds the independent opinion to physical calls and the historical author
checkpoint. It deliberately does not open a model transaction or edit truth.
"""
from __future__ import annotations

import hashlib
import json

from pipeline.post_corpus_world_review import (
    BOUNDARY, GATE, LINEAGE, MIGRATION, PHYSICAL, REVIEW, _read_bound,
)
from pipeline.world_review_physical import audit_new_review, digest, sha_file


VERSION = "cumulative-original-owner-route/v1"


def _checkpoint_name(wp):
    identity = json.dumps({"whitepaper": wp, "existing": None}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))
    return "02_world_agent_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20] + ".json"


def plan_owner_route(run):
    """Resolve proven intrinsic targets to existing authors without writing.

    Only a completed, physically bound repair opinion can enter this route.
    The returned plan is evidence for a later bounded transaction, not a cap
    authorization, an author reply, or a changed world.
    """
    from pipeline import world_semantics
    from pipeline.cumulative_materializer import _encode
    from pipeline.instance_executor import VALUE_TRANSACTION_VERSION, review_value_groups
    from pipeline.world_gen import WORLD_DRAFT_VERSION, _world_digest
    from pipeline.world_state import WorldState

    cfg = (run.manifest.get("config") or {}).get("cumulative_recovery") or {}
    if cfg.get("version") != "explicit-cumulative-source-migration/v1":
        raise ValueError("Owner route lacks explicit cumulative migration control")
    for key in ("lineage_sha256", "migration_sha256", "first_request_boundary_sha256"):
        if not isinstance(cfg.get(key), str) or len(cfg[key]) != 64:
            raise ValueError("Owner route lacks bound cumulative input: " + key)
    lineage = _read_bound(run, LINEAGE, cfg["lineage_sha256"])
    migration = _read_bound(run, MIGRATION, cfg["migration_sha256"])
    _read_bound(run, BOUNDARY, cfg["first_request_boundary_sha256"])
    if (migration.get("status") != "complete" or
            migration.get("complete_source_migration") is not True or
            migration.get("parent_lineage_sha256") != cfg["lineage_sha256"]):
        raise ValueError("Owner route lacks complete cumulative migration")
    for name in (REVIEW, PHYSICAL, GATE):
        if not run.has(name):
            raise ValueError("Owner route lacks new independent review evidence: " + name)
    review, physical, gate = (run.read(name) for name in (REVIEW, PHYSICAL, GATE))
    if (gate.get("version") != "post-corpus-world-review/v1" or
            gate.get("status") != "responsibility_repair_required" or
            gate.get("historical_lineage_sha256") != cfg["lineage_sha256"] or
            gate.get("new_review_sha256") != sha_file(run.dir / REVIEW) or
            gate.get("new_physical_receipt_sha256") != sha_file(run.dir / PHYSICAL) or
            gate.get("first_request_boundary_sha256") != cfg["first_request_boundary_sha256"] or
            gate.get("new_physical_requests") != physical.get("new_physical_requests") or
            gate.get("truth_changed") is not False or
            gate.get("author_repair_started") is not False or
            gate.get("production_ready") is not False or
            gate.get("provider_calls_by_gate") != 0):
        raise ValueError("New world gate does not admit original owner routing")
    if (review.get("status") != "failed" or
            not (review.get("repair_targets") or {}).get("intrinsic") or
            (review.get("repair_targets") or {}).get("structure") is not False):
        raise ValueError("Independent opinion has no permitted intrinsic-only repair")
    expected_physical = audit_new_review(review, run.dir / "llm_attempts.jsonl",
        run.dir / "prompts.jsonl",
        original_trace_bytes=migration["historical_trace_bytes"],
        original_trace_sha256=migration["historical_trace_sha256"],
        original_prompt_bytes=migration["historical_prompt_bytes"],
        original_prompt_sha256=migration["historical_prompt_sha256"])
    if physical != expected_physical or physical.get("review_sha256") != digest(review):
        raise ValueError("New independent opinion is not physically bound")
    wp, task = run.read("01_whitepaper.json"), run.read("00_input.json")
    ws = WorldState.from_dict(run.read("02_world.json"))
    if world_semantics.validate_review(review, wp, ws, task_input=task) != [
            {"code": "world_review_not_passed", "status": "failed"}]:
        raise ValueError("New independent opinion cannot replay against the candidate")
    if (sha_file(run.dir / "02_world_review.json") !=
            lineage["historical_world_review_sha256"]):
        raise ValueError("Historical world opinion changed before owner routing")
    materialization = run.read("00_cumulative_materialization_receipt.json")
    derivation_path = run.dir / "00_cumulative_derivation_receipt.json"
    if not derivation_path.is_file() or derivation_path.is_symlink():
        raise ValueError("Original inherited file inventory is missing")
    derivation = run.read("00_cumulative_derivation_receipt.json")
    inherited = derivation.get("inherited_run_files")
    if (not isinstance(inherited, dict) or
            materialization.get("version") != "cumulative-child-materialization/v1" or
            materialization.get("derivation_receipt_sha256") != sha_file(derivation_path) or
            materialization.get("original_files") != len(inherited) or
            materialization.get("original_file_inventory_sha256") !=
                hashlib.sha256(_encode(inherited)).hexdigest() or
            derivation.get("archive_sha256") != lineage.get("archive_sha256") or
            materialization.get("evidence_sha256", {}).get(LINEAGE) !=
                cfg["lineage_sha256"] or
            materialization.get("evidence_sha256", {}).get(MIGRATION) !=
                cfg["migration_sha256"]):
        raise ValueError("Original inherited file inventory is not bound to this child")
    checkpoint_name = _checkpoint_name(wp)
    for name in ("02_world_draft.json", checkpoint_name):
        entry = inherited.get(name)
        path = run.dir / name
        if (not isinstance(entry, dict) or path.is_symlink() or not path.is_file() or
                path.stat().st_size != entry.get("bytes") or
                sha_file(path) != entry.get("sha256")):
            raise ValueError("Original author file differs from the inherited bytes: " + name)
    draft = run.read("02_world_draft.json")
    if (draft.get("version") != WORLD_DRAFT_VERSION or
            draft.get("generation_strategy") != "agentic" or
            draft.get("status") != "completed" or
            draft.get("existing_base") is not None or
            draft.get("wp_hash") != _world_digest(wp) or
            draft.get("draft_hash") != _world_digest({k: v for k, v in draft.items() if k != "draft_hash"}) or
            draft.get("candidate_world_hash") != _world_digest(draft.get("candidate_world"))):
        raise ValueError("Original author draft is stale or changed")
    agent = draft.get("agent") or {}
    if (agent.get("status") != "completed" or
            agent.get("checkpoint_hash") != _world_digest({k: v for k, v in agent.items()
                                                            if k != "checkpoint_hash"}) or
            run.read(checkpoint_name) != agent):
        raise ValueError("Original author checkpoint differs from its completed draft")
    if any(row.get("status") == "pending" for row in agent.get("feedback_history", [])):
        raise ValueError("An original author review transaction is already pending")
    if any(row.get("version") == VALUE_TRANSACTION_VERSION and row.get("status") == "running"
           for row in agent.get("local_repairs", [])):
        raise ValueError("An original local repair transaction is already running")
    groups = review_value_groups(wp, agent, review)
    used = {unit_id: sum(row.get("unit_id") == unit_id
                         for row in agent.get("local_repairs", [])) for unit_id in groups}
    if any(count >= 3 for count in used.values()):
        raise ValueError("Original local repair allowance is exhausted")
    targets = review["repair_targets"]
    call_limit = min(12, max(4, len(targets["intrinsic"]) + 2))
    return {"version": VERSION, "status": "original_owner_transaction_required",
            "new_review_sha256": sha_file(run.dir / REVIEW),
            "new_physical_receipt_sha256": sha_file(run.dir / PHYSICAL),
            "historical_draft_sha256": sha_file(run.dir / "02_world_draft.json"),
            "historical_checkpoint_sha256": sha_file(run.dir / checkpoint_name),
            "owner_groups": groups, "local_repairs_used": used,
            "original_factory_repair_call_limit": call_limit,
            "truth_changed": False, "author_repair_started": False,
            "provider_calls_by_plan": 0, "paid_resume_authorization": False,
            "production_ready": False}
