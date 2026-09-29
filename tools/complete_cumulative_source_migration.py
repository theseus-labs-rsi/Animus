"""Close the offline source-and-history migration inventory for one frozen run.

Complete here means every historical byte has a declared fate under the target
source epoch. It does not certify new business truth, make the child runnable,
authorize spending, or accept any downstream stage under the changed source.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
import os
import sys

VERSION = "complete-cumulative-source-migration/v1"
FIRST = "post_corpus_world.semantic_review"


def install_offline_guard():
    def denied(*_args, **_kwargs):
        raise AssertionError("Complete source migration forbids network calls")
    socket.socket.connect = socket.socket.connect_ex = denied
    socket.create_connection = socket.getaddrinfo = denied
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["OPENAI_API_KEY"] = "offline-disabled"
    os.environ["MODEL"] = "offline-disabled"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def complete(*, derived_root: Path, source_root: Path, lineage_path: Path,
             stage_path: Path, source_epoch_path: Path, world_migration_path: Path,
             scope_path: Path, cost_path: Path, physical_path: Path,
             boundary_path: Path, downstream_evidence_path: Path) -> dict:
    install_offline_guard()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.audit_cumulative_migration_scope import audit as audit_scope
    lineage, stage, epoch, world, scope, cost, physical, boundary, evidence = (
        load(path) for path in (lineage_path, stage_path, source_epoch_path,
            world_migration_path, scope_path, cost_path, physical_path,
            boundary_path, downstream_evidence_path))
    source_boundary_path = boundary_path.parent / "first_downstream_review_request_boundary.json"
    source_boundary = load(source_boundary_path)
    recalculated = audit_scope(derived_root, lineage_path, stage_path,
        source_epoch_path, world_migration_path, source_root=source_root)
    if scope != recalculated:
        raise ValueError("The complete migration scope no longer matches the frozen run")
    budget = lineage.get("budget") or {}
    prices = cost.get("reconciliation") or {}
    physical_count = 52
    if (cost.get("version") != "offline-cumulative-cost-baseline/v1"
            or cost.get("source", {}).get("trace_sha256") != budget.get("trace_sha256")
            or cost.get("source", {}).get("profile_sha256") != budget.get("profile_sha256")
            or prices.get("requests") != budget.get("admitted_calls")
            or prices.get("responses_with_usage") != budget.get("settled_usage_calls")
            or prices.get("unsettled_reservations") != budget.get("unsettled_reservation_calls")
            or prices.get("max_calls") != budget.get("max_calls")
            or Decimal(str(prices.get("max_cny"))) != Decimal(str(budget.get("max_cny")))
            or Decimal(str(prices.get("reported_usage_cost_cny")))
               != Decimal(str(budget.get("physical_trace_usage_cost_cny")))
            or physical.get("version") != "offline-world-review-physical-binding/v1"
            or physical.get("review_sha256") != lineage.get("historical_world_review_sha256")
            or physical.get("trace_sha256") != budget.get("trace_sha256")
            or physical.get("prompt_log_sha256") != lineage.get("historical_prompt_sha256")
            or any(physical.get(key) != physical_count for key in (
                "bound_transcript_calls", "physical_requests_with_usage",
                "parsed_results_equal_saved_output", "prompt_log_outputs_equal_saved_output"))
            or physical.get("new_world_opinion") is not False
            or physical.get("business_truth_certified") is not False
            or physical.get("provider_calls_by_this_audit") != 0):
        raise ValueError("Original cumulative cost or historical physical review is unbound")
    if (boundary.get("version") != "derived-run-first-review-boundary/v1"
            or boundary.get("provider_calls") != 0
            or boundary.get("budget_stop_preserved") is not True
            or boundary.get("new_semantic_opinion") is not False
            or boundary.get("first_step") != "world.semantic_review"
            or boundary.get("derived_manifest_sha256")
               != lineage.get("historical_copy", {}).get("reconciled_manifest_sha256")
            or boundary.get("ledger_sha256") != budget.get("profile_sha256")
            or boundary.get("trace_sha256") != budget.get("trace_sha256")
            or boundary.get("prompt_archive_sha256") != lineage.get("historical_prompt_sha256")
            or boundary.get("model") != budget.get("model")
            or source_boundary.get("version") != "first-downstream-review-request-boundary/v1"
            or source_boundary.get("provider_calls") != 0
            or source_boundary.get("step") != boundary.get("first_step")
            or source_boundary.get("model") != boundary.get("model")
            or source_boundary.get("messages_hash") != boundary.get("messages_hash")
            or source_boundary.get("params_hash") != boundary.get("params_hash")
            or source_boundary.get("evidence_hash") != boundary.get("evidence_hash")
            or boundary.get("evidence_hash") != evidence.get("envelope_hash")
            or evidence.get("new_world_opinion") is not False
            or evidence.get("paid_resume_authorization") is not False):
        raise ValueError("First new review boundary or evidence envelope changed")
    if (stage.get("historical_world_review_attempts")
            != lineage.get("historical_world_review_attempts")
            or stage.get("production", {}).get("identity")
               != lineage.get("identity", {}).get("production_identity")
            or stage.get("corpus", {}).get("release_eligible") is not False
            or stage.get("author_slots", {}).get("attempts_reset") is not False
            or stage.get("orders", {}).get("raw") != 42
            or stage.get("orders", {}).get("effective") != 42
            or stage.get("author_slots", {}).get("active_groups") != 119
            or stage.get("author_slots", {}).get("admitted_slots") != 131
            or stage.get("author_slots", {}).get("saved_author_replies") != 130):
        raise ValueError("Historical stage identities, attempts, or denominators changed")
    if (scope.get("historical_file_count") != 348
            or scope.get("historical_ledger_append_only") is not True
            or scope.get("historical_stage_attempts_preserved") is not True
            or scope.get("first_new_operation") != FIRST
            or scope.get("downstream_results_current") is not False
            or scope.get("paid_resume_authorization") is not False
            or world.get("semantic_revalidation") is not False
            or world.get("publication_or_signal_migration") is not False):
        raise ValueError("Historical work was promoted to current acceptance")
    return {
        "version": VERSION, "status": "complete",
        "meaning": "all historical files classified and bound; downstream stages require current review",
        "parent_lineage_sha256": sha(lineage_path),
        "historical_stage_inventory_sha256": sha(stage_path),
        "source_epoch_sha256": sha(source_epoch_path),
        "world_migration_sha256": sha(world_migration_path),
        "migration_scope_sha256": sha(scope_path),
        "historical_cost_baseline_sha256": sha(cost_path),
        "historical_review_physical_binding_sha256": sha(physical_path),
        "first_review_boundary_sha256": sha(boundary_path),
        "source_review_boundary_sha256": sha(source_boundary_path),
        "downstream_evidence_sha256": sha(downstream_evidence_path),
        "historical_profile_sha256": budget["profile_sha256"],
        "historical_trace_sha256": budget["trace_sha256"],
        "historical_trace_bytes": lineage["historical_trace_bytes"],
        "historical_prompt_sha256": lineage["historical_prompt_sha256"],
        "historical_prompt_bytes": lineage["historical_prompt_bytes"],
        "historical_review_sha256": lineage["historical_world_review_sha256"],
        "historical_review_attempts_sha256": lineage["historical_world_review_attempts_sha256"],
        "historical_production_sha256": lineage["historical_production_sha256"],
        "initial_source_map_sha256": lineage["initial_source_map_sha256"],
        "ending_source_map_sha256": lineage["ending_source_map_sha256"],
        "historical_file_count": scope["historical_file_count"],
        "historical_file_bytes": scope["historical_file_bytes"],
        "historical_run_id": lineage["identity"]["run_id"],
        "historical_production_identity": lineage["identity"]["production_identity"],
        "historical_attempt_counters_preserved": True,
        "historical_candidate_documents": stage["corpus"]["candidate_documents"],
        "historical_author_slots": stage["author_slots"],
        "historical_orders": stage["orders"],
        "target_head": epoch["target_head"], "target_tree": epoch["target_tree"],
        "target_source_hashes": epoch["target_source_hashes"],
        "source_delta": epoch["source_delta"],
        "first_new_operation": FIRST,
        "stage_validity": {
            "input_and_seed": "historical_identity_retained",
            "world_facts": "compiled_projection_migrated_only",
            "world_semantic_review": "pending_new_independent_review",
            "disclosure_and_orders": "historical_only_pending_truth_decision",
            "corpus": "historical_failed_candidate_not_release_eligible",
            "grounding_quality_release": "not_run"},
        "complete_source_migration": True,
        "stage_revalidation_completed": False,
        "downstream_results_current": False,
        "new_world_opinion": False,
        "runnable_run_created": False,
        "paid_resume_authorization": False,
        "provider_calls": 0,
        "historical_run_modified": False,
    }


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("lineage", "stage", "source_epoch", "world_migration", "scope",
                 "cost", "physical", "boundary", "downstream_evidence", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.derived_root.resolve()):
        parser.error("Use fresh evidence outside the immutable derived copy")
    result = complete(derived_root=args.derived_root, source_root=args.source_root,
        lineage_path=args.lineage, stage_path=args.stage,
        source_epoch_path=args.source_epoch, world_migration_path=args.world_migration,
        scope_path=args.scope, cost_path=args.cost, physical_path=args.physical,
        boundary_path=args.boundary, downstream_evidence_path=args.downstream_evidence)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "historical_file_count", "complete_source_migration",
        "stage_revalidation_completed", "paid_resume_authorization", "provider_calls")}))


if __name__ == "__main__":
    main()
