"""Dedicated independent world review after a failed corpus candidate.

This is a transaction for an explicitly migrated cumulative run. It preserves
the historical world opinion, refuses a stopped ledger, records a new opinion
separately, and verifies new physical calls before classifying any repair.
The resulting gate never marks production ready or edits canonical truth.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
import subprocess

from pipeline.world_review_physical import audit_new_review, digest, sha_file


VERSION = "post-corpus-world-review/v1"
LINEAGE = "00_cumulative_lineage.json"
MIGRATION = "00_cumulative_source_migration.json"
EVIDENCE = "12_downstream_evidence.json"
AUTHORIZATION = "00_cumulative_cap_authorization.json"
BOUNDARY = "00_first_world_review_boundary.json"
STAGE_INVENTORY = "00_cumulative_stage_inventory.json"
SOURCE_EPOCH = "00_cumulative_source_epoch.json"
REVIEW = "12_post_corpus_world_review.json"
PHYSICAL = "12_post_corpus_world_review_physical.json"
GATE = "12_post_corpus_world_review_gate.json"
HISTORICAL_STAGE_FILES = (
    "02_world_review.json", "02_world_review_attempts.json",
    "02_disclosure_ready.json", "02_disclosure_plan_attempts.json",
    "03_raw_orders.json", "03_orders.json", "05_corpus.json",
    "05_corpus_candidate.json", "05_corpus.ckpt.json",
    "05_corpus_warning.json", "05_corpus_token_scale.json",
    "11_production.json",
)


def _read_bound(run, name, expected_sha):
    path = run.dir / name
    if not path.is_file() or sha_file(path) != expected_sha:
        raise ValueError("Cumulative recovery input is missing or changed: " + name)
    return json.loads(path.read_text(encoding="utf-8"))


def _money(value):
    if type(value) not in (int, float, str):
        raise ValueError("Cumulative cap is not a finite number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Cumulative cap is not a finite number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cumulative cap is not a finite nonnegative amount")
    return amount


def _verify_migration_layers(run, lineage, migration, stage_inventory, source_epoch,
                             lineage_sha, stage_sha, epoch_sha):
    """Bind the hypothetical full migration to audited stage and Git epochs."""
    if (migration.get("historical_stage_inventory_sha256") != stage_sha
            or migration.get("source_epoch_sha256") != epoch_sha
            or stage_inventory.get("version") != "cumulative-stage-retention-audit/v1"
            or stage_inventory.get("status") != "historical_stage_inventory_verified"
            or stage_inventory.get("parent_lineage_sha256") != lineage_sha
            or stage_inventory.get("provider_calls") != 0
            or stage_inventory.get("complete_source_migration") is not False
            or stage_inventory.get("runnable_run_created") is not False
            or stage_inventory.get("business_truth_certified") is not False
            or stage_inventory.get("new_world_opinion") is not False
            or (stage_inventory.get("production") or {}).get("identity")
               != lineage["identity"]["production_identity"]
            or (stage_inventory.get("corpus") or {}).get("release_eligible") is not False
            or (stage_inventory.get("author_slots") or {}).get("attempts_reset") is not False):
        raise ValueError("Historical stage retention is not proved for cumulative migration")
    historical = stage_inventory.get("historical_artifact_sha256") or {}
    if set(historical) != set(HISTORICAL_STAGE_FILES):
        raise ValueError("Historical stage artifact denominator is incomplete")
    for name in HISTORICAL_STAGE_FILES:
        if (not isinstance(historical[name], str) or len(historical[name]) != 64
                or sha_file(run.dir / name) != historical[name]):
            raise ValueError("Historical stage artifact changed before the independent review: " + name)
    if (source_epoch.get("version") != "cumulative-source-epoch-audit/v1"
            or source_epoch.get("status") != "source_epochs_inventoried"
            or source_epoch.get("historical_lineage_sha256") != lineage_sha
            or source_epoch.get("legacy_source_map_sha256")
               != lineage["ending_source_map_sha256"]
            or source_epoch.get("target_fully_committed") is not True
            or source_epoch.get("target_uncommitted_source_paths") != []
            or source_epoch.get("complete_source_migration") is not False
            or source_epoch.get("provider_calls") != 0
            or migration.get("target_head") != source_epoch.get("target_head")
            or migration.get("target_tree") != source_epoch.get("target_tree")
            or migration.get("target_source_hashes")
               != source_epoch.get("target_source_hashes")):
        raise ValueError("Target source epoch is not proved for cumulative migration")
    from tools.run_original_bc_smoke import ROOT, implementation_hashes
    if source_epoch["target_source_hashes"] != implementation_hashes():
        raise ValueError("Target source bytes changed after the source-epoch audit")
    for ref, field in (("HEAD", "target_head"), ("HEAD^{tree}", "target_tree")):
        resolved = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT).decode().strip()
        if source_epoch[field] != resolved:
            raise ValueError("Target Git commit or tree changed after the source-epoch audit")
    source_paths = ("pipeline", "eval", "config.py", "execution_control.py",
                    "llm_transport.py", "llm_trace.py", "tools/diversity_metrics.py",
                    "tools/run_original_bc_smoke.py")
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1",
        "--untracked-files=all", "--", *source_paths], cwd=ROOT)
    if dirty:
        raise ValueError("Target production source has uncommitted changes")


class _FirstRequestBoundTracer:
    """Reject changed first-request bytes before delegating to the call gate."""

    def __init__(self, tracer, boundary):
        self.tracer = tracer
        self.boundary = boundary
        self.pfile = getattr(tracer, "pfile", None)
        self.first_checked = False

    def chat_json(self, step, messages, **params):
        if not self.first_checked:
            if (step != self.boundary["first_step"]
                    or digest(messages) != self.boundary["messages_hash"]
                    or digest(params) != self.boundary["params_hash"]
                    or params.get("model") != self.boundary["model"]):
                raise ValueError("First world-review provider request differs from the frozen offline boundary")
            self.first_checked = True
        return self.tracer.chat_json(step, messages, **params)

    def chat_text(self, *_args, **_kwargs):
        raise ValueError("The post-corpus world-review transaction only permits JSON review requests")


def _closed_before_review(run):
    """Validate all authority and source anchors without a provider call."""
    from pipeline.cumulative_entry_guard import require_bounded_run
    require_bounded_run(run.manifest.get("run_id"))
    if run.has(REVIEW) or run.has(PHYSICAL) or run.has(GATE):
        raise ValueError("An existing post-corpus review needs reconciliation, not a repeated call")
    cfg = (run.manifest.get("config") or {}).get("cumulative_recovery") or {}
    if cfg.get("version") != "explicit-cumulative-source-migration/v1":
        raise ValueError("No explicit cumulative migration control is installed")
    for key in ("lineage_sha256", "migration_sha256", "evidence_sha256",
                "authorization_sha256", "first_request_boundary_sha256",
                "stage_inventory_sha256", "source_epoch_sha256"):
        if not isinstance(cfg.get(key), str) or len(cfg[key]) != 64:
            raise ValueError("Cumulative recovery control lacks " + key)
    lineage = _read_bound(run, LINEAGE, cfg["lineage_sha256"])
    migration = _read_bound(run, MIGRATION, cfg["migration_sha256"])
    evidence = _read_bound(run, EVIDENCE, cfg["evidence_sha256"])
    authorization = _read_bound(run, AUTHORIZATION, cfg["authorization_sha256"])
    boundary = _read_bound(run, BOUNDARY, cfg["first_request_boundary_sha256"])
    stage_inventory = _read_bound(run, STAGE_INVENTORY, cfg["stage_inventory_sha256"])
    source_epoch = _read_bound(run, SOURCE_EPOCH, cfg["source_epoch_sha256"])
    if (lineage.get("status") != "historical_lineage_verified"
            or lineage.get("complete_source_migration") is not False
            or lineage.get("runnable_run_created") is not False
            or type(lineage.get("identity", {}).get("layout_revisions")) is not int
            or not 0 <= lineage["identity"]["layout_revisions"] <= 3
            or migration.get("version") != "complete-cumulative-source-migration/v1"
            or migration.get("status") != "complete"
            or migration.get("complete_source_migration") is not True
            or migration.get("historical_attempt_counters_preserved") is not True
            or migration.get("first_new_operation") != "post_corpus_world.semantic_review"
            or migration.get("stage_revalidation_completed") is not False
            or migration.get("downstream_results_current") is not False
            or migration.get("new_world_opinion") is not False
            or migration.get("paid_resume_authorization") is not False
            or migration.get("parent_lineage_sha256") != cfg["lineage_sha256"]
            or migration.get("historical_profile_sha256")
               != lineage.get("budget", {}).get("profile_sha256")
            or migration.get("historical_trace_sha256")
               != lineage.get("budget", {}).get("trace_sha256")
            or migration.get("historical_trace_bytes") != lineage.get("historical_trace_bytes")
            or migration.get("historical_prompt_sha256") != lineage.get("historical_prompt_sha256")
            or migration.get("historical_prompt_bytes") != lineage.get("historical_prompt_bytes")
            or migration.get("historical_review_sha256") != lineage.get("historical_world_review_sha256")
            or migration.get("historical_review_attempts_sha256")
               != lineage.get("historical_world_review_attempts_sha256")
            or migration.get("historical_production_sha256")
               != lineage.get("historical_production_sha256")
            or migration.get("initial_source_map_sha256") != lineage.get("initial_source_map_sha256")
            or migration.get("ending_source_map_sha256") != lineage.get("ending_source_map_sha256")
            or migration.get("provider_calls") != 0):
        raise ValueError("Full cumulative source migration is not proved")
    _verify_migration_layers(run, lineage, migration, stage_inventory, source_epoch,
        cfg["lineage_sha256"], cfg["stage_inventory_sha256"], cfg["source_epoch_sha256"])
    if (boundary.get("version") != "derived-run-first-review-boundary/v1"
            or boundary.get("provider_calls") != 0
            or boundary.get("new_semantic_opinion") is not False
            or boundary.get("budget_stop_preserved") is not True
            or boundary.get("first_step") != "world.semantic_review"
            or boundary.get("derived_manifest_sha256")
               != lineage.get("historical_copy", {}).get("reconciled_manifest_sha256")
            or boundary.get("ledger_sha256") != lineage["budget"]["profile_sha256"]
            or boundary.get("trace_sha256") != lineage["budget"]["trace_sha256"]
            or boundary.get("prompt_archive_sha256") != lineage["historical_prompt_sha256"]
            or boundary.get("evidence_hash") != evidence.get("envelope_hash")
            or boundary.get("model") != lineage["budget"]["model"]
            or any(not isinstance(boundary.get(key), str) or len(boundary[key]) != 64
                   for key in ("messages_hash", "params_hash"))):
        raise ValueError("First paid reviewer request is not bound to the frozen offline boundary")
    from tools.run_original_bc_smoke import implementation_hashes
    if migration.get("target_source_hashes") != implementation_hashes():
        raise ValueError("Target source epoch differs from the cumulative migration")
    profile = run.read("experiment_profile.json")
    original_calls = lineage["budget"]["admitted_calls"]
    old_max_calls = lineage["budget"]["max_calls"]
    old_max_cny = _money(lineage["budget"]["max_cny"])
    new_max_cny = _money(authorization.get("new_max_cny"))
    if (authorization.get("policy") != "explicit-cumulative-cap-authorization/v1"
            or authorization.get("user_authorization") is not True
            or authorization.get("migration_sha256") != cfg["migration_sha256"]
            or authorization.get("first_request_boundary_sha256") != cfg["first_request_boundary_sha256"]
            or authorization.get("allowed_first_step") != "world.semantic_review"
            or authorization.get("parent_profile_sha256") != lineage["budget"]["profile_sha256"]
            or authorization.get("parent_trace_sha256") != lineage["budget"]["trace_sha256"]
            or authorization.get("model") != lineage["budget"]["model"]
            or authorization.get("old_max_calls") != old_max_calls
            or _money(authorization.get("old_max_cny")) != old_max_cny
            or type(authorization.get("new_max_calls")) is not int
            or authorization["new_max_calls"] < old_max_calls
            or new_max_cny <= old_max_cny
            or profile.get("max_calls") != authorization["new_max_calls"]
            or not old_max_cny < _money(profile.get("max_cny")) <= new_max_cny):
        raise ValueError("No matching explicit cumulative cap authorization")
    if (profile.get("stopped") is not False or profile.get("source_drift")
            or profile.get("unsettled_reservation_calls") != 0
            or profile.get("admitted_calls") != original_calls
            or profile.get("settled_usage_calls") != original_calls
            or profile.get("model") != lineage["budget"]["model"]
            or _money(profile.get("reported_usage_estimate_cny"))
               != _money(lineage["budget"]["reported_usage_estimate_cny"])
            or profile.get("max_calls", 0) <= original_calls):
        raise ValueError("Cumulative call ledger is stopped, changed, or unsettled")
    trace = run.dir / "llm_attempts.jsonl"
    prompts = run.dir / "prompts.jsonl"
    trace_bytes = migration.get("historical_trace_bytes")
    prompt_bytes = migration.get("historical_prompt_bytes")
    prompt_sha = migration.get("historical_prompt_sha256")
    if (type(trace_bytes) is not int or type(prompt_bytes) is not int
            or trace.stat().st_size != trace_bytes or prompts.stat().st_size != prompt_bytes
            or sha_file(trace) != migration["historical_trace_sha256"]
            or sha_file(prompts) != prompt_sha):
        raise ValueError("The first new physical request boundary changed")
    if (run.manifest.get("run_id") != lineage["identity"]["run_id"]
            or sha_file(run.dir / "02_world_review.json")
               != lineage["historical_world_review_sha256"]
            or sha_file(run.dir / "02_world_review_attempts.json")
               != lineage["historical_world_review_attempts_sha256"]
            or sha_file(run.dir / "11_production.json")
               != lineage["historical_production_sha256"]
            or sha_file(run.dir / "experiment_source_hashes.json")
               != lineage["initial_source_map_sha256"]
            or sha_file(run.dir / "experiment_source_hashes_at_end.json")
               != lineage["ending_source_map_sha256"]
            or run.read("11_production.json").get("identity")
               != lineage["identity"]["production_identity"]):
        raise ValueError("Historical production identity or world opinion changed")
    return evidence, lineage, migration, boundary, trace, prompts


def expected_gate(run, lineage, review, physical):
    """Derive the gate from a saved, physically checked independent opinion."""
    targets = review.get("repair_targets") or {}
    actionable_truth = bool(targets.get("intrinsic") or targets.get("structure"))
    # Structure feedback belongs to the original, exhausted layout allowance.
    layout_used = lineage.get("identity", {}).get("layout_revisions")
    status = ("original_layout_allowance_exhausted"
              if review.get("status") == "failed" and targets.get("structure") is True
              and layout_used >= 3
              else "responsibility_repair_required"
              if review.get("status") == "failed" and actionable_truth
              else "unresolved_corpus_failure")
    return {"version": VERSION, "status": status,
            "historical_lineage_sha256": sha_file(run.dir / LINEAGE),
            "new_review_sha256": sha_file(run.dir / REVIEW),
            "new_physical_receipt_sha256": sha_file(run.dir / PHYSICAL),
            "new_physical_requests": physical["new_physical_requests"],
            "first_request_boundary_sha256": sha_file(run.dir / BOUNDARY),
            "historical_layout_revisions": layout_used,
            "truth_changed": False, "author_repair_started": False,
            "production_ready": False, "provider_calls_by_gate": 0}


def run_once(run):
    """Call the original independent reviewer once, then stop at its decision."""
    evidence, lineage, migration, boundary, trace, prompts = _closed_before_review(run)
    from pipeline import world_semantics
    from pipeline.world_state import WorldState

    wp = run.read("01_whitepaper.json")
    task = run.read("00_input.json")
    ws = WorldState.from_dict(run.read("02_world.json"))
    historical = run.read(world_semantics.REVIEW_ARTIFACT)
    if historical.get("status") != "passed":
        raise ValueError("The historical passed opinion must remain the previous review")
    world_semantics._validate_downstream_evidence(evidence, wp, ws, task, historical)
    boundary_tracer = _FirstRequestBoundTracer(run.tracer, boundary)
    review = world_semantics.review_world(wp, ws, boundary_tracer, task_input=task,
        previous=historical, author_responses=None, downstream_evidence=evidence)
    run.write(REVIEW, review)
    if review.get("status") == "error":
        raise ValueError("The new independent world review did not produce an opinion")
    if not boundary_tracer.first_checked:
        raise ValueError("The new independent review never reached its bound first request")
    issues = world_semantics.validate_review(review, wp, ws, task_input=task)
    allowed = [] if review.get("status") == "passed" else [
        {"code": "world_review_not_passed", "status": review.get("status")}]
    if issues != allowed:
        raise ValueError("New independent world review cannot replay against current inputs")
    physical = audit_new_review(review, trace, prompts,
        original_trace_bytes=migration["historical_trace_bytes"],
        original_trace_sha256=migration["historical_trace_sha256"],
        original_prompt_bytes=migration["historical_prompt_bytes"],
        original_prompt_sha256=migration["historical_prompt_sha256"])
    run.write(PHYSICAL, physical)
    gate = expected_gate(run, lineage, review, physical)
    run.write(GATE, gate)
    return gate
