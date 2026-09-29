"""Read-only admission proof before constructing a cumulative child run.

This module never copies files, opens a Run, or calls a provider. A validated
permit only authorizes the dedicated entry to begin its own child-directory
transaction; the bounded call wrapper still enforces every money/call cap.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


_SEAL = object()


class PrewritePermit:
    __slots__ = ("run_id", "destination", "migration_sha256", "authorization_sha256", "_seal")

    def __init__(self, run_id: str, destination: Path, migration_sha256: str,
                 authorization_sha256: str, *, _seal=None):
        if _seal is not _SEAL:
            raise ValueError("A cumulative prewrite permit requires validated evidence")
        self.run_id = run_id
        self.destination = destination
        self.migration_sha256 = migration_sha256
        self.authorization_sha256 = authorization_sha256
        self._seal = _SEAL


def is_valid_permit(value, run_id: str) -> bool:
    return (isinstance(value, PrewritePermit) and value._seal is _SEAL
            and value.run_id == run_id and value.destination.is_absolute()
            and value.destination.name == run_id)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _amount(value) -> Decimal:
    if type(value) not in (str, int, float):
        raise ValueError("Cumulative cap must be a finite nonnegative number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Cumulative cap must be a finite nonnegative number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cumulative cap must be a finite nonnegative number")
    return amount


def _safe_member(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith("/") and "\\" not in name and ".." not in parts and not any(
        ":" in part or part.lower().startswith(".env") or "credential" in part.lower()
        or "secret" in part.lower() for part in parts)


def validate_prewrite(*, derived_root: Path, destination: Path, source_root: Path,
                      lineage_path: Path, stage_path: Path, source_epoch_path: Path,
                      scope_path: Path, migration_path: Path, authorization_path: Path,
                      boundary_path: Path, world_migration_path: Path,
                      downstream_evidence_path: Path, expected_archive_sha256: str,
                      _allow_existing_pristine: bool = False) -> PrewritePermit:
    """Bind frozen history, a full migration, and specific cap authority with zero writes."""
    derived_root, destination, source_root = (path.resolve() for path in
        (derived_root, destination, source_root))
    if ((destination.exists() and not _allow_existing_pristine)
            or (_allow_existing_pristine and (not destination.is_dir() or destination.is_symlink()))
            or destination.parent != (source_root / "output" / "runs").resolve()):
        raise ValueError("Cumulative child must use a fresh dedicated run directory")
    receipt_path = derived_root / "evidence" / "derivation_receipt.json"
    receipt = _load(receipt_path)
    inherited = receipt.get("inherited_run_files")
    lineage, stage, epoch, scope, migration, authorization, boundary, world, evidence = (
        _load(path) for path in (lineage_path, stage_path, source_epoch_path, scope_path,
                                migration_path, authorization_path, boundary_path,
                                world_migration_path, downstream_evidence_path))
    lineage_sha, stage_sha, epoch_sha, scope_sha = (
        _sha(path) for path in (lineage_path, stage_path, source_epoch_path, scope_path))
    if (not isinstance(inherited, dict) or len(inherited) != 348
            or receipt.get("runnable_run_created") is not False
            or receipt.get("admission_reopened") is not False
            or lineage.get("status") != "historical_lineage_verified"
            or lineage.get("archive_sha256") != receipt.get("archive_sha256")
            or lineage.get("archive_sha256") != expected_archive_sha256
            or lineage.get("provider_calls") != 0
            or stage.get("status") != "historical_stage_inventory_verified"
            or stage.get("parent_lineage_sha256") != lineage_sha
            or stage.get("derivation_receipt_sha256") != _sha(receipt_path)
            or epoch.get("status") != "source_epochs_inventoried"
            or epoch.get("historical_lineage_sha256") != lineage_sha
            or epoch.get("target_fully_committed") is not True
            or epoch.get("target_uncommitted_source_paths") != []
            or scope.get("status") != "historical_migration_scope_classified"
            or scope.get("parent_lineage_sha256") != lineage_sha
            or scope.get("historical_stage_inventory_sha256") != stage_sha
            or scope.get("source_epoch_sha256") != epoch_sha
            or scope.get("world_migration_sha256") != _sha(world_migration_path)
            or scope.get("derivation_receipt_sha256") != _sha(receipt_path)
            or scope.get("historical_file_count") != 348
            or scope.get("historical_file_bytes") != receipt.get("inherited_run_bytes")
            or scope.get("complete_source_migration") is not False
            or scope.get("paid_resume_authorization") is not False
            or scope.get("provider_calls") != 0
            or set(scope.get("files") or {}) != set(inherited)):
        raise ValueError("Frozen cumulative lineage or migration scope is incomplete")
    if (world.get("version") != "world-checkpoint-source-migration/v1"
            or world.get("provider_calls") != 0
            or world.get("all_original_fields_retained") is not True
            or world.get("semantic_revalidation") is not False
            or world.get("publication_or_signal_migration") is not False
            or world.get("paid_resume_authorization") is not False
            or world.get("parent_sha256")
               != inherited.get("02_world_agent_5903517308903f10c8b0.json", {}).get("sha256")
            or not (set(world.get("target_compiler_source_hashes") or {})
                    & set(epoch.get("target_source_hashes") or {}))
            or any(epoch.get("target_source_hashes", {}).get(name) != expected for name, expected
                   in world.get("target_compiler_source_hashes", {}).items()
                   if name in epoch.get("target_source_hashes", {}))):
        raise ValueError("The old world checkpoint has no bound source migration")
    for ref, field in (("HEAD", "target_head"), ("HEAD^{tree}", "target_tree")):
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=source_root).decode().strip()
        if actual != epoch.get(field) or actual != scope.get(field):
            raise ValueError("Target Git source epoch changed before cumulative child creation")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all",
            "--", "pipeline", "eval", "config.py", "execution_control.py",
            "llm_transport.py", "llm_trace.py", "tools/diversity_metrics.py",
            "tools/run_original_bc_smoke.py"], cwd=source_root):
        raise ValueError("Target production source is not committed cleanly")
    for name, expected in epoch.get("target_source_hashes", {}).items():
        if not _safe_member(name) or not name.endswith(".py") or _sha(source_root / name) != expected:
            raise ValueError("Target production source changed: " + name)
    if not epoch.get("target_source_hashes"):
        raise ValueError("Target source epoch is empty")
    if (migration.get("version") != "complete-cumulative-source-migration/v1"
            or migration.get("status") != "complete"
            or migration.get("complete_source_migration") is not True
            or migration.get("parent_lineage_sha256") != lineage_sha
            or migration.get("historical_stage_inventory_sha256") != stage_sha
            or migration.get("source_epoch_sha256") != epoch_sha
            or migration.get("migration_scope_sha256") != scope_sha
            or migration.get("world_migration_sha256") != _sha(world_migration_path)
            or migration.get("target_head") != epoch["target_head"]
            or migration.get("target_tree") != epoch["target_tree"]
            or migration.get("target_source_hashes") != epoch["target_source_hashes"]
            or migration.get("provider_calls") != 0
            or migration.get("downstream_results_current") is not False
            or migration.get("stage_revalidation_completed") is not False
            or migration.get("new_world_opinion") is not False
            or migration.get("paid_resume_authorization") is not False):
        raise ValueError("Complete cumulative source migration is absent or unbound")
    historical_bindings = {
        "historical_profile_sha256": lineage.get("budget", {}).get("profile_sha256"),
        "historical_trace_sha256": lineage.get("budget", {}).get("trace_sha256"),
        "historical_trace_bytes": lineage.get("historical_trace_bytes"),
        "historical_prompt_sha256": lineage.get("historical_prompt_sha256"),
        "historical_prompt_bytes": lineage.get("historical_prompt_bytes"),
        "historical_review_sha256": lineage.get("historical_world_review_sha256"),
        "historical_review_attempts_sha256": lineage.get("historical_world_review_attempts_sha256"),
        "historical_production_sha256": lineage.get("historical_production_sha256"),
        "initial_source_map_sha256": lineage.get("initial_source_map_sha256"),
        "ending_source_map_sha256": lineage.get("ending_source_map_sha256"),
        "historical_file_count": 348,
        "historical_file_bytes": receipt["inherited_run_bytes"],
        "first_new_operation": "post_corpus_world.semantic_review",
    }
    if any(migration.get(name) != expected for name, expected in historical_bindings.items()):
        raise ValueError("Complete cumulative migration omitted a historical binding")
    run = derived_root / "run"
    if (_sha(derived_root / "evidence" / "parent_manifest.json")
            != receipt.get("parent_manifest_sha256")
            or _sha(run / "manifest.json") != receipt.get("reconciled_manifest_sha256")):
        raise ValueError("Historical manifest changed")
    for name, expected in inherited.items():
        if (not _safe_member(name)
                or scope["files"].get(name, {}).get("sha256") != expected.get("sha256")
                or scope["files"].get(name, {}).get("bytes") != expected.get("bytes")):
            raise ValueError("Historical file inventory differs from classified migration scope")
        path = run.joinpath(*PurePosixPath(name).parts)
        if (not path.is_file() or (name != "manifest.json" and
                (path.stat().st_size != expected.get("bytes") or _sha(path) != expected.get("sha256")))):
            raise ValueError("Historical run file changed before child creation: " + name)
    from tools.audit_cumulative_migration_scope import classify
    categories = Counter()
    for name, row in scope["files"].items():
        if row.get("migration_scope") != classify(name):
            raise ValueError("Historical migration scope relabeled a file: " + name)
        categories[row["migration_scope"]] += 1
    if dict(categories) != scope.get("migration_scope_counts"):
        raise ValueError("Historical migration scope denominator changed")
    for name, expected in (stage.get("historical_artifact_sha256") or {}).items():
        if name not in inherited or inherited[name].get("sha256") != expected:
            raise ValueError("Historical stage artifact is not retained: " + name)
    if (stage.get("production", {}).get("identity")
            != lineage.get("identity", {}).get("production_identity")):
        raise ValueError("Historical production identity is not retained")
    profile, trace, prompts = (run / name for name in
        ("experiment_profile.json", "llm_attempts.jsonl", "prompts.jsonl"))
    old = _load(profile)
    budget = lineage.get("budget") or {}
    original_calls = budget.get("admitted_calls")
    if (type(original_calls) is not int or original_calls != 1026
            or old.get("admitted_calls") != original_calls
            or old.get("settled_usage_calls") != original_calls
            or old.get("unsettled_reservation_calls") != 0
            or old.get("stopped") is not True
            or _amount(old.get("reported_usage_estimate_cny"))
               != _amount(budget.get("reported_usage_estimate_cny"))
            or _sha(profile) != budget.get("profile_sha256")
            or _sha(trace) != budget.get("trace_sha256")
            or _sha(prompts) != lineage.get("historical_prompt_sha256")):
        raise ValueError("Frozen cumulative call ledger or trace changed")
    run_id = lineage.get("identity", {}).get("run_id")
    if (not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id
            or destination.name != run_id):
        raise ValueError("Cumulative child must preserve the original run identity")
    if (authorization.get("policy") != "explicit-cumulative-cap-authorization/v1"
            or authorization.get("user_authorization") is not True
            or authorization.get("migration_sha256") != _sha(migration_path)
            or authorization.get("first_request_boundary_sha256") != _sha(boundary_path)
            or authorization.get("allowed_first_step") != "world.semantic_review"
            or authorization.get("parent_profile_sha256") != budget.get("profile_sha256")
            or authorization.get("parent_trace_sha256") != budget.get("trace_sha256")
            or authorization.get("model") != budget.get("model")
            or authorization.get("old_max_calls") != budget.get("max_calls")
            or _amount(authorization.get("old_max_cny")) != _amount(budget.get("max_cny"))
            or type(authorization.get("new_max_calls")) is not int
            or authorization["new_max_calls"] < budget["max_calls"]
            or _amount(authorization.get("new_max_cny")) <= _amount(budget.get("max_cny"))):
        raise ValueError("A matching explicit cumulative cap authorization is required")
    if (boundary.get("version") != "derived-run-first-review-boundary/v1"
            or boundary.get("provider_calls") != 0
            or boundary.get("new_semantic_opinion") is not False
            or boundary.get("budget_stop_preserved") is not True
            or boundary.get("first_step") != "world.semantic_review"
            or boundary.get("derived_manifest_sha256") != receipt.get("reconciled_manifest_sha256")
            or boundary.get("ledger_sha256") != budget.get("profile_sha256")
            or boundary.get("trace_sha256") != budget.get("trace_sha256")
            or boundary.get("prompt_archive_sha256") != lineage.get("historical_prompt_sha256")):
        raise ValueError("First independent review request is not bound to the old ledger")
    if any(not isinstance(boundary.get(name), str) or len(boundary[name]) != 64
           for name in ("messages_hash", "params_hash")):
        raise ValueError("First independent review request bytes are not bound")
    if (boundary.get("evidence_hash") != evidence.get("envelope_hash")
            or evidence.get("new_world_opinion") is not False
            or evidence.get("paid_resume_authorization") is not False
            or boundary.get("model") != budget.get("model")):
        raise ValueError("Downstream evidence is not bound as review input only")
    return PrewritePermit(run_id, destination, _sha(migration_path),
                          _sha(authorization_path), _seal=_SEAL)
