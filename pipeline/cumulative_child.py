"""Build the next cumulative control documents in memory, without a run write.

The caller must first obtain a sealed prewrite permit and must still create a
fresh child transaction. These values are deliberately not an entrypoint: no
directory, provider, or ordinary resume is opened here.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path


VERSION = "explicit-cumulative-source-migration/v1"
AUTH_POLICY = "explicit-cumulative-cap-authorization/v1"


def _money(value):
    if type(value) not in (int, float, str):
        raise ValueError("Cumulative money cap must be finite and nonnegative")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Cumulative money cap must be finite and nonnegative") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cumulative money cap must be finite and nonnegative")
    return amount


def build_control_documents(*, permit, parent_manifest_bytes: bytes,
                            parent_profile_bytes: bytes,
                            lineage: dict, migration: dict, authorization: dict,
                            lineage_sha256: str, migration_sha256: str,
                            evidence_sha256: str, authorization_sha256: str,
                            boundary_sha256: str, stage_sha256: str,
                            source_epoch_sha256: str, parent_manifest_sha256: str,
                            parent_profile_sha256: str) -> tuple[dict, dict]:
    """Describe a legal child manifest and ledger; preserve all old counts.

    Every supplied SHA must come from the read-only prewrite validation. The
    returned documents are merely candidates for a later atomic copy/install.
    """
    from pipeline.cumulative_prewrite import is_valid_permit

    hashes = (lineage_sha256, migration_sha256, evidence_sha256,
              authorization_sha256, boundary_sha256, stage_sha256,
              source_epoch_sha256, parent_manifest_sha256, parent_profile_sha256)
    if any(not isinstance(value, str) or len(value) != 64
           or any(ch not in "0123456789abcdef" for ch in value) for value in hashes):
        raise ValueError("Cumulative control has an invalid SHA-256 anchor")
    if (hashlib.sha256(parent_manifest_bytes).hexdigest() != parent_manifest_sha256
            or hashlib.sha256(parent_profile_bytes).hexdigest() != parent_profile_sha256):
        raise ValueError("Original manifest or ledger bytes changed after prewrite")
    parent_manifest = json.loads(parent_manifest_bytes)
    parent_profile = json.loads(parent_profile_bytes)
    identity = lineage.get("identity") or {}
    budget = lineage.get("budget") or {}
    run_id = identity.get("run_id")
    if (not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id
            or not is_valid_permit(permit, run_id)
            or permit.migration_sha256 != migration_sha256
            or permit.authorization_sha256 != authorization_sha256
            or parent_manifest.get("run_id") != run_id
            or parent_manifest.get("scenario") != identity.get("scenario")
            or not isinstance(parent_manifest.get("config"), dict)
            or (parent_manifest.get("config") or {}).get("cumulative_recovery") is not None
            or parent_manifest.get("status") != "failed"
            or parent_profile.get("stopped") is not True
            or (parent_profile.get("stop_reason") or {}).get("reason") != "estimated_budget_limit"
            or parent_profile.get("source_drift")
            or parent_profile.get("model") != budget.get("model")
            or parent_profile.get("admitted_calls") != budget.get("admitted_calls")
            or parent_profile.get("settled_usage_calls") != budget.get("settled_usage_calls")
            or parent_profile.get("unsettled_reservation_calls") != 0
            or parent_profile.get("max_calls") != budget.get("max_calls")
            or _money(parent_profile.get("max_cny")) != _money(budget.get("max_cny"))
            or _money(parent_profile.get("reported_usage_estimate_cny"))
               != _money(budget.get("reported_usage_estimate_cny"))
            or migration.get("version") != "complete-cumulative-source-migration/v1"
            or migration.get("status") != "complete"
            or migration.get("complete_source_migration") is not True
            or migration.get("paid_resume_authorization") is not False
            or migration.get("historical_profile_sha256") != parent_profile_sha256
            or migration.get("parent_lineage_sha256") != lineage_sha256
            or authorization.get("policy") != AUTH_POLICY
            or authorization.get("user_authorization") is not True
            or authorization.get("migration_sha256") != migration_sha256
            or authorization.get("first_request_boundary_sha256") != boundary_sha256
            or authorization.get("parent_profile_sha256") != parent_profile_sha256
            or authorization.get("parent_trace_sha256") != budget.get("trace_sha256")
            or authorization.get("model") != budget.get("model")
            or authorization.get("allowed_first_step") != "world.semantic_review"
            or authorization.get("old_max_calls") != budget.get("max_calls")
            or _money(authorization.get("old_max_cny")) != _money(budget.get("max_cny"))
            or type(authorization.get("new_max_calls")) is not int
            or authorization["new_max_calls"] < budget["max_calls"]
            or _money(authorization.get("new_max_cny")) <= _money(budget.get("max_cny"))):
        raise ValueError("Cumulative child needs intact history and matching cap authority")
    if _money(parent_profile.get("budget_consumed_cny")) >= _money(authorization["new_max_cny"]):
        raise ValueError("Cumulative cap cannot admit the preserved historical liability")
    manifest, profile = deepcopy(parent_manifest), deepcopy(parent_profile)
    manifest.setdefault("config", {})["cumulative_recovery"] = {
        "version": VERSION,
        "lineage_sha256": lineage_sha256,
        "migration_sha256": migration_sha256,
        "evidence_sha256": evidence_sha256,
        "authorization_sha256": authorization_sha256,
        "first_request_boundary_sha256": boundary_sha256,
        "stage_inventory_sha256": stage_sha256,
        "source_epoch_sha256": source_epoch_sha256,
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_profile_sha256": parent_profile_sha256,
    }
    profile.setdefault("resume_history", []).append({
        "reason": "explicit_cumulative_source_migration",
        "parent_profile_sha256": parent_profile_sha256,
        "parent_manifest_sha256": parent_manifest_sha256,
        "migration_sha256": migration_sha256,
        "authorization_sha256": authorization_sha256,
        "previous_stop_reason": deepcopy(parent_profile["stop_reason"]),
        "admitted_calls": parent_profile["admitted_calls"],
        "settled_usage_calls": parent_profile["settled_usage_calls"],
        "unsettled_reservation_calls": parent_profile["unsettled_reservation_calls"],
        "budget_consumed_cny": parent_profile["budget_consumed_cny"],
        "reported_usage_estimate_cny": parent_profile["reported_usage_estimate_cny"],
    })
    authorized_cap = _money(authorization["new_max_cny"])
    new_cap = float(authorized_cap)
    if not math.isfinite(new_cap):
        raise ValueError("Cumulative cap exceeds the bounded runner's finite range")
    if Decimal(str(new_cap)) > authorized_cap:
        new_cap = math.nextafter(new_cap, -math.inf)
    if new_cap <= float(_money(budget["max_cny"])):
        raise ValueError("Cumulative cap increase is too small for the bounded runner")
    profile["max_calls"] = authorization["new_max_calls"]
    profile["max_cny"] = new_cap
    profile["stopped"] = False
    profile.pop("stop_reason", None)
    profile["execution_status"] = "pending_post_corpus_world_review"
    return manifest, profile
