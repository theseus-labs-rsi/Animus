"""Bounded experiment wrapper around the original factory stages."""
import argparse
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REUSE_STAGES = {
    "whitepaper": ("input", "whitepaper"),
    "world": ("input", "whitepaper", "world"),
    "questions": ("input", "whitepaper", "world", "orders", "well_posed", "questions"),
    "corpus": ("input", "whitepaper", "world", "orders", "well_posed", "questions", "corpus"),
    "disclosure": ("input", "whitepaper", "world", "orders", "well_posed", "questions",
                   "disclosure", "corpus"),
}
REUSE_ARTIFACTS = {"input": "00_input.json", "whitepaper": "01_whitepaper.json",
    "world": "02_world.json", "orders": "03_orders.json",
    "well_posed": "03_well_posed_report.json", "questions": "04_questions.json",
    "corpus": "05_corpus.json", "disclosure": "02_disclosure_ready.json"}
REUSE_NEXT = {"whitepaper": "world", "world": "orders", "questions": "disclosure",
              "corpus": "disclosure", "disclosure": "grounding"}
_TRACER_STEP = ContextVar("original_bc_smoke_tracer_step", default=None)
_JSON_REQUEST_ACTIVE = ContextVar("original_bc_smoke_json_request_active", default=False)
_CLEAN_HEADER_RETRY_ACTIVE = ContextVar("original_bc_smoke_clean_header_retry_active", default=False)
_CALL_USAGE = ContextVar("original_bc_smoke_call_usage", default=None)


def _claim_cumulative_execution(directory: Path, permit) -> None:
    """Atomically claim one published child before any ledger or stage write.

    A second process must reconcile the first execution rather than race its
    cumulative ledger. The claim remains as evidence after a crash or stop.
    """
    path = directory / "00_cumulative_execution_claim.json"
    body = (json.dumps({"version": "cumulative-execution-claim/v1",
        "run_id": permit.run_id,
        "migration_sha256": permit.migration_sha256,
        "authorization_sha256": permit.authorization_sha256,
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls_at_claim": 0}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive claim is retained if recording failed: another process
        # cannot assume that execution never began without reconciliation.
        raise


def _validate_plan_retry_policy_option(selected, permit):
    """A flag selects an evidence-bound migration; it cannot alter policy alone."""
    actual = permit.get("plan_retry_policy") if isinstance(permit, dict) else None
    if selected != actual or (selected is not None and selected != "separate-protocol-v2"):
        raise ValueError("Plan retry policy flag differs from the explicit checkpoint receipt")
    if selected is not None:
        policy = permit.get("retry_policy")
        encoded = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if (not isinstance(policy, dict) or hashlib.sha256(encoded).hexdigest() != permit.get("retry_policy_sha256")
                or permit.get("source_migration", {}).get("retry_policy_sha256") != permit.get("retry_policy_sha256")):
            raise ValueError("Plan retry policy is not bound to the source migration")


def _validate_plan_escalation_option(selected, permit):
    actual = permit.get("plan_escalation") if isinstance(permit, dict) else None
    if selected != actual or (selected is not None and selected not in {"original-business-slot-v1", "approved-business-checkpoint-v2"}):
        raise ValueError("Plan escalation differs from its explicit checkpoint receipt")
    if selected is not None:
        context = permit.get("escalation")
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if (not isinstance(context, dict) or hashlib.sha256(encoded).hexdigest() != permit.get("escalation_sha256")
                or permit.get("source_migration", {}).get("escalation_sha256") != permit.get("escalation_sha256")
                or permit.get("first_new_step") != ("council.instance_plan_business_repair" if selected == "approved-business-checkpoint-v2" else "council.blueprint_feasibility")):
            raise ValueError("Business escalation is not source-bound to its required first step")
        if selected == "approved-business-checkpoint-v2":
            business = permit.get("business_resume_checkpoint")
            encoded = json.dumps(business, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            if (not isinstance(business, dict) or digest != permit.get("business_resume_checkpoint_sha256")
                    or digest != permit.get("source_migration", {}).get("business_resume_checkpoint_sha256")
                    or business.get("completed_business_attempts") != 1 or business.get("max_business_attempts") != 3
                    or business.get("max_additional_physical_requests") != 36 or business.get("stage_start_admitted_calls") != 25
                    or permit.get("prior_physical_calls") != 25 or permit.get("max_calls") != 5000 or permit.get("max_cny") != 100):
                raise ValueError("Approved business checkpoint allowance is not source-bound")


def _install_business_checkpoint_stage(ledger, permit):
    """Persist the approved 36-physical-request stage once, never rebase it."""
    if not permit or not permit.get("business_resume_checkpoint"):
        return
    _validate_plan_escalation_option("approved-business-checkpoint-v2", permit)
    policy = permit["business_resume_checkpoint"]
    expected = {"version": "business-checkpoint-physical-stage/v1", "receipt_sha256": permit["receipt_sha256"],
        "business_resume_checkpoint_sha256": permit["business_resume_checkpoint_sha256"],
        "start_admitted_calls": policy["stage_start_admitted_calls"],
        "max_additional_physical_requests": policy["max_additional_physical_requests"],
        "max_admitted_calls": min(ledger["max_calls"], policy["stage_start_admitted_calls"] + policy["max_additional_physical_requests"])}
    old = ledger.get("business_checkpoint_stage")
    if old is not None:
        if any(old.get(k) != v for k, v in expected.items()):
            raise ValueError("Business checkpoint stage cannot be reset or rebound")
        return
    if ledger["admitted_calls"] != expected["start_admitted_calls"]:
        raise ValueError("Business checkpoint stage must start at its original 25-call boundary")
    ledger["business_checkpoint_stage"] = {**expected, "status": "active"}


def _business_checkpoint_admission_reason(directory, ledger, step):
    """Called under the physical admission lock, including JSON parser retries."""
    stage = ledger.get("business_checkpoint_stage")
    if stage is None:
        return None
    if (stage.get("start_admitted_calls") != 25 or stage.get("max_additional_physical_requests") != 36
            or stage.get("max_admitted_calls") != min(ledger["max_calls"], 61)
            or ledger["admitted_calls"] < 25):
        return "business_checkpoint_stage_binding"
    if stage.get("status") == "completed":
        return None
    if stage.get("status") != "active":
        return "business_checkpoint_stage_binding"
    if step not in {"council.instance_plan_business_repair", "council.blueprint_feasibility"}:
        try:
            accepted = json.loads((Path(directory) / "01_initial_plan_recovery.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "business_checkpoint_unaccepted_step"
        if (accepted.get("status") != "accepted" or accepted.get("request_sha256") != stage["receipt_sha256"]
                or accepted.get("business_resume_checkpoint_sha256") != stage["business_resume_checkpoint_sha256"]
                or accepted.get("final_business_review", {}).get("decision") != "accept"):
            return "business_checkpoint_unaccepted_step"
        stage.update(status="completed", completed_admitted_calls=ledger["admitted_calls"],
            physical_requests_used=ledger["admitted_calls"] - stage["start_admitted_calls"])
        return None
    if ledger["admitted_calls"] >= stage["max_admitted_calls"]:
        return "business_checkpoint_stage_call_limit"
    return None


def _claim_initial_plan_execution(directory: Path, permit) -> None:
    """Win the execution entry before changing the cumulative profile or sources."""
    directory = Path(directory)
    if (not isinstance(permit, dict) or directory.name != permit.get("run_id")
            or not directory.samefile(Path(permit["directory"]))):
        raise ValueError("Initial-plan execution claim targets a different run")
    for name, bound in permit["evidence"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != bound["sha256"]:
            raise ValueError("Initial-plan evidence changed before execution claim: " + name)
    body = {"version": "initial-plan-execution-claim/v1", "run_id": permit["run_id"],
        "request_sha256": permit["receipt_sha256"],
        "original_profile_sha256": permit["evidence"]["experiment_profile.json"]["sha256"],
        "claimed_utc": datetime.now(timezone.utc).isoformat(), "provider_calls_at_claim": 0}
    if permit.get("epoch"):
        body["epoch"] = permit["epoch"]
        body["parent_request_sha256"] = permit.get("parent_request_sha256")
    if permit.get("retry_policy"):
        body["retry_policy_sha256"] = permit["retry_policy_sha256"]
    if permit.get("plan_escalation"):
        body["escalation_sha256"] = permit["escalation_sha256"]
    if permit.get("business_resume_checkpoint"):
        body["business_resume_checkpoint_sha256"] = permit["business_resume_checkpoint_sha256"]
    name = permit.get("entry_claim", "00_initial_plan_execution_claim.json")
    if Path(name).name != name:
        raise ValueError("Unsafe initial-plan execution claim path")
    path = directory / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    # Retain the exclusive claim even when its write fails. An incomplete entry
    # needs reconciliation; a second process must never reset its ledger.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _finish_full_material_acceptance(directory, ledger, *, enabled, model, max_tokens):
    """Run the final reader under the same physical call gate, after generation."""
    directory = Path(directory)
    state_path = directory / "11_production.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    ledger["experiment_target_passed"] = False
    summary = {"status": "not_run", "requested": bool(enabled), "current": False,
        "passed": False, "production_status": state.get("status"),
        "artifact": "12_full_material_acceptance.json"}
    ledger["full_material_acceptance"] = summary
    if not enabled:
        summary["reason"] = "not_requested"
        return
    if state.get("status") != "generation_ready_awaiting_selection":
        summary["reason"] = "generation_not_ready"
        return
    from pipeline.cumulative_entry_guard import require_bounded_run
    require_bounded_run(directory.name)
    from pipeline.run import Run
    from pipeline.full_material_acceptance import run_full_material_acceptance, snapshot, target_checks
    from pipeline.production import outcome
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    original_config = deepcopy(manifest["config"])
    run = Run(manifest["scenario"], manifest["run_id"])
    if run.manifest["config"] != original_config or not Path(run.dir).samefile(directory):
        raise ValueError("Final reader changed the frozen run configuration or destination")
    summary["status"] = "running"
    try:
        with run.stage_write_lock("full_material_acceptance", lock_name=".production.lock"):
            contract = target_checks(run)
            ledger["target_checks"] = deepcopy(contract)
            summary["target_checks_passed"] = contract.get("passed") is True
            if contract.get("passed") is not True:
                summary.update(status="not_run", reason="experiment_target_contract_not_met")
                return
            run_full_material_acceptance(run, model=model, max_tokens=max_tokens)
            receipt = snapshot(run)
            current_outcome = outcome(run)
            current_state = run.read("11_production.json")
            summary.update({key: deepcopy(receipt[key]) for key in
                ("status", "current", "passed", "identity", "released_count", "document_count", "issues")
                if key in receipt})
            summary["production_status"] = current_state.get("status")
            summary["production_outcome_passed"] = current_outcome.get("passed") is True
            ledger["experiment_target_passed"] = bool(
                current_state.get("status") == "generation_ready_awaiting_selection"
                and current_outcome.get("passed") is True
                and contract.get("passed") is True
                and receipt.get("current") is True and receipt.get("passed") is True)
    except BaseException as error:
        summary.update(status="execution_failed", passed=False, current=False,
                       error_type=type(error).__name__)
        ledger["experiment_target_passed"] = False
        raise


def _cumulative_review_exit_status(directory: Path) -> str:
    """Confirm the new gate exists before declaring a bounded invocation done.

    This checks the saved control receipt only. The review transaction itself
    verifies semantic replay and physical calls before writing that receipt.
    """
    names = ("12_post_corpus_world_review.json",
             "12_post_corpus_world_review_physical.json",
             "12_post_corpus_world_review_gate.json")
    paths = [directory / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError("Cumulative invocation ended without a post-corpus world-review gate")
    review, physical, gate = [json.loads(path.read_text(encoding="utf-8"))
                              for path in paths]
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths[:2]]
    statuses = ("responsibility_repair_required",
                "original_layout_allowance_exhausted", "unresolved_corpus_failure")
    if (gate.get("version") != "post-corpus-world-review/v1" or
            gate.get("status") not in statuses or
            gate.get("new_review_sha256") != digests[0] or
            gate.get("new_physical_receipt_sha256") != digests[1] or
            physical.get("version") != "post-corpus-world-review-physical/v1" or
            type(physical.get("new_physical_requests")) is not int or
            physical["new_physical_requests"] < 1 or
            gate.get("new_physical_requests") != physical["new_physical_requests"] or
            review.get("status") not in ("passed", "failed", "unresolved") or
            gate.get("truth_changed") is not False or
            gate.get("author_repair_started") is not False or
            gate.get("production_ready") is not False or
            gate.get("provider_calls_by_gate") != 0):
        raise ValueError("Cumulative world-review gate is missing or inconsistent")
    return gate["status"]


def _reuse_layout(whitepaper, through):
    """Use the current production graph for V2; retain historical prefix support."""
    contract = whitepaper.get("generation_contract") or {}
    material_first = (contract.get("version") == "generation-first/v2"
                      and contract.get("material_first") is True)
    if not material_first:
        return set(REUSE_STAGES[through]), REUSE_NEXT[through], False
    from pipeline.factory import STAGES
    names = [stage.name for stage in STAGES]
    if names.index("corpus") > names.index("questions"):
        raise ValueError("V2 recovery requires the production material-first stage graph")
    end = names.index(through)
    return set(names[:end + 1]), names[end + 1], True


def recoverable_unit_failure(step, exc):
    """Let bounded original units handle transient failures; never relax money/auth guards."""
    import config
    steps = {"phrase", "phrase.review", "orders.process_propose", "world.semantic_review",
             "world.disclosure_plan", "corpus.review", "render.plan", "render.signal", "render.filler",
             "semantic_review.blind_read", "semantic_review.adjudicate",
             "agent_editing.independent_reference_audit"}
    kind = config.chat_error_kind(exc)
    if step == "render.filler":
        from pipeline.filler_failure import is_content_rejection
        if is_content_rejection(exc):
            return True
    return step in steps and (kind in {"json_syntax", "output_truncated", "empty_completion",
        "http_connect_timeout", "http_pool_timeout", "http_connect_error", "http_read_error",
        "http_write_error", "http_protocol_error", "http_read_timeout", "http_write_timeout"}
        or (kind == "total_deadline" and getattr(exc, "local_cleanup_confirmed", False))
        or (kind == "http_status_429" and config.is_retryable_chat_error(getattr(exc, "__cause__", None) or exc))
        or kind in {"http_status_500", "http_status_502", "http_status_503", "http_status_504"})


def transient_resume_evidence(profile, trace_path):
    """Verify an explicitly requested transport recovery; retain all liabilities.

    Older wrappers did not record a stop cause. Their full physical trace must
    account for every admitted call and show only recoverable transport errors.
    A model, money, authentication, source-drift or unknown stop stays closed.
    """
    if not profile.get("stopped"):
        return None
    if (profile.get("source_drift") or profile["admitted_calls"] >= profile["max_calls"]
            or profile["budget_consumed_cny"] >= profile["max_cny"]):
        raise ValueError("A drifted or exhausted run cannot reopen admission")
    first = profile.get("stop_reason")
    if first and first.get("reason") != "provider_failure":
        raise ValueError("Only a confirmed provider transport stop can resume")
    if not first and not (profile.get("error_type") == "MaterialAuthoringStopped"
            and "Original experiment model/call/estimated budget limit" in profile.get("error", "")):
        raise ValueError("Legacy stop has no recognized engineering failure record")
    raw = Path(trace_path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    requests = [row for row in rows if row.get("event") == "request"]
    errors = [row for row in rows if row.get("event") == "call_error"]
    ids = {row.get("call_id") for row in requests}
    kinds = {"http_connect_error", "http_protocol_error", "http_connect_timeout", "http_pool_timeout",
             "http_read_error", "http_write_error", "http_status_500", "http_status_502",
             "http_status_503", "http_status_504"}
    if (len(requests) != profile["admitted_calls"] or len(ids) != len(requests) or None in ids
            or any(row.get("model") != profile["model"] for row in requests) or not errors):
        raise ValueError("Transport recovery requires complete matching physical-call evidence")
    for row in errors:
        error = RuntimeError("recorded transport failure")
        error.kind = row.get("kind")
        joint_revision = row.get("step") == "council.joint_design_revision"
        clean_header = (joint_revision and row.get("kind") == "total_deadline"
                        and row.get("phase") == "awaiting_http_headers"
                        and row.get("local_cleanup_confirmed") is True
                        and row.get("retryable") is False)
        ordinary = (row.get("kind") in kinds and row.get("retryable") is True
                    and row.get("local_cleanup_confirmed") is True
                    and (joint_revision or recoverable_unit_failure(row.get("step"), error)))
        if row.get("call_id") not in ids or not (clean_header or ordinary):
            raise ValueError("A nonrecoverable or unclassified provider error keeps admission closed")
    last = errors[-1]
    if first and (first.get("kind") not in kinds | {"total_deadline"}
                  or first.get("call_id") not in (None, last.get("call_id"))
                  or first.get("step") not in (None, last.get("step"))):
        raise ValueError("First stop cause is not a recoverable transport error")
    result = {"reason": "explicit_transient_transport_resume", "previous_stop_reason": first,
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "error_call_ids": [row["call_id"] for row in errors],
            "prior_admitted_calls": profile["admitted_calls"],
            "prior_budget_consumed_cny": profile["budget_consumed_cny"],
            "unknown_reservations_preserved": profile["unsettled_reservation_calls"]}
    if last.get("step") == "council.joint_design_revision":
        audit_path = Path(trace_path).with_name("01_joint_design_audit.json")
        audit_raw = audit_path.read_bytes()
        audit = json.loads(audit_raw)
        call = (audit.get("calls") or [])[-1]
        metadata = (call.get("response") or {}).get("__error_metadata__") or {}
        request = requests[-1]
        from pipeline.capability_contract import digest
        try:
            request_hash = digest(json.loads(request["messages"][-1]["content"]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Joint design failed request payload is missing") from exc
        if (audit.get("status") != "designing" or call.get("status") != "provider_error"
                or call.get("step") != last["step"] or request.get("call_id") != last.get("call_id")
                or request_hash != call.get("request_hash")
                or metadata.get("call_id") != last.get("call_id")
                or metadata.get("kind") != last.get("kind")
                or not audit.get("actions") or audit["actions"][-1].get("status") != "rejected"):
            raise ValueError("Joint design checkpoint is not bound to the failed physical request")
        result.update(run_id=audit_path.parent.name,
                      joint_design_checkpoint_sha256=hashlib.sha256(audit_raw).hexdigest(),
                      failed_request_hash=request_hash)
    return result


def budget_cap_resume_evidence(profile, trace_path, authorization_path):
    """Apply an explicit cumulative budget increase to a recorded budget stop.

    The authorization binds the original ledger and physical trace. No call,
    usage, unknown reservation, model or attempt allowance is reset.
    """
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    ceiling = authorization.get("new_max_cny")
    if (authorization.get("policy") != "explicit-cumulative-budget-increase/v1"
            or not authorization.get("user_authorization")
            or type(ceiling) not in (int, float) or not math.isfinite(ceiling)
            or authorization.get("model") != profile["model"]
            or authorization.get("max_calls") != profile["max_calls"]):
        raise ValueError("Budget recovery requires explicit matching cumulative authorization")
    if not profile.get("stopped"):
        if profile["max_cny"] != ceiling:
            raise ValueError("A budget change requires a recorded budget stop")
        return None
    raw = Path(trace_path).read_bytes()
    canonical = json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    requests = [row for row in rows if row.get("event") == "request"]
    first = profile.get("stop_reason") or {}
    if (first.get("reason") != "estimated_budget_limit" or profile.get("source_drift")
            or profile["admitted_calls"] >= profile["max_calls"]
            or ceiling <= profile["max_cny"] or ceiling <= profile["budget_consumed_cny"]
            or authorization.get("previous_max_cny") != profile["max_cny"]
            or authorization.get("profile_sha256") != hashlib.sha256(canonical).hexdigest()
            or authorization.get("trace_sha256") != hashlib.sha256(raw).hexdigest()
            or len(requests) != profile["admitted_calls"]
            or len({row.get("call_id") for row in requests}) != len(requests)
            or any(not row.get("call_id") or row.get("model") != profile["model"] for row in requests)):
        raise ValueError("Budget recovery requires an intact stopped ledger and complete original trace")
    return {"reason": "explicit_cumulative_budget_resume", "previous_stop_reason": first,
            "authorization_sha256": hashlib.sha256(Path(authorization_path).read_bytes()).hexdigest(),
            "trace_sha256": hashlib.sha256(raw).hexdigest(), "prior_max_cny": profile["max_cny"],
            "model": profile["model"], "max_calls": profile["max_calls"],
            "new_max_cny": ceiling, "prior_admitted_calls": profile["admitted_calls"],
            "prior_budget_consumed_cny": profile["budget_consumed_cny"],
            "unknown_reservations_preserved": profile["unsettled_reservation_calls"]}


def output_truncation_resume_evidence(profile, trace_path, ceiling, partition=None):
    """Reopen a completed, billed truncation with an explicitly larger ceiling."""
    if not profile.get("stopped"):
        if ceiling != profile["max_output_tokens"]:
            raise ValueError("An output ceiling change requires a recorded truncation stop")
        return None
    first = profile.get("stop_reason") or {}
    if (profile.get("source_drift") or profile["admitted_calls"] >= profile["max_calls"]
            or profile["budget_consumed_cny"] >= profile["max_cny"]
            or first.get("reason") != "json_execution_failure" or first.get("kind") != "output_truncated"
            or type(ceiling) is not int or not 1 <= ceiling <= 128000
            or (ceiling <= profile["max_output_tokens"] and partition is None)):
        raise ValueError("Output recovery requires a nonexhausted truncation stop and a larger explicit ceiling")
    raw = Path(trace_path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    requests = [r for r in rows if r.get("event") == "request"]
    ids = {r.get("call_id") for r in requests}
    call_id = first.get("call_id")
    errors = [r for r in rows if r.get("event") == "call_error" and r.get("call_id") == call_id]
    replies = [r for r in rows if r.get("event") == "response" and r.get("call_id") == call_id]
    if (len(requests) != profile["admitted_calls"] or len(ids) != len(requests) or None in ids
            or any(r.get("model") != profile["model"] for r in requests)
            or call_id not in ids or len(errors) != 1 or len(replies) != 1):
        raise ValueError("Output recovery requires complete matching physical request and response evidence")
    error, response = errors[0], replies[0].get("response") or {}
    if (error.get("kind") != "output_truncated" or error.get("step") != first.get("step")
            or error.get("usage_status") != "reported"
            or not response.get("choices") or response["choices"][0].get("finish_reason") != "length"
            or usage_cost(response.get("usage"), 1, 1) is None
            or type(error.get("actual_output_cap")) is not int
            or (ceiling <= error["actual_output_cap"] and partition is None)):
        raise ValueError("Only a closed, billed truncation can reopen with a higher actual output cap")
    if partition is not None:
        policy=json.loads(Path(partition).read_text(encoding="utf-8"))
        request=next(r for r in requests if r["call_id"]==call_id)
        canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        if (ceiling != profile["max_output_tokens"] or ceiling != error["actual_output_cap"]
                or policy.get("parent_call_id") != call_id or policy.get("parent_step") != first["step"]
                or policy.get("parent_messages_sha256") != canonical(request.get("messages"))
                or policy.get("policy") != "resource-allocation-then-unit-edits/v1"
                or policy.get("first_new_steps") != ["council.instance_resources"]
                or policy.get("max_semantic_attempts") != 3
                or policy.get("used_semantic_attempts") != 1
                or policy.get("remaining_semantic_attempts") != 2):
            raise ValueError("Task partition recovery requires a bound, unchanged-budget decomposition policy")
        proposals=[r.get("response",{}).get("choices",[{}])[0].get("content","") for r in rows
                   if r.get("event")=="response" and r.get("step")=="council.instance_plan"]
        if len(proposals)!=1 or canonical(json.loads(proposals[0])) != policy.get("original_proposal_sha256"):
            raise ValueError("Task partition must retain the actual complete original proposal")
    return {"reason":"explicit_task_partition_resume" if partition else "explicit_output_ceiling_resume", "previous_stop_reason":first,
        "trace_sha256":hashlib.sha256(raw).hexdigest(), "error_call_ids":[call_id],
        "prior_admitted_calls":profile["admitted_calls"], "prior_budget_consumed_cny":profile["budget_consumed_cny"],
        "unknown_reservations_preserved":profile["unsettled_reservation_calls"],
        "prior_output_ceiling":profile["max_output_tokens"], "new_output_ceiling":ceiling,
        "task_partition_sha256":hashlib.sha256(Path(partition).read_bytes()).hexdigest() if partition else None}


def json_format_resume_evidence(profile, trace_path):
    """Explicitly reopen a closed, billed syntax failure after interface repair."""
    if not profile.get("stopped"):
        return None
    first = profile.get("stop_reason") or {}
    if (profile.get("source_drift") or profile["admitted_calls"] >= profile["max_calls"]
            or profile["budget_consumed_cny"] >= profile["max_cny"]
            or first.get("reason") != "json_execution_failure" or first.get("kind") != "json_syntax"):
        raise ValueError("Format recovery requires a nonexhausted recorded JSON syntax stop")
    raw = Path(trace_path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    requests = [r for r in rows if r.get("event") == "request"]
    ids = {r.get("call_id") for r in requests}
    replies = {r.get("call_id"):r for r in rows if r.get("event") == "response"}
    failures = [r for r in rows if r.get("event") == "json_error"
        and r.get("step") == first.get("step") and r.get("error_type") == "JSONDecodeError"]
    if (len(requests) != profile["admitted_calls"] or len(ids) != len(requests) or None in ids
            or any(r.get("model") != profile["model"] for r in requests) or not failures):
        raise ValueError("Format recovery requires complete matching physical-call evidence")
    failure = failures[-1]
    operation = failure.get("operation_id")
    failed_requests = [r for r in requests if r.get("operation_id") == operation]
    if (not operation or not failed_requests or failure.get("step") != first.get("step")
            or failure.get("kind") != "json_syntax" or failure.get("error_type") != "JSONDecodeError"
            or not isinstance(failure.get("raw_output"), str)):
        raise ValueError("Format recovery requires a specific failed JSON operation")
    try:
        json.loads(failure["raw_output"])
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError("Recorded format failure is valid JSON")
    for request in failed_requests:
        response = (replies.get(request["call_id"]) or {}).get("response") or {}
        if (request.get("step") != first.get("step") or not response.get("choices")
                or response["choices"][0].get("finish_reason") != "stop"
                or usage_cost(response.get("usage"), 1, 1) is None):
            raise ValueError("Only fully returned and billed format attempts can reopen")
    last_reply = replies[failed_requests[-1]["call_id"]]
    if (last_reply.get("operation_id") != operation
            or last_reply["response"]["choices"][0].get("content", "").strip() != failure["raw_output"].strip()):
        raise ValueError("Format error must bind the actual final provider body")
    return {"reason":"explicit_json_format_resume", "previous_stop_reason":first,
        "trace_sha256":hashlib.sha256(raw).hexdigest(), "failed_operation_id":operation,
        "error_call_ids":[r["call_id"] for r in failed_requests],
        "prior_admitted_calls":profile["admitted_calls"], "prior_budget_consumed_cny":profile["budget_consumed_cny"],
        "unknown_reservations_preserved":profile["unsettled_reservation_calls"]}


def filler_content_resume_evidence(profile, trace_path):
    """Reopen only a fully evidenced background-content rejection, retaining its cost."""
    from pipeline.filler_failure import is_content_rejection
    first = profile.get("stop_reason") or {}
    if (not profile.get("stopped") or profile.get("source_drift")
            or profile["admitted_calls"] >= profile["max_calls"]
            or profile["budget_consumed_cny"] >= profile["max_cny"]
            or first.get("reason") != "provider_failure" or first.get("step") != "render.filler"
            or first.get("kind") != "http_status_400"):
        raise ValueError("Background-content recovery requires a nonexhausted matching provider stop")
    raw = Path(trace_path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    requests = [row for row in rows if row.get("event") == "request"]
    errors = [row for row in rows if row.get("event") == "call_error"]
    ids = {row.get("call_id") for row in requests}
    if (len(requests) != profile["admitted_calls"] or len(ids) != len(requests) or None in ids
            or any(row.get("model") != profile["model"] for row in requests)
            or not errors or first.get("call_id") not in {row.get("call_id") for row in errors}
            or any(row.get("call_id") not in ids or row.get("step") != "render.filler"
                   or row.get("local_cleanup_confirmed") is not True or not is_content_rejection(row)
                   for row in errors)):
        raise ValueError("Background-content recovery requires complete matching physical-call evidence")
    return {"reason": "explicit_background_content_resume", "previous_stop_reason": first,
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "error_call_ids": [row["call_id"] for row in errors],
            "prior_admitted_calls": profile["admitted_calls"],
            "prior_budget_consumed_cny": profile["budget_consumed_cny"],
            "unknown_reservations_preserved": profile["unsettled_reservation_calls"]}


def quantity_arguments(args):
    """Forward quantity controls to the existing factory/closed_loop stages."""
    if (args.source_run and (not args.reuse_world_checkpoint
                            or (args.min_questions is None and args.question_budget is None))) or args.resume_existing:
        # A derived recovery run keeps the frozen source quantities.  Supplying
        # the wrapper's tiny smoke defaults would silently turn a large saved
        # world into a four-question experiment.
        return []
    if args.min_questions is None:
        if getattr(args, "delivery_target", None) and args.question_budget is None:
            result = ["--max-world-entities", str(args.max_world_entities)]
            if args.time_span_weeks is not None:
                result += ["--time-span-weeks", str(args.time_span_weeks)]
            return result
        result = ["--question-budget", str(args.question_budget if args.question_budget is not None else 4),
                  "--max-world-entities", str(args.max_world_entities)]
        if args.time_span_weeks is not None:
            result += ["--time-span-weeks", str(args.time_span_weeks)]
        return result
    result = ["--min-questions", str(args.min_questions), "--max-rounds", str(args.max_rounds),
              "--max-world-entities", str(args.max_world_entities)]
    if args.total_only:
        result.append("--total-only")
    if args.time_span_weeks is not None:
        result += ["--time-span-weeks", str(args.time_span_weeks)]
    for value in args.per_line or []:
        result += ["--per-line", value]
    return result


def _inherit_resume_settings(args, profile, explicit_options):
    """A resume keeps the admitted calls, liabilities and original execution caps."""
    required = ("model", "max_calls", "max_cny", "admitted_calls", "budget_consumed_cny",
                "reported_usage_estimate_cny", "reserved_upper_estimate_cny",
                "settled_usage_calls", "unsettled_reservation_calls", "stopped", "transport")
    if any(key not in profile for key in required):
        raise ValueError("Existing-run resume requires the complete original budget ledger")
    for key in required[1:9]:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid saved budget value: {key}")
    if type(profile["stopped"]) is not bool:
        raise ValueError("Invalid saved budget stop flag")
    for key in ("max_calls", "admitted_calls", "settled_usage_calls", "unsettled_reservation_calls"):
        if type(profile[key]) is not int:
            raise ValueError(f"Invalid saved call counter: {key}")
    frozen = ("model", "max_calls", "max_cny", "json_attempts", "min_output_tokens",
              "max_output_tokens", "call_timeout_seconds", "llm_concurrency", "semantic_workers",
              "disclosure_format_attempts", "settle_reported_usage")
    for key in frozen:
        if key not in profile:
            continue
        option = "--" + key.replace("_", "-")
        if option in explicit_options and getattr(args, key) != profile[key]:
            raise ValueError(f"Resume must preserve frozen {option}={profile[key]!r}")
        setattr(args, key, deepcopy(profile[key]))


def usage_cost(usage, input_price, output_price):
    """Missing/invalid provider usage cannot release a conservative reservation."""
    if not isinstance(usage, dict):
        return None
    counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if any(type(n) is not int or n < 0 for n in counts):
        return None
    return (counts[0] * input_price + counts[1] * output_price) / 1e6


def implementation_hashes():
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (ROOT / "pipeline", ROOT / "eval") for path in folder.rglob("*.py")}
    hashes.update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ("config.py", "execution_control.py", "llm_transport.py",
                                "llm_trace.py", "tools/diversity_metrics.py",
                                "tools/run_original_bc_smoke.py")})
    return hashes


def verify_resume_source(directory, current_hashes):
    """Reject an implicit implementation change before touching a cumulative run.

    The end-of-previous-execution map is the source epoch that produced the
    saved ledger. A source migration needs its own explicit, audited entrypoint;
    a plain resume must not silently replace that map with today's source.
    """
    if not (directory / "experiment_source_hashes.json").is_file():
        raise ValueError("Existing-run resume requires its original source snapshot")
    path = directory / "experiment_source_hashes_at_end.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Existing-run resume requires its previous source snapshot") from exc
    if (not isinstance(previous, dict) or not previous
            or any(not isinstance(name, str) or not isinstance(digest, str)
                   or len(digest) != 64 for name, digest in previous.items())):
        raise ValueError("Existing-run source snapshot is malformed")
    changed = sorted(name for name in set(previous) | set(current_hashes)
                     if previous.get(name) != current_hashes.get(name))
    if changed:
        raise ValueError(f"Existing-run source epoch differs in {len(changed)} files; "
                         "an explicit source migration is required")
    return previous


def reuse_original_stages(source, directory, through, *, reuse_world_checkpoint=False, model=None,
                          repair_disclosure=False, reuse_review_checkpoints=False):
    """Freeze completed original inputs, including seed identity, before a comparison."""
    from pipeline.run import _io_path, _path_label
    source, directory = _io_path(source), _io_path(directory)
    from pipeline.seed_run import validate_seed_identity
    from pipeline import world_semantics
    previous = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_status = previous.get("status")
    source_current_stage = previous.get("current_stage")
    whitepaper = json.loads((source / "01_whitepaper.json").read_text(encoding="utf-8"))
    stages, resume_from, material_first = _reuse_layout(whitepaper, through)
    stale_downstream_run = (repair_disclosure and previous.get("status") == "running"
                            and previous.get("current_stage") in
                            {"disclosure", "grounding", "quality", ""})
    source_stage_states = {name: deepcopy(previous.get("stages", {}).get(name, {})) for name in stages}
    def reusable_stage(name):
        state = source_stage_states[name]
        if state.get("done"):
            return True
        # A later closed-loop world attempt can fail after the first complete
        # world has already produced orders, questions and corpus.  The
        # derived recovery validates that exact saved world/review before use.
        return (repair_disclosure and name == "world" and through in {"questions", "corpus", "disclosure"}
                and (source / "02_world.json").is_file()
                and previous.get("stages", {}).get("questions", {}).get("done") is True)
    if ((previous.get("status") == "running" and not stale_downstream_run) or any(
            not reusable_stage(name) for name in stages)):
        raise ValueError("Source must be stopped with all selected original stages completed")
    view = SimpleNamespace(manifest=previous, dir=source,
        has=lambda name: (source / name).is_file(),
        read=lambda name: json.loads((source / name).read_text(encoding="utf-8")))
    pack = validate_seed_identity(view, view.read("01_whitepaper.json"))
    if pack is None and previous.get("scenario") != "bc_small_office":
        raise ValueError("Only the bounded office scenario or a verified frozen seed is supported")
    names = {previous["stages"][name].get("artifact", REUSE_ARTIFACTS[name])
             for name in stages} | {"00_about.json"}
    if (previous.get("config", {}).get("production_control") or {}).get("version") == "supply-driven/v6":
        # The joint-design receipt is part of the whitepaper's executable
        # contract, and the world stage reads it again after a checkpoint reuse.
        names |= {"01_joint_design_audit.json", "01_joint_design_receipt.json",
                    "01_supply_plan.json"}
        if whitepaper.get("business_instance_plan"):
            names.add("01_instance_plan.json")
        if "world" in stages:
            names.add("02_source_supply_gate.json")
            names |= {name for name in ("02_instance_fulfillment.json",
                "02_publication_obligations.json", "02_world_review_attempts.json",
                "02_disclosure_plan_attempts.json") if (source / name).is_file()}
    if pack is not None:
        names |= {"00_seed_pack.json", "01_seed_audit.json"}
        if pack["schema_version"] == 2:
            from pipeline.seed_run import AUDIT_ARTIFACT, GENERATION_ARTIFACT
            names |= {AUDIT_ARTIFACT, GENERATION_ARTIFACT}
        if "world" in stages:
            names.add("02_seed_audit.json")
    checkpoint_name = None
    if reuse_world_checkpoint:
        if through != "whitepaper" or not model:
            raise ValueError("World checkpoint reuse requires whitepaper reuse and an explicit model")
        from pipeline.world_agent import _binding, _digest, supply_shortfall_checkpoint
        wp = view.read("01_whitepaper.json")
        from pipeline import joint_design
        allow_shortfall = (joint_design.exploratory_after_limit(view, wp)
                          and view.read("01_joint_design_audit.json").get("used_revisions", 0)
                          >= joint_design.MAX_REVISIONS)
        checkpoints = [(path.name, view.read(path.name)) for path in source.glob("02_world_agent_*.json")]
        checkpoints = [(name, state) for name, state in checkpoints
                       if state.get("status") == "completed"
                       or (allow_shortfall and supply_shortfall_checkpoint(state))]
        if not checkpoints:
            raise ValueError("No completed or design-limit supply checkpoint to reuse")
        for checkpoint_name, state in checkpoints:
            expected = {**_binding(wp, None), "model": model,
                        "wp_hash": (state.get("binding") or {}).get("wp_hash")}
            if (state.get("binding") != expected or not expected["wp_hash"]
                    or state.get("checkpoint_hash") != _digest({k:v for k,v in state.items() if k != "checkpoint_hash"})):
                raise ValueError("Completed world checkpoint input/model/source binding mismatch")
            names.add(checkpoint_name)
        # A failed quantity loop rolls the published whitepaper back. It will
        # derive the same scaled contract again; generate_world checks its full
        # binding before using the matching checkpoint. Never relabel its hash.
        # Only construction progress is reused. The normal world stage recompiles
        # the units, checks seed requirements and runs disclosure/business review.
    if "questions" in stages:
        names.add("04_wording_report.json")
    if "orders" in stages:
        # Reused orders retain their exact capacity decision and warnings.
        # Omitting these receipts can silently lose an exploratory restriction.
        names |= {name for name in ("03_capacity_gate.json", "03_orders_warning.json")
                  if (source / name).is_file()}
    if "corpus" in stages:
        names |= {name for name in ("05_corpus_scale.json", "05_corpus_token_scale.json",
            "05_corpus_review.json",
            "05_corpus_warning.json", "05_corpus_candidate.json") if (source / name).is_file()}
    if through == "questions" and (source / "05_corpus.ckpt.json").is_file():
        # The original corpus stage validates its input/target/scope identity
        # before resuming. Copy the exact bytes and include their provenance.
        names.add("05_corpus.ckpt.json")
    if "world" in stages and world_semantics.enabled(view.read("01_whitepaper.json"), previous.get("config", {})):
        from pipeline.world_state import WorldState
        wp, task = view.read("01_whitepaper.json"), view.read("00_input.json")
        ws = WorldState.from_dict(view.read("02_world.json"))
        review = view.read(world_semantics.REVIEW_ARTIFACT)
        errors = world_semantics.validate_review(review, wp, ws, task_input=task)
        warning_ok = False
        if (source / world_semantics.WARNING_ARTIFACT).is_file():
            warning = view.read(world_semantics.WARNING_ARTIFACT)
            warning_ok = not world_semantics.validate_generation_warning(
                warning, review, wp, ws, task_input=task)
        if (review.get("status") != "passed" or errors) and not (warning_ok or repair_disclosure):
            raise ValueError("Source world has no current passing review or exact bound recovery warning")
        names.add(world_semantics.REVIEW_ARTIFACT)
        # In explicit disclosure-recovery mode an old/stale review is input
        # evidence only. stage_disclosure must produce a fresh current binding
        # before release; copying a historical warning cannot certify it.
        if warning_ok or (repair_disclosure and (source / world_semantics.WARNING_ARTIFACT).is_file()):
            names.add(world_semantics.WARNING_ARTIFACT)
        if (previous.get("config", {}).get("production_control") or {}).get("version") == "supply-driven/v6":
            from pipeline.factory import _require_v6_supply_gate
            _require_v6_supply_gate(view, wp, ws.to_dict())
    if repair_disclosure:
        if through not in {"questions", "corpus", "disclosure"}:
            raise ValueError("Disclosure recovery requires reuse through questions, corpus or disclosure")
        auxiliaries = {
            "02_world_draft.json", "02_world_candidate.json", "02_world_review_attempts.json",
            "02_disclosure_plan_attempts.json", "03_orders_warning.json", "04_questions_warning.json",
            "03_process_proposals.json", "04_wording_report.json", "05_corpus_warning.json",
            "05_corpus_candidate.json", "05_corpus_review.json",
        }
        names |= {name for name in auxiliaries if (source / name).is_file()}
        names |= {path.name for path in source.glob("02_disclosure_checkpoint_*.json") if path.is_file()}
        # An errored review checkpoint is execution debris, not reusable work.
        # Copying it into recovery can repeatedly replay the same incompatible
        # first action and prevent the reviewer from ever starting fresh.
        review_status = (view.read(world_semantics.REVIEW_ARTIFACT).get("status")
                         if (source / world_semantics.REVIEW_ARTIFACT).is_file() else None)
        review_error = (view.read(world_semantics.REVIEW_ARTIFACT).get("error", "")
                        if (source / world_semantics.REVIEW_ARTIFACT).is_file() else "")
        recoverable_review_error = any(marker in str(review_error) for marker in (
            "WinError 5",
            "Submitted review revision did not change the opinion",
            "Witnessed mechanism must locate actual candidate nodes",
            "Witnessed mechanism must cite at least one actual world ref",
            "Review submission differs from original opinion schema",
            "Inspect needs 1 to 12 distinct existing ids",
        ))
        if review_status != "error" or recoverable_review_error:
            names |= {path.name for path in source.glob("02_world_review_*.ckpt.json") if path.is_file()}
    if any(Path(name).name != name or not (source / name).is_file() for name in names):
        raise ValueError("Source artifacts must be existing simple file names")
    if material_first:
        # These are the same checks the real downstream stages use. Validate
        # the source before creating a destination; never invent receipts.
        from pipeline.factory import (_corpus_is_current, _disclosure_is_current,
                                      _require_current_question_sources)
        try:
            if "disclosure" in stages and not repair_disclosure and not _disclosure_is_current(view):
                raise ValueError("Public disclosure binding is missing or stale")
            if "corpus" in stages and not _corpus_is_current(view):
                raise ValueError("Corpus source/review binding is missing or stale")
            if "questions" in stages:
                _require_current_question_sources(view, whitepaper)
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Source production binding is missing or stale: {exc}") from exc
    # Read every artifact before creating the destination; invalid inputs leave no partial run.
    artifacts = {name: (source / name).read_bytes() for name in sorted(names)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}
    review_checkpoint_sources = []
    review_checkpoint_receipt = None
    if reuse_review_checkpoints:
        if not {"corpus", "questions"}.issubset(stages):
            raise ValueError("Semantic review checkpoints require exact corpus reuse")
        checkpoint_source = source / "06_review_checkpoints"
        review_checkpoint_sources = sorted(checkpoint_source.glob("*.json"))
        if not review_checkpoint_sources:
            raise ValueError("No semantic review checkpoints are available to reuse")
        checkpoint_hashes = []
        total_bytes = 0
        for path in review_checkpoint_sources:
            if path.name != Path(path.name).name or not path.is_file():
                raise ValueError("Semantic review checkpoints must be existing simple JSON files")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint_hashes.append((path.name, digest))
            total_bytes += path.stat().st_size
        review_checkpoint_receipt = {
            "source_directory": _path_label(checkpoint_source.resolve()),
            "count": len(checkpoint_hashes),
            "total_bytes": total_bytes,
            "set_sha256": hashlib.sha256(json.dumps(
                checkpoint_hashes, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "validation": "Each checkpoint remains subject to the grounding stage's exact binding checks",
        }
    previous.update(run_id=directory.name, status="created", current_stage=None, llm_calls=0,
        created=datetime.now().isoformat(), stages={k:v for k,v in previous["stages"].items() if k in stages},
        derived_from={"operation": "resume_original_stages", "run_id": source.name,
                      "source_directory": _path_label(source.resolve()),
                      "input_files": hashes, "reused_through": through,
                      "resume_from": "disclosure" if repair_disclosure else resume_from})
    if checkpoint_name:
        previous["derived_from"]["world_construction_checkpoint"] = checkpoint_name
        previous["derived_from"]["world_construction_checkpoints"] = [name for name, _ in checkpoints]
        previous["derived_from"]["world_review_required"] = True
    if repair_disclosure:
        for name in stages:
            state = previous["stages"][name]
            state.pop("error", None)
            state.update(status="succeeded", done=True, reused_from_source=True,
                         artifact=state.get("artifact", REUSE_ARTIFACTS[name]))
        previous["derived_from"].update(
            recovery="complete_or_validate_disclosure_then_resume_downstream",
            source_artifacts_immutable=True,
            source_manifest_status=source_status,
            source_manifest_current_stage=source_current_stage,
            stale_downstream_run_accepted=stale_downstream_run,
            source_stage_states=source_stage_states)
    if review_checkpoint_receipt is not None:
        previous["derived_from"]["semantic_review_checkpoints"] = review_checkpoint_receipt
    cleared = ["grounding", "quality", "met_status", "per_line_final"]
    if through not in ("questions", "corpus", "disclosure"):
        cleared += ["orders", "orders_by_line", "question_supply", "well_posed", "questions", "corpus"]
    elif "corpus" not in stages:
        cleared += ["corpus", "docs", "chars", "diversity", "render_strategy"]
    if through == "whitepaper":
        cleared += ["entities", "sessions", "world_semantic_review"]
    for key in cleared:
        previous.get("algo", {}).pop(key, None)
    if "corpus" not in stages:
        previous["derived_from"]["cleared_transient_config"] = {
            key: previous.get("config", {}).pop(key) for key in
            ("augment", "render_only", "render_only_pairs", "quotas")
            if key in previous.get("config", {})}
    directory.mkdir(parents=True)
    for name, data in artifacts.items():
        (directory / name).write_bytes(data)
    if review_checkpoint_sources:
        checkpoint_target = directory / "06_review_checkpoints"
        checkpoint_target.mkdir()
        for path in review_checkpoint_sources:
            shutil.copyfile(path, checkpoint_target / path.name)
    (directory / "manifest.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    return previous


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--to", default="quality", choices=["input", "whitepaper", "world", "orders",
                    "well_posed", "questions", "disclosure", "corpus", "grounding", "quality"])
    ap.add_argument("--max-calls", type=int, default=48)
    ap.add_argument("--max-cny", type=float, default=15)
    ap.add_argument("--json-attempts", type=int, choices=range(1, 6), default=3,
                    help="Maximum provider attempts per JSON request; each attempt consumes the same call/cost budget")
    ap.add_argument("--model", choices=["gpt-5.4-mini", "glm-4.7-nothinking", "glm-5.3-flash"], default="gpt-5.4-mini")
    ap.add_argument("--min-output-tokens", type=int, default=0)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--call-timeout-seconds", type=float, default=150)
    ap.add_argument("--llm-concurrency", type=int, default=1,
                    help="Maximum simultaneous provider requests inside this one bounded run")
    ap.add_argument("--semantic-workers", type=int, default=1,
                    help="Independent questions reviewed concurrently; final validation remains whole-run")
    ap.add_argument("--disclosure-format-attempts", type=int, choices=range(2, 7), default=2)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "max"], default="low")
    ap.add_argument("--world-review-reasoning-effort", choices=["low", "medium"], default=None,
                    help="Mini-only experiment: override reasoning for exact world.semantic_review calls; keep their token limit and timeout")
    ap.add_argument("--corpus-review-reasoning-effort", choices=["low", "medium"], default=None,
                    help="Mini-only experiment: override reasoning for exact corpus.review calls; keep their token limit")
    ap.add_argument("--corpus-review-max-tokens", type=int, choices=[4096, 8192], default=None,
                    help="Mini-only experiment: output limit for exact corpus.review calls; reserve the same limit before dispatch")
    ap.add_argument("--reference-audit-prompt", type=Path,
                    help="Experimental override for the existing audit role; copied and hashed into this run")
    ap.add_argument("--seed-pack", type=Path, help="Use the original seed entry instead of the small office scenario")
    ap.add_argument("--delivery-target", type=Path, help="Frozen post-selection quantity and tokenizer contract")
    ap.add_argument("--delivery-survival-rates", type=Path, help="Optional sourced line retention evidence")
    ap.add_argument("--world-semantic-review", action="store_true",
                    help="Enable the original world business gate for a frozen older whitepaper")
    ap.add_argument("--full-material-acceptance", action="store_true",
                    help="After the seven-line target and generation readiness, run the formal reader over all released materials under this same cumulative budget")
    quantities = ap.add_mutually_exclusive_group()
    quantities.add_argument("--question-budget", type=int)
    quantities.add_argument("--min-questions", type=int)
    ap.add_argument("--total-only", action="store_true")
    ap.add_argument("--per-line", action="append")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--max-world-entities", type=int, default=80)
    ap.add_argument("--time-span-weeks", type=int)
    ap.add_argument("--settle-reported-usage", action="store_true",
                    help="Reserve before each call; settle valid reported usage afterwards; keep full reservation for missing usage/errors")
    ap.add_argument("--target-mtokens", type=float, default=0.004)
    ap.add_argument("--haystack-ratio", type=float, default=None,
                    help="Forward the original factory's background-to-core ratio")
    ap.add_argument("--process-questions", action="store_true",
                    help="Exercise original L3 process proposals in a fresh run")
    ap.add_argument("--source-run", help="Existing run name or absolute directory; copy selected original stages into a fresh run")
    ap.add_argument("--resume-existing", action="store_true",
                    help="Continue the named run from its actual stages, preserving artifacts and the original budget ledger")
    ap.add_argument("--cumulative-child-config", type=Path,
                    help="Dedicated first-review entry from full source migration and explicit cap authority")
    ap.add_argument("--resume-after-transient-error", action="store_true",
                    help="With existing-run resume, reopen a verified transport-only stop without changing caps or accounting")
    ap.add_argument("--resume-after-filler-content-rejection", action="store_true",
                    help="Reopen a verified background-content provider rejection; omit that slot without retry")
    ap.add_argument("--resume-output-ceiling", type=int,
                    help="Reopen a fully recorded billed truncation with an explicitly larger output ceiling; money/model/call caps stay fixed")
    ap.add_argument("--resume-task-partition", type=Path,
                    help="Use a recorded scoped task decomposition to reopen billed truncation at the same output cap")
    ap.add_argument("--resume-after-json-format-error", action="store_true",
                    help="After diagnosed interface repair, reopen a fully billed syntax stop with unchanged caps")
    ap.add_argument("--resume-plan-recovery", type=Path,
                    help="Recover the bound initial empty-route failure with complete source migration and cumulative accounting")
    ap.add_argument("--resume-plan-checkpoint", type=Path,
                    help="Continue a physically settled initial-plan checkpoint in an appended source epoch")
    ap.add_argument("--resume-budget-authorization", type=Path,
                    help="Apply an explicitly authorized cumulative cap increase bound to the original stopped ledger and trace")
    ap.add_argument("--plan-retry-policy", choices=("separate-protocol-v2",),
                    help="Use the exact source/evidence-bound protocol-vs-semantic policy in --resume-plan-checkpoint")
    ap.add_argument("--plan-escalation", choices=("original-business-slot-v1", "approved-business-checkpoint-v2"),
                    help="Use the original unused business revision only after source-bound independent diagnosis")
    ap.add_argument("--reuse-world-checkpoint", action="store_true",
                    help="With whitepaper reuse, retain a compatible completed world construction checkpoint; run all world reviews")
    ap.add_argument("--reuse-through", choices=list(REUSE_STAGES), default="corpus",
                    help="Reuse frozen original inputs through whitepaper, world, questions (including a corpus checkpoint), or corpus")
    ap.add_argument("--repair-disclosure", action="store_true",
                    help="Accept an exactly bound world warning, resume saved disclosure checkpoints, and preserve upstream artifacts")
    ap.add_argument("--reuse-review-checkpoints", action="store_true",
                    help="With exact corpus reuse, copy bound per-question semantic checkpoints; the grounding stage revalidates each one")
    args = ap.parse_args()
    if not args.run or Path(args.run).name != args.run:
        ap.error("A simple run name is required")
    cumulative_child = None
    if args.cumulative_child_config is not None:
        supplied = {value.split("=", 1)[0] for value in sys.argv[1:]
                    if value.startswith("--")}
        if supplied != {"--run", "--cumulative-child-config"}:
            ap.error("The cumulative child entry accepts only --run and its evidence config")
        if (".env" in args.cumulative_child_config.name.lower()
                or "credential" in args.cumulative_child_config.name.lower()
                or "secret" in args.cumulative_child_config.name.lower()):
            ap.error("Cumulative control cannot read a credential path")
        try:
            controls = json.loads(args.cumulative_child_config.read_text(encoding="utf-8"))
            path_keys = ("derived_root", "destination", "source_root", "lineage_path",
                "stage_path", "source_epoch_path", "scope_path", "migration_path",
                "authorization_path", "boundary_path", "world_migration_path",
                "downstream_evidence_path")
            if (not isinstance(controls, dict)
                    or set(controls) != set(path_keys) | {"expected_archive_sha256"}
                    or any(not isinstance(controls[key], str) for key in path_keys)
                    or not isinstance(controls["expected_archive_sha256"], str)):
                raise ValueError("Cumulative child evidence config is incomplete")
            if any(".env" in Path(controls[key]).name.lower()
                   or "credential" in Path(controls[key]).name.lower()
                   or "secret" in Path(controls[key]).name.lower() for key in path_keys):
                raise ValueError("Cumulative control cannot read a credential path")
            paths = {key: Path(controls[key]) for key in path_keys}
            if (any(not path.is_absolute() for path in paths.values())
                    or paths["source_root"].resolve() != ROOT.resolve()
                    or paths["destination"].resolve()
                       != (ROOT / "output" / "runs" / args.run).resolve()):
                raise ValueError("Cumulative child config targets a different run or source root")
            from pipeline.cumulative_materializer import materialize_child, recover_pristine_child
            installer = (recover_pristine_child if paths["destination"].exists()
                         else materialize_child)
            cumulative_child = installer(**paths,
                expected_archive_sha256=controls["expected_archive_sha256"])
            args.resume_existing = True
        except (OSError, KeyError, TypeError, ValueError) as exc:
            ap.error(str(exc))
    start_source_hashes = implementation_hashes()
    if cumulative_child is not None:
        cumulative_start = cumulative_child.destination / "experiment_source_hashes_cumulative_start.json"
        if json.loads(cumulative_start.read_text(encoding="utf-8")) != start_source_hashes:
            ap.error("Cumulative child source snapshot differs before review")
    resume_profile = None
    resume_evidence = None
    plan_recovery_permit = None
    if args.plan_escalation is not None and (args.resume_plan_checkpoint is None or args.plan_retry_policy != "separate-protocol-v2"):
        ap.error("Business escalation requires an explicit checkpoint and its persisted retry policy")
    if args.plan_retry_policy is not None and args.resume_plan_checkpoint is None:
        ap.error("Plan retry policy requires explicit --resume-plan-checkpoint")
    if args.resume_plan_checkpoint is not None and not args.resume_existing:
        ap.error("Initial-plan checkpoint continuation requires --resume-existing")
    if args.resume_plan_recovery is not None and not args.resume_existing:
        ap.error("Initial-plan recovery requires --resume-existing")
    if args.resume_after_transient_error and not args.resume_existing:
        ap.error("Transient-error recovery requires --resume-existing")
    if args.resume_after_filler_content_rejection and not args.resume_existing:
        ap.error("Background-content recovery requires --resume-existing")
    if args.resume_output_ceiling is not None and not args.resume_existing:
        ap.error("Output-ceiling recovery requires --resume-existing")
    if args.resume_task_partition is not None and (not args.resume_existing or args.resume_output_ceiling is None):
        ap.error("Task partition requires an existing billed output recovery and its unchanged ceiling")
    if args.resume_after_json_format_error and not args.resume_existing:
        ap.error("Format recovery requires --resume-existing")
    if args.resume_budget_authorization is not None and not args.resume_existing:
        ap.error("Budget authorization requires --resume-existing")
    if sum((args.resume_after_transient_error, args.resume_after_filler_content_rejection,
            args.resume_output_ceiling is not None, args.resume_after_json_format_error,
            args.resume_budget_authorization is not None, args.resume_plan_recovery is not None,
            args.resume_plan_checkpoint is not None)) > 1:
        ap.error("Choose one explicitly evidenced recovery cause")
    if args.resume_existing:
        from pipeline.run import _io_path
        saved_profile = _io_path(ROOT / "output/runs" / args.run / "experiment_profile.json")
        try:
            resume_profile = json.loads(saved_profile.read_text(encoding="utf-8"))
            _inherit_resume_settings(args, resume_profile,
                                     {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")})
            if args.resume_plan_recovery is not None:
                from pipeline.initial_plan_recovery import validate_receipt
                plan_recovery_permit = validate_receipt(args.resume_plan_recovery,
                    saved_profile.parent, start_source_hashes)
                resume_evidence = {"reason": "explicit_initial_plan_recovery",
                    "receipt_sha256": plan_recovery_permit["receipt_sha256"],
                    "source_migration": plan_recovery_permit["source_migration"]}
            if args.resume_plan_checkpoint is not None:
                from pipeline.initial_plan_recovery import validate_checkpoint_receipt
                plan_recovery_permit = validate_checkpoint_receipt(args.resume_plan_checkpoint,
                    saved_profile.parent, start_source_hashes, plan_retry_policy=args.plan_retry_policy,
                    plan_escalation=args.plan_escalation)
                _validate_plan_retry_policy_option(args.plan_retry_policy, plan_recovery_permit)
                _validate_plan_escalation_option(args.plan_escalation, plan_recovery_permit)
                resume_evidence = {"reason": "explicit_initial_plan_checkpoint_continuation",
                    "epoch": plan_recovery_permit["epoch"],
                    "receipt_sha256": plan_recovery_permit["receipt_sha256"],
                    "source_migration": plan_recovery_permit["source_migration"]}
                if plan_recovery_permit.get("retry_policy"):
                    resume_evidence.update(plan_retry_policy=plan_recovery_permit["plan_retry_policy"],
                        retry_policy=deepcopy(plan_recovery_permit["retry_policy"]),
                        retry_policy_sha256=plan_recovery_permit["retry_policy_sha256"])
                if plan_recovery_permit.get("plan_escalation"):
                    resume_evidence.update(plan_escalation=plan_recovery_permit["plan_escalation"],
                        escalation=deepcopy(plan_recovery_permit["escalation"]),
                        escalation_sha256=plan_recovery_permit["escalation_sha256"])
                if plan_recovery_permit.get("business_resume_checkpoint"):
                    resume_evidence.update(business_resume_checkpoint=deepcopy(plan_recovery_permit["business_resume_checkpoint"]),
                        business_resume_checkpoint_sha256=plan_recovery_permit["business_resume_checkpoint_sha256"])
            if args.resume_after_transient_error:
                resume_evidence = transient_resume_evidence(resume_profile,
                                                           saved_profile.with_name("llm_attempts.jsonl"))
            if args.resume_after_filler_content_rejection:
                resume_evidence = filler_content_resume_evidence(resume_profile,
                                                                saved_profile.with_name("llm_attempts.jsonl"))
            if args.resume_output_ceiling is not None:
                resume_evidence = output_truncation_resume_evidence(resume_profile,
                    saved_profile.with_name("llm_attempts.jsonl"), args.resume_output_ceiling, args.resume_task_partition)
                args.max_output_tokens = args.resume_output_ceiling
            if args.resume_after_json_format_error:
                resume_evidence = json_format_resume_evidence(resume_profile,
                    saved_profile.with_name("llm_attempts.jsonl"))
            if args.resume_budget_authorization is not None:
                resume_evidence = budget_cap_resume_evidence(resume_profile,
                    saved_profile.with_name("llm_attempts.jsonl"), args.resume_budget_authorization)
                if resume_evidence is not None:
                    args.max_cny = resume_evidence["new_max_cny"]
        except (OSError, ValueError, TypeError) as exc:
            ap.error(str(exc))
        if resume_profile.get("stopped") and resume_evidence is None:
            ap.error("A stopped run requires an explicitly evidenced recovery cause; plain resume cannot reopen admission")
        if resume_profile.get("source_drift"):
            ap.error("The previous execution reported source drift; an explicit source migration is required")
        if cumulative_child is None and plan_recovery_permit is None:
            try:
                verify_resume_source(saved_profile.parent, start_source_hashes)
            except ValueError as exc:
                ap.error(str(exc))
    if not 0 <= args.min_output_tokens <= args.max_output_tokens <= 128000 or args.max_output_tokens < 1:
        ap.error("Output token limits must satisfy 0 <= minimum <= maximum <= 128000")
    if not math.isfinite(args.call_timeout_seconds) or args.call_timeout_seconds <= 0:
        ap.error("Call timeout must be finite and positive")
    from llm_transport import original_run_transport
    try:
        transport = original_run_transport(args.model, args.reasoning_effort, args.call_timeout_seconds)
    except ValueError as exc:
        ap.error(str(exc))
    if resume_profile is not None:
        transport = deepcopy(resume_profile["transport"])
    if args.world_review_reasoning_effort is not None and args.model != "gpt-5.4-mini":
        ap.error("--world-review-reasoning-effort is supported only for gpt-5.4-mini")
    if args.corpus_review_reasoning_effort is not None and args.model != "gpt-5.4-mini":
        ap.error("--corpus-review-reasoning-effort is supported only for gpt-5.4-mini")
    if args.corpus_review_max_tokens is not None and args.model != "gpt-5.4-mini":
        ap.error("--corpus-review-max-tokens is supported only for gpt-5.4-mini")
    audit_prompt = args.reference_audit_prompt.read_text(encoding="utf-8") if args.reference_audit_prompt else None
    if audit_prompt is not None and not audit_prompt.strip():
        ap.error("An experimental audit prompt cannot be empty")
    if args.seed_pack and args.source_run:
        ap.error("A source run supplies its verified frozen seed; do not also select a mutable seed file")
    if args.resume_existing and (args.source_run or args.seed_pack):
        ap.error("Existing-run resume cannot also select a source run or seed pack")
    if args.seed_pack and not args.seed_pack.is_file():
        ap.error("The seed pack must be an existing file")
    if ((args.question_budget is not None and args.question_budget < 1)
            or (args.min_questions is not None and args.min_questions < 1)
            or not math.isfinite(args.target_mtokens) or args.target_mtokens < 0):
        ap.error("Question budget must be positive and corpus size must be nonnegative")
    if (args.total_only or args.per_line) and args.min_questions is None:
        ap.error("Closed-loop options require --min-questions")
    if args.reuse_world_checkpoint and not (args.source_run and args.reuse_through == "whitepaper"):
        ap.error("World checkpoint reuse requires --source-run and --reuse-through whitepaper")
    if args.repair_disclosure and not (args.source_run
            and args.reuse_through in {"questions", "corpus", "disclosure"}):
        ap.error("Disclosure recovery requires --source-run and reuse through questions, corpus or disclosure")
    if args.reuse_review_checkpoints and not (args.source_run and args.reuse_through in {"corpus", "questions"}):
        ap.error("Semantic review checkpoint reuse requires a source prefix containing questions and corpus")
    if args.min_questions is not None and (args.to != "quality" or (args.source_run and not args.reuse_world_checkpoint)):
        ap.error("Quantity-target production requires a fresh run or explicit completed world checkpoint reuse through quality")
    if not 8 <= args.max_world_entities <= 80 or args.max_rounds < 1:
        ap.error("Invalid world/round limits")
    if not 1 <= args.llm_concurrency <= 16 or not 1 <= args.semantic_workers <= 16:
        ap.error("LLM concurrency and semantic workers must be between 1 and 16")
    if args.time_span_weeks is not None and not 6 <= args.time_span_weeks <= 26:
        ap.error("time-span-weeks must be between 6 and 26")
    if Path(args.run).name != args.run or args.max_calls < 1 or not math.isfinite(args.max_cny) or args.max_cny <= 0:
        ap.error("A fresh simple run name and positive limits are required")
    allowed_after_reuse = {"disclosure": {"grounding", "quality"},
                          "corpus": {"disclosure", "questions", "corpus", "grounding", "quality"},
                          "questions": {"disclosure", "corpus", "grounding", "quality"},
                          "world": {"orders", "well_posed", "questions", "corpus", "grounding", "quality"},
                          "whitepaper": {"world", "orders", "well_posed", "questions", "corpus", "grounding", "quality"}}
    if args.source_run and args.to not in allowed_after_reuse[args.reuse_through]:
        ap.error("The last stage must follow the reused stage")
    from pipeline.run import _io_path
    directory = _io_path(ROOT / "output/runs" / args.run)
    if args.resume_existing:
        if not (directory / "manifest.json").is_file():
            ap.error("Existing-run resume requires an existing run manifest")
        previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("status") == "running":
            ap.error("Existing run is marked running; stop and verify its process before continuing its budget")
    elif directory.exists():
        ap.error("Use a fresh run name; historical experiments are never overwritten")
    else:
        previous = None
    if plan_recovery_permit is not None:
        try:
            _claim_initial_plan_execution(directory, plan_recovery_permit)
        except (OSError, ValueError) as exc:
            ap.error("Initial-plan execution is already claimed or its evidence changed: " + str(exc))
    if args.source_run:
        source = Path(args.source_run)
        if not source.is_absolute():
            if source.name != args.source_run:
                ap.error("source-run must be a simple existing run name or absolute directory")
            source = ROOT / "output/runs" / args.source_run
        try:
            previous = reuse_original_stages(source, directory, args.reuse_through,
                reuse_world_checkpoint=args.reuse_world_checkpoint, model=args.model,
                repair_disclosure=args.repair_disclosure,
                reuse_review_checkpoints=args.reuse_review_checkpoints)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
    elif not args.resume_existing:
        directory.mkdir(parents=True)
    import config
    model = args.model
    config.MODEL = config.STRUCTURE_MODEL = config.DISCRIMINATOR_MODEL = model
    config.REVIEWER_MODEL = config.JUDGE_MODEL = model
    config.DISCLOSURE_FORMAT_ATTEMPTS = args.disclosure_format_attempts
    config.LLM_CONCURRENCY = args.llm_concurrency
    config._LLM_SEM = threading.BoundedSemaphore(args.llm_concurrency)
    corpus_review_transport = None
    if args.corpus_review_reasoning_effort is not None or args.corpus_review_max_tokens is not None:
        corpus_review_transport = deepcopy(transport)
        if args.corpus_review_reasoning_effort is not None:
            corpus_review_transport["profiles"]["low"]["reasoning_effort"] = args.corpus_review_reasoning_effort
    world_review_transport = None
    if args.world_review_reasoning_effort is not None:
        world_review_transport = deepcopy(transport)
        world_review_transport["profiles"]["low"]["reasoning_effort"] = args.world_review_reasoning_effort
    if resume_profile is not None:
        overrides = resume_profile.get("step_transport_overrides") or {}
        corpus_override = overrides.get("corpus.review") or {}
        world_override = overrides.get("world.semantic_review") or {}
        corpus_review_transport = deepcopy(corpus_override.get("transport"))
        world_review_transport = deepcopy(world_override.get("transport"))
        saved_cap = corpus_override.get("max_tokens")
        args.corpus_review_max_tokens = saved_cap if type(saved_cap) is int else None
    price_input, price_output = {"gpt-5.4-mini": (3.75, 22.5),
                                 "glm-4.7-nothinking": (2.37, 2.37 * 4.683544),
                                 "glm-5.3-flash": (0.632, 2.212)}[model]
    if resume_profile is not None:
        price_input = resume_profile.get("input_cny_per_m", price_input)
        price_output = resume_profile.get("output_cny_per_m", price_output)
    lock = threading.Lock()
    admission = threading.Condition(lock)
    active_reservations = {}
    # Admission and settlement are serialized, while admitted network calls may
    # run concurrently under config._LLM_SEM.  A terminal failure closes future
    # admissions; already admitted calls retain their recorded reservations.
    ledger = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": model,
        "transport": transport, "max_calls": args.max_calls, "max_cny": args.max_cny,
        "json_attempts": args.json_attempts,
        "min_output_tokens": args.min_output_tokens, "max_output_tokens": args.max_output_tokens,
        "call_timeout_seconds": args.call_timeout_seconds,
        "llm_concurrency": args.llm_concurrency,
        "semantic_workers": args.semantic_workers,
        "disclosure_format_attempts": args.disclosure_format_attempts,
        "json_retry_policy": {"version": "classified/v2", "caller_attempt_limit": "min(caller, runner)",
            "output_truncated": "double actual cap up to frozen maximum, stop when no increase remains",
            "json_syntax": "bounded feedback", "connection_429_5xx": "bounded backoff",
            "empty_deadline_read_timeout": "stop", "model_fallback": False},
        "request_lifecycle": "cancellable_async_http/v1",
        "logical_failure_policy": {"version": "agent-author-replan/v1",
            "recoverable_step": "world.agent.write", "required_kind": "output_truncated",
            "requires_received_response": True, "preserve_existing_stop": True,
            "next_call_checks": ["model", "calls", "estimated_budget"], "all_other_failures": "unit_recovery_policy_or_stop"},
        "unit_recovery_policy": {"version": "original-bounded-units/v1",
            "scope": "Transient formatting/transport failure returns to existing bounded unit; no budget/auth bypass",
            "implementation": "recoverable_unit_failure"},
        "deferred_agent_author_failures": [],
        "step_transport_overrides": ({"corpus.review": {
            "reasoning_effort": (args.corpus_review_reasoning_effort
                                 if args.corpus_review_reasoning_effort is not None else "unchanged global setting"),
            "transport": corpus_review_transport,
            "max_tokens": (args.corpus_review_max_tokens
                           if args.corpus_review_max_tokens is not None else "unchanged caller limit")}}
            if corpus_review_transport is not None else {}),
        "admitted_calls": 0, "reserved_upper_estimate_cny": 0.0, "stopped": False,
        "settle_reported_usage": args.settle_reported_usage,
        "budget_consumed_cny": 0.0, "reported_usage_estimate_cny": 0.0,
        "settled_usage_calls": 0, "unsettled_reservation_calls": 0,
        "quantity_arguments": quantity_arguments(args),
        "pricing_source": "https://rmb.dmxapi.cn/", "input_cny_per_m": price_input, "output_cny_per_m": price_output,
        "scope": "Original factory stages with a frozen experimental input. Not a quality guarantee.",
        "input": ({"kind": "seed_pack", "path": str(args.seed_pack.resolve()),
                   "sha256": hashlib.sha256(args.seed_pack.read_bytes()).hexdigest()}
                  if args.seed_pack else
                  {"kind": "frozen_seed", "seed_id": previous["config"]["seed_id"],
                   "digest": previous["config"]["seed_pack_digest"], "artifact": "00_seed_pack.json"}
                  if previous and previous.get("config", {}).get("seed_pack_digest") else
                  {"kind": "scenario", "name": "bc_small_office"}),
        "source_run": args.source_run,
        "reused_through": args.reuse_through if args.source_run else None,
        "budget_basis": "UTF-8 prompt bytes plus framing allowance and maximum completion reservation, not an invoice"}
    if world_review_transport is not None:
        ledger["step_transport_overrides"]["world.semantic_review"] = {
            "reasoning_effort": args.world_review_reasoning_effort,
            "transport": world_review_transport, "max_tokens": "unchanged caller limit"}
    if cumulative_child is not None:
        try:
            _claim_cumulative_execution(directory, cumulative_child.permit)
        except OSError as exc:
            ap.error("Cumulative child is already claimed or its execution claim failed: " + str(exc))
    ledger_path = directory / "experiment_profile.json"
    if args.resume_existing and ledger_path.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = directory / f"experiment_profile_before_resume_{suffix}.json"
        archived.write_bytes(ledger_path.read_bytes())
        if cumulative_child is None:
            for name in ("experiment_source_hashes.json", "experiment_source_hashes_at_end.json"):
                old_path = directory / name
                (directory / f"{old_path.stem}_before_resume_{suffix}.json").write_bytes(old_path.read_bytes())
        ledger = deepcopy(resume_profile)
        if resume_evidence and resume_evidence.get("reason") == "explicit_output_ceiling_resume":
            ledger["max_output_tokens"] = resume_evidence["new_output_ceiling"]
        if resume_evidence and resume_evidence.get("reason") == "explicit_cumulative_budget_resume":
            ledger["max_cny"] = resume_evidence["new_max_cny"]
        ledger.setdefault("resume_history", []).append({
            "resumed_utc": datetime.now(timezone.utc).isoformat(),
            "previous_execution_status": ledger.get("execution_status"),
            "admitted_calls": ledger["admitted_calls"],
            "budget_consumed_cny": ledger["budget_consumed_cny"],
            "unsettled_reservation_calls": ledger["unsettled_reservation_calls"],
            "previous_error_type": ledger.get("error_type"),
            "transport_recovery": resume_evidence,
            "target_stage": args.to})
        if resume_evidence is not None:
            ledger["stopped"] = False
            ledger.pop("stop_reason", None)
        ledger["execution_status"] = "running"
        ledger.pop("error_type", None)
        ledger.pop("error", None)
    _install_business_checkpoint_stage(ledger, plan_recovery_permit)
    args.full_material_acceptance = bool(args.full_material_acceptance
        or (resume_profile or {}).get("full_material_acceptance_requested", False))
    ledger["full_material_acceptance_requested"] = args.full_material_acceptance
    ledger["experiment_target_passed"] = False
    ledger["full_material_acceptance"] = {"status": "not_run", "requested": args.full_material_acceptance,
        "current": False, "passed": False, "reason": "generation_not_finished"}
    from pipeline.run import _atomic_write_json
    def save():
        _atomic_write_json(ledger_path, ledger)
    def stop(reason, *, error=None, **details):
        # Keep the first cause. Later siblings rejected by admission must not
        # replace a provider failure with a misleading budget-limit diagnosis.
        ledger["stopped"] = True
        if "stop_reason" not in ledger:
            value = {"reason": reason, "step": _TRACER_STEP.get(),
                     "at_utc": datetime.now(timezone.utc).isoformat(), **details}
            if error is not None:
                value.update(error_type=type(error).__name__, kind=config.chat_error_kind(error),
                             call_id=getattr(error, "call_id", None))
            ledger["stop_reason"] = value
        save()
        admission.notify_all()
    actual_chat, actual_json = config.chat, config.chat_json
    chat_signature, json_signature = inspect.signature(actual_chat), inspect.signature(actual_json)
    actual_emit = config.emit
    def observe_usage(event, **fields):
        actual_emit(event, **fields)
        box = _CALL_USAGE.get()
        if event == "response" and box is not None:
            box["usage"] = (fields.get("response") or {}).get("usage")
    if args.settle_reported_usage:
        config.emit = observe_usage
    def bounded_chat(messages, *a, **kw):
        bound = chat_signature.bind(messages, *a, **kw)
        kw = dict(bound.arguments); kw.pop("messages")
        a = ()
        chosen = kw.get("model") or config.MODEL
        requested_cap = kw.get("max_tokens", 4096)
        if args.corpus_review_max_tokens is not None and _TRACER_STEP.get() == "corpus.review":
            requested_cap = args.corpus_review_max_tokens
        cap = min(max(requested_cap, args.min_output_tokens), args.max_output_tokens)
        prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 512
        reserve = (prompt_bound * price_input + cap * price_output) / 1e6
        reservation_key = object()
        with lock:
            while True:
                reason = ("already_stopped" if ledger["stopped"] else
                          "model_mismatch" if chosen != model else
                          "call_limit" if ledger["admitted_calls"] >= args.max_calls else
                          _business_checkpoint_admission_reason(directory, ledger, _TRACER_STEP.get()) or
                          ("estimated_budget_limit" if ledger["budget_consumed_cny"] + reserve > args.max_cny else None))
                # Only this process's active, settleable reservations may free
                # capacity. Historical unknown liabilities remain in the floor.
                # Wait for a real settlement, then recheck every original cap.
                floor = ledger["budget_consumed_cny"] - sum(active_reservations.values())
                if (reason == "estimated_budget_limit" and args.settle_reported_usage
                        and active_reservations and floor + reserve <= args.max_cny):
                    admission.wait()
                    continue
                break
            if reason is not None:
                stop(reason, requested_model=chosen, estimated_reservation_cny=reserve)
                first = ledger["stop_reason"]
                from execution_control import ExecutionStopped
                denied = ExecutionStopped("Original experiment model/call/estimated budget limit; no provider dispatch"
                                      f"; reason={reason}; first_reason={first['reason']};"
                                      f" first_kind={first.get('kind')}; first_step={first.get('step')}",
                                      kind=first.get("kind") or reason)
                raise denied
            ledger["admitted_calls"] += 1
            ledger["reserved_upper_estimate_cny"] += reserve
            ledger["budget_consumed_cny"] += reserve
            active_reservations[reservation_key] = reserve
            save()
        box = {}
        usage_token = _CALL_USAGE.set(box)
        try:
            selected_transport = (corpus_review_transport
                if corpus_review_transport is not None and _TRACER_STEP.get() == "corpus.review"
                else world_review_transport
                if world_review_transport is not None and _TRACER_STEP.get() == "world.semantic_review"
                else transport)
            kw.update(model=model, max_tokens=cap, transport=selected_transport)
            return actual_chat(messages, *a, **kw)
        except Exception as exc:
            if not ((_JSON_REQUEST_ACTIVE.get() and (config.is_retryable_chat_error(exc)
                        or (_CLEAN_HEADER_RETRY_ACTIVE.get() and config.is_clean_header_deadline(exc))))
                    or recoverable_unit_failure(_TRACER_STEP.get(), exc)):
                with lock:
                    stop("provider_failure", error=exc)
            raise
        finally:
            _CALL_USAGE.reset(usage_token)
            settled = usage_cost(box.get("usage"), price_input, price_output) if args.settle_reported_usage else None
            with lock:
                active_reservations.pop(reservation_key)
                if settled is not None:
                    ledger["budget_consumed_cny"] += settled - reserve
                    ledger["reported_usage_estimate_cny"] += settled
                    ledger["settled_usage_calls"] += 1
                    if ledger["budget_consumed_cny"] > args.max_cny:
                        stop("reported_usage_budget_limit")
                else:
                    ledger["unsettled_reservation_calls"] += 1
                save()
                admission.notify_all()
    def bounded_json(messages, *a, **kw):
        bound = json_signature.bind(messages, *a, **kw)
        kw = dict(bound.arguments); kw.pop("messages")
        a = ()
        caller_attempts = kw.get("retries", args.json_attempts)
        if type(caller_attempts) is not int or caller_attempts < 1:
            raise ValueError("Caller retries must be a positive integer upper bound")
        cap = min(max(kw.get("max_tokens", 4096), args.min_output_tokens), args.max_output_tokens)
        maximum = min(kw.get("max_output_tokens") or args.max_output_tokens, args.max_output_tokens)
        if args.corpus_review_max_tokens is not None and _TRACER_STEP.get() == "corpus.review":
            cap = min(max(args.corpus_review_max_tokens, args.min_output_tokens), args.max_output_tokens)
            maximum = cap  # An exact step override remains an actual upper bound.
        if maximum < cap:
            raise ValueError("Caller output maximum conflicts with the frozen minimum")
        kw.update(retries=min(caller_attempts, args.json_attempts), max_tokens=cap,
                  max_output_tokens=maximum, strict_json=True, retry_delay_base=1.0)
        token = _JSON_REQUEST_ACTIVE.set(True)
        header_token = _CLEAN_HEADER_RETRY_ACTIVE.set(kw.get("retry_clean_header_deadline") is True)
        try:
            return actual_json(messages, *a, **kw)
        except Exception as exc:
            with lock:
                deferred = ((_TRACER_STEP.get() == "world.agent.write"
                    and isinstance(exc, config.ChatJSONError)
                    and exc.kind == "output_truncated"
                    and exc.response_received is True)
                    or recoverable_unit_failure(_TRACER_STEP.get(), exc)) and not ledger["stopped"]
                if deferred:
                    # The Agent receives this failed result and must ask its
                    # planner to narrow/revise the task. Every next physical
                    # call still enters bounded_chat's unchanged admission.
                    ledger["deferred_agent_author_failures"].append({
                        "step": _TRACER_STEP.get(), "kind": config.chat_error_kind(exc),
                        "call_id": getattr(exc, "call_id", None), "token_cap": getattr(exc, "token_cap", None),
                        "usage_status": getattr(exc, "usage_status", None), "admitted_calls": ledger["admitted_calls"]})
                else:
                    stop("json_execution_failure", error=exc)
                save()
            raise
        finally:
            _CLEAN_HEADER_RETRY_ACTIVE.reset(header_token)
            _JSON_REQUEST_ACTIVE.reset(token)
    config.chat, config.chat_json = bounded_chat, bounded_json
    from pipeline import factory
    factory.JOINT_DESIGN_RECOVERY = (resume_evidence if resume_evidence and
        resume_evidence.get("joint_design_checkpoint_sha256") else None)
    if audit_prompt is not None:
        from pipeline import reference_audit
        reference_audit.SYSTEM = audit_prompt
        (directory / "experiment_reference_audit_prompt.txt").write_text(audit_prompt, encoding="utf-8")
        ledger["experimental_reference_audit_prompt"] = {
            "source": str(args.reference_audit_prompt.resolve()),
            "sha256": hashlib.sha256(audit_prompt.encode("utf-8")).hexdigest(),
            "production_default_changed": False}
    factory.SCENARIOS["bc_small_office"] = {
        "description": "构造小型项目办公室的合成记录，用于验证原生成流程。规模保持很小："
            "两份项目记录、两位负责人、四个记录周，只需要项目状态、预算、负责人关系的少量变化。"
            "采用周报和任命通知两类正文。业务关系要合理，正文与当期事实一致。不要扩成大型企业。",
        "few_shot": [{"title": "项目周报", "date": "2025-01-06", "doc_type": "周报",
            "content": "松河项目本周状态为立项，预算10万元，负责人为许华。"}]}
    save()
    source_hashes = start_source_hashes
    if plan_recovery_permit is not None:
        start_name = (f"experiment_source_hashes_initial_plan_epoch_{plan_recovery_permit['epoch']:04d}_start.json"
                      if args.resume_plan_checkpoint is not None else
                      "experiment_source_hashes_initial_plan_recovery_start.json")
        (directory / start_name).write_text(
            json.dumps(source_hashes, indent=2), encoding="utf-8")
    elif cumulative_child is None:
        (directory / "experiment_source_hashes.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    sys.argv = ["pipeline.factory", "--run", args.run, "--to", args.to,
                "--semantic-workers", str(args.semantic_workers)] + quantity_arguments(args)
    if not args.source_run and not args.resume_existing and not args.delivery_target:
        sys.argv += ["--target-mtokens", str(args.target_mtokens)]
    if args.delivery_target:
        sys.argv += ["--delivery-target", str(args.delivery_target.resolve())]
    if args.haystack_ratio is not None and not args.resume_existing and not args.source_run:
        sys.argv += ["--haystack-ratio", str(args.haystack_ratio)]
    if args.delivery_survival_rates:
        sys.argv += ["--delivery-survival-rates", str(args.delivery_survival_rates.resolve())]
    if args.process_questions:
        sys.argv += ["--process-questions"]
    if args.world_semantic_review:
        sys.argv += ["--world-semantic-review"]
    if args.seed_pack:
        sys.argv += ["--seed-pack", str(args.seed_pack.resolve())]
    elif not args.source_run and not args.resume_existing:
        sys.argv += ["--scenario", "bc_small_office"]
    if args.source_run:
        sys.argv += ["--from", previous.get("derived_from", {}).get(
            "resume_from", REUSE_NEXT[args.reuse_through])]
    from pipeline.run import Tracer
    actual_tracer_json = Tracer.chat_json
    actual_tracer_text = Tracer.chat_text
    def scoped_tracer_json(self, step, messages, *a, **kw):
        token = _TRACER_STEP.set(step)
        try:
            return actual_tracer_json(self, step, messages, *a, **kw)
        finally:
            _TRACER_STEP.reset(token)
    def scoped_tracer_text(self, step, messages, *a, **kw):
        token = _TRACER_STEP.set(step)
        try:
            return actual_tracer_text(self, step, messages, *a, **kw)
        finally:
            _TRACER_STEP.reset(token)
    from pipeline.cumulative_entry_guard import (enter_bounded_run, leave_bounded_run,
        enter_explicit_cumulative_run, leave_explicit_cumulative_run)
    bounded_token = enter_bounded_run(args.run)
    cumulative_token = None
    plan_recovery_token = None
    try:
        if plan_recovery_permit is not None:
            from pipeline.initial_plan_recovery import enter_recovery
            plan_recovery_token = enter_recovery(plan_recovery_permit)
        if cumulative_child is not None:
            cumulative_token = enter_explicit_cumulative_run(cumulative_child.permit)
        Tracer.chat_json = scoped_tracer_json
        Tracer.chat_text = scoped_tracer_text
        factory.main()
        _finish_full_material_acceptance(directory, ledger, enabled=args.full_material_acceptance,
                                         model=model, max_tokens=args.max_output_tokens)
        if cumulative_child is not None:
            gate_status = _cumulative_review_exit_status(directory)
            ledger["execution_status"] = "post_corpus_world_review_gate_recorded"
            ledger["post_corpus_world_review_gate_status"] = gate_status
        else:
            ledger["execution_status"] = "completed"
    except BaseException as exc:
        ledger.update(execution_status="failed", error_type=type(exc).__name__, error=str(exc)[:600])
        raise
    finally:
        factory.JOINT_DESIGN_RECOVERY = None
        if plan_recovery_token is not None:
            from pipeline.initial_plan_recovery import leave_recovery
            leave_recovery(plan_recovery_token)
        if cumulative_token is not None:
            leave_explicit_cumulative_run(cumulative_token)
        leave_bounded_run(bounded_token)
        if args.settle_reported_usage:
            config.emit = actual_emit
        config.chat, config.chat_json = actual_chat, actual_json
        Tracer.chat_json = actual_tracer_json
        Tracer.chat_text = actual_tracer_text
        end_hashes = implementation_hashes()
        ending_name = ("experiment_source_hashes_cumulative_at_end.json" if cumulative_child is not None
                       else "experiment_source_hashes_at_end.json")
        (directory / ending_name).write_text(
            json.dumps(end_hashes, indent=2), encoding="utf-8")
        ledger["source_drift"] = sorted(name for name in set(source_hashes) | set(end_hashes)
                                         if source_hashes.get(name) != end_hashes.get(name))
        save()


if __name__ == "__main__":
    main()
