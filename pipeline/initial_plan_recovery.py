"""Explicit, cumulative recovery of the fresh run's empty initial-plan route.

This deliberately narrow entry never relaxes ordinary source guards. A receipt
binds the original failed execution before the wrapper inherits its ledger.
"""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

VERSION = "initial-plan-empty-route-recovery/v1"
CHECKPOINT_VERSION = "initial-plan-checkpoint-continuation/v1"
ESCALATION = "original-business-slot-v1"
BUSINESS_CHECKPOINT = "approved-business-checkpoint-v2"
RUN_ID = "showcase_supply_v1_fresh_20260928"
ARTIFACT = "01_initial_plan_recovery.json"
CLAIM = "01_initial_plan_recovery_claim.json"
FILES = ("manifest.json", "01_whitepaper.json", "01_instance_plan_audit.json",
         "11_production.json", "experiment_profile.json", "llm_attempts.jsonl",
         "prompts.jsonl", "experiment_source_hashes.json", "experiment_source_hashes_at_end.json")
ALLOWED_SOURCE_CHANGES = {"pipeline/plan_transactions.py", "pipeline/production.py",
                          "pipeline/initial_plan_recovery.py", "pipeline/full_material_acceptance.py",
                          "pipeline/instance_plan.py", "pipeline/layout_revision.py",
                          "pipeline/supply_capacity.py", "pipeline/blueprint_feasibility.py", "pipeline/render.py",
                          "tools/run_original_bc_smoke.py"}
_SEAL = object()
_permit = ContextVar("initial_plan_recovery_permit", default=None)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _require(value, message):
    if not value:
        raise ValueError("Initial-plan recovery: " + message)


def _map(value):
    _require(isinstance(value, dict) and value, "source map is empty")
    _require(all(isinstance(k, str) and k.endswith(".py") and not k.startswith("/")
                 and ".." not in Path(k).parts and isinstance(v, str) and len(v) == 64
                 and all(c in "0123456789abcdef" for c in v) for k, v in value.items()),
             "source map is malformed")
    return value


def _implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    result = {path.relative_to(root).as_posix(): _sha(path.read_bytes())
              for folder in (root / "pipeline", root / "eval") for path in folder.rglob("*.py")}
    result.update({name: _sha((root / name).read_bytes()) for name in
                   ("config.py", "execution_control.py", "llm_transport.py", "llm_trace.py",
                    "tools/diversity_metrics.py", "tools/run_original_bc_smoke.py")})
    return result


def _prepared_whitepaper(original):
    from pipeline.capability_contract import requirements
    from pipeline.instance_plan import VERSION as PLAN_VERSION
    wp = deepcopy(original)
    wp["supply_plan"]["instance_policy"] = PLAN_VERSION
    wp["supply_plan"]["requirements"] = [requirements(line, amount)
        for line, amount in wp["supply_plan"]["candidate_allocation"].items() if amount]
    return wp


def _records(raw):
    return [json.loads(row) for row in raw.decode("utf-8").splitlines() if row.strip()]


def _original_binding(raw):
    """Return bound controls only after every historical physical call matches."""
    manifest, wp, audit, state, profile = [json.loads(raw[name]) for name in FILES[:5]]
    _require(manifest.get("run_id") == RUN_ID, "different run identity")
    _require(profile.get("execution_status") == "failed" and profile.get("error_type") == "PlanConflict"
             and profile.get("stopped") is False and not profile.get("source_drift")
             and not profile.get("resume_history") and profile.get("admitted_calls") == 11
             and profile.get("settled_usage_calls") == 11 and profile.get("unsettled_reservation_calls") == 0
             and profile.get("max_calls") == 5000 and profile.get("max_cny") == 100
             and profile.get("model") == "glm-5.3-flash", "different or unsettled failed execution")
    _require(state.get("status") == "plan_recovery_required" and state.get("round") == 1
             and state.get("supply_round") == 1 and state.get("layout_revisions") == 0
             and not state.get("history"), "different production stage or allowance")
    _require(not wp.get("business_instance_plan") and audit.get("status") == "mechanical_recovery_required"
             and not audit.get("recovery_history") and len(audit.get("attempts", [])) == 1
             and len(audit.get("transactions", [])) == 1, "different initial plan transaction")
    proposal = audit["attempts"][0].get("proposal")
    _require(audit["attempts"][0].get("attempt") == 1 and isinstance(proposal, dict), "missing original proposal")
    tx = audit["transactions"][0]
    drafts = tx.get("drafts", [])
    _require(tx.get("mechanical_attempt") == 1 and len(drafts) == 3
             and not tx.get("edit_attempts") and not tx.get("resource_attempts")
             and not tx.get("unit_proposals"), "a real repair allowance was already used")
    findings = tx.get("findings", [])
    _require(len(findings) == 1 and findings[0].get("code") == "missing_declared_instances"
             and findings[0].get("kind") == "structure"
             and len(findings[0].get("affected_ids", [])) == 1
             and findings[0].get("evidence") == {"required": 2, "actual": 1}
             and audit.get("findings") == findings and state.get("findings") == findings,
             "failure is not the bound empty type route")
    from pipeline.plan_transactions import derived_dependencies
    candidate = derived_dependencies(proposal)
    _require(all(d.get("iteration") == i + 1 and not d.get("unit_proposals")
                 and d.get("candidate") == candidate and d.get("findings") == findings
                 for i, d in enumerate(drafts)), "empty drafts or candidate changed")
    prepared = _prepared_whitepaper(wp)
    _require(audit.get("before_hash") == _digest(prepared), "whitepaper input changed")
    trace, prompts = _records(raw["llm_attempts.jsonl"]), _records(raw["prompts.jsonl"])
    requests = [r for r in trace if r.get("event") == "request"]
    responses = [r for r in trace if r.get("event") == "response"]
    results = [r for r in trace if r.get("event") == "json_result"]
    request_ids, response_ids = [r.get("call_id") for r in requests], [r.get("call_id") for r in responses]
    _require(len(requests) == len(responses) == 11 and None not in request_ids
             and len(set(request_ids)) == 11 and Counter(request_ids) == Counter(response_ids)
             and all(r.get("model") == profile["model"] for r in requests), "physical-call ledger mismatch")
    _require(not any(r.get("event") in {"call_error", "json_error"} for r in trace), "uncertain physical response")
    steps = Counter(r.get("step") for r in requests)
    _require(steps["council.instance_plan"] == 1 and not any(steps[s] for s in
             ("council.instance_resources", "council.instance_plan_unit_repair", "council.instance_plan_repair",
              "council.instance_plan_business_repair")), "original author attempts differ")
    money = Decimal(0)
    for row in responses:
        usage = (row.get("response") or {}).get("usage") or {}
        _require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in
                     ("prompt_tokens", "completion_tokens")), "missing physical usage")
        request = next(r for r in requests if r["call_id"] == row["call_id"])
        _require(row.get("step") == request.get("step") and row.get("operation_id") == request.get("operation_id"),
                 "response identity differs")
        money += (Decimal(usage["prompt_tokens"]) * Decimal(str(profile["input_cny_per_m"]))
                  + Decimal(usage["completion_tokens"]) * Decimal(str(profile["output_cny_per_m"]))) / Decimal(1000000)
    _require(money == Decimal("0.13562878") and money == Decimal(str(profile["budget_consumed_cny"]))
             and money == Decimal(str(profile["reported_usage_estimate_cny"])), "historical cost differs")
    request = next(r for r in requests if r["step"] == "council.instance_plan")
    reply = next(r for r in responses if r["call_id"] == request["call_id"])
    parsed = [r for r in results if r.get("operation_id") == request.get("operation_id")]
    logged = [r for r in prompts if r.get("step") == "council.instance_plan"]
    content = reply.get("response", {}).get("choices", [{}])[0].get("content")
    _require(len(parsed) == 1 and parsed[0].get("step") == request["step"]
             and parsed[0].get("parsed") == proposal and isinstance(content, str)
             and json.loads(content) == proposal and len(logged) == 1 and logged[0].get("ok") is True
             and logged[0].get("output") == proposal and logged[0].get("messages") == request.get("messages"),
             "original proposal is not its actual physical response and prompt log")
    return {"manifest": manifest, "whitepaper": wp, "audit": audit, "production": state,
            "profile": profile, "candidate": candidate, "findings": findings,
            "proposal_call_id": request["call_id"], "proposal_operation_id": request["operation_id"]}


def build_receipt(directory, current_hashes):
    """Build a deterministic read-only proposal; caller saves it separately."""
    directory = Path(directory).resolve()
    _require(not (directory / CLAIM).exists() and not (directory / ARTIFACT).exists(), "recovery already claimed")
    raw = {name: (directory / name).read_bytes() for name in FILES}
    original = _original_binding(raw)
    start, end = [_map(json.loads(raw[name])) for name in FILES[-2:]]
    current = _map(deepcopy(current_hashes))
    _require(current == _implementation_hashes(), "target source map is incomplete or differs on disk")
    _require(start == end, "original execution source drifted")
    changed = sorted(k for k in set(end) | set(current) if end.get(k) != current.get(k))
    _require(changed and set(changed) <= ALLOWED_SOURCE_CHANGES and set(end) <= set(current),
             "undeclared source migration or deleted source")
    _require("pipeline/initial_plan_recovery.py" in current, "recovery implementation absent")
    return {"version": VERSION, "run_id": RUN_ID,
            "evidence": {name: {"sha256": _sha(body), "bytes": len(body)} for name, body in raw.items()},
            "source_migration": {"previous_source_hashes": end, "target_source_hashes": current,
                                 "changed_paths": changed, "ordinary_resume_guard_unchanged": True},
            "production_identity": original["production"]["identity"],
            "candidate_sha256": _digest(original["candidate"]),
            "proposal_call_id": original["proposal_call_id"],
            "proposal_operation_id": original["proposal_operation_id"],
            "first_new_step": "council.instance_plan_unit_repair", "used_unit_author_attempts": 0,
            "max_unit_author_attempts": 3, "prior_physical_calls": 11,
            "resuming_logical_plan_attempt": 2, "max_logical_plan_attempts": 3,
            "remaining_business_revisions_after_mechanical_accept": 1,
            "prior_estimated_cny": "0.13562878", "max_calls": 5000, "max_cny": 100,
            "provider_calls_by_receipt": 0, "quality_accepted": False}


def validate_receipt(path, directory, current_hashes):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    _require(receipt == build_receipt(directory, current_hashes), "receipt or source/evidence changed")
    directory = Path(directory).resolve()
    original = _original_binding({name: (directory / name).read_bytes() for name in FILES})
    permit = {**deepcopy(receipt), "receipt_sha256": _sha(raw), "directory": str(directory),
              "original": original}
    permit["_binding"] = _digest(permit)
    permit["_seal"] = _SEAL
    return permit


def _checkpoint_files(directory, *, escalation=False):
    names = set(FILES) | {ARTIFACT, CLAIM, "00_initial_plan_execution_claim.json",
                         "experiment_source_hashes_initial_plan_recovery_start.json"}
    if escalation:
        names.update(("00_input.json", "00_seed_pack.json"))
    for pattern in ("experiment_profile_before_resume_*.json",
                    "experiment_source_hashes*before_resume_*.json",
                    "00_initial_plan_execution_epoch_*.json",
                    "01_initial_plan_recovery_epoch_*.json",
                    "01_initial_plan_recovery_snapshot_epoch_*.json",
                    "experiment_source_hashes_initial_plan_epoch_*_start.json"):
        names.update(p.name for p in directory.glob(pattern))
    return {name: (directory / name).read_bytes() for name in sorted(names)}


def _author_operation_bindings(trace, prompts, requests, responses, *, steps=None):
    """Bind each saved logical author reply to all of its settled JSON attempts."""
    from config import _strip_code_fence
    bound = []
    seen = set()
    for request in requests:
        if request.get("step") not in (steps or {"council.instance_plan", "council.instance_plan_unit_repair"}):
            continue
        operation = request.get("operation_id")
        if operation in seen:
            continue
        seen.add(operation)
        group = [r for r in trace if r.get("operation_id") == operation]
        physical = [r for r in requests if r.get("operation_id") == operation]
        results = [r for r in group if r.get("event") == "json_result"]
        errors = [r for r in group if r.get("event") == "json_error"]
        _require(operation and 1 <= len(physical) <= 3 and len(results) == 1
                 and len(errors) == len(physical) - 1
                 and all(r.get("step") == request["step"] for r in group),
                 "author JSON retry chain is incomplete or exceeds original three attempts")
        original_messages = request.get("messages")
        for index, actual in enumerate(physical):
            response = next(r for r in responses if r["call_id"] == actual["call_id"])
            content = response.get("response", {}).get("choices", [{}])[0].get("content")
            _require(isinstance(content, str), "physical response content absent")
            if index == len(physical) - 1:
                _require(results[0].get("parse_mode", "complete") == "complete"
                         and json.loads(_strip_code_fence(content)) == results[0].get("parsed")
                         and group.index(response) < group.index(results[0]),
                         "saved author response differs from strict complete physical parse")
                continue
            error = errors[index]
            attempts = [r for r in group if r.get("event") == "json_attempt"]
            feedback = [r for r in group if r.get("event") == "json_retry_feedback"
                        and r.get("attempt_id") == error.get("attempt_id")]
            _require(len(attempts) == len(physical) and attempts[index].get("attempt") == index + 1
                     and attempts[index].get("attempt_id") == error.get("attempt_id")
                     and error.get("kind") == "json_syntax" and error.get("error_type") == "JSONDecodeError"
                     and error.get("retryable") is True and error.get("retry_action") == "json_feedback"
                     and error.get("raw_output") == content
                     and len(feedback) == 1 and feedback[0].get("next_attempt") == index + 2,
                     "unproven author syntax retry")
            try:
                json.loads(_strip_code_fence(content))
            except json.JSONDecodeError:
                pass
            else:
                _require(False, "claimed syntax failure was valid JSON")
            excerpt = content if len(content) <= 8192 else content[:4096] + "\n[中间内容已截去]\n" + content[-4096:]
            messages = physical[index + 1].get("messages", [])
            _require(messages[:-2] == original_messages
                     and messages[-2] == {"role": "assistant", "content": excerpt}
                     and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str)
                     and group.index(actual) < group.index(response) < group.index(error) < group.index(physical[index + 1]),
                     "syntax retry changed original task or event order")
        value = results[0].get("parsed")
        logged = [r for r in prompts if r.get("step") == request["step"]
                  and r.get("messages") == original_messages and r.get("output") == value and r.get("ok") is True]
        _require(len(logged) == 1, "saved author prompt differs")
        payload = None if request["step"] == "council.instance_plan" else json.loads(original_messages[-1]["content"])
        uid = payload["unit"]["unit_id"] if request["step"] == "council.instance_plan_unit_repair" else None
        bound.append({"step": request["step"], "unit_id": uid, "proposal": value, "call_id": physical[-1]["call_id"],
                      "physical_call_ids": [r["call_id"] for r in physical], "operation_id": operation, "request_payload": payload})
    return bound


def _bind_checkpoint_author_rows(attempts, bound, policy=None):
    """Every returned protocol reply owns a logical operation and its physical calls."""
    bindings = []
    prefixes = policy.get("legacy_attempt_prefix", {}) if isinstance(policy, dict) else {}
    for uid, rows in attempts.items():
        logical = [r for r in bound if r["unit_id"] == uid]
        cursor = 0
        _require(rows and (policy is not None or len(rows) <= 3), "unit allowance differs from persisted policy")
        for n, row in enumerate(rows, 1):
            _require(row.get("attempt") == n and row.get("status") not in
                     {"requested", "request_pending", "call_failed", "provider_error"},
                     "unit checkpoint is incomplete")
            is_v2 = policy is not None and n > len(prefixes.get(uid, []))
            protocols = row.get("protocol_attempts") if is_v2 else [row]
            _require(isinstance(protocols, list) and 1 <= len(protocols) <= 3,
                     "protocol checkpoint absent or exceeds original allowance")
            if is_v2:
                _require(row.get("retry_policy") == policy.get("version")
                         and row.get("semantic_attempt") == n - sum(p["unit_id"] == uid for p in policy["historical_protocol_pairs"]),
                         "semantic slot was reset or changed")
            base_payload = None
            for number, protocol in enumerate(protocols, 1):
                _require(cursor < len(logical) and "proposal" in protocol,
                         "saved protocol reply lacks a complete physical operation")
                actual = logical[cursor]
                cursor += 1
                _require(protocol["proposal"] == actual["proposal"], "saved protocol reply differs from physical response")
                if is_v2:
                    payload = deepcopy(actual.get("request_payload"))
                    _require(isinstance(payload, dict), "missing v2 physical request payload")
                    declared = payload.get("retry_policy", {})
                    _require(declared.get("version") == policy.get("version")
                             and declared.get("semantic_attempt") == row["semantic_attempt"]
                             and all(declared.get(k) == 3 for k in ("max_semantic_attempts", "max_protocol_attempts", "max_json_attempts")),
                             "physical request differs from its semantic policy slot")
                    feedback = payload.pop("protocol_feedback", None)
                    if number == 1:
                        _require(feedback is None, "new semantic slot inherited stale protocol feedback")
                        base_payload = payload
                    else:
                        _require(payload == base_payload and isinstance(feedback, dict)
                                 and "previous_response" in feedback
                                 and feedback["previous_response"] == protocols[number - 2]["proposal"],
                                 "protocol correction changed the original semantic task")
                    _require(protocol.get("protocol_attempt") == number
                             and protocol.get("status") in {"protocol_rejected", "protocol_valid"}
                             and (number == len(protocols) or protocol.get("status") == "protocol_rejected"),
                             "protocol checkpoint is pending or its order changed")
                else:
                    _require(isinstance(protocol["proposal"], dict), "legacy proposal is not an object")
                item = {"unit_id": uid, "attempt": n, "call_id": actual["call_id"],
                        "operation_id": actual["operation_id"], "physical_call_ids": actual["physical_call_ids"],
                        "proposal_sha256": _digest(protocol["proposal"])}
                if is_v2:
                    item.update(semantic_attempt=row["semantic_attempt"], protocol_attempt=number)
                bindings.append(item)
            if is_v2:
                _require(row.get("proposal") == protocols[-1]["proposal"]
                         and ((row.get("status") == "protocol_exhausted" and len(protocols) == 3
                               and protocols[-1]["status"] == "protocol_rejected")
                              or (row.get("status") != "protocol_exhausted" and protocols[-1]["status"] == "protocol_valid")),
                         "parent semantic attempt differs from its final protocol reply")
        _require(cursor == len(logical), "unbound physical protocol operation")
    return bindings


def _checkpoint_binding(raw):
    """Reconcile persisted authors with settled physical responses, never status labels."""
    manifest, wp, audit, state, profile = [json.loads(raw[name]) for name in FILES[:5]]
    record = json.loads(raw[ARTIFACT])
    history = audit.get("recovery_history", [])
    _require(manifest.get("run_id") == RUN_ID and manifest.get("status") != "running",
             "different or live checkpoint run")
    _require(profile.get("execution_status") == "failed" and profile.get("error_type") == "PlanConflict"
             and profile.get("stopped") is False and not profile.get("source_drift")
             and profile.get("unsettled_reservation_calls") == 0
             and profile.get("max_calls") == 5000 and profile.get("max_cny") == 100
             and profile.get("model") == "glm-5.3-flash", "different or unsettled checkpoint execution")
    _require(state.get("status") == "plan_recovery_required" and state.get("round") == 1
             and state.get("supply_round") == 1 and state.get("layout_revisions") == 0
             and not state.get("history") and not wp.get("business_instance_plan"),
             "checkpoint passed or changed its original production stage")
    _require(history and history[-1] == record and record.get("status") == "repair_failed"
             and record.get("error_type") == "PlanConflict" and record.get("business_review") is None
             and record.get("logical_plan_attempt") == 2 and record.get("max_logical_plan_attempts") == 3
             and record.get("business_revision_remaining") == 1 and not record.get("business_revision"),
             "checkpoint is not an uncommitted mechanical transaction")
    _require(audit.get("status") == "mechanical_recovery_required"
             and len(audit.get("attempts", [])) == len(audit.get("transactions", [])) == 1,
             "original logical plan attempts changed")
    original_audit = {k: v for k, v in audit.items() if k != "recovery_history"}
    serialized = json.dumps(original_audit, ensure_ascii=False, indent=2).encode("utf-8")
    serializations = (serialized, serialized + b"\n", serialized.replace(b"\n", b"\r\n"),
                      (serialized + b"\n").replace(b"\n", b"\r\n"))
    _require(history[0].get("original_audit_sha256") in {_sha(value) for value in serializations},
             "original audit bytes are no longer recoverable")
    proposal = audit["attempts"][0].get("proposal")
    _require(isinstance(proposal, dict) and audit["attempts"][0].get("attempt") == 1
             and audit.get("before_hash") == _digest(_prepared_whitepaper(wp)), "original plan input differs")
    original_tx = audit["transactions"][0]
    _require(original_tx.get("mechanical_attempt") == 1 and len(original_tx.get("drafts", [])) == 3
             and not original_tx.get("edit_attempts") and not original_tx.get("resource_attempts")
             and not original_tx.get("unit_proposals"), "original empty-route history changed")
    entry = json.loads(raw["00_initial_plan_execution_claim.json"])
    claim = json.loads(raw[CLAIM])
    _require(entry.get("run_id") == claim.get("run_id") == RUN_ID
             and entry.get("request_sha256") == claim.get("request_sha256") == history[0].get("request_sha256"),
             "original claim chain differs")
    archives = [(name, json.loads(body)) for name, body in raw.items()
                if name.startswith("experiment_profile_before_resume_")
                and _sha(body) == entry.get("original_profile_sha256")]
    _require(len(archives) == 1, "original 11-call ledger archive absent or ambiguous")
    base_profile = archives[0][1]
    stable = ("model", "max_calls", "max_cny", "transport", "max_output_tokens", "json_attempts",
              "input_cny_per_m", "output_cny_per_m")
    _require(base_profile.get("admitted_calls") == base_profile.get("settled_usage_calls") == 11
             and base_profile.get("budget_consumed_cny") == 0.13562878
             and all(base_profile.get(k) == profile.get(k) for k in stable), "original budget/settings changed")
    resume_history = profile.get("resume_history", [])
    _require(len(resume_history) == len(history), "execution epoch count differs")
    previous_source = _map(json.loads(raw["experiment_source_hashes.json"]))
    for index, (old, resume) in enumerate(zip(history, resume_history), 1):
        migration = old.get("source_migration", {})
        _require(old.get("epoch", index) == index
                 and migration.get("previous_source_hashes") == previous_source
                 and resume.get("transport_recovery", {}).get("receipt_sha256") == old.get("request_sha256")
                 and resume.get("transport_recovery", {}).get("source_migration") == migration,
                 "source or resume epoch chain differs")
        if old.get("retry_policy") is not None:
            policy = old["retry_policy"]
            policy_sha = _digest(policy)
            transport = resume.get("transport_recovery", {})
            _require(old.get("plan_retry_policy") == transport.get("plan_retry_policy") == "separate-protocol-v2"
                     and old.get("retry_policy_sha256") == transport.get("retry_policy_sha256") == policy_sha
                     and transport.get("retry_policy") == policy
                     and migration.get("retry_policy_sha256") == policy_sha
                     and migration.get("retry_policy_source_sha256") == migration.get("target_source_hashes", {}).get("pipeline/plan_transactions.py"),
                     "retry policy is not bound to its execution/source epoch")
            previous_policies = [r["retry_policy"] for r in history[:index - 1] if r.get("retry_policy") is not None]
            _require(not previous_policies or all(p == policy for p in previous_policies), "retry policy changed after migration")
        else:
            _require(not any(r.get("retry_policy") is not None for r in history[:index - 1]), "retry policy was removed")
        previous_source = _map(migration.get("target_source_hashes"))
        start_name = ("experiment_source_hashes_initial_plan_recovery_start.json" if index == 1 else
                      f"experiment_source_hashes_initial_plan_epoch_{index:04d}_start.json")
        _require(json.loads(raw[start_name]) == previous_source, "source epoch start differs")
        if index > 1:
            snapshot_name = f"01_initial_plan_recovery_snapshot_epoch_{index - 1:04d}.json"
            _require(old.get("parent_request_sha256") == history[index - 2].get("request_sha256")
                     and _sha(raw[snapshot_name]) == old.get("parent_checkpoint_sha256")
                     and json.loads(raw[snapshot_name]) == history[index - 2],
                     "parent checkpoint epoch was changed")
            for prefix in ("00_initial_plan_execution_epoch_", "01_initial_plan_recovery_epoch_"):
                saved = json.loads(raw[f"{prefix}{index:04d}.json"])
                _require(saved.get("request_sha256") == old["request_sha256"]
                         and saved.get("run_id") == RUN_ID, "checkpoint claim chain differs")
                if old.get("retry_policy") is not None:
                    _require(saved.get("retry_policy_sha256") == old.get("retry_policy_sha256"), "retry policy claim differs")
    _require(_map(json.loads(raw["experiment_source_hashes_at_end.json"])) == previous_source,
             "checkpoint source drifted during execution")
    trace, prompts = _records(raw["llm_attempts.jsonl"]), _records(raw["prompts.jsonl"])
    requests = [r for r in trace if r.get("event") == "request"]
    responses = [r for r in trace if r.get("event") == "response"]
    results = [r for r in trace if r.get("event") == "json_result"]
    ids = [r.get("call_id") for r in requests]
    _require(ids and None not in ids and len(set(ids)) == len(ids)
             and Counter(ids) == Counter(r.get("call_id") for r in responses)
             and len(ids) == profile.get("admitted_calls") == profile.get("settled_usage_calls")
             and not any(r.get("event") == "call_error" for r in trace),
             "unknown, duplicated, or unsettled physical calls")
    money = Decimal(0)
    original_money = Decimal(0)
    bound = []
    for request_index, request in enumerate(requests):
        response = next(r for r in responses if r["call_id"] == request["call_id"])
        _require(request.get("model") == profile["model"] and response.get("step") == request.get("step")
                 and response.get("operation_id") == request.get("operation_id"), "physical identity differs")
        usage = response.get("response", {}).get("usage", {})
        _require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in
                     ("prompt_tokens", "completion_tokens")), "physical usage absent")
        amount = (Decimal(usage["prompt_tokens"]) * Decimal(str(profile["input_cny_per_m"]))
                  + Decimal(usage["completion_tokens"]) * Decimal(str(profile["output_cny_per_m"]))) / Decimal(1000000)
        money += amount
        if request_index < 11:
            original_money += amount
            _require(request.get("step") != "council.instance_plan_unit_repair", "original trace prefix was replaced")
        else:
            _require(request.get("step") == "council.instance_plan_unit_repair", "unbound post-prefix physical request")
        if request.get("step") not in {"council.instance_plan", "council.instance_plan_unit_repair"}:
            _require(request.get("step") not in {"council.instance_resources", "council.instance_plan_repair",
                     "council.instance_plan_business_repair"}, "unbound repair call")
            continue
    bound = _author_operation_bindings(trace, prompts, requests, responses)
    syntax_operations = {r.get("operation_id") for r in trace if r.get("event") == "json_error"}
    _require(syntax_operations <= {r["operation_id"] for r in bound if r["unit_id"] is not None},
             "syntax errors outside the bound local author operations")
    # The immutable wrapper ledger was accumulated as binary floats. Keep its
    # bytes, but compare to exact physical token pricing with only a sub-token
    # representation tolerance. This is not a budget reset or reconciliation.
    tolerance = Decimal("0.000000000001")
    _require(all(Decimal(str(profile.get(key))).is_finite()
                 and abs(money - Decimal(str(profile[key]))) <= tolerance for key in
                 ("budget_consumed_cny", "reported_usage_estimate_cny")), "cumulative settled cost differs")
    _require(original_money == Decimal(str(base_profile["budget_consumed_cny"])), "original physical cost prefix changed")
    initial = [r for r in bound if r["unit_id"] is None]
    _require(len(initial) == 1 and initial[0]["proposal"] == proposal, "original proposal physical binding differs")
    tx = record.get("transaction", {})
    drafts = tx.get("drafts", [])
    _require(drafts and [d.get("iteration") for d in drafts] == list(range(1, len(drafts) + 1))
             and not tx.get("resource_attempts"), "checkpoint drafts are absent/nonconsecutive or used other authors")
    attempts = tx.get("edit_attempts", {})
    author_bindings = _bind_checkpoint_author_rows(attempts, bound, record.get("retry_policy"))
    _require(len(ids) == 11 + sum(len(r["physical_call_ids"]) for r in author_bindings)
             and len(bound) == 1 + len(author_bindings),
             "physical calls outside the persisted original transaction")
    from pipeline.plan_transactions import derived_dependencies
    candidate = derived_dependencies(proposal)
    unit_ids = [u["unit_id"] for u in candidate["units"]]
    _require(set(attempts) <= set(unit_ids), "author identity outside the original plan")
    from pipeline.plan_transactions import retry_policy_counts
    semantic_counts = retry_policy_counts(tx, record.get("retry_policy"))
    remaining = {uid: 3 - semantic_counts.get(uid, 0) for uid in unit_ids}
    _require(any(remaining.values()), "all original unit author attempts are exhausted")
    _require(all(d.get("candidate") == candidate and not d.get("unit_proposals")
                 for d in original_tx["drafts"]), "original failed draft changed")
    return {"manifest": manifest, "whitepaper": wp, "audit": audit, "production": state,
            "profile": profile, "candidate": candidate, "findings": original_tx["findings"],
            "record": record, "author_bindings": author_bindings,
            "estimated_cny_exact": str(money), "remaining_unit_author_slots": remaining}


def _serialized_matching(value, expected):
    body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    for raw in (body, body + b"\n", body.replace(b"\n", b"\r\n"), (body + b"\n").replace(b"\n", b"\r\n")):
        if _sha(raw) == expected:
            return raw
    _require(False, "archived JSON snapshot cannot reproduce its original byte hash")


def _business_checkpoint_binding(raw):
    """Reconcile the original 23-call chain plus real diagnostic and B1 calls."""
    audit, profile, record = [json.loads(raw[n]) for n in ("01_instance_plan_audit.json", "experiment_profile.json", ARTIFACT)]
    history = audit.get("recovery_history", [])
    _require(len(history) == 5 and history[-1] == record and record.get("epoch") == 5
             and record.get("status") == "business_repair_required"
             and record.get("logical_plan_attempt") == record.get("max_logical_plan_attempts") == 3
             and record.get("business_revision_remaining") == 0
             and record.get("plan_escalation") == ESCALATION,
             "not the bound exhausted first business revision checkpoint")
    migration = record["source_migration"]
    escalation = record["escalation"]
    _require(_digest(escalation) == record.get("escalation_sha256") == migration.get("escalation_sha256")
             and _digest(record.get("retry_policy")) == record.get("retry_policy_sha256") == migration.get("retry_policy_sha256")
             and json.loads(raw["experiment_source_hashes_at_end.json"]) == migration.get("target_source_hashes")
             and json.loads(raw["experiment_source_hashes_initial_plan_epoch_0005_start.json"]) == migration.get("target_source_hashes"),
             "business checkpoint source/policy/escalation binding differs")
    entry = json.loads(raw["00_initial_plan_execution_epoch_0005.json"])
    claim = json.loads(raw["01_initial_plan_recovery_epoch_0005.json"])
    for value in (entry, claim):
        _require(value.get("run_id") == RUN_ID and value.get("request_sha256") == record.get("request_sha256")
                 and value.get("retry_policy_sha256") == record["retry_policy_sha256"]
                 and value.get("escalation_sha256") == record["escalation_sha256"], "business checkpoint claim differs")
    saved_profiles = [body for name, body in raw.items() if name.startswith("experiment_profile_before_resume_")
                      and _sha(body) == entry.get("original_profile_sha256")]
    _require(len(saved_profiles) == 1, "original 23-call ledger archive missing or ambiguous")
    prior_profile = json.loads(saved_profiles[0])
    _require(prior_profile.get("admitted_calls") == prior_profile.get("settled_usage_calls") == 23
             and profile.get("admitted_calls") == profile.get("settled_usage_calls") == 25
             and profile.get("unsettled_reservation_calls") == 0 and profile.get("execution_status") == "failed"
             and profile.get("error_type") == "PlanConflict" and profile.get("stopped") is False
             and not profile.get("source_drift") and profile.get("max_cny") == 100 and profile.get("max_calls") == 5000,
             "different or unsettled business checkpoint ledger")
    stable = ("model", "max_calls", "max_cny", "transport", "max_output_tokens", "json_attempts",
              "input_cny_per_m", "output_cny_per_m", "quantity_arguments", "input")
    _require(all(profile.get(k) == prior_profile.get(k) for k in stable), "business checkpoint frozen settings changed")
    resumes = profile.get("resume_history", [])
    _require(len(resumes) == 5 and resumes[:-1] == prior_profile.get("resume_history"), "business resume history changed")
    recovery = resumes[-1].get("transport_recovery", {})
    _require(recovery.get("receipt_sha256") == record["request_sha256"] and recovery.get("source_migration") == migration
             and recovery.get("retry_policy") == record["retry_policy"] and recovery.get("retry_policy_sha256") == record["retry_policy_sha256"]
             and recovery.get("plan_escalation") == ESCALATION and recovery.get("escalation") == escalation
             and recovery.get("escalation_sha256") == record["escalation_sha256"], "business profile migration differs")
    trace, prompts = _records(raw["llm_attempts.jsonl"]), _records(raw["prompts.jsonl"])
    requests = [r for r in trace if r.get("event") == "request"]
    responses = [r for r in trace if r.get("event") == "response"]
    ids = [r.get("call_id") for r in requests]
    _require(len(ids) == len(set(ids)) == 25 and None not in ids
             and Counter(ids) == Counter(r.get("call_id") for r in responses)
             and not any(r.get("event") == "call_error" for r in trace), "business physical calls are unknown or unsettled")
    total = Decimal(0)
    for request in requests:
        response = next(r for r in responses if r["call_id"] == request["call_id"])
        usage = response.get("response", {}).get("usage", {})
        _require(request.get("model") == profile["model"] and response.get("step") == request.get("step")
                 and response.get("operation_id") == request.get("operation_id")
                 and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")),
                 "business physical identity or usage differs")
        total += (Decimal(usage["prompt_tokens"]) * Decimal(str(profile["input_cny_per_m"]))
                  + Decimal(usage["completion_tokens"]) * Decimal(str(profile["output_cny_per_m"]))) / Decimal(1000000)
    _require(total == Decimal("0.414118632") and all(Decimal(str(profile[k])).is_finite()
             and abs(Decimal(str(profile[k])) - total) <= Decimal("1e-12") for k in
             ("budget_consumed_cny", "reported_usage_estimate_cny")), "cumulative business cost differs")
    tail = requests[23:]
    _require([r["step"] for r in tail] == ["council.blueprint_feasibility", "council.instance_plan_business_repair"],
             "diagnostic or original B1 physical step differs")
    tail_ops = {r["operation_id"] for r in tail}
    cut = next(i for i, row in enumerate(trace) if row.get("operation_id") in tail_ops)
    _require([r for r in trace[:cut] if r.get("event") == "request"] == requests[:23]
             and all(r.get("operation_id") in tail_ops for r in trace[cut:] if r.get("event") == "request"),
             "business requests overlap the original physical prefix")
    tail_prompts = prompts[-2:]
    _require([r.get("step") for r in tail_prompts] == [t["step"] for t in tail], "unbound business prompt tail")
    # Validate the exact parent audit, artifact, source epoch and 23-call physical
    # prefix through the unchanged checkpoint validator, using in-memory views.
    parent_bytes = raw["01_initial_plan_recovery_snapshot_epoch_0004.json"]
    parent = json.loads(parent_bytes)
    _require(_sha(parent_bytes) == record.get("parent_checkpoint_sha256") and parent == history[-2]
             and parent.get("request_sha256") == record.get("parent_request_sha256")
             and record.get("transaction") == parent.get("transaction"), "local history was changed in business phase")
    parent_audit = deepcopy(audit); parent_audit["recovery_history"] = history[:-1]
    prefix = dict(raw)
    prefix.update({"01_instance_plan_audit.json": _serialized_matching(parent_audit, escalation["original_audit_sha256"]),
                   ARTIFACT: parent_bytes, "experiment_profile.json": saved_profiles[0],
                   "experiment_source_hashes_at_end.json": json.dumps(migration["previous_source_hashes"]).encode(),
                   "llm_attempts.jsonl": b"".join(raw["llm_attempts.jsonl"].splitlines(keepends=True)[:cut]),
                   "prompts.jsonl": b"".join(raw["prompts.jsonl"].splitlines(keepends=True)[:-2])})
    original = _checkpoint_binding(prefix)
    _require(original["estimated_cny_exact"] == "0.314097680"
             and escalation.get("candidate") == parent["transaction"]["drafts"][-1]["candidate"],
             "original failed candidate or monetary prefix changed")
    diagnostic = record.get("business_diagnostic", {})
    from pipeline.blueprint_feasibility import validate_uncompiled_diagnostic
    validate_uncompiled_diagnostic(_prepared_whitepaper(original["whitepaper"]), escalation["candidate"], diagnostic)
    _require(diagnostic.get("decision") == "repair" and diagnostic.get("author_allowed") is True
             and record.get("business_review") == diagnostic.get("review"), "original diagnostic did not authorize repair")
    revision = record.get("business_revision", {})
    rows = revision.get("business_revisions", [])
    _require(revision.get("status") == "business_repair_required" and len(rows) == 1
             and rows[0].get("attempt") == 1 and rows[0].get("status") == "compiler_rejected"
             and len(rows[0].get("protocol_attempts", [])) == 1
             and rows[0]["protocol_attempts"][0].get("status") == "protocol_valid"
             and rows[0]["protocol_attempts"][0].get("protocol_attempt") == 1
             and rows[0]["protocol_attempts"][0].get("proposal") == rows[0].get("proposal")
             and revision.get("uncompiled_diagnostic") == diagnostic
             and revision.get("uncompiled_candidate_sha256") == _digest(escalation["candidate"]),
             "B1 is not its complete compiler-rejected original business attempt")
    bindings = _author_operation_bindings(trace[cut:], tail_prompts, tail,
        [r for r in responses if r.get("operation_id") in tail_ops], steps={t["step"] for t in tail})
    _require(len(bindings) == 2 and bindings[0]["proposal"] == diagnostic["review"]["review"]
             and bindings[0]["request_payload"] == diagnostic["review"]["input"]
             and bindings[1]["proposal"] == rows[0]["proposal"], "business replies are not their real physical results")
    calls = record.get("business_call_history", [])
    _require(len(calls) == 2, "business checkpoint has unbound logical calls")
    for logical, physical, request in zip(calls, bindings, tail):
        _require(logical.get("status") == "returned" and logical.get("step") == request["step"]
                 and logical.get("messages_sha256") == _digest(request["messages"])
                 and logical.get("response") == physical["proposal"]
                 and logical.get("response_sha256") == _digest(physical["proposal"]), "business call history binding differs")
    original.update(audit=audit, profile=profile, record=record, estimated_cny_exact=str(total),
                    business_bindings=[{k: v for k, v in b.items() if k not in {"request_payload", "proposal"}} for b in bindings])
    return original


def _business_escalation(original, raw, current, retry_policy):
    """Bind the untouched business slot to this settled failed candidate only."""
    from pipeline.instance_plan import compile_plan, PlanConflict
    from pipeline.plan_transactions import retry_policy_counts
    record = original["record"]
    tx = record["transaction"]
    _require(len(original["audit"]["recovery_history"]) == 4
             and original["profile"]["admitted_calls"] == 23
             and original["estimated_cny_exact"] == "0.314097680"
             and retry_policy is not None and record.get("retry_policy") == retry_policy
             and record.get("business_revision_remaining") == 1
             and record.get("logical_plan_attempt") == 2 and not record.get("business_revision")
             and len(tx.get("drafts", [])) == 5
             and retry_policy_counts(tx, retry_policy).get("u_tower_case") == 3,
             "business escalation requires the settled 23-call checkpoint and original unused slot")
    candidate = deepcopy(tx["drafts"][-1]["candidate"])
    findings = deepcopy(tx["drafts"][-1].get("findings", []))
    _require(len(record.get("findings", [])) == 1
             and record["findings"][0].get("code") == "repair_author_exhausted"
             and record["findings"][0].get("affected_ids") == ["u_tower_case"]
             and record["findings"][0].get("evidence", {}).get("compiler_findings") == findings
             and [f.get("code") for f in findings] == ["native_trend_temporal_capacity"],
             "business escalation does not match the persisted exhausted local transaction")
    wp = _prepared_whitepaper(original["whitepaper"])
    try:
        compile_plan(wp, candidate)
    except PlanConflict as error:
        _require(error.findings == findings, "current compiler differs from the bound uncommitted findings")
    else:
        _require(False, "candidate no longer needs the bound escalation")
    seed_input = json.loads(raw["00_input.json"])
    seed_pack = json.loads(raw["00_seed_pack.json"])
    target = deepcopy(original["manifest"].get("config", {}).get("delivery_target"))
    _require(isinstance(seed_input, dict) and isinstance(seed_pack, dict)
             and isinstance(target, dict) and target.get("max_supply_rounds") == 3,
             "original seed and supply target are absent")
    # Keep every failure, local raw reply and protocol transcript. The many
    # repeated whole-plan snapshots remain SHA-bound in immutable evidence;
    # the independent reviewer receives the latest complete plan once.
    def diagnostic_projection(value):
        if isinstance(value, list):
            return [diagnostic_projection(row) for row in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        projected = {}
        for key, item in value.items():
            if key in {"candidate", "base", "proposal"} and isinstance(item, dict) and "units" in item:
                projected[key + "_sha256"] = _digest(item)
            elif key == "source_migration":
                projected["source_migration_sha256"] = _digest(item)
            else:
                projected[key] = diagnostic_projection(item)
        return projected
    history = diagnostic_projection({"initial_attempts": original["audit"]["attempts"],
        "initial_transactions": original["audit"]["transactions"],
        "cumulative_transaction": tx, "terminal_findings": record["findings"],
        "prior_recovery_epochs": [{key: old[key] for key in
             ("epoch", "request_sha256", "status", "findings", "source_migration", "retry_policy_sha256") if key in old}
             for old in original["audit"]["recovery_history"]]})
    history["projection"] = {"version": "complete-failures-with-plan-snapshot-digests/v1",
        "complete_source_audit_sha256": _sha(raw["01_instance_plan_audit.json"]),
        "full_snapshot_location": "01_instance_plan_audit.json", "all_local_raw_replies_retained": True}
    return {"version": ESCALATION, "phase": "uncompiled_plan_escalation",
        "candidate": candidate, "candidate_sha256": _digest(candidate),
        "compiler_findings": findings, "compiler_findings_sha256": _digest(findings),
        "failure_history": history, "failure_history_sha256": _digest(history),
        "original_audit_sha256": _sha(raw["01_instance_plan_audit.json"]),
        "seed_input": seed_input, "seed_input_sha256": _sha(raw["00_input.json"]),
        "seed_pack": seed_pack, "seed_pack_sha256": _sha(raw["00_seed_pack.json"]),
        "delivery_target": target, "delivery_target_sha256": _digest(target),
        "blueprint_sha256": _digest(wp["world_blueprint"]),
        "requirements_sha256": _digest(wp["supply_plan"]["requirements"]),
        "limits_sha256": _digest(wp["supply_plan"]["world_limits"]),
        "seed_contract_sha256": _digest(wp.get("seed_contract", {})),
        "target_source_hashes_sha256": _digest(current), "retry_policy_sha256": _digest(retry_policy),
        "original_business_revision_remaining": 1, "max_business_revisions": 1,
        "original_local_semantic_limit": 3, "max_supply_rounds": 3,
        "first_new_step": "council.blueprint_feasibility",
        "author_requires_independent_repair_decision": True,
        "accept_requires_original_compiler": True}


def build_checkpoint_receipt(directory, current_hashes, *, plan_retry_policy=None, plan_escalation=None):
    """Explicit source migration for a fully settled, persisted mechanical epoch."""
    directory = Path(directory).resolve()
    _require(plan_escalation in (None, ESCALATION, BUSINESS_CHECKPOINT), "unknown business escalation mode")
    _require(plan_escalation is None or plan_retry_policy == "separate-protocol-v2", "business escalation requires the persisted retry policy")
    raw = _checkpoint_files(directory, escalation=plan_escalation is not None)
    original = (_business_checkpoint_binding(raw) if plan_escalation == BUSINESS_CHECKPOINT else _checkpoint_binding(raw))
    epoch = len(original["audit"]["recovery_history"]) + 1
    entry_claim = f"00_initial_plan_execution_epoch_{epoch:04d}.json"
    repair_claim = f"01_initial_plan_recovery_epoch_{epoch:04d}.json"
    _require(not (directory / entry_claim).exists() and not (directory / repair_claim).exists(), "checkpoint epoch already claimed")
    previous = _map(json.loads(raw["experiment_source_hashes_at_end.json"]))
    current = _map(deepcopy(current_hashes))
    _require(current == _implementation_hashes(), "target source map is incomplete or differs on disk")
    changed = sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))
    _require(set(changed) <= ALLOWED_SOURCE_CHANGES and set(previous) <= set(current), "undeclared checkpoint source migration")
    retry_policy = deepcopy(original["record"].get("retry_policy"))
    if retry_policy is not None:
        _require(plan_retry_policy == "separate-protocol-v2", "persisted policy requires explicit matching checkpoint option")
    elif plan_retry_policy is not None:
        _require(plan_retry_policy == "separate-protocol-v2", "unknown checkpoint retry policy")
        from pipeline.plan_transactions import build_retry_policy_migration
        _require(original["profile"]["admitted_calls"] == 22 and original["estimated_cny_exact"] == "0.295167700"
                 and epoch == 4, "retry policy migration requires the bound 22-call checkpoint")
        retry_policy = build_retry_policy_migration(original["record"]["transaction"],
            [{"unit_id": "u_tower_case", "first_attempt": 2, "second_attempt": 3}])
    receipt = {"version": CHECKPOINT_VERSION, "run_id": RUN_ID, "epoch": epoch,
        "entry_claim": entry_claim, "repair_claim": repair_claim,
        "parent_request_sha256": original["record"]["request_sha256"],
        "parent_checkpoint_sha256": _sha(raw[ARTIFACT]),
        "evidence": {name: {"sha256": _sha(body), "bytes": len(body)} for name, body in raw.items()},
        "source_migration": {"previous_source_hashes": previous, "target_source_hashes": current,
                             "changed_paths": changed, "ordinary_resume_guard_unchanged": True},
        "production_identity": original["production"]["identity"],
        "author_bindings": original["author_bindings"],
        "used_unit_author_attempts": {uid: len(rows) for uid, rows in original["record"]["transaction"]["edit_attempts"].items()},
        "max_unit_author_attempts": 3, "resuming_logical_plan_attempt": 2, "max_logical_plan_attempts": 3,
        "joint_batch_policy": "consume_remaining_original_author_slots",
        "joint_drafts_consumed": len(original["record"]["transaction"]["drafts"]),
        "remaining_unit_author_slots": original["remaining_unit_author_slots"],
        "maximum_additional_joint_batches": sum(original["remaining_unit_author_slots"].values()),
        "remaining_business_revisions_after_mechanical_accept": 1,
        "prior_physical_calls": original["profile"]["admitted_calls"],
        "prior_estimated_cny": original["estimated_cny_exact"],
        "ledger_representation_tolerance_cny": "0.000000000001",
        "max_calls": 5000, "max_cny": 100, "first_new_step": "council.instance_plan_unit_repair",
        "provider_calls_by_receipt": 0, "quality_accepted": False}
    if retry_policy is not None:
        policy_sha = _digest(retry_policy)
        receipt.update(plan_retry_policy="separate-protocol-v2", retry_policy=retry_policy,
                       retry_policy_sha256=policy_sha)
        receipt["source_migration"].update(retry_policy_sha256=policy_sha,
            retry_policy_source_sha256=current["pipeline/plan_transactions.py"])
        from pipeline.plan_transactions import retry_policy_counts
        counts = retry_policy_counts(original["record"]["transaction"], retry_policy)
        remaining = {uid: 3 - counts.get(uid, 0) for uid in receipt["remaining_unit_author_slots"]}
        receipt.update(used_unit_semantic_attempts=counts, remaining_unit_author_slots=remaining,
                       maximum_additional_joint_batches=sum(remaining.values()),
                       joint_batch_policy="consume_original_semantic_slots_with_separate_protocol")
    if plan_escalation is not None:
        escalation = (deepcopy(original["record"]["escalation"]) if plan_escalation == BUSINESS_CHECKPOINT
                      else _business_escalation(original, raw, current, retry_policy))
        receipt.update(plan_escalation=plan_escalation, escalation=escalation,
                       escalation_sha256=_digest(escalation), first_new_step=escalation["first_new_step"])
        receipt["source_migration"]["escalation_sha256"] = receipt["escalation_sha256"]
        if plan_escalation == BUSINESS_CHECKPOINT:
            record = original["record"]
            business = {"version": "business-revision-checkpoint/v1",
                "candidate_sha256": _digest(escalation["candidate"]),
                "diagnostic_sha256": _digest(record["business_diagnostic"]),
                "business_revisions_sha256": _digest(record["business_revision"]["business_revisions"]),
                "completed_business_attempts": 1, "max_business_attempts": 3,
                "remaining_business_attempts": 2, "max_additional_physical_requests": 36,
                "stage_start_admitted_calls": 25, "original_max_logical_plan_attempts": 3,
                "max_logical_plan_attempts": 5, "next_logical_plan_attempt": 4,
                "authorization": "user_explicit_add_two_business_attempts_same_run_100_5000"}
            receipt.update(business_resume_checkpoint=business, business_resume_checkpoint_sha256=_digest(business),
                business_bindings=original["business_bindings"], first_new_step="council.instance_plan_business_repair",
                max_logical_plan_attempts=5, resuming_logical_plan_attempt=4,
                remaining_business_revisions_after_mechanical_accept=2)
            receipt["source_migration"]["business_resume_checkpoint_sha256"] = receipt["business_resume_checkpoint_sha256"]
    return receipt


def validate_checkpoint_receipt(path, directory, current_hashes, *, plan_retry_policy=None, plan_escalation=None):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    _require(receipt == build_checkpoint_receipt(directory, current_hashes, plan_retry_policy=plan_retry_policy, plan_escalation=plan_escalation), "checkpoint receipt or source/evidence changed")
    directory = Path(directory).resolve()
    evidence = _checkpoint_files(directory, escalation=plan_escalation is not None)
    original = (_business_checkpoint_binding(evidence) if plan_escalation == BUSINESS_CHECKPOINT
                else _checkpoint_binding(evidence))
    permit = {**deepcopy(receipt), "receipt_sha256": _sha(raw), "directory": str(directory), "original": original}
    permit["_binding"] = _digest(permit)
    permit["_seal"] = _SEAL
    return permit


def _validate_permit(permit):
    _require(isinstance(permit, dict) and permit.get("_seal") is _SEAL
             and permit.get("_binding") == _digest({k: v for k, v in permit.items()
                                                   if k not in {"_seal", "_binding"}}), "unvalidated or mutated permit")


def enter_recovery(permit):
    _validate_permit(permit)
    return _permit.set(permit)


def leave_recovery(token):
    _permit.reset(token)


def _run_business_escalation(wp, permit, record, tracer, checkpoint):
    from pipeline.blueprint_feasibility import diagnose_uncompiled_plan, validate_uncompiled_diagnostic, assess
    from pipeline.instance_plan import compile_plan, PlanConflict, finding
    from pipeline.layout_revision import revise
    escalation = permit["escalation"]
    _require(_digest(escalation) == permit["escalation_sha256"]
             and record["business_revision_remaining"] == 1, "escalation or business allowance changed")
    candidate = deepcopy(escalation["candidate"])
    class RecordedTracer:
        def chat_json(self, step, messages, *args, **kwargs):
            entry = {"step": step, "messages_sha256": _digest(messages), "status": "request_pending"}
            record.setdefault("business_call_history", []).append(entry)
            checkpoint()
            try:
                response = tracer.chat_json(step, messages, *args, **kwargs)
            except BaseException as error:
                entry.update(status="call_failed", error_type=type(error).__name__)
                checkpoint()
                raise
            entry.update(status="returned", response=deepcopy(response), response_sha256=_digest(response))
            checkpoint()
            return response
    audit_tracer = RecordedTracer()
    record.update(status="business_diagnostic", business_diagnostic={"status": "request_pending"})
    checkpoint()
    diagnostic = diagnose_uncompiled_plan(wp, candidate, deepcopy(escalation["compiler_findings"]), audit_tracer,
        history=deepcopy(escalation["failure_history"]),
        source_binding={"run_id": RUN_ID, "source_migration": deepcopy(permit["source_migration"]),
            "seed_input": deepcopy(escalation["seed_input"]), "seed_pack": deepcopy(escalation["seed_pack"]),
            "delivery_target": deepcopy(escalation["delivery_target"]),
            "manifest_config": deepcopy(permit["original"]["manifest"].get("config")),
            "escalation_sha256": permit["escalation_sha256"]},
        retry_policy_binding={"policy": deepcopy(permit["retry_policy"]), "sha256": permit["retry_policy_sha256"]},
        remaining_business_attempts=1)
    record["business_diagnostic"] = deepcopy(diagnostic)
    record["business_review"] = deepcopy(diagnostic.get("review"))
    checkpoint()
    validate_uncompiled_diagnostic(wp, candidate, diagnostic)
    decision = diagnostic.get("decision")
    _require(decision != "repair" or diagnostic.get("author_allowed") is True,
             "independent review has not authorized a responsibility-bound repair")
    uncompiled = diagnostic if decision == "repair" else None
    if decision == "unresolved":
        record["status"] = "design_unresolved"
        checkpoint()
        raise PlanConflict([finding("business_unresolved", "Original independent reviewer cannot resolve the failed plan",
                                    kind="design", evidence=diagnostic)])
    if decision == "accept":
        # A positive opinion cannot make a mechanically invalid candidate true.
        revised = compile_plan(wp, candidate)
        review = assess(revised, audit_tracer, feedback={"phase": "after_uncompiled_plan_diagnostic",
                        "recovery_request_sha256": permit["receipt_sha256"]})
        record["post_compile_business_review"] = deepcopy(review)
        checkpoint()
        if review.get("decision") == "accept":
            return revised
        decision = review.get("decision")
        if decision != "repair":
            record["status"] = "design_unresolved"
            checkpoint()
            raise PlanConflict([finding("business_unresolved", "Compiled plan was not accepted by the independent reviewer",
                                        kind="design", evidence=review)])
        business_evidence = review
    elif decision == "repair":
        business_evidence = diagnostic
    else:
        _require(False, "independent diagnostic response has no valid decision")
    record.update(status="business_repair", logical_plan_attempt=3,
                  business_revision_remaining=0, business_revision={})
    checkpoint()
    feedback = {"original": {"phase": "initial_business_layout"}, "rejected_plan": deepcopy(candidate),
        "findings": [finding("business_repair", "Original independent review requests its one remaining revision",
                             kind="business", evidence=business_evidence)]}
    try:
        revised = revise(wp, candidate, feedback, audit_tracer, record["business_revision"],
                         max_attempts=1, checkpoint=checkpoint,
                         **({"uncompiled_diagnostic": uncompiled} if uncompiled is not None else {}))
    except BaseException:
        record["status"] = record["business_revision"].get("status", "business_repair_required")
        checkpoint()
        raise
    reviews = record["business_revision"].get("business_revisions", [])
    _require(reviews and reviews[-1].get("business_review", {}).get("decision") == "accept",
             "remaining business revision lacks final independent acceptance")
    record["final_business_review"] = deepcopy(reviews[-1]["business_review"])
    checkpoint()
    return revised


def _run_business_checkpoint(wp, permit, record, tracer, checkpoint):
    from pipeline.layout_revision import revise
    from pipeline.instance_plan import finding
    policy = permit["business_resume_checkpoint"]
    _require(_digest(policy) == permit["business_resume_checkpoint_sha256"], "business authorization binding changed")
    candidate = deepcopy(permit["escalation"]["candidate"])
    old_rows = deepcopy(record["business_revision"]["business_revisions"])
    def save(*_args, **_kwargs):
        rows = record["business_revision"]["business_revisions"]
        _require(rows[:len(old_rows)] == old_rows and len(old_rows) <= len(rows) <= 3,
                 "old B1 was rewritten or approved business allowance exceeded")
        record.update(logical_plan_attempt=2 + len(rows), business_revision_remaining=3 - len(rows))
        checkpoint()
    class RecordedTracer:
        def chat_json(self, step, messages, *args, **kwargs):
            _require(step in {"council.instance_plan_business_repair", "council.blueprint_feasibility"},
                     "unexpected paid step during approved business checkpoint")
            call = {"step": step, "messages_sha256": _digest(messages), "status": "request_pending"}
            record["business_call_history"].append(call); save()
            try:
                response = tracer.chat_json(step, messages, *args, **kwargs)
            except BaseException as error:
                call.update(status="call_failed", error_type=type(error).__name__); save(); raise
            call.update(status="returned", response=deepcopy(response), response_sha256=_digest(response)); save()
            return response
    record["status"] = "business_repair"; save()
    feedback = {"original": {"phase": "initial_business_layout"}, "rejected_plan": deepcopy(candidate),
        "findings": [finding("business_repair", "Continue the two explicitly approved remaining business attempts",
            kind="business", evidence={"diagnostic": record["business_diagnostic"], "prior_business_revisions": old_rows})]}
    try:
        revised = revise(wp, candidate, feedback, RecordedTracer(), record["business_revision"], max_attempts=3,
                         checkpoint=save, uncompiled_diagnostic=record["business_diagnostic"],
                         business_resume_checkpoint={k: deepcopy(policy[k]) for k in (
                             "version", "candidate_sha256", "diagnostic_sha256", "business_revisions_sha256",
                             "completed_business_attempts", "max_business_attempts", "max_additional_physical_requests")})
    except BaseException:
        record["status"] = record["business_revision"].get("status", "business_repair_required"); save(); raise
    rows = record["business_revision"]["business_revisions"]
    _require(rows and rows[-1].get("business_review", {}).get("decision") == "accept",
             "business checkpoint lacks final independent acceptance")
    record["final_business_review"] = deepcopy(rows[-1]["business_review"]); save()
    return revised


def recover_if_permitted(run):
    permit = _permit.get()
    if permit is None:
        return False
    _validate_permit(permit)
    from pipeline.cumulative_entry_guard import require_bounded_run
    require_bounded_run(run.run_id)
    directory = Path(run.dir).resolve()
    # Run uses Windows extended paths while the receipt uses a normal resolved
    # path. Bind the actual directory identity, not those two spellings.
    _require(run.run_id == permit["run_id"] and directory.samefile(Path(permit["directory"])),
             "different active run")
    original = permit["original"]
    continuing = permit["version"] == CHECKPOINT_VERSION
    _require(_implementation_hashes() == permit["source_migration"]["target_source_hashes"],
             "implementation changed after source migration validation")
    _require(run.manifest.get("run_id") == RUN_ID
             and run.manifest.get("config") == original["manifest"].get("config")
             and run.manifest.get("stages") == original["manifest"].get("stages"), "active production identity changed")
    for name in (set(permit["evidence"]) - {"manifest.json", "experiment_profile.json",
                                            "llm_attempts.jsonl", "prompts.jsonl"}):
        _require(_sha((directory / name).read_bytes()) == permit["evidence"][name]["sha256"], "historical artifact changed: " + name)
    for name in ("llm_attempts.jsonl", "prompts.jsonl"):
        body = (directory / name).read_bytes()
        anchor = permit["evidence"][name]
        _require(len(body) == anchor["bytes"] and _sha(body) == anchor["sha256"], "physical trace changed before first repair")
    live = run.read("experiment_profile.json")
    keys = ("model", "max_calls", "max_cny", "admitted_calls", "settled_usage_calls",
            "unsettled_reservation_calls", "budget_consumed_cny", "reported_usage_estimate_cny",
            "reserved_upper_estimate_cny", "transport", "max_output_tokens", "json_attempts")
    _require(all(live.get(k) == original["profile"].get(k) for k in keys)
             and not live.get("stopped") and not live.get("source_drift"), "cumulative ledger or execution settings changed")
    # An exclusive file survives crashes. No second process can spend this permit.
    with (directory / permit.get("repair_claim", CLAIM)).open("x", encoding="utf-8") as claim:
        json.dump({"version": permit["version"], "request_sha256": permit["receipt_sha256"],
                   "run_id": RUN_ID, "epoch": permit.get("epoch", 1),
                   **({"retry_policy_sha256": permit["retry_policy_sha256"]} if permit.get("retry_policy") else {}),
                   **({"escalation_sha256": permit["escalation_sha256"]} if permit.get("plan_escalation") else {}),
                   **({"business_resume_checkpoint_sha256": permit["business_resume_checkpoint_sha256"]} if permit.get("business_resume_checkpoint") else {})}, claim, sort_keys=True)
    if continuing:
        # Preserve the exact previous checkpoint bytes as well as the cumulative
        # audit. Claims and previous epochs remain immutable evidence.
        with (directory / f"01_initial_plan_recovery_snapshot_epoch_{permit['epoch'] - 1:04d}.json").open("xb") as saved:
            saved.write((directory / ARTIFACT).read_bytes())
    record = {"version": permit["version"], "status": "repairing", "request_sha256": permit["receipt_sha256"],
              "epoch": permit.get("epoch", 1),
              "source_migration": deepcopy(permit["source_migration"]),
              "original_audit_sha256": permit["evidence"]["01_instance_plan_audit.json"]["sha256"],
              "transaction": deepcopy(original["record"]["transaction"]) if continuing else {},
              "business_review": None, "paid_call_budget_reset": False,
              "logical_plan_attempt": 2, "max_logical_plan_attempts": 3, "business_revision_remaining": 1}
    if continuing:
        record.update(parent_request_sha256=permit["parent_request_sha256"],
                      parent_checkpoint_sha256=permit["parent_checkpoint_sha256"],
                      prior_physical_calls=permit["prior_physical_calls"],
                      prior_estimated_cny=permit["prior_estimated_cny"],
                      joint_batch_policy=permit["joint_batch_policy"],
                      joint_drafts_consumed_at_entry=permit["joint_drafts_consumed"],
                      remaining_unit_author_slots_at_entry=deepcopy(permit["remaining_unit_author_slots"]),
                      maximum_additional_joint_batches=permit["maximum_additional_joint_batches"])
    if permit.get("retry_policy"):
        record.update(plan_retry_policy=permit["plan_retry_policy"], retry_policy=deepcopy(permit["retry_policy"]),
                      retry_policy_sha256=permit["retry_policy_sha256"])
    if permit.get("plan_escalation"):
        record.update(plan_escalation=permit["plan_escalation"], escalation=deepcopy(permit["escalation"]),
                      escalation_sha256=permit["escalation_sha256"])
    if permit.get("business_resume_checkpoint"):
        record.update(business_resume_checkpoint=deepcopy(permit["business_resume_checkpoint"]),
            business_resume_checkpoint_sha256=permit["business_resume_checkpoint_sha256"],
            business_diagnostic=deepcopy(original["record"]["business_diagnostic"]),
            business_review=deepcopy(original["record"]["business_review"]),
            business_revision=deepcopy(original["record"]["business_revision"]),
            business_call_history=deepcopy(original["record"]["business_call_history"]),
            logical_plan_attempt=3, max_logical_plan_attempts=5, business_revision_remaining=2)
    audit = deepcopy(original["audit"])
    prior_history = deepcopy(audit.get("recovery_history", []))
    def checkpoint(*_args, **_kwargs):
        audit["recovery_history"] = prior_history + [deepcopy(record)]
        run.write("01_instance_plan_audit.json", audit)
        run.write(ARTIFACT, record)
    checkpoint()
    from pipeline.plan_transactions import repair
    from pipeline.instance_plan import compile_plan, PlanConflict, finding
    from pipeline.blueprint_feasibility import assess
    wp = _prepared_whitepaper(original["whitepaper"])
    class FirstStepTracer:
        def __init__(self): self.first = True
        def chat_json(self, step, *args, **kwargs):
            if self.first:
                _require(step == permit["first_new_step"], "unexpected first paid repair step")
                self.first = False
            return run.tracer.chat_json(step, *args, **kwargs)
    try:
        repair_tracer = FirstStepTracer()
        if permit.get("plan_escalation"):
            revised = (_run_business_checkpoint(wp, permit, record, repair_tracer, checkpoint)
                       if permit.get("business_resume_checkpoint") else
                       _run_business_escalation(wp, permit, record, repair_tracer, checkpoint))
            _require(not repair_tracer.first, "required first business recovery request was never called")
            record.update(status="accepted", after_hash=_digest(revised), after=deepcopy(revised))
            checkpoint()
            run.write("01_whitepaper.json", revised)
            run.write("01_supply_plan.json", revised["supply_plan"])
            run.write("01_instance_plan.json", revised["business_instance_plan"])
            return True
        candidate = repair(wp, deepcopy(original["candidate"]), {"findings": deepcopy(original["findings"])},
                           repair_tracer, record["transaction"], checkpoint=checkpoint,
                           **({"resume_checkpoint": True} if continuing else {}),
                           **({"retry_policy": deepcopy(permit["retry_policy"])} if permit.get("retry_policy") else {}))
        _require(not repair_tracer.first, "required original unit author was never called")
        revised = compile_plan(wp, candidate)
        record["candidate_sha256"] = _digest(candidate)
        record["status"] = "business_review"
        checkpoint()
        review = assess(revised, run.tracer, feedback={"business_instance_plan": revised["business_instance_plan"],
                        "original_request": {"phase": "initial_business_layout", "recovery_request_sha256": permit["receipt_sha256"]}})
        record["business_review"] = deepcopy(review)
        if review.get("decision") == "repair":
            # Normal create() would now advance from logical attempt 2 to 3 and
            # call revise(max_attempts=1). Continue that exact remaining slot.
            from pipeline.layout_revision import revise
            record.update(status="business_repair", logical_plan_attempt=3,
                          business_revision_remaining=0, business_revision={})
            checkpoint()
            class BusinessCheckpointTracer:
                def chat_json(self, step, messages, *args, **kwargs):
                    entry = {"step": step, "messages_sha256": _digest(messages), "status": "request_pending"}
                    record.setdefault("business_call_history", []).append(entry)
                    checkpoint()
                    try:
                        result = run.tracer.chat_json(step, messages, *args, **kwargs)
                    except BaseException as error:
                        entry.update(status="call_failed", error_type=type(error).__name__)
                        checkpoint()
                        raise
                    entry.update(status="returned", response=deepcopy(result))
                    checkpoint()
                    return result
            business_feedback = {"original": {"phase": "initial_business_layout"},
                "rejected_plan": deepcopy(candidate), "findings": [finding("business_repair",
                    "Original business review requests revision", kind="business", evidence=review)]}
            try:
                revised = revise(wp, candidate, business_feedback, BusinessCheckpointTracer(),
                                 record["business_revision"], max_attempts=1, checkpoint=checkpoint)
            except BaseException:
                record["status"] = record["business_revision"].get("status", "business_repair_required")
                checkpoint()
                raise
            record["final_business_review"] = deepcopy(record["business_revision"]["business_revisions"][-1]["business_review"])
        elif review.get("decision") != "accept":
            record["status"] = "design_unresolved"
            checkpoint()
            raise PlanConflict([finding("business_unresolved", "Original business review did not accept recovered plan",
                                        kind="design", evidence=review)])
        record.update(status="accepted", after_hash=_digest(revised), after=deepcopy(revised))
        checkpoint()
        run.write("01_whitepaper.json", revised)
        run.write("01_supply_plan.json", revised["supply_plan"])
        run.write("01_instance_plan.json", revised["business_instance_plan"])
        return True
    except BaseException as error:
        if record["status"] not in {"business_repair_required", "design_unresolved", "accepted"}:
            record.update(status="repair_failed", error_type=type(error).__name__,
                          findings=deepcopy(getattr(error, "findings", [])))
        checkpoint()
        raise
