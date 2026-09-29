"""Independent, fail-closed final reading of the complete delivered materials.

This adds an acceptance receipt; it never edits grounding, release or scores.
All three original semantic roles read every public document for every released
question. Paging proves exact delivery, while the existing semantic validators
and model opinions retain responsibility for understanding and correctness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pipeline.benchmark_export import public_view
from pipeline.cumulative_entry_guard import check_before_production, require_bounded_run
from pipeline.grounding_review import (candidates_with_evidence, execution_complete,
    selection, validate_current_review)
from pipeline import paged_read
from pipeline.quality import quality_snapshot
from pipeline.run import _atomic_write_json, _io_path
from pipeline.semantic_review import (SemanticReviewAuditError, fingerprint,
    review_questions)

VERSION = "full-material-acceptance/v1"
ARTIFACT = "12_full_material_acceptance.json"
DIRECTORY = "full_material_acceptance"
ROLES = ("blind_read", "reference_audit", "adjudicate")
INPUTS = ("00_about.json", "01_whitepaper.json", "02_world.json", "03_orders.json",
          "04_questions.json", "05_corpus.json", "06_grounded_questions.json",
          "06_grounding_report.json", "06_semantic_review.json", "07_release.json")
SEVEN_LINES = ("L1_timeline", "L2_relational", "L3_process", "L5_conflict",
               "L6_refusal", "L7_consolidation", "L8_transition")
SHOWCASE_SEED_DIGEST = "ab4b3c049b264cf9380fd56ef4894f2f2136b8d006ab8c2b2b8efd8bb71f68f3"


def _offline_token_measurement(corpus):
    """Load only verified local vocabulary bytes; never download or repair cache."""
    import os
    import tempfile
    from unittest.mock import patch
    import tiktoken.load
    from pipeline.corpus_tokens import CorpusTokenCounter

    def cached_only(blobpath, expected_hash=None):
        directory = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR",
                                    str(Path(tempfile.gettempdir()) / "data-gym-cache")))
        if not directory:
            raise ValueError("Offline token validation requires a prepared vocabulary cache")
        key = hashlib.sha1(blobpath.encode()).hexdigest()
        data = (Path(directory) / key).read_bytes()
        if not expected_hash or _sha(data) != expected_hash:
            raise ValueError("Offline tokenizer vocabulary hash mismatch")
        return data

    with patch.object(tiktoken.load, "read_file_cached", cached_only):
        counter = CorpusTokenCounter("cl100k_base", expected_version="0.12.0")
    return counter.measure(corpus)  # Recount actual bodies; do not trust cached token totals.


def target_checks(run):
    """Read-only acceptance of this authorized seven-line Showcase objective.

    This is separate from the generic quality receipt and selection target.
    Physical cost/authorization and the full reader receipt remain additional
    wrapper gates. No original artifact or ledger is changed here.
    """
    from pipeline.capability_contract import actual_capacity, family_key
    from pipeline.world_state import WorldState
    checks, by_line = {}, []
    cfg = run.manifest.get("config") or {}

    def check(name, passed, **evidence):
        checks[name] = {"passed": bool(passed), **evidence}

    def read(name, fallback):
        return run.read(name) if run.has(name) else deepcopy(fallback)

    def indexed(name):
        rows = read(name, [])
        valid = isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("qid"), str)
                                              and row["qid"] for row in rows)
        values = {row["qid"]: row for row in rows} if valid else {}
        check(name + ":identity", valid and bool(rows) and len(values) == len(rows),
              rows=len(rows) if isinstance(rows, list) else None, distinct_qids=len(values))
        return values

    try:
        wp = read("01_whitepaper.json", {})
        target = cfg.get("delivery_target") or {}
        plan = wp.get("supply_plan") or {}
        check("frozen_target", set(target.get("requested_lines", [])) == set(SEVEN_LINES)
              and all(target.get("per_line_min", {}).get(line, 0) >= 2 for line in SEVEN_LINES)
              and target.get("corpus_tokens", 0) >= 4000
              and target.get("tokenizer") == "cl100k_base@0.12.0"
              and target.get("max_supply_rounds") == 3,
              delivery_target=deepcopy(target), count_stage="generation_quality")
        try:
            from pipeline.seed_run import validate_seed_identity
            from pipeline.seed_pack import seed_digest
            pack = validate_seed_identity(run, wp)
            actual_seed = seed_digest(pack) if pack else None
            check("same_seed", actual_seed == SHOWCASE_SEED_DIGEST, actual_digest=actual_seed,
                  expected_digest=SHOWCASE_SEED_DIGEST)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check("same_seed", False, error=f"{type(exc).__name__}: {exc}")
        world = read("02_world.json", {})
        entities = world.get("entities") or {}
        check("world_shape", isinstance(entities, dict) and 24 <= len(entities) <= 32
              and type(world.get("n_sessions")) is int and world["n_sessions"] == 6,
              n_sessions=world.get("n_sessions"), entity_count=len(entities),
              expected_sessions=6, entity_range=[24, 32])
        corpus = read("05_corpus.json", {})
        sessions = corpus.get("corpus", corpus).get("sessions", [])
        session_ids = [s.get("session_id") for s in sessions]
        check("all_six_periods_published", len(sessions) == 6
              and all(type(i) is int for i in session_ids) and set(session_ids) == set(range(6))
              and all(isinstance(s.get("docs"), list) and s["docs"] for s in sessions),
              session_ids=session_ids)
        state = read("11_production.json", {})
        supply_round = state.get("supply_round", state.get("round"))
        check("supply_rounds", type(supply_round) is int and 1 <= supply_round <= 3,
              supply_round=supply_round, maximum=3,
              controller_round=state.get("round"), layout_revisions=state.get("layout_revisions"))
        check("production_state", state.get("status") == "generation_ready_awaiting_selection",
              status=state.get("status"))
        fulfillment = read("02_instance_fulfillment.json", {})
        check("independent_supply_policy", fulfillment.get("verification_policy") == "actual-independent-supply/v1"
              and bool(wp.get("business_instance_plan")) and bool(plan.get("instance_policy")),
              verification_policy=fulfillment.get("verification_policy"))
        raw, effective, formed = (indexed(name) for name in (
            "03_raw_orders.json", "03_orders.json", "04_questions.json"))
        quality = quality_snapshot(run.dir)
        partition = quality.get("checks", {}).get("partition", {}).get("qids", {})
        check("quality_eligible", quality.get("eligible"), status=quality.get("status"),
              partition=deepcopy(partition))
        released = set(partition.get("released", [])) if quality.get("eligible") else set()
        certificates, capacity_error = {}, None
        try:
            capacity = actual_capacity(wp, WorldState.from_dict(world))
            certificates = {(row["line"], row["family_key"]): row for row in capacity["certificates"]
                            if row.get("passed") is True}
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            capacity_error = f"{type(exc).__name__}: {exc}"
        check("current_capacity", bool(certificates) and not capacity_error,
              certified_families=len(certificates), error=capacity_error)
        ancestry_errors = []
        for child, parent, label in ((effective, raw, "effective"), (formed, effective, "formed")):
            for qid, row in child.items():
                if qid not in parent or family_key(row) != family_key(parent[qid]) or row.get("gt") != parent[qid].get("gt"):
                    ancestry_errors.append({"qid": qid, "stage": label})
        check("original_candidate_ancestry", not ancestry_errors and released <= set(formed),
              mismatches=ancestry_errors, unknown_released=sorted(released - set(formed)))
        for line in SEVEN_LINES:
            pools = {label: {qid: row for qid, row in values.items() if row.get("line") == line}
                     for label, values in (("raw", raw), ("effective", effective), ("formed", formed),
                                           ("released", {qid: formed[qid] for qid in released if qid in formed}))}
            row = {"line": line}
            for label, values in pools.items():
                families = sorted({family_key(value) for value in values.values()})
                valid = [key for key in families if (line, key) in certificates]
                row[label] = {"count": len(values), "qids": sorted(values), "independent_count": len(families),
                              "family_keys": families, "certified_count": len(valid)}
            row["quality_partition"] = {status: sorted(set(qids) & set(pools["formed"]))
                                         for status, qids in partition.items()}
            row["passed"] = (row["raw"]["independent_count"] >= 6
                             and row["effective"]["certified_count"] >= 6
                             and row["released"]["certified_count"] >= 2)
            by_line.append(row)
        check("seven_line_denominators", all(row["passed"] for row in by_line),
              required_raw_independent=6, required_effective_independent=6, required_released_independent=2)
        native = sorted({family_key(effective[qid]) for qid in released if qid in effective
            and effective[qid].get("line") == "L7_consolidation"
            and (effective[qid].get("aux") or {}).get("sub") == "S1_trend"
            and (effective[qid].get("aux") or {}).get("supply_origin") == "native_numeric_trajectory/v1"
            and ("L7_consolidation", family_key(effective[qid])) in certificates})
        check("released_native_l7", len(native) >= 1, required=1, family_keys=native)
        try:
            measurement = _offline_token_measurement(corpus)
            from pipeline.factory import _corpus_is_current
            current = _corpus_is_current(run)
            check("body_tokens", current and measurement["core"]["tokens"] >= 4000,
                  current=current, tokenizer=measurement["tokenizer"],
                  core=measurement["core"], total=measurement["total"],
                  corpus_sha256=measurement["corpus_sha256"], documents=measurement["documents"])
            check("haystack_zero", cfg.get("haystack_ratio") == 0
                  and measurement["filler"]["documents"] == 0, configured_ratio=cfg.get("haystack_ratio"),
                  filler=measurement["filler"])
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, ImportError) as exc:
            check("body_tokens", False, error=f"{type(exc).__name__}: {exc}")
            check("haystack_zero", False, configured_ratio=cfg.get("haystack_ratio"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        check("target_check_execution", False, error=f"{type(exc).__name__}: {exc}")
    selection_files = [name for name in ("08_calibration.json", "09_selection.json", "09_selected_questions.json")
                       if run.has(name)]
    selection_scope = {"required_for_this_experiment": False,
                       "status": "not_run" if not selection_files else "artifacts_present_not_accepted_here",
                       "artifacts": selection_files}
    return {"version": "showcase-seven-line-target/v1", "count_stage": "generation_quality",
            "passed": bool(checks) and all(value["passed"] for value in checks.values()),
            "checks": checks, "by_line": by_line,
            "failed_checks": [key for key, value in checks.items() if not value["passed"]],
            "four_athletes_and_easy_filter": selection_scope,
            "full_material_reader": "separate_required_snapshot",
            "budget_and_zero_call_resume": "separate_required_wrapper_checks"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _implementation():
    root = Path(__file__).resolve().parent
    files = ("full_material_acceptance.py", "semantic_review.py", "reference_audit.py",
             "grounding_review.py", "paged_read.py", "world_context.py", "benchmark_export.py")
    return {name: _sha((root / name).read_bytes()) for name in files}


def _inputs(run):
    quality = quality_snapshot(run.dir)
    if not quality.get("eligible"):
        raise ValueError("Full-material acceptance requires a current eligible release")
    questions = deepcopy(run.read("06_grounded_questions.json"))
    qids = [q.get("qid") for q in questions]
    released = quality["checks"]["partition"]["qids"]["released"]
    if not qids or len(set(qids)) != len(qids) or set(qids) != set(released):
        raise ValueError("Full-material question denominator differs from released qids")
    material, protocol = public_view(run)
    if not material or not protocol.strip():
        raise ValueError("Full-material acceptance requires actual public documents and protocol")
    ids = [doc["doc_id"] for doc in material]
    if len(set(ids)) != len(ids):
        raise ValueError("Full-material public document IDs are not unique")
    corpus = run.read("05_corpus.json")
    for question in questions:
        # Audit the unchanged original answer, never an earlier reader's repair.
        if "gt" in question:
            question.pop("reference_proposal", None)
    candidates = candidates_with_evidence(questions, corpus, isolated_reference=True)
    for question in candidates:
        question.pop("semantic_scope_doc_ids", None)
    documents = [{"doc_id": doc["doc_id"], "utf8_bytes": len(_json_bytes(doc)),
                  "sha256": _sha(_json_bytes(doc)),
                  "content_utf8_bytes": len(doc["content"].encode("utf-8")),
                  "content_sha256": _sha(doc["content"].encode("utf-8"))} for doc in material]
    binding = {"version": VERSION, "run_id": run.run_id,
               "implementation": _implementation(),
               "files": {name: _sha((run.dir / name).read_bytes()) for name in INPUTS if run.has(name)},
               "released_qids": qids, "candidates_hash": fingerprint(candidates),
               "documents": documents, "public_protocol_sha256": _sha(protocol.encode("utf-8"))}
    return candidates, corpus, material, protocol, binding


def _ledger(run):
    profile = run.read("experiment_profile.json")
    keys = ("max_calls", "max_cny", "admitted_calls", "settled_usage_calls",
            "unsettled_reservation_calls", "budget_consumed_cny", "stopped", "execution_status")
    return {key: deepcopy(profile.get(key)) for key in keys}


def _sealed(value):
    value = deepcopy(value)
    value["hash"] = fingerprint(value)
    return value


def _read_sealed(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.pop("hash")
    if fingerprint(value) != expected:
        raise ValueError("Full-material checkpoint hash mismatch")
    return value


def _trace_rows(run, offset=0):
    path = run.dir / "prompts.jsonl"
    rows = {}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            stream.seek(offset)
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("step", "")).startswith("full_material_acceptance."):
                    if row.get("i") in rows:
                        raise ValueError("Duplicate full-material tracer index")
                    rows[row["i"]] = row
    return rows


def _verify_physical(records, traces):
    for record in records:
        if record.get("state") != "returned":
            raise ValueError("Unsettled full-material physical request")
        trace = traces.get(record["tracer_index"])
        if not trace or fingerprint(trace) != record["trace_hash"]:
            raise ValueError("Full-material reply lacks its unchanged original tracer record")
        if (trace.get("step") != record["trace_step"]
                or fingerprint(trace.get("messages")) != record["messages_hash"]
                or trace.get("params") != record["params"]
                or trace.get("output") != record["output"]):
            raise ValueError("Full-material physical input/output binding mismatch")


def _coverage(review, material, physical):
    """Reconstruct role/page coverage from exact original requests and replies."""
    result = []
    all_ids = [doc["doc_id"] for doc in material]
    for item in review.get("items", []):
        if item.get("document_scope", {}).get("visible_doc_ids") != all_ids:
            raise ValueError("Full-material role retained a restricted document scope")
        for role in ROLES:
            events = [row for row in review.get("records", [])
                      if row.get("candidate_id") == item["candidate_id"]
                      and row.get("stage") == role and row.get("event") == "finished"]
            if len(events) != 1:
                raise ValueError("Full-material role lacks its complete original finished record")
            event = events[0]
            raw = ((item.get("reference_audit") or {}).get("raw_output") if role == "reference_audit"
                   else item.get(role + "_raw_output"))
            if raw != event["output"] or event["execution"] != item.get("stage_execution", {}).get(role):
                raise ValueError("Full-material item differs from its original role output")
            if json.loads(event["messages"][-1]["content"]).get("documents") != material:
                raise ValueError("Full-material role input differs from delivered public view")
            output = event["output"]
            page = output.get("_paged_read") if isinstance(output, dict) else None
            if page:
                paged_read.validate(output, material)
                requests = page["transcript"]
            else:
                requests = [{"messages_hash": fingerprint(event["messages"]), "raw_output": output}]
            matched = []
            for request in requests:
                matches = [entry for entry in physical
                           if entry["candidate_id"] == item["candidate_id"] and entry["role"] == role
                           and entry["messages_hash"] == request["messages_hash"]
                           and entry["output"] == request["raw_output"]]
                if len(matches) != 1:
                    raise ValueError("Full-material page lacks unique original physical evidence")
                matched.append(matches[0])
            result.append({"qid": item["source_qid"], "role": role,
                           "document_ids": all_ids, "document_count": len(all_ids),
                           "transport": "exact_public_pages" if page else "complete_direct_input",
                           "page_requests": [{key: entry[key] for key in (
                               "messages_hash", "request_utf8_bytes", "tracer_index", "trace_hash")}
                               for entry in matched],
                           "execution": deepcopy(item.get("stage_execution", {}).get(role)),
                           "delivery_verified": True})
    return result


def _validate(candidates, corpus, material, protocol, review, physical, run):
    _verify_physical(physical, _trace_rows(run))
    coverage = _coverage(review, material, physical)
    if review.get("documents") != material or not execution_complete(review):
        raise ValueError("Full-material semantic execution is incomplete")
    validate_current_review(candidates, corpus, protocol, review)
    kept, routing = selection(candidates, review)
    return coverage, routing, len(kept) == len(candidates) and not routing["n_pending"] and not routing["n_dropped"]


def snapshot(run):
    """Read-only replay; absence, stale inputs and unverified opinions fail closed."""
    base = {"version": VERSION, "status": "not_run", "current": False, "passed": False, "issues": []}
    if not run.has(ARTIFACT):
        return base
    try:
        receipt = _read_sealed(run.dir / ARTIFACT)
        candidates, corpus, material, protocol, binding = _inputs(run)
        if receipt["binding"] != binding:
            return {**base, "status": "stale", "issues": ["Acceptance inputs or implementation changed"]}
        identity = fingerprint({"binding": binding, "settings": receipt["settings"]})
        if receipt["identity"] != identity:
            raise ValueError("Full-material acceptance identity mismatch")
        folder = _io_path(run.dir / DIRECTORY / identity)
        review = _read_sealed(folder / "review.json")
        if fingerprint(review) != receipt["review_hash"]:
            raise ValueError("Full-material review changed")
        physical = [_read_sealed(folder / "calls" / (key + ".json")) for key in receipt["physical_keys"]]
        if fingerprint(physical) != receipt["physical_hash"]:
            raise ValueError("Full-material physical history changed")
        coverage, routing, passed = _validate(candidates, corpus, material, protocol, review, physical, run)
        if coverage != receipt["coverage"] or routing != receipt["routing"] or passed != receipt["passed"]:
            raise ValueError("Full-material acceptance decision differs from replay")
        return {**base, "status": "passed" if passed else "failed", "current": True,
                "passed": passed, "identity": identity, "released_count": len(candidates),
                "document_count": len(material), "coverage": coverage,
                "routing": routing, "ledger_before": receipt["ledger_before"],
                "ledger_after": receipt["ledger_after"]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**base, "status": "failed", "issues": [f"{type(exc).__name__}: {exc}"]}


def run_full_material_acceptance(run, *, model=None, max_calls=None, max_tokens=4096):
    """Run once per bound input/settings; wrapper owns all physical cost/call caps.

    max_calls bounds logical roles, not physical pages. The original bounded
    Tracer wrapper remains the sole physical admission and cumulative ledger.
    Completed negative receipts are retained; retrying them spends zero calls.
    An interrupted request with an unknown reply stops instead of resending.
    """
    require_bounded_run(run.run_id)
    check_before_production(run)
    if not run.has("experiment_profile.json"):
        raise ValueError("Full-material acceptance requires the original cumulative ledger")
    state = run.read("11_production.json")
    if state.get("status") != "generation_ready_awaiting_selection":
        raise ValueError("Full-material acceptance requires generation_ready_awaiting_selection")
    candidates, corpus, material, protocol, binding = _inputs(run)
    model = model or (run.manifest.get("env") or {}).get("reviewer_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Full-material acceptance requires an explicit reader model")
    if max_calls is None:
        max_calls = len(candidates) * 3
    if type(max_calls) is not int or max_calls < 0 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("Invalid full-material logical call/output allowance")
    settings = {"model": model, "max_calls": max_calls, "max_tokens": max_tokens}
    identity = fingerprint({"binding": binding, "settings": settings})
    folder = _io_path(run.dir / DIRECTORY / identity)
    calls = folder / "calls"
    if (folder / "receipt.json").is_file():
        _atomic_write_json(run.dir / ARTIFACT, json.loads((folder / "receipt.json").read_text(encoding="utf-8")))
        return snapshot(run)
    calls.mkdir(parents=True, exist_ok=True)
    begin = folder / "begin.json"
    if not begin.exists():
        _atomic_write_json(begin, _sealed({"binding": binding, "settings": settings,
                                          "ledger_before": _ledger(run)}))
    initial = _read_sealed(begin)
    if initial["binding"] != binding or initial["settings"] != settings:
        raise ValueError("Full-material initial checkpoint changed")
    active, used = {}, {}
    historical_traces = None
    stopped = False

    def record(event):
        active.update(candidate_id=event.get("candidate_id"), role=event.get("stage"))

    def physical_call(step, messages, **params):
        nonlocal stopped, historical_traces
        if stopped:
            raise RuntimeError("Earlier full-material execution failed; no further calls")
        key = fingerprint({"identity": identity, **active, "step": step, "messages": messages, "params": params})
        path = calls / (key + ".json")
        if path.exists():
            entry = _read_sealed(path)
            if historical_traces is None:
                historical_traces = _trace_rows(run)
            _verify_physical([entry], historical_traces)
        else:
            entry = {"key": key, **active, "state": "reserved", "messages_hash": fingerprint(messages),
                     "params": deepcopy(params),
                     "request_utf8_bytes": len(_json_bytes(messages)),
                     "trace_step": "full_material_acceptance." + str(active["candidate_id"]) + "." + step}
            _atomic_write_json(path, _sealed(entry))
            try:
                prompt_path = run.dir / "prompts.jsonl"
                offset = prompt_path.stat().st_size if prompt_path.exists() else 0
                output = run.tracer.chat_json(entry["trace_step"], messages, **params)
                index = run.tracer.n
                trace = _trace_rows(run, offset).get(index)
                entry.update(state="returned", output=deepcopy(output), tracer_index=index,
                             trace_hash=fingerprint(trace))
                _verify_physical([entry], {index: trace})
                _atomic_write_json(path, _sealed(entry))
                if historical_traces is not None:
                    historical_traces[index] = trace
            except BaseException:
                stopped = True
                raise
        used[key] = entry
        if not isinstance(entry["output"], dict) or "__error__" in entry["output"]:
            stopped = True
        return deepcopy(entry["output"])

    def invoke(step, messages, **params):
        nonlocal stopped
        try:
            return paged_read.call(step, messages, chat_json=physical_call, **params)
        except (paged_read.PagedReadExecutionError, paged_read.PagedReadProtocolError) as exc:
            stopped = True
            return (deepcopy(exc.response) if isinstance(exc, paged_read.PagedReadExecutionError)
                    else {"__error__": str(exc), "__error_metadata__": {"kind": exc.kind}})
        except BaseException:
            stopped = True
            raise

    try:
        review = review_questions(candidates, corpus, protocol, chat_json=invoke,
            reviewer_model=model, reader_model=model, reference_auditor_model=model,
            max_calls=max_calls, max_tokens=max_tokens,
            max_input_chars=max(200000, len(json.dumps(corpus, ensure_ascii=False)) * 3 + 100000),
            record=record)
    except SemanticReviewAuditError as exc:
        review = exc.report
    _atomic_write_json(folder / "review.json", _sealed(review))
    physical = list(used.values())
    issues = []
    try:
        coverage, routing, passed = _validate(candidates, corpus, material, protocol, review, physical, run)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        coverage, routing, passed = [], {}, False
        issues.append(f"{type(exc).__name__}: {exc}")
    receipt = _sealed({"version": VERSION, "identity": identity, "binding": binding,
        "settings": settings, "review_hash": fingerprint(review), "physical_keys": list(used),
        "physical_hash": fingerprint(physical), "coverage": coverage, "routing": routing,
        "passed": passed, "issues": issues, "ledger_before": initial["ledger_before"],
        "ledger_after": _ledger(run),
        "limitations": ["Exact delivery does not prove comprehension or model factual correctness.",
                        "This independent receipt does not modify historical grounding or scores."]})
    _atomic_write_json(folder / "receipt.json", receipt)
    _atomic_write_json(run.dir / ARTIFACT, receipt)
    return snapshot(run)
