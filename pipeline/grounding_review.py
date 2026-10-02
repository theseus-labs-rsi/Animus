"""Public semantic review inside the original questions/corpus merge stage.

This adapts the existing readers; it never creates a world, question or gold.
Legacy lexical grounding is retained as a diagnostic, not a semantic veto.
"""
from collections import Counter
from copy import deepcopy
import threading

from pipeline.grounding import gather_evidence, _sessions_of, run_grounding
from pipeline.semantic_review import (SemanticReviewAuditError, review_questions,
                                      review_certification)

VERSION = "original-grounding-semantic/v2"
REVIEW_ARTIFACT = "06_semantic_review.json"


def _local_execution_failure(raw):
    if not isinstance(raw, dict) or "__error__" not in raw:
        return False
    kind = (raw.get("__error_metadata__") or {}).get("kind")
    return kind in {"json_syntax", "output_truncated", "empty_completion", "http_connect_timeout",
        "http_pool_timeout", "http_connect_error", "http_read_error", "http_write_error",
        "http_protocol_error", "http_read_timeout", "http_write_timeout", "total_deadline",
        "http_status_500", "http_status_502", "http_status_503", "http_status_504",
        "paged_invalid_action", "paged_allowance_exhausted"}


def _global_execution_failure(exc):
    """Return true only when later provider calls must not be admitted.

    Question-bound transport, JSON, paging, context and model-output failures
    remain local to the current review stage.  The outer bounded runner already
    owns the hard call/cost fuse; authentication failures cannot recover on a
    later question either.  Keeping this list narrow prevents one malformed
    candidate response from suppressing the rest of a large review.
    """
    from execution_control import global_failure
    return global_failure(exc)


def enabled(wp):
    return (wp.get("quality_contract") or {}).get("public_semantic_review") is True


def execution_complete(review):
    """Require actual complete stage outputs, independently of semantic verdicts."""
    if not isinstance(review, dict) or review.get("audit_error"):
        return False
    items = review.get("items")
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        stages = item.get("stage_execution")
        required = {"blind_read", "adjudicate"}
        if (item.get("binding") or review.get("binding") or {}).get("reference_auditor_model"):
            required.add("reference_audit")
        if not isinstance(stages, dict) or not required <= stages.keys():
            return False
        if any(not isinstance(value, dict) or value.get("status") != "ok" for value in stages.values()):
            return False
    return True


def candidates_with_evidence(questions, corpus, *, isolated_reference=False):
    # Scan the complete public history, then give the semantic roles every
    # document that explicitly mentions the questioned canonical entity.  The
    # previous union also included *all* documents in every declared session;
    # in recurring-entity worlds that inflated one question to hundreds of
    # mostly unrelated documents and made the paging controller dominate the
    # review.  Exact entity retrieval is transport only: filler/conflict rows
    # remain visible and the three LLM roles still decide aliases, meaning,
    # support, absence and validity.  A machine receipt makes the full-corpus
    # scan explicit without claiming that an exact-string scan proves absence.
    sessions = _sessions_of(corpus)
    public = [(session.get("session_id"), doc) for session in sessions
              for doc in session.get("docs", []) if doc.get("doc_id")]
    rows = []
    from eval.grading import requires_semantic_grading
    from pipeline.reference_audit import validate_reference_proposal
    for question in questions:
        declared = set(question.get("evidence_sessions") or [])
        entity = str(question.get("entity") or "").strip()
        if entity:
            scoped = [(sid, doc) for sid, doc in public
                      if entity in str(doc.get("content") or "")]
            scope_kind = "all_canonical_entity_mentions/v2"
        elif declared:
            scoped = [(sid, doc) for sid, doc in public if sid in declared]
            scope_kind = "declared_sessions_without_entity/v2"
        else:
            scoped = list(public)
            scope_kind = "complete_public_corpus_without_entity/v2"
        # Candidate evidence remains signal-only.  The semantic readers also
        # receive filler/distractor documents inside the same bounded scope so
        # they can detect ambiguity and counterevidence.
        pool = list(dict.fromkeys(str(doc["doc_id"]) for _, doc in scoped
                                  if doc.get("is_filler") is not True))
        semantic_pool = list(dict.fromkeys(str(doc["doc_id"]) for _, doc in scoped))
        scope_fallback = False
        if not pool or not semantic_pool:
            # A declared session can contain only distractors after an upstream
            # partial render.  Expanding this one question is safer than either
            # crashing the batch or letting an empty scope pass as complete.
            scope_fallback = True
            scope_kind = "complete_public_corpus_fallback/v2"
            pool = list(dict.fromkeys(str(doc["doc_id"]) for _, doc in public
                                      if doc.get("is_filler") is not True))
            semantic_pool = list(dict.fromkeys(str(doc["doc_id"]) for _, doc in public))
        if not pool or not semantic_pool:
            raise ValueError("Public corpus contains no signal documents")
        if requires_semantic_grading(question):
            # Never manufacture a natural reference from the canonical witness.
            validate_reference_proposal(question.get("reference_proposal"))
        rows.append({**deepcopy(question), "evidence_doc_ids": pool,
                     "candidate_evidence_doc_ids": pool,
                     "semantic_scope_doc_ids": semantic_pool,
                     "semantic_scope_receipt": {
                         "version": "full-public-exact-entity-scan/v1",
                         "selection_kind": scope_kind,
                         "canonical_entity": entity or None,
                         "scanned_fields": ["content"],
                         "full_public_document_count": len(public),
                         "matched_document_count": len(semantic_pool),
                         "declared_sessions": sorted(declared, key=str),
                         "semantic_limit": "Exact canonical-name retrieval does not prove semantic absence or exclude aliases."},
                     "evidence_scope": {"kind": scope_kind,
                                        "declared_sessions": sorted(declared, key=str),
                                        "entity_expansion": bool(entity),
                                        "semantic_document_count": len(semantic_pool),
                                        "full_public_document_count": len(public),
                                        "full_corpus_fallback": scope_fallback,
                                        "minimal_proof": "not_measured",
                                        "necessary_document_count": None, "difficulty_verified": False}})
        if isolated_reference and "reference_proposal" not in rows[-1]:
            # Expose the original reference as an explicit audit target, never
            # substitute the blind reader's answer. Keep the raw gt unchanged.
            answer = deepcopy(question.get("gt"))
            encoding = None
            if (question.get("capability") in {"IE", "MR"}
                    and isinstance(answer, dict) and "value" in answer):
                # These original capabilities have always graded gt.value;
                # at_week is a zero-based internal locator, not an extra answer.
                # Preserve gt verbatim on the row and record the source pointer.
                answer = deepcopy(answer["value"])
                encoding = {"raw_value": deepcopy(question["gt"]), "answer_pointer": "/value",
                            "scope": "Original value-answer wire projection; metadata is not public evidence"}
            if answer == "INSUFFICIENT_EVIDENCE":
                subtype = (question.get("question_contract") or {}).get("abstention_kind")
                answer = {"never_known": "无此项/查无此记录", "forgotten": "已停止统计/不再跟踪",
                          "out_of_scope": "信息不足/不在记录范围内"}.get(subtype, "信息不足，无法确定")
                encoding = {"raw_value": question["gt"], "declared_subtype": subtype,
                            "scope": "Reference encoding only, not public factual evidence"}
            rows[-1]["reference_proposal"] = {"answer": answer, "rationale": ""}
            rows[-1]["reference_projection"] = {"version": "original-gt-audit-target/v2",
                                                "source": "gt", "encoding": encoding}
    return rows


def validate_delivery(questions, corpus, protocol, review):
    """Certify the selected subset's procedure; keep failed candidates pending.

    Completed candidate-bound failures may be isolated. Provider, budget,
    input-limit, audit-sink and missing-stage failures retain the whole-run stop.
    This never edits an opinion, repairs a reference or judges business truth.
    """
    result = {"version": "grounding-candidate-isolation/v1", "delivery_safe": False,
              "execution_complete": execution_complete(review), "source_count": len(questions),
              "selected_count": 0, "pending_count": 0, "parsed_format_failures": [], "issues": []}
    result["isolated_execution_failures"] = []
    try:
        if not isinstance(review, dict) or review.get("audit_error"):
            raise ValueError("Missing review or global audit failure")
        accounting = review.get("execution_accounting") or {}
        if accounting.get("execution_stopped") or accounting.get("suppressed_after_failure", 0):
            raise ValueError("Review provider execution stopped")
        if not questions or len({q.get("qid") for q in questions}) != len(questions):
            raise ValueError("Source candidates must be nonempty with unique identities")
        # Existing evaluator replay covers full input, current references and
        # every completed item's exact outputs/locators/certification.
        validate_current_review(questions, corpus, protocol, review)
        kept, routing = selection(questions, review)
        result.update(selected_count=len(kept), pending_count=routing["n_pending"],
                      rejected_count=routing["n_dropped"])
        items = review["items"]
        if len({item.get("candidate_id") for item in items}) != len(items):
            raise ValueError("Duplicate candidate trace identities")
        for item in items:
            stages = item.get("stage_execution")
            required = {"blind_read", "adjudicate"}
            if (item.get("binding") or review.get("binding") or {}).get("reference_auditor_model"):
                required.add("reference_audit")
            if not isinstance(stages, dict) or set(stages) != required:
                raise ValueError("Missing or unknown review stages")
            for stage, execution in stages.items():
                status = execution.get("status") if isinstance(execution, dict) else None
                if status == "ok":
                    continue
                allowed = "invalid_output" if stage == "reference_audit" else "invalid_review"
                raw = ((item.get("reference_audit") or {}).get("raw_output") if stage == "reference_audit"
                       else item.get(stage + "_raw_output"))
                # A bounded worker records one started/finished trace per
                # candidate stage.  The complete-trace check below proves that
                # the failure belongs to this candidate, even when an internal
                # exception happened before a raw model response was available.
                # The candidate remains pending and can never enter ``kept``.
                local = (status == "model_error"
                         and accounting.get("unit_failure_isolation") == "bounded-units/v1")
                if status != allowed and not local:
                    raise ValueError(f"Non-isolatable execution failure: {stage}:{status}")
                if not local and (not isinstance(raw, (dict, list))
                        or isinstance(raw, dict) and "__error__" in raw):
                    raise ValueError("Format failure lacks an actual parsed raw opinion")
                if item.get("review_state") != "pending" or review_certification(item)["status"] == "certified":
                    raise ValueError("Incomplete candidate must remain pending and uncertified")
                result["isolated_execution_failures" if local else "parsed_format_failures"].append({"qid": item["source_qid"], "stage": stage,
                                                        "execution": deepcopy(execution)})
        if result["parsed_format_failures"] or result["isolated_execution_failures"]:
            # A partial delivery must retain the complete original attempt log,
            # including the malformed replies, rather than drop failed rows.
            records = review.get("records")
            if not isinstance(records, list):
                raise ValueError("Partial review lacks its original trace records")
            for item in items:
                for stage, execution in item["stage_execution"].items():
                    events = [r for r in records if isinstance(r, dict)
                              and r.get("candidate_id") == item["candidate_id"] and r.get("stage") == stage]
                    starts = [r for r in events if r.get("event") == "started"]
                    ends = [r for r in events if r.get("event") == "finished"]
                    if len(events) != 2 or len(starts) != 1 or len(ends) != 1:
                        raise ValueError("Partial review lacks unique started/finished stage records")
                    start, end = starts[0], ends[0]
                    raw = ((item.get("reference_audit") or {}).get("raw_output") if stage == "reference_audit"
                           else item.get(stage + "_raw_output"))
                    if (end.get("output") != raw or end.get("execution") != execution
                            or any(start.get(k) != end.get(k) for k in ("messages", "params", "binding"))
                            or end.get("binding") != item.get("binding")):
                        raise ValueError("Partial review trace differs from its original stage opinion")
        if not kept:
            raise ValueError("No completely certified candidate is available for delivery")
        validate_current_review(kept, corpus, protocol, review)
        result["delivery_safe"] = True
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        result["issues"].append({"code": "semantic_review_execution_not_delivery_safe",
                                 "message": f"{type(exc).__name__}: {exc}"})
    return result


def selection(questions, review):
    """Route recorded LLM decisions, without interpreting free-text reasons."""
    if len(questions) != len(review.get("items", [])):
        raise ValueError("Review must retain every original candidate")
    kept, pending, rejected = [], [], []
    for question, item in zip(questions, review["items"]):
        if item.get("source_qid") != question.get("qid"):
            raise ValueError("Review candidate order/identity mismatch")
        ready = (item.get("review_state") == "completed"
                 and review_certification(item)["status"] == "certified")
        verdict = {k: item.get(k) for k in ("item_validity", "answerability", "reference_status")}
        row = {"qid": question.get("qid"), "line": question.get("line"),
               "capability": question.get("capability"), **verdict,
               "reason": (item.get("adjudication") or {}).get("reasoning"),
               "concerns": (item.get("adjudication") or {}).get("concerns", [])}
        if not ready or any(v in ("ambiguous", "unresolved", None) for v in verdict.values()):
            pending.append(row)
        elif (verdict["item_validity"] == "valid" and verdict["reference_status"] == "supported"
              and verdict["answerability"] in ("answerable", "unanswerable")):
            kept.append(question)
        else:
            rejected.append(row)
    def counts(field):
        total = Counter(q.get(field, "?") for q in questions)
        final = Counter(q.get(field, "?") for q in kept)
        return {k: {"n": n, "grounded": final[k], "survival": round(final[k] / n, 3)}
                for k, n in sorted(total.items())}
    return kept, {"version": VERSION,
                  "overall": {"n": len(questions), "grounded": len(kept),
                              "survival": round(len(kept) / len(questions), 3) if questions else None},
                  "by_line": counts("line"), "by_capability": counts("capability"),
                  "n_dropped": len(rejected), "n_pending": len(pending),
                  "drops": rejected, "pending": pending,
                  "decision_basis": "fallible_public_semantic_review",
                  "limitations": ["Model decisions are not verified truth.",
                                  "Candidate evidence is not necessary evidence or measured difficulty."]}


def _merge_partition_reviews(candidates, reports):
    """Merge independent question shards without changing candidate order.

    Each shard sees the same immutable corpus/protocol and therefore produces
    the same global binding.  Only item/record/call collections are shard-local.
    The merge restores source order before the existing whole-review validator
    replays every binding and audit record.
    """
    if not reports:
        raise ValueError("Parallel semantic review produced no reports")
    first = reports[0]
    common = ("version", "mode", "publication_effect", "input_manifest", "binding",
              "public_protocol", "documents", "source_map")
    for report in reports[1:]:
        if any(report.get(key) != first.get(key) for key in common):
            raise ValueError("Parallel semantic review shards have different bindings")
    by_qid = {}
    records = []
    calls_used = 0
    audit_errors = []
    for report in reports:
        calls_used += int(report.get("calls_used", 0) or 0)
        records.extend(report.get("records") or [])
        if report.get("audit_error"):
            audit_errors.append(deepcopy(report["audit_error"]))
        for item in report.get("items") or []:
            qid = item.get("source_qid")
            if not qid or qid in by_qid:
                raise ValueError("Parallel semantic review has missing/duplicate source identity")
            by_qid[qid] = item
    order = [row.get("qid") for row in candidates]
    if len(set(order)) != len(order) or set(order) != set(by_qid):
        raise ValueError("Parallel semantic review does not cover the original candidates")
    first["items"] = [by_qid[qid] for qid in order]
    first["records"] = records
    first["calls_used"] = calls_used
    first["summary"] = {"candidates": len(first["items"]),
        "completed": sum(item.get("review_state") == "completed" for item in first["items"]),
        "pending": sum(item.get("review_state") == "pending" for item in first["items"])}
    if audit_errors:
        first["audit_error"] = {"error_type": "ParallelSemanticReviewAuditError",
                                "shard_errors": audit_errors}
    return first


def review_grounding(questions, corpus, protocol, *, chat_json, model,
                     max_calls=None, max_tokens=4096, max_input_chars=200000, record=None,
                     isolated_reference=False, checkpoint_dir=None, workers=1, cache_namespace=None):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Semantic review workers must be an integer from 1 to 16")
    candidates = candidates_with_evidence(questions, corpus, isolated_reference=isolated_reference)
    _, lexical = run_grounding(questions, corpus)
    stopped = False
    caller_invocations = 0
    suppressed_after_failure = 0
    from pathlib import Path
    import json
    from pipeline.semantic_review import fingerprint
    from pipeline.run import _atomic_write_json
    from pipeline import paged_read
    cache = Path(checkpoint_dir) if checkpoint_dir else None
    active = threading.local()
    state_lock = threading.Lock()
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        # This is the logical full-input envelope. Every actual paged request
        # remains below paged_read.LIMIT and retains all public documents.
        max_input_chars = max(max_input_chars, len(json.dumps(corpus, ensure_ascii=False)) * 3 + 100000)
    paging_contract = Path(paged_read.__file__).read_text(encoding="utf-8")
    def cache_key(messages, params, candidate, contract=None):
        payload = {"messages": messages, "params": params,
                   "candidate_id": candidate, "paging": paging_contract if contract is None else contract}
        if cache_namespace is not None:
            payload["public_input"] = cache_namespace
        return fingerprint(payload)
    def record_progress(event):
        active.candidate = event.get("candidate_id")
        if record:
            record(deepcopy(event))
        if cache and event.get("event") == "finished" and event.get("execution", {}).get("status") == "ok":
            key = cache_key(event["messages"], event["params"], event.get("candidate_id"))
            value = {"key": key, "output": deepcopy(event["output"])}
            value["hash"] = fingerprint(value)
            _atomic_write_json(cache / (key + ".json"), value)

    def invoke(step, messages, **kwargs):
        nonlocal stopped, caller_invocations, suppressed_after_failure
        candidate = getattr(active, "candidate", None)
        with state_lock:
            if stopped:
                suppressed_after_failure += 1
                raise RuntimeError("Previous review execution failed; no further provider calls")
        try:
            if cache:
                from pipeline import paged_read_v2
                contracts = [paging_contract, Path(paged_read_v2.__file__).read_text(encoding="utf-8")]
                for contract in contracts:
                    path = cache / (cache_key(messages, kwargs, candidate, contract) + ".json")
                    if not path.exists():
                        continue
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("key") != path.stem or value.get("hash") != fingerprint({k:v for k,v in value.items() if k != "hash"}):
                        raise ValueError("Semantic review checkpoint changed")
                    if contract != paging_contract:
                        documents = json.loads(messages[-1]["content"]).get("documents", [])
                        paged_read.validate(value["output"], documents)
                    return deepcopy(value["output"])
            with state_lock:
                caller_invocations += 1
            result = (paged_read.call(step, messages, chat_json=chat_json, **{**kwargs, "retries": 3})
                      if cache else chat_json(step, messages, **kwargs))
            if isinstance(result, dict) and "__error__" in result:
                if cache and _local_execution_failure(result):
                    return result
                raise RuntimeError(str(result["__error__"]))
            return result
        except paged_read.PagedReadExecutionError as exc:
            if cache and _local_execution_failure(exc.response):
                return exc.response
            if _global_execution_failure(exc):
                with state_lock:
                    stopped = True
            raise
        except paged_read.PagedReadProtocolError as exc:
            # A model that repeatedly emits an invalid paging action makes only
            # the current question incomplete.  Preserve the exact error and
            # let review_questions continue with the remaining candidates.
            return {"__error__": str(exc),
                    "__error_metadata__": {"kind": exc.kind,
                                             "candidate_id": candidate}}
        except Exception as exc:
            if _global_execution_failure(exc):
                with state_lock:
                    stopped = True
            raise

    try:
        worker_count = min(workers, len(candidates))
        if worker_count == 1:
            review = review_questions(candidates, corpus, protocol, chat_json=invoke,
                                      reviewer_model=model, reader_model=model, max_calls=max_calls,
                                      max_tokens=max_tokens, max_input_chars=max_input_chars, record=record_progress,
                                      reference_auditor_model=model if isolated_reference else None)
        else:
            indexed_candidates = [{**candidate, "_semantic_candidate_id": f"q{index + 1:06d}"}
                                  for index, candidate in enumerate(candidates)]
            shards = [indexed_candidates[index::worker_count] for index in range(worker_count)]
            if max_calls is None:
                shard_budgets = [None] * worker_count
            else:
                quotient, remainder = divmod(max_calls, worker_count)
                shard_budgets = [quotient + int(index < remainder) for index in range(worker_count)]
            def review_shard(index):
                try:
                    return review_questions(shards[index], corpus, protocol, chat_json=invoke,
                        reviewer_model=model, reader_model=model, max_calls=shard_budgets[index],
                        max_tokens=max_tokens, max_input_chars=max_input_chars, record=record_progress,
                        reference_auditor_model=model if isolated_reference else None)
                except SemanticReviewAuditError as exc:
                    report = deepcopy(exc.report)
                    report.setdefault("audit_error", {
                        "error_type": type(exc).__name__, "message": str(exc)})
                    return report
            import config
            reports = config.pmap(review_shard, range(worker_count), workers=worker_count)
            review = _merge_partition_reviews(candidates, reports)
    except SemanticReviewAuditError as exc:
        # The exception deliberately carries the complete in-memory audit.
        # Stop all later provider calls, retain that evidence, and let delivery
        # validation reject the incomplete set instead of aborting the factory.
        review = deepcopy(exc.report)
        stopped = True
        review.setdefault("audit_error", {
            "error_type": type(exc).__name__, "message": str(exc)})
    kept, report = selection(candidates, review)
    report["lexical_diagnostic"] = lexical
    report["execution_stopped"] = stopped
    report["execution_complete"] = execution_complete(review)
    accounting = {"execution_stopped": stopped, "caller_invocations": caller_invocations,
                  "suppressed_after_failure": suppressed_after_failure,
                  "logical_calls_used": review["calls_used"],
                  "paid_provider_calls": None,
                  "unit_failure_isolation": "bounded-units/v1",
                  "scope": "Forwarded callable invocations; actual provider attempts and charges require the provider trace/bill."}
    report.update(accounting)
    review["execution_accounting"] = deepcopy(accounting)
    delivery = validate_delivery(candidates, corpus, protocol, review)
    report["delivery_safe"] = delivery["delivery_safe"]
    report["delivery_validation"] = delivery
    return kept, report, review


def validate_current_review(questions, corpus, protocol, review):
    """Replay the same version and receipt checks used by the original evaluator.

    No grading is performed and the caller has a zero API allowance.
    Full-review receipts can serve a selected subset without changing identity.
    """
    from eval.semantic_judge import SemanticJudge
    from eval.provenance import protocol_scoring_policy
    scoring_policy = protocol_scoring_policy(protocol)
    def no_call(*args, **kwargs):
        raise AssertionError("Release validation cannot call a provider")
    SemanticJudge(review, questions, corpus, protocol,
                  model=review["binding"]["reviewer_model"], chat_json=no_call, max_calls=0,
                  scoring_policy=scoring_policy)
