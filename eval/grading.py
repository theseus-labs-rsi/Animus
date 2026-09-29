"""Dependency-free grading record contract shared by caches and consumers."""
import hashlib
import json
from eval.provenance import (digest, question_hash, reference_hash, valid_context, CONTEXT_FIELDS,
                             make_evaluation_context)
JUDGE_VERSION = "typed-primary-v4"
SEMANTIC_JUDGE_VERSION = "semantic-primary-v6"
ANSWER_TASK_JUDGE_VERSION = "semantic-task-support/v1"
DIRECT_METHOD = "direct/v1"
SEMANTIC_CITATION_POLICY = "explicit-field-or-verbatim/v1"


def requires_semantic_grading(question):
    """Capability routing only; never evidence that a proposed answer is correct."""
    return isinstance(question, dict) and question.get("capability") == "L3_process_trace"


def is_semantic_grade_version(version):
    return version in (SEMANTIC_JUDGE_VERSION, ANSWER_TASK_JUDGE_VERSION)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def answer_task_binding(grade):
    """Versioned identity only; these hashes do not certify a semantic decision."""
    keys = ("question_hash", "reference_hash", "source_qid", "reference_provided", "answer_hash",
            "input_manifest_hash", "review_hash", "documents_hash", "public_context", "public_question",
            "reader_configuration", "grading_execution")
    body = {key: grade.get(key) for key in keys}
    body.update(version=ANSWER_TASK_JUDGE_VERSION,
                scoring_configuration_hash=_fingerprint(grade.get("scoring_configuration")),
                reader_report_hash=_fingerprint(grade.get("answer_task_review")))
    return {**body, "binding_hash": _fingerprint(body)}


def valid_answer_task_grade(grade, record=None, *, require_opinion=True):
    """Recheck public/answer/opinion identity and shape, never reason relevance."""
    if not isinstance(grade, dict) or grade.get("version") != ANSWER_TASK_JUDGE_VERSION:
        return False
    try:
        # Lazy imports keep the legacy record reader independent of the helpers.
        from eval import answer_task_review as reader
        from eval.semantic_judge import resolve_scoring_config
        from pathlib import Path
        cfg = grade.get("scoring_configuration")
        if (not isinstance(cfg, dict) or cfg != resolve_scoring_config(
                cfg.get("scoring_policy"), cfg.get("grading_method"))
                or cfg["scoring_policy"] != reader.POLICY_VERSION):
            return False
        if (not valid_context(grade.get("public_context"))
                or grade.get("answer_task_binding") != answer_task_binding(grade)):
            return False
        if (not isinstance(grade.get("documents_hash"), str) or len(grade["documents_hash"]) != 64
                or any(c not in "0123456789abcdef" for c in grade["documents_hash"])
                or not isinstance(grade.get("judge_model"), str) or not grade["judge_model"].strip()
                or not isinstance(grade.get("public_question"), dict)
                or grade["public_question"].get("qid") != grade.get("source_qid")
                or not isinstance(grade["public_question"].get("question"), str)):
            return False
        if record is not None:
            provenance = record.get("evaluation_provenance")
            if (not valid_context(provenance) or any(provenance[k] != grade["public_context"][k]
                    for k in CONTEXT_FIELDS)):
                return False
            if grade["public_question"] != {"qid": record.get("qid"), "question": record.get("question")}:
                return False
        report = grade.get("answer_task_review")
        reader_cfg = grade.get("reader_configuration")
        if not cfg["reader_enabled"]:
            return report is None and reader_cfg is None
        if report is None and not require_opinion:
            return True  # A bound pre-reader item dispute is still a real dispute.
        if not isinstance(report, dict) or not isinstance(reader_cfg, dict):
            return False
        raw, inputs, identity = report.get("raw_output"), report.get("inputs"), report.get("answer_task_review_binding")
        if (report.get("opinion_ready") is not True or report.get("execution", {}).get("status") != "ok"
                or report.get("calls_used") != 1 or report.get("opinion") != raw
                or not isinstance(inputs, dict) or not isinstance(identity, dict)):
            return False
        docs, answer, protocol = inputs["documents"], inputs["solver_answer"], inputs["public_protocol"]
        if reader.append_scoring_policy(protocol) != protocol:
            return False
        public_question = {"qid": inputs["source_qid"], "question": inputs["question"]}
        if (public_question != grade["public_question"]
                or _fingerprint(answer) != grade["answer_hash"]
                or _fingerprint(docs) != grade["documents_hash"]
                or make_evaluation_context([(int(d.get("session", 0)), d.get("date", ""), d["content"])
                    for d in docs], protocol) != grade["public_context"]):
            return False
        if record is not None and (public_question != {"qid": record.get("qid"), "question": record.get("question")}
                                   or answer != record.get("pred")):
            return False
        validation = reader.validate_answer_opinion(raw, answer, docs)
        if (validation["shape_errors"] or validation["location_errors"]
                or report.get("opinion_validation") != validation):
            return False
        expected = {"version": reader.VERSION,
            "implementation_hash": _fingerprint(Path(reader.__file__).read_text(encoding="utf-8")),
            "prompt_hash": _fingerprint(reader.SYSTEM), "model": grade.get("judge_model"),
            "max_tokens": reader_cfg.get("max_tokens")}
        if (reader_cfg != expected or type(reader_cfg["max_tokens"]) is not int or reader_cfg["max_tokens"] <= 0
                or report.get("budget", {}).get("max_tokens") != reader_cfg["max_tokens"]):
            return False
        expected_identity = {key: expected[key] for key in ("version", "implementation_hash", "prompt_hash", "model")}
        expected_identity.update(policy_version=reader.POLICY_VERSION,
            policy_text_hash=cfg["public_scoring_policy"]["text_hash"],
            question_hash=_fingerprint(public_question), answer_hash=_fingerprint(answer),
            protocol_hash=_fingerprint(protocol), documents_hash=_fingerprint(docs))
        if any(identity.get(k) != v for k, v in expected_identity.items()):
            return False
        if (inputs.get("review_identity") != identity or report.get("public_scoring_policy") != cfg["public_scoring_policy"]
                or report.get("binding", {}).get("input_hash") != digest(inputs)):
            return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        return False


def is_scored(record: dict) -> bool:
    grade = record.get("judgement") or record
    semantic_valid = (isinstance(grade, dict) and is_semantic_grade_version(grade.get("version"))
                      and grade.get("citation_policy") == SEMANTIC_CITATION_POLICY
                      and isinstance(grade.get("evidence"), list)
                      and isinstance(grade.get("resolved_evidence"), list)
                      and len(grade["resolved_evidence"]) == len(grade["evidence"])
                      and grade.get("execution_status") == "ok"
                      and grade.get("requires_item_reassessment") is False
                      and isinstance(grade.get("answer_review"), dict)
                      and grade.get("item_validity") == "valid"
                      and grade.get("answerability") in {"answerable", "unanswerable"}
                      and grade.get("reference_status") in {"supported", "not_provided"}
                      and type(grade.get("reference_provided")) is bool
                      and ((grade["reference_status"] == "not_provided") is (not grade["reference_provided"]))
                      and (grade.get("coverage") or {}).get("status") == "complete"
                      and (grade.get("coverage") or {}).get("scope_conflict") is False
                      and grade.get("coverage_claim_source") == "model_self_report"
                      and isinstance(grade.get("input_manifest_hash"), str)
                      and isinstance(grade.get("answer_hash"), str) and isinstance(grade.get("review_hash"), str))
    if semantic_valid and grade.get("version") == ANSWER_TASK_JUDGE_VERSION:
        semantic_valid = (valid_answer_task_grade(grade, record if "judgement" in record else None)
                          and isinstance(grade.get("answer_task_review_response"), str)
                          and bool(grade["answer_task_review_response"].strip()))
        if semantic_valid:
            cfg = grade["scoring_configuration"]
            expected_steps = (["agent_editing.independent_answer_task_review"] if cfg["reader_enabled"] else [])
            expected_steps.append("semantic_judge.answer")
            semantic_valid = grade.get("grading_execution") == {
                "status": "completed", "calls_used": cfg["calls_per_answer"], "steps": expected_steps}
    if semantic_valid and "judgement" in record:
        answer_hash = hashlib.sha256(json.dumps(record.get("pred"), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        semantic_valid = (grade["answer_hash"] == answer_hash
                          and grade.get("question_hash") == question_hash(record)
                          and grade.get("reference_hash") == reference_hash(record)
                          and "source_qid" in grade and digest(grade["source_qid"]) == digest(record.get("qid"))
                          and grade["reference_provided"] == any(key in record for key in ("reference_proposal", "gold", "gt")))
    return (isinstance(grade, dict) and (grade.get("version") == JUDGE_VERSION or semantic_valid)
            and all(isinstance(grade.get(key), str) and grade[key].strip() for key in ("path", "reason"))
            and grade.get("verdict") in {"correct", "incorrect"}
            and type(grade.get("correct")) is bool
            and grade["correct"] == (grade["verdict"] == "correct")
            and ("judgement" not in record or record.get("correct") is grade["correct"])
            and not record.get("error") and not record.get("judge_error")
            and record.get("execution_status", "ok") == "ok")
