"""Bind an incomplete L3 proposal warning to the questions it can affect.

The order stage emits ``03_orders_warning.json`` only when the optional
``L3_process`` proposal agent did not complete.  Retain that warning as
evidence, withhold every question from the affected line, and let unrelated
lines proceed only when an exact, reproducible receipt validates the scope.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


WARNING_ARTIFACT = "03_orders_warning.json"
PROPOSAL_ARTIFACT = "03_process_proposals.json"
RESOLUTION_ARTIFACT = "03_orders_warning_resolution.json"
VERSION = "order-generation-warning-resolution/v1"
AFFECTED_LINE = "L3_process"


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_resolution(whitepaper, world, questions, warning, proposal) -> dict:
    if (not isinstance(warning, dict)
            or warning.get("version") != "order-generation-warning/v1"
            or warning.get("status") != "warning"
            or warning.get("release_eligible") is not False):
        raise ValueError("Unsupported order warning schema")
    if not isinstance(proposal, dict):
        raise ValueError("Order warning lacks its proposal evidence")
    if (not isinstance(questions, list) or not questions
            or any(not isinstance(q, dict) or not isinstance(q.get("qid"), str) or not q["qid"]
                   for q in questions)
            or len({q["qid"] for q in questions}) != len(questions)):
        raise ValueError("Order warning scope requires unique question identities")
    binding = warning.get("binding") or {}
    expected_binding = {
        "whitepaper_hash": canonical_hash(whitepaper),
        "world_hash": canonical_hash(world),
        "proposal_hash": canonical_hash(proposal),
    }
    if binding != expected_binding:
        raise ValueError("Order warning does not bind the current world and proposal")
    excluded = [q["qid"] for q in questions if q.get("line") == AFFECTED_LINE]
    return {
        "version": VERSION,
        "status": "scoped_exclusion",
        "release_eligible_unaffected": True,
        "affected_lines": [AFFECTED_LINE],
        "excluded_qids": excluded,
        "binding": {
            "warning_hash": canonical_hash(warning),
            "whitepaper_hash": expected_binding["whitepaper_hash"],
            "world_hash": expected_binding["world_hash"],
            "proposal_hash": expected_binding["proposal_hash"],
            "questions_hash": canonical_hash(questions),
        },
        "policy": "retain_warning_and_withhold_every_question_from_affected_line",
    }


def validate_resolution(receipt, whitepaper, world, questions, warning, proposal) -> list[dict]:
    try:
        expected = build_resolution(whitepaper, world, questions, warning, proposal)
        if receipt != expected:
            raise ValueError("Order warning resolution differs from its bound inputs")
        return []
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return [{"code": "invalid_order_warning_resolution",
                 "message": f"{type(exc).__name__}: {exc}"}]


def apply_resolution(selected, report, receipt):
    """Withhold affected selected questions and recompute only selection counts."""
    excluded = set(receipt.get("excluded_qids", []))
    kept = [q for q in selected if q.get("qid") not in excluded]
    removed = [q for q in selected if q.get("qid") in excluded]
    report = deepcopy(report)
    overall = report.get("overall") or {}
    overall["grounded"] = len(kept)
    overall["survival"] = round(len(kept) / overall["n"], 3) if overall.get("n") else None
    report["overall"] = overall
    for field, group_name in (("line", "by_line"), ("capability", "by_capability")):
        counts = {}
        for question in kept:
            key = question.get(field, "?")
            counts[key] = counts.get(key, 0) + 1
        for key, row in (report.get(group_name) or {}).items():
            if isinstance(row, dict):
                row["grounded"] = counts.get(key, 0)
                row["survival"] = round(row["grounded"] / row["n"], 3) if row.get("n") else None
    report["order_warning_resolution"] = {
        "version": receipt["version"],
        "status": receipt["status"],
        "affected_lines": deepcopy(receipt["affected_lines"]),
        "excluded_qids": deepcopy(receipt["excluded_qids"]),
        "selected_withheld": [q.get("qid") for q in removed],
    }
    report["n_scoped_excluded"] = len(removed)
    report.setdefault("limitations", []).append(
        "Questions from a production line with an incomplete order proposal were withheld from release.")
    return kept, report
