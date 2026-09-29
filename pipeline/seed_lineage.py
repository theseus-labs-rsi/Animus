"""Offline seed uptake diagnostics from the original pipeline's saved records.

This report has no model calls and cannot certify causal reasoning or quality.
Per-question stage-06 decisions remain authoritative for the usable subset.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ARTIFACT = "07_seed_lineage.json"


def seed_lineage_report(directory):
    directory = Path(directory)
    inputs, missing = {}, []

    def read(name, fallback):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            return fallback
        raw = path.read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    pack = read("00_seed_pack.json", {})
    whitepaper = read("01_seed_audit.json", {})
    world = read("02_seed_audit.json", {})
    review = read("02_world_review.json", {})
    candidates = read("04_questions.json", [])
    final = read("06_grounded_questions.json", [])
    routing = read("06_grounding_report.json", {})
    wp_rows = {row["id"]: row for row in whitepaper.get("mechanism_coverage", [])}
    world_rows = {row["mechanism_id"]: row for row in world.get("mechanism_coverage", [])}
    review_rows = {row["mechanism_id"]: row for row in review.get("mechanism_coverage", [])}

    def associated(question, witnesses):
        refs = ((question.get("aux") or {}).get("process") or {}).get("witness_refs") or {}
        return any(set(refs.get(key) or []) & set(ids)
                   for key, ids in witnesses.items())

    rows = []
    for mechanism in pack.get("mechanisms", []):
        identity = mechanism["id"]
        structure = world_rows.get(identity, {})
        witnesses = {key: [value for values in structure.get(group, {}).values() for value in values]
                     for key, group in (("event_ids", "events"), ("relation_ids", "relations"))}
        def selection(questions):
            return [{"index": i, "qid": q.get("qid")} for i, q in enumerate(questions, 1)
                    if associated(q, witnesses)]
        rows.append({"mechanism_id": identity,
                     "whitepaper_structural_retention": wp_rows.get(identity),
                     "world_structural_witnesses": structure or None,
                     "world_business_review": review_rows.get(identity),
                     "candidate_process_associations": selection(candidates),
                     "final_process_associations": selection(final),
                     "public_mechanism_expression": "not_assessed_by_this_report",
                     "mechanism_necessary_to_answer": "not_assessed_by_this_report"})
    return {"version": "seed-lineage/v1", "seed_id": pack.get("seed_id"),
            "schema_version": pack.get("schema_version"), "input_sha256": inputs,
            "missing_artifacts": missing,
            "extraction": {"declared_source_count": len(pack.get("sources", [])),
                "source_kinds": dict(Counter(s.get("source_kind", "unspecified") for s in pack.get("sources", []))),
                "mapped_segment_count": sum(len(item.get("segments", [])) for item in pack.get("document_map", [])),
                "declared_blocking_issues": (pack.get("review") or {}).get("blocking_issues", [])},
            "questions": {"candidate_count": len(candidates), "final_count": len(final),
                "candidates_by_line": dict(Counter(q.get("line") for q in candidates)),
                "final_by_line": dict(Counter(q.get("line") for q in final)),
                "routing_counts": {key: routing[key] for key in ("n_in", "n_out", "n_pending", "n_dropped") if key in routing}},
            "mechanisms": rows,
            "limitations": [
                "Extraction counts describe declared metadata; they do not measure complete source reading.",
                "Recorded world review is displayed without independent semantic reassessment.",
                "Process associations use exact event/relation references and may overlap across mechanisms.",
                "Missing association does not establish absence of a mechanism in other question types.",
                "Reference overlap does not establish public expression or necessity for answering.",
                "This diagnostic does not grant release eligibility."]}
