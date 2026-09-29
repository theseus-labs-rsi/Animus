"""Build and validate the lightweight final benchmark receipt.

Question quality is decided in stage 06, one question at a time. This module
does not repeat those semantic decisions and does not veto a sound subset
because another question, a planned quota, or a world-level advisory failed.
Stage 07 only checks that the recorded question partition is structurally
complete and binds the released subset to the artifacts that produced it.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


VERSION = 2
RELEASE_ARTIFACT = "07_release.json"
INPUTS = (
    "00_about.json",
    "02_world.json",
    "04_questions.json",
    "05_corpus.json",
    "06_grounded_questions.json",
    "06_grounding_report.json",
)


class ReleaseError(ValueError):
    pass


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_fingerprint() -> str:
    """Bind receipts only to the code that interprets the final partition."""
    return file_hash(Path(__file__).resolve())


def _read(directory, name):
    return json.loads((Path(directory) / name).read_text(encoding="utf-8"))


def _release_inputs(directory):
    directory = Path(directory)
    names = list(INPUTS)
    for optional in ("06_semantic_review.json", "03_orders_warning_resolution.json"):
        if (directory / optional).is_file():
            names.append(optional)
    return tuple(names)


def _delivery_contract(directory) -> dict:
    """Retain planned yield as metadata; it never rejects surviving questions."""
    path = Path(directory) / "manifest.json"
    manifest = _read(directory, "manifest.json") if path.is_file() else {}
    target = ((manifest.get("algo") or {}).get("targetspec") or {})
    return {
        "min_questions": target.get("min_questions", 0),
        "per_line_min": target.get("per_line_min", {}),
    }


def _qid_rows(value, label, issues):
    if not isinstance(value, list):
        issues.append({"code": "invalid_question_collection", "artifact": label})
        return [], []
    rows, qids = [], []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str) or not row["qid"]:
            issues.append({"code": "missing_question_identity", "artifact": label, "index": index})
            continue
        rows.append(row)
        qids.append(row["qid"])
    duplicates = sorted(qid for qid, count in Counter(qids).items() if count > 1)
    if duplicates:
        issues.append({"code": "duplicate_question_identity", "artifact": label, "qids": duplicates})
    return rows, qids


def _decision_qids(report, key, issues):
    rows = report.get(key, [])
    if not isinstance(rows, list):
        issues.append({"code": "invalid_grounding_decisions", "decision": key})
        return []
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str) or not row["qid"]:
            issues.append({"code": "missing_decision_identity", "decision": key, "index": index})
        else:
            result.append(row["qid"])
    return result


def _partition(source_qids, released_qids, report, issues):
    rejected = _decision_qids(report, "drops", issues)
    pending = _decision_qids(report, "pending", issues)
    scoped = [*(report.get("scoped_excluded") or []),
              *((report.get("order_warning_resolution") or {}).get("selected_withheld") or [])]
    if not isinstance(scoped, list) or any(not isinstance(qid, str) or not qid for qid in scoped):
        issues.append({"code": "invalid_scoped_exclusion"})
        scoped = []

    groups = {
        "released": list(released_qids),
        "rejected": rejected,
        "pending_review": pending,
        "scoped_excluded": list(scoped),
    }
    source_set = set(source_qids)
    membership = Counter(qid for qids in groups.values() for qid in qids)
    overlap = sorted(qid for qid, count in membership.items() if count > 1)
    unknown = sorted(set(membership) - source_set)
    missing = sorted(source_set - set(membership))
    if overlap:
        issues.append({"code": "question_status_overlap", "qids": overlap})
    if unknown:
        issues.append({"code": "question_status_unknown", "qids": unknown})
    if missing:
        issues.append({"code": "question_status_missing", "qids": missing})
    return groups


def evaluate_release(directory) -> dict:
    """Summarize stage-06 decisions and bind the usable subset to its inputs."""
    directory = Path(directory)
    issues, warnings, hashes = [], [], {}
    for name in _release_inputs(directory):
        path = directory / name
        if path.is_file():
            hashes[name] = file_hash(path)
        else:
            issues.append({"code": "missing_release_input", "artifact": name})

    report = {
        "version": VERSION,
        "status": "failed",
        "eligible": False,
        "scope": ["stage_06_question_partition", "artifact_identity"],
        "inputs": hashes,
        "implementation_fingerprint": implementation_fingerprint(),
        "delivery_contract": _delivery_contract(directory),
        "checks": {},
        "issues": issues,
        "warnings": warnings,
        "limitations": [
            "Question decisions inherit the limits of the stage-06 model review.",
            "Planned yield and line coverage are reported as capacity signals only.",
        ],
    }
    if issues:
        return report

    try:
        source, source_qids = _qid_rows(_read(directory, "04_questions.json"), "04_questions.json", issues)
        released, released_qids = _qid_rows(
            _read(directory, "06_grounded_questions.json"), "06_grounded_questions.json", issues)
        grounding_report = _read(directory, "06_grounding_report.json")
        if not isinstance(grounding_report, dict):
            raise ValueError("06_grounding_report.json must be an object")
        partition = _partition(source_qids, released_qids, grounding_report, issues)

        source_by_qid = {row["qid"]: row for row in source}
        for row in released:
            original = source_by_qid.get(row["qid"])
            if original is None:
                continue
            for key in ("question", "gt", "line", "capability", "entity", "field"):
                if row.get(key) != original.get(key):
                    issues.append({"code": "released_question_changed", "qid": row["qid"], "field": key})

        counts = {name: len(qids) for name, qids in partition.items()}
        by_line = Counter(row.get("line", "?") for row in released)
        target = report["delivery_contract"]
        floors = target.get("per_line_min") if isinstance(target.get("per_line_min"), dict) else {}
        minimum = target.get("min_questions") if type(target.get("min_questions")) is int else 0
        missing_by_line = {
            line: floor - by_line[line]
            for line, floor in floors.items()
            if type(floor) is int and floor > by_line[line]
        }
        if len(released) < minimum or missing_by_line:
            warnings.append({
                "code": "delivery_target_unmet",
                "effect": "planning_shortfall_only",
                "actual_questions": len(released),
                "min_questions": minimum,
                "per_line_missing": missing_by_line,
            })
        for status, code in (("pending_review", "questions_pending_review"),
                             ("rejected", "questions_rejected"),
                             ("scoped_excluded", "questions_scoped_excluded")):
            if counts[status]:
                warnings.append({"code": code, "n": counts[status]})

        report["checks"] = {
            "partition": {
                "status": "passed" if not issues else "failed",
                "source_count": len(source_qids),
                "counts": counts,
                "qids": partition,
            },
            "coverage": {
                "final_count": len(released),
                "by_line": dict(sorted(by_line.items())),
                "delivery_target_met": len(released) >= minimum and not missing_by_line,
            },
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        issues.append({"code": "release_summary_error", "message": f"{type(exc).__name__}: {exc}"[:500]})

    released_count = report.get("checks", {}).get("partition", {}).get("counts", {}).get("released", 0)
    report.update(status="passed" if not issues else "failed",
                  eligible=bool(released_count) and not issues)
    return report


def quality_snapshot(run_dir) -> dict:
    directory = Path(run_dir)
    base = {"version": VERSION, "status": "not_run", "eligible": False,
            "scope": [], "checks": {}, "issues": [], "warnings": []}
    path = directory / RELEASE_ARTIFACT
    if not path.is_file():
        return base
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(receipt, dict)
                or receipt.get("version") != VERSION
                or receipt.get("status") not in ("passed", "failed")
                or type(receipt.get("eligible")) is not bool
                or not isinstance(receipt.get("scope"), list)
                or not isinstance(receipt.get("checks"), dict)
                or not isinstance(receipt.get("issues"), list)
                or not isinstance(receipt.get("warnings", []), list)
                or not isinstance(receipt.get("inputs"), dict)):
            raise ValueError("invalid release receipt schema")
        base.update({key: receipt.get(key, base[key])
                     for key in ("scope", "checks", "issues", "warnings")})
        if (receipt.get("implementation_fingerprint") != implementation_fingerprint()
                or set(receipt["inputs"]) != set(_release_inputs(directory))
                or any(not (directory / name).is_file()
                       or file_hash(directory / name) != expected
                       for name, expected in receipt["inputs"].items())):
            return {**base, "status": "stale", "eligible": False}
        valid = receipt["status"] == "passed" and not receipt["issues"]
        return {**base, "status": "passed" if valid else "failed",
                "eligible": valid and receipt["eligible"]}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {**base, "status": "failed", "eligible": False,
                "issues": [{"code": "invalid_release_receipt", "message": str(exc)}]}


def require_release(bench_path, *, allow_unverified=False, corpus_path=None) -> dict:
    bench_path = Path(bench_path).resolve()
    snapshot = quality_snapshot(bench_path.parent)
    reasons = []
    if bench_path.name != "06_grounded_questions.json":
        reasons.append("benchmark is not the receipt-bound released subset")
    if corpus_path is not None and Path(corpus_path).resolve() != bench_path.parent / "05_corpus.json":
        reasons.append("corpus differs from receipt-bound corpus")
    if reasons:
        snapshot = {**snapshot, "eligible": False, "status": "stale",
                    "issues": snapshot["issues"] + [
                        {"code": "release_input_mismatch", "message": reason} for reason in reasons]}
    if snapshot["eligible"]:
        return snapshot
    if allow_unverified:
        return {**snapshot, "override": True, "evaluation_mode": "unverified_research"}
    raise ReleaseError(
        f"Benchmark subset is not eligible: {snapshot['status']}. "
        "Run the quality summary, or use --allow-unverified for research artifacts.")
