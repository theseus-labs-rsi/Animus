"""Audit frozen stage meaning and spent corpus slots before a source migration.

This is an offline, read-only inventory of one already verified historical
copy.  It deliberately leaves semantic revalidation and paid admission closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket


VERSION = "cumulative-stage-retention-audit/v1"
STAGE_FILES = (
    "02_world_review.json", "02_world_review_attempts.json",
    "02_disclosure_ready.json", "02_disclosure_plan_attempts.json",
    "03_raw_orders.json", "03_orders.json", "05_corpus.json",
    "05_corpus_candidate.json", "05_corpus.ckpt.json",
    "05_corpus_warning.json", "05_corpus_token_scale.json",
    "11_production.json",
)


def install_offline_guard():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stage retention audit forbids network calls")
    socket.socket.connect = socket.socket.connect_ex = forbidden
    socket.create_connection = socket.getaddrinfo = forbidden
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["OPENAI_API_KEY"] = "offline-disabled"
    os.environ["MODEL"] = "offline-disabled"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bound_file(run: Path, inherited: dict, name: str):
    row = inherited.get(name)
    path = run / name
    if (not isinstance(row, dict) or not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or sha(path) != row.get("sha256")):
        raise ValueError("Frozen stage artifact is absent or changed: " + name)
    return load(path)


def _orders(raw: list, effective: list) -> dict:
    lines = ("L1_timeline", "L2_relational", "L3_process",
             "L5_conflict", "L6_refusal", "L7_consolidation", "L8_transition")
    if not isinstance(raw, list) or not isinstance(effective, list):
        raise ValueError("Historical order format changed")
    raw_counts = Counter(row.get("line") for row in raw if isinstance(row, dict))
    effective_counts = Counter(row.get("line") for row in effective if isinstance(row, dict))
    raw_ids = [row.get("qid") for row in raw]
    effective_ids = [row.get("qid") for row in effective]
    expected_counts = Counter({line: 6 for line in lines})
    if (len(raw) != 42 or len(effective) != 42
            or raw_counts != expected_counts or effective_counts != expected_counts
            or len(set(raw_ids)) != 42 or len(set(effective_ids)) != 42
            or any(not isinstance(qid, str) or not qid for qid in raw_ids + effective_ids)):
        raise ValueError("Historical seven-line order denominator changed")
    return {"raw": len(raw), "effective": len(effective),
            "per_line_raw": dict(raw_counts), "per_line_effective": dict(effective_counts)}


def _corpus(candidate: dict, checkpoint: dict, warning: dict, token: dict,
            terminal: dict) -> dict:
    sessions = (candidate.get("corpus") or {}).get("sessions")
    checkpoint_sessions = (checkpoint.get("corpus") or {}).get("sessions")
    if not isinstance(sessions, list) or not isinstance(checkpoint_sessions, list):
        raise ValueError("Historical candidate corpus is malformed")
    weeks = [row.get("session_id") for row in sessions]
    docs = sum(len(row.get("docs", [])) for row in sessions)
    terminal_candidate = terminal.get("05_corpus_candidate.json") or {}
    if (candidate.get("done_weeks") != weeks or checkpoint.get("done_weeks") != weeks
            or weeks != terminal_candidate.get("session_ids")
            or docs != terminal_candidate.get("documents")
            or len(weeks) != 4 or len(set(weeks)) != 4 or docs != 207
            or checkpoint_sessions != sessions
            or warning.get("status") != "warning"
            or warning.get("release_eligible") is not False
            or token.get("status") != "incomplete"
            or token.get("attempt_complete") is not False):
        raise ValueError("Administrative corpus success conceals a changed failed candidate")
    return {"candidate_weeks": weeks, "candidate_documents": docs,
            "required_weeks": 6, "release_eligible": False,
            "semantic_warning_sha256": terminal["05_corpus_warning.json"]["sha256"],
            "token_scale_status": token["status"],
            "historical_corpus_stage_is_acceptance": False}


def _slots(run: Path, inherited: dict, prepared: dict, prior: dict) -> dict:
    groups = prepared.get("groups")
    if (not isinstance(groups, list) or not groups
            or prepared.get("public_plan_validated_without_changes") is not True
            or prior.get("version") != "legacy-signal-inheritance-audit/v1"
            or prior.get("provider_calls") != 0
            or prior.get("saved_reply_complete_messages_match_real_requests") is not True):
        raise ValueError("Historical corpus group audit is not bound")
    keys = [row.get("legacy_key") for row in groups]
    if len(set(keys)) != len(keys) or any(not isinstance(k, str) or len(k) != 64 for k in keys):
        raise ValueError("Historical corpus group identities repeat or are malformed")
    slots = replies = completed = 0
    for row in groups:
        name = "05_signal_checkpoints/" + row["legacy_key"] + ".json"
        old = _bound_file(run, inherited, name)
        drafts = (old.get("progress") or {}).get("drafts")
        expected = row.get("original_drafts")
        if (old.get("key") != row["legacy_key"] or not isinstance(drafts, list)
                or not isinstance(expected, list) or len(drafts) != len(expected)
                or row.get("prepared_slot_verified") is not True
                or ("documents" in old) != (row.get("legacy_complete") is True)):
            raise ValueError("Historical author group or slot changed: " + row["legacy_key"])
        for position, (draft, receipt) in enumerate(zip(drafts, expected)):
            has_reply = "author_output" in draft
            if (receipt.get("position") != position or draft.get("admitted") is not True
                    or receipt.get("saved_author_reply") is not has_reply):
                raise ValueError("Historical author position or reply changed")
            replies += int(has_reply)
        slots += len(drafts)
        completed += int(row["legacy_complete"] is True)
    if (len(groups) != prior.get("actual_production_keys_enumerated")
            or slots != prior.get("original_draft_positions")
            or replies != prior.get("saved_author_replies")
            or slots - replies != prior.get("admitted_positions_without_reply")
            or (len(groups), slots, replies, completed) != (119, 131, 130, 117)):
        raise ValueError("Historical corpus denominator or spent attempts changed")
    all_checkpoints = set(path.name for path in (run / "05_signal_checkpoints").glob("*.json"))
    if not {key + ".json" for key in keys} <= all_checkpoints:
        raise ValueError("Historical corpus checkpoint set is incomplete")
    return {"active_groups": len(groups), "completed_groups": completed,
            "incomplete_groups": len(groups) - completed, "admitted_slots": slots,
            "saved_author_replies": replies, "admitted_without_reply": slots - replies,
            "historical_checkpoint_files": len(all_checkpoints),
            "new_author_positions": 0, "attempts_reset": False}


def audit(derived_root: Path, lineage_path: Path, terminal_path: Path,
          prepared_path: Path, prior_slot_path: Path) -> dict:
    install_offline_guard()
    run = derived_root / "run"
    evidence = derived_root / "evidence"
    lineage = load(lineage_path)
    derivation_path = evidence / "derivation_receipt.json"
    derivation = load(derivation_path)
    terminal = load(terminal_path)
    inherited = derivation.get("inherited_run_files") or {}
    if (lineage.get("status") != "historical_lineage_verified"
            or lineage.get("complete_source_migration") is not False
            or lineage.get("runnable_run_created") is not False
            or lineage.get("historical_copy", {}).get("files") != len(inherited)
            or lineage.get("historical_copy", {}).get("parent_manifest_sha256")
               != derivation.get("parent_manifest_sha256")
            or terminal.get("version") != "terminal-corpus-state-audit/v1"
            or terminal.get("provider_calls") != 0):
        raise ValueError("Stage inventory is not attached to the frozen lineage")
    parent = evidence / "parent_manifest.json"
    if sha(parent) != derivation["parent_manifest_sha256"]:
        raise ValueError("Original manifest changed")
    original_manifest = load(parent)
    current_manifest = load(run / "manifest.json")
    if (sha(run / "manifest.json") != lineage["historical_copy"]["reconciled_manifest_sha256"]
            or original_manifest.get("stages") != current_manifest.get("stages")
            or sha(parent) != terminal["artifacts"]["manifest.json"]["sha256"]):
        raise ValueError("Original stage attempts or terminal manifest changed")
    artifacts = {name: _bound_file(run, inherited, name) for name in STAGE_FILES}
    for name, row in terminal["artifacts"].items():
        if row is not None and name != "manifest.json":
            if name not in inherited or inherited[name]["sha256"] != row["sha256"]:
                raise ValueError("Terminal corpus evidence differs from the frozen copy")
    production = artifacts["11_production.json"]
    if (production.get("identity") != lineage["identity"]["production_identity"]
            or production.get("status") != "execution_error"
            or len(production.get("history", [])) != lineage["identity"]["production_history_count"]
            or production.get("round") != lineage["identity"]["production_round"]
            or production.get("supply_round") != lineage["identity"]["supply_round"]
            or production.get("layout_revisions") != lineage["identity"]["layout_revisions"]
            or sha(run / "11_production.json") != lineage["historical_production_sha256"]):
        raise ValueError("Historical production identity or attempts changed")
    review = artifacts["02_world_review.json"]
    review_attempts = artifacts["02_world_review_attempts.json"].get("attempts")
    if (review.get("status") != "passed" or not isinstance(review_attempts, list)
            or len(review_attempts) != lineage["historical_world_review_attempts"]
            or review_attempts[-1] != review):
        raise ValueError("Historical world review or attempts changed")
    disclosure_attempts = artifacts["02_disclosure_plan_attempts.json"].get("attempts")
    if not isinstance(disclosure_attempts, list):
        raise ValueError("Historical disclosure attempts changed")
    orders = _orders(artifacts["03_raw_orders.json"], artifacts["03_orders.json"])
    corpus = _corpus(artifacts["05_corpus_candidate.json"],
        artifacts["05_corpus.ckpt.json"], artifacts["05_corpus_warning.json"],
        artifacts["05_corpus_token_scale.json"], terminal["artifacts"])
    slots = _slots(run, inherited, load(prepared_path), load(prior_slot_path))
    stage_rows = {name: {"status": row.get("status"), "attempt": row.get("attempt")}
                  for name, row in original_manifest["stages"].items()}
    return {"version": VERSION, "status": "historical_stage_inventory_verified",
            "parent_lineage_sha256": sha(lineage_path),
            "derivation_receipt_sha256": sha(derivation_path),
            "terminal_corpus_audit_sha256": sha(terminal_path),
            "prepared_signal_audit_sha256": sha(prepared_path),
            "physical_slot_audit_sha256": sha(prior_slot_path),
            "historical_stage_attempts": stage_rows,
            "historical_artifact_sha256": {name: inherited[name]["sha256"] for name in STAGE_FILES},
            "historical_world_review_attempts": len(review_attempts),
            "historical_disclosure_plan_attempts": len(disclosure_attempts),
            "production": {"identity": production["identity"], "round": production["round"],
                "supply_round": production["supply_round"],
                "layout_revisions": production["layout_revisions"],
                "history_count": len(production["history"]), "status": production["status"]},
            "orders": orders, "corpus": corpus, "author_slots": slots,
            "new_world_opinion": False, "business_truth_certified": False,
            "complete_source_migration": False, "runnable_run_created": False,
            "paid_resume_authorization": False, "provider_calls": 0,
            "historical_run_modified": False}


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--terminal-corpus-audit", type=Path, required=True)
    parser.add_argument("--prepared-signal-audit", type=Path, required=True)
    parser.add_argument("--physical-slot-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.derived_root.resolve()):
        parser.error("Use a new evidence path outside the immutable derived copy")
    result = audit(args.derived_root, args.lineage, args.terminal_corpus_audit,
        args.prepared_signal_audit, args.physical_slot_audit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "orders": result["orders"]["effective"],
        "candidate_documents": result["corpus"]["candidate_documents"],
        "spent_slots": result["author_slots"]["admitted_slots"],
        "complete_source_migration": False, "provider_calls": 0}))


if __name__ == "__main__":
    main()
