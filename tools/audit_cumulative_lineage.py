"""Read-only verification of a frozen run's cumulative history.

This checks the independent derived copy before any source migration or Run
construction. It never imports provider configuration, changes a run, or opens
call admission. A passing lineage audit is necessary, not sufficient, for a
cross-source resume.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import tarfile


def _forbid_network(*_args, **_kwargs):
    raise AssertionError("Cumulative lineage audit forbids network and provider calls")


def install_offline_guard():
    """Close provider and dotenv paths before inspecting any project data."""
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = _forbid_network
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["OPENAI_API_KEY"] = "offline-disabled"
    os.environ["MODEL"] = "offline-disabled"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"


VERSION = "cumulative-lineage-audit/v1"
REQUIRED = {"manifest.json", "experiment_profile.json", "llm_attempts.jsonl",
            "prompts.jsonl", "11_production.json", "02_world_review.json",
            "02_world_review_attempts.json", "05_corpus_candidate.json",
            "05_corpus_warning.json", "experiment_source_hashes.json",
            "experiment_source_hashes_at_end.json"}


def sha_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checked_relative(name: str) -> Path:
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in name or ".." in parts
            or any(":" in part or part.lower().startswith(".env")
                   or "credential" in part.lower() or "secret" in part.lower()
                   for part in parts)):
        raise ValueError("Unsafe inherited run path")
    return Path(*parts)


def _verify_archive(archive_path: Path, expected_sha256: str, run_prefix: str,
                    receipt: dict, parent_manifest: bytes) -> None:
    if (len(expected_sha256) != 64 or receipt.get("archive_sha256") != expected_sha256
            or not run_prefix.endswith("/") or run_prefix.startswith("/")
            or ".." in PurePosixPath(run_prefix).parts):
        raise ValueError("Frozen archive identity or run prefix changed")
    if sha_file(archive_path) != expected_sha256:
        raise ValueError("Frozen archive differs from its external SHA")
    expected = receipt["inherited_run_files"]
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.name.startswith(run_prefix) or member.isdir():
                continue
            relative = member.name[len(run_prefix):]
            _checked_relative(relative)
            if (relative not in expected or relative in seen or not member.isfile()
                    or member.issym() or member.islnk()):
                raise ValueError("Frozen archive contains an unexpected run member")
            seen.add(relative)
            digest = hashlib.sha256()
            size = 0
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
            if (size != member.size or size != expected[relative]["bytes"]
                    or digest.hexdigest() != expected[relative]["sha256"]):
                raise ValueError("Derivation receipt differs from frozen archive: " + relative)
            if relative == "manifest.json" and hashlib.sha256(parent_manifest).hexdigest() != digest.hexdigest():
                raise ValueError("Saved original manifest differs from frozen archive")
    if seen != set(expected):
        raise ValueError("Derivation receipt does not cover the complete frozen run")


def _verify_copy(run_dir: Path, receipt: dict, parent_manifest: bytes) -> dict:
    files = receipt.get("inherited_run_files")
    if (receipt.get("version") != "terminal-state-reconciliation/v1"
            or receipt.get("runnable_run_created") is not False
            or receipt.get("ledger_changed") is not False
            or receipt.get("admission_reopened") is not False
            or not isinstance(files, dict) or not REQUIRED <= set(files)):
        raise ValueError("Incomplete or runnable derivation receipt")
    if (receipt.get("parent_manifest_sha256") != hashlib.sha256(parent_manifest).hexdigest()
            or receipt.get("inherited_run_file_count") != len(files)):
        raise ValueError("Original manifest or inherited file count changed")
    expected = {_checked_relative(name).as_posix() for name in files}
    actual = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Derived copy has missing or extra files")
    total = 0
    for name, entry in files.items():
        path = run_dir / _checked_relative(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Derived copy contains a link or non-file")
        if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int
                or entry["bytes"] < 0 or not isinstance(entry.get("sha256"), str)):
            raise ValueError("Malformed inherited file entry")
        if name != "manifest.json" and (path.stat().st_size != entry["bytes"]
                or sha_file(path) != entry["sha256"]):
            raise ValueError("Derived file differs from frozen archive: " + name)
        total += entry["bytes"]
    if total != receipt.get("inherited_run_bytes"):
        raise ValueError("Inherited byte count changed")
    if (files["manifest.json"]["sha256"] != receipt["parent_manifest_sha256"]
            or sha_file(run_dir / "manifest.json") != receipt.get("reconciled_manifest_sha256")):
        raise ValueError("Original or reconciled manifest binding changed")
    return {"files": len(files), "original_bytes": total,
            "parent_manifest_sha256": receipt["parent_manifest_sha256"],
            "reconciled_manifest_sha256": receipt["reconciled_manifest_sha256"]}


def _verify_manifest(run_dir: Path, original: dict, receipt: dict) -> dict:
    child = read_json(run_dir / "manifest.json")
    historical = deepcopy(child)
    proof = historical.pop("terminal_reconciliation", None)
    historical["status"] = original.get("status")
    historical["current_stage"] = original.get("current_stage")
    if (historical != original or proof is None
            or proof.get("parent_manifest_sha256") != receipt["parent_manifest_sha256"]
            or child.get("status") != "failed" or child.get("current_stage") != ""):
        raise ValueError("Reconciled manifest changed historical identity or stages")
    production = read_json(run_dir / "11_production.json")
    if (not child.get("run_id") or production.get("status") != "execution_error"
            or not production.get("identity") or not isinstance(production.get("history"), list)):
        raise ValueError("Production identity or terminal history missing")
    return {"run_id": child["run_id"], "scenario": child.get("scenario"),
            "production_identity": production["identity"],
            "production_history_count": len(production["history"]),
            "production_round": production.get("round"),
            "supply_round": production.get("supply_round"),
            "layout_revisions": production.get("layout_revisions"),
            "stage_statuses_preserved": True}


def _verify_physical_trace(path: Path, profile: dict) -> tuple[int, Decimal]:
    requests = set()
    responses = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event")
            if event == "request":
                call_id = row.get("call_id")
                if not isinstance(call_id, str) or not call_id or call_id in requests:
                    raise ValueError("Physical trace has a missing or repeated request ID")
                requests.add(call_id)
            elif event == "response":
                call_id = row.get("call_id")
                usage = (row.get("response") or {}).get("usage") or {}
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if (not isinstance(call_id, str) or call_id in responses
                        or type(prompt) is not int or type(completion) is not int
                        or prompt < 0 or completion < 0):
                    raise ValueError("Physical trace has duplicate or unsettled usage")
                responses[call_id] = (prompt, completion)
    if requests != set(responses) or len(requests) != profile["admitted_calls"]:
        raise ValueError("Cumulative physical requests and usage do not balance")
    input_price = Decimal(str(profile["input_cny_per_m"]))
    output_price = Decimal(str(profile["output_cny_per_m"]))
    cost = sum((Decimal(prompt) * input_price + Decimal(completion) * output_price)
               / Decimal(1_000_000) for prompt, completion in responses.values())
    return len(requests), cost


def _verify_budget(run_dir: Path, receipt: dict, baseline: dict) -> dict:
    profile_path = run_dir / "experiment_profile.json"
    trace_path = run_dir / "llm_attempts.jsonl"
    profile = read_json(profile_path)
    physical_calls, physical_cost = _verify_physical_trace(trace_path, profile)
    source = baseline.get("source") or {}
    reconciliation = baseline.get("reconciliation") or {}
    if (baseline.get("version") != "offline-cumulative-cost-baseline/v1"
            or sha_file(profile_path) != receipt.get("profile_sha256")
            or source.get("profile_sha256") != receipt.get("profile_sha256")
            or source.get("trace_sha256") != sha_file(trace_path)
            or profile.get("source_drift") or profile.get("stopped") is not True
            or (profile.get("stop_reason") or {}).get("reason") != "estimated_budget_limit"
            or type(profile.get("admitted_calls")) is not int
            or profile["admitted_calls"] != profile.get("settled_usage_calls")
            or profile.get("unsettled_reservation_calls") != 0
            or profile["admitted_calls"] != reconciliation.get("requests")
            or physical_calls != profile["admitted_calls"]
            or reconciliation.get("responses_with_usage") != profile["admitted_calls"]
            or profile.get("max_calls") != reconciliation.get("max_calls")
            or Decimal(str(profile.get("max_cny"))) != Decimal(reconciliation.get("max_cny", "NaN"))
            or Decimal(str(profile.get("reported_usage_estimate_cny")))
               != Decimal(reconciliation.get("reported_usage_cost_cny", "NaN"))
            or physical_cost != Decimal(reconciliation.get("reported_usage_cost_cny", "NaN"))):
        raise ValueError("Cumulative ledger, trace, or independently measured usage changed")
    return {"admitted_calls": profile["admitted_calls"],
            "settled_usage_calls": profile["settled_usage_calls"],
            "unsettled_reservation_calls": 0,
            "model": profile["model"],
            "profile_sha256": receipt["profile_sha256"],
            "reported_usage_estimate_cny": reconciliation["reported_usage_cost_cny"],
            "physical_trace_usage_cost_cny": str(physical_cost),
            "max_calls": profile["max_calls"], "max_cny": str(profile["max_cny"]),
            "stop_reason": "estimated_budget_limit", "trace_sha256": source["trace_sha256"]}


def audit(run_dir: Path, evidence_dir: Path, baseline_path: Path, *,
          archive_path: Path, archive_sha256: str, run_prefix: str) -> dict:
    """Verify the complete historical copy; make no claim of migration readiness."""
    run_dir = run_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    receipt = read_json(evidence_dir / "derivation_receipt.json")
    parent_bytes = (evidence_dir / "parent_manifest.json").read_bytes()
    _verify_archive(archive_path, archive_sha256, run_prefix, receipt, parent_bytes)
    lineage = _verify_copy(run_dir, receipt, parent_bytes)
    manifest = _verify_manifest(run_dir, json.loads(parent_bytes), receipt)
    budget = _verify_budget(run_dir, receipt, read_json(baseline_path))
    old_review = read_json(run_dir / "02_world_review.json")
    attempts = read_json(run_dir / "02_world_review_attempts.json")
    if (old_review.get("status") != "passed"
            or not isinstance(attempts.get("attempts"), list)
            or not attempts["attempts"] or attempts["attempts"][-1] != old_review):
        raise ValueError("Historical world review or its attempt record changed")
    initial = read_json(run_dir / "experiment_source_hashes.json")
    ending = read_json(run_dir / "experiment_source_hashes_at_end.json")
    if (not isinstance(initial, dict) or not initial or not isinstance(ending, dict)
            or not ending or any(not isinstance(k, str) or not isinstance(v, str)
                              or len(v) != 64 for mapping in (initial, ending)
                              for k, v in mapping.items())):
        raise ValueError("Original source epoch is missing or malformed")
    return {"version": VERSION, "status": "historical_lineage_verified",
            "archive_sha256": archive_sha256,
            "historical_copy": lineage, "identity": manifest, "budget": budget,
            "historical_prompt_sha256": sha_file(run_dir / "prompts.jsonl"),
            "historical_prompt_bytes": (run_dir / "prompts.jsonl").stat().st_size,
            "historical_trace_bytes": (run_dir / "llm_attempts.jsonl").stat().st_size,
            "historical_world_review_sha256": sha_file(run_dir / "02_world_review.json"),
            "historical_world_review_attempts_sha256": sha_file(run_dir / "02_world_review_attempts.json"),
            "historical_production_sha256": sha_file(run_dir / "11_production.json"),
            "initial_source_map_sha256": sha_file(run_dir / "experiment_source_hashes.json"),
            "ending_source_map_sha256": sha_file(run_dir / "experiment_source_hashes_at_end.json"),
            "historical_world_review_attempts": len(attempts["attempts"]),
            "initial_source_files": len(initial), "ending_source_files": len(ending),
            "provider_calls": 0, "historical_run_modified": False,
            "complete_source_migration": False, "post_corpus_review_completed": False,
            "runnable_run_created": False, "paid_resume_authorization": False}


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--cost-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence path; prior receipts remain immutable")
    result = audit(args.derived_root / "run", args.derived_root / "evidence", args.cost_baseline,
        archive_path=args.archive, archive_sha256=args.archive_sha256, run_prefix=args.run_prefix)
    if args.output.resolve().is_relative_to(args.derived_root.resolve()):
        parser.error("Write the audit outside the immutable derived copy")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files": result["historical_copy"]["files"],
        "admitted_calls": result["budget"]["admitted_calls"], "runnable_run_created": False}))


if __name__ == "__main__":
    main()
