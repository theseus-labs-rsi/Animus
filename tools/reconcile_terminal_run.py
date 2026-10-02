"""Prepare an offline, non-runnable terminal-state reconciliation copy.

The archived run remains immutable. A later cumulative recovery may copy the
verified control files to an independent checkout, but this tool never changes
the physical-call ledger, opens admission, or invokes production.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


VERSION = "terminal-state-reconciliation/v1"
RUN_FILES = ("manifest.json", "11_production.json", "experiment_profile.json",
             "05_corpus_warning.json", "05_corpus.json", "05_corpus_candidate.json")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8"))


def reconcile(raw_files, *, archive_sha256):
    """Return a changed manifest and proof, without mutating any input."""
    missing = set((*RUN_FILES, "controller_report.json")) - set(raw_files)
    if missing:
        raise ValueError("Missing terminal evidence: " + ", ".join(sorted(missing)))
    data = {name: json.loads(raw) for name, raw in raw_files.items()}
    manifest, production = data["manifest.json"], data["11_production.json"]
    profile, controller = data["experiment_profile.json"], data["controller_report.json"]
    warning = data["05_corpus_warning.json"]
    phases = controller.get("phases") or []
    if (manifest.get("status") != "running" or manifest.get("current_stage") != "corpus"
            or production.get("status") != "execution_error"
            or controller.get("status") != "needs_analysis"
            or len(phases) != 1 or phases[0].get("phase") != "generate"
            or phases[0].get("status") != "failed" or phases[0].get("exit_code") != 1
            or not controller.get("finished_utc")):
        raise ValueError("The frozen controller and run do not prove this terminal state")
    if (not profile.get("stopped") or (profile.get("stop_reason") or {}).get("reason")
            != "estimated_budget_limit" or profile.get("source_drift")
            or profile.get("admitted_calls") != profile.get("settled_usage_calls")
            or profile.get("unsettled_reservation_calls") != 0):
        raise ValueError("The stopped cumulative ledger is missing or unsettled")
    binding = warning.get("binding") or {}
    if (warning.get("status") != "warning" or warning.get("release_eligible") is not False
            or binding.get("corpus_hash") != canonical_hash(data["05_corpus.json"])
            or binding.get("candidate_hash") != canonical_hash(data["05_corpus_candidate.json"])):
        raise ValueError("The published corpus and incomplete candidate are not bound to the warning")
    if manifest.get("terminal_reconciliation") is not None:
        raise ValueError("The manifest already has a terminal reconciliation")
    child = deepcopy(manifest)
    child["status"] = "failed"
    child["current_stage"] = ""
    proof = {"version": VERSION, "archive_sha256": archive_sha256,
        "parent_manifest_sha256": sha256(raw_files["manifest.json"]),
        "controller_report_sha256": sha256(raw_files["controller_report.json"]),
        "production_sha256": sha256(raw_files["11_production.json"]),
        "profile_sha256": sha256(raw_files["experiment_profile.json"]),
        "published_corpus_sha256": sha256(raw_files["05_corpus.json"]),
        "candidate_corpus_sha256": sha256(raw_files["05_corpus_candidate.json"]),
        "warning_sha256": sha256(raw_files["05_corpus_warning.json"]),
        "terminal_finished_utc": controller["finished_utc"],
        "original_status": manifest["status"], "reconciled_status": child["status"],
        "original_current_stage": manifest["current_stage"],
        "provider_calls": 0, "ledger_changed": False, "admission_reopened": False,
        "stage_statuses_changed": False, "production_history_changed": False,
        "runnable_run_created": False}
    child["terminal_reconciliation"] = deepcopy(proof)
    retained = deepcopy(child)
    retained.pop("terminal_reconciliation")
    retained["status"] = manifest["status"]
    retained["current_stage"] = manifest["current_stage"]
    if retained != manifest:
        raise ValueError("Reconciliation changed a historical manifest field")
    return child, proof


def read_archive(archive_path, *, expected_sha256, run_prefix):
    hasher = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected_sha256:
        raise ValueError("The terminal archive differs from its external SHA")
    if not run_prefix.endswith("/") or ".." in Path(run_prefix).parts:
        raise ValueError("An exact run prefix ending in / is required")
    raw_files = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in (*RUN_FILES, "controller_report.json"):
            member_name = name if name == "controller_report.json" else run_prefix + name
            member = archive.getmember(member_name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Unsafe terminal artifact: " + name)
            raw_files[name] = archive.extractfile(member).read()
    return raw_files


def preview(archive_path, *, expected_sha256, run_prefix, destination):
    if destination.exists():
        raise ValueError("Use a new preview directory; never overwrite previous evidence")
    raw_files = read_archive(archive_path, expected_sha256=expected_sha256, run_prefix=run_prefix)
    child, proof = reconcile(raw_files, archive_sha256=expected_sha256)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "parent_manifest.json").write_bytes(raw_files["manifest.json"])
    child_bytes = (json.dumps(child, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    (destination / "reconciled_manifest.json").write_bytes(child_bytes)
    proof["reconciled_manifest_sha256"] = sha256(child_bytes)
    (destination / "reconciliation_receipt.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return proof


def derive_run(archive_path, *, expected_sha256, run_prefix, destination):
    """Copy every frozen run file before changing only its derived manifest.

    The output lives outside the factory's run root and keeps the budget stop.
    A receipt written last distinguishes a complete copy from interruption.
    """
    if destination.exists():
        raise ValueError("Use a fresh destination; never overwrite previous evidence")
    raw_files = read_archive(archive_path, expected_sha256=expected_sha256, run_prefix=run_prefix)
    child, proof = reconcile(raw_files, archive_sha256=expected_sha256)
    output_run = destination / "run"
    copied = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = []
        for item in archive:
            if not item.name.startswith(run_prefix):
                continue
            if item.isdir():
                continue
            if not item.isfile() or item.issym() or item.islnk():
                raise ValueError("The frozen run contains a non-regular artifact")
            members.append(item)
        paths = []
        for member in members:
            relative = member.name[len(run_prefix):]
            parts = PurePosixPath(relative).parts
            if (not parts or relative.startswith("/") or ".." in parts
                    or any(":" in part or ".env" in part.lower() or "credential" in part.lower()
                           or "secret" in part.lower() for part in parts)
                    or member.issym() or member.islnk()):
                raise ValueError("The frozen run contains a disallowed file path")
            target = output_run.joinpath(*parts)
            if os.name == "nt" and len(str(target.resolve())) >= 240:
                raise ValueError("Destination is too deep for a portable Windows run path")
            paths.append((member, relative, target))
        output_run.mkdir(parents=True, exist_ok=False)
        for member, relative, target in paths:
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            total = 0
            with archive.extractfile(member) as source, target.open("xb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)
                    digest.update(block)
                    total += len(block)
            if total != member.size:
                raise ValueError("A frozen run file was truncated during derivation")
            copied[relative] = {"sha256": digest.hexdigest(), "bytes": total}
    if not set(RUN_FILES) <= set(copied) or copied["manifest.json"]["sha256"] != proof["parent_manifest_sha256"]:
        raise ValueError("The derived run is missing a required frozen artifact")
    evidence = destination / "evidence"
    evidence.mkdir()
    (evidence / "parent_manifest.json").write_bytes(raw_files["manifest.json"])
    child_bytes = (json.dumps(child, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    (output_run / "manifest.json").write_bytes(child_bytes)
    proof.update(reconciled_manifest_sha256=sha256(child_bytes),
                 inherited_run_file_count=len(copied),
                 inherited_run_bytes=sum(item["bytes"] for item in copied.values()),
                 inherited_run_files=copied,
                 derived_run_dir="run", runnable_run_created=False)
    (evidence / "derivation_receipt.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--run-prefix", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--preview-dir", type=Path)
    destination.add_argument("--derive-root", type=Path)
    args = parser.parse_args()
    if args.preview_dir is not None:
        result = preview(args.archive, expected_sha256=args.archive_sha256,
            run_prefix=args.run_prefix, destination=args.preview_dir)
    else:
        result = derive_run(args.archive, expected_sha256=args.archive_sha256,
            run_prefix=args.run_prefix, destination=args.derive_root)
    print(json.dumps({key: result[key] for key in ("version", "parent_manifest_sha256",
        "reconciled_manifest_sha256", "provider_calls", "runnable_run_created")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
