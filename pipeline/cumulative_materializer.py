"""Install a separately authorized cumulative child from a frozen run copy.

The only public operation requires the full read-only prewrite proof. It copies
the old run byte for byte into a private staging directory, archives the two
control documents it must replace, and publishes the child with one rename.
It never opens a Run, installs a provider, or consumes a model call.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import uuid

from pipeline.cumulative_child import build_control_documents
from pipeline.cumulative_prewrite import PrewritePermit, _safe_member, validate_prewrite


def _io_path(path: Path) -> Path:
    """Keep deep historical checkpoint names within the Windows file API."""
    if os.name != "nt":
        return Path(path)
    value = os.path.abspath(path)
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


EVIDENCE_FILES = {
    "00_cumulative_lineage.json": "lineage_path",
    "00_cumulative_source_migration.json": "migration_path",
    "12_downstream_evidence.json": "downstream_evidence_path",
    "00_cumulative_cap_authorization.json": "authorization_path",
    "00_first_world_review_boundary.json": "boundary_path",
    "00_cumulative_stage_inventory.json": "stage_path",
    "00_cumulative_source_epoch.json": "source_epoch_path",
    "00_cumulative_migration_scope.json": "scope_path",
    "00_cumulative_world_migration.json": "world_migration_path",
}


@dataclass(frozen=True)
class MaterializedChild:
    destination: Path
    permit: PrewritePermit


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _encode(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_new(path: Path, raw: bytes) -> None:
    path = _io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_verified(source: Path, target: Path, expected_sha: str, expected_bytes: int) -> None:
    source, target = _io_path(source), _io_path(target)
    if source.is_symlink() or not source.is_file() or type(expected_bytes) is not int:
        raise ValueError("Historical member is not a regular file")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        for block in iter(lambda: reader.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != expected_sha or size != expected_bytes:
        raise ValueError("Historical file changed while copying: " + source.name)


def materialize_child(*, derived_root: Path, destination: Path, source_root: Path,
                      lineage_path: Path, stage_path: Path, source_epoch_path: Path,
                      scope_path: Path, migration_path: Path, authorization_path: Path,
                      boundary_path: Path, world_migration_path: Path,
                      downstream_evidence_path: Path,
                      expected_archive_sha256: str) -> MaterializedChild:
    """Make a new run directory only after exact history and cap authority pass.

    All inputs are checked again immediately before publication. The caller
    must install a separate bounded execution entry before any later Run open.
    """
    args = dict(derived_root=derived_root, destination=destination,
                source_root=source_root, lineage_path=lineage_path,
                stage_path=stage_path, source_epoch_path=source_epoch_path,
                scope_path=scope_path, migration_path=migration_path,
                authorization_path=authorization_path, boundary_path=boundary_path,
                world_migration_path=world_migration_path,
                downstream_evidence_path=downstream_evidence_path,
                expected_archive_sha256=expected_archive_sha256)
    permit = validate_prewrite(**args)
    derived_root, destination = derived_root.resolve(), destination.resolve()
    receipt_path = derived_root / "evidence" / "derivation_receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    inherited = receipt["inherited_run_files"]
    source_files = {name: Path(args[arg]) for name, arg in EVIDENCE_FILES.items()}
    evidence_raw = {name: path.read_bytes() for name, path in source_files.items()}
    evidence_sha = {name: _sha(raw) for name, raw in evidence_raw.items()}
    lineage = json.loads(evidence_raw["00_cumulative_lineage.json"])
    migration = json.loads(evidence_raw["00_cumulative_source_migration.json"])
    authorization = json.loads(evidence_raw["00_cumulative_cap_authorization.json"])
    epoch = json.loads(evidence_raw["00_cumulative_source_epoch.json"])
    run = derived_root / "run"
    manifest_bytes = (run / "manifest.json").read_bytes()
    profile_bytes = (run / "experiment_profile.json").read_bytes()
    if (_sha(receipt_raw) != _sha(receipt_path.read_bytes())
            or _sha(manifest_bytes) != receipt["reconciled_manifest_sha256"]
            or _sha(profile_bytes) != lineage["budget"]["profile_sha256"]):
        raise ValueError("Frozen child inputs changed before staging")
    manifest, profile = build_control_documents(
        permit=permit, parent_manifest_bytes=manifest_bytes,
        parent_profile_bytes=profile_bytes, lineage=lineage, migration=migration,
        authorization=authorization,
        lineage_sha256=evidence_sha["00_cumulative_lineage.json"],
        migration_sha256=evidence_sha["00_cumulative_source_migration.json"],
        evidence_sha256=evidence_sha["12_downstream_evidence.json"],
        authorization_sha256=evidence_sha["00_cumulative_cap_authorization.json"],
        boundary_sha256=evidence_sha["00_first_world_review_boundary.json"],
        stage_sha256=evidence_sha["00_cumulative_stage_inventory.json"],
        source_epoch_sha256=evidence_sha["00_cumulative_source_epoch.json"],
        parent_manifest_sha256=_sha(manifest_bytes),
        parent_profile_sha256=_sha(profile_bytes))
    if permit.destination != destination or destination.exists():
        raise ValueError("Cumulative child destination changed before staging")
    runs_parent = source_root.resolve() / "output" / "runs"
    if ((source_root.resolve() / "output").is_symlink()
            or runs_parent.is_symlink()
            or destination.parent != runs_parent.resolve()):
        raise ValueError("Cumulative child directory escaped the source workspace")
    _io_path(runs_parent).mkdir(parents=True, exist_ok=True)
    staging_id = uuid.uuid4().hex
    staging = destination.with_name(f".{destination.name}.cumulative-{staging_id}.staging")
    if staging.resolve().parent != destination.parent or staging.exists():
        raise ValueError("Cumulative staging path escaped its run directory")
    _io_path(staging).mkdir(parents=False, exist_ok=False)
    try:
        _write_new(staging / ".cumulative_staging_marker", staging_id.encode("ascii"))
        for name, entry in sorted(inherited.items()):
            if not _safe_member(name):
                raise ValueError("Unsafe historical member: " + name)
            expected_sha = (receipt["reconciled_manifest_sha256"] if name == "manifest.json"
                            else entry["sha256"])
            expected_size = (len(manifest_bytes) if name == "manifest.json"
                             else entry["bytes"])
            parts = PurePosixPath(name).parts
            _copy_verified(run.joinpath(*parts), staging.joinpath(*parts),
                           expected_sha, expected_size)
        history = staging / "cumulative_history"
        _write_new(history / "manifest_before_cumulative.json", manifest_bytes)
        _write_new(history / "experiment_profile_before_cumulative.json", profile_bytes)
        # Keep the complete inherited file inventory in the child so later
        # owner routing can recheck its original draft and checkpoint bytes.
        _write_new(staging / "00_cumulative_derivation_receipt.json", receipt_raw)
        for name, raw in evidence_raw.items():
            _write_new(staging / name, raw)
        _write_new(staging / "experiment_source_hashes_cumulative_start.json",
                   _encode(epoch["target_source_hashes"]))
        manifest_raw, profile_raw = _encode(manifest), _encode(profile)
        _io_path(staging / "manifest.json").write_bytes(manifest_raw)
        _io_path(staging / "experiment_profile.json").write_bytes(profile_raw)
        materialization = {
            "version": "cumulative-child-materialization/v1",
            "status": "materialized_offline",
            "original_files": len(inherited),
            "original_file_inventory_sha256": _sha(_encode(inherited)),
            "derivation_receipt_sha256": _sha(receipt_raw),
            "historical_manifest_sha256": _sha(manifest_bytes),
            "historical_profile_sha256": _sha(profile_bytes),
            "child_manifest_sha256": _sha(manifest_raw),
            "child_profile_sha256": _sha(profile_raw),
            "evidence_sha256": evidence_sha,
            "provider_calls": 0,
            "stage_revalidation_completed": False,
            "new_world_opinion": False,
            "experiment_target_passed": False,
        }
        _write_new(staging / "00_cumulative_materialization_receipt.json",
                   _encode(materialization))
        renewed = validate_prewrite(**args)
        if (renewed.migration_sha256 != permit.migration_sha256
                or renewed.authorization_sha256 != permit.authorization_sha256
                or _sha(receipt_path.read_bytes()) != _sha(receipt_raw)
                or any(_sha(path.read_bytes()) != evidence_sha[name]
                       for name, path in source_files.items())
                or destination.exists()):
            raise ValueError("Cumulative evidence changed before child publication")
        os.rename(_io_path(staging), _io_path(destination))
        return MaterializedChild(destination, renewed)
    except BaseException:
        # This is only our fresh private staging tree, never the parent run.
        if (not staging.is_symlink() and staging.resolve().parent == destination.parent
                and staging.name == f".{destination.name}.cumulative-{staging_id}.staging"
                and _io_path(staging).is_dir()
                and _io_path(staging / ".cumulative_staging_marker").is_file()
                and _io_path(staging / ".cumulative_staging_marker").read_bytes()
                    == staging_id.encode("ascii")):
            shutil.rmtree(_io_path(staging))
        raise


def recover_pristine_child(*, derived_root: Path, destination: Path, source_root: Path,
                           lineage_path: Path, stage_path: Path, source_epoch_path: Path,
                           scope_path: Path, migration_path: Path, authorization_path: Path,
                           boundary_path: Path, world_migration_path: Path,
                           downstream_evidence_path: Path,
                           expected_archive_sha256: str) -> MaterializedChild:
    """Re-enter only a published child on which execution has never started.

    A crash between atomic publication and the wrapper's first ledger write
    must not strand the authorized copy. Any changed child file or extra stage
    artifact requires separate reconciliation, so this operation stays read-only.
    """
    args = dict(derived_root=derived_root, destination=destination,
        source_root=source_root, lineage_path=lineage_path, stage_path=stage_path,
        source_epoch_path=source_epoch_path, scope_path=scope_path,
        migration_path=migration_path, authorization_path=authorization_path,
        boundary_path=boundary_path, world_migration_path=world_migration_path,
        downstream_evidence_path=downstream_evidence_path,
        expected_archive_sha256=expected_archive_sha256)
    permit = validate_prewrite(**args, _allow_existing_pristine=True)
    destination = destination.resolve()
    receipt_path = derived_root.resolve() / "evidence" / "derivation_receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    inherited = receipt["inherited_run_files"]
    source_files = {name: Path(args[arg]) for name, arg in EVIDENCE_FILES.items()}
    evidence_raw = {name: path.read_bytes() for name, path in source_files.items()}
    evidence_sha = {name: _sha(raw) for name, raw in evidence_raw.items()}
    run = derived_root.resolve() / "run"
    old_manifest = _io_path(run / "manifest.json").read_bytes()
    old_profile = _io_path(run / "experiment_profile.json").read_bytes()
    lineage = json.loads(evidence_raw["00_cumulative_lineage.json"])
    migration = json.loads(evidence_raw["00_cumulative_source_migration.json"])
    authorization = json.loads(evidence_raw["00_cumulative_cap_authorization.json"])
    epoch = json.loads(evidence_raw["00_cumulative_source_epoch.json"])
    manifest, profile = build_control_documents(
        permit=permit, parent_manifest_bytes=old_manifest,
        parent_profile_bytes=old_profile, lineage=lineage, migration=migration,
        authorization=authorization,
        lineage_sha256=evidence_sha["00_cumulative_lineage.json"],
        migration_sha256=evidence_sha["00_cumulative_source_migration.json"],
        evidence_sha256=evidence_sha["12_downstream_evidence.json"],
        authorization_sha256=evidence_sha["00_cumulative_cap_authorization.json"],
        boundary_sha256=evidence_sha["00_first_world_review_boundary.json"],
        stage_sha256=evidence_sha["00_cumulative_stage_inventory.json"],
        source_epoch_sha256=evidence_sha["00_cumulative_source_epoch.json"],
        parent_manifest_sha256=_sha(old_manifest),
        parent_profile_sha256=_sha(old_profile))
    expected_manifest, expected_profile = _encode(manifest), _encode(profile)
    materialization_path = _io_path(destination / "00_cumulative_materialization_receipt.json")
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if (materialization.get("version") != "cumulative-child-materialization/v1"
            or materialization.get("status") != "materialized_offline"
            or materialization.get("original_files") != len(inherited)
            or materialization.get("original_file_inventory_sha256") != _sha(_encode(inherited))
            or materialization.get("derivation_receipt_sha256") != _sha(receipt_raw)
            or materialization.get("historical_manifest_sha256") != _sha(old_manifest)
            or materialization.get("historical_profile_sha256") != _sha(old_profile)
            or materialization.get("child_manifest_sha256") != _sha(expected_manifest)
            or materialization.get("child_profile_sha256") != _sha(expected_profile)
            or materialization.get("evidence_sha256") != evidence_sha
            or materialization.get("provider_calls") != 0
            or materialization.get("new_world_opinion") is not False
            or materialization.get("stage_revalidation_completed") is not False):
        raise ValueError("Published cumulative child has no matching pristine receipt")
    expected = {
        "manifest.json": expected_manifest,
        "experiment_profile.json": expected_profile,
        "cumulative_history/manifest_before_cumulative.json": old_manifest,
        "cumulative_history/experiment_profile_before_cumulative.json": old_profile,
        "00_cumulative_derivation_receipt.json": receipt_raw,
        "experiment_source_hashes_cumulative_start.json": _encode(epoch["target_source_hashes"]),
    }
    expected.update(evidence_raw)
    for name, raw in expected.items():
        if _io_path(destination.joinpath(*PurePosixPath(name).parts)).read_bytes() != raw:
            raise ValueError("Published cumulative child changed before its first execution: " + name)
    for name, entry in inherited.items():
        if name in ("manifest.json", "experiment_profile.json"):
            continue
        path = _io_path(destination.joinpath(*PurePosixPath(name).parts))
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Published cumulative history changed: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("Published cumulative history changed: " + name)
    marker = _io_path(destination / ".cumulative_staging_marker")
    if (not marker.is_file() or len(marker.read_bytes()) != 32
            or any(byte not in b"0123456789abcdef" for byte in marker.read_bytes())):
        raise ValueError("Published cumulative child has no staging identity")
    allowed = (set(inherited) | set(expected) |
        {"00_cumulative_materialization_receipt.json", ".cumulative_staging_marker"})
    root = _io_path(destination)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Published cumulative child contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != allowed:
        raise ValueError("Published cumulative child has unexpected execution files")
    return MaterializedChild(destination, permit)
