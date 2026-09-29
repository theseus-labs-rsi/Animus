"""Validate user-supplied seed contracts offline; optionally verify local source bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.seed_pack import SeedPackError, generation_context, load_seed_pack, seed_digest
from pipeline.seed_run import require_raw_seed


def inspect_pack(path: Path, *, verify_sources: bool = False,
                 source_root: Path | None = None) -> dict:
    """Check a stored seed and optionally its files, without model calls.

    ``generation_ready`` means structural checks and the original factory's
    generation-entry checks pass. It does not certify extraction or business truth.
    Source paths resolve against the selected workspace; resolving first also
    prevents symlinks from escaping the workspace during hash verification.
    """
    pack = load_seed_pack(path)
    errors = []
    source_root = (source_root or ROOT).resolve()
    if verify_sources:
        for source in pack["sources"]:
            try:
                source_path = (source_root / source["path"]).resolve()
                source_path.relative_to(source_root)
            except (OSError, RuntimeError, ValueError):
                errors.append(f"{source['id']}: source path must resolve within source_root")
                continue
            try:
                digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError:
                errors.append(f"{source['id']}: source file unavailable")
                continue
            if digest != source["sha256"].lower():
                errors.append(f"{source['id']}: source SHA256 mismatch")
    blocking_issues = pack.get("review", {}).get("blocking_issues", [])
    generation_blockers = []
    try:
        require_raw_seed(pack)
    except SeedPackError as error:
        generation_blockers.append(str(error))
    try:
        generation_context(pack)
    except SeedPackError as error:
        generation_blockers.append(str(error))
    return {"path": str(path), "seed_id": pack["seed_id"], "digest": seed_digest(pack),
            "schema_version": pack["schema_version"], "structure_valid": True,
            "mechanisms": len(pack["mechanisms"]), "sources": len(pack["sources"]),
            "source_root": str(source_root),
            "source_bytes_verified": verify_sources and not errors,
            "blocking_issues": blocking_issues,
            "generation_blockers": generation_blockers,
            "generation_ready": not errors and not generation_blockers,
            "semantic_review_performed": False,
            "passed": not errors, "issues": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Seed JSON paths to validate")
    parser.add_argument("--verify-sources", action="store_true",
                        help="Also check provenance hashes against local original files")
    parser.add_argument("--source-root", type=Path,
                        help="Resolve source paths inside this original document workspace; default: repository root")
    parser.add_argument("--require-generation-ready", action="store_true",
                        help="Require the original factory's generation-entry checks as well as structure and selected source hashes")
    args = parser.parse_args()
    paths = args.paths
    reports = []
    for path in paths:
        try:
            reports.append(inspect_pack(path, verify_sources=args.verify_sources,
                                        source_root=args.source_root))
        except SeedPackError as error:
            reports.append({"path": str(path), "passed": False, "structure_valid": False,
                            "generation_ready": False, "semantic_review_performed": False,
                            "generation_blockers": [str(error)],
                            "issues": [str(error)]})
    passed = all(report["passed"] for report in reports)
    generation_ready = all(report["generation_ready"] for report in reports)
    print(json.dumps({"passed": passed,
                      "generation_ready": generation_ready,
                      "supported_schema_versions": [1, 2],
                      "semantic_review_performed": False,
                      "generation_ready_scope": "structural and factory generation-entry checks only",
                      "packs": reports}, ensure_ascii=False, indent=2))
    if not passed or (args.require_generation_ready and not generation_ready):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
