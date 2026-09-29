"""Classify every frozen file before any cumulative child can be created.

The result is an offline migration input, not a complete migration receipt or
an admission decision.  Historical output stays historical until a separate
review, repair, and stage-specific replay proves a current result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess


VERSION = "cumulative-migration-scope-audit/v1"
LEDGER = {"experiment_profile.json", "llm_attempts.jsonl", "prompts.jsonl"}
SOURCES = {"experiment_source_hashes.json", "experiment_source_hashes_at_end.json"}
INPUTS = {"00_input.json", "00_seed_pack.json", "00_about.json",
          "01_whitepaper.json", "01_instance_plan.json", "01_supply_plan.json"}
WORLD = {"02_world.json"}
LOCKS = {".production.lock", ".run.lock"}


def install_offline_guard():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Migration scope audit forbids network calls")
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


def classify(name: str) -> str:
    if name in LEDGER:
        return "cumulative_ledger_append_only"
    if name in SOURCES:
        return "historical_source_epoch_immutable"
    if name == "manifest.json":
        return "historical_manifest_reconciled_only"
    if name in LOCKS:
        return "historical_lock_never_reused"
    if name in INPUTS:
        return "historical_input_retained_not_revalidated"
    if name in WORLD or name.startswith("02_world_agent_"):
        return "historical_world_input_requires_new_review"
    return "historical_output_not_current_acceptance"


def _safe(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (bool(parts) and not name.startswith("/") and "\\" not in name
            and ".." not in parts and not any(":" in part or part.lower().startswith(".env")
                or "credential" in part.lower() or "secret" in part.lower() for part in parts))


def _verify_world_receipt(world: dict, inherited: dict, epoch: dict, root: Path) -> None:
    checkpoint = "02_world_agent_5903517308903f10c8b0.json"
    target = world.get("target_compiler_source_hashes")
    if (world.get("version") != "world-checkpoint-source-migration/v1"
            or world.get("parent_sha256") != inherited[checkpoint]["sha256"]
            or world.get("all_original_fields_retained") is not True
            or world.get("provider_calls") != 0
            or world.get("semantic_revalidation") is not False
            or world.get("publication_or_signal_migration") is not False
            or world.get("ledger_or_budget_changes") is not False
            or world.get("paid_resume_authorization") is not False
            or not isinstance(target, dict) or not target):
        raise ValueError("World-only migration is absent, changed, or overstated")
    epoch_sources = epoch["target_source_hashes"]
    overlap = set(target) & set(epoch_sources)
    if not overlap or any(target[name] != epoch_sources[name] for name in overlap):
        raise ValueError("World compiler migration targets a different source epoch")
    for name, expected in target.items():
        if (not _safe(name) or not name.endswith(".py") or not (root / name).is_file()
                or sha(root / name) != expected):
            raise ValueError("World compiler source changed after its migration proof: " + name)


def audit(derived_root: Path, lineage_path: Path, stage_path: Path,
          epoch_path: Path, world_path: Path, *, source_root: Path) -> dict:
    install_offline_guard()
    derived_root = derived_root.resolve()
    source_root = source_root.resolve()
    run = derived_root / "run"
    receipt_path = derived_root / "evidence" / "derivation_receipt.json"
    receipt = load(receipt_path)
    inherited = receipt.get("inherited_run_files")
    lineage, stage, epoch, world = (load(path) for path in
        (lineage_path, stage_path, epoch_path, world_path))
    if (not isinstance(inherited, dict) or len(inherited) != 348
            or lineage.get("status") != "historical_lineage_verified"
            or lineage.get("archive_sha256") != receipt.get("archive_sha256")
            or lineage.get("historical_copy", {}).get("files") != len(inherited)
            or stage.get("status") != "historical_stage_inventory_verified"
            or stage.get("parent_lineage_sha256") != sha(lineage_path)
            or stage.get("derivation_receipt_sha256") != sha(receipt_path)
            or epoch.get("status") != "source_epochs_inventoried"
            or epoch.get("historical_lineage_sha256") != sha(lineage_path)
            or epoch.get("legacy_source_map_sha256")
               != lineage.get("ending_source_map_sha256")
            or epoch.get("target_fully_committed") is not True
            or epoch.get("target_uncommitted_source_paths") != []
            or any(obj.get("provider_calls") != 0 for obj in (lineage, stage, epoch))
            or any(obj.get("complete_source_migration") is not False
                   for obj in (lineage, stage, epoch))):
        raise ValueError("Historical lineage, stage, or source epoch is incomplete")
    for ref, field in (("HEAD", "target_head"), ("HEAD^{tree}", "target_tree")):
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=source_root).decode().strip()
        if actual != epoch.get(field):
            raise ValueError("Target Git source epoch changed")
    if subprocess.check_output(["git", "status", "--porcelain=v1",
            "--untracked-files=all", "--", "pipeline", "eval", "config.py",
            "execution_control.py", "llm_transport.py", "llm_trace.py",
            "tools/diversity_metrics.py", "tools/run_original_bc_smoke.py"], cwd=source_root):
        raise ValueError("Target production source is not committed cleanly")
    _verify_world_receipt(world, inherited, epoch, source_root)
    old_manifest = derived_root / "evidence" / "parent_manifest.json"
    if sha(old_manifest) != receipt.get("parent_manifest_sha256"):
        raise ValueError("Original manifest changed")
    if (sha(run / "manifest.json") != receipt.get("reconciled_manifest_sha256")
            or receipt.get("runnable_run_created") is not False
            or receipt.get("admission_reopened") is not False):
        raise ValueError("Derived terminal copy became runnable or changed")
    categories = Counter()
    rows = {}
    total_bytes = 0
    for name, record in inherited.items():
        if not _safe(name):
            raise ValueError("Unsafe historical run member")
        path = run.joinpath(*PurePosixPath(name).parts)
        if (not path.is_file() or (name != "manifest.json"
                and (path.stat().st_size != record.get("bytes")
                     or sha(path) != record.get("sha256")))):
            raise ValueError("Frozen historical file changed or disappeared: " + name)
        category = classify(name)
        categories[category] += 1
        rows[name] = {"sha256": record["sha256"], "bytes": record["bytes"],
                      "migration_scope": category}
        total_bytes += record["bytes"]
    if (total_bytes != receipt.get("inherited_run_bytes")
            or not LEDGER <= set(rows) or not SOURCES <= set(rows)
            or not INPUTS <= set(rows) or not WORLD <= set(rows)
            or not LOCKS <= set(rows)
            or stage.get("orders", {}).get("raw") != 42
            or stage.get("orders", {}).get("effective") != 42
            or stage.get("author_slots", {}).get("active_groups") != 119
            or stage.get("author_slots", {}).get("admitted_slots") != 131
            or stage.get("author_slots", {}).get("saved_author_replies") != 130
            or stage.get("corpus", {}).get("candidate_documents") != 207
            or stage.get("corpus", {}).get("release_eligible") is not False):
        raise ValueError("Cumulative history or stage denominator is incomplete")
    return {"version": VERSION, "status": "historical_migration_scope_classified",
            "parent_lineage_sha256": sha(lineage_path),
            "historical_stage_inventory_sha256": sha(stage_path),
            "source_epoch_sha256": sha(epoch_path),
            "world_migration_sha256": sha(world_path),
            "derivation_receipt_sha256": sha(receipt_path),
            "archive_sha256": lineage["archive_sha256"],
            "target_head": epoch["target_head"], "target_tree": epoch["target_tree"],
            "historical_file_count": len(rows), "historical_file_bytes": total_bytes,
            "migration_scope_counts": dict(categories), "files": rows,
            "historical_stage_attempts_preserved": True,
            "historical_ledger_append_only": True,
            "first_new_operation": "post_corpus_world.semantic_review",
            "downstream_results_current": False,
            "complete_source_migration": False, "runnable_run_created": False,
            "paid_resume_authorization": False, "provider_calls": 0,
            "historical_run_modified": False}


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--stage-inventory", type=Path, required=True)
    parser.add_argument("--source-epoch", type=Path, required=True)
    parser.add_argument("--world-migration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.derived_root.resolve()):
        parser.error("Use a fresh evidence path outside the immutable historical copy")
    result = audit(args.derived_root, args.lineage, args.stage_inventory,
        args.source_epoch, args.world_migration, source_root=args.source_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files": result["historical_file_count"],
        "bytes": result["historical_file_bytes"],
        "complete_source_migration": False, "provider_calls": 0}))


if __name__ == "__main__":
    main()
