"""Derive a source-bound world checkpoint without buying or relabeling old calls.

Both source environments recompile the original accepted facts in separate,
offline processes. This converts only completed business-value checkpoints;
publication/corpus acceptance and their attempt counters are never migrated.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION = "world-checkpoint-source-migration/v1"
EXECUTOR_VERSION = "business-value-executor/v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def inspect_worker(root, checkpoint):
    # No configuration import occurs until the process has closed its network
    # boundary. The exact selected checkout supplies every compiler import.
    import socket

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Checkpoint migration forbids network and provider calls")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = forbidden
    sys.path.insert(0, str(root.resolve()))
    import config
    config.chat = config.chat_json = forbidden
    from pipeline import instance_executor, world_agent
    from pipeline.world_blueprint import normalize_world_blueprint

    def sources():
        paths = sorted(set(root.glob("*.py")) | set((root / "pipeline").rglob("*.py")))
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}

    source_before = sources()

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (state.get("version") != EXECUTOR_VERSION or state.get("status") != "completed"
            or state.get("checkpoint_hash") != digest({k: v for k, v in state.items() if k != "checkpoint_hash"})):
        raise ValueError("Migration requires an intact completed business-value checkpoint")
    wp = state["input_snapshot"]["whitepaper"]
    config.STRUCTURE_MODEL = state["binding"]["model"]
    if state["binding"]["existing_hash"] != digest(None):
        raise ValueError("An existing-world checkpoint requires its original existing input")
    blueprint = normalize_world_blueprint(wp)
    if blueprint != state["input_snapshot"]["blueprint"]:
        raise ValueError("Original blueprint normalization changed")
    for unit in state["units"]:
        if unit["raw_hash"] != digest(unit["raw"]):
            raise ValueError("Accepted unit raw facts changed")
    table = instance_executor.assembled_facts(wp, state["units"], complete=True)
    world, applied, initial, issues = world_agent._compiled(table, blueprint, None, True, wp)
    binding = world_agent._binding(wp, None)
    if sources() != source_before:
        raise ValueError("Compiler sources changed during inspection")
    return {"binding": binding, "table": table,
        "world": world.to_dict(), "applied": applied, "initial": initial, "issues": issues,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_hashes": source_before, "provider_calls": 0}


def compare_inspections(state, old, current):
    if old["binding"] != state["binding"]:
        raise ValueError("Legacy source/model/input does not match the original binding")
    if old["checkpoint_sha256"] != current["checkpoint_sha256"]:
        raise ValueError("The compiler environments did not inspect the same original bytes")
    for key in ("version", "wp_hash", "existing_hash", "model"):
        if current["binding"][key] != state["binding"][key]:
            raise ValueError("Migration may change implementation bytes only: " + key)
    if current["binding"] == state["binding"]:
        raise ValueError("The checkpoint already has the target source binding")
    for key in ("table", "world", "applied", "initial", "issues"):
        if old[key] != current[key]:
            raise ValueError("The target compiler changed the original accepted projection: " + key)
    if old["issues"]:
        raise ValueError("Original accepted facts do not pass the complete compiler")


def inspect(root, checkpoint):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1",
        OPENAI_API_KEY="offline-disabled", MODEL="offline-disabled",
        OPENAI_BASE_URL="http://127.0.0.1:9/v1")
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
        "--inspect-worker", str(root.resolve()), str(checkpoint.resolve())],
        cwd=root, env=environment, capture_output=True, encoding="utf-8", timeout=60)
    if result.returncode:
        raise ValueError("Offline compiler inspection failed: " + result.stderr[-3000:])
    return json.loads(result.stdout)


def migrate(checkpoint, *, expected_sha256, legacy_root, target_root, destination, inspect_fn=inspect):
    parent_bytes = checkpoint.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    if parent_sha != expected_sha256:
        raise ValueError("Original checkpoint differs from the externally supplied SHA")
    state = json.loads(parent_bytes)
    if (state.get("version") != EXECUTOR_VERSION or state.get("status") != "completed"
            or state.get("checkpoint_hash") != digest({k: v for k, v in state.items() if k != "checkpoint_hash"})):
        raise ValueError("Migration requires an intact completed business-value checkpoint")
    if destination.exists():
        raise ValueError("Use a new destination; existing recovery evidence remains immutable")
    if destination.resolve().is_relative_to(legacy_root.resolve()):
        raise ValueError("The frozen legacy source tree remains read-only")
    old = inspect_fn(legacy_root, checkpoint)
    current = inspect_fn(target_root, checkpoint)
    compare_inspections(state, old, current)
    if checkpoint.read_bytes() != parent_bytes:
        raise ValueError("Original checkpoint changed during inspection")
    if old["checkpoint_sha256"] != parent_sha:
        raise ValueError("Compiler inspection is not bound to the supplied parent checkpoint")
    candidate = deepcopy(state)
    candidate["binding"] = deepcopy(current["binding"])
    proof = {"version": VERSION, "parent_sha256": parent_sha,
        "parent_checkpoint_hash": state["checkpoint_hash"], "parent_binding": deepcopy(state["binding"]),
        "target_binding": deepcopy(current["binding"]),
        "compiled_projection_hash": digest({k: current[k] for k in ("table", "world", "applied", "initial", "issues")}),
        "legacy_compiler_source_hashes": deepcopy(old["source_hashes"]),
        "target_compiler_source_hashes": deepcopy(current["source_hashes"]),
        "provider_calls": 0, "semantic_revalidation": False,
        "publication_or_signal_migration": False,
        "original_fields_hash": digest({k: v for k, v in state.items() if k not in ("binding", "checkpoint_hash")})}
    candidate.setdefault("implementation_migrations", []).append(deepcopy(proof))
    candidate["checkpoint_hash"] = digest({k: v for k, v in candidate.items() if k != "checkpoint_hash"})
    # Explicit child provenance is the only new record. Old records, protocol
    # versions, pending receipts, raw responses and every counter keep identity.
    retained = deepcopy(candidate)
    retained["binding"] = state["binding"]
    retained["checkpoint_hash"] = state["checkpoint_hash"]
    if "implementation_migrations" in state:
        retained["implementation_migrations"] = deepcopy(state["implementation_migrations"])
    else:
        retained.pop("implementation_migrations")
    if retained != state:
        raise ValueError("Migration changed a historical input, record or counter")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "parent_checkpoint.json").write_bytes(parent_bytes)
    child_path = destination / checkpoint.name
    child_bytes = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    child_path.write_bytes(child_bytes)
    proof.update(child_file=child_path.name, child_sha256=hashlib.sha256(child_bytes).hexdigest(),
        child_checkpoint_hash=candidate["checkpoint_hash"], all_original_fields_retained=True,
        ledger_or_budget_changes=False, paid_resume_authorization=False)
    (destination / "migration_receipt.json").write_bytes(
        (json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return proof


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--inspect-worker":
        print(json.dumps(inspect_worker(Path(sys.argv[2]), Path(sys.argv[3])), ensure_ascii=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    proof = migrate(args.checkpoint, expected_sha256=args.checkpoint_sha256,
        legacy_root=args.legacy_root, target_root=args.target_root, destination=args.destination)
    print(json.dumps({key: proof[key] for key in ("version", "parent_sha256", "child_sha256",
        "provider_calls", "semantic_revalidation", "all_original_fields_retained", "paid_resume_authorization")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
