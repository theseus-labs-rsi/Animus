"""Bind the frozen source map to Git and inventory the target source epoch.

This is an offline, read-only prerequisite for a complete cumulative source
migration. It does not declare old stage results valid under changed code and
does not create a runnable run or open call admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys


VERSION = "cumulative-source-epoch-audit/v1"


def _forbid_network(*_args, **_kwargs):
    raise AssertionError("Source epoch audit forbids network and provider calls")


def install_offline_guard():
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = _forbid_network
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["OPENAI_API_KEY"] = "offline-disabled"
    os.environ["MODEL"] = "offline-disabled"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if result.returncode:
        raise ValueError("The required local Git source object is unavailable")
    return result.stdout


def _safe_source_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (name.endswith(".py") and bool(parts) and not name.startswith("/")
            and "\\" not in name and ".." not in parts
            and not any(":" in item or item.lower().startswith(".env")
                        or "credential" in item.lower() or "secret" in item.lower()
                        for item in parts))


def _blob_style(blob: bytes, expected_sha256: str) -> str | None:
    if sha(blob) == expected_sha256:
        return "git_blob"
    if sha(blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")) == expected_sha256:
        return "windows_crlf"
    return None


def _git_equivalent_checkout(blob: bytes, checkout: bytes, expected_sha256: str) -> bool:
    """Git may clean mixed Windows line endings while production hashes bytes."""
    return (sha(checkout) == expected_sha256
            and checkout.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"))


def audit(root: Path, old_map_path: Path, lineage_path: Path, *, legacy_commit: str) -> dict:
    install_offline_guard()
    root = root.resolve()
    old_raw = old_map_path.read_bytes()
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    old = json.loads(old_raw)
    if (lineage.get("status") != "historical_lineage_verified"
            or lineage.get("complete_source_migration") is not False
            or lineage.get("ending_source_map_sha256") != sha(old_raw)
            or not isinstance(old, dict) or not old
            or any(not _safe_source_path(name) or not isinstance(digest, str)
                   or len(digest) != 64 for name, digest in old.items())
            or len(legacy_commit) != 40
            or any(char not in "0123456789abcdef" for char in legacy_commit)):
        raise ValueError("Frozen source map is not bound to the historical lineage")
    legacy_head = _git(root, "rev-parse", "--verify", legacy_commit + "^{commit}").decode().strip()
    if legacy_head != legacy_commit:
        raise ValueError("Legacy Git commit identity changed")
    legacy_tree = _git(root, "rev-parse", legacy_commit + "^{tree}").decode().strip()
    old_styles = {"git_blob": 0, "windows_crlf": 0}
    for name, expected in old.items():
        style = _blob_style(_git(root, "show", f"{legacy_commit}:{name}"), expected)
        if style is None:
            raise ValueError("Frozen source map differs from the legacy Git commit: " + name)
        old_styles[style] += 1
    # Import the exact production inventory function only after the offline
    # boundary is installed. It has no provider or dotenv import at module load.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools import run_original_bc_smoke

    if run_original_bc_smoke.ROOT.resolve() != root:
        raise ValueError("Target checkout differs from the production source inventory root")

    target = run_original_bc_smoke.implementation_hashes()
    if any(not _safe_source_path(name) for name in target):
        raise ValueError("Target source inventory contains an unsafe path")
    target_head = _git(root, "rev-parse", "HEAD").decode().strip()
    target_tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    uncommitted = []
    for name, expected in target.items():
        result = subprocess.run(["git", "show", f"{target_head}:{name}"], cwd=root,
                                capture_output=True)
        if (result.returncode or not _git_equivalent_checkout(result.stdout,
                (root / name).read_bytes(), expected)):
            uncommitted.append(name)
    old_names, new_names = set(old), set(target)
    modified = sorted(name for name in old_names & new_names if old[name] != target[name])
    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)
    return {"version": VERSION,
            "status": "source_epochs_inventoried",
            "historical_lineage_sha256": sha(lineage_path.read_bytes()),
            "legacy_commit": legacy_head, "legacy_tree": legacy_tree,
            "legacy_source_map_sha256": sha(old_raw),
            "legacy_source_files": len(old), "legacy_byte_style": old_styles,
            "target_head": target_head, "target_tree": target_tree,
            "target_source_hashes": target,
            "target_source_files": len(target),
            "target_fully_committed": not uncommitted,
            "target_uncommitted_source_paths": sorted(uncommitted),
            "source_delta": {"unchanged": len(old_names & new_names) - len(modified),
                             "modified": modified, "added": added, "removed": removed},
            "provider_calls": 0, "historical_run_modified": False,
            "complete_source_migration": False,
            "stage_revalidation_completed": False,
            "runnable_run_created": False,
            "paid_resume_authorization": False}


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-map", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--legacy-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence path; prior receipts remain immutable")
    if args.output.resolve().is_relative_to(args.old_map.parent.resolve()):
        parser.error("Write evidence outside the immutable historical run copy")
    result = audit(args.root, args.old_map, args.lineage, legacy_commit=args.legacy_commit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    delta = result["source_delta"]
    print(json.dumps({"status": result["status"], "legacy_source_files": result["legacy_source_files"],
        "modified": len(delta["modified"]), "added": len(delta["added"]),
        "removed": len(delta["removed"]), "target_fully_committed": result["target_fully_committed"],
        "complete_source_migration": False}))


if __name__ == "__main__":
    main()
