"""Prepare explicit corpus-slot inheritance from a frozen run, without dispatch.

This prepares a manifest and copies the historical signal checkpoints into a
new evidence directory.  It does not create a production run or authorize a
paid resume.  The renderer checks the complete current task again on use.
"""
from collections import Counter
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import tarfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VERSION = "signal-slot-inheritance/v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(archive_path, trace_path, audit_path, destination, *,
            archive_sha256, trace_sha256, audit_sha256, run_prefix):
    os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-disabled",
                      MODEL="offline-disabled")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Signal inheritance preparation forbids network calls")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = forbidden
    from pipeline.semantic_review import fingerprint
    from pipeline.run import _io_path
    for path, expected in ((archive_path, archive_sha256),
                           (trace_path, trace_sha256), (audit_path, audit_sha256)):
        if _sha(path.read_bytes()) != expected:
            raise ValueError("Externally pinned inheritance input changed: " + path.name)
    if destination.exists():
        raise ValueError("Use a fresh output directory; never modify old run evidence")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    groups = audit["groups"]
    wanted = {row["legacy_key"] for row in groups}
    if (not groups or len({row["key"] for row in groups}) != len(groups)
            or len({row["legacy_key"] for row in groups}) != len(groups)
            or not audit["public_plan_validated_without_changes"]):
        raise ValueError("Incomplete or repeated audited corpus groups")
    archive_rows = {}
    profile = None
    run_manifest = None
    prompts = Counter()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name
            if name == run_prefix + "experiment_profile.json":
                profile = json.load(archive.extractfile(member))
            elif name == run_prefix + "manifest.json":
                run_manifest = json.load(archive.extractfile(member))
            elif name == run_prefix + "prompts.jsonl":
                for line in archive.extractfile(member):
                    row = json.loads(line)
                    if row.get("step") == "render.signal":
                        prompts[(fingerprint(row["messages"]), fingerprint(row["output"]))] += 1
            elif name.startswith(run_prefix + "05_signal_checkpoints/") and name.endswith(".json"):
                if Path(name).stem in wanted:
                    archive_rows[Path(name).stem] = archive.extractfile(member).read()
    if profile is None or run_manifest is None or set(archive_rows) != wanted:
        raise ValueError("The frozen run profile or exact audited signal set is missing")
    env = run_manifest["env"]
    if env.get("model") != profile["model"]:
        raise ValueError("Historical run and paid profile model identities differ")
    physical = Counter()
    physical_inputs = {}
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "request" and row.get("step") == "render.signal":
                message_hash = fingerprint(row["messages"])
                physical[message_hash] += 1
                physical_inputs.setdefault(message_hash, []).append(row)
    entries = []
    for row in groups:
        old_bytes = archive_rows[row["legacy_key"]]
        state = json.loads(old_bytes)
        if (state.get("key") != row["legacy_key"]
                or state.get("progress_hash") != fingerprint(state.get("progress"))
                or fingerprint(state) != row["original_cache_hash"]):
            raise ValueError("Audited historical signal bytes or progress changed")
        slots = state["progress"].get("drafts", [])
        if len(slots) != len(row["original_drafts"]):
            raise ValueError("Historical draft positions changed")
        draft_receipts = []
        for index, (slot, draft) in enumerate(zip(slots, row["original_drafts"])):
            if (draft["position"] != index or slot.get("admitted") is not True
                    or draft["saved_author_reply"] != ("author_output" in slot)):
                raise ValueError("Historical draft admission or output changed")
            count = physical[draft["messages_hash"]]
            for request in physical_inputs.get(draft["messages_hash"], ()):
                if (request.get("model") != profile["model"]
                        or any(request.get("logical_parameters", {}).get(name) != value
                               for name, value in draft["parameters"].items())):
                    raise ValueError("Historical physical author parameters or model changed")
            if "author_output" in slot:
                key = (draft["messages_hash"], fingerprint(slot["author_output"]))
                if (draft["author_output_hash"] != key[1] or not count or not prompts[key]):
                    raise ValueError("A saved author reply has no exact real request and output")
            draft_receipts.append({"position": index, "messages_hash": draft["messages_hash"],
                "parameters": draft["parameters"], "saved_author_reply": draft["saved_author_reply"],
                "author_output_hash": draft["author_output_hash"],
                "matching_original_physical_requests": count})
        entries.append({"key": row["key"], "legacy_key": row["legacy_key"],
            "group_hash": row["group_hash"], "task_hash": row["complete_task_hash"],
            "material_needs_hash": row["material_needs_hash"],
            "legacy_sha256": _sha(old_bytes), "drafts": draft_receipts})
    manifest = {"version": VERSION, "model": profile["model"],
        "reviewer_model": env["reviewer_model"],
        "discriminator_model": env["discriminator_model"],
        "parent_archive_sha256": archive_sha256, "parent_trace_sha256": trace_sha256,
        "parent_audit_sha256": audit_sha256, "groups": entries}
    manifest["hash"] = fingerprint(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    cache_dir = destination / "05_signal_checkpoints"
    cache_dir.mkdir()
    for key, data in archive_rows.items():
        _io_path(cache_dir / (key + ".json")).write_bytes(data)
    (destination / "05_signal_inheritance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {"version": VERSION, "manifest_hash": manifest["hash"],
        "historical_groups": len(entries),
        "historical_slots": sum(len(row["drafts"]) for row in entries),
        "replies_matching_physical_trace_and_prompt_log": sum(d["saved_author_reply"]
            for row in entries for d in row["drafts"]),
        "provider_calls": 0, "new_author_slots": 0, "semantic_revalidation": False,
        "paid_resume_authorization": False}
    (destination / "inheritance_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "trace", "audit", "destination"):
        parser.add_argument(name, type=Path)
    for name in ("archive_sha256", "trace_sha256", "audit_sha256", "run_prefix"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.trace, args.audit, args.destination,
        archive_sha256=args.archive_sha256, trace_sha256=args.trace_sha256,
        audit_sha256=args.audit_sha256, run_prefix=args.run_prefix)))


if __name__ == "__main__":
    main()
