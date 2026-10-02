"""Disconnected owner-routing checks; fixtures are not new semantic opinions."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.cumulative_owner_route import _checkpoint_name, plan_owner_route
from pipeline.world_review_physical import digest

_previous_config = sys.modules.get("config")
sys.modules["config"] = SimpleNamespace(STRUCTURE_MODEL="offline",
                                         REVIEWER_MODEL="offline")
try:
    from pipeline.world_gen import WORLD_DRAFT_VERSION, _world_digest
finally:
    if _previous_config is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = _previous_config


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


class FakeRun:
    def __init__(self, root):
        self.dir = root
        self.manifest = {"config": {"cumulative_recovery": {
            "version": "explicit-cumulative-source-migration/v1"}}}

    def write(self, name, value):
        (self.dir / name).write_bytes(raw(value))

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def has(self, name):
        return (self.dir / name).is_file()


class OwnerRouteTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        previous_config = sys.modules.get("config")
        sys.modules["config"] = SimpleNamespace(STRUCTURE_MODEL="offline",
                                                 REVIEWER_MODEL="offline")
        def restore_config():
            if previous_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
        self.addCleanup(restore_config)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = FakeRun(Path(self.temp.name))
        wp = {"business_instance_plan": {"units": [{"unit_id": "u_one"}]}}
        self.run.write("01_whitepaper.json", wp)
        self.run.write("00_input.json", {"seed": "synthetic"})
        self.run.write("02_world.json", {"synthetic": True})
        self.run.write("02_world_review.json", {"status": "passed"})
        agent = {"status": "completed", "units": [{"unit_id": "u_one"}],
                 "feedback_history": [], "local_repairs": []}
        agent["checkpoint_hash"] = _world_digest(agent)
        draft = {"version": WORLD_DRAFT_VERSION, "generation_strategy": "agentic",
                 "status": "completed", "existing_base": None, "wp_hash": _world_digest(wp),
                 "candidate_world": {"synthetic": True}, "agent": agent}
        draft["candidate_world_hash"] = _world_digest(draft["candidate_world"])
        draft["draft_hash"] = _world_digest(draft)
        self.run.write("02_world_draft.json", draft)
        self.run.write(_checkpoint_name(wp), agent)
        self.run.write("llm_attempts.jsonl", {"synthetic": True})
        self.run.write("prompts.jsonl", {"synthetic": True})
        self.run.write("00_cumulative_lineage.json", {
            "archive_sha256": "f" * 64,
            "historical_world_review_sha256": sha(
                (self.run.dir / "02_world_review.json").read_bytes())})
        cfg = self.run.manifest["config"]["cumulative_recovery"]
        cfg["lineage_sha256"] = sha((self.run.dir / "00_cumulative_lineage.json").read_bytes())
        self.run.write("00_cumulative_source_migration.json", {
            "status": "complete", "complete_source_migration": True,
            "parent_lineage_sha256": cfg["lineage_sha256"],
            "historical_trace_bytes": 0, "historical_trace_sha256": sha(b""),
            "historical_prompt_bytes": 0, "historical_prompt_sha256": sha(b"")})
        cfg["migration_sha256"] = sha((self.run.dir / "00_cumulative_source_migration.json").read_bytes())
        from pipeline.cumulative_materializer import _encode
        inherited = {name: {"sha256": sha((self.run.dir / name).read_bytes()),
                            "bytes": (self.run.dir / name).stat().st_size}
                     for name in ("02_world_draft.json", _checkpoint_name(wp))}
        self.run.write("00_cumulative_derivation_receipt.json", {
            "archive_sha256": "f" * 64, "inherited_run_files": inherited})
        inherited = self.run.read("00_cumulative_derivation_receipt.json")["inherited_run_files"]
        self.run.write("00_cumulative_materialization_receipt.json", {
            "version": "cumulative-child-materialization/v1",
            "derivation_receipt_sha256": sha((self.run.dir / "00_cumulative_derivation_receipt.json").read_bytes()),
            "original_files": len(inherited),
            "original_file_inventory_sha256": sha(_encode(inherited)),
            "evidence_sha256": {"00_cumulative_lineage.json": cfg["lineage_sha256"],
                                "00_cumulative_source_migration.json": cfg["migration_sha256"]}})
        self.run.write("00_first_world_review_boundary.json", {"synthetic": True})
        cfg["first_request_boundary_sha256"] = sha((self.run.dir / "00_first_world_review_boundary.json").read_bytes())
        review = {"status": "failed", "repair_targets": {
            "intrinsic": [{"entity": "old-entity", "fields": ["old-field"]}], "structure": False}}
        self.run.write("12_post_corpus_world_review.json", review)
        physical = {"review_sha256": digest(review), "new_physical_requests": 1}
        self.run.write("12_post_corpus_world_review_physical.json", physical)
        self.run.write("12_post_corpus_world_review_gate.json", {
            "version": "post-corpus-world-review/v1", "status": "responsibility_repair_required",
            "historical_lineage_sha256": cfg["lineage_sha256"],
            "new_review_sha256": sha((self.run.dir / "12_post_corpus_world_review.json").read_bytes()),
            "new_physical_receipt_sha256": sha((self.run.dir / "12_post_corpus_world_review_physical.json").read_bytes()),
            "first_request_boundary_sha256": cfg["first_request_boundary_sha256"],
            "new_physical_requests": 1, "truth_changed": False,
            "author_repair_started": False, "production_ready": False,
            "provider_calls_by_gate": 0})
        from pipeline import instance_executor, world_semantics, world_state
        patch.object(instance_executor, "review_value_groups", return_value={
            "u_one": [{"code": "world_semantic_review"}]}).start()
        patch.object(world_semantics, "validate_review", return_value=[
            {"code": "world_review_not_passed", "status": "failed"}]).start()
        patch.object(world_state.WorldState, "from_dict", return_value=SimpleNamespace()).start()
        patch("pipeline.cumulative_owner_route.audit_new_review", return_value=physical).start()

    def test_read_only_owner_plan_preserves_history_and_quota(self):
        before = {p.name: sha(p.read_bytes()) for p in self.run.dir.iterdir()}
        plan = plan_owner_route(self.run)
        self.assertEqual(plan["status"], "original_owner_transaction_required")
        self.assertEqual(plan["local_repairs_used"], {"u_one": 0})
        self.assertEqual(plan["original_factory_repair_call_limit"], 4)
        self.assertFalse(plan["author_repair_started"])
        self.assertEqual({p.name: sha(p.read_bytes()) for p in self.run.dir.iterdir()}, before)

    def test_exhausted_original_owner_quota_rejects(self):
        name = _checkpoint_name(self.run.read("01_whitepaper.json"))
        draft = self.run.read("02_world_draft.json")
        agent = draft["agent"]
        agent["local_repairs"] = [{"unit_id": "u_one"}] * 3
        agent["checkpoint_hash"] = _world_digest({k: v for k, v in agent.items()
                                                    if k != "checkpoint_hash"})
        draft["draft_hash"] = _world_digest({k: v for k, v in draft.items() if k != "draft_hash"})
        self.run.write("02_world_draft.json", draft)
        self.run.write(name, agent)
        from pipeline.cumulative_materializer import _encode
        derivation = self.run.read("00_cumulative_derivation_receipt.json")
        inherited = derivation["inherited_run_files"]
        for member in ("02_world_draft.json", name):
            inherited[member] = {"sha256": sha((self.run.dir / member).read_bytes()),
                                 "bytes": (self.run.dir / member).stat().st_size}
        self.run.write("00_cumulative_derivation_receipt.json", derivation)
        inherited = self.run.read("00_cumulative_derivation_receipt.json")["inherited_run_files"]
        materialization = self.run.read("00_cumulative_materialization_receipt.json")
        materialization["derivation_receipt_sha256"] = sha(
            (self.run.dir / "00_cumulative_derivation_receipt.json").read_bytes())
        materialization["original_file_inventory_sha256"] = sha(_encode(inherited))
        self.run.write("00_cumulative_materialization_receipt.json", materialization)
        with self.assertRaisesRegex(ValueError, "allowance is exhausted"):
            plan_owner_route(self.run)

    def test_rehashed_draft_without_original_inventory_rejects(self):
        draft = self.run.read("02_world_draft.json")
        draft["candidate_world"]["synthetic"] = False
        draft["candidate_world_hash"] = _world_digest(draft["candidate_world"])
        draft["draft_hash"] = _world_digest({k: v for k, v in draft.items() if k != "draft_hash"})
        self.run.write("02_world_draft.json", draft)
        with self.assertRaisesRegex(ValueError, "inherited bytes"):
            plan_owner_route(self.run)

    def test_changed_gate_cannot_route(self):
        gate = self.run.read("12_post_corpus_world_review_gate.json")
        gate["new_review_sha256"] = "0" * 64
        self.run.write("12_post_corpus_world_review_gate.json", gate)
        with self.assertRaisesRegex(ValueError, "gate does not admit"):
            plan_owner_route(self.run)


if __name__ == "__main__":
    unittest.main()
