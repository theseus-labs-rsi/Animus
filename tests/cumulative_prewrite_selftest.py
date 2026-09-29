"""A cumulative child cannot be opened until history and authority agree."""
from collections import Counter
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.cumulative_prewrite import (PrewritePermit, is_valid_permit,
                                          validate_prewrite)
from tools.audit_cumulative_migration_scope import classify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class CumulativePrewriteTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "repo"
        self.source.mkdir()
        self.derived = self.root / "derived"
        run = self.derived / "run"
        run.mkdir(parents=True)
        evidence_dir = self.derived / "evidence"
        evidence_dir.mkdir()
        self.run_id = "oldrun"
        self.destination = self.source / "output" / "runs" / self.run_id
        self.archive = "f" * 64
        self.head, self.tree = "a" * 40, "b" * 40
        src = self.source / "source.py"
        src.write_text("ORIGINAL = 1\n", encoding="utf-8")
        self.source_hash = sha(src)
        profile = {"stopped": True, "admitted_calls": 1026,
                   "settled_usage_calls": 1026, "unsettled_reservation_calls": 0,
                   "reported_usage_estimate_cny": "34.796145028",
                   "budget_consumed_cny": 34.796145028, "max_calls": 1200,
                   "max_cny": 35, "model": "dummy", "source_drift": [],
                   "stop_reason": {"reason": "estimated_budget_limit"}}
        save(run / "experiment_profile.json", profile)
        (run / "llm_attempts.jsonl").write_bytes(b"old trace\n")
        (run / "prompts.jsonl").write_bytes(b"old prompts\n")
        save(run / "manifest.json", {"run_id": self.run_id,
             "scenario": "seed-showcase", "status": "failed", "config": {},
             "terminal_reconciliation": True})
        save(run / "02_world_agent_5903517308903f10c8b0.json", {"old": True})
        for name in ("02_world_review.json", "02_world_review_attempts.json",
                     "11_production.json", "experiment_source_hashes.json",
                     "experiment_source_hashes_at_end.json"):
            save(run / name, {"historical": name})
        save(evidence_dir / "parent_manifest.json", {"run_id": self.run_id, "status": "failed"})
        existing = list(run.iterdir())
        for number in range(348 - len(existing)):
            (run / f"history_{number:03}.json").write_bytes(b"{}\n")
        inherited = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                     for p in run.iterdir()}
        self.assertEqual(len(inherited), 348)
        receipt = {"archive_sha256": self.archive, "inherited_run_files": inherited,
                   "inherited_run_bytes": sum(x["bytes"] for x in inherited.values()),
                   "parent_manifest_sha256": sha(evidence_dir / "parent_manifest.json"),
                   "reconciled_manifest_sha256": sha(run / "manifest.json"),
                   "runnable_run_created": False, "admission_reopened": False}
        receipt_path = save(evidence_dir / "derivation_receipt.json", receipt)
        self.paths = {}
        def put(name, value):
            self.paths[name] = save(self.root / (name + ".json"), value)
            return self.paths[name]
        budget = {"admitted_calls": 1026, "model": "dummy",
                  "settled_usage_calls": 1026,
                  "profile_sha256": sha(run / "experiment_profile.json"),
                  "trace_sha256": sha(run / "llm_attempts.jsonl"),
                  "reported_usage_estimate_cny": "34.796145028",
                  "max_calls": 1200, "max_cny": "35"}
        lineage = {"status": "historical_lineage_verified", "archive_sha256": self.archive,
                   "provider_calls": 0, "budget": budget,
                   "historical_prompt_sha256": sha(run / "prompts.jsonl"),
                   "historical_prompt_bytes": (run / "prompts.jsonl").stat().st_size,
                   "historical_trace_bytes": (run / "llm_attempts.jsonl").stat().st_size,
                   "historical_world_review_sha256": sha(run / "02_world_review.json"),
                   "historical_world_review_attempts_sha256": sha(run / "02_world_review_attempts.json"),
                   "historical_production_sha256": sha(run / "11_production.json"),
                   "initial_source_map_sha256": sha(run / "experiment_source_hashes.json"),
                   "ending_source_map_sha256": sha(run / "experiment_source_hashes_at_end.json"),
                   "identity": {"run_id": self.run_id, "scenario": "seed-showcase",
                                "production_identity": {"seed": "old"}}}
        lineage_path = put("lineage", lineage)
        stage = {"status": "historical_stage_inventory_verified",
                 "parent_lineage_sha256": sha(lineage_path),
                 "derivation_receipt_sha256": sha(receipt_path),
                 "production": {"identity": {"seed": "old"}},
                 "historical_artifact_sha256": {
                     "02_world_agent_5903517308903f10c8b0.json":
                     inherited["02_world_agent_5903517308903f10c8b0.json"]["sha256"]}}
        stage_path = put("stage", stage)
        epoch = {"status": "source_epochs_inventoried",
                 "historical_lineage_sha256": sha(lineage_path),
                 "target_fully_committed": True, "target_uncommitted_source_paths": [],
                 "target_head": self.head, "target_tree": self.tree,
                 "target_source_hashes": {"source.py": self.source_hash}}
        epoch_path = put("epoch", epoch)
        world = {"version": "world-checkpoint-source-migration/v1",
                 "provider_calls": 0, "all_original_fields_retained": True,
                 "semantic_revalidation": False, "publication_or_signal_migration": False,
                 "paid_resume_authorization": False,
                 "parent_sha256": inherited["02_world_agent_5903517308903f10c8b0.json"]["sha256"],
                 "target_compiler_source_hashes": {"source.py": self.source_hash}}
        world_path = put("world", world)
        categories = Counter(classify(name) for name in inherited)
        scope = {"status": "historical_migration_scope_classified",
                 "parent_lineage_sha256": sha(lineage_path),
                 "historical_stage_inventory_sha256": sha(stage_path),
                 "source_epoch_sha256": sha(epoch_path),
                 "world_migration_sha256": sha(world_path),
                 "derivation_receipt_sha256": sha(receipt_path),
                 "historical_file_count": 348,
                 "historical_file_bytes": receipt["inherited_run_bytes"],
                 "complete_source_migration": False,
                 "paid_resume_authorization": False, "provider_calls": 0,
                 "target_head": self.head, "target_tree": self.tree,
                 "migration_scope_counts": dict(categories),
                 "files": {name: {**entry, "migration_scope": classify(name)}
                           for name, entry in inherited.items()}}
        scope_path = put("scope", scope)
        migration = {"version": "complete-cumulative-source-migration/v1",
                     "status": "complete", "complete_source_migration": True,
                     "parent_lineage_sha256": sha(lineage_path),
                     "historical_stage_inventory_sha256": sha(stage_path),
                     "source_epoch_sha256": sha(epoch_path),
                     "migration_scope_sha256": sha(scope_path),
                     "world_migration_sha256": sha(world_path),
                     "target_head": self.head, "target_tree": self.tree,
                     "target_source_hashes": epoch["target_source_hashes"],
                     "provider_calls": 0, "downstream_results_current": False,
                     "stage_revalidation_completed": False, "new_world_opinion": False,
                     "paid_resume_authorization": False}
        migration.update({
            "historical_profile_sha256": budget["profile_sha256"],
            "historical_trace_sha256": budget["trace_sha256"],
            "historical_trace_bytes": lineage["historical_trace_bytes"],
            "historical_prompt_sha256": lineage["historical_prompt_sha256"],
            "historical_prompt_bytes": lineage["historical_prompt_bytes"],
            "historical_review_sha256": lineage["historical_world_review_sha256"],
            "historical_review_attempts_sha256": lineage["historical_world_review_attempts_sha256"],
            "historical_production_sha256": lineage["historical_production_sha256"],
            "initial_source_map_sha256": lineage["initial_source_map_sha256"],
            "ending_source_map_sha256": lineage["ending_source_map_sha256"],
            "historical_file_count": 348,
            "historical_file_bytes": receipt["inherited_run_bytes"],
            "first_new_operation": "post_corpus_world.semantic_review"})
        put("migration", migration)
        authorization = {"policy": "explicit-cumulative-cap-authorization/v1",
                         "user_authorization": True,
                         "parent_profile_sha256": budget["profile_sha256"],
                         "parent_trace_sha256": budget["trace_sha256"],
                         "model": "dummy", "old_max_calls": 1200, "old_max_cny": 35,
                         "new_max_calls": 1200, "new_max_cny": 40}
        put("authorization", authorization)
        evidence = {"envelope_hash": "e" * 64, "new_world_opinion": False,
                    "paid_resume_authorization": False}
        put("evidence", evidence)
        boundary = {"version": "derived-run-first-review-boundary/v1",
                    "provider_calls": 0, "new_semantic_opinion": False,
                    "budget_stop_preserved": True, "first_step": "world.semantic_review",
                    "derived_manifest_sha256": receipt["reconciled_manifest_sha256"],
                    "ledger_sha256": budget["profile_sha256"],
                    "trace_sha256": budget["trace_sha256"],
                    "prompt_archive_sha256": lineage["historical_prompt_sha256"],
                    "evidence_hash": evidence["envelope_hash"], "model": "dummy",
                    "messages_hash": "c" * 64, "params_hash": "d" * 64}
        put("boundary", boundary)
        authorization.update(migration_sha256=sha(self.paths["migration"]),
                             first_request_boundary_sha256=sha(self.paths["boundary"]),
                             allowed_first_step="world.semantic_review")
        save(self.paths["authorization"], authorization)

    def validate(self, **overrides):
        args = {"derived_root": self.derived, "destination": self.destination,
                "source_root": self.source, "lineage_path": self.paths["lineage"],
                "stage_path": self.paths["stage"], "source_epoch_path": self.paths["epoch"],
                "scope_path": self.paths["scope"], "migration_path": self.paths["migration"],
                "authorization_path": self.paths["authorization"],
                "boundary_path": self.paths["boundary"],
                "world_migration_path": self.paths["world"],
                "downstream_evidence_path": self.paths["evidence"],
                "expected_archive_sha256": self.archive}
        args.update(overrides)
        def git(command, **_kwargs):
            return ({"HEAD": self.head, "HEAD^{tree}": self.tree}.get(command[2], "") + "\n").encode() \
                if command[:2] == ["git", "rev-parse"] else b""
        with patch("pipeline.cumulative_prewrite.subprocess.check_output", side_effect=git):
            return validate_prewrite(**args)

    def test_happy_path_is_read_only_and_sealed(self):
        permit = self.validate()
        self.assertTrue(is_valid_permit(permit, self.run_id))
        self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ValueError, "requires validated evidence"):
            PrewritePermit(self.run_id, self.destination, "a" * 64, "b" * 64)

    def test_missing_authorization_and_source_drift_fail_before_child_creation(self):
        self.paths["authorization"].unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.assertFalse(self.destination.exists())
        save(self.paths["authorization"], {"user_authorization": False})
        with self.assertRaisesRegex(ValueError, "authorization"):
            self.validate()
        self.assertFalse(self.destination.exists())
        self.source.joinpath("source.py").write_text("CHANGED\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.validate()

    def test_frozen_file_and_archive_changes_fail_without_writing(self):
        with self.assertRaisesRegex(ValueError, "lineage or migration scope"):
            self.validate(expected_archive_sha256="0" * 64)
        self.derived.joinpath("run", "history_000.json").write_bytes(b"forged\n")
        with self.assertRaisesRegex(ValueError, "Historical run file changed"):
            self.validate()
        self.assertFalse(self.destination.exists())

    def test_cap_authorization_cannot_be_reused_for_another_migration(self):
        authorization = json.loads(self.paths["authorization"].read_text(encoding="utf-8"))
        authorization["migration_sha256"] = "0" * 64
        save(self.paths["authorization"], authorization)
        with self.assertRaisesRegex(ValueError, "authorization"):
            self.validate()
        self.assertFalse(self.destination.exists())

    def test_materializes_isolated_child_with_frozen_history(self):
        from pipeline.cumulative_materializer import materialize_child
        original_manifest = (self.derived / "run" / "manifest.json").read_bytes()
        original_profile = (self.derived / "run" / "experiment_profile.json").read_bytes()
        args = {"derived_root": self.derived, "destination": self.destination,
                "source_root": self.source, "lineage_path": self.paths["lineage"],
                "stage_path": self.paths["stage"], "source_epoch_path": self.paths["epoch"],
                "scope_path": self.paths["scope"], "migration_path": self.paths["migration"],
                "authorization_path": self.paths["authorization"],
                "boundary_path": self.paths["boundary"],
                "world_migration_path": self.paths["world"],
                "downstream_evidence_path": self.paths["evidence"],
                "expected_archive_sha256": self.archive}
        def git(command, **_kwargs):
            return ({"HEAD": self.head, "HEAD^{tree}": self.tree}.get(command[2], "") + "\n").encode() \
                if command[:2] == ["git", "rev-parse"] else b""
        with patch("pipeline.cumulative_prewrite.subprocess.check_output", side_effect=git):
            result = materialize_child(**args)
        self.assertEqual(result.destination, self.destination.resolve())
        self.assertTrue(is_valid_permit(result.permit, self.run_id))
        from pipeline.cumulative_entry_guard import (check_before_run_init,
            enter_bounded_run, leave_bounded_run, enter_explicit_cumulative_run,
            leave_explicit_cumulative_run)
        with self.assertRaisesRegex(ValueError, "dedicated entry"):
            check_before_run_init(result.destination / "manifest.json")
        bounded = enter_bounded_run(self.run_id)
        try:
            explicit = enter_explicit_cumulative_run(result.permit)
            try:
                check_before_run_init(result.destination / "manifest.json")
            finally:
                leave_explicit_cumulative_run(explicit)
        finally:
            leave_bounded_run(bounded)
        result = result.destination
        self.assertEqual((self.derived / "run" / "manifest.json").read_bytes(), original_manifest)
        self.assertEqual((self.derived / "run" / "experiment_profile.json").read_bytes(), original_profile)
        child_manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
        child_profile = json.loads((result / "experiment_profile.json").read_text(encoding="utf-8"))
        self.assertEqual(child_manifest["run_id"], self.run_id)
        self.assertTrue(child_manifest["terminal_reconciliation"])
        self.assertEqual(child_profile["admitted_calls"], 1026)
        self.assertEqual(child_profile["settled_usage_calls"], 1026)
        self.assertEqual(child_profile["max_cny"], 40)
        self.assertFalse(child_profile["stopped"])
        self.assertEqual((result / "cumulative_history" / "manifest_before_cumulative.json").read_bytes(),
                         original_manifest)
        self.assertEqual((result / "cumulative_history" / "experiment_profile_before_cumulative.json").read_bytes(),
                         original_profile)
        self.assertEqual((result / "00_cumulative_derivation_receipt.json").read_bytes(),
                         (self.derived / "evidence" / "derivation_receipt.json").read_bytes())
        for name in ("02_world_review.json", "02_world_review_attempts.json",
                     "11_production.json", "llm_attempts.jsonl", "prompts.jsonl",
                     "experiment_source_hashes.json", "experiment_source_hashes_at_end.json"):
            self.assertEqual((result / name).read_bytes(), (self.derived / "run" / name).read_bytes())
        receipt = json.loads((result / "00_cumulative_materialization_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["original_files"], 348)
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertFalse(receipt["experiment_target_passed"])

    def test_materializer_refuses_absent_cap_authority_before_write(self):
        from pipeline.cumulative_materializer import materialize_child
        self.paths["authorization"].unlink()
        args = {"derived_root": self.derived, "destination": self.destination,
                "source_root": self.source, "lineage_path": self.paths["lineage"],
                "stage_path": self.paths["stage"], "source_epoch_path": self.paths["epoch"],
                "scope_path": self.paths["scope"], "migration_path": self.paths["migration"],
                "authorization_path": self.paths["authorization"],
                "boundary_path": self.paths["boundary"],
                "world_migration_path": self.paths["world"],
                "downstream_evidence_path": self.paths["evidence"],
                "expected_archive_sha256": self.archive}
        with self.assertRaises(FileNotFoundError):
            materialize_child(**args)
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.source / "output" / "runs").exists())

    def test_pristine_published_child_can_reenter_without_writes(self):
        from pipeline.cumulative_materializer import materialize_child, recover_pristine_child
        args = {"derived_root": self.derived, "destination": self.destination,
                "source_root": self.source, "lineage_path": self.paths["lineage"],
                "stage_path": self.paths["stage"], "source_epoch_path": self.paths["epoch"],
                "scope_path": self.paths["scope"], "migration_path": self.paths["migration"],
                "authorization_path": self.paths["authorization"],
                "boundary_path": self.paths["boundary"],
                "world_migration_path": self.paths["world"],
                "downstream_evidence_path": self.paths["evidence"],
                "expected_archive_sha256": self.archive}
        def git(command, **_kwargs):
            return ({"HEAD": self.head, "HEAD^{tree}": self.tree}.get(command[2], "") + "\n").encode() \
                if command[:2] == ["git", "rev-parse"] else b""
        with patch("pipeline.cumulative_prewrite.subprocess.check_output", side_effect=git):
            created = materialize_child(**args)
            before = {name: sha(self.destination / name) for name in (
                "manifest.json", "experiment_profile.json", "llm_attempts.jsonl",
                "00_cumulative_materialization_receipt.json",
                "00_cumulative_derivation_receipt.json")}
            recovered = recover_pristine_child(**args)
            self.assertEqual(recovered.destination, created.destination)
            self.assertTrue(is_valid_permit(recovered.permit, self.run_id))
            self.assertEqual(before, {name: sha(self.destination / name) for name in before})
            derivation_path = self.destination / "00_cumulative_derivation_receipt.json"
            derivation_raw = derivation_path.read_bytes()
            derivation_path.write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "changed before its first execution"):
                recover_pristine_child(**args)
            derivation_path.write_bytes(derivation_raw)
            (self.destination / "12_post_corpus_world_review.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected execution files"):
                recover_pristine_child(**args)
            (self.destination / "12_post_corpus_world_review.json").unlink()
            (self.destination / "experiment_profile.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed before its first execution"):
                recover_pristine_child(**args)

    def test_materializer_discards_only_its_private_partial_copy(self):
        from pipeline import cumulative_materializer
        original = cumulative_materializer._copy_verified
        def interrupted(source, target, expected_sha, expected_bytes):
            if source.name == "history_000.json":
                raise RuntimeError("simulated interrupted historical copy")
            return original(source, target, expected_sha, expected_bytes)
        args = {"derived_root": self.derived, "destination": self.destination,
                "source_root": self.source, "lineage_path": self.paths["lineage"],
                "stage_path": self.paths["stage"], "source_epoch_path": self.paths["epoch"],
                "scope_path": self.paths["scope"], "migration_path": self.paths["migration"],
                "authorization_path": self.paths["authorization"],
                "boundary_path": self.paths["boundary"],
                "world_migration_path": self.paths["world"],
                "downstream_evidence_path": self.paths["evidence"],
                "expected_archive_sha256": self.archive}
        def git(command, **_kwargs):
            return ({"HEAD": self.head, "HEAD^{tree}": self.tree}.get(command[2], "") + "\n").encode() \
                if command[:2] == ["git", "rev-parse"] else b""
        old_trace_sha = sha(self.derived / "run" / "llm_attempts.jsonl")
        with patch("pipeline.cumulative_prewrite.subprocess.check_output", side_effect=git), \
             patch("pipeline.cumulative_materializer._copy_verified", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted historical copy"):
                cumulative_materializer.materialize_child(**args)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob(".oldrun.cumulative-*.staging")), [])
        self.assertEqual(sha(self.derived / "run" / "llm_attempts.jsonl"), old_trace_sha)

    def test_dedicated_wrapper_denies_missing_control_before_run_write(self):
        from tools.run_original_bc_smoke import main
        absent = self.root / "nonexistent-cumulative-control.json"
        argv = ["run_original_bc_smoke.py", "--run", "synthetic-unused-child",
                "--cumulative-child-config", str(absent)]
        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(absent.exists())
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
