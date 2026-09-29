"""Synthetic, disconnected control-flow checks; no semantic or paid acceptance."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_semantics_selftest import fixture, positive, negative
from pipeline import world_semantics
from pipeline.post_corpus_world_review import (run_once, LINEAGE, MIGRATION, EVIDENCE,
    AUTHORIZATION, BOUNDARY, STAGE_INVENTORY, SOURCE_EPOCH,
    HISTORICAL_STAGE_FILES, REVIEW, PHYSICAL, GATE)
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from pipeline.cumulative_review_reconcile import audit_saved_review
from pipeline.world_review_physical import digest
from tools.run_original_bc_smoke import implementation_hashes


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


class FakeRun:
    def __init__(self, root):
        self.dir = root
        self.manifest = {"run_id": "synthetic-cumulative", "config": {}}
        self.tracer = None

    def has(self, name):
        return (self.dir / name).is_file()

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.dir / name).write_bytes(raw(value))


class PhysicalFixtureTracer:
    def __init__(self, root, output):
        self.pfile = root / "prompts.jsonl"
        self.trace = root / "llm_attempts.jsonl"
        self.output = output
        self.calls = 0

    def chat_json(self, step, messages, **params):
        self.calls += 1
        request = {"event": "request", "step": step, "call_id": "new-1",
                   "operation_id": "new-op", "messages": messages,
                   "model": params["model"], "parameters": params}
        response = {"event": "response", "step": step, "call_id": "new-1",
                    "operation_id": "new-op", "response": {"usage": {
                        "prompt_tokens": 10, "completion_tokens": 20}}}
        result = {"event": "json_result", "step": step,
                  "operation_id": "new-op", "parsed": self.output}
        with self.trace.open("ab") as stream:
            for row in (request, response, result):
                stream.write(raw(row))
        with self.pfile.open("ab") as stream:
            stream.write(raw({"step": step, "messages": messages, "output": self.output,
                "params": {key: value for key, value in params.items()
                           if key in ("model", "temperature", "max_tokens", "retries",
                                      "strict_json", "complete_containers")}}))
        return self.output


class PostCorpusWorldReviewTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(REVIEWER_MODEL="dummy")}).start()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = FakeRun(self.root)
        wp, ws, task = fixture()
        self.run.write("01_whitepaper.json", wp)
        self.run.write("02_world.json", ws.to_dict())
        self.run.write("00_input.json", task)
        class PreviousTracer:
            def chat_json(self, *_args, **_kwargs):
                return positive()
        previous = world_semantics.review_world(wp, ws, PreviousTracer(), task_input=task)
        self.assertEqual(previous["status"], "passed")
        self.run.write("02_world_review.json", previous)
        self.run.write("02_world_review_attempts.json", {"attempts": [previous]})
        self.run.write("11_production.json", {"identity": {"seed": "synthetic"},
            "status": "execution_error", "history": []})
        self.run.write("experiment_source_hashes.json", {"legacy.py": "d" * 64})
        self.run.write("experiment_source_hashes_at_end.json", {"legacy.py": "d" * 64})
        for name in HISTORICAL_STAGE_FILES:
            if not self.run.has(name):
                self.run.write(name, {"synthetic_historical_artifact": name})
        body = {"version": "verified-downstream-evidence/v1", "offline_only": True,
                "new_provider_calls": 0, "new_world_opinion": False,
                "paid_resume_authorization": False,
                "world_hash": world_semantics._hash(ws.to_dict()),
                "whitepaper_hash": world_semantics._hash(wp),
                "original_task_hash": world_semantics._hash(task),
                "historical_passed_review_hash": world_semantics._hash(previous),
                "historical_world_review_status": "passed",
                "downstream_negative_reviews": [{"call_id": "old-negative",
                    "negative_opinion": {"verdict": "fail"}}]}
        self.run.write(EVIDENCE, {**body, "envelope_hash": world_semantics._hash(body)})
        class CaptureFirstRequest:
            pfile = None
            first = None
            def chat_json(self, step, messages, **params):
                if self.first is None:
                    self.first = (step, messages, params)
                return negative()
        capture = CaptureFirstRequest()
        from pipeline.world_state import WorldState
        preview = world_semantics.review_world(self.run.read("01_whitepaper.json"),
            WorldState.from_dict(self.run.read("02_world.json")), capture,
            task_input=self.run.read("00_input.json"),
            previous=self.run.read("02_world_review.json"),
            downstream_evidence=self.run.read(EVIDENCE))
        self.assertEqual(preview["status"], "failed")
        step, messages, params = capture.first
        trace = raw({"event": "request", "call_id": "historical"})
        prompt = raw({"step": "old", "output": {"status": "historical"}})
        (self.root / "llm_attempts.jsonl").write_bytes(trace)
        (self.root / "prompts.jsonl").write_bytes(prompt)
        lineage = {"status": "historical_lineage_verified",
                   "complete_source_migration": False, "runnable_run_created": False,
                   "historical_copy": {"reconciled_manifest_sha256": "f" * 64},
                   "identity": {"run_id": "synthetic-cumulative",
                                "layout_revisions": 3,
                                "production_identity": {"seed": "synthetic"}},
                   "historical_trace_bytes": len(trace),
                   "historical_prompt_bytes": len(prompt),
                   "historical_prompt_sha256": sha(prompt),
                   "historical_world_review_sha256": sha((self.root / "02_world_review.json").read_bytes()),
                   "historical_world_review_attempts_sha256": sha((self.root / "02_world_review_attempts.json").read_bytes()),
                   "historical_production_sha256": sha((self.root / "11_production.json").read_bytes()),
                   "initial_source_map_sha256": sha((self.root / "experiment_source_hashes.json").read_bytes()),
                   "ending_source_map_sha256": sha((self.root / "experiment_source_hashes_at_end.json").read_bytes()),
                   "budget": {"admitted_calls": 1, "max_calls": 2, "max_cny": "5",
                              "model": "dummy",
                              "profile_sha256": "a" * 64, "trace_sha256": sha(trace),
                              "reported_usage_estimate_cny": "0.1"}}
        self.run.write(LINEAGE, lineage)
        target_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        target_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT).decode().strip()
        self.run.write(STAGE_INVENTORY, {
            "version": "cumulative-stage-retention-audit/v1",
            "status": "historical_stage_inventory_verified",
            "parent_lineage_sha256": sha((self.root / LINEAGE).read_bytes()),
            "provider_calls": 0, "complete_source_migration": False,
            "runnable_run_created": False, "business_truth_certified": False,
            "new_world_opinion": False,
            "production": {"identity": {"seed": "synthetic"}},
            "corpus": {"release_eligible": False},
            "author_slots": {"attempts_reset": False},
            "historical_artifact_sha256": {
                name: sha((self.root / name).read_bytes()) for name in HISTORICAL_STAGE_FILES}})
        self.run.write(SOURCE_EPOCH, {
            "version": "cumulative-source-epoch-audit/v1",
            "status": "source_epochs_inventoried",
            "historical_lineage_sha256": sha((self.root / LINEAGE).read_bytes()),
            "legacy_source_map_sha256": lineage["ending_source_map_sha256"],
            "target_fully_committed": True, "target_uncommitted_source_paths": [],
            "complete_source_migration": False, "provider_calls": 0,
            "target_head": target_head, "target_tree": target_tree,
            "target_source_hashes": implementation_hashes()})
        self.run.write(BOUNDARY, {"version": "derived-run-first-review-boundary/v1",
            "provider_calls": 0, "new_semantic_opinion": False,
            "budget_stop_preserved": True, "first_step": step,
            "derived_manifest_sha256": "f" * 64,
            "ledger_sha256": lineage["budget"]["profile_sha256"],
            "trace_sha256": sha(trace), "prompt_archive_sha256": sha(prompt),
            "evidence_hash": self.run.read(EVIDENCE)["envelope_hash"],
            "model": "dummy", "messages_hash": digest(messages),
            "params_hash": digest(params)})
        migration = {"version": "complete-cumulative-source-migration/v1",
                     "status": "complete", "complete_source_migration": True,
                     "historical_attempt_counters_preserved": True,
                     "first_new_operation": "post_corpus_world.semantic_review",
                     "stage_revalidation_completed": False,
                     "downstream_results_current": False,
                     "new_world_opinion": False,
                     "paid_resume_authorization": False,
                     "parent_lineage_sha256": sha((self.root / LINEAGE).read_bytes()),
                     "historical_profile_sha256": "a" * 64,
                     "historical_trace_sha256": sha(trace), "historical_trace_bytes": len(trace),
                     "historical_prompt_sha256": sha(prompt), "historical_prompt_bytes": len(prompt),
                     "historical_review_sha256": lineage["historical_world_review_sha256"],
                     "historical_review_attempts_sha256": lineage["historical_world_review_attempts_sha256"],
                     "historical_production_sha256": lineage["historical_production_sha256"],
                     "initial_source_map_sha256": lineage["initial_source_map_sha256"],
                     "ending_source_map_sha256": lineage["ending_source_map_sha256"],
                     "target_source_hashes": implementation_hashes(), "provider_calls": 0,
                     "historical_stage_inventory_sha256": sha((self.root / STAGE_INVENTORY).read_bytes()),
                     "source_epoch_sha256": sha((self.root / SOURCE_EPOCH).read_bytes()),
                     "target_head": target_head, "target_tree": target_tree}
        self.run.write(MIGRATION, migration)
        authorization = {"policy": "explicit-cumulative-cap-authorization/v1",
                         "user_authorization": True, "parent_profile_sha256": "a" * 64,
                         "parent_trace_sha256": sha(trace), "old_max_calls": 2,
                         "old_max_cny": 5, "new_max_calls": 3, "new_max_cny": 6,
                         "model": "dummy",
                         "migration_sha256": sha((self.root / MIGRATION).read_bytes()),
                         "first_request_boundary_sha256": sha((self.root / BOUNDARY).read_bytes()),
                         "allowed_first_step": "world.semantic_review"}
        self.run.write(AUTHORIZATION, authorization)
        self.run.write("experiment_profile.json", {"stopped": False, "source_drift": [],
            "unsettled_reservation_calls": 0, "admitted_calls": 1, "settled_usage_calls": 1,
            "max_calls": 3, "max_cny": 6, "reported_usage_estimate_cny": 0.1,
            "model": "dummy"})
        self.run.manifest["config"]["cumulative_recovery"] = {
            "version": "explicit-cumulative-source-migration/v1",
            "lineage_sha256": sha((self.root / LINEAGE).read_bytes()),
            "migration_sha256": sha((self.root / MIGRATION).read_bytes()),
            "evidence_sha256": sha((self.root / EVIDENCE).read_bytes()),
            "authorization_sha256": sha((self.root / AUTHORIZATION).read_bytes()),
            "first_request_boundary_sha256": sha((self.root / BOUNDARY).read_bytes()),
            "stage_inventory_sha256": sha((self.root / STAGE_INVENTORY).read_bytes()),
            "source_epoch_sha256": sha((self.root / SOURCE_EPOCH).read_bytes())}
        self.run.tracer = PhysicalFixtureTracer(self.root, negative())

    def run_in_bounded_fixture(self):
        token = enter_bounded_run("synthetic-cumulative")
        try:
            return run_once(self.run)
        finally:
            leave_bounded_run(token)

    def rebind_synthetic_authorization_to_boundary(self):
        authorization = self.run.read(AUTHORIZATION)
        authorization["first_request_boundary_sha256"] = sha((self.root / BOUNDARY).read_bytes())
        self.run.write(AUTHORIZATION, authorization)
        self.run.manifest["config"]["cumulative_recovery"]["authorization_sha256"] = sha(
            (self.root / AUTHORIZATION).read_bytes())

    def test_direct_review_without_bounded_wrapper_stops_before_tracer(self):
        with self.assertRaisesRegex(ValueError, "active bounded call wrapper"):
            run_once(self.run)
        self.assertEqual(self.run.tracer.calls, 0)

    def test_bound_repair_stops_at_existing_author_transaction(self):
        old_review_sha = sha((self.root / "02_world_review.json").read_bytes())
        gate = self.run_in_bounded_fixture()
        self.assertEqual(gate["status"], "responsibility_repair_required")
        self.assertEqual(gate["new_physical_requests"], 1)
        self.assertFalse(gate["production_ready"])
        self.assertEqual(gate["first_request_boundary_sha256"], sha((self.root / BOUNDARY).read_bytes()))
        self.assertEqual(self.run.tracer.calls, 1)
        self.assertEqual(sha((self.root / "02_world_review.json").read_bytes()), old_review_sha)
        self.assertTrue(self.run.has(REVIEW) and self.run.has(PHYSICAL) and self.run.has(GATE))
        with self.assertRaisesRegex(ValueError, "existing post-corpus review"):
            self.run_in_bounded_fixture()

    def test_conservative_float_cap_stays_under_explicit_authority(self):
        profile = self.run.read("experiment_profile.json")
        profile["max_cny"] = 5.9
        self.run.write("experiment_profile.json", profile)
        gate = self.run_in_bounded_fixture()
        self.assertEqual(gate["new_physical_requests"], 1)

    def test_runner_cap_above_authorized_amount_stops_before_tracer(self):
        profile = self.run.read("experiment_profile.json")
        profile["max_cny"] = 6.1
        self.run.write("experiment_profile.json", profile)
        with self.assertRaisesRegex(ValueError, "matching explicit cumulative cap"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_structure_feedback_cannot_reopen_exhausted_layout_transaction(self):
        output = negative()
        output["repair_targets"] = {"intrinsic": [], "structure": True}
        self.run.tracer.output = output
        gate = self.run_in_bounded_fixture()
        self.assertEqual(gate["status"], "original_layout_allowance_exhausted")
        self.assertEqual(gate["historical_layout_revisions"], 3)
        self.assertEqual(gate["new_physical_requests"], 1)
        self.assertFalse(gate["author_repair_started"])
        self.assertFalse(gate["production_ready"])

    def test_missing_authorization_stops_before_tracer(self):
        (self.root / AUTHORIZATION).unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_incomplete_source_migration_stops_before_tracer(self):
        migration = self.run.read(MIGRATION)
        migration["complete_source_migration"] = False
        self.run.write(MIGRATION, migration)
        migration_sha = sha((self.root / MIGRATION).read_bytes())
        self.run.manifest["config"]["cumulative_recovery"]["migration_sha256"] = migration_sha
        authorization = self.run.read(AUTHORIZATION)
        authorization["migration_sha256"] = migration_sha
        self.run.write(AUTHORIZATION, authorization)
        self.run.manifest["config"]["cumulative_recovery"]["authorization_sha256"] = sha(
            (self.root / AUTHORIZATION).read_bytes())
        with self.assertRaisesRegex(ValueError, "Full cumulative source migration is not proved"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_closed_budget_stops_before_tracer(self):
        profile = self.run.read("experiment_profile.json")
        profile["stopped"] = True
        self.run.write("experiment_profile.json", profile)
        with self.assertRaisesRegex(ValueError, "stopped, changed, or unsettled"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_stale_downstream_evidence_stops_before_tracer(self):
        evidence = self.run.read(EVIDENCE)
        evidence["world_hash"] = "0" * 64
        evidence["envelope_hash"] = world_semantics._hash(
            {key: value for key, value in evidence.items() if key != "envelope_hash"})
        self.run.write(EVIDENCE, evidence)
        self.run.manifest["config"]["cumulative_recovery"]["evidence_sha256"] = sha(
            (self.root / EVIDENCE).read_bytes())
        boundary = self.run.read(BOUNDARY)
        boundary["evidence_hash"] = evidence["envelope_hash"]
        self.run.write(BOUNDARY, boundary)
        self.run.manifest["config"]["cumulative_recovery"]["first_request_boundary_sha256"] = sha(
            (self.root / BOUNDARY).read_bytes())
        self.rebind_synthetic_authorization_to_boundary()
        with self.assertRaisesRegex(ValueError, "does not bind current world inputs"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_historical_attempt_record_tamper_stops_before_tracer(self):
        attempts = self.run.read("02_world_review_attempts.json")
        attempts["attempts"].append({"status": "passed", "forged": True})
        self.run.write("02_world_review_attempts.json", attempts)
        with self.assertRaisesRegex(ValueError, "Historical stage artifact changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_missing_stage_inventory_stops_before_tracer(self):
        (self.root / STAGE_INVENTORY).unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_changed_stage_artifact_stops_before_tracer(self):
        self.run.write("05_corpus_warning.json", {"release_eligible": True})
        with self.assertRaisesRegex(ValueError, "Historical stage artifact changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_missing_source_epoch_stops_before_tracer(self):
        (self.root / SOURCE_EPOCH).unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_dirty_target_source_stops_before_tracer(self):
        real_check_output = subprocess.check_output
        def fake_check_output(args, **kwargs):
            if args[:2] == ["git", "status"]:
                return b" M pipeline/factory.py\n"
            return real_check_output(args, **kwargs)
        with patch.object(subprocess, "check_output", side_effect=fake_check_output):
            with self.assertRaisesRegex(ValueError, "uncommitted changes"):
                self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_historical_source_map_tamper_stops_before_tracer(self):
        self.run.write("experiment_source_hashes_at_end.json", {"legacy.py": "e" * 64})
        with self.assertRaisesRegex(ValueError, "Historical production identity or world opinion changed"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_nonfinite_cap_stops_before_tracer(self):
        authorization = self.run.read(AUTHORIZATION)
        authorization["new_max_cny"] = "NaN"
        self.run.write(AUTHORIZATION, authorization)
        self.run.manifest["config"]["cumulative_recovery"]["authorization_sha256"] = sha(
            (self.root / AUTHORIZATION).read_bytes())
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)

    def test_changed_first_request_cannot_reach_tracer(self):
        boundary = self.run.read(BOUNDARY)
        boundary["messages_hash"] = "0" * 64
        self.run.write(BOUNDARY, boundary)
        self.run.manifest["config"]["cumulative_recovery"]["first_request_boundary_sha256"] = sha(
            (self.root / BOUNDARY).read_bytes())
        self.rebind_synthetic_authorization_to_boundary()
        with self.assertRaisesRegex(ValueError, "did not produce an opinion"):
            self.run_in_bounded_fixture()
        self.assertEqual(self.run.tracer.calls, 0)
        self.assertTrue(self.run.has(REVIEW))
        self.assertFalse(self.run.has(PHYSICAL) or self.run.has(GATE))

    def test_read_only_audit_verifies_saved_gate_without_new_call(self):
        self.run_in_bounded_fixture()
        before = {path.name: sha(path.read_bytes()) for path in self.root.iterdir()
                  if path.is_file()}
        result = audit_saved_review(self.run)
        self.assertEqual(result["status"], "saved_gate_verified")
        self.assertEqual(result["new_physical_requests"], 1)
        self.assertEqual(result["provider_calls_by_audit"], 0)
        self.assertFalse(result["execution_claim_reopened"])
        self.assertEqual(self.run.tracer.calls, 1)
        self.assertEqual(before, {path.name: sha(path.read_bytes())
                                  for path in self.root.iterdir() if path.is_file()})

    def test_read_only_audit_classifies_missing_gate(self):
        self.run_in_bounded_fixture()
        (self.root / GATE).unlink()
        result = audit_saved_review(self.run)
        self.assertEqual(result["status"], "gate_missing_after_verified_opinion")
        self.assertFalse(result["production_ready"])
        self.assertFalse(self.run.has(GATE))
        self.assertEqual(self.run.tracer.calls, 1)

    def test_read_only_audit_classifies_missing_physical_and_gate(self):
        self.run_in_bounded_fixture()
        (self.root / GATE).unlink()
        (self.root / PHYSICAL).unlink()
        result = audit_saved_review(self.run)
        self.assertEqual(result["status"],
                         "physical_and_gate_missing_after_verified_opinion")
        self.assertFalse(self.run.has(PHYSICAL) or self.run.has(GATE))
        self.assertEqual(self.run.tracer.calls, 1)

    def test_read_only_audit_distinguishes_no_opinion_from_unresolved_activity(self):
        self.assertEqual(audit_saved_review(self.run)["status"],
                         "no_new_reviewer_activity")
        self.run_in_bounded_fixture()
        for name in (REVIEW, PHYSICAL, GATE):
            (self.root / name).unlink()
        result = audit_saved_review(self.run)
        self.assertEqual(result["status"],
                         "unresolved_activity_without_saved_opinion")
        self.assertGreater(result["new_trace_events"], 0)

    def test_read_only_audit_rejects_changed_physical_or_gate(self):
        self.run_in_bounded_fixture()
        physical = self.run.read(PHYSICAL)
        physical["usage_prompt_tokens"] += 1
        self.run.write(PHYSICAL, physical)
        with self.assertRaisesRegex(ValueError, "physical receipt differs"):
            audit_saved_review(self.run)
        physical["usage_prompt_tokens"] -= 1
        self.run.write(PHYSICAL, physical)
        gate = self.run.read(GATE)
        gate["production_ready"] = True
        self.run.write(GATE, gate)
        with self.assertRaisesRegex(ValueError, "gate differs"):
            audit_saved_review(self.run)

    def test_read_only_audit_rejects_changed_run_identity(self):
        self.run_in_bounded_fixture()
        self.run.manifest["run_id"] = "another-run"
        with self.assertRaisesRegex(ValueError, "history or authority bindings"):
            audit_saved_review(self.run)


if __name__ == "__main__":
    unittest.main()
