"""Offline wrapper-to-factory boundary with a synthetic, sealed child."""

from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.cumulative_prewrite import PrewritePermit, _SEAL
from pipeline.cumulative_materializer import MaterializedChild
from llm_transport import original_run_transport
from tools import run_original_bc_smoke as smoke

import original_smoke_corpus_review_override_selftest as fixture


class CumulativeWrapperBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = "synthetic-cumulative"
        self.destination = self.root / "output" / "runs" / self.run_id
        self.source_hashes = {"synthetic.py": "a" * 64}
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "offline-disabled",
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "PYTHONUTF8": "1"}).start()
        self.api = fixture.FakeSDK()
        self.cfg = fixture.load_config(self.api)
        previous_config = sys.modules.get("config")
        sys.modules["config"] = self.cfg
        def restore_config():
            if previous_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
        self.addCleanup(restore_config)

    def _install_synthetic_child(self, **kwargs):
        self.assertEqual(kwargs["destination"], self.destination)
        self.destination.mkdir(parents=True)
        (self.destination / "manifest.json").write_text(json.dumps({
            "run_id": self.run_id, "scenario": "office", "status": "failed",
            "config": {"cumulative_recovery": {"version": "explicit-cumulative-source-migration/v1"}},
            "stages": {}, "terminal_reconciliation": {"status": "stopped"}}), encoding="utf-8")
        (self.destination / "experiment_profile.json").write_text(json.dumps({
            "model": "glm-5.3-flash", "transport": original_run_transport("glm-5.3-flash", "low", 150),
            "max_calls": 1300, "max_cny": 40, "admitted_calls": 1026,
            "settled_usage_calls": 1026, "unsettled_reservation_calls": 0,
            "reserved_upper_estimate_cny": 34.8, "budget_consumed_cny": 34.8,
            "reported_usage_estimate_cny": 34.796145028, "stopped": False,
            "source_drift": [], "json_attempts": 3, "min_output_tokens": 0,
            "max_output_tokens": 32768, "call_timeout_seconds": 150,
            "llm_concurrency": 1, "semantic_workers": 1,
            "disclosure_format_attempts": 1, "settle_reported_usage": True,
            "execution_status": "materialized_offline"}), encoding="utf-8")
        (self.destination / "experiment_source_hashes_cumulative_start.json").write_text(
            json.dumps(self.source_hashes), encoding="utf-8")
        permit = PrewritePermit(self.run_id, self.destination.resolve(), "b" * 64,
                                "c" * 64, _seal=_SEAL)
        return MaterializedChild(self.destination.resolve(), permit)

    def test_real_factory_routes_through_bounded_fake_provider(self):
        from pipeline import cumulative_materializer, factory, post_corpus_world_review, run
        reached = []

        def review(current):
            reached.append(current.run_id)
            self.assertEqual(current.tracer.chat_json("world.semantic_review",
                [{"role": "user", "content": "offline boundary"}],
                model="glm-5.3-flash", max_tokens=4096), {"ok": True})
            current.write("12_post_corpus_world_review.json", {"status": "unresolved"})
            current.write("12_post_corpus_world_review_physical.json", {
                "version": "post-corpus-world-review-physical/v1", "new_physical_requests": 1})
            review_path = current.dir / "12_post_corpus_world_review.json"
            physical_path = current.dir / "12_post_corpus_world_review_physical.json"
            current.write("12_post_corpus_world_review_gate.json", {
                "version": "post-corpus-world-review/v1", "status": "unresolved_corpus_failure",
                "new_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                "new_physical_receipt_sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest(),
                "new_physical_requests": 1, "truth_changed": False,
                "author_repair_started": False, "production_ready": False,
                "provider_calls_by_gate": 0})
            return {"status": "unresolved"}

        config = self.root / "cumulative-control.json"
        keys = ("derived_root", "destination", "source_root", "lineage_path", "stage_path",
                "source_epoch_path", "scope_path", "migration_path", "authorization_path",
                "boundary_path", "world_migration_path", "downstream_evidence_path")
        values = {key: str(self.root / key) for key in keys}
        values.update(source_root=str(self.root), destination=str(self.destination),
                      expected_archive_sha256="d" * 64)
        config.write_text(json.dumps(values), encoding="utf-8")
        with ExitStack() as stack:
            stack.enter_context(patch.object(smoke, "ROOT", self.root))
            stack.enter_context(patch.object(smoke, "implementation_hashes", return_value=self.source_hashes))
            stack.enter_context(patch.object(cumulative_materializer, "materialize_child",
                                             side_effect=self._install_synthetic_child))
            stack.enter_context(patch.object(factory, "RUNS_DIR", self.destination.parent))
            stack.enter_context(patch.object(run, "RUNS_DIR", self.destination.parent))
            stack.enter_context(patch.object(post_corpus_world_review, "run_once", side_effect=review))
            stack.enter_context(patch.object(sys, "argv", ["smoke", "--run", self.run_id,
                "--cumulative-child-config", str(config)]))
            smoke.main()
        self.assertEqual(reached, [self.run_id])
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.api.calls[0]["kwargs"]["model"], "glm-5.3-flash")
        profile = json.loads((self.destination / "experiment_profile.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["admitted_calls"], 1027)
        self.assertEqual(profile["source_drift"], [])
        self.assertEqual(profile["execution_status"], "post_corpus_world_review_gate_recorded")
        self.assertEqual(profile["post_corpus_world_review_gate_status"], "unresolved_corpus_failure")
        claim = json.loads((self.destination / "00_cumulative_execution_claim.json").read_text(encoding="utf-8"))
        self.assertEqual(claim["run_id"], self.run_id)
        before = (self.destination / "experiment_profile.json").read_bytes()
        with self.assertRaises(FileExistsError):
            smoke._claim_cumulative_execution(self.destination,
                PrewritePermit(self.run_id, self.destination.resolve(), "b" * 64,
                               "c" * 64, _seal=_SEAL))
        self.assertEqual((self.destination / "experiment_profile.json").read_bytes(), before)

    def test_cumulative_exit_rejects_absent_or_changed_gate(self):
        self.destination.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "without a post-corpus world-review gate"):
            smoke._cumulative_review_exit_status(self.destination)
        review_path = self.destination / "12_post_corpus_world_review.json"
        physical_path = self.destination / "12_post_corpus_world_review_physical.json"
        gate_path = self.destination / "12_post_corpus_world_review_gate.json"
        review_path.write_text('{"status":"unresolved"}', encoding="utf-8")
        physical_path.write_text('{"version":"post-corpus-world-review-physical/v1","new_physical_requests":1}',
                                 encoding="utf-8")
        gate_path.write_text(json.dumps({
            "version": "post-corpus-world-review/v1", "status": "unresolved_corpus_failure",
            "new_review_sha256": "0" * 64,
            "new_physical_receipt_sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest(),
            "new_physical_requests": 1, "truth_changed": False,
            "author_repair_started": False, "production_ready": False,
            "provider_calls_by_gate": 0}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or inconsistent"):
            smoke._cumulative_review_exit_status(self.destination)


if __name__ == "__main__":
    unittest.main()
