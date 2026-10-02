"""A copied terminal run cannot reach factory or production writes."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cumulative_entry_guard import (check_before_run_init, check_before_production,
    enter_bounded_run, leave_bounded_run, enter_explicit_cumulative_run,
    leave_explicit_cumulative_run)
from pipeline.cumulative_prewrite import PrewritePermit, _SEAL


class CumulativeEntryGuardTests(unittest.TestCase):
    def synthetic_permit(self):
        # The test constructs a sealed value to exercise routing; production
        # obtains it only from the read-only validate_prewrite function.
        return PrewritePermit(self.root.name, self.root.resolve(), "a" * 64,
                              "b" * 64, _seal=_SEAL)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"status": "failed", "config": {}}), encoding="utf-8")
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict("os.environ", {"PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "offline-disabled",
            "MODEL": "offline-disabled", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}).start()

    def test_terminal_copy_rejected_before_manifest_change(self):
        original = {"status": "failed", "env": {"model": "historical"},
                    "terminal_reconciliation": {"parent_manifest_sha256": "a" * 64}}
        self.manifest.write_text(json.dumps(original), encoding="utf-8")
        before = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "dedicated entry"):
            check_before_run_init(self.manifest)
        self.assertEqual(hashlib.sha256(self.manifest.read_bytes()).hexdigest(), before)

    def test_stopped_ledger_rejected_before_run_init(self):
        profile = self.root / "experiment_profile.json"
        profile.write_text(json.dumps({"stopped": True, "source_drift": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stopped cumulative budget"):
            check_before_run_init(self.manifest)
        self.assertEqual(json.loads(profile.read_text(encoding="utf-8"))["stopped"], True)

    def test_claimed_cumulative_migration_cannot_use_plain_factory(self):
        self.manifest.write_text(json.dumps({"config": {"cumulative_recovery": {
            "version": "explicit-cumulative-source-migration/v1"}}}), encoding="utf-8")
        before = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "dedicated entry"):
            check_before_run_init(self.manifest)
        self.assertEqual(hashlib.sha256(self.manifest.read_bytes()).hexdigest(), before)

    def test_explicit_child_requires_both_dedicated_and_bounded_contexts(self):
        self.manifest.write_text(json.dumps({"run_id": self.root.name,
            "terminal_reconciliation": {"status": "stopped"},
            "config": {"cumulative_recovery": {
                "version": "explicit-cumulative-source-migration/v1"}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "validated cumulative prewrite permit"):
            enter_explicit_cumulative_run(self.root.name)
        with self.assertRaisesRegex(ValueError, "active bounded call wrapper"):
            enter_explicit_cumulative_run(self.synthetic_permit())
        bounded = enter_bounded_run(self.root.name)
        try:
            with self.assertRaisesRegex(ValueError, "dedicated entry"):
                check_before_run_init(self.manifest)
            explicit = enter_explicit_cumulative_run(self.synthetic_permit())
            try:
                check_before_run_init(self.manifest)
            finally:
                leave_explicit_cumulative_run(explicit)
            with self.assertRaisesRegex(ValueError, "dedicated entry"):
                check_before_run_init(self.manifest)
        finally:
            leave_bounded_run(bounded)

    def test_direct_production_rejects_without_writing(self):
        class FakeRun:
            manifest = {"terminal_reconciliation": {"version": "terminal-state-reconciliation/v1"}}
            def has(self, _name):
                raise AssertionError("Run data should not be consulted before terminal gate")
            def write(self, *_args):
                raise AssertionError("Production write forbidden")
        with self.assertRaisesRegex(ValueError, "cannot skip"):
            check_before_production(FakeRun())

    def test_plain_open_run_can_enter_its_existing_path(self):
        (self.root / "experiment_profile.json").write_text(
            json.dumps({"stopped": False, "source_drift": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "active bounded wrapper"):
            check_before_run_init(self.manifest)
        token = enter_bounded_run(self.root.name)
        try:
            check_before_run_init(self.manifest)
        finally:
            leave_bounded_run(token)

    def test_direct_production_cannot_use_an_open_ledger_without_wrapper(self):
        class FakeRun:
            run_id = "example"
            manifest = {"config": {}}
            def has(self, name):
                return name == "experiment_profile.json"
            def read(self, _name):
                return {"stopped": False, "source_drift": []}
        with self.assertRaisesRegex(ValueError, "call wrapper"):
            check_before_production(FakeRun())
        token = enter_bounded_run("example")
        try:
            check_before_production(FakeRun())
        finally:
            leave_bounded_run(token)

    def test_claimed_cumulative_migration_cannot_skip_review_gate(self):
        class FakeRun:
            manifest = {"config": {"cumulative_recovery": {
                "version": "explicit-cumulative-source-migration/v1"}}}
            def has(self, _name):
                raise AssertionError("No production data should be read")
        with self.assertRaisesRegex(ValueError, "before review and repair acceptance"):
            check_before_production(FakeRun())

    def test_dedicated_factory_route_reaches_review_before_production(self):
        from pipeline import factory, post_corpus_world_review, production
        self.manifest.write_text(json.dumps({"run_id": self.root.name,
            "scenario": "synthetic", "config": {
                "cumulative_recovery": {"version": "explicit-cumulative-source-migration/v1"},
                "target_tokens": 100000}}), encoding="utf-8")
        class FakeRun:
            def __init__(self, _scenario, _run_id, **_kwargs):
                self.manifest = json.loads(self_manifest.read_text(encoding="utf-8"))
        self_manifest = self.manifest
        bounded = enter_bounded_run(self.root.name)
        explicit = enter_explicit_cumulative_run(self.synthetic_permit())
        try:
            with (patch.object(factory, "RUNS_DIR", self.root.parent),
                  patch.object(factory, "Run", FakeRun),
                  patch.object(post_corpus_world_review, "run_once", return_value={"status": "stopped"}) as review,
                  patch.object(production, "produce", side_effect=AssertionError("production bypass")),
                  patch.object(sys, "argv", ["pipeline.factory", "--run", self.root.name, "--to", "quality"])):
                factory.main()
            review.assert_called_once()
        finally:
            leave_explicit_cumulative_run(explicit)
            leave_bounded_run(bounded)

    def test_produce_rejects_before_creating_a_lock_file(self):
        from pipeline.production import produce
        class FakeRun:
            manifest = {"terminal_reconciliation": {"version": "terminal-state-reconciliation/v1"}}
            def stage_write_lock(self, *_args, **_kwargs):
                raise AssertionError("Production lock must not be opened")
        with self.assertRaisesRegex(ValueError, "cannot skip"):
            produce(FakeRun())


if __name__ == "__main__":
    unittest.main()
