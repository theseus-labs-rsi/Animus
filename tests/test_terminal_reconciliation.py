"""The recovery control preview must not promote incomplete work or clear a stop."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.reconcile_terminal_run import canonical_hash, derive_run, reconcile


def fixture():
    published = {"corpus": {"sessions": [{"session_id": 0, "docs": []}]}, "done_weeks": [0]}
    candidate = {"corpus": {"sessions": [{"session_id": 1, "docs": []}]}, "done_weeks": [1]}
    objects = {
        "manifest.json": {"status": "running", "current_stage": "corpus",
                          "stages": {"corpus": {"status": "succeeded", "done": True}},
                          "config": {"seed_id": "test"}},
        "11_production.json": {"status": "execution_error", "history": ["original"]},
        "experiment_profile.json": {"stopped": True,
                                    "stop_reason": {"reason": "estimated_budget_limit"},
                                    "source_drift": [], "admitted_calls": 7,
                                    "settled_usage_calls": 7, "unsettled_reservation_calls": 0},
        "controller_report.json": {"status": "needs_analysis",
                                   "finished_utc": "2026-09-27T17:40:40+00:00",
                                   "phases": [{"phase": "generate", "status": "failed", "exit_code": 1}]},
        "05_corpus.json": published,
        "05_corpus_candidate.json": candidate,
        "05_corpus_warning.json": {"status": "warning", "release_eligible": False,
                                   "binding": {"corpus_hash": canonical_hash(published),
                                               "candidate_hash": canonical_hash(candidate)}},
    }
    return {name: json.dumps(obj, ensure_ascii=False).encode("utf-8") for name, obj in objects.items()}


def mutate(raw, name, change):
    modified = deepcopy(raw)
    obj = json.loads(modified[name])
    change(obj)
    modified[name] = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return modified


def archive_fixture(path, *, unsafe=False, link=False):
    raw = fixture()
    prefix = "candidate/output/runs/example/"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in raw.items():
            member_name = name if name == "controller_report.json" else prefix + name
            item = tarfile.TarInfo(member_name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
        if unsafe:
            content = b"unread"
            item = tarfile.TarInfo(prefix + ".env")
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
        if link:
            item = tarfile.TarInfo(prefix + "linked-artifact")
            item.type = tarfile.SYMTYPE
            item.linkname = "manifest.json"
            archive.addfile(item)
    return raw, prefix, hashlib.sha256(path.read_bytes()).hexdigest()


class TerminalReconciliationTests(unittest.TestCase):
    def test_only_status_and_reconciliation_proof_change(self):
        raw = fixture()
        child, receipt = reconcile(raw, archive_sha256="a" * 64)
        parent = json.loads(raw["manifest.json"])
        self.assertEqual(child.pop("terminal_reconciliation"), receipt)
        self.assertEqual(child.pop("status"), "failed")
        self.assertEqual(child.pop("current_stage"), "")
        parent.pop("status")
        parent.pop("current_stage")
        self.assertEqual(child, parent)
        self.assertFalse(receipt["admission_reopened"])
        self.assertFalse(receipt["runnable_run_created"])

    def test_running_controller_is_not_a_terminal_proof(self):
        raw = mutate(fixture(), "controller_report.json", lambda x: x.update(status="running"))
        with self.assertRaisesRegex(ValueError, "terminal state"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_successful_child_is_not_a_failed_controller(self):
        raw = mutate(fixture(), "controller_report.json", lambda x: x["phases"][0].update(exit_code=0))
        with self.assertRaisesRegex(ValueError, "terminal state"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_unsettled_ledger_is_rejected(self):
        raw = mutate(fixture(), "experiment_profile.json", lambda x: x.update(unsettled_reservation_calls=1))
        with self.assertRaisesRegex(ValueError, "ledger"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_candidate_mutation_is_rejected(self):
        raw = mutate(fixture(), "05_corpus_candidate.json", lambda x: x["done_weeks"].append(2))
        with self.assertRaisesRegex(ValueError, "warning"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_release_eligible_warning_is_rejected(self):
        raw = mutate(fixture(), "05_corpus_warning.json", lambda x: x.update(release_eligible=True))
        with self.assertRaisesRegex(ValueError, "warning"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_nonrunning_manifest_is_not_reconciled_twice(self):
        raw = mutate(fixture(), "manifest.json", lambda x: x.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "terminal state"):
            reconcile(raw, archive_sha256="a" * 64)

    def test_complete_derivation_preserves_every_original_except_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, prefix, archive_hash = archive_fixture(root / "terminal.tar.gz")
            proof = derive_run(root / "terminal.tar.gz", expected_sha256=archive_hash,
                run_prefix=prefix, destination=root / "derived")
            output = root / "derived" / "run"
            for name, content in raw.items():
                if name != "controller_report.json" and name != "manifest.json":
                    self.assertEqual((output / name).read_bytes(), content)
            self.assertEqual((root / "derived/evidence/parent_manifest.json").read_bytes(),
                             raw["manifest.json"])
            self.assertEqual(json.loads((output / "manifest.json").read_bytes())["status"], "failed")
            self.assertEqual(proof["inherited_run_file_count"], len(raw) - 1)
            self.assertFalse(proof["runnable_run_created"])

    def test_derivation_requires_external_archive_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prefix, _ = archive_fixture(root / "terminal.tar.gz")
            with self.assertRaisesRegex(ValueError, "external SHA"):
                derive_run(root / "terminal.tar.gz", expected_sha256="0" * 64,
                    run_prefix=prefix, destination=root / "derived")
            self.assertFalse((root / "derived").exists())

    def test_derivation_rejects_env_file_without_opening_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prefix, archive_hash = archive_fixture(root / "terminal.tar.gz", unsafe=True)
            with self.assertRaisesRegex(ValueError, "disallowed file path"):
                derive_run(root / "terminal.tar.gz", expected_sha256=archive_hash,
                    run_prefix=prefix, destination=root / "derived")
            self.assertFalse((root / "derived/run/.env").exists())

    def test_derivation_rejects_nonregular_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prefix, archive_hash = archive_fixture(root / "terminal.tar.gz", link=True)
            with self.assertRaisesRegex(ValueError, "non-regular artifact"):
                derive_run(root / "terminal.tar.gz", expected_sha256=archive_hash,
                    run_prefix=prefix, destination=root / "derived")
            self.assertFalse((root / "derived").exists())

    @unittest.skipUnless(os.name == "nt", "Windows path-length guard")
    def test_deep_destination_fails_before_any_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prefix, archive_hash = archive_fixture(root / "terminal.tar.gz")
            destination = root / ("long-" * 40) / "derived"
            with self.assertRaisesRegex(ValueError, "too deep"):
                derive_run(root / "terminal.tar.gz", expected_sha256=archive_hash,
                    run_prefix=prefix, destination=destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
