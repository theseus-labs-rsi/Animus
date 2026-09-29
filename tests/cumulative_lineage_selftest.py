"""Fault-injection checks for the read-only cumulative lineage audit."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_cumulative_lineage import audit


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CumulativeLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "derived/run"
        self.evidence = self.root / "derived/evidence"
        self.run.mkdir(parents=True)
        self.evidence.mkdir(parents=True)
        self.prefix = "candidate/output/runs/example/"
        parent = {"run_id": "example", "scenario": "seed_s6", "status": "running",
                  "current_stage": "corpus", "stages": {"world": {"done": True}},
                  "config": {"production_control": {"version": "supply-driven/v5"}}}
        self.parent = encoded(parent)
        parent_sha = sha(self.parent)
        child = {**parent, "status": "failed", "current_stage": "",
                 "terminal_reconciliation": {"parent_manifest_sha256": parent_sha}}
        profile = {"model": "dummy", "source_drift": [], "stopped": True,
                   "stop_reason": {"reason": "estimated_budget_limit"},
                   "admitted_calls": 1, "settled_usage_calls": 1,
                   "unsettled_reservation_calls": 0, "max_calls": 2, "max_cny": 5,
                   "reported_usage_estimate_cny": 0.1,
                   "input_cny_per_m": 1, "output_cny_per_m": 0}
        previous = {"status": "passed", "raw_output": {"decision": "accept"}}
        files = {
            "manifest.json": self.parent,
            "experiment_profile.json": encoded(profile),
            "llm_attempts.jsonl": (b'{"event":"request","call_id":"c1"}\n'
                                   b'{"event":"response","call_id":"c1",'
                                   b'"response":{"usage":{"prompt_tokens":100000,"completion_tokens":0}}}\n'),
            "prompts.jsonl": b'{"i":1}\n',
            "11_production.json": encoded({"status": "execution_error", "identity": "identity",
                                            "history": [], "round": 3, "supply_round": 1,
                                            "layout_revisions": 3}),
            "02_world_review.json": encoded(previous),
            "02_world_review_attempts.json": encoded({"attempts": [previous]}),
            "05_corpus_candidate.json": encoded({"done_weeks": [0]}),
            "05_corpus_warning.json": encoded({"release_eligible": False}),
            "experiment_source_hashes.json": encoded({"config.py": "a" * 64}),
            "experiment_source_hashes_at_end.json": encoded({"config.py": "a" * 64}),
        }
        for name, raw in files.items():
            (self.run / name).write_bytes(encoded(child) if name == "manifest.json" else raw)
        (self.evidence / "parent_manifest.json").write_bytes(self.parent)
        self.archive = self.root / "archive.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            for name, raw in files.items():
                member = tarfile.TarInfo(self.prefix + name)
                member.size = len(raw)
                tar.addfile(member, io.BytesIO(raw))
        self.archive_sha = sha(self.archive.read_bytes())
        self.receipt = {"version": "terminal-state-reconciliation/v1",
            "archive_sha256": self.archive_sha, "runnable_run_created": False,
            "ledger_changed": False, "admission_reopened": False,
            "parent_manifest_sha256": parent_sha,
            "reconciled_manifest_sha256": sha(encoded(child)),
            "profile_sha256": sha(files["experiment_profile.json"]),
            "inherited_run_files": {name: {"bytes": len(raw), "sha256": sha(raw)}
                                    for name, raw in files.items()},
            "inherited_run_file_count": len(files),
            "inherited_run_bytes": sum(map(len, files.values()))}
        (self.evidence / "derivation_receipt.json").write_bytes(encoded(self.receipt))
        self.baseline = self.root / "baseline.json"
        self.baseline.write_bytes(encoded({"version": "offline-cumulative-cost-baseline/v1",
            "source": {"profile_sha256": sha(files["experiment_profile.json"]),
                       "trace_sha256": sha(files["llm_attempts.jsonl"])},
            "reconciliation": {"requests": 1, "responses_with_usage": 1,
                               "reported_usage_cost_cny": "0.1", "max_calls": 2,
                               "max_cny": "5"}}))

    def run_audit(self):
        return audit(self.run, self.evidence, self.baseline,
            archive_path=self.archive, archive_sha256=self.archive_sha, run_prefix=self.prefix)

    def test_complete_historical_copy_is_read_only_and_not_runnable(self):
        before = sha((self.run / "manifest.json").read_bytes())
        result = self.run_audit()
        self.assertEqual(result["status"], "historical_lineage_verified")
        self.assertEqual(result["historical_copy"]["files"], 11)
        self.assertFalse(result["complete_source_migration"])
        self.assertFalse(result["runnable_run_created"])
        self.assertEqual(sha((self.run / "manifest.json").read_bytes()), before)

    def test_changed_or_extra_derived_file_is_rejected(self):
        path = self.run / "02_world_review.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Derived file differs"):
            self.run_audit()
        path.write_bytes(original)
        (self.run / "invented.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            self.run_audit()

    def test_receipt_cannot_substitute_different_archive(self):
        receipt = dict(self.receipt)
        receipt["inherited_run_files"] = dict(receipt["inherited_run_files"])
        receipt["inherited_run_files"]["02_world_review.json"] = {
            "bytes": 1, "sha256": "0" * 64}
        (self.evidence / "derivation_receipt.json").write_bytes(encoded(receipt))
        with self.assertRaisesRegex(ValueError, "frozen archive"):
            self.run_audit()

    def test_lied_cost_baseline_is_rejected(self):
        baseline = json.loads(self.baseline.read_text(encoding="utf-8"))
        baseline["reconciliation"]["reported_usage_cost_cny"] = "0.2"
        self.baseline.write_bytes(encoded(baseline))
        with self.assertRaisesRegex(ValueError, "Cumulative ledger"):
            self.run_audit()

    def test_archive_sha_must_match_external_identity(self):
        with self.assertRaisesRegex(ValueError, "Frozen archive"):
            audit(self.run, self.evidence, self.baseline, archive_path=self.archive,
                  archive_sha256="0" * 64, run_prefix=self.prefix)


if __name__ == "__main__":
    unittest.main()
