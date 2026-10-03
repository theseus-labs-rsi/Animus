"""Offline accounting transition checks; no child directory or provider exists."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.cumulative_child import build_control_documents
from pipeline.cumulative_prewrite import PrewritePermit, _SEAL


class CumulativeChildControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "output" / "runs" / "same-run"
        self.h = {key: char * 64 for key, char in zip(
            ("lineage", "migration", "evidence", "authorization", "boundary",
             "stage", "epoch", "manifest", "profile", "trace"),
            "abcdef0123")}
        self.permit = PrewritePermit("same-run", self.destination.resolve(),
            self.h["migration"], self.h["authorization"], _seal=_SEAL)
        self.manifest = {"run_id": "same-run", "scenario": "seed-showcase",
            "status": "failed", "config": {"seed_id": "original-seed"},
            "stages": {"corpus": {"attempt": 2, "status": "succeeded"}},
            "terminal_reconciliation": {"budget_stop_preserved": True}}
        self.profile = {"model": "dummy", "max_calls": 1200, "max_cny": 35,
            "admitted_calls": 1026, "settled_usage_calls": 1026,
            "unsettled_reservation_calls": 0, "reported_usage_estimate_cny": 34.796145028,
            "budget_consumed_cny": 34.79614502800003, "stopped": True,
            "stop_reason": {"reason": "estimated_budget_limit"},
            "source_drift": [], "execution_status": "failed"}
        self.manifest_bytes = json.dumps(self.manifest, ensure_ascii=False).encode("utf-8")
        self.profile_bytes = json.dumps(self.profile, ensure_ascii=False).encode("utf-8")
        self.h["manifest"] = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.h["profile"] = hashlib.sha256(self.profile_bytes).hexdigest()
        self.lineage = {"identity": {"run_id": "same-run", "scenario": "seed-showcase"},
            "budget": {"model": "dummy", "admitted_calls": 1026,
                "settled_usage_calls": 1026, "max_calls": 1200, "max_cny": 35,
                "reported_usage_estimate_cny": "34.796145028",
                "trace_sha256": self.h["trace"]}}
        self.migration = {"version": "complete-cumulative-source-migration/v1",
            "status": "complete", "complete_source_migration": True,
            "paid_resume_authorization": False,
            "historical_profile_sha256": self.h["profile"],
            "parent_lineage_sha256": self.h["lineage"]}
        self.authorization = {"policy": "explicit-cumulative-cap-authorization/v1",
            "user_authorization": True,
            "migration_sha256": self.h["migration"],
            "first_request_boundary_sha256": self.h["boundary"],
            "parent_profile_sha256": self.h["profile"],
            "parent_trace_sha256": self.h["trace"],
            "model": "dummy", "allowed_first_step": "world.semantic_review",
            "old_max_calls": 1200, "old_max_cny": 35,
            "new_max_calls": 1300, "new_max_cny": 36}

    def build(self):
        return build_control_documents(permit=self.permit,
            parent_manifest_bytes=self.manifest_bytes,
            parent_profile_bytes=self.profile_bytes,
            lineage=self.lineage, migration=self.migration,
            authorization=self.authorization,
            lineage_sha256=self.h["lineage"],
            migration_sha256=self.h["migration"],
            evidence_sha256=self.h["evidence"],
            authorization_sha256=self.h["authorization"],
            boundary_sha256=self.h["boundary"], stage_sha256=self.h["stage"],
            source_epoch_sha256=self.h["epoch"],
            parent_manifest_sha256=self.h["manifest"],
            parent_profile_sha256=self.h["profile"])

    def test_preserves_cumulative_counts_and_old_documents(self):
        before_manifest, before_profile = deepcopy(self.manifest), deepcopy(self.profile)
        manifest, profile = self.build()
        self.assertEqual((self.manifest, self.profile), (before_manifest, before_profile))
        self.assertEqual(manifest["run_id"], before_manifest["run_id"])
        self.assertEqual(manifest["stages"], before_manifest["stages"])
        self.assertEqual(manifest["terminal_reconciliation"], before_manifest["terminal_reconciliation"])
        self.assertEqual(manifest["config"]["cumulative_recovery"]["migration_sha256"], self.h["migration"])
        for key in ("admitted_calls", "settled_usage_calls", "unsettled_reservation_calls",
                    "reported_usage_estimate_cny", "budget_consumed_cny", "source_drift"):
            self.assertEqual(profile[key], before_profile[key])
        self.assertFalse(profile["stopped"])
        self.assertEqual(profile["max_cny"], 36)
        self.assertEqual(profile["resume_history"][-1]["previous_stop_reason"],
                         before_profile["stop_reason"])
        self.assertFalse(self.destination.exists())

    def test_missing_authority_and_mismatched_seal_fail_closed(self):
        self.authorization["user_authorization"] = False
        with self.assertRaisesRegex(ValueError, "matching cap authority"):
            self.build()
        self.authorization["user_authorization"] = True
        self.permit = PrewritePermit("same-run", self.destination.resolve(),
            "0" * 64, self.h["authorization"], _seal=_SEAL)
        with self.assertRaisesRegex(ValueError, "matching cap authority"):
            self.build()
        self.assertFalse(self.destination.exists())

    def test_preserved_liability_must_fit_the_new_cap(self):
        self.authorization["new_max_cny"] = "34.7"
        with self.assertRaisesRegex(ValueError, "matching cap authority"):
            self.build()
        self.authorization["new_max_cny"] = "35.2"
        self.profile["budget_consumed_cny"] = 35.3
        self.profile_bytes = json.dumps(self.profile, ensure_ascii=False).encode("utf-8")
        self.h["profile"] = hashlib.sha256(self.profile_bytes).hexdigest()
        self.migration["historical_profile_sha256"] = self.h["profile"]
        self.authorization["parent_profile_sha256"] = self.h["profile"]
        with self.assertRaisesRegex(ValueError, "historical liability"):
            self.build()
        self.assertFalse(self.destination.exists())

    def test_authorized_decimal_cap_cannot_round_up_in_float_runner(self):
        self.authorization["new_max_cny"] = "35.000000000000001"
        with self.assertRaisesRegex(ValueError, "too small for the bounded runner"):
            self.build()
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
