"""Fault checks for classifying the frozen run before cumulative migration."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools import audit_cumulative_migration_scope as scope


class CumulativeMigrationScopeTest(unittest.TestCase):
    def test_never_accepts_unsafe_or_secret_historical_members(self):
        for name in ("../outside.json", "/root/old.json", "C:/old.json",
                     "nested\\old.json", ".env", "nested/credentials.json"):
            with self.subTest(name=name):
                self.assertFalse(scope._safe(name))

    def test_ledger_and_old_world_are_not_current_acceptance(self):
        self.assertEqual(scope.classify("llm_attempts.jsonl"),
                         "cumulative_ledger_append_only")
        self.assertEqual(scope.classify("02_world.json"),
                         "historical_world_input_requires_new_review")
        self.assertEqual(scope.classify("02_world_review.json"),
                         "historical_output_not_current_acceptance")
        self.assertEqual(scope.classify("05_corpus.json"),
                         "historical_output_not_current_acceptance")

    def test_world_receipt_rejects_source_drift_and_false_authorization(self):
        checkpoint = "02_world_agent_5903517308903f10c8b0.json"
        with TemporaryDirectory() as td:
            root = Path(td)
            code = root / "world.py"
            code.write_text("OLD\n", encoding="utf-8")
            digest = hashlib.sha256(code.read_bytes()).hexdigest()
            inherited = {checkpoint: {"sha256": "a" * 64}}
            epoch = {"target_source_hashes": {"world.py": digest}}
            world = {"version": "world-checkpoint-source-migration/v1",
                     "parent_sha256": "a" * 64,
                     "all_original_fields_retained": True,
                     "provider_calls": 0, "semantic_revalidation": False,
                     "publication_or_signal_migration": False,
                     "ledger_or_budget_changes": False,
                     "paid_resume_authorization": False,
                     "target_compiler_source_hashes": {"world.py": digest}}
            scope._verify_world_receipt(world, inherited, epoch, root)
            code.write_text("NEW\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                scope._verify_world_receipt(world, inherited, epoch, root)
            code.write_text("OLD\n", encoding="utf-8")
            world["paid_resume_authorization"] = True
            with self.assertRaisesRegex(ValueError, "overstated"):
                scope._verify_world_receipt(world, inherited, epoch, root)


if __name__ == "__main__":
    unittest.main()
