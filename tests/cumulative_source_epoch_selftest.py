"""Offline source-epoch binding rejects line-ending and path ambiguities."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_cumulative_source_epoch import (_blob_style, _git_equivalent_checkout,
    _safe_source_path, audit)


class SourceEpochAuditTests(unittest.TestCase):
    def test_git_blob_and_windows_checkout_are_explicit_distinct_styles(self):
        blob = b"first\nsecond\n"
        self.assertEqual(_blob_style(blob, hashlib.sha256(blob).hexdigest()), "git_blob")
        checked_out = blob.replace(b"\n", b"\r\n")
        self.assertEqual(_blob_style(blob, hashlib.sha256(checked_out).hexdigest()), "windows_crlf")
        self.assertIsNone(_blob_style(blob, "0" * 64))
        mixed = b"first\r\nsecond\n"
        self.assertTrue(_git_equivalent_checkout(blob, mixed, hashlib.sha256(mixed).hexdigest()))
        self.assertFalse(_git_equivalent_checkout(blob, b"first\r\nchanged\n",
            hashlib.sha256(b"first\r\nchanged\n").hexdigest()))

    def test_source_path_cannot_reach_credentials_or_leave_repository(self):
        for name in ("../config.py", "C:/config.py", "nested\\config.py", ".env.py",
                     "pipeline/credentials.py", "pipeline/secret_model.py"):
            self.assertFalse(_safe_source_path(name), name)
        self.assertTrue(_safe_source_path("pipeline/world_semantics.py"))

    def test_unbound_historical_map_rejected_before_git_or_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_map = root / "old.json"
            lineage = root / "lineage.json"
            old_map.write_text(json.dumps({"pipeline/world.py": "a" * 64}), encoding="utf-8")
            lineage.write_text(json.dumps({"status": "historical_lineage_verified",
                "complete_source_migration": False, "ending_source_map_sha256": "0" * 64}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                audit(root, old_map, lineage, legacy_commit="a" * 40)


if __name__ == "__main__":
    unittest.main()
