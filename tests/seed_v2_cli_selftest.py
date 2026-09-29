"""Offline, real-process checks of v1/v2 seed validation and source boundaries."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "tools" / "validate_seed_packs.py"
sys.path.insert(0, str(ROOT))
from tests.seed_contract_selftest import fixture as synthetic_seed


class SeedCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source_root = self.work / "source_workspace"
        self.source_root.mkdir()
        self.source = self.source_root / "synthetic.txt"
        self.source.write_text("Synthetic source for offline hash verification.\n", encoding="utf-8")
        self.pack = synthetic_seed()
        for source in self.pack["sources"]:
            source["path"] = self.source.name
            source["sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.path = self.work / "seed.json"

    def write(self, pack=None):
        self.path.write_text(json.dumps(self.pack if pack is None else pack, ensure_ascii=False),
                             encoding="utf-8")

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(CLI),
                                 str(self.path), *map(str, args)], cwd=self.work,
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.returncode, json.loads(result.stdout)

    def v2(self):
        pack = deepcopy(self.pack)
        pack["schema_version"] = 2
        pack["task"]["instructions"] = [pack["task"]["instructions"]]
        for source in pack["sources"]:
            source["source_kind"] = "example"
        pack.update(document_map=[], generation_contract={}, review={"blocking_issues": []})
        return pack

    def test_v1_structure_only_does_not_read_sources(self):
        self.source.unlink()
        self.write()
        code, report = self.run_cli()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertTrue(report["generation_ready"])
        self.assertFalse(report["semantic_review_performed"])
        self.assertFalse(report["packs"][0]["source_bytes_verified"])
        self.assertEqual(report["packs"][0]["schema_version"], 1)
        self.assertEqual(Path(report["packs"][0]["source_root"]), ROOT.resolve())

    def test_input_paths_are_required(self):
        result = subprocess.run([sys.executable, "-B", str(CLI)], cwd=self.work,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("paths", result.stderr)

    def test_explicit_source_root_verifies_hashes(self):
        self.write()
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root)
        self.assertEqual(code, 0)
        self.assertTrue(report["packs"][0]["source_bytes_verified"])

    def test_source_root_without_verify_remains_read_free(self):
        self.source.unlink()
        self.write()
        code, report = self.run_cli("--source-root", self.work / "missing")
        self.assertEqual(code, 0)
        self.assertFalse(report["packs"][0]["source_bytes_verified"])

    def test_escaped_source_is_rejected_before_read(self):
        outside = self.work / "outside.txt"
        outside.write_text("Synthetic outside file", encoding="utf-8")
        self.pack["sources"][0]["path"] = "../outside.txt"
        self.pack["sources"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
        self.write()
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root)
        self.assertEqual(code, 1)
        self.assertFalse(report["generation_ready"])
        self.assertTrue(any("within source_root" in item for item in report["packs"][0]["issues"]))

    def test_wrong_hash_fails_generation_ready(self):
        self.source.write_text("Changed synthetic content", encoding="utf-8")
        self.write()
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root)
        self.assertEqual(code, 1)
        self.assertTrue(report["packs"][0]["structure_valid"])
        self.assertFalse(report["generation_ready"])
        self.assertTrue(any("SHA256 mismatch" in item for item in report["packs"][0]["issues"]))

    def test_missing_source_reports_unavailable(self):
        self.source.unlink()
        self.write()
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root)
        self.assertEqual(code, 1)
        self.assertIn("unavailable", report["packs"][0]["issues"][0])

    def test_duplicate_json_key_is_rejected(self):
        self.path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertFalse(report["packs"][0]["structure_valid"])
        self.assertIn("Duplicate JSON key", report["packs"][0]["issues"][0])

    def test_v2_is_loaded_by_real_cli(self):
        self.write(self.v2())
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root,
                                    "--require-generation-ready")
        self.assertEqual(code, 0, report)
        self.assertTrue(report["generation_ready"])
        self.assertEqual(report["packs"][0]["schema_version"], 2)

    def test_documented_example_passes_real_hash_verification(self):
        example_root = ROOT / "skills/realfiles-to-seedjson/examples"
        self.path = example_root / "seed_v2_example.json"
        code, report = self.run_cli("--verify-sources", "--source-root", example_root / "source_workspace",
                                    "--require-generation-ready")
        self.assertEqual(code, 0, report)
        self.assertTrue(report["packs"][0]["source_bytes_verified"])
        self.assertTrue(report["generation_ready"])

    def test_blocked_v2_can_be_archived(self):
        pack = self.v2()
        pack["review"] = {"blocking_issues": ["Synthetic unresolved source conflict"]}
        self.write(pack)
        code, report = self.run_cli()
        self.assertEqual(code, 0, report)
        self.assertTrue(report["passed"])
        self.assertFalse(report["generation_ready"])
        self.assertEqual(report["packs"][0]["blocking_issues"], pack["review"]["blocking_issues"])

    def test_require_generation_ready_rejects_blocked_v2(self):
        pack = self.v2()
        pack["review"] = {"blocking_issues": ["Synthetic unresolved source conflict"]}
        self.write(pack)
        code, report = self.run_cli("--require-generation-ready")
        self.assertEqual(code, 1, report)
        self.assertTrue(report["passed"])
        self.assertFalse(report["generation_ready"])

    def test_unresolved_core_can_archive_but_cannot_generate(self):
        pack = self.v2()
        pack["blueprint_requirements"]["entity_types"][0]["status"] = "unresolved"
        self.write(pack)
        code, report = self.run_cli()
        self.assertEqual(code, 0, report)
        self.assertTrue(report["passed"])
        self.assertFalse(report["generation_ready"])
        self.assertEqual(report["packs"][0]["blocking_issues"], [])
        self.assertIn("entity_types", report["packs"][0]["generation_blockers"][0])
        code, report = self.run_cli("--require-generation-ready")
        self.assertEqual(code, 1, report)

    def test_conditional_mechanism_cannot_generate(self):
        pack = self.v2()
        pack["mechanisms"][0].update(status="conditional", conditions=["Synthetic unresolved condition"])
        self.write(pack)
        code, report = self.run_cli("--require-generation-ready")
        self.assertEqual(code, 1, report)
        self.assertTrue(report["passed"])
        self.assertIn("mechanisms", report["packs"][0]["generation_blockers"][0])

    def test_uncertain_semantic_boundary_stays_generation_context(self):
        pack = self.v2()
        pack["generation_contract"]["unresolved"] = [
            {"text": "Synthetic optional approval remains unresolved", "status": "unresolved", "source_refs": []}]
        self.write(pack)
        code, report = self.run_cli("--require-generation-ready")
        self.assertEqual(code, 0, report)
        self.assertTrue(report["generation_ready"])
        self.assertEqual(report["packs"][0]["generation_blockers"], [])

    def test_internal_sanitized_contract_cannot_replace_raw_seed(self):
        from pipeline.seed_pack import seed_contract
        self.write(seed_contract(self.v2()))
        code, report = self.run_cli()
        self.assertEqual(code, 0, report)
        self.assertTrue(report["passed"])
        self.assertFalse(report["generation_ready"])
        self.assertIn("full raw seed", report["packs"][0]["generation_blockers"][0])
        code, report = self.run_cli("--require-generation-ready")
        self.assertEqual(code, 1, report)

    def test_symlink_escape_is_rejected(self):
        outside = self.work / "outside.txt"
        outside.write_text("Synthetic outside file", encoding="utf-8")
        link = self.source_root / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not permit creating a test symlink")
        self.pack["sources"][0]["path"] = link.name
        self.pack["sources"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
        self.write()
        code, report = self.run_cli("--verify-sources", "--source-root", self.source_root)
        self.assertEqual(code, 1)
        self.assertTrue(any("within source_root" in item for item in report["packs"][0]["issues"]))


if __name__ == "__main__":
    unittest.main()
