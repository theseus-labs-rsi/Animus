"""Offline seed-v2 freeze, reuse, release and diagnostic integration checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from seed_v2_contract_selftest import sample_v2
from pipeline.seed_pack import SeedPackError, seed_contract, seed_digest
from pipeline.seed_run import (AUDIT_ARTIFACT, GENERATION_ARTIFACT, SEED_ARTIFACT,
    prepare_seed_input, seed_config, validate_seed_identity, validate_seed_input)
from pipeline.quality import _release_inputs, evaluate_release, INPUTS
from pipeline.seed_lineage import seed_lineage_report
from tools.run_original_bc_smoke import reuse_original_stages


class SeedV2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "run"
        self.source.mkdir()
        self.pack = sample_v2()
        self.path = self.root / "seed.json"
        self.path.write_text(json.dumps(self.pack, ensure_ascii=False), encoding="utf-8")
        self.manifest = {"config": seed_config(self.path), "status": "done", "scenario": "seed_v2",
            "stages": {"input": {"done": True, "artifact": "00_input.json"},
                       "whitepaper": {"done": True, "artifact": "01_whitepaper.json"}}}
        self.run = SimpleNamespace(manifest=self.manifest, write=self.write,
            has=lambda name: (self.source / name).is_file(), read=self.read)
        self.scenario = prepare_seed_input(self.run)
        self.write("00_input.json", self.scenario)
        self.wp = {"seed_contract": seed_contract(self.pack)}
        self.write("01_whitepaper.json", self.wp)
        self.write("manifest.json", self.manifest)

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def read(self, name):
        return json.loads((self.source / name).read_text(encoding="utf-8"))

    def test_audit_and_generation_views_frozen_separately(self):
        audit = self.read(AUDIT_ARTIFACT)
        context = self.read(GENERATION_ARTIFACT)
        self.assertIn("EVAL_AUDIT_SECRET", json.dumps(audit))
        self.assertIn("AUDIT_ONLY_SECRET", json.dumps(audit))
        self.assertNotIn("SECRET", json.dumps(context))
        self.assertEqual(audit["seed_digest"], seed_digest(self.pack))
        self.assertIn("state_machines", context["context"]["blueprint_requirements"])
        self.assertEqual(validate_seed_identity(self.run, self.wp), self.pack)

    def test_resume_uses_frozen_seed_when_mutable_source_disappears(self):
        self.path.unlink()
        self.assertEqual(prepare_seed_input(self.run), self.scenario)
        self.assertEqual(validate_seed_input(self.run, self.scenario), self.pack)

    def test_tampered_or_missing_snapshots_rejected(self):
        for name in (AUDIT_ARTIFACT, GENERATION_ARTIFACT):
            original = self.read(name)
            with self.subTest(name=name):
                self.write(name, {**original, "tampered": True})
                with self.assertRaises(SeedPackError):
                    validate_seed_identity(self.run, self.wp)
                self.write(name, original)
                (self.source / name).unlink()
                with self.assertRaises(SeedPackError):
                    validate_seed_identity(self.run, self.wp)
                self.write(name, original)

    def test_blocking_seed_rejected_before_run_configuration(self):
        self.pack["review"]["blocking_issues"] = ["unknown field ownership"]
        self.path.write_text(json.dumps(self.pack), encoding="utf-8")
        with self.assertRaisesRegex(SeedPackError, "blocking_issues"):
            seed_config(self.path)

    def test_internal_contract_cannot_replace_full_raw_seed(self):
        contract = seed_contract(self.pack)
        self.path.write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaisesRegex(SeedPackError, "full raw seed"):
            seed_config(self.path)
        (self.source / SEED_ARTIFACT).unlink()
        with self.assertRaisesRegex(SeedPackError, "full raw seed"):
            prepare_seed_input(self.run)
        self.assertFalse(self.run.has(SEED_ARTIFACT))

    def test_original_stage_reuse_carries_both_views(self):
        self.write("00_about.json", {})
        self.write("01_seed_audit.json", {})
        target = self.root / "reused"
        result = reuse_original_stages(self.source, target, "whitepaper")
        for name in (SEED_ARTIFACT, AUDIT_ARTIFACT, GENERATION_ARTIFACT):
            self.assertEqual((self.source / name).read_bytes(), (target / name).read_bytes())
            self.assertIn(name, result["derived_from"]["input_files"])




    def test_filtered_bundle_retains_private_seed_identity(self):
        from eval.question_filter import export_filtered_benchmark
        self.write("06_grounded_questions.json", [{"qid": "q1", "question": "Offline fixture question"}])
        with patch("pipeline.quality.require_release", return_value={"eligible": True}), \
             patch("pipeline.quality.evaluate_release", return_value={"eligible": True, "issues": [], "status": "passed"}), \
             patch("eval.question_filter.filter_questions", return_value=([{"qid": "q1"}], {
                 "result_scope": "verified", "systems": ["stub"], "preserve_capabilities": [], "counts": {}})), \
             patch("eval.question_filter._markdown_report", return_value="offline fixture"):
            target = self.root / "filtered"
            export_filtered_benchmark(self.source / "06_grounded_questions.json", {}, target)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        view = SimpleNamespace(manifest=manifest, has=lambda name: (target / name).is_file(),
            read=lambda name: json.loads((target / name).read_text(encoding="utf-8")))
        self.assertEqual(validate_seed_identity(view, self.wp), self.pack)

    def test_lineage_keeps_denominators_without_certifying_mechanism_necessity(self):
        identity = self.pack["mechanisms"][0]["id"]
        self.write("01_seed_audit.json", {"mechanism_coverage": [{"id": identity, "passed": True}]})
        self.write("02_seed_audit.json", {"mechanism_coverage": [{"mechanism_id": identity,
            "events": {"change": ["event1"]}, "relations": {}}]})
        self.write("02_world_review.json", {"mechanism_coverage": [{"mechanism_id": identity, "status": "witnessed"}]})
        questions = [{"qid": "associated", "line": "L3_process", "aux": {"process": {"witness_refs": {"event_ids": ["event1"]}}}},
                     {"qid": "other", "line": "L1_state"}]
        self.write("04_questions.json", questions)
        self.write("06_grounded_questions.json", questions[1:])
        self.write("06_grounding_report.json", {"n_in": 2, "n_out": 1, "n_pending": 1, "n_dropped": 0})
        result = seed_lineage_report(self.source)
        self.assertEqual(result["questions"]["candidate_count"], 2)
        self.assertEqual(result["questions"]["final_count"], 1)
        self.assertEqual(result["questions"]["routing_counts"]["n_pending"], 1)
        row = result["mechanisms"][0]
        self.assertEqual(len(row["candidate_process_associations"]), 1)
        self.assertEqual(row["final_process_associations"], [])
        self.assertEqual(row["mechanism_necessary_to_answer"], "not_assessed_by_this_report")


if __name__ == "__main__":
    unittest.main()
