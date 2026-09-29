"""Offline identity checks for comparisons through the original factory stages."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.seed_pack import seed_contract, seed_digest, seed_input
from tools.run_original_bc_smoke import reuse_original_stages
from tests.seed_contract_selftest import fixture as synthetic_seed


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.target = Path(self.temp.name) / "target"
        self.source.mkdir()
        pack = synthetic_seed()
        description, examples = seed_input(pack)
        digest = seed_digest(pack)
        self.write("00_seed_pack.json", pack)
        self.write("00_input.json", {"description": description, "few_shot": examples,
            "seed": {"seed_id": pack["seed_id"], "family": pack["family"], "digest": digest,
                     "artifact": "00_seed_pack.json", "schema_version": pack["schema_version"]}})
        self.write("01_whitepaper.json", {"seed_contract": seed_contract(pack)})
        for name in ("00_about.json", "01_seed_audit.json", "02_seed_audit.json", "02_world.json"):
            self.write(name, {})
        self.manifest = {"scenario": "seed_" + pack["seed_id"], "status": "done",
            "config": {"seed_id": pack["seed_id"], "seed_pack_digest": digest,
                       "seed_pack_path": "missing mutable source.json", "augment": True},
            "algo": {"seed": {"digest": digest}, "entities": 8, "quality": "stale"},
            "stages": {name: {"done": True, "artifact": artifact} for name, artifact in
                       [("input", "00_input.json"), ("whitepaper", "01_whitepaper.json"), ("world", "02_world.json")]}}
        self.write("manifest.json", self.manifest)

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_whitepaper_reuse_keeps_frozen_seed_and_excludes_old_world(self):
        result = reuse_original_stages(self.source, self.target, "whitepaper")
        for name in ("00_input.json", "00_seed_pack.json", "01_whitepaper.json", "01_seed_audit.json"):
            self.assertEqual((self.source / name).read_bytes(), (self.target / name).read_bytes())
        self.assertFalse((self.target / "02_world.json").exists())
        self.assertNotIn("entities", result["algo"])
        self.assertNotIn("quality", result["algo"])
        self.assertNotIn("augment", result["config"])
        self.assertEqual(set(result["stages"]), {"input", "whitepaper"})

    def test_world_reuse_keeps_seed_world_audit(self):
        result = reuse_original_stages(self.source, self.target, "world")
        self.assertTrue((self.target / "02_seed_audit.json").is_file())
        self.assertEqual(result["algo"]["entities"], 8)

    def test_changed_seed_identity_rejected_before_destination_created(self):
        pack = json.loads((self.source / "00_seed_pack.json").read_text(encoding="utf-8"))
        pack["title"] += " changed"
        self.write("00_seed_pack.json", pack)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "whitepaper")
        self.assertFalse(self.target.exists())

    def test_incomplete_or_active_source_rejected(self):
        self.manifest["status"] = "running"
        self.write("manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "world")
        self.assertFalse(self.target.exists())

    def test_office_reuse_remains_unseeded(self):
        self.manifest.update(scenario="bc_small_office", config={})
        self.write("manifest.json", self.manifest)
        self.write("01_whitepaper.json", {})
        (self.source / "00_seed_pack.json").unlink()
        result = reuse_original_stages(self.source, self.target, "world")
        self.assertEqual(result["scenario"], "bc_small_office")
        self.assertFalse((self.target / "00_seed_pack.json").exists())

    def question_source(self):
        for name, artifact in [("orders", "03_orders.json"),
                               ("well_posed", "03_well_posed_report.json"),
                               ("questions", "04_questions.json")]:
            self.write(artifact, {"frozen": name})
            self.manifest["stages"][name] = {"done": True, "artifact": artifact}
        self.write("04_wording_report.json", {"status": "passed"})
        self.manifest["stages"]["corpus"] = {"done": False, "status": "failed"}
        self.manifest["status"] = "failed"
        self.manifest["algo"].update(questions=4, docs=3, render_strategy="stale")
        self.write("manifest.json", self.manifest)

    def test_question_resume_preserves_checkpoint_bytes_and_provenance(self):
        self.question_source()
        self.write("05_corpus.ckpt.json", {"identity": "factory-validates-this",
            "done_weeks": [2, 3], "corpus": {"sessions": []}})
        result = reuse_original_stages(self.source, self.target, "questions")
        for name in ("04_questions.json", "04_wording_report.json", "05_corpus.ckpt.json"):
            self.assertEqual((self.source / name).read_bytes(), (self.target / name).read_bytes())
            self.assertIn(name, result["derived_from"]["input_files"])
        self.assertNotIn("corpus", result["stages"])
        self.assertEqual(result["algo"]["questions"], 4)
        self.assertNotIn("docs", result["algo"])
        self.assertNotIn("render_strategy", result["algo"])

    def test_question_resume_also_supports_no_checkpoint(self):
        self.question_source()
        reuse_original_stages(self.source, self.target, "questions")
        self.assertFalse((self.target / "05_corpus.ckpt.json").exists())

    def test_exact_corpus_resume_copies_review_checkpoints_with_receipt(self):
        self.question_source()
        self.write("05_corpus.json", {"corpus": {"sessions": []}})
        self.manifest["stages"]["corpus"] = {
            "done": True, "status": "succeeded", "artifact": "05_corpus.json"}
        self.manifest["status"] = "done"
        self.write("manifest.json", self.manifest)
        checkpoints = self.source / "06_review_checkpoints"
        checkpoints.mkdir()
        (checkpoints / "a.json").write_bytes(b'{"saved":1}')
        (checkpoints / "b.json").write_bytes(b'{"saved":2}')
        result = reuse_original_stages(
            self.source, self.target, "corpus", reuse_review_checkpoints=True)
        for name in ("a.json", "b.json"):
            self.assertEqual((checkpoints / name).read_bytes(),
                             (self.target / "06_review_checkpoints" / name).read_bytes())
        receipt = result["derived_from"]["semantic_review_checkpoints"]
        self.assertEqual(receipt["count"], 2)
        self.assertEqual(receipt["total_bytes"], 22)
        self.assertEqual(len(receipt["set_sha256"]), 64)

    def test_review_checkpoints_require_exact_corpus_and_existing_files(self):
        self.question_source()
        with self.assertRaisesRegex(ValueError, "exact corpus reuse"):
            reuse_original_stages(
                self.source, self.target, "questions", reuse_review_checkpoints=True)
        self.assertFalse(self.target.exists())

    def test_disclosure_recovery_accepts_stale_downstream_running_manifest(self):
        self.question_source()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        self.manifest.update(status="running", current_stage="grounding")
        self.write("manifest.json", self.manifest)
        result = reuse_original_stages(
            self.source, self.target, "questions", repair_disclosure=True)
        provenance = result["derived_from"]
        self.assertTrue(provenance["stale_downstream_run_accepted"])
        self.assertEqual(provenance["source_manifest_status"], "running")
        self.assertEqual(provenance["source_manifest_current_stage"], "grounding")
        self.assertEqual((self.source / "04_questions.json").read_bytes(),
                         (self.target / "04_questions.json").read_bytes())

    def test_disclosure_recovery_accepts_interrupted_disclosure_manifest(self):
        self.question_source()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        self.manifest.update(status="running", current_stage="disclosure")
        self.write("manifest.json", self.manifest)
        result = reuse_original_stages(
            self.source, self.target, "questions", repair_disclosure=True)
        provenance = result["derived_from"]
        self.assertTrue(provenance["stale_downstream_run_accepted"])
        self.assertEqual(provenance["source_manifest_current_stage"], "disclosure")
        self.assertEqual((self.source / "04_questions.json").read_bytes(),
                         (self.target / "04_questions.json").read_bytes())

    def test_disclosure_recovery_refreshes_stale_review_instead_of_reusing_it(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "passed", "historical": True})
        with patch("pipeline.world_semantics.enabled", return_value=True), \
             patch("pipeline.world_semantics.validate_review",
                   return_value=[{"code": "missing_or_stale_world_review"}]):
            result = reuse_original_stages(
                self.source, self.target, "questions", repair_disclosure=True)
        self.assertEqual((self.target / "02_world_review.json").read_text(encoding="utf-8"),
                         (self.source / "02_world_review.json").read_text(encoding="utf-8"))
        self.assertEqual(result["derived_from"]["recovery"],
                         "complete_or_validate_disclosure_then_resume_downstream")

    def test_disclosure_recovery_drops_errored_review_checkpoint(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"bad": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertFalse((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_nonerror_review_checkpoint(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "unresolved"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_checkpoint_after_temporary_windows_lock(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error",
                   "error": "Original reviewer execution failed: [WinError 5] access denied"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_checkpoint_after_repeated_review_revision(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error",
                   "error": "Four consecutive invalid world review actions: "
                            "Submitted review revision did not change the opinion"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_incomplete_questions_cannot_be_reused(self):
        self.question_source()
        self.manifest["stages"]["questions"]["done"] = False
        self.write("manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "questions")
        self.assertFalse(self.target.exists())

    def test_historical_world_checkpoint_recompiles_without_generation_calls(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        result = reuse_original_stages(source, self.target, "whitepaper",
                                      reuse_world_checkpoint=True, model="glm-5.3-flash")
        checkpoint = result["derived_from"]["world_construction_checkpoint"]
        self.assertEqual((source / checkpoint).read_bytes(), (self.target / checkpoint).read_bytes())
        self.assertTrue(result["derived_from"]["world_review_required"])
        self.assertFalse((self.target / "02_world.json").exists())
        from pipeline import world_agent, disclosure, closed_loop, run as run_module
        from pipeline.world_state import WorldState
        from pipeline.targetspec import TargetSpec
        captured = []
        class DisclosureReached(Exception):
            pass
        def denied(*args, **kwargs):
            self.fail("A completed construction checkpoint must not call any author")
        def at_disclosure(wp, ws, *args, **kwargs):
            captured.append(disclosure._world(ws))
            raise DisclosureReached()
        with patch.object(world_agent.config, "STRUCTURE_MODEL", "glm-5.3-flash"), \
             patch.object(run_module, "RUNS_DIR", self.target.parent), \
             patch.object(disclosure, "author_plan", side_effect=at_disclosure):
            run = run_module.Run(result["scenario"], self.target.name)
            run.log = lambda *_: None
            run.tracer.chat_json = denied
            with self.assertRaises(DisclosureReached):
                closed_loop.build_to_target(run, TargetSpec(min_questions=200, total_only=True,
                                            max_world_entities=80, time_span_weeks=24))
        old = WorldState.from_dict(json.loads((source / "02_world_candidate.json").read_text(encoding="utf-8")))
        self.assertEqual(captured, [disclosure._world(old)])

    def test_wrong_world_checkpoint_model_rejected_before_creating_destination(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            reuse_original_stages(source, self.target, "whitepaper",
                                  reuse_world_checkpoint=True, model="wrong-model")
        self.assertFalse(self.target.exists())

    def test_corrupt_world_checkpoint_rejected_and_never_promoted_to_reviewed_world(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        result = reuse_original_stages(source, self.target, "whitepaper",
                                      reuse_world_checkpoint=True, model="glm-5.3-flash")
        checkpoint = self.target / result["derived_from"]["world_construction_checkpoint"]
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["steps"] += 1
        checkpoint.write_text(json.dumps(state), encoding="utf-8")
        other = Path(self.temp.name) / "rejected"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            reuse_original_stages(self.target, other, "whitepaper",
                                  reuse_world_checkpoint=True, model="glm-5.3-flash")
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
