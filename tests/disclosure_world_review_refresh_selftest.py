from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import factory, world_semantics


class Run:
    def __init__(self, review, warning=False):
        self.files = {world_semantics.REVIEW_ARTIFACT: review}
        if warning:
            self.files[world_semantics.WARNING_ARTIFACT] = {"status": "warning"}
    def has(self, name):
        return name in self.files
    def read(self, name):
        return self.files[name]


class DisclosureWorldReviewRefreshTests(unittest.TestCase):
    def test_exact_execution_warning_still_requires_fresh_review(self):
        run = Run({"status": "error"}, warning=True)
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertFalse(factory._release_current_world_review(run, {}, object(), {}))

    def test_current_passed_review_can_be_reused(self):
        run = Run({"status": "passed"})
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertTrue(factory._release_current_world_review(run, {}, object(), {}))

    def test_invalid_or_errored_review_requires_refresh(self):
        for review, errors in (({"status": "error"}, []), ({"status": "passed"}, [{"code": "stale"}])):
            with self.subTest(review=review, errors=errors), \
                    patch.object(world_semantics, "validate_review", return_value=errors):
                self.assertFalse(factory._release_current_world_review(Run(review), {}, object(), {}))

    def test_public_repair_retains_old_review_transcript_and_counts_each_call_once(self):
        old = {"status": "failed", "repair_targets": {"intrinsic": [], "structure": False,
                "disclosure": True}}
        fresh = {"status": "passed", "repair_targets": {"intrinsic": [], "structure": False}}

        class World:
            disclosure = {"strategy": "direct-disclosure/v2", "plan_hash": "old", "records": [],
                          "undisclosed": []}

            def to_dict(self):
                return {"disclosure": self.disclosure}

        class StageRun:
            def __init__(self, path):
                self.dir = path
                self.tracer = object()
                self.files = {factory.ART["whitepaper"]: {}, factory.ART["input"]: {},
                              factory.ART["world"]: {"disclosure": World.disclosure}}

            def has(self, name):
                return name in self.files

            def read(self, name):
                return self.files[name]

            def write(self, name, value):
                self.files[name] = value

        with tempfile.TemporaryDirectory() as folder:
            run = StageRun(Path(folder))
            checkpoint = run.dir / "02_world_review_historical.ckpt.json"
            checkpoint.write_bytes(b"historical review transcript")
            with patch.object(factory.WorldState, "from_dict", return_value=World()), \
                    patch.object(factory, "validate_seed_identity"), \
                    patch.object(factory, "validate_seed_world"), \
                    patch.object(factory, "_release_current_world_review", return_value=False), \
                    patch.object(factory, "_recoverable_current_world_review", return_value=None), \
                    patch.object(factory, "_publish_world_bundle"), \
                    patch("pipeline.disclosure.enabled", return_value=True), \
                    patch("pipeline.disclosure.validate_plan", return_value=[]), \
                    patch("pipeline.disclosure._world", return_value={"truth": "unchanged"}), \
                    patch("pipeline.disclosure_batches.repair", return_value={
                        "status": "ready", "repaired_record_ids": ["d14"]}), \
                    patch.object(world_semantics, "review_world", side_effect=[old, fresh]) as reviewer, \
                    patch.object(world_semantics, "validate_review", return_value=[]):
                factory.stage_disclosure(run)

            self.assertEqual(reviewer.call_count, 2)
            self.assertEqual(run.read("02_world_review_attempts.json")["attempts"], [old, fresh])
            self.assertEqual(checkpoint.read_bytes(), b"historical review transcript")
            self.assertEqual(run.read("02_disclosure_semantic_repair.json")[
                "retained_review_checkpoints"], 1)


if __name__ == "__main__":
    unittest.main()
