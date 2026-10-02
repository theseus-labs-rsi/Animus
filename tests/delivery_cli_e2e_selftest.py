"""Offline public-CLI path for an explicit final-delivery and tokenizer target."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from generation_cli_e2e_selftest import ScriptedServices, offline_factory
from pipeline import calibration, factory, run as run_module


class DeliveryCliE2eTests(unittest.TestCase):
    def test_fresh_seed_through_four_athletes_and_current_delivery_report(self):
        services = ScriptedServices(construction=True)
        original_council = services.council.chat_json
        def ten_line_map(step, messages, **kwargs):
            if step in ("council.map", "council.map_repair"):
                from pipeline.delivery_target import LINE_IDS
                return {"per_line": [{"line": line, "applicable": line == "L1_timeline",
                    "gt_feasible": line == "L1_timeline", "instantiation": "Fixture chronology",
                    "weight_hint": .5 if line == "L1_timeline" else 0} for line in LINE_IDS]}
            return original_council(step, messages, **kwargs)
        services.council.chat_json = ten_line_map
        with tempfile.TemporaryDirectory(prefix="delivery-cli-") as directory, ExitStack() as cleanup, offline_factory(directory, services):
            root = Path(directory)
            resolved = root.resolve()
            if resolved.parent != Path(tempfile.gettempdir()).resolve() or not resolved.name.startswith("delivery-cli-"):
                raise AssertionError("Unexpected temporary test directory")
            cleanup.callback(shutil.rmtree, run_module._io_path(resolved))
            seed = root / "seed.json"
            seed.write_text(json.dumps(services.pack, ensure_ascii=False), encoding="utf-8")
            target = root / "target.json"
            target.write_text(json.dumps({"version": 1, "final_questions": 1,
                "count_stage": "selection_complete", "requested_lines": ["L1_timeline"],
                "per_line_min": {}, "per_line_max": {}, "corpus_tokens": 1,
                "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 1}), encoding="utf-8")
            argv = ["factory", "--seed-pack", str(seed), "--delivery-target", str(target),
                    "--run", "small", "--question-budget", "4", "--haystack-ratio", "0",
                    "--semantic-workers", "1", "--release"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                factory.main()
            run = run_module.Run("seed_" + services.pack["seed_id"], "small")
            report = run.read("10_delivery_target.json")
            measurement = report["corpus_target"]["measurement"]
            self.assertEqual(measurement["tokenizer"]["encoding"], "cl100k_base")
            self.assertEqual(measurement["tokenizer"]["version"], "0.12.0")
            self.assertEqual(len(measurement["tokenizer_id"]), 64)
            self.assertTrue(report["corpus_target"]["current"])
            self.assertEqual(report["totals"]["primary"]["raw_orders"], 4)
            self.assertTrue(report["delivery_target_met"], report)
            self.assertEqual(report["totals"]["primary"]["final_selected"], 1)
            self.assertEqual(len(run.read(calibration.SELECTED_ARTIFACT)), 1)
            self.assertEqual(len(services.athlete_calls), 4 * len(run.read("06_grounded_questions.json")))
            self.assertTrue(run.read(calibration.SELECTION_ARTIFACT)["release_ready"])
            before = (len(services.calls), len(services.athlete_calls))
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                factory.main()
            self.assertEqual((len(services.calls), len(services.athlete_calls)), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
