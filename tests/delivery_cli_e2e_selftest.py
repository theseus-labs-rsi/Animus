"""Offline public-CLI path for an explicit final-delivery and tokenizer target."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
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
from pipeline import calibration, factory, production, run as run_module


class DeliveryCliE2eTests(unittest.TestCase):
    def test_generation_target_finishes_quality_and_resumes_without_selection(self):
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
            cleanup.callback(shutil.rmtree, run_module._io_path(root.resolve()))
            seed = root / "seed.json"
            seed.write_text(json.dumps(services.pack, ensure_ascii=False), encoding="utf-8")
            target = root / "target.json"
            target.write_text(json.dumps({"version": 2, "candidate_questions": 4,
                "count_stage": "generation_quality", "requested_lines": ["L1_timeline"],
                "corpus_tokens": 1, "core_tokens": 1, "filler_ratio": 0,
                "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 1}), encoding="utf-8")
            argv = ["factory", "--seed-pack", str(seed), "--delivery-target", str(target),
                    "--run", "generation", "--semantic-workers", "1"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                factory.main()
            run = run_module.Run("seed_" + services.pack["seed_id"], "generation")
            report = run.read("10_delivery_target.json")
            state = run.read(production.STATE)
            self.assertEqual(state["status"], "generation_complete", state)
            self.assertEqual(state["result"]["counts"], {"L1_timeline": 4})
            self.assertTrue(state["result"]["passed"], state["result"])
            self.assertTrue(report["candidate_target_met"])
            self.assertTrue(report["delivery_target_met"])
            self.assertTrue(report["corpus_current"])
            self.assertTrue(report["corpus_target"]["target_met"])
            self.assertEqual(report["selection_scope"], "not_run")
            self.assertEqual(run.manifest["config"]["acceptance_scope"]["count_stage"], "generation_quality")
            self.assertFalse(run.has(calibration.SELECTION_ARTIFACT))
            self.assertEqual(services.athlete_calls, [])
            before = (len(services.calls), services.judge_calls)
            with patch.object(sys, "argv", ["factory", "--run", "generation"]), redirect_stdout(io.StringIO()):
                factory.main()
            self.assertEqual((len(services.calls), services.judge_calls), before)
            self.assertEqual(run.read(production.STATE)["status"], "generation_complete")
            for flags in (["--release"], ["--to", "selection"], ["--only", "calibration"], ["--from", "calibration"]):
                with self.subTest(flags=flags), patch.object(sys, "argv", ["factory", "--run", "generation", *flags]), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as stopped:
                        factory.main()
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn("V2 delivery targets end at generation quality", errors.getvalue())
            self.assertEqual((len(services.calls), services.judge_calls), before)

    def test_generation_target_rejects_fresh_and_frozen_selection_config_before_calls(self):
        target = {"version": 2, "candidate_questions": 4, "count_stage": "generation_quality",
            "requested_lines": ["L1_timeline"], "corpus_tokens": 1, "core_tokens": 1,
            "filler_ratio": 0, "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 1}
        services = ScriptedServices(construction=True)
        with tempfile.TemporaryDirectory(prefix="delivery-guard-") as directory, offline_factory(directory, services) as runs:
            path = Path(directory) / "target.json"
            path.write_text(json.dumps(target), encoding="utf-8")
            runs.joinpath("frozen").mkdir(parents=True)
            runs.joinpath("frozen", "manifest.json").write_text(json.dumps({"scenario": "office", "config": {
                "delivery_target": target, "calibration": {"fixture": True}}}), encoding="utf-8")
            commands = (["factory", "--delivery-target", str(path), "--run", "fresh", "--release"],
                        ["factory", "--run", "frozen"])
            for argv in commands:
                with self.subTest(argv=argv), patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as stopped:
                        factory.main()
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn("V2 delivery targets end at generation quality", errors.getvalue())
            self.assertFalse(runs.joinpath("fresh").exists())
            runs.joinpath("bad-scope").mkdir()
            runs.joinpath("bad-scope", "manifest.json").write_text(json.dumps({"scenario": "office", "config": {
                "delivery_target": target, "acceptance_scope": {"version": 1, "count_stage": "selection_complete"}}}), encoding="utf-8")
            with patch.object(sys, "argv", ["factory", "--run", "bad-scope"]), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    factory.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("Frozen V2 acceptance scope must be generation_quality", errors.getvalue())
            self.assertEqual(services.calls, [])
            self.assertEqual(services.athlete_calls, [])

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
