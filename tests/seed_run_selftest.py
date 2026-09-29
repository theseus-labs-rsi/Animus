"""Offline factory integration: frozen inputs, restart safety and shared guards."""
from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import factory
from pipeline import run as run_module
from pipeline.closed_loop import (_ensure_event_role_capacity, _ensure_l7_capacity,
                                  _scale_world_contract)
from pipeline.run import Run, drive
from pipeline.seed_pack import (SeedPackError, attach_seed_contract, load_seed_pack,
                                seed_digest, validate_seed_blueprint)
from pipeline.seed_run import (SEED_ARTIFACT, frozen_seed, seed_config,
                               validate_seed_identity, validate_seed_input)
from pipeline.seed_world import SeedWorldError
from pipeline.world_blueprint import normalize_world_blueprint
from pipeline.world_state import WorldState
from tests.seed_contract_selftest import fixture as synthetic_seed


def minimal_whitepaper(pack):
    bp = deepcopy(pack["blueprint_requirements"])
    minimum = bp["temporal_model"]["min_sessions"]
    bp["temporal_model"] = {"unit": "week", "cadence": "weekly", "step_days": 7,
                            "n_sessions": minimum}
    bp["version"] = 1
    for item in bp["entity_types"]:
        item.setdefault("noun", item["id"])
        item.setdefault("count", 2)
    if not any(item.get("primary") for item in bp["entity_types"]):
        bp["entity_types"][0]["primary"] = True
    wp = {"world_blueprint": normalize_world_blueprint(bp),
          "domain_profile": {"field_schema": [], "medium": "documents"},
          "shared_world_spec": {"entities": {"count": sum(t["count"] for t in bp["entity_types"])},
                                "timeline": {"n_sessions": minimum}},
          "active_lines": []}
    return attach_seed_contract(wp, pack)


class SeedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.patch_run = patch.object(run_module, "RUNS_DIR", self.runs)
        self.patch_factory = patch.object(factory, "RUNS_DIR", self.runs)
        self.patch_run.start()
        self.patch_factory.start()
        self.addCleanup(self.patch_run.stop)
        self.addCleanup(self.patch_factory.stop)
        self.pack = synthetic_seed()
        self.pack_path = self.root / "pack.json"
        self.pack_path.write_text(json.dumps(self.pack, ensure_ascii=False), encoding="utf-8")

    def new_run(self, identity="seed_test"):
        return Run("seed_" + self.pack["seed_id"], identity,
                   config_meta=seed_config(self.pack_path))

    def input_run(self):
        run = self.new_run()
        drive(run, factory.STAGES, to_stage="input")
        return run

    def test_synthetic_v1_and_v2_packs_are_executable_blueprint_requirements(self):
        example = ROOT / "skills/realfiles-to-seedjson/examples/seed_v2_example.json"
        for pack in (synthetic_seed(), load_seed_pack(example)):
            with self.subTest(schema_version=pack["schema_version"]):
                wp = minimal_whitepaper(pack)
                self.assertTrue(wp["seed_audit"]["passed"])
                self.assertEqual(seed_digest(wp["seed_contract"]), seed_digest(pack))

    def test_input_freezes_and_forced_resume_uses_snapshot(self):
        run = self.input_run()
        expected_input = run.read("00_input.json")
        self.assertEqual(frozen_seed(run), self.pack)
        # Source file may be moved/edited after a run starts; its frozen input
        # must remain identical even when force reruns the input stage.
        self.pack_path.write_text("not JSON any more", encoding="utf-8")
        resumed = Run(run.scenario, run.run_id)
        drive(resumed, factory.STAGES, to_stage="input", force=True)
        self.assertEqual(resumed.read("00_input.json"), expected_input)
        self.assertEqual(validate_seed_input(resumed, expected_input), self.pack)

    def test_changed_snapshot_and_input_are_rejected(self):
        run = self.input_run()
        original = run.read(SEED_ARTIFACT)
        altered = deepcopy(original)
        altered["title"] += " edited"
        run.write(SEED_ARTIFACT, altered)
        with self.assertRaises(SeedPackError):
            frozen_seed(run)
        run.write(SEED_ARTIFACT, original)
        scenario = run.read("00_input.json")
        scenario["few_shot"][0]["content"] += " stale input"
        with self.assertRaises(SeedPackError):
            validate_seed_input(run, scenario)

    def test_seed_cannot_be_added_or_replaced_on_existing_run(self):
        cfg = seed_config(self.pack_path)
        self.assertEqual(seed_config(self.pack_path, cfg), cfg)
        with self.assertRaises(SeedPackError):
            seed_config(self.pack_path, {})
        altered = deepcopy(self.pack)
        altered["title"] += " changed"
        self.pack_path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaises(SeedPackError):
            seed_config(self.pack_path, cfg)

    def test_whitepaper_stage_passes_pack_and_writes_audit(self):
        run = self.input_run()
        wp = minimal_whitepaper(self.pack)
        with patch.object(factory, "central_office", return_value=wp) as council:
            drive(run, factory.STAGES, only="whitepaper")
        self.assertEqual(council.call_args.kwargs["seed_pack"], self.pack)
        self.assertTrue(run.read("01_seed_audit.json")["passed"])
        self.assertEqual(validate_seed_identity(run, run.read("01_whitepaper.json")), self.pack)

    def test_wrong_whitepaper_identity_rejected_before_generation(self):
        run = self.input_run()
        altered = deepcopy(self.pack)
        altered["description"] += " different"
        run.write("01_whitepaper.json", minimal_whitepaper(altered))
        with patch.object(factory, "build_world") as build:
            with self.assertRaises(SeedPackError):
                factory.stage_world(run)
        build.assert_not_called()

    def test_world_guard_rejects_before_publish_and_on_reload(self):
        run = self.input_run()
        wp = minimal_whitepaper(self.pack)
        run.write("01_whitepaper.json", wp)
        invalid_world = WorldState(n_sessions=wp["world_blueprint"]["temporal_model"]["n_sessions"],
                                   world_blueprint=wp["world_blueprint"])
        with patch.object(factory, "build_world", return_value=invalid_world), \
             patch.object(factory, "_prepare_lines"):
            with self.assertRaises(SeedWorldError):
                factory.stage_world(run)
        self.assertFalse(run.has("02_world.json"))
        run.write("02_world.json", invalid_world.to_dict())
        with patch.object(factory, "run_lines") as orders, \
             patch.object(factory, "render_corpus") as render:
            with self.assertRaises(SeedWorldError):
                factory.stage_orders(run)
            with self.assertRaises(SeedWorldError):
                factory.stage_corpus(run)
        orders.assert_not_called()
        render.assert_not_called()

    def test_missing_identity_cannot_turn_seed_into_legacy(self):
        run = self.input_run()
        run.manifest["config"].clear()
        with self.assertRaises(SeedPackError):
            validate_seed_identity(run, {"active_lines": []})

    def test_closed_loop_scaling_preserves_required_minima(self):
        wp = minimal_whitepaper(self.pack)
        _scale_world_contract(wp, 1, 2)
        _ensure_event_role_capacity(wp)
        _ensure_l7_capacity(wp, 100)
        report = validate_seed_blueprint(wp, self.pack)
        self.assertTrue(report["passed"], report["issues"])

    def test_cli_to_input_and_changed_seed_leave_original_manifest_intact(self):
        with patch.object(sys, "argv", ["factory", "--seed-pack", str(self.pack_path),
                                         "--run", "seed_cli_test", "--to", "input"]):
            factory.main()
        manifest = self.runs / "seed_cli_test" / "manifest.json"
        original = manifest.read_bytes()
        altered = deepcopy(self.pack)
        altered["title"] += " new version"
        self.pack_path.write_text(json.dumps(altered), encoding="utf-8")
        with patch.object(sys, "argv", ["factory", "--seed-pack", str(self.pack_path),
                                         "--run", "seed_cli_test", "--to", "input"]), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failed:
                factory.main()
        self.assertEqual(failed.exception.code, 2)
        self.assertEqual(manifest.read_bytes(), original)

    def test_legacy_input_and_whitepaper_keep_existing_call_contract(self):
        run = Run("office", "legacy_test")
        factory.stage_input(run)
        self.assertNotIn("seed", run.read("00_input.json"))
        with patch.object(factory, "central_office", return_value={"active_lines": []}) as council:
            factory.stage_whitepaper(run)
        self.assertEqual(council.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
