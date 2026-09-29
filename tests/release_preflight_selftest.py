"""Release environment checks must precede generation and never call a model."""
from __future__ import annotations

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "offline-preflight")
os.environ.setdefault("MODEL", "offline-preflight")
from pipeline.calibration import load_config, preflight_release
from pipeline import native_evaluation, calibration, factory
from agent_harnesses.runners import native_cli


class ReleasePreflightTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config(Path(__file__).resolve().parents[1] / "examples/release_four.json")
        self.chat, self.chat_json = Mock(), Mock()
        self.source = SimpleNamespace(API_KEY="hidden-test-key", BASE_URL="https://example.invalid/v1",
                                      chat=self.chat, chat_json=self.chat_json)
        self.judge = SimpleNamespace(_module=SimpleNamespace(config=self.source))
        self.reports = []

    def runtime(self, system, model):
        self.reports.append((system.system_id, deepcopy(model)))
        return {"ok": True, "errors": [], "warnings": [], "binary": "offline-binary",
                "binary_version": "offline-1", "required_env_present": {"KEY": True}}

    def test_checks_every_target_and_judge_without_creating_artifacts_or_calls(self):
        with patch.object(native_cli, "preflight_system", side_effect=self.runtime), \
             patch.object(native_evaluation, "load_judge", return_value=self.judge), \
             patch.object(native_evaluation, "_write", side_effect=AssertionError("read only")), \
             patch.object(native_evaluation, "execute_many", side_effect=AssertionError("no answers")):
            report = preflight_release(self.settings, log=lambda *_: None)
        self.assertTrue(report["ok"])
        self.assertFalse(report["remote_access_verified"])
        self.assertEqual(len(self.reports), 4)
        self.assertEqual(report["judge"]["model"], self.settings["judge_model"])
        self.assertNotIn("hidden-test-key", str(report))
        self.chat.assert_not_called()
        self.chat_json.assert_not_called()

    def test_missing_cli_or_endpoint_is_reported_before_any_generation(self):
        def bad_runtime(system, model):
            return {"ok": False, "errors": ["找不到 dsh", "缺少 DEEPSEEK_API_KEY"]}
        with patch.object(native_cli, "preflight_system", side_effect=bad_runtime), \
             patch.object(native_evaluation, "load_judge", return_value=self.judge):
            with self.assertRaisesRegex(native_evaluation.ConfigurationError, "找不到 dsh.*DEEPSEEK_API_KEY"):
                preflight_release(self.settings, log=lambda *_: None)
        self.chat.assert_not_called()

    def test_missing_judge_key_and_invalid_endpoint_fail_without_secrets(self):
        self.source.API_KEY = ""
        self.source.BASE_URL = "https://username:secret@example.invalid"
        with patch.object(native_cli, "preflight_system", side_effect=self.runtime), \
             patch.object(native_evaluation, "load_judge", return_value=self.judge):
            with self.assertRaises(native_evaluation.ConfigurationError) as caught:
                preflight_release(self.settings, log=lambda *_: None)
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertIn("OPENAI_BASE_URL", str(caught.exception))
        self.assertNotIn("username:secret", str(caught.exception))

    def test_judge_import_system_exit_becomes_clear_config_error(self):
        with patch.object(native_cli, "preflight_system", side_effect=self.runtime), \
             patch.object(native_evaluation, "load_judge", side_effect=SystemExit("hidden-secret")):
            with self.assertRaises(native_evaluation.ConfigurationError) as caught:
                preflight_release(self.settings, log=lambda *_: None)
        self.assertIn("裁判配置读取失败", str(caught.exception))
        self.assertNotIn("hidden-secret", str(caught.exception))

    def test_imported_results_do_not_require_runtime_or_judge(self):
        settings = {**self.settings, "result_dirs": {}, "source_benchmark": "existing-benchmark"}
        with patch.object(native_cli, "preflight_system", side_effect=AssertionError("no runtime needed")), \
             patch.object(native_evaluation, "load_judge", side_effect=AssertionError("no judge needed")):
            self.assertEqual(preflight_release(settings, log=lambda *_: None)["mode"], "import")

    def test_version_warning_is_observable_without_requiring_exact_pin(self):
        messages = []
        def version_warning(system, model):
            return {"ok": True, "errors": [], "warnings": ["CLI 版本与 pin 不一致"], "binary_version": "2"}
        with patch.object(native_cli, "preflight_system", side_effect=version_warning), \
             patch.object(native_evaluation, "load_judge", return_value=self.judge):
            result = preflight_release(self.settings, log=messages.append)
        self.assertTrue(result["ok"])
        self.assertEqual(sum("pin 不一致" in m for m in messages), 4)

    def factory_main(self, arguments, root):
        def run_stub(scenario, run_id, **kwargs):
            return SimpleNamespace(manifest={"config": kwargs["config_meta"]},
                log=lambda *_: None, tracer=SimpleNamespace(n=0), dir=root / run_id)
        with patch.object(factory, "RUNS_DIR", root), \
             patch.object(factory, "Run", side_effect=run_stub), \
             patch.object(sys, "argv", ["factory", "--run", "preflight-fixture", *arguments]):
            factory.main()

    def test_factory_rejects_missing_runtime_before_drive_or_supply_loop(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(calibration, "preflight_release", side_effect=native_evaluation.ConfigurationError("missing dsh")), \
             patch.object(factory, "drive") as drive, \
             patch.object(factory, "build_to_target") as supply, \
             patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.factory_main(["--release", "--min-questions", "10"], Path(temporary))
        self.assertEqual(caught.exception.code, 2)
        drive.assert_not_called()
        supply.assert_not_called()

    def test_factory_only_input_skips_release_runtime_check(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(calibration, "preflight_release", side_effect=AssertionError("no runtime needed")) as preflight, \
             patch.object(factory, "drive") as drive:
            self.factory_main(["--release", "--only", "input"], Path(temporary))
        preflight.assert_not_called()
        drive.assert_called_once()
        self.assertEqual(drive.call_args.args[4], "input")

    def test_factory_import_mode_reaches_driver_without_native_or_judge_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {k: v for k, v in self.settings.items() if k != "systems"}
            settings.update(result_dirs={}, source_benchmark="previous-benchmark")
            config_path = root / "import.json"
            config_path.write_text(json.dumps(settings), encoding="utf-8")
            with patch.object(native_cli, "preflight_system", side_effect=AssertionError("no runtime needed")), \
                 patch.object(native_evaluation, "load_judge", side_effect=AssertionError("no judge needed")), \
                 patch.object(factory, "drive") as drive:
                self.factory_main(["--calibration-config", str(config_path)], root)
            drive.assert_called_once()


if __name__ == "__main__":
    unittest.main()
