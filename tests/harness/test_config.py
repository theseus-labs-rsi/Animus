from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_harnesses.config import (
    CONFIG_ROOT,
    ConfigurationError,
    load_experiment,
    load_registry,
    parse_system,
    resolve_systems,
)


TEMPLATES = CONFIG_ROOT / "experiments" / "templates"


class RegistryTests(unittest.TestCase):
    def test_registry_contains_both_tracks_and_diagnostic(self):
        registry = load_registry()
        self.assertIn("native.codex", registry)
        self.assertIn("memory.simplemem", registry)
        self.assertTrue(registry["diagnostic.builtin"].executable)
        self.assertEqual({item.track for item in registry.values()}, {"native", "memory"})

    def test_benchmark_native_systems_keep_native_tools(self):
        registry = load_registry()
        formal = [
            item
            for item in registry.values()
            if item.track == "native" and item.role == "benchmark"
        ]
        self.assertTrue(formal)
        for item in formal:
            self.assertEqual(item.spec["native_tool_policy"]["mode"], "native")

    def test_memory_answering_model_is_fixed_by_experiment(self):
        registry = load_registry()
        experiment = load_experiment(TEMPLATES / "memory-smoke.toml")
        systems = resolve_systems(experiment, registry)
        self.assertTrue(systems)
        self.assertTrue(all(system.track == "memory" for system in systems))
        self.assertEqual(experiment.answering_model["model_id"], "deepseek-v4-flash")
        self.assertTrue(all("answering_model" not in system.spec for system in systems))
        mem0_target = next(target for target in experiment.targets if target.system_id == "memory.mem0-oss")
        self.assertEqual(mem0_target.memory_config["top_k"], 3)
        self.assertEqual(
            mem0_target.memory_config["internal_model"]["endpoint_profile"], "DEEPSEEK"
        )
        self.assertTrue(registry["memory.mem0-oss"].executable)

    def test_memory_config_rejects_inline_credentials(self):
        template = (TEMPLATES / "mem0-smoke.toml").read_text(encoding="utf-8")
        template = template.replace(
            '[targets.memory_config.internal_model]\nmodel_id = "deepseek-v4-flash"',
            '[targets.memory_config.internal_model]\nmodel_id = "deepseek-v4-flash"\napi_key = "must-not-live-here"',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text(template, encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "不得保存凭证"):
                load_experiment(path)

    def test_duplicate_system_ids_are_rejected(self):
        row = """
schema_version = 2
[[systems]]
schema_version = 2
id = "x.y"
track = "native"
display_name = "x"
status = "candidate"
role = "benchmark"
implementation = { runner = "planned", executable = false }
spec = { harness = { id = "x" }, native_tool_policy = { mode = "native" } }
[[systems]]
schema_version = 2
id = "x.y"
track = "native"
display_name = "x again"
status = "candidate"
role = "benchmark"
implementation = { runner = "planned", executable = false }
spec = { harness = { id = "x" }, native_tool_policy = { mode = "native" } }
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "systems.toml").write_text(row, encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_registry(root)

    def test_active_system_cannot_keep_tbd_or_planned_runner(self):
        row = {
            "schema_version": 2,
            "id": "vendor.harness",
            "track": "native",
            "display_name": "candidate promoted too early",
            "status": "active",
            "role": "benchmark",
            "implementation": {"runner": "planned", "executable": False},
            "spec": {
                "harness": {"id": "harness"},
                "native_tool_policy": {"mode": "native"},
            },
        }
        with self.assertRaises(ConfigurationError):
            parse_system(row, Path("test.toml"))


if __name__ == "__main__":
    unittest.main()
