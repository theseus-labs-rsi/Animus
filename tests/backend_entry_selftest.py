"""Offline backend imports, plus retained checks for optional historical frontend paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = "import socket; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden')); "


class BackendEntryTests(unittest.TestCase):
    def execute(self, cwd: Path, code: str):
        env = dict(os.environ, OPENAI_API_KEY="offline-entry-test", MODEL="offline-model",
                   OPENAI_BASE_URL="http://127.0.0.1:9", PYTHONPATH="", PYTHONIOENCODING="utf-8")
        return subprocess.run([sys.executable, "-B", "-c", OFFLINE + code], cwd=cwd,
                              env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)

    def test_each_first_import_resolves_root(self):
        # Start a fresh interpreter each time so one import cannot hide a broken entry point.
        modules = ("pipeline.factory", "pipeline.grounding", "pipeline.lines.L2_relational",
                   "eval.judge", "pipeline.calibration", "agent_harnesses.cli",
                   "tools.run_original_bc_smoke", "config")
        directories = (ROOT, ROOT / "frontend") if (ROOT / "frontend").is_dir() else (ROOT,)
        for cwd in directories:
            for name in modules:
                with self.subTest(cwd=cwd.name, module=name):
                    code = f"import importlib,json; m=importlib.import_module({name!r}); print(json.dumps(m.__file__))"
                    result = self.execute(cwd, code)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(Path(json.loads(result.stdout.strip())).resolve(),
                                     ROOT.joinpath(*name.split('.')).with_suffix('.py').resolve())

    @unittest.skipUnless((ROOT / "frontend").is_dir(), "Frontend removed from the CLI release; historical test retained")
    def test_old_direct_script_entry_points(self):
        for script in ("frontend/pipeline/factory.py", "frontend/tools/live_demo_worker.py"):
            with self.subTest(script=script):
                code = f"import runpy,sys; sys.argv=[{script!r},'--help']; runpy.run_path({str(ROOT / script)!r},run_name='__main__')"
                result = self.execute(ROOT / "frontend", code)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    @unittest.skipUnless((ROOT / "frontend").is_dir(), "Frontend removed from the CLI release; historical test retained")
    def test_frontend_without_backend_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            detached = Path(temporary) / "frontend"
            detached.mkdir()
            for item in ("pipeline", "tools", "eval"):
                target = detached / item
                target.mkdir()
                shutil.copyfile(ROOT / "frontend" / item / "__init__.py", target / "__init__.py")
            shutil.copyfile(ROOT / "frontend/config.py", detached / "config.py")
            for name in ("pipeline", "tools", "eval", "config"):
                with self.subTest(module=name):
                    result = self.execute(detached, f"import {name}")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("backend is missing", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
