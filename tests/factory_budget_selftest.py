"""Factory budget defaults and safe resume configuration (no model calls)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import factory, run as run_module


class BudgetTests(unittest.TestCase):
    def test_new_run_has_bounded_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(factory, "RUNS_DIR", root), patch.object(run_module, "RUNS_DIR", root), \
                 patch.object(sys, "argv", ["factory", "--run", "new", "--to", "input"]):
                factory.main()
            manifest = json.loads((root / "new/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["question_budget"], 30)

    def test_changing_completed_budget_needs_orders_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directory = root / "existing"
            directory.mkdir()
            path = directory / "manifest.json"
            path.write_text(json.dumps({"scenario": "office", "config": {"question_budget": 30},
                                        "stages": {"orders": {"done": True}}, "algo": {}}), encoding="utf-8")
            before = path.read_bytes()
            for suffix in (["--only", "quality", "--force"], ["--from", "questions"], []):
                with self.subTest(suffix=suffix), patch.object(factory, "RUNS_DIR", root), \
                     patch.object(sys, "argv", ["factory", "--run", "existing", "--question-budget", "10", *suffix]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    factory.main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
