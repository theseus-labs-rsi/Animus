"""No-network native process tests, including real subprocess descendants."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harnesses.runners.native_cli import _run_cli_process
from agent_harnesses.tracks.native import NativeTrackAdapter


class NativeProcessTests(unittest.TestCase):
    def test_utf8_cli_output_in_non_utf8_python(self):
        # Exercise a new interpreter with UTF-8 mode explicitly off. The runner
        # must decode the CLI bytes itself, independently of the local locale.
        script = (
            "import json,sys;from agent_harnesses.runners.native_cli import _run_cli_process;"
            "r=_run_cli_process([sys.executable,'-c',"
            "'import sys;sys.stdout.buffer.write(bytes.fromhex(\"e4b8ade69687\"))'],timeout=5);"
            "print(json.dumps({'answer':r.stdout,'mode':sys.flags.utf8_mode}))"
        )
        completed = subprocess.run([sys.executable, "-X", "utf8=0", "-c", script],
                                   cwd=ROOT, capture_output=True, encoding="utf-8", timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {"answer": "中文", "mode": 0})

    def test_timeout_kills_descendant_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "descendant-survived.txt"
            child = (
                "import pathlib,time;time.sleep(2);"
                f"pathlib.Path({str(marker)!r}).write_text('still running')"
            )
            parent = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
                "sys.stdout.buffer.write(bytes.fromhex('e4b8ade69687'));sys.stdout.flush();"
                "time.sleep(20)"
            )
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                _run_cli_process([sys.executable, "-c", parent], timeout=0.5)
            self.assertLess(time.monotonic() - started, 4)
            self.assertEqual(caught.exception.stdout, "中文")
            # A surviving child would write this marker even after its launcher
            # died. Only wait for this small local probe; no model is involved.
            time.sleep(2.2)
            self.assertFalse(marker.exists(), "CLI descendant survived timeout")

    def test_runner_command_enables_utf8_in_child_interpreter(self):
        command = NativeTrackAdapter().build_command(
            SimpleNamespace(execution={}), SimpleNamespace(runner="native_cli"),
            SimpleNamespace(output_dir="unused", benchmark={"path": "unused"}),
        )
        self.assertEqual(command[:4], [sys.executable, "-X", "utf8", "-m"])

    def test_invalid_completed_answer_is_reported_as_decode_error(self):
        # A malformed answer must not silently turn into replacement characters
        # and then be scored as if it were the athlete's actual answer.
        with self.assertRaises(UnicodeDecodeError):
            _run_cli_process([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'\\xff')"],
                             timeout=5)


if __name__ == "__main__":
    unittest.main()
