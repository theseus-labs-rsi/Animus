"""Offline real-filesystem regression for long run/checkpoint paths on Windows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import run as runtime
from llm_trace import emit, trace_scope


class LongPathTests(unittest.TestCase):
    def setUp(self):
        # No config import is needed for IO tests. Recovery imports the existing
        # factory; prevent dotenv loading and all outbound model/network calls.
        self.enterContext(patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-long-path-test", "MODEL": "offline-long-path-test",
            "OPENAI_BASE_URL": "http://127.0.0.1:1", "PYTHON_DOTENV_DISABLED": "1"}))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.object(socket, "getaddrinfo", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.object(runtime, "_env_snapshot", return_value={}))
        self.root = Path(tempfile.mkdtemp(prefix="long-path-selftest-"))
        self.addCleanup(self.cleanup)

    def cleanup(self):
        resolved = self.root.resolve()
        self.assertTrue(resolved.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(resolved.name.startswith("long-path-selftest-"))
        shutil.rmtree(runtime._io_path(resolved))

    def directory(self, length, tag="case"):
        """Create ordinary directory components, never an overlong component."""
        path = self.root / tag
        while len(str(path)) < length:
            size = min(48, length - len(str(path)) - 1)
            if size <= 0:
                path = path.with_name(path.name + "x")
            else:
                path = path / ("d" * size)
        runtime._io_path(path).mkdir(parents=True)
        return path

    def write(self, path, value):
        runtime._atomic_write_json(path, value)

    def test_atomic_temp_over_260_with_existing_short_parent(self):
        parent = self.directory(187)
        self.assertTrue(parent.is_dir())
        path = parent / ("a" * 64 + ".json")
        legacy_tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        self.assertLess(len(str(path)), 260)
        self.assertGreater(len(str(legacy_tmp)), 260)
        replacements = []
        original_replace = os.replace
        def capture(source, target):
            replacements.append((source, target))
            return original_replace(source, target)
        with patch.object(runtime.os, "replace", side_effect=capture):
            self.write(path, {"中文": "saved", "state": [1, 2, 3]})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["中文"], "saved")
        self.assertEqual(len(replacements), 1)
        self.assertGreater(len(runtime._path_label(replacements[0][0])), 260)
        self.assertFalse(list(runtime._io_path(parent).glob(".*.tmp")))
        if os.name == "nt":
            self.assertTrue(str(replacements[0][0]).startswith("\\\\?\\"))
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
                long_paths_enabled = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
            if not long_paths_enabled:
                with self.assertRaises(OSError):
                    legacy_tmp.write_text("old code cannot write this", encoding="utf-8")

    def test_atomic_target_and_temp_over_260_read_and_cleanup(self):
        parent = self.directory(220)
        path = parent / ("b" * 64 + ".json")
        self.assertGreater(len(str(path)), 260)
        self.write(path, {"checkpoint": "complete"})
        io_path = runtime._io_path(path)
        self.assertTrue(io_path.exists())
        self.assertEqual(json.loads(io_path.read_text(encoding="utf-8")), {"checkpoint": "complete"})
        io_path.unlink()
        self.assertFalse(io_path.exists())
        self.assertFalse(list(runtime._io_path(parent).glob(".*.tmp")))

    def test_replace_failure_keeps_old_bytes_and_cleans_temp(self):
        parent = self.directory(220)
        path = parent / ("c" * 64 + ".json")
        self.write(path, {"old": [1, 2]})
        before = runtime._io_path(path).read_bytes()
        with patch.object(runtime.os, "replace", side_effect=PermissionError("replace blocked")) as replaced, \
             patch.object(runtime.time, "sleep") as slept, self.assertRaisesRegex(PermissionError, "replace blocked"):
            self.write(path, {"new": [3, 4]})
        self.assertEqual(replaced.call_count, 8)
        self.assertEqual(slept.call_count, 7)
        self.assertEqual(runtime._io_path(path).read_bytes(), before)
        self.assertFalse(list(runtime._io_path(parent).glob(".*.tmp")))

    def test_cleanup_failure_does_not_hide_original_replace_error(self):
        parent = self.directory(220)
        path = parent / ("e" * 64 + ".json")
        self.write(path, {"old": True})
        before = runtime._io_path(path).read_bytes()
        original_unlink = Path.unlink
        def blocked_tmp_unlink(target, *args, **kwargs):
            if target.suffix == ".tmp":
                raise PermissionError("cleanup blocked")
            return original_unlink(target, *args, **kwargs)
        with patch.object(runtime.os, "replace", side_effect=OSError("primary replace failure")), \
             patch.object(Path, "unlink", blocked_tmp_unlink), self.assertRaisesRegex(OSError, "primary replace failure"):
            self.write(path, {"new": True})
        self.assertEqual(runtime._io_path(path).read_bytes(), before)
        leftovers = list(runtime._io_path(parent).glob(".*.tmp"))
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(json.loads(leftovers[0].read_text(encoding="utf-8")), {"new": True})
        leftovers[0].unlink()

    def test_json_encoding_failure_keeps_old_checkpoint(self):
        parent = self.directory(220)
        path = parent / ("f" * 64 + ".json")
        self.write(path, {"complete": True})
        before = runtime._io_path(path).read_bytes()
        with self.assertRaises(TypeError):
            self.write(path, {"bad": object()})
        self.assertEqual(runtime._io_path(path).read_bytes(), before)
        self.assertFalse(list(runtime._io_path(parent).glob(".*.tmp")))

    def test_run_resume_trace_lock_and_listing_beyond_260(self):
        base = self.directory(265)
        with patch.object(runtime, "RUNS_DIR", base):
            run = runtime.Run("offline", "resume")
            artifact = "05_signal_checkpoints/" + "1" * 64 + ".json"
            (run.dir / "05_signal_checkpoints").mkdir()
            run.write(artifact, {"bound": "unchanged", "documents": [{"text": "evidence"}]})
            run.log("offline long-path log")
            run.tracer._log(1, "offline.fixture", [], {}, {})
            with trace_scope(run.tracer.pfile.with_name("llm_attempts.jsonl"), "offline.fixture"):
                emit("fixture_event")
            with run.stage_write_lock("one"):
                with self.assertRaises(RuntimeError):
                    with run.stage_write_lock("two"):
                        pass
            resumed = runtime.Run("offline", "resume")
            self.assertEqual(resumed.read(artifact), run.read(artifact))
            self.assertTrue(resumed.has(artifact))
            self.assertEqual(resumed.tracer.n, 1)
            self.assertEqual(runtime._jsonl_record_count(base / "resume/prompts.jsonl"), 1)
            self.assertEqual(runtime.list_runs()[0]["run_id"], "resume")
            self.assertIn("offline long-path log", (resumed.dir / "run.log").read_text(encoding="utf-8"))
            self.assertIn("fixture_event", (resumed.dir / "llm_attempts.jsonl").read_text(encoding="utf-8"))
            (resumed.dir / artifact).unlink()
            self.assertFalse(resumed.has(artifact))

    @unittest.skipUnless(os.name == "nt", "Windows drive/UNC path conversion")
    def test_windows_relative_unc_and_prefix_identity(self):
        plain = self.root / "uncreated" / ".." / "same"
        converted = runtime._io_path(plain)
        self.assertEqual(runtime._io_path(converted), converted)
        self.assertEqual(runtime._path_label(converted), os.path.abspath(plain))
        relative = Path(os.path.relpath(self.root, Path.cwd()))
        self.assertEqual(runtime._io_path(relative), runtime._io_path(self.root))
        unc = Path(r"\\server\share\folder\checkpoint.json")
        extended = runtime._io_path(unc)
        self.assertEqual(str(extended), r"\\?\UNC\server\share\folder\checkpoint.json")
        self.assertEqual(runtime._path_label(extended), str(unc))

    def test_reuse_entry_copies_long_checkpoints_preserving_binding_and_identity(self):
        from tools.run_original_bc_smoke import reuse_original_stages, REUSE_STAGES, REUSE_ARTIFACTS
        source = self.directory(222, "source")
        target_base = self.directory(223, "target")
        target = target_base / "resumed"
        artifacts = {REUSE_ARTIFACTS[name]: {} for name in REUSE_STAGES["corpus"]}
        artifacts.update({"00_about.json": {}, "04_wording_report.json": {"status": "passed"}})
        manifest = {"scenario": "bc_small_office", "status": "done", "config": {}, "algo": {},
            "stages": {name: {"done": True, "artifact": REUSE_ARTIFACTS[name]} for name in REUSE_STAGES["corpus"]}}
        artifacts["manifest.json"] = manifest
        for name, value in artifacts.items():
            self.write(source / name, value)
        checkpoint_dir = runtime._io_path(source / "06_review_checkpoints")
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / ("2" * 64 + ".json")
        payload = b'{"binding":{"source":"original"},"result":{"status":"passed"}}'
        checkpoint.write_bytes(payload)
        self.assertGreater(len(runtime._path_label(checkpoint)), 260)
        before = {p.name: p.read_bytes() for p in runtime._io_path(source).glob("*.json")}
        result = reuse_original_stages(source, target, "corpus", reuse_review_checkpoints=True)
        target_checkpoint = runtime._io_path(target) / "06_review_checkpoints" / checkpoint.name
        self.assertEqual(target_checkpoint.read_bytes(), payload)
        expected_set_hash = hashlib.sha256(json.dumps([(checkpoint.name, hashlib.sha256(payload).hexdigest())],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        receipt = result["derived_from"]["semantic_review_checkpoints"]
        self.assertEqual(receipt["set_sha256"], expected_set_hash)
        self.assertEqual(result["derived_from"]["source_directory"], str(source.resolve()))
        self.assertEqual(receipt["source_directory"], str((source / "06_review_checkpoints").resolve()))
        # An already-prefixed caller records exactly the same path identity and
        # hashes; the copy does not rewrite checkpoint payloads or source files.
        again = reuse_original_stages(runtime._io_path(source), runtime._io_path(target_base / "again"),
                                      "corpus", reuse_review_checkpoints=True)
        self.assertEqual(result["derived_from"], again["derived_from"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in runtime._io_path(source).glob("*.json")})
        self.assertEqual(checkpoint.read_bytes(), payload)
        with patch.object(runtime, "RUNS_DIR", target_base):
            resumed = runtime.Run("bc_small_office", "resumed")
            self.assertEqual(resumed.read("06_review_checkpoints/" + checkpoint.name), json.loads(payload))


if __name__ == "__main__":
    unittest.main(verbosity=2)
