"""Offline cancellation regressions with actual local processes, no provider."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_batch_stop_target", ROOT / "tools/run_original_bc_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reparented zombie has stopped executing even before its new parent reaps it.
    status = Path(f"/proc/{pid}/stat")
    if status.exists() and status.read_text().split(") ", 1)[1].split()[0] == "Z":
        return False
    return True


class StopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="original_batch_stop_offline_")
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        for folder in ("pipeline", "eval", "seeds", "tools"):
            (self.root / folder).mkdir()
        for name in ("pipeline/a.py", "eval/a.py", "config.py", "llm_transport.py", "llm_trace.py",
                     "tools/run_original_bc_smoke.py", "tools/run_original_bc_batch.py"):
            (self.root / name).write_text("# offline fixture\n", encoding="utf-8")
        for name in ("sample_a", "sample_b", "sample_c"):
            batch.write(self.root / "seeds" / (name + ".json"), {"seed_id": name})
        root_patch = patch.object(batch, "ROOT", self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        network.start(); self.addCleanup(network.stop)

    def prepare(self, *extra):
        seeds = [] if "--seeds" in extra else ["--seeds", "sample_a", "sample_b", "sample_c"]
        return batch.prepare(batch.parser().parse_args(["--batch", "offline_stop", *seeds, *extra]))

    def request_after_marker(self, directory, marker):
        errors = []
        def request():
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if marker.is_file():
                        try:
                            json.loads(marker.read_text(encoding="utf-8"))
                        except (ValueError, OSError):
                            pass
                        else:
                            batch.request_stop(directory, "Offline test: stop known active child")
                            return
                    time.sleep(0.02)
                raise AssertionError("Local child never became ready")
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        return thread, errors

    def test_stop_request_is_prepared_reasoned_and_idempotent(self):
        with self.assertRaises(ValueError):
            batch.request_stop(self.root, "no plan")
        directory, _ = self.prepare()
        with self.assertRaises(ValueError):
            batch.request_stop(directory, " ")
        first = batch.request_stop(directory, "The operator found a reproducible defect")
        self.assertEqual(first, batch.request_stop(directory, "second request"))
        self.assertEqual(batch.digest(directory / "plan.json"), first["plan_sha256"])

    def test_stop_before_batch_keeps_all_slots_unstarted(self):
        directory, plan = self.prepare("--continue-on-failure")
        batch.request_stop(directory, "Stop before dispatch")
        with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("Must not spawn")):
            report = batch.execute(directory)
        self.assertEqual("cancelled", report["status"])
        self.assertEqual({"completed": 0, "failed": 0, "not_started": 3, "cancelled": 0}, report["counts"])
        self.assertEqual("cancelled", batch.read(directory / "execution/report.json")["status"])
        self.assertTrue(all(not Path(row["run_dir"]).exists() for row in plan["runs"]))
        self.assertTrue(all(row["reason"] == "stop_requested" for row in report["runs"]))

    def test_real_child_and_grandchild_terminated_with_stop_file(self):
        directory, _ = self.prepare()
        marker = self.root / "tree.json"
        script = self.root / "tree.py"
        script.write_text('''import json, os, pathlib, subprocess, sys, time
options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **options)
pathlib.Path(sys.argv[1]).write_text(json.dumps({"child": os.getpid(), "grandchild": child.pid}), encoding="utf-8")
time.sleep(30)
''', encoding="utf-8")
        thread, errors = self.request_after_marker(directory, marker)
        result = batch.launch([sys.executable, "-B", str(script), str(marker)], self.root / "tree.log", 15,
                              stop_request_path=directory / "stop_request.json")
        thread.join(11)
        self.assertFalse(errors, errors)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["child_terminated"])
        pids = json.loads(marker.read_text(encoding="utf-8"))
        # Windows venv python may spawn a redirector child with a separate PID.
        self.assertFalse(alive(result["pid"]))
        self.assertFalse(alive(pids["child"]))
        self.assertFalse(alive(pids["grandchild"]))

    def test_real_batch_cancelled_no_second_run_even_continue_enabled(self):
        # This exact fake wrapper is launched by the original prepared command.
        script = '''import json, os, pathlib, sys, time
name=sys.argv[sys.argv.index("--run")+1]
folder=pathlib.Path.cwd()/"output"/"runs"/name
folder.mkdir(parents=True)
(pathlib.Path.cwd()/"started.json").write_text(json.dumps({"pid": os.getpid(), "run": name}), encoding="utf-8")
time.sleep(30)
'''
        (self.root / "tools/run_original_bc_smoke.py").write_text(script, encoding="utf-8")
        directory, plan = self.prepare("--continue-on-failure")  # default timeout=None
        thread, errors = self.request_after_marker(directory, self.root / "started.json")
        report = batch.execute(directory)
        thread.join(11)
        self.assertFalse(errors, errors)
        self.assertFalse(thread.is_alive())
        self.assertEqual("cancelled", report["status"])
        self.assertEqual({"completed": 0, "failed": 0, "not_started": 2, "cancelled": 1}, report["counts"])
        self.assertEqual("cancelled", report["runs"][0]["status"])
        self.assertTrue(report["runs"][0]["process"]["child_terminated"])
        self.assertFalse(alive(report["runs"][0]["process"]["pid"]))
        self.assertFalse(Path(plan["runs"][1]["run_dir"]).exists())
        self.assertFalse((directory / "execution/002.log").exists())
        self.assertEqual(report, batch.read(directory / "execution/report.json"))
        self.assertEqual(0, report["trace_totals"]["requests"])

    def test_stop_between_runs_preserves_completed_row_and_no_next_launch(self):
        directory, _ = self.prepare("--continue-on-failure")
        calls = []
        def launcher(*args):
            calls.append(args)
            batch.request_stop(directory, "Stop after the completed first child")
            return {"returncode": 0, "timed_out": False, "child_terminated": True}
        report = batch.execute(directory, launcher=launcher)
        self.assertEqual(1, len(calls))
        self.assertEqual("completed", report["runs"][0]["status"])
        self.assertEqual("cancelled", report["status"])
        self.assertEqual(2, report["counts"]["not_started"])

    def test_normal_exit_with_no_timeout_still_reaped(self):
        result = batch.launch([sys.executable, "-c", "print('local only')"], self.root / "normal.log", None)
        self.assertEqual(0, result["returncode"])
        self.assertTrue(result["child_terminated"])
        self.assertFalse(result.get("cancelled", False))
        self.assertFalse(result["timed_out"])

    def test_role_override_defaults_and_explicit_value_are_forwarded_once(self):
        directory, plan = self.prepare()
        for run in plan["runs"]:
            command = run["command"]
            self.assertEqual(1, command.count("--world-review-reasoning-effort"))
            self.assertEqual("medium", command[command.index("--world-review-reasoning-effort") + 1])
            self.assertEqual("low", command[command.index("--reasoning-effort") + 1])
        args = batch.parser().parse_args(["--batch", "offline_low", "--seeds", "sample_a",
                                          "--world-review-reasoning-effort", "low"])
        _, other = batch.prepare(args)
        self.assertTrue(all(r["command"][r["command"].index("--world-review-reasoning-effort") + 1] == "low"
                            for r in other["runs"]))

    def test_first_real_call_error_is_retained_but_json_error_not_mislabeled(self):
        trace = self.root / "llm_attempts.jsonl"
        events = [{"event": "json_error", "step": "world.author", "error": "json only"}]
        trace.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        self.assertIsNone(batch.summarize_run(self.root)["first_call_error"])
        events += [{"event": "call_error", "step": "world.skeptic", "error_type": "TimeoutError",
                    "error": "e" * 1200, "elapsed_ms": 150000},
                   {"event": "call_error", "step": "world.later", "error_type": "Other", "error": "later"}]
        trace.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        summary = batch.summarize_run(self.root)
        self.assertEqual({"step": "world.skeptic", "error_type": "TimeoutError", "error": "e" * 1000,
                          "elapsed_ms": 150000}, summary["first_call_error"])
        self.assertEqual(1, summary["trace"]["json_errors"])
        self.assertEqual(2, summary["trace"]["call_errors"])

    def test_malformed_stop_file_cannot_allow_dispatch(self):
        directory, _ = self.prepare()
        path = directory / "stop_request.json"
        path.write_text("{", encoding="utf-8")
        with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("Must not spawn")):
            result = batch.launch([sys.executable, "-c", "pass"], self.root / "no.log", None, stop_request_path=path)
        self.assertTrue(result["cancelled"])
        self.assertIsNone(result["pid"])
        self.assertIn("read_error", result["stop_request"])

    def test_summary_counts_inflight_timeout_and_reported_truncation_usage_separately(self):
        events = [{"event": "request", "call_id": ident} for ident in ("truncated", "timed_out", "inflight", "missing_usage")]
        events += [{"event": "response", "call_id": "truncated", "response": {"usage": {
            "prompt_tokens": 10, "completion_tokens": 32768, "total_tokens": 32778}}},
            {"event": "call_error", "call_id": "truncated", "kind": "output_truncated"},
            {"event": "call_error", "call_id": "timed_out", "kind": "total_deadline"},
            {"event": "response", "call_id": "missing_usage", "response": {"usage": None}}]
        (self.root / "llm_attempts.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        batch.write(self.root / "experiment_profile.json", {"reserved_upper_estimate_cny": 8,
            "reported_usage_estimate_cny": 2, "budget_consumed_cny": 5, "unsettled_reservation_calls": 2})
        summary = batch.summarize_run(self.root)
        self.assertEqual(summary["trace"]["requests_without_response"], 2)
        self.assertEqual(summary["trace"]["usage_missing_responses"], 1)
        self.assertEqual(summary["trace"]["calls_without_valid_usage"], 3)
        self.assertEqual(summary["budget_unsettled_estimate_cny"], 3)
        self.assertTrue(summary["trace"]["call_id_accounting_complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
