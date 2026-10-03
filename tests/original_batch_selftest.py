"""Offline original-batch orchestration tests; all run artifacts are temporary."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("original_batch", ROOT / "tools/run_original_bc_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="original_batch_offline_")
        self.root = Path(self.tmp.name).resolve()
        for folder in ("pipeline", "eval", "seeds", "tools"):
            (self.root / folder).mkdir()
        for name in ("pipeline/a.py", "eval/a.py", "config.py", "llm_transport.py", "llm_trace.py",
                     "tools/run_original_bc_smoke.py", "tools/run_original_bc_batch.py"):
            (self.root / name).write_text("# offline source fixture\n", encoding="utf-8")
        for name in ("sample_a", "sample_b", "sample_c"):
            (self.root / "seeds" / (name + ".json")).write_text(json.dumps({"seed_id": name}), encoding="utf-8")
        self.patch = patch.object(batch, "ROOT", self.root)
        self.patch.start()
        self.network = patch.object(socket, "socket", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def args(self, *extra):
        seeds = [] if "--seeds" in extra else ["--seeds", "sample_a", "sample_b", "sample_c"]
        return batch.parser().parse_args(["--batch", "offline", *seeds, *extra])

    def test_seed_arguments_are_required(self):
        with self.assertRaises(SystemExit) as caught:
            batch.parser().parse_args(["--batch", "offline"])
        self.assertEqual(caught.exception.code, 2)

    def prepare(self, *extra):
        return batch.prepare(self.args(*extra))

    def fake(self, returncode=0, eligible=True, after=None):
        def launcher(command, log, timeout):
            name = command[command.index("--run") + 1]
            directory = self.root / "output/runs" / name
            directory.mkdir(parents=True)
            log.write_text("offline fake child\n", encoding="utf-8")
            batch.write(directory / "manifest.json", {"status": "completed" if returncode == 0 else "failed",
                         "current_stage": "quality", "stages": {"quality": {"done": returncode == 0}}})
            batch.write(directory / "experiment_profile.json", {"execution_status": "completed" if returncode == 0 else "failed",
                        "reserved_upper_estimate_cny": 0.25, "admitted_calls": 1, "source_drift": []})
            batch.write(directory / "07_release.json", {"status": "passed" if eligible else "failed", "eligible": eligible})
            batch.write(directory / "04_questions.json", [{"qid": "q1"}, {"qid": "q2"}])
            batch.write(directory / "06_grounded_questions.json", [{"qid": "q1"}])
            batch.write(directory / "06_grounding_report.json", {"overall": {"n": 2, "grounded": 1},
                        "n_dropped": 0, "n_pending": 1, "execution_complete": True})
            # Temp directory is outside the actual output tree; no paid ledger contamination.
            trace = [{"event": "request"}, {"event": "response", "response": {"usage": {"total_tokens": 12}}}]
            (directory / "llm_attempts.jsonl").write_text("\n".join(json.dumps(e) for e in trace), encoding="utf-8")
            if after:
                after(directory)
            return {"returncode": returncode, "timed_out": False, "child_terminated": True, "elapsed_seconds": 0.01}
        return launcher

    def test_prepare_is_zero_child_and_does_not_import_provider_config(self):
        with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("child forbidden")):
            before = set(sys.modules)
            directory, plan = self.prepare()
        self.assertFalse({"config", "openai"} & (set(sys.modules) - before))
        self.assertEqual(3, len(plan["runs"]))
        self.assertFalse((directory / "execution").exists())
        self.assertEqual(0, batch.read(directory / "prepared.json")["provider_calls"])

    def test_whole_batch_cap_rejected_before_files_or_launch(self):
        with self.assertRaisesRegex(ValueError, "caps exceed"):
            self.prepare("--repeats", "2", "--max-total-cny", "359")
        self.assertFalse((self.root / "output").exists())

    def test_commands_are_explicit_mini_with_all_limits_and_overrides(self):
        _, plan = self.prepare("--repeats", "2", "--max-total-cny", "360", "--question-budget", "4",
            "--process-questions", "--world-semantic-review", "--corpus-review-reasoning-effort", "medium",
            "--corpus-review-max-tokens", "8192")
        self.assertEqual(6, len(plan["runs"]))
        self.assertEqual(6, len({r["run_id"] for r in plan["runs"]}))
        for run in plan["runs"]:
            command = run["command"]
            self.assertEqual(batch.MODEL, command[command.index("--model") + 1])
            self.assertEqual("quality", command[command.index("--to") + 1])
            self.assertIn("--max-calls", command)
            self.assertIn("--max-cny", command)
            self.assertIn("--corpus-review-max-tokens", command)
            self.assertNotIn("--source-run", command)

    def test_relative_seed_and_output_paths_do_not_depend_on_cwd(self):
        prior = Path.cwd()
        try:
            os.chdir(self.root / "tools")
            directory, plan = self.prepare("--seeds", "seeds/sample_b.json", "--output-dir", "batch_result")
        finally:
            os.chdir(prior)
        self.assertEqual(self.root / "batch_result", directory)
        self.assertEqual(str(self.root / "seeds/sample_b.json"), plan["seeds"][0]["path"])

    def test_repetitions_cover_all_seeds_before_next_round(self):
        _, plan = self.prepare("--repeats", "2", "--max-total-cny", "360")
        self.assertEqual([("sample_a", 1), ("sample_b", 1), ("sample_c", 1),
                          ("sample_a", 2), ("sample_b", 2), ("sample_c", 2)],
                         [(r["seed_id"], r["repeat"]) for r in plan["runs"]])

    def test_new_seed_name_resolves_through_same_repository_entry(self):
        batch.write(self.root / "seeds/sample_d.json", {"seed_id": "sample_d_v2"})
        _, plan = self.prepare("--seeds", "sample_d")
        self.assertEqual("sample_d_v2", plan["seeds"][0]["seed_id"])
        self.assertEqual(str(self.root / "seeds/sample_d.json"), plan["seeds"][0]["path"])

    def test_quantity_target_is_forwarded_without_a_question_cap(self):
        _, plan = self.prepare("--min-questions", "50", "--total-only", "--max-world-entities", "36",
            "--time-span-weeks", "12", "--workers", "2", "--settle-reported-usage")
        self.assertIsNone(plan["question_budget"])
        self.assertEqual(50, plan["min_questions"])
        for run in plan["runs"]:
            command = run["command"]
            self.assertNotIn("--question-budget", command)
            self.assertIn("--total-only", command)
            self.assertIn("--settle-reported-usage", command)
            self.assertEqual("36", command[command.index("--max-world-entities") + 1])

    def test_two_workers_overlap_but_never_exceed_the_bound(self):
        directory, _ = self.prepare("--workers", "2", "--continue-on-failure")
        barrier, lock = threading.Barrier(2), threading.Lock()
        active, peak, calls = 0, 0, 0
        fake = self.fake()
        def concurrent(*args):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
                number = calls
            try:
                if number <= 2:
                    barrier.wait(timeout=5)
                return fake(*args)
            finally:
                with lock:
                    active -= 1
        report = batch.execute(directory, launcher=concurrent)
        self.assertEqual("completed", report["status"])
        self.assertEqual(2, peak)
        self.assertEqual(3, calls)
        self.assertEqual(0, active)

    def test_parallel_stop_reaps_inflight_and_leaves_remaining_unstarted(self):
        directory, _ = self.prepare("--workers", "2", "--continue-on-failure")
        barrier = threading.Barrier(2)
        def cancelled(*args):
            leader = barrier.wait(timeout=5)
            if leader == 0:
                batch.request_stop(directory, "operator stop")
            barrier.wait(timeout=5)
            return {"returncode": -1, "timed_out": False, "child_terminated": True,
                    "cancelled": True, "stop_request": batch.read(directory / "stop_request.json")}
        report = batch.execute(directory, launcher=cancelled)
        self.assertEqual("cancelled", report["status"])
        self.assertEqual(2, report["counts"]["cancelled"])
        self.assertEqual(1, report["counts"]["not_started"])

    def test_live_concurrency_increase_preserves_frozen_plan(self):
        directory, _ = self.prepare("--workers", "3", "--initial-workers", "1")
        original_hash = batch.digest(directory / "plan.json")
        barrier = threading.Barrier(3)
        fake = self.fake()
        def controlled(command, log, timeout):
            if "_01_sample_a_" in command[command.index("--run") + 1]:
                batch.request_concurrency(directory, 3, "Expand after first worker starts")
            barrier.wait(timeout=8)
            return fake(command, log, timeout)
        report = batch.execute(directory, launcher=controlled)
        self.assertEqual("completed", report["status"])
        self.assertEqual(3, report["peak_active_runs"])
        self.assertEqual(3, report["concurrency_limit"])
        self.assertEqual(original_hash, batch.digest(directory / "plan.json"))
        self.assertEqual(1, len(report["concurrency_history"]))
        with self.assertRaisesRegex(ValueError, "finished"):
            batch.request_concurrency(directory, 2, "finished cannot restart")

    def test_concurrency_cap_supports_twenty_one_and_rejects_overshoot(self):
        directory, plan = self.prepare("--workers", "21", "--initial-workers", "7")
        self.assertEqual((7, None), batch.concurrency_limit(directory, plan))
        batch.request_concurrency(directory, 21, "authorized maximum")
        self.assertEqual(21, batch.concurrency_limit(directory, plan)[0])
        for count in (0, 22, True):
            with self.assertRaises(ValueError):
                batch.request_concurrency(directory, count, "out of range")
        request = batch.read(directory / "concurrency_request.json")
        request["plan_sha256"] = "changed"
        batch.write(directory / "concurrency_request.json", request)
        with self.assertRaises(ValueError):
            batch.concurrency_limit(directory, plan)


    def test_incomplete_automatic_receipt_does_not_certify_questions(self):
        directory, _ = self.prepare("--seeds", "sample_a")
        report = batch.execute(directory, launcher=self.fake(eligible=True))
        self.assertEqual(1, report["recorded_quality_eligible"])
        self.assertEqual(0, report["current_quality_eligible"])
        self.assertEqual(0, report["quality_eligible_question_count"])
        self.assertEqual("invalid_release_receipt",
            report["runs"][0]["artifacts"]["quality"]["issues"][0]["code"])

    def test_current_valid_receipt_counts_only_its_bound_questions(self):
        directory, _ = self.prepare("--seeds", "sample_a")
        with patch("pipeline.quality.quality_snapshot", return_value={
                "status": "passed", "eligible": True, "issues": []}):
            report = batch.execute(directory, launcher=self.fake(eligible=True))
        self.assertEqual(1, report["current_quality_eligible"])
        self.assertEqual(1, report["quality_eligible_question_count"])

    def test_three_children_and_original_yield_quality_observed(self):
        directory, _ = self.prepare()
        report = batch.execute(directory, launcher=self.fake(eligible=False))
        self.assertEqual("completed", report["status"])
        self.assertEqual(3, report["counts"]["completed"])
        self.assertEqual(0, report["recorded_quality_eligible"])
        for row in report["runs"]:
            self.assertEqual(1, row["artifacts"]["yield"]["pending"])
            self.assertEqual(12, row["artifacts"]["trace"]["reported_tokens"])
            self.assertIs(row["artifacts"]["quality"]["eligible"], False)

    def test_failure_stops_and_preserves_all_planned_slots(self):
        directory, _ = self.prepare()
        report = batch.execute(directory, launcher=self.fake(returncode=1, eligible=False))
        self.assertEqual({"completed": 0, "failed": 1, "not_started": 2}, report["counts"])
        self.assertTrue((directory / "execution/001.log").exists())
        self.assertFalse((directory / "execution/002.log").exists())

    def test_explicit_continue_keeps_failed_runs_and_never_retries_them(self):
        directory, _ = self.prepare("--continue-on-failure")
        report = batch.execute(directory, launcher=self.fake(returncode=1, eligible=False))
        self.assertEqual(3, report["counts"]["failed"])
        self.assertEqual(3, len(list((directory / "execution").glob("*.log"))))

    def test_source_drift_stops_even_when_continue_enabled(self):
        directory, _ = self.prepare("--continue-on-failure")
        report = batch.execute(directory, launcher=self.fake(after=lambda _: (self.root / "pipeline/a.py").write_text("changed")))
        self.assertEqual("failed", report["status"])
        self.assertEqual(["pipeline/a.py"], report["source_drift"])
        self.assertEqual(2, report["counts"]["not_started"])

    def test_frozen_seed_or_plan_mutation_rejected(self):
        directory, plan = self.prepare()
        Path(plan["seeds"][0]["copy"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "Frozen seed"):
            batch.execute(directory, launcher=self.fake())
        self.assertFalse((directory / "execution").exists())

    def test_cannot_restart_or_overwrite_run(self):
        directory, plan = self.prepare()
        batch.execute(directory, launcher=self.fake())
        with self.assertRaises(FileExistsError):
            batch.execute(directory, launcher=self.fake())
        self.assertTrue(Path(plan["runs"][0]["run_dir"]).is_dir())

    def test_spawn_failure_has_report_and_unstarted_denominator(self):
        directory, _ = self.prepare()
        def failed(*_):
            raise OSError("offline spawn failed")
        report = batch.execute(directory, launcher=failed)
        self.assertEqual(1, report["counts"]["failed"])
        self.assertEqual(2, report["counts"]["not_started"])
        self.assertEqual("OSError", report["error"]["type"])

    def test_bad_trace_is_visible_and_not_silently_complete_usage(self):
        directory, _ = self.prepare("--seeds", "sample_a")
        report = batch.execute(directory, launcher=self.fake(after=lambda p: (p / "llm_attempts.jsonl").write_text("{")))
        error = report["runs"][0]["artifacts"]["artifact_read_errors"][0]
        self.assertEqual("llm_attempts.jsonl", error["artifact"])
        self.assertTrue(error["counts_partial"])
        self.assertEqual("failed", report["status"])

    def test_corrupt_cost_field_keeps_failed_report_and_unknown_cost(self):
        directory, _ = self.prepare()
        def corrupt(path):
            profile = batch.read(path / "experiment_profile.json")
            profile["reserved_upper_estimate_cny"] = "broken"
            batch.write(path / "experiment_profile.json", profile)
        report = batch.execute(directory, launcher=self.fake(after=corrupt))
        self.assertEqual("failed", report["status"])
        self.assertEqual(2, report["counts"]["not_started"])
        self.assertEqual(1, len(report["budget"]["missing_reservation_runs"]))
        self.assertEqual("failed", batch.read(directory / "execution/report.json")["status"])

    def test_real_local_child_timeout_is_terminated_and_reaped(self):
        result = batch.launch([sys.executable, "-c", "import time; time.sleep(30)"], self.root / "timeout.log", 0.15)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["child_terminated"])
        self.assertIsNotNone(result["returncode"])
        self.assertLess(result["elapsed_seconds"], 20)


class PublicPoolTests(unittest.TestCase):
    def fixture(self):
        q = {"qid": "q", "question": "第1期负责人是谁？", "capability": "IE", "gt": {"value": "乙"},
             "entity": "合成项目", "evidence_sessions": [0]}
        corpus = {"sessions": [{"session_id": 0, "docs": [{"doc_id": "early", "content": "合成项目：待披露"}]},
                  {"session_id": 2, "docs": [{"doc_id": "later", "content": "合成项目回顾：第1期负责人是乙。"},
                                              {"doc_id": "filler", "content": "无关", "is_filler": True}]}]}
        return q, corpus

    def test_semantic_pool_contains_later_retrospective_document(self):
        from pipeline.grounding_review import candidates_with_evidence
        q, corpus = self.fixture()
        row = candidates_with_evidence([q], corpus)[0]
        self.assertEqual(["early", "later"], row["candidate_evidence_doc_ids"])
        self.assertEqual([0], row["evidence_sessions"])
        self.assertEqual(q["gt"], row["gt"])
        self.assertIsNone(row["evidence_scope"]["necessary_document_count"])

    def test_legacy_pool_still_uses_canonical_sessions(self):
        from pipeline.grounding import gather_evidence
        q, corpus = self.fixture()
        pool = gather_evidence(q, {s["session_id"]: s for s in corpus["sessions"]})
        self.assertEqual(["early"], [d["doc_id"] for d in pool])




class RealDryCliTests(unittest.TestCase):
    def test_real_cli_in_fresh_process_blocks_config_provider_network_and_child(self):
        with tempfile.TemporaryDirectory(prefix="original_batch_cli_offline_") as temporary:
            destination = Path(temporary) / "prepared"
            program = r'''
import importlib.abc, runpy, socket, subprocess, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'config', 'openai'}:
            raise AssertionError('provider/config import forbidden: ' + fullname)
sys.meta_path.insert(0, Block())
def forbidden(*a, **k): raise AssertionError('network or child forbidden')
socket.socket = forbidden
subprocess.Popen = forbidden
script, output, seed = sys.argv[1:]
sys.argv = [script, '--batch', 'offline_cli_test', '--output-dir', output, '--seeds', seed]
runpy.run_path(script, run_name='__main__')
'''
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", program,
                                     str(ROOT / "tools/run_original_bc_batch.py"), str(destination),
                                     str(ROOT / "skills/realfiles-to-seedjson/examples/seed_v2_example.json")],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30, cwd=temporary)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(0, json.loads(result.stdout)["provider_calls"])
            self.assertFalse((destination / "execution").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
