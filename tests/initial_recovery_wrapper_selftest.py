"""Offline entry-race and final reader integration checks for the paid wrapper."""
import os
import socket
import sys
import types
os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-dummy",
                  OPENAI_BASE_URL="http://offline.invalid", MODEL="offline-model")
def blocked(*args, **kwargs): raise AssertionError("Network/provider forbidden")
socket.socket.connect = blocked
socket.create_connection = blocked
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules["dotenv"] = dotenv
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import openai
openai.OpenAI = openai.AsyncOpenAI = blocked
import config
config.chat = config.chat_json = blocked
from copy import deepcopy
import hashlib
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from tools import run_original_bc_smoke as smoke
from pipeline import run as runmod
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run, require_bounded_run


class WrapperTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="initial-recovery-wrapper-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_id = "showcase_supply_v1_fresh_20260928"
        self.directory = self.root / self.run_id
        self.directory.mkdir()
        self.ledger = {"model": "glm-5.3-flash", "max_calls": 5000, "max_cny": 100,
            "admitted_calls": 11, "settled_usage_calls": 11, "budget_consumed_cny": .13562878,
            "reported_usage_estimate_cny": .13562878, "unsettled_reservation_calls": 0,
            "stopped": False, "source_drift": []}
        self.config = {"delivery_target": {"fixed": "original-target"}, "seed_pack_digest": "original-seed"}
        self.manifest = {"run_id": self.run_id, "scenario": "fixture", "config": deepcopy(self.config),
            "stages": {"quality": {"done": True}}, "status": "completed", "llm_calls": 11}
        self.write("experiment_profile.json", self.ledger)
        self.write("experiment_source_hashes.json", {"config.py": "1" * 64})
        self.write("manifest.json", self.manifest)
        self.write("11_production.json", {"status": "generation_ready_awaiting_selection", "result": {"passed": True}})
        names = ("manifest.json", "experiment_profile.json", "experiment_source_hashes.json", "11_production.json")
        self.before = {name: (self.directory / name).read_bytes() for name in names}
        self.permit = {"run_id": self.run_id, "directory": str(self.directory), "receipt_sha256": "a" * 64,
            "evidence": {name: {"sha256": hashlib.sha256(raw).hexdigest()} for name, raw in self.before.items()}}
        self.source_patch = patch.object(runmod, "RUNS_DIR", self.root)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def finish(self, *, enabled=True, receipt=None, outcome=None, reader=None, target=None):
        receipt = receipt if receipt is not None else {"status": "passed", "current": True, "passed": True,
            "identity": "bound-reader", "released_count": 42, "document_count": 207}
        outcome = outcome if outcome is not None else {"passed": True}
        target = target if target is not None else {"passed": True}
        self.calls = []
        def invoked(run, **kwargs):
            require_bounded_run(self.run_id)
            self.assertEqual(run.run_id, self.run_id)
            self.assertIsInstance(run.tracer, runmod.Tracer)
            self.assertEqual(run.manifest["config"], self.config)
            self.assertEqual(kwargs, {"model": "glm-5.3-flash", "max_tokens": 32768})
            self.calls.append(run)
            if reader is not None: return reader(run)
            return receipt
        with patch("pipeline.full_material_acceptance.run_full_material_acceptance", side_effect=invoked), \
             patch("pipeline.full_material_acceptance.target_checks", return_value=target, create=True), \
             patch("pipeline.full_material_acceptance.snapshot", return_value=receipt), \
             patch("pipeline.production.outcome", return_value=outcome):
            smoke._finish_full_material_acceptance(self.directory, self.ledger, enabled=enabled,
                                                  model="glm-5.3-flash", max_tokens=32768)

    def test_entry_claim_preserves_profile_and_source_then_refuses_duplicate(self):
        smoke._claim_initial_plan_execution(self.directory, self.permit)
        claim = (self.directory / "00_initial_plan_execution_claim.json").read_bytes()
        with self.assertRaises(FileExistsError): smoke._claim_initial_plan_execution(self.directory, self.permit)
        self.assertEqual(claim, (self.directory / "00_initial_plan_execution_claim.json").read_bytes())
        self.assertTrue(all((self.directory / name).read_bytes() == data for name, data in self.before.items()))

    def test_concurrent_entry_claim_has_one_winner_and_zero_ledger_writes(self):
        start = threading.Barrier(2)
        outcomes = []
        def enter():
            start.wait()
            try: smoke._claim_initial_plan_execution(self.directory, self.permit)
            except FileExistsError: outcomes.append("refused")
            else: outcomes.append("claimed")
        threads = [threading.Thread(target=enter) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=3)
        self.assertCountEqual(outcomes, ["claimed", "refused"])
        self.assertTrue(all((self.directory / name).read_bytes() == data for name, data in self.before.items()))

    def test_changed_prior_profile_refused_before_claim(self):
        self.ledger["admitted_calls"] = 10
        self.write("experiment_profile.json", self.ledger)
        with self.assertRaises(ValueError): smoke._claim_initial_plan_execution(self.directory, self.permit)
        self.assertFalse((self.directory / "00_initial_plan_execution_claim.json").exists())

    def test_final_reader_not_requested_is_not_run(self):
        self.finish(enabled=False)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger["full_material_acceptance"]["status"], "not_run")
        self.assertFalse(self.ledger["experiment_target_passed"])

    def test_final_reader_waits_for_generation_readiness(self):
        self.write("11_production.json", {"status": "plan_recovery_required"})
        with patch.object(runmod, "Run", side_effect=AssertionError("Run must not initialize")):
            self.finish()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger["full_material_acceptance"]["reason"], "generation_not_ready")
        self.assertFalse(self.ledger["experiment_target_passed"])

    def test_final_reader_requires_same_active_bounded_context(self):
        before = (self.directory / "manifest.json").read_bytes()
        with self.assertRaises(ValueError): self.finish()
        self.assertEqual(self.calls, [])
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)

    def test_final_reader_real_run_keeps_config_state_and_cumulative_ledger(self):
        token = enter_bounded_run(self.run_id)
        try: self.finish()
        finally: leave_bounded_run(token)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.ledger["experiment_target_passed"])
        self.assertEqual(self.ledger["admitted_calls"], 11)
        self.assertEqual(self.ledger["budget_consumed_cny"], .13562878)
        self.assertEqual(json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))["config"], self.config)
        self.assertEqual((self.directory / "11_production.json").read_bytes(), self.before["11_production.json"])

    def test_passed_reader_without_current_generation_outcome_is_false(self):
        token = enter_bounded_run(self.run_id)
        try: self.finish(outcome={"passed": False})
        finally: leave_bounded_run(token)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.ledger["experiment_target_passed"])

    def test_target_contract_failure_stops_before_paid_reader(self):
        target = {"passed": False, "issues": ["seven-line independent denominator short"]}
        token = enter_bounded_run(self.run_id)
        try: self.finish(target=target)
        finally: leave_bounded_run(token)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger["target_checks"], target)
        self.assertEqual(self.ledger["full_material_acceptance"]["status"], "not_run")
        self.assertFalse(self.ledger["experiment_target_passed"])

    def test_stale_reader_cannot_pass_even_with_generation_outcome(self):
        token = enter_bounded_run(self.run_id)
        try: self.finish(receipt={"status": "stale", "current": False, "passed": True})
        finally: leave_bounded_run(token)
        self.assertFalse(self.ledger["experiment_target_passed"])

    def test_reader_exception_keeps_failure_and_propagates(self):
        def fail(run): raise RuntimeError("fixture physical stop")
        token = enter_bounded_run(self.run_id)
        try:
            with self.assertRaisesRegex(RuntimeError, "fixture physical stop"): self.finish(reader=fail)
        finally: leave_bounded_run(token)
        self.assertFalse(self.ledger["experiment_target_passed"])
        self.assertEqual(self.ledger["full_material_acceptance"]["status"], "execution_failed")


import original_smoke_corpus_review_override_selftest as fixture


class WrapperWireTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def test_actual_wrapper_keeps_reader_physical_call_in_original_ledger(self):
        frozen_config = {"delivery_target": {"fixture": "frozen"}}
        def factory(directory, cfg):
            (directory / "manifest.json").write_text(json.dumps({"run_id": directory.name,
                "scenario": "fixture", "config": frozen_config, "stages": {"quality": {"done": True}},
                "status": "completed", "llm_calls": 0}), encoding="utf-8")
            (directory / "11_production.json").write_text(json.dumps({"status": "generation_ready_awaiting_selection"}), encoding="utf-8")
            self.assertEqual(runmod.Tracer(types.SimpleNamespace(dir=directory)).chat_json("fixture.generation", [], max_tokens=4096), {"ok": True})
        def reader(run, **kwargs):
            require_bounded_run(run.run_id)
            self.assertEqual(kwargs, {"model": "gpt-5.4-mini", "max_tokens": 16384})
            self.assertEqual(run.tracer.chat_json("full_material_acceptance.fixture.blind", [], max_tokens=4096), {"ok": True})
        with patch.object(runmod, "RUNS_DIR", self.directory / "output/runs"), \
             patch("pipeline.full_material_acceptance.run_full_material_acceptance", side_effect=reader) as paid_reader, \
             patch("pipeline.full_material_acceptance.target_checks", return_value={"passed": True}, create=True), \
             patch("pipeline.full_material_acceptance.snapshot", return_value={"status": "passed", "current": True, "passed": True}), \
             patch("pipeline.production.outcome", return_value={"passed": True}):
            result = self.run_tool(factory, ["--full-material-acceptance"])
        self.assertIsNone(result.error)
        self.assertEqual(paid_reader.call_count, 1)
        self.assertEqual(result.profile["admitted_calls"], 2)
        self.assertEqual(len(result.api.calls), 2)
        self.assertEqual(result.profile["max_calls"], 12)
        self.assertEqual(result.profile["execution_status"], "completed")
        self.assertTrue(result.profile["full_material_acceptance_requested"])
        self.assertTrue(result.profile["experiment_target_passed"])
        self.assertEqual(json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))["config"], frozen_config)
        with self.assertRaises(ValueError): require_bounded_run(result.directory.name)

    def test_actual_wrapper_exit_completed_is_not_target_acceptance(self):
        def factory(directory, cfg):
            (directory / "11_production.json").write_text(json.dumps({"status": "round_limit"}), encoding="utf-8")
        with patch("pipeline.full_material_acceptance.run_full_material_acceptance", side_effect=AssertionError("not ready")) as reader:
            result = self.run_tool(factory, ["--full-material-acceptance"])
        self.assertIsNone(result.error)
        self.assertEqual(result.profile["execution_status"], "completed")
        self.assertEqual(result.profile["full_material_acceptance"]["status"], "not_run")
        self.assertFalse(result.profile["experiment_target_passed"])
        self.assertEqual(result.profile["admitted_calls"], 0)
        reader.assert_not_called()


if __name__ == "__main__": unittest.main(verbosity=2)
