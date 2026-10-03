"""Offline explicit initial-plan recovery and adversarial evidence tests."""
import os
import sys
import socket
import types
os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-dummy", MODEL="offline-model",
                  OPENAI_BASE_URL="http://offline.invalid")
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
import json
import tempfile
import unittest
from unittest.mock import patch
from pipeline import initial_plan_recovery as recovery
from pipeline import instance_plan as ip
from pipeline.plan_transactions import derived_dependencies
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from instance_plan_selftest import setup_plan, ACCEPT


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="initial-plan-evidence-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / recovery.RUN_ID
        self.directory.mkdir()
        self.wp, proposal, *_ = setup_plan()
        next(t for t in self.wp["world_blueprint"]["event_types"] if t["id"] == "reviewed")["min_count"] = 2
        candidate = derived_dependencies(proposal)
        prepared = recovery._prepared_whitepaper(self.wp)
        try: ip.compile_plan(prepared, candidate)
        except ip.PlanConflict as error: findings = error.findings
        self.assertEqual([f["code"] for f in findings], ["missing_declared_instances"])
        self.audit = {"status": "mechanical_recovery_required", "before_hash": recovery._digest(prepared),
                      "attempts": [{"attempt": 1, "proposal": proposal, "findings": findings}],
                      "transactions": [{"mechanical_attempt": 1, "unit_proposals": {}, "findings": findings,
                          "drafts": [{"iteration": n, "candidate": deepcopy(candidate), "unit_proposals": {}, "findings": findings}
                                     for n in (1, 2, 3)]}], "findings": findings}
        self.manifest = {"run_id": recovery.RUN_ID, "status": "failed", "config": {"seed": "immutable"}, "stages": {"whitepaper": {"done": True}}}
        self.state = {"status": "plan_recovery_required", "round": 1, "supply_round": 1,
                      "layout_revisions": 0, "history": [], "identity": "fixture-identity", "findings": findings}
        self.profile = {"execution_status": "failed", "error_type": "PlanConflict", "stopped": False,
                        "source_drift": [], "admitted_calls": 11, "settled_usage_calls": 11,
                        "unsettled_reservation_calls": 0, "max_calls": 5000, "max_cny": 100,
                        "model": "glm-5.3-flash", "budget_consumed_cny": 0.13562878,
                        "reported_usage_estimate_cny": 0.13562878, "reserved_upper_estimate_cny": 0,
                        "input_cny_per_m": 0.632, "output_cny_per_m": 2.212,
                        "transport": {}, "max_output_tokens": 32768, "json_attempts": 5}
        self.previous = {"pipeline/plan_transactions.py": "a" * 64, "pipeline/production.py": "b" * 64,
                         "pipeline/instance_plan.py": "c" * 64, "tools/run_original_bc_smoke.py": "d" * 64,
                         "config.py": "8" * 64}
        self.current = dict(self.previous, **{"pipeline/plan_transactions.py": "e" * 64,
                                             "pipeline/initial_plan_recovery.py": "f" * 64})
        source_guard = patch.object(recovery, "_implementation_hashes", side_effect=lambda: deepcopy(self.current))
        source_guard.start()
        self.addCleanup(source_guard.stop)
        self.trace = []
        messages = [{"role": "user", "content": "fixture plan request"}]
        for n in range(11):
            step = "council.instance_plan" if n == 10 else "fixture.context"
            self.trace += [{"event": "request", "step": step, "call_id": str(n), "operation_id": str(n),
                            "model": "glm-5.3-flash", "messages": messages},
                           {"event": "response", "step": step, "call_id": str(n), "operation_id": str(n),
                            "response": {"usage": {"prompt_tokens": 122346 if n == 10 else 0,
                                                   "completion_tokens": 26359 if n == 10 else 0},
                                         "choices": [{"content": json.dumps(proposal if n == 10 else {}), "finish_reason": "stop"}]}},
                           {"event": "json_result", "step": step, "operation_id": str(n), "parsed": proposal if n == 10 else {}}]
        self.prompts = [{"step": "council.instance_plan", "ok": True, "output": proposal, "messages": messages}]
        for name, value in [("manifest.json", self.manifest), ("01_whitepaper.json", self.wp),
                            ("01_instance_plan_audit.json", self.audit), ("11_production.json", self.state),
                            ("experiment_profile.json", self.profile), ("experiment_source_hashes.json", self.previous),
                            ("experiment_source_hashes_at_end.json", self.previous)]: self.write(name, value)
        self.write_rows("llm_attempts.jsonl", self.trace)
        self.write_rows("prompts.jsonl", self.prompts)
        self.receipt_path = self.directory / "explicit_request.json"

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_rows(self, name, rows):
        (self.directory / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def permit(self):
        self.write(self.receipt_path.name, recovery.build_receipt(self.directory, self.current))
        return recovery.validate_receipt(self.receipt_path, self.directory, self.current)

    def test_bound_original_receipt_is_read_only(self):
        before = {name: (self.directory / name).read_bytes() for name in recovery.FILES}
        permit = self.permit()
        self.assertEqual(permit["prior_physical_calls"], 11)
        self.assertEqual(permit["used_unit_author_attempts"], 0)
        self.assertEqual(permit["max_cny"], 100)
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in recovery.FILES})

    def test_changed_bound_file_rejected_before_provider(self):
        self.permit()
        self.profile["admitted_calls"] = 10
        self.write("experiment_profile.json", self.profile)
        with self.assertRaises(ValueError): recovery.validate_receipt(self.receipt_path, self.directory, self.current)

    def test_undeclared_source_change_rejected(self):
        self.current["config.py"] = "1" * 64
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_deleted_source_rejected(self):
        del self.current["pipeline/production.py"]
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_incomplete_current_source_map_rejected(self):
        supplied = deepcopy(self.current)
        supplied.pop("pipeline/instance_plan.py")
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, supplied)

    def test_original_source_drift_rejected(self):
        self.write("experiment_source_hashes_at_end.json", self.current)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_duplicate_call_rejected(self):
        self.trace.append(deepcopy(self.trace[0]))
        self.write_rows("llm_attempts.jsonl", self.trace)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_unsettled_usage_rejected(self):
        del self.trace[1]["response"]["usage"]
        self.write_rows("llm_attempts.jsonl", self.trace)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_forged_json_result_rejected(self):
        self.trace[-1]["parsed"] = {"decision": "plan", "units": []}
        self.write_rows("llm_attempts.jsonl", self.trace)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_forged_prompt_output_rejected(self):
        self.prompts[0]["output"] = {}
        self.write_rows("prompts.jsonl", self.prompts)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_prior_real_author_attempt_rejected(self):
        self.audit["transactions"][0]["edit_attempts"] = {"report-case": [{"proposal": {}}]}
        self.write("01_instance_plan_audit.json", self.audit)
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_mutated_permit_rejected(self):
        permit = self.permit()
        permit["max_cny"] = 101
        with self.assertRaises(ValueError): recovery.enter_recovery(permit)

    def make_run(self):
        directory, manifest = self.directory, deepcopy(self.manifest)
        class Tracer:
            def __init__(self): self.calls = []
            def chat_json(self, step, *args, **kwargs):
                self.calls.append(step)
                if step == "council.instance_plan_business_repair":
                    return {"decision": "repair", "reason": "Publish the review one period later",
                        "changes": [{"unit_id": "report-case", "collection": "publications", "operation": "replace",
                                     "selector": {"fact_slot": "review-three", "channel": "复核记录", "session": 3},
                                     "value": {"fact_slot": "review-three", "channel": "复核记录", "session": 4}}]}
                return {"decision": "repair"}
        class Run:
            run_id = recovery.RUN_ID
            dir = directory
            def __init__(self): self.manifest, self.tracer = manifest, Tracer()
            def read(self, name): return json.loads((directory / name).read_text(encoding="utf-8"))
            def write(self, name, value): (directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return Run()

    def simulated_repair(self, wp, candidate, feedback, tracer, audit, checkpoint):
        tracer.chat_json("council.instance_plan_unit_repair", [])
        event = {"slot": "report-case-extra-review", "type": "reviewed", "participants": {"report": "report-slot"},
                 "session": 1, "caused_by": "revision-zero"}
        candidate["units"][0]["events"].append(event)
        audit["edit_attempts"] = {"report-case": [{"attempt": 1, "proposal": {"event": event}, "status": "locally_compiled"}]}
        checkpoint()
        return candidate

    def execute(self, review):
        permit = self.permit()
        # The real wrapper updates lifecycle only after validating the receipt.
        live = deepcopy(self.profile)
        live.update(execution_status="running", resume_history=[{"initial_plan_recovery": permit["receipt_sha256"]}])
        self.write("experiment_profile.json", live)
        run = self.make_run()
        self.last_run = run
        bounded, token = enter_bounded_run(run.run_id), recovery.enter_recovery(permit)
        try:
            review_patch = {"side_effect": review} if isinstance(review, list) else {"return_value": review}
            with patch("pipeline.plan_transactions.repair", side_effect=self.simulated_repair), patch("pipeline.blueprint_feasibility.assess", **review_patch):
                result = recovery.recover_if_permitted(run)
            return run, result
        finally:
            recovery.leave_recovery(token)
            leave_bounded_run(bounded)

    def test_no_permit_does_nothing(self):
        run = self.make_run()
        self.assertFalse(recovery.recover_if_permitted(run))
        self.assertEqual(run.tracer.calls, [])

    def test_bounded_wrapper_required(self):
        token = recovery.enter_recovery(self.permit())
        try:
            with self.assertRaises(ValueError): recovery.recover_if_permitted(self.make_run())
        finally: recovery.leave_recovery(token)
        self.assertFalse((self.directory / recovery.CLAIM).exists())

    def test_source_change_after_validation_rejected_before_claim(self):
        permit = self.permit()
        self.current["pipeline/plan_transactions.py"] = "9" * 64
        bounded, token = enter_bounded_run(recovery.RUN_ID), recovery.enter_recovery(permit)
        try:
            with self.assertRaises(ValueError): recovery.recover_if_permitted(self.make_run())
        finally:
            recovery.leave_recovery(token)
            leave_bounded_run(bounded)
        self.assertFalse((self.directory / recovery.CLAIM).exists())

    def test_accept_installs_plan_keeps_history_and_ledger(self):
        trace_before = (self.directory / "llm_attempts.jsonl").read_bytes()
        run, result = self.execute(ACCEPT)
        self.assertTrue(result)
        saved = run.read("01_instance_plan_audit.json")
        self.assertEqual(saved["attempts"], self.audit["attempts"])
        self.assertEqual(saved["transactions"], self.audit["transactions"])
        self.assertEqual(len(saved["recovery_history"]), 1)
        self.assertEqual(run.read(recovery.ARTIFACT)["status"], "accepted")
        self.assertIn("business_instance_plan", run.read("01_whitepaper.json"))
        self.assertEqual(run.read("experiment_profile.json")["admitted_calls"], 11)
        self.assertEqual(run.read("11_production.json")["status"], "plan_recovery_required")
        self.assertEqual((self.directory / "llm_attempts.jsonl").read_bytes(), trace_before)
        self.assertEqual(run.tracer.calls, ["council.instance_plan_unit_repair"])
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_real_negative_review_is_preserved_and_does_not_install(self):
        before = (self.directory / "01_whitepaper.json").read_bytes()
        review = deepcopy(ACCEPT); review["decision"] = "repair"
        with self.assertRaises(ip.PlanConflict): self.execute([review, review])
        record = json.loads((self.directory / recovery.ARTIFACT).read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "business_repair_required")
        self.assertEqual(record["business_review"], review)
        self.assertEqual((self.directory / "01_whitepaper.json").read_bytes(), before)
        self.assertFalse((self.directory / "01_instance_plan.json").exists())
        self.assertEqual(self.last_run.tracer.calls.count("council.instance_plan_business_repair"), 1)
        self.assertEqual(record["business_revision_remaining"], 0)

    def test_review_repair_uses_single_original_remaining_revision_then_accepts(self):
        review = deepcopy(ACCEPT); review["decision"] = "repair"
        run, result = self.execute([review, ACCEPT])
        self.assertTrue(result)
        record = run.read(recovery.ARTIFACT)
        self.assertEqual(record["logical_plan_attempt"], 3)
        self.assertEqual(record["business_revision_remaining"], 0)
        self.assertEqual(record["business_review"], review)
        self.assertEqual(record["final_business_review"], ACCEPT)
        self.assertEqual(len(record["business_revision"]["business_revisions"]), 1)
        self.assertEqual(run.tracer.calls.count("council.instance_plan_business_repair"), 1)
        self.assertEqual(record["business_call_history"][0]["status"], "returned")

    def test_unresolved_review_uses_no_business_author_slot(self):
        review = deepcopy(ACCEPT); review["decision"] = "unresolved"
        with self.assertRaises(ip.PlanConflict): self.execute(review)
        record = self.last_run.read(recovery.ARTIFACT)
        self.assertEqual(record["status"], "design_unresolved")
        self.assertEqual(record["business_revision_remaining"], 1)
        self.assertNotIn("council.instance_plan_business_repair", self.last_run.tracer.calls)

    def test_missing_first_unit_author_cannot_advance_to_business_review(self):
        with patch.object(self, "simulated_repair", side_effect=lambda wp, candidate, *args, **kwargs: candidate):
            with self.assertRaisesRegex(ValueError, "unit author was never called"):
                self.execute(ACCEPT)
        self.assertEqual(self.last_run.tracer.calls, [])
        self.assertFalse((self.directory / "01_instance_plan.json").exists())
        self.assertEqual(self.last_run.read(recovery.ARTIFACT)["status"], "repair_failed")

    @unittest.skipUnless(os.name == "nt", "Windows extended-path Run identity")
    def test_real_run_extended_path_retains_same_directory_identity(self):
        from pipeline.run import _io_path
        permit = self.permit()
        run = self.make_run()
        run.dir = _io_path(run.dir)
        self.assertNotEqual(Path(run.dir).resolve(), Path(permit["directory"]))
        bounded, token = enter_bounded_run(run.run_id), recovery.enter_recovery(permit)
        try:
            with patch("pipeline.plan_transactions.repair", side_effect=self.simulated_repair), patch("pipeline.blueprint_feasibility.assess", return_value=ACCEPT):
                self.assertTrue(recovery.recover_if_permitted(run))
        finally:
            recovery.leave_recovery(token)
            leave_bounded_run(bounded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
