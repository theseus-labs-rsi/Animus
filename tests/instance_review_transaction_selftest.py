"""Original review targets use the existing value owners and durable allowance."""
import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-disabled")
os.environ.setdefault("MODEL", "offline-disabled")
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden during import")):
    import config
    from instance_plan_selftest import setup_plan, authored_values, ScriptedTracer
    from pipeline import instance_plan, instance_executor, world_agent, world_gen
    from pipeline.supply_capacity import CapacityReviewRequested
    from pipeline.world_blueprint import WorldBlueprintError


class ReviewTransactionTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Provider/network forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        self.wp, self.plan, _table, self.full, _action = setup_plan()
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "world.json"

    def build(self, separate=True, maximum=None):
        if separate:
            company = self.plan["units"][0]["objects"].pop(0)
            self.plan["units"][0]["depends_on"] = ["foundation"]
            self.plan["units"].insert(0, {"unit_id":"foundation", "business_purpose":"Original company", "objects":[company]})
        wp = instance_plan.compile_plan(self.wp, self.plan)
        if maximum is not None:
            wp.setdefault("world_generation", {})["max_steps"] = maximum
        values = {u["unit_id"]:authored_values(wp, u, self.full) for u in wp["business_instance_plan"]["units"]}
        trace = ScriptedTracer([("world.instance.values", values[u["unit_id"]]) for u in wp["business_instance_plan"]["units"]])
        table, state = world_agent.generate_world(wp, trace, checkpoint_path=self.path, log=lambda *_:None)
        return wp, values, table, state

    def feedback(self, entity="Aster", field="sector", identity="review-1", structure=False):
        return {"status":"failed", "issues":[{"id":identity, "finding":"Reconsider the original value against its business evidence"}],
            "repair_targets":{"intrinsic":[{"entity":entity, "fields":[field]}], "structure":structure}}

    def response(self, raw, disposition="addressed", identity="review-1"):
        raw = deepcopy(raw)
        raw["issue_responses"] = [{"issue_id":identity, "disposition":disposition,
            "response":"The original business evidence supports this value and its history."}]
        return raw

    def revise(self, wp, trace, feedback=None):
        return world_agent.generate_world(wp, trace, checkpoint_path=self.path, feedback=feedback, log=lambda *_:None)

    def test_only_selected_owner_changes_and_other_units_keep_original_bytes(self):
        wp, values, table, original = self.build()
        revised = self.response(values["foundation"])
        revised["objects"]["company-slot"]["fields"]["sector"]["value"] = "banking"
        feedback = self.feedback(); trace = ScriptedTracer([("world.instance.values", revised)])
        changed, state = self.revise(wp, trace, feedback)
        self.assertEqual(len(trace.calls), 1)
        self.assertEqual(trace.calls[0]["input"]["task"]["unit_id"], "foundation")
        self.assertEqual(trace.calls[0]["input"]["local_repair"]["previous_response"], values["foundation"])
        self.assertEqual(trace.calls[0]["input"]["feedback"], feedback)
        self.assertEqual(state["units"][1], original["units"][1])
        self.assertEqual(state["retired_units"], [original["units"][0]])
        self.assertEqual(state["local_repairs"][0]["status"], "completed")
        self.assertEqual(state["steps"], original["steps"] + 1)
        self.assertEqual(changed["events"], table["events"])
        self.assertEqual(changed["relations"], table["relations"])
        self.assertEqual(state["issue_responses"], revised["issue_responses"])
        again, resumed = self.revise(wp, ScriptedTracer([]), feedback)
        self.assertEqual(again, changed); self.assertEqual(resumed, state)

    def test_multiple_intrinsic_owners_share_existing_transaction_contract(self):
        wp, values, _table, original = self.build()
        feedback = self.feedback()
        feedback["repair_targets"]["intrinsic"].append({"entity":"Yearbook", "fields":["capital"]})
        trace = ScriptedTracer([("world.instance.values", self.response(values[uid], "disputed"))
            for uid in ("foundation", "report-case")])
        _table, state = self.revise(wp, trace, feedback)
        self.assertEqual([x["input"]["task"]["unit_id"] for x in trace.calls], ["foundation", "report-case"])
        self.assertEqual(state["steps"], original["steps"] + 2)
        self.assertEqual(len(state["local_repairs"]), 2)
        self.assertEqual(len(state["issue_responses"]), 2)

    def test_unknown_structure_owned_and_duplicate_targets_buy_no_calls(self):
        wp, _values, _table, original = self.build()
        bad = [self.feedback("Unknown"), self.feedback("Yearbook", "profit"), {"issues":[{"id":"unscoped"}]}]
        duplicate = self.feedback(); duplicate["repair_targets"]["intrinsic"] *= 2; bad.append(duplicate)
        for feedback in bad:
            with self.subTest(feedback=feedback):
                trace = ScriptedTracer([])
                with self.assertRaises(WorldBlueprintError): self.revise(wp, trace, feedback)
                self.assertEqual(trace.calls, [])
                self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), original)

    def test_structural_review_returns_original_layout_request_without_retiring_values(self):
        wp, _values, _table, original = self.build()
        feedback = self.feedback(structure=True); trace = ScriptedTracer([])
        with self.assertRaises(CapacityReviewRequested) as error: self.revise(wp, trace, feedback)
        self.assertEqual(error.exception.evidence["findings"][0]["evidence"], feedback)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["units"], original["units"])
        self.assertEqual(saved["steps"], original["steps"])
        self.assertEqual(saved["retired_units"], [])
        self.assertEqual(saved["local_repairs"], [])
        self.assertEqual(trace.calls, [])

    def test_unrelated_field_or_initial_value_deletion_returns_to_same_owner(self):
        self.full["initial_states"] = [{"entity":"Yearbook", "field":"review", "value":"unreviewed"}]
        wp, values, _table, _original = self.build()
        bad = self.response(values["report-case"])
        bad["objects"]["report-slot"].pop("initial_values")
        good = self.response(values["report-case"])
        trace = ScriptedTracer([("world.instance.values", bad), ("world.instance.values", good)])
        _table, state = self.revise(wp, trace, self.feedback("Yearbook", "capital"))
        self.assertIn("Local correction changed unrelated values", trace.calls[1]["input"]["correction"]["compiler_issues"][0])
        self.assertEqual(state["local_repairs"][0]["findings"][0]["carrier"], {"entities":["Yearbook"], "field":"capital"})
        self.assertEqual(len(state["local_repairs"]), 1)

    def test_disputed_or_deferred_opinion_is_preserved_for_independent_review(self):
        wp, values, table, _original = self.build()
        for n, disposition in enumerate(("disputed", "deferred")):
            identity = "review-" + str(n + 1)
            raw = self.response(values["foundation"], disposition, identity)
            changed, state = self.revise(wp, ScriptedTracer([("world.instance.values", raw)]), self.feedback(identity=identity))
            self.assertEqual(changed, table)
            self.assertEqual(state["issue_responses"][-1]["disposition"], disposition)
            self.assertNotIn("world_semantic_review", state["fulfillment"])

    def test_review_cannot_finish_without_original_author_response(self):
        wp, values, _table, _original = self.build()
        good = self.response(values["foundation"])
        trace = ScriptedTracer([("world.instance.values", values["foundation"]), ("world.instance.values", good)])
        _table, state = self.revise(wp, trace, self.feedback())
        self.assertIn("explicit issue_responses", trace.calls[1]["input"]["correction"]["compiler_issues"][0])
        self.assertEqual(len(state["local_repairs"]), 1)

    def test_returned_local_reply_resumes_without_extra_call_or_transaction(self):
        wp, values, _table, original = self.build(separate=False)
        real_save = world_agent._save; interrupted = False
        def stop_after_return(path, state):
            nonlocal interrupted
            real_save(path, state)
            if (not interrupted and state["local_repairs"] and state["log"][-1]["status"] == "ready"
                    and state["log"][-1]["input"].get("local_repair")):
                interrupted = True
                raise RuntimeError("Simulated process interruption after saved validation")
        trace = ScriptedTracer([("world.instance.values", self.response(values["report-case"]))])
        with patch.object(world_agent, "_save", stop_after_return):
            with self.assertRaises(RuntimeError): self.revise(wp, trace, self.feedback())
        resume_trace = ScriptedTracer([])
        _table, state = self.revise(wp, resume_trace)
        self.assertEqual(resume_trace.calls, [])
        self.assertEqual(len(state["local_repairs"]), 1)
        self.assertEqual(state["steps"], original["steps"] + 1)
        self.assertEqual(len(state["retired_units"]), 1)

    def test_unreturned_reply_consumes_original_slot_and_resumes_remaining_slots(self):
        wp, values, _table, original = self.build(separate=False)
        with self.assertRaises(RuntimeError):
            self.revise(wp, ScriptedTracer([("world.instance.values", RuntimeError("Lost transport reply"))]), self.feedback())
        lost = json.loads(self.path.read_text(encoding="utf-8"))["log"][-1]
        trace = ScriptedTracer([("world.instance.values", self.response(values["report-case"]))])
        _table, state = self.revise(wp, trace)
        self.assertEqual(state["log"][-2], lost)
        self.assertEqual(state["log"][-1]["attempt"], 2)
        self.assertEqual(state["steps"], original["steps"] + 2)
        self.assertEqual(len(state["local_repairs"]), 1)

    def test_three_lost_slots_or_global_step_limit_cannot_be_reset_by_resume(self):
        wp, _values, _table, original = self.build(separate=False)
        for n in range(3):
            with self.assertRaises(RuntimeError):
                self.revise(wp, ScriptedTracer([("world.instance.values", RuntimeError("Lost reply"))]), self.feedback() if n == 0 else None)
        trace = ScriptedTracer([])
        with self.assertRaisesRegex(WorldBlueprintError, "content remains unresolved"): self.revise(wp, trace)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["steps"], original["steps"] + 3)
        self.assertEqual(len(saved["local_repairs"]), 1)
        self.assertEqual(trace.calls, [])
        self.plan = setup_plan()[1]
        self.path = Path(self.temp.name) / "short.json"
        wp, _values, _table, _state = self.build(separate=False, maximum=2)
        with self.assertRaises(RuntimeError): self.revise(wp, ScriptedTracer([("world.instance.values", RuntimeError("Lost reply"))]), self.feedback())
        trace = ScriptedTracer([])
        with self.assertRaisesRegex(WorldBlueprintError, "call budget exhausted"): self.revise(wp, trace)
        self.assertEqual(trace.calls, [])

    def test_original_three_local_transactions_are_not_increased(self):
        wp, values, _table, _original = self.build(separate=False)
        for n in range(3):
            identity = "review-" + str(n + 1)
            self.revise(wp, ScriptedTracer([("world.instance.values", self.response(values["report-case"], "disputed", identity))]), self.feedback(identity=identity))
        trace = ScriptedTracer([])
        with self.assertRaisesRegex(WorldBlueprintError, "allowance exhausted"):
            self.revise(wp, trace, self.feedback(identity="review-4"))
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["local_repairs"]), 3)
        self.assertEqual(saved["steps"], 4)
        self.assertEqual(trace.calls, [])

    def test_original_factory_call_ceiling_stays_bound_across_resume(self):
        wp, values, _table, original = self.build(separate=False)
        trace = ScriptedTracer([("world.instance.values", values["report-case"])])
        with self.assertRaisesRegex(WorldBlueprintError, "review correction call allowance exhausted"):
            world_agent.generate_world(wp, trace, checkpoint_path=self.path, feedback=self.feedback(),
                repair_max_calls=1, log=lambda *_:None)
        self.assertEqual(len(trace.calls), 1)
        empty = ScriptedTracer([])
        with self.assertRaisesRegex(WorldBlueprintError, "call allowance cannot change"):
            world_agent.generate_world(wp, empty, checkpoint_path=self.path, repair_max_calls=2, log=lambda *_:None)
        with self.assertRaisesRegex(WorldBlueprintError, "review correction call allowance exhausted"):
            self.revise(wp, empty)
        self.assertEqual(empty.calls, [])
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["steps"], original["steps"] + 1)

    def test_legacy_interrupted_receipt_is_preserved_and_still_uses_allowance(self):
        wp, values, _table, _original = self.build(separate=False)
        for n in range(2):
            identity = "review-" + str(n + 1)
            _table, state = self.revise(wp, ScriptedTracer([("world.instance.values",
                self.response(values["report-case"], "disputed", identity))]), self.feedback(identity=identity))
        # Legacy local receipts had no durable protocol or frozen request. An
        # earlier interrupted entry could coexist with a later accepted one.
        for receipt in state["local_repairs"]:
            for key in ("version", "dependency_world", "feedback", "review_hash"):
                receipt.pop(key)
        state["local_repairs"][0]["status"] = "running"
        legacy = deepcopy(state["local_repairs"])
        world_agent._save(self.path, state)
        _table, resumed = self.revise(wp, ScriptedTracer([]))
        self.assertEqual(resumed["local_repairs"], legacy)
        _table, repaired = self.revise(wp, ScriptedTracer([("world.instance.values",
            self.response(values["report-case"], "disputed", "review-3"))]), self.feedback(identity="review-3"))
        self.assertEqual(repaired["local_repairs"][:2], legacy)
        self.assertEqual(len(repaired["local_repairs"]), 3)
        trace = ScriptedTracer([])
        with self.assertRaisesRegex(WorldBlueprintError, "allowance exhausted"):
            self.revise(wp, trace, self.feedback(identity="review-4"))
        self.assertEqual(trace.calls, [])


if __name__ == "__main__": unittest.main()
