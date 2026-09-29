"""Offline contract tests for agent-directed original-world generation."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.world_agent import generate_world
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_state import assemble_world
from seed_world_selftest import fixture


def write(unit_id, intent="生成该业务过程", reads=()):
    return {"action": "write", "unit_id": unit_id, "intent": intent,
            "read_entities": list(reads), "read_events": [], "plan": "完成来源与复核"}


def table_part(table, entity_indices, structural=False):
    return {"entities": [deepcopy(table["entities"][index]) for index in entity_indices],
            "relations": deepcopy(table["relations"]) if structural else [],
            "events": deepcopy(table["events"]) if structural else [], "initial_states": []}


class ScriptedTracer:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat_json(self, step, messages, **parameters):
        self.calls.append({"step": step, "input": json.loads(messages[1]["content"]), "parameters": parameters})
        if not self.script:
            raise AssertionError("unexpected call " + step)
        expected, response = self.script.pop(0)
        if expected != step:
            raise AssertionError(f"expected {expected}, got {step}")
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


PLAN, WRITE = "world.agent.plan", "world.agent.write"
FINISH = {"action": "finish", "reason": "实际来源、采用变化与复核已经落实"}


class WorldAgentTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("No network/provider allowed"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, self.world, self.table = fixture()
        self.full = table_part(self.table, [0, 1], True)
        self.company = table_part(self.table, [0])
        self.report = table_part(self.table, [1], True)

    def generate(self, script, **kwargs):
        tracer = ScriptedTracer(script)
        table, metadata = generate_world(self.wp, tracer, log=lambda *_: None, **kwargs)
        self.assertFalse(tracer.script)
        for call in tracer.calls:
            self.assertEqual(call["parameters"]["retries"], 3)
            self.assertEqual(call["parameters"]["response_format"], {"type": "json_object"})
            self.assertEqual(call["parameters"]["model"], config.STRUCTURE_MODEL)
        return table, metadata, tracer

    def test_complete_unit_compiles_original_world(self):
        table, metadata, _ = self.generate([(PLAN, write("one")), (WRITE, self.full), (PLAN, FINISH)])
        world, issues = assemble_world(table, blueprint=self.wp["world_blueprint"], include_shape_diagnostics=False)
        self.assertFalse(issues)
        self.assertEqual(world.to_dict(), self.world.to_dict())
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(len(metadata["units"]), 1)

    def test_inspect_and_business_units_have_actual_shared_context(self):
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, {"action": "inspect", "read_entities": ["Aster"]}),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)])
        plans = [call["input"] for call in tracer.calls if call["step"] == PLAN]
        self.assertEqual(plans[2]["observation"]["inspection"]["entities"][0]["canonical_fields"]["sector"][0]["value"], "insurance")
        second_writer = [call for call in tracer.calls if call["step"] == WRITE][1]
        self.assertEqual(second_writer["input"]["read_context"]["entities"][0]["name"], "Aster")
        self.assertEqual(len(metadata["units"]), 2)
        self.assertNotIn("SECRET", json.dumps(second_writer["input"]))

    def test_bad_unit_atomic_rejection_and_planner_adapts_reads(self):
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, write("bad_report")), (WRITE, self.report),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)])
        adapted = [call for call in tracer.calls if call["step"] == PLAN][2]["input"]
        self.assertEqual(adapted["observation"]["rejected_unit"], "bad_report")
        self.assertEqual(len(adapted["current"]["entities"]), 1)
        self.assertEqual([unit["unit_id"] for unit in metadata["units"]], ["company", "report"])

    def test_finish_cannot_turn_incomplete_prefix_into_complete_world(self):
        _, _, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company), (PLAN, FINISH),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)])
        fourth = [call for call in tracer.calls if call["step"] == PLAN][2]
        self.assertIn("finish_rejected", fourth["input"]["observation"])

    def test_revision_rolls_back_dependents_and_can_merge_work(self):
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report),
            (PLAN, {"action": "revise", "unit_id": "company", "reason": "共同重新核对公司与报告归属"}),
            (PLAN, write("merged")), (WRITE, self.full), (PLAN, FINISH)])
        after_rollback = [call for call in tracer.calls if call["step"] == PLAN][3]
        self.assertEqual(after_rollback["input"]["current"]["entities"], [])
        self.assertEqual([unit["unit_id"] for unit in metadata["retired_units"]], ["company", "report"])
        self.assertEqual([unit["unit_id"] for unit in metadata["units"]], ["merged"])

    def test_author_can_request_context_without_committing(self):
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, write("report")), (WRITE, {"needs_context": "需要完整公司事实"}),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)])
        plans = [call for call in tracer.calls if call["step"] == PLAN]
        self.assertIn("author_needs_context", plans[2]["input"]["observation"])
        self.assertEqual(len(metadata["units"]), 2)

    def test_error_checkpoints_and_resume_reuses_accepted_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            tracer = ScriptedTracer([(PLAN, write("company")), (WRITE, self.company), (PLAN, TimeoutError("offline timeout"))])
            with self.assertRaises(TimeoutError):
                generate_world(self.wp, tracer, checkpoint_path=path, log=lambda *_: None)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "execution_error")
            self.assertEqual(len(saved["units"]), 1)
            _, metadata, resumed = self.generate([
                (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)], checkpoint_path=path)
            self.assertEqual(len(metadata["units"]), 2)
            self.assertEqual(sum(call["step"] == WRITE for call in resumed.calls), 1)

    def test_memory_resume_and_feedback_can_dispute_without_forced_rewrite(self):
        table, state, _ = self.generate([(PLAN, write("one")), (WRITE, self.full), (PLAN, FINISH)])
        response = {**FINISH, "issue_responses": [{"issue_id": "review-1", "disposition": "disputed",
                     "response": "既有e1与e2明确具有同一报告参与者且相差一期。"}]}
        new_table, new_state, tracer = self.generate([(PLAN, response)], resume_state=state,
                     feedback={"issue_id": "review-1", "description": "缺少复核因果"})
        self.assertEqual(table, new_table)
        self.assertEqual(new_state["issue_responses"][0]["disposition"], "disputed")
        self.assertEqual(len(tracer.calls), 1)

    def test_checkpoint_tampering_and_changed_inputs_rejected(self):
        _, state, _ = self.generate([(PLAN, write("one")), (WRITE, self.full), (PLAN, FINISH)])
        state["units"][0]["intent"] = "tampered"
        with self.assertRaisesRegex(WorldBlueprintError, "binding mismatch"):
            generate_world(self.wp, ScriptedTracer([]), resume_state=state)
        _, state, _ = self.generate([(PLAN, write("one")), (WRITE, self.full), (PLAN, FINISH)])
        altered = deepcopy(self.wp)
        altered["scenario"] = "different task"
        with self.assertRaisesRegex(WorldBlueprintError, "binding mismatch"):
            generate_world(altered, ScriptedTracer([]), resume_state=state)

    def test_initial_states_applied_once_and_in_metadata(self):
        raw = deepcopy(self.full)
        raw["initial_states"] = [{"entity": "Yearbook", "field": "profit", "session": 0, "value": "100"}]
        raw["events"][0]["session"] = 1
        table, metadata, _ = self.generate([(PLAN, write("one")), (WRITE, raw), (PLAN, FINISH)])
        report = next(entity for entity in table["entities"] if entity["name"] == "Yearbook")
        self.assertEqual(report["fields"]["profit"], {"type": "stable", "value": "100"})
        self.assertEqual(metadata["initial_states"], raw["initial_states"])

    def test_received_truncation_requires_changed_work_before_retry(self):
        failure = {"__error__": "truncated", "__error_metadata__": {
            "kind": "output_truncated", "response_received": True, "token_cap": 16384}}
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, write("report", reads=["Aster"])), (WRITE, failure),
            (PLAN, write("renamed_report", reads=["Aster"])),
            (PLAN, write("report", intent="只生成本公司这一份报告的修订复核过程", reads=["Aster"])),
            (WRITE, self.report), (PLAN, FINISH)])
        plans = [call for call in tracer.calls if call["step"] == PLAN]
        self.assertIn("output_truncated", plans[2]["input"]["observation"])
        self.assertIn("identical truncated work", plans[3]["input"]["observation"]["action_rejected"])
        self.assertEqual(len(metadata["truncated_work"]), 1)
        self.assertEqual(sum(call["step"] == WRITE for call in tracer.calls), 3)

    def test_timeout_sentinel_cannot_be_treated_as_replan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            tracer = ScriptedTracer([(PLAN, write("one")), (WRITE, {
                "__error__": "deadline", "__error_metadata__": {"kind": "total_deadline", "response_received": False}})])
            with self.assertRaisesRegex(RuntimeError, "execution error"):
                generate_world(self.wp, tracer, checkpoint_path=path, log=lambda *_: None)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["units"], [])
            self.assertEqual(saved["status"], "execution_error")

    def test_budget_is_finite_even_if_planner_only_inspects(self):
        self.wp["world_generation"] = {"max_steps": 2}
        tracer = ScriptedTracer([(PLAN, {"action": "inspect"}), (PLAN, {"action": "inspect"})])
        with self.assertRaisesRegex(WorldBlueprintError, "step budget exhausted"):
            generate_world(self.wp, tracer, log=lambda *_: None)
        self.assertEqual(len(tracer.calls), 2)

    def test_existing_world_can_add_structure_without_recreating_entities(self):
        baseline = deepcopy(self.world.to_dict())
        extra = {"entities": [], "relations": [], "initial_states": [], "events": [{
            "id": "e3", "type": "revision", "session": 4, "participants": {"report": "Yearbook"},
            "effects": [{"entity": "Yearbook", "field": "profit", "set": "160"}]}]}
        table, _, _ = self.generate([
            (PLAN, write("next_revision", reads=["Yearbook"])), (WRITE, extra), (PLAN, FINISH)], existing=self.world)
        result, issues = assemble_world(table, blueprint=self.wp["world_blueprint"], existing=self.world,
                                        include_shape_diagnostics=False)
        self.assertFalse(issues)
        self.assertEqual(len(result.events), len(self.world.events) + 1)
        self.assertEqual(self.world.to_dict(), baseline)
        self.assertEqual(result.events[:-1], self.world.events)

    def test_unresolved_action_stops_without_claiming_completion(self):
        tracer = ScriptedTracer([(PLAN, {"action": "report_unresolved", "reason": "业务依赖需要澄清"})])
        with self.assertRaisesRegex(WorldBlueprintError, "unresolved"):
            generate_world(self.wp, tracer, log=lambda *_: None)

    def test_recreating_accepted_entity_is_not_silently_merged(self):
        changed_company = deepcopy(self.company)
        changed_company["entities"][0]["fields"]["sector"]["value"] = "changed"
        _, metadata, tracer = self.generate([
            (PLAN, write("company")), (WRITE, self.company),
            (PLAN, write("overwrite", reads=["Aster"])), (WRITE, changed_company),
            (PLAN, write("report", reads=["Aster"])), (WRITE, self.report), (PLAN, FINISH)])
        observation = [call for call in tracer.calls if call["step"] == PLAN][2]["input"]["observation"]
        self.assertTrue(any("cannot recreate" in issue for issue in observation["issues"]))
        self.assertEqual(metadata["units"][0]["raw"], self.company)


if __name__ == "__main__":
    unittest.main(verbosity=2)
