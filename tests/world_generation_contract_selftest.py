"""Offline production-agent replay for shared instance format and local context."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import config
from pipeline import world_agent as agent
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint
from pipeline.world_state import assemble_world
from world_agent_selftest import ScriptedTracer, fixture, table_part, write, PLAN, WRITE, FINISH


class WorldGenerationContractTests(unittest.TestCase):
    def setUp(self):
        for target, name in ((socket.socket, "connect"), (socket, "getaddrinfo"),
                             (config, "chat"), (config, "chat_json")):
            self.enterContext(patch.object(target, name, side_effect=AssertionError("Network forbidden")))
        self.wp, self.world, self.table = fixture()
        self.company = table_part(self.table, [0])
        self.report = table_part(self.table, [1], True)
        self.full = table_part(self.table, [0, 1], True)

    def test_planner_and_writer_share_instance_format_with_full_dependencies(self):
        self.wp.update(active_lines=[{"line": "L1", "weight": 1}],
                       line_mapping={"L1": "跨期更新"}, traps=["不要消除版本差异"],
                       corpus_plan={"filler_topics": ["global background"]},
                       shared_world_spec={"goal": "all-world-only"})
        action = write("report", "为 Aster 生成报告的修订与复核", ["Aster"])
        action["plan"] = "GLOBAL-REMAINDER-NOT-FOR-WRITER"
        action["events"] = [{"id": "PLANNER-MUST-NOT-AUTHOR-FACTS"}]
        class Capture(ScriptedTracer):
            systems = []
            def chat_json(inner, step, messages, **params):
                inner.systems.append((step, messages[0]["content"]))
                return super().chat_json(step, messages, **params)
        tracer = Capture([(PLAN, write("company")), (WRITE, self.company),
                          (PLAN, action), (WRITE, self.report), (PLAN, FINISH)])
        _, state = agent.generate_world(self.wp, tracer, log=lambda *_: None)
        planner = next(c["input"] for c in tracer.calls if c["step"] == PLAN)
        writers = [c["input"] for c in tracer.calls if c["step"] == WRITE]
        self.assertEqual(planner["generation_brief"], {k: self.wp[k] for k in
            ("active_lines", "line_mapping", "traps", "corpus_plan")})
        for local in writers:
            for key in ("business", "blueprint", "calendar", "field_write_routes", "fact_time_contract"):
                self.assertEqual(local[key], planner[key])
            self.assertNotIn("generation_brief", local)
            self.assertNotIn("world_brief", local)
            self.assertNotIn("plan", local["task"])
            self.assertNotIn("events", local["task"])
        self.assertEqual(writers[-1]["read_context"]["entities"][0]["name"], "Aster")
        self.assertEqual(writers[-1]["task"]["intent"], action["intent"])
        for _, system in tracer.systems:
            self.assertIn(agent.INSTANCE_FORMAT, system)
        self.assertEqual(len(state["units"]), 2)

    def test_historical_role_shapes_report_expected_participant_without_converting(self):
        for shape in ("entity", "role", "role_type", "role_name", "mapping"):
            with self.subTest(shape=shape):
                bad = deepcopy(self.full)
                event = bad["events"][0]
                role = next(iter(event["participants"]))
                entity = event["participants"][role]
                effect = event["effects"][0]
                if shape == "mapping":
                    event["effects"] = {role: {effect["field"]: effect["set"]}}
                elif shape == "entity":
                    effect["entity"] = role
                else:
                    effect[shape] = role
                    effect.pop("entity")
                before = deepcopy(bad)
                _, _, _, issues = agent._compiled(bad, self.wp["world_blueprint"], None, False, self.wp)
                message = next(issue for issue in issues if "恰好覆盖 effect_fields" in issue)
                self.assertIn(f"'role': '{role}'", message)
                self.assertIn(f"'entity': '{entity}'", message)
                self.assertIn("participants[expected.role]", message)
                if shape != "mapping":
                    self.assertIn(f"events[{event['id']}].effects[0].entity", message)
                self.assertEqual(bad, before)

    def test_rejected_role_proposal_preserves_accepted_prefix_and_exact_feedback(self):
        bad = deepcopy(self.report)
        event = bad["events"][0]
        role = next(iter(event["participants"]))
        event["effects"][0]["entity"] = role
        wrong = write("bad", "旧建议：effects 使用角色键而非 entity", ["Aster"])
        tracer = ScriptedTracer([(PLAN, write("company")), (WRITE, self.company),
            (PLAN, wrong), (WRITE, bad), (PLAN, write("report", reads=["Aster"])),
            (WRITE, self.report), (PLAN, FINISH)])
        _, state = agent.generate_world(self.wp, tracer, log=lambda *_: None)
        repaired_input = [c["input"] for c in tracer.calls if c["step"] == WRITE][-1]
        self.assertEqual(repaired_input["feedback"]["rejected_proposal"], bad)
        self.assertTrue(any("'entity': 'Yearbook'" in issue for issue in repaired_input["feedback"]["issues"]))
        self.assertIn("废弃冲突的格式建议", agent.AUTHOR_SYSTEM)
        self.assertEqual([u["unit_id"] for u in state["units"]], ["company", "report"])
        self.assertEqual(state["units"][0]["raw"], self.company)
        self.assertEqual(len(state["prior_failures"]), 1)
        self.assertEqual(len(tracer.calls), 7)  # No added author/repair loop.

    def test_business_value_errors_still_fail_after_valid_instance_serialization(self):
        bad = deepcopy(self.full)
        bad["events"][0]["effects"][0]["set"] = "unlisted-business-value"
        _, _, _, issues = agent._compiled(bad, self.wp["world_blueprint"], None, True, self.wp)
        self.assertTrue(issues)
        self.assertFalse(any("恰好覆盖 effect_fields" in issue for issue in issues))
        compiled, issues = assemble_world(self.table, blueprint=self.wp["world_blueprint"],
                                         include_shape_diagnostics=False)
        self.assertFalse(issues)
        self.assertEqual(compiled.to_dict(), self.world.to_dict())

    def test_actual_input_snapshot_survives_failure_and_replays_exact_binding(self):
        actual = deepcopy(self.wp)
        actual["world_generation"] = {"max_steps": 32}
        actual["world_blueprint"]["temporal_model"]["n_sessions"] = 6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            tracer = ScriptedTracer([(PLAN, write("company")), (WRITE, self.company),
                                     (PLAN, TimeoutError("offline failure"))])
            with self.assertRaises(TimeoutError):
                agent.generate_world(actual, tracer, checkpoint_path=path, log=lambda *_: None)
            saved = json.loads(path.read_text(encoding="utf-8"))
            snapshot = saved["input_snapshot"]
            self.assertEqual(snapshot["whitepaper"], actual)
            self.assertEqual(snapshot["blueprint"], normalize_world_blueprint(actual))
            self.assertEqual(agent._digest(snapshot["whitepaper"]), saved["binding"]["wp_hash"])
            with self.assertRaisesRegex(WorldBlueprintError, "binding mismatch"):
                agent.generate_world(self.wp, ScriptedTracer([]), checkpoint_path=path)
            resumed = ScriptedTracer([(PLAN, write("report", reads=["Aster"])),
                                      (WRITE, self.report), (PLAN, FINISH)])
            _, state = agent.generate_world(snapshot["whitepaper"], resumed,
                                            checkpoint_path=path, log=lambda *_: None)
            self.assertEqual(len(state["units"]), 2)
            self.assertEqual(sum(c["step"] == WRITE for c in resumed.calls), 1)

    def test_oversized_brief_preserves_original_input_without_silent_truncation(self):
        self.wp["corpus_plan"] = {"instructions": "x" * agent.MAX_CONTEXT_CHARS}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            tracer = ScriptedTracer([])
            with self.assertRaisesRegex(WorldBlueprintError, "requirements were not truncated"):
                agent.generate_world(self.wp, tracer, checkpoint_path=path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["input_snapshot"]["whitepaper"], self.wp)
            self.assertEqual(tracer.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
