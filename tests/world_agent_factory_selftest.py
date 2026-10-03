"""Original factory integration for agent-authored worlds; offline fixtures only."""
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
from pipeline import factory, world_gen, world_semantics, disclosure
from pipeline.world_blueprint import WorldBlueprintError
from seed_world_selftest import fixture


class LocalRun:
    scenario = "office"
    tracer = object()

    def __init__(self, directory, wp):
        self.dir = Path(directory)
        self.manifest = {"config": {}, "algo": {}}
        self.write("00_input.json", {"description": "Track business revisions", "few_shot": []})
        self.write("01_whitepaper.json", wp)

    def write(self, name, value):
        (self.dir / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def has(self, name):
        return (self.dir / name).is_file()

    def log(self, *args):
        pass

    def set_algo(self, **values):
        self.manifest["algo"].update(values)


class AgentFactoryTests(unittest.TestCase):
    def setUp(self):
        self.wp, self.world, self.table = fixture()
        self.wp["world_generation"] = {"strategy": "agentic", "version": 1}
        self.wp["active_lines"] = []
        self.meta = {"units": [], "initial_states": [], "status": "completed"}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Network forbidden"))
            guard.start(); self.addCleanup(guard.stop)

    def generator(self, **kw):
        return patch("pipeline.world_agent.generate_world", return_value=(deepcopy(self.table), deepcopy(self.meta)), **kw)

    def test_same_world_state_contract_and_bound_draft(self):
        draft = {}
        with self.generator() as generate:
            world = world_gen.build_world(self.wp, object(), log=lambda *_: None, draft_out=draft)
        self.assertEqual(world.entities, self.world.entities)
        self.assertEqual(world.relations, self.world.relations)
        self.assertEqual(world.events, self.world.events)
        self.assertEqual(draft["generation_strategy"], "agentic")
        self.assertEqual(draft["candidate_world_hash"], world_gen._world_digest(world.to_dict()))
        self.assertEqual(generate.call_count, 1)

    def test_invalid_complete_result_never_becomes_completed_draft(self):
        draft = {}
        table = deepcopy(self.table); table["events"] = []
        with patch("pipeline.world_agent.generate_world", return_value=(table, self.meta)):
            with self.assertRaises(WorldBlueprintError):
                world_gen.build_world(self.wp, object(), log=lambda *_: None, draft_out=draft)
        self.assertEqual(draft["status"], "error")
        self.assertNotIn("candidate_world", draft)

    def test_semantic_feedback_reuses_agent_state_and_preserves_published_base(self):
        draft = {}
        with self.generator():
            world_gen.build_world(self.wp, object(), log=lambda *_: None, draft_out=draft)
        feedback = {"status": "failed", "issues": [{"id": "i1", "message": "Check source ownership"}],
                    "messages": ["full redundant world"], "binding": {"world": "old"}}
        with self.generator() as generate:
            world_gen.build_world(self.wp, object(), log=lambda *_: None,
                repair_input={"draft": draft, "feedback": feedback})
        self.assertEqual(generate.call_args.kwargs["resume_state"], self.meta)
        self.assertEqual(generate.call_args.kwargs["feedback"]["issues"], feedback["issues"])
        self.assertNotIn("messages", generate.call_args.kwargs["feedback"])

    def test_agent_repair_cannot_substitute_a_different_review_target(self):
        draft = {}
        with self.generator():
            world_gen.build_world(self.wp, object(), log=lambda *_: None, draft_out=draft)
        feedback = {"status":"failed", "repair_targets":{"intrinsic":[], "structure":True}}
        with self.generator() as generate:
            with self.assertRaisesRegex(WorldBlueprintError, "targets differ"):
                world_gen.build_world(self.wp, object(), log=lambda *_:None,
                    repair_input={"draft":draft, "feedback":feedback,
                                  "targets":{"intrinsic":[], "structure":False}})
            generate.assert_not_called()

    def test_tampered_repair_draft_rejected_before_author(self):
        draft = {}
        with self.generator():
            world_gen.build_world(self.wp, object(), log=lambda *_: None, draft_out=draft)
        draft["candidate_world"]["n_sessions"] += 1
        with self.generator() as generate:
            with self.assertRaisesRegex(WorldBlueprintError, "stale or changed"):
                world_gen.build_world(self.wp, object(), repair_input={"draft": draft, "feedback": {"status": "failed"}})
        generate.assert_not_called()

    def test_new_seed_whitepaper_freezes_agent_strategy(self):
        run = LocalRun(self.temp.name, self.wp)
        original = deepcopy(self.wp); original.pop("world_generation")
        with patch.object(factory, "validate_seed_input", return_value={"seed_id": "fixture"}), \
             patch.object(factory, "validate_seed_identity"), \
             patch.object(factory, "central_office", return_value=original):
            factory.stage_whitepaper(run)
        self.assertEqual(run.read("01_whitepaper.json")["world_generation"],
                         {"strategy": "agentic", "version": 2, "blueprint_repair_attempts": 1,
                          "disclosure_strategy": "direct_batches_v1", "value_repair_attempts": 6})

    def test_factory_uses_checkpoint_and_preserves_existing_final_gates(self):
        wp = deepcopy(self.wp)
        wp["quality_contract"] = {"world_semantic_review": True, "public_disclosure": True}
        run = LocalRun(self.temp.name, wp)
        def author(wp, ws, tracer, **kw):
            ws.disclosure = {"records": [], "undisclosed": []}
            return {"status": "ready"}
        with self.generator() as generate, patch.object(factory, "validate_seed_identity"), \
             patch.object(factory, "_prepare_lines"), \
             patch.object(disclosure, "author_plan", side_effect=author) as disclosure_call, \
             patch.object(disclosure, "validate_plan", return_value=[]), \
             patch.object(world_semantics, "review_world", return_value={"status": "passed"}) as review_call, \
             patch.object(world_semantics, "validate_review", return_value=[]):
            factory.stage_world(run)
        self.assertTrue(generate.call_args.kwargs["checkpoint_path"].name.startswith("02_world_agent_"))
        self.assertEqual(disclosure_call.call_count, 1)
        self.assertEqual(review_call.call_count, 1)
        self.assertTrue(run.has("02_world.json"))

    def test_failed_final_review_publishes_candidate_with_warning(self):
        wp = deepcopy(self.wp); wp["quality_contract"] = {"world_semantic_review": True}
        run = LocalRun(self.temp.name, wp)
        with self.generator(), patch.object(factory, "validate_seed_identity"), \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", return_value={"status": "unresolved", "repair_targets": {}}):
            factory.stage_world(run)
        self.assertTrue(run.has("02_world.json"))
        self.assertTrue(run.has("02_world_candidate.json"))
        self.assertTrue(run.has(world_semantics.WARNING_ARTIFACT))

    def full_tracer(self, reject=False):
        from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part
        from world_scoped_agents_selftest import ScopedAgentFixture
        generator = ScriptedTracer([
            (PLAN, write("company")), (WRITE, table_part(self.table, [0])),
            (PLAN, {"action": "inspect", "read_entities": ["Aster"]}),
            (PLAN, write("report", reads=["Aster"])), (WRITE, table_part(self.table, [1], True)),
            (PLAN, FINISH)])
        reviewer = ScopedAgentFixture(reject=reject)
        class Combined:
            def chat_json(self, step, messages, **params):
                target = generator if step in (PLAN, WRITE) else reviewer
                return target.chat_json(step, messages, **params)
        return Combined(), generator, reviewer

    def test_real_original_stage_with_agent_disclosure_review_and_orders(self):
        wp = deepcopy(self.wp)
        wp["active_lines"] = [{"line": "L7_consolidation", "weight": 1}]
        wp["quality_contract"] = {"world_semantic_review": True, "public_disclosure": True}
        run = LocalRun(self.temp.name, wp)
        run.manifest["config"]["question_budget"] = 4
        run.tracer, author, reviewer = self.full_tracer()
        with patch.object(factory, "validate_seed_identity"):
            factory.stage_world(run)
            factory.stage_orders(run)
            factory.stage_well_posed(run)
        self.assertFalse(author.script)
        self.assertEqual(run.read(world_semantics.REVIEW_ARTIFACT)["status"], "passed")
        self.assertTrue(run.has("03_supply_report.json"))
        self.assertTrue(run.has("03_well_posed_report.json"))
        self.assertIsInstance(run.read("03_orders.json"), list)
        self.assertGreater(len(reviewer.calls), 4)
        self.assertEqual(len(list(run.dir.glob("02_world_agent_*.json"))), 1)

    def test_real_final_negative_review_continues_with_warning(self):
        wp = deepcopy(self.wp)
        wp["quality_contract"] = {"world_semantic_review": True, "public_disclosure": True}
        run = LocalRun(self.temp.name, wp)
        run.tracer, _, _ = self.full_tracer(reject=True)
        with patch.object(factory, "validate_seed_identity"):
            factory.stage_world(run)
        self.assertTrue(run.has("02_world.json"))
        self.assertTrue(run.has(world_semantics.WARNING_ARTIFACT))
        self.assertEqual(run.read("02_world_review_attempts.json")["attempts"][-1]["status"], "unresolved")

    def test_direct_disclosure_runs_through_real_original_world_and_order_gates(self):
        from disclosure_batches_selftest import DirectFixture
        wp = deepcopy(self.wp)
        wp["world_generation"]["disclosure_strategy"] = "direct_batches_v1"
        wp["active_lines"] = [{"line": "L7_consolidation", "weight": 1}]
        wp["quality_contract"] = {"world_semantic_review": True, "public_disclosure": True}
        run = LocalRun(self.temp.name, wp)
        run.manifest["config"]["question_budget"] = 4
        original, generator, reviewer = self.full_tracer()
        direct = DirectFixture()
        class Combined:
            def chat_json(inner, step, messages, **params):
                return (direct if step == disclosure.STEP else original).chat_json(step, messages, **params)
        run.tracer = Combined()
        with patch.object(factory, "validate_seed_identity"):
            factory.stage_world(run)
            factory.stage_orders(run)
            factory.stage_well_posed(run)
        self.assertTrue(run.has("02_world.json"))
        self.assertEqual(run.read(world_semantics.REVIEW_ARTIFACT)["status"], "passed")
        self.assertTrue(run.has("03_well_posed_report.json"))
        self.assertEqual(run.read("02_world.json")["disclosure"]["strategy"], "direct-disclosure/v2")
        self.assertTrue(list(run.dir.glob("02_disclosure_checkpoint_*.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
