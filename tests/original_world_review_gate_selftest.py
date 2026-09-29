"""Original factory routing, bounded repair and restart checks; no provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import factory, world_semantics
from pipeline import run as run_module
from pipeline.run import Run, Stage, drive
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_state import WorldState, Timeline, Op, SET


class LocalRun:
    scenario = "office"
    tracer = object()
    def __init__(self, directory):
        self.dir = Path(directory)
        self.manifest = {"config": {"world_semantic_review": True}, "algo": {}}
        self.write("00_input.json", {"description": "Maintain an office record", "few_shot": []})
        self.write("01_whitepaper.json", {"quality_contract": {"world_semantic_review": True}})
    def write(self, name, value):
        (self.dir / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))
    def has(self, name):
        return (self.dir / name).is_file()
    def log(self, *args): pass
    def set_algo(self, **kwargs): self.manifest["algo"].update(kwargs)


def candidate(value="pending"):
    return WorldState(entities={"Project": {"status": Timeline([Op(0, "2025-01-06", SET, value)])}},
                      n_sessions=2)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = LocalRun(self.temp.name)
        for target, name, value in [(factory, "validate_seed_identity", None),
                                     (factory, "validate_seed_world", {"passed": True})]:
            p = patch.object(target, name, return_value=value)
            p.start(); self.addCleanup(p.stop)

    def builder(self, wp, tracer, log, **kwargs):
        kwargs.get("draft_out", {}).update({"marker": "original author draft", "repair_log": []})
        return candidate("reviewed" if kwargs.get("repair_input") else "pending")

    def test_review_observes_prepared_world_and_receipt_is_published(self):
        observed = []
        def prepare(wp, ws, log): ws.conflicts = [{"marker": "prepared"}]
        def review(wp, ws, tracer, **kwargs):
            observed.append(deepcopy(ws.to_dict()))
            self.assertIn("description", kwargs["task_input"])
            return {"status": "passed"}
        with patch.object(factory, "build_world", side_effect=self.builder), \
             patch.object(factory, "_prepare_lines", side_effect=prepare), \
             patch.object(world_semantics, "review_world", side_effect=review), \
             patch.object(world_semantics, "validate_review", return_value=[]):
            factory.stage_world(self.run)
        self.assertEqual(observed[0]["conflicts"], [{"marker": "prepared"}])
        self.assertTrue(self.run.has("02_world.json"))
        self.assertEqual(self.run.read(world_semantics.REVIEW_ARTIFACT)["status"], "passed")

    def test_only_one_repair_returns_to_original_author(self):
        failure = {"status": "failed", "repair_targets": {"intrinsic": [], "structure": True}}
        with patch.object(factory, "build_world", side_effect=self.builder) as build, \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", side_effect=[failure, {"status": "passed"}]) as review, \
             patch.object(world_semantics, "validate_review", return_value=[]):
            factory.stage_world(self.run)
        self.assertEqual(build.call_count, 2)
        repair = build.call_args_list[1].kwargs["repair_input"]
        self.assertEqual(repair["draft"]["marker"], "original author draft")
        self.assertEqual(repair["feedback"], failure)
        self.assertEqual(repair["max_calls"], 4)
        self.assertEqual(review.call_args_list[1].kwargs["previous"], failure)
        self.assertEqual(len(self.run.read("02_world_review_attempts.json")["attempts"]), 2)

    def test_persistent_problem_publishes_candidate_with_release_warning(self):
        failure = {"status": "failed", "repair_targets": {"structure": True}}
        with patch.object(factory, "build_world", side_effect=self.builder) as build, \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", return_value=failure):
            factory.stage_world(self.run)
        self.assertEqual(build.call_count, 2)
        self.assertTrue(self.run.has("02_world.json"))
        self.assertTrue(self.run.has("02_world_candidate.json"))
        self.assertEqual(self.run.read(world_semantics.REVIEW_ARTIFACT)["status"], "failed")
        self.assertEqual(self.run.read(world_semantics.WARNING_ARTIFACT)["category"],
                         "semantic_review_not_passed")
        factory._require_current_world_review(self.run, self.run.read("01_whitepaper.json"))

    def test_failed_optional_repair_keeps_pre_repair_candidate_and_continues(self):
        failure = {"status": "failed", "repair_targets": {"structure": True}}
        original = candidate("pre-repair")
        with patch.object(factory, "build_world",
                          side_effect=[original, WorldBlueprintError("repair planning budget exhausted")]) as build, \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", return_value=failure):
            factory.stage_world(self.run)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(self.run.read("02_world.json"), original.to_dict())
        self.assertEqual(self.run.read(world_semantics.REVIEW_ARTIFACT), failure)
        self.assertEqual(self.run.read(world_semantics.WARNING_ARTIFACT)["release_eligible"], False)
        repair = self.run.read(factory.WORLD_REPAIR_FAILURE)
        self.assertEqual(repair["error_type"], "WorldBlueprintError")
        self.assertEqual(repair["binding"]["world_hash"], factory._canonical_hash(original.to_dict()))
        factory._require_current_world_review(self.run, self.run.read("01_whitepaper.json"))

    def test_review_execution_error_continues_without_business_rewrite(self):
        with patch.object(factory, "build_world", side_effect=self.builder) as build, \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", return_value={"status": "error"}):
            factory.stage_world(self.run)
        self.assertEqual(build.call_count, 1)
        self.assertTrue(self.run.has("02_world.json"))
        self.assertEqual(self.run.read(world_semantics.WARNING_ARTIFACT)["category"],
                         "review_execution_error")

    def test_new_candidate_replaces_previous_world_with_bound_warning(self):
        old = candidate("previous").to_dict()
        receipt = {"status": "passed", "marker": "previous receipt"}
        self.run.write("02_world.json", old)
        self.run.write(world_semantics.REVIEW_ARTIFACT, receipt)
        with patch.object(factory, "build_world", side_effect=self.builder), \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world", return_value={"status": "unresolved"}):
            factory.stage_world(self.run)
        self.assertNotEqual(self.run.read("02_world.json"), old)
        self.assertEqual(self.run.read(world_semantics.REVIEW_ARTIFACT)["status"], "unresolved")
        self.assertTrue(self.run.has(world_semantics.WARNING_ARTIFACT))

    def test_publish_write_failure_restores_whole_bundle_byte_for_byte(self):
        names = ["02_world.json", world_semantics.REVIEW_ARTIFACT, "02_seed_audit.json", factory.CORPUS_CKPT]
        for name in names: self.run.write(name, {"old": name})
        before = {name: (self.run.dir / name).read_bytes() for name in names}
        self.run.manifest["algo"] = {"entities": 1, "sessions": 2}
        write = self.run.write
        def fail(name, value):
            if name == world_semantics.REVIEW_ARTIFACT: raise OSError("injected receipt write failure")
            write(name, value)
        with patch.object(self.run, "write", side_effect=fail):
            with self.assertRaises(OSError):
                factory._publish_world_bundle(self.run, {name: {"new": name} for name in names[:3]},
                                              {"entities": 20})
        self.assertEqual(before, {name: (self.run.dir / name).read_bytes() for name in names})
        self.assertEqual(self.run.manifest["algo"], {"entities": 1, "sessions": 2})

    def test_publish_metadata_failure_leaves_no_new_world_or_receipt(self):
        with patch.object(self.run, "set_algo", side_effect=RuntimeError("injected manifest failure")):
            with self.assertRaises(RuntimeError):
                factory._publish_world_bundle(self.run,
                    {"02_world.json": {}, world_semantics.REVIEW_ARTIFACT: {"status": "passed"}},
                    {"entities": 8})
        self.assertFalse(self.run.has("02_world.json"))
        self.assertFalse(self.run.has(world_semantics.REVIEW_ARTIFACT))

    def test_orders_cannot_bypass_missing_or_stale_review(self):
        self.run.write("02_world.json", candidate().to_dict())
        with patch.object(factory, "run_lines") as author:
            with self.assertRaises(WorldBlueprintError): factory.stage_orders(self.run)
            author.assert_not_called()
        self.run.write(world_semantics.REVIEW_ARTIFACT, {"status": "passed"})
        with patch.object(world_semantics, "validate_review", return_value=["stale world"]), \
             patch.object(factory, "run_lines") as author:
            with self.assertRaises(WorldBlueprintError): factory.stage_orders(self.run)
            author.assert_not_called()

    def test_corpus_checks_frozen_whitepaper_before_local_render_upgrade(self):
        original = {"quality_contract": {"world_semantic_review": True}}
        self.run.write("01_whitepaper.json", original)
        self.run.write("02_world.json", candidate().to_dict())
        self.run.write(world_semantics.REVIEW_ARTIFACT, {"status": "passed"})
        def validate(report, wp, ws, **kwargs):
            self.assertEqual(wp, original)
            return []
        def render(wp, ws, target, tracer, corpus, done, save, log, **kwargs):
            self.assertTrue(wp["quality_contract"]["corpus_review"])
            corpus["sessions"] = [{"session_id": sid, "docs": []} for sid in ws.sessions()]
            done.update(ws.sessions())
            save()
        with patch.object(world_semantics, "validate_review", side_effect=validate), \
             patch.object(factory, "render_corpus", side_effect=render):
            factory.stage_corpus(self.run)
        self.assertEqual(self.run.read("01_whitepaper.json"), original)

    def test_legacy_world_stage_keeps_original_call_shape(self):
        self.run.manifest["config"] = {}
        self.run.write("01_whitepaper.json", {})
        with patch.object(factory, "build_world", return_value=candidate()) as build, \
             patch.object(factory, "_prepare_lines"), \
             patch.object(world_semantics, "review_world") as review:
            factory.stage_world(self.run)
        self.assertNotIn("draft_out", build.call_args.kwargs)
        self.assertNotIn("repair_input", build.call_args.kwargs)
        review.assert_not_called()


class FreshnessDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = patch.object(run_module, "RUNS_DIR", Path(self.temp.name))
        p.start(); self.addCleanup(p.stop)
        self.run = Run("office", "offline_world_review")
        self.current = True
        self.calls = []
        def upstream(run):
            self.calls.append("world")
            self.current = True
            run.write("world.json", {"revision": len(self.calls)})
        def downstream(run):
            self.calls.append("orders")
            run.write("orders.json", run.read("world.json"))
        self.stages = [Stage("world", [], upstream, "world.json", is_current=lambda _: self.current),
                       Stage("orders", ["world"], downstream, "orders.json")]
        drive(self.run, self.stages)

    def test_stale_world_rebuild_invalidates_completed_descendants(self):
        self.current = False
        drive(self.run, self.stages)
        self.assertEqual(self.calls, ["world", "orders", "world", "orders"])
        self.assertEqual(self.run.read("world.json"), self.run.read("orders.json"))

    def test_midstream_restart_checks_dependency_freshness_before_skip(self):
        self.current = False
        with self.assertRaises(SystemExit): drive(self.run, self.stages, from_stage="orders")
        self.assertEqual(self.calls, ["world", "orders"])

    def test_declared_refresh_stage_can_revalidate_stale_dependency(self):
        self.current = False
        def refresh(run):
            self.calls.append("review")
            self.current = True
            run.write("review.json", {"current": True})
        stages = [self.stages[0],
                  Stage("review", ["world"], refresh, "review.json", refreshes=("world",))]
        drive(self.run, stages, only="review")
        self.assertEqual(self.calls, ["world", "orders", "review"])
        self.assertTrue(self.current)

    def test_one_drive_replays_unchanged_freshness_once(self):
        checks = []
        stages = [Stage("world", [], self.stages[0].fn, "world.json",
                        is_current=lambda _: checks.append("world") or True),
                  self.stages[1],
                  Stage("grounding", ["orders"],
                        lambda run: run.write("grounding.json", {}), "grounding.json")]
        drive(self.run, stages)
        checks.clear()
        drive(self.run, stages, from_stage="orders")
        self.assertEqual(checks, ["world"])

    def test_descendant_freshness_can_cover_ancestor_replay(self):
        checks = []
        stages = [Stage("world", [], self.stages[0].fn, "world.json",
                        is_current=lambda _: checks.append("world") or True),
                  Stage("disclosure", ["world"], self.stages[1].fn, "orders.json",
                        is_current=lambda _: checks.append("disclosure") or True,
                        freshness_covers=("world",)),
                  Stage("corpus", ["world", "disclosure"],
                        lambda run: run.write("corpus.json", {}), "corpus.json")]
        drive(self.run, stages)
        checks.clear()
        drive(self.run, stages, only="corpus")
        self.assertEqual(checks, ["disclosure"])

    def test_completed_indirect_descendant_cannot_skip_stale_world(self):
        for name in ("questions", "grounding"):
            self.stages.append(Stage(name, [self.stages[-1].name],
                lambda run, name=name: run.write(name + ".json", {}), name + ".json"))
        drive(self.run, self.stages)
        self.current = False
        for name in ("questions", "grounding"):
            with self.assertRaises(SystemExit): drive(self.run, self.stages, only=name)
        self.assertEqual(self.calls, ["world", "orders"])


if __name__ == "__main__":
    unittest.main()
