"""Real closed-loop/world/Run transaction paths with provider-free model fixtures.

These tests establish rollback and audit retention, not model semantic ability.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from pipeline import factory, run as run_module, world_semantics
from pipeline.closed_loop import build_to_target
from pipeline.targetspec import TargetSpec
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_state import Op, SET, Timeline, WorldState


ACCEPT = {"decision": "accept", "reason": "Offline transaction fixture.", "issues": [],
          "mechanism_coverage": [], "repair_targets": {"intrinsic": [], "structure": False},
          "limitations": "Does not establish business correctness."}
PUBLISHED = ("01_whitepaper.json", "01_seed_audit.json", "02_world.json", world_semantics.REVIEW_ARTIFACT,
             world_semantics.WARNING_ARTIFACT, "02_seed_audit.json", factory.CORPUS_CKPT)


class StopAfterWorld(Exception):
    pass


class ClosedLoopWorldTransactionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="bc_world_transaction_"))
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network prohibited")))
        self.stack.enter_context(patch.object(config, "chat", side_effect=AssertionError("API prohibited")))
        self.provider = self.stack.enter_context(patch.object(config, "chat_json", return_value=deepcopy(ACCEPT)))
        self.stack.enter_context(patch.object(run_module, "RUNS_DIR", Path(directory)))
        self.run = run_module.Run("office", "test", config_meta={"world_semantic_review": True})
        self.run.log = lambda *_: None
        self.wp = {"active_lines": [{"line": "L1_timeline", "weight": 1}],
                   "shared_world_spec": {"entities": {"count": 1}, "timeline": {"n_sessions": 2}},
                   "domain_profile": {"field_schema": []},
                   "seed_contract": {"task": {"objective": "Offline transaction fixture"}},
                   "quality_contract": {"world_semantic_review": True}}
        self.task = {"description": "Keep record status in an offline fixture.", "few_shot": []}
        self.run.write("00_input.json", self.task)
        self.run.write("01_whitepaper.json", self.wp)
        self.stack.enter_context(patch.object(factory, "validate_seed_identity"))
        self.stack.enter_context(patch.object(factory, "validate_seed_world", return_value={"passed": True, "new": True}))
        self.stack.enter_context(patch.object(factory, "_prepare_lines"))
        self.author_calls = 0

        def author(wp, tracer, log, **kwargs):
            self.author_calls += 1
            kwargs["draft_out"].update({"author_attempt": self.author_calls})
            return self.world(f"candidate{self.author_calls}", wp["shared_world_spec"]["timeline"]["n_sessions"])

        self.stack.enter_context(patch.object(factory, "build_world", side_effect=author))
        self.orders = self.stack.enter_context(patch.object(factory, "stage_orders", side_effect=StopAfterWorld))

    @staticmethod
    def world(value, n_sessions=2):
        return WorldState({"Record": {"status": Timeline([Op(0, "2025-01-06", SET, value)])}},
                          n_sessions=n_sessions)

    def old_publication(self):
        ws = self.world("old")
        review = world_semantics.review_world(self.wp, ws, self.run.tracer, task_input=self.task)
        self.assertEqual(world_semantics.validate_review(review, self.wp, ws, task_input=self.task), [])
        self.run.write("02_world.json", ws.to_dict())
        self.run.write(world_semantics.REVIEW_ARTIFACT, review)
        self.run.write("02_seed_audit.json", {"old": True})
        self.run.write(factory.CORPUS_CKPT, {"old_render_progress": 1})
        # Preserve exact source bytes, including noncanonical whitespace.
        wp_path = self.run.dir / "01_whitepaper.json"
        wp_path.write_bytes(wp_path.read_bytes() + b"\n\n")
        self.run.set_algo(entities=1, sessions=2, old_metadata={"kept": True})
        self.run.mark("world", "02_world.json", 0)

    def snapshot(self):
        return {name: (self.run.dir / name).read_bytes() if (self.run.dir / name).exists() else None
                for name in PUBLISHED}

    def invoke(self, *, order_subrounds=1):
        build_to_target(self.run, TargetSpec(min_questions=1, per_line_min={"L1_timeline": 1}),
                        max_rounds=1, order_subrounds=order_subrounds)

    def assert_rollback(self, before, algo, *, expected_calls):
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.run.manifest["algo"], algo)
        disk = self.run.read("manifest.json")
        self.assertEqual(disk["algo"], algo)
        self.assertEqual(disk["status"], "failed")
        self.assertEqual(disk["stages"]["world"]["status"], "failed")
        self.assertFalse(disk["stages"]["world"]["done"])
        self.assertNotIn("finished_ts", disk["stages"]["world"])
        self.assertEqual(disk["llm_calls"], expected_calls)
        rows = [json.loads(line) for line in self.run.tracer.pfile.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), expected_calls)
        self.assertEqual([row["i"] for row in rows], list(range(1, expected_calls + 1)))
        self.assertTrue(self.run.has("02_world_draft.json"))
        self.assertTrue(self.run.has("02_world_candidate.json"))
        self.assertTrue(self.run.has("02_world_review_attempts.json"))
        if before["02_world.json"] is not None:
            self.assertEqual(world_semantics.validate_review(self.run.read(world_semantics.REVIEW_ARTIFACT),
                self.run.read("01_whitepaper.json"), WorldState.from_dict(self.run.read("02_world.json")),
                task_input=self.task), [])

    def test_actual_mark_manifest_save_failure_rolls_back_published_bundle(self):
        self.old_publication()
        before, algo = self.snapshot(), deepcopy(self.run.manifest["algo"])
        original_save = self.run._save_manifest
        injected = False

        def fail_once_on_mark():
            nonlocal injected
            if not injected and self.run.is_done("world"):
                injected = True
                raise OSError("Injected mark manifest save failure")
            original_save()

        with patch.object(self.run, "_save_manifest", side_effect=fail_once_on_mark):
            with self.assertRaisesRegex(OSError, "Injected mark"):
                self.invoke()
        self.assertTrue(injected)
        self.orders.assert_not_called()
        self.assert_rollback(before, algo, expected_calls=2)

    def test_accepted_upstream_definition_reaches_later_supply_round(self):
        from pipeline.closed_loop import _run_world_attempt
        wp = deepcopy(self.wp)
        def revised_stage(run):
            changed = run.read("01_whitepaper.json")
            changed["blueprint_revisions"] = [{"reason": "Reviewed cyclic state"}]
            run.write("01_whitepaper.json", changed)
            run.write("01_seed_audit.json", {"passed": True, "revised": True})
            run.write("02_world.json", self.world("revised").to_dict())
        _run_world_attempt(self.run, wp, revised_stage, factory.ART, factory.CORPUS_CKPT)
        self.assertEqual(wp, self.run.read("01_whitepaper.json"))
        self.assertEqual(wp["blueprint_revisions"][0]["reason"], "Reviewed cyclic state")
        before = self.snapshot()
        def failed_later_stage(run):
            run.write("01_seed_audit.json", {"passed": False, "injected": True})
            raise WorldBlueprintError("Later attempt rejected")
        with self.assertRaises(WorldBlueprintError):
            _run_world_attempt(self.run, wp, failed_later_stage, factory.ART, factory.CORPUS_CKPT)
        self.assertEqual(self.snapshot(), before)

    def test_interrupt_after_successful_mark_restores_old_world_and_failed_status(self):
        self.old_publication()
        before, algo = self.snapshot(), deepcopy(self.run.manifest["algo"])
        mark = self.run.mark

        def interrupted_mark(*args, **kwargs):
            mark(*args, **kwargs)
            raise KeyboardInterrupt("Injected interruption after mark")

        with patch.object(self.run, "mark", side_effect=interrupted_mark):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke()
        self.assert_rollback(before, algo, expected_calls=2)

    def test_failed_first_publication_removes_new_world_but_keeps_attempts(self):
        before, algo = self.snapshot(), deepcopy(self.run.manifest["algo"])
        mark = self.run.mark

        def failed_mark(*args, **kwargs):
            mark(*args, **kwargs)
            raise OSError("Injected post-mark failure")

        with patch.object(self.run, "mark", side_effect=failed_mark):
            with self.assertRaises(OSError):
                self.invoke()
        self.assert_rollback(before, algo, expected_calls=1)
        self.assertEqual(self.run.read("02_world_review_attempts.json")["attempts"][0]["status"], "passed")

    def test_real_world_semantic_negative_commits_candidate_then_reaches_orders(self):
        self.old_publication()
        before = self.snapshot()
        self.provider.return_value = {**ACCEPT, "decision": "unresolved", "reason": "Offline negative fixture."}
        with self.assertRaises(StopAfterWorld):
            self.invoke()
        self.orders.assert_called_once()
        self.assertNotEqual(self.snapshot(), before)
        self.assertEqual(self.run.manifest["stages"]["world"]["status"], "succeeded")
        self.assertTrue(self.run.has(world_semantics.WARNING_ARTIFACT))
        self.assertEqual(self.run.read(world_semantics.REVIEW_ARTIFACT)["status"], "unresolved")
        self.assertEqual(self.run.read("02_world_review_attempts.json")["attempts"][0]["status"], "unresolved")

    def test_second_attempt_rolls_back_to_first_committed_world(self):
        self.old_publication()
        committed = {}

        def no_supply(run):
            committed.update(files=self.snapshot(), algo=deepcopy(run.manifest["algo"]))
            run.write(factory.ART["orders"], [])

        self.orders.side_effect = no_supply
        self.stack.enter_context(patch.object(factory, "stage_well_posed",
            side_effect=lambda run: run.write("03_well_posed_report.json", {})))
        mark = self.run.mark

        def fail_second_world(*args, **kwargs):
            mark(*args, **kwargs)
            if args[0] == "world" and self.author_calls == 2:
                raise OSError("Injected second world mark failure")

        with patch.object(self.run, "mark", side_effect=fail_second_world):
            with self.assertRaisesRegex(OSError, "second world"):
                self.invoke(order_subrounds=2)
        self.assertEqual(self.author_calls, 2)
        self.assert_rollback(committed["files"], committed["algo"], expected_calls=3)
        self.assertIsNone(committed["files"][factory.CORPUS_CKPT])
        self.assertEqual(self.run.read("02_world_candidate.json")["entities"]["Record"]["status"][0]["value"], "candidate2")

    def test_successful_world_stays_committed_when_later_orders_stop(self):
        self.old_publication()
        before = self.snapshot()
        with self.assertRaises(StopAfterWorld):
            self.invoke()
        self.assertNotEqual(self.snapshot()["01_whitepaper.json"], before["01_whitepaper.json"])
        self.assertNotEqual(self.snapshot()["02_world.json"], before["02_world.json"])
        self.assertFalse(self.run.has(factory.CORPUS_CKPT))
        self.assertTrue(self.run.is_done("world"))
        self.assertEqual(self.run.manifest["stages"]["orders"]["status"], "failed")
        self.assertEqual(world_semantics.validate_review(self.run.read(world_semantics.REVIEW_ARTIFACT),
            self.run.read("01_whitepaper.json"), WorldState.from_dict(self.run.read("02_world.json")),
            task_input=self.task), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
