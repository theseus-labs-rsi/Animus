from copy import deepcopy
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import factory, world_semantics


class Run:
    def __init__(self, review, warning=False):
        self.files = {world_semantics.REVIEW_ARTIFACT: review}
        if warning:
            self.files[world_semantics.WARNING_ARTIFACT] = {"status": "warning"}
    def has(self, name):
        return name in self.files
    def read(self, name):
        return self.files[name]


class DisclosureWorldReviewRefreshTests(unittest.TestCase):
    def test_exact_execution_warning_still_requires_fresh_review(self):
        run = Run({"status": "error"}, warning=True)
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertFalse(factory._release_current_world_review(run, {}, object(), {}))

    def test_current_passed_review_can_be_reused(self):
        run = Run({"status": "passed"})
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertTrue(factory._release_current_world_review(run, {}, object(), {}))

    def test_invalid_or_errored_review_requires_refresh(self):
        for review, errors in (({"status": "error"}, []), ({"status": "passed"}, [{"code": "stale"}])):
            with self.subTest(review=review, errors=errors), \
                    patch.object(world_semantics, "validate_review", return_value=errors):
                self.assertFalse(factory._release_current_world_review(Run(review), {}, object(), {}))

    def test_public_repair_retains_old_review_transcript_and_counts_each_call_once(self):
        old = {"status": "failed", "repair_targets": {"intrinsic": [], "structure": False,
                "disclosure": True}}
        fresh = {"status": "passed", "repair_targets": {"intrinsic": [], "structure": False}}

        class World:
            disclosure = {"strategy": "direct-disclosure/v2", "plan_hash": "old", "records": [],
                          "undisclosed": []}

            def to_dict(self):
                return {"disclosure": self.disclosure}

        class StageRun:
            def __init__(self, path):
                self.dir = path
                self.tracer = object()
                self.manifest = {"config": {}}
                self.files = {factory.ART["whitepaper"]: {}, factory.ART["input"]: {},
                              factory.ART["world"]: {"disclosure": World.disclosure}}

            def has(self, name):
                return name in self.files

            def read(self, name):
                return self.files[name]

            def write(self, name, value):
                self.files[name] = value

        with tempfile.TemporaryDirectory() as folder:
            run = StageRun(Path(folder))
            checkpoint = run.dir / "02_world_review_historical.ckpt.json"
            checkpoint.write_bytes(b"historical review transcript")
            with patch.object(factory.WorldState, "from_dict", return_value=World()), \
                    patch.object(factory, "validate_seed_identity"), \
                    patch.object(factory, "validate_seed_world"), \
                    patch.object(factory, "_release_current_world_review", return_value=False), \
                    patch.object(factory, "_recoverable_current_world_review", return_value=None), \
                    patch.object(factory, "_publish_world_bundle"), \
                    patch("pipeline.disclosure.enabled", return_value=True), \
                    patch("pipeline.disclosure.validate_plan", return_value=[]), \
                    patch("pipeline.disclosure._world", return_value={"truth": "unchanged"}), \
                    patch("pipeline.disclosure_batches.repair", return_value={
                        "status": "ready", "repaired_record_ids": ["d14"]}), \
                    patch.object(world_semantics, "review_world", side_effect=[old, fresh]) as reviewer, \
                    patch.object(world_semantics, "validate_review", return_value=[]):
                factory.stage_disclosure(run)

            self.assertEqual(reviewer.call_count, 2)
            self.assertEqual(run.read("02_world_review_attempts.json")["attempts"], [old, fresh])
            self.assertEqual(checkpoint.read_bytes(), b"historical review transcript")
            self.assertEqual(run.read("02_disclosure_semantic_repair.json")[
                "retained_review_checkpoints"], 1)


class DisclosureSupplyRefreshTests(unittest.TestCase):
    """Publication-only recovery rechecks the real original supply, offline."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from instance_plan_selftest import setup_plan
        from original_world_review_gate_selftest import LocalRun
        from pipeline import instance_plan as ip
        from pipeline.capability_contract import digest
        from pipeline.world_state import assemble_world
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.run = LocalRun(temporary.name)
        self.run.manifest["config"]["production_control"] = {"version": "supply-driven/v6"}
        wp, proposal, table, full, _ = setup_plan()
        self.wp = ip.compile_plan(wp, proposal)
        self.world, issues = assemble_world(table, blueprint=self.wp["world_blueprint"],
                                            include_shape_diagnostics=False)
        self.assertEqual(issues, [])
        self.world.disclosure = {"strategy": "direct-disclosure/v2", "plan_hash": "old",
                                 "records": [], "undisclosed": []}
        state = {"units": [{"instance_units": ["report-case"],
                            "instance_mapping": full["slot_mapping"]}]}
        from pipeline.construction import resolve
        self.world.supply_construction = resolve(self.wp, ip.active_mapping(state))
        proof = ip.fulfillment(self.wp, self.world, state)
        self.assertTrue(proof["passed"], proof["findings"])
        self.digest = digest
        self.old_review = {"status": "passed", "repair_targets": {}}
        self.run.write("01_whitepaper.json", self.wp)
        self.run.write("02_world.json", self.world.to_dict())
        self.run.write("02_instance_fulfillment.json", proof)
        self.run.write("01_joint_design_audit.json", {"used_revisions": 0})
        self.run.write("01_joint_design_receipt.json", {
            "status": "ready_for_world_probe", "whitepaper_hash": digest(self.wp),
            "supply_plan_hash": digest(self.wp["supply_plan"]),
            "business_instance_plan_hash": digest(self.wp["business_instance_plan"])})
        self.run.write(world_semantics.REVIEW_ARTIFACT, self.old_review)
        self.run.write("02_source_supply_gate.json", {
            "status": "source_supply_ready", "whitepaper_hash": digest(self.wp),
            "world_hash": digest(self.world.to_dict()), "actual_supply_hash": digest(proof),
            "business_review_hash": digest(self.old_review)})
        identity = patch.object(factory, "validate_seed_identity")
        identity.start(); self.addCleanup(identity.stop)

    def recover(self, *, review=None):
        fresh = review if review is not None else {"status": "passed", "repair_targets": {}}
        def author(_wp, ws, *_args, **_kwargs):
            ws.disclosure["plan_hash"] = "new"
            return {"status": "ready"}
        with patch("pipeline.disclosure.enabled", return_value=True), \
                patch("pipeline.disclosure.validate_plan", side_effect=[["incomplete"], []]), \
                patch("pipeline.disclosure.author_plan", side_effect=author) as author_call, \
                patch.object(world_semantics, "review_world", return_value=fresh) as reviewer, \
                patch.object(world_semantics, "validate_review", return_value=[]):
            factory.stage_disclosure(self.run)
        self.assertEqual((author_call.call_count, reviewer.call_count), (1, 1))

    def test_bound_warning_does_not_buy_a_fourth_repair_or_review(self):
        old = {"status": "failed", "repair_targets": {"disclosure": True}}
        warning = {"status": "warning", "release_eligible": False}
        self.run.write(world_semantics.REVIEW_ARTIFACT, old)
        self.run.write(world_semantics.WARNING_ARTIFACT, warning)
        gate = self.run.read("02_source_supply_gate.json")
        gate.update(status="exploratory_source_observed", release_eligible=False,
                    business_review_hash=self.digest(old), world_review_warning=True,
                    world_review_warning_hash=self.digest(warning))
        self.run.write("02_source_supply_gate.json", gate)
        with patch("pipeline.disclosure.enabled", return_value=True), \
                patch("pipeline.disclosure.validate_plan", return_value=[]), \
                patch.object(factory, "_require_current_world_review"), \
                patch("pipeline.disclosure_batches.repair") as repair, \
                patch.object(world_semantics, "review_world") as review:
            factory.stage_disclosure(self.run)
        repair.assert_not_called(); review.assert_not_called()
        self.assertEqual(self.run.read(world_semantics.WARNING_ARTIFACT), warning)
        self.assertEqual(self.run.read("02_source_supply_gate.json"), gate)

    def test_public_recovery_recomputes_supply_and_its_world_binding(self):
        from pipeline.disclosure import _world
        truth = _world(self.world)
        old = self.run.read("02_instance_fulfillment.json")
        self.recover()
        world = self.run.read("02_world.json")
        proof = self.run.read("02_instance_fulfillment.json")
        gate = self.run.read("02_source_supply_gate.json")
        self.assertEqual(_world(factory.WorldState.from_dict(world)), truth)
        self.assertTrue(proof["passed"])
        self.assertNotEqual(proof["world_hash"], old["world_hash"])
        self.assertEqual(proof["world_hash"], self.digest(world))
        self.assertEqual(gate["world_hash"], self.digest(world))
        self.assertEqual(gate["actual_supply_hash"], self.digest(proof))
        factory._require_v6_supply_gate(self.run, self.wp, world)
        with patch.object(factory, "_require_current_world_review"):
            factory.stage_orders(self.run)
        self.assertEqual(len(self.run.read("03_orders.json")), 1)

    def test_unresolved_review_keeps_the_refreshed_source_release_ineligible(self):
        failure = {"status": "unresolved", "repair_targets": {}}
        self.recover(review=failure)
        gate = self.run.read("02_source_supply_gate.json")
        self.assertEqual(gate["status"], "exploratory_source_observed")
        self.assertFalse(gate["release_eligible"])
        self.assertEqual(gate["world_review_warning_hash"],
                         self.digest(self.run.read(world_semantics.WARNING_ARTIFACT)))
        with patch.object(world_semantics, "validate_generation_warning", return_value=[]):
            factory._require_v6_supply_gate(self.run, self.wp, self.run.read("02_world.json"))

    def test_an_actual_shortage_is_not_signed_as_a_ready_source(self):
        from pipeline import instance_plan as ip
        from pipeline.supply_capacity import CapacityReviewRequested
        proof = self.run.read("02_instance_fulfillment.json")
        proof.update(passed=False, findings=[{"code": "actual_supply_shortfall"}])
        before = self.run.read("02_world.json")
        with patch.object(ip, "fulfillment", return_value=proof), \
                self.assertRaises(CapacityReviewRequested):
            self.recover()
        self.assertEqual(self.run.read("02_world.json"), before)
        self.assertEqual(self.run.read("02_source_supply_gate.json")["world_hash"], self.digest(before))

    def test_design_limit_remains_exploratory_after_a_fresh_positive_review(self):
        from pipeline.joint_design import MAX_REVISIONS
        design = self.run.read("01_joint_design_receipt.json")
        design.update(release_eligible=False, fallback="retained_compiled_world")
        self.run.write("01_joint_design_receipt.json", design)
        self.run.write("01_joint_design_audit.json", {"used_revisions": MAX_REVISIONS})
        self.recover()
        self.assertFalse(self.run.read("02_source_supply_gate.json")["release_eligible"])
        factory._require_v6_supply_gate(self.run, self.wp, self.run.read("02_world.json"))

    def test_real_shortage_stays_exploratory_after_a_fresh_positive_review(self):
        from pipeline import instance_plan as ip
        from pipeline.joint_design import MAX_REVISIONS
        self.wp["supply_plan"]["candidate_allocation"]["L7_consolidation"] = 2
        self.wp["supply_plan"]["candidate_budget"] = 2
        old = self.run.read("02_instance_fulfillment.json")
        progress = old["progress"]
        proof = ip.fulfillment(self.wp, self.world, {
            "identity_registry": progress["slot_mapping"],
            "units": [{"instance_units": progress["completed_units"]}]})
        self.assertFalse(proof["passed"])
        self.run.write("01_whitepaper.json", self.wp)
        self.run.write("02_instance_fulfillment.json", proof)
        self.run.write("01_joint_design_audit.json", {"used_revisions": MAX_REVISIONS})
        design = self.run.read("01_joint_design_receipt.json")
        design.update(whitepaper_hash=self.digest(self.wp),
                      supply_plan_hash=self.digest(self.wp["supply_plan"]),
                      release_eligible=False, fallback="retained_compiled_world")
        self.run.write("01_joint_design_receipt.json", design)
        gate = self.run.read("02_source_supply_gate.json")
        gate.update(status="exploratory_source_observed", release_eligible=False,
                    whitepaper_hash=self.digest(self.wp), actual_supply_hash=self.digest(proof))
        self.run.write("02_source_supply_gate.json", gate)
        self.recover()
        refreshed = self.run.read("02_instance_fulfillment.json")
        self.assertFalse(refreshed["passed"])
        self.assertEqual(refreshed["actual_supply"]["L7_consolidation"]["available"], 1)
        self.assertFalse(self.run.read("02_source_supply_gate.json")["release_eligible"])
        factory._require_v6_supply_gate(self.run, self.wp, self.run.read("02_world.json"))

    def test_source_write_failure_rolls_back_world_review_and_fulfillment(self):
        names = ["02_world.json", world_semantics.REVIEW_ARTIFACT,
                 "02_source_supply_gate.json", "02_instance_fulfillment.json"]
        before = {name: (self.run.dir / name).read_bytes() for name in names}
        writer = self.run.write
        def fail(name, value):
            if name == "02_instance_fulfillment.json":
                raise OSError("injected supply receipt write failure")
            writer(name, value)
        with patch.object(self.run, "write", side_effect=fail), self.assertRaises(OSError):
            self.recover()
        self.assertEqual({name: (self.run.dir / name).read_bytes() for name in names}, before)
        self.assertFalse(self.run.has(factory.ART["disclosure"]))


if __name__ == "__main__":
    unittest.main()
