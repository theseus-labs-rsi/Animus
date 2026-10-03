"""Explicit source migration retains old records and the normal restore guard."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENAI_API_KEY"] = "offline-disabled"
os.environ["MODEL"] = "offline-disabled"
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from tools import migrate_world_checkpoint as migration
with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
    import config
    from instance_plan_selftest import setup_plan, authored_values, ScriptedTracer
    from pipeline import instance_plan, world_agent
    from pipeline.world_blueprint import WorldBlueprintError


class MigrationTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Provider/network forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        wp, plan, _table, full, _action = setup_plan()
        self.wp = instance_plan.compile_plan(wp, plan)
        values = [authored_values(self.wp, unit, full) for unit in self.wp["business_instance_plan"]["units"]]
        self.table, state = world_agent.generate_world(self.wp,
            ScriptedTracer([("world.instance.values", value) for value in values]), log=lambda *_: None)
        self.current_binding = deepcopy(state["binding"])
        # A synthetic changed source, never real migration evidence. The real
        # frozen environments are exercised by the archived-fixture proof.
        state["binding"]["implementation"]["instance_executor.py"] = "synthetic-legacy-source"
        state["local_repairs"] = [{"unit_id": state["units"][0]["unit_id"],
            "status": "running", "original_evidence": "historical receipt without a new protocol"}]
        self.parent = self.directory / "parent.json"
        world_agent._save(self.parent, state)
        self.state = json.loads(self.parent.read_text(encoding="utf-8"))
        self.parent_bytes = self.parent.read_bytes()
        self.legacy_root = self.directory / "legacy"
        self.legacy_root.mkdir()
        self.output = self.directory / "child"
        self.inspections = 0

    def inspect(self, root, _checkpoint):
        self.inspections += 1
        return {"binding": deepcopy(self.state["binding"] if root == self.legacy_root else self.current_binding),
            "checkpoint_sha256": hashlib.sha256(self.parent_bytes).hexdigest(),
            "table": deepcopy(self.table), "world": {"synthetic_inspection": True},
            "applied": deepcopy(self.table), "initial": {}, "issues": [], "source_hashes": {"synthetic": "only-tests"}}

    def migrate(self, inspector=None, expected=None):
        return migration.migrate(self.parent,
            expected_sha256=expected or hashlib.sha256(self.parent_bytes).hexdigest(),
            legacy_root=self.legacy_root, target_root=Path(__file__).resolve().parents[1],
            destination=self.output, inspect_fn=inspector or self.inspect)

    def test_child_restores_normally_without_relabeling_old_records_or_calls(self):
        with self.assertRaises(WorldBlueprintError):
            world_agent.generate_world(self.wp, ScriptedTracer([]), resume_state=self.state, log=lambda *_: None)
        receipt = self.migrate()
        child = json.loads((self.output / self.parent.name).read_text(encoding="utf-8"))
        for key, value in self.state.items():
            if key not in ("binding", "checkpoint_hash"):
                self.assertEqual(child[key], value, key)
        self.assertEqual((self.output / "parent_checkpoint.json").read_bytes(), self.parent_bytes)
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)
        self.assertNotIn("version", child["local_repairs"][0])
        self.assertFalse(receipt["paid_resume_authorization"])
        self.assertFalse(receipt["semantic_revalidation"])
        trace = ScriptedTracer([])
        table, resumed = world_agent.generate_world(self.wp, trace, resume_state=child, log=lambda *_: None)
        self.assertEqual(table, self.table)
        self.assertEqual(trace.calls, [])
        self.assertEqual(resumed["log"], self.state["log"])
        self.assertEqual(resumed["steps"], self.state["steps"])
        self.assertEqual(resumed["local_repairs"], self.state["local_repairs"])

    def test_external_parent_hash_precedes_inspection_and_writes(self):
        with self.assertRaisesRegex(ValueError, "externally supplied SHA"):
            self.migrate(expected="wrong")
        self.assertEqual(self.inspections, 0)
        self.assertFalse(self.output.exists())

    def test_changed_input_or_projection_buys_no_migration(self):
        for changed in ("wp_hash", "model", "existing_hash", "version", "world", "issues", "inventory"):
            with self.subTest(changed=changed):
                def inspect(root, checkpoint):
                    row = self.inspect(root, checkpoint)
                    if root != self.legacy_root:
                        if changed in row["binding"]:
                            row["binding"][changed] = "changed input"
                        else:
                            row[changed] = ["changed projection"]
                    return row
                with self.assertRaises(ValueError): self.migrate(inspect)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_wrong_legacy_environment_or_observed_parent_bytes_rejected(self):
        for wrong in ("binding", "checkpoint_sha256"):
            def inspect(root, checkpoint):
                row = self.inspect(root, checkpoint)
                if root == self.legacy_root:
                    row[wrong] = deepcopy(self.current_binding) if wrong == "binding" else "wrong"
                return row
            with self.assertRaises(ValueError): self.migrate(inspect)
            self.assertFalse(self.output.exists())

    def test_incomplete_parent_and_existing_destination_are_not_overwritten(self):
        changed = deepcopy(self.state); changed["status"] = "building"
        world_agent._save(self.parent, changed)
        with self.assertRaisesRegex(ValueError, "completed"):
            self.migrate(expected=hashlib.sha256(self.parent.read_bytes()).hexdigest())
        self.assertEqual(self.inspections, 0)
        self.parent.write_bytes(self.parent_bytes)
        self.output.mkdir(); marker = self.output / "preserved"
        marker.write_text("old evidence", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new destination"): self.migrate()
        self.assertEqual(marker.read_text(encoding="utf-8"), "old evidence")

    def test_legacy_source_tree_is_read_only(self):
        self.output = self.legacy_root / "child"
        with self.assertRaisesRegex(ValueError, "read-only"): self.migrate()
        self.assertEqual(self.inspections, 0)
        self.assertFalse(self.output.exists())

    def ordinary_parent(self):
        from seed_world_selftest import fixture
        from supply_pipeline_selftest import target
        from world_agent_selftest import ScriptedTracer as OrdinaryTracer, PLAN, WRITE, FINISH, write, table_part
        from pipeline import supply, supply_capacity
        self.parent = self.directory / "ordinary-parent.json"
        self.wp, _world, table = fixture()
        self.wp["line_mapping"] = [{"line": line, "applicable": True, "instantiation": "Synthetic carrier"}
            for line in ("L1_timeline", "L3_process", "L5_conflict", "L7_consolidation", "L8_transition")]
        supply.attach_plan(self.wp, target(["L1_timeline", "L3_process"], 4), 10, capacity_driven=True)
        trace = OrdinaryTracer([(PLAN, write("one")), (WRITE, table_part(table, [0, 1], True)),
                                (PLAN, FINISH), (PLAN, FINISH)])
        with self.assertRaises(supply_capacity.CapacityReviewRequested):
            world_agent.generate_world(self.wp, trace, checkpoint_path=self.parent, log=lambda *_: None)
        state = json.loads(self.parent.read_text(encoding="utf-8"))
        self.current_binding = deepcopy(state["binding"])
        state["binding"]["implementation"]["world_agent.py"] = "synthetic-legacy-source"
        world_agent._save(self.parent, state)
        self.state = json.loads(self.parent.read_text(encoding="utf-8"))
        self.parent_bytes = self.parent.read_bytes()

    def test_ordinary_supply_stop_inspects_migrates_and_restores_without_new_calls(self):
        self.ordinary_parent()
        current = migration.inspect(Path(__file__).resolve().parents[1], self.parent)
        self.assertEqual(current["issues"], [])
        self.assertEqual(current["inventory"], self.state["upstream_request"]["inventory"])
        self.assertEqual(current["provider_calls"], 0)

        def inspect(root, _checkpoint):
            row = deepcopy(current)
            if root == self.legacy_root:
                row["binding"] = deepcopy(self.state["binding"])
                row["source_hashes"] = {"synthetic": "legacy-inspection"}
            return row

        receipt = self.migrate(inspect)
        self.assertEqual(receipt["supply_inventory"], current["inventory"])
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertFalse(receipt["ledger_or_budget_changes"])
        child = json.loads((self.output / self.parent.name).read_text(encoding="utf-8"))
        trace = ScriptedTracer([])
        _table, resumed = world_agent.generate_world(self.wp, trace, resume_state=child,
            allow_supply_shortfall=True, log=lambda *_: None)
        self.assertEqual(trace.calls, [])
        self.assertEqual(resumed["status"], "completed")
        self.assertFalse(resumed["release_eligible"])
        self.assertGreater(resumed["supply_warning"]["shortfall"], 0)
        for key in ("units", "log", "steps", "retired_units", "upstream_request"):
            self.assertEqual(resumed[key], self.state[key], key)
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_ordinary_migration_rejects_other_stop_reasons_before_inspection(self):
        self.ordinary_parent()
        parent = deepcopy(self.state)
        for change in ({"status": "execution_error"}, {"upstream_request": {}},
                       {"feedback_revision_required": True}, {"version": "unknown"}):
            with self.subTest(change=change):
                world_agent._save(self.parent, {**deepcopy(parent), **change})
                with self.assertRaisesRegex(ValueError, "completed or supply-review"):
                    self.migrate(expected=hashlib.sha256(self.parent.read_bytes()).hexdigest())
                self.assertFalse(self.output.exists())
                self.assertEqual(self.inspections, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
