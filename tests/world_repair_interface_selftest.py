"""Original world draft/repair integration, with network and providers disabled."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_gen import build_world, _world_digest, _RepairTracer
from pipeline.seed_world import validate_seed_world
from seed_world_selftest import fixture, FixtureTracer
from world_structure_context_selftest import TableTracer, section


class Script:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": params})
        if not self.outputs:
            raise AssertionError("Unexpected extra author call")
        response = self.outputs.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class WorldRepairInterfaceTests(unittest.TestCase):
    def setUp(self):
        for target, key in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(target, key, side_effect=AssertionError("Network/API prohibited"))
            guard.start()
            self.addCleanup(guard.stop)
        self.wp, _, self.table = fixture()
        self.draft = {}
        self.world = build_world(self.wp, FixtureTracer(self.table), log=lambda *_: None,
                                 draft_out=self.draft)

    def proposal(self, **updates):
        value = {key: deepcopy(self.table[key]) for key in ("relations", "events")}
        value.update(updates)
        return value

    def repair(self, outputs, *, intrinsic=(), structure=False, max_calls=4,
               draft=None, wp=None, existing=None):
        tracer, result = Script(outputs), {}
        kwargs = {"draft": deepcopy(self.draft if draft is None else draft),
                  "feedback": {"issues": [{"id": "i1", "description": "Fallible business reading"}]},
                  "targets": {"intrinsic": list(intrinsic), "structure": structure},
                  "max_calls": max_calls}
        world = build_world(self.wp if wp is None else wp, tracer, log=lambda *_: None,
                            draft_out=result, repair_input=kwargs, existing=existing)
        return world, result, tracer

    def test_default_and_captured_world_are_identical_without_extra_calls(self):
        tracer = FixtureTracer(self.table)
        ordinary = build_world(self.wp, tracer, log=lambda *_: None)
        self.assertEqual(ordinary.to_dict(), self.world.to_dict())
        self.assertEqual(self.draft["candidate_world"], self.world.to_dict())
        self.assertEqual(self.draft["merged"], {**self.table, "cascades": [], "absent_fields": []})
        self.assertEqual(len(tracer.calls), 4)  # One shared original-author plan.
        self.assertEqual(self.draft["repair_log"], [])
        self.assertEqual(self.draft["status"], "completed")

    def test_capture_uses_mechanically_repaired_candidate_not_first_structure(self):
        output = {}
        build_world(self.wp, FixtureTracer(self.table, bad_first=True), log=lambda *_: None,
                    draft_out=output)
        self.assertEqual(len(output["raw_merged"]["events"]), 2)
        self.assertEqual(len(output["merged"]["events"]), 3)
        world, _, tracer = self.repair([self.proposal()], structure=True, draft=output)
        self.assertEqual(world.to_dict(), output["candidate_world"])
        self.assertEqual(len(tracer.calls), 1)

    def test_intrinsic_author_repairs_only_target_and_recompiles(self):
        answer = {"fields": {"capital": {"type": "stable", "value": "125"}},
                  "issue_responses": [{"issue_id": "i1", "disposition": "addressed",
                                       "response": "The original task allows this correction."}]}
        world, draft, tracer = self.repair([answer], intrinsic=[{"entity": "Yearbook", "fields": ["capital"]}])
        self.assertEqual(world.timeline("Yearbook", "capital").latest_valid(), "125")
        self.assertEqual(world.events, self.world.events)
        self.assertEqual(world.relations, self.world.relations)
        self.assertTrue(validate_seed_world(self.wp, world)["passed"])
        self.assertEqual([c["step"] for c in tracer.calls], ["world.repair"])
        self.assertNotIn("峰【或】谷", tracer.calls[0]["messages"][0]["content"])
        self.assertEqual(draft["repair_log"][0]["issue_responses"], answer["issue_responses"])
        self.assertEqual(draft["repair_log"][0]["actual_changes"][0]["field"], "capital")
        self.assertEqual(draft["candidate_world"], world.to_dict())

    def test_structure_repair_recompiles_event_effect_and_keeps_intrinsic_raw(self):
        answer = self.proposal()
        answer["events"][-1]["effects"][0]["set"] = "review complete"
        world, draft, tracer = self.repair([answer], structure=True)
        self.assertEqual(world.timeline("Yearbook", "review").latest_valid(), "review complete")
        self.assertEqual(draft["merged"]["entities"], self.draft["merged"]["entities"])
        self.assertEqual([c["step"] for c in tracer.calls], ["world.structure"])
        self.assertEqual(draft["repair_log"][0]["actual_changes"][0]["structure"], "events")

    def test_intrinsic_then_structure_share_changed_input_and_budget(self):
        world, _, tracer = self.repair([
            {"fields": {"capital": {"type": "stable", "value": "125"}}}, self.proposal()],
            intrinsic=[{"entity": "Yearbook", "fields": ["capital"]}], structure=True, max_calls=2)
        self.assertEqual([c["step"] for c in tracer.calls], ["world.repair", "world.structure"])
        self.assertTrue(all(c["params"]["retries"] == 1 and c["params"]["strict_json"]
                            for c in tracer.calls))
        catalog = section(tracer.calls[1]["messages"][1]["content"], "【typed entities】")
        self.assertEqual(next(e for e in catalog if e["name"] == "Yearbook")["intrinsic_fields"]["capital"],
                         {"type": "stable", "value": "125"})
        self.assertEqual(world.timeline("Yearbook", "capital").latest_valid(), "125")

    def test_disputed_feedback_can_preserve_all_original_facts(self):
        response = {"issue_id": "i1", "disposition": "disputed", "response": "The task explicitly permits this."}
        world, draft, _ = self.repair([{"fields": {}, "issue_responses": [response]}],
                                      intrinsic=[{"entity": "Yearbook", "fields": ["capital"]}])
        self.assertEqual(world.to_dict(), self.world.to_dict())
        self.assertEqual(draft["repair_log"][0]["issue_responses"], [response])
        self.assertEqual(draft["repair_log"][0]["actual_changes"], [])
        self.assertNotIn("semantic_passed", draft)

    def test_structure_dispute_can_keep_original_structure(self):
        output = self.proposal(issue_responses=[{"issue_id": "i1", "disposition": "deferred", "response": "Unresolved."}])
        world, draft, _ = self.repair([output], structure=True)
        self.assertEqual(world.to_dict(), self.world.to_dict())
        self.assertEqual(draft["repair_log"][0]["actual_changes"], [])

    def assert_rejected(self, *, edit=None, targets=None, outputs=(), max_calls=4, wp=None, draft=None):
        source = deepcopy(self.draft if draft is None else draft)
        if edit:
            edit(source)
        tracer, result = Script(outputs), {}
        with self.assertRaises((WorldBlueprintError, OSError)):
            build_world(self.wp if wp is None else wp, tracer, log=lambda *_: None, draft_out=result,
                        repair_input={"draft": source, "feedback": {"issues": ["i1"]},
                                      "targets": targets or {"intrinsic": [], "structure": True},
                                      "max_calls": max_calls})
        self.assertEqual(result["status"], "error")
        return tracer, result

    def test_stale_draft_whitepaper_and_candidate_all_rejected_before_dispatch(self):
        tracer, _ = self.assert_rejected(edit=lambda d: d["merged"]["entities"][0].update(name="Other"))
        self.assertEqual(tracer.calls, [])
        wp = deepcopy(self.wp)
        wp["description"] = "Different author task"
        tracer, _ = self.assert_rejected(wp=wp)
        self.assertEqual(tracer.calls, [])
        tracer, _ = self.assert_rejected(edit=lambda d: d["candidate_world"].update(n_sessions=99))
        self.assertEqual(tracer.calls, [])

    def test_rehashed_raw_tamper_does_not_match_recompiled_candidate(self):
        def edit(d):
            d["merged"]["entities"][1]["fields"]["capital"] = {"type": "stable", "value": "125"}
            d["draft_hash"] = _world_digest({k: v for k, v in d.items() if k != "draft_hash"})
        tracer, result = self.assert_rejected(edit=edit)
        self.assertEqual(tracer.calls, [])
        self.assertIn("reproduce", result["error"])

    def test_target_and_returned_field_scope_are_enforced(self):
        for field in ("profit", "issuer", "unlisted"):
            tracer, _ = self.assert_rejected(targets={"intrinsic": [{"entity": "Yearbook", "fields": [field]}], "structure": False})
            self.assertEqual(tracer.calls, [])
        tracer, result = self.assert_rejected(
            targets={"intrinsic": [{"entity": "Yearbook", "fields": ["capital"]}], "structure": False},
            outputs=[{"fields": {"profit": {"type": "stable", "value": "900"}}}])
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(result["repair_log"][0]["raw_output"]["fields"]["profit"]["value"], "900")
        self.assertEqual(result["merged"], self.draft["merged"])

    def test_one_budget_covers_semantic_and_original_mechanical_structure_calls(self):
        bad = self.proposal()
        bad["events"] = bad["events"][:2]
        tracer, result = self.assert_rejected(
            targets={"intrinsic": [{"entity": "Yearbook", "fields": ["capital"]}], "structure": True},
            outputs=[{"fields": {}}, bad], max_calls=2)
        self.assertEqual([c["step"] for c in tracer.calls], ["world.repair", "world.structure"])
        self.assertEqual(len(result["repair_log"]), 2)
        self.assertIn("budget exhausted", result["error"])

    def test_zero_budget_and_provider_error_stop_without_hidden_fallback(self):
        tracer, result = self.assert_rejected(max_calls=0)
        self.assertEqual(tracer.calls, [])
        self.assertEqual(result["repair_log"], [])
        for output in ({"__error__": "transport failed"}, OSError("transport failed")):
            tracer, result = self.assert_rejected(outputs=[output])
            self.assertEqual(len(tracer.calls), 1)
            self.assertEqual(result["repair_log"][0]["status"], "error")

    def augment_draft(self):
        wp, existing, table = fixture()
        wp["world_blueprint"]["entity_types"][0]["count"] = 2
        table["entities"] = [{"name": "Beryl", "type": "company", "fields": {
            "sector": {"type": "stable", "value": "insurance"}}}]
        draft = {}
        build_world(wp, TableTracer(table), existing=existing, log=lambda *_: None, draft_out=draft)
        return wp, draft

    def test_incremental_old_intrinsic_entities_rejected_without_calls(self):
        wp, draft = self.augment_draft()
        tracer, _ = self.assert_rejected(wp=wp, draft=draft, targets={
            "intrinsic": [{"entity": "Yearbook", "fields": ["capital"]}], "structure": False})
        self.assertEqual(tracer.calls, [])

    def test_incremental_changed_published_event_is_not_silently_filtered(self):
        wp, draft = self.augment_draft()
        changed = self.proposal()
        changed["events"][0]["effects"][0]["set"] = "999"
        tracer, result = self.assert_rejected(wp=wp, draft=draft, outputs=[changed])
        self.assertEqual(len(tracer.calls), 1)
        self.assertIn("published", result["error"])
        self.assertEqual(result["existing_base"], draft["existing_base"])

    def test_incremental_delta_repair_retains_all_old_canonical_operations(self):
        wp, draft = self.augment_draft()
        world, result, _ = self.repair([{"fields": {"sector": {"type": "stable", "value": "mutual"}}}],
                                      wp=wp, draft=draft,
                                      intrinsic=[{"entity": "Beryl", "fields": ["sector"]}])
        for name, fields in draft["existing_base"]["entities"].items():
            self.assertEqual(world.to_dict()["entities"][name], fields)
        self.assertEqual(result["existing_base"], draft["existing_base"])
        self.assertEqual(world.timeline("Beryl", "sector").latest_valid(), "mutual")

    def test_capture_existing_noop_is_explicit_and_cannot_rewrite_old_canon(self):
        draft, tracer = {}, Script([])
        build_world(self.wp, tracer, existing=deepcopy(self.world), log=lambda *_: None, draft_out=draft)
        self.assertEqual(tracer.calls, [])
        self.assertEqual(draft["mode"], "existing_noop")
        tracer, _ = self.assert_rejected(draft=draft)
        self.assertEqual(tracer.calls, [])

    def test_provider_failure_stops_new_reservations_but_keeps_inflight_return(self):
        first_started, second_started, release_second = (threading.Event() for _ in range(3))
        class ParallelProvider:
            def chat_json(self, step, messages, **params):
                if step == "first":
                    first_started.set()
                    if not second_started.wait(3):
                        raise AssertionError("Second request did not enter")
                    raise OSError("first failed")
                if step == "second":
                    second_started.set()
                    if not release_second.wait(3):
                        raise AssertionError("In-flight request not released")
                    return {"fields": {}}
                raise AssertionError("Late provider request should not enter")
        state = {"repair_log": []}
        guarded = _RepairTracer(ParallelProvider(), state, 4)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(guarded.chat_json, "first", [])
            self.assertTrue(first_started.wait(3))
            second = pool.submit(guarded.chat_json, "second", [])
            with self.assertRaises(OSError):
                first.result(timeout=3)
            with self.assertRaises(WorldBlueprintError):
                guarded.chat_json("late", [])
            release_second.set()
            self.assertEqual(second.result(timeout=3), {"fields": {}})
        self.assertEqual(len(state["repair_log"]), 2)
        self.assertEqual([entry["status"] for entry in state["repair_log"]], ["error", "returned"])
        self.assertEqual(state["repair_log"][1]["raw_output"], {"fields": {}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
