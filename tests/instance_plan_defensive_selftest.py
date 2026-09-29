"""Malformed records keep concrete compiler findings and original author identity."""
import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENAI_API_KEY"] = "offline-compiler-dummy-key"
os.environ["MODEL"] = "offline-compiler-model"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
import socket
import sys
import types
from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch

def forbidden(*args, **kwargs):
    raise AssertionError("Network/provider forbidden")

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
dotenv.find_dotenv = lambda *args, **kwargs: ""
sys.modules["dotenv"] = dotenv
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import openai
openai.OpenAI = forbidden
openai.AsyncOpenAI = forbidden
from pipeline.instance_plan import PlanConflict, apply_repair, compile_plan
from pipeline.plan_transactions import _route_findings
from instance_plan_selftest import setup_plan


class DefensiveCompilerTests(unittest.TestCase):
    def setUp(self):
        self.wp, self.plan, *_ = setup_plan()
        self.uid = self.plan["units"][0]["unit_id"]

    def findings(self, proposal):
        before = deepcopy(proposal)
        with self.assertRaises(PlanConflict) as caught:
            compile_plan(self.wp, proposal)
        self.assertEqual(proposal, before)
        self.assertNotIn("plan_shape", {row["code"] for row in caught.exception.findings})
        return caught.exception.findings

    def test_missing_required_fields_retain_unit_and_available_record_id(self):
        required = {
            "objects": ("slot", "type"),
            "events": ("slot", "type", "participants", "session"),
            "relations": ("slot", "type", "from", "to", "session"),
            "observations": ("entity", "field", "sessions"),
            "obligations": ("id", "line", "carrier"),
            "publications": ("fact_slot", "channel", "session"),
        }
        for collection, fields in required.items():
            for field in fields:
                with self.subTest(collection=collection, missing=field):
                    plan = deepcopy(self.plan)
                    row = plan["units"][0][collection][0]
                    if collection == "relations" and field == "session":
                        # Only temporal bindings require an authored session.
                        spec = next(s for s in self.wp["world_blueprint"]["relation_types"]
                                    if s["id"] == row["type"])
                        spec["temporal"] = True
                    row.pop(field)
                    found = self.findings(plan)
                    shapes = [f for f in found if f["code"] == "record_shape"
                              and f["evidence"]["collection"] == collection]
                    self.assertTrue(shapes)
                    self.assertIn(field, shapes[0]["evidence"]["missing_fields"])
                    self.assertIn(self.uid, shapes[0]["affected_ids"])
                    if row.get("slot", row.get("id")):
                        self.assertIn(row.get("slot", row.get("id")), shapes[0]["affected_ids"])
                    if collection == "relations" and field == "session":
                        spec["temporal"] = False

    def test_malformed_nested_fields_do_not_escape_as_unowned_exception(self):
        cases = [
            ("events", "participants", {"report": ["report-slot"]}),
            ("events", "session", True),
            ("relations", "from", ["report-slot"]),
            ("observations", "sessions", [[0]]),
            ("obligations", "carrier", {"entities": [["report-slot"]]}),
        ]
        for collection, field, value in cases:
            with self.subTest(collection=collection, field=field):
                plan = deepcopy(self.plan)
                plan["units"][0][collection][0][field] = value
                found = self.findings(plan)
                self.assertTrue(any(f["code"] == "record_shape" and self.uid in f["affected_ids"] for f in found))
        plan = deepcopy(self.plan)
        plan["units"][0]["depends_on"] = [["other-unit"]]
        self.assertIn("record_shape", {f["code"] for f in self.findings(plan)})

    def test_partial_event_replace_rejected_without_filling_original_fields(self):
        old_event = self.plan["units"][0]["events"][0]
        reply = {"decision": "repair", "changes": [{"unit_id": self.uid,
            "collection": "events", "operation": "replace", "key": old_event["slot"],
            "value": {"participants": deepcopy(old_event["participants"])}}]}
        partial = apply_repair(self.plan, reply)
        event = partial["units"][0]["events"][0]
        self.assertNotIn("type", event)
        self.assertNotIn("session", event)
        found = self.findings(partial)
        shape = next(f for f in found if f["code"] == "record_shape")
        self.assertEqual(set(shape["evidence"]["missing_fields"]), {"type", "session"})
        self.assertIn(old_event["slot"], shape["affected_ids"])
        routes = _route_findings(self.wp, partial, found)[6]
        self.assertEqual([uid for uid, fs in routes.items() if fs], [self.uid])

    def test_unknown_types_keep_original_semantic_codes_and_owners(self):
        for collection, code in (("objects", "unknown_entity_type"),
                                 ("events", "unknown_event"), ("relations", "unknown_relation")):
            with self.subTest(collection=collection):
                plan = deepcopy(self.plan)
                row = plan["units"][0][collection][0]
                row["type"] = "not-declared"
                matches = [f for f in self.findings(plan) if f["code"] == code]
                self.assertTrue(matches)
                self.assertIn(row["slot"], matches[0]["affected_ids"])
                self.assertIn(self.uid, matches[0]["affected_ids"])

    def test_nonrecord_values_keep_collection_and_owner(self):
        for collection in ("objects", "events", "relations", "observations", "obligations", "publications"):
            with self.subTest(collection=collection):
                plan = deepcopy(self.plan)
                plan["units"][0][collection][0] = []
                shapes = [f for f in self.findings(plan) if f["code"] == "record_shape"]
                self.assertEqual(shapes[0]["evidence"]["collection"], collection)
                self.assertEqual(shapes[0]["affected_ids"], [self.uid])

    def test_static_relation_with_later_session_cannot_change_target(self):
        plan = deepcopy(self.plan)
        unit = plan["units"][0]
        original = deepcopy(unit["relations"][0])
        self.assertFalse(next(row for row in self.wp["world_blueprint"]["relation_types"]
                              if row["id"] == original["type"]).get("temporal"))
        unit["objects"].append({"slot": "other-company", "type": "company"})
        unit["relations"].append(dict(original, slot="new-static-target", to="other-company", session=4))
        found = self.findings(plan)
        conflict = next(f for f in found if f["code"] == "scalar_binding_conflict")
        self.assertIn("new-static-target", conflict["affected_ids"])
        self.assertIn(self.uid, conflict["affected_ids"])

    def test_static_relation_omitted_session_retains_original_lowering(self):
        plan = deepcopy(self.plan)
        original = plan["units"][0]["relations"][0]
        self.assertFalse(next(row for row in self.wp["world_blueprint"]["relation_types"]
                              if row["id"] == original["type"]).get("temporal"))
        original.pop("session")
        before = deepcopy(plan)
        compiled = compile_plan(self.wp, plan)
        self.assertIn("business_instance_plan", compiled)
        self.assertEqual(plan, before)

    def test_valid_plan_still_compiles_without_mutating_input(self):
        before = deepcopy(self.plan)
        result = compile_plan(self.wp, self.plan)
        self.assertIn("business_instance_plan", result)
        self.assertEqual(self.plan, before)


if __name__ == "__main__":
    unittest.main()
