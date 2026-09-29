"""Offline v2 context wiring; fixed opinions make no quality claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("OPENAI_API_KEY", "offline-seed-v2-context")

from pipeline import corpus_contract as cc, disclosure, process_proposals, render, world_joint_plan, world_semantics
from pipeline.seed_pack import attach_seed_contract, generation_context, validate_seed_pack
from pipeline.seed_world import seed_business_context, seed_entity_prompt, seed_world_prompt, seed_world_report
from pipeline.semantic_review import visible_documents
from pipeline.world_state import WorldState
from seed_world_selftest import fixture as original_fixture
from public_disclosure_render_selftest import Trace
from eval.answer_task_review import POLICY_VERSION
import config


def fixture():
    wp, ws, _ = original_fixture()
    pack = deepcopy(wp.pop("seed_contract"))
    pack.pop("digest", None)
    pack.pop("contract_digest", None)
    pack["schema_version"] = 2
    pack["task"]["instructions"] = [pack["task"]["instructions"], "PRIVATE_BUSINESS_GUIDANCE"]
    pack["task"]["forbidden_inferences"] = ["Missing profit must remain unknown, never zero."]
    pack["mechanisms"][0]["failure_modes"] = ["Revision recorded before it actually happened."]
    pack["mechanisms"][0]["capability_hooks"] = ["L3_process"]
    field = pack["blueprint_requirements"]["entity_types"][1]["fields"][1]
    field["description"] = "Profit uses the stated reporting period, not the publication date."
    field["source_refs"] = [{"source_id": "fixture", "locator": "fixture()/profit"}]
    pack["blueprint_requirements"]["metric_definitions"] = [{
        "name": "profit", "entity_type": "report", "kind": "numeric", "unit": "万元",
        "definition": "Quarterly profit under the declared accounting scope.", "period_basis": "quarter",
        "comparators": ["same quarter of preceding year"], "source_refs": deepcopy(field["source_refs"])}]
    pack["blueprint_requirements"]["state_machines"] = [{
        "id": "report_review", "entity_type": "report", "field": "review",
        "states": ["pending", "checked"], "allowed_transitions": [["pending", "checked"]],
        "conditions": ["Review evidence is available."], "source_refs": deepcopy(field["source_refs"])}]
    pack["generation_contract"] = {
        "must_distinguish": ["Reporting period, occurrence date and acquisition date."],
        "must_not_infer": ["No implicit metric formulas or universal causal delays."],
    }
    pack["review"] = {"blocking_issues": [], "conflicts": ["AUDIT_ONLY_SENTINEL"]}
    pack["document_map"] = [{"source_id": "fixture", "segments": [
        {"locator": "fixture()", "topic": "DOCUMENT_MAP_SENTINEL"}]}]
    pack["sources"].append({"id": "eval_private", "path": "EVALUATOR_PATH_SENTINEL",
                            "sha256": "1" * 64, "role": "evaluator_only"})
    for source in pack["sources"]:
        source["source_kind"] = "factual_record"
    pack = validate_seed_pack(pack)
    wp = attach_seed_contract(wp, pack)
    wp["quality_contract"] = {"public_disclosure": True, "corpus_review": True,
        "world_semantic_review": True, "public_semantic_review": True,
        "isolated_reference_audit": True, "scoring_policy": POLICY_VERSION}
    task = {"description": "Offline seed v2 integration fixture."}
    rows = disclosure.catalogue(ws)
    raw = {"records": [{"session": ws.n_sessions - 1, "refs": [row["ref"] for row in rows],
        "channel": "Archive overview", "acquisition_context": "The final period reviews dated historical records."}],
        "undisclosed": [], "reason": "Offline context fixture only."}
    ws.disclosure = disclosure.compile_plan(wp, ws, raw, task)
    return wp, ws, task, pack


class SeedV2ContextTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        self.wp, self.ws, self.task, self.pack = fixture()

    def test_common_context_preserves_semantics_and_omits_audit_sources(self):
        context = seed_business_context(self.wp)
        full = generation_context(self.pack)
        for key in ("task", "mechanisms", "blueprint_requirements", "generation_contract", "sources"):
            self.assertEqual(context[key], full[key])
        text = json.dumps(context)
        self.assertIn("PRIVATE_BUSINESS_GUIDANCE", text)
        self.assertIn("allowed_transitions", text)
        self.assertIn("period_basis", text)
        self.assertIn("context_policy", context)
        for marker in ("AUDIT_ONLY_SENTINEL", "DOCUMENT_MAP_SENTINEL", "EVALUATOR_PATH_SENTINEL"):
            self.assertNotIn(marker, text)
        self.assertEqual(context["sources"][0]["source_kind"], "factual_record")
        self.assertTrue(all("path" not in source and "sha256" not in source
                            and source["role"] != "evaluator_only" for source in context["sources"]))

    def test_entity_and_structure_authors_receive_extended_definitions(self):
        entity = seed_entity_prompt(self.wp, "report")
        world = seed_world_prompt(self.wp)
        for text in (entity, world):
            self.assertIn("Profit uses the stated reporting period", text)
            self.assertIn("No implicit metric formulas", text)
            self.assertNotIn("AUDIT_ONLY_SENTINEL", text)
        self.assertTrue(seed_world_report(self.wp, self.ws)["passed"])

    def test_world_plan_reviewer_disclosure_and_process_share_context(self):
        expected = seed_business_context(self.wp)
        joint = world_joint_plan.make_context(self.wp, self.ws.world_blueprint, [])
        reviewed, _ = world_semantics._project(self.wp, self.ws, self.task)
        plan = disclosure._payload(self.wp, self.ws, self.task, None)
        proposed = process_proposals.prepare_process_proposals(self.wp, self.ws, target=1,
                                                               model="offline-mini")
        payload = json.loads(proposed["messages"][-1]["content"])
        for key in expected:
            self.assertEqual(joint[key], expected[key])
        self.assertEqual(reviewed["seed"], expected)
        self.assertEqual(plan["whitepaper"]["seed_requirements"], expected)
        self.assertEqual(payload["whitepaper"]["seed_business_context"], expected)
        for value in (joint, reviewed, plan, payload):
            text = json.dumps(value)
            for marker in ("AUDIT_ONLY_SENTINEL", "DOCUMENT_MAP_SENTINEL", "EVALUATOR_PATH_SENTINEL"):
                self.assertNotIn(marker, text)

    def test_business_guidance_does_not_become_a_public_fact_or_requirement(self):
        context = cc.canonical_context(self.ws, 0)
        self.assertEqual(context["facts"], [])
        self.assertEqual(context["events"], [])
        self.assertEqual(cc.fidelity_requirements(self.ws, 0), [])
        self.assertEqual(cc.public_stage_rules(self.ws), [])
        self.assertIn("PRIVATE_BUSINESS_GUIDANCE", json.dumps(context["business_interpretation"]))
        public = disclosure.context(self.ws, 0)
        self.assertNotIn("business_interpretation", public)
        self.assertNotIn("PRIVATE_BUSINESS_GUIDANCE", json.dumps(public))

    def test_render_review_receipt_roundtrip_and_public_reader_boundary(self):
        tracer, corpus = Trace(), {"sessions": []}
        with patch.object(config, "pmap", side_effect=lambda fn, items, workers=6: [fn(x) for x in items]):
            render.render_corpus(self.wp, self.ws, 0, tracer, corpus, set(), lambda: None, lambda *args: None)
        marker = "【生成与审阅共享的截至时点事实；不得把未知业务状态写成已发生】\n"
        author = next(messages[-1]["content"] for step, messages, _ in tracer.calls if step == "render.signal")
        context, _ = json.JSONDecoder().raw_decode(author.split(marker, 1)[1])
        reviewed = next(json.loads(messages[-1]["content"]) for step, messages, _ in tracer.calls if step == "corpus.review")
        self.assertEqual(context, reviewed["CANON"])
        self.assertEqual(context["business_interpretation"]["definitions"], seed_business_context(self.wp))
        self.assertEqual(cc.validate_corpus(self.ws, corpus)["status"], "passed")
        restored = WorldState.from_dict(self.ws.to_dict())
        self.assertEqual(cc.validate_corpus(restored, corpus)["status"], "passed")
        blind = [messages for step, messages, _ in tracer.calls if step == "render.discriminate"]
        public, _ = visible_documents(corpus)
        self.assertNotIn("PRIVATE_BUSINESS_GUIDANCE", json.dumps(blind))
        self.assertNotIn("PRIVATE_BUSINESS_GUIDANCE", json.dumps(public))
        self.assertNotIn("AUDIT_ONLY_SENTINEL", json.dumps(tracer.calls))

    def test_tampered_frozen_seed_invalidates_context_and_old_receipts(self):
        self.ws.disclosure["source_inputs"]["whitepaper"]["seed_contract"]["task"]["objective"] = "changed"
        with self.assertRaises(ValueError):
            cc.canonical_context(self.ws, 0)
        self.assertEqual(cc.validate_corpus(self.ws, {"sessions": []})["status"], "failed")

    def test_reviewer_cannot_receive_unbound_business_context(self):
        context = cc.canonical_context(self.ws, self.ws.n_sessions - 1)
        context["business_interpretation"]["definitions"]["task"]["objective"] = "silently changed"
        tracer = Trace()
        report = cc.review_documents(tracer, self.ws, self.ws.n_sessions - 1,
            [{"title": "Offline document", "content": "A fixed original body."}], context=context)
        self.assertEqual(report["status"], "error")
        self.assertIn("differs from the frozen public disclosure projection", report["issues"][0]["message"])
        self.assertEqual(tracer.calls, [])

    def test_v1_context_has_unchanged_shape_and_no_business_interpretation(self):
        wp, ws, _ = original_fixture()
        contract = wp["seed_contract"]
        expected = {"task": {key: contract["task"][key] for key in ("objective", "instructions")},
            "mechanisms": [{key: item[key] for key in ("id", "description", "required_entity_types",
                "required_relation_types", "required_event_types", "required_causal_rules")}
                for item in contract["mechanisms"]]}
        self.assertEqual(seed_business_context(wp), expected)
        self.assertNotIn("business_interpretation", cc.canonical_context(ws, 0))


if __name__ == "__main__":
    unittest.main()
