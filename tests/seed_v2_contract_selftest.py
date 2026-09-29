"""Offline v2 interface and v1 compatibility checks; no API or source reads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_contract_selftest import binding_fixture, blueprint
from pipeline.seed_pack import (SeedPackError, attach_seed_contract, core_requirements,
    generation_context, seed_context, seed_contract, seed_digest, seed_input,
    validate_seed_blueprint, validate_seed_pack)


def sample_v2():
    pack = binding_fixture()
    pack["schema_version"] = 2
    pack["seed_id"] = "test_report_v2"
    for source in pack["sources"]:
        source["source_kind"] = "task_specification" if source["id"] == "design" else "factual_record"
    pack["sources"][1].update(date_scope={"reporting_period": "2025Q4", "published": "2026-01-01"},
                              allowed_use=["mechanism", "exemplar_style"], sensitivity="public")
    pack["task"]["instructions"] = [pack["task"]["instructions"]]
    pack["task"]["forbidden_inferences"] = ["缺失值不得自动记零"]
    ref = [{"source_id": "design", "locator": "修订程序"}]
    field = pack["blueprint_requirements"]["entity_types"][0]["fields"][0]
    field.update(allowed_values=["初版", "修订版"], evolving=True, source_refs=deepcopy(ref))
    pack["blueprint_requirements"]["state_machines"] = [{
        "id": "version_progression", "entity_type": "report", "field": "version",
        "states": ["初版", "修订版"], "allowed_transitions": [["初版", "修订版"]],
        "source_refs": deepcopy(ref), "conditions": ["正式修订已到达"],
    }]
    pack["blueprint_requirements"]["metric_definitions"] = [{
        "name": "amount", "entity_type": "metric", "kind": "numeric", "unit": "万元",
        "definition": "报告期内采用的测算值", "period_basis": "季度", "comparators": ["上年同期"],
        "source_refs": deepcopy(ref),
    }]
    pack["blueprint_requirements"]["causal_rules"][0].update(
        basis="间隔一个 session 为合成设计", status="synthetic_design", source_refs=deepcopy(ref))
    pack["mechanisms"][0].update(failure_modes=["沿用旧版测算"], capability_hooks=["L3_process"])
    pack["generation_contract"] = {
        "must_distinguish": ["报告期与发布日期"],
        "conditional_requirements": [{"text": "修订到达后重算", "status": "conditional",
                                       "conditions": ["正式修订到达"], "source_refs": deepcopy(ref)}],
        "unresolved": [{"text": "是否存在额外审批", "status": "unresolved", "source_refs": deepcopy(ref)}],
        "synthetic_designs": [{"text": "使用合成机构名称", "status": "synthetic_design", "source_refs": []}],
        "syntheticization": {"names": "replace", "dates": "shift_or_generate"},
    }
    pack["document_map"] = [{"source_id": "judge", "segments": [{"locator": "sheet1", "topic": "EVAL_AUDIT_SECRET"}]}]
    pack["review"] = {"blocking_issues": [], "quality_status": "draft", "notes": "AUDIT_ONLY_SECRET"}
    return pack


class SeedV2ContractTests(unittest.TestCase):
    def test_v1_output_unchanged(self):
        pack = binding_fixture()
        self.assertEqual(core_requirements(pack), pack["blueprint_requirements"])
        self.assertEqual(generation_context(pack), seed_contract(pack))
        self.assertEqual(seed_context(pack), json.dumps(seed_contract(pack), ensure_ascii=False, indent=2, allow_nan=False))

    def test_v2_lossless_archive_validation(self):
        pack = sample_v2()
        self.assertEqual(validate_seed_pack(pack), pack)

    def test_raw_v2_requires_explicit_audit_and_contract(self):
        for key in ("document_map", "review", "generation_contract"):
            pack = sample_v2()
            del pack[key]
            with self.subTest(key=key), self.assertRaises(SeedPackError):
                validate_seed_pack(pack)

    def test_core_projection_does_not_compile_metadata(self):
        pack = sample_v2()
        actual = core_requirements(pack)
        self.assertEqual(actual, binding_fixture()["blueprint_requirements"])
        self.assertNotIn("states", actual["entity_types"][0]["fields"][0])

    def test_metadata_survives_context_without_audit(self):
        context = generation_context(sample_v2())
        text = json.dumps(context, ensure_ascii=False)
        self.assertIn("allowed_transitions", text)
        self.assertIn("period_basis", text)
        self.assertIn("synthetic_design", text)
        for secret in ("AUDIT_ONLY_SECRET", "EVAL_AUDIT_SECRET", "DO_NOT_EXPOSE", "document_map", '"review"'):
            self.assertNotIn(secret, text)
        self.assertEqual({s["id"] for s in context["sources"]}, {"design", "public"})

    def test_contract_identity_and_roundtrip(self):
        pack = sample_v2()
        contract = seed_contract(pack)
        self.assertEqual(seed_digest(contract), seed_digest(pack))
        self.assertEqual(seed_contract(contract), contract)
        self.assertEqual(validate_seed_pack(contract), contract)
        changed = deepcopy(pack)
        changed["review"]["notes"] = "different audit"
        self.assertNotEqual(seed_digest(changed), seed_digest(pack))

    def test_instruction_array_adapts_to_original_input(self):
        desc, examples = seed_input(sample_v2())
        self.assertIn("依据正式修订重算指标", desc)
        self.assertNotIn("['", desc)
        self.assertEqual(examples[0]["seed_origin"], "synthetic")

    def test_blocking_review_can_archive_but_not_generate(self):
        pack = sample_v2()
        pack["review"]["blocking_issues"] = ["核心资料缺页"]
        pack["review"]["quality_status"] = "ready_for_whitepaper"
        validate_seed_pack(pack)
        for func in (seed_input, generation_context, seed_context, seed_contract):
            with self.subTest(func=func.__name__), self.assertRaises(SeedPackError):
                func(pack)

    def test_evaluator_derivation_rejected(self):
        for location in ("mechanism", "field", "contract", "task"):
            pack = sample_v2()
            ref = [{"source_id": "judge", "locator": "secret"}]
            target = {"mechanism": pack["mechanisms"][0],
                      "field": pack["blueprint_requirements"]["entity_types"][0]["fields"][0],
                      "contract": pack["generation_contract"]["unresolved"][0], "task": pack["task"]}[location]
            target["source_refs"] = ref
            with self.subTest(location=location), self.assertRaises(SeedPackError):
                validate_seed_pack(pack)

    def test_uncertain_core_can_archive_but_cannot_be_executed(self):
        for location in ("entity_types", "relation_types", "event_types", "causal_rules", "field", "mechanism"):
            pack = sample_v2()
            if location == "field":
                target = pack["blueprint_requirements"]["entity_types"][0]["fields"][0]
            elif location == "mechanism":
                target = pack["mechanisms"][0]
            else:
                target = pack["blueprint_requirements"][location][0]
            target["status"] = "unresolved"
            validate_seed_pack(pack)
            for func in (core_requirements, generation_context, seed_input, seed_contract):
                with self.subTest(location=location, function=func.__name__), self.assertRaises(SeedPackError):
                    func(pack)

    def test_missing_minimum_never_invented(self):
        pack = sample_v2()
        del pack["blueprint_requirements"]["event_types"][0]["min_count"]
        with self.assertRaises(SeedPackError):
            validate_seed_pack(pack)

    def test_unknown_semantic_key_rejected(self):
        pack = sample_v2()
        pack["blueprint_requirements"]["state_machines"][0]["transition_predicate"] = "unknown"
        with self.assertRaises(SeedPackError):
            validate_seed_pack(pack)

    def test_conditional_claim_requires_condition(self):
        pack = sample_v2()
        del pack["generation_contract"]["conditional_requirements"][0]["conditions"]
        with self.assertRaises(SeedPackError):
            validate_seed_pack(pack)

    def test_invalid_transition_rejected(self):
        pack = sample_v2()
        pack["blueprint_requirements"]["state_machines"][0]["allowed_transitions"] = [["初版", "unknown"]]
        with self.assertRaises(SeedPackError):
            validate_seed_pack(pack)

    def test_malformed_role_and_identity_return_validation_error(self):
        for key in ("role", "id"):
            pack = sample_v2()
            target = pack["sources"][0] if key == "role" else pack["blueprint_requirements"]["entity_types"][0]
            target[key] = []
            with self.subTest(key=key), self.assertRaises(SeedPackError):
                validate_seed_pack(pack)

    def test_original_blueprint_checks_only_executable_projection(self):
        pack = sample_v2()
        bp = blueprint(binding_fixture())
        self.assertTrue(validate_seed_blueprint(bp, pack)["passed"])
        self.assertEqual(attach_seed_contract({"world_blueprint": bp}, pack)["seed_contract"], seed_contract(pack))
        bp["causal_rules"][0]["shared_roles"] = []
        self.assertFalse(validate_seed_blueprint(bp, pack)["passed"])

    def test_full_schema_accepts_fixture_and_contract(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/seed_pack_v2.schema.json").read_text(encoding="utf8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        for pack in (sample_v2(), seed_contract(sample_v2())):
            jsonschema.validate(pack, schema)


if __name__ == "__main__":
    unittest.main()
