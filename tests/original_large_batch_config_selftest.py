"""Offline regressions for required seed fields and economical large runs."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from pipeline.central_office import central_office, _remove_inferred_identity_fields
from pipeline.seed_pack import core_requirements, validate_seed_pack
from pipeline.run import Tracer
from pipeline import disclosure
from disclosure_selftest import data, raw_plan, SequenceTracer
import original_smoke_corpus_review_override_selftest as wrapper_fixture
import original_batch_selftest as batch_fixture
from seed_contract_selftest import fixture as synthetic_seed


class RequiredSeedFieldTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def test_required_seed_field_survives_original_council_without_observed_field(self):
        pack = synthetic_seed()
        metric = next(item for item in pack["blueprint_requirements"]["entity_types"] if item["id"] == "metric")
        metric["fields"].append({"name": "指标名称", "kind": "category"})
        pack = validate_seed_pack(pack)
        blueprint = core_requirements(pack)
        blueprint["version"] = 1
        blueprint["temporal_model"] = {"unit": "week", "n_sessions": 4, "cadence": "weekly", "step_days": 7}
        calls = []
        def answer(step, messages, **kw):
            calls.append(step)
            if step == "council.world":
                return {"world_blueprint": deepcopy(blueprint)}
            if step == "council.blueprint_feasibility":
                payload = json.loads(messages[-1]["content"])
                return {"decision": "accept", "reason": "Offline fixture for field preservation",
                        "issues": [], "mechanism_checks": [{"mechanism_id": m["id"], "status": "feasible",
                            "walkthrough": "Offline control-flow fixture, no semantic claim"}
                            for m in payload["business"].get("mechanisms", [])]}
            if step == "council.map":
                return {"per_line": [{"line": lid, "applicable": lid == "L1_timeline",
                    "gt_feasible": lid == "L1_timeline", "instantiation": "Read dated records",
                    "weight_hint": 1 if lid == "L1_timeline" else 0}
                    for lid in ("L1_timeline", "L2_relational", "L3_process", "L4_preference",
                                "L5_conflict", "L6_refusal", "L7_consolidation")]}
            if step in {"council.observe", "council.skeptic", "council.medium", "council.style", "council.traps"}:
                return {}
            raise AssertionError("Unexpected repair or provider call: " + step)
        wp = central_office("Offline synthetic field regression", [], SimpleNamespace(chat_json=answer),
                            log=lambda *a: None, seed_pack=pack)
        observed = next(e for e in wp["world_blueprint"]["entity_types"] if e["id"] == "metric")
        self.assertIn("指标名称", [f["name"] for f in observed["fields"]])
        self.assertEqual(calls.count("council.world"), 1)

    def test_protection_is_typed_and_never_invents_missing_field(self):
        candidate = {"entity_types": [
            {"id": "observation", "fields": [{"name": "指标名称", "kind": "category"}]},
            {"id": "unrelated", "fields": [{"name": "指标名称", "kind": "category"}]}]}
        required = {("observation", "指标名称"), ("observation", "合同编号")}
        _remove_inferred_identity_fields(candidate, {}, seed_required_fields=required)
        self.assertEqual(candidate["entity_types"][0]["fields"], [{"name": "指标名称", "kind": "category"}])
        self.assertEqual(candidate["entity_types"][1]["fields"], [])

    def test_multiple_disclosure_patches_use_merged_proposal_and_keep_evidence(self):
        wp, ws, task = data()
        good = raw_plan(ws)
        refs = good["records"][0]["refs"]
        self.assertGreaterEqual(len(refs), 2)
        initial = deepcopy(good); initial["records"] = []
        record1 = {**deepcopy(good["records"][0]), "refs": refs[:1]}
        record2 = {**deepcopy(good["records"][0]), "refs": refs[1:]}
        outputs = [initial,
            {"record_updates": [], "append_records": [record1], "undisclosed": None, "reason": "First partial repair"},
            {"record_updates": [], "append_records": [record2], "undisclosed": None, "reason": "Remaining references"}]
        tracer = SequenceTracer(outputs)
        cfg = SimpleNamespace(STRUCTURE_MODEL="offline", DISCLOSURE_FORMAT_ATTEMPTS=4)
        with patch.dict(sys.modules, {"config": cfg}):
            result = disclosure.author_plan(wp, ws, tracer, task)
        self.assertEqual(result["status"], "ready", result.get("error"))
        self.assertEqual(result["logical_calls"], 3)
        last_input = json.loads(tracer.calls[-1]["messages"][1]["content"])
        self.assertEqual(last_input["format_repair"]["previous_output"]["records"], [record1])
        self.assertFalse(disclosure.validate_plan(ws))


class LargeWrapperTests(unittest.TestCase):
    setUp = wrapper_fixture.CorpusReasoningOverrideTests.setUp
    run_tool = wrapper_fixture.CorpusReasoningOverrideTests.run_tool

    def test_glm_wire_uses_large_output_timeout_and_every_role_same_model(self):
        def calls(directory, cfg):
            self.assertEqual({cfg.MODEL, cfg.STRUCTURE_MODEL, cfg.DISCRIMINATOR_MODEL,
                              cfg.REVIEWER_MODEL, cfg.JUDGE_MODEL}, {"glm-5.3-flash"})
            self.assertEqual(cfg.DISCLOSURE_FORMAT_ATTEMPTS, 4)
            for limit in [2048, 48000, 128000]:
                Tracer(SimpleNamespace(dir=directory)).chat_json("world.structure", [], max_tokens=limit)
        result = self.run_tool(calls, ["--model", "glm-5.3-flash", "--min-output-tokens", "32768",
            "--max-output-tokens", "65536", "--call-timeout-seconds", "600",
            "--json-attempts", "5", "--disclosure-format-attempts", "4"])
        self.assertIsNone(result.error)
        self.assertEqual([x["kwargs"]["max_tokens"] for x in result.api.calls], [32768, 48000, 65536])
        self.assertTrue(all(x["kwargs"].get("reasoning_effort") == "low" and "max_completion_tokens" not in x["kwargs"]
                            and x["options"]["timeout"] == 600 for x in result.api.calls))
        self.assertTrue(all(r["max_attempts"] == 5 for r in result.rows if r.get("event") == "json_attempt"))
        self.assertEqual(result.profile["input_cny_per_m"], 0.632)
        self.assertEqual(result.profile["output_cny_per_m"], 2.212)

    def test_larger_limit_reserves_before_dispatch(self):
        result = self.run_tool(lambda directory, cfg: Tracer(SimpleNamespace(dir=directory)).chat_json(
            "render.discriminate", [], max_tokens=2048), ["--model", "glm-5.3-flash",
            "--min-output-tokens", "32768", "--max-output-tokens", "65536", "--max-cny", "0.001"])
        self.assertEqual(result.api.calls, [])
        self.assertEqual(result.profile["admitted_calls"], 0)


class LargeBatchTests(unittest.TestCase):
    setUp = batch_fixture.BatchTests.setUp
    args = batch_fixture.BatchTests.args
    prepare = batch_fixture.BatchTests.prepare

    def test_frozen_batch_carries_actual_model_and_permissive_execution_limits(self):
        _, plan = self.prepare("--model", "glm-5.3-flash", "--min-questions", "200", "--total-only",
            "--workers", "7", "--min-output-tokens", "32768", "--max-output-tokens", "65536",
            "--call-timeout-seconds", "600", "--json-attempts", "5", "--disclosure-format-attempts", "4")
        self.assertEqual(plan["model"], "glm-5.3-flash")
        self.assertEqual(plan["transport"]["profiles"]["low"]["reasoning_effort"], "low")
        self.assertEqual(plan["request_lifecycle"], "cancellable_async_http/v1")
        self.assertEqual(plan["min_questions"], 200)
        for run in plan["runs"]:
            command = run["command"]
            self.assertEqual(command[command.index("--model") + 1], "glm-5.3-flash")
            self.assertEqual(command[command.index("--min-output-tokens") + 1], "32768")
            self.assertNotIn("--world-review-reasoning-effort", command)
            self.assertNotIn("--corpus-review-max-tokens", command)

    def test_unsupported_glm_reasoning_rejected_before_preparing_run(self):
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            self.prepare("--model", "glm-5.3-flash", "--reasoning-effort", "medium")


if __name__ == "__main__":
    unittest.main(verbosity=2)
