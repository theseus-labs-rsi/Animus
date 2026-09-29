"""Provider-free original-world reviewer checks, not semantic-accuracy tests."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import world_semantics as review
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE


def fixture(seed=True):
    blueprint = {"version": 1, "entity_types": [{"id": "record", "fields": [
        {"name": "利润", "kind": "numeric"}, {"name": "负责人", "kind": "person"}]}],
        "temporal_model": {"unit": "day", "step_days": 2, "n_sessions": 2},
        "relation_types": [], "event_types": [], "causal_rules": []}
    ws = WorldState(entities={"甲/~记录": {
        "利润": Timeline([Op(0, "2025-01-06", SET, "10"), Op(1, "2025-01-08", UPDATE, "12", "10")]),
        "负责人": Timeline([Op(0, "2025-01-06", SET, "李明")])}},
        n_sessions=2, entity_types={"甲/~记录": "record"}, world_blueprint=deepcopy(blueprint))
    ws.conflicts = [{"entity": "甲/~记录", "field": "负责人", "session": 0, "date": "2025-01-06",
        "authoritative_value": "李明", "authoritative_source": "签发记录",
        "rumor_value": "王刚", "rumor_source": "待核转述", "rule": "source_reliability",
        "gt": "BAKED_GOLD_SENTINEL"}]
    wp = {"world_blueprint": blueprint, "quality_contract": {"world_semantic_review": True},
          "domain_profile": {"entity_noun": "记录"}, "shared_world_spec": {"scope": "可修订记录"},
          "traps": [{"trap": "可选干扰，不是必须"}],
          "rubric": "RUBRIC_SENTINEL", "questions": ["QUESTION_SENTINEL"],
          "provenance": {"path": "SOURCE_PATH_SENTINEL"},
          "_council_views": [{"opinion": "COUNCIL_SENTINEL"}]}
    if seed:
        wp["seed_contract"] = {"task": {"objective": "公开变化后的最新记录", "instructions": "保留两个时点",
                "rubric": "TASK_RUBRIC_SENTINEL"},
            "mechanisms": [{"id": "m1", "description": "更新记录",
                "required_entity_types": ["record"], "required_relation_types": [],
                "required_event_types": [], "required_causal_rules": [],
                "source_refs": ["SOURCE_REF_SENTINEL"]}],
            "blueprint_requirements": {"entity_types": [{"id": "record"}]},
            "sources": [{"path": "PRIVATE_SOURCE_FILE_SENTINEL", "role": "builder_only"}],
            "rubric": "PRIVATE_RUBRIC_SENTINEL"}
    task = {"description": "允许记录修订，核对对象和时间。", "few_shot": [{"title": "体裁示例",
        "content": "例子实体无需复制。", "date": "2020-01-01", "doc_type": "记录",
        "source_refs": ["INPUT_SOURCE_REF_SENTINEL"]}], "private": "INPUT_PRIVATE_SENTINEL"}
    return wp, ws, task


def reference_id(source, pointer, payload=None):
    if payload is None:
        payload, _ = review._project(*fixture())
    return next(row["ref_id"] for row in payload["reference_index"]
                if row["source"] == source and row["json_pointer"] == pointer)


def positive(seed=True, payload=None):
    return {"decision": "accept", "reason": "离线假意见只验证控制流及归属。", "issues": [],
        "mechanism_coverage": ([{"mechanism_id": "m1", "status": "witnessed",
             "refs": [reference_id("world", "/entities/甲~1~0记录/利润", payload)],
             "observed_sequence": "甲/~记录在2025-01-06记10，2025-01-08更新为12。",
             "reason": "此处两时点仅为测试定位，不宣称语义已证明。"}] if seed else []),
        "repair_targets": {"intrinsic": [], "structure": False},
        "limitations": "模型意见不是业务真值证明。"}


def negative(kind="contradiction", disposition="repair"):
    raw = positive()
    raw["decision"] = "repair" if disposition == "repair" else "unresolved"
    raw["issues"] = [{"id": "i1", "finding": "本测试设定为一项待核问题，会影响原任务依据。",
        "refs": [reference_id("task", "/description"),
                 reference_id("world", "/entities/甲~1~0记录/利润")],
        "alternative_reading": "若声明允许该变化也可能成立。",
        "disposition": disposition}]
    raw["repair_targets"] = {"intrinsic": [{"entity": "甲/~记录", "fields": ["利润"]}], "structure": False}
    return raw


class FakeTracer:
    def __init__(self, raw=None, error=None):
        self.raw = positive() if raw is None else raw
        self.error, self.calls = error, []

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
        if self.error:
            raise self.error
        return deepcopy(self.raw)


class WorldSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(REVIEWER_MODEL="gpt-5.4-mini")}).start()
        self.wp, self.ws, self.task = fixture()

    def run_review(self, raw=None, **kwargs):
        tracer = FakeTracer(raw)
        result = review.review_world(self.wp, self.ws, tracer, task_input=self.task, **kwargs)
        return result, tracer

    def test_one_original_tracer_call_exact_params_and_full_prepared_world(self):
        result, tracer = self.run_review()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(tracer.calls[0]["step"], "world.semantic_review")
        self.assertEqual(tracer.calls[0]["params"], {"model": "gpt-5.4-mini", "temperature": 0,
                                                   "max_tokens": 16384, "retries": 3, "strict_json": True,
                                                   "response_format": {"type": "json_object"}})
        payload = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertEqual(payload["world"]["entities"], self.ws.to_dict()["entities"])
        self.assertEqual(payload["calendar"], [{"session": 0, "period_number": 1, "date": "2025-01-06"},
            {"session": 1, "period_number": 2, "date": "2025-01-08"}])
        self.assertEqual(payload["world"]["conflicts"][0]["rumor_value"], "王刚")
        self.assertEqual(result["logical_calls"], 1)
        self.assertIsNone(result["physical_requests"])

    def test_private_sources_rubric_gold_questions_never_in_messages(self):
        result, tracer = self.run_review()
        text = json.dumps(tracer.calls[0]["messages"], ensure_ascii=False)
        self.assertNotIn("SENTINEL", text)
        self.assertEqual(result["binding"]["wp_hash"], review._hash(self.wp))
        self.assertEqual(result["binding"]["world_hash"], review._hash(self.ws.to_dict()))
        self.assertEqual(result["binding"]["task_hash"], review._hash(self.task))
        self.assertIn("advisory", result["input_snapshot"])

    def test_actual_world_call_has_its_stage_scope_without_hiding_required_facts(self):
        self.task["description"] += "最后生成自然周报和题目。"
        self.task["stage_scope"] = {"current_stage": "all_outputs_already_due"}
        result, tracer = self.run_review()
        payload = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertEqual(payload["stage_scope"]["current_stage"], "world")
        self.assertIn("corpus", payload["stage_scope"]["later_original_stages"])
        self.assertNotIn("stage_scope", payload["task"])
        self.assertEqual(payload["task"]["description"], self.task["description"])
        self.assertEqual(payload["world"]["entities"], self.ws.to_dict()["entities"])
        self.assertEqual(payload["seed"]["mechanisms"][0]["id"], "m1")
        stale = deepcopy(result)
        stale["input_snapshot"].pop("stage_scope")
        self.assertTrue(review.validate_review(stale, self.wp, self.ws, self.task))

    def test_semantic_opinion_is_not_overridden_by_lexical_or_domain_rules(self):
        passed, _ = self.run_review()
        failed, _ = self.run_review(negative())
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["repair_targets"], negative()["repair_targets"])
        self.assertEqual(review.validate_review(failed, self.wp, self.ws, self.task),
                         [{"code": "world_review_not_passed", "status": "failed"}])

    def test_legal_unknown_or_optional_editorial_note_can_pass(self):
        raw = negative("editorial", "non_blocking")
        raw["decision"] = "accept"
        raw["repair_targets"] = {"intrinsic": [], "structure": False}
        result, _ = self.run_review(raw)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])

    def test_schema_capacity_can_remain_unresolved_without_invented_target(self):
        raw = positive(); raw["decision"] = "unresolved"
        raw["reason"] = "冻结蓝图无法表达所需身份区分，不能擅改白皮书。"
        raw["mechanism_coverage"][0].update(status="unresolved", refs=[])
        result, _ = self.run_review(raw)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["repair_targets"], {"intrinsic": [], "structure": False})
        self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task))

    def test_malformed_raw_and_provider_error_are_preserved_no_retry(self):
        for raw in ({}, {"__error__": "timeout"}, {"decision": "accept"}, [], "truncated"):
            with self.subTest(raw=raw):
                result, tracer = self.run_review(raw)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["raw_output"], raw)
                self.assertEqual(len(tracer.calls), 1)
                self.assertEqual(result["repair_targets"], {"intrinsic": [], "structure": False})
                self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task))
        tracer = FakeTracer(error=TimeoutError("offline transport failure"))
        result = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertEqual(len(tracer.calls), 1)

    def test_invalid_input_is_error_before_caller(self):
        tracer = FakeTracer()
        result = review.review_world(self.wp, self.ws, tracer, task_input=["not an input object"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["logical_calls"], 0)
        self.assertFalse(result["caller_entered"])
        self.assertEqual(tracer.calls, [])

    def test_short_reference_ids_bound_to_actual_nodes_and_invented_refs_rejected(self):
        result, _ = self.run_review()
        node = result["locations"]["mechanisms"][0]["refs"][0]
        self.assertEqual(node["source_value"], self.ws.to_dict()["entities"]["甲/~记录"]["利润"])
        for pointer in ("/entities/absent", "/entities/甲~1~0记录/利润/01", "/entities/甲~1~0记录/利润/-1",
                        "/entities/甲~2记录", "/entities/甲~1~0记录/利润/99", "r_0123456789", "r", "r999999"):
            raw = positive(); raw["mechanism_coverage"][0]["refs"] = [pointer]
            result, tracer = self.run_review(raw)
            self.assertEqual(result["status"], "error", pointer)
            self.assertEqual(result["raw_output"], raw)
            self.assertEqual(len(tracer.calls), 1)

    def test_no_advisory_or_nonexistent_nodes_in_reference_index(self):
        payload, _ = review._project(self.wp, self.ws, self.task)
        for source, pointer in (("advisory", "/traps/0"), ("whitepaper", "/traps/0"), ("task", "/missing")):
            self.assertFalse(any(row["source"] == source and row["json_pointer"] == pointer
                                 for row in payload["reference_index"]))
            raw = negative("coverage_gap")
            raw["issues"][0]["refs"] = [source + ":" + pointer]
            result, _ = self.run_review(raw)
            self.assertEqual(result["status"], "error")
        raw = negative("coverage_gap"); raw["issues"][0]["refs"] = []
        self.assertEqual(self.run_review(raw)[0]["status"], "error")

    def test_repair_target_existing_fields_structure_or_no_route(self):
        raw = negative(); raw["repair_targets"] = {"intrinsic": [], "structure": True}
        self.assertEqual(self.run_review(raw)[0]["status"], "failed")
        for target in ({"intrinsic": [{"entity": "新实体", "fields": ["利润"]}], "structure": False},
                       {"intrinsic": [{"entity": "甲/~记录", "fields": ["新字段"]}], "structure": False},
                       {"intrinsic": [], "structure": False}, {"intrinsic": [], "structure": 1}):
            raw["repair_targets"] = target
            self.assertEqual(self.run_review(raw)[0]["status"], "error")

    def test_all_seed_mechanisms_required_with_real_node_location_for_witnessed(self):
        for mode in ("missing", "repeated", "unknown", "no_witness", "not_covered", "unresolved"):
            raw = positive()
            if mode == "missing": raw["mechanism_coverage"] = []
            elif mode == "repeated": raw["mechanism_coverage"] *= 2
            elif mode == "unknown": raw["mechanism_coverage"][0]["mechanism_id"] = "m2"
            elif mode == "no_witness": raw["mechanism_coverage"][0]["refs"] = []
            else: raw["mechanism_coverage"][0]["status"] = mode
            self.assertEqual(self.run_review(raw)[0]["status"], "error", mode)
        self.wp, self.ws, self.task = fixture(False)
        result, _ = self.run_review(positive(False))
        self.assertEqual(result["status"], "passed")

    def test_observed_sequence_required_without_backfill_or_domain_parsing(self):
        for value in (None, "", "   ", [], {"before": "10"}):
            raw = positive()
            raw["mechanism_coverage"][0]["observed_sequence"] = value
            result, tracer = self.run_review(raw)
            self.assertEqual(result["status"], "error", value)
            self.assertEqual(result["raw_output"], raw)
            self.assertEqual(len(tracer.calls), 1)
        legacy = positive(); del legacy["mechanism_coverage"][0]["observed_sequence"]
        result, _ = self.run_review(legacy)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("observed_sequence", result["raw_output"]["mechanism_coverage"][0])
        # Content accuracy stays with semantic review, not a hard-coded lifecycle parser.
        raw = positive(); raw["mechanism_coverage"][0]["observed_sequence"] = "自由自然表达，不含任何规定阶段词。"
        self.assertEqual(self.run_review(raw)[0]["status"], "passed")

    def test_observed_sequence_is_replayed_and_cannot_promote_v3_receipt(self):
        original, _ = self.run_review()
        self.assertEqual(original["mechanism_coverage"], original["raw_output"]["mechanism_coverage"])
        for where in ("summary", "raw", "raw_with_hash", "old_version"):
            changed = deepcopy(original)
            if where == "summary": changed["mechanism_coverage"][0]["observed_sequence"] = "改写"
            elif where == "old_version": changed["version"] = "original-world-semantics/v3"
            else:
                changed["raw_output"]["mechanism_coverage"][0]["observed_sequence"] = "改写"
                if where == "raw_with_hash": changed["raw_output_hash"] = review._hash(changed["raw_output"])
            self.assertTrue(review.validate_review(changed, self.wp, self.ws, self.task), where)

    def test_missing_required_witness_cannot_pass_or_repair_without_finding(self):
        for status in ("not_covered", "unresolved"):
            raw = positive(); raw["mechanism_coverage"][0]["status"] = status
            raw["mechanism_coverage"][0]["observed_sequence"] = "只见首次建立；另一条记录合法待处理。"
            self.assertEqual(self.run_review(raw)[0]["status"], "error")
            raw["decision"] = "repair"; raw["repair_targets"]["structure"] = True
            self.assertEqual(self.run_review(raw)[0]["status"], "error")
            raw["issues"] = negative()["issues"]
            self.assertEqual(self.run_review(raw)[0]["status"], "failed")

    def test_one_complete_output_template_emits_facts_before_decision(self):
        template = json.loads(review.SYSTEM[review.SYSTEM.index('{"mechanism_coverage":'):])
        self.assertEqual(list(template)[0], "mechanism_coverage")
        self.assertEqual(list(template)[-1], "decision")
        row = template["mechanism_coverage"][0]
        self.assertLess(list(row).index("observed_sequence"), list(row).index("status"))
        self.assertLess(list(row).index("observed_sequence"), list(row).index("reason"))

    def test_failed_and_unresolved_cannot_be_promoted_by_top_level_status(self):
        for raw in (negative(), negative(disposition="unresolved"), {}):
            result, _ = self.run_review(raw)
            result["status"] = "passed"
            self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task))

    def test_input_mutations_invalidate_exact_receipt(self):
        result, _ = self.run_review()
        for mode in ("world", "prepared_conflict", "blueprint", "seed", "task", "private_input_hash"):
            wp, ws, task = deepcopy(self.wp), deepcopy(self.ws), deepcopy(self.task)
            if mode == "world": ws.entities["甲/~记录"]["利润"].ops[0].value = "999"
            elif mode == "prepared_conflict": ws.conflicts[0]["authoritative_source"] = "changed"
            elif mode == "blueprint": wp["world_blueprint"]["temporal_model"]["step_days"] = 3
            elif mode == "seed": wp["seed_contract"]["task"]["instructions"] += "changed"
            elif mode == "task": task["description"] += "changed"
            else: wp["provenance"]["path"] = "another opaque identity"
            self.assertTrue(review.validate_review(result, wp, ws, task), mode)

    def test_raw_summary_location_and_message_tampering_is_rejected(self):
        original, _ = self.run_review()
        for mode in ("raw", "raw_with_new_hash", "reason", "location", "message", "payload", "binding"):
            result = deepcopy(original)
            if mode.startswith("raw"):
                result["raw_output"]["reason"] = "changed"
                if mode.endswith("new_hash"): result["raw_output_hash"] = review._hash(result["raw_output"])
            elif mode == "reason": result["reason"] = "changed"
            elif mode == "location": result["locations"]["mechanisms"][0]["refs"][0]["source_value"] = []
            elif mode == "message": result["messages"][-1]["content"] += " "
            elif mode == "payload": result["input_snapshot"]["world"]["entities"] = {}
            else: result["binding"]["world_hash"] = "0" * 64
            self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task), mode)

    def test_previous_and_author_rebuttal_preserved_and_bound(self):
        previous, _ = self.run_review(negative())
        responses = [{"issue_id": "i1", "status": "disputed", "reason": "原任务明确允许修订。"}]
        result, _ = self.run_review(previous=previous, author_responses=responses)
        self.assertEqual(result["previous"], previous)
        self.assertEqual(result["author_responses"], responses)
        self.assertEqual(result["input_snapshot"]["previous_review"]["raw_output"], previous["raw_output"])
        self.assertEqual(result["input_snapshot"]["previous_review"]["locations"], previous["locations"])
        self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])
        result["author_responses"][0]["reason"] = "已修改"
        self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task))

    def test_no_mutation_of_world_whitepaper_task_or_model_raw(self):
        originals = deepcopy((self.wp, self.ws.to_dict(), self.task))
        raw = positive(); before = deepcopy(raw)
        self.run_review(raw)
        self.assertEqual((self.wp, self.ws.to_dict(), self.task), originals)
        self.assertEqual(raw, before)

    def test_enabled_is_explicit_and_pure_no_provider_config_import(self):
        self.assertTrue(review.enabled(self.wp))
        self.assertTrue(review.enabled({}, {"world_semantic_review": True}))
        self.assertFalse(review.enabled({}))
        self.assertFalse(review.enabled({"quality_contract": {"world_semantic_review": "true"}}))
        original_import = builtins.__import__
        def blocked(name, *a, **kw):
            if name in ("config", "openai"): raise AssertionError("provider import forbidden")
            return original_import(name, *a, **kw)
        result, _ = self.run_review()
        with patch("builtins.__import__", side_effect=blocked):
            self.assertTrue(review.enabled(self.wp))
            self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])

    def test_call_and_version_contract_and_physical_count_cannot_be_forged(self):
        original, _ = self.run_review()
        for mode in ("step", "model", "tokens", "retries", "json", "version", "physical", "caller"):
            result = deepcopy(original)
            if mode == "step": result["call"]["step"] = "another.step"
            elif mode == "model": result["call"]["params"]["model"] = "another-model"
            elif mode == "tokens": result["call"]["params"]["max_tokens"] = 128
            elif mode == "retries": result["call"]["params"]["retries"] = 5
            elif mode == "json": result["call"]["params"]["strict_json"] = False
            elif mode == "version": result["version"] = "old"
            elif mode == "physical": result["physical_requests"] = 1
            else: result["caller_entered"] = False
            self.assertTrue(review.validate_review(result, self.wp, self.ws, self.task), mode)
        with patch.object(review, "SYSTEM", review.SYSTEM + " changed"):
            self.assertTrue(review.validate_review(original, self.wp, self.ws, self.task))

    def test_complete_long_task_and_world_are_not_truncated(self):
        self.task["description"] = "原任务" * 20000 + "结尾仍保留"
        self.ws.entities["甲/~记录"]["负责人"].ops[0].value = "记录" * 10000 + "末尾事实"
        result, tracer = self.run_review()
        payload = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertEqual(payload["task"]["description"], self.task["description"])
        self.assertEqual(payload["world"]["entities"], self.ws.to_dict()["entities"])
        self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])

    def test_reference_index_deterministic_local_ids_and_actual_array_instance_labels(self):
        self.ws.events = [{"id": "evt-1", "session": 1, "date": "2025-01-08", "effects": []}]
        payload, _ = review._project(self.wp, self.ws, self.task)
        index = payload["reference_index"]
        event = next(row for row in index if row["source"] == "world" and row["json_pointer"] == "/events/0")
        self.assertEqual(event["node_id"], "evt-1")
        self.assertEqual(len({row["ref_id"] for row in index}), len(index))
        self.assertEqual([row["ref_id"] for row in index], [f"r{i}" for i in range(1, len(index) + 1)])
        self.assertEqual(index, review._project(self.wp, self.ws, self.task)[0]["reference_index"])
        for row in index:
            located = review._refs([row["ref_id"]], payload)[0]
            self.assertEqual(located["json_pointer"], row["json_pointer"])
        changed = deepcopy(self.ws)
        changed.events[0]["effects"] = [{"entity": "甲/~记录", "field": "利润", "set": "13"}]
        new, _ = review._project(self.wp, changed, self.task)
        self.assertEqual(event["ref_id"], next(x for x in new["reference_index"] if x.get("node_id") == "evt-1")["ref_id"])
        self.assertNotEqual(review._refs([event["ref_id"]], payload)[0]["source_value_hash"],
                            review._refs([event["ref_id"]], new)[0]["source_value_hash"])

    def test_renumbered_node_cannot_reuse_old_world_or_v2_receipt(self):
        original, _ = self.run_review()
        old_node = original["locations"]["mechanisms"][0]["refs"][0]
        # A new earlier-sorted entity changes local numbering, not old history.
        changed = deepcopy(self.ws)
        changed.entities["AAA"] = {"负责人": Timeline([Op(0, "2025-01-06", SET, "赵宁")])}
        changed.entity_types["AAA"] = "record"
        payload, _ = review._project(self.wp, changed, self.task)
        rebound = review._refs([old_node["ref_id"]], payload)[0]
        self.assertNotEqual(rebound["json_pointer"], old_node["json_pointer"])
        self.assertNotEqual(rebound["source_value_hash"], old_node["source_value_hash"])
        self.assertTrue(review.validate_review(original, self.wp, changed, self.task))
        old_version = deepcopy(original)
        old_version["version"] = "original-world-semantics/v2"
        self.assertTrue(review.validate_review(old_version, self.wp, self.ws, self.task))

    def test_previous_local_id_keeps_old_value_while_current_refs_use_new_index(self):
        previous, _ = self.run_review(negative())
        before = deepcopy(previous)
        old_node = previous["locations"]["issues"][0]["refs"][1]
        self.ws.entities["AAA"] = {"负责人": Timeline([Op(0, "2025-01-06", SET, "赵宁")])}
        self.ws.entity_types["AAA"] = "record"
        payload, _ = review._project(self.wp, self.ws, self.task)
        raw = positive(payload=payload)
        current, tracer = self.run_review(raw, previous=previous)
        self.assertEqual(current["status"], "passed")
        sent = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertEqual(previous, before)
        self.assertEqual(sent["previous_review"]["raw_output"], previous["raw_output"])
        self.assertEqual(sent["previous_review"]["locations"], previous["locations"])
        self.assertEqual(sent["previous_review"]["locations"]["issues"][0]["refs"][1], old_node)
        same_local_id = review._refs([old_node["ref_id"]], sent)[0]
        self.assertNotEqual(same_local_id["json_pointer"], old_node["json_pointer"])
        new_node = current["locations"]["mechanisms"][0]["refs"][0]
        self.assertNotEqual(new_node["ref_id"], old_node["ref_id"])
        self.assertEqual(new_node["json_pointer"], old_node["json_pointer"])
        self.assertEqual(new_node["source_value_hash"], old_node["source_value_hash"])
        self.assertEqual(review.validate_review(current, self.wp, self.ws, self.task), [])
        current["previous"]["locations"]["issues"][0]["refs"][1]["source_value"] = []
        self.assertTrue(review.validate_review(current, self.wp, self.ws, self.task))

    def add_owned_fields(self):
        blueprint = self.ws.world_blueprint
        blueprint["entity_types"].append({"id": "manager", "fields": []})
        self.ws.entities["乙负责人"] = {}
        self.ws.entity_types["乙负责人"] = "manager"
        # Exercise the reverse (to-owned) relation, using the original helper.
        blueprint["relation_types"] = [{"id": "r1", "from_type": "manager", "to_type": "record", "field": "负责人"}]
        blueprint["event_types"] = [{"id": "e1", "roles": {"record": "record"},
                                      "effect_fields": [{"role": "record", "field": "利润"}]}]
        self.wp["world_blueprint"] = deepcopy(blueprint)

    def test_author_routes_match_relation_owner_and_event_effect_fields(self):
        self.add_owned_fields()
        payload, _ = review._project(self.wp, self.ws, self.task)
        routes = {row["entity"]: row for row in payload["author_routes"]}
        self.assertEqual(routes["甲/~记录"]["intrinsic_fields"], [])
        self.assertEqual(set(routes["甲/~记录"]["structure_fields"]), {"负责人", "利润"})
        self.assertEqual(routes["乙负责人"]["intrinsic_fields"], [])
        self.assertEqual(routes["乙负责人"]["structure_fields"], [])

    def test_owned_intrinsic_route_errors_before_any_repair_not_silently_reclassified(self):
        self.add_owned_fields()
        for field in ("利润", "负责人"):
            raw = negative(); raw["repair_targets"]["intrinsic"][0]["fields"] = [field]
            raw["repair_targets"]["structure"] = True
            result, tracer = self.run_review(raw)
            self.assertEqual(result["status"], "error")
            self.assertIn("allowed intrinsic", result["error"])
            self.assertEqual(result["raw_output"], raw)
            self.assertEqual(result["repair_targets"], {"intrinsic": [], "structure": False})
            self.assertEqual(len(tracer.calls), 1)
        raw["repair_targets"] = {"intrinsic": [], "structure": True}
        self.assertEqual(self.run_review(raw)[0]["status"], "failed")

    def test_implementation_index_and_route_changes_invalidate_receipt(self):
        original, _ = self.run_review()
        hashes = review._implementation_hashes()
        self.assertEqual(set(hashes), {"world_semantics.py", "reference_locations.py", "world_blueprint.py",
                                     "world_gen.py", "world_joint_plan.py", "seed_world.py", "seed_pack.py", "seed_v2.py",
                                     "world_state.py", "value_types.py", "prompts.py", "disclosure.py", "world_context.py", "world_agent.py"})
        for source in ("world_semantics.py", "prompts.py"):
            with self.subTest(source=source), patch.object(
                    review, "_implementation_hashes", return_value={**hashes, source: "0" * 64}):
                self.assertTrue(review.validate_review(original, self.wp, self.ws, self.task))
        for mode in ("index", "route"):
            edited = deepcopy(original)
            if mode == "index": edited["input_snapshot"]["reference_index"][0]["json_pointer"] = "/changed"
            else: edited["input_snapshot"]["author_routes"][0]["intrinsic_fields"] = []
            self.assertTrue(review.validate_review(edited, self.wp, self.ws, self.task))

    def test_saved_failed_calibrations_preserved_and_actual_bad_routes_detected_offline(self):
        # The prior failures are development evidence, not successful opinions.
        # Only their returned targets are tested against newly derived ownership.
        paths = [ROOT / "output/original_bc_20260918/calibration" / name / "review.json"
                 for name in ("world_gate_v1", "world_gate_v2")]
        if not all(path.exists() for path in paths):
            self.skipTest("real failed calibration artifacts unavailable")
        for path in paths:
            before = path.read_bytes()
            old = json.loads(before)
            self.assertEqual(old["status"], "error")
            world = old["input_snapshot"]["world"]
            routes = {row["entity"]: set(row["intrinsic_fields"]) for row in review._author_routes(world)}
            targets = old["raw_output"]["repair_targets"]["intrinsic"]
            invalid = [(row["entity"], field) for row in targets for field in row["fields"]
                       if field not in routes[row["entity"]]]
            self.assertTrue(invalid, path.name)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
