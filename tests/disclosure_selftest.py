"""Original disclosure interface controls. Scripted opinions are not semantic evidence."""
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
from pipeline import disclosure as d, world_semantics as review
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE
from world_semantics_selftest import fixture, positive, FakeTracer


def data():
    wp, ws, task = fixture(False)
    wp["quality_contract"]["public_disclosure"] = True
    ws.conflicts = []
    ws.entities["秘密实体"] = {"隐藏值": Timeline([Op(0, ws.date_of_session(0), SET, "SECRET_VALUE")])}
    ws.entity_types["秘密实体"] = "record"
    return wp, ws, task


def raw_plan(ws, choose=None):
    refs = d.catalogue(ws)
    chosen = [r["ref"] for r in refs if (choose(r) if choose else r.get("field") == "利润")]
    fixed = {r for row in d._fixed(ws, refs) for r in row["refs"]}
    return {"records": ([{"session": 1, "refs": chosen, "channel": "档案回顾",
              "acquisition_context": "本期收到先前记录，并明确列出原记录时点。"}] if chosen else []),
            "undisclosed": [r["ref"] for r in refs if r["ref"] not in set(chosen) | fixed],
            "reason": "仅测试公开投影，不宣称自然安排正确。"}


def install(wp, ws, task, raw=None):
    ws.disclosure = d.compile_plan(wp, ws, raw or raw_plan(ws), task)
    assert not d.validate_plan(ws), d.validate_plan(ws)
    return ws.disclosure


def positive_disclosure(wp, ws, task):
    """Complete fixed test opinion; never fill missing fields in real outputs."""
    payload, _ = review._project(wp, ws, task)
    raw = positive(False)
    raw["disclosure_reviews"] = []
    for index, record in enumerate(ws.disclosure["records"]):
        ref = next(x["ref_id"] for x in payload["reference_index"] if x["source"] == "world"
                   and x["json_pointer"] == f"/disclosure/records/{index}")
        raw["disclosure_reviews"].append({"record_id": record["id"], "understanding": "离线固定原记录理解",
            "refs": [ref], "reason": "离线桩只检查接口完整性", "status": "compatible"})
    return raw


class SequenceTracer(FakeTracer):
    def __init__(self, outputs):
        super().__init__(); self.outputs = list(outputs)

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


class DisclosureTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(STRUCTURE_MODEL="cheap-author", REVIEWER_MODEL="cheap-review")}).start()
        self.wp, self.ws, self.task = data()

    def test_private_and_public_clocks_and_exact_history(self):
        before = self.ws.to_dict()
        install(self.wp, self.ws, self.task)
        self.assertEqual({k: v for k, v in self.ws.to_dict().items() if k != "disclosure"}, before)
        self.assertEqual(d.session_facts(self.ws, 0), [])
        rows = d.session_facts(self.ws, 1)
        self.assertEqual([(r["value"], r["fact_session"], r["disclosure_session"]) for r in rows], [("10", 0, 1), ("12", 1, 1)])
        self.assertEqual(len({r["target_ref"] for r in rows}), 2)
        public = d.context(self.ws, 1)
        self.assertNotIn("SECRET_VALUE", json.dumps(public, ensure_ascii=False))
        self.assertNotIn("秘密实体", json.dumps(public, ensure_ascii=False))
        self.assertEqual(public["disclosures"][0]["channel"], "档案回顾")
        self.assertEqual(self.ws.entities["秘密实体"]["隐藏值"].value_at_session(1), "SECRET_VALUE")

    def test_repeated_acquisition_has_separate_occurrences(self):
        raw = raw_plan(self.ws); raw["records"].append(deepcopy(raw["records"][0]))
        raw["records"][1]["channel"] = "复取档案"
        install(self.wp, self.ws, self.task, raw)
        rows = d.session_facts(self.ws, 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["disclosure_id"] for r in rows}, {"d1", "d2"})

    def test_one_author_call_json_mode_and_private_input_projection(self):
        tracer = FakeTracer(raw_plan(self.ws))
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(len(tracer.calls), 1)
        self.assertEqual(tracer.calls[0]["params"]["response_format"], {"type": "json_object"})
        text = json.dumps(tracer.calls[0]["messages"])
        for forbidden in ("RUBRIC_SENTINEL", "SOURCE_PATH_SENTINEL", "INPUT_PRIVATE_SENTINEL", "QUESTION_SENTINEL"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(d.validate_plan(self.ws))

    def test_provider_failure_keeps_previous_plan(self):
        before = deepcopy(install(self.wp, self.ws, self.task))
        tracer = FakeTracer(error=RuntimeError("offline failure"))
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["logical_calls"], 1)
        self.assertEqual(self.ws.disclosure, before)

    def test_missing_refs_preserves_invalid_raw_and_no_plan(self):
        raw = raw_plan(self.ws); raw["undisclosed"] = []
        result = d.author_plan(self.wp, self.ws, FakeTracer(raw), self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["raw_output"], raw)
        self.assertEqual(self.ws.disclosure, {})
        self.assertEqual(result["logical_calls"], 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual([x["raw_output"] for x in result["attempts"]], [raw, raw])

    def test_bounded_reference_repair_binds_actual_second_payload(self):
        good = raw_plan(self.ws); bad = deepcopy(good); bad["undisclosed"] = []
        tracer = SequenceTracer([bad, good])
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["logical_calls"], 2)
        self.assertEqual([x["raw_output"] for x in result["attempts"]], [bad, good])
        repair = self.ws.disclosure["author_input"]["format_repair"]
        self.assertEqual(repair["previous_output"], bad)
        self.assertEqual(repair["validation_error"]["missing_refs"], sorted(good["undisclosed"]))
        self.assertEqual(self.ws.disclosure["binding"]["messages_hash"], d._hash(tracer.calls[1]["messages"]))
        self.assertNotEqual(d._hash(tracer.calls[0]["messages"]), d._hash(tracer.calls[1]["messages"]))
        self.assertFalse(d.validate_plan(self.ws))

    def test_fixed_normal_refs_explicit_and_overlap_precise(self):
        self.wp, self.ws, self.task = fixture(False)
        self.wp["quality_contract"]["public_disclosure"] = True
        good = raw_plan(self.ws)
        payload = d._payload(self.wp, self.ws, self.task, None)
        fixed = payload["fixed_refs_already_public"]
        self.assertTrue(any(x.startswith("f") for x in fixed))
        self.assertFalse(set(fixed).intersection(payload["refs_requiring_arrangement"]))
        self.assertEqual(set(fixed) | set(payload["refs_requiring_arrangement"]), {r["ref"] for r in payload["catalogue"]})
        bad = deepcopy(good); bad["undisclosed"] += fixed
        with self.assertRaises(d.PlanFormatError) as caught:
            d.compile_plan(self.wp, self.ws, bad, self.task)
        self.assertEqual(caught.exception.details["overlap_refs"], fixed)
        self.assertEqual(caught.exception.details["fixed_overlap_refs"], fixed)
        tracer = SequenceTracer([bad, good])
        self.assertEqual(d.author_plan(self.wp, self.ws, tracer, self.task)["status"], "ready")
        self.assertFalse(d.validate_plan(self.ws))

    def test_shape_repair_and_no_silent_assignment(self):
        good = raw_plan(self.ws)
        tracer = SequenceTracer([{"records": []}, good])
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.ws.disclosure["raw_output"], good)
        self.assertEqual(self.ws.disclosure["undisclosed"], good["undisclosed"])
        self.assertFalse(d.validate_plan(self.ws))

    def test_patch_adds_missing_arrangement_without_rewriting_existing_records(self):
        good = raw_plan(self.ws)
        missing = good["undisclosed"][0]
        bad = deepcopy(good)
        bad["undisclosed"].remove(missing)
        addition = {"session": 1, "refs": [missing], "channel": "补充档案",
                    "acquisition_context": "本期获取补充记录并交代其原时点。"}
        patch_output = {"record_updates": [], "append_records": [addition],
                        "undisclosed": None, "reason": "作者安排遗漏引用的获取途径。"}
        tracer = SequenceTracer([bad, patch_output])
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual("ready", result["status"], result)
        self.assertEqual(patch_output, result["attempts"][1]["raw_output"])
        self.assertEqual(patch_output, self.ws.disclosure["author_output"])
        compiled = self.ws.disclosure["raw_output"]
        self.assertEqual(bad["records"], compiled["records"][:-1])
        self.assertEqual(addition, compiled["records"][-1])
        self.assertEqual(bad["undisclosed"], compiled["undisclosed"])
        self.assertFalse(d.validate_plan(self.ws))
        self.ws.disclosure["author_output"]["append_records"][0]["session"] = 0
        self.ws.disclosure["plan_hash"] = d._hash({k:v for k,v in self.ws.disclosure.items() if k != "plan_hash"})
        self.assertTrue(d.validate_plan(self.ws))

    def test_patch_feedback_lists_coverage_gaps_even_when_first_error_is_duplicate(self):
        good = raw_plan(self.ws)
        bad = deepcopy(good)
        bad["records"][0]["refs"].append(bad["records"][0]["refs"][0])
        missing = bad["undisclosed"].pop()
        patch_output = {"record_updates": [{"record_index": 0, "record": good["records"][0]}],
                        "append_records": [], "undisclosed": good["undisclosed"], "reason": "保留作者原有安排并修复漏项。"}
        tracer = SequenceTracer([bad, patch_output])
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual("ready", result["status"], result)
        repair = self.ws.disclosure["author_input"]["format_repair"]
        self.assertEqual("record_refs", repair["validation_error"]["code"])
        self.assertIn(missing, repair["missing_refs_all"])
        self.assertFalse(d.validate_plan(self.ws))

    def test_patch_cannot_move_an_unknown_index_or_silently_fill_a_gap(self):
        good = raw_plan(self.ws)
        for operations in ([], [{"record_index": 999, "record": good["records"][0]}]):
            bad = deepcopy(good); bad["undisclosed"] = []
            patch_output = {"record_updates": operations, "append_records": [],
                            "undisclosed": None, "reason": "遗漏未解决"}
            result = d.author_plan(self.wp, self.ws, SequenceTracer([bad, patch_output]), self.task)
            self.assertEqual("error", result["status"])
            self.assertFalse(self.ws.disclosure)

    def test_provider_valueerror_is_not_repaired_and_second_failure_preserved(self):
        before = deepcopy(install(self.wp, self.ws, self.task))
        for values, expected in (([ValueError("provider invalid JSON")], 1),
                                 ([{"records": []}, TimeoutError("provider timeout")], 2)):
            with self.subTest(expected=expected):
                tracer = SequenceTracer(values)
                result = d.author_plan(self.wp, self.ws, tracer, self.task)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["logical_calls"], expected)
                self.assertEqual(len(tracer.calls), expected)
                self.assertEqual(result["attempts"][-1]["status"], "execution_error")
                self.assertEqual(self.ws.disclosure, before)

    def test_invalid_world_fixed_precondition_gets_no_call(self):
        self.ws.conflicts = [{"entity": "甲/~记录", "field": "利润", "session": 0,
                             "authoritative_value": "not-canonical", "authoritative_source": "公告"}]
        tracer = SequenceTracer([])
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["logical_calls"], 0)
        self.assertFalse(tracer.calls)

    def test_repaired_plan_rejects_changed_first_error_or_output(self):
        good = raw_plan(self.ws); bad = deepcopy(good); bad["undisclosed"] = []
        d.author_plan(self.wp, self.ws, SequenceTracer([bad, good]), self.task)
        plan = deepcopy(self.ws.disclosure)
        for key in ("validation_error", "previous_output"):
            self.ws.disclosure = deepcopy(plan)
            ctx = self.ws.disclosure["author_input"]["format_repair"]
            if key == "validation_error": ctx[key]["missing_refs"] = []
            else: ctx[key] = deepcopy(good)
            self.ws.disclosure["binding"]["messages_hash"] = d._hash(d._messages(self.ws.disclosure["author_input"]))
            self.ws.disclosure["plan_hash"] = d._hash({k:v for k,v in self.ws.disclosure.items() if k != "plan_hash"})
            self.assertTrue(d.validate_plan(self.ws))

    def test_duplicate_unknown_and_outside_period_rejected(self):
        for mode in ("duplicate", "unknown", "period", "both"):
            raw = raw_plan(self.ws)
            if mode == "duplicate": raw["records"][0]["refs"] *= 2
            elif mode == "unknown": raw["records"][0]["refs"][0] = "fake"
            elif mode == "period": raw["records"][0]["session"] = 999
            else: raw["undisclosed"].append(raw["records"][0]["refs"][0])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                d.compile_plan(self.wp, self.ws, raw, self.task)

    def test_stale_truth_and_plan_edits_rejected(self):
        plan = deepcopy(install(self.wp, self.ws, self.task))
        self.ws.entities["甲/~记录"]["利润"].ops[0].value = "changed"
        self.assertTrue(d.validate_plan(self.ws))
        self.wp, self.ws, self.task = data(); self.ws.disclosure = plan
        self.ws.disclosure["records"][0]["channel"] = "forged"
        self.assertTrue(d.validate_plan(self.ws))

    def test_roundtrip_and_no_plan_legacy_shape(self):
        self.assertNotIn("disclosure", self.ws.to_dict())
        self.assertEqual(WorldState.from_dict(self.ws.to_dict()).disclosure, {})
        install(self.wp, self.ws, self.task)
        copy = WorldState.from_dict(self.ws.to_dict())
        self.assertEqual(copy.disclosure, self.ws.disclosure)
        self.assertFalse(d.validate_plan(copy))

    def test_fixed_l5_authority_and_rumor_timing(self):
        self.wp, self.ws, self.task = fixture(False)
        self.wp["quality_contract"]["public_disclosure"] = True
        install(self.wp, self.ws, self.task)
        self.assertTrue(any(x["value"] == "李明" for x in d.session_facts(self.ws, 0)))
        claims = d.context(self.ws, 0)["public_claims"]
        self.assertEqual({x["value"] for x in claims}, {"李明", "王刚"})
        self.assertNotIn("BAKED_GOLD_SENTINEL", json.dumps(d.context(self.ws, 0)))
        self.assertEqual(d.source_assertions(self.ws, 0)[0]["source"], "签发记录")
        raw = raw_plan(self.ws); raw["undisclosed"].append("c1_rumor")
        with self.assertRaises(ValueError): d.compile_plan(self.wp, self.ws, raw, self.task)

    def test_fixed_l9_l10_and_benign_witness(self):
        self.ws.rule_instances = [{"inst_id": "i1", "session": 0, "x": 2, "surface_action": "登记", "canon_action": "PRIVATE_RULE"}]
        self.ws.sensitive = [{"entity": "甲/~记录", "field": "密钥", "stype": "token", "value": "sensitive-public", "session": 0,
                              "witness_field": "负责人", "witness_value": "李明"}]
        install(self.wp, self.ws, self.task)
        self.assertTrue(any(x["field"] == "负责人" for x in d.session_facts(self.ws, 0)))
        text = json.dumps(d.context(self.ws, 0))
        self.assertIn("sensitive-public", text)
        self.assertNotIn("PRIVATE_RULE", text)
        self.assertEqual(len(self.ws.disclosure["records"]), 3)

    def test_event_atomic_projection_without_private_ancestors(self):
        self.ws.events = [{"id": "ev1", "type": "revise", "session": 1, "date": self.ws.date_of_session(1),
                          "participants": {"record": "甲/~记录"}, "effects": [{"entity": "甲/~记录", "field": "利润", "value": "12"}],
                          "caused_by": ["PRIVATE_ANCESTOR"]}]
        raw = raw_plan(self.ws, lambda r: r["kind"] == "event")
        install(self.wp, self.ws, self.task, raw)
        event = d.session_events(self.ws, 1)[0]
        self.assertEqual(event["effects"][0]["value"], "12")
        self.assertEqual(event["channel"], "档案回顾")
        self.assertEqual(event["caused_by"], ["PRIVATE_ANCESTOR"])
        self.assertEqual(len(d.context(self.ws, 1)["events"]), 1)
        self.assertNotIn("SECRET_VALUE", json.dumps(d.context(self.ws, 1)))

    def test_review_missing_plan_fails_before_call(self):
        tracer = FakeTracer(positive(False))
        result = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(tracer.calls, [])

    def test_review_plan_input_mismatch_fails_before_call(self):
        install(self.wp, self.ws, self.task)
        for part in ("task", "wp"):
            wp, task = deepcopy(self.wp), deepcopy(self.task)
            (task if part == "task" else wp)["description"] = "different"
            tracer = FakeTracer(positive(False))
            self.assertEqual(review.review_world(wp, self.ws, tracer, task)["status"], "error")
            self.assertFalse(tracer.calls)

    def test_review_compact_plan_and_replay(self):
        install(self.wp, self.ws, self.task)
        tracer = FakeTracer(positive_disclosure(self.wp, self.ws, self.task))
        report = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "passed", report)
        self.assertFalse(review.validate_review(report, self.wp, self.ws, self.task))
        plan = report["input_snapshot"]["world"]["disclosure"]
        self.assertEqual(set(plan), {"version", "catalogue", "records", "undisclosed", "reason"})
        self.assertTrue(any(x["json_pointer"].startswith("/disclosure/records/") for x in report["input_snapshot"]["reference_index"]))

    def test_disclosure_only_repair_legal_without_truth_target(self):
        install(self.wp, self.ws, self.task)
        payload, _ = review._project(self.wp, self.ws, self.task)
        ref = next(x["ref_id"] for x in payload["reference_index"] if x["json_pointer"] == "/disclosure/records/0" and x["source"] == "world")
        raw = positive_disclosure(self.wp, self.ws, self.task)
        raw["disclosure_reviews"][0]["status"] = "repair"
        raw.update(decision="repair", issues=[{"id": "i1", "finding": "取得过程有待澄清。", "refs": [ref],
                    "alternative_reading": "可回顾旧记录，需说明。", "disposition": "repair"}],
                    repair_targets={"intrinsic": [], "structure": False, "disclosure": True})
        report = review.review_world(self.wp, self.ws, FakeTracer(raw), self.task)
        self.assertEqual(report["status"], "failed", report)
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task)[0]["code"], "world_review_not_passed")

    def test_old_mode_rejects_new_route_and_preserves_system(self):
        wp, ws, task = fixture(False)
        raw = positive(False); raw["repair_targets"]["disclosure"] = False
        tracer = FakeTracer(raw)
        self.assertEqual(review.review_world(wp, ws, tracer, task)["status"], "error")
        self.assertEqual(tracer.calls[0]["messages"][0]["content"], review.SYSTEM)


if __name__ == "__main__":
    unittest.main()
