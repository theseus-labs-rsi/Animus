"""Offline scripted agent protocol tests; these do not prove model quality."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from pipeline import disclosure as d, world_semantics as review
from pipeline.world_context import ReadWindow, ProgressGuard
from world_semantics_selftest import fixture


class ScopedAgentFixture:
    """Reusable by original factory integration tests; exact two legacy steps."""

    def __init__(self, *, reject=False):
        self.calls = []
        self.nodes = {}
        self.reject = reject

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
        payload = json.loads(messages[-1]["content"])
        self.nodes.setdefault(step, {}).update(payload["exact_reads"])
        pending_parts = {key: [part for part in parts if
                              part != payload["exact_reads"].get(key, {}).get("part")]
                         for key, parts in payload.get("unread_parts", {}).items()}
        remaining = [key for key in payload["unread_ids"]
                     if key not in payload["exact_reads"] or pending_parts.get(key)]
        if remaining:
            if remaining[0] in pending_parts:
                return {"action": "inspect", "ids": [remaining[0]], "part": pending_parts[remaining[0]][0]}
            return {"action": "inspect", "ids": remaining[:8]}
        if step == d.STEP:
            if not payload["working_state"]["records"]:
                refs = payload["requirements"]["refs_requiring_arrangement"]
                return {"action": "arrange", "records": ([{"session": payload["requirements"]["calendar"][-1]["session"],
                    "refs": refs, "channel": "离线场景记录", "acquisition_context": "在本期汇总此前各时点的原始记录。"}] if refs else []),
                    "undisclosed": [], "record_updates": [], "reason": "仅用于接口测试的固定作者响应。"}
            return {"action": "finish", "reason": "离线安排完成；模型语义质量尚未实测。"}
        if step != review.STEP:
            raise AssertionError("Unsupported fixture step: " + step)
        world_refs = [r["ref_id"] for node in self.nodes[step].values() for r in node.get("reference_index", [])]
        record_contexts = [node["value"] for node in self.nodes[step].values()
                           if isinstance(node.get("value"), dict) and "record_ref_id" in node["value"]]
        state = payload["working_state"]
        mechanisms = payload["requirements"]["seed"]["mechanisms"]
        if (len(state["mechanism_coverage"]) < len(mechanisms)
                or len(state["disclosure_reviews"]) < len(record_contexts)
                or self.reject and not state["issues"]):
            return {"action": "submit", "issues": ([{"id": "offline_issue", "finding": "离线注入阻断问题。",
                "refs": [world_refs[0]], "alternative_reading": "本测试没有足以消除该问题的证据。", "disposition": "unresolved"}] if self.reject else []),
                "mechanism_coverage": [{"mechanism_id": m["id"], "status": "witnessed", "refs": [world_refs[0]],
                    "observed_sequence": "离线假意见仅验证引用与控制流。", "reason": "此意见不作为业务质量结论。"} for m in mechanisms],
                "disclosure_reviews": [{"record_id": r["record_id"], "understanding": "离线固定记录理解。",
                    "refs": [r["record_ref_id"]], "reason": "只验证已读取并逐条提交。", "status": "compatible"} for r in record_contexts],
                "note": "读取完毕，保留原始节点的引用。"}
        targets = {"intrinsic": [], "structure": False}
        if payload["requirements"].get("public_disclosure_scope"):
            targets["disclosure"] = False
        return {"action": "finish", "decision": "unresolved" if self.reject else "accept", "reason": "离线固定最终意见。",
                "limitations": "本轮只验证工程协议；没有调用真实模型。", "repair_targets": targets}


class Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(STRUCTURE_MODEL="cheap-author", REVIEWER_MODEL="cheap-review")}).start()
        self.wp, self.ws, self.task = fixture()
        self.wp["world_generation"] = {"strategy": "agentic"}
        self.wp["quality_contract"]["public_disclosure"] = True
        self.ws.conflicts = []

    def author(self):
        tracer = ScopedAgentFixture()
        report = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(d.validate_plan(self.ws), [])
        return tracer, report

    def test_original_author_compiler_and_review_parser_end_to_end(self):
        tracer, report = self.author()
        result = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(review.validate_review(result, self.wp, self.ws, self.task), [])
        self.assertGreater(report["logical_calls"], 1)
        self.assertGreater(result["logical_calls"], 1)
        for call in tracer.calls:
            payload = json.loads(call["messages"][-1]["content"])
            self.assertNotIn("world", payload["requirements"])
            self.assertNotIn("catalogue", payload["requirements"])

    def test_disclosure_exact_read_tampering_rejected_even_with_new_plan_hash(self):
        self.author()
        self.ws.disclosure["transcript"][1]["messages_hash"] = "0" * 64
        self.ws.disclosure["plan_hash"] = d._hash({k: v for k, v in self.ws.disclosure.items() if k != "plan_hash"})
        self.assertTrue(d.validate_plan(self.ws))

    def test_disclosure_original_fact_or_action_edits_invalidate_replay(self):
        self.author()
        original = deepcopy(self.ws.disclosure)
        self.ws.disclosure["transcript"][0]["raw_output"]["ids"] = []
        self.assertTrue(d.validate_plan(self.ws))
        self.ws.disclosure = original
        self.ws.disclosure["author_input"]["catalogue"][0]["value"] = "tampered original"
        self.ws.disclosure["plan_hash"] = d._hash({k: v for k, v in self.ws.disclosure.items() if k != "plan_hash"})
        self.assertTrue(d.validate_plan(self.ws))

    def test_review_action_and_exact_read_hash_edits_invalidate_replay(self):
        self.author()
        original = review.review_world(self.wp, self.ws, ScopedAgentFixture(), self.task)
        self.assertEqual(original["status"], "passed", original)
        for kind in ("hash", "action", "source"):
            report = deepcopy(original)
            if kind == "hash":
                report["transcript"][1]["messages_hash"] = "0" * 64
            elif kind == "action":
                report["transcript"][0]["raw_output"]["ids"] = []
            else:
                report["input_snapshot"]["calendar"][0]["date"] = "2099-01-01"
            self.assertTrue(review.validate_review(report, self.wp, self.ws, self.task), kind)

    def test_reviewer_receipt_replay_and_negative_preserved(self):
        self.author()
        report = review.review_world(self.wp, self.ws, ScopedAgentFixture(reject=True), self.task)
        self.assertEqual(report["status"], "unresolved", report)
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task)[0]["code"], "world_review_not_passed")
        report["status"] = "passed"
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task)[0]["code"], "missing_or_stale_world_review")

    def test_cannot_finish_without_reading_or_silently_hide_missing_refs(self):
        class Premature:
            def __init__(self): self.calls = 0
            def chat_json(self, step, messages, **params):
                self.calls += 1
                return {"action": "finish", "reason": "省略读取"}
        tracer = Premature()
        report = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "error")
        self.assertEqual(tracer.calls, 4)
        self.assertFalse(self.ws.disclosure)
        self.author()
        tracer = Premature()
        report = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "error")
        self.assertGreater(tracer.calls, 4)
        self.assertIn("Invalid world review decision", report["error"])

    def test_premature_finish_forces_next_exact_read_without_certifying(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        keys = list(window.nodes)
        window.read = {keys[0]}
        state = {"issues": [], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "", "revisions": []}
        result, advance = review._review_action(
            {"action": "finish", "reason": "过早结束"}, window, state, payload)
        self.assertIsNone(result)
        self.assertTrue(advance)
        self.assertNotEqual(window.read, set(window.nodes))

    def test_witness_without_world_ref_is_rejected_at_submission_time(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        first = next(iter(window.nodes))
        window.read = {first}
        mechanism = payload["seed"]["mechanisms"][0]["id"]
        state = {"issues": [], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "", "revisions": []}
        action = {"action": "submit", "issues": [], "disclosure_reviews": [],
                  "mechanism_coverage": [{"mechanism_id": mechanism,
                      "status": "witnessed", "refs": [], "reason": "声称已见证",
                      "observed_sequence": "缺少世界节点"}], "note": "当前批意见"}
        with self.assertRaisesRegex(ValueError, "actual world ref"):
            review._review_action(action, window, state, payload)
        self.assertEqual(state["mechanism_coverage"], [])

    def test_error_receipt_reports_original_execution_failure(self):
        self.author()
        report = {"version": review.VERSION, "strategy": "agentic", "status": "error",
                  "error_type": "ValueError", "error": "invalid finish action"}
        errors = review.validate_review(report, self.wp, self.ws, self.task)
        self.assertEqual(errors, [{"code": "world_review_execution_error",
                                  "error_type": "ValueError",
                                  "message": "invalid finish action"}])

    def test_cannot_arrange_unread_fact_and_can_recover_with_actual_read(self):
        class Recover(ScopedAgentFixture):
            def chat_json(self, step, messages, **params):
                if not self.calls:
                    self.calls.append({})
                    return {"action": "arrange", "records": [], "undisclosed": ["f1"], "reason": "未经读取的决定。"}
                return super().chat_json(step, messages, **params)
        report = d.author_plan(self.wp, self.ws, Recover(), self.task)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(self.ws.disclosure["undisclosed"], [])
        self.assertEqual(d.validate_plan(self.ws), [])

    def test_provider_sentinel_stops_without_semantic_retry(self):
        class Broken:
            def __init__(self): self.calls = 0
            def chat_json(self, *args, **kwargs):
                self.calls += 1
                return {"__error__": "offline timeout"}
        tracer = Broken()
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(tracer.calls, 1)
        self.assertEqual(result["logical_calls"], 1)

    def test_inspect_is_not_counted_until_original_content_is_sent(self):
        window = ReadWindow({}, {"a": {"label": "a", "value": "actual"}})
        window.control({"action": "inspect", "ids": ["a"]})
        self.assertEqual(window.read, set())
        self.assertIn("actual", window.messages("test", {})[-1]["content"])
        window.mark_sent()
        self.assertEqual(window.read, {"a"})

    def test_read_window_rejects_oversized_selection_without_losing_prior_state(self):
        window = ReadWindow({}, {key: {"label": key, "value": "x" * 13000} for key in ("a", "b")})
        with self.assertRaises(ValueError):
            window.control({"action": "inspect", "ids": ["a", "b"]})
        self.assertEqual(window.read, set())
        window.control({"action": "inspect", "ids": ["a"]})
        self.assertEqual(set(window.visible), {"a"})

    def test_large_original_node_pages_preserve_exact_bytes_and_require_all_reads(self):
        original = {"label": "large", "value": "原始资料。" * 11000}
        window = ReadWindow({}, {"large": original})
        fragments = []
        count = window.parts("large")
        for part in range(count):
            window.control({"action": "inspect", "ids": ["large"], "part": part})
            window.messages("offline", {})
            fragments.append(window.visible["large"]["serialized_json_fragment"])
            self.assertNotIn("large", window.read)
            window.mark_sent()
        self.assertEqual(json.loads("".join(fragments)), original)
        self.assertEqual(window.read, {"large"})

    def test_input_room_checked_before_accepting_next_inspection(self):
        window = ReadWindow({"frozen": "x" * 108000}, {"a": {"value": "x" * 15000}})
        window.messages("offline", {})
        with self.assertRaisesRegex(ValueError, "total input"):
            window.control({"action": "inspect", "ids": ["a"]})
        self.assertEqual(window.visible, {})

    def test_progress_guard_counts_new_index_pages_but_rejects_page_loop(self):
        window = ReadWindow({}, {str(i): {"value": i} for i in range(130)})
        guard = ProgressGuard("offline reader", limit=2)
        window.messages("offline", {}); window.mark_sent()
        guard.observe(window.progress_marker())
        window.control({"action": "index", "offset": 60})
        window.messages("offline", {}); window.mark_sent()
        guard.observe(window.progress_marker())
        window.control({"action": "index", "offset": 0})
        window.messages("offline", {}); window.mark_sent()
        guard.observe(window.progress_marker())
        with self.assertRaisesRegex(ValueError, "no cumulative progress"):
            guard.observe(window.progress_marker())

    def test_world_reviewer_directory_loop_fails_fast(self):
        self.author()
        class Loop:
            def __init__(self): self.calls = 0
            def chat_json(self, *args, **kwargs):
                self.calls += 1
                return {"action": "index", "offset": 0 if self.calls % 2 else 60}
        tracer = Loop()
        result = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(tracer.calls, 4)
        self.assertIn("delivery is automatic", result["error"])

    def test_world_reviewer_does_not_advance_without_batch_record(self):
        self.author()
        seen = []
        class OneBadThenReview(ScopedAgentFixture):
            def chat_json(inner, step, messages, **params):
                payload = json.loads(messages[-1]["content"])
                if step == review.STEP:
                    seen.append((tuple(payload["unread_ids"]), tuple(payload["exact_reads"])))
                    if len(seen) == 1:
                        return {"action": "continue"}
                return super().chat_json(step, messages, **params)
        result = review.review_world(self.wp, self.ws, OneBadThenReview(), self.task)
        self.assertEqual(result["status"], "passed", result)
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(seen[0][1], seen[1][1])

    def test_validator_feedback_allows_same_identity_revision_with_audit(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        window.read = set(window.nodes)
        ref_id = next(r["ref_id"] for key in window.read
                      for r in window.nodes[key].get("reference_index", []))
        old = {"id": "i1", "finding": "旧意见", "refs": [ref_id],
               "alternative_reading": "旧解释", "disposition": "non_blocking"}
        new = {**old, "finding": "收到校验反馈后的修订意见", "alternative_reading": "修订解释"}
        state = {"issues": [deepcopy(old)], "mechanism_coverage": [],
                 "disclosure_reviews": [], "note": "旧批次", "revisions": []}
        action = {"action": "submit", "issues": [new], "mechanism_coverage": [],
                  "disclosure_reviews": [], "note": "根据 finish 校验错误修订同一意见。"}
        with self.assertRaisesRegex(ValueError, "Repeated submitted review identity"):
            review._review_action(action, window, deepcopy(state), payload)
        _, advance = review._review_action(action, window, state, payload,
            allow_revision=True, revision_reason="finish validator rejected the old citation")
        self.assertTrue(advance)
        self.assertEqual(state["issues"], [new])
        self.assertEqual(state["revisions"][0]["previous"], old)
        self.assertEqual(state["revisions"][0]["replacement"], new)

    def test_unchanged_opinion_can_acknowledge_a_new_batch_but_cannot_loop_after_all_reads(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        keys = list(window.nodes)
        window.read = set(keys[:-1])
        ref_id = next(r["ref_id"] for key in window.read
                      for r in window.nodes[key].get("reference_index", []))
        old = {"id": "i1", "finding": "意见保持", "refs": [ref_id],
               "alternative_reading": "已核对", "disposition": "non_blocking"}
        state = {"issues": [deepcopy(old)], "mechanism_coverage": [],
                 "disclosure_reviews": [], "note": "此前批次", "revisions": []}
        action = {"action": "submit", "issues": [deepcopy(old)],
                  "mechanism_coverage": [], "disclosure_reviews": [],
                  "note": "新交付材料没有改变既有意见。"}
        _, advance = review._review_action(
            action, window, state, payload, allow_revision=True,
            revision_reason="premature finish")
        self.assertTrue(advance)
        self.assertEqual(state["issues"], [old])
        window.read = set(window.nodes)
        with self.assertRaisesRegex(ValueError, "output finish"):
            review._review_action(
                action, window, state, payload, allow_revision=True,
                revision_reason="premature finish")

    def test_migrated_checkpoint_keeps_maximal_compatible_prefix(self):
        old_rows = [{"messages_hash": f"old-{i}", "call": {}, "raw_output": {}}
                    for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / "02_world_review_historical.ckpt.json"
            checkpoint.write_text(json.dumps({
                "identity": "historical", "transcript": old_rows,
                "hash": review._hash(old_rows), "legacy_resume_prefix": 3,
            }), encoding="utf-8")
            tracer = SimpleNamespace(pfile=folder / "prompts.jsonl")
            calls = []
            def session(*args, **kwargs):
                calls.append(deepcopy(kwargs))
                if len(calls) == 1:
                    raise ValueError(
                        "Scoped review exact-read messages or call changed at entry 2")
                return ({}, {"protocol": "test"}, {"step": review.STEP, "params": {}},
                        [], {"decision": "accept"})
            with patch.object(review, "_scoped_review_session", side_effect=session), \
                 patch.object(review, "_parse", return_value={"status": "passed"}):
                result = review._review_agentic(
                    self.wp, self.ws, tracer, self.task, None, None)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(calls[0]["resume"]), 3)
            self.assertEqual(len(calls[1]["resume"]), 2)
            self.assertEqual(calls[1]["legacy_resume_prefix"], 2)
            self.assertEqual(calls[1]["compatibility_resume_length"], 2)

    def test_reviewer_accepts_bounded_state_echo_and_long_batch_note(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        state = {"issues": [], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "", "revisions": []}
        action = {"action": "submit", "issues": [], "mechanism_coverage": [],
                  "disclosure_reviews": [], "note": "审" * 1436,
                  "revision_count": 0,
                  "working_state": review._scoped_review_state(state)}
        _, advance = review._review_action(action, window, state, payload)
        self.assertTrue(advance)
        self.assertEqual(len(state["note"]), 1436)

        bad = deepcopy(action)
        bad["revision_count"] = 3
        _, advance = review._review_action(bad, window, {"issues": [], "mechanism_coverage": [],
            "disclosure_reviews": [], "note": "", "revisions": []}, payload)
        self.assertTrue(advance)

        malformed = deepcopy(action)
        malformed["revision_count"] = "stale"
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            review._review_action(malformed, window, {"issues": [], "mechanism_coverage": [],
                "disclosure_reviews": [], "note": "", "revisions": []}, payload)

        with self.assertRaisesRegex(ValueError, "Unknown scoped review submission fields"):
            review._review_action(action, window, {"issues": [], "mechanism_coverage": [],
                "disclosure_reviews": [], "note": "", "revisions": []}, payload, legacy=True)

    def test_premature_consolidated_final_is_saved_as_batch_and_keeps_reading(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        state = {"issues": [], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "", "revisions": []}
        raw = {"mechanism_coverage": [], "disclosure_reviews": [], "issues": [],
               "repair_targets": {"intrinsic": [], "structure": False, "disclosure": False},
               "limitations": "尚有原始节点未读。", "reason": "当前批综合判断。",
               "decision": "unresolved"}
        result, advance = review._review_action(raw, window, state, payload)
        self.assertIsNone(result)
        self.assertTrue(advance)
        self.assertEqual(state["note"], "当前批综合判断。")

    def test_complete_consolidated_final_withdraws_progress_only_issue(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        window.read = set(window.nodes)
        ref_id = next(r["ref_id"] for key in window.read
                      for r in window.nodes[key].get("reference_index", []))
        early = {"id": "progress_only", "finding": "仍有材料未读。", "refs": [ref_id],
                 "alternative_reading": "全部读取后重新判断。", "disposition": "unresolved"}
        state = {"issues": [early], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "早期进度意见", "revisions": []}
        raw = {"mechanism_coverage": [], "disclosure_reviews": [], "issues": [],
               "repair_targets": {"intrinsic": [], "structure": False, "disclosure": False},
               "limitations": "全部原始节点已读取。", "reason": "最终没有阻断问题。",
               "decision": "accept"}
        with patch.object(review, "_parse") as parse:
            result, advance = review._review_action(raw, window, state, payload)
        self.assertFalse(advance)
        self.assertEqual(result["issues"], [])
        self.assertEqual(state["issues"], [])
        self.assertEqual(state["revisions"][-1]["previous"], early)
        self.assertIsNone(state["revisions"][-1]["replacement"])
        parse.assert_called_once()

    def test_complete_final_can_echo_compact_disclosure_working_state(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        window.read = set(window.nodes)
        record = payload["disclosure_record_contexts"][0]
        full = {"record_id": record["record_id"], "understanding": "已提交完整理解。",
                "refs": [record["record_ref_id"]], "reason": "完整依据已在前一批提交。",
                "status": "compatible"}
        state = {"issues": [], "mechanism_coverage": [],
                 "disclosure_reviews": [deepcopy(full)], "note": "此前批次",
                 "revisions": []}
        raw = {"mechanism_coverage": [],
               "disclosure_reviews": [{"record_id": full["record_id"],
                                        "status": "compatible"}],
               "issues": [],
               "repair_targets": {"intrinsic": [], "structure": False,
                                  "disclosure": False},
               "limitations": "全部节点已读。", "reason": "确认已提交意见。",
               "decision": "accept"}
        with patch.object(review, "_parse") as parse:
            result, advance = review._review_action(raw, window, state, payload)
        self.assertFalse(advance)
        self.assertEqual(result["disclosure_reviews"], [full])
        parse.assert_called_once()

    def test_finish_reaffirms_consolidated_opinion_after_last_transport_read(self):
        self.author()
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        keys = list(window.nodes)
        window.read = set(keys[:-1])
        ref_id = next(r["ref_id"] for key in window.read
                      for r in window.nodes[key].get("reference_index", []))
        early = {"id": "unread_progress", "finding": "仍有节点未读。", "refs": [ref_id],
                 "alternative_reading": "读完后可撤回。", "disposition": "unresolved"}
        state = {"issues": [early], "mechanism_coverage": [], "disclosure_reviews": [],
                 "note": "早期进度", "revisions": []}
        consolidated = {"mechanism_coverage": [], "disclosure_reviews": [], "issues": [],
                        "repair_targets": {"intrinsic": [], "structure": False,
                                           "disclosure": False},
                        "limitations": "完整意见已形成。", "reason": "当前结论通过。",
                        "decision": "accept"}
        result, advance = review._review_action(consolidated, window, state, payload)
        self.assertIsNone(result)
        self.assertTrue(advance)
        window.read = set(window.nodes)
        finish = {"action": "finish", "decision": "accept",
                  "repair_targets": deepcopy(consolidated["repair_targets"]),
                  "reason": "读完最后节点后确认通过。", "limitations": "全部节点已读。"}
        with patch.object(review, "_parse") as parse:
            result, advance = review._review_action(finish, window, state, payload)
        self.assertFalse(advance)
        self.assertEqual(result["issues"], [])
        self.assertEqual(state["issues"], [])
        self.assertIsNone(state["revisions"][-1]["replacement"])
        parse.assert_called_once()

    def test_raised_provider_error_records_entered_call(self):
        class Broken:
            def chat_json(self, *args, **kwargs):
                raise TimeoutError("local test deadline")
        result = d.author_plan(self.wp, self.ws, Broken(), self.task)
        self.assertTrue(result["caller_entered"])
        self.assertEqual(result["logical_calls"], 1)
        self.assertEqual(result["attempts"][0]["error_type"], "TimeoutError")

    def test_feedback_omits_repeated_prompt_transcript_preserves_opinion(self):
        feedback = {"version": review.VERSION, "raw_output": {"reason": "原业务意见"},
                    "locations": {"issues": [{"source_value": "原事实"}]},
                    "transcript": [{"messages": "huge repeated inputs"}]}
        projected = d._feedback_projection(feedback)
        self.assertNotIn("transcript", projected)
        self.assertEqual(projected["raw_output"], feedback["raw_output"])
        self.assertEqual(projected["locations"], feedback["locations"])


if __name__ == "__main__":
    unittest.main()
