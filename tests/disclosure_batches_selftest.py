"""No-provider regression of direct disclosure on small and historical worlds.

Scripted authors verify delivery/recovery, never semantic benchmark quality.
"""
from collections import defaultdict
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
from pipeline import disclosure as d, disclosure_batches as batch, factory, world_semantics
from pipeline.world_state import WorldState, Timeline, Op, SET
from disclosure_selftest import data
from world_agent_factory_selftest import LocalRun
from world_scoped_agents_selftest import ScopedAgentFixture


class DirectFixture:
    def __init__(self, hook=None):
        self.calls, self.hook = [], hook

    def chat_json(self, step, messages, **params):
        self.calls.append({"step": step, "messages": deepcopy(messages), "params": params})
        body = json.loads(messages[-1]["content"])
        if self.hook:
            response = self.hook(body, len(self.calls))
            if response is not None:
                if isinstance(response, Exception):
                    raise response
                return response
        grouped = defaultdict(list)
        facts = {f["ref"]: f for f in body["exact_facts"]}
        for ref in body["target_refs"]:
            f = facts[ref]
            grouped[f.get("entity", (f.get("event") or {}).get("id", ref))].append(ref)
        return {"records": [{"session": body["requirements"]["calendar"][-1]["session"],
                            "refs": refs, "channel": "离线接口测试",
                            "acquisition_context": "离线固定作者汇总原始历史；此响应不代表业务审阅通过。"}
                           for refs in grouped.values()], "undisclosed": [], "reason": "Offline fixture only"}


class Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("No network")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(STRUCTURE_MODEL="cheap-author", REVIEWER_MODEL="cheap-review",
                                                           DISCLOSURE_FORMAT_ATTEMPTS=4)}).start()
        self.wp, self.ws, self.task = data()
        self.wp["world_generation"] = {"strategy": "agentic", "disclosure_strategy": "direct_batches_v1"}

    def test_direct_delivery_and_original_compiler_keep_exact_history(self):
        before = d._world(self.ws)
        tracer = DirectFixture()
        result = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["logical_calls"], 1)
        self.assertEqual(d.validate_plan(self.ws), [])
        self.assertEqual(d._world(self.ws), before)
        body = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertGreater(len(body["exact_facts"]), 0)
        self.assertNotIn("exact_reads", body)
        self.assertNotIn("refs_requiring_arrangement", tracer.calls[0]["messages"][0]["content"])
        self.assertEqual(tracer.calls[0]["params"]["response_format"], {"type": "json_object"})

    def test_missing_channel_gets_full_bad_group_and_local_feedback(self):
        bad = {}
        def hook(body, number):
            if number == 1:
                bad.update(records=[{"session": 0, "refs": body["target_refs"]}], undisclosed=[], reason="bad fields")
                return deepcopy(bad)
            self.assertEqual(body["previous_output"], bad)
            self.assertIn("records[0]", body["validation_error"])
            self.assertIn("channel", body["validation_error"])
            self.assertIn("acquisition_context", body["validation_error"])
        tracer = DirectFixture(hook)
        report = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(len(tracer.calls), 2)
        self.assertFalse(d.validate_plan(self.ws))

    def test_unread_reference_cannot_be_published_and_missing_refs_not_auto_hidden(self):
        def hook(body, number):
            return {"records": [], "undisclosed": [], "reason": "No decision"}
        report = d.author_plan(self.wp, self.ws, DirectFixture(hook), self.task)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["accepted_groups"], 0)
        self.assertFalse(self.ws.disclosure)
        with self.assertRaisesRegex(ValueError, "only exact_facts"):
            batch._merge(self.ws, {"records": [], "undisclosed": []},
                         {"records": [], "undisclosed": ["f999"], "reason": "x"}, ["f999"], {"f1"})

    def test_more_than_twelve_refs_delivered_without_read_calls(self):
        for i in range(35):
            self.ws.entities[f"entity{i}"] = {"value": Timeline([Op(0, self.ws.date_of_session(0), SET, i)])}
        tracer = DirectFixture()
        self.assertEqual(d.author_plan(self.wp, self.ws, tracer, self.task)["status"], "ready")
        body = json.loads(tracer.calls[0]["messages"][-1]["content"])
        self.assertGreater(len(body["target_refs"]), 12)
        self.assertEqual(len(tracer.calls), 1)

    def test_cross_entity_context_requested_by_author_without_twelve_id_cap(self):
        for i in range(40):
            self.ws.entities[f"entity{i}"] = {"value": Timeline([Op(0, self.ws.date_of_session(0), SET, i)])}
        patch.object(batch, "FACT_CHARS", 1800).start()
        requested = []
        def hook(body, number):
            if number == 1:
                requested.extend(row["ref"] for row in body["remaining_index"][:15])
                return {"need_refs": requested, "reason": "Check related original facts"}
            if number == 2:
                self.assertTrue(set(requested) <= {r["ref"] for r in body["exact_facts"]})
        tracer = DirectFixture(hook)
        report = d.author_plan(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(len(requested), 15)
        self.assertFalse(d.validate_plan(self.ws))

    def test_input_capacity_trims_optional_context_and_keeps_required_targets(self):
        payload = d._payload(self.wp, self.ws, self.task, None)
        refs = payload["refs_requiring_arrangement"]
        targets, context = refs[:1], refs[1:]
        raw = {"records": [], "undisclosed": [], "reason": "pending"}
        baseline = batch._messages(payload, raw, targets, [], None)
        limit = len(baseline[-1]["content"])
        with patch.object(batch, "INPUT_CHARS", limit):
            messages, retained = batch._fit_messages(payload, raw, targets, context, None)
            self.assertEqual(retained, [])
            self.assertEqual(json.loads(messages[-1]["content"])["target_refs"], targets)
            with self.assertRaises(batch.InputCapacityError):
                batch._fit_messages(payload, raw, targets, context, None, required=context)

    def test_pending_index_keeps_only_undecided_references(self):
        payload = d._payload(self.wp, self.ws, self.task, None)
        refs = payload["refs_requiring_arrangement"]
        raw = {"records": [{"session": 0, "refs": refs[:1], "channel": "原文",
                            "acquisition_context": "已由作者安排"}],
               "undisclosed": refs[1:2], "reason": "已有决定"}
        targets = refs[2:3]
        legacy = json.loads(batch._messages(payload, raw, targets, [], None)[-1]["content"])
        compact = json.loads(batch._messages(payload, raw, targets, [], None,
                                             index_scope="pending_only")[-1]["content"])
        self.assertEqual(compact["accepted_records"], legacy["accepted_records"])
        self.assertEqual(compact["accepted_undisclosed"], legacy["accepted_undisclosed"])
        self.assertNotIn(refs[0], {row["ref"] for row in compact["remaining_index"]})
        self.assertNotIn(refs[1], {row["ref"] for row in compact["remaining_index"]})
        self.assertIn(refs[0], {row["ref"] for row in legacy["remaining_index"]})

    def test_legacy_checkpoint_migrates_without_rewriting_accepted_decisions(self):
        for i in range(40):
            self.ws.entities[f"entity{i}"] = {"value": Timeline([Op(0, self.ws.date_of_session(0), SET, i)])}
        payload = d._payload(self.wp, self.ws, self.task, None)
        refs = payload["refs_requiring_arrangement"]
        targets = refs[:1]
        messages = batch._messages(payload, {"records": [], "undisclosed": [],
                                              "reason": "尚未完成安排"}, targets, [], None)
        output = DirectFixture().chat_json(d.STEP, messages)
        part = {"target_refs": targets, "context_refs": [], "repair": None,
                "messages_hash": d._hash(messages), "output": output}
        self.assertEqual(batch._replay(payload, self.ws, [part])["records"][0]["refs"], targets)
        with patch.object(batch, "FACT_CHARS", 700), tempfile.TemporaryDirectory() as directory:
            current = batch._binding(self.wp, self.ws, payload, "cheap-author")
            legacy = deepcopy(current)
            legacy["implementation"] = "earlier-source-implementation"
            old_path = Path(directory) / ("02_disclosure_checkpoint_" + d._hash(legacy)[:20] + ".json")
            batch._save(old_path, {"binding": legacy, "parts": [part],
                                   "attempts": [], "status": "building"})
            old_bytes = old_path.read_bytes()
            tracer = DirectFixture()
            result = d.author_plan(self.wp, self.ws, tracer, self.task, checkpoint_dir=directory)
            self.assertEqual(result["status"], "ready", result)
            self.assertEqual(result["migrated_checkpoint"]["accepted_groups"], 1)
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(self.ws.disclosure["parts"][0], part)
            self.assertTrue(all(row.get("index_scope") == "pending_only"
                                for row in self.ws.disclosure["parts"][1:]))
            self.assertEqual(d.validate_plan(self.ws), [])

    def test_tighter_transport_limit_preserves_exact_completed_decisions(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        original = deepcopy(self.ws.disclosure)
        with patch.object(batch, "INPUT_CHARS", 1):
            self.assertEqual(d.validate_plan(self.ws), [])
            self.assertEqual(self.ws.disclosure, original)
        self.ws.disclosure["parts"][0]["output"]["records"][0]["channel"] = "tampered"
        with patch.object(batch, "INPUT_CHARS", 1):
            self.assertTrue(d.validate_plan(self.ws))

    def test_implementation_fingerprint_drift_replays_decisions_without_reauthoring(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        original = deepcopy(self.ws.disclosure)
        self.ws.disclosure["batch_binding"]["implementation"] = "historical-code-hash"
        self.ws.disclosure["batch_binding"]["dependencies"] = {"historical.py": "historical-code-hash"}
        self.ws.disclosure["binding"]["implementation_hashes"] = {"historical.py": "historical-code-hash"}
        saved = {key: deepcopy(value) for key, value in self.ws.disclosure.items() if key != "plan_hash"}
        self.ws.disclosure["plan_hash"] = d._hash(saved)
        self.assertEqual(d.validate_plan(self.ws), [])
        self.assertEqual(self.ws.disclosure["raw_output"], original["raw_output"])

    def test_protocol_fingerprint_drift_replays_decisions_without_reauthoring(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        original = deepcopy(self.ws.disclosure)
        self.ws.disclosure["binding"]["protocol_hash"] = "historical-protocol-hash"
        self.ws.disclosure["batch_binding"]["protocol_hash"] = "historical-protocol-hash"
        saved = {key: deepcopy(value) for key, value in self.ws.disclosure.items() if key != "plan_hash"}
        self.ws.disclosure["plan_hash"] = d._hash(saved)
        self.assertEqual(d.validate_plan(self.ws), [])
        self.assertEqual(self.ws.disclosure["raw_output"], original["raw_output"])

    def test_world_review_repairs_only_named_disclosure_records(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        before = deepcopy(self.ws.disclosure["raw_output"])
        review = {"status": "unresolved", "repair_targets": {"disclosure": True},
                  "issues": [{"id": "i1", "finding": "d1说明范围超过refs", "disposition": "repair"}],
                  "disclosure_reviews": [{"record_id": "d1", "status": "repair",
                                           "reason": "说明范围超过refs"}]}
        def local_patch(body, number):
            self.assertEqual(body["requested_record_indices"], [0])
            replacement = deepcopy(body["records"][0]["record"])
            replacement["acquisition_context"] = "仅按本记录所列原始引用公开，不扩大范围。"
            return {"record_updates": [{"record_index": 0, "record": replacement}],
                    "reason": "收敛公开范围"}
        report = batch.repair(self.wp, self.ws, DirectFixture(local_patch), review)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(report["repaired_record_ids"], ["d1"])
        self.assertEqual(self.ws.disclosure["raw_output"]["records"][1:], before["records"][1:])
        self.assertEqual(self.ws.disclosure["raw_output"]["records"][0]["refs"], before["records"][0]["refs"])
        self.assertEqual(d.validate_plan(self.ws), [])
        again = batch.repair(self.wp, self.ws, DirectFixture(local_patch), review)
        self.assertEqual(again["status"], "error")
        self.assertIn("already attempted", again["error"])

    def test_all_repair_issues_include_unresolved_disclosure_rows(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        before = deepcopy(self.ws.disclosure["raw_output"])
        review = {"status": "unresolved", "repair_targets": {"disclosure": True},
                  "issues": [{"id": "i1", "finding": "公开说明均可由作者收窄", "disposition": "repair"}],
                  "disclosure_reviews": [
                      {"record_id": "d1", "status": "repair", "reason": "说明过宽"},
                      {"record_id": "d2", "status": "unresolved", "reason": "需要作者澄清"},
                  ]}
        def local_patch(body, number):
            self.assertEqual(body["requested_record_indices"], [0, 1])
            updates = []
            for row in body["records"]:
                replacement = deepcopy(row["record"])
                replacement["acquisition_context"] = "仅说明本记录所列引用的取得途径。"
                updates.append({"record_index": row["record_index"], "record": replacement})
            return {"record_updates": updates, "reason": "收窄公开说明"}
        report = batch.repair(self.wp, self.ws, DirectFixture(local_patch), review)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(report["repaired_record_ids"], ["d1", "d2"])
        self.assertEqual(self.ws.disclosure["raw_output"]["records"][2:], before["records"][2:])

    def test_publication_repair_can_move_the_record_without_changing_truth(self):
        def original_arrangement(body, number):
            return {"records": [{"session": 0, "refs": body["target_refs"],
                "channel": "离线接口测试", "acquisition_context": "原作者的公开安排。"}],
                "undisclosed": [], "reason": "Offline arrangement fixture"}
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(original_arrangement),
                                      self.task)["status"], "ready")
        before = deepcopy(self.ws.disclosure)
        truth = d._world(self.ws)
        review = {"status": "failed", "repair_targets": {"disclosure": True},
            "issues": [{"id": "i1", "finding": "这条安排需要延后公开，不能只改说明。",
                        "disposition": "repair"}],
            "disclosure_reviews": [{"record_id": "d1", "status": "repair",
                                    "reason": "由原公开作者选择可行公开期。"}]}
        def move_record(body, number):
            replacement = deepcopy(body["records"][0]["record"])
            replacement["session"] = self.ws.n_sessions - 1
            replacement["acquisition_context"] = "原作者选择在窗口末期公开同一组原引用。"
            return {"record_updates": [{"record_index": 0, "record": replacement}],
                    "reason": "延后公开；离线响应不代表语义正确。"}
        author = DirectFixture(move_record)
        report = batch.repair(self.wp, self.ws, author, review)
        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(self.ws.disclosure["records"][0]["session"], self.ws.n_sessions - 1)
        self.assertEqual(self.ws.disclosure["parts"][:-1], before["parts"])
        self.assertEqual(self.ws.disclosure["parts"][-1]["protocol"], "publication-repair/v2")
        self.assertEqual(d._world(self.ws), truth)
        self.assertEqual(d.validate_plan(self.ws), [])
        no_more_calls = DirectFixture(lambda *_: AssertionError("Attempt allowance must not grow"))
        self.assertEqual(batch.repair(self.wp, self.ws, no_more_calls, review)["status"], "error")
        self.assertFalse(no_more_calls.calls)

    def test_publication_repair_keeps_original_compiler_and_legacy_scope(self):
        self.assertEqual(d.author_plan(self.wp, self.ws, DirectFixture(), self.task)["status"], "ready")
        original = deepcopy(self.ws.disclosure["raw_output"])
        replacement = deepcopy(original["records"][0])
        replacement["session"] = 0
        output = {"record_updates": [{"record_index": 0, "record": replacement}],
                  "reason": "Offline transport test"}
        with self.assertRaisesRegex(ValueError, "preserve session"):
            batch._merge_semantic_repair(self.ws, original, output, [0])
        replacement["session"] = self.ws.n_sessions
        with self.assertRaises(d.PlanFormatError):
            batch._merge_semantic_repair(self.ws, original, output, [0], protocol="publication-repair/v2")
        replacement["session"] = 0
        replacement["channel"] = "changed channel"
        with self.assertRaisesRegex(ValueError, "preserve refs and channel"):
            batch._merge_semantic_repair(self.ws, original, output, [0], protocol="publication-repair/v2")
        replacement["channel"] = original["records"][0]["channel"]
        replacement["refs"] = original["records"][1]["refs"]
        with self.assertRaisesRegex(ValueError, "preserve refs and channel"):
            batch._merge_semantic_repair(self.ws, original, output, [0], protocol="publication-repair/v2")
        self.assertEqual(self.ws.disclosure["raw_output"], original)

    def test_partial_success_is_saved_and_resume_requests_only_remaining_refs(self):
        first_ref = []
        def hook(body, number):
            if number == 1:
                first_ref.append(body["target_refs"][0])
                return {"records": [{"session": 0, "refs": first_ref[:], "channel": "材料", "acquisition_context": "作者说明"}],
                        "undisclosed": [], "reason": "完成第一部分"}
            return TimeoutError("Interrupted after partial progress")
        with tempfile.TemporaryDirectory() as directory:
            result = d.author_plan(self.wp, self.ws, DirectFixture(hook), self.task, checkpoint_dir=directory)
            self.assertEqual(result["accepted_groups"], 1, result)
            saved = json.loads(Path(result["checkpoint"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["parts"][0]["output"]["records"][0]["refs"], first_ref)
            resumed = DirectFixture()
            result = d.author_plan(self.wp, self.ws, resumed, self.task, checkpoint_dir=directory)
            self.assertEqual(result["status"], "ready", result)
            body = json.loads(resumed.calls[0]["messages"][-1]["content"])
            self.assertNotIn(first_ref[0], body["target_refs"])
            self.assertEqual(self.ws.disclosure["raw_output"]["records"][0], saved["parts"][0]["output"]["records"][0])
            self.assertFalse(d.validate_plan(self.ws))

    def test_explicit_updates_preserve_other_records_and_cannot_lose_coverage(self):
        refs = d._payload(self.wp, self.ws, self.task, None)["refs_requiring_arrangement"]
        def record(refs, channel="原文"):
            return {"session": 0, "refs": refs, "channel": channel, "acquisition_context": "作者说明"}
        original = {"records": [record(refs[:1]), record(refs[1:2])], "undisclosed": [], "reason": "此前安排"}
        output = {"record_updates": [{"record_index": 0, "record": record(refs[:1], "明确修订渠道")}],
                  "records": [record(refs[2:])], "undisclosed": [], "reason": "修订并补充"}
        result = batch._merge(self.ws, original, output, refs[2:], set(refs))
        self.assertEqual(result["records"][0]["channel"], "明确修订渠道")
        self.assertEqual(result["records"][1], original["records"][1])
        self.assertEqual(original["records"][0]["channel"], "原文")
        bad = deepcopy(output)
        bad["record_updates"][0]["record"] = record(refs[2:])
        with self.assertRaisesRegex(ValueError, "lost earlier decisions"):
            batch._merge(self.ws, original, bad, refs[2:], set(refs))

    def test_historical_v7_real_replies_complete_by_append_without_inventing_content(self):
        root = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not root.exists():
            self.skipTest("Local historical run is optional")
        wp = json.loads((root / "01_whitepaper.json").read_text(encoding="utf-8"))
        task = json.loads((root / "00_input.json").read_text(encoding="utf-8"))
        ws = WorldState.from_dict(json.loads((root / "02_world_candidate.json").read_text(encoding="utf-8")))
        before = d._world(ws)
        state = json.loads(next(root.glob("02_disclosure_checkpoint_*.json")).read_text(encoding="utf-8"))
        failed = [a["raw_output"] for a in state["attempts"] if a["part"] == 3]
        replies = [p["output"] for p in state["parts"]] + failed[:2]
        tracer = DirectFixture(lambda body, number: deepcopy(replies[number - 1]))
        result = d.author_plan(wp, ws, tracer, task)
        self.assertEqual(result["status"], "ready", result)
        self.assertFalse(d.validate_plan(ws))
        self.assertEqual(d._world(ws), before)
        self.assertEqual(ws.disclosure["raw_output"]["records"], [r for out in replies for r in out["records"]])
        self.assertFalse(d._missing_refs(ws, ws.disclosure["raw_output"]))
        self.assertEqual(len(tracer.calls), 5)
        print(json.dumps({"v7_original_model_replies_replayed": 5, "required_refs": 442,
                          "all_covered": True, "new_provider_calls": 0, "semantic_review": "not_claimed"}))

    def test_interrupted_group_resume_preserves_completed_work(self):
        patch.object(batch, "FACT_CHARS", 500).start()
        with tempfile.TemporaryDirectory() as directory:
            def interrupted(body, number):
                return TimeoutError("injected provider interruption") if number == 2 else None
            report = d.author_plan(self.wp, self.ws, DirectFixture(interrupted), self.task, checkpoint_dir=directory)
            self.assertEqual(report["status"], "error", report)
            checkpoint = Path(report["checkpoint"])
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            first = deepcopy(saved["parts"])
            self.assertEqual(len(first), 1)
            resumed = DirectFixture()
            report = d.author_plan(self.wp, self.ws, resumed, self.task, checkpoint_dir=directory)
            self.assertEqual(report["status"], "ready", report)
            self.assertEqual(self.ws.disclosure["parts"][:1], first)
            self.assertEqual(self.ws.disclosure["raw_output"]["reason"], "\n".join(
                part["output"]["reason"] for part in self.ws.disclosure["parts"]))
            no_calls = DirectFixture(lambda *_: AssertionError("Completed work must be replayed offline"))
            self.assertEqual(d.author_plan(self.wp, self.ws, no_calls, self.task, checkpoint_dir=directory)["status"], "ready")
            self.assertFalse(no_calls.calls)
            self.assertFalse(d.validate_plan(self.ws))

    def test_checkpoint_corruption_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            result = d.author_plan(self.wp, self.ws, DirectFixture(), self.task, checkpoint_dir=directory)
            path = Path(result["checkpoint"])
            saved = json.loads(path.read_text(encoding="utf-8")); saved["parts"][0]["output"]["records"][0]["channel"] = "edited"
            path.write_text(json.dumps(saved), encoding="utf-8"); before = path.read_bytes()
            tracer = DirectFixture()
            result = d.author_plan(self.wp, self.ws, tracer, self.task, checkpoint_dir=directory)
            self.assertEqual(result["status"], "error")
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(tracer.calls)

    def test_exact_input_and_world_tampering_invalidate_plan(self):
        d.author_plan(self.wp, self.ws, DirectFixture(), self.task)
        original = deepcopy(self.ws.disclosure)
        self.ws.disclosure["parts"][0]["messages_hash"] = "0" * 64
        self.assertTrue(d.validate_plan(self.ws))
        self.ws.disclosure = original
        self.ws.entities["secret_new"] = {"v": Timeline([Op(0, self.ws.date_of_session(0), SET, "changed")])}
        self.assertTrue(d.validate_plan(self.ws))

    def test_downstream_proposals_keep_semantics_and_omit_batch_audit(self):
        from pipeline.process_proposals import _author_world
        d.author_plan(self.wp, self.ws, DirectFixture(), self.task)
        original = self.ws.to_dict()
        projected = _author_world(original)
        for key in ("parts", "batch_binding", "source_inputs", "author_input"):
            self.assertNotIn(key, projected["disclosure"])
        for key in ("records", "catalogue", "undisclosed", "raw_output"):
            self.assertEqual(projected["disclosure"][key], original["disclosure"][key])

    def test_negative_review_is_preserved_as_recoverable_world_warning(self):
        from world_semantics_selftest import fixture
        self.wp["seed_contract"] = fixture()[0]["seed_contract"]
        with tempfile.TemporaryDirectory() as directory:
            run = LocalRun(directory, self.wp)
            class Combined:
                author, reviewer = DirectFixture(), ScopedAgentFixture(reject=True)
                def chat_json(inner, step, messages, **params):
                    return (inner.author if step == d.STEP else inner.reviewer).chat_json(step, messages, **params)
            run.tracer = Combined()
            with patch.object(factory, "validate_seed_identity"), patch.object(factory, "validate_seed_world", return_value={"passed": True}), \
                 patch.object(factory, "build_world", return_value=self.ws), patch.object(factory, "_prepare_lines"):
                factory.stage_world(run)
            self.assertTrue(run.has("02_world.json"))
            self.assertTrue(run.has(world_semantics.WARNING_ARTIFACT))
            result = run.read("02_world_review_attempts.json")["attempts"][-1]
            self.assertEqual(result["status"], "unresolved", result.get("error"))
            self.assertTrue(list(run.dir.glob("02_disclosure_checkpoint_*.json")))

    def test_historical_large_world_delivery_and_exact_failed_shape_repair(self):
        root = ROOT / "output/experiment_snapshots/insurance_large_20260919_v6/output/runs/insurance_large_20260919_v6_01_insurance_r001"
        if not root.exists():
            self.skipTest("Local historical run is optional")
        wp = json.loads((root / "01_whitepaper.json").read_text(encoding="utf-8"))
        wp["world_generation"]["disclosure_strategy"] = "direct_batches_v1"
        ws = WorldState.from_dict(json.loads((root / "02_world_candidate.json").read_text(encoding="utf-8")))
        task = json.loads((root / "00_input.json").read_text(encoding="utf-8"))
        failure = json.loads((root / "02_disclosure_plan_attempts.json").read_text(encoding="utf-8"))["attempts"][0]
        bad = failure["attempts"][-1]["raw_output"]
        def inject(body, number):
            return deepcopy(bad) if number == 1 else None
        tracer = DirectFixture(inject)
        with tempfile.TemporaryDirectory() as directory:
            result = d.author_plan(wp, ws, tracer, task, checkpoint_dir=directory)
        self.assertEqual(result["status"], "ready", result.get("error"))
        self.assertFalse(d.validate_plan(ws))
        self.assertLess(len(tracer.calls), 30)
        self.assertFalse(d._missing_refs(ws, ws.disclosure["raw_output"]))
        self.assertEqual(len(d._payload(wp, ws, task, None)["refs_requiring_arrangement"]), 453)
        self.assertIn("previous_output", json.loads(tracer.calls[1]["messages"][-1]["content"]))
        print(json.dumps({"historical_world_entities": len(ws.entities), "offline_author_calls": len(tracer.calls),
                          "groups": result["accepted_groups"], "provider_calls": 0,
                          "scope": "Scripted transport and recovery verification only"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
