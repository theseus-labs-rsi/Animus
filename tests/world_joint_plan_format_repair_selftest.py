"""Joint author JSON-shape correction; real Tracer, no network or saved cases."""
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
from pipeline import world_joint_plan as joint
from pipeline.run import Tracer

GOOD = {"plan": "Keep a stable object and a legitimate untriggered condition.", "limitations": "A proposal, not evidence."}
BAD = {"plan": GOOD["plan"], "limitation": GOOD["limitations"]}
WP = {"frozen_task": "Do not change business content to fix JSON structure."}
CONTEXT = {"task": "Synthetic original task", "calendar": [{"session": 0, "date": "2026-01-01"}], "existing_world": {}}


def run(outputs):
    calls = []; values = list(outputs)
    def provider(messages, **params):
        calls.append({"messages": deepcopy(messages), "parameters": deepcopy(params)})
        output = values.pop(0)
        if isinstance(output, Exception):
            raise output
        return deepcopy(output)
    with tempfile.TemporaryDirectory(prefix="joint-format-offline-") as tmp:
        tracer = Tracer(SimpleNamespace(dir=Path(tmp)))
        with patch.dict(sys.modules, {"config": SimpleNamespace(chat_json=provider)}):
            result = joint.author_plan(WP, CONTEXT, tracer, model="offline-mini", max_tokens=2048)
        logs = [json.loads(line) for line in (Path(tmp)/"prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    return result, calls, logs


def rehash(result):
    result["binding"]["attempts_hash"] = joint.digest(result["attempts"])
    result["binding"]["messages_hash"] = joint.digest(result["messages"])
    result["binding"]["raw_hash"] = joint.digest(result["raw_output"])
    result["plan_hash"] = joint.digest({k:v for k,v in result.items() if k != "plan_hash"})


class Tests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def test_valid_original_once_same_business_input_and_json_mode(self):
        report, calls, logs = run([GOOD])
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(json.loads(calls[0]["messages"][1]["content"]), CONTEXT)
        self.assertTrue(calls[0]["messages"][0]["content"].startswith(joint.SYSTEM))
        self.assertEqual(calls[0]["parameters"], {"model":"offline-mini", "temperature":0.4,
            "max_tokens":2048, "retries":1, "strict_json":True, "response_format":{"type":"json_object"}})
        self.assertEqual(joint.validate_plan(report, WP, CONTEXT), [])
        self.assertEqual(report["raw_output"], logs[0]["output"])

    def test_singular_key_model_corrects_once_original_raw_unchanged(self):
        before = deepcopy(BAD)
        report, calls, logs = run([BAD, GOOD])
        self.assertEqual(BAD, before)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["attempts"][0]["raw_output"], BAD)
        self.assertEqual(report["attempts"][0]["issues"], {"missing_keys":["limitations"], "unexpected_keys":["limitation"]})
        self.assertEqual(json.loads(calls[1]["messages"][2]["content"]), BAD)
        self.assertEqual(calls[1]["messages"][:2], calls[0]["messages"])
        self.assertEqual(report["messages"], calls[1]["messages"])
        self.assertEqual(report["binding"]["messages_hash"], joint.digest(calls[1]["messages"]))
        self.assertEqual([a["raw_output"] for a in report["attempts"]], [l["output"] for l in logs])
        self.assertEqual(joint.validate_plan(report, WP, CONTEXT), [])
        self.assertNotIn("format_repair", joint.shared_prompt(report))
        self.assertEqual(report["raw_output"], GOOD)

    def test_nonobject_wrong_type_blank_and_extra_all_shape_only(self):
        for value in ([], {"plan":3,"limitations":"none"}, {"plan":" ","limitations":"none"}, {**GOOD,"extra":"field"}):
            with self.subTest(value=value):
                report, calls, _ = run([value, GOOD])
                self.assertEqual(len(calls), 2)
                self.assertEqual(report["status"], "ready")
                self.assertEqual(joint.validate_plan(report, WP, CONTEXT), [])

    def test_second_invalid_stops_two_no_alias_no_third(self):
        report, calls, _ = run([BAD, BAD, GOOD])
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["raw_output"], BAD)
        self.assertTrue(joint.validate_plan(report, WP, CONTEXT))

    def test_real_tracer_provider_or_parser_error_not_retried(self):
        for value in (TimeoutError("timeout"), ValueError("invalid JSON"), {"__error__":"execution failed"}):
            with self.subTest(value=value):
                report, calls, logs = run([value, GOOD])
                self.assertEqual(len(calls), 1)
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["attempts"][0]["status"], "execution_error")
                self.assertEqual(report["raw_output"], logs[0]["output"])
                self.assertFalse(logs[0]["ok"])

    def test_second_execution_error_stops_two_and_preserves_failed_raw(self):
        report, calls, logs = run([BAD, TimeoutError("second timeout"), GOOD])
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["raw_output"], logs[1]["output"])
        self.assertEqual(report["attempts"][1]["status"], "execution_error")

    def test_repaired_raw_diagnostics_and_messages_replayed(self):
        original, _, _ = run([BAD, GOOD])
        for field in ("first_raw", "issues", "second_message", "final_message", "fake_sentinel", "first_ready"):
            report = deepcopy(original)
            with self.subTest(field=field):
                if field == "first_raw": report["attempts"][0]["raw_output"]["plan"] = "changed archived business"
                elif field == "issues": report["attempts"][0]["issues"] = {"missing_keys":["unrelated"]}
                elif field == "second_message": report["attempts"][1]["messages"][3]["content"] = "different request"
                elif field == "final_message": report["messages"] = report["attempts"][0]["messages"]
                elif field == "fake_sentinel": report["attempts"][0]["raw_output"] = {"__error__":"cannot retry execution"}
                else: report["attempts"][0]["raw_output"] = deepcopy(GOOD)
                rehash(report)
                self.assertTrue(joint.validate_plan(report, WP, CONTEXT))

    def test_stale_task_context_source_and_version_fail(self):
        report, _, _ = run([GOOD])
        self.assertTrue(joint.validate_plan(report, {**WP,"changed":True}, CONTEXT))
        self.assertTrue(joint.validate_plan(report, WP, {**CONTEXT,"changed":True}))
        with patch.object(joint, "implementation_hashes", return_value={"changed":"source"}):
            self.assertTrue(joint.validate_plan(report, WP, CONTEXT))
        report["version"] = "original-world-joint-plan/v1"; rehash(report)
        self.assertTrue(joint.validate_plan(report, WP, CONTEXT))

    def test_source_drift_on_invalid_response_stops_before_repair(self):
        hashes = joint.implementation_hashes(); state = deepcopy(hashes); calls = []
        class Drift:
            def chat_json(self, step, messages, **kwargs):
                calls.append(step); state["world_joint_plan.py"] = "changed"
                return deepcopy(BAD)
        with patch.object(joint,"implementation_hashes",side_effect=lambda:deepcopy(state)):
            report=joint.author_plan(WP,CONTEXT,Drift(),model="offline-mini",max_tokens=2048)
        self.assertEqual(calls,["world.joint_plan"])
        self.assertEqual(report["status"],"error")
        self.assertIn("drift",report["error"])

    def test_real_builder_uses_corrected_plan_and_semantic_repair_reuses_it(self):
        from world_joint_plan_selftest import Tracer as BuilderTracer
        from seed_world_selftest import fixture
        from pipeline.world_gen import build_world
        wp, _, table = fixture()
        class RepairTracer(BuilderTracer):
            def chat_json(self, step, messages, **params):
                if step == "world.joint_plan":
                    self.plan = BAD if not self.calls else GOOD
                return super().chat_json(step, messages, **params)
        tracer = RepairTracer(table); draft = {}
        world = build_world(wp, tracer, draft_out=draft, log=lambda *_: None)
        self.assertEqual([step for step, _ in tracer.calls],
                         ["world.joint_plan", "world.joint_plan", "world.batch", "world.batch", "world.structure"])
        self.assertEqual(draft["joint_plan"]["raw_output"], GOOD)
        follow = BuilderTracer(table); repaired = {}
        build_world(wp, follow, draft_out=repaired, log=lambda *_: None, repair_input={
            "draft": draft, "feedback": {"issues": ["fallible business opinion"]},
            "targets": {"intrinsic": [], "structure": True}, "max_calls": 1})
        self.assertEqual([step for step, _ in follow.calls], ["world.structure"])
        self.assertEqual(repaired["joint_plan"], draft["joint_plan"])
        self.assertEqual(repaired["candidate_world"], world.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
