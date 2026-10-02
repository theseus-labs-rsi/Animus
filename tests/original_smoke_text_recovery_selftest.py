"""Actual wrapper text dispatch, stop provenance and conservative recovery."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_corpus_review_override_selftest as fixture
from original_smoke_json_retry_selftest import ReplaySDK
from tools import run_original_bc_smoke as smoke


def error(kind):
    exc = RuntimeError("offline transport failure")
    exc.kind = kind
    return exc


class TextRecoveryTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def text_calls(self, outcomes, extra=()):
        api, answers = ReplaySDK(outcomes), []
        original = fixture.Tracer.chat_text
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            for _ in outcomes:
                answers.append(tracer.chat_text("render.filler", [{"role": "user", "content": "plain prose"}],
                                                max_tokens=4096))
        with patch.object(fixture, "FakeSDK", return_value=api):
            result = self.run_tool(calls, extra)
        self.assertIs(fixture.Tracer.chat_text, original)
        self.assertIsNone(smoke._TRACER_STEP.get())
        return result, answers

    def test_text_connection_failure_leaves_siblings_and_budget_open(self):
        for kind in ("http_connect_error", "http_protocol_error"):
            with self.subTest(kind=kind):
                result, answers = self.text_calls([error(kind), "saved prose"])
                self.assertIsNone(result.error)
                self.assertIn("__error__", answers[0])
                self.assertEqual(answers[1], "saved prose")
                self.assertFalse(result.profile["stopped"])
                self.assertNotIn("stop_reason", result.profile)
                self.assertEqual(result.profile["admitted_calls"], 2)
                self.assertEqual([row["step"] for row in result.requests], ["render.filler"] * 2)
                self.assertEqual(result.profile["unsettled_reservation_calls"], 2)

    def test_text_auth_failure_preserves_first_cause_and_blocks_later_calls(self):
        result, answers = self.text_calls([error("http_status_401"), "must not dispatch"])
        self.assertEqual(len(result.api.calls), 1)
        self.assertTrue(result.profile["stopped"])
        cause = result.profile["stop_reason"]
        self.assertEqual((cause["reason"], cause["step"], cause["kind"]),
                         ("provider_failure", "render.filler", "http_status_401"))
        self.assertEqual(answers[1]["__error_metadata__"]["kind"], "http_status_401")

    def test_real_budget_limit_remains_closed_and_distinct(self):
        result, answers = self.text_calls(["first", "must not dispatch"], ["--max-cny", "0.12"])
        self.assertEqual(len(result.api.calls), 1)
        self.assertEqual(result.profile["stop_reason"]["reason"], "estimated_budget_limit")
        self.assertIn("reason=estimated_budget_limit", answers[-1]["__error__"])

    def test_json_failure_does_not_inherit_text_context(self):
        api = ReplaySDK(["text", error("http_connect_error"), "json sibling"])
        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            tracer.chat_text("render.filler", [])
            tracer.chat_json("unrecognized-step", [], retries=1)
            tracer.chat_text("render.filler", [])
        with patch.object(fixture, "FakeSDK", return_value=api):
            result = self.run_tool(calls)
        self.assertEqual(result.profile["stop_reason"]["step"], "unrecognized-step")
        self.assertEqual(len(api.calls), 2)


class ResumeEvidenceTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp

    def evidence(self):
        profile = {"model": "glm-5.3-flash", "stopped": True, "max_calls": 900,
            "admitted_calls": 2, "max_cny": 4, "budget_consumed_cny": 0.5,
            "unsettled_reservation_calls": 1, "source_drift": [],
            "error_type": "MaterialAuthoringStopped",
            "error": "Original experiment model/call/estimated budget limit; no provider dispatch"}
        rows = [{"event": "request", "model": profile["model"], "call_id": str(i)} for i in range(2)]
        rows.append({"event": "call_error", "call_id": "1", "step": "render.filler",
            "kind": "http_connect_error", "retryable": True, "local_cleanup_confirmed": True})
        return profile, rows

    def check(self, profile, rows):
        path = self.directory / "trace.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return smoke.transient_resume_evidence(profile, path)

    def test_explicit_legacy_transport_recovery_keeps_all_prior_liabilities(self):
        profile, rows = self.evidence()
        before = deepcopy(profile)
        result = self.check(profile, rows)
        self.assertEqual(profile, before)
        self.assertEqual(result["prior_budget_consumed_cny"], 0.5)
        self.assertEqual(result["unknown_reservations_preserved"], 1)
        self.assertEqual(result["error_call_ids"], ["1"])

    def test_auth_unknown_and_unproven_errors_cannot_resume(self):
        for change in ({"kind": "http_status_401"}, {"kind": "RuntimeError"},
                       {"retryable": False}, {"local_cleanup_confirmed": False},
                       {"step": "unknown"}):
            with self.subTest(change=change):
                profile, rows = self.evidence()
                rows[-1].update(change)
                with self.assertRaises(ValueError):
                    self.check(profile, rows)

    def test_budget_model_trace_and_drift_guards_remain(self):
        for key, value in (("budget_consumed_cny", 4), ("admitted_calls", 900),
                           ("source_drift", ["config.py"]), ("model", "another-model"),
                           ("stop_reason", {"reason": "estimated_budget_limit"})):
            with self.subTest(key=key):
                profile, rows = self.evidence()
                profile[key] = value
                with self.assertRaises(ValueError):
                    self.check(profile, rows)
        profile, rows = self.evidence()
        with self.assertRaises(ValueError):
            self.check(profile, rows[1:])

    def test_joint_design_header_deadline_needs_matching_saved_request(self):
        from pipeline.capability_contract import digest
        profile, _ = self.evidence()
        profile["admitted_calls"] = 1
        profile["stop_reason"] = {"reason": "provider_failure", "step": "council.joint_design_revision",
                                  "kind": "total_deadline", "call_id": "x"}
        payload = {"candidate": "unchanged"}
        rows = [{"event": "request", "model": profile["model"], "call_id": "x",
                 "messages": [{"role": "user", "content": json.dumps(payload)}]},
                {"event": "call_error", "call_id": "x", "step": "council.joint_design_revision",
                 "kind": "total_deadline", "phase": "awaiting_http_headers", "retryable": False,
                 "local_cleanup_confirmed": True}]
        audit = {"status": "designing", "actions": [{"status": "rejected"}],
                 "calls": [{"step": "council.joint_design_revision", "status": "provider_error",
                            "request_hash": digest(payload), "response": {"__error_metadata__": {
                                "kind": "total_deadline", "call_id": "x"}}}]}
        (self.directory / "01_joint_design_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        result = self.check(profile, rows)
        self.assertEqual(result["failed_request_hash"], digest(payload))
        self.assertEqual(result["unknown_reservations_preserved"], 1)
        rows[-1]["phase"] = "reading_http_body"
        with self.assertRaises(ValueError):
            self.check(profile, rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
