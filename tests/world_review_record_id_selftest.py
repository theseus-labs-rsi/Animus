"""Offline regression for the 4145c689 unknown-disclosure-ID reply shape."""
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
from pipeline import disclosure, world_semantics as review
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE
from world_scoped_agents_selftest import ScopedAgentFixture
from world_semantics_selftest import fixture


UNKNOWN = "world/disclosure/undisclosed"
OLD_ERROR = "Disclosure opinion requires reading full joined record context"


class Author(ScopedAgentFixture):
    def __init__(self, ws):
        super().__init__()
        self.inventory = disclosure.catalogue(ws)

    def chat_json(self, step, messages, **params):
        raw = super().chat_json(step, messages, **params)
        if raw.get("action") == "arrange":
            raw["records"] = [{"session": item["fact_session"], "refs": [item["ref"]],
                "channel": "Offline record", "acquisition_context": item["fact_date"] + " received."}
                for item in self.inventory]
        return raw


class WrongIDReader(ScopedAgentFixture):
    """Valid d4/d5 opinions plus a list mistaken for an additional record."""
    def __init__(self, path):
        super().__init__()
        self.pfile = path / "prompts.jsonl"
        self.bad_index = None
        self.bad_payload = self.corrected = self.correction_payload = None

    def chat_json(self, step, messages, **params):
        body = json.loads(messages[-1]["content"])
        if body["working_state"].get("action_error"):
            self.calls.append({"step": step, "messages": deepcopy(messages), "params": deepcopy(params)})
            self.correction_payload = deepcopy(body)
            return deepcopy(self.corrected)
        raw = super().chat_json(step, messages, **params)
        if raw.get("action") == "submit" and self.bad_index is None:
            nodes = self.nodes[step]
            wanted = [key for key, node in nodes.items()
                if isinstance(node.get("value"), dict) and node["value"].get("record_id") in ("d4", "d5")]
            hidden = next(key for key, node in nodes.items() if node["label"] == UNKNOWN)
            if not set(wanted + [hidden]) <= set(body["exact_reads"]):
                return {"action": "inspect", "ids": wanted + [hidden]}
            for row in raw["disclosure_reviews"]:
                if row["record_id"] in ("d4", "d5"):
                    day = "2025-01-27" if row["record_id"] == "d4" else "2025-02-03"
                    row.update(understanding="Current record received on " + day,
                               reason="Current acquisition date agrees with session date " + day)
            self.corrected = deepcopy(raw)
            raw["disclosure_reviews"].append({"record_id": UNKNOWN,
                "understanding": "The undisclosed list is empty.", "refs": [hidden],
                "reason": "All listed targets are disclosed.", "status": "compatible"})
            self.bad_index = len(self.calls) - 1
            self.bad_payload = deepcopy(body)
        return raw


class ContinueReader:
    def __init__(self, path, corrected):
        self.pfile = path / "prompts.jsonl"
        self.corrected = corrected
        self.calls = []

    def chat_json(self, step, messages, **params):
        self.calls.append(deepcopy(messages))
        if len(self.calls) == 1:
            return deepcopy(self.corrected)
        return {"action": "finish", "decision": "accept", "reason": "Offline original final opinion.",
            "limitations": "No model quality claim.",
            "repair_targets": {"intrinsic": [], "structure": False, "disclosure": False}}


class Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")).start()
        patch.dict(sys.modules, {"config": SimpleNamespace(
            STRUCTURE_MODEL="cheap-author", REVIEWER_MODEL="cheap-review")}).start()
        self.wp, self.ws, self.task = fixture()
        self.wp["world_generation"] = {"strategy": "agentic"}
        self.wp["quality_contract"]["public_disclosure"] = True
        temporal = {"unit": "week", "step_days": 7, "n_sessions": 6}
        self.wp["world_blueprint"]["temporal_model"] = deepcopy(temporal)
        self.ws.world_blueprint["temporal_model"] = deepcopy(temporal)
        self.ws.n_sessions = 6
        self.ws.entities["甲/~记录"]["利润"] = Timeline([
            Op(i, date, SET if i == 0 else UPDATE, str(10+i), None if i == 0 else str(9+i))
            for i, date in enumerate(("2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03"))])
        self.ws.conflicts = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        result = disclosure.author_plan(self.wp, self.ws, Author(self.ws), self.task)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(disclosure.validate_plan(self.ws), [])

    def test_unknown_id_rolls_back_batch_retains_reads_then_accepts_corrected_batch(self):
        tracer = WrongIDReader(self.path)
        report = review.review_world(self.wp, self.ws, tracer, self.task)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task), [])
        current = tracer.correction_payload
        error = current["working_state"]["action_error"]
        self.assertIn("Unknown disclosure record_id", error)
        self.assertIn('"d4"', error)
        self.assertIn('"d5"', error)
        self.assertIn(UNKNOWN + " is a list, not a disclosure record", error)
        self.assertIn("resubmit this same batch", error)
        self.assertEqual(current["exact_reads"], tracer.bad_payload["exact_reads"])
        self.assertEqual(current["working_state"]["disclosure_reviews"], [])
        self.assertEqual(current["working_state"]["mechanism_coverage"], [])
        self.assertEqual(current["working_state"]["note"], "")
        records = [node["value"] for node in current["exact_reads"].values()
                   if isinstance(node.get("value"), dict) and "record_id" in node["value"]]
        self.assertEqual({record["record_id"] for record in records}, {"d4", "d5"})
        for identity, date in (("d4", "2025-01-27"), ("d5", "2025-02-03")):
            self.assertIn(date, json.dumps(next(r for r in records if r["record_id"] == identity)))
        submitted = {row["record_id"]: row for row in report["raw_output"]["disclosure_reviews"]}
        self.assertNotIn(UNKNOWN, submitted)
        self.assertEqual(submitted["d4"]["status"], "compatible")
        self.assertEqual(submitted["d5"]["status"], "compatible")

    def test_real_unread_record_keeps_original_error(self):
        payload, _, _ = review._inputs(self.wp, self.ws, self.task, None, None)
        window = review._scoped_review_context(payload)
        record = next(row for row in payload["disclosure_record_contexts"] if row["record_id"] == "d4")
        window.read = set(window.nodes) - {record["record_ref_id"]}
        state = {"issues": [], "mechanism_coverage": [], "disclosure_reviews": [], "note": "", "revisions": []}
        raw = {"action": "submit", "disclosure_reviews": [{"record_id": "d4", "refs": [],
            "understanding": "Unread record", "reason": "Offline test", "status": "compatible"}], "note": "test"}
        with self.assertRaisesRegex(ValueError, "^" + OLD_ERROR + "$"):
            review._review_action(raw, window, state, payload)

    def test_old_error_checkpoint_reuses_only_exact_prefix_then_forks_feedback(self):
        original_action = review._review_action

        def old_feedback(*args, **kwargs):
            try:
                return original_action(*args, **kwargs)
            except ValueError as exc:
                if str(exc).startswith("Unknown disclosure record_id"):
                    raise ValueError(OLD_ERROR) from exc
                raise

        old_reader = WrongIDReader(self.path)
        with patch.object(review, "_review_action", side_effect=old_feedback):
            old_report = review.review_world(self.wp, self.ws, old_reader, self.task)
        self.assertEqual(old_report["status"], "passed", old_report)
        self.assertEqual(old_reader.correction_payload["working_state"]["action_error"], OLD_ERROR)
        checkpoint = next(self.path.glob("02_world_review_*.ckpt.json"))
        old = json.loads(checkpoint.read_text(encoding="utf-8"))
        checkpoint.unlink()
        old["identity"] = "old-generic-error-implementation"
        old_path = self.path / "02_world_review_old_feedback.ckpt.json"
        old_path.write_text(json.dumps(old), encoding="utf-8")
        old_bytes = old_path.read_bytes()
        reader = ContinueReader(self.path, old_reader.corrected)
        result = review.review_world(self.wp, self.ws, reader, self.task)
        self.assertEqual(result["status"], "passed", result)
        prefix = old_reader.bad_index + 1
        self.assertEqual(result["transcript"][:prefix], old_report["transcript"][:prefix])
        self.assertNotEqual(result["transcript"][prefix]["messages_hash"],
                            old_report["transcript"][prefix]["messages_hash"])
        self.assertEqual(len(reader.calls), len(result["transcript"]) - prefix)
        self.assertIn("Unknown disclosure record_id", json.loads(reader.calls[0][-1]["content"])["working_state"]["action_error"])
        self.assertEqual(old_path.read_bytes(), old_bytes)

    def test_real_pre_fix_passed_candidate_transcript_replays_without_calls(self):
        run = ROOT / "output/v2b_review_fix/gpu_evidence/terminal/candidate/output/runs/generation_first_v2_b_candidate_20260924"
        if not run.is_dir():
            self.skipTest("Optional immutable GPU evidence is not distributed in source releases")
        read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
        report = read("02_world_review.json")
        self.assertEqual(report["status"], "passed")
        ws = WorldState.from_dict(read("02_world_candidate.json"))
        payload, binding, call, transcript, raw = review._scoped_review_session(
            read("01_whitepaper.json"), ws, read("00_input.json"),
            report.get("previous"), report.get("author_responses"), report["call"]["params"]["model"],
            transcript=report["transcript"],
            legacy_resume_prefix=report["binding"].get("legacy_resume_prefix", 0),
            compatibility_resume_length=report["binding"].get("compatibility_resume_length", 0))
        self.assertEqual(transcript, report["transcript"])
        self.assertEqual(raw, report["raw_output"])
        self.assertEqual(review._parse(raw, payload)["status"], "passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
