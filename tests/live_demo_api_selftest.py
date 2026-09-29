"""Offline API contract tests: execution, quality, counts and selected-run replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if (ROOT / "tools" / "live_demo_api.py").is_file():
    from fastapi.testclient import TestClient
    from tools import live_demo_api as api
else:
    TestClient = api = None


@unittest.skipIf(api is None, "Demo API removed from the CLI release; historical tests retained")
class LiveDemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(api, "RUNS_DIR", self.runs).start()
        patch.object(api, "JOBS", {}).start()
        original_connect = socket.socket.connect
        def offline_connect(sock, address):
            # Windows builds asyncio's local self-pipe through socketpair's loopback
            # implementation. Permit only that stdlib call; all service connections fail.
            caller = sys._getframe(1).f_code
            if caller.co_filename == socket.__file__ and caller.co_name == "_fallback_socketpair":
                return original_connect(sock, address)
            raise AssertionError("network forbidden")
        patch.object(socket.socket, "connect", offline_connect).start()
        self.client = TestClient(api.app)

    def write(self, directory, name, value):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def run_fixture(self, run_id="live__current-1234", status="done"):
        directory = self.runs / run_id
        directory.mkdir()
        self.write(directory, "manifest.json", {"status": status, "current_stage": "grounding", "stages": {
            "input": {"done": True, "started_ts": 10, "elapsed_s": 1},
            "grounding": {"done": True, "started_ts": 11, "elapsed_s": 2},
        }})
        self.write(directory, "06_grounded_questions.json", [{"qid": "q1", "question": "fixture"}])
        return directory

    def test_health_without_model_keeps_replay_available(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "MODEL": ""}), patch.object(api, "dotenv_values", return_value={}):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["live_ready"], False)
            self.assertEqual(response.json()["stop_at"], "07_quality")
            self.assertTrue(response.json()["ok"])
            with patch.object(api.subprocess, "Popen") as spawn:
                result = self.client.post("/api/runs", json={"scenario_text": "A sufficiently long scenario", "sample_name": "one.txt", "sample_text": "example"})
                self.assertEqual(result.status_code, 503)
                spawn.assert_not_called()

    def test_empty_environment_does_not_inherit_dotenv_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "MODEL": ""}), patch.object(api, "dotenv_values", return_value={"OPENAI_API_KEY": "file-key", "MODEL": "file-model"}):
            self.assertEqual(api._live_configuration(), (False, "not-configured"))

    def test_execution_done_is_not_release_qualification(self):
        self.run_fixture()
        payload = self.client.get("/api/runs/live__current-1234").json()
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["execution_status"], "succeeded")
        self.assertEqual(payload["quality"]["status"], "not_run")
        self.assertFalse(payload["eligible"])
        self.assertEqual(payload["stages"][-1]["name"], "quality")
        self.assertIsNone(payload["metrics"]["continuity_conflicts"])
        self.assertIsNone(payload["metrics"]["questions"])

    def test_historical_manual_review_does_not_replace_question_summary(self):
        directory = self.run_fixture()
        self.write(directory, "08_semantic_review.json", {"review_status": "manual_failed"})
        payload = self.client.get("/api/runs/live__current-1234").json()
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["quality"]["status"], "not_run")
        self.assertFalse(payload["eligible"])
        self.assertEqual(self.client.get("/api/runs").json()["runs"][0]["quality"]["status"], "not_run")

    def test_quality_failure_preserves_generation_completion(self):
        directory = self.run_fixture(status="failed")
        self.write(directory, "manifest.json", {"status": "failed", "current_stage": "quality", "stages": {
            name: {"done": name != "quality", "started_ts": 10 + index, "elapsed_s": 1}
            for index, name in enumerate(api.STAGE_ORDER)
        }})
        self.write(directory, "07_release.json", {"version": 2})
        payload = api._run_snapshot(directory.name)
        self.assertEqual(payload["execution_status"], "failed")
        self.assertEqual(payload["generation_status"], "succeeded")
        self.assertEqual(payload["quality"]["status"], "failed")
        self.assertFalse(payload["eligible"])

    def test_stale_receipt_is_not_eligible(self):
        directory = self.run_fixture()
        self.write(directory, "07_release.json", {"version": 1, "status": "failed", "eligible": False,
                                                   "scope": [], "checks": {}, "issues": [],
                                                   "inputs": {}, "implementation_fingerprint": "old-code"})
        payload = self.client.get("/api/runs/live__current-1234").json()
        self.assertEqual(payload["quality"]["status"], "failed")
        self.assertFalse(payload["eligible"])

    def test_api_passes_through_valid_release_state(self):
        self.run_fixture()
        quality = {"version": 1, "status": "passed", "eligible": True,
                   "scope": ["fixture_check"], "checks": {"fixture_check": {"passed": True}}, "issues": []}
        with patch.object(api, "quality_snapshot", return_value=quality):
            snapshot = api._run_snapshot("live__current-1234")
        self.assertEqual(snapshot["quality"], quality)
        self.assertTrue(snapshot["eligible"])

    def test_known_zero_and_unknown_are_distinct(self):
        directory = self.run_fixture()
        self.write(directory, "02_world.json", {"entities": {f"entity-{i}": {} for i in range(13)}, "n_sessions": 12, "events": []})
        self.write(directory, "04_questions.json", [])
        self.write(directory, "05_corpus.json", {"corpus": {"sessions": [
            {"session_id": i, "docs": [{"doc_id": str(i), "content": "abc"}]} for i in range(12)]}})
        snapshot = api._run_snapshot(directory.name)
        self.assertEqual(len(snapshot["views"]["world"]["entity_names"]), 12)
        self.assertEqual(snapshot["metrics"]["entities"], 13)
        self.assertEqual(snapshot["metrics"]["docs"], 12)
        self.assertEqual(snapshot["metrics"]["chars"], 36)
        self.assertEqual(snapshot["metrics"]["questions"], 0)
        self.assertEqual(snapshot["metrics"]["events"], 0)
        self.assertIsNone(snapshot["metrics"]["grounding"]["grounded"])

    def test_replay_keeps_requested_run_and_quality(self):
        self.run_fixture("office__20260717-064826")
        current = self.run_fixture()
        self.write(current, "07_release.json", {"version": 2})
        payload = self.client.get("/api/replay/live__current-1234").json()
        self.assertEqual(payload["run_id"], "live__current-1234")
        self.assertEqual(payload["final_snapshot"]["run_id"], "live__current-1234")
        self.assertEqual(payload["final_snapshot"]["quality"]["status"], "failed")

    def test_create_uses_canonical_worker(self):
        process = Mock()
        process.poll.return_value = None
        request = {"scenario_text": "A sufficiently long scenario", "sample_name": "one.txt", "sample_text": "example"}
        with patch.object(api, "_live_configuration", return_value=(True, "offline-model")), patch.object(api.subprocess, "Popen", return_value=process) as spawn:
            response = self.client.post("/api/runs", json=request)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(spawn.call_args.kwargs["cwd"], ROOT)
            self.assertIn("tools.live_demo_worker", spawn.call_args.args[0])
        request["sample_name"] = "one.csv"
        self.assertEqual(self.client.post("/api/runs", json=request).status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
