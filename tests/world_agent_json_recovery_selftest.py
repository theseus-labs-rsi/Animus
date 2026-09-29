"""Replay observed GLM JSON tails through the real bounded transport, offline."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import original_smoke_json_retry_selftest as transport
import world_agent_selftest as agent_fixture
from pipeline import world_agent


class WorldJSONRecoveryTests(unittest.TestCase):
    setUp = transport.fixture.CorpusReasoningOverrideTests.setUp
    run_tool = transport.fixture.CorpusReasoningOverrideTests.run_tool

    def test_all_seven_observed_failures_use_feedback_and_accounted_retry(self):
        cases = json.loads((ROOT / "tests/fixtures/world_agent_json_tail_failures.json").read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(cases), 7)
        for case in cases:
            with self.subTest(seed=case["seed"]):
                raw = case["raw_output"]
                # This is an offline simulated corrected response, never a production repair.
                parsed, end = json.JSONDecoder().raw_decode(raw.lstrip())
                self.assertTrue(raw.lstrip()[end:].strip())
                corrected = json.dumps(parsed, ensure_ascii=False)
                sdk = transport.ReplaySDK([raw, corrected, '{"verdict":"fail"}'])
                answers = []

                def calls(directory, cfg):
                    tracer = transport.Tracer(transport.SimpleNamespace(dir=directory))
                    answers.append(tracer.chat_json("world.agent.write", [
                        {"role": "system", "content": world_agent.AUTHOR_SYSTEM},
                        {"role": "user", "content": "生成一个完整的 JSON 业务单元"}],
                        retries=world_agent.FORMAT_ATTEMPTS, strict_json=True,
                        response_format={"type": "json_object"}, max_tokens=4096))
                    answers.append(tracer.chat_json("world.review", [], strict_json=True))

                with patch.object(transport.fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
                    result = self.run_tool(calls, ["--model", "glm-5.3-flash", "--json-attempts", "5"])
                self.assertIsNone(result.error)
                self.assertEqual(answers, [parsed, {"verdict": "fail"}])
                self.assertEqual(result.profile["admitted_calls"], 3)
                self.assertFalse(result.profile["stopped"])
                self.assertEqual(len(sdk.calls), 3)
                first, retry = [call["kwargs"] for call in sdk.calls[:2]]
                self.assertEqual(first["response_format"], {"type": "json_object"})
                self.assertEqual(retry["response_format"], first["response_format"])
                self.assertEqual(retry["messages"][-2], {"role": "assistant", "content": raw})
                self.assertIn("Extra data", retry["messages"][-1]["content"])
                attempts = [row for row in result.rows if row.get("event") == "json_attempt"]
                self.assertEqual(attempts[0]["max_attempts"], world_agent.FORMAT_ATTEMPTS)

    def test_real_agent_continues_to_finish_after_author_format_recovery(self):
        wp, _, table = agent_fixture.fixture()
        full = agent_fixture.table_part(table, [0, 1], True)
        sdk = transport.ReplaySDK([
            json.dumps(agent_fixture.write("one")),
            json.dumps(full) + "\n说明：本单元完成。",
            json.dumps(full), json.dumps(agent_fixture.FINISH)])
        completed = []

        def calls(directory, cfg):
            tracer = transport.Tracer(transport.SimpleNamespace(dir=directory))
            with patch.object(world_agent, "config", cfg):
                completed.append(world_agent.generate_world(deepcopy(wp), tracer,
                    checkpoint_path=directory / "world_checkpoint.json", log=lambda *_: None))

        with patch.object(transport.fixture, "FakeSDK", return_value=sdk), patch("time.sleep"):
            result = self.run_tool(calls, ["--model", "glm-5.3-flash", "--json-attempts", "5"])
        self.assertIsNone(result.error)
        self.assertEqual(completed[0][1]["status"], "completed")
        self.assertEqual(len(completed[0][1]["units"]), 1)
        self.assertEqual(result.profile["admitted_calls"], 4)
        self.assertFalse(result.profile["stopped"])
        self.assertTrue(all(call["kwargs"]["response_format"] == {"type": "json_object"}
                            for call in sdk.calls))


if __name__ == "__main__":
    unittest.main()
