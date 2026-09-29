"""事件归一模块的离线测试。fixture 抄自 output/ 里真实落盘的 raw stdout 形状。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harnesses import events


def _codex_stream(*objects: dict) -> str:
    return "\n".join(json.dumps(obj, ensure_ascii=False) for obj in objects) + "\n"


CODEX_SAMPLE = _codex_stream(
    {"type": "thread.started", "thread_id": "t-1"},
    {"type": "turn.started"},
    {
        "type": "item.started",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/zsh -lc 'rg -n \"秦二\" INDEX.md sessions'",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/zsh -lc 'rg -n \"秦二\" INDEX.md sessions'",
            "aggregated_output": "sessions/s0000/x.md:11:命中",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {"type": "item.completed", "item": {"id": "item_2", "type": "agent_message", "text": "袁芳"}},
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 38432,
            "cached_input_tokens": 20466,
            "cache_write_input_tokens": 17957,
            "output_tokens": 336,
            "reasoning_output_tokens": 48,
        },
    },
)

CLAUDE_SAMPLE = json.dumps(
    {
        "result": "袁芳",
        "usage": {"input_tokens": 1200, "output_tokens": 12},
        "num_turns": 4,
        "duration_ms": 4211,
        "total_cost_usd": 0.03,
    },
    ensure_ascii=False,
)


class EventNormalizerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _normalize(self, adapter, stdout, **kwargs):
        return events.normalize(
            adapter,
            stdout,
            kwargs.pop("stderr", ""),
            question_index=0,
            workspace=self.workspace,
            question_id="q0",
            **kwargs,
        )

    def test_codex_stream_maps_tool_call_result_and_answer(self):
        normalized = self._normalize("codex", CODEX_SAMPLE)
        kinds = [event["kind"] for event in normalized]
        self.assertEqual(kinds[0], "run_start")
        self.assertEqual(kinds[-1], "run_end")
        self.assertEqual(kinds.count("tool_call"), 1)
        self.assertEqual(kinds.count("tool_result"), 1)

        call = next(event for event in normalized if event["kind"] == "tool_call")
        result = next(event for event in normalized if event["kind"] == "tool_result")
        self.assertEqual(call["call_id"], result["call_id"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("INDEX.md", call["args_preview"])

        message = next(event for event in normalized if event["kind"] == "message")
        self.assertEqual(message["text"], "袁芳")

    def test_codex_usage_event_carries_all_token_fields(self):
        normalized = self._normalize("codex", CODEX_SAMPLE)
        usage = next(event for event in normalized if event["kind"] == "usage")
        self.assertEqual(usage["source"], "codex_turn_completed")
        self.assertEqual(usage["input_tokens"], 38432)
        self.assertEqual(usage["cache_write_input_tokens"], 17957)
        self.assertEqual(usage["reasoning_output_tokens"], 48)

    def test_codex_in_workspace_tool_call_is_not_flagged(self):
        normalized = self._normalize("codex", CODEX_SAMPLE)
        call = next(event for event in normalized if event["kind"] == "tool_call")
        self.assertIsNone(call["boundary"]["outside_workspace"])
        self.assertIsNone(call["boundary"]["write_intent"])

    def test_codex_gold_read_is_flagged_outside_workspace(self):
        stdout = _codex_stream(
            {
                "type": "item.started",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "cat ../06_grounded_questions.json",
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "cat ../06_grounded_questions.json",
                    "aggregated_output": '{"questions": [...]}',
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {"type": "item.completed", "item": {"id": "item_2", "type": "agent_message", "text": "42"}},
        )
        normalized = self._normalize("codex", stdout)
        call = next(event for event in normalized if event["kind"] == "tool_call")
        self.assertEqual(call["boundary"]["outside_workspace"], "../06_grounded_questions.json")
        self.assertIsNotNone(call["boundary"]["sensitive_name"])

    def test_codex_unparseable_line_becomes_note_not_silence(self):
        normalized = self._normalize("codex", "not json at all\n")
        notes = [event for event in normalized if event["kind"] == "note"]
        self.assertTrue(any(note["code"] == "unparsed_stdout_line" for note in notes))

    def test_claude_single_object_maps_answer_and_usage(self):
        normalized = self._normalize("claude", CLAUDE_SAMPLE)
        message = next(event for event in normalized if event["kind"] == "message")
        self.assertEqual(message["text"], "袁芳")
        usage = next(event for event in normalized if event["kind"] == "usage")
        self.assertEqual(usage["source"], "claude_json")
        self.assertEqual(usage["input_tokens"], 1200)

    def test_claude_records_tool_events_unavailable(self):
        normalized = self._normalize("claude", CLAUDE_SAMPLE)
        codes = [event.get("code") for event in normalized if event["kind"] == "note"]
        self.assertIn("claude_tool_events_unavailable", codes)
        self.assertNotIn("tool_call", [event["kind"] for event in normalized])

    def test_dsh_plain_text_is_unavailable_but_keeps_answer(self):
        normalized = self._normalize("dsh", "袁芳\n")
        message = next(event for event in normalized if event["kind"] == "message")
        self.assertEqual(message["text"], "袁芳")
        codes = [event.get("code") for event in normalized if event["kind"] == "note"]
        self.assertIn("dsh_structured_events_unavailable", codes)

    def test_telemetry_status_reflects_coverage_honestly(self):
        codex = self._normalize("codex", CODEX_SAMPLE)
        self.assertEqual(events.telemetry_summary(codex)["status"], "partial")
        for adapter, payload in (("claude", CLAUDE_SAMPLE), ("dsh", "袁芳\n")):
            summary = events.telemetry_summary(self._normalize(adapter, payload))
            self.assertEqual(summary["status"], "unavailable")
            self.assertEqual(summary["tool_calls"], 0)

    def test_secret_leak_event_never_contains_the_value(self):
        secret = "cheat-secret-42"
        normalized = self._normalize(
            "dsh", f"answer\nkey={secret}\n", secrets={"DEEPSEEK_API_KEY": secret}
        )
        leaks = [event for event in normalized if event["kind"] == "secret_leak"]
        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0]["var"], "DEEPSEEK_API_KEY")
        self.assertEqual(leaks[0]["line_number"], 2)
        self.assertNotIn(secret, json.dumps(leaks[0], ensure_ascii=False))
        self.assertIn("<redacted>", leaks[0]["context"])
        self.assertEqual(events.telemetry_summary(normalized)["secret_leaks"], 1)

    def test_secret_scan_ignores_unrelated_secret_values(self):
        normalized = self._normalize(
            "dsh", "no leak here\n", secrets={"DEEPSEEK_API_KEY": "different-secret"}
        )
        self.assertEqual(events.telemetry_summary(normalized)["secret_leaks"], 0)

    def test_resume_attempt_is_recorded_on_run_bounds(self):
        first = self._normalize("dsh", "袁芳\n", attempt=0)
        second = self._normalize("dsh", "袁芳\n", attempt=1)
        self.assertEqual(first[0]["attempt"], 0)
        self.assertEqual(second[0]["attempt"], 1)
        self.assertEqual(second[-1]["attempt"], 1)

    def test_write_emits_one_json_object_per_line(self):
        normalized = self._normalize("codex", CODEX_SAMPLE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                written = events.write(handle, normalized)
            self.assertEqual(written, len(normalized))
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), len(normalized))
            for line in lines:
                self.assertEqual(json.loads(line)["schema"], events.EVENT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
