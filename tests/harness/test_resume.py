from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harnesses.runners.native_cli import _load_done, run

# 离线测试夹具，见 fixtures/README.md。
OFFICE_FILTERED = Path(__file__).resolve().parent / "fixtures" / "office__20260902-121616"


def _write_env(env_file: Path, cli: Path) -> None:
    env_file.write_text(
        "ANTHROPIC_API_KEY=secret\n"
        "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
        f"CLAUDE_BIN={cli}\n",
        encoding="utf-8",
    )


def _fake_cli(path: Path, script: str) -> Path:
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _plan(out: Path, env_file: Path) -> Path:
    plan = {
        "system_id": "native.claude-code",
        "model": {"model_id": "fake-model", "endpoint_profile": "ANTHROPIC"},
        "system": {
            "implementation": {
                "runner": "native_cli",
                "adapter": "claude",
                "env_file": str(env_file),
                "executable": True,
            }
        },
    }
    plan_path = out / "run_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


class ResumeTests(unittest.TestCase):
    def test_resume_reruns_failed_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failing = _fake_cli(root / "fail-claude", "#!/bin/sh\nexit 1\n")
            working = _fake_cli(
                root / "ok-claude",
                "#!/bin/sh\nprintf '%s\\n' '{\"result\":\"good-answer\"}'\n",
            )
            changed = _fake_cli(
                root / "changed-claude",
                "#!/bin/sh\nprintf '%s\\n' '{\"result\":\"should-not-appear\"}'\n",
            )
            env_file = root / "claude.env"
            out = root / "out"
            out.mkdir()
            plan_path = _plan(out, env_file)

            _write_env(env_file, failing)
            summary = run(plan_path, OFFICE_FILTERED, out, limit=2)
            self.assertEqual(summary["n_errors"], 2)
            first_lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(first_lines), 2)

            # resume：failed 重跑（追加行、末行胜出），summary 反映最终状态
            _write_env(env_file, working)
            summary = run(plan_path, OFFICE_FILTERED, out, limit=2, resume=True)
            self.assertEqual(summary["n_errors"], 0)
            lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            done = _load_done(out / "results.jsonl")
            self.assertEqual(len(done), 2)
            self.assertTrue(all(r["status"] == "completed" for r in done.values()))
            self.assertTrue(all(r["answer"] == "good-answer" for r in done.values()))

            # 再次 resume：completed 全部跳过，不追加新行、不覆盖旧答案
            _write_env(env_file, changed)
            summary = run(plan_path, OFFICE_FILTERED, out, limit=2, resume=True)
            self.assertEqual(summary["n_errors"], 0)
            lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            done = _load_done(out / "results.jsonl")
            self.assertTrue(all(r["answer"] == "good-answer" for r in done.values()))

    def test_load_done_takes_last_line_per_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            rows = [
                {"question_id": "a", "status": "failed", "error_type": "timeout"},
                {"question_id": "b", "status": "completed", "error_type": None},
                {"question_id": "a", "status": "completed", "error_type": None},
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            done = _load_done(path)
            self.assertEqual(len(done), 2)
            self.assertEqual(done["a"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
