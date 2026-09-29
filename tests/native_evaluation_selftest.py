"""Offline bridge coverage: real plans, execute_many and scoring, fake CLI/API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "harness"))
from standard_light_fixture import build_standard_light
from pipeline import native_evaluation as n
from agent_harnesses.artifacts import load_benchmark_questions
from agent_harnesses.runners import native_cli


def fixture_chat_json(messages, model=None):
    return json.loads(chat(messages, model=model))  # replaced in module-local clone


class FixtureJudge:
    def __init__(self, provider, version="fixture-v1"):
        self._module = SimpleNamespace(config=SimpleNamespace(chat=provider, chat_json=fixture_chat_json))
        self.version = {"judge_sha256": version, "judge_model": "unconfigured"}

    def judge_answer(self, q, pred, use_llm=True):
        if pred == q["reference_answer"]:
            return True
        return self._module.config.chat_json([{"role": "user", "content": pred}],
                                             model=self._module.config.JUDGE_MODEL)["correct"]

    def judge_mode(self, q):
        return "value"

    def is_judgeable(self, q):
        return True

    def gold_display(self, q):
        return q["reference_answer"]


class NativeEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = [{"qid": f"q{i}", "question": f"value{i}?", "line": "L1", "capability": "IE",
                           "quality_status": "released"} for i in range(2)]
        refs = [{"qid": f"q{i}", "answer": f"v{i}", "answer_raw": f"v{i}",
                 "answer_projection": "raw_gt", "quality_status": "released"} for i in range(2)]
        self.benchmark = build_standard_light(self.root / "benchmark", self.questions, refs)
        self.directory = self.root / "calibration"
        self.settings = {"backend": "native_four", "targets": [
            {"id": f"athlete-{i}", "system": "native.codex" if i < 2 else "native.dsh", "model_id": f"model{i}",
             "model_label": f"model{i}", "endpoint_profile": "TEST", "protocol_style": "responses" if i < 2 else "openai-responses"}
            for i in range(4)], "judge_model": "glm-5.3-flash", "max_judge_calls": 20,
            "workers": 2, "parallel_targets": 2, "timeout_s": 77}
        self.provider = Mock(return_value='{"correct": true, "reason": "equivalent"}')
        self.version = "fixture-v1"
        self.answer_calls = []
        self.patches = [
            patch("agent_harnesses.tracks.native.preflight_system", side_effect=self.preflight),
            patch("agent_harnesses.execution.subprocess.run", side_effect=self.fake_cli),
            patch.object(n, "load_judge", side_effect=lambda root: FixtureJudge(self.provider, self.version)),
        ]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def preflight(self, system, model):
        return {"ok": True, "errors": [], "binary": "offline-cli", "binary_version": "fixture-v1",
                "adapter": system.implementation["adapter"], "model": model["model_id"],
                "runtime_options": {"timeout_s": "11"}}

    def fake_cli(self, command, **kwargs):
        self.assertIn("--resume", command)
        plan = json.loads(Path(command[command.index("--plan") + 1]).read_text(encoding="utf-8"))
        out = Path(plan["output_dir"])
        path = out / "results.jsonl"
        existing = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        completed = {r["question_index"] for r in existing if r["status"] == "completed"}
        self.assertEqual(plan["execution"]["timeout_s"], 77)
        self.assertEqual(plan["runtime"]["runtime_options"]["timeout_s"], 77)
        with path.open("a", encoding="utf-8") as stream:
            for i, q in enumerate(load_benchmark_questions(Path(plan["benchmark"]["path"]))):
                if i not in completed:
                    self.answer_calls.append((plan["target_id"], i))
                    stream.write(json.dumps({"question_index": i, "question_id": q["qid"], "status": "completed",
                        "error_type": None, "answer": "v0" if i == 0 else "equivalent-v1"}) + "\n")
        return SimpleNamespace(returncode=0)

    def test_real_plan_execute_score_chain_and_repeated_run_reuses_results(self):
        outputs = n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(self.answer_calls), 8)
        self.assertEqual(self.provider.call_count, 4)
        for target, run in outputs.items():
            plan = json.loads((run / "run_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["target_id"], target)
            self.assertEqual(plan["runner"], "native_cli")
            rows = [json.loads(s) for s in (run / "judged.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["correct"] for r in rows], [True, True])
            self.assertEqual(rows[0]["judge"]["version"]["judge_model"], "glm-5.3-flash")
        self.assertEqual(n.execute_native_evaluation(self.benchmark, self.directory, self.settings), outputs)
        self.assertEqual(len(self.answer_calls), 8)
        self.assertEqual(self.provider.call_count, 4)
        self.assertEqual(n._read(self.directory / "judge_budget.json")["reserved_calls"], 4)

    def test_preflight_failure_blocks_all_answer_and_judge_calls(self):
        self.mocks[0].side_effect = lambda system, model: {"ok": model["model_id"] != "model3", "errors": ["missing dsh"]}
        with self.assertRaisesRegex(n.ConfigurationError, "missing dsh"):
            n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.mocks[1].assert_not_called()
        self.mocks[2].assert_not_called()
        self.provider.assert_not_called()
        self.assertTrue((self.directory / "evaluation_plan.json").exists())

    def test_judge_budget_is_shared_and_persists_across_resume(self):
        self.settings["max_judge_calls"] = 2
        outputs = n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.assertEqual(self.provider.call_count, 2)
        rows = [json.loads(s) for p in outputs.values() for s in (p / "judged.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(r["correct"] is None for r in rows), 2)
        self.assertEqual(sum(r["correct"] is True for r in rows), 6)
        n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.assertEqual(self.provider.call_count, 2)
        self.assertEqual(len(self.answer_calls), 8)

    def test_changed_judge_rescores_without_reanswering(self):
        outputs = n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.version = "fixture-v2"
        self.assertEqual(n.execute_native_evaluation(self.benchmark, self.directory, self.settings), outputs)
        self.assertEqual(len(self.answer_calls), 8)
        self.assertEqual(self.provider.call_count, 8)

    def test_changed_latest_answer_rescores_only_that_answer(self):
        outputs = n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        run = outputs["athlete-0"]
        with (run / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"question_index": 1, "question_id": "q1", "status": "completed",
                                    "error_type": None, "answer": "different-v1"}) + "\n")
        n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        self.assertEqual(len(self.answer_calls), 8)
        self.assertEqual(self.provider.call_count, 5)

    def test_invalid_targets_and_caps_fail_without_calls(self):
        for key, value in (("targets", self.settings["targets"][:3]), ("max_judge_calls", 0), ("timeout_s", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                n.execute_native_evaluation(self.benchmark, self.directory, {**self.settings, key: value})
        self.provider.assert_not_called()
        self.mocks[1].assert_not_called()

    def test_json_retry_cannot_exceed_physical_judge_budget(self):
        # Exercise the production parser/retry loop; its HTTP transport is fake.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "MODEL": "offline"}):
            import config
        judge = FixtureJudge(Mock(return_value="invalid-json"))
        provider = judge._module.config.chat
        judge._module.config.chat_json = config.chat_json
        budget = n._Budget(self.root / "retry_budget.json", 1)
        wrapped = n._bounded_judge(judge, self.settings, budget)
        with self.assertRaisesRegex(config.ChatJSONError, "budget exhausted"):
            wrapped._module.config.chat_json([], retries=3, strict_json=True, retry_delay_base=0)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(budget.used, 1)
        self.assertEqual(provider.call_args.kwargs["model"], "glm-5.3-flash")
        profile = provider.call_args.kwargs["transport"]["profiles"]["low"]
        self.assertEqual(profile["deadline_seconds"], 77)

    def test_real_native_question_dispatch_uses_plan_timeout(self):
        outputs = n.execute_native_evaluation(self.benchmark, self.directory, self.settings)
        run = outputs["athlete-0"]
        # Run the real question scheduler, replacing the process creation edge.
        # The timeout must reach the process-tree runner, not just the saved plan.
        seen = []
        def question_process(command, **kwargs):
            if "--version" in command:
                return SimpleNamespace(returncode=0, stdout="offline-cli 1.0", stderr="")
            seen.append(kwargs["timeout"])
            return SimpleNamespace(returncode=0, stdout='{"type":"item.completed","item":{"type":"agent_message","text":"v0"}}\n', stderr="")
        local = {"TEST_API_KEY": "offline", "TEST_BASE_URL": "https://example.invalid",
                 "NATIVE_TIMEOUT_S": "11", "CODEX_BIN": "offline-cli"}
        with patch.object(native_cli, "load_env_file", return_value=local), \
                patch.object(native_cli, "_binary_path", return_value="offline-cli"), \
                patch.object(native_cli, "_run_cli_process", side_effect=question_process):
            native_cli.run(run / "run_plan.json", self.benchmark, run, limit=1, resume=False)
        self.assertEqual(seen, [77])


if __name__ == "__main__":
    unittest.main()
