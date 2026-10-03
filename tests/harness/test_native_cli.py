from __future__ import annotations

import json
import re
import sys
from cli_fixture import write_cli
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from agent_harnesses.artifacts import inspect_benchmark
from agent_harnesses.config import (
    CONFIG_ROOT,
    ConfigurationError,
    load_experiment,
    load_registry,
)
from agent_harnesses.runners.native_cli import (
    DSH_PROFILE_NAME,
    _classify_failure,
    _cli_report,
    _command,
    _materialize_workspace,
    _parse_answer,
    _prepare_dsh_home,
    _prompt,
    _render_protocol,
    preflight_system,
    run,
)
from agent_harnesses.planning import make_plans
from agent_harnesses.execution import execute
from standard_light_fixture import build_standard_light


# 测试夹具：office 历史候选快照(filtered, UNMET)，见 fixtures/README.md。
OFFICE_FILTERED = Path(__file__).resolve().parent / "fixtures" / "office__20260902-121616"


class NativeCliTests(unittest.TestCase):
    def test_claude_error_result_is_not_misparsed_as_answer(self):
        payload = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": "API Error: Unable to connect to API",
            }
        )
        self.assertEqual(_parse_answer("claude", payload, None), "")

    def test_preflight_reads_ignored_env_without_exposing_key(self):
        system = load_registry()["native.claude-code"]
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "claude.env"
            env_file.write_text(
                "ALT_GATEWAY_API_KEY=super-secret-value\n"
                "ALT_GATEWAY_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={sys.executable}\n",
                encoding="utf-8",
            )
            implementation = dict(system.implementation)
            implementation["env_file"] = str(env_file)
            report = preflight_system(
                replace(system, implementation=implementation),
                {
                    "model_id": "test-model",
                    "label": "test-model",
                    "endpoint_profile": "ALT_GATEWAY",
                    "protocol_style": "anthropic-messages",
                },
            )
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["model"], "test-model")
            self.assertEqual(report["endpoint_profile"], "ALT_GATEWAY")
            self.assertEqual(report["endpoint_host"], "example.invalid")
            self.assertEqual(len(report["endpoint_fingerprint"]), 16)
            self.assertEqual(report["protocol_style"], "anthropic-messages")
            self.assertNotIn("super-secret-value", json.dumps(report))

    def test_workspace_contains_only_visible_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            sessions, docs = _materialize_workspace(OFFICE_FILTERED, workspace)
            self.assertEqual(sessions, 8)
            self.assertEqual(docs, 221)
            self.assertTrue((workspace / "INDEX.md").is_file())
            self.assertTrue((workspace / "sessions").is_dir())
            self.assertFalse((workspace / "06_grounded_questions.json").exists())
            self.assertFalse((workspace / "00_about.json").exists())
            # 根目录只允许 INDEX.md 与 sessions/：不能有会被 CLI 当作项目指令
            # 自动加载的文件（codex 读 AGENTS.md，claude 读 CLAUDE.md，dsh 两者都读）。
            self.assertEqual(
                sorted(p.name for p in workspace.iterdir()),
                ["INDEX.md", "sessions"],
            )

    def test_standard_light_workspace_and_protocol_use_public_exports(self):
        questions = [
            {"qid": "q1", "question": "状态？", "line": "L1", "capability": "KU", "quality_status": "rejected"}
        ]
        references = [
            {"qid": "q1", "answer": "进行中", "answer_raw": "进行中", "answer_projection": "raw_gt", "quality_status": "rejected"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = build_standard_light(Path(tmp) / "benchmark", questions, references)
            workspace = Path(tmp) / "workspace"
            sessions, docs = _materialize_workspace(root, workspace)
            self.assertEqual((sessions, docs), (1, 1))
            self.assertIn("第一周，项目状态为进行中。", next((workspace / "sessions").rglob("*.md")).read_text(encoding="utf-8"))
            self.assertEqual(
                _render_protocol(root),
                (root / "public" / "protocol.txt").read_text(encoding="utf-8"),
            )
            self.assertFalse((workspace / "references").exists())
            self.assertFalse((workspace / "private").exists())

    def test_prompt_carries_all_run_rules(self):
        prompt = _prompt("Answer protocol:\n- rule", "谁向谁汇报？")
        for fragment in (
            "INDEX.md and sessions/ are the only observable memory",
            "do not use external knowledge",
            "native read/search tools",
            "judge files",
            "credentials",
            "Do not write files",
            "Check all relevant sessions for temporal questions",
            "concise final answer",
            "Answer protocol:",
            "谁向谁汇报？",
        ):
            self.assertIn(fragment, prompt)
        # 工作区不再有指令文件，规则不能只留在文件里。
        self.assertNotIn("AGENTS.md", prompt)
        self.assertNotIn("CLAUDE.md", prompt)

    def test_preflight_rejects_invalid_protocol_and_embedded_credentials(self):
        system = load_registry()["native.dsh"]
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "deepseek.env"
            env_file.write_text(
                "DEEPSEEK_API_KEY=secret\n"
                "DEEPSEEK_BASE_URL=https://user:password@example.invalid/v1\n"
                f"DSH_BIN={sys.executable}\n",
                encoding="utf-8",
            )
            implementation = dict(system.implementation)
            implementation["env_file"] = str(env_file)
            report = preflight_system(
                replace(system, implementation=implementation),
                {
                    "model_id": "test-model",
                    "endpoint_profile": "DEEPSEEK",
                    "protocol_style": "invalid",
                },
            )
            self.assertFalse(report["ok"])
            self.assertTrue(any("不得内嵌" in item for item in report["errors"]))
            self.assertTrue(any("protocol_style" in item for item in report["errors"]))

    def test_fake_claude_cli_runs_one_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-claude"
            fake_cli = write_cli(fake_cli, 'print(\'{"result":"fake-answer"}\')\n')
            env_file = root / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=secret\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
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
            summary = run(plan_path, OFFICE_FILTERED, out, limit=1)
            self.assertEqual(summary["n_errors"], 0)
            result = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["answer"], "fake-answer")
            self.assertFalse(result["judgeable"])

    def test_fake_codex_cli_runs_one_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-codex"
            fake_cli = write_cli(fake_cli, 'print(\'{"item":{"text":"fake-codex-answer"}}\')\n')
            env_file = root / "gpt.env"
            env_file.write_text(
                "GPT_API_KEY=secret\n"
                "GPT_BASE_URL=https://example.invalid/v1\n"
                f"CODEX_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
            plan = {
                "system_id": "native.codex",
                "model": {"model_id": "fake-model", "endpoint_profile": "GPT", "protocol_style": "responses"},
                "system": {
                    "implementation": {
                        "runner": "native_cli",
                        "adapter": "codex",
                        "env_file": str(env_file),
                        "executable": True,
                    }
                },
            }
            plan_path = out / "run_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            summary = run(plan_path, OFFICE_FILTERED, out, limit=1)

            self.assertEqual(summary["n_errors"], 0)
            result = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["answer"], "fake-codex-answer")
            self.assertFalse(summary["scratch"]["retained"])
            self.assertFalse((out / "runtime").exists())

    def test_codex_last_message_is_read_from_scratch_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-codex"
            fake_cli = write_cli(fake_cli, 'import sys\nfrom pathlib import Path\nif "--output-last-message" in sys.argv:\n    Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("answer-from-last-message\\n", encoding="utf-8")\nprint(\'{"item":{"text":"answer-from-stdout"}}\')\n')
            env_file = root / "gpt.env"
            env_file.write_text(
                "GPT_API_KEY=secret\n"
                "GPT_BASE_URL=https://example.invalid/v1\n"
                f"CODEX_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
            plan = {
                "system_id": "native.codex",
                "model": {
                    "model_id": "fake-model",
                    "endpoint_profile": "GPT",
                    "protocol_style": "responses",
                },
                "system": {
                    "implementation": {
                        "runner": "native_cli",
                        "adapter": "codex",
                        "env_file": str(env_file),
                        "executable": True,
                    }
                },
            }
            plan_path = out / "run_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            summary = run(plan_path, OFFICE_FILTERED, out, limit=1, keep_scratch=True)

            self.assertEqual(summary["n_errors"], 0)
            result = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            # 中间副本放在该题自己的临时区子目录里（runtime/qNNNN/），
            # 这样并发答题时不会互相覆盖；解析出的答案必须进 results.jsonl。
            self.assertEqual(result["answer"], "answer-from-last-message")
            self.assertTrue((out / "runtime" / "q0000" / "last" / "0000.txt").is_file())
            self.assertEqual(
                sorted(p.name for p in (out / "raw").iterdir()),
                ["0000.stderr.txt", "0000.stdout.txt"],
            )

    def test_scratch_removed_by_default_and_kept_on_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-claude"
            fake_cli = write_cli(fake_cli, 'print(\'{"result":"fake-answer"}\')\n')
            env_file = root / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=sk-do-not-leak\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={fake_cli}\n",
                encoding="utf-8",
            )
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
            for keep, expect_workspace in ((False, False), (True, True)):
                out = root / f"out-{keep}"
                out.mkdir()
                plan_path = out / "run_plan.json"
                plan_path.write_text(json.dumps(plan), encoding="utf-8")
                summary = run(
                    plan_path, OFFICE_FILTERED, out, limit=1, keep_scratch=keep
                )
                self.assertEqual(summary["scratch"]["retained"], keep)
                self.assertEqual(
                    (out / "workspace").is_dir(),
                    expect_workspace,
                )
                # 无论是否保留临时区，实验结果都必须落盘。
                self.assertTrue((out / "results.jsonl").is_file())
                self.assertTrue((out / "events.jsonl").is_file())
                self.assertTrue((out / "summary.json").is_file())
                self.assertNotIn(
                    "sk-do-not-leak", (out / "summary.json").read_text(encoding="utf-8")
                )

    def test_codex_command_disables_nonessential_plugin_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "GPT_MODEL": "fake-model",
                "GPT_BASE_URL": "https://example.invalid/v1",
                "GPT_API_KEY": "secret",
            }
            command, _ = _command(
                "codex", "/usr/bin/true", env, root, root, "question", 0
            )
            disabled = [
                command[index + 1]
                for index, value in enumerate(command[:-1])
                if value == "--disable"
            ]
            self.assertEqual(
                disabled,
                ["plugins", "remote_plugin", "recommended_plugins", "apps"],
            )

    def test_claude_command_has_no_turn_cap(self):
        """claude 不应被单独加上轮数上限。

        `--max-turns` 实测不可靠（同一数值下 30~173 轮都可能完成，也可能在 31 轮
        报 error_max_turns），而且 codex/dsh 没有对应开关——只卡 claude 会让三家
        harness 不可比。硬边界统一用 NATIVE_TIMEOUT_S。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "CLAUDE_MODEL": "fake-model",
                "CLAUDE_MAX_TURNS": "30",  # 即使 env 里还留着，也不能再传下去
                "CLAUDE_MAX_BUDGET_USD": "5.00",
            }
            command, _ = _command(
                "claude", "/usr/bin/true", env, root, root, "question", 0
            )
            self.assertNotIn("--max-turns", command)
            self.assertNotIn("30", command)
            self.assertIn("--max-budget-usd", command)

    def test_claude_runtime_options_have_no_max_turns(self):
        system = load_registry()["native.claude-code"]
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=secret\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={sys.executable}\n"
                "CLAUDE_MAX_TURNS=30\n",
                encoding="utf-8",
            )
            implementation = dict(system.implementation)
            implementation["env_file"] = str(env_file)
            report = preflight_system(
                replace(system, implementation=implementation),
                {
                    "model_id": "test-model",
                    "label": "test-model",
                    "endpoint_profile": "ANTHROPIC",
                    "protocol_style": "anthropic-messages",
                },
            )
        self.assertTrue(report["ok"], report)
        # runtime_options 进 fingerprint：留着一个不生效的 max_turns 会误导读者。
        self.assertEqual(
            sorted(report["runtime_options"]), ["max_budget_usd", "timeout_s"]
        )

    def test_run_records_failure_classification_and_cli_report(self):
        """失败分型必须落进 results.jsonl 与 summary.json，而不是只留在 raw 里。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = json.dumps(
                {
                    "type": "result",
                    "subtype": "error_max_turns",
                    "is_error": True,
                    "api_error_status": None,
                    "terminal_reason": "max_turns",
                    "num_turns": 31,
                    "total_cost_usd": 2.44,
                    "result": None,
                }
            )
            fake_cli = root / "failing-claude"
            fake_cli = write_cli(fake_cli, "import sys\nprint(" + repr(payload) + ")\nsys.exit(1)\n")
            env_file = root / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=secret\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
            plan_path = out / "run_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
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
                ),
                encoding="utf-8",
            )

            summary = run(plan_path, OFFICE_FILTERED, out, limit=1)

            self.assertEqual(summary["n_errors"], 1)
            self.assertEqual(summary["n_errors_by_type"], {"budget_exceeded": 1})
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["error_type"], "budget_exceeded")
            self.assertEqual(row["error_detail"], {"budget_dimension": "turns"})
            self.assertEqual(row["returncode"], 1)
            self.assertEqual(row["cli_report"]["num_turns"], 31)
            self.assertEqual(row["cli_report"]["terminal_reason"], "max_turns")
            self.assertEqual(row["cli_report"]["total_cost_usd"], 2.44)
            self.assertIsNone(row["answer"])

    def test_fake_dsh_cli_runs_multiple_questions_without_persisting_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-dsh"
            fake_cli = write_cli(fake_cli, "print('fake-dsh-answer')\n")
            env_file = root / "deepseek.env"
            env_file.write_text(
                "DEEPSEEK_API_KEY=dsh-secret\n"
                "DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
                f"DSH_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
            plan = {
                "system_id": "native.dsh",
                "model": {"model_id": "fake-model", "endpoint_profile": "DEEPSEEK", "protocol_style": "openai-responses"},
                "system": {
                    "implementation": {
                        "runner": "native_cli",
                        "adapter": "dsh",
                        "env_file": str(env_file),
                        "executable": True,
                    }
                },
            }
            plan_path = out / "run_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            summary = run(plan_path, OFFICE_FILTERED, out, limit=2, keep_scratch=True)

            self.assertEqual(summary["n_errors"], 0)
            results = [
                json.loads(line)
                for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(results), 2)
            self.assertTrue(
                all(result["answer"] == "fake-dsh-answer" for result in results)
            )
            settings = (
                out / "runtime" / "q0000" / "dsh-home" / "settings.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("apiKeyEnv: DEEPSEEK_API_KEY", settings)
            self.assertNotIn("dsh-secret", settings)
            # 仓库钉住的 profile 必须被复制进 $DSH_HOME/profiles/<name>，且启动参数用同一个名字。
            # 只分发 package.json：cordis.yml / cordis.patch.yml / pnpm-workspace.yaml
            # 由 dsh 自己在 profile 目录里生成（见 tests/test_dsh_profile.py）。
            copied = (
                out / "runtime" / "q0000" / "dsh-home" / "profiles" / DSH_PROFILE_NAME
            )
            self.assertEqual(sorted(p.name for p in copied.iterdir()), ["package.json"])
            # 每题一个独立 $DSH_HOME，第二题不能复用第一题的 session 缓存。
            self.assertTrue((out / "runtime" / "q0001" / "dsh-home" / "settings.yaml").is_file())
            command, _ = _command(
                "dsh",
                "/usr/bin/true",
                {
                    "DEEPSEEK_MODEL": "fake-model",
                    "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                    "DEEPSEEK_API_KEY": "secret",
                },
                root,
                out,
                "question",
                0,
            )
            self.assertEqual(command[1:3], ["--profile", DSH_PROFILE_NAME])

    def test_dsh_profile_missing_is_a_preflight_error(self):
        system = load_registry()["native.dsh"]
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "deepseek.env"
            env_file.write_text(
                "DEEPSEEK_API_KEY=secret\n"
                "DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
                f"DSH_BIN={sys.executable}\n",
                encoding="utf-8",
            )
            implementation = dict(system.implementation)
            implementation["env_file"] = str(env_file)
            system = replace(system, implementation=implementation)
            model = {
                "model_id": "test-model",
                "label": "test-model",
                "endpoint_profile": "DEEPSEEK",
                "protocol_style": "openai-responses",
            }
            self.assertTrue(preflight_system(system, model)["ok"])
            with mock.patch(
                "agent_harnesses.runners.native_cli.DSH_PROFILES_DIR",
                Path(tmp) / "missing-profiles",
            ):
                report = preflight_system(system, model)
            self.assertFalse(report["ok"])
            self.assertTrue(
                any("缺少 dsh profile" in item for item in report["errors"]),
                report["errors"],
            )
            with mock.patch(
                "agent_harnesses.runners.native_cli.DSH_PROFILES_DIR",
                Path(tmp) / "missing-profiles",
            ):
                with self.assertRaises(ConfigurationError):
                    _prepare_dsh_home(Path(tmp) / "out", {})

    def test_timeout_output_is_normalized_on_python39(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "slow-codex"
            fake_cli = write_cli(fake_cli, 'import time\ntime.sleep(2)\n')
            env_file = root / "gpt.env"
            env_file.write_text(
                "GPT_API_KEY=secret\n"
                "GPT_BASE_URL=https://example.invalid/v1\n"
                "NATIVE_TIMEOUT_S=1\n"
                f"CODEX_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            out = root / "out"
            out.mkdir()
            plan = {
                "system_id": "native.codex",
                "model": {"model_id": "fake-model", "endpoint_profile": "GPT", "protocol_style": "responses"},
                "system": {
                    "implementation": {
                        "runner": "native_cli",
                        "adapter": "codex",
                        "env_file": str(env_file),
                        "executable": True,
                    }
                },
            }
            plan_path = out / "run_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            summary = run(plan_path, OFFICE_FILTERED, out, limit=1)

            self.assertEqual(summary["n_errors"], 1)
            result = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["error_type"], "timeout")

    def test_execute_records_public_runtime_lineage_without_secret(self):
        registry = load_registry()
        base = load_experiment(
            CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-claude"
            fake_cli = write_cli(fake_cli, 'print(\'{"result":"fake-answer"}\')\n')
            env_file = root / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=runtime-secret\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            implementation = dict(registry["native.claude-code"].implementation)
            implementation["env_file"] = str(env_file)
            system = replace(
                registry["native.claude-code"], implementation=implementation
            )
            experiment = replace(
                base,
                output_root=root / "output",
                execution={**base.execution, "limit": 1},
                targets=(base.targets[0],),
            )
            benchmark = inspect_benchmark(experiment.benchmark)
            plan = make_plans(experiment, [system], benchmark)[0]

            result = execute(experiment, system, plan)

            effective_plan = json.loads(Path(result["plan"]).read_text(encoding="utf-8"))
            serialized = json.dumps(effective_plan)
            self.assertNotIn("runtime-secret", serialized)
            self.assertEqual(effective_plan["runtime"]["model"], "claude-4-6")
            self.assertEqual(len(effective_plan["runtime_fingerprint"]), 64)
            # run 目录 = <family>/<开始时间>__<run_fingerprint[:12]>，且 family 是人可读的三段。
            run_dir = Path(result["plan"]).parent
            self.assertEqual(run_dir.parent, Path(plan.output_dir))
            self.assertEqual(Path(result["output_dir"]), run_dir)
            match = re.fullmatch(r"(\d{8}-\d{6})__([0-9a-f]{12})", run_dir.name)
            self.assertIsNotNone(match, run_dir.name)
            self.assertEqual(match.group(1), effective_plan["run_stamp"])
            self.assertEqual(
                match.group(2), effective_plan["run_fingerprint"][:12]
            )
            self.assertEqual(
                run_dir.parent.parent,
                experiment.output_root / experiment.experiment_id / benchmark.scenario,
            )
            # 默认只留实验结果，运行期临时区不落盘。
            self.assertFalse((run_dir / "workspace").exists())
            self.assertFalse((run_dir / "runtime").exists())
            for name in ("run_plan.json", "results.jsonl", "events.jsonl", "summary.json", "execution.json"):
                self.assertTrue((run_dir / name).is_file(), name)

    def test_resume_reuses_the_same_run_dir(self):
        registry = load_registry()
        base = load_experiment(
            CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "fake-claude"
            fake_cli = write_cli(fake_cli, 'print(\'{"result":"fake-answer"}\')\n')
            env_file = root / "claude.env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=secret\n"
                "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
                f"CLAUDE_BIN={fake_cli}\n",
                encoding="utf-8",
            )
            implementation = dict(registry["native.claude-code"].implementation)
            implementation["env_file"] = str(env_file)
            system = replace(
                registry["native.claude-code"], implementation=implementation
            )
            experiment = replace(
                base,
                output_root=root / "output",
                execution={**base.execution, "limit": 1},
                targets=(base.targets[0],),
            )
            benchmark = inspect_benchmark(experiment.benchmark)
            plan = make_plans(experiment, [system], benchmark)[0]

            first = execute(experiment, system, plan)
            first_rows = (Path(first["output_dir"]) / "results.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(Path(first["output_dir"]).name.split("__")[0], plan.run_stamp)

            # 非 resume 重跑（模拟稍后的另一次运行）：新建目录，不覆盖上一次 run。
            later = replace(
                plan,
                created_at="2099-01-01T00:00:00+00:00",
                run_stamp="20990101-000000",
            )
            second = execute(experiment, system, later)
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertEqual(
                Path(second["output_dir"]).name.split("__")[0], "20990101-000000"
            )

            # resume：复用同 fingerprint 的最近一个目录，并保留它原始的开始时间。
            resumed = execute(
                experiment,
                system,
                replace(
                    plan,
                    created_at="2099-06-06T06:06:06+00:00",
                    run_stamp="20990606-060606",
                ),
                resume=True,
            )
            self.assertEqual(resumed["output_dir"], second["output_dir"])
            resumed_plan = json.loads(Path(resumed["plan"]).read_text(encoding="utf-8"))
            self.assertEqual(resumed_plan["created_at"], "2099-01-01T00:00:00+00:00")
            self.assertEqual(resumed_plan["run_stamp"], "20990101-000000")
            # 该题已完成，resume 不重复追加结果行。
            resumed_rows = [
                json.loads(line)
                for line in (
                    Path(resumed["output_dir"]) / "results.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                len(resumed_rows), len(first_rows.strip().splitlines())
            )
            self.assertEqual(resumed_rows[0]["answer"], "fake-answer")
            self.assertEqual(resumed_rows[0]["question_index"], 0)
            execution = json.loads(
                (Path(resumed["output_dir"]) / "execution.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(execution["resumed"])
            self.assertIn("--resume", execution["command"])
            # 成功时 runner stdout 与 summary.json 重复，不再单独留存。
            self.assertIsNone(execution["stdout"])
            self.assertEqual(execution["schema"], "agent-harnesses.execution/v2")


class FailureClassificationTests(unittest.TestCase):
    """失败分型：不得把"预算耗尽/API 错误"混记为进程崩溃（本次冒烟靠翻 raw 才看清）。"""

    def _classify(self, adapter, stdout, *, returncode=1, answer="", timed_out=False):
        return _classify_failure(
            adapter,
            returncode=returncode,
            answer=answer,
            timed_out=timed_out,
            cli_report=_cli_report(adapter, stdout),
        )

    def test_claude_max_turns_is_budget_exceeded(self):
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "api_error_status": None,
                "terminal_reason": "max_turns",
                "num_turns": 31,
                "total_cost_usd": 2.44,
                "result": None,
            }
        )
        error_type, detail = self._classify("claude", stdout)
        self.assertEqual(error_type, "budget_exceeded")
        self.assertEqual(detail["budget_dimension"], "turns")
        report = _cli_report("claude", stdout)
        self.assertEqual(report["num_turns"], 31)
        self.assertEqual(report["terminal_reason"], "max_turns")
        self.assertEqual(report["total_cost_usd"], 2.44)

    def test_claude_max_budget_usd_is_budget_exceeded_on_cost(self):
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_budget_usd",
                "is_error": True,
                "terminal_reason": "budget_usd",
                "num_turns": 40,
            }
        )
        error_type, detail = self._classify("claude", stdout)
        self.assertEqual(error_type, "budget_exceeded")
        self.assertEqual(detail["budget_dimension"], "cost")

    def test_claude_api_error_status_is_api_error(self):
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "api_error_status": 429,
                "terminal_reason": "api_error",
            }
        )
        error_type, detail = self._classify("claude", stdout)
        self.assertEqual(error_type, "api_error")
        self.assertEqual(detail["api_error_status"], 429)

    def test_claude_api_error_prefix_without_status_is_api_error(self):
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "api_error_status": None,
                "result": "API Error: Unable to connect to API",
            }
        )
        error_type, _ = self._classify("claude", stdout)
        self.assertEqual(error_type, "api_error")

    def test_claude_unknown_error_subtype_is_cli_error(self):
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_structured_output_retries",
                "is_error": True,
                "terminal_reason": "completed",
            }
        )
        error_type, _ = self._classify("claude", stdout)
        self.assertEqual(error_type, "cli_error")

    def test_codex_error_event_is_cli_error_with_message(self):
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t"}),
                json.dumps({"type": "error", "message": "stream disconnected before completion"}),
            ]
        )
        report = _cli_report("codex", stdout)
        self.assertEqual(report["message"], "stream disconnected before completion")
        error_type, _ = self._classify("codex", stdout)
        self.assertEqual(error_type, "cli_error")

    def test_dsh_has_no_cli_report_and_falls_back_to_exit_code(self):
        # dsh headless 是纯文本，拿不到结构化原因：只按退出码归类，不猜。
        self.assertIsNone(_cli_report("dsh", "some plain text output"))
        self.assertEqual(self._classify("dsh", "some plain text")[0], "process_error")

    def test_timeout_and_parse_error_and_success(self):
        self.assertEqual(self._classify("claude", "", timed_out=True)[0], "timeout")
        self.assertEqual(
            self._classify("claude", "", returncode=0, answer="")[0], "parse_error"
        )
        self.assertIsNone(
            self._classify("claude", "", returncode=0, answer="answer")[0]
        )


if __name__ == "__main__":
    unittest.main()
