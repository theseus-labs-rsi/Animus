"""并行执行的离线回归：不访问网络，用 fake CLI 的 sleep 模拟慢题。

覆盖三件事：
1. 内层逐题并行确实重叠执行（wall-clock 显著小于串行），且每题一个独立
   `workspace/qNNNN/` 与 `runtime/qNNNN/`，不会互相污染；
2. 并发只改调度与落盘顺序，不改每题结果：并发=1 与并发=N 的 results/events
   在去掉时间戳与顺序后语义等价，`question_id`/`attempt` 语义一致；
3. `--resume` 在并发下仍然跳过已完成题（不追加新行、不重算），失败题会重跑。
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from cli_fixture import write_cli
from dataclasses import replace
from pathlib import Path

from agent_harnesses.artifacts import inspect_benchmark
from agent_harnesses.config import (
    CONFIG_ROOT,
    ConfigurationError,
    ExperimentConfig,
    load_experiment,
    load_registry,
)
from agent_harnesses.execution import execute_many
from agent_harnesses.planning import make_plans
from agent_harnesses.runners.native_cli import run

# 离线测试夹具，见 fixtures/README.md。
OFFICE_FILTERED = Path(__file__).resolve().parent / "fixtures" / "office__20260902-121616"


def _fake_sleepy_claude(path: Path, seconds: int) -> Path:
    return write_cli(path, "import json, os, time\n" + f"time.sleep({seconds})\n" +
                     "print(json.dumps({'result': 'ans-' + str(os.getpid())}))\n")


def _claude_plan(out: Path, env_file: Path, parallel: int) -> Path:
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
        "execution": {"questions_in_parallel": parallel},
    }
    plan_path = out / "run_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def _write_claude_env(root: Path, cli: Path, native_timeout_s: int = 30) -> Path:
    env_file = root / "claude.env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=secret\n"
        "ANTHROPIC_BASE_URL=https://example.invalid/v1\n"
        f"NATIVE_TIMEOUT_S={native_timeout_s}\n"
        f"CLAUDE_BIN={cli}\n",
        encoding="utf-8",
    )
    return env_file


def _rows(out: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_rows(out: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class QuestionParallelismTests(unittest.TestCase):
    def test_parallel_questions_overlap_and_isolate_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 2)
            env_file = _write_claude_env(root, cli)

            serial_out = root / "serial"
            serial_out.mkdir()
            serial_plan = _claude_plan(serial_out, env_file, 1)
            started = time.monotonic()
            serial = run(serial_plan, OFFICE_FILTERED, serial_out, limit=3)
            serial_wall = time.monotonic() - started

            parallel_out = root / "parallel"
            parallel_out.mkdir()
            parallel_plan = _claude_plan(parallel_out, env_file, 3)
            started = time.monotonic()
            parallel = run(parallel_plan, OFFICE_FILTERED, parallel_out, limit=3)
            parallel_wall = time.monotonic() - started

            self.assertEqual(serial["n_errors"], 0)
            self.assertEqual(parallel["n_errors"], 0)
            self.assertEqual(serial["n_questions"], 3)
            self.assertEqual(parallel["n_questions"], 3)
            # 3 题 × 2 s 串行约 6 s；并发 3 应接近 2 s。留宽裕余量避免机器抖动导致假失败。
            self.assertLess(parallel_wall, serial_wall * 0.75, (parallel_wall, serial_wall))
            self.assertEqual(serial["concurrency"]["workers_used"], 1)
            self.assertEqual(parallel["concurrency"]["workers_used"], 3)

            # 每题一个独立进程：答案里的 pid 互不相同。
            serial_answers = {row["answer"] for row in _rows(serial_out)}
            parallel_rows = _rows(parallel_out)
            self.assertEqual(len(parallel_rows), 3)
            self.assertEqual(len({row["answer"] for row in parallel_rows}), 3)
            self.assertEqual(len(serial_answers), 3)

    def test_parallel_uses_per_question_workspace_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 0)
            env_file = _write_claude_env(root, cli)
            out = root / "out"
            out.mkdir()
            plan = _claude_plan(out, env_file, 2)

            run(plan, OFFICE_FILTERED, out, limit=2, keep_scratch=True)

            # 并发>1：每题一份工作区，语料内容确定性重建且内容一致。
            workspaces = sorted(p.name for p in (out / "workspace").iterdir())
            self.assertEqual(workspaces, ["q0000", "q0001"])
            for index in range(2):
                workspace = out / "workspace" / f"q{index:04d}"
                self.assertTrue((workspace / "INDEX.md").is_file())
                self.assertTrue((workspace / "sessions").is_dir())
            first = (out / "workspace" / "q0000" / "INDEX.md").read_text(encoding="utf-8")
            second = (out / "workspace" / "q0001" / "INDEX.md").read_text(encoding="utf-8")
            self.assertEqual(first, second)

    def test_parallel_results_are_equivalent_to_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 0)
            env_file = _write_claude_env(root, cli)
            runs = {}
            for name, parallel in (("serial", 1), ("parallel", 3)):
                out = root / name
                out.mkdir()
                plan = _claude_plan(out, env_file, parallel)
                run(plan, OFFICE_FILTERED, out, limit=3)
                runs[name] = out

            # 每题语义必须一致：同 question_id 的 status/error_type/answer 相同。
            by_id = {}
            for name, out in runs.items():
                by_id[name] = {row["question_id"]: row for row in _rows(out)}
            self.assertEqual(set(by_id["serial"]), set(by_id["parallel"]))
            for question_id, serial_row in by_id["serial"].items():
                parallel_row = by_id["parallel"][question_id]
                # answer 里带进程 pid，两次 run 之间必然不同，因此不参与等价比较；
                # 它是否真的每题独立由 test_parallel_questions_overlap_and_isolate_scratch 覆盖。
                for key in ("question_index", "status", "error_type", "judgeable"):
                    self.assertEqual(serial_row[key], parallel_row[key], key)
            # events.jsonl 逐题归一结果一致（时间戳 ts/duration 天然不同，不比较）。
            def shape(out: Path) -> list[tuple]:
                collected = []
                for event in _event_rows(out):
                    # 信封字段：question_index 每题必有，question_id 只有 run_start/run_end 带。
                    collected.append(
                        (
                            event.get("question_index"),
                            event.get("kind"),
                            event.get("question_id"),
                            event.get("attempt"),
                            event.get("status"),
                        )
                    )
                # 按题、再按事件顺序比较：并发只改块与块的相对顺序，不改题内顺序。
                return sorted(collected, key=lambda row: (row[0] or 0, row[1] or ""))

            self.assertEqual(shape(runs["serial"]), shape(runs["parallel"]))

    def test_resume_under_parallelism_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working = _fake_sleepy_claude(root / "ok-claude", 0)
            failing = write_cli(root / "fail-claude", "import sys\nsys.exit(1)\n")
            env_file = _write_claude_env(root, failing)
            out = root / "out"
            out.mkdir()
            plan = _claude_plan(out, env_file, 2)

            first = run(plan, OFFICE_FILTERED, out, limit=2)
            self.assertEqual(first["n_errors"], 2)
            self.assertEqual(len(_rows(out)), 2)

            # 换成能成功的 CLI 后 resume：失败题重跑，行数追加，末行胜出。
            _write_claude_env(root, working)
            second = run(plan, OFFICE_FILTERED, out, limit=2, resume=True)
            self.assertEqual(second["n_errors"], 0)
            rows = _rows(out)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["status"] == "completed" for row in rows[-2:]))

            # 第三次 resume：全部 completed，跳过，不追加新行。
            third = run(plan, OFFICE_FILTERED, out, limit=2, resume=True)
            self.assertEqual(third["n_errors"], 0)
            self.assertEqual(len(_rows(out)), 4)
            self.assertEqual(third["concurrency"]["questions_dispatched"], 0)
            # attempt 编号递增：同一题第二次尝试记 attempt=1。
            attempts = [
                event["attempt"]
                for event in _event_rows(out)
                if event["kind"] == "run_start"
            ]
            self.assertEqual(sorted(attempts), [0, 0, 1, 1])

    def test_worker_exception_is_recorded_as_a_row(self):
        """worker 内部的非 subprocess 异常不能静默丢题，必须落成一行 worker_error。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 0)
            env_file = _write_claude_env(root, cli)
            out = root / "out"
            out.mkdir()
            plan = _claude_plan(out, env_file, 2)

            from agent_harnesses.runners import native_cli

            original = native_cli._command

            def exploding_command(adapter, binary, env, workspace, out_dir, prompt, index):
                raise RuntimeError(f"boom-{index}")

            native_cli._command = exploding_command
            try:
                summary = run(plan, OFFICE_FILTERED, out, limit=2)
            finally:
                native_cli._command = original

            self.assertEqual(summary["n_errors"], 2)
            self.assertEqual(summary["n_errors_by_type"], {"worker_error": 2})
            rows = _rows(out)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["error_type"], "worker_error")
                self.assertIn("boom", row["error_detail"]["message"])
                # raw 文件必须存在，否则下游取证据会缺文件。
                self.assertTrue((out / row["raw_stdout"]).is_file())
                self.assertTrue((out / row["raw_stderr"]).is_file())


class PlanParallelismTests(unittest.TestCase):
    def _experiment(self, root: Path, targets, execution_extra=None) -> ExperimentConfig:
        base = load_experiment(
            CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
        )
        execution = {**base.execution, "limit": 1}
        if execution_extra:
            execution.update(execution_extra)
        return replace(
            base,
            output_root=root / "output",
            execution=execution,
            targets=targets,
        )

    def _systems(self, root: Path, cli: Path, env_file: Path):
        registry = load_registry()
        implementation = dict(registry["native.claude-code"].implementation)
        implementation["env_file"] = str(env_file)
        return [replace(registry["native.claude-code"], implementation=implementation)]

    def test_parallel_plans_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 2)
            env_file = _write_claude_env(root, cli)
            system = self._systems(root, cli, env_file)[0]
            base = load_experiment(
                CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
            )
            # 两个 target 用不同 model_id，保证 run fingerprint 不同、目录不冲突。
            first = replace(base.targets[0], model_id="fake-a", model_label="fake-a")
            second = replace(base.targets[0], model_id="fake-b", model_label="fake-b")

            def wall_for(name: str, parallel: int) -> float:
                experiment = self._experiment(root / name, (first, second))
                plans = make_plans(experiment, [system, system], inspect_benchmark(experiment.benchmark))
                started = time.monotonic()
                results = execute_many(
                    experiment, [system, system], plans, parallel_plans=parallel
                )
                elapsed = time.monotonic() - started
                self.assertTrue(all(not result.get("error") for result in results), results)
                self.assertEqual(len({result["output_dir"] for result in results}), 2)
                return elapsed

            serial_wall = wall_for("serial", 1)
            parallel_wall = wall_for("parallel", 2)
            self.assertLess(parallel_wall, serial_wall * 0.8, (parallel_wall, serial_wall))

    def test_same_fingerprint_plans_are_rejected(self):
        """同一 adapter + 同一 model_id 的两个 target 会争同一个 run 目录，必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 0)
            env_file = _write_claude_env(root, cli)
            system = self._systems(root, cli, env_file)[0]
            base = load_experiment(
                CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
            )
            experiment = self._experiment(root, (base.targets[0], base.targets[0]))
            plans = make_plans(
                experiment, [system, system], inspect_benchmark(experiment.benchmark)
            )
            with self.assertRaises(ConfigurationError) as ctx:
                execute_many(experiment, [system, system], plans, parallel_plans=2)
            self.assertIn("会写同一个 run 目录", str(ctx.exception))

    def test_execute_many_keeps_plan_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = _fake_sleepy_claude(root / "fake-claude", 1)
            env_file = _write_claude_env(root, cli)
            system = self._systems(root, cli, env_file)[0]
            base = load_experiment(
                CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
            )
            first = replace(base.targets[0], model_id="fake-slow", model_label="fake-slow")
            second = replace(base.targets[0], model_id="fake-fast", model_label="fake-fast")
            experiment = self._experiment(root, (first, second))
            plans = make_plans(
                experiment, [system, system], inspect_benchmark(experiment.benchmark)
            )
            results = execute_many(experiment, [system, system], plans, parallel_plans=2)
            # 返回值顺序必须与 plans 顺序一致，调用方才不需要按 target 重新找。
            self.assertEqual(len(results), 2)
            self.assertEqual(
                [json.loads(Path(result["plan"]).read_text(encoding="utf-8"))["target_id"] for result in results],
                [plan.target_id for plan in plans],
            )
            self.assertTrue(all(result.get("returncode") == 0 for result in results))


class ConfigValidationTests(unittest.TestCase):
    def _load_with_execution(self, execution_lines: str) -> ExperimentConfig:
        template = (
            CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
        ).read_text(encoding="utf-8")
        template = template.split("[execution]")[0]
        template += "[execution]\nmode = \"smoke\"\nlimit = 1\n" + execution_lines
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(template, encoding="utf-8")
            return load_experiment(path)

    def test_parallel_fields_are_validated(self):
        ok = self._load_with_execution("questions_in_parallel = 3\nparallel_plans = 2\n")
        self.assertEqual(ok.execution["questions_in_parallel"], 3)
        self.assertEqual(ok.execution["parallel_plans"], 2)
        for bad in ("questions_in_parallel = 0", "parallel_plans = -1", "questions_in_parallel = 1.5"):
            with self.assertRaises(ConfigurationError, msg=bad):
                self._load_with_execution(bad + "\n")

    def test_parallel_fields_do_not_change_fingerprint(self):
        registry = load_registry()
        base = load_experiment(
            CONFIG_ROOT / "experiments" / "templates" / "native-smoke.toml"
        )
        systems = [
            registry[target.system_id] for target in base.targets
        ]
        benchmark = inspect_benchmark(base.benchmark)
        plain = make_plans(base, systems, benchmark)
        parallel = make_plans(
            replace(
                base,
                execution={**base.execution, "questions_in_parallel": 4, "parallel_plans": 3},
            ),
            systems,
            benchmark,
        )
        self.assertEqual(
            [plan.config_fingerprint for plan in plain],
            [plan.config_fingerprint for plan in parallel],
        )
        # 但并发度必须留在 run_plan 里可追溯。
        self.assertEqual(parallel[0].execution["questions_in_parallel"], 4)


if __name__ == "__main__":
    unittest.main()
