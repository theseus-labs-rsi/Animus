from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_harnesses.artifacts import inspect_benchmark
from agent_harnesses.config import (
    CONFIG_ROOT,
    BenchmarkRef,
    ConfigurationError,
    load_experiment,
    load_registry,
    resolve_systems,
)
from agent_harnesses.planning import (
    allocate_run_dir,
    find_latest_run_dir,
    make_plans,
    run_fingerprint,
)
from agent_harnesses.execution import execute
from agent_harnesses.runners.diagnostic import run as run_diagnostic
from standard_light_fixture import build_standard_light


# 离线测试夹具，见 fixtures/README.md。
OFFICE_FILTERED = Path(__file__).resolve().parent / "fixtures" / "office__20260902-121616"
DIAGNOSTIC_TEMPLATE = (
    CONFIG_ROOT / "experiments" / "templates" / "diagnostic-smoke.toml"
)


class ArtifactTests(unittest.TestCase):
    def test_standard_light_validates_manifest_and_keeps_all_statuses(self):
        questions = [
            {"qid": "q-release", "question": "状态？", "line": "L1", "capability": "KU", "quality_status": "released"},
            {"qid": "q-reject", "question": "未知项？", "line": "L6", "capability": "L6_refusal", "quality_status": "rejected"},
        ]
        references = [
            {"qid": "q-release", "answer": "进行中", "answer_raw": "进行中", "answer_projection": "raw_gt", "quality_status": "released"},
            {"qid": "q-reject", "answer": "查无此记录", "answer_raw": "INSUFFICIENT_EVIDENCE", "answer_projection": "abstention:never_known", "quality_status": "rejected"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = build_standard_light(Path(tmp), questions, references)
            inspection = inspect_benchmark(BenchmarkRef(root, False, "as_provided"))
            self.assertEqual(inspection.schema, "memory-bench-standard-light/v1")
            self.assertEqual(inspection.n_questions, 2)
            self.assertEqual(len(inspection.files), 6)

            out = Path(tmp) / "diagnostic"
            summary = run_diagnostic(root, out)
            self.assertEqual(summary["n_questions_checked"], 2)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual([row["question_id"] for row in rows], ["q-release", "q-reject"])
            self.assertEqual([row["quality_status"] for row in rows], ["released", "rejected"])

    def test_standard_light_rejects_tampered_declared_file(self):
        questions = [
            {"qid": "q1", "question": "状态？", "line": "L1", "capability": "KU", "quality_status": "released"}
        ]
        references = [
            {"qid": "q1", "answer": "进行中", "answer_raw": "进行中", "answer_projection": "raw_gt", "quality_status": "released"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = build_standard_light(Path(tmp), questions, references)
            (root / "public" / "protocol.txt").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "身份不匹配"):
                inspect_benchmark(BenchmarkRef(root, False, "as_provided"))

    def test_benchmark_can_live_outside_repository_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / "factory-output" / "benchmark-a"
            external.mkdir(parents=True)
            for name in (
                "00_about.json",
                "05_corpus.json",
                "06_grounded_questions.json",
                "manifest.json",
                "validated.json",
            ):
                shutil.copy2(OFFICE_FILTERED / name, external / name)
            inspection = inspect_benchmark(
                BenchmarkRef(
                    path=external,
                    allow_unmet=True,
                    corpus_variant="filtered",
                )
            )
            self.assertEqual(inspection.path, external)
            self.assertEqual(inspection.n_questions, 40)

    def test_malformed_nested_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "00_about.json").write_text(
                json.dumps({"answer_protocol": {"rules": ["rule"]}}),
                encoding="utf-8",
            )
            (root / "05_corpus.json").write_text(
                json.dumps(
                    {
                        "corpus": {
                            "sessions": [
                                {
                                    "session_id": 0,
                                    "date": "2026-01-01",
                                    "docs": [{"doc_id": "d0", "type": "note"}],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "06_grounded_questions.json").write_text(
                json.dumps(
                    [
                        {
                            "line": "L1",
                            "capability": "value",
                            "question": "q?",
                            "gt": {"value": "a"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "content"):
                inspect_benchmark(
                    BenchmarkRef(
                        path=root,
                        allow_unmet=True,
                        corpus_variant="as_provided",
                    )
                )

    def test_unmet_input_is_rejected_by_default(self):
        with self.assertRaises(ConfigurationError):
            inspect_benchmark(
                BenchmarkRef(
                    path=OFFICE_FILTERED,
                    allow_unmet=False,
                    corpus_variant="filtered",
                )
            )

    def test_diagnostic_can_explicitly_accept_unmet(self):
        inspection = inspect_benchmark(
            BenchmarkRef(
                path=OFFICE_FILTERED,
                allow_unmet=True,
                corpus_variant="filtered",
            )
        )
        self.assertTrue(inspection.met_status.startswith("UNMET"))
        self.assertEqual(inspection.n_questions, 40)
        self.assertEqual(inspection.n_docs, 221)
        self.assertEqual(set(inspection.files), {
            "00_about.json",
            "05_corpus.json",
            "06_grounded_questions.json",
        })


class PlanTests(unittest.TestCase):
    def test_fingerprint_is_stable(self):
        registry = load_registry()
        experiment = load_experiment(DIAGNOSTIC_TEMPLATE)
        systems = resolve_systems(experiment, registry)
        benchmark = inspect_benchmark(experiment.benchmark)
        first = make_plans(experiment, systems, benchmark)[0]
        second = make_plans(experiment, systems, benchmark)[0]
        self.assertEqual(first.config_fingerprint, second.config_fingerprint)
        self.assertEqual(first.system["spec"]["native_tool_policy"]["mode"], "none")
        # run family 目录不含 hash，具体 run 目录才带时间戳与 fingerprint。
        self.assertEqual(
            Path(first.output_dir),
            experiment.output_root
            / experiment.experiment_id
            / benchmark.scenario
            / first.target_id,
        )
        self.assertRegex(first.run_stamp, r"^\d{8}-\d{6}$")

    def test_memory_config_enters_plan_and_fingerprint(self):
        registry = load_registry()
        experiment = load_experiment(
            CONFIG_ROOT / "experiments" / "templates" / "mem0-smoke.toml"
        )
        systems = resolve_systems(experiment, registry)
        benchmark = inspect_benchmark(experiment.benchmark)
        first = make_plans(experiment, systems, benchmark)[0]
        changed_target = replace(
            experiment.targets[0],
            memory_config={**experiment.targets[0].memory_config, "top_k": 4},
        )
        changed = make_plans(
            replace(experiment, targets=(changed_target,)), systems, benchmark
        )[0]
        self.assertEqual(first.memory_config["top_k"], 3)
        self.assertNotEqual(first.config_fingerprint, changed.config_fingerprint)

    def test_run_dir_naming_disambiguates_same_second(self):
        fp = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp)
            first = allocate_run_dir(family, "20260918-120000", fp)
            self.assertEqual(first.name, "20260918-120000__aaaaaaaaaaaa")
            first.mkdir()
            second = allocate_run_dir(family, "20260918-120000", fp)
            self.assertEqual(second.name, "20260918-120000__aaaaaaaaaaaa-2")
            second.mkdir()
            third = allocate_run_dir(family, "20260918-120000", fp)
            self.assertEqual(third.name, "20260918-120000__aaaaaaaaaaaa-3")

    def test_find_latest_run_dir_matches_fingerprint_and_prefers_newest(self):
        fp = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp)
            for name in (
                "20260918-120000__bbbbbbbbbbbb",
                "20260918-130000__bbbbbbbbbbbb-2",
                "20260918-140000__cccccccccccc",  # 别的 fingerprint，不匹配
                "not-a-run-dir",
            ):
                (family / name).mkdir()
            found = find_latest_run_dir(family, fp)
            self.assertEqual(found.name, "20260918-130000__bbbbbbbbbbbb-2")
            self.assertIsNone(find_latest_run_dir(family, "d" * 64))
            self.assertIsNone(find_latest_run_dir(family / "missing", fp))

    def test_keep_scratch_does_not_change_fingerprint(self):
        registry = load_registry()
        experiment = load_experiment(DIAGNOSTIC_TEMPLATE)
        systems = resolve_systems(experiment, registry)
        benchmark = inspect_benchmark(experiment.benchmark)
        plain = make_plans(experiment, systems, benchmark)[0]
        keeping = make_plans(
            replace(experiment, execution={**experiment.execution, "keep_scratch": True}),
            systems,
            benchmark,
        )[0]
        # 只影响产物保留的字段不进 fingerprint，否则排障时开会话会丢 --resume 落点。
        self.assertEqual(plain.config_fingerprint, keeping.config_fingerprint)
        # 但完整配置仍写进 run_plan，读者能看到这次是否保留了临时区。
        self.assertTrue(keeping.execution["keep_scratch"])
        self.assertNotIn("keep_scratch", plain.execution)

    def test_run_fingerprint_separates_config_and_runtime(self):
        config_fp = "1" * 64
        runtime = {"adapter": "claude", "binary_version": "1.0"}
        self.assertEqual(run_fingerprint(config_fp, None), config_fp)
        self.assertNotEqual(run_fingerprint(config_fp, runtime), config_fp)
        self.assertEqual(
            run_fingerprint(config_fp, runtime),
            run_fingerprint(config_fp, dict(runtime)),
        )
        self.assertNotEqual(
            run_fingerprint(config_fp, runtime),
            run_fingerprint(config_fp, {"adapter": "claude", "binary_version": "2.0"}),
        )

    def test_offline_diagnostic_executes_through_unified_runner(self):
        registry = load_registry()
        base = DIAGNOSTIC_TEMPLATE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            base = base.replace('output_root = "output/eval"',
                                f'output_root = "{(Path(tmp) / "output").as_posix()}"')
            base = base.replace("limit = 2", "limit = 1")
            config_path = Path(tmp) / "experiment.toml"
            config_path.write_text(base, encoding="utf-8")
            experiment = load_experiment(config_path)
            systems = resolve_systems(experiment, registry)
            benchmark = inspect_benchmark(experiment.benchmark)
            plan = make_plans(experiment, systems, benchmark)[0]
            result = execute(experiment, systems[0], plan)
            self.assertEqual(result["returncode"], 0)
            # plan.output_dir 是 run family 目录，具体 run 落在它的时间戳子目录里。
            out = Path(result["output_dir"])
            self.assertEqual(out.parent, Path(plan.output_dir))
            self.assertRegex(out.name, r"^\d{8}-\d{6}__[0-9a-f]{12}$")
            self.assertTrue((out / "run_plan.json").is_file())
            self.assertTrue((out / "execution.json").is_file())
            self.assertTrue((out / "summary.json").is_file())
            self.assertFalse((out / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
