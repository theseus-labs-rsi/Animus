"""Memory System Track 的离线回归。

不访问网络、不需要 embedding 依赖：把 runner 的 factory 入口与 secrets 读入替换成
确定性 fake，验证 harness 这一侧负责的东西——ingest 顺序、逐题并行语义等价、错误分型、
results.jsonl / summary.json 的字段与 native 赛道对齐、以及并发下 retrieve 只读。

真模型/真记忆系统的行为由真实 run 验证，不在本文件里。
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import unittest.mock
from pathlib import Path

from agent_harnesses import execution
from agent_harnesses.config import ConfigurationError, SystemConfig
from agent_harnesses.planning import RunPlan
from agent_harnesses.runners import memory as memory_runner
from agent_harnesses.tracks.memory import MemoryTrackAdapter
from eval.memory_systems import mem0_adapter


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class _FakeMemoryError(Exception):
    """与 factory MemoryExecutionError 同形的测试替身。"""

    def __init__(self, stage: str, code: str, details: dict | None = None):
        self.stage, self.code, self.details = stage, code, dict(details or {})
        super().__init__(f"memory {stage}: {code}")

    def as_dict(self) -> dict:
        return {"stage": self.stage, "code": self.code, "details": self.details}


class _FakeSystem:
    """确定性记忆系统：记录 ingest 顺序，retrieve 只读且可并发。"""

    def __init__(self, *, top_k: int = 3, fail_retrieve_on: set[int] | None = None,
                 slow_retrieve_s: float = 0.0):
        self.top_k = top_k
        self.fail_retrieve_on = fail_retrieve_on or set()
        self.slow_retrieve_s = slow_retrieve_s
        self.sessions: list[dict] = []
        self.finalized = False
        self.cleaned = False
        self.retrieve_calls: list[str] = []
        self._lock = threading.Lock()

    def ingest_session(self, session: dict) -> dict:
        self.sessions.append(session)
        return {"status": "ok", "n_docs": len(session["docs"]),
                "requested_docs": len(session["docs"]), "completion": "accepted"}

    def finalize_ingest(self, on_progress=None) -> None:
        self.finalized = True

    def retrieve(self, question: str, top_k: int = None) -> str:
        if self.slow_retrieve_s:
            time.sleep(self.slow_retrieve_s)
        with self._lock:
            index = len(self.retrieve_calls)
            self.retrieve_calls.append(question)
        if index in self.fail_retrieve_on:
            raise _FakeMemoryError("retrieve", "empty_index", {"index": index})
        return f"[片段1] 关于「{question}」的记忆"

    def get_diagnostics(self) -> dict:
        return {"top_k": self.top_k}

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "top_k": self.top_k}

    def cleanup(self) -> dict:
        self.cleaned = True
        return {"status": "ok", "completion": "not_applicable"}


def _fake_factory(system: _FakeSystem, *, answers: dict[str, str] | None = None,
                  answer_error_on: set[int] | None = None, seen_models: list | None = None):
    answers = answers or {}
    answer_error_on = answer_error_on or set()
    counter = {"n": 0}
    lock = threading.Lock()

    class _Config:
        MODEL = "unset"
        API_KEY = "unset"
        BASE_URL = "unset"

    def make_system(name: str, **kwargs):
        system.name = name
        system.kwargs = kwargs
        return system

    def unified_answer(question: str, context: str, protocol: str = "", max_tokens: int = 2048) -> str:
        with lock:
            index = counter["n"]
            counter["n"] += 1
        if index in answer_error_on:
            raise RuntimeError("answer transport broke")
        if seen_models is not None:
            seen_models.append(_Config.MODEL)
        return answers.get(question, f"答案:{question}")

    return {
        "config": _Config,
        "make_system": make_system,
        "unified_answer": unified_answer,
        "MemoryExecutionError": _FakeMemoryError,
        # 与真实 _load_factory 的契约一致：ingest 路由的 lineage 指纹由它计算。
        "configuration_fingerprint": lambda value: (
            hashlib.sha256(value.encode()).hexdigest() if value else None
        ),
        "load_public_protocol": lambda about_path: "【答题约定】只给最终值",
        "load_visible_corpus": lambda corpus_path: [
            (0, "2025-01-06", "docs-a"),
            (1, "2025-01-13", "docs-b"),
        ],
        "make_evaluation_context": lambda docs, protocol: {"docs": len(docs), "protocol": bool(protocol)},
    }


def _plan_dict(tmp: Path, *, system_id: str = "memory.simplemem",
               memory_id: str = "simplemem", top_k: int = 3,
               limit: int = 4, parallel: int = 1) -> dict:
    spec = {
        "memory_system": {"id": memory_id, "version": "repository"},
        "ingestion": {"mode": "session_ordered", "reset_scope": "run"},
        "retrieval": {"mode": "dense", "top_k": top_k},
    }
    if memory_id == "fullcontext":
        spec["memory_system"]["full_context_char_budget"] = 120_000
    return {
        "schema": "agent-harnesses.run-plan/v2",
        "experiment_id": "memory-test",
        "target_id": memory_id,
        "track": "memory",
        "system_id": system_id,
        "system_status": "candidate",
        "system_role": "benchmark",
        "runner": "memory",
        "executable": True,
        "system": {
            "display_name": memory_id,
            "implementation": {"runner": "memory", "executable": True},
            "spec": spec,
            "source": str(tmp / "systems.toml"),
        },
        "benchmark": {"path": str(tmp / "bench")},
        "protocol": {},
        "execution": {"mode": "smoke", "limit": limit, "questions_in_parallel": parallel},
        "judge": {},
        "pricing": {},
        "output_dir": str(tmp / "out"),
        "config_fingerprint": "x",
        "created_at": "2026-09-20T00:00:00+00:00",
        "run_stamp": "20260920-000000",
        "model": None,
        "answering_model": {
            "label": "deepseek-v4.1-flash",
            "model_id": "deepseek-v4.1-flash",
            "endpoint_profile": "DEEPSEEK",
            "protocol_style": None,
        },
    }


class MemoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        bench = self.tmp / "bench"
        bench.mkdir()
        self.questions = [{"question": f"q{i}?", "capability": "IE"} for i in range(5)]
        _write_json(bench / "06_grounded_questions.json", self.questions)
        _write_json(bench / "05_corpus.json", {"corpus": {"sessions": []}})
        _write_json(bench / "00_about.json", {"answer_protocol": {"rules": ["极简"]}})
        self.out = self.tmp / "out"
        self.plan_path = self.tmp / "run_plan.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, plan: dict, factory: dict, *, limit=None, parallel=None, env=None):
        _write_json(self.plan_path, plan)
        patches = [
            unittest.mock.patch.object(memory_runner, "_load_factory", lambda: factory),
            unittest.mock.patch.object(
                memory_runner, "load_env_file",
                lambda path: env or {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_BASE_URL": "http://example.invalid/v1"},
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return memory_runner.run(
            self.plan_path,
            self.out,
            plan["execution"]["limit"] if limit is None else limit,
            questions_in_parallel=parallel,
        )

    def _rows(self):
        return [json.loads(line) for line in (self.out / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_ingest_once_then_answer_all_questions(self):
        system = _FakeSystem()
        summary = self._run(_plan_dict(self.tmp), _fake_factory(system))
        self.assertEqual(system.sessions, [
            {"session_id": 0, "date": "2025-01-06", "docs": ["docs-a"]},
            {"session_id": 1, "date": "2025-01-13", "docs": ["docs-b"]},
        ])
        self.assertTrue(system.finalized)
        self.assertTrue(system.cleaned)
        self.assertEqual(summary["n_questions"], 4)
        self.assertEqual(summary["n_docs"], 2)
        self.assertEqual(summary["ingest"]["n_sessions"], 2)
        self.assertEqual(summary["cleanup"]["receipt"]["status"], "ok")
        self.assertEqual(summary["effective_memory_config"]["top_k"], 3)
        rows = self._rows()
        self.assertEqual([r["question_index"] for r in rows], [0, 1, 2, 3])
        self.assertTrue(all(r["status"] == "completed" for r in rows))
        self.assertEqual(rows[0]["answer"], "答案:q0?")
        self.assertEqual(rows[0]["retrieval"]["context_chars"], len("[片段1] 关于「q0?」的记忆"))

    def test_schema_fields_align_with_native(self):
        self._run(_plan_dict(self.tmp), _fake_factory(_FakeSystem()))
        row = self._rows()[0]
        self.assertEqual(row["schema"], "agent-harnesses.result/v1")
        for field in ("question_id", "question_index", "status", "answer", "error_type", "elapsed_s"):
            self.assertIn(field, row)
        for field in ("cli_report", "raw_stdout", "raw_stderr", "events", "telemetry",
                      "boundary_violation", "contaminated"):
            self.assertIn(field, row)
            self.assertIn(row[field], (None, False))
        self.assertIsNone(row["correct"])          # 判分由 score 负责
        self.assertFalse(row["judgeable"])

    def test_parallel_matches_serial_semantics(self):
        serial_plan = _plan_dict(self.tmp, parallel=1)
        serial = self._run(serial_plan, _fake_factory(_FakeSystem()), parallel=1)
        serial_rows = self._rows()
        serial_answers = {r["question_index"]: r["answer"] for r in serial_rows}

        self.out = self.tmp / "out-parallel"
        parallel = self._run(_plan_dict(self.tmp, parallel=4), _fake_factory(_FakeSystem()), parallel=4)
        parallel_rows = self._rows()
        self.assertEqual(serial["n_questions"], parallel["n_questions"])
        self.assertEqual(parallel["concurrency"]["workers_used"], 4)
        for row in parallel_rows:
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["answer"], serial_answers[row["question_index"]])
        self.assertEqual([r["question_index"] for r in parallel_rows], [0, 1, 2, 3])

    def test_parallel_workers_actually_overlap(self):
        system = _FakeSystem(slow_retrieve_s=0.4)
        started = time.monotonic()
        summary = self._run(_plan_dict(self.tmp, limit=4, parallel=4), _fake_factory(system), parallel=4)
        elapsed = time.monotonic() - started
        self.assertEqual(summary["concurrency"]["questions_dispatched"], 4)
        # 串行下 4 × 0.4s ≈ 1.6s；并发应显著更快。
        self.assertLess(elapsed, 1.2)

    def test_retrieve_failure_is_typed_and_isolated(self):
        system = _FakeSystem(fail_retrieve_on={1})
        summary = self._run(_plan_dict(self.tmp), _fake_factory(system))
        rows = {r["question_index"]: r for r in self._rows()}
        self.assertEqual(rows[1]["status"], "failed")
        self.assertEqual(rows[1]["error_type"], "memory_retrieve_error")
        self.assertEqual(rows[1]["error_detail"]["code"], "empty_index")
        self.assertEqual(summary["n_errors"], 1)
        self.assertEqual(summary["n_errors_by_type"], {"memory_retrieve_error": 1})
        self.assertEqual(rows[0]["status"], "completed")

    def test_answer_failure_is_typed(self):
        summary = self._run(_plan_dict(self.tmp), _fake_factory(_FakeSystem(), answer_error_on={0}))
        rows = {r["question_index"]: r for r in self._rows()}
        self.assertEqual(rows[0]["error_type"], "memory_answer_error")
        self.assertEqual(summary["n_errors"], 1)

    def test_ingest_failure_marks_every_question_without_calling_retrieve(self):
        system = _FakeSystem()
        system.ingest_session = lambda session: (_ for _ in ()).throw(_FakeMemoryError("ingest", "boom"))
        summary = self._run(_plan_dict(self.tmp), _fake_factory(system))
        rows = self._rows()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["error_type"] == "memory_ingest_error" for r in rows))
        self.assertTrue(all(r["answer"] is None for r in rows))
        self.assertEqual(system.retrieve_calls, [])
        self.assertEqual(summary["n_errors"], 4)

    def test_answering_model_is_bound_before_factory_import(self):
        seen: list[str] = []
        factory = _fake_factory(_FakeSystem(), seen_models=seen)
        import os

        self._run(_plan_dict(self.tmp), factory)
        # runner 必须把 experiment 的模型写进进程环境，供 factory config 读取。
        self.assertEqual(os.environ.get("MODEL"), "deepseek-v4.1-flash")
        self.assertEqual(os.environ.get("OPENAI_BASE_URL"), "http://example.invalid/v1")
        self.assertEqual(factory["config"].MODEL, "deepseek-v4.1-flash")
        self.assertTrue(seen and set(seen) == {"deepseek-v4.1-flash"})

    def test_missing_answering_model_is_rejected(self):
        plan = _plan_dict(self.tmp)
        plan["answering_model"] = {}
        with self.assertRaises(ConfigurationError):
            self._run(plan, _fake_factory(_FakeSystem()))

    def test_system_kwargs_mapping(self):
        self.assertEqual(memory_runner._system_kwargs(_plan_dict(self.tmp)["system"]["spec"]),
                         ("simplemem", {"top_k": 3}))
        iterative_spec = {
            "memory_system": {"id": "iterative"},
            "retrieval": {"mode": "iterative", "top_k": 3, "hop_max_tokens": 4096},
        }
        self.assertEqual(memory_runner._system_kwargs(iterative_spec),
                         ("iterative", {"top_k": 3, "max_tokens": 4096}))
        # 未声明 hop_max_tokens 时回落到 legacy 默认 2048。
        iterative_default = {"memory_system": {"id": "iterative"}, "retrieval": {"top_k": 3}}
        self.assertEqual(memory_runner._system_kwargs(iterative_default),
                         ("iterative", {"top_k": 3, "max_tokens": 2048}))
        fullctx = _plan_dict(self.tmp, system_id="memory.fullcontext", memory_id="fullcontext")["system"]["spec"]
        self.assertEqual(memory_runner._system_kwargs(fullctx),
                         ("fullcontext", {"budget": 120_000}))
        mem0_spec = {
            "memory_system": {"id": "mem0"},
            "retrieval": {"mode": "native", "top_k": 3},
        }
        mem0_config = {
            "top_k": 3,
            "namespace_prefix": "mem0_eval",
            "internal_model": {"model_id": "extractor", "endpoint_profile": "INGEST"},
            "embedder": {"provider": "huggingface", "model_id": "BAAI/bge-small-zh-v1.5"},
            "vector_store": {"provider": "qdrant", "host": "127.0.0.1", "port": 6333},
        }
        name, public = memory_runner._system_kwargs(mem0_spec, mem0_config)
        self.assertEqual(name, "mem0")
        self.assertEqual(public["top_k"], 3)
        self.assertEqual(public["llm_endpoint_profile"], "INGEST")
        runtime = memory_runner._runtime_system_kwargs(
            name,
            public,
            {"INGEST_API_KEY": "secret", "INGEST_BASE_URL": "http://ingest.invalid/v1"},
        )
        self.assertNotIn("llm_endpoint_profile", runtime)
        self.assertEqual(runtime["llm_model"], "extractor")
        self.assertEqual(runtime["llm_api_key"], "secret")

    def test_mem0_runtime_state_is_pinned_and_telemetry_disabled(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            memory_runner._prepare_external_runtime("mem0")
            import os
            self.assertTrue(os.environ["MEM0_DIR"].endswith("output/eval/_mem0_state"))
            self.assertEqual(os.environ["MEM0_TELEMETRY"], "False")


class MemoryTrackAdapterTests(unittest.TestCase):
    def test_build_command_uses_run_dir_plan(self):
        adapter = MemoryTrackAdapter()
        run_plan = RunPlan(
            schema="agent-harnesses.run-plan/v2", experiment_id="e", target_id="t",
            track="memory", system_id="memory.simplemem", system_status="candidate",
            system_role="benchmark", runner="memory", executable=True,
            system={}, benchmark={"path": "/bench"}, protocol={},
            execution={"mode": "smoke", "limit": 6}, judge={}, pricing={},
            output_dir="/tmp/run-dir", config_fingerprint="x",
            created_at="2026-09-20T00:00:00+00:00", run_stamp="20260920-000000",
        )
        system = SystemConfig(
            schema_version=2, system_id="memory.simplemem", track="memory",
            display_name="s", status="candidate", role="benchmark",
            implementation={"runner": "memory", "executable": True},
            spec={"memory_system": {"id": "simplemem"}}, source=Path("/tmp/systems.toml"),
        )
        command = adapter.build_command(None, system, run_plan)
        self.assertEqual(command[:3], [__import__("sys").executable, "-m", "agent_harnesses.runners.memory"])
        self.assertIn("--plan", command)
        self.assertEqual(command[command.index("--plan") + 1], "/tmp/run-dir/run_plan.json")
        self.assertEqual(command[command.index("--limit") + 1], "6")

    def test_unknown_memory_system_is_rejected(self):
        from agent_harnesses.runners.memory import _system_kwargs

        with self.assertRaises(ConfigurationError):
            _system_kwargs({"memory_system": {}})


class Mem0AdapterContractTests(unittest.TestCase):
    def test_preflight_records_versions_and_fingerprints_without_secret(self):
        response = unittest.mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"version": "1.15.5"}
        versions = {
            "mem0ai": "2.0.7",
            "qdrant-client": "1.18.0",
            "sentence-transformers": "6.1.0",
        }
        with (
            unittest.mock.patch.object(mem0_adapter.importlib.util, "find_spec", return_value=object()),
            unittest.mock.patch.object(
                mem0_adapter.importlib.metadata,
                "version",
                side_effect=lambda name: versions[name],
            ),
            unittest.mock.patch.object(mem0_adapter.httpx, "get", return_value=response),
        ):
            report = mem0_adapter.Mem0Adapter.preflight(
                top_k=3,
                llm_model="extractor",
                llm_base_url="https://user:private@model.invalid/v1",
                llm_api_key="super-secret",
                qdrant_host="127.0.0.1",
                qdrant_port=6333,
                qdrant_expected_version="1.15.5",
                embedding_model="BAAI/bge-small-zh-v1.5",
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["memory_runtime"]["package_versions"], versions)
        self.assertEqual(
            report["memory_runtime"]["vector_store"]["service_version"], "1.15.5"
        )
        encoded = json.dumps(report)
        self.assertNotIn("super-secret", encoded)
        self.assertNotIn("private", encoded)

    def test_cleanup_deletes_current_collection_without_rebuilding(self):
        client = unittest.mock.Mock()
        qdrant = types.SimpleNamespace(QdrantClient=unittest.mock.Mock(return_value=client))
        instance = mem0_adapter.Mem0Adapter.__new__(mem0_adapter.Mem0Adapter)
        instance._qdrant_host = "127.0.0.1"
        instance._qdrant_port = 6333
        instance._collection = "mem0_eval_private"
        instance._last_context = "old"
        instance._rebuild_memory = unittest.mock.Mock()
        with unittest.mock.patch.dict(sys.modules, {"qdrant_client": qdrant}):
            receipt = instance.cleanup()
        self.assertEqual(receipt["completion"], "deleted")
        client.delete_collection.assert_called_once_with("mem0_eval_private")
        client.close.assert_called_once()
        instance._rebuild_memory.assert_not_called()


class IngestGatewayRoutingTest(unittest.TestCase):
    """ingest LLM 的 no-think 网关路由（正式链路）与 lineage 记录。

    背景：推理模型的 reasoning_content 会吃掉补全预算，直连时抽取结果被截断成非法
    JSON，触发 adapter 的 json 守卫 fail-closed —— 长语料必然中途全盘中止，且调大
    max_tokens 只能把失败点往后推。网关统一关 thinking 是唯一确定性修法，
    见 output/local_services/mem0_smoke_report.md 的三次 run 对照。
    """

    PROFILE_ENV = {
        "DEEPSEEK_API_KEY": "profile-key",
        "DEEPSEEK_BASE_URL": "https://upstream.invalid/v1",
    }
    PUBLIC = {
        "llm_model": "deepseek-v4-flash",
        "llm_endpoint_profile": "DEEPSEEK",
        "top_k": 3,
    }

    def setUp(self):
        # 测试必须对宿主机环境免疫：只由用例自己决定网关是否配置。
        patcher = unittest.mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("INGEST_LLM_BASE_URL", "INGEST_LLM_API_KEY", "INGEST_LLM_MODEL"):
            os.environ.pop(name, None)

    def _kwargs(self, extra):
        env = dict(self.PROFILE_ENV)
        env.update(extra)
        return memory_runner._runtime_system_kwargs("mem0", dict(self.PUBLIC), env)

    def test_direct_when_gateway_not_configured(self):
        runtime = self._kwargs({})
        self.assertEqual(runtime["llm_base_url"], "https://upstream.invalid/v1")
        self.assertEqual(runtime["llm_api_key"], "profile-key")

    def test_gateway_overrides_declared_profile(self):
        runtime = self._kwargs({"INGEST_LLM_BASE_URL": "http://127.0.0.1:9800/v1"})
        self.assertEqual(runtime["llm_base_url"], "http://127.0.0.1:9800/v1")
        # 网关不校验 key，但 mem0 的 OpenAIConfig 要求 api_key 非空 → 回退 profile key。
        self.assertEqual(runtime["llm_api_key"], "profile-key")

    def test_gateway_key_and_model_override(self):
        runtime = self._kwargs({
            "INGEST_LLM_BASE_URL": "http://127.0.0.1:9800/v1",
            "INGEST_LLM_API_KEY": "gateway-key",
            "INGEST_LLM_MODEL": "Qwen3-8B",
        })
        self.assertEqual(runtime["llm_api_key"], "gateway-key")
        self.assertEqual(runtime["llm_model"], "Qwen3-8B")

    def test_gateway_falls_back_to_process_env(self):
        """services/run_eval.sh 用 export 注入网关地址，不走 secrets.env。"""
        with unittest.mock.patch.dict(
            os.environ, {"INGEST_LLM_BASE_URL": "http://127.0.0.1:9999/v1"}
        ):
            runtime = self._kwargs({})
        self.assertEqual(runtime["llm_base_url"], "http://127.0.0.1:9999/v1")

    def test_route_records_both_fingerprints(self):
        env = dict(self.PROFILE_ENV, INGEST_LLM_BASE_URL="http://127.0.0.1:9800/v1")
        route = memory_runner._ingest_route(
            "mem0", dict(self.PUBLIC), env, lambda v: f"fp:{v}" if v else None
        )
        self.assertEqual(route["route"], "gateway")
        self.assertEqual(route["declared_profile"], "DEEPSEEK")
        # 网关是本地地址，真实上游必须单独可见：否则「同网关不同上游」的两次 run
        # 会长得一模一样（与 max_tokens 不入 fingerprint 是同一类 lineage 缺口）。
        self.assertEqual(
            route["declared_upstream_fingerprint"], "fp:https://upstream.invalid/v1"
        )
        self.assertEqual(route["gateway_fingerprint"], "fp:http://127.0.0.1:9800/v1")

    def test_route_is_direct_without_gateway(self):
        route = memory_runner._ingest_route(
            "mem0", dict(self.PUBLIC), dict(self.PROFILE_ENV),
            lambda v: f"fp:{v}" if v else None,
        )
        self.assertEqual(route["route"], "direct")
        self.assertIsNone(route["gateway_fingerprint"])

    def test_non_mem0_system_has_no_route(self):
        self.assertIsNone(memory_runner._ingest_route("simplemem", {}, {}, lambda v: v))


class RuntimeIdentityTest(unittest.TestCase):
    """ingest 路由必须进 fingerprint：换路由会改变抽取行为，两次 run 不该同名。"""

    def test_ingest_llm_enters_fingerprint_when_declared(self):
        identity = execution.runtime_identity({"ingest_llm": {"route": "gateway"}})
        self.assertEqual(identity["ingest_llm"], {"route": "gateway"})

    def test_native_report_fingerprint_unchanged(self):
        """不声明 ingest_llm 的赛道（native）fingerprint 不应被这次改动影响。"""
        identity = execution.runtime_identity({"adapter": "native_cli", "model": "m"})
        self.assertNotIn("ingest_llm", identity)

    def test_route_change_changes_fingerprint(self):
        direct = execution.runtime_identity({"ingest_llm": {"route": "direct"}})
        gateway = execution.runtime_identity({"ingest_llm": {"route": "gateway"}})
        self.assertNotEqual(direct, gateway)


if __name__ == "__main__":
    unittest.main()
