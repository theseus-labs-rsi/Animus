"""Memory System Track adapter。

把统一的 RunPlan 翻成 memory runner 的执行动作。与 native 赛道共用 harness 的
配置校验、run plan、产物与判分；只有「准备 + 作答」这层由 `runners/memory.py` 负责。

冻结口径（D-20260920-02，2026-09-20 由用户确认）：

- 固定回答模型：experiment 顶层 `answering_model`（本 run 的唯一模型，抽桥同用）；
- 检索：`simplemem` / `iterative` 的 `retrieval.top_k`；`fullcontext` 的
  `memory_system.full_context_char_budget`；
- `context_token_budget` 暂不强制（记实测值），`witness_required` 暂不强制为门；
- 一个世界：ingest 一次且串行；并行只作用于逐题作答。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..config import REPOSITORY_ROOT, ConfigurationError, ExperimentConfig, SystemConfig
from ..planning import RunPlan
from .base import TrackAdapter

# 需要本地 bge embedding 的内置实现（fullcontext 不需要）。
EMBEDDING_SYSTEMS = {"simplemem", "iterative"}
EMBEDDING_REQUIREMENTS = ("numpy", "sentence_transformers")


def _memory_system_id(system: SystemConfig) -> str:
    spec = system.spec if isinstance(system.spec, dict) else {}
    memory = spec.get("memory_system") or {}
    name = str(memory.get("id") or "").strip()
    if not name:
        raise ConfigurationError(f"{system.system_id}: spec.memory_system.id 不能为空")
    return name


def _probe_gateway(base_url: str, timeout: float = 3.0) -> str | None:
    """探测 no-think 网关可达性。返回 None 表示可达，否则返回简短的异常类型。

    显式绕过 HTTP(S)_PROXY：沙箱里代理会劫持 localhost 请求，本地网关会被误判不可达。
    只回传异常类型，不带 URL 细节、响应体或任何凭证。
    """
    import urllib.request  # noqa: PLC0415

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:
            response.read(1)
    except Exception as exc:  # noqa: BLE001 - 任何失败都等价于「网关不可用」
        return type(exc).__name__
    return None


class MemoryTrackAdapter(TrackAdapter):
    def preflight(
        self,
        system: SystemConfig,
        experiment: ExperimentConfig,
        plan: RunPlan,
    ) -> dict:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            name = _memory_system_id(system)
        except ConfigurationError as exc:
            name, errors = "", [str(exc)]

        # 1) 固定回答模型：experiment 顶层必须声明，且 system 不得带 target 级模型。
        answering = experiment.answering_model or {}
        if not answering:
            errors.append(
                "Memory Track 必须在 experiment 顶层固定 answering_model（model_id + endpoint_profile）"
            )
        if system.implementation.get("runner") != "memory":
            errors.append(
                f"{system.system_id}: implementation.runner 必须是 memory，"
                f"当前 {system.implementation.get('runner')!r}"
            )

        # 2) benchmark 与 endpoint 密钥。
        benchmark = Path(plan.benchmark.get("path", ""))
        if not benchmark.is_dir():
            errors.append(f"benchmark 目录不存在: {benchmark}")
        endpoint = {}
        env: dict[str, str] = {}
        if answering:
            profile = str(answering.get("endpoint_profile") or "")
            from ..runners.native_cli import _profile_names, load_env_file

            key_name, base_name = _profile_names(profile)
            env_file = REPOSITORY_ROOT / "configs" / "env" / "secrets.env"
            if not env_file.is_file():
                errors.append(f"缺失本地配置: {env_file}（从 secrets.env.example 复制）")
            else:
                env = load_env_file(env_file)
                for field, value in ((key_name, env.get(key_name)), (base_name, env.get(base_name))):
                    if not value:
                        errors.append(f"answering_model.endpoint_profile={profile} 缺少 {field}")
                endpoint = {
                    "endpoint_profile": profile,
                    "endpoint_host": str(env.get(base_name) or "").split("//", 1)[-1].split("/", 1)[0],
                }

        # 3) 内置实现的运行期依赖（fail-closed：缺依赖时不要跑到一半才炸）。
        missing = [
            module for module in (EMBEDDING_REQUIREMENTS if name in EMBEDDING_SYSTEMS else ())
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            errors.append(
                f"{system.system_id}({name}) 需要本地 bge embedding 依赖: "
                f"{', '.join(missing)} 未安装（pip install numpy sentence-transformers）"
            )
        elif name in EMBEDDING_SYSTEMS:
            if importlib.util.find_spec("transformers") is None:
                warnings.append("transformers 未安装；sentence-transformers 首次加载会自行解析")

        # 4) adapter-owned preflight。公共控制面只负责解析公开配置与 endpoint
        # profile；SDK/服务版本和健康检查由具体 adapter 声明，结果进入 runtime
        # fingerprint，但不得包含凭证或随机 namespace ID。
        memory_runtime: dict = {}
        adapter_version = None
        ingest_llm: dict = {}
        if name and not errors:
            try:
                from ..runners.memory import (
                    _ingest_llm_gateway,
                    _ingest_route,
                    _runtime_system_kwargs,
                    _system_kwargs,
                )
                from eval.memory_systems import preflight_system
                from eval.memory_systems.base import configuration_fingerprint

                _, public_kwargs = _system_kwargs(
                    system.spec if isinstance(system.spec, dict) else {},
                    plan.memory_config or {},
                )
                runtime_kwargs = _runtime_system_kwargs(name, public_kwargs, env)
                route = _ingest_route(name, public_kwargs, env, configuration_fingerprint) or {}
                ingest_llm = route

                # ingest LLM 走 no-think 网关时必须确认网关已在跑：否则要到 ingest 中途
                # 才炸（长语料几十分钟），成本极高。这里提前 fail-fast。
                gateway = _ingest_llm_gateway(env)
                if gateway["base_url"]:
                    failure = _probe_gateway(gateway["base_url"])
                    if failure:
                        errors.append(
                            f"{system.system_id}: INGEST_LLM_BASE_URL="
                            f"{gateway['base_url']} 不可达（{failure}）；"
                            "请先启动 no-think 网关（services/no_think_proxy.py 或 "
                            "services/embed_server.py），或清空 INGEST_LLM_BASE_URL 走直连"
                        )
                    else:
                        fingerprint = str(route.get("gateway_fingerprint") or "")
                        warnings.append(
                            f"{system.system_id}: ingest LLM 经 no-think 网关 "
                            f"{gateway['base_url']}（指纹 {fingerprint[:12]}）"
                        )
                adapter_report = preflight_system(name, **runtime_kwargs)
                errors.extend(str(item) for item in adapter_report.get("errors") or [])
                warnings.extend(str(item) for item in adapter_report.get("warnings") or [])
                runtime_value = adapter_report.get("memory_runtime") or {}
                if isinstance(runtime_value, dict):
                    memory_runtime = runtime_value
                    adapter_version = runtime_value.get("adapter_version")
            except Exception as exc:  # fail closed, never expose provider payloads
                errors.append(f"{system.system_id} adapter preflight 失败（{type(exc).__name__}）")

        spec = system.spec if isinstance(system.spec, dict) else {}
        retrieval = spec.get("retrieval") or {}
        return {
            "system_id": system.system_id,
            "adapter": "memory",
            "adapter_version": adapter_version,
            "ok": not errors,
            "memory_system": name,
            "track": "memory",
            "benchmark": str(benchmark),
            "answering_model": dict(answering),
            **endpoint,
            "retrieval": dict(retrieval),
            "memory_runtime": memory_runtime,
            "ingest_llm": ingest_llm,
            "python": sys.executable,
            "warnings": warnings,
            "errors": errors,
        }

    def build_command(
        self,
        experiment: ExperimentConfig,
        system: SystemConfig,
        plan: RunPlan,
    ) -> list[str]:
        if not plan.executable:
            raise ConfigurationError(f"{system.system_id}: system 未标记为可执行")
        limit = int(plan.execution.get("limit") or 0)
        return [
            sys.executable,
            "-m",
            "agent_harnesses.runners.memory",
            "--plan",
            str(Path(plan.output_dir) / "run_plan.json"),
            "--run",
            str(plan.benchmark.get("path")),
            "--out",
            str(plan.output_dir),
            "--limit",
            str(limit),
        ]
