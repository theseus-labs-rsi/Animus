"""统一执行分发：离线 diagnostic 与候选 Native CLI smoke 已接入。"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import REPOSITORY_ROOT, ConfigurationError, ExperimentConfig, SystemConfig
from .events import NORMALIZER_VERSION
from .boundaries import DETECTOR_VERSION
from .planning import (
    RunPlan,
    allocate_run_dir,
    find_latest_run_dir,
    fingerprint,
    run_dir_name,
    run_fingerprint,
    write_plan,
)
from .tracks import get_track_adapter

# runtime 身份字段：preflight 报告里哪些内容属于「运行环境」而不是「实验配置」。
RUNTIME_KEYS = (
    "adapter",
    "adapter_version",
    "model",
    "endpoint_host",
    "endpoint_fingerprint",
    "protocol_style",
    "runtime_options",
    "binary",
    "binary_version",
    "memory_runtime",
)

# run 目录分配 + run_plan.json 落盘必须互斥：两个 target 可能解析出完全相同的
# run_fingerprint（同一 adapter、同一 model_id、同一 endpoint），此时
# allocate_run_dir 的 `while exists()` 探测不是原子的——并发下两个 worker 会选到
# 同一个目录名，一个先建目录、另一个 mkdir 直接 FileExistsError。
# 锁只覆盖「选目录 + 落盘」这一小段，不覆盖执行。
_RUN_DIR_LOCK = threading.Lock()


def command_for(
    experiment: ExperimentConfig,
    system: SystemConfig,
    plan: RunPlan,
) -> list[str]:
    return get_track_adapter(system.track).build_command(experiment, system, plan)


def preflight(
    experiment: ExperimentConfig,
    system: SystemConfig,
    plan: RunPlan,
) -> dict[str, Any]:
    return get_track_adapter(system.track).preflight(system, experiment, plan)


def runtime_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """把 preflight 报告收敛成可入 fingerprint 的 runtime 身份。"""
    runtime = {key: report.get(key) for key in RUNTIME_KEYS}
    # ingest LLM 的实际路由（直连 vs no-think 网关）是 runtime 身份的一部分：换路由
    # 会改变抽取行为，两次 run 不该长得一样。只在报告确实声明了它时加入，因此
    # 不声明该字段的赛道（如 native）fingerprint 保持不变。
    ingest_llm = report.get("ingest_llm")
    if isinstance(ingest_llm, dict) and ingest_llm:
        runtime["ingest_llm"] = ingest_llm
    # 归一器/检测器版本进入 fingerprint：启发式变了必须换输出目录，
    # 而不是静默重新解释旧 run 的事件流。
    runtime["event_normalizer_version"] = NORMALIZER_VERSION
    runtime["boundary_detector_version"] = DETECTOR_VERSION
    return runtime


def planned_run_dir(plan: RunPlan, report: Mapping[str, Any] | None = None) -> Path:
    """算出这次 run 会写入的目录（与 execute 同一套命名）。

    同一秒内重复运行同一配置时，execute 会给目录追加 `-2`、`-3` 后缀；
    这里给出的是首选名。
    """
    has_runtime = report is not None and plan.runner in {"native_cli", "memory"}
    runtime = runtime_identity(report) if has_runtime else None
    run_fp = run_fingerprint(plan.config_fingerprint, runtime)
    return Path(plan.output_dir) / run_dir_name(plan.run_stamp, run_fp)


def _read_previous_plan(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_plan.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _execute_one(
    experiment: ExperimentConfig,
    system: SystemConfig,
    plan: RunPlan,
    report: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """执行一个 plan，并把它收敛成 run 目录、写 run_plan.json。

    preflight 由调用方在主线程统一做完（并发时不能每个 worker 各跑一次），
    目录也由调用方在主线程分配，worker 只负责写这个已定的目录。
    """
    runtime = None
    runtime_fp = None
    if system.runner in {"native_cli", "memory"}:
        runtime = runtime_identity(report)
        runtime_fp = fingerprint(runtime)
    run_fp = run_fingerprint(plan.config_fingerprint, runtime)

    # 目录 = <output_root>/<experiment>/<scenario>/<target>/<开始时间>__<run_fp[:12]>。
    # 新 run 一定新建目录（不覆盖历史）；--resume 才复用同 fingerprint 的最近一个。
    family = Path(plan.output_dir)
    # 选目录与落 run_plan 在锁内完成：并发 worker 不能同时探测到同一个空闲目录名。
    with _RUN_DIR_LOCK:
        existing = find_latest_run_dir(family, run_fp) if resume else None
        run_dir = existing or allocate_run_dir(family, plan.run_stamp, run_fp)
        created_at = plan.created_at
        run_stamp = plan.run_stamp
        if existing is not None:
            # 复用旧目录时保留它的原始开始时间：目录名里的时间戳必须和 run_plan 一致。
            previous = _read_previous_plan(existing)
            created_at = str(previous.get("created_at") or created_at)
            run_stamp = str(previous.get("run_stamp") or run_stamp)

        effective_plan = replace(
            plan,
            output_dir=str(run_dir),
            run_stamp=run_stamp,
            created_at=created_at,
            runtime=runtime,
            runtime_fingerprint=runtime_fp,
            run_fingerprint=run_fp,
        )
        # write_plan 用 mkdir(exist_ok=...) 建目录；新建路径必须 exist_ok=False，
        # 否则「目录已被别人占掉」会被静默忽略、两个 plan 共用一个 run 目录。
        plan_path = write_plan(
            effective_plan, exist_ok=existing is not None or resume
        )
    cmd = command_for(experiment, system, effective_plan)
    if resume:
        if system.runner != "native_cli":
            raise ConfigurationError(f"{system.system_id}: --resume 仅支持 native_cli runner")
        cmd.append("--resume")
    # Real files let us wait for the runner process itself without waiting for pipe EOF.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            cmd,
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
        stdout_file.seek(0)
        stderr_file.seek(0)
        runner_stdout = stdout_file.read()
        runner_stderr = stderr_file.read()
    record = {
        "schema": "agent-harnesses.execution/v2",
        "system_id": system.system_id,
        "output_dir": str(run_dir),
        "plan": str(plan_path),
        "resumed": bool(resume),
        "command": cmd,
        "returncode": completed.returncode,
        # 成功时 runner stdout 与 summary.json 逐字重复，只在失败时保留以便定位。
        "stdout_policy": "kept only when returncode != 0; on success it duplicates summary.json",
        "stdout": runner_stdout if completed.returncode else None,
        "stderr": runner_stderr or None,
    }
    execution_path = run_dir / "execution.json"
    execution_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "plan": str(plan_path),
        "execution": str(execution_path),
        "output_dir": str(run_dir),
        "returncode": completed.returncode,
    }
    if completed.returncode:
        result["error"] = (
            f"{system.system_id} 执行失败，returncode={completed.returncode}，见 {execution_path}"
        )
    return result


def execute(
    experiment: ExperimentConfig,
    system: SystemConfig,
    plan: RunPlan,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if not system.executable:
        raise ConfigurationError(f"{system.system_id}: 配置标记为不可执行（候选/TBD）")
    report = preflight(experiment, system, plan)
    if not report.get("ok"):
        details = "; ".join(str(item) for item in report.get("errors") or [])
        raise ConfigurationError(f"{system.system_id} preflight 失败: {details}")

    result = _execute_one(experiment, system, plan, report, resume=resume)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def preflight_all(
    experiment: ExperimentConfig,
    systems: Iterable[SystemConfig],
    plans: Iterable[RunPlan],
) -> dict[str, Mapping[str, Any]]:
    """在主线程一次性做完全部 preflight，返回 {target_id: report}。

    并发执行时不能在每个 worker 里各跑一次 preflight：它读 env 文件、探测二进制
    版本并可能打印警告，重复执行既浪费也会让诊断信息互相覆盖。
    """
    by_target = {
        target.target_id: system
        for target, system in zip(experiment.targets, tuple(systems), strict=True)
    }
    reports: dict[str, Mapping[str, Any]] = {}
    for plan in plans:
        reports[plan.target_id] = preflight(experiment, by_target[plan.target_id], plan)
    return reports


def _reject_colliding_plans(
    plans: Iterable[RunPlan],
    reports: Mapping[str, Mapping[str, Any]],
    systems_by_id: Mapping[str, SystemConfig],
) -> None:
    """拒绝两个 plan 解析到同一个 run 目录的情况。

    run 目录身份 = 配置 + system + model 的 fingerprint，不含 target_id。因此两个
    只有 target_id 不同、其余完全相同的 target（同一 adapter、同一 model_id、
    同一 endpoint）会解析出同一个 family 和同一个 run_fingerprint。新建 run 时
    allocate_run_dir 会用 `-2` 后缀区分，但 `--resume` 的「按 fingerprint 找最近
    目录」无法区分这两者，两个 plan 会写同一个 results.jsonl（后写的截断先写的）。
    这里显式拒绝，把静默的数据丢失换成可诊断的配置错误。
    """
    seen: dict[tuple[str, str], str] = {}
    for plan in plans:
        system = systems_by_id[plan.system_id]
        runtime = (
            runtime_identity(reports[plan.target_id])
            if system.runner in {"native_cli", "memory"}
            else None
        )
        key = (plan.output_dir, run_fingerprint(plan.config_fingerprint, runtime))
        other = seen.get(key)
        if other is not None:
            raise ConfigurationError(
                f"target {other} 与 {plan.target_id} 会写同一个 run 目录"
                f"（{plan.output_dir}，run fingerprint 相同）：run 目录身份不含 target_id，"
                "并发执行或 --resume 都会互相覆盖。请让这两个 target 的 model_id / "
                "endpoint_profile / protocol_style 至少有一项不同。"
            )
        seen[key] = plan.target_id


def execute_many(
    experiment: ExperimentConfig,
    systems: Iterable[SystemConfig],
    plans: Iterable[RunPlan],
    *,
    resume: bool = False,
    parallel_plans: int = 1,
) -> list[dict[str, Any]]:
    """按 plan 并行执行（每个 plan = 一个 benchmark × target 的独立 run）。

    串行路径（parallel_plans<=1）保持历史语义：preflight 失败或执行失败直接抛错。
    并行路径不可能「抛出并同时保留其它 plan 的产物」，因此改为把失败收敛成
    `{"error": ...}` 项：全部 plan 都跑完后，若存在失败则由调用方决定如何报错。
    每个失败项的 execution.json 仍然落盘，失败证据不丢。
    """
    plan_list = list(plans)
    system_list = list(systems)
    if not plan_list:
        return []
    systems_by_id = {system.system_id: system for system in system_list}
    for plan in plan_list:
        system = systems_by_id[plan.system_id]
        if not system.executable:
            raise ConfigurationError(f"{system.system_id}: 配置标记为不可执行（候选/TBD）")
    reports = preflight_all(experiment, system_list, plan_list)
    for plan in plan_list:
        report = reports[plan.target_id]
        if not report.get("ok"):
            details = "; ".join(str(item) for item in report.get("errors") or [])
            raise ConfigurationError(f"{plan.system_id} preflight 失败: {details}")
    _reject_colliding_plans(plan_list, reports, systems_by_id)

    def worker(plan: RunPlan) -> dict[str, Any]:
        try:
            return _execute_one(
                experiment,
                systems_by_id[plan.system_id],
                plan,
                reports[plan.target_id],
                resume=resume,
            )
        except Exception as exc:  # 单个 plan 失败不阻断其它 plan
            return {
                "target_id": plan.target_id,
                "system_id": plan.system_id,
                "returncode": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    workers = min(max(1, parallel_plans), len(plan_list))
    if workers <= 1:
        results = [worker(plan) for plan in plan_list]
        # 串行路径保持历史失败语义：直接抛错。
        for result in results:
            if result.get("error"):
                raise RuntimeError(result["error"])
        return results
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mb-plan") as pool:
        futures = {pool.submit(worker, plan): index for index, plan in enumerate(plan_list)}
        results = [{} for _ in plan_list]
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
