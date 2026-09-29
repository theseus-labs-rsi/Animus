"""将配置解析为稳定、可审阅的 run plan。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifacts import BenchmarkInspection
from .config import ExperimentConfig, ExperimentTarget, SystemConfig

# run 目录名：<运行开始本地时间>__<run fingerprint 前 12 位>[-序号]。
# 时间部分让人一眼看出这次运行何时开始，hash 后缀保证同配置可被 --resume 找回；
# 同一秒内重复运行同一配置时用尾部序号区分（`-2`、`-3`…），绝不覆盖已有目录。
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_DIR_RE = re.compile(r"^(\d{8}-\d{6})__([0-9a-f]{12})(?:-(\d+))?$")

# 只决定「产物留不留」、不改变实验结果的 execution 字段不进 fingerprint。
# 否则为了排障开 keep_scratch 会让 fingerprint 变化，--resume 找不到原目录并重跑全部题目。
# 并发度同理：它只改变调度顺序和 results.jsonl/events.jsonl 的落盘顺序，每题语义与
# 环境完全一致，下游（scoring）按 question_index 重排；因此换并发度必须能 --resume。
ARTIFACT_ONLY_EXECUTION_KEYS = ("keep_scratch", "questions_in_parallel", "parallel_plans")


@dataclass(frozen=True)
class RunPlan:
    schema: str
    experiment_id: str
    target_id: str
    track: str
    system_id: str
    system_status: str
    system_role: str
    runner: str
    executable: bool
    system: dict[str, Any]
    benchmark: dict[str, Any]
    protocol: dict[str, Any]
    execution: dict[str, Any]
    judge: dict[str, Any]
    pricing: dict[str, Any]
    # output_dir 是 run family 目录：一次 experiment 的同一 scenario + target 共用；
    # 具体 run 目录是它下面的 <run_stamp>__<run_fingerprint[:12]>，由 execution 决定。
    output_dir: str
    config_fingerprint: str
    created_at: str
    run_stamp: str
    model: dict[str, Any] | None = None
    answering_model: dict[str, Any] | None = None
    memory_config: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    runtime_fingerprint: str | None = None
    run_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_stamp_for(instant: datetime) -> str:
    """把某一时刻渲染为本地时间的 run 目录时间戳。"""
    return instant.astimezone().strftime(RUN_STAMP_FORMAT)


def run_fingerprint(config_fingerprint: str, runtime: dict[str, Any] | None) -> str:
    """一次 run 的身份 = 配置身份 + 运行环境身份。

    两者分开记录，但目录名只用一个合并 hash：配置变了或 CLI/endpoint 变了都必须
    落到新目录，不能复用旧 run 的部分结果。
    """
    if not runtime:
        return config_fingerprint
    return fingerprint({"config": config_fingerprint, "runtime": runtime})


def run_dir_name(run_stamp: str, run_fp: str) -> str:
    return f"{run_stamp}__{run_fp[:12]}"


def find_latest_run_dir(family: Path, run_fp: str) -> Path | None:
    """在同一 run family 下按 fingerprint 找回最近的 run 目录，供 --resume 复用。"""
    family = Path(family)
    if not family.is_dir():
        return None
    suffix = run_fp[:12]
    matches: list[Path] = []
    for path in family.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.fullmatch(path.name)
        if match and match.group(2) == suffix:
            matches.append(path)
    # 时间戳定宽且在最前，字典序即时间序；尾部序号不影响跨时间戳比较。
    return sorted(matches)[-1] if matches else None


def allocate_run_dir(family: Path, run_stamp: str, run_fp: str) -> Path:
    """为新 run 分配一个不冲突的目录；同一秒重复运行用尾部序号区分。"""
    family = Path(family)
    base = run_dir_name(run_stamp, run_fp)
    candidate = family / base
    index = 2
    while candidate.exists():
        candidate = family / f"{base}-{index}"
        index += 1
    return candidate


def model_for(target: ExperimentTarget, system: SystemConfig) -> dict[str, Any] | None:
    """Return the endpoint-scoped model identity declared by this run target."""
    model = target.model_dict()
    if model is not None:
        model["adapter"] = str(system.implementation.get("adapter") or "")
    return model


def make_plans(
    experiment: ExperimentConfig,
    systems: Iterable[SystemConfig],
    benchmarks: BenchmarkInspection | Iterable[BenchmarkInspection],
) -> list[RunPlan]:
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    run_stamp = run_stamp_for(now)
    if isinstance(benchmarks, BenchmarkInspection):
        benchmark_list = [benchmarks]
    else:
        benchmark_list = list(benchmarks)
    plans: list[RunPlan] = []
    for benchmark in benchmark_list:
        for target, system in zip(experiment.targets, systems, strict=True):
            model = model_for(target, system)
            # checkout 路径不改变实验语义；输入内容由文件 hash 锁定。
            benchmark_for_fingerprint = benchmark.to_dict()
            benchmark_for_fingerprint.pop("path", None)
            stable = {
                "experiment_id": experiment.experiment_id,
                "target_id": target.target_id,
                "track": experiment.track,
                "benchmark": benchmark_for_fingerprint,
                "system": {
                    "system_id": system.system_id,
                    "status": system.status,
                    "role": system.role,
                    "implementation": dict(system.implementation),
                    "spec": dict(system.spec),
                },
                "model": model,
                "answering_model": dict(experiment.answering_model or {}),
                "memory_config": dict(target.memory_config or {}),
                "protocol": dict(experiment.protocol),
                "execution": {
                    key: value
                    for key, value in experiment.execution.items()
                    if key not in ARTIFACT_ONLY_EXECUTION_KEYS
                },
                "judge": dict(experiment.judge),
                "pricing": dict(experiment.pricing),
            }
            fp = fingerprint(stable)
            out = (
                experiment.output_root
                / experiment.experiment_id
                / benchmark.scenario
                / target.target_id
            )
            plans.append(
                RunPlan(
                    schema="agent-harnesses.run-plan/v2",
                    experiment_id=experiment.experiment_id,
                    target_id=target.target_id,
                    track=experiment.track,
                    system_id=system.system_id,
                    system_status=system.status,
                    system_role=system.role,
                    runner=system.runner,
                    executable=system.executable,
                    system={
                        "display_name": system.display_name,
                        "implementation": dict(system.implementation),
                        "spec": dict(system.spec),
                        "source": str(system.source),
                    },
                    benchmark=benchmark.to_dict(),
                    protocol=dict(experiment.protocol),
                    execution=dict(experiment.execution),
                    judge=dict(experiment.judge),
                    pricing=dict(experiment.pricing),
                    output_dir=str(out),
                    config_fingerprint=fp,
                    created_at=created_at,
                    run_stamp=run_stamp,
                    model=model,
                    answering_model=dict(experiment.answering_model or {}) or None,
                    memory_config=dict(target.memory_config or {}) or None,
                )
            )
    return plans


def write_plan(plan: RunPlan, *, exist_ok: bool = False) -> Path:
    out = Path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=exist_ok)
    path = out / "run_plan.json"
    path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
