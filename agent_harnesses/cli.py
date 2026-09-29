"""统一评测入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import inspect_benchmark
from .config import (
    CONFIG_ROOT,
    ConfigurationError,
    load_experiment,
    load_registry,
    resolve_systems,
)
from .judging import load_judge, resolve_factory_root
from .planning import make_plans
from .execution import command_for, execute_many, planned_run_dir, preflight
from .scoring import score_run


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load(experiment_path: Path, config_root: Path):
    registry = load_registry(config_root)
    experiment = load_experiment(experiment_path, config_root)
    systems = resolve_systems(experiment, registry)
    benchmarks = [inspect_benchmark(ref) for ref in experiment.benchmarks]
    plans = make_plans(experiment, systems, benchmarks)
    return registry, experiment, systems, benchmarks, plans


def cmd_systems(args) -> int:
    registry = load_registry(Path(args.config_root))
    rows = []
    for item in sorted(registry.values(), key=lambda x: (x.track, x.system_id)):
        if args.track and item.track != args.track:
            continue
        rows.append(
            {
                "system_id": item.system_id,
                "track": item.track,
                "status": item.status,
                "role": item.role,
                "runner": item.runner,
                "executable": item.executable,
                "display_name": item.display_name,
            }
        )
    _json(rows)
    return 0


def cmd_validate(args) -> int:
    _, experiment, systems, benchmarks, plans = _load(
        Path(args.experiment), Path(args.config_root)
    )
    _json(
        {
            "ok": True,
            "experiment_id": experiment.experiment_id,
            "track": experiment.track,
            "systems": [s.system_id for s in systems],
            "targets": [
                {
                    "id": target.target_id,
                    "system": target.system_id,
                    "model": target.model_dict(),
                    "memory_config": dict(target.memory_config or {}) or None,
                }
                for target in experiment.targets
            ],
            "benchmarks": [b.to_dict() for b in benchmarks],
            "plans": [p.to_dict() for p in plans],
        }
    )
    return 0


def cmd_preflight(args) -> int:
    _, experiment, systems, benchmarks, plans = _load(
        Path(args.experiment), Path(args.config_root)
    )
    reports = []
    for target, system in zip(experiment.targets, systems, strict=True):
        target_plan = next(item for item in plans if item.target_id == target.target_id)
        report = preflight(experiment, system, target_plan)
        report["target_id"] = target.target_id
        report["planned_output_dirs"] = [
            str(planned_run_dir(item, report)) if report.get("ok") else item.output_dir
            for item in plans
            if item.target_id == target.target_id
        ]
        reports.append(report)
    ok = all(report.get("ok") for report in reports)
    _json(
        {
            "ok": ok,
            "experiment_id": experiment.experiment_id,
            "benchmarks": [b.to_dict() for b in benchmarks],
            "systems": reports,
        }
    )
    return 0 if ok else 2


def _execution_option(experiment, args, attribute: str, key: str) -> int:
    """并发度取值：命令行显式给出优先，其次 experiment.execution.<key>，最后 1（串行）。"""
    value = getattr(args, attribute, None)
    if value is not None:
        return int(value)
    raw = experiment.execution.get(key, 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigurationError(f"execution.{key} 必须是正整数，当前为 {raw!r}")
    return raw


def cmd_run(args) -> int:
    _, experiment, systems, _, plans = _load(
        Path(args.experiment), Path(args.config_root)
    )
    parallel_plans = _execution_option(experiment, args, "parallel_plans", "parallel_plans")
    if not args.execute:
        rendered = []
        by_target = {
            target.target_id: system
            for target, system in zip(experiment.targets, systems, strict=True)
        }
        for plan in plans:
            system = by_target[plan.target_id]
            command = None
            blocked_reason = None
            try:
                command = command_for(experiment, system, plan) if system.executable else None
                if not system.executable:
                    blocked_reason = "candidate/TBD: implementation.executable=false"
            except ConfigurationError as exc:
                blocked_reason = str(exc)
            row = plan.to_dict()
            row["command"] = command
            row["blocked_reason"] = blocked_reason
            rendered.append(row)
        _json({"dry_run": True, "plans": rendered})
        return 0

    results = execute_many(
        experiment,
        systems,
        plans,
        resume=args.resume,
        parallel_plans=parallel_plans,
    )
    failures = [result for result in results if result.get("error")]
    _json({"dry_run": False, "results": results})
    if failures:
        # 并行时其它 plan 已经跑完，不能抛错丢掉它们的产物；把失败汇总到 stderr 并返回非零。
        for result in failures:
            print(f"error: {result['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_score(args) -> int:
    judge_cfg: dict = {}
    pricing: dict = {}
    use_llm = not args.no_llm
    if args.experiment:
        experiment = load_experiment(Path(args.experiment))
        judge_cfg = dict(experiment.judge)
        pricing = dict(experiment.pricing)
        if not args.no_llm:
            use_llm = bool(judge_cfg.get("use_llm", True))
    if args.factory_root:
        judge_cfg["factory_root"] = args.factory_root
    factory_root = resolve_factory_root(judge_cfg)
    judge = load_judge(factory_root)
    summary = score_run(
        Path(args.out),
        judge,
        use_llm=use_llm,
        force=args.force,
        pricing=pricing,
    )
    if summary["advisory"]:
        print(
            "warning: 该 run 的 protocol.scoring=disabled_for_chain_smoke，"
            "判分结果仅供链路验证，不进正式榜。",
            file=sys.stderr,
        )
    _json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harnesses",
        description="Memory Bench 的配置驱动统一评测入口",
    )
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    systems = sub.add_parser("systems", help="列出 registry 中的被测对象")
    systems.add_argument("--track", choices=["native", "memory"])
    systems.set_defaults(func=cmd_systems)
    validate = sub.add_parser("validate", help="校验 experiment、system 与 benchmark")
    validate.add_argument("--experiment", required=True)
    validate.set_defaults(func=cmd_validate)
    check = sub.add_parser("preflight", help="检查本地 env、CLI 与 benchmark，不调用模型")
    check.add_argument("--experiment", required=True)
    check.set_defaults(func=cmd_preflight)
    run = sub.add_parser("run", help="生成 run plan；显式 --execute 才执行")
    run.add_argument("--experiment", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--resume",
        action="store_true",
        help="复用已有输出目录，跳过 results.jsonl 中已完成的题",
    )
    run.add_argument(
        "--parallel-plans",
        type=int,
        default=None,
        help="并发执行的 plan 数（benchmark × target）；缺省读 experiment 的 "
        "execution.parallel_plans，再缺省为 1（串行）",
    )
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score", help="对已完成 run 的 results.jsonl 判分并聚合")
    score.add_argument("--out", required=True, help="run 输出目录（含 run_plan.json 与 results.jsonl）")
    score.add_argument("--experiment", help="可选：从 experiment 读取 judge/pricing 配置")
    score.add_argument("--factory-root", help="memory-bench-factory 仓库路径（覆盖配置与环境变量）")
    score.add_argument("--no-llm", action="store_true", help="只用字面判分，不调用 LLM 兜底")
    score.add_argument("--force", action="store_true", help="覆盖已有 judged.jsonl")
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigurationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
