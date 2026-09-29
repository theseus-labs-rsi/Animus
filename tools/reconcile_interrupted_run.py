#!/usr/bin/env python3
"""修复中断恢复造成的 01/02 有效规模漂移，并重算最终闭环账本。

该工具不生成、补写或修改世界事实，只允许在 01/02 结构签名一致时，把
02 内嵌的实际 world_blueprint 重新发布为 01 的有效规模合同；随后完全根据
现有 06_grounding_report 与 manifest.targetspec 重算 MET/UNMET。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from pipeline.run import RUNS_DIR, _atomic_write_json
from pipeline.world_blueprint import normalize_world_blueprint, structure_signature


def _read(path: Path) -> dict:
    """读取一个 JSON 对象。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须是 JSON object")
    return value


def _sha(path: Path) -> str:
    """返回文件 SHA256，作为修复前证据。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(run_dir: Path, *, write: bool) -> dict:
    """计算并可选提交一次可审计的恢复收口。"""
    whitepaper_path = run_dir / "01_whitepaper.json"
    world_path = run_dir / "02_world.json"
    report_path = run_dir / "06_grounding_report.json"
    manifest_path = run_dir / "manifest.json"
    whitepaper = _read(whitepaper_path)
    world = _read(world_path)
    report = _read(report_path)
    manifest = _read(manifest_path)

    base = normalize_world_blueprint(whitepaper)
    effective = normalize_world_blueprint({"world_blueprint": world.get("world_blueprint")})
    if structure_signature(base) != structure_signature(effective):
        raise ValueError("01/02 语义结构签名不一致，拒绝机械修复")
    if int(world.get("n_sessions", 0) or 0) != int(effective["temporal_model"]["n_sessions"]):
        raise ValueError("02.n_sessions 与其内嵌 blueprint 不一致，拒绝机械修复")

    repaired_whitepaper = copy.deepcopy(whitepaper)
    repaired_whitepaper["world_blueprint"] = copy.deepcopy(world["world_blueprint"])
    shared = repaired_whitepaper.setdefault("shared_world_spec", {})
    shared.setdefault("entities", {})["count"] = sum(
        int(item.get("count", 0) or 0) for item in effective.get("entity_types", []))
    temporal = effective.get("temporal_model", {})
    timeline = shared.setdefault("timeline", {})
    timeline.update({key: temporal[key] for key in ("n_sessions", "unit", "cadence", "step_days")
                     if key in temporal})

    target = (manifest.get("algo") or {}).get("targetspec") or {}
    floors = target.get("per_line_min") or {}
    by_line = report.get("by_line") or {}
    per_line_final = {
        line_id: int((by_line.get(line_id) or {}).get("grounded", 0) or 0)
        for line_id in floors
    }
    grounded = int((report.get("overall") or {}).get("grounded", 0) or 0)
    met = (grounded >= int(target.get("min_questions", 0) or 0)
           and all(per_line_final[line_id] >= int(floor)
                   for line_id, floor in floors.items()))

    before = {
        "whitepaper_sha256": _sha(whitepaper_path),
        "world_sha256": _sha(world_path),
        "manifest_sha256": _sha(manifest_path),
        "whitepaper_sessions": int(base["temporal_model"]["n_sessions"]),
        "world_sessions": int(world["n_sessions"]),
        "met_status": (manifest.get("algo") or {}).get("met_status"),
    }
    result = {
        "run_id": run_dir.name,
        "action": "publish_02_effective_contract_and_recompute_06_ledger",
        "before": before,
        "after": {
            "effective_sessions": int(effective["temporal_model"]["n_sessions"]),
            "effective_entities": len(world.get("entities") or {}),
            "grounded": grounded,
            "per_line_final": per_line_final,
            "met_status": "MET" if met else "UNMET_GROUNDING",
        },
    }
    if write:
        _atomic_write_json(whitepaper_path, repaired_whitepaper)
        manifest.setdefault("algo", {}).update({
            "per_line_final": per_line_final,
            "met_status": "MET" if met else "UNMET_GROUNDING",
        })
        result["committed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest.setdefault("recovery_history", []).append(result)
        _atomic_write_json(manifest_path, manifest)
    return result


def main() -> None:
    """解析 Run 并默认 dry-run；显式 --write 才提交。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run id 或 Run 目录")
    parser.add_argument("--write", action="store_true", help="提交修复；默认只预览")
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser()
    if not run_dir.is_dir():
        run_dir = RUNS_DIR / args.run
    print(json.dumps(reconcile(run_dir.resolve(), write=args.write),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
