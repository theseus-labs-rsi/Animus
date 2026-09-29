"""Read-only status and acceptance check for a v6 seed-to-questions canary."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def audit(root: Path):
    def read(name):
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    manifest, profile = read("manifest.json"), read("experiment_profile.json")
    result = {"run_dir": str(root.resolve()), "checked_utc": datetime.now(timezone.utc).isoformat(),
              "status": "not_started", "passed": False, "issues": []}
    if manifest is None:
        return result
    stages = {name: value.get("status") for name, value in (manifest.get("stages") or {}).items()}
    result.update(run_id=manifest.get("run_id"), manifest_status=manifest.get("status"), stages=stages)
    if profile:
        result["model_budget"] = {key: profile.get(key) for key in
            ("model", "admitted_calls", "max_calls", "budget_consumed_cny", "max_cny",
             "stopped", "stop_reason", "execution_status")}
    wp, design, world, supply = (read(name) for name in
        ("01_whitepaper.json", "01_joint_design_receipt.json", "02_world.json", "02_source_supply_gate.json"))
    if wp and design:
        result["design"] = {"status": design.get("status"), "used_revisions": design.get("used_revisions"),
                            "bound": design.get("whitepaper_hash") == digest(wp)}
        if not result["design"]["bound"]:
            result["issues"].append("Joint design receipt does not bind the whitepaper")
    if world and supply:
        result["source_supply"] = {"status": supply.get("status"),
            "bound": supply.get("world_hash") == digest(world) and
                     (wp is not None and supply.get("whitepaper_hash") == digest(wp)),
            "per_line": {line: {"required": row.get("required"), "available": row.get("available"),
                                "pool_at_limit": row.get("pool_at_limit")}
                         for line, row in (supply.get("per_line") or {}).items()},
            "native_trend": {key: (supply.get("native_trend") or {}).get(key)
                             for key in ("required", "actual")}}
        if supply.get("status") != "source_supply_ready" or not result["source_supply"]["bound"]:
            result["issues"].append("Original source supply gate has not accepted this world")
    orders, questions = read("03_orders.json"), read("04_questions.json")
    if orders is not None:
        result["orders_by_line"] = dict(Counter(row.get("line") for row in orders))
    if questions is not None:
        result["questions_by_line"] = dict(Counter(row.get("line") for row in questions))
    scale = read("05_corpus_token_scale.json")
    if scale is not None:
        result["corpus_tokens"] = {key: scale.get(key) for key in
            ("target_met", "actual_tokens", "target_tokens", "tokenizer")}
    warning = read("04_questions_warning.json")
    if warning is not None:
        result["question_warning"] = warning
        result["issues"].append("Question author recorded a generation warning")
    if questions is None:
        ledger = read("01_joint_design_audit.json")
        if ledger and ledger.get("status") not in ("accepted", "designing", "initial_schema_requested",
                                                   "initial_schema_ready", "initial_draft_ready",
                                                   "revising_after_world_probe"):
            result["issues"].append("Joint design stopped: " + str(ledger.get("status")))
        if (profile or {}).get("stopped") or (profile or {}).get("execution_status") == "failed":
            result["issues"].append("Experiment process stopped before questions")
        result["status"] = "failed" if result["issues"] else "running"
        return result
    target = (manifest.get("config") or {}).get("delivery_target") or {}
    desired = (wp or {}).get("supply_plan", {}).get("candidate_allocation", {})
    for line, amount in desired.items():
        if result.get("orders_by_line", {}).get(line, 0) < amount:
            result["issues"].append(f"Orders below reserve: {line}")
        if result.get("questions_by_line", {}).get(line, 0) < amount:
            result["issues"].append(f"Questions below reserve: {line}")
    if not desired or len(desired) != len(target.get("requested_lines", [])):
        result["issues"].append("Frozen per-line candidate allocation is missing")
    if not result.get("design", {}).get("bound") or not result.get("source_supply", {}).get("bound"):
        result["issues"].append("Design/source proof is missing or stale")
    if not scale or scale.get("target_met") is not True:
        result["issues"].append("Corpus token target has not been certified")
    if stages.get("questions") != "succeeded":
        result["issues"].append("Questions stage has not succeeded")
    result["passed"] = not result["issues"]
    result["status"] = "passed" if result["passed"] else "failed"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.run_dir), ensure_ascii=False, indent=2))
