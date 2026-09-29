"""按公开单价把 usage 折成人民币，并从各 native CLI 的原始输出里抠 token。

迁移自 eval/costing.py；单价表可被 experiment 的 pricing 字段覆盖。
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

# 阿里云百炼 DeepSeek-V4-Flash 公开列表价（元 / 百万 token，文章口径 2026）。
# 思考/思维链按输出计。实际账单以控制台为准。
PRICE_CNY_PER_M = {
    "deepseek-v4-flash": {
        "input": 1.0,
        "input_cached": 0.02,
        "output": 2.0,
    }
}


def price_for(model: str, pricing: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, float] | None:
    name = (model or "").lower()
    table = {**PRICE_CNY_PER_M, **{k.lower(): v for k, v in (pricing or {}).items()}}
    for key, val in table.items():
        if key in name:
            return dict(val)
    return None


def cost_cny(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> float | None:
    p = price_for(model, pricing)
    if p is None:
        return None
    cached = max(int(cached_tokens or 0), 0)
    prompt = max(int(prompt_tokens or 0), 0)
    uncached = max(prompt - cached, 0)
    out = max(int(completion_tokens or 0), 0)
    return (uncached * p["input"] + cached * p["input_cached"] + out * p["output"]) / 1_000_000


def _field(obj: Any, *names: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def normalize_usage(raw: dict[str, Any] | None, *, source: str) -> dict[str, Any] | None:
    if not raw:
        return None
    prompt = _field(raw, "prompt_tokens", "input_tokens")
    completion = _field(raw, "completion_tokens", "output_tokens")
    details = _field(raw, "prompt_tokens_details", "input_tokens_details", default={})
    cached = int(_field(raw, "cached_tokens", "cache_read_input_tokens") or _field(details, "cached_tokens") or 0)
    if int(prompt) and cached > int(prompt):
        cached = int(prompt)
    out_details = _field(raw, "output_tokens_details", default={})
    reasoning = int(_field(out_details, "reasoning_tokens") or 0)
    # 思考 token 已计入 output_tokens 时不再加；只作备注
    total = _field(raw, "total_tokens", default=int(prompt) + int(completion))
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "cached_tokens": int(cached),
        "reasoning_tokens": reasoning,
        "total_tokens": int(total),
        "n_usage_events": 1,
        "source": source,
    }


def usage_from_claude_json(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    if not isinstance(data, dict):
        return None
    raw = data.get("usage")
    if not isinstance(raw, dict) and isinstance(data.get("result"), dict):
        raw = data["result"].get("usage")
    return normalize_usage(raw if isinstance(raw, dict) else None, source="claude_json")


def usage_from_codex_text(stdout: str, stderr: str = "") -> dict[str, Any] | None:
    blob = f"{stdout or ''}\n{stderr or ''}"
    nums = [int(x) for x in re.findall(r"tokens used[:\s]+(\d+)", blob, flags=re.I)]
    if not nums:
        return None
    total = nums[-1]
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": total,
        "n_usage_events": 1,
        "source": "codex_tokens_used",
        "unsplit_total": True,
    }


def extract_usage(adapter: str, stdout: str, stderr: str = "") -> dict[str, Any] | None:
    if adapter == "claude":
        return usage_from_claude_json(stdout)
    if adapter == "codex":
        return usage_from_codex_text(stdout, stderr)
    return None


def attach_cost(
    rec: dict[str, Any],
    model: str,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    usage = rec.get("usage")
    if not usage:
        rec["cost_cny"] = None
        rec["cost_cny_low"] = None
        rec["cost_cny_high"] = None
        return rec
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    if usage.get("unsplit_total") and total and not (prompt or completion):
        # Codex 只给总量：按「全算输入」低估输出；同时给一档「全算输出」上界。
        rec["cost_cny"] = None
        rec["cost_cny_low"] = cost_cny(model, prompt_tokens=total, completion_tokens=0, pricing=pricing)
        rec["cost_cny_high"] = cost_cny(model, prompt_tokens=0, completion_tokens=total, pricing=pricing)
        rec["cost_note"] = "codex 只回报 total tokens，给出输入/输出两档夹逼"
        return rec
    value = cost_cny(
        model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        pricing=pricing,
    )
    rec["cost_cny"] = round(value, 6) if value is not None else None
    rec["cost_cny_low"] = rec["cost_cny"]
    rec["cost_cny_high"] = rec["cost_cny"]
    if value is None:
        rec["cost_note"] = "模型无单价表；请在 experiment.pricing 中补充"
    return rec


def summarize_costs(
    records: list[dict],
    model: str,
    n_full: int,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    recs = [r for r in records if r.get("judgeable")]
    prompt = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in recs)
    completion = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in recs)
    cached = sum(int((r.get("usage") or {}).get("cached_tokens") or 0) for r in recs)
    total = sum(int((r.get("usage") or {}).get("total_tokens") or 0) for r in recs)
    known = [r for r in recs if r.get("cost_cny") is not None]
    lows = [float(r.get("cost_cny_low") or 0) for r in recs if r.get("cost_cny_low") is not None]
    highs = [float(r.get("cost_cny_high") or 0) for r in recs if r.get("cost_cny_high") is not None]
    n_priced = len(lows) or 1
    spent_low = round(sum(lows), 6) if lows else 0.0
    spent_high = round(sum(highs), 6) if highs else 0.0
    spent = round(sum(float(r["cost_cny"]) for r in known), 6) if known else None
    scale = n_full / (len(lows) or len(recs) or 1)
    return {
        "model": model,
        "price_cny_per_m": price_for(model, pricing),
        "n_scored": len(recs),
        "n_with_cost": len(known),
        "n_usage_missing": sum(1 for r in recs if not r.get("usage")),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "total_tokens": total,
        "spent_cny": spent,
        "spent_cny_low": spent_low,
        "spent_cny_high": spent_high,
        "per_question_cny_low": round(spent_low / n_priced, 6),
        "per_question_cny_high": round(spent_high / n_priced, 6),
        "estimate_full_cny_low": round(spent_low * scale, 4),
        "estimate_full_cny_high": round(spent_high * scale, 4),
        "estimate_n": n_full,
        "note": "全量预估 = 已跑均价 × 题数；L3 等长工具环会高于均值。含思考输出时按输出单价。未含判分器调用。",
    }
