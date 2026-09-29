"""score 后处理：读 results.jsonl，判分 + usage/cost 回填，写 judged.jsonl 与聚合报告。

判分是独立于 execute 的显式阶段（统一入口行为链的 judge and aggregate），
可对历史 run 输出补判；不改写 results.jsonl 与 runner 生成的 summary.json。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    STANDARD_LIGHT_SCHEMA,
    benchmark_schema,
    load_benchmark_questions,
    load_benchmark_references,
    question_id,
)
from .config import ConfigurationError
from .costing import attach_cost, extract_usage, summarize_costs
from .judging import Judge

RESULT_SCHEMA_V2 = "agent-harnesses.result/v2"
SUMMARY_SCHEMA = "agent-harnesses.score-summary/v1"

# 可判分的 runner。两条赛道共用同一套判分/聚合，只是 usage 来源不同：
# native_cli 从 CLI 输出提取，memory 由 runner 直接记录答题模型用量。
SCORABLE_RUNNERS = ("native_cli", "memory")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ConfigurationError(f"缺少文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(run_dir: Path) -> list[dict]:
    questions = load_benchmark_questions(run_dir)
    if benchmark_schema(run_dir) != STANDARD_LIGHT_SCHEMA:
        return questions

    references = load_benchmark_references(run_dir)
    if len(references) != len(questions):
        raise ConfigurationError("public/references 题目数量或 qid 唯一性不一致")
    adapted = []
    for public in questions:
        qid = question_id(public)
        try:
            reference = references[qid]
        except KeyError as exc:
            raise ConfigurationError(f"qid={qid!r} 缺少判分真值") from exc
        if reference.get("quality_status") != public.get("quality_status"):
            raise ConfigurationError(f"qid={qid!r} 的 public/reference quality_status 不一致")
        question = dict(public)
        question.update(_standard_scoring_fields(question, reference))
        adapted.append(question)
    return adapted


def _standard_scoring_fields(public: dict, reference: dict) -> dict[str, Any]:
    """Map standard-light references onto the current factory judge contract.

    This is deliberately evaluator-side: the exported public question remains
    untouched.  Per the evaluation protocol, `answer` is the sole scoring truth;
    `answer_raw` and `answer_projection` are not consulted.  The generated
    contract only adapts that public reference value to the existing
    capability-aware factory judge's expected shape.
    """
    capability = public.get("capability")
    answer = reference.get("answer")
    if answer is None:
        raise ConfigurationError("references/questions.json.answer 不能为空")
    answer_kind = "value"
    contract: dict[str, Any] = {
        "version": 1,
        "answer_kind": answer_kind,
        "value_schema": {},
        "allowed_aliases": [],
        "scoring_scope": "primary_answer",
    }
    aux: dict[str, Any] = {}
    if capability in {"L6_refusal", "FORGET", "ABS"}:
        contract["answer_kind"] = "abstention"
    elif capability == "L3_order":
        contract["answer_kind"] = "order"
    elif capability == "TR":
        contract["answer_kind"] = "time"
        # The standard-light public protocol uses 第N周/期 for the same
        # one-based period.  The factory judge already accepts 周 and dates;
        # preserve those answers and supply the missing 期 spellings here.
        week = answer.get("week") if isinstance(answer, dict) else None
        if type(week) is int and week > 0:
            contract["allowed_aliases"] = [f"第{week}期", f"{week}期"]
    elif capability == "L8_next":
        # The legacy judge requires a declared state vocabulary.  The standard
        # export only carries the canonical answer, so use an explicit sentinel
        # as the second closed-set member without inventing another valid label.
        contract["answer_kind"] = "enum"
        aux["states"] = [answer, "__STANDARD_LIGHT_OTHER__"]

    # IE/MR are represented as {value: ...} by the legacy capability judge.
    # Wrapping is an evaluator adapter detail; the source of truth remains the
    # exported `answer` value and no field from `answer_raw` is copied.
    gt = {"value": answer} if capability in {"IE", "MR"} else answer

    return {
        "gt": gt,
        "aux": aux,
        "question_contract": contract,
        "reference_answer": answer,
        "truth_source": "references/questions.json.answer",
        "_benchmark_schema": STANDARD_LIGHT_SCHEMA,
    }


def read_results(out_dir: Path) -> list[dict]:
    """读 results.jsonl；resume 会对同题追加多行，按 question_index 取末行。"""
    path = out_dir / "results.jsonl"
    if not path.is_file():
        raise ConfigurationError(f"缺少 results.jsonl: {path}")
    latest: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        latest[int(rec["question_index"])] = rec
    return [latest[idx] for idx in sorted(latest)]


def _score_record(
    record: dict,
    question: dict,
    judge: Judge,
    *,
    use_llm: bool,
) -> dict:
    out = dict(record)
    out["schema"] = RESULT_SCHEMA_V2
    out["line"] = question.get("line")
    out["capability"] = question.get("capability")
    out["entity"] = question.get("entity")
    out["field"] = question.get("field")
    out["quality_status"] = question.get("quality_status")
    out["truth_source"] = question.get("truth_source")
    out["reference_answer"] = question.get("reference_answer")
    out["gold"] = judge.gold_display(question)
    judge_info: dict[str, Any] = {
        "version": judge.version,
        "mode": judge.judge_mode(question),
        "use_llm": use_llm,
        "error": None,
    }
    judgeable = judge.is_judgeable(question)
    out["judgeable"] = judgeable
    pred = str(record.get("answer") or "")
    infra_failed = bool(record.get("error_type"))
    if infra_failed or not judgeable:
        # 基础设施失败不计错也不删分母；不可判分题保持 correct=None 显式暴露。
        out["correct"] = None
    else:
        try:
            out["correct"] = judge.judge_answer(question, pred, use_llm=use_llm)
            if question.get("capability") == "L2_multihop":
                out["partial"] = judge.judge_l2_partial(question, pred, use_llm=use_llm)
            if question.get("capability") == "L6_refusal":
                lure = ((question.get("aux") or {}).get("lure") or {}).get("value")
                out["refusal_bucket"] = judge.classify_refusal(pred, lure)
        except Exception as exc:
            out["correct"] = None
            judge_info["error"] = f"{type(exc).__name__}: {exc}"
    out["judge"] = judge_info
    return out


def _attach_usage(record: dict, out_dir: Path, adapter: str) -> None:
    """把 usage 归一到统一形状。

    native_cli 从 raw stdout/stderr 提取；memory run 没有 CLI 输出，usage 由 runner
    直接记录在结果行里（答题模型的 token 计数），此时保留原值、不做覆盖。
    """
    stdout = stderr = ""
    raw_stdout = record.get("raw_stdout")
    raw_stderr = record.get("raw_stderr")
    if raw_stdout and (out_dir / raw_stdout).is_file():
        stdout = (out_dir / raw_stdout).read_text(encoding="utf-8", errors="replace")
    if raw_stderr and (out_dir / raw_stderr).is_file():
        stderr = (out_dir / raw_stderr).read_text(encoding="utf-8", errors="replace")
    if not stdout and not stderr:
        # memory runner 已经写好 usage；没有 raw 证据时保留它，同时也保证字段存在。
        record.setdefault("usage", None)
        return
    record["usage"] = extract_usage(adapter, stdout, stderr)


def aggregate(records: list[dict]) -> dict[str, Any]:
    def bucket(rows: list[dict]) -> dict[str, Any]:
        judged = [r for r in rows if r.get("correct") is not None]
        n_correct = sum(1 for r in judged if r["correct"])
        n_infra = sum(1 for r in rows if r.get("error_type"))
        partials = [r["partial"] for r in rows if r.get("partial") is not None]
        out = {
            "n_total": len(rows),
            "n_judgeable": sum(1 for r in rows if r.get("judgeable")),
            "n_judged": len(judged),
            "n_infra_failed": n_infra,
            "n_judge_error": sum(1 for r in rows if (r.get("judge") or {}).get("error")),
            "n_correct": n_correct,
            "accuracy": round(n_correct / len(rows), 3) if rows else None,
            "accuracy_over_judged": round(n_correct / len(judged), 3) if judged else None,
        }
        if partials:
            out["partial_avg"] = round(sum(partials) / len(partials), 3)
        return out

    return {
        "overall": bucket(records),
        "by_line": {
            line: bucket([r for r in records if r.get("line") == line])
            for line in sorted({r.get("line") for r in records if r.get("line")})
        },
        "by_capability": {
            cap: bucket([r for r in records if r.get("capability") == cap])
            for cap in sorted({r.get("capability") for r in records if r.get("capability")})
        },
        "by_quality_status": {
            status: bucket([r for r in records if r.get("quality_status") == status])
            for status in sorted(
                {r.get("quality_status") for r in records if r.get("quality_status")}
            )
        },
    }


def _summary_md(summary: dict[str, Any]) -> str:
    overall = summary["aggregate"]["overall"]
    acc = overall["accuracy"]
    acc_text = "n/a" if acc is None else f"{acc:.1%} ({overall['n_correct']}/{overall['n_total']})"
    lines = [
        f"# Score {summary['experiment_id']} / {summary['system_id']}",
        "",
        f"- model: `{summary['model']}`",
        f"- judge: factory_commit `{(summary['judge_version'] or {}).get('factory_commit')}` "
        f"judge_sha256 `{(summary['judge_version'] or {}).get('judge_sha256')}` "
        f"(use_llm={summary['use_llm']})",
        f"- scoring_policy: `{summary['scoring_policy']}`"
        + ("（advisory：该 run 的 protocol 声明不计分，本结果仅供链路验证）" if summary["advisory"] else ""),
        f"- 准确率（分母含全部题）: **{acc_text}**；仅可判分口径: "
        + ("n/a" if overall["accuracy_over_judged"] is None else f"{overall['accuracy_over_judged']:.1%}"),
        f"- infra 失败: {overall['n_infra_failed']}（不计错、不删分母）; judge 异常: {overall['n_judge_error']}",
        f"- usage 缺失: {summary['costs']['n_usage_missing']}",
        "",
    ]
    for title, key in (
        ("按能力线", "by_line"),
        ("按能力", "by_capability"),
        ("按质量标签（仅诊断，不过滤）", "by_quality_status"),
    ):
        lines += [f"## {title}", "", "| 组 | n | judged | correct | acc | infra |", "|---|---|---|---|---|---|"]
        for name, b in summary["aggregate"][key].items():
            acc_cell = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.1%}"
            lines.append(
                f"| {name} | {b['n_total']} | {b['n_judged']} | {b['n_correct']} | {acc_cell} | {b['n_infra_failed']} |"
            )
        lines.append("")
    return "\n".join(lines)


def score_run(
    out_dir: Path,
    judge: Judge,
    *,
    use_llm: bool,
    force: bool = False,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    judged_path = out_dir / "judged.jsonl"
    if judged_path.exists() and not force:
        raise ConfigurationError(f"{judged_path} 已存在；重判请加 --force")
    plan = _read_json(out_dir / "run_plan.json")
    runner = plan.get("runner")
    if runner not in SCORABLE_RUNNERS:
        raise ConfigurationError(
            f"score 仅支持 {' / '.join(SCORABLE_RUNNERS)} run，当前 runner={runner!r}"
        )
    runtime = plan.get("runtime") or {}
    # memory run 没有 CLI adapter：模型身份取 experiment 固定的 answering_model。
    answering = plan.get("answering_model") or {}
    adapter = str(
        runtime.get("adapter")
        or plan["system"]["implementation"].get("adapter")
        or ("memory" if runner == "memory" else "")
    )
    model = str(runtime.get("model") or answering.get("model_id") or "")
    run_dir = Path(plan["benchmark"]["path"])
    questions = load_questions(run_dir)
    records = read_results(out_dir)

    scored: list[dict] = []
    for record in records:
        index = int(record["question_index"])
        if index >= len(questions):
            raise ConfigurationError(f"question_index={index} 超出题库范围 ({len(questions)})")
        question = questions[index]
        expected_id = question_id(question)
        if record.get("question_id") != expected_id:
            raise ConfigurationError(
                f"question_index={index}: question_id 不一致（结果 {record.get('question_id')!r} "
                f"vs 题库 {expected_id!r}）；results.jsonl 可能与该 benchmark 不匹配"
            )
        out = _score_record(record, question, judge, use_llm=use_llm)
        _attach_usage(out, out_dir, adapter)
        attach_cost(out, model, pricing)
        scored.append(out)

    with judged_path.open("w", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scoring_policy = str((plan.get("protocol") or {}).get("scoring") or "")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": plan.get("experiment_id"),
        "system_id": plan.get("system_id"),
        "model": model or None,
        "adapter": adapter,
        "scoring_policy": scoring_policy,
        "advisory": scoring_policy == "disabled_for_chain_smoke",
        "use_llm": use_llm,
        "judge_version": judge.version,
        "n_questions_full": len(questions),
        "aggregate": aggregate(scored),
        "costs": summarize_costs(scored, model, n_full=len(questions), pricing=pricing),
        "judged_path": str(judged_path),
    }
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "score_summary.md").write_text(_summary_md(summary) + "\n", encoding="utf-8")
    return summary
