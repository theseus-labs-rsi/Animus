#!/usr/bin/env python3
"""机械验收一个 00–06 Benchmark Run 是否具备完整、可追溯的出厂条件。

输入可以是 run_id 或 Run 目录；输出逐项 PASS/FAIL/WARN，并以退出码 0/1
表示是否通过硬门。该工具只读产物，不调用模型，也不修改 Run。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.world_blueprint import (WorldBlueprintError,
                                      normalize_world_blueprint,
                                      structure_signature)


REQUIRED_ARTIFACTS = [
    "00_input.json",
    "01_whitepaper.json",
    "02_world.json",
    "03_orders.json",
    "03_well_posed_report.json",
    "04_questions.json",
    "05_corpus.json",
    "06_grounded_questions.json",
    "06_grounding_report.json",
    "manifest.json",
]

_CREDENTIAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,})(?![A-Za-z0-9_-])")


class Audit:
    """收集硬门和警告，并统一打印审计结论。"""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passes if condition else self.failures).append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    def print(self) -> None:
        for message in self.passes:
            print(f"PASS  {message}")
        for message in self.warnings:
            print(f"WARN  {message}")
        for message in self.failures:
            print(f"FAIL  {message}")
        print(
            f"\nSUMMARY pass={len(self.passes)} warn={len(self.warnings)} "
            f"fail={len(self.failures)} status={'PASS' if not self.failures else 'FAIL'}"
        )


def _read_json(path: Path):
    """读取 UTF-8 JSON；异常由调用方转成清晰的硬门失败。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value, nested_key: str | None = None) -> list:
    """兼容直接数组以及带 corpus/sessions 等包装的历史产物。"""
    if isinstance(value, list):
        return value
    if nested_key and isinstance(value, dict) and isinstance(value.get(nested_key), list):
        return value[nested_key]
    return []


def _resolve_run(value: str) -> Path:
    """将 run_id 或目录解析为绝对 Run 路径。"""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return (Path(__file__).resolve().parent.parent / "output" / "runs" / value).resolve()


def audit_run(run_dir: Path) -> Audit:
    """执行完整静态验收，返回包含全部证据的 Audit。"""
    audit = Audit()
    audit.check(run_dir.is_dir(), f"Run 目录存在：{run_dir}")
    if not run_dir.is_dir():
        return audit

    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    audit.check(not missing, "00–06 与 manifest 十项产物齐全" if not missing else f"缺少产物：{missing}")
    if missing:
        return audit

    data: dict[str, object] = {}
    for name in REQUIRED_ARTIFACTS:
        try:
            data[name] = _read_json(run_dir / name)
        except (OSError, json.JSONDecodeError) as exc:
            audit.failures.append(f"{name} 不是有效 JSON：{exc}")
    if audit.failures:
        return audit

    manifest = data["manifest.json"]
    whitepaper = data["01_whitepaper.json"]
    world = data["02_world.json"]
    audit.check(isinstance(manifest, dict) and manifest.get("status") == "done", "manifest.status=done")
    audit.check(
        isinstance(manifest, dict) and manifest.get("run_id") == run_dir.name,
        "manifest.run_id 与 Run 目录名一致",
    )

    blueprint = None
    try:
        blueprint = normalize_world_blueprint(whitepaper)
        audit.check(not blueprint.get("legacy_adapter"), "使用显式 world_blueprint，而非 legacy 适配")
    except (WorldBlueprintError, TypeError, ValueError) as exc:
        audit.failures.append(f"world_blueprint 非法：{exc}")

    entities = world.get("entities", {}) if isinstance(world, dict) else {}
    entity_types = world.get("entity_types", {}) if isinstance(world, dict) else {}
    relations = _as_list(world.get("relations", []) if isinstance(world, dict) else [])
    events = _as_list(world.get("events", []) if isinstance(world, dict) else [])
    cascades = _as_list(world.get("cascades", []) if isinstance(world, dict) else [])
    audit.check(bool(entities), "真值世界含实体")
    audit.check(bool(entity_types) and set(entity_types) == set(entities), "每个实体都有且仅有一个类型")
    audit.check(bool(relations), "世界含关系实例")
    audit.check(bool(events), "世界含领域事件实例")

    if blueprint:
        world_blueprint = world.get("world_blueprint") if isinstance(world, dict) else None
        audit.check(
            isinstance(world_blueprint, dict)
            and structure_signature(world_blueprint) == structure_signature(blueprint),
            "白皮书与世界使用同一份冻结结构签名",
        )
        declared_types = {item.get("id") for item in blueprint.get("entity_types", [])}
        audit.check(set(entity_types.values()) <= declared_types, "实体类型均来自冻结蓝图")

        relation_counts = Counter(item.get("type") for item in relations)
        for declaration in blueprint.get("relation_types", []):
            relation_id = declaration.get("id")
            floor = int(declaration.get("min_count", 1) or 1)
            audit.check(relation_counts[relation_id] >= floor,
                        f"关系 {relation_id} 实例数 {relation_counts[relation_id]}≥{floor}")

        event_counts = Counter(item.get("type") for item in events)
        for declaration in blueprint.get("event_types", []):
            event_id = declaration.get("id")
            floor = int(declaration.get("min_count", 1) or 1)
            audit.check(event_counts[event_id] >= floor,
                        f"事件 {event_id} 实例数 {event_counts[event_id]}≥{floor}")

        cascade_counts = Counter(item.get("rule_id") for item in cascades)
        for rule in blueprint.get("causal_rules", []):
            rule_id = rule.get("id")
            audit.check(cascade_counts[rule_id] >= 1, f"因果规则 {rule_id} 有真实事件对见证")

        # 结构签名冻结语义骨架；闭环允许在 01 中发布有效规模旋钮。02 自带的
        # blueprint 才是生成该世界的直接合同，时间片/基数必须按它验。
        effective_blueprint = world_blueprint if isinstance(world_blueprint, dict) else blueprint
        expected_sessions = int((effective_blueprint.get("temporal_model") or {}).get("n_sessions", 0) or 0)
        audit.check(int(world.get("n_sessions", 0) or 0) == expected_sessions,
                    f"世界时间片数量与蓝图一致（{expected_sessions}）")

    orders = _as_list(data["03_orders.json"])
    questions = _as_list(data["04_questions.json"])
    grounded = _as_list(data["06_grounded_questions.json"])
    report = data["06_grounding_report.json"]
    audit.check(bool(orders), "订单非空")
    audit.check(bool(questions) and all(item.get("question") and "gt" in item for item in questions),
                "题库非空且每题含题面与机械 gold")
    audit.check(bool(grounded), "接地后题库非空")
    # 06 只能给 04 增加证据池，不能在接地阶段悄悄改题面、gold 或能力标签。
    def question_signature(question: dict) -> str:
        return json.dumps(
            {key: value for key, value in question.items() if key != "evidence_doc_ids"},
            ensure_ascii=False,
            sort_keys=True,
        )

    question_counts = Counter(question_signature(item) for item in questions)
    grounded_counts = Counter(question_signature(item) for item in grounded)
    lineage_ok = all(count <= question_counts[signature]
                     for signature, count in grounded_counts.items())
    audit.check(lineage_ok, "06 是 04 的未篡改子集")
    report_grounded = ((report.get("overall") or {}).get("grounded")
                       if isinstance(report, dict) else None)
    audit.check(report_grounded == len(grounded), "接地报告数量与 06 题库一致")

    algo = manifest.get("algo", {}) if isinstance(manifest, dict) else {}
    target = algo.get("targetspec", {}) if isinstance(algo, dict) else {}
    min_questions = int(target.get("min_questions", 0) or 0)
    audit.check(algo.get("met_status") == "MET", f"闭环状态为 MET（实际 {algo.get('met_status')}）")
    audit.check(len(grounded) >= min_questions, f"接地题数 {len(grounded)}≥总下限 {min_questions}")
    by_line = report.get("by_line", {}) if isinstance(report, dict) else {}
    for line_id, floor in (target.get("per_line_min", {}) or {}).items():
        count = int((by_line.get(line_id) or {}).get("grounded", 0) or 0)
        audit.check(count >= int(floor), f"{line_id} 接地题数 {count}≥下限 {floor}")

    corpus_wrapper = data["05_corpus.json"]
    corpus = corpus_wrapper.get("corpus", corpus_wrapper) if isinstance(corpus_wrapper, dict) else {}
    sessions = _as_list(corpus, "sessions")
    docs = [doc for session in sessions for doc in _as_list(session.get("docs", []))]
    doc_ids = [doc.get("doc_id") for doc in docs]
    doc_sessions = {
        doc.get("doc_id"): session.get("session_id")
        for session in sessions
        for doc in _as_list(session.get("docs", []))
    }
    docs_by_id = {
        doc.get("doc_id"): doc
        for session in sessions
        for doc in _as_list(session.get("docs", []))
    }
    audit.check(bool(sessions) and bool(docs), "语料含时间片与文档")
    audit.check(None not in doc_ids and len(doc_ids) == len(set(doc_ids)), "所有 doc_id 存在且唯一")
    missing_evidence_ids = sorted({
        doc_id
        for question in grounded
        for doc_id in (question.get("evidence_doc_ids") or [])
        if doc_id not in doc_sessions
    })
    out_of_scope_evidence = [
        (question.get("qid"), doc_id, doc_sessions[doc_id])
        for question in grounded
        for doc_id in (question.get("evidence_doc_ids") or [])
        if doc_id in doc_sessions and doc_sessions[doc_id] not in (question.get("evidence_sessions") or [])
    ]
    empty_evidence_pools = [
        question.get("qid") or f"{question.get('line')}:{index}"
        for index, question in enumerate(grounded)
        if not question.get("evidence_doc_ids")
    ]
    duplicate_evidence_pools = [
        question.get("qid") or f"{question.get('line')}:{index}"
        for index, question in enumerate(grounded)
        if len(question.get("evidence_doc_ids") or [])
        != len(set(question.get("evidence_doc_ids") or []))
    ]
    filler_evidence = [
        (question.get("qid") or f"{question.get('line')}:{index}", doc_id)
        for index, question in enumerate(grounded)
        for doc_id in (question.get("evidence_doc_ids") or [])
        if doc_id in docs_by_id and docs_by_id[doc_id].get("is_filler") is True
    ]
    audit.check(
        not empty_evidence_pools,
        "06 每道题都发布非空 evidence_doc_ids"
        if not empty_evidence_pools
        else f"06 存在空证据池：{empty_evidence_pools[:12]}",
    )
    audit.check(
        not missing_evidence_ids,
        "题目 evidence_doc_ids 全部存在"
        if not missing_evidence_ids
        else f"题目引用不存在的 evidence_doc_ids：{missing_evidence_ids[:12]}",
    )
    audit.check(
        not out_of_scope_evidence,
        "evidence_doc_ids 全部落在声明 evidence_sessions 内"
        if not out_of_scope_evidence
        else f"证据文档落在声明时间窗外：{out_of_scope_evidence[:12]}",
    )
    audit.check(
        not duplicate_evidence_pools,
        "每题 evidence_doc_ids 均无重复"
        if not duplicate_evidence_pools
        else f"题目证据池含重复 ID：{duplicate_evidence_pools[:12]}",
    )
    audit.check(
        not filler_evidence,
        "evidence_doc_ids 全部指向非 filler 文档"
        if not filler_evidence
        else f"题目把 filler 当证据：{filler_evidence[:12]}",
    )
    signal_docs = [doc for doc in docs if "_sig_" in str(doc.get("doc_id", ""))]
    filler_docs = [doc for doc in docs if doc.get("is_filler") is True]
    corpus_chars = sum(len(str(doc.get("content", ""))) for doc in docs)
    target_chars = int((manifest.get("config") or {}).get("target_tokens") or 0)
    audit.check(
        corpus_chars >= target_chars,
        f"语料规模 {corpus_chars} 字符≥目标 {target_chars}"
        if corpus_chars >= target_chars
        else f"语料规模不足：{corpus_chars}/{target_chars}",
    )
    audit.check(bool(signal_docs) and all(doc.get("fact_refs") for doc in signal_docs),
                "信号文档非空且全部带 fact_refs")
    valid_fact_refs = {
        f"{entity}.{field}"
        for entity, fields in entities.items()
        for field in fields
    }
    valid_fact_refs.update(item.get("id") for item in events)
    valid_fact_refs.update(item.get("id") for item in relations)
    valid_fact_refs.update(item.get("rule_id") for item in cascades)
    dangling_fact_refs = sorted({
        ref
        for doc in signal_docs
        for ref in (doc.get("fact_refs") or [])
        if ref not in valid_fact_refs
    })
    audit.check(
        not dangling_fact_refs,
        "全部 fact_refs 可解析到世界事实"
        if not dangling_fact_refs
        else f"fact_refs 存在悬空引用：{dangling_fact_refs[:12]}",
    )
    audit.check(all(not doc.get("fact_refs") for doc in filler_docs), "filler 全部不携带 fact_refs")
    credential_like_fillers = [
        doc.get("doc_id") for doc in filler_docs
        if _CREDENTIAL_TOKEN_RE.search(str(doc.get("content") or ""))
    ]
    audit.check(
        not credential_like_fillers,
        "filler 不含凭据形态 token"
        if not credential_like_fillers
        else f"filler 含凭据形态 token：{credential_like_fillers[:12]}",
    )

    tracked_terms = set(entities)
    profile = whitepaper.get("domain_profile", {}) if isinstance(whitepaper, dict) else {}
    person_fields = {
        item.get("name") for item in profile.get("field_schema", [])
        if isinstance(item, dict) and item.get("kind") == "person"
    }
    tracked_terms.update(
        str(point.get("value"))
        for fields in entities.values()
        for field, timeline in fields.items()
        if field in person_fields and isinstance(timeline, list)
        for point in timeline
        if isinstance(point, dict) and point.get("value") not in (None, "")
    )
    filler_text = "\n".join(str(doc.get("content", "")) for doc in filler_docs)
    # 与生产 blocklist 对齐：单字状态占位（如“无”）不是可定位的人物专名。
    leaks = sorted(term for term in tracked_terms if term and len(str(term)) >= 2 and term in filler_text)
    audit.check(not leaks, "filler 未触碰被追踪实体/人物专名" if not leaks else f"filler 泄漏追踪专名：{leaks[:10]}")

    if manifest.get("scenario") == "game":
        office_markers = ["OA系统", "办公区", "行政部", "人力资源部", "员工培训", "公司员工", "食堂管理部"]
        crossed = [marker for marker in office_markers if marker in filler_text]
        audit.check(not crossed, "游戏 filler 未串入现代办公模板" if not crossed else f"游戏 filler 跨域：{crossed}")

    filler_ratio = len(filler_docs) / len(docs) if docs else 0.0
    audit.warn(filler_ratio <= 0.8, f"filler 文档占比偏高：{filler_ratio:.1%}")
    audit.warn(not report.get("drops"), f"接地闸丢弃 {report.get('n_dropped', 0)} 题，需人工抽检原因")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="机械验收一个 Memory Forge 00–06 Run")
    parser.add_argument("run", help="run_id 或 Run 目录")
    args = parser.parse_args()
    run_dir = _resolve_run(args.run)
    print(f"AUDIT {run_dir}\n")
    result = audit_run(run_dir)
    result.print()
    raise SystemExit(0 if not result.failures else 1)


if __name__ == "__main__":
    main()
