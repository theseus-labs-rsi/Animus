"""根据已有逐题判分剔除全员答对题；纯离线，不发起模型调用。

python -m eval.question_filter --bench 06_grounded_questions.json \
    --results results.json --out-dir filtered --keep-easy-ratio 0.2
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
from eval.grading import JUDGE_VERSION, is_scored
from eval.provenance import (REFERENCE_FIELDS, load_public_protocol, load_visible_corpus,
                             make_evaluation_context, preserve_result_contexts, provenance_issue)


DISPOSITIONS = ("removed_easy", "kept_easy_sample", "kept_not_all_correct", "kept_incomplete")
IDENTITY_FIELDS = ("question", *REFERENCE_FIELDS)


def question_key(item: dict) -> str:
    """按题面和评分合同匹配，避免题号重排或旧答案造成错配。"""
    payload = {field: item.get(field) for field in IDENTITY_FIELDS}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_options(systems: list[str], keep_easy_ratio: float, seed: int) -> None:
    """提前检查筛选参数，供离线命令和在线评测共用。"""
    if (len(systems) < 2 or any(not isinstance(s, str) or not s.strip() for s in systems)
            or len(systems) != len(set(systems))):
        raise ValueError("至少需要两个名称不同的系统，才能判断全员答对")
    if not math.isfinite(keep_easy_ratio) or not 0 <= keep_easy_ratio <= 1:
        raise ValueError("keep_easy_ratio 必须在 0 到 1 之间")
    if type(seed) is not int:
        raise ValueError("seed 必须为整数")


def _rows(value, label: str) -> list[dict]:
    """校验逐题列表；损坏输入直接报错，避免静默漏读。"""
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} 必须是题目对象列表")
    for row in value:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{label} 中存在空题面或缺失 question 的记录")
        # Semantic tasks may have no legacy capability label or code-computable
        # gold. Their versioned review/grade provenance is checked separately.
    return value


def _invalid_reason(record: dict) -> str | None:
    """只有成功、可判分且带布尔判分的回答才参与难度判断。"""
    if (record.get("error") or record.get("judge_error")
            or record.get("execution_status", "ok") not in ("ok", "success")):
        return "evaluation_error"
    if record.get("judgeable", True) is not True:
        return "not_judgeable"
    grade = record.get("judgement")
    if not isinstance(grade, dict):
        return "missing_judgement_provenance"
    if grade.get("verdict") in {"error", "unjudgeable"}:
        return "judge_" + grade["verdict"]
    if not is_scored(record):
        return "invalid_or_stale_judgement"
    if type(record.get("correct")) is not bool:
        return "invalid_judgement"
    pred = record.get("pred")
    if not isinstance(pred, str) or not pred.strip():
        return "missing_answer"
    if pred.lstrip().startswith("[") and "ERROR" in pred:
        return "evaluation_error"
    return None


def filter_questions(questions: list[dict], results: dict[str, list[dict]], *,
                      keep_easy_ratio: float = 0.0, seed: int = 0,
                      preserve_capabilities: tuple[str, ...] | list[str] = (),
                      expected_context: dict | None = None,
                      stratify_by: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    """筛选全员答对题并返回审计报告；输入不变，缺测或异常题一律保留。"""
    systems = sorted(results)
    validate_options(systems, keep_easy_ratio, seed)
    preserved = set(preserve_capabilities)
    _rows(questions, "bench")
    keys = [question_key(q) for q in questions]
    key_counts = Counter(keys)
    indexes = {}
    unmatched = {}
    for system in systems:
        index = defaultdict(list)
        for record in _rows(results[system], system):
            index[question_key(record)].append(record)
        indexes[system] = index
        unmatched[system] = sum(len(rows) for key, rows in index.items() if key not in key_counts)

    items = []
    easy = []
    for position, (q, key) in enumerate(zip(questions, keys)):
        issues = {}
        if key_counts[key] > 1:
            issues["bench"] = "duplicate_question"
        if q.get("judgeable", True) is not True:
            issues["bench"] = "not_judgeable"
        if q.get("capability") in preserved:
            issues["review"] = "capability_pending_review"
        grades = {}
        for system in systems:
            records = indexes[system].get(key, [])
            if len(records) != 1:
                issues[system] = "missing_result" if not records else "duplicate_result"
                grades[system] = None
                continue
            record = records[0]
            reason = _invalid_reason(record) or provenance_issue(record, q, expected_context)
            if q.get("qid") is not None and record.get("qid") is not None and q["qid"] != record["qid"]:
                reason = "qid_mismatch"
            grades[system] = None if reason else record["correct"]
            if reason:
                issues[system] = reason
        if issues:
            disposition = "kept_incomplete"
        elif all(grades.values()):
            disposition = "removed_easy"
            easy.append(key)
        else:
            disposition = "kept_not_all_correct"
        items.append({"index": position, "key": key, "qid": q.get("qid"),
                      "line": q.get("line"), "capability": q.get("capability"),
                      "question": q["question"], "correct": grades,
                      "issues": issues, "disposition": disposition})

    # The old evaluation CLI keeps its global floor rule. Production selects
    # within each world/line, rounding upward so a small easy-only line survives.
    if any(field not in {"world", "line", "capability"} for field in stratify_by):
        raise ValueError("Unsupported easy-question sampling stratum")
    groups = defaultdict(list)
    easy_keys = set(easy)
    for q, key in zip(questions, keys):
        if key in easy_keys:
            groups[tuple(str(q.get(field, "")) for field in stratify_by)].append(key)
    sampled = set()
    for group in groups.values():
        amount = len(group) * Fraction(str(keep_easy_ratio))
        keep_n = math.ceil(amount) if stratify_by else int(amount)
        ranked = sorted(group, key=lambda key: (hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
        sampled.update(ranked[:keep_n])
    kept = []
    for q, item in zip(questions, items):
        if item["key"] in sampled:
            item["disposition"] = "kept_easy_sample"
        if item["disposition"] != "removed_easy":
            kept.append(q)

    def counts(rows):
        counter = Counter(item["disposition"] for item in rows)
        return {"input": len(rows), "kept": len(rows) - counter["removed_easy"],
                **{name: counter[name] for name in DISPOSITIONS}}

    by_line = defaultdict(list)
    by_capability = defaultdict(list)
    for item in items:
        by_line[str(item["line"])].append(item)
        by_capability[str(item["capability"])].append(item)
    report = {
        "schema_version": 3, "rule": "all_selected_systems_correct", "judge_version": JUDGE_VERSION,
        "systems": systems, "keep_easy_ratio": keep_easy_ratio, "seed": seed,
        "preserve_capabilities": sorted(preserved),
        "sampling": ("ceil(n_easy_in_stratum * keep_easy_ratio); seeded SHA-256 order" if stratify_by
                     else "floor(n_easy * keep_easy_ratio); seeded SHA-256 order"),
        "stratify_by": list(stratify_by),
        "identity_fields": list(IDENTITY_FIELDS),
        "counts": {**counts(items), "all_correct": len(easy)},
        "by_line": {key: counts(rows) for key, rows in sorted(by_line.items())},
        "by_capability": {key: counts(rows) for key, rows in sorted(by_capability.items())},
        "unmatched_result_rows": unmatched,
        "expected_evaluation_context": deepcopy(expected_context),
        "judgement_basis": "context_and_reference_bound_primary_judgements_without_rejudging",
        "legacy_results": "readable_but_unidentified_records_cannot_establish_all_correct",
        "items": items,
        "result_scope": "research_only" if any(
            "research_only" in getattr(rows, "result_scopes", []) or any(
                row.get("evaluation_scope") == "research_only" or "research_only" in (row.get("_result_scopes") or [])
                for row in rows) for rows in results.values()) else "validated_input_identity",
    }
    return kept, report


def _read_json(path: Path):
    """兼容 UTF-8 BOM 的 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_questions(path: Path) -> list[dict]:
    """兼容题目列表及 {questions: [...]} 包装。"""
    data = _read_json(path)
    return _rows(data.get("questions") if isinstance(data, dict) else data, str(path))


def load_results(*, aggregate: Path | None = None, system_files: dict[str, Path] | None = None,
                 systems: list[str] | None = None) -> dict[str, list[dict]]:
    """读本项目汇总 JSON 或每个系统独立的 JSON/JSONL，禁止静默省略所选系统。"""
    if (aggregate is None) == (not system_files):
        raise ValueError("必须且只能提供汇总 results 或逐系统结果文件")
    if aggregate is not None:
        data = _read_json(aggregate)
        raw = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("汇总文件缺少 results 对象")
        selected = systems if systems is not None else data.get("systems", list(raw))
        if not isinstance(selected, list):
            raise ValueError("systems 必须为系统名称列表")
        validate_options(selected, 0.0, 0)
        result = {}
        for system in selected:
            if system not in raw or not isinstance(raw[system], dict):
                raise ValueError(f"缺少所选系统的结果: {system}")
            result[system] = preserve_result_contexts(_rows(raw[system].get("records"), system),
                                                       data, raw[system])
        return result
    selected = list(system_files) if systems is None else systems
    validate_options(selected, 0.0, 0)
    result = {}
    for system in selected:
        if system not in system_files:
            raise ValueError(f"缺少所选系统的结果文件: {system}")
        path = system_files[system]
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            envelope = None
        else:
            data = _read_json(path)
            rows = data.get("records") if isinstance(data, dict) else data
            envelope = data if isinstance(data, dict) else None
        result[system] = preserve_result_contexts(_rows(rows, system), envelope)
    return result


def _markdown_report(report: dict) -> str:
    """生成人可读的筛选摘要；逐题去留依据保存在配套 JSON。"""
    c = report["counts"]
    lines = ["# 全员答对题筛选", "", f"系统：{', '.join(report['systems'])}", "",
             f"输入 {c['input']} 题；全员答对 {c['all_correct']} 题；剔除 {c['removed_easy']} 题；最终保留 {c['kept']} 题。", "",
             f"简单题保留比例：{report['keep_easy_ratio']}；种子：{report['seed']}；向下取整保留 {c['kept_easy_sample']} 道简单题。", "",
             f"缺测、异常、重复记录或人工暂缓筛选的 {c['kept_incomplete']} 题保留待核查。", "",
             f"暂缓筛选的能力：{', '.join(report['preserve_capabilities']) or '无'}。", "",
             f"导出用途：{report.get('result_scope', 'unknown')}；发布检查：{report.get('release', {}).get('status', 'not_run')}。", "",
             "| 能力 | 原题数 | 剔除 | 保留 | 待核查 |", "| --- | ---: | ---: | ---: | ---: |"]
    for capability, row in report["by_capability"].items():
        lines.append(f"| {capability} | {row['input']} | {row['removed_easy']} | {row['kept']} | {row['kept_incomplete']} |")
    lines += ["", "筛选依据是所选系统当前版本的结构化判分；缺失来源、判分异常和旧版本标签进入待复核，本步骤不重新作答或判分。",
              "全员答对仅指本次所选系统；筛后发布检查保留原交付数量约束，判分错误仍需先修正再重跑筛选。",
              "逐题判分、去留原因、输入文件 SHA-256 见 filter_report.json。", ""]
    return "\n".join(lines)


def export_filtered_benchmark(bench: Path, results: dict[str, list[dict]], out_dir: Path, *,
                              keep_easy_ratio: float = 0.0, seed: int = 0,
                               preserve_capabilities: tuple[str, ...] | list[str] = (),
                               corpus: Path | None = None, about: Path | None = None,
                               result_paths: list[Path] = (), allow_unverified: bool = False,
                               protocol_enabled: bool = True) -> dict:
    """导出独立派生题库并重验发布资格，保留源交付约束和研究用途边界。"""
    bench, out_dir = Path(bench), Path(out_dir)
    questions = load_questions(bench)
    from pipeline.quality import require_release, evaluate_release
    release = require_release(bench, allow_unverified=allow_unverified, corpus_path=corpus)
    source_manifest_path = bench.parent / "manifest.json"
    source_manifest = _read_json(source_manifest_path) if source_manifest_path.exists() else {}
    if not isinstance(source_manifest, dict):
        raise ValueError("源 manifest 必须为对象")
    copies = {}
    from pipeline.grounding_review import REVIEW_ARTIFACT
    for name, supplied in (("01_whitepaper.json", None), ("02_world.json", None),
                           ("04_questions.json", None), ("05_corpus.json", corpus), ("00_about.json", about),
                           ("06_grounding_report.json", None), (REVIEW_ARTIFACT, None)):
        path = Path(supplied) if supplied is not None else bench.parent / name
        if supplied is not None or path.exists():
            if not path.is_file():
                raise ValueError(f"配套文件不存在: {path}")
            copies[name] = path
    # Filtering does not change the world. Carry its actual review inputs and
    # effective obligation with it, including old whitepapers opted in by config.
    # Freeze the bytes we validate so later copying cannot substitute another
    # world/review pair. Explicit research exports keep failures, never approval.
    from pipeline import world_semantics
    wp_path = bench.parent / "01_whitepaper.json"
    wp_bytes = wp_path.read_bytes() if wp_path.is_file() else None
    wp = json.loads(wp_bytes) if wp_bytes is not None else {}
    seed_v2 = (wp.get("seed_contract") or {}).get("schema_version") == 2
    world_review_enabled = world_semantics.enabled(wp, source_manifest.get("config", {}))
    world_snapshot, world_review_provenance = {}, None
    if seed_v2:
        from pipeline.seed_run import SEED_ARTIFACT, AUDIT_ARTIFACT, GENERATION_ARTIFACT
        for name in (SEED_ARTIFACT, AUDIT_ARTIFACT, GENERATION_ARTIFACT, "00_input.json"):
            path = bench.parent / name
            if path.is_file():
                world_snapshot[name] = path.read_bytes()
                copies[name] = path
            elif not allow_unverified:
                raise ValueError(f"Seed v2 source snapshot missing: {name}")
    if world_review_enabled:
        required = ("01_whitepaper.json", "02_world.json", "00_input.json", world_semantics.REVIEW_ARTIFACT)
        for name in required:
            path = bench.parent / name
            if path.is_file():
                world_snapshot[name] = wp_bytes if name == "01_whitepaper.json" else path.read_bytes()
                copies[name] = path
        world_errors = [f"missing_world_review_input:{name}" for name in required if name not in world_snapshot]
        world_review = None
        if not world_errors:
            try:
                from pipeline.world_state import WorldState
                world_review = json.loads(world_snapshot[world_semantics.REVIEW_ARTIFACT])
                world_errors = world_semantics.validate_review(world_review, wp,
                    WorldState.from_dict(json.loads(world_snapshot["02_world.json"])),
                    task_input=json.loads(world_snapshot["00_input.json"]))
                if not isinstance(world_review, dict) or world_review.get("status") != "passed":
                    world_errors = [*world_errors, "source_world_review_not_passed"]
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                world_errors = [f"invalid_world_review_input:{type(exc).__name__}:{exc}"]
        if world_errors and not allow_unverified:
            raise ValueError("源世界业务审阅缺失、未通过或已过期: " + str(world_errors))
        world_review_provenance = {
            "artifact": world_semantics.REVIEW_ARTIFACT,
            "scope": "unchanged_source_world_and_review_inputs",
            "required": True,
            "validation": {"status": "failed" if world_errors else "passed", "issues": world_errors},
            "input_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in world_snapshot.items()},
        }
    # A release override permits historical research, never an invented result
    # identity. Missing corpus means there is no verified current input to match.
    expected_context = None
    if "05_corpus.json" in copies:
        protocol = (load_public_protocol(copies.get("00_about.json", bench.parent / "00_about.json"))
                    if protocol_enabled else "")
        expected_context = make_evaluation_context(load_visible_corpus(copies["05_corpus.json"]), protocol)
    filtered, report = filter_questions(questions, results, keep_easy_ratio=keep_easy_ratio, seed=seed,
                                        preserve_capabilities=preserve_capabilities,
                                        expected_context=expected_context)
    report["source_release"] = release
    report["protocol_enabled"] = protocol_enabled
    targetspec = deepcopy((source_manifest.get("algo") or {}).get("targetspec") or {})
    if not isinstance(targetspec, dict):
        raise ValueError("源 targetspec 必须为对象")
    source_receipt_path = bench.parent / "07_release.json"
    provenance_files = [p for p in (source_manifest_path, source_receipt_path) if p.is_file()]
    inputs = list(dict.fromkeys([bench, *map(Path, result_paths), *copies.values(), *provenance_files]))
    frozen_by_path = {copies[name].resolve(): raw for name, raw in world_snapshot.items()}
    report["input_files"] = [{"path": str(path.resolve()),
                              "sha256": hashlib.sha256(frozen_by_path[path.resolve()]
                                  if path.resolve() in frozen_by_path else path.read_bytes()).hexdigest()} for path in inputs]
    world_review_failed = (world_review_provenance is not None
                           and world_review_provenance["validation"]["status"] != "passed")
    research_only = bool(release.get("override")) or report["result_scope"] == "research_only" or world_review_failed
    provenance = {"operation": "filter_all_selected_systems_correct", "source_directory": str(bench.parent.resolve()),
                  "source_benchmark": str(bench.resolve()), "source_release_eligible": release.get("eligible") is True,
                  "research_only": research_only, "input_files": deepcopy(report["input_files"]),
                  "evaluation_context": deepcopy(expected_context), "protocol_enabled": protocol_enabled,
                  "source_result_scope": report["result_scope"]}
    if REVIEW_ARTIFACT in copies:
        provenance["semantic_review"] = {
            "artifact": REVIEW_ARTIFACT,
            "sha256": hashlib.sha256(copies[REVIEW_ARTIFACT].read_bytes()).hexdigest(),
            "scope": "unchanged_complete_source_review",
            "source_final_count": len(questions),
            "selected_qids": [q.get("qid") for q in filtered]}
    if world_review_provenance is not None:
        provenance["world_semantic_review"] = world_review_provenance
    manifest = {"schema_version": 1, "status": "research_only" if research_only else "derived_pending_quality",
                "evaluation_mode": "unverified_research" if research_only else "release_required",
                "algo": {"targetspec": targetspec}, "derived_from": provenance,
                "filter": {"systems": report["systems"], "keep_easy_ratio": keep_easy_ratio, "seed": seed,
                           "preserve_capabilities": report["preserve_capabilities"], "counts": report["counts"]}}
    if world_review_enabled:
        manifest["config"] = {"world_semantic_review": True}
    if seed_v2:
        manifest.setdefault("config", {}).update({key: source_manifest.get("config", {}).get(key)
            for key in ("seed_pack_digest", "seed_id")})
    if research_only:
        manifest["release_policy"] = {"inherited_research_only": True,
                                      "source_status": release.get("status"),
                                      "source_benchmark_sha256": hashlib.sha256(bench.read_bytes()).hexdigest(),
                                      "source_receipt_sha256": hashlib.sha256(source_receipt_path.read_bytes()).hexdigest()
                                      if source_receipt_path.is_file() else None}
    report["derived_from"] = provenance
    report["outputs"] = ["06_grounded_questions.json", "manifest.json", "07_release.json",
                         "filter_report.json", "filter_report.md", *copies]
    # exist_ok=False 阻止覆盖源目录和旧导出；所有解析、参数校验在创建目录前完成。
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "06_grounded_questions.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, source in copies.items():
        if name in world_snapshot:
            (out_dir / name).write_bytes(world_snapshot[name])
        else:
            shutil.copyfile(source, out_dir / name)
    def write_json(name, value):
        (out_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_candidates = (_read_json(copies["04_questions.json"])
                         if "04_questions.json" in copies else deepcopy(questions))
    if "04_questions.json" not in copies:
        write_json("04_questions.json", source_candidates)
    source_routing = (_read_json(copies["06_grounding_report.json"])
                      if "06_grounding_report.json" in copies else {
                          "drops": [],
                          "pending": [{"qid": row.get("qid"), "reason": "legacy_source_status_unknown"}
                                      for row in source_candidates
                                      if row.get("qid") not in {q.get("qid") for q in questions}],
                      })
    if not isinstance(source_routing, dict):
        raise ValueError("源 06_grounding_report.json 必须为对象")
    source_released = {q.get("qid") for q in questions}
    filtered_qids = {q.get("qid") for q in filtered}
    existing_excluded = source_routing.get("scoped_excluded") or []
    if not isinstance(existing_excluded, list):
        raise ValueError("源 scoped_excluded 必须为数组")
    source_routing["scoped_excluded"] = list(dict.fromkeys([
        *existing_excluded,
        *(q.get("qid") for q in questions
          if q.get("qid") in source_released - filtered_qids),
    ]))
    write_json("06_grounding_report.json", source_routing)
    write_json("manifest.json", manifest)
    receipt = evaluate_release(out_dir)
    receipt["derived_from"] = provenance
    if not filtered:
        receipt["issues"].append({"code": "empty_filtered_benchmark"})
        receipt.update(status="failed", eligible=False)
    if research_only:
        if not any(issue.get("code") == "unverified_source_derivation" for issue in receipt["issues"]):
            receipt["issues"].append({"code": "unverified_source_derivation",
                                      "message": "Research-only evaluation results or source cannot certify a derived release."})
        receipt.update(status="failed", eligible=False)
    else:
        manifest["status"] = "done" if receipt["eligible"] else "quality_failed"
    write_json("manifest.json", manifest)
    write_json("07_release.json", receipt)
    report["release"] = receipt
    report["result_scope"] = ("research_only" if research_only else
                              "release_eligible" if receipt["eligible"] else "filtered_release_failed")
    (out_dir / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "filter_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main(argv=None) -> int:
    """离线筛题入口；可直接使用四种 harness 的原始 results.jsonl。"""
    parser = argparse.ArgumentParser(description="剔除全员答对题，或固定种子按比例保留；不调用模型")
    parser.add_argument("--bench", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="eval.multi_system 生成的 results.json")
    source.add_argument("--system-result", action="append", metavar="NAME=PATH", help="逐系统 JSON/JSONL；每个系统指定一次")
    parser.add_argument("--systems", help="逗号分隔的系统名单；默认使用输入声明的全部系统")
    parser.add_argument("--keep-easy-ratio", type=float, default=0.0, help="全员答对题保留比例，0 全剔除，1 全保留")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preserve-capability", action="append", default=[], help="暂缓筛选的能力名，可重复指定；适用于判分待复核的题")
    parser.add_argument("--out-dir", required=True, type=Path, help="尚不存在的输出目录")
    parser.add_argument("--corpus", type=Path, help="默认复制 bench 同目录的 05_corpus.json")
    parser.add_argument("--about", type=Path, help="默认复制 bench 同目录的 00_about.json")
    parser.add_argument("--no-protocol", action="store_true", help="核对未注入公开答题约定的实验成绩")
    parser.add_argument("--allow-unverified", action="store_true", help="仅历史研究：允许未取得发布资格的输入")
    args = parser.parse_args(argv)
    try:
        system_files = {}
        for spec in args.system_result or []:
            name, separator, path = spec.partition("=")
            name = name.strip()
            if not separator or not name or not path or name in system_files:
                raise ValueError(f"无效或重复的 --system-result: {spec}")
            system_files[name] = Path(path)
        systems = None if args.systems is None else args.systems.split(",")
        systems = [s.strip() for s in systems] if systems is not None else None
        results = load_results(aggregate=args.results, system_files=system_files, systems=systems)
        paths = [args.results] if args.results else [system_files[s] for s in results]
        report = export_filtered_benchmark(args.bench, results, args.out_dir,
            keep_easy_ratio=args.keep_easy_ratio, seed=args.seed, corpus=args.corpus,
            about=args.about, result_paths=paths, preserve_capabilities=args.preserve_capability,
            allow_unverified=args.allow_unverified, protocol_enabled=not args.no_protocol)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"out_dir": str(args.out_dir), "result_scope": report["result_scope"],
                      "release_status": report["release"]["status"], **report["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
