"""Import identified native-harness votes for generation's existing easy filter.

This module never runs a solver or regrades an answer. Source votes and their
judge metadata remain visible; absent or failed votes cannot remove a question.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent_harnesses.artifacts import STANDARD_LIGHT_SCHEMA, inspect_benchmark
from agent_harnesses.config import BenchmarkRef
from eval.grading import JUDGE_VERSION, requires_semantic_grading
from eval.provenance import (digest, load_visible_corpus,
                             make_evaluation_context, record_provenance)
from pipeline.benchmark_export import public_view
from pipeline.run import _atomic_write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index(rows, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label}: expected a question list")
    result = {}
    for row in rows:
        qid = row.get("qid") if isinstance(row, dict) else None
        if not isinstance(qid, str) or not qid or qid in result:
            raise ValueError(f"{label}: missing or duplicate qid {qid!r}")
        result[qid] = row
    return result


def _answer(q):
    # Same public-reference projection as benchmark_export.export_selected.
    if "gt" not in q:
        raise ValueError(f"Missing reference answer for {q['qid']}")
    raw = q["gt"]
    if raw in ("INSUFFICIENT", "INSUFFICIENT_EVIDENCE"):
        return "无此项/查无此记录", raw, "abstention:never_known"
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"], raw, "gt/value"
    return raw, raw, "raw_gt"


def _targets(settings):
    systems, targets = settings.get("systems"), settings.get("targets")
    if (settings.get("backend") != "native_four" or not isinstance(systems, list)
            or len(systems) != 4 or any(not isinstance(s, str) or not s for s in systems)
            or len(set(systems)) != 4 or not isinstance(targets, list) or len(targets) != 4):
        raise ValueError("native_four requires four explicit distinct target IDs")
    by_id = {}
    for target in targets:
        if (not isinstance(target, dict) or any(not isinstance(target.get(k), str) or not target[k]
                for k in ("id", "system", "model_id", "model_label", "endpoint_profile", "protocol_style"))
                or target["id"] in by_id or target["system"] not in {"native.codex", "native.dsh"}):
            raise ValueError("Invalid or duplicate native target metadata")
        by_id[target["id"]] = target
    if set(by_id) != set(systems):
        raise ValueError("settings systems must exactly match target IDs")
    return systems, by_id


def _validate_plan(plan, target, source_files):
    model, runtime = plan.get("model") or {}, plan.get("runtime") or {}
    adapter = target["system"].removeprefix("native.")
    expected = {"label": target["model_label"], "model_id": target["model_id"],
                "endpoint_profile": target["endpoint_profile"], "protocol_style": target["protocol_style"]}
    if (plan.get("target_id") != target["id"] or plan.get("system_id") != target["system"]
            or plan.get("runner") != "native_cli" or plan.get("track") != "native"
            or any(model.get(k) != v for k, v in expected.items())
            or runtime.get("model") != target["model_id"] or runtime.get("adapter") != adapter
            or runtime.get("protocol_style") != target["protocol_style"]):
        raise ValueError(f"Native target/model identity mismatch: {target['id']}")
    benchmark = plan.get("benchmark") or {}
    if benchmark.get("schema") != STANDARD_LIGHT_SCHEMA or benchmark.get("files") != source_files:
        raise ValueError(f"Native source benchmark hashes mismatch: {target['id']}")


def _native_rows(path, public, *, latest):
    """Only execution logs may append retries for the same index and qid."""
    if not path.is_file():
        return {}
    rows, seen_ids = {}, {}
    for text in path.read_text(encoding="utf-8-sig").splitlines():
        if not text.strip():
            continue
        row = json.loads(text)
        index = row.get("question_index") if isinstance(row, dict) else None
        qid = row.get("question_id") if isinstance(row, dict) else None
        if (type(index) is not int or not 0 <= index < len(public)
                or qid != public[index]["qid"]):
            raise ValueError(f"{path.name}: unknown or mismatched question index/qid")
        if qid in seen_ids and (not latest or seen_ids[qid] != index):
            raise ValueError(f"{path.name}: duplicate qid {qid}")
        if index in rows and not latest:
            raise ValueError(f"{path.name}: duplicate question index {index}")
        seen_ids[qid] = index
        rows[index] = row
    return rows


def _normalized(q, execution, judged, context, target, source):
    pred = execution.get("answer") if execution else None
    judge = judged.get("judge") if judged else None
    reason = None
    if not execution or not judged:
        reason = "missing native answer or score"
    elif (execution.get("status") != "completed" or execution.get("error_type")
          or execution.get("returncode") not in (None, 0)):
        reason = "native answer execution failed"
    elif not isinstance(pred, str) or not pred.strip():
        reason = "missing native answer text"
    elif (execution.get("contaminated") or execution.get("boundary_violation")
          or judged.get("contaminated") or judged.get("boundary_violation")):
        reason = "native result reports evaluation boundary contamination"
    elif (judged.get("judgeable") is not True or type(judged.get("correct")) is not bool
          or not isinstance(judge, dict) or judge.get("error") or not judge.get("version")):
        reason = "native score missing, unjudgeable or failed"
    if requires_semantic_grading(q):
        reason = "native legacy vote does not provide the bound semantic process grade"
    correct = None if reason else judged["correct"]
    grade = {"version": JUDGE_VERSION, "path": "native_harness",
             "verdict": "unjudgeable" if requires_semantic_grading(q) else
                        "error" if reason else "correct" if correct else "incorrect",
             "correct": correct,
             "reason": reason or "Imported the identified native source vote without regrading",
             "source_judge": deepcopy(judge), "regraded": False}
    return {**deepcopy(q), "pred": pred, "correct": correct, "judgeable": reason is None,
            "execution_status": "incomplete" if reason else "ok", "judgement": grade,
            "evaluation_provenance": record_provenance(q, context),
            "native_source": {**source, "target": deepcopy(target),
                              "execution_status": execution.get("status") if execution else None,
                              "error_type": execution.get("error_type") if execution else None},
            "native_judge": deepcopy(judge)}


def import_native_results(run, source_benchmark: Path, result_dirs: dict[str, Path],
                          directory: Path, settings: dict) -> Path:
    """Validate source material/answers and normalize four native result streams.

    Source all-labeled benchmarks may contain extra questions. Missing targets
    or scores produce incomplete votes; changed inputs or stale scores fail.
    The only written artifact is ``directory/results.json``.
    """
    systems, targets = _targets(settings)
    if not isinstance(result_dirs, dict) or set(result_dirs) - set(systems):
        raise ValueError("Unknown native result target")
    source_benchmark, directory = Path(source_benchmark), Path(directory)
    inspected = inspect_benchmark(BenchmarkRef(source_benchmark, False, "as_provided"))
    if inspected.schema != STANDARD_LIGHT_SCHEMA:
        raise ValueError("Native result import requires a standard-light benchmark")
    public = _read(source_benchmark / "public/questions.json")
    public_by_id = _index(public, "source public questions")
    references = _index(_read(source_benchmark / "references/questions.json"), "source references")
    questions = run.read("06_grounded_questions.json")
    if isinstance(questions, dict):
        questions = questions.get("questions")
    current = _index(questions, "current eligible questions")
    for qid, q in current.items():
        p, ref = public_by_id.get(qid), references.get(qid)
        if p is None or ref is None or any(q.get(k) != p.get(k) for k in ("question", "line", "capability")):
            raise ValueError(f"Current question is absent or changed in source: {qid}")
        if digest(_answer(q)) != digest((ref.get("answer"), ref.get("answer_raw"), ref.get("answer_projection"))):
            raise ValueError(f"Current reference answer differs from native source: {qid}")
    material, protocol = public_view(run)
    if (_read(source_benchmark / "public/material.json") != material
            or (source_benchmark / "public/protocol.txt").read_text(encoding="utf-8") != protocol):
        raise ValueError("Native source public corpus/protocol differs from current run")
    context = make_evaluation_context(load_visible_corpus(run.dir / "05_corpus.json"), protocol)
    result = {"schema": "memory-bench-native-results/v1", "schema_version": 1,
              "backend": "native_four", "systems": systems,
              "evaluation_context": context, "source_benchmark": str(source_benchmark.resolve()),
              "source_files": inspected.files, "regraded": False, "results": {}}
    for system in systems:
        source_dir = Path(result_dirs[system]) if system in result_dirs else None
        source = {"directory": str(source_dir.resolve()) if source_dir else None}
        execution, judged = {}, {}
        if source_dir is not None and (source_dir / "run_plan.json").is_file():
            plan = _read(source_dir / "run_plan.json")
            _validate_plan(plan, targets[system], inspected.files)
            execution = _native_rows(source_dir / "results.jsonl", public, latest=True)
            judged = _native_rows(source_dir / "judged.jsonl", public, latest=False)
            for index, row in judged.items():
                latest = execution.get(index)
                if latest is None or any(digest(row.get(k)) != digest(latest.get(k)) for k in
                        ("answer", "status", "error_type", "question_id", "question_index", "returncode")):
                    raise ValueError(f"Stale native score differs from latest answer: {system}/{index}")
                if (row.get("truth_source") != "references/questions.json.answer"
                        or digest(row.get("reference_answer")) != digest(references[row["question_id"]]["answer"])):
                    raise ValueError(f"Native scored reference answer mismatch: {system}/{index}")
            source["files"] = {name: _sha(source_dir / name) for name in
                               ("run_plan.json", "results.jsonl", "judged.jsonl") if (source_dir / name).is_file()}
        elif source_dir is not None and any((source_dir / name).is_file() for name in ("results.jsonl", "judged.jsonl")):
            raise ValueError(f"Native results lack their source run plan: {system}")
        by_qid = {q["qid"]: index for index, q in enumerate(public)}
        rows = [_normalized(q, execution.get(by_qid[q["qid"]]), judged.get(by_qid[q["qid"]]),
                            context, targets[system], source) for q in questions]
        result["results"][system] = {"records": rows, "evaluation_context": context}
    if (directory.resolve() == source_benchmark.resolve()
            or any(source_dir is not None and directory.resolve() == Path(source_dir).resolve()
                   for source_dir in result_dirs.values())):
        raise ValueError("Normalized output must be separate from native source directories")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    if path.exists():
        previous = _read(path)
        if previous == result:
            return path
        if not isinstance(previous, dict) or previous.get("schema") != result["schema"]:
            raise ValueError("Refusing to overwrite results not produced by the native import bridge")
    _atomic_write_json(path, result)
    return path
