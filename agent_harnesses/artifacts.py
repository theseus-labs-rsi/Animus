"""工厂评测输入的只读校验和指纹。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkRef, ConfigurationError


REQUIRED_FILES = ("00_about.json", "05_corpus.json", "06_grounded_questions.json")
LEGACY_SCHEMA = "memory-bench-factory-native/v1"
STANDARD_LIGHT_SCHEMA = "memory-bench-standard-light/v1"
STANDARD_FILES = (
    "private/world.json",
    "public/material.json",
    "public/protocol.txt",
    "public/questions.json",
    "references/questions.json",
)
ANSWER_PROJECTIONS = {"raw_gt", "gt/value", "abstention:never_known"}


@dataclass(frozen=True)
class BenchmarkInspection:
    path: Path
    schema: str
    scenario: str
    factory_run_id: str
    factory_status: str
    met_status: str | None
    n_sessions: int
    n_docs: int
    n_questions: int
    files: dict[str, str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "schema": self.schema,
            "scenario": self.scenario,
            "factory_run_id": self.factory_run_id,
            "factory_status": self.factory_status,
            "met_status": self.met_status,
            "n_sessions": self.n_sessions,
            "n_docs": self.n_docs,
            "n_questions": self.n_questions,
            "files": self.files,
            "warnings": list(self.warnings),
        }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"benchmark 缺少文件: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"benchmark JSON 无法解析: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_schema(root: Path) -> str:
    """Detect the on-disk benchmark contract without guessing from content."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if isinstance(manifest, dict) and manifest.get("schema_version") == STANDARD_LIGHT_SCHEMA:
            return STANDARD_LIGHT_SCHEMA
    return LEGACY_SCHEMA


def question_id(question: dict[str, Any]) -> str:
    """Use the factory/export identity first; hashes are legacy-only fallback."""
    explicit = question.get("qid") or question.get("question_id")
    if explicit is not None:
        return str(explicit)
    text = _require_nonempty_text(question.get("question"), "question.question")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_benchmark_questions(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if benchmark_schema(root) == STANDARD_LIGHT_SCHEMA:
        value = _load_json(root / "public" / "questions.json")
    else:
        value = _load_json(root / "06_grounded_questions.json")
    questions = value if isinstance(value, list) else value.get("questions") if isinstance(value, dict) else None
    if not isinstance(questions, list):
        raise ConfigurationError("benchmark questions 必须是数组")
    return questions


def load_benchmark_references(root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root)
    if benchmark_schema(root) != STANDARD_LIGHT_SCHEMA:
        return {}
    rows = _load_json(root / "references" / "questions.json")
    if not isinstance(rows, list):
        raise ConfigurationError("references/questions.json 必须是数组")
    return {str(row["qid"]): row for row in rows}


def load_benchmark_protocol(root: Path) -> str:
    root = Path(root)
    if benchmark_schema(root) == STANDARD_LIGHT_SCHEMA:
        path = root / "public" / "protocol.txt"
        try:
            protocol = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigurationError(f"benchmark 缺少文件: {path}") from exc
        if not protocol.strip():
            raise ConfigurationError("public/protocol.txt 不能为空")
        return protocol
    from eval.provenance import load_public_protocol

    return load_public_protocol(root / "00_about.json")


def load_benchmark_sessions(root: Path) -> list[dict[str, Any]]:
    """Return the exact public material grouped by session for both tracks."""
    root = Path(root)
    if benchmark_schema(root) == STANDARD_LIGHT_SCHEMA:
        documents = _load_json(root / "public" / "material.json")
        if not isinstance(documents, list):
            raise ConfigurationError("public/material.json 必须是数组")
        grouped: dict[int, dict[str, Any]] = {}
        for document in documents:
            sid = int(document["session"])
            date = str(document["date"])
            session = grouped.setdefault(sid, {"session_id": sid, "date": date, "docs": []})
            if session["date"] != date:
                raise ConfigurationError(f"session={sid} 出现多个 date")
            session["docs"].append(
                {
                    "doc_id": str(document["doc_id"]),
                    "type": "Document",
                    "content": str(document["content"]),
                }
            )
        return [grouped[sid] for sid in sorted(grouped)]
    value = _load_json(root / "05_corpus.json")
    corpus = value.get("corpus", value) if isinstance(value, dict) else None
    sessions = corpus.get("sessions") if isinstance(corpus, dict) else None
    if not isinstance(sessions, list):
        raise ConfigurationError("05_corpus.json.corpus.sessions 必须是数组")
    return sessions


def load_visible_documents(root: Path) -> list[tuple[int, str, str]]:
    """The Memory Track view: session, date and public document body only."""
    return [
        (int(session["session_id"]), str(session.get("date") or ""), str(document["content"]))
        for session in load_benchmark_sessions(root)
        for document in session["docs"]
    ]


def _require_mapping(value: Any, location: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} 必须是 object")
    return value


def _require_nonempty_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} 必须是非空字符串")
    return value


def _validate_about(value: Any) -> None:
    about = _require_mapping(value, "00_about.json")
    protocol = _require_mapping(
        about.get("answer_protocol"), "00_about.json.answer_protocol"
    )
    rules = protocol.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ConfigurationError("00_about.json.answer_protocol.rules 必须是非空数组")
    for index, rule in enumerate(rules):
        _require_nonempty_text(rule, f"00_about.json.answer_protocol.rules[{index}]")
    sentinel_map = protocol.get("gold_sentinel_map", {})
    if not isinstance(sentinel_map, dict):
        raise ConfigurationError(
            "00_about.json.answer_protocol.gold_sentinel_map 必须是 object"
        )


def _validate_corpus(value: Any) -> tuple[list[dict], set[int]]:
    root = _require_mapping(value, "05_corpus.json")
    inner = root.get("corpus", root)
    inner = _require_mapping(inner, "05_corpus.json.corpus")
    sessions = inner.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ConfigurationError("05_corpus.json.corpus.sessions 必须是非空数组")
    session_ids: set[int] = set()
    doc_ids: set[str] = set()
    for session_index, raw_session in enumerate(sessions):
        location = f"05_corpus.json.corpus.sessions[{session_index}]"
        session = _require_mapping(raw_session, location)
        try:
            session_id = int(session["session_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"{location}.session_id 必须是整数") from exc
        if session_id in session_ids:
            raise ConfigurationError(f"{location}.session_id 重复: {session_id}")
        session_ids.add(session_id)
        _require_nonempty_text(session.get("date"), f"{location}.date")
        docs = session.get("docs")
        if not isinstance(docs, list):
            raise ConfigurationError(f"{location}.docs 必须是数组")
        for doc_index, raw_doc in enumerate(docs):
            doc_location = f"{location}.docs[{doc_index}]"
            doc = _require_mapping(raw_doc, doc_location)
            doc_id = _require_nonempty_text(doc.get("doc_id"), f"{doc_location}.doc_id")
            if doc_id in doc_ids:
                raise ConfigurationError(f"{doc_location}.doc_id 重复: {doc_id}")
            doc_ids.add(doc_id)
            _require_nonempty_text(doc.get("type"), f"{doc_location}.type")
            _require_nonempty_text(doc.get("content"), f"{doc_location}.content")
            fact_refs = doc.get("fact_refs", [])
            if not isinstance(fact_refs, list) or not all(
                isinstance(item, str) for item in fact_refs
            ):
                raise ConfigurationError(f"{doc_location}.fact_refs 必须是字符串数组")
    return sessions, session_ids


def _validate_questions(value: Any, session_ids: set[int]) -> tuple[list[dict], list[str]]:
    questions = (
        value
        if isinstance(value, list)
        else value.get("questions") if isinstance(value, dict) else None
    )
    if not isinstance(questions, list) or not questions:
        raise ConfigurationError("06_grounded_questions.json.questions 必须是非空数组")
    warnings: list[str] = []
    explicit_ids: set[str] = set()
    missing_explicit_id = 0
    for index, raw_question in enumerate(questions):
        location = f"06_grounded_questions.json.questions[{index}]"
        question = _require_mapping(raw_question, location)
        _require_nonempty_text(question.get("question"), f"{location}.question")
        _require_nonempty_text(question.get("line"), f"{location}.line")
        _require_nonempty_text(question.get("capability"), f"{location}.capability")
        if "gt" not in question:
            raise ConfigurationError(f"{location}.gt 缺失")
        question_id = question.get("qid") or question.get("question_id")
        if question_id is None:
            missing_explicit_id += 1
        else:
            question_id = _require_nonempty_text(question_id, f"{location}.qid")
            if question_id in explicit_ids:
                raise ConfigurationError(f"{location}.qid 重复: {question_id}")
            explicit_ids.add(question_id)
        evidence = question.get("evidence_sessions", [])
        if not isinstance(evidence, list):
            raise ConfigurationError(f"{location}.evidence_sessions 必须是数组")
        for raw_session_id in evidence:
            try:
                session_id = int(raw_session_id)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{location}.evidence_sessions 包含非整数值"
                ) from exc
            if session_id not in session_ids:
                raise ConfigurationError(
                    f"{location}.evidence_sessions 引用了不存在的 session_id={session_id}"
                )
    if missing_explicit_id:
        warnings.append(
            f"{missing_explicit_id}/{len(questions)} 道题缺少稳定 qid/question_id；"
            "运行器只能使用内容 hash 作为临时 ID"
        )
    return questions, warnings


def _standard_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"manifest 文件路径越界: {relative!r}") from exc
    return path


def _validate_standard_light(root: Path) -> BenchmarkInspection:
    manifest_path = root / "manifest.json"
    manifest = _require_mapping(_load_json(manifest_path), "manifest.json")
    if manifest.get("schema_version") != STANDARD_LIGHT_SCHEMA:
        raise ConfigurationError("manifest.json.schema_version 非法")
    declared = _require_mapping(manifest.get("files"), "manifest.json.files")
    if set(declared) != set(STANDARD_FILES):
        raise ConfigurationError(
            f"manifest.json.files 必须恰好声明 {list(STANDARD_FILES)!r}"
        )

    files = {"manifest.json": _sha256(manifest_path)}
    for relative in STANDARD_FILES:
        path = _standard_path(root, relative)
        if not path.is_file():
            raise ConfigurationError(f"benchmark 缺少文件: {path}")
        info = _require_mapping(declared[relative], f"manifest.json.files.{relative}")
        expected_hash = _require_nonempty_text(info.get("sha256"), f"{relative}.sha256")
        expected_bytes = info.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ConfigurationError(f"{relative}.bytes 必须是非负整数")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash or path.stat().st_size != expected_bytes:
            raise ConfigurationError(f"manifest 文件身份不匹配: {relative}")
        files[relative] = actual_hash

    protocol = load_benchmark_protocol(root)
    if not protocol.strip():  # kept explicit even though the loader already rejects it
        raise ConfigurationError("public/protocol.txt 不能为空")

    documents = _load_json(root / "public" / "material.json")
    if not isinstance(documents, list) or not documents:
        raise ConfigurationError("public/material.json 必须是非空数组")
    doc_ids: set[str] = set()
    sessions: set[int] = set()
    for index, raw in enumerate(documents):
        location = f"public/material.json[{index}]"
        document = _require_mapping(raw, location)
        doc_id = _require_nonempty_text(document.get("doc_id"), f"{location}.doc_id")
        if doc_id in doc_ids:
            raise ConfigurationError(f"{location}.doc_id 重复: {doc_id}")
        doc_ids.add(doc_id)
        if type(document.get("session")) is not int:
            raise ConfigurationError(f"{location}.session 必须是整数")
        sessions.add(document["session"])
        _require_nonempty_text(document.get("date"), f"{location}.date")
        _require_nonempty_text(document.get("content"), f"{location}.content")

    public = _load_json(root / "public" / "questions.json")
    references = _load_json(root / "references" / "questions.json")
    if not isinstance(public, list) or not public:
        raise ConfigurationError("public/questions.json 必须是非空数组")
    if not isinstance(references, list) or len(references) != len(public):
        raise ConfigurationError("references/questions.json 必须与 public/questions.json 等长")

    public_ids: list[str] = []
    status_counts: Counter[str] = Counter()
    for index, raw in enumerate(public):
        location = f"public/questions.json[{index}]"
        question = _require_mapping(raw, location)
        qid = _require_nonempty_text(question.get("qid"), f"{location}.qid")
        public_ids.append(qid)
        for field in ("question", "line", "capability", "quality_status"):
            _require_nonempty_text(question.get(field), f"{location}.{field}")
        status_counts[str(question["quality_status"])] += 1
    if len(set(public_ids)) != len(public_ids):
        raise ConfigurationError("public/questions.json.qid 必须唯一")

    reference_ids: list[str] = []
    for index, raw in enumerate(references):
        location = f"references/questions.json[{index}]"
        reference = _require_mapping(raw, location)
        qid = _require_nonempty_text(reference.get("qid"), f"{location}.qid")
        reference_ids.append(qid)
        projection = _require_nonempty_text(
            reference.get("answer_projection"), f"{location}.answer_projection"
        )
        if projection not in ANSWER_PROJECTIONS:
            raise ConfigurationError(f"{location}.answer_projection 不支持: {projection!r}")
        if "answer_raw" not in reference or "answer" not in reference:
            raise ConfigurationError(f"{location} 缺少 answer/answer_raw")
        if reference.get("quality_status") != public[index].get("quality_status"):
            raise ConfigurationError(f"{location}.quality_status 与 public 不一致")
        raw_answer, answer = reference["answer_raw"], reference["answer"]
        if projection == "raw_gt" and answer != raw_answer:
            raise ConfigurationError(f"{location}: raw_gt 要求 answer == answer_raw")
        if projection == "gt/value" and (
            not isinstance(raw_answer, dict) or answer != raw_answer.get("value")
        ):
            raise ConfigurationError(f"{location}: gt/value 投影不一致")
        if projection == "abstention:never_known" and (
            raw_answer != "INSUFFICIENT_EVIDENCE"
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ConfigurationError(f"{location}: abstention 投影不一致")
    if public_ids != reference_ids:
        raise ConfigurationError("public/references 的 qid 顺序必须完全一致")

    counts = _require_mapping(manifest.get("counts"), "manifest.json.counts")
    if counts.get("documents") != len(documents) or counts.get("questions") != len(public):
        raise ConfigurationError("manifest counts 与实际文件不一致")
    declared_statuses = counts.get("by_quality_status")
    if (
        not isinstance(declared_statuses, dict)
        or any(type(value) is not int or value < 0 for value in declared_statuses.values())
        or sum(declared_statuses.values()) != len(public)
        or any(status_counts.get(status, 0) != count for status, count in declared_statuses.items())
        or any(status not in declared_statuses for status in status_counts)
    ):
        raise ConfigurationError("manifest by_quality_status 与实际题目不一致")

    return BenchmarkInspection(
        path=root,
        schema=STANDARD_LIGHT_SCHEMA,
        scenario=_require_nonempty_text(manifest.get("domain"), "manifest.json.domain"),
        factory_run_id=_require_nonempty_text(manifest.get("source_run"), "manifest.json.source_run"),
        factory_status="standardized",
        met_status=None,
        n_sessions=len(sessions),
        n_docs=len(documents),
        n_questions=len(public),
        files=files,
        warnings=("quality_status 仅保留用于诊断；本评测协议执行全部题目",),
    )


def inspect_benchmark(ref: BenchmarkRef) -> BenchmarkInspection:
    root = ref.path
    if not root.is_dir():
        raise ConfigurationError(f"benchmark 目录不存在: {root}")
    if benchmark_schema(root) == STANDARD_LIGHT_SCHEMA:
        return _validate_standard_light(root)
    paths = {name: root / name for name in REQUIRED_FILES}
    for path in paths.values():
        if not path.is_file():
            raise ConfigurationError(f"benchmark 缺少 {path.name}: {root}")

    _validate_about(_load_json(paths["00_about.json"]))
    sessions, session_ids = _validate_corpus(_load_json(paths["05_corpus.json"]))
    questions_obj = _load_json(paths["06_grounded_questions.json"])
    questions, schema_warnings = _validate_questions(questions_obj, session_ids)

    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    validated_path = root / "validated.json"
    validated = _load_json(validated_path) if validated_path.is_file() else {}
    algo = manifest.get("algo") if isinstance(manifest.get("algo"), dict) else {}
    raw_met_status = validated.get("met_status") or algo.get("met_status")
    met_status = str(raw_met_status) if raw_met_status else None
    warnings: list[str] = list(schema_warnings)
    if met_status and not met_status.startswith("MET"):
        warnings.append(f"factory release gate 未满足: {met_status}")
        if not ref.allow_unmet:
            raise ConfigurationError(
                f"benchmark 是 {met_status}；仅诊断实验可显式设置 benchmark.allow_unmet=true"
            )
    if validated_path.is_file() and ref.corpus_variant == "full":
        raise ConfigurationError("当前目录是 filtered/validated 产物，但 corpus_variant 声明为 full")
    if not validated_path.is_file() and ref.corpus_variant == "filtered":
        warnings.append("corpus_variant=filtered，但目录没有 validated.json，无法证明过滤 lineage")

    scenario = str(validated.get("scenario") or manifest.get("scenario") or root.name.split("__", 1)[0])
    run_id = str(manifest.get("run_id") or validated.get("source_run") or root.name)
    return BenchmarkInspection(
        path=root,
        schema=LEGACY_SCHEMA,
        scenario=scenario,
        factory_run_id=run_id,
        factory_status=str(manifest.get("status") or "unknown"),
        met_status=met_status,
        n_sessions=len(sessions),
        n_docs=sum(len(s.get("docs") or []) for s in sessions if isinstance(s, dict)),
        n_questions=len(questions),
        files={name: _sha256(path) for name, path in paths.items()},
        warnings=tuple(warnings),
    )
