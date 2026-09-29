from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_standard_light(root: Path, questions: list[dict[str, Any]], references: list[dict[str, Any]]) -> Path:
    """Create a minimal, manifest-verified standard-light benchmark."""
    files: dict[str, Any] = {
        "private/world.json": {"entities": []},
        "public/material.json": [
            {
                "doc_id": "doc-1",
                "content": "第一周，项目状态为进行中。",
                "date": "2026-01-05",
                "session": 1,
            }
        ],
        "public/protocol.txt": "只依据公开材料回答；信息不足时明确拒答。\n",
        "public/questions.json": questions,
        "references/questions.json": references,
    }
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".json"):
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            path.write_text(str(value), encoding="utf-8")

    file_manifest = {}
    for relative in files:
        path = root / relative
        file_manifest[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    statuses = Counter(str(question["quality_status"]) for question in questions)
    manifest = {
        "schema_version": "memory-bench-standard-light/v1",
        "benchmark_id": "test_standard_light",
        "domain": "test-domain",
        "seed_id": "test-seed",
        "source_run": "test-source-run",
        "question_selection": "all_candidates_with_quality_status",
        "default_evaluation_statuses": ["released"],
        "counts": {
            "documents": 1,
            "questions": len(questions),
            "by_quality_status": dict(statuses),
        },
        "files": file_manifest,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root
