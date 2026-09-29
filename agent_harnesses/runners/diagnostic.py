"""不依赖旧实现的离线控制面 diagnostic runner。

它只验证 benchmark 读取、逐题迭代和规范化产物写出，不生成可计分答案。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..artifacts import load_benchmark_questions, load_benchmark_sessions, question_id


def run(run_dir: Path, out_dir: Path, limit: int = 0) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    sessions = load_benchmark_sessions(run_dir)
    questions = load_benchmark_questions(run_dir)
    selected = questions[:limit] if limit else questions

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for index, question in enumerate(selected):
            record = {
                "schema": "agent-harnesses.result/v1",
                "question_id": question_id(question),
                "question_index": index,
                "quality_status": question.get("quality_status"),
                "status": "diagnostic_only",
                "answer": None,
                "judgeable": False,
                "correct": None,
                "error_type": None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema": "agent-harnesses.diagnostic-summary/v1",
        "backend": "builtin_diagnostic",
        "n_sessions": len(sessions),
        "n_docs": sum(len(session["docs"]) for session in sessions),
        "n_questions_available": len(questions),
        "n_questions_checked": len(selected),
        "scorable": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline control-plane diagnostic")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(Path(args.run), Path(args.out), args.limit),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
