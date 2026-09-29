"""Standard-light delivery from an existing run; no generation or model calls."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

from eval.provenance import load_public_protocol
from pipeline.run import _atomic_write_json


def public_view(run) -> tuple[list[dict], str]:
    """The standard-light public document IDs and verbatim published rules.

    This is the same serialization used by the delivered standard-light packs;
    it does not append the separate legacy evaluator's additional instructions.
    """
    corpus = run.read("05_corpus.json")
    sessions = corpus.get("corpus", corpus)["sessions"]
    material = [{"doc_id": f"d{index:06d}", "content": doc["content"],
                 "date": session["date"], "session": session["session_id"]}
                for index, (session, doc) in enumerate(
                    ((session, doc) for session in sorted(sessions, key=lambda s: int(s["session_id"]))
                     for doc in session["docs"]), 1)]
    about = run.read("00_about.json")
    if "public_protocol" in about:
        return material, load_public_protocol(run.dir / "00_about.json")
    ap = about.get("answer_protocol") or {}
    if not ap:
        return material, ""
    lines = []
    for label, key in (("answer_protocol_version", "version"),
                       ("scoring_policy", "scoring_policy"), ("scoring_scope", "scoring_scope")):
        if key in ap:
            lines.append(f"{label}: {ap[key]}")
    lines.extend(["", "rules:", *[f"{index}. {rule}" for index, rule in enumerate(ap.get("rules", []), 1)]])
    lines.extend(["", "gold_sentinel_map:",
                  *[f"- {key}: {value}" for key, value in (ap.get("gold_sentinel_map") or {}).items()]])
    return material, "\n".join(lines) + "\n"


def export_selected(run, questions: list[dict], destination: Path, selection: dict) -> dict:
    """Export the selected released subset, retaining the source run unchanged."""
    from agent_harnesses.artifacts import inspect_benchmark
    from agent_harnesses.config import BenchmarkRef

    destination = Path(destination)
    material, protocol = public_view(run)
    if not protocol.strip():
        raise ValueError("Cannot export a benchmark without its actual public protocol")
    public, references = [], []
    for q in questions:
        public.append({**{key: q[key] for key in ("qid", "question", "line", "capability")},
                       "quality_status": "released"})
        raw = deepcopy(q.get("gt"))
        if "gt" not in q:
            raise ValueError(f"Missing export answer for {q['qid']}")
        if raw in ("INSUFFICIENT", "INSUFFICIENT_EVIDENCE"):
            answer, projection = "无此项/查无此记录", "abstention:never_known"
        elif isinstance(raw, dict) and "value" in raw:
            answer, projection = raw["value"], "gt/value"
        else:
            answer, projection = raw, "raw_gt"
        references.append({"qid": q["qid"], "answer": answer, "answer_raw": raw,
                           "answer_projection": projection, "quality_status": "released"})
    files = {"private/world.json": run.read("02_world.json"),
             "public/material.json": material, "public/questions.json": public,
             "references/questions.json": references}
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, value)
    (destination / "public/protocol.txt").write_text(protocol, encoding="utf-8")
    receipts = {}
    for name in [*files, "public/protocol.txt"]:
        data = (destination / name).read_bytes()
        receipts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"schema_version": "memory-bench-standard-light/v1",
                "benchmark_id": f"{run.run_id}-selected", "domain": run.scenario, "source_run": run.run_id,
                "seed_id": run.manifest.get("config", {}).get("seed_id", run.scenario),
                "question_selection": "generation_calibrated_subset",
                "default_evaluation_statuses": ["released"],
                "selection": selection,
                "counts": {"documents": len(material), "questions": len(public),
                           "by_quality_status": dict(Counter(q["quality_status"] for q in public))},
                "files": receipts}
    _atomic_write_json(destination / "manifest.json", manifest)
    inspect_benchmark(BenchmarkRef(destination, False, "as_provided"))
    return manifest
