"""Prepare or explicitly execute an independent, non-publishing semantic review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.semantic_review import prepare_review, review_questions


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new_output(path: Path, source_paths: list[Path]) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or any(path == source.resolve() for source in source_paths):
        raise ValueError("Output must be a new path; overwriting inputs/results is forbidden")
    runs = (ROOT / "output" / "runs").resolve()
    if path == runs or runs in path.parents or any(
        (parent / "manifest.json").exists() and
        ((parent / "00_input.json").exists() or (parent / "02_world.json").exists())
        for parent in path.parents
    ):
        raise ValueError("Review output must be outside historical run directories")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="Public protocol as JSON object/string, or UTF-8 text")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report outside historical runs")
    parser.add_argument("--reviewer-model", required=True)
    parser.add_argument("--reader-model")
    parser.add_argument("--include-titles", action="store_true", help="Only when titles are public to the solver; current adapter omits them")
    parser.add_argument("--execute", action="store_true", help="Explicitly permit model calls; default only prepares inputs")
    parser.add_argument("--max-calls", type=int, help="Required explicit call budget when --execute is used")
    parser.add_argument("--max-input-chars", type=int, default=200000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.execute and args.max_calls is None:
        parser.error("--execute requires an explicit --max-calls budget")
    if args.max_calls is not None and args.max_calls < 0:
        parser.error("--max-calls must be nonnegative")
    if args.max_input_chars < 1 or args.max_tokens < 1:
        parser.error("Input and output limits must be positive")
    try:
        output = _new_output(args.output, [args.questions, args.corpus, args.protocol])
        trace = output.with_name(output.name + ".calls.jsonl")
        attempts = output.with_name(output.name + ".attempts.jsonl")
        if trace.exists() or attempts.exists():
            raise ValueError("Review call trace already exists")
        questions, corpus = _load(args.questions), _load(args.corpus)
        protocol_text = args.protocol.read_text(encoding="utf-8")
        protocol = json.loads(protocol_text) if args.protocol.suffix.lower() == ".json" else protocol_text
        if isinstance(protocol, dict) and ("answer_protocol" in protocol or "public_protocol" in protocol):
            # Match the exact public text injected by the solver harness. Never
            # send the rest of an about artifact as extra instructions/evidence.
            from eval.provenance import load_public_protocol
            protocol = load_public_protocol(args.protocol)
        kwargs = {"reviewer_model": args.reviewer_model, "reader_model": args.reader_model,
                  "include_titles": args.include_titles}
        # Validate inputs before creating output files or importing config.
        prepared = prepare_review(questions, corpus, protocol, **kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not args.execute or args.max_calls == 0:
            prepared["mode"] = "prepared"
            prepared["planned_max_calls"] = args.max_calls
            report = prepared
        else:
            # Only explicit execution loads credentials/provider configuration.
            import config
            from llm_trace import redact, trace_scope
            secrets = config._trace_secrets()
            with trace.open("x", encoding="utf-8") as sink:
                def record(event):
                    sink.write(json.dumps(redact(event, secrets), ensure_ascii=False, allow_nan=False) + "\n")
                    sink.flush()

                def chat_json(step, messages, **params):
                    with trace_scope(attempts, step):
                        return config.chat_json(messages, **params)

                report = review_questions(questions, corpus, protocol, chat_json=chat_json,
                                          max_calls=args.max_calls, max_input_chars=args.max_input_chars,
                                          max_tokens=args.max_tokens, record=record, **kwargs)
                report = redact(report, secrets)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output": str(output), "mode": report["mode"],
                          "calls_used": report["calls_used"], "publication_effect": "none"}))
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
