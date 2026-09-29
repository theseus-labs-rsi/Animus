"""Bounded subprocess batches of the original factory smoke wrapper; dry by default."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = "original-bc-batch/v3"
MODEL = "gpt-5.4-mini"
MODELS = (MODEL, "glm-4.7-nothinking", "glm-5.3-flash")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = [p for folder in (ROOT / "pipeline", ROOT / "eval") for p in folder.rglob("*.py")]
    paths += [ROOT / name for name in ("config.py", "llm_transport.py", "llm_trace.py",
              "tools/run_original_bc_smoke.py", "tools/run_original_bc_batch.py")]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True, help="Fresh simple name, also used for original run names")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="User-supplied JSON paths or local seed names; relative paths resolve from the repository")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--source-run", type=Path,
                    help="Single-world production: reuse this stopped run's completed world construction checkpoint")
    quantities = ap.add_mutually_exclusive_group()
    quantities.add_argument("--question-budget", type=int)
    quantities.add_argument("--min-questions", type=int)
    ap.add_argument("--total-only", action="store_true")
    ap.add_argument("--per-line", action="append")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--max-world-entities", type=int, default=80)
    ap.add_argument("--time-span-weeks", type=int)
    ap.add_argument("--workers", type=int, choices=range(1, 22), default=1,
                    help="Maximum independent run concurrency")
    ap.add_argument("--initial-workers", type=int,
                    help="Start below the maximum; request_concurrency can adjust the live limit without changing frozen code/inputs")
    ap.add_argument("--keep-awake", action="store_true", help="While running on Windows, request system wakefulness; restore at exit")
    ap.add_argument("--settle-reported-usage", action="store_true")
    ap.add_argument("--target-mtokens", type=float, default=0.004)
    ap.add_argument("--max-calls-per-run", type=int, default=160)
    ap.add_argument("--max-cny-per-run", type=float, default=60)
    ap.add_argument("--max-total-cny", type=float, default=180)
    ap.add_argument("--model", choices=MODELS, default=MODEL)
    ap.add_argument("--min-output-tokens", type=int, default=0)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--call-timeout-seconds", type=float, default=150)
    ap.add_argument("--json-attempts", type=int, choices=range(1, 6), default=3)
    ap.add_argument("--disclosure-format-attempts", type=int, choices=range(2, 7), default=2)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "max"], default="low")
    ap.add_argument("--world-review-reasoning-effort", choices=["low", "medium"], default="medium")
    ap.add_argument("--corpus-review-reasoning-effort", choices=["low", "medium"], default="medium")
    ap.add_argument("--corpus-review-max-tokens", type=int, choices=[4096, 8192], default=8192)
    ap.add_argument("--world-semantic-review", action="store_true")
    ap.add_argument("--process-questions", action="store_true")
    ap.add_argument("--timeout-seconds", type=float, help="Optional wall-clock limit for each complete child run")
    ap.add_argument("--continue-on-failure", action="store_true",
                    help="Continue independent runs after a terminated failure; source drift always stops the batch")
    ap.add_argument("--execute", action="store_true", help="Without this flag only freeze/validate the zero-call plan")
    return ap


def prepare(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,90}", args.batch):
        raise ValueError("batch must be a simple ASCII identifier")
    if args.min_questions is None and args.question_budget is None:
        args.question_budget = 3
    for key in ("repeats", "max_calls_per_run", "max_rounds"):
        if getattr(args, key) < 1:
            raise ValueError(key + " must be positive")
    if any(v is not None and v < 1 for v in (args.min_questions, args.question_budget)):
        raise ValueError("Question quantity must be positive")
    if (args.total_only or args.per_line or args.time_span_weeks is not None) and args.min_questions is None:
        raise ValueError("Closed-loop settings require min-questions")
    if not 8 <= args.max_world_entities <= 80:
        raise ValueError("max-world-entities must be between 8 and 80")
    if args.initial_workers is not None and not 1 <= args.initial_workers <= args.workers:
        raise ValueError("initial-workers must be within 1..workers")
    if not 0 <= args.min_output_tokens <= args.max_output_tokens <= 128000 or args.max_output_tokens < 1:
        raise ValueError("Invalid output token limits")
    if not math.isfinite(args.call_timeout_seconds) or args.call_timeout_seconds <= 0:
        raise ValueError("call-timeout-seconds must be finite and positive")
    from llm_transport import original_run_transport
    transport = original_run_transport(args.model, args.reasoning_effort, args.call_timeout_seconds)
    if args.time_span_weeks is not None and not 6 <= args.time_span_weeks <= 26:
        raise ValueError("time-span-weeks must be between 6 and 26")
    for key in ("max_cny_per_run", "max_total_cny", "target_mtokens"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key != "target_mtokens" and value == 0):
            raise ValueError(key + " must be finite and within its allowed range")
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    count = len(args.seeds) * args.repeats
    source = getattr(args, "source_run", None)
    reuse_source = None
    if source:
        source = source.resolve()
        if count != 1 or args.min_questions is None:
            raise ValueError("World checkpoint reuse requires one seed/world and a quantity target")
        previous = read(source / "manifest.json")
        if previous.get("status") == "running" or not previous.get("stages", {}).get("whitepaper", {}).get("done"):
            raise ValueError("World source must be stopped with completed whitepaper")
        reuse_source = {"directory": str(source),
                        "files": {p.name: digest(p) for p in source.glob("*.json")}}
    reserved = Decimal(str(args.max_cny_per_run)) * count
    if reserved > Decimal(str(args.max_total_cny)):
        raise ValueError("Planned run caps exceed max-total-cny before any child is launched")
    directory = args.output_dir or ROOT / "output/original_bc_batches" / args.batch
    directory = (directory if directory.is_absolute() else ROOT / directory).resolve()
    seeds = []
    for index, name in enumerate(args.seeds, 1):
        path = (ROOT / "seeds" / (name + ".json")
                if Path(name).name == name and not Path(name).suffix else Path(name))
        path = (path if path.is_absolute() else ROOT / path).resolve()
        value = read(path)
        if not isinstance(value, dict) or not isinstance(value.get("seed_id"), str):
            raise ValueError("Each seed must be a JSON object with a seed_id")
        seeds.append({"index": index, "name": re.sub(r"[^A-Za-z0-9_-]", "_", path.stem),
                      "path": str(path), "sha256": digest(path), "seed_id": value["seed_id"],
                      "copy": str(directory / "inputs" / f"seed_{index:03d}.json")})
    if len({s["path"] for s in seeds}) != len(seeds):
        raise ValueError("Duplicate seed paths; use repeats for deliberate repetition")
    if source and read(source / "00_seed_pack.json").get("seed_id") != seeds[0]["seed_id"]:
        raise ValueError("Selected seed differs from the source world")
    runs = []
    # Cover every seed once before starting another repetition of any seed.
    for repeat in range(1, args.repeats + 1):
        for seed in seeds:
            name = f"{args.batch}_{seed['index']:02d}_{seed['name']}_r{repeat:03d}"
            run_dir = ROOT / "output/runs" / name
            command = [sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/run_original_bc_smoke.py"),
                "--run", name, "--to", "quality", "--seed-pack", seed["copy"], "--model", args.model,
                "--target-mtokens", str(args.target_mtokens),
                "--max-calls", str(args.max_calls_per_run), "--max-cny", str(args.max_cny_per_run),
                "--reasoning-effort", args.reasoning_effort,
                "--min-output-tokens", str(args.min_output_tokens),
                "--max-output-tokens", str(args.max_output_tokens),
                "--call-timeout-seconds", str(args.call_timeout_seconds),
                "--json-attempts", str(args.json_attempts),
                "--disclosure-format-attempts", str(args.disclosure_format_attempts)]
            if source:
                index = command.index("--seed-pack")
                command[index:index + 2] = ["--source-run", str(source), "--reuse-through", "whitepaper", "--reuse-world-checkpoint"]
            if args.min_questions is None:
                command += ["--question-budget", str(args.question_budget)]
            else:
                command += ["--min-questions", str(args.min_questions), "--max-rounds", str(args.max_rounds),
                            "--max-world-entities", str(args.max_world_entities)]
                if args.time_span_weeks is not None:
                    command += ["--time-span-weeks", str(args.time_span_weeks)]
                for value in args.per_line or []:
                    command += ["--per-line", value]
            for field in ("world_review_reasoning_effort", "corpus_review_reasoning_effort", "corpus_review_max_tokens"):
                if args.model == MODEL and getattr(args, field) is not None:
                    command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
            for field in ("world_semantic_review", "process_questions", "total_only", "settle_reported_usage"):
                if getattr(args, field):
                    command.append("--" + field.replace("_", "-"))
            runs.append({"run_id": name, "run_dir": str(run_dir), "seed_id": seed["seed_id"],
                         "repeat": repeat, "command": command})
    plan = {"version": VERSION, "batch": args.batch, "model": args.model, "seeds": seeds, "runs": runs,
            "transport": transport, "reasoning_effort": args.reasoning_effort,
            "request_lifecycle": "cancellable_async_http/v1", "json_retry_policy_version": "classified/v2",
            "min_output_tokens": args.min_output_tokens, "max_output_tokens": args.max_output_tokens,
            "call_timeout_seconds": args.call_timeout_seconds, "json_attempts": args.json_attempts,
            "disclosure_format_attempts": args.disclosure_format_attempts,
            "sources": source_hashes(), "timeout_seconds": args.timeout_seconds,
            "continue_on_failure": args.continue_on_failure, "question_budget": args.question_budget,
            "min_questions": args.min_questions, "total_only": args.total_only, "workers": args.workers,
            "initial_workers": args.initial_workers or args.workers,
            "keep_awake": args.keep_awake,
            "settle_reported_usage": args.settle_reported_usage,
            "run_order": "round_robin_by_seed",
            "max_calls_per_run": args.max_calls_per_run, "max_cny_per_run": args.max_cny_per_run,
            "max_total_cny": args.max_total_cny, "reserved_all_run_caps_cny": float(reserved),
            "budget_basis": "Sum of per-run caps. Each request reserves its maximum; optional valid provider usage settlement releases unused reservation. Estimates are not invoices.",
            "scope": "Original factory generation through quality; no solver/scoring/context experiment or automatic release claim."}
    if reuse_source:
        plan["reuse_source"] = reuse_source
    if directory.exists():
        if read(directory / "plan.json") != plan:
            raise ValueError("Prepared settings/inputs/source changed; use a fresh batch")
        validate(directory)
        return directory, plan
    if any(Path(run["run_dir"]).exists() for run in runs):
        raise ValueError("An original run directory already exists; no overwrite/resume allowed")
    directory.mkdir(parents=True)
    (directory / "inputs").mkdir()
    for seed in seeds:
        Path(seed["copy"]).write_bytes(Path(seed["path"]).read_bytes())
    write(directory / "plan.json", plan)
    write(directory / "prepared.json", {"created_utc": now(), "plan_sha256": digest(directory / "plan.json"),
                                      "provider_calls": 0})
    validate(directory)
    return directory, plan


def validate(directory):
    directory = Path(directory)
    plan = read(directory / "plan.json")
    if digest(directory / "plan.json") != read(directory / "prepared.json")["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    current = source_hashes()
    drift = [k for k in set(plan["sources"]) | set(current) if plan["sources"].get(k) != current.get(k)]
    if drift:
        raise ValueError("Source drift: " + ", ".join(sorted(drift)))
    for seed in plan["seeds"]:
        if digest(seed["path"]) != seed["sha256"] or digest(seed["copy"]) != seed["sha256"]:
            raise ValueError("Frozen seed changed: " + seed["seed_id"])
    for name, expected in plan.get("reuse_source", {}).get("files", {}).items():
        if digest(Path(plan["reuse_source"]["directory"]) / name) != expected:
            raise ValueError("World source artifact changed: " + name)
    return plan


def request_stop(directory, reason):
    """Request cancellation of this prepared batch; never dispatch or resume."""
    directory = Path(directory)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A human-readable stop reason is required")
    if not (directory / "plan.json").is_file():
        raise ValueError("Stop request needs an existing prepared batch")
    path = directory / "stop_request.json"
    if path.exists():
        return read(path)
    request = {"version": "original-bc-stop-request/v1", "requested_utc": now(),
               "reason": reason, "plan_sha256": digest(directory / "plan.json")}
    write(path, request)
    return request


def request_concurrency(directory, workers, reason):
    """Change only dispatch concurrency, within the frozen maximum and budget."""
    directory = Path(directory)
    plan = validate(directory)
    if type(workers) is not int or not 1 <= workers <= plan.get("workers", 1):
        raise ValueError("Requested workers exceed frozen concurrency bounds")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Concurrency change requires a reason")
    if (directory / "stop_request.json").exists():
        raise ValueError("Cannot adjust a stopped batch")
    report_path = directory / "execution/report.json"
    if report_path.exists() and read(report_path).get("status") != "running":
        raise ValueError("Cannot adjust a finished batch")
    request = {"workers": workers, "reason": reason, "requested_utc": now(),
               "plan_sha256": digest(directory / "plan.json")}
    write(directory / "concurrency_request.json", request)
    return request


def concurrency_limit(directory, plan):
    path = Path(directory) / "concurrency_request.json"
    if not path.exists():
        return plan.get("initial_workers", plan.get("workers", 1)), None
    request = read(path)
    workers = request.get("workers")
    if (type(workers) is not int or not 1 <= workers <= plan.get("workers", 1)
            or request.get("plan_sha256") != digest(Path(directory) / "plan.json")
            or not isinstance(request.get("reason"), str) or not request["reason"].strip()):
        raise ValueError("Invalid concurrency request; frozen plan/bounds must match")
    return workers, request


def _stop_request(path):
    if path is None or not Path(path).exists():
        return None
    try:
        request = read(path)
        if not isinstance(request, dict) or not isinstance(request.get("reason"), str) or not request["reason"].strip():
            raise ValueError("Stop request requires a reason")
        return request
    except (ValueError, OSError) as exc:
        # An unreadable stop file cannot become permission for another dispatch.
        return {"reason": "Stop request file exists but is unreadable", "read_error": str(exc)}


def terminate_tree(process):
    """Stop this spawned process tree, then reap the child before any next run."""
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=15, check=False)
        note = {"method": "taskkill_tree", "returncode": result.returncode}
    else:
        os.killpg(process.pid, signal.SIGKILL)
        note = {"method": "kill_process_group"}
    process.wait(timeout=15)
    return note


def launch(command, log_path, timeout, *, stop_request_path=None):
    started = time.monotonic()
    requested = _stop_request(stop_request_path)
    if requested is not None:
        return {"pid": None, "timed_out": False, "cancelled": True, "stop_request": requested,
                "child_terminated": True, "returncode": None, "elapsed_seconds": 0.0}
    with Path(log_path).open("xb") as log:
        options = {"cwd": str(ROOT), "stdout": log, "stderr": subprocess.STDOUT,
                   "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        child = subprocess.Popen(command, **options)
        result = {"pid": child.pid, "timed_out": False, "child_terminated": False}
        try:
            deadline = started + timeout if timeout is not None else None
            while child.poll() is None:
                requested = _stop_request(stop_request_path)
                if requested is not None:
                    result.update(cancelled=True, stop_request=requested)
                    result["termination"] = terminate_tree(child)
                    break
                remaining = deadline - time.monotonic() if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    result["timed_out"] = True
                    result["termination"] = terminate_tree(child)
                    break
                try:
                    child.wait(timeout=min(1.0, remaining) if remaining is not None else 1.0)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if child.poll() is None:
                terminate_tree(child)
            raise
        finally:
            result.update(returncode=child.poll(), child_terminated=child.poll() is not None,
                          elapsed_seconds=round(time.monotonic() - started, 3))
        return result


def summarize_run(directory):
    directory = Path(directory)
    errors = []
    def optional(name):
        path = directory / name
        if not path.exists():
            return None
        try:
            return read(path)
        except (ValueError, OSError) as exc:
            errors.append({"artifact": name, "error": str(exc)})
            return None
    def object_artifact(name):
        value = optional(name)
        if value is not None and not isinstance(value, dict):
            errors.append({"artifact": name, "error": "Expected JSON object"})
            return {}
        return value or {}
    manifest = object_artifact("manifest.json")
    profile = object_artifact("experiment_profile.json")
    quality = optional("07_release.json")
    if quality is not None and not isinstance(quality, dict):
        errors.append({"artifact": "07_release.json", "error": "Expected JSON object"})
    # Use the same receipt/independent-rejection policy as downstream consumers.
    # Merely recording an old automatic pass must never certify this output.
    import_root = str(Path(__file__).resolve().parents[1])
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    from pipeline.quality import quality_snapshot
    current_quality = quality_snapshot(directory)
    grounding = object_artifact("06_grounding_report.json")
    reserved = profile.get("reserved_upper_estimate_cny")
    if reserved is not None and (type(reserved) not in (int, float)
                                 or not math.isfinite(reserved) or reserved < 0):
        errors.append({"artifact": "experiment_profile.json", "error": "Invalid reserved_upper_estimate_cny"})
        reserved = None
    consumed = profile.get("budget_consumed_cny", reserved)
    if consumed is not None and (type(consumed) not in (int, float) or not math.isfinite(consumed) or consumed < 0):
        errors.append({"artifact": "experiment_profile.json", "error": "Invalid budget_consumed_cny"})
        consumed = None
    questions = optional("04_questions.json")
    kept = optional("06_grounded_questions.json")
    trace = {"requests": 0, "responses": 0, "call_errors": 0, "json_errors": 0,
             "reported_tokens": 0, "usage_missing_responses": 0}
    request_ids, response_ids, usage_ids = set(), set(), set()
    ids_complete = True
    first_call_error = None
    path = directory / "llm_attempts.jsonl"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    kind = event.get("event")
                    call_id = event.get("call_id")
                    if kind in {"request", "response"}:
                        if isinstance(call_id, str) and call_id:
                            (request_ids if kind == "request" else response_ids).add(call_id)
                        else:
                            ids_complete = False
                    key = {"request": "requests", "response": "responses",
                           "call_error": "call_errors", "json_error": "json_errors"}.get(kind)
                    if key:
                        trace[key] += 1
                    if kind == "call_error" and first_call_error is None:
                        first_call_error = {key: event.get(key) for key in ("step", "error_type", "error", "elapsed_ms")}
                        first_call_error.update({key: event[key] for key in (
                            "kind", "phase", "retryable", "actual_output_cap", "usage_status") if key in event})
                        if isinstance(first_call_error["error"], str):
                            first_call_error["error"] = first_call_error["error"][:1000]
                    if kind == "response":
                        usage = event.get("response", {}).get("usage") or {}
                        tokens = usage.get("total_tokens")
                        if type(tokens) is int and tokens >= 0:
                            trace["reported_tokens"] += tokens
                        if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                            if isinstance(call_id, str) and call_id:
                                usage_ids.add(call_id)
                        else:
                            trace["usage_missing_responses"] += 1
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            errors.append({"artifact": "llm_attempts.jsonl", "error": str(exc), "counts_partial": True})
    trace.update(call_id_accounting_complete=ids_complete,
        requests_without_response=len(request_ids - response_ids) if ids_complete else max(0, trace["requests"] - trace["responses"]),
        calls_without_valid_usage=len(request_ids - usage_ids) if ids_complete else
            max(0, trace["requests"] - trace["responses"]) + trace["usage_missing_responses"])
    return {"manifest_status": manifest.get("status"), "current_stage": manifest.get("current_stage"),
            "stages": manifest.get("stages", {}),
            "quality": {"artifact_present": quality is not None,
                        "status": current_quality["status"],
                        "eligible": current_quality["eligible"],
                        "issues": current_quality["issues"],
                        "authority": "Current pipeline.quality.quality_snapshot, including independent semantic rejection and artifact binding."},
            "recorded_release": {"status": quality.get("status") if isinstance(quality, dict) else None,
                                 "eligible": quality.get("eligible") if isinstance(quality, dict) else None},
            "yield": {"original_candidates": len(questions) if isinstance(questions, list) else None,
                      "kept": len(kept) if isinstance(kept, list) else None,
                      "grounding_overall": grounding.get("overall"), "drop": grounding.get("n_dropped"),
                      "pending": grounding.get("n_pending"), "execution_complete": grounding.get("execution_complete")},
            "execution_status": profile.get("execution_status"),
            "execution_error": {k: profile[k] for k in ("error_type", "error") if k in profile},
            "reserved_upper_estimate_cny": reserved,
            "budget_consumed_cny": consumed,
            "reported_usage_estimate_cny": profile.get("reported_usage_estimate_cny"),
            "unsettled_reservation_calls": profile.get("unsettled_reservation_calls"),
            "budget_unsettled_estimate_cny": (max(0, consumed - profile["reported_usage_estimate_cny"])
                if consumed is not None and type(profile.get("reported_usage_estimate_cny")) in (int, float)
                and math.isfinite(profile["reported_usage_estimate_cny"]) else None),
            "target_status": manifest.get("algo", {}).get("met_status"),
            "orders_by_line": manifest.get("algo", {}).get("orders_by_line"),
            "grounded_by_line": grounding.get("by_line", {}),
            "admitted_calls": profile.get("admitted_calls"), "source_drift": profile.get("source_drift"),
            "trace": trace, "first_call_error": first_call_error, "artifact_read_errors": errors}


def execute(directory, *, launcher=None):
    directory = Path(directory)
    plan = validate(directory)
    execution = directory / "execution"
    execution.mkdir()  # A completed/failed batch cannot be restarted.
    stop_path = directory / "stop_request.json"
    launcher = launcher or (lambda command, log, timeout: launch(command, log, timeout, stop_request_path=stop_path))
    started = time.monotonic()
    report = {"version": VERSION, "status": "running", "started_utc": now(),
              "plan_sha256": digest(directory / "plan.json"), "sources_start": plan["sources"],
              "planned_runs": len(plan["runs"]), "reserved_all_run_caps_cny": plan["reserved_all_run_caps_cny"],
              "concurrency_limit": plan.get("initial_workers", plan.get("workers", 1)),
              "concurrency_history": [], "peak_active_runs": 0,
              "runs": [{"run_id": r["run_id"], "run_dir": r["run_dir"], "status": "not_started"}
                       for r in plan["runs"]], "quality_semantics": "Child exit0 is not a semantic quality guarantee."}
    stopped = None
    prior_execution_state = None
    try:
        if plan.get("keep_awake") and os.name == "nt":
            import ctypes
            prior_execution_state = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            if not prior_execution_state:
                raise OSError("Windows refused the batch's temporary keep-awake request")
            report["keep_awake"] = {"active": True, "method": "SetThreadExecutionState", "screen_kept_on": False}
        write(execution / "report.json", report)
        next_index, pending = 0, {}
        fatal = None
        with ThreadPoolExecutor(max_workers=plan.get("workers", 1)) as pool:
            while pending or (next_index < len(plan["runs"]) and stopped is None):
                try:
                    validate(directory)
                    requested = _stop_request(stop_path)
                    if requested is not None and fatal is None:
                        report["stop_request"] = requested
                        stopped = "stop_requested"
                    if stopped is None:
                        limit, adjustment = concurrency_limit(directory, plan)
                        if adjustment is not None and (not report["concurrency_history"] or report["concurrency_history"][-1] != adjustment):
                            report["concurrency_history"].append(adjustment)
                        if limit != report["concurrency_limit"]:
                            report["concurrency_limit"] = limit
                            write(execution / "report.json", report)
                    while (stopped is None and next_index < len(plan["runs"])
                           and len(pending) < report["concurrency_limit"]):
                        index = next_index
                        run, row = plan["runs"][index], report["runs"][index]
                        if Path(run["run_dir"]).exists():
                            raise ValueError("Run directory exists before dispatch: " + run["run_id"])
                        row.update(status="running", started_utc=now(), log=f"{index + 1:03d}.log")
                        write(execution / "report.json", report)
                        print(json.dumps({"run": run["run_id"], "status": "starting", "index": index + 1,
                                          "total": len(plan["runs"])}, ensure_ascii=False), flush=True)
                        future = pool.submit(launcher, run["command"], execution / row["log"], plan["timeout_seconds"])
                        pending[future] = index
                        report["peak_active_runs"] = max(report["peak_active_runs"], len(pending))
                        next_index += 1
                except BaseException as exc:
                    fatal = fatal or exc
                    stopped = "orchestration_error"
                    request_stop(directory, "Orchestration failed; cancel active children: " + str(exc)[:400])
                if not pending:
                    break
                ready, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = pending.pop(future)
                    run, row = plan["runs"][index], report["runs"][index]
                    try:
                        row["process"] = future.result()
                        row["status"] = ("cancelled" if row["process"].get("cancelled") else
                            "completed" if row["process"]["returncode"] == 0
                            and not row["process"]["timed_out"] else "failed")
                        if not row["process"].get("child_terminated"):
                            raise RuntimeError("Child termination not confirmed; no next run")
                        if row["status"] == "cancelled" and fatal is None:
                            report["stop_request"] = row["process"]["stop_request"]
                            stopped = "stop_requested"
                        if row["status"] != "completed" and not plan["continue_on_failure"]:
                            stopped = stopped or "child_failed"
                    except BaseException as exc:
                        row.update(status="failed", process_error={"type": type(exc).__name__, "error": str(exc)})
                        fatal = fatal or exc
                        stopped = "orchestration_error"
                        request_stop(directory, "Child control failed; cancel active children: " + str(exc)[:400])
                    finally:
                        row["finished_utc"] = now()
                        try:
                            row["artifacts"] = summarize_run(run["run_dir"])
                            validate(directory)
                            if row["artifacts"]["artifact_read_errors"] or row["artifacts"].get("source_drift"):
                                row["status"] = "failed"
                                raise RuntimeError("Child artifact audit failed; no next run")
                        except BaseException as exc:
                            fatal = fatal or exc
                            stopped = "orchestration_error"
                            request_stop(directory, "Artifact audit failed; cancel active children: " + str(exc)[:400])
                        write(execution / "report.json", report)
        if fatal is not None:
            raise fatal
        report["status"] = ("cancelled" if stopped == "stop_requested" else
            "completed" if all(r["status"] == "completed" for r in report["runs"]) else "failed")
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        stopped = "orchestration_error"
    finally:
        if prior_execution_state:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(prior_execution_state)
            report["keep_awake"]["active"] = False
        try:
            current = source_hashes()
            report["sources_end"] = current
            report["source_drift"] = sorted(k for k in set(plan["sources"]) | set(current)
                if plan["sources"].get(k) != current.get(k))
            if report["source_drift"]:
                report["status"] = "failed"
        except Exception as exc:
            report.update(status="failed", source_check_error=str(exc))
        for row in report["runs"]:
            if row["status"] == "not_started":
                row["reason"] = stopped or "batch_did_not_dispatch"
        report.update(finished_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
        report["counts"] = {status: sum(r["status"] == status for r in report["runs"])
                            for status in ("completed", "failed", "not_started")}
        if report["status"] == "cancelled" or any(r["status"] == "cancelled" for r in report["runs"]):
            report["counts"]["cancelled"] = sum(r["status"] == "cancelled" for r in report["runs"])
        report["recorded_quality_eligible"] = sum(
            r.get("artifacts", {}).get("recorded_release", {}).get("eligible") is True for r in report["runs"])
        report["current_quality_eligible"] = sum(
            r.get("artifacts", {}).get("quality", {}).get("eligible") is True for r in report["runs"])
        report["quality_eligible_question_count"] = sum(
            r.get("artifacts", {}).get("yield", {}).get("kept") or 0 for r in report["runs"]
            if r.get("artifacts", {}).get("quality", {}).get("eligible") is True)
        report["trace_totals"] = {key: sum(r.get("artifacts", {}).get("trace", {}).get(key, 0)
                                          for r in report["runs"])
            for key in ("requests", "responses", "call_errors", "json_errors", "reported_tokens", "usage_missing_responses",
                        "requests_without_response", "calls_without_valid_usage")}
        report["trace_totals"]["counts_may_be_partial"] = any(
            r.get("artifacts", {}).get("artifact_read_errors") for r in report["runs"])
        report["requested_question_slots"] = (plan["question_budget"] * len(plan["runs"])
                                               if plan["question_budget"] is not None else None)
        report["minimum_qualified_question_target"] = ((plan.get("min_questions") or 0) * len(plan["runs"]))
        report["budget"] = {"max_total_cny": plan["max_total_cny"],
            "reserved_all_run_caps_cny": plan["reserved_all_run_caps_cny"],
            "reported_child_reservations_cny": sum(
                r.get("artifacts", {}).get("reserved_upper_estimate_cny") or 0 for r in report["runs"]),
            "reported_child_budget_consumed_cny": sum(
                r.get("artifacts", {}).get("budget_consumed_cny") or 0 for r in report["runs"]),
            "reported_child_usage_estimate_cny": sum(
                r.get("artifacts", {}).get("reported_usage_estimate_cny") or 0 for r in report["runs"]),
            "unsettled_child_reservation_estimate_cny": sum(
                r.get("artifacts", {}).get("budget_unsettled_estimate_cny") or 0 for r in report["runs"]),
            "missing_reservation_runs": [r["run_id"] for r in report["runs"]
                if r["status"] != "not_started" and r.get("artifacts", {}).get("reserved_upper_estimate_cny") is None],
            "not_an_invoice": True}
        write(execution / "report.json", report)
    return report


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        directory, plan = prepare(args)
        if not args.execute:
            print(json.dumps({"ready": True, "provider_calls": 0, "directory": str(directory),
                              "planned_runs": len(plan["runs"]), "plan_sha256": digest(directory / "plan.json"),
                              "reserved_all_run_caps_cny": plan["reserved_all_run_caps_cny"]}, ensure_ascii=False))
            return 0
        report = execute(directory)
        print(json.dumps({"status": report["status"], "counts": report["counts"],
                          "current_quality_eligible": report["current_quality_eligible"],
                          "quality_eligible_question_count": report["quality_eligible_question_count"],
                          "elapsed_seconds": report["elapsed_seconds"],
                          "report": str(directory / "execution/report.json")}, ensure_ascii=False))
        return 0 if report["status"] == "completed" else 1
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
