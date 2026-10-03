"""Generation-owned calibration and question selection, using existing evaluators.

The world/corpus and the stage-06 quality partition remain immutable. Selection
produces a derived standard package; it never changes semantic quality labels.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from eval.question_filter import filter_questions, load_results, validate_options
from eval.provenance import load_public_protocol, load_visible_corpus, make_evaluation_context
from pipeline.run import ROOT

CALIBRATION_ARTIFACT = "08_calibration.json"
SELECTION_ARTIFACT = "09_selection.json"
SELECTED_ARTIFACT = "09_selected_questions.json"
INPUTS = ("02_world.json", "00_about.json", "05_corpus.json", "06_grounded_questions.json",
          "06_semantic_review.json", "07_release.json")


def load_config(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("calibration config must be a JSON object")
    if value.get("backend") == "native_four":
        return _native_config(value, Path(path).resolve().parent)
    allowed = {"systems", "answering_model", "judge_model", "workers", "max_judge_calls",
               "timeout_s", "keep_easy_ratio", "seed"}
    if set(value) - allowed:
        raise ValueError(f"Unknown calibration settings: {sorted(set(value) - allowed)}")
    value = {"workers": 4, "timeout_s": 7200, "keep_easy_ratio": 0.2, "seed": 0, **value}
    if not isinstance(value.get("systems"), list):
        raise ValueError("calibration systems must be an explicit list")
    if type(value["keep_easy_ratio"]) not in (int, float):
        raise ValueError("calibration keep_easy_ratio must be a number between 0 and 1")
    value["systems"] = [name.strip() if isinstance(name, str) else name for name in value["systems"]]
    validate_options(value["systems"], value["keep_easy_ratio"], value["seed"])
    # Match the evaluator's result keys for its built-in A/B/C aliases.
    value["systems"] = [name.upper() if name.upper() in {"A", "B", "C"} else name
                        for name in value["systems"]]
    # No default model or silent fallbacks: enabling this stage explicitly
    # declares both paid roles. Credentials continue to come from the environment.
    for key in ("answering_model", "judge_model"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"calibration requires an explicit {key}")
        value[key] = value[key].strip()
    for key in ("workers", "timeout_s", "max_judge_calls"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"calibration {key} must be a positive integer")
    if value["workers"] > 16:
        raise ValueError("calibration workers must be at most 16")
    from eval.memory_systems import canonical_system_name
    identities = [canonical_system_name(name) for name in value["systems"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Calibration systems must have distinct implementations, not aliases")
    return value


def _native_config(value: dict, base: Path) -> dict:
    allowed = {"backend", "targets", "judge_model", "max_judge_calls", "workers",
               "parallel_targets", "timeout_s", "keep_easy_ratio", "seed", "config_root",
               "source_benchmark", "result_dirs"}
    if set(value) - allowed:
        raise ValueError(f"Unknown native calibration settings: {sorted(set(value) - allowed)}")
    value = {"workers": 4, "parallel_targets": 2, "timeout_s": 600,
             "keep_easy_ratio": 0, "seed": 0, **value}
    targets = value.get("targets")
    if not isinstance(targets, list) or len(targets) != 4:
        raise ValueError("native_four requires exactly four explicit athletes")
    required = {"id", "system", "model_id", "model_label", "endpoint_profile", "protocol_style"}
    for target in targets:
        if (not isinstance(target, dict) or set(target) != required
                or any(not isinstance(target[k], str) or not target[k].strip() for k in required)):
            raise ValueError(f"Each athlete requires exactly {sorted(required)}")
        if target["system"] not in {"native.codex", "native.dsh"}:
            raise ValueError("Release athletes must use the existing native.codex/native.dsh runners")
    value["systems"] = [t["id"] for t in targets]
    if len({(t["system"], t["model_id"]) for t in targets}) != 4:
        raise ValueError("Four athletes must have distinct system/model pairs")
    if type(value["keep_easy_ratio"]) not in (int, float):
        raise ValueError("keep_easy_ratio must be numeric")
    validate_options(value["systems"], value["keep_easy_ratio"], value["seed"])
    if value["keep_easy_ratio"] != 0:
        raise ValueError("native_four release removes every all-correct question; keep_easy_ratio must be 0")
    if not isinstance(value.get("judge_model"), str) or not value["judge_model"].strip():
        raise ValueError("Specify judge_model explicitly")
    for key in ("workers", "parallel_targets", "timeout_s", "max_judge_calls"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["workers"] > 16 or value["parallel_targets"] > 4:
        raise ValueError("workers must be <=16 and parallel_targets <=4")
    for key in ("config_root", "source_benchmark"):
        if key in value:
            value[key] = str((base / value[key]).resolve())
    if "result_dirs" in value:
        if "source_benchmark" not in value or not isinstance(value["result_dirs"], dict):
            raise ValueError("Import requires source_benchmark and result_dirs")
        if set(value["result_dirs"]) - set(value["systems"]):
            raise ValueError("result_dirs contains an unknown athlete")
        value["result_dirs"] = {key: str((base / folder).resolve()) for key, folder in value["result_dirs"].items()}
    elif "source_benchmark" in value:
        raise ValueError("source_benchmark requires result_dirs; imported runs never launch new answers")
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def preflight_release(settings: dict, log=print) -> dict:
    """Read-only local checks for the release tail before costly generation.

    Imported results are validated against the actual generated inputs in the
    existing importer. They require neither local athlete CLIs nor judge keys.
    """
    if settings.get("backend") != "native_four":
        return {"ok": True, "mode": "legacy"}
    if "result_dirs" in settings:
        log("  ⓘ 发布评测使用已有结果，跳过本地选手及裁判运行环境检查")
        return {"ok": True, "mode": "import", "remote_access_verified": False}
    from pipeline.native_evaluation import preflight_native_runtime
    return preflight_native_runtime(settings, log=log)


def production_calibration_support(settings: dict, control: dict) -> dict:
    """Validate the shared judge-cap interface before production pays for inputs."""
    if not isinstance(control, dict) or control.get("version") not in ("supply-driven/v3", "supply-driven/v4", "supply-driven/v5", "supply-driven/v6"):
        raise ValueError("production_calibration_invalid_control: expected supply-driven/v3 through v6")
    if settings.get("backend") != "native_four":
        raise ValueError("production_calibration_unsupported_backend: the legacy memory_systems "
                         "subprocess has no shared cross-round judge ledger; use native_four")
    limit = settings.get("max_judge_calls")
    if type(limit) is not int or limit < 1:
        raise ValueError("production_calibration_invalid_judge_cap: max_judge_calls must be positive")
    imported = "result_dirs" in settings
    if not imported:
        targets = settings.get("targets")
        if (not isinstance(targets, list) or len(targets) != 4
                or any(not isinstance(t, dict) or t.get("system") not in
                       {"native.codex", "native.dsh"} for t in targets)):
            raise ValueError("production_calibration_unsupported_athletes: require four "
                             "native.codex/native.dsh targets")
    return {"supported": True, "backend": "native_four", "mode": "import" if imported else "execute",
            "judge_budget_scope": "no_new_calls" if imported else "all_production_rounds",
            "max_judge_calls": limit, "athlete_internal_calls_bounded": False,
            "athlete_total_cost_bounded": False,
            "diagnostic": "Imported results launch no athletes or judge" if imported else
                "Native CLI episodes retain their timeout; athlete internal model/tool calls "
                "and total fees require a separate global budget"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_identity(run) -> str:
    settings = run.manifest["config"]["calibration"]
    settings = {key: value for key, value in settings.items()
                if key not in {"keep_easy_ratio", "seed", "workers", "timeout_s"}}
    sources = [ROOT / "eval" / name for name in (
        "multi_system.py", "semantic_judge.py", "grading.py", "baseline_r1.py", "memory_interface.py")]
    sources += list((ROOT / "eval/memory_systems").glob("*.py"))
    external = {}
    if settings.get("backend") == "native_four":
        sources = [ROOT / name for name in (
            "pipeline/native_evaluation.py", "pipeline/native_results.py", "pipeline/benchmark_export.py",
            "agent_harnesses/scoring.py", "agent_harnesses/judging.py", "eval/judge.py", "eval/grading.py")]
        sources += list((ROOT / "agent_harnesses/runners").glob("*.py"))
        if "result_dirs" in settings:
            for target, folder in settings["result_dirs"].items():
                for name in ("run_plan.json", "results.jsonl", "judged.jsonl"):
                    file = Path(folder) / name
                    external[f"{target}/{name}"] = _file_hash(file) if file.is_file() else None
            from agent_harnesses.artifacts import STANDARD_FILES
            for name in STANDARD_FILES:
                file = Path(settings["source_benchmark"]) / name
                external[name] = _file_hash(file) if file.is_file() else None
    return _digest({"version": 2, "settings": settings,
                    "inputs": _evaluation_inputs(run),
                    "implementation": {str(p.relative_to(ROOT)): _file_hash(p) for p in sources},
                    "imported_sources": external})

def _evaluation_inputs(run):
    """Identity of consumed inputs, independent of paths and audit receipts."""
    settings = run.manifest["config"]["calibration"]
    if settings.get("backend") != "native_four":
        # Legacy graders can consume additional semantic-review data.
        return {name: _file_hash(run.dir / name) if (run.dir / name).is_file() else None
                for name in INPUTS}
    from pipeline.benchmark_export import public_view
    material, protocol = public_view(run)
    return {"material": material, "protocol": protocol,
            "questions": [{key: q.get(key) for key in
                           ("qid", "question", "line", "capability", "gt")}
                          for q in run.read("06_grounded_questions.json")]}


def calibration_is_current(run) -> bool:
    if not run.has(CALIBRATION_ARTIFACT):
        return False
    report = run.read(CALIBRATION_ARTIFACT)
    if report.get("identity") != calibration_identity(run):
        return False
    result = report.get("results")
    return (report.get("status") == "empty" if not result else
            (run.dir / result).is_file() and _file_hash(run.dir / result) == report.get("results_sha256"))


def _context(run):
    if run.manifest["config"]["calibration"].get("backend") == "native_four":
        from pipeline.benchmark_export import public_view
        _, protocol = public_view(run)
    else:
        protocol = load_public_protocol(run.dir / "00_about.json")
    return make_evaluation_context(load_visible_corpus(run.dir / "05_corpus.json"), protocol)


def calibration_can_skip(run) -> bool:
    """Fresh partial results remain readable, but native resumes must fill gaps."""
    return (calibration_is_current(run) and
            (run.manifest["config"]["calibration"].get("backend") != "native_four"
             or run.read(CALIBRATION_ARTIFACT).get("status") != "partial"))


def _select(run, result_path: Path):
    settings = run.manifest["config"]["calibration"]
    results = load_results(aggregate=result_path, systems=settings["systems"])
    return filter_questions(run.read("06_grounded_questions.json"), results,
        keep_easy_ratio=settings["keep_easy_ratio"], seed=settings["seed"],
        expected_context=_context(run), stratify_by=("line",))


def execute_calibration(run, directory: Path, settings: dict) -> Path:
    """Reuse the existing solver, semantic judge and per-answer resume caches."""
    control = run.manifest.get("config", {}).get("production_control")
    production = production_calibration_support(settings, control) if control is not None else None
    directory.mkdir(parents=True, exist_ok=True)
    if settings.get("backend") == "native_four":
        from pipeline.native_results import import_native_results
        if "result_dirs" in settings:
            benchmark = Path(settings["source_benchmark"])
            results = {target: Path(folder) for target, folder in settings["result_dirs"].items()}
        else:
            from pipeline.benchmark_export import export_selected
            from pipeline.native_evaluation import execute_native_evaluation
            from eval.grading import requires_semantic_grading
            if any(requires_semantic_grading(q) for q in run.read("06_grounded_questions.json")):
                raise ValueError("Native four-athlete scoring does not yet support L3_process_trace; "
                                 "use the supported generation configuration without process proposals")
            benchmark = directory / "evaluation_input"
            if not benchmark.exists():
                holder = Path(tempfile.mkdtemp(prefix="input-", dir=directory))
                prepared = holder / "benchmark"
                export_selected(run, run.read("06_grounded_questions.json"), prepared,
                                {"stage": "before_difficulty_selection"})
                prepared.rename(benchmark)
                holder.rmdir()
            if production is None:
                results = execute_native_evaluation(benchmark, directory / "evaluation", settings)
            else:
                results = execute_native_evaluation(benchmark, directory / "evaluation", settings,
                    shared_judge_budget=run.dir / "production" / "judge_budget.json")
        return import_native_results(run, benchmark, results, directory, settings)
    command = [sys.executable, "-B", "-X", "utf8", "-m", "eval.multi_system",
               "--bench", str(run.dir / "06_grounded_questions.json"),
               "--corpus", str(run.dir / "05_corpus.json"),
               "--about", str(run.dir / "00_about.json"),
               "--systems", ",".join(settings["systems"]), "--judge-mode", "semantic",
               "--semantic-review", str(run.dir / "06_semantic_review.json"),
               "--judge-model", settings["judge_model"],
               "--max-judge-calls", str(settings["max_judge_calls"]),
               "--workers", str(settings["workers"]), "--output-dir", str(directory)]
    env = {**os.environ, "MODEL": settings["answering_model"], "JUDGE_MODEL": settings["judge_model"]}
    with (directory / "execution.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=settings["timeout_s"], check=False)
    if completed.returncode:
        raise RuntimeError(f"Calibration stopped; saved answers remain reusable. See {directory / 'execution.log'}")
    return directory / "results.json"


def stage_calibration(run):
    control = run.manifest.get("config", {}).get("production_control")
    if control is not None:
        production_calibration_support(run.manifest["config"].get("calibration", {}), control)
    identity = calibration_identity(run)
    questions = run.read("06_grounded_questions.json")
    if not questions:
        run.write(CALIBRATION_ARTIFACT, {"identity": identity, "status": "empty", "results": None})
        return
    from pipeline.quality import require_release
    require_release(run.dir / "06_grounded_questions.json")
    settings = run.manifest["config"]["calibration"]
    directory = run.dir / "calibration" / identity[:16]
    if settings.get("backend") == "native_four":
        # Keep the answer cache stable when only the judge implementation changes.
        # The native runner validates its own input/model/runtime fingerprints.
        wp = run.read("01_whitepaper.json") if run.has("01_whitepaper.json") else {}
        if (wp.get("generation_contract") or {}).get("version") == "generation-first/v2":
            answer_inputs = _evaluation_inputs(run)
        else:
            # Keep historical native answer directories reachable. Their runner
            # still validates the public bundle, model and runtime before reuse.
            answer_inputs = {name: _file_hash(run.dir / name) for name in INPUTS
                             if (run.dir / name).is_file()}
        answer_identity = _digest({"targets": settings["targets"], "inputs": answer_inputs})
        directory = run.dir / "calibration" / ("native-" + answer_identity[:16])
    result_path = execute_calibration(run, directory, settings)
    if calibration_identity(run) != identity:
        raise RuntimeError("Generation inputs changed during calibration; saved results were retained")
    _, report = _select(run, result_path)
    incomplete = report["counts"]["kept_incomplete"]
    status = "partial" if incomplete else "complete"
    run.write(CALIBRATION_ARTIFACT, {"version": 1, "identity": identity, "status": status,
        "systems": settings["systems"], "backend": settings.get("backend", "memory_systems"),
        "answering_model": settings.get("answering_model"), "targets": settings.get("targets"),
        "judge_model": settings["judge_model"], "questions": len(questions),
        "incomplete_questions": incomplete, "results": str(result_path.relative_to(run.dir)),
        "results_sha256": _file_hash(result_path)})
    run.set_algo(calibration={"status": status, "questions": len(questions), "incomplete": incomplete})
    run.log(f"  校准试答完成: {len(questions)} 题 × {len(settings['systems'])} 系统，{incomplete} 题保留待核查")


def selection_identity(run) -> str:
    settings = run.manifest["config"]["calibration"]
    return _digest({"calibration": run.read(CALIBRATION_ARTIFACT),
                    "ratio": settings["keep_easy_ratio"], "seed": settings["seed"],
                    "strata": "world/line", "implementation": {
                        name: _file_hash(ROOT / name) for name in (
                            "eval/question_filter.py", "pipeline/benchmark_export.py", "pipeline/calibration.py")}})


def selection_is_current(run) -> bool:
    if not calibration_is_current(run):
        return False
    if not run.has(SELECTION_ARTIFACT) or not run.has(SELECTED_ARTIFACT):
        return False
    report = run.read(SELECTION_ARTIFACT)
    if report.get("identity") != selection_identity(run):
        return False
    if _file_hash(run.dir / SELECTED_ARTIFACT) != report.get("selected_sha256"):
        return False
    if not report.get("benchmark"):
        if (run.manifest.get("config", {}).get("production_control")
                and report.get("status") == "complete"
                and report.get("release_ready") is False
                and run.read(CALIBRATION_ARTIFACT).get("status") == "complete"
                and report.get("delivery_subset")):
            return True
        return report.get("status") in {"empty", "partial"}
    from agent_harnesses.artifacts import inspect_benchmark
    from agent_harnesses.config import BenchmarkRef
    try:
        inspect_benchmark(BenchmarkRef(run.dir / report["benchmark"], False, "as_provided"))
        return True
    except (OSError, ValueError):
        return False


def stage_selection(run):
    if not calibration_is_current(run):
        raise ValueError("Run calibration before selecting questions")
    calibration = run.read(CALIBRATION_ARTIFACT)
    identity = selection_identity(run)
    if calibration["status"] == "empty":
        selected, report = [], {"counts": {"input": 0, "kept": 0, "all_correct": 0,
                                           "removed_easy": 0, "kept_incomplete": 0}, "items": []}
    else:
        selected, report = _select(run, run.dir / calibration["results"])
    delivery_subset = None
    if (run.manifest.get("config", {}).get("delivery_target")
            and calibration["status"] == "complete"
            and run.manifest["config"]["calibration"].get("backend") == "native_four"):
        from pipeline.supply import balanced_selection_subset
        selected, delivery_subset = balanced_selection_subset(
            selected, report, run.manifest["config"]["delivery_target"],
            strict_allocation=bool(run.manifest["config"].get("production_control")))
    run.write(SELECTED_ARTIFACT, selected)
    destination = None
    complete = calibration["status"] == "complete"
    native = run.manifest["config"]["calibration"].get("backend") == "native_four"
    balanced = (not run.manifest["config"].get("production_control") or
                bool(delivery_subset and all(delivery_subset["selected_by_line"].get(line, 0) >= amount
                     for line, amount in delivery_subset["ideal_allocation"].items())))
    if selected and (complete or not native) and balanced:
        from pipeline.benchmark_export import export_selected
        delivery = run.dir / "delivery"
        delivery.mkdir(exist_ok=True)
        # A new attempt owns a new destination; even an interrupted export never
        # removes or overwrites the previous benchmark or any generation inputs.
        holder = Path(tempfile.mkdtemp(prefix=identity[:12] + "-", dir=delivery))
        destination = holder / "benchmark"
        export_selected(run, selected, destination, {
            "calibration_status": calibration["status"],
            "systems": calibration.get("systems", []), "counts": report["counts"],
            "keep_easy_ratio": run.manifest["config"]["calibration"]["keep_easy_ratio"],
            "seed": run.manifest["config"]["calibration"]["seed"], "stratify_by": ["world", "line"]})
    report.update(version=1, identity=identity, status=(calibration["status"] if selected else "empty"),
                  benchmark=str(destination.relative_to(run.dir)) if destination else None,
                  selected_sha256=_file_hash(run.dir / SELECTED_ARTIFACT),
                  source_questions="06_grounded_questions.json", all_candidates="04_questions.json")
    if delivery_subset is not None:
        report["delivery_subset"] = delivery_subset
        report["exported_question_count"] = len(selected) if destination else 0
    if native:
        report["release_ready"] = bool(complete and selected and balanced)
        report["rule"] = "all_four_athletes_correct"
        report["note"] = ("All four scores complete; per-line delivery shortfall requires capacity refill" if complete and selected and not balanced else
                          "All four scores complete; release subset exported" if complete and selected else
                          "Every candidate was removed as easy" if complete else
                          "Saved selected candidates; resume missing athlete scores before final export")
    run.write(SELECTION_ARTIFACT, report)
    if run.manifest.get("config", {}).get("delivery_target"):
        from pipeline.supply import write_delivery_report
        write_delivery_report(run)
    run.set_algo(selection={"status": report["status"], **report["counts"]}, benchmark=report["benchmark"])
    run.log(f"  成品筛选: 原 {report['counts']['input']} 题，剔除简单题 {report['counts']['removed_easy']}，保留 {len(selected)}")
