"""Run the four configured native athletes, then grade their saved answers.

Generation remains unchanged. This adapter uses the same harness/plans/scorer as
standalone evaluation; rerunning it resumes answers and reuses bound judgments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from agent_harnesses.artifacts import inspect_benchmark
from agent_harnesses.config import CONFIG_ROOT, REPOSITORY_ROOT, ConfigurationError, load_experiment, load_registry, resolve_systems, _targets_for
from agent_harnesses.execution import command_for, execute_many, preflight_all
from agent_harnesses.judging import load_judge
from agent_harnesses.planning import fingerprint, make_plans
from agent_harnesses.scoring import read_results, score_run


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(settings: dict, name: str, default: int) -> int:
    value = settings.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"native evaluation {name} must be a positive integer")
    return value


def preflight_native_runtime(settings: dict, log=print) -> dict:
    """Check local CLI/profile/judge configuration before generation, with no API calls.

    The ordinary benchmark-dependent preflight still runs after generation.
    Native CLI --version and the existing judge loader only inspect local state;
    successful checks do not establish remote connectivity or model access.
    """
    from agent_harnesses.runners.native_cli import preflight_system

    targets = _targets_for(settings.get("targets"))
    if len(targets) != 4 or len({(t.system_id, t.model_id) for t in targets}) != 4:
        raise ConfigurationError("native_four requires four distinct system/model targets")
    model = settings.get("judge_model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("native_four requires an explicit judge_model")
    for key, default in (("max_judge_calls", 2000), ("workers", 4),
                         ("parallel_targets", 2), ("timeout_s", 600)):
        _positive(settings, key, default)
    systems = resolve_systems(SimpleNamespace(track="native", targets=targets, answering_model=None),
        load_registry(Path(settings.get("config_root") or CONFIG_ROOT)))
    reports = {}
    for target, system in zip(targets, systems):
        if system.runner != "native_cli" or system.role != "benchmark" or not system.executable:
            raise ConfigurationError(f"{target.target_id}: release requires an executable benchmark native_cli system")
        report = preflight_system(system, target.model_dict())
        report.setdefault("runtime_options", {})["timeout_s"] = settings.get("timeout_s", 600)
        reports[target.target_id] = report

    judge_report = {"model": model, "ok": False, "errors": []}
    try:
        judge = load_judge(REPOSITORY_ROOT)
        source = judge._module.config
        if not str(getattr(source, "API_KEY", "") or "").strip():
            judge_report["errors"].append("裁判缺少 OPENAI_API_KEY，请配置项目 .env 或环境变量")
        endpoint = str(getattr(source, "BASE_URL", "") or "").strip()
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                judge_report["errors"].append("裁判 OPENAI_BASE_URL 必须是未内嵌凭据的 http(s) URL")
            else:
                judge_report["endpoint_host"] = parsed.hostname
        else:
            judge_report["endpoint_host"] = "api.openai.com"
        if not callable(getattr(source, "chat", None)) or not callable(getattr(source, "chat_json", None)):
            judge_report["errors"].append("裁判调用接口不可用，请检查项目依赖安装")
        judge_report["ok"] = not judge_report["errors"]
    except (Exception, SystemExit) as exc:
        # Never include configuration values or an arbitrary import exception
        # message in diagnostics; those may contain credentials.
        judge_report["errors"].append(
            f"裁判配置读取失败 ({type(exc).__name__})，请检查项目 .env 的 OPENAI_API_KEY、MODEL 和 Python 依赖")
    failures = [f"{target}: {'; '.join(map(str, report.get('errors') or []))}"
                for target, report in reports.items() if not report.get("ok")]
    if not judge_report["ok"]:
        failures.append("judge: " + "; ".join(judge_report["errors"]))
    if failures:
        raise ConfigurationError("发布前环境检查未通过，尚未开始生成或模型调用：" + " | ".join(failures))
    for target, report in reports.items():
        for warning in report.get("warnings") or []:
            log(f"  ⚠ {target}: {warning}")
    log("  ✓ 发布前环境检查：四选手 CLI、接口配置与裁判配置可读取；远程可用性尚未调用验证")
    return {"ok": True, "mode": "native_four", "targets": reports, "judge": judge_report,
            "remote_access_verified": False}


def _experiment(benchmark: Path, directory: Path, settings: dict):
    targets = settings.get("targets")
    if not isinstance(targets, list) or len(targets) != 4:
        raise ValueError("native_four requires four explicit targets")
    required = ("id", "system", "model_id", "model_label", "endpoint_profile", "protocol_style")
    for target in targets:
        if not isinstance(target, dict) or any(not isinstance(target.get(k), str) or not target[k].strip() for k in required):
            raise ValueError("each native target requires " + ", ".join(required))
    if len({(t["system"], t["model_id"], t["endpoint_profile"], t["protocol_style"]) for t in targets}) != 4:
        raise ValueError("native_four requires four distinct athlete configurations")
    judge_model = settings.get("judge_model")
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("native_four requires an explicit judge_model")
    _positive(settings, "max_judge_calls", 2000)
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ['schema_version = 2', 'experiment_id = "generation-four-athletes"', 'track = "native"',
             f'output_root = {quote(directory / "runs")}', '', '[[benchmarks]]',
             f'path = {quote(benchmark)}', 'corpus_variant = "as_provided"', 'allow_unmet = false']
    for target in targets:
        lines += ['', '[[targets]]', *(f'{key} = {quote(target[key])}' for key in required)]
    lines += ['', '[protocol]', 'version = "native-four-release-v1"', 'network_mode = "closed"',
              'filler_policy = "as_provided"', 'hidden_asset_policy = "deny"',
              'scoring = "references.questions.answer"', '', '[execution]', 'mode = "full"', 'limit = 0',
              f'questions_in_parallel = {_positive(settings, "workers", 4)}',
              f'parallel_plans = {_positive(settings, "parallel_targets", 2)}',
              f'timeout_s = {_positive(settings, "timeout_s", 600)}']
    # Judging has its own binding below. Changing a judge must not invalidate
    # completed athlete answers by changing the native answer-run fingerprint.
    path = directory / "experiment.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_experiment(path)


class _Budget:
    """Reserve before each physical judge request; retain usage across resumes."""
    def __init__(self, path: Path, limit: int, *, shared: bool = False):
        self.path, self.limit, self.shared = Path(path), limit, shared
        if shared:
            if type(limit) is not int or limit < 1:
                raise ValueError("production judge cap must be a positive integer")
            with self._ledger_lock():
                self.used = self._shared_used()
                if not self.path.exists():
                    self._save_shared()
            return
        previous = _read(path, {})
        self.used = int(previous.get("reserved_calls", 0))
        if self.used < 0:
            raise ValueError("invalid native judge budget ledger")

    def reserve(self):
        if self.shared:
            with self._ledger_lock():
                self.used = self._shared_used()
                if self.used >= self.limit:
                    raise RuntimeError(f"production judge call budget exhausted ({self.used}/{self.limit})")
                self.used += 1
                self._save_shared()
            return
        if self.used >= self.limit:
            raise RuntimeError(f"native judge call budget exhausted ({self.used}/{self.limit})")
        self.used += 1
        _write(self.path, {"max_judge_calls": self.limit, "reserved_calls": self.used,
                           "policy": "reserved before physical call; failed calls remain counted"})

    def _shared_used(self):
        previous = _read(self.path, None)
        if previous is None:
            return 0
        if (not isinstance(previous, dict)
                or previous.get("version") != "production-judge-budget/v1"
                or type(previous.get("max_judge_calls")) is not int
                or previous["max_judge_calls"] != self.limit):
            raise ValueError("production judge cap/ledger binding changed; retain the frozen shared ledger")
        used = previous.get("reserved_calls")
        if type(used) is not int or not 0 <= used <= self.limit:
            raise ValueError("invalid production judge reservation count")
        return used

    def _save_shared(self):
        _write(self.path, {"version": "production-judge-budget/v1",
                          "max_judge_calls": self.limit, "reserved_calls": self.used,
                          "policy": "reserved before physical call; failed calls remain counted"})

    def _ledger_lock(self):
        """Serialize independent context budgets without storing any query payload."""
        from contextlib import contextmanager
        import os
        @contextmanager
        def locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as handle:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if os.name == "nt":
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return locked()


def _bounded_judge(judge, settings: dict, budget: _Budget):
    """Use a module-local config proxy: no shared config/model mutation."""
    source = judge._module.config
    model = settings["judge_model"]
    timeout = _positive(settings, "timeout_s", 600)
    from llm_transport import RUN_MODEL_CAPABILITIES, original_run_transport
    transport = original_run_transport(model, "low", timeout) if model in RUN_MODEL_CAPABILITIES else {
        "version": "chat-completions-transport/v1", "default_profile": "judge", "model_profiles": {},
        "profiles": {"judge": {"token_limit_parameter": "max_tokens", "deadline_seconds": timeout,
                               "http_timeout_seconds": timeout}},
    }

    def bounded_chat(*args, **kwargs):
        kwargs.update(model=model, transport=transport)
        budget.reserve()
        return source.chat(*args, **kwargs)

    # Preserve the existing JSON retry implementation and count each retry at
    # its actual chat dispatch, without monkey-patching global config.chat.
    original = source.chat_json
    chat_json = FunctionType(original.__code__, {**original.__globals__, "chat": bounded_chat},
                             original.__name__, original.__defaults__, original.__closure__)
    chat_json.__kwdefaults__ = original.__kwdefaults__

    class ConfigProxy:
        JUDGE_MODEL = model
        def __getattr__(self, key):
            return getattr(source, key)

    proxy = ConfigProxy()
    proxy.chat_json = chat_json
    judge._module.config = proxy
    judge.version = {**judge.version, "judge_model": model,
                     "scoring_adapter_sha256": _sha(REPOSITORY_ROOT / "agent_harnesses" / "scoring.py")}
    return judge


class _CachedJudge:
    """Cache successful judge decisions, including false, by exact input/version."""
    def __init__(self, judge, path: Path):
        self.judge, self.path, self.cache = judge, path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["value"]

    def __getattr__(self, key):
        return getattr(self.judge, key)

    def _call(self, method, q, pred, use_llm):
        key = fingerprint({"method": method, "question": q, "answer": pred, "use_llm": use_llm})
        if key not in self.cache:
            value = getattr(self.judge, method)(q, pred, use_llm=use_llm)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            self.cache[key] = value
        return self.cache[key]

    def judge_answer(self, q, pred, use_llm=True):
        return self._call("judge_answer", q, pred, use_llm)

    def judge_l2_partial(self, q, pred, use_llm=True):
        return self._call("judge_l2_partial", q, pred, use_llm)


def execute_native_evaluation(benchmark: Path, directory: Path, settings: dict, *,
                              shared_judge_budget: Path | None = None) -> dict[str, Path]:
    """Return target -> native run with judged.jsonl. Makes calls only after preflight.

    Native answer count is bounded by the benchmark size times four; each native
    episode has timeout_s. Native CLIs control their internal model/tool calls.
    max_judge_calls is the shared physical judge-call cap across all four runs.
    """
    if settings.get("backend", "native_four") != "native_four":
        raise ValueError("execute_native_evaluation requires backend=native_four")
    benchmark, directory = Path(benchmark).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    experiment = _experiment(benchmark, directory, settings)
    systems = resolve_systems(experiment, load_registry(Path(settings.get("config_root") or CONFIG_ROOT)))
    if any(s.runner != "native_cli" or s.role != "benchmark" for s in systems):
        raise ValueError("native_four targets must use benchmark native_cli systems")
    inspection = inspect_benchmark(experiment.benchmark)
    if inspection.n_questions < 1:
        raise ValueError("native_four requires a nonempty benchmark")
    plans = make_plans(experiment, systems, inspection)
    reports = preflight_all(experiment, systems, plans)
    _write(directory / "evaluation_plan.json", {"plans": [p.to_dict() for p in plans],
        "commands": [command_for(experiment, s, p) for s, p in zip(systems, plans)],
        "preflight": reports, "scoring": {"model": settings["judge_model"],
            "max_judge_calls": _positive(settings, "max_judge_calls", 2000)}})
    failures = [f"{key}: {'; '.join(map(str, r.get('errors') or []))}" for key, r in reports.items() if not r.get("ok")]
    if failures:
        raise ConfigurationError("native evaluation preflight failed: " + " | ".join(failures))
    budget = _Budget(shared_judge_budget if shared_judge_budget is not None else directory / "judge_budget.json",
                     _positive(settings, "max_judge_calls", 2000), shared=shared_judge_budget is not None)
    judge = _bounded_judge(load_judge(REPOSITORY_ROOT), settings, budget)
    results = execute_many(experiment, systems, plans, resume=True,
                           parallel_plans=_positive(settings, "parallel_targets", 2))
    _write(directory / "execution.json", results)
    failures = [str(r["error"]) for r in results if r.get("error")]
    if failures:
        raise RuntimeError("native evaluation failed; saved answers can be resumed: " + " | ".join(failures))
    if len(results) != len(plans):
        raise RuntimeError("native evaluation returned an incomplete target list")
    outputs = {}
    identity = fingerprint({k: v for k, v in judge.version.items() if k not in {"factory_commit", "factory_root"}})
    for plan, result in zip(plans, results):
        run_dir = Path(result["output_dir"])
        records = read_results(run_dir)
        if len(records) != inspection.n_questions or {r["question_index"] for r in records} != set(range(inspection.n_questions)):
            raise RuntimeError(f"{plan.target_id}: incomplete native answer coverage")
        binding = {"results_sha256": _sha(run_dir / "results.jsonl"), "judge_identity": identity}
        receipt = run_dir / "native_score_receipt.json"
        previous = _read(receipt, {})
        judged_path = run_dir / "judged.jsonl"
        if (previous.get("binding") != binding or not judged_path.exists()
                or previous.get("judged_sha256") != _sha(judged_path) or previous.get("judge_errors", 0)):
            cached = _CachedJudge(judge, run_dir / f"native_judge_cache_{identity[:16]}.jsonl")
            summary = score_run(run_dir, cached, use_llm=True, force=True)
            _write(receipt, {"binding": binding, "judged_sha256": _sha(judged_path),
                "judge_errors": summary["aggregate"]["overall"]["n_judge_error"]})
        outputs[plan.target_id] = run_dir
    _write(directory / "native_results.json", {key: str(path) for key, path in outputs.items()})
    return outputs
