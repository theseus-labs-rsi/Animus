"""TOML-backed system registry and per-run experiment configuration."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
SYSTEMS_PATH = CONFIG_ROOT / "systems.toml"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
PROFILE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
TRACKS = {"native", "memory"}
STATUSES = {"active", "candidate", "diagnostic", "disabled"}
ROLES = {"benchmark", "reference", "diagnostic"}
CORPUS_VARIANTS = {"full", "filtered", "as_provided"}
SECRET_FIELD_PARTS = ("api_key", "password", "secret", "token", "credential")


class ConfigurationError(ValueError):
    """Configuration is incomplete, contradictory, or not reproducible."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} 必须是 table")
    return value


def _read_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as fh:
            value = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"配置不存在: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"TOML 无法解析: {path}: {exc}") from exc
    return _mapping(value, str(path))


def _text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ConfigurationError(f"{field} 不能为空")
    return text


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _contains_tbd(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() == "TBD"
    if isinstance(value, Mapping):
        return any(_contains_tbd(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tbd(item) for item in value)
    return False


def _public_memory_config(value: Any, field: str) -> Mapping[str, Any] | None:
    """Parse target-scoped memory settings while keeping credentials out of TOML.

    Memory-system internal models and storage backends are part of the evaluated
    system identity, so their public settings belong in the experiment and enter
    the RunPlan fingerprint. Credentials remain in ``secrets.env`` and are
    referenced only through ``endpoint_profile``.
    """
    if value is None:
        return None
    config = dict(_mapping(value, field))

    def reject_secrets(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                name = str(key).lower()
                if any(part in name for part in SECRET_FIELD_PARTS):
                    raise ConfigurationError(
                        f"{path}.{key} 不得保存凭证；请改用 endpoint_profile 引用 secrets.env"
                    )
                reject_secrets(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                reject_secrets(nested, f"{path}[{index}]")

    reject_secrets(config, field)
    return config


@dataclass(frozen=True)
class BenchmarkRef:
    path: Path
    allow_unmet: bool
    corpus_variant: str


@dataclass(frozen=True)
class SystemConfig:
    schema_version: int
    system_id: str
    track: str
    display_name: str
    status: str
    role: str
    implementation: Mapping[str, Any]
    spec: Mapping[str, Any]
    source: Path

    @property
    def executable(self) -> bool:
        return bool(self.implementation.get("executable", False))

    @property
    def runner(self) -> str:
        return str(self.implementation.get("runner") or "planned")


def parse_system(raw: Mapping[str, Any], source: Path) -> SystemConfig:
    version = int(raw.get("schema_version", 0))
    if version != 2:
        raise ConfigurationError(f"{source}: system schema_version 只支持 2")
    system_id = _text(raw.get("id"), "system.id")
    if not ID_RE.fullmatch(system_id):
        raise ConfigurationError(f"{source}: 非法 system id: {system_id!r}")
    track = _text(raw.get("track"), f"{system_id}.track")
    if track not in TRACKS:
        raise ConfigurationError(f"{system_id}: track 必须是 native 或 memory")
    status = _text(raw.get("status"), f"{system_id}.status")
    if status not in STATUSES:
        raise ConfigurationError(f"{system_id}: 非法 status={status!r}")
    role = _text(raw.get("role"), f"{system_id}.role")
    if role not in ROLES:
        raise ConfigurationError(f"{system_id}: 非法 role={role!r}")
    implementation = _mapping(raw.get("implementation", {}), f"{system_id}.implementation")
    spec = _mapping(raw.get("spec"), f"{system_id}.spec")

    if track == "native":
        for field in ("harness", "native_tool_policy"):
            if field not in spec:
                raise ConfigurationError(f"{system_id}: native spec 缺少 {field}")
        tool_policy = _mapping(spec["native_tool_policy"], f"{system_id}.native_tool_policy")
        mode = _text(tool_policy.get("mode"), f"{system_id}.native_tool_policy.mode")
        if role == "benchmark" and mode != "native":
            raise ConfigurationError(
                f"{system_id}: 正式 Native Track 必须使用 mode=native；统一工具只能作为 diagnostic"
            )
    else:
        for field in ("memory_system", "ingestion", "retrieval"):
            if field not in spec:
                raise ConfigurationError(f"{system_id}: memory spec 缺少 {field}")
            _mapping(spec[field], f"{system_id}.{field}")

    if bool(implementation.get("executable")) and implementation.get("runner") in (None, "planned"):
        raise ConfigurationError(f"{system_id}: executable system 必须声明 runner")
    if status == "active":
        if not bool(implementation.get("executable")):
            raise ConfigurationError(f"{system_id}: active system 必须可执行")
        if _contains_tbd(spec) or _contains_tbd(implementation):
            raise ConfigurationError(f"{system_id}: active system 不得包含 TBD")
    return SystemConfig(
        schema_version=version,
        system_id=system_id,
        track=track,
        display_name=_text(raw.get("display_name"), f"{system_id}.display_name"),
        status=status,
        role=role,
        implementation=implementation,
        spec=spec,
        source=source,
    )


def load_registry(config_root: Path = CONFIG_ROOT) -> dict[str, SystemConfig]:
    path = Path(config_root) / "systems.toml"
    data = _read_toml(path)
    rows = data.get("systems")
    if not isinstance(rows, list):
        raise ConfigurationError(f"{path}: 顶层必须包含 [[systems]] 数组")
    registry: dict[str, SystemConfig] = {}
    for row in rows:
        system = parse_system(_mapping(row, str(path)), path)
        if system.system_id in registry:
            raise ConfigurationError(f"{path}: 重复 system id={system.system_id}")
        registry[system.system_id] = system
    if not registry:
        raise ConfigurationError(f"未发现 system 配置: {path}")
    return registry


@dataclass(frozen=True)
class ExperimentTarget:
    target_id: str
    system_id: str
    model_label: str | None
    model_id: str | None
    endpoint_profile: str | None
    protocol_style: str | None
    memory_config: Mapping[str, Any] | None

    def model_dict(self) -> dict[str, Any] | None:
        if self.model_id is None:
            return None
        return {
            "label": self.model_label or self.model_id,
            "model_id": self.model_id,
            "endpoint_profile": self.endpoint_profile,
            "protocol_style": self.protocol_style,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    track: str
    benchmarks: tuple[BenchmarkRef, ...]
    targets: tuple[ExperimentTarget, ...]
    answering_model: Mapping[str, Any] | None
    output_root: Path
    protocol: Mapping[str, Any]
    execution: Mapping[str, Any]
    judge: Mapping[str, Any]
    pricing: Mapping[str, Any]
    source: Path

    @property
    def benchmark(self) -> BenchmarkRef:
        return self.benchmarks[0]

    @property
    def system_ids(self) -> tuple[str, ...]:
        return tuple(target.system_id for target in self.targets)


def _repo_path(value: Any, field: str, root: Path = REPOSITORY_ROOT) -> Path:
    raw = _text(value, field)
    path = Path(raw)
    return path if path.is_absolute() else (Path(root) / path).resolve()


def _benchmarks_for(raw: Any) -> tuple[BenchmarkRef, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError("[[benchmarks]] 至少包含一个 benchmark")
    refs: list[BenchmarkRef] = []
    for index, item in enumerate(raw):
        location = f"benchmarks[{index}]"
        entry = _mapping(item, location)
        variant = _text(entry.get("corpus_variant"), f"{location}.corpus_variant")
        if variant not in CORPUS_VARIANTS:
            raise ConfigurationError(f"{location}.corpus_variant 非法: {variant!r}")
        refs.append(
            BenchmarkRef(
                path=_repo_path(entry.get("path"), f"{location}.path"),
                allow_unmet=bool(entry.get("allow_unmet", False)),
                corpus_variant=variant,
            )
        )
    return tuple(refs)


def _targets_for(raw: Any) -> tuple[ExperimentTarget, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError("[[targets]] 至少包含一个运行目标")
    targets: list[ExperimentTarget] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        location = f"targets[{index}]"
        entry = _mapping(item, location)
        target_id = _text(entry.get("id"), f"{location}.id")
        if not ID_RE.fullmatch(target_id):
            raise ConfigurationError(f"{location}: 非法 id={target_id!r}")
        if target_id in seen:
            raise ConfigurationError(f"重复 target id={target_id}")
        seen.add(target_id)
        endpoint_profile = _optional_text(entry.get("endpoint_profile"), f"{location}.endpoint_profile")
        if endpoint_profile and not PROFILE_RE.fullmatch(endpoint_profile):
            raise ConfigurationError(f"{location}: 非法 endpoint_profile={endpoint_profile!r}")
        targets.append(
            ExperimentTarget(
                target_id=target_id,
                system_id=_text(entry.get("system"), f"{location}.system"),
                model_label=_optional_text(entry.get("model_label"), f"{location}.model_label"),
                model_id=_optional_text(entry.get("model_id"), f"{location}.model_id"),
                endpoint_profile=endpoint_profile,
                protocol_style=_optional_text(entry.get("protocol_style"), f"{location}.protocol_style"),
                memory_config=_public_memory_config(
                    entry.get("memory_config"), f"{location}.memory_config"
                ),
            )
        )
    return tuple(targets)


def _execution_positive_int(execution: Mapping[str, Any], key: str) -> None:
    """并发度字段：可省略（=1，完全串行），写了就必须是正整数。

    TOML 里 `3.0` 和 `3` 是不同的类型；浮点一律拒绝而不是静默截断——`1.5` 被当成 1
    会让「我明明设了 1.5」变成难以发现的配置错误。bool 是 int 的子类，也要显式挡掉。
    """
    if key not in execution:
        return
    raw = execution[key]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigurationError(f"execution.{key} 必须是正整数，当前为 {raw!r}")


def _answering_model_for(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    model = dict(_mapping(value, "answering_model"))
    for field in ("model_id", "endpoint_profile"):
        _text(model.get(field), f"answering_model.{field}")
    profile = str(model["endpoint_profile"])
    if not PROFILE_RE.fullmatch(profile):
        raise ConfigurationError(f"answering_model.endpoint_profile 非法: {profile!r}")
    model.setdefault("label", model["model_id"])
    return model


def load_experiment(path: Path, config_root: Path = CONFIG_ROOT) -> ExperimentConfig:
    del config_root  # Kept for CLI/API compatibility; schema v2 has no secondary registry.
    path = Path(path).resolve()
    raw = _read_toml(path)
    version = int(raw.get("schema_version", 0))
    if version != 2:
        raise ConfigurationError(f"{path}: experiment schema_version 只支持 2")
    experiment_id = _text(raw.get("experiment_id"), "experiment_id")
    if not ID_RE.fullmatch(experiment_id):
        raise ConfigurationError(f"非法 experiment_id: {experiment_id!r}")
    track = _text(raw.get("track"), "track")
    if track not in TRACKS:
        raise ConfigurationError("experiment.track 必须是 native 或 memory")
    execution = _mapping(raw.get("execution"), "execution")
    mode = _text(execution.get("mode"), "execution.mode")
    if mode not in {"smoke", "full"}:
        raise ConfigurationError("execution.mode 必须是 smoke 或 full")
    limit = int(execution.get("limit", 0))
    if mode == "smoke" and limit <= 0:
        raise ConfigurationError("smoke 必须设置正数 execution.limit")
    # 并行度是运行条件，不是被测对象语义；只影响调度与写入顺序，不进 fingerprint
    # （见 planning.ARTIFACT_ONLY_EXECUTION_KEYS），因此同一 run 可以换并发度续跑。
    _execution_positive_int(execution, "questions_in_parallel")
    _execution_positive_int(execution, "parallel_plans")
    return ExperimentConfig(
        schema_version=version,
        experiment_id=experiment_id,
        track=track,
        benchmarks=_benchmarks_for(raw.get("benchmarks")),
        targets=_targets_for(raw.get("targets")),
        answering_model=_answering_model_for(raw.get("answering_model")),
        output_root=_repo_path(raw.get("output_root"), "output_root"),
        protocol=_mapping(raw.get("protocol"), "protocol"),
        execution=execution,
        judge=_mapping(raw.get("judge", {}), "judge"),
        pricing=_mapping(raw.get("pricing", {}), "pricing"),
        source=path,
    )


def resolve_systems(
    experiment: ExperimentConfig,
    registry: Mapping[str, SystemConfig],
) -> tuple[SystemConfig, ...]:
    resolved: list[SystemConfig] = []
    for target in experiment.targets:
        try:
            system = registry[target.system_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"target={target.target_id} 引用了未知 system={target.system_id}"
            ) from exc
        if system.track != experiment.track:
            raise ConfigurationError(
                f"{target.system_id}: system track={system.track} 与 experiment track={experiment.track} 不一致"
            )
        if system.status == "disabled":
            raise ConfigurationError(f"{target.system_id}: system 已 disabled")
        if system.runner == "native_cli":
            if not target.model_id or not target.endpoint_profile:
                raise ConfigurationError(
                    f"target={target.target_id}: native_cli 必须声明 model_id 和 endpoint_profile"
                )
        elif target.model_id or target.endpoint_profile:
            raise ConfigurationError(
                f"target={target.target_id}: runner={system.runner} 不接受 target 级模型配置"
            )
        if system.runner != "memory" and target.memory_config is not None:
            raise ConfigurationError(
                f"target={target.target_id}: runner={system.runner} 不接受 memory_config"
            )
        if (
            system.runner == "memory"
            and bool(system.implementation.get("requires_memory_config"))
            and not target.memory_config
        ):
            raise ConfigurationError(
                f"target={target.target_id}: {system.system_id} 必须声明 memory_config"
            )
        resolved.append(system)
    if experiment.track == "memory" and experiment.answering_model is None:
        raise ConfigurationError("Memory System Track 必须在 experiment 固定 answering_model")
    if experiment.track == "native" and experiment.answering_model is not None:
        raise ConfigurationError("Native Agent Track 不使用 experiment.answering_model")
    return tuple(resolved)
