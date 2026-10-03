"""Claude Code、Codex 与 DeepSeek Harness 的小规模原生工具 smoke runner。

真实密钥只从被 gitignore 的 configs/env/secrets.env 读取；模型名由 experiment 解析后
随 run_plan 传入；run plan 和结果都不保存密钥。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .. import boundaries, events
from ..artifacts import (
    load_benchmark_protocol,
    load_benchmark_questions,
    load_benchmark_sessions,
    question_id as benchmark_question_id,
)
from ..config import CONFIG_ROOT, REPOSITORY_ROOT, ConfigurationError, SystemConfig


NATIVE_CLI_ADAPTER_VERSION = "native-cli-smoke-v11"
SECRETS_ENV_FILE = REPOSITORY_ROOT / "configs" / "env" / "secrets.env"
# dsh 的 profile 是仓库钉住的配置（不是 Python 包数据）：它声明该 harness 启动哪些
# dsh bundle，跟着根 package.json 的 @deepseek-ai/dsh pin 解析。每次 run 复制进
# $DSH_HOME/profiles，`dsh --profile <name>` 从那里启动。
DSH_PROFILE_NAME = "headless"
DSH_PROFILES_DIR = CONFIG_ROOT / "dsh" / "profiles"


def dsh_profile_path() -> Path:
    return DSH_PROFILES_DIR / DSH_PROFILE_NAME
ADAPTER_ENV = {
    # 每个 adapter 的密钥/base_url/模型变量名；模型值来自 plan["model"]，不来自 env 文件。
    "claude": {
        "key": "ANTHROPIC_API_KEY",
        "base_url": "ANTHROPIC_BASE_URL",
        "model": "CLAUDE_MODEL",
        "binary": "CLAUDE_BIN",
        "binary_default": "claude",
    },
    "codex": {
        "key": "GPT_API_KEY",
        "base_url": "GPT_BASE_URL",
        "model": "GPT_MODEL",
        "binary": "CODEX_BIN",
        "binary_default": "codex",
    },
    "dsh": {
        "key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
        "model": "DEEPSEEK_MODEL",
        "binary": "DSH_BIN",
        "binary_default": "dsh",
    },
}


def _profile_names(profile: str) -> tuple[str, str]:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", profile).strip("_").upper()
    if not prefix:
        raise ConfigurationError("endpoint_profile 不能为空")
    return f"{prefix}_API_KEY", f"{prefix}_BASE_URL"


def resolve_env_file(system: SystemConfig) -> Path:
    """所有 adapter 共用一份 secrets.env；仍允许 system 显式覆盖以便测试。"""
    raw = str(system.implementation.get("env_file") or "").strip()
    if not raw:
        return SECRETS_ENV_FILE
    path = Path(raw)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ConfigurationError(
            f"本地配置不存在: {path}；请从同名 .env.example 复制后填写"
        )
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"{path}:{line_number}: 需要 KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ConfigurationError(f"{path}:{line_number}: 非法变量名 {key!r}")
        values[key] = value.strip().strip("'").strip('"')
    return values


def _binary_path(value: str, env: Mapping[str, str]) -> str | None:
    if "/" in value:
        path = Path(value).expanduser()
        return str(path.resolve()) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value, path=env.get("PATH"))


def _run_cli_process(command, *, timeout, cwd=None, env=None):
    """Bound the CLI process tree and read UTF-8 without pipe-reader threads.

    npm launchers create children. Killing just the launcher leaves those
    children running and can leave communicate() waiting for inherited pipes.
    Reuse the batch runner's platform tree termination and capture to files so
    collecting partial output never waits for a descendant to close a pipe.
    """
    from tools.run_original_bc_batch import terminate_tree

    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr, **options)
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_tree(child)
        except BaseException:
            if child.poll() is None:
                terminate_tree(child)
            raise
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode("utf-8", errors="replace" if timed_out else "strict")
        error = stderr.read().decode("utf-8", errors="replace")
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout, output=output, stderr=error)
        return subprocess.CompletedProcess(command, child.returncode, output, error)


def _version(binary: str) -> str | None:
    try:
        result = _run_cli_process([binary, "--version"], timeout=10)
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return text[0][:200] if text else None


def _validate_positive_number(
    value: str,
    name: str,
    errors: list[str],
    *,
    integer: bool,
) -> None:
    try:
        parsed = int(value) if integer else float(value)
    except ValueError:
        errors.append(f"{name} 必须是{'正整数' if integer else '正数'}")
        return
    if parsed <= 0:
        errors.append(f"{name} 必须大于 0")


def preflight_system(
    system: SystemConfig,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = str(system.implementation.get("adapter") or "")
    mapping = ADAPTER_ENV.get(adapter)
    errors: list[str] = []
    warnings: list[str] = []
    if not mapping:
        return {
            "system_id": system.system_id,
            "adapter": adapter,
            "ok": False,
            "errors": [f"未知 native CLI adapter: {adapter!r}"],
        }
    env_file = resolve_env_file(system)
    try:
        local = load_env_file(env_file)
    except ConfigurationError as exc:
        local = {}
        errors.append(str(exc))
    merged = dict(os.environ)
    merged.update(local)
    model = model or {}
    endpoint_profile = str(model.get("endpoint_profile") or "").strip()
    if endpoint_profile:
        source_key, source_base_url = _profile_names(endpoint_profile)
    else:
        source_key, source_base_url = mapping["key"], mapping["base_url"]
        errors.append("run target 缺少 endpoint_profile")
    required = [source_key, source_base_url]
    present = {name: bool(merged.get(name, "").strip()) for name in required}
    for name, ok in present.items():
        if not ok:
            errors.append(f"缺少 {name}")
    resolved_model = str(model.get("model_id") or "").strip() or None
    if resolved_model is None:
        errors.append("run target 缺少 model_id")
    base_url = merged.get(source_base_url, "").strip()
    endpoint_host = None
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append(f"{source_base_url} 必须是 http(s) URL")
        else:
            endpoint_host = parsed.hostname
            if parsed.username or parsed.password:
                errors.append(f"{source_base_url} 不得内嵌用户名或密码")
    endpoint_fingerprint = (
        hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()[:16]
        if base_url
        else None
    )
    binary_name = merged.get(mapping["binary"], "").strip() or mapping["binary_default"]
    binary = _binary_path(binary_name, merged)
    if not binary:
        errors.append(
            f"找不到可执行文件 {binary_name!r}；可在 {mapping['binary']} 中填写绝对路径"
        )
    binary_version = _version(binary) if binary else None
    harness = system.spec.get("harness")
    package_pin = harness.get("package_pin") if isinstance(harness, Mapping) else None
    if binary and not binary_version:
        warnings.append("CLI 存在，但无法读取版本")
    elif package_pin and binary_version and str(package_pin) not in binary_version:
        warnings.append(
            f"CLI 版本 {binary_version!r} 与 package pin {package_pin!r} 不一致；"
            "smoke 会记录实际版本"
        )
    if adapter == "codex":
        protocol_style = str(model.get("protocol_style") or "responses").strip()
        runtime_options = {"timeout_s": merged.get("NATIVE_TIMEOUT_S", "240")}
        if protocol_style not in {"responses", "chat"}:
            errors.append("target.protocol_style 必须是 responses 或 chat")
    elif adapter == "dsh":
        protocol_style = str(model.get("protocol_style") or "openai-responses").strip()
        runtime_options = {"timeout_s": merged.get("NATIVE_TIMEOUT_S", "240")}
        if protocol_style not in {"openai-responses", "openai-completions"}:
            errors.append(
                "target.protocol_style 必须是 openai-responses 或 openai-completions"
            )
        if not (dsh_profile_path() / "package.json").is_file():
            errors.append(f"缺少 dsh profile: {dsh_profile_path()}")
    else:
        protocol_style = str(model.get("protocol_style") or "anthropic-messages").strip()
        if protocol_style != "anthropic-messages":
            errors.append("Claude target.protocol_style 必须是 anthropic-messages")
        # 不设轮数上限：实测同一个 --max-turns 30 会让 30/102/129/173 轮的题跑完，
        # 却在 31 轮报 error_max_turns；而且 codex/dsh 没有这个开关，只卡 claude 会让
        # 三家 harness 不可比。硬边界统一用 timeout_s（+ max_budget_usd 作成本保险）。
        runtime_options = {
            "max_budget_usd": merged.get("CLAUDE_MAX_BUDGET_USD", "5.00"),
            "timeout_s": merged.get("NATIVE_TIMEOUT_S", "240"),
        }
        _validate_positive_number(
            runtime_options["max_budget_usd"],
            "CLAUDE_MAX_BUDGET_USD",
            errors,
            integer=False,
        )
    _validate_positive_number(
        runtime_options["timeout_s"], "NATIVE_TIMEOUT_S", errors, integer=True
    )
    return {
        "system_id": system.system_id,
        "adapter": adapter,
        "adapter_version": NATIVE_CLI_ADAPTER_VERSION,
        "ok": not errors,
        "env_file": str(env_file),
        "env_file_exists": env_file.is_file(),
        "required_env_present": present,
        "model": resolved_model,
        "model_label": model.get("label"),
        "endpoint_profile": endpoint_profile,
        "endpoint_host": endpoint_host,
        "endpoint_fingerprint": endpoint_fingerprint,
        "protocol_style": protocol_style,
        "runtime_options": runtime_options,
        "binary": binary,
        "binary_version": binary_version,
        "warnings": warnings,
        "errors": errors,
    }


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or fallback)[:80]


def _corpus_sessions(run_dir: Path) -> list[Mapping[str, Any]]:
    return load_benchmark_sessions(run_dir)


def _materialize_workspace(run_dir: Path, workspace: Path) -> tuple[int, int]:
    """把 benchmark 的公开材料物化成 agent 唯一可见的工作区。

    工作区根目录**只有** `INDEX.md` 与 `sessions/`：不写 `AGENTS.md`/`CLAUDE.md`。
    这两类文件名会被 codex/dsh（以及 dsh 的候选列表）当作项目指令自动加载，
    放在工作区里既与「只有 INDEX.md 和 sessions/ 是可见记忆」自相矛盾，也会诱导
    agent 去找「题目提示文件」——已有的 dsh run 日志中出现过这种搜索行为。
    运行规则统一走 prompt / system prompt。

    正文按 `(index, session, doc)` 一次算出，因此同一份语料可以在多份工作区之间
    确定性重建（每题一份工作区时用得到），每份的字节内容完全一致。
    """
    sessions = _corpus_sessions(run_dir)
    workspace.mkdir(parents=True, exist_ok=False)
    index = ["# Benchmark corpus", ""]
    doc_count = 0
    for session in sessions:
        session_id = int(session["session_id"])
        date = str(session["date"])
        session_dir = workspace / "sessions" / f"s{session_id:04d}_{_safe_name(date, 'date')}"
        session_dir.mkdir(parents=True)
        index.append(f"## Session {session_id} — {date}")
        for doc_index, doc in enumerate(session["docs"]):
            doc_id = str(doc["doc_id"])
            path = session_dir / f"{doc_index:04d}_{_safe_name(doc_id, 'doc')}.md"
            body = (
                f"# {doc.get('type') or 'Document'}\n\n"
                f"- session_id: {session_id}\n- date: {date}\n- doc_id: {doc_id}\n\n"
                f"{doc['content']}\n"
            )
            path.write_text(body, encoding="utf-8")
            index.append(f"- `{path.relative_to(workspace)}` — {doc_id}")
            doc_count += 1
        index.append("")
    (workspace / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    return len(sessions), doc_count


def _render_protocol(run_dir: Path) -> str:
    return load_benchmark_protocol(run_dir)


def _prompt(protocol: str, question: str) -> str:
    """所有运行规则都在这里；工作区里不放 AGENTS.md/CLAUDE.md 之类的指令文件。"""
    return (
        "You are answering a closed-world benchmark question. "
        "INDEX.md and sessions/ are the only observable memory; do not use external knowledge. "
        "Use the harness's native read/search tools. "
        "Do not read parent directories, other runs, judge files, hidden answers, or credentials. "
        "Do not write files. Check all relevant sessions for temporal questions. "
        "Return only the concise final answer, without reasoning.\n\n"
        f"{protocol}\n\nQuestion: {question}\nFinal answer:"
    )


def _runtime_env(
    local: Mapping[str, str],
    adapter: str,
    model: Mapping[str, Any],
) -> dict[str, str]:
    env = dict(os.environ)
    mapping = ADAPTER_ENV[adapter]
    profile = str(model["endpoint_profile"])
    source_key, source_base_url = _profile_names(profile)
    for name in tuple(env):
        if name.endswith("_API_KEY"):
            env.pop(name, None)
    for name, value in local.items():
        if not name.endswith("_API_KEY"):
            env[name] = value
    env[mapping["key"]] = local[source_key]
    env[mapping["base_url"]] = local[source_base_url]
    env[mapping["model"]] = str(model["model_id"])
    protocol_style = str(model.get("protocol_style") or "").strip()
    if adapter == "codex" and protocol_style:
        env["GPT_WIRE_API"] = protocol_style
    elif adapter == "dsh" and protocol_style:
        env["DEEPSEEK_API_STYLE"] = protocol_style
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    return env


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _secret_values(
    local: Mapping[str, str],
    endpoint_profile: str,
) -> dict[str, str]:
    """只收集本 adapter 的密钥值，供事件层检测「是否被打印进了输出」。

    返回值只在内存里比对，不写进任何产物；事件里只记变量名和行号。
    """
    name, _ = _profile_names(endpoint_profile)
    value = str(local.get(name) or "").strip()
    return {name: value} if value else {}


def _prepare_dsh_home(runtime_root: Path, env: Mapping[str, str]) -> Path:
    """在 `runtime_root` 下建一个独立的 $DSH_HOME。

    `runtime_root` 由调用方给到「这一题」的 runtime 子目录，因此并发 worker
    不会共享 $DSH_HOME（也就不会共享 session 缓存与 sqlite/WAL）。
    """
    runtime_home = Path(runtime_root) / "dsh-home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    if not DSH_PROFILES_DIR.is_dir():
        raise ConfigurationError(f"缺少 dsh profiles 目录: {DSH_PROFILES_DIR}")
    if not (dsh_profile_path() / "package.json").is_file():
        raise ConfigurationError(f"缺少 dsh profile: {dsh_profile_path()}")
    # 整棵 profiles/ 都复制，$DSH_HOME/profiles/<name> 由 --profile 选择。
    shutil.copytree(DSH_PROFILES_DIR, runtime_home / "profiles", dirs_exist_ok=True)
    api_style = env.get("DEEPSEEK_API_STYLE", "openai-responses").strip()
    if api_style not in {"openai-responses", "openai-completions"}:
        raise ConfigurationError(
            "DEEPSEEK_API_STYLE 必须是 openai-responses 或 openai-completions"
        )
    model = env["DEEPSEEK_MODEL"]
    base_url = env["DEEPSEEK_BASE_URL"].rstrip("/")
    settings = (
        "agent-default-model:\n"
        "  provider: benchmark\n"
        f"  model: {_toml_string(model)}\n\n"
        "llm-pi-ai:\n"
        "  providers:\n"
        "    benchmark:\n"
        "      displayName: benchmark\n"
        "      apiKeyEnv: DEEPSEEK_API_KEY\n"
        f"      api: {api_style}\n"
        f"      baseURL: {_toml_string(base_url)}\n"
        "      models:\n"
        f"        - id: {_toml_string(model)}\n"
    )
    (runtime_home / "settings.yaml").write_text(settings, encoding="utf-8")
    return runtime_home


def _runtime_dir(out_dir: Path, question_index: int) -> Path:
    """每题一个 runtime 子目录：CLI home、sqlite/WAL、session 缓存和
    `--output-last-message` 中间副本都不能跨题（更不能跨并发 worker）共享。
    """
    return out_dir / "runtime" / f"q{question_index:04d}"


def _command(
    adapter: str,
    binary: str,
    env: dict[str, str],
    workspace: Path,
    out_dir: Path,
    prompt: str,
    question_index: int,
) -> tuple[list[str], Path | None]:
    if adapter == "claude":
        model = env["CLAUDE_MODEL"]
        command = [
            binary,
            "--bare",
            "-p",
            "--model",
            model,
            "--tools",
            "Read,Grep,Glob",
            "--allowedTools",
            "Read,Grep,Glob",
            "--permission-mode",
            "dontAsk",
            "--add-dir",
            str(workspace),
            "--max-budget-usd",
            env.get("CLAUDE_MAX_BUDGET_USD", "5.00"),
            "--output-format",
            "json",
            "--no-session-persistence",
            "--append-system-prompt",
            "Only use native Read, Grep, and Glob in the current benchmark workspace. Never write files.",
            "--",
            prompt,
        ]
        return command, None
    if adapter == "codex":
        model = env["GPT_MODEL"]
        base_url = env["GPT_BASE_URL"].rstrip("/")
        wire_api = env.get("GPT_WIRE_API", "responses").strip()
        if wire_api not in {"responses", "chat"}:
            raise ConfigurationError("GPT_WIRE_API 必须是 responses 或 chat")
        codex_home = _runtime_dir(out_dir, question_index) / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        # CLI 落盘的「最后一条消息」只是 answer 的中间副本，放进临时区随 run 一起清理。
        last_dir = _runtime_dir(out_dir, question_index) / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        last_message = last_dir / f"{question_index:04d}.txt"
        command = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--disable",
            "plugins",
            "--disable",
            "remote_plugin",
            "--disable",
            "recommended_plugins",
            "--disable",
            "apps",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            "-C",
            str(workspace),
            "-m",
            model,
            "-c",
            'model_provider="benchmark"',
            "-c",
            'model_providers.benchmark.name="benchmark"',
            "-c",
            f"model_providers.benchmark.base_url={_toml_string(base_url)}",
            "-c",
            'model_providers.benchmark.env_key="GPT_API_KEY"',
            "-c",
            f"model_providers.benchmark.wire_api={_toml_string(wire_api)}",
            "-c",
            "model_providers.benchmark.requires_openai_auth=false",
            "--output-last-message",
            str(last_message),
            prompt,
        ]
        return command, last_message
    if adapter == "dsh":
        runtime_home = _prepare_dsh_home(_runtime_dir(out_dir, question_index), env)
        env["DSH_HOME"] = str(runtime_home)
        env["DSH_PERMISSION_MODE"] = "read-only"
        return [binary, "--profile", DSH_PROFILE_NAME, "--", prompt], None
    raise ConfigurationError(f"未知 adapter: {adapter}")


def _parse_answer(adapter: str, stdout: str, last_message: Path | None) -> str:
    if last_message and last_message.is_file():
        text = last_message.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    text = stdout.strip()
    if adapter == "claude" and text:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(obj, dict) and obj.get("is_error") is True:
            return ""
        for key in ("result", "text", "answer"):
            value = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if adapter == "codex":
        for line in reversed(text.splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") if isinstance(event, dict) else None
            value = (item or event).get("text") if isinstance(item or event, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    return text


def _decoded_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


# claude 的 result 对象里表示"预算耗尽"的 subtype / terminal_reason（取自 2.1.148 二进制）。
# error_max_structured_output_retries 不属于预算维度，单独归 cli_error 并保留 subtype。
BUDGET_SUBTYPES = {"error_max_turns": "turns", "error_max_budget_usd": "cost"}
BUDGET_TERMINAL_REASONS = {"max_turns": "turns", "budget_usd": "cost", "budget_tokens": "tokens"}
CLAUDE_REPORT_KEYS = (
    "type",
    "subtype",
    "is_error",
    "api_error_status",
    "terminal_reason",
    "num_turns",
    "total_cost_usd",
    "duration_ms",
)
# claude 在 API 层失败时 result 里会带这类前缀（is_error=true 且没有 api_error_status）。
_API_ERROR_HINT = re.compile(r"^\s*api[\s_-]*error", re.IGNORECASE)


def _json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    for candidate in (stripped, stripped[stripped.find("{") : stripped.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _cli_report(adapter: str, stdout: str) -> dict[str, Any] | None:
    """CLI 自报的结束状态（含它自己的轮数与成本）。

    拿不到就返回 None——不猜测。dsh headless 是纯文本输出，没有结构化上报；
    codex 只在我们实际见到 error 事件时记录其 message（不推断错误类别）。
    """
    if adapter == "claude":
        obj = _json_object(stdout)
        if not isinstance(obj, dict):
            return None
        report = {key: obj[key] for key in CLAUDE_REPORT_KEYS if key in obj}
        # 失败时 result 字段放的是诊断消息；成功时它就是答案本身，不重复记录。
        if report.get("is_error") and isinstance(obj.get("result"), str):
            report["result_excerpt"] = obj["result"][:500]
        return report or None
    if adapter == "codex":
        for line in reversed((stdout or "").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "error":
                return {"type": "error", "message": str(event.get("message") or "")[:500]}
        return None
    return None


def _classify_failure(
    adapter: str,
    *,
    returncode: int | None,
    answer: str,
    timed_out: bool,
    cli_report: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    """把一次调用归类为可诊断的失败类型。

    类别：timeout（我们自己的 wall-clock 上限）、budget_exceeded（CLI 自报的轮数/成本/
    token 预算耗尽）、api_error（API 层返回错误）、cli_error（CLI 报错但类别未知，原样
    保留 subtype/message）、process_error（非零退出且 CLI 没给原因）、parse_error
    （退出正常但取不到答案）。拿不到证据时不下结论，把原始标识留在 cli_report 里。
    """
    if timed_out:
        return "timeout", {}
    report = cli_report or {}
    subtype = str(report.get("subtype") or "")
    terminal = str(report.get("terminal_reason") or "")
    cli_error = bool(report.get("is_error")) or report.get("type") == "error"
    if cli_error:
        dimension = BUDGET_SUBTYPES.get(subtype) or BUDGET_TERMINAL_REASONS.get(terminal)
        if dimension:
            return "budget_exceeded", {"budget_dimension": dimension}
        status = report.get("api_error_status")
        if status is not None:
            return "api_error", {"api_error_status": status}
        message = str(report.get("result_excerpt") or report.get("message") or "")
        if subtype == "error_during_execution" and _API_ERROR_HINT.match(message):
            return "api_error", {}
        return "cli_error", {}
    if returncode:
        return "process_error", {}
    if not answer:
        return "parse_error", {}
    return None, {}


def _corpus_counts(run_dir: Path) -> tuple[int, int]:
    sessions = _corpus_sessions(run_dir)
    return len(sessions), sum(len(session["docs"]) for session in sessions)


def _load_done(results_path: Path) -> dict[str, dict]:
    """已有结果按 question_id 取末行（resume 追加写、末行胜出）。"""
    done: dict[str, dict] = {}
    if not results_path.is_file():
        return done
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done[str(record["question_id"])] = record
    return done


def _question_id(question: Mapping[str, Any]) -> str:
    return benchmark_question_id(dict(question))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 运行期临时区（相对 run 目录）：可从 benchmark + adapter 版本重建，不属于实验结果。
SCRATCH_DIRS = ("workspace", "runtime")


def _clean_scratch(out_dir: Path) -> tuple[list[str], list[str]]:
    """删除运行期临时区，返回 (已删除目录, 删除失败目录)。"""
    removed: list[str] = []
    failed: list[str] = []
    for name in SCRATCH_DIRS:
        path = out_dir / name
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            failed.append(str(path))
        else:
            removed.append(str(path))
    return removed, failed


def _plan_concurrency(plan: Mapping[str, Any], key: str) -> int:
    """读取 plan.execution 里的并发度。缺省 1 = 完全串行（与历史行为一致）。"""
    execution = plan.get("execution") or {}
    raw = execution.get(key, 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigurationError(f"execution.{key} 必须是正整数，当前为 {raw!r}")
    return raw


def _load_attempts(events_path: Path) -> dict[str, int]:
    """resume 时按题统计已有 run_start 次数，给新尝试编号。只读不写，单线程调用。"""
    attempts: dict[str, int] = {}
    if not events_path.is_file():
        return attempts
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "run_start":
            key = str(event.get("question_id") or "")
            attempts[key] = attempts.get(key, 0) + 1
    return attempts


def _error_record(
    *,
    question_index: int,
    question_id: str,
    error_type: str,
    error_detail: dict[str, Any] | None,
    returncode: int | None,
    elapsed_s: float,
    out_dir: Path,
    raw_dir: Path,
    events_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """没有 CLI 输出的失败记录（worker 内的非 subprocess 异常）。

    手工拼出与正常路径同样的形状：raw 文件照写（空），因此下游按
    `raw_stdout`/`raw_stderr` 取证据时不会缺文件。
    """
    for name in ("stdout", "stderr"):
        (raw_dir / f"{question_index:04d}.{name}.txt").write_text("", encoding="utf-8")
    record = {
        "schema": "agent-harnesses.result/v1",
        "question_id": question_id,
        "question_index": question_index,
        "status": "failed",
        "answer": None,
        "judgeable": False,
        "correct": None,
        "error_type": error_type,
        "cli_report": None,
        "error_detail": error_detail or None,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "raw_stdout": str((raw_dir / f"{question_index:04d}.stdout.txt").relative_to(out_dir)),
        "raw_stderr": str((raw_dir / f"{question_index:04d}.stderr.txt").relative_to(out_dir)),
        "events": events_path.name,
        "telemetry": events.telemetry_summary([]),
        "boundary_violation": False,
        "boundary_violation_details": [],
        "contaminated": False,
    }
    if extra:
        record.update(dict(extra))
    return record


def _run_questions(
    plan_path: Path,
    run_dir: Path,
    out_dir: Path,
    limit: int,
    resume: bool = False,
    parallel_questions: int | None = None,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    system_block = plan["system"]
    implementation = system_block["implementation"]
    adapter = str(implementation["adapter"])
    model_block = plan.get("model") or {}
    model_id = str(model_block.get("model_id") or "").strip() or None
    if model_id is None:
        raise ConfigurationError("run plan 缺少 model.model_id")
    endpoint_profile = str(model_block.get("endpoint_profile") or "").strip()
    if not endpoint_profile:
        raise ConfigurationError("run plan 缺少 model.endpoint_profile")
    env_file_raw = str(implementation.get("env_file") or "").strip()
    env_file = Path(env_file_raw) if env_file_raw else SECRETS_ENV_FILE
    if not env_file.is_absolute():
        env_file = (REPOSITORY_ROOT / env_file).resolve()
    local = load_env_file(env_file)
    env = _runtime_env(local, adapter, model_block)
    mapping = ADAPTER_ENV[adapter]
    binary_name = env.get(mapping["binary"], "").strip() or mapping["binary_default"]
    binary = _binary_path(binary_name, env)
    if not binary:
        raise ConfigurationError(f"找不到可执行文件: {binary_name}")

    if parallel_questions is None:
        parallel_questions = _plan_concurrency(plan, "questions_in_parallel")
    if parallel_questions <= 0:
        raise ConfigurationError("parallel_questions 必须是正整数")

    protocol = _render_protocol(run_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    events_path = out_dir / "events.jsonl"
    # 历史行为：非 resume 的 run 从空文件开始；即使本次一题都不跑（全部已完成），
    # results.jsonl / events.jsonl 也必须存在。
    if not resume:
        results_path.write_text("", encoding="utf-8")
        events_path.write_text("", encoding="utf-8")
    timeout_s = (plan.get("execution") or {}).get("timeout_s")
    if timeout_s is None:
        timeout_s = int(env.get("NATIVE_TIMEOUT_S", "240"))
    if type(timeout_s) is not int or timeout_s <= 0:
        raise ConfigurationError("execution.timeout_s 必须是正整数")
    secrets = _secret_values(local, endpoint_profile)
    done = _load_done(results_path) if resume else {}
    # resume 判定必须在派发前完成：把已完成的题排除掉后，remaining 才是这次要跑的题；
    # 工作区也只给 remaining 建（已完成题不再需要，避免白复制语料）。
    done_ids = {
        str(question_id)
        for question_id, record in done.items()
        if record.get("status") == "completed"
    }
    questions = load_benchmark_questions(run_dir)
    limited = questions[:limit] if limit else questions
    indexed_all = [(index, question) for index, question in enumerate(limited)]
    remaining = [
        (index, question)
        for index, question in indexed_all
        if _question_id(question) not in done_ids
    ]
    n_sessions, n_docs = _corpus_counts(run_dir)
    if parallel_questions <= 1:
        workspace = out_dir / "workspace"
        if not (workspace.is_dir() and any(workspace.iterdir())):
            _materialize_workspace(run_dir, workspace)
        work_items = [(index, question, workspace) for index, question in remaining]
    else:
        work_items = []
        for index, question in remaining:
            workspace = out_dir / "workspace" / f"q{index:04d}"
            _materialize_workspace(run_dir, workspace)
            work_items.append((index, question, workspace))
    # 慢题优先派发：并发窗口里最长的任务应该最早开始，否则 wall-clock 由尾部空转决定。
    # 优先用上一次 run 的实测耗时（resume 场景），其次用题库声明，最后才按下标。
    def _priority(item: tuple[int, Mapping[str, Any], Path]) -> tuple[float, int]:
        index, question, _ = item
        previous = done.get(_question_id(question)) or {}
        elapsed = previous.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            return (-float(elapsed), index)
        declared = question.get("expected_elapsed_s")
        if isinstance(declared, (int, float)):
            return (-float(declared), index)
        return (0.0, index)

    work_items.sort(key=_priority)
    attempts = _load_attempts(events_path) if resume else {}
    telemetry_by_status = {"full": 0, "partial": 0, "unavailable": 0}
    n_secret_leaks = 0
    violations: list[dict[str, Any]] = []
    contaminated_run = False
    write_lock = threading.Lock()

    def worker(
        item: tuple[int, Mapping[str, Any], Path],
    ) -> dict[str, Any]:
        index, question, workspace = item
        question_id = _question_id(question)
        prompt = _prompt(protocol, question["question"])
        started = time.monotonic()
        # _command 会往 env 里写 CODEX_HOME/DSH_HOME，必须用每题自己的副本，
        # 否则并发时后启动的题会改掉先启动的题的环境（同一进程内的竞态）。
        question_env = dict(env)
        try:
            command, last_message = _command(
                adapter, binary, question_env, workspace, out_dir, prompt, index
            )
            timed_out = False
            try:
                completed = _run_cli_process(
                    command,
                    cwd=workspace,
                    env=question_env,
                    timeout=timeout_s,
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                answer = _parse_answer(adapter, stdout, last_message)
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = _decoded_output(exc.stdout)
                stderr = _decoded_output(exc.stderr)
                answer = ""
                returncode = None
                timed_out = True
            # 失败分型：先看 CLI 自报的原因，再看退出码/答案，避免把"预算耗尽"记成进程崩溃。
            cli_report = _cli_report(adapter, stdout)
            error_type, error_detail = _classify_failure(
                adapter,
                returncode=returncode,
                answer=answer,
                timed_out=timed_out,
                cli_report=cli_report,
            )
            elapsed_s = round(time.monotonic() - started, 3)
            (raw_dir / f"{index:04d}.stdout.txt").write_text(stdout, encoding="utf-8")
            (raw_dir / f"{index:04d}.stderr.txt").write_text(stderr, encoding="utf-8")
            with write_lock:
                attempt = attempts.get(question_id, 0)
                normalized = events.normalize(
                    adapter,
                    stdout,
                    stderr,
                    question_index=index,
                    workspace=workspace,
                    question_id=question_id,
                    secrets=secrets,
                    attempt=attempt,
                    status="completed" if not error_type else "failed",
                    error_type=error_type,
                    returncode=returncode,
                    elapsed_s=elapsed_s,
                )
                attempts[question_id] = attempt + 1
                record = {
                    "schema": "agent-harnesses.result/v1",
                    "question_id": question_id,
                    "question_index": index,
                    "quality_status": question.get("quality_status"),
                    "status": "completed" if not error_type else "failed",
                    "answer": answer or None,
                    "judgeable": False,
                    "correct": None,
                    "error_type": error_type,
                    # CLI 自报的结束状态（claude 含 is_error/subtype/terminal_reason/num_turns/
                    # total_cost_usd）；拿不到时为 None，不做推断。
                    "cli_report": cli_report,
                    "error_detail": error_detail or None,
                    "returncode": returncode,
                    "elapsed_s": elapsed_s,
                    "raw_stdout": str((raw_dir / f"{index:04d}.stdout.txt").relative_to(out_dir)),
                    "raw_stderr": str((raw_dir / f"{index:04d}.stderr.txt").relative_to(out_dir)),
                    "events": events_path.name,
                    "telemetry": events.telemetry_summary(normalized),
                    "boundary_violation": False,
                    "boundary_violation_details": [],
                    "contaminated": False,
                }
                # 结果行与事件块在同一次加锁内写盘：并发 worker 的行不会交错，
                # 且 results.jsonl 的末行胜出语义（resume 依赖）保持成立。
                with results_path.open("a", encoding="utf-8") as results:
                    results.write(json.dumps(record, ensure_ascii=False) + "\n")
                with events_path.open("a", encoding="utf-8") as events_file:
                    events.write(events_file, normalized)
        except Exception as exc:  # worker 自身的异常也要落成一行，不能静默丢题
            elapsed_s = round(time.monotonic() - started, 3)
            error_type = "worker_error"
            error_detail = {"exception": type(exc).__name__, "message": str(exc)[:500]}
            cli_report = None
            normalized = []
            with write_lock:
                attempt = attempts.get(question_id, 0)
                attempts[question_id] = attempt + 1
                record = _error_record(
                    question_index=index,
                    question_id=question_id,
                    error_type=error_type,
                    error_detail=error_detail,
                    returncode=None,
                    elapsed_s=elapsed_s,
                    out_dir=out_dir,
                    raw_dir=raw_dir,
                    events_path=events_path,
                    extra={"quality_status": question.get("quality_status")},
                )
                with results_path.open("a", encoding="utf-8") as results:
                    results.write(json.dumps(record, ensure_ascii=False) + "\n")
        # 每题的分析只依赖本题的题内状态，因此按题算完再回主线程归并，
        # 避免并发累加共享计数器丢更新。
        scan = boundaries.scan_events(normalized, workspace)
        return {
            "index": index,
            "record": record,
            "telemetry": events.telemetry_summary(normalized),
            "scan": scan,
        }

    workers = min(parallel_questions, len(work_items))
    if workers <= 1:
        outcomes = [worker(item) for item in work_items]
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mb-question"
        ) as pool:
            futures = [pool.submit(worker, item) for item in work_items]
            outcomes = [future.result() for future in as_completed(futures)]
    for outcome in outcomes:
        telemetry = outcome["telemetry"]
        telemetry_by_status[telemetry["status"]] = (
            telemetry_by_status.get(telemetry["status"], 0) + 1
        )
        n_secret_leaks += telemetry["secret_leaks"]
        scan = outcome["scan"]
        contaminated = bool(scan["attempted_outside"] and scan["succeeded"])
        contaminated_run = contaminated_run or contaminated
        record = outcome["record"]
        if scan["boundary_violation"]:
            violations.append(
                {
                    "question_index": outcome["index"],
                    "question_id": record["question_id"],
                    "attempted_outside": scan["attempted_outside"],
                    "succeeded": scan["succeeded"],
                    "details": scan["details"],
                }
            )
        done[record["question_id"]] = record
    errors = sum(1 for record in done.values() if record.get("error_type"))
    errors_by_type: dict[str, int] = {}
    for record in done.values():
        key = record.get("error_type")
        if key:
            errors_by_type[str(key)] = errors_by_type.get(str(key), 0) + 1
    return {
        "schema": "agent-harnesses.native-smoke-summary/v3",
        "system_id": plan["system_id"],
        "adapter": adapter,
        "model": env[mapping["model"]],
        "endpoint_host": urlsplit(env[mapping["base_url"]]).hostname,
        "binary_version": _version(binary),
        "n_sessions": n_sessions,
        "n_docs": n_docs,
        "n_questions": len(limited),
        "n_errors": errors,
        "n_errors_by_type": errors_by_type,
        "concurrency": {
            "questions_in_parallel": parallel_questions,
            "workers_used": workers,
            "questions_dispatched": len(work_items),
            "questions_already_completed": len(done_ids),
            "note": "并发度只影响调度与落盘顺序，不进 fingerprint；换并发度可以 --resume 同一目录",
        },
        "scorable": False,
        "contaminated_run": contaminated_run,
        "secret_leaks": n_secret_leaks > 0,
        "telemetry": {
            "by_status": telemetry_by_status,
            "required_by_architecture": "原生工具清单与 telemetry completeness",
            "note": "unavailable 表示该 CLI 的当前输出格式没有工具级事件，不等于没有发生工具调用",
        },
        "boundary": {
            "n_questions_with_violation": len(violations),
            "violations": violations,
            "contaminated_run": contaminated_run,
            "detector": boundaries.DETECTOR_VERSION,
            "note": "telemetry-based detection, not OS-level enforcement",
        },
    }


def run(
    plan_path: Path,
    run_dir: Path,
    out_dir: Path,
    limit: int,
    resume: bool = False,
    keep_scratch: bool = False,
    parallel_questions: int | None = None,
) -> dict[str, Any]:
    """跑完一次 smoke，并把 output 目录收敛成「只留实验结果」。

    `workspace/`（由公开 material 确定性物化）与 `runtime/`（各 CLI 的 home、sqlite
    日志、session 缓存、node_modules 链接等）只是运行期临时区：既不是实验结果，也
    能从 benchmark + adapter 版本重建。默认在 run 结束后删除，`keep_scratch=True`
    时保留用于排障。

    `parallel_questions` 为 None 时取 run plan 的 `execution.questions_in_parallel`
    （缺省 1 = 完全串行）。并发只改调度与写入顺序，不改每题语义，因此不进 fingerprint。
    """
    out_dir = Path(out_dir)
    started_at = _now_iso()
    removed: list[str] = []
    cleanup_failed: list[str] = []
    try:
        if resume:
            # A killed runner cannot execute the finally block below. Rebuild
            # its scratch directories while preserving answers and telemetry.
            _, stale_cleanup_failed = _clean_scratch(out_dir)
            if stale_cleanup_failed:
                raise ConfigurationError(
                    "Cannot clear interrupted native scratch: " + ", ".join(stale_cleanup_failed)
                )
        summary = _run_questions(
            plan_path,
            run_dir,
            out_dir,
            limit,
            resume=resume,
            parallel_questions=parallel_questions,
        )
    finally:
        # 临时区清理必须在所有 worker 结束之后：_run_questions 内部是 join 完才返回。
        if not keep_scratch:
            removed, cleanup_failed = _clean_scratch(out_dir)
    summary["started_at"] = started_at
    summary["finished_at"] = _now_iso()
    summary["scratch"] = {
        "retained": bool(keep_scratch),
        "removed": removed,
        "cleanup_failed": cleanup_failed,
        "note": "workspace/runtime 是运行期临时区，默认删除；execution.keep_scratch=true 可保留",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native CLI smoke runner")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="保留 workspace/ 与 runtime/ 临时区（默认删除）",
    )
    parser.add_argument(
        "--parallel-questions",
        type=int,
        default=None,
        help="并发答题数；缺省读 run plan 的 execution.questions_in_parallel（默认 1）",
    )
    args = parser.parse_args(argv)
    try:
        summary = run(
            Path(args.plan),
            Path(args.run),
            Path(args.out),
            args.limit,
            resume=args.resume,
            keep_scratch=args.keep_scratch,
            parallel_questions=args.parallel_questions,
        )
    except (ConfigurationError, KeyError, ValueError) as exc:
        print(f"preflight/runtime error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
