"""把三套原生 CLI 的原始 stdout 归一为统一事件流（events.jsonl）。

三家的输出格式完全不同：codex 是 JSONL 事件流，claude 是单个 JSON 对象
（默认 `--output-format json` 不含工具级事件），dsh 是纯文本。

本模块只解析和归一，不执行命令。遥测覆盖度按题分为 full/partial/unavailable，
**绝不把「解析不到」当成「没有发生」**——拿不到证据就是 unavailable，并写明原因。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import boundaries

EVENT_SCHEMA = "agent-harnesses.event/v1"
NORMALIZER_VERSION = "native-events-v1"

PREVIEW = 256
OUTPUT_PREVIEW = 2048


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(value: Any, limit: int) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + f"…(+{len(text) - limit})"
    return text


def _envelope(question_index: int, kind: str, adapter: str, workspace: Path) -> dict[str, Any]:
    return {
        "schema": EVENT_SCHEMA,
        "question_index": question_index,
        "kind": kind,
        "adapter": adapter,
        "ts": _now(),
        "workspace": {"path": str(workspace), "root_rel": workspace.name},
        "normalizer": NORMALIZER_VERSION,
    }


def _iter_jsonl(text: str) -> list[dict[str, Any]]:
    objects = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _single_json(text: str) -> dict[str, Any] | None:
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


def _final_text(obj: Mapping[str, Any]) -> str:
    for key in ("result", "text", "answer"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _codex_events(
    stdout: str, question_index: int, adapter: str, workspace: Path
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    open_calls: list[str] = []
    seq = 0
    for line_number, line in enumerate((stdout or "").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            note = _envelope(question_index, "note", adapter, workspace)
            note.update({"code": "unparsed_stdout_line", "line_number": line_number})
            events.append(note)
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
        item_type = item.get("type")
        if kind == "thread.started":
            note = _envelope(question_index, "note", adapter, workspace)
            note.update({"code": "thread_started", "thread_id": obj.get("thread_id")})
            events.append(note)
        elif kind == "turn.completed":
            usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
            if usage:
                event = _envelope(question_index, "usage", adapter, workspace)
                event.update({"source": "codex_turn_completed"})
                event.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
                events.append(event)
        elif kind == "item.started" and item_type in {"command_execution", "shell"}:
            call_id = f"c{seq}"
            seq += 1
            open_calls.append(call_id)
            command = item.get("command")
            event = _envelope(question_index, "tool_call", adapter, workspace)
            event.update(
                {
                    "call_id": call_id,
                    "tool_name": "command",
                    "tool_name_raw": item_type,
                    "args": {"command": command},
                    "args_preview": _clip(command, PREVIEW),
                    "boundary": boundaries.classify_call("command", {"command": command}, workspace),
                }
            )
            events.append(event)
        elif kind == "item.completed" and item_type in {"command_execution", "shell"}:
            call_id = open_calls.pop(0) if open_calls else None
            event = _envelope(question_index, "tool_result", adapter, workspace)
            event.update(
                {
                    "call_id": call_id,
                    "status": item.get("status"),
                    "exit_code": item.get("exit_code"),
                    "output_preview": _clip(item.get("aggregated_output"), OUTPUT_PREVIEW),
                }
            )
            events.append(event)
        elif kind == "item.completed" and item_type == "agent_message":
            event = _envelope(question_index, "message", adapter, workspace)
            event.update({"role": "assistant", "text": item.get("text")})
            events.append(event)
        elif kind in {"item.started", "item.completed"} and item_type:
            note = _envelope(question_index, "note", adapter, workspace)
            note.update({"code": "item_completed", "item_type": item_type})
            events.append(note)
    return events


def _claude_events(
    stdout: str, question_index: int, adapter: str, workspace: Path
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    obj = _single_json(stdout)
    if obj is None:
        note = _envelope(question_index, "note", adapter, workspace)
        note.update({"code": "claude_stdout_not_json", "bytes": len(stdout or "")})
        events.append(note)
        return events
    text = _final_text(obj)
    if text:
        event = _envelope(question_index, "message", adapter, workspace)
        event.update({"role": "assistant", "text": text})
        events.append(event)
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    if usage:
        event = _envelope(question_index, "usage", adapter, workspace)
        event.update({"source": "claude_json"})
        event.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
        events.append(event)
    for key in ("num_turns", "duration_ms", "total_cost_usd"):
        if obj.get(key) is not None:
            note = _envelope(question_index, "note", adapter, workspace)
            note.update({"code": key, "value": obj.get(key)})
            events.append(note)
    note = _envelope(question_index, "note", adapter, workspace)
    note.update(
        {
            "code": "claude_tool_events_unavailable",
            "detail": "output_format=json 不含工具级事件",
        }
    )
    events.append(note)
    return events


def _dsh_events(
    stdout: str, question_index: int, adapter: str, workspace: Path
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    structured = _iter_jsonl(stdout)
    # dsh 目前是纯文本；若将来出现结构化事件，此处按同一信封映射，无需改 schema。
    tool_like = [o for o in structured if o.get("type") or o.get("tool") or o.get("event")]
    if tool_like:
        for obj in tool_like:
            event = _envelope(question_index, "note", adapter, workspace)
            event.update({"code": "dsh_structured_event", "raw_type": obj.get("type")})
            events.append(event)
        return events
    text = (stdout or "").strip()
    if text:
        event = _envelope(question_index, "message", adapter, workspace)
        event.update({"role": "assistant", "text": text})
        events.append(event)
    note = _envelope(question_index, "note", adapter, workspace)
    note.update(
        {
            "code": "dsh_structured_events_unavailable",
            "detail": "headless profile 输出为纯文本，无工具级事件",
        }
    )
    events.append(note)
    return events


_PARSERS = {"codex": _codex_events, "claude": _claude_events, "dsh": _dsh_events}


def normalize(
    adapter: str,
    stdout: str,
    stderr: str,
    *,
    question_index: int,
    workspace: Path,
    question_id: str | None = None,
    secrets: Mapping[str, str] | None = None,
    attempt: int = 0,
    status: str = "completed",
    error_type: str | None = None,
    returncode: int | None = None,
    elapsed_s: float | None = None,
) -> list[dict[str, Any]]:
    """把一次 CLI 输出归一为事件列表，首尾补 run_start / run_end。

    resume 会对同一题重跑并追加新的事件块，`attempt` 用来区分同一题的第几次尝试。
    """
    workspace = Path(workspace)
    parser = _PARSERS.get(adapter)
    events: list[dict[str, Any]] = []
    start = _envelope(question_index, "run_start", adapter, workspace)
    start.update({"question_id": question_id, "attempt": attempt})
    events.append(start)
    if parser is None:
        note = _envelope(question_index, "note", adapter, workspace)
        note.update({"code": "unknown_adapter", "adapter": adapter})
        events.append(note)
    else:
        events.extend(parser(stdout, question_index, adapter, workspace))
    if secrets:
        for channel, text in (("stdout", stdout), ("stderr", stderr)):
            for leak in boundaries.secret_leaks(text or "", secrets, channel=channel):
                event = _envelope(question_index, "secret_leak", adapter, workspace)
                event.update(leak)
                events.append(event)
    end = _envelope(question_index, "run_end", adapter, workspace)
    end.update(
        {
            "question_id": question_id,
            "attempt": attempt,
            "status": status,
            "error_type": error_type,
            "returncode": returncode,
            "duration_s": elapsed_s,
        }
    )
    events.append(end)
    return events


def write(handle, events: Iterable[Mapping[str, Any]]) -> int:
    written = 0
    for event in events:
        handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
        written += 1
    return written


def telemetry_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """按题统计遥测覆盖度。claude/dsh 拿不到工具事件就是 unavailable，不是失败。"""
    tool_calls = 0
    tool_results = 0
    messages = 0
    usage_events = 0
    leaks = 0
    notes: list[str] = []
    for event in events:
        kind = event.get("kind")
        if kind == "tool_call":
            tool_calls += 1
        elif kind == "tool_result":
            tool_results += 1
        elif kind == "message":
            messages += 1
        elif kind == "usage":
            usage_events += 1
        elif kind == "secret_leak":
            leaks += 1
        elif kind == "note" and event.get("code"):
            code = str(event["code"])
            if code not in notes:
                notes.append(code)
    if tool_calls:
        status = "partial"
    else:
        status = "unavailable"
    return {
        "status": status,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "messages": messages,
        "usage_events": usage_events,
        "secret_leaks": leaks,
        "notes": notes,
    }
