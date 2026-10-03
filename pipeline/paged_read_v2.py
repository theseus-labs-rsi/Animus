"""Exact public-document paging for the original semantic readers.

Only transport and read progress are managed here. The same role supplies its
final opinion, and the original semantic/evidence validators still decide use.
"""
from copy import deepcopy
import json

from pipeline.world_context import ReadWindow, AUTO_DELIVERY_INSTRUCTION, ProgressGuard
from pipeline.semantic_review import fingerprint

LIMIT = 100000
class PagedReadExecutionError(RuntimeError):
    def __init__(self, response):
        self.response = deepcopy(response)
        super().__init__("Paged reader execution failed; original failure retained")


class PagedReadProtocolError(ValueError):
    """A bounded reader failed to operate its own paging controls.

    This is local to one role on one candidate.  It must remain auditable and
    pending, while later candidates continue.
    """
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)
RULES = """
材料较长，现逐页提供原始 documents。继续承担原审阅角色，最终意见的 schema 与判定要求完全保留。
本角色的原文读取窗口为60000字符；每次 inspect 仍最多12个文档，超长单文档按目录提供的 parts 分页。
读完当前窗口后，可输出 {"action":"continue","notes":"累积阅读笔记，最多6000字符；保留支持、反证、时点和未决内容"}，程序保存笔记并提供下一页。
可用 inspect 主动重读关联原文；不输出 index，笔记不能替代需要核对的原文。请先读完所有文档的全部页。
全部读完后输出 {"action":"finish","opinion":{按原审阅 schema 的完整意见}}。
需要核对细节时先 inspect。阅读完成只代表材料已送达；理解不足仍须保留 unresolved/partial/unknown，不能为结束而通过。
"""


def _window(messages):
    payload = json.loads(messages[-1]["content"])
    documents = payload.pop("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Paged public reading requires documents")
    nodes = {doc["doc_id"]: {"label": doc["doc_id"], "value": doc} for doc in documents}
    if len(nodes) != len(documents):
        raise ValueError("Public document IDs must be unique")
    return ReadWindow(payload, nodes, max_read_chars=60000, part_chars=20000)


def _next(window):
    # Exact transfer pages; no relevance filter and no dropped documents.
    return window.deliver_next_unread()


def _session(messages, step, params, chat_json=None, transcript=None):
    window = _window(messages)
    system = messages[0]["content"] + AUTO_DELIVERY_INSTRUCTION + RULES
    notes, records, failures, feedback = "", [], 0, None
    progress = ProgressGuard("Paged semantic reader")
    _next(window)
    allowance = min(2400, max(24, sum(window.parts(key) for key in window.nodes) * 4))
    for number in range(allowance):
        request = window.messages(system, {"reading_notes": notes, "action_error": feedback})
        if sum(len(m["content"]) for m in request) > LIMIT:
            raise ValueError("Non-document review context exceeds paged request allowance")
        window.mark_sent()
        entry = {"messages_hash": fingerprint(request)}
        if transcript is not None:
            if number >= len(transcript) or transcript[number]["messages_hash"] != entry["messages_hash"]:
                raise ValueError("Paged reader transcript differs from exact public input")
            raw = deepcopy(transcript[number]["raw_output"])
        else:
            raw = chat_json(step, request, **params)
        entry["raw_output"] = deepcopy(raw)
        records.append(entry)
        if not isinstance(raw, dict) or "__error__" in raw:
            raise PagedReadExecutionError(raw)
        try:
            if raw.get("action") == "index":
                raise ValueError("Document pages are delivered automatically; use continue with notes")
            if window.control(raw):
                pass
            elif raw.get("action") == "continue":
                if not isinstance(raw.get("notes"), str) or len(raw["notes"]) > 6000:
                    raise ValueError("Reading notes must be text within 6000 characters")
                notes = raw["notes"]
                _next(window)
            elif raw.get("action") == "finish":
                if window.read != set(window.nodes):
                    raise ValueError("All original document pages must be read before a final opinion")
                opinion = raw.get("opinion")
                if not isinstance(opinion, dict) or "_paged_read" in opinion or "__error__" in opinion:
                    raise ValueError("Final opinion must use the original role schema")
                if transcript is not None and number + 1 != len(transcript):
                    raise ValueError("Extra paged reader transcript")
                return deepcopy(opinion), records
            else:
                raise ValueError("Use continue/inspect/finish")
            progress.observe(window.progress_marker())
            failures, feedback = 0, None
        except (ValueError, TypeError, KeyError) as exc:
            failures, feedback = failures + 1, str(exc)
            if failures >= 4:
                raise PagedReadProtocolError(
                    "paged_invalid_action",
                    "Repeated invalid paged reader actions: " + feedback) from exc
    raise PagedReadProtocolError("paged_allowance_exhausted", "Paged reader allowance exhausted")


def call(step, messages, *, chat_json, **params):
    if sum(len(m["content"]) for m in messages) <= LIMIT:
        return chat_json(step, messages, **params)
    opinion, transcript = _session(messages, step, params, chat_json=chat_json)
    opinion["_paged_read"] = {"version": "exact-public-pages/v2", "messages": deepcopy(messages),
        "step": step, "params": deepcopy(params), "transcript": transcript}
    return opinion


def validate(output, documents):
    receipt = output.get("_paged_read") if isinstance(output, dict) else None
    if receipt is None:
        return
    if receipt.get("version") != "exact-public-pages/v2":
        raise ValueError("Unknown public paging protocol")
    if json.loads(receipt["messages"][-1]["content"]).get("documents") != documents:
        raise ValueError("Paged reader public corpus changed")
    opinion, _ = _session(receipt["messages"], receipt["step"], receipt["params"], transcript=receipt["transcript"])
    if opinion != {k:v for k,v in output.items() if k != "_paged_read"}:
        raise ValueError("Paged reader final opinion changed")
