"""Bounded, resumable authorship inside the original L3 proposal stage."""
from copy import deepcopy
import json
from pathlib import Path

from pipeline import process_proposals as p
from pipeline.world_context import ReadWindow, INSTRUCTION, MAX_READ_CHARS, ProgressGuard

STRATEGY = "process-batches/v1"
BATCH_SIZE = 8
RULES = """
逐批提出业务过程题。程序提供一组原事件及参与对象作为起点，你决定哪些材料应一起理解；需要跨对象或跨期信息时用 inspect/index 读取原节点。
每次可输出 {"action":"submit","proposals":[按原 schema],"reason":"说明"}，数量不超过 working_state.batch_maximum。
每次只提交新问题，避免重复已有意图。witness_refs 只能引用已读过的事件、关系、实体字段；未读事实须先 inspect。
程序保存每批原回复和有效提案。没有更多合适问题时输出 {"action":"finish","reason":"说明"}；允许少于总目标，不能凑题。
当前窗口不够支持判断时继续读取。全部原材料可通过目录读取，默认窗口不限定业务问题的范围。
"""


def context(wp, ws):
    prepared = p.prepare_process_proposals(wp, ws, target=0, model="context")
    payload = json.loads(prepared["messages"][-1]["content"])
    world = payload.pop("world")
    payload.pop("target_maximum")
    nodes = {}
    # Events first supplies useful starting points; the author can read any node.
    for key in ["events", "entities", "relations", *[k for k in world if k not in ("events", "entities", "relations")]]:
        value = world.get(key)
        if key == "entities":
            rows = [(f"entity/{name}", fields) for name, fields in value.items()]
        elif key == "disclosure" and isinstance(value, dict):
            rows = [(f"disclosure/{field}/{i}", item) for field, items in value.items()
                    for i, item in (enumerate(items) if isinstance(items, list) else [("value", items)])]
        elif isinstance(value, list):
            rows = [(f"{key}/{i}", item) for i, item in enumerate(value)]
        else:
            rows = [(key, value)]
        for identity, item in rows:
            label = identity
            if isinstance(item, dict):
                label += " " + str(item.get("id", "")) + " " + str(item.get("type", ""))
            nodes[identity] = {"label": label, "value": deepcopy(item)}
    return ReadWindow(payload, nodes)


def _next_window(window):
    unread = [key for key in window.nodes if key not in window.read]
    if not unread:
        window.visible = {}
        return
    anchor = unread[0]
    value = window.nodes[anchor]["value"]
    related = ([f"entity/{name}" for name in (value.get("participants") or {}).values()]
               if anchor.startswith("events/") else [])
    candidates = list(dict.fromkeys([anchor, *related, *unread[:8]]))
    chosen = {}
    for key in candidates:
        if key not in window.nodes or window.parts(key) != 1:
            continue
        trial = {**chosen, key: deepcopy(window.nodes[key])}
        if len(json.dumps(trial, ensure_ascii=False)) <= MAX_READ_CHARS:
            chosen = trial
        if len(chosen) >= 8:
            break
    window.visible = chosen


def _known_witness(proposal, ws, window):
    refs = proposal.get("witness_refs") or {}
    read = [window.nodes[key]["value"] for key in window.read]
    for group, ids in (("events", refs.get("event_ids", [])), ("relations", refs.get("relation_ids", []))):
        known = {window.nodes[key]["value"].get("id") for key in window.read if key.startswith(group + "/")}
        if any(identity not in known for identity in ids):
            raise ValueError("Witness includes unread " + group + "; inspect its original node")
    if any("entity/" + ref.get("entity", "") not in window.read for ref in refs.get("facts", [])):
        raise ValueError("Witness includes unread entity fields; inspect their original node")


def _binding(wp, ws, target, model, max_tokens):
    return {"strategy": STRATEGY, "wp": p._hash(wp), "world": p._hash(ws.to_dict()),
            "target": target, "model": model, "max_tokens": max_tokens,
            "source": p._hash([Path(__file__).read_text(encoding="utf-8"),
                               Path(p.__file__).read_text(encoding="utf-8")])}


def _session(wp, ws, *, target, model, max_tokens, report, chat_json=None, saved=(), save=None):
    window = context(wp, ws)
    _next_window(window)
    system = p.SYSTEM + INSTRUCTION + RULES
    orders, items, identities, intents = [], [], set(), []
    feedback, invalid, empty = None, 0, 0
    progress = ProgressGuard("Process proposal author")
    max_steps = min(1200, max(32, (target + 7) // 8 * 10 + len(window.nodes)))
    for number in range(max_steps):
        state = {"target_maximum": target, "accepted": len(orders),
                 "batch_maximum": min(BATCH_SIZE, target - len(orders)),
                 "recent_intents": intents[-16:], "action_error": feedback}
        messages = window.messages(system, state)
        window.mark_sent()
        call = {"model": model, "temperature": 0.3, "max_tokens": max_tokens,
                "retries": 3, "strict_json": True, "response_format": {"type": "json_object"}}
        entry = {"messages_hash": p._hash(messages), "call": call}
        if number < len(saved):
            old = saved[number]
            if any(old.get(key) != value for key, value in entry.items()):
                raise ValueError("Process checkpoint request changed")
            raw = deepcopy(old["raw_output"])
        else:
            if chat_json is None:
                raise ValueError("Incomplete process proposal transcript")
            raw = chat_json(p.STEP, messages, **call)
        if not isinstance(raw, dict) or "__error__" in raw:
            raise RuntimeError("Process batch execution failed; completed batches retained")
        entry["raw_output"] = deepcopy(raw)
        report["transcript"].append(entry)
        finished = False
        try:
            if window.control(raw):
                pass
            elif raw.get("action") == "finish":
                p._text(raw.get("reason"), "reason")
                finished = True
            elif raw.get("action") == "submit":
                original = {"proposals": deepcopy(raw.get("proposals")), "reason": raw.get("reason")}
                # Namespace local IDs without modifying the preserved model reply.
                if isinstance(original["proposals"], list):
                    for row in original["proposals"]:
                        if isinstance(row, dict) and isinstance(row.get("id"), str):
                            row["id"] = f"batch{number}/" + row["id"]
                batch_items, batch_orders = p._derive(original, ws, state["batch_maximum"])
                gained = 0
                for item in batch_items:
                    item["batch"] = number
                    order = item.get("order")
                    if order is not None:
                        try:
                            _known_witness(item["proposal"], ws, window)
                            identity = p._hash({k:v for k,v in item["proposal"].items() if k != "id"})
                            if identity in identities:
                                raise ValueError("Duplicate proposal across batches")
                            identities.add(identity)
                            orders.append(order); intents.append(item["proposal"]["intent"]); gained += 1
                        except (ValueError, TypeError, KeyError) as exc:
                            item.update(status="rejected", reason=str(exc)); item.pop("order", None)
                    items.append(item)
                empty = 0 if gained else empty + 1
                if empty >= 4:
                    raise ValueError("Four batches made no valid proposal progress; finish or inspect better evidence")
                _next_window(window)
                finished = len(orders) >= target
            else:
                raise ValueError("Use inspect/index/submit/finish")
            progress.observe((window.progress_marker(), len(orders)))
            feedback, invalid = None, 0
        except (ValueError, TypeError, KeyError) as exc:
            feedback, invalid = str(exc), invalid + 1
            entry["action_error"] = feedback
        report.update(items=deepcopy(items), orders=deepcopy(orders), calls_used=len(report["transcript"]))
        if save:
            save(report)
        if invalid >= 4:
            raise ValueError("Repeated invalid process author actions: " + feedback)
        if finished:
            if len(saved) > number + 1:
                raise ValueError("Extra process transcript entries")
            report["status"] = "completed"
            return
    raise ValueError("Process author step allowance exhausted; progress retained")


def propose(wp, ws, *, target, chat_json, model, max_tokens=8192, checkpoint_path=None):
    binding = _binding(wp, ws, target, model, max_tokens)
    saved = []
    path = Path(checkpoint_path) if checkpoint_path else None
    if path and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("binding") == binding:
            if previous.get("checkpoint_hash") != p._hash({k:v for k,v in previous.items() if k != "checkpoint_hash"}):
                raise ValueError("Process checkpoint content changed")
            saved = previous["transcript"]
    report = {"version": p.VERSION, "strategy": STRATEGY, "binding": binding,
              "status": "running", "transcript": [], "items": [], "orders": [], "calls_used": 0}
    def save(value):
        if path:
            from pipeline.run import _atomic_write_json
            frozen = deepcopy(value)
            frozen["checkpoint_hash"] = p._hash(frozen)
            _atomic_write_json(path, frozen)
    try:
        _session(wp, ws, target=target, model=model, max_tokens=max_tokens, report=report,
                 chat_json=chat_json, saved=saved, save=save)
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    save(report)
    return report


def validate(report, wp, ws):
    binding = report["binding"]
    if report.get("status") != "completed" or binding != _binding(wp, ws, binding["target"], binding["model"], binding["max_tokens"]):
        raise ValueError("Process batch binding/status changed")
    rebuilt = {"transcript": []}
    _session(wp, ws, target=binding["target"], model=binding["model"], max_tokens=binding["max_tokens"],
             report=rebuilt, saved=report["transcript"])
    if any(rebuilt[key] != report.get(key) for key in ("transcript", "orders", "items", "calls_used", "status")):
        raise ValueError("Process batch receipt changed")
    return deepcopy(rebuilt["orders"])
