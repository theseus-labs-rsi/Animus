"""Direct fact delivery and resumable local authoring in the original world gate.

Packing bounds transport only. The LLM owns grouping, publication times,
channels, non-public decisions, and requests for additional business context.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pipeline import disclosure as d

VERSION = "direct-disclosure/v2"
FACT_CHARS = 24000
CONTEXT_CHARS = 48000
INPUT_CHARS = 120000
NEW_GROUP_HEADROOM_CHARS = 8000
SEMANTIC_REPAIR_STEP = "world.disclosure.semantic_repair"
LEGACY_SEMANTIC_REPAIR_PROTOCOL = "acquisition-repair/v1"
SEMANTIC_REPAIR_PROTOCOL = "publication-repair/v2"

SEMANTIC_REPAIR_RULES = """你负责局部修正已经完成的公开安排。只处理输入列出的记录。
每条 replacement 必须保留原 record_index、session、refs 和 channel，只把 acquisition_context
改成与 exact_facts 完全一致的获取时间、途径和公开范围。不得补写未提供的事实，不得扩大 refs，
不得修改其他记录。返回 {"record_updates":[{"record_index":0,"record":{完整替换记录}}],
"reason":"修正依据"}。record_updates 必须逐条覆盖 requested_record_indices，且每个索引恰好一次。"""

PUBLICATION_REPAIR_RULES = """你负责局部修正已经完成的公开安排。只处理输入列出的记录。
每条 replacement 必须保留原 record_index、refs 和 channel；你可以修订公开期 session 和
acquisition_context。session 是这条记录实际公开的期，publication_scope.calendar 给出原日历。
若需要延后公开，应实际修改 session，不能只在获取说明中写后来补录而保留原公开期。
获取说明与 exact_facts 的确切版本、事实时点及所选公开期须能同时成立。事实/事件原值和时点
不在本次修改范围，不能通过公开说明把原事实的日期改掉。只在原窗口内安排，不得补写未提供的
事实、扩大 refs、改变其他记录或要求后续正文替你修正安排。固定通道仍受原编译器约束。
返回 {"record_updates":[{"record_index":0,"record":{完整替换记录}}],"reason":"修正依据"}。
record_updates 必须逐条覆盖 requested_record_indices，且每个索引恰好一次。"""


def _semantic_repair_rules(protocol):
    if protocol == LEGACY_SEMANTIC_REPAIR_PROTOCOL:
        return SEMANTIC_REPAIR_RULES
    if protocol == SEMANTIC_REPAIR_PROTOCOL:
        return PUBLICATION_REPAIR_RULES
    raise ValueError("Unsupported disclosure repair protocol")


class InputCapacityError(ValueError):
    pass


RULES = """
本轮直接交付一组事实原文；无需 index/inspect/finish 操作。
target_refs 是当前待处理的引用；exact_facts 是已提供的完整原文。程序分包仅控制输入大小，业务分组与公开期由你决定，可按自然业务过程跨对象/跨期合并，不按包统一公开时间。
remaining_index 仅供定位。若需要尚未提供的关联事实，返回 {"need_refs":["目录中的编号"],"reason":"需要哪些业务上下文"}；程序会直接补给原文，不要求手动按12条切分。未读目录不能当事实。
安排回复：{"records":[{"session":0,"refs":["f1"],"channel":"必须填写的获取渠道","acquisition_context":"必须说明谁何时怎样获知，覆盖所引全部版本"}],"undisclosed":[],"reason":"本组安排依据","next_refs":[]}。
每次 records 和 undisclosed 都是追加内容，至少解决一个 target_ref；本次没写完的引用会在下一轮继续，不必重写此前已接收内容。全部引用最终必须由你安排公开或明确不公开，不得为了凑覆盖默认隐藏。可同时安排 exact_facts 中其他尚未安排的引用。fixed_refs_already_public 无需重复处理。
accepted_records 是此前作者已完成的安排原文；本组不能覆盖它们。需要核对相关原事实时通过 need_refs 获取。next_refs 可选择下一组优先处理的引用，程序会在输入额度内交付。
validation_error 表示本次追加尚未接收。对照 previous_output 修正本次追加；此前 accepted_records 保留。程序不填充渠道、日期、业务理由或公开决定。
如确实需要修改此前记录，可另外给出 record_updates:[{"record_index":0,"record":{完整替换记录}}]，下标指 accepted_records；未指定的记录原样保留。新增安排仍放 records。替换不得使已有公开/不公开决定丢失；涉及整体公开策略的业务返修交原世界审阅后处理。
不要输出 action 或将工作状态摘要当成安排正文。全部事实处理后由程序合并并提交原世界业务审阅。
"""


def _size(value):
    return len(json.dumps(value, ensure_ascii=False))


def _entities(row):
    if row.get("entity"):
        return {row["entity"]}
    return set((row.get("event") or {}).get("participants", {}).values())


def _pack(ids, by_ref, limit):
    selected, size = [], 0
    for ref in dict.fromkeys(ids):
        row = by_ref[ref]
        n = _size(row)
        if n > limit:
            if not selected:
                raise ValueError(f"Fact {ref} exceeds direct disclosure input capacity")
            break
        if size + n > limit:
            break
        selected.append(ref)
        size += n
    return selected


def _messages(payload, raw, targets, context, repair, *, index_scope="all"):
    if index_scope not in {"all", "pending_only"}:
        raise ValueError("Unsupported disclosure index scope")
    by_ref = {row["ref"]: row for row in payload["catalogue"]}
    visible = set(targets) | set(context)
    pending = None
    if index_scope == "pending_only":
        pending = set(payload["refs_requiring_arrangement"]) - set(raw["undisclosed"])
        for record in raw["records"]:
            pending.difference_update(record["refs"])
    common = {k: deepcopy(v) for k, v in payload.items()
              if k not in {"catalogue", "fixed_records", "refs_requiring_arrangement"}}
    common["fixed_records"] = [{k: row[k] for k in ("session", "refs", "channel")} for row in payload["fixed_records"]]
    # Labels guide optional requests; only exact_facts authorize new assertions.
    index = [{"ref": ref, "entity": row.get("entity"), "field": row.get("field"),
              "event": (row.get("event") or {}).get("id"),
              "session": row.get("fact_session", (row.get("event") or {}).get("session"))}
             for ref, row in by_ref.items() if ref not in visible
             and (pending is None or ref in pending)]
    body = {"requirements": common, "target_refs": targets,
            "exact_facts": [by_ref[ref] for ref in dict.fromkeys(targets + context)],
            "remaining_index": index, "accepted_records": raw["records"],
            "accepted_undisclosed": raw["undisclosed"], "accepted_reason": raw["reason"],
            **deepcopy(repair or {})}
    if _size(body) > INPUT_CHARS:
        raise InputCapacityError("Disclosure input exceeds direct delivery capacity")
    system = d.SYSTEM.split("只输出三键 JSON：")[0]
    system = system.replace("catalogue 的 fact", "exact_facts 的 fact").replace(
        "refs_requiring_arrangement 才是你必须逐一安排的引用清单", "target_refs 是本组必须逐一安排的引用清单").replace(
        "catalogue 是完整只读定位目录，不代表所有条目都待你安排。",
        "exact_facts 是当前已提供的事实原文，remaining_index 只提供其他引用的定位信息。")
    return [{"role": "system", "content": system + RULES},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def _fit_messages(payload, raw, targets, context, repair, required=(), *, index_scope="all"):
    """Trim only optional neighbouring facts; explicit author requests stay exact."""
    required = list(dict.fromkeys(required))
    context = list(dict.fromkeys(required + list(context)))
    while True:
        try:
            return _messages(payload, raw, targets, context, repair, index_scope=index_scope), context
        except InputCapacityError:
            if len(context) <= len(required):
                raise
            context.pop()


def _select_batch(payload, raw, by_ref, pending, preferred):
    """Size the next fact group against the complete current message.

    Accepted decisions grow after every group, so a fixed fact-only batch
    size cannot guarantee that later messages fit the transport envelope.
    """
    targets = _pack(preferred + pending, by_ref, FACT_CHARS)
    while True:
        names = set().union(*(_entities(by_ref[ref]) for ref in targets))
        related = [ref for ref, row in by_ref.items()
                   if ref not in targets and _entities(row) & names]
        context = _pack(related, by_ref, CONTEXT_CHARS)
        try:
            messages, context = _fit_messages(payload, raw, targets, context, None,
                                               index_scope="pending_only")
        except InputCapacityError:
            if len(targets) == 1:
                raise
        else:
            if (len(messages[-1]["content"]) <= INPUT_CHARS - NEW_GROUP_HEADROOM_CHARS
                    or len(targets) == 1):
                return targets, context
        targets = targets[:max(1, len(targets) // 2)]


def _merge(ws, raw, output, targets, visible):
    if not isinstance(output, dict) or not {"records", "undisclosed", "reason"} <= set(output):
        raise ValueError("Return records, undisclosed and reason for this group")
    if not isinstance(output["records"], list) or not isinstance(output["undisclosed"], list):
        raise ValueError("records and undisclosed must be arrays")
    updates = output.get("record_updates", [])
    if not isinstance(updates, list):
        raise ValueError("record_updates must be an array")
    replacements, seen = [], set()
    for edit in updates:
        if not isinstance(edit, dict) or set(edit) != {"record_index", "record"}:
            raise ValueError("record_updates needs record_index and a complete record")
        index = edit["record_index"]
        if type(index) is not int or not 0 <= index < len(raw["records"]) or index in seen:
            raise ValueError("record_updates index must be existing and distinct")
        seen.add(index)
        replacements.append(edit["record"])
    refs = list(output["undisclosed"])
    for index, record in enumerate(output["records"] + replacements):
        if not isinstance(record, dict) or not isinstance(record.get("refs"), list):
            raise ValueError("Each record requires a refs array and full channel/acquisition_context")
        missing_fields = [key for key in ("channel", "acquisition_context")
                          if not isinstance(record.get(key), str) or not record[key].strip()]
        if missing_fields:
            raise ValueError(f"Current group records[{index}] missing required text fields: {missing_fields}")
        refs.extend(record["refs"])
    if any(not isinstance(ref, str) or ref not in visible for ref in refs):
        raise ValueError("Use only exact_facts references; request missing context via need_refs")
    if not set(targets).intersection(refs):
        raise ValueError("No pending target was resolved; append decisions for the supplied target_refs")
    reason = d._text(output["reason"], "reason")
    if raw["records"] or raw["undisclosed"]:
        reason = raw["reason"] + "\n" + reason
    records = deepcopy(raw["records"])
    for edit in updates:
        records[edit["record_index"]] = deepcopy(edit["record"])
    result = {"records": records + deepcopy(output["records"]),
              "undisclosed": raw["undisclosed"] + deepcopy(output["undisclosed"]),
              "reason": reason}
    try:
        d._compile(ws, result)
    except d.PlanFormatError as exc:
        if exc.details.get("code") != "reference_accounting" or any(exc.details.get(k)
                for k in ("overlap_refs", "unknown_refs", "duplicate_refs")):
            raise
    if set(d._missing_refs(ws, result)) - set(d._missing_refs(ws, raw)):
        raise ValueError("Updates lost earlier decisions; keep existing coverage while adding or replacing records")
    return result


def _semantic_repair_messages(payload, raw, record_indices, feedback, repair=None, *,
                              protocol=LEGACY_SEMANTIC_REPAIR_PROTOCOL):
    rules = _semantic_repair_rules(protocol)
    by_ref = {row["ref"]: row for row in payload["catalogue"]}
    records = [{"record_index": index, "record": deepcopy(raw["records"][index])}
               for index in record_indices]
    refs = list(dict.fromkeys(ref for row in records for ref in row["record"]["refs"]))
    body = {"requested_record_indices": record_indices, "records": records,
            "exact_facts": [deepcopy(by_ref[ref]) for ref in refs],
            "review_feedback": deepcopy(feedback), **deepcopy(repair or {})}
    if protocol == SEMANTIC_REPAIR_PROTOCOL:
        body["publication_scope"] = {"editable_fields": ["session", "acquisition_context"],
            "fixed_fields": ["refs", "channel"], "calendar": deepcopy(payload["calendar"])}
    if _size(body) > INPUT_CHARS:
        raise InputCapacityError("Semantic disclosure repair input exceeds direct delivery capacity")
    return [{"role": "system", "content": rules},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def _merge_semantic_repair(ws, raw, output, record_indices, *,
                           protocol=LEGACY_SEMANTIC_REPAIR_PROTOCOL):
    _semantic_repair_rules(protocol)
    if not isinstance(output, dict) or set(output) != {"record_updates", "reason"}:
        raise ValueError("Semantic repair requires exactly record_updates and reason")
    updates = output["record_updates"]
    if not isinstance(updates, list) or len(updates) != len(record_indices):
        raise ValueError("Semantic repair must update every requested record exactly once")
    by_index = {}
    for edit in updates:
        if not isinstance(edit, dict) or set(edit) != {"record_index", "record"}:
            raise ValueError("Each semantic repair needs record_index and a complete record")
        index, record = edit["record_index"], edit["record"]
        if index not in record_indices or index in by_index or not isinstance(record, dict):
            raise ValueError("Semantic repair used an unrequested or duplicate record index")
        original = raw["records"][index]
        fixed_fields = ("session", "refs", "channel") if protocol == LEGACY_SEMANTIC_REPAIR_PROTOCOL else ("refs", "channel")
        if any(record.get(key) != original.get(key) for key in fixed_fields):
            detail = "session, refs and channel" if protocol == LEGACY_SEMANTIC_REPAIR_PROTOCOL else "refs and channel"
            raise ValueError("Semantic repair must preserve " + detail)
        if (set(record) != set(original) or not isinstance(record.get("acquisition_context"), str)
                or not record["acquisition_context"].strip()):
            raise ValueError("Semantic repair must return the complete record with nonempty acquisition_context")
        by_index[index] = deepcopy(record)
    if set(by_index) != set(record_indices):
        raise ValueError("Semantic repair omitted a requested record")
    reason = d._text(output["reason"], "semantic repair reason")
    merged = deepcopy(raw)
    for index in record_indices:
        merged["records"][index] = by_index[index]
    merged["reason"] = raw["reason"] + "\n局部业务审阅返修:" + reason
    d._compile(ws, merged)
    return merged


def _replay(payload, ws, parts):
    raw = {"records": [], "undisclosed": [], "reason": "尚未完成安排"}
    all_refs = {r["ref"] for r in payload["catalogue"]}
    for part in parts:
        if part.get("kind") == "semantic_repair":
            indices = part.get("record_indices")
            if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                    or any(type(i) is not int or i < 0 or i >= len(raw["records"]) for i in indices)):
                raise ValueError("Semantic disclosure repair indices are stale or invalid")
            protocol = part.get("protocol", LEGACY_SEMANTIC_REPAIR_PROTOCOL)
            messages = _semantic_repair_messages(payload, raw, indices, part.get("feedback"), part.get("repair"),
                                                protocol=protocol)
            if part.get("messages_hash") != d._hash(messages):
                raise ValueError("Semantic disclosure repair input changed")
            raw = _merge_semantic_repair(ws, raw, part.get("output"), indices, protocol=protocol)
            continue
        targets, context = part["target_refs"], part["context_refs"]
        pending = set(d._missing_refs(ws, raw))
        if (not targets or len(set(targets)) != len(targets) or not set(targets) <= pending
                or not set(context) <= all_refs):
            raise ValueError("Disclosure checkpoint targets/context are stale or invalid")
        messages = _messages(payload, raw, targets, context, part.get("repair"),
                             index_scope=part.get("index_scope", "all"))
        if part["messages_hash"] != d._hash(messages):
            raise ValueError("Disclosure exact input or earlier accepted work changed")
        raw = _merge(ws, raw, part["output"], targets, set(targets) | set(context))
    return raw


def _replay_preserved_decisions(payload, ws, parts):
    """Replay accepted author decisions without rebuilding an obsolete envelope.

    Historical large plans can exceed a newer transport limit solely because
    every later request repeated all previously accepted prose.  Their exact
    outputs still remain checkable against the current world.  This path never
    invents or edits a decision; it verifies the saved plan hash, target order,
    visible references and every merge through the current compiler.
    """
    raw = {"records": [], "undisclosed": [], "reason": "尚未完成安排"}
    all_refs = {row["ref"] for row in payload["catalogue"]}
    for part in parts:
        if part.get("kind") == "semantic_repair":
            indices = part.get("record_indices")
            protocol = part.get("protocol", LEGACY_SEMANTIC_REPAIR_PROTOCOL)
            messages = _semantic_repair_messages(payload, raw, indices, part.get("feedback"), part.get("repair"),
                                                protocol=protocol)
            if part.get("messages_hash") != d._hash(messages):
                raise ValueError("Preserved semantic disclosure repair input changed")
            raw = _merge_semantic_repair(ws, raw, part.get("output"), indices, protocol=protocol)
            continue
        targets, context = part["target_refs"], part["context_refs"]
        pending = set(d._missing_refs(ws, raw))
        if (not targets or len(set(targets)) != len(targets)
                or not set(targets) <= pending or not set(context) <= all_refs):
            raise ValueError("Preserved disclosure targets/context are stale or invalid")
        raw = _merge(ws, raw, part["output"], targets, set(targets) | set(context))
    return raw


def _binding(wp, ws, payload, model):
    return {"version": VERSION, "truth_hash": d._hash(d._world(ws)), "wp_hash": d._hash(wp),
            "input_hash": d._hash(payload), "call_hash": d._hash(d._call(model)),
            "protocol_hash": d._hash(RULES), "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": d._implementations(), "limits": [FACT_CHARS, CONTEXT_CHARS, INPUT_CHARS]}


def _save(path, state):
    if path is None:
        return
    state["checkpoint_hash"] = d._hash({k: v for k, v in state.items() if k != "checkpoint_hash"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _migrate_checkpoint(directory, current_path, binding, payload, ws):
    """Copy an exactly replayable earlier envelope into the new implementation.

    The original checkpoint remains untouched.  Business outputs and the
    historical message hashes are checked before any continuation is allowed.
    """
    source_fields = ("truth_hash", "wp_hash", "input_hash", "call_hash", "protocol_hash")
    matches = []
    for old_path in sorted(Path(directory).glob("02_disclosure_checkpoint_*.json")):
        if old_path == current_path:
            continue
        source = old_path.read_bytes()
        saved = json.loads(source)
        old_binding = saved.get("binding") or {}
        if any(old_binding.get(key) != binding[key] for key in source_fields):
            continue
        if old_binding.get("limits") != binding["limits"]:
            continue
        if saved.get("checkpoint_hash") != d._hash(
                {key: value for key, value in saved.items() if key != "checkpoint_hash"}):
            raise ValueError("Matching disclosure checkpoint changed; refusing migration")
        _replay(payload, ws, saved["parts"])
        matches.append((old_path, source, saved))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Multiple matching disclosure checkpoints; choose a source explicitly")
    old_path, source, saved = matches[0]
    saved["binding"] = binding
    saved["migrated_from"] = {"checkpoint": old_path.name,
                              "sha256": hashlib.sha256(source).hexdigest(),
                              "accepted_groups": len(saved["parts"])}
    _save(current_path, saved)
    return saved


def _plan(wp, ws, task, feedback, payload, state, model):
    try:
        raw = _replay(payload, ws, state["parts"])
    except InputCapacityError:
        raw = _replay_preserved_decisions(payload, ws, state["parts"])
    plan = d._compile(ws, raw)
    plan["binding"].update(wp_hash=d._hash(wp), task_hash=d._hash(task), feedback_hash=d._hash(feedback),
                           messages_hash=d._hash(state["parts"]), call_hash=d._hash(d._call(model)),
                           protocol_hash=d._hash(RULES))
    plan.update(strategy=VERSION, parts=deepcopy(state["parts"]), author_input=deepcopy(payload),
                source_inputs={"whitepaper": deepcopy(wp), "task": deepcopy(task), "feedback": deepcopy(feedback)},
                call=d._call(model), batch_binding=_binding(wp, ws, payload, model))
    plan["plan_hash"] = d._hash(plan)
    return plan


def repair(wp, ws, tracer, review):
    """Repair only disclosure records explicitly rejected by the bound world review."""
    import config
    report = {"version": "direct-disclosure-semantic-repair/v2", "status": "not_needed",
              "logical_calls": 0, "attempts": []}
    try:
        plan = ws.disclosure
        if plan.get("strategy") not in ("direct-disclosure/v1", "direct-disclosure/v2"):
            raise ValueError("Targeted semantic repair requires a direct-disclosure plan")
        if any(part.get("kind") == "semantic_repair" for part in plan.get("parts", [])):
            raise ValueError("Targeted semantic disclosure repair was already attempted for this plan")
        errors = validate(ws)
        if errors:
            raise ValueError(errors[0]["message"])
        issues = [row for row in (review.get("issues") or []) if isinstance(row, dict)]
        # An unresolved record is author-repairable only when the reviewer has
        # explicitly requested disclosure repair and every reported issue has a
        # repair disposition.  A mixed review (for example, one genuinely
        # unresolved issue) remains conservative and repairs only rows named
        # ``repair``.  This lets the author clarify a bounded set without
        # converting an unresolved business judgment into a program decision.
        include_unresolved = bool(issues) and all(
            row.get("disposition") == "repair" for row in issues
        ) and (review.get("repair_targets") or {}).get("disclosure") is True
        allowed_statuses = {"repair", "unresolved"} if include_unresolved else {"repair"}
        rows = [row for row in (review.get("disclosure_reviews") or [])
                if isinstance(row, dict) and row.get("status") in allowed_statuses]
        record_ids = list(dict.fromkeys(row.get("record_id") for row in rows))
        if not record_ids:
            return report
        if any(not isinstance(record_id, str) or not record_id.startswith("d")
                or not record_id[1:].isdigit() for record_id in record_ids):
            raise ValueError("World review repair targets need direct disclosure record IDs")
        indices = [int(record_id[1:]) - 1 for record_id in record_ids]
        source = plan["source_inputs"]
        payload = d._payload(source["whitepaper"], ws, source["task"], source["feedback"])
        raw = _replay_preserved_decisions(payload, ws, plan["parts"])
        if raw != plan.get("raw_output"):
            raise ValueError("Published disclosure does not replay before semantic repair")
        issue_rows = [deepcopy(row) for row in issues
                      if isinstance(row, dict) and row.get("disposition") == "repair"]
        feedback = {"status": review.get("status"), "issues": issue_rows,
                    "disclosure_reviews": deepcopy(rows)}
        repair_context = None
        for number in range(3):
            messages = _semantic_repair_messages(payload, raw, indices, feedback, repair_context,
                                                protocol=SEMANTIC_REPAIR_PROTOCOL)
            entry = {"number": number + 1, "messages_hash": d._hash(messages), "status": "started"}
            report["attempts"].append(entry)
            report["logical_calls"] += 1
            output = tracer.chat_json(SEMANTIC_REPAIR_STEP, messages,
                model=config.STRUCTURE_MODEL, temperature=0.1, max_tokens=d.MAX_TOKENS,
                retries=3, strict_json=True, response_format={"type": "json_object"})
            entry["raw_output"] = deepcopy(output)
            if isinstance(output, dict) and "__error__" in output:
                raise RuntimeError("Semantic disclosure repair execution failed: " + str(output["__error__"]))
            try:
                _merge_semantic_repair(ws, raw, output, indices, protocol=SEMANTIC_REPAIR_PROTOCOL)
            except (ValueError, KeyError, TypeError, d.PlanFormatError) as exc:
                entry.update(status="local_repair_required", error=str(exc))
                repair_context = {"previous_output": deepcopy(output), "validation_error": str(exc)}
                continue
            part = {"kind": "semantic_repair", "protocol": SEMANTIC_REPAIR_PROTOCOL, "record_indices": indices,
                    "feedback": feedback, "repair": repair_context,
                    "messages_hash": d._hash(messages), "output": deepcopy(output)}
            rebuilt = _plan(source["whitepaper"], ws, source["task"], source["feedback"], payload,
                            {"parts": [*deepcopy(plan["parts"]), part]}, config.STRUCTURE_MODEL)
            ws.disclosure = rebuilt
            entry["status"] = "accepted"
            report.update(status="ready", repaired_record_ids=record_ids,
                          plan_hash=rebuilt["plan_hash"])
            return report
        raise ValueError("Targeted semantic disclosure repair remained invalid after three attempts")
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        return report


def _validate_preserved_plan(plan, wp, ws, task, feedback, payload):
    """Validate saved business decisions under the current world/compiler.

    This deliberately ignores only the old execution-envelope fingerprints
    (Python implementation, transport limits, and author protocol hash) and
    their derived outer plan hash.  The original plan hash, model outputs,
    source inputs, exact reference accounting and compiled public records must
    all still match.
    """
    saved = {key: deepcopy(value) for key, value in plan.items() if key != "plan_hash"}
    if plan.get("plan_hash") != d._hash(saved):
        raise ValueError("Preserved disclosure plan hash changed")
    raw = _replay_preserved_decisions(payload, ws, plan["parts"])
    if raw != plan.get("raw_output"):
        raise ValueError("Preserved disclosure outputs do not reproduce the published plan")
    compiled = d._compile(ws, raw)
    for key in ("version", "records", "undisclosed", "catalogue", "raw_output"):
        if compiled[key] != plan.get(key):
            raise ValueError("Preserved disclosure differs from current compiled world: " + key)
    for key in ("truth_hash", "catalogue_hash"):
        if plan.get("binding", {}).get(key) != compiled["binding"][key]:
            raise ValueError("Preserved disclosure world binding is stale: " + key)
    model = plan["call"]["params"]["model"]
    expected = {"truth_hash": d._hash(d._world(ws)), "wp_hash": d._hash(wp),
                "input_hash": d._hash(payload), "call_hash": d._hash(d._call(model))}
    if any(plan.get("batch_binding", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Preserved disclosure source binding is stale")


def author(wp, ws, tracer, task=None, feedback=None, checkpoint_dir=None):
    import config
    model = config.STRUCTURE_MODEL
    report = {"version": VERSION, "strategy": VERSION, "status": "error", "logical_calls": 0,
              "caller_entered": False, "attempts": []}
    state, path = None, None
    try:
        if not d.enabled(wp):
            raise ValueError("Public disclosure author was not enabled")
        payload = d._payload(wp, ws, task, feedback)
        binding = _binding(wp, ws, payload, model)
        state = {"binding": binding, "parts": [], "attempts": [], "status": "building"}
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / ("02_disclosure_checkpoint_" + d._hash(binding)[:20] + ".json")
            report["checkpoint"] = str(path)
            if not path.exists():
                migrated = _migrate_checkpoint(checkpoint_dir, path, binding, payload, ws)
                if migrated is not None:
                    report["migrated_checkpoint"] = migrated["migrated_from"]
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if (loaded.get("binding") != binding or loaded.get("checkpoint_hash") !=
                            d._hash({k: v for k, v in loaded.items() if k != "checkpoint_hash"})):
                        raise ValueError("Disclosure checkpoint changed; refusing reuse")
                    _replay(payload, ws, loaded["parts"])
                except Exception:
                    state, path = None, None
                    raise
                state = loaded
        raw = _replay(payload, ws, state["parts"])
        by_ref = {row["ref"]: row for row in payload["catalogue"]}
        attempt_limit = getattr(config, "DISCLOSURE_FORMAT_ATTEMPTS", 4)
        if type(attempt_limit) is not int or not 2 <= attempt_limit <= 6:
            raise ValueError("Disclosure format attempts must be between 2 and 6")
        next_refs = state["parts"][-1]["output"].get("next_refs", []) if state["parts"] else []
        while d._missing_refs(ws, raw):
            missing = set(d._missing_refs(ws, raw))
            pending = [ref for ref in by_ref if ref in missing]
            preferred = [ref for ref in next_refs if ref in pending] if isinstance(next_refs, list) else []
            # Transport order, not a business partition. Author may reprioritize.
            targets, context = _select_batch(payload, raw, by_ref, pending, preferred)
            required_context = []
            repair = None
            for number in range(attempt_limit):
                messages, context = _fit_messages(payload, raw, targets, context, repair,
                                                  required_context, index_scope="pending_only")
                entry = {"part": len(state["parts"]), "attempt": number+1,
                         "messages_hash": d._hash(messages), "status": "started"}
                state["attempts"].append(entry)
                report["attempts"].append(entry)
                report.update(caller_entered=True, logical_calls=report["logical_calls"]+1)
                _save(path, state)
                try:
                    output = tracer.chat_json(d.STEP, messages, **d._call(model)["params"])
                    entry["raw_output"] = deepcopy(output)
                    if isinstance(output, dict) and "__error__" in output:
                        raise RuntimeError("Original author execution failed: " + str(output["__error__"]))
                except Exception as exc:
                    entry.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
                    raise
                try:
                    if isinstance(output, dict) and "need_refs" in output:
                        requested = output["need_refs"]
                        if (not isinstance(requested, list) or not requested or
                                any(not isinstance(ref, str) or ref not in by_ref for ref in requested)):
                            raise ValueError("need_refs must contain existing catalogue references")
                        requested = list(dict.fromkeys(ref for ref in requested if ref not in targets))
                        if set(requested) <= set(context):
                            raise ValueError("Requested facts are already supplied; author this group")
                        if _size([by_ref[ref] for ref in requested]) > CONTEXT_CHARS:
                            raise ValueError("Additional facts exceed context capacity; request the needed related subset")
                        # Explicit business requests take precedence over the
                        # automatically supplied neighbouring facts.
                        required = list(dict.fromkeys(required_context + requested))
                        if _size([by_ref[ref] for ref in required]) > CONTEXT_CHARS:
                            raise ValueError("Requested context exceeds capacity; finish this group before requesting more")
                        proposed = _pack(required + context, by_ref, CONTEXT_CHARS)
                        _, context = _fit_messages(payload, raw, targets, proposed, None,
                                                   required, index_scope="pending_only")
                        required_context = required
                        repair = None
                        entry["status"] = "context_supplied"
                        continue
                    merged = _merge(ws, raw, output, targets, set(targets) | set(context))
                except (ValueError, KeyError, TypeError) as exc:
                    entry.update(status="local_repair_required", error=str(exc))
                    repair = {"previous_output": deepcopy(output), "validation_error": str(exc)}
                    continue
                part = {"target_refs": targets, "context_refs": context, "repair": repair,
                        "index_scope": "pending_only",
                        "messages_hash": d._hash(messages), "output": deepcopy(output)}
                state["parts"].append(part)
                raw, next_refs = merged, output.get("next_refs", [])
                entry.update(status="accepted", remaining_refs=len(d._missing_refs(ws, raw)))
                _save(path, state)
                break
            else:
                raise ValueError("Current disclosure group remains unresolved; prior groups retained in checkpoint")
        plan = _plan(wp, ws, task, feedback, payload, state, model)
        ws.disclosure = plan
        state["status"] = "completed"
        report.update(status="ready", plan_hash=plan["plan_hash"], binding=plan["binding"])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        if state is not None:
            report["accepted_groups"] = len(state["parts"])
            if report["status"] != "ready":
                state["status"] = "incomplete"
            _save(path, state)
    return report


def validate(ws):
    try:
        plan = ws.disclosure
        source = plan["source_inputs"]
        wp, task, feedback = source["whitepaper"], source["task"], source["feedback"]
        payload = d._payload(wp, ws, task, feedback)
        try:
            rebuilt = _plan(wp, ws, task, feedback, payload, {"parts": plan["parts"]},
                            plan["call"]["params"]["model"])
        except InputCapacityError:
            # Transport limits may become tighter after a run.  Do not force a
            # new LLM to rewrite already accepted business decisions merely to
            # refresh an oversized audit envelope.  Verify the immutable old
            # receipt and replay its exact outputs with the current compiler.
            _validate_preserved_plan(plan, wp, ws, task, feedback, payload)
            return []
        if rebuilt != plan:
            differences = {key for key in set(rebuilt) | set(plan) if rebuilt.get(key) != plan.get(key)}
            binding_differences = {key for key in set(rebuilt.get("binding", {}))
                                   | set(plan.get("binding", {}))
                                   if rebuilt.get("binding", {}).get(key)
                                   != plan.get("binding", {}).get(key)}
            batch_differences = {key for key in set(rebuilt.get("batch_binding", {}))
                                 | set(plan.get("batch_binding", {}))
                                 if rebuilt.get("batch_binding", {}).get(key)
                                 != plan.get("batch_binding", {}).get(key)}
            if (differences <= {"binding", "batch_binding", "plan_hash"}
                    and binding_differences <= {"protocol_hash", "implementation_hashes"}
                    and batch_differences <= {
                        "implementation", "dependencies", "limits", "protocol_hash"}):
                _validate_preserved_plan(plan, wp, ws, task, feedback, payload)
                return []
            raise ValueError("Disclosure differs from exact author inputs/outputs or current world")
        return []
    except Exception as exc:
        return [{"code": "missing_or_stale_disclosure", "message": f"{type(exc).__name__}: {exc}"}]
