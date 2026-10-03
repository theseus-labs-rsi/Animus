"""Public observations inside the original world stage; no second truth timeline.

The author chooses plausible acquisition and publication. Code resolves exact
frozen versions and checks identity/coverage, never business chronology by name.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "original-public-disclosure/v5"
STEP = "world.disclosure_plan"
MAX_TOKENS = 16384
FORMAT_REPAIR_INSTRUCTION = "仅修复上次输出的结构和引用对账，返回完整三键JSON。按本轮目录与明确列表逐项检查；保持已合法安排和业务理由，遗漏项仍由你据原场景选择公开安排或未披露。不能改世界、fixed_records、增加新事实，也不能把错误当语义通过意见。"
PATCH_REPAIR_INSTRUCTION = "只修改出错的记录；未指定修改的记录由程序原样保留。根据原场景为遗漏引用选择公开时间与途径，或明确未披露。record_updates 的 record_index 从0开始，record 为该条的完整替换；append_records 为新增记录；undisclosed 为 null 表示保留原列表，否则替换完整列表。不得默认把遗漏项藏入未披露，不得增加事实。只输出四键 JSON：{\"record_updates\":[{\"record_index\":0,\"record\":{\"session\":0,\"refs\":[\"f1\"],\"channel\":\"渠道\",\"acquisition_context\":\"场景中何时如何获知\"}}],\"append_records\":[],\"undisclosed\":null,\"reason\":\"本次修改理由\"}。对照 validation_error 和 missing_refs_all 检查全部遗漏及重叠，编号和数组是形状示例。"
SYSTEM = """你是原世界的作者，现在为已完成的合成世界安排公开记录。
不改世界真值、事件、白皮书或既有来源声明。按本场景合理交代谁何时、通过什么途径获得哪些信息。
后台存在的事实不等于读者开局知道；也不把所有信息的获取时点统一绑到某个接收字段。
catalogue 的 fact 引用是一条确切历史版本；event 引用公开该事件列明的参与者、效果及已有因果链接，不自动展开参与者其他属性或链接父事件。
如正文需说明链接所指事件，须显式安排相应 event 引用；未公开的父事件不能由作者补造内容。因果链接本身仍须结合场景审阅，不能凭标签推出一般规则。
record.session 是公开文档所在期，acquisition_context 自然说明获取过程及其时点；允许后出的记录回顾过去。
未来日期可作为当时已知的文件标注或计划，但不能据此声称未来事件已发生；这由场景证据决定，不按词语判断。
同一记录的 refs 与 acquisition_context 共同构成公开主张。写完每组后逐一核对所引对象、确切版本和原时点，确认获取说明能够同时解释全部引用，不能只围绕其中一个对象或事件写说明。若说明保留未完成状态，引用却指向后来完成的结果，应重选引用、调整公开期或明确有依据的获取含义，不能要求后续正文同时兑现冲突说法。
提前取得未来生效文件、已知计划与延后回顾均可合理；说明该记录当时实际知道的内容和获取途径，保留事实成立时间与公开时间的区别。不按时间先后一刀切，也不为解释提前公开凭空补造已完成结果。
可多次获取或披露同一版本，可在同一记录对照新旧版本；每次记录独立。不能把历史值换成当前值。
fixed_records 是已有公开通道的固定安排，由程序原样加入；fixed_refs_already_public 明列所有已固定公开引用，其中也有普通 f 编号，绝不能放入 undisclosed，也无需重复生成。
refs_requiring_arrangement 才是你必须逐一安排的引用清单：每个编号须进入 records 或 undisclosed；确实不公开的可放 undisclosed，不补造事实凑公开量。
catalogue 是完整只读定位目录，不代表所有条目都待你安排。不能移动固定通道；来源主张不自动等于真值，敏感项在材料中出现也不授权答案复述。
已由 fixed_records 引用的条目可以不重复，不能同时称未披露。按一次自然获取/记录把相关引用合并成组，通常每期少量记录；不把每个事实拆成一条记录，也不强设组数。
获取说明不能凭空新增业务任命、采用、批准或事件结果；不把将来问题或金标作为公开安排依据。
只输出三键 JSON：{"records":[{"session":0,"refs":["f1"],"channel":"获取与记录渠道","acquisition_context":"谁在何时经何途径得知，正文应如何清楚交代"}],"undisclosed":["f2"],"reason":"安排理由及未披露范围"}。
数组仅为形状示例，不能照抄不存在的编号。所有 refs 复制本轮 catalogue；不输出新事实值。"""


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def enabled(wp):
    return (wp.get("quality_contract") or {}).get("public_disclosure") is True


def _world(ws):
    value = deepcopy(ws.to_dict())
    value.pop("disclosure", None)
    return value


def _implementations():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("disclosure.py", "world_state.py", "seed_world.py", "seed_pack.py", "seed_v2.py", "world_context.py")}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be nonempty text")
    return value


def _session(ws, value):
    if type(value) is not int or value not in ws.sessions():
        raise ValueError("Disclosure session is outside this world")
    return value


def catalogue(ws):
    """Full private reference directory for the world author/reviewer only."""
    rows = []
    for name in sorted(ws.entities):
        for field in sorted(ws.entities[name]):
            for index, op in enumerate(ws.entities[name][field]._sorted()):
                rows.append({"ref": f"f{len(rows)+1}", "kind": "fact",
                    "entity": name, "entity_type": ws.entity_types.get(name),
                    "field": field, "op_index": index, "fact_session": op.session,
                    "fact_date": op.date, "operation": op.op,
                    "value": deepcopy(op.value), "stopped": op.op in ("EXPIRE", "DELETE")})
    for index, event in enumerate(ws.events):
        # Preserve the existing link, without recursively expanding its parent.
        # The author must separately select parent events needed by the account.
        item = deepcopy(event)
        rows.append({"ref": f"e{index+1}", "kind": "event", "event": item})
    for index, claim in enumerate(ws.conflicts):
        for side in ("authoritative", "rumor"):
            if claim.get(side + "_value") in (None, ""):
                continue
            rows.append({"ref": f"c{index+1}_{side}", "kind": "source_claim",
                "session": claim["session"], "date": claim.get("date"),
                "entity": claim["entity"], "field": claim["field"],
                "source": claim.get(side + "_source"),
                "value": deepcopy(claim[side + "_value"]), "side": side,
                "fixed_channel": "L5"})
    for index, item in enumerate(ws.rule_instances):
        if item.get("surface_action") is not None:
            public = {k: deepcopy(item[k]) for k in
                      ("inst_id", "trigger_field", "x", "unit", "surface_action", "session") if k in item}
            rows.append({"ref": f"r{index+1}", "kind": "rule_instance", **public, "fixed_channel": "L9"})
    for index, item in enumerate(ws.sensitive):
        if item.get("value"):
            public = {k: deepcopy(item[k]) for k in ("entity", "field", "stype", "value", "session", "witness_field", "witness_value") if k in item}
            rows.append({"ref": f"s{index+1}", "kind": "sensitive_record", **public, "fixed_channel": "L10"})
    return rows


def _fixed(ws, inventory):
    rows = []
    for item in inventory:
        if not item.get("fixed_channel"):
            continue
        refs = [item["ref"]]
        if item.get("side") == "authoritative":
            matches = [f for f in inventory if f["kind"] == "fact"
                       and (f["entity"], f["field"]) == (item["entity"], item["field"])
                       and f["fact_session"] <= item["session"]]
            latest = max(matches, key=lambda f: (f["fact_date"], f["fact_session"], f["op_index"]), default=None)
            if latest is None or latest["stopped"] or latest["value"] != item["value"]:
                raise ValueError("Fixed authoritative claim does not match its canonical version")
            refs.append(latest["ref"])
        if item.get("fixed_channel") == "L10" and item.get("witness_field"):
            matches = [f for f in inventory if f["kind"] == "fact"
                       and (f["entity"], f["field"]) == (item["entity"], item["witness_field"])
                       and f["fact_session"] <= item["session"]]
            latest = max(matches, key=lambda f: (f["fact_date"], f["fact_session"], f["op_index"]), default=None)
            if latest is None or latest["stopped"] or latest["value"] != item.get("witness_value"):
                raise ValueError("Fixed L10 public witness does not match its canonical version")
            refs.append(latest["ref"])
        channel = {"L5": item.get("source") or "来源记录", "L9": "业务执行记录", "L10": "登记记录"}[item["fixed_channel"]]
        acquisition = (f"本期通过{channel}记录该来源对所列事项的说法，保留其来源归属。"
                       if item["fixed_channel"] == "L5" else "本期登记所列既有记录，正文沿用其原有记录形式及适用范围。")
        rows.append({"id": "fixed_" + item["ref"], "session": _session(ws, item["session"]),
            "refs": refs, "channel": channel, "fixed_channel": item["fixed_channel"],
            "acquisition_context": acquisition,
            "fixed": True})
    return rows


class PlanFormatError(ValueError):
    """Only returned-output shape/reference defects permit one author repair."""
    def __init__(self, code, message, **details):
        self.details = {"code": code, "message": message, **details}
        super().__init__(json.dumps(self.details, ensure_ascii=False, sort_keys=True))


def _compile(ws, raw):
    # Original-world/fixed-channel failures are preconditions, not format repairs.
    inventory = catalogue(ws)
    by_ref = {row["ref"]: row for row in inventory}
    fixed = _fixed(ws, inventory)
    if not isinstance(raw, dict) or not {"records", "undisclosed", "reason"} <= set(raw):
        raise PlanFormatError("output_shape", "Disclosure output needs records, undisclosed and reason")
    try:
        _text(raw["reason"], "reason")
    except ValueError as exc:
        raise PlanFormatError("output_shape", str(exc)) from exc
    if not isinstance(raw["records"], list) or not isinstance(raw["undisclosed"], list):
        raise PlanFormatError("output_shape", "Disclosure arrays required")
    fixed_refs = {ref for row in fixed for ref in row["refs"]}
    records, mentioned = [], set(fixed_refs)
    for number, row in enumerate(raw["records"], 1):
        if not isinstance(row, dict):
            raise PlanFormatError("record_shape", "Disclosure record must be an object", record_index=number-1)
        try:
            session = _session(ws, row.get("session"))
            channel = _text(row.get("channel"), "channel")
            acquisition = _text(row.get("acquisition_context"), "acquisition_context")
        except ValueError as exc:
            raise PlanFormatError("record_shape", str(exc), record_index=number-1) from exc
        refs = row.get("refs")
        if not isinstance(refs, list) or not refs or any(not isinstance(x, str) for x in refs):
            raise PlanFormatError("record_refs_shape", "refs must be a nonempty list of strings", record_index=number-1)
        unknown, duplicate = sorted(set(refs)-set(by_ref)), sorted({x for x in refs if refs.count(x)>1})
        if unknown or duplicate:
            raise PlanFormatError("record_refs", "Use existing distinct refs", record_index=number-1, unknown_refs=unknown, duplicate_refs=duplicate)
        if any(by_ref[ref].get("fixed_channel") and by_ref[ref]["session"] != session for ref in refs):
            raise PlanFormatError("fixed_timing", "Fixed source-channel timing cannot be moved", record_index=number-1,
                refs=[ref for ref in refs if by_ref[ref].get("fixed_channel") and by_ref[ref]["session"] != session])
        records.append({"id": f"d{number}", "session": session, "refs": deepcopy(refs),
            "channel": channel, "acquisition_context": acquisition, "fixed": False})
        mentioned.update(refs)
    hidden = raw["undisclosed"]
    if any(not isinstance(ref, str) for ref in hidden):
        raise PlanFormatError("undisclosed_shape", "undisclosed must contain only reference strings")
    missing = sorted(set(by_ref) - mentioned - set(hidden))
    overlap = sorted(mentioned.intersection(hidden))
    unknown = sorted(set(hidden)-set(by_ref))
    duplicate = sorted({x for x in hidden if hidden.count(x)>1})
    if missing or overlap or unknown or duplicate:
        raise PlanFormatError("reference_accounting", "Every nonfixed ref needs an explicit public or undisclosed arrangement; fixed refs are already public",
            missing_refs=missing, overlap_refs=overlap, fixed_overlap_refs=sorted(fixed_refs.intersection(hidden)),
            unknown_refs=unknown, duplicate_refs=duplicate)
    records.extend(fixed)
    records.sort(key=lambda row: (row["session"], row["id"]))
    return {"version": VERSION, "records": records, "undisclosed": deepcopy(hidden),
            "catalogue": inventory, "raw_output": deepcopy(raw),
            "binding": {"truth_hash": _hash(_world(ws)), "catalogue_hash": _hash(inventory),
                        "system_hash": _hash(SYSTEM), "implementation_hashes": _implementations()}}


def _feedback_projection(feedback):
    """Keep semantic content; strip only the known original review's audit envelope.

    Unknown feedback shapes/fields remain intact. Previous original opinions
    are projected recursively, while raw opinions, located evidence, author
    responses and any extension business fields are never summarized/truncated.
    """
    if not (isinstance(feedback, dict) and isinstance(feedback.get("version"), str)
            and feedback["version"].startswith("original-world-semantics/")):
        return deepcopy(feedback)
    # These four fields duplicate caller parameters or the full world/task
    # already supplied independently, not additional semantic review findings.
    envelope = {"messages", "input_snapshot", "binding", "call", "transcript"}
    projected = {k: deepcopy(v) for k, v in feedback.items() if k not in envelope}
    if "previous" in projected:
        projected["previous"] = _feedback_projection(feedback["previous"])
    projected["source_hash"] = _hash(feedback)
    return projected


def _payload(wp, ws, task_input, feedback, format_repair=None):
    inventory = catalogue(ws)
    fixed = _fixed(ws, inventory)
    fixed_refs = {ref for row in fixed for ref in row["refs"]}
    task = task_input or {}
    projected_task = {k: deepcopy(task[k]) for k in ("description", "objective", "instructions", "few_shot") if k in task}
    if "few_shot" in projected_task:
        projected_task["few_shot"] = [{k: deepcopy(x[k]) for k in ("title", "content", "date", "doc_type") if k in x}
                                       for x in projected_task["few_shot"]]
    paper = {k: deepcopy(wp[k]) for k in ("domain_profile", "world_blueprint", "shared_world_spec") if k in wp}
    from pipeline.seed_world import seed_business_context
    paper["seed_requirements"] = seed_business_context(wp)
    if wp.get("business_instance_plan"):
        paper["business_instance_plan"] = deepcopy(wp["business_instance_plan"])
    payload = {"task": projected_task, "whitepaper": paper, "catalogue": inventory,
            "fixed_records": fixed, "fixed_refs_already_public": sorted(fixed_refs),
            "refs_requiring_arrangement": [row["ref"] for row in inventory if row["ref"] not in fixed_refs],
            "calendar": [{"session": s, "date": ws.date_of_session(s)} for s in ws.sessions()],
            "feedback": _feedback_projection(feedback)}
    if format_repair is not None:
        payload["format_repair"] = deepcopy(format_repair)
    return payload


def _messages(payload):
    system = SYSTEM
    if payload.get("format_repair", {}).get("instruction") == PATCH_REPAIR_INSTRUCTION:
        system = SYSTEM.split("只输出三键 JSON：")[0] + PATCH_REPAIR_INSTRUCTION
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]


def _apply_author_patch(previous, raw):
    """Mechanical edits only; the model supplies every changed disclosure decision."""
    if not isinstance(raw, dict) or "record_updates" not in raw:
        return deepcopy(raw)  # A full replacement remains a valid author response.
    if set(raw) != {"record_updates", "append_records", "undisclosed", "reason"}:
        raise PlanFormatError("patch_shape", "Patch requires exactly record_updates, append_records, undisclosed and reason")
    if not isinstance(previous, dict) or not isinstance(previous.get("records"), list):
        raise PlanFormatError("patch_precondition", "Previous records are invalid; return a full three-key replacement")
    if not isinstance(raw["record_updates"], list) or not isinstance(raw["append_records"], list):
        raise PlanFormatError("patch_shape", "Patch operations must be arrays")
    result = deepcopy(previous)
    seen = set()
    for edit in raw["record_updates"]:
        if not isinstance(edit, dict) or set(edit) != {"record_index", "record"}:
            raise PlanFormatError("patch_shape", "Each update needs record_index and a complete record")
        index = edit["record_index"]
        if type(index) is not int or not 0 <= index < len(result["records"]) or index in seen:
            raise PlanFormatError("patch_index", "Update indices must be existing and distinct")
        seen.add(index)
        result["records"][index] = deepcopy(edit["record"])
    result["records"].extend(deepcopy(raw["append_records"]))
    if raw["undisclosed"] is not None:
        result["undisclosed"] = deepcopy(raw["undisclosed"])
    result["reason"] = _text(raw["reason"], "patch reason")
    return result


def _missing_refs(ws, raw):
    inventory = catalogue(ws)
    mentioned = {ref for record in _fixed(ws, inventory) for ref in record["refs"]}
    if isinstance(raw, dict):
        for record in raw.get("records", []) if isinstance(raw.get("records"), list) else []:
            if isinstance(record, dict) and isinstance(record.get("refs"), list):
                mentioned.update(ref for ref in record["refs"] if isinstance(ref, str))
        if isinstance(raw.get("undisclosed"), list):
            mentioned.update(ref for ref in raw["undisclosed"] if isinstance(ref, str))
    return sorted({row["ref"] for row in inventory} - mentioned)


def _call(model):
    return {"step": STEP, "params": {"model": _text(model, "author model"), "temperature": 0.2,
            "max_tokens": MAX_TOKENS, "retries": 3, "strict_json": True,
            "response_format": {"type": "json_object"}}}


def compile_plan(wp, ws, raw, task_input=None, feedback=None, model="offline-fixture", format_repair=None):
    """Pure compilation; produces no execution claim and never invokes a model."""
    payload, call = _payload(wp, ws, task_input, feedback, format_repair), _call(model)
    authored = (_apply_author_patch(format_repair["previous_output"], raw)
                if format_repair is not None else deepcopy(raw))
    plan = _compile(ws, authored)
    plan["binding"].update(wp_hash=_hash(wp), task_hash=_hash(task_input), feedback_hash=_hash(feedback),
                          messages_hash=_hash(_messages(payload)), call_hash=_hash(call))
    plan.update(author_input=payload, author_output=deepcopy(raw), call=call,
                source_inputs={"whitepaper": deepcopy(wp), "task": deepcopy(task_input), "feedback": deepcopy(feedback)})
    plan["plan_hash"] = _hash(plan)
    return plan


def author_plan(wp, ws, tracer, task_input=None, feedback=None, *, checkpoint_dir=None):
    """Author then perform bounded shape/ref repairs; provider failures stop."""
    if (wp.get("world_generation") or {}).get("disclosure_strategy") == "direct_batches_v1":
        from pipeline.disclosure_batches import author
        return author(wp, ws, tracer, task_input, feedback, checkpoint_dir)
    from pipeline.world_context import enabled as agentic_enabled
    if agentic_enabled(wp):
        return _author_agentic(wp, ws, tracer, task_input, feedback)
    report = {"version": VERSION, "status": "error", "raw_output": None,
              "logical_calls": 0, "caller_entered": False, "attempts": []}
    try:
        if not enabled(wp):
            raise ValueError("Public disclosure author was not enabled")
        _payload(wp, ws, task_input, feedback)  # Preconditions before any dispatch.
        import config
        attempt_limit = getattr(config, "DISCLOSURE_FORMAT_ATTEMPTS", 2)
        if type(attempt_limit) is not int or not 2 <= attempt_limit <= 6:
            raise ValueError("Disclosure format attempts must be between 2 and 6")
        report["attempt_limit"] = attempt_limit
        call = _call(config.STRUCTURE_MODEL)
        repair = None
        for number in range(attempt_limit):
            messages = _messages(_payload(wp, ws, task_input, feedback, repair))
            attempt = {"number": number+1, "kind": "initial" if number == 0 else "format_reference_repair",
                       "messages": deepcopy(messages), "call": deepcopy(call), "raw_output": None, "status": "started"}
            report["attempts"].append(attempt)
            report.update(messages=messages, call=call, caller_entered=True, logical_calls=number+1)
            # Exceptions here are execution failures, even if they are ValueError.
            try:
                raw = tracer.chat_json(STEP, messages, **call["params"])
            except Exception as exc:
                attempt.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
                raise
            attempt["raw_output"] = deepcopy(raw)
            report["raw_output"] = deepcopy(raw)
            # Original Tracer preserves provider exceptions as this reserved
            # result. It is execution failure, never a model shape to repair.
            if isinstance(raw, dict) and "__error__" in raw:
                attempt.update(status="execution_error", error_type="TracerExecutionError",
                               error=str(raw["__error__"]))
                raise RuntimeError("Original author execution failed: " + str(raw["__error__"]))
            try:
                plan = compile_plan(wp, ws, raw, task_input, feedback, config.STRUCTURE_MODEL, repair)
            except PlanFormatError as exc:
                attempt.update(status="format_reference_error", validation_error=deepcopy(exc.details))
                report["validation_error"] = deepcopy(exc.details)
                if number + 1 >= attempt_limit:
                    raise
                # The next patch must apply to the fully merged last proposal,
                # including earlier accepted edits and untouched records.
                previous = (_apply_author_patch(repair["previous_output"], raw)
                            if repair is not None else deepcopy(raw))
                repair = {"previous_output": previous, "validation_error": deepcopy(exc.details),
                    "missing_refs_all": _missing_refs(ws, previous), "instruction": PATCH_REPAIR_INSTRUCTION}
                continue
            attempt["status"] = "ready"
            ws.disclosure = plan
            report.update(status="ready", plan_hash=plan["plan_hash"], binding=deepcopy(plan["binding"]))
            report.pop("validation_error", None)
            break
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    return report


def validate_plan(ws):
    try:
        plan = getattr(ws, "disclosure", None)
        if not isinstance(plan, dict) or not plan:
            raise ValueError("Missing public disclosure plan")
        if plan.get("strategy") in ("direct-disclosure/v1", "direct-disclosure/v2"):
            from pipeline.disclosure_batches import validate
            return validate(ws)
        if plan.get("strategy") == "agentic":
            return _validate_agentic_plan(ws)
        saved = {k: deepcopy(v) for k, v in plan.items() if k != "plan_hash"}
        if plan.get("plan_hash") != _hash(saved):
            raise ValueError("Public disclosure plan hash changed")
        rebuilt = _compile(ws, plan.get("raw_output"))
        for key in ("version", "records", "undisclosed", "catalogue", "raw_output"):
            if rebuilt[key] != plan.get(key):
                raise ValueError("Disclosure plan differs from compiled original output: " + key)
        if any(plan.get("binding", {}).get(k) != v for k, v in rebuilt["binding"].items()):
            raise ValueError("Disclosure truth/catalogue/implementation binding is stale")
        payload = plan["author_input"]
        source = plan["source_inputs"]
        repair = payload.get("format_repair")
        if repair is not None:
            if not isinstance(repair, dict) or set(repair) not in (
                    {"previous_output", "validation_error", "instruction"},
                    {"previous_output", "validation_error", "instruction", "missing_refs_all"}):
                raise ValueError("Invalid saved format-repair context")
            if repair["instruction"] not in (FORMAT_REPAIR_INSTRUCTION, PATCH_REPAIR_INSTRUCTION):
                raise ValueError("Saved format-repair instruction changed")
            if repair["instruction"] == PATCH_REPAIR_INSTRUCTION and repair.get("missing_refs_all") != _missing_refs(ws, repair["previous_output"]):
                raise ValueError("Saved missing-reference diagnostics changed")
            try:
                _compile(ws, repair["previous_output"])
            except PlanFormatError as exc:
                if exc.details != repair["validation_error"]:
                    raise ValueError("Saved format-repair diagnostics changed")
            else:
                raise ValueError("Format-repair context did not follow a shape/ref failure")
        authored = (_apply_author_patch(repair["previous_output"], plan["author_output"])
                    if repair is not None else plan["author_output"])
        if authored != plan["raw_output"]:
            raise ValueError("Disclosure differs from the model's recorded output/patch")
        if payload != _payload(source["whitepaper"], ws, source["task"], source["feedback"], repair):
            raise ValueError("Disclosure author original input projection changed")
        if plan["call"] != _call(plan["call"]["params"]["model"]):
            raise ValueError("Disclosure author call contract changed")
        messages = _messages(payload)
        if any(plan["binding"].get(k) != v for k, v in {
                "wp_hash": _hash(source["whitepaper"]), "task_hash": _hash(source["task"]), "feedback_hash": _hash(source["feedback"]),
                "messages_hash": _hash(messages), "call_hash": _hash(plan["call"])}.items()):
            raise ValueError("Disclosure original author input binding changed")
        return []
    except Exception as exc:
        return [{"code": "missing_or_stale_disclosure", "message": f"{type(exc).__name__}: {exc}"}]


def _occurrences(ws, session, *, history=False, validated=False):
    if not validated:
        errors = validate_plan(ws)
        if errors:
            raise ValueError(errors[0]["message"])
    _session(ws, session)
    plan = ws.disclosure
    by_ref = {item["ref"]: item for item in plan["catalogue"]}
    for record in plan["records"]:
        if record["session"] > session or (not history and record["session"] != session):
            continue
        for ref in record["refs"]:
            yield record, by_ref[ref]


SCOPED_AUTHOR_RULES = """
分步公开安排协议：自行判断哪些业务事实需要一起理解，先 inspect 读取确切版本，再安排。
{"action":"arrange","records":[原record对象],"undisclosed":["已读ref"],"record_updates":[],"remove_undisclosed":[],"reason":"安排理由"}
records 追加公开记录；undisclosed 追加明确决定不公开的引用；remove_undisclosed 移除先前未披露安排；
record_updates 用 {"record_index":0,"record":完整原record对象} 修订已安排记录。它们只处理你本次明确提交的修改。
你可多次安排，程序保留其他记录。记录必须同时解释同组全部 refs 的版本和时点；主动读取同对象其他版本和必要关联材料。
working_state 只列记录索引、refs 和公开期。需要复核先前文字时使用 {"action":"recall","record_indices":[0]}，原文会进入下一轮窗口。
全部引用有明确安排后输出 {"action":"finish","reason":"整个公开安排的理由和限制"}。
缺少引用的安排要继续读取并作决定。代码不会替你添加公开或未披露内容。
"""


def _scoped_author_context(payload):
    from pipeline.world_context import ReadWindow
    nodes = {}
    for row in payload["catalogue"]:
        event = row.get("event") or {}
        label = "/".join(str(v) for v in (row.get("entity", event.get("id", "")),
            row.get("field", event.get("type", row["kind"])), row.get("fact_session", event.get("session", ""))))
        nodes[row["ref"]] = {"label": label, "catalogue_entry": deepcopy(row)}
    common = {key: deepcopy(value) for key, value in payload.items() if key != "catalogue"}
    return ReadWindow(common, nodes)


def _scoped_author_state(raw, ws):
    return {"records": [{"record_index": index, "session": row["session"], "refs": row["refs"]}
                        for index, row in enumerate(raw["records"])],
            "undisclosed": raw["undisclosed"], "missing_refs": _missing_refs(ws, raw), "reason": raw["reason"]}


def _author_action(action, window, raw, ws):
    if window.control(action):
        return raw, False
    if action.get("action") == "recall":
        indices = action.get("record_indices")
        if (not isinstance(indices, list) or not indices or len(indices) > 12
                or any(type(i) is not int or i < 0 or i >= len(raw["records"]) for i in indices)):
            raise ValueError("Disclosure recall requires existing record indices")
        # Separate from fact ids: recall cannot count uninspected facts as read.
        window.visible = {"record_" + str(i): {"record_index": i, "record": deepcopy(raw["records"][i])}
                          for i in indices}
        return raw, False
    if action.get("action") == "finish":
        raw = {**deepcopy(raw), "reason": _text(action.get("reason"), "final arrangement reason")}
        _compile(ws, raw)
        return raw, True
    if action.get("action") != "arrange":
        raise ValueError("Unknown disclosure agent action")
    result = deepcopy(raw)
    for key in ("records", "undisclosed", "record_updates", "remove_undisclosed"):
        if not isinstance(action.get(key, []), list):
            raise ValueError("Disclosure arrangement operations must be arrays")
    patch = {"record_updates": action.get("record_updates", []), "append_records": action.get("records", []),
             "undisclosed": None, "reason": action.get("reason")}
    result = _apply_author_patch(result, patch)
    removed = action.get("remove_undisclosed", [])
    if any(ref not in result["undisclosed"] for ref in removed):
        raise ValueError("Cannot remove an unknown undisclosed arrangement")
    result["undisclosed"] = [ref for ref in result["undisclosed"] if ref not in removed] + action.get("undisclosed", [])
    changed_records = action.get("records", []) + [item["record"] for item in action.get("record_updates", [])]
    from pipeline.world_context import MAX_READ_CHARS
    for record in changed_records:
        joined = {ref: window.nodes.get(ref) for ref in record.get("refs", [])}
        if len(json.dumps(joined, ensure_ascii=False)) > MAX_READ_CHARS - 2000:
            raise ValueError("One disclosure record exceeds a review window; choose smaller coherent groups")
    refs = [ref for row in changed_records for ref in row.get("refs", [])] + action.get("undisclosed", [])
    if any(ref not in window.read for ref in refs):
        raise ValueError("Disclosure decisions require reading each exact original reference")
    try:
        _compile(ws, result)
    except PlanFormatError as exc:
        # Incomplete work is retained only when its existing records are valid.
        if (exc.details.get("code") != "reference_accounting" or any(exc.details.get(key)
                for key in ("overlap_refs", "unknown_refs", "duplicate_refs"))):
            raise
    window.visible = {}
    return result, False


def _scoped_author_session(wp, ws, task_input, feedback, model, tracer=None, transcript=None, records=None):
    from pipeline.world_context import INSTRUCTION, MAX_STEPS, ProgressGuard
    payload = _payload(wp, ws, task_input, feedback)
    window = _scoped_author_context(payload)
    system = SYSTEM.split("只输出三键 JSON：")[0] + INSTRUCTION + SCOPED_AUTHOR_RULES
    call = _call(model)
    raw = {"records": [], "undisclosed": [], "reason": "尚未完成安排"}
    records = [] if records is None else records
    action_error, failures = None, 0
    progress = ProgressGuard("Disclosure author")
    for number in range(MAX_STEPS):
        state = _scoped_author_state(raw, ws)
        if action_error is not None:
            state["action_error"] = action_error
        messages = window.messages(system, state)
        window.mark_sent()
        entry = {"messages_hash": _hash(messages), "call": deepcopy(call), "raw_output": None}
        records.append(entry)
        if transcript is not None:
            if number >= len(transcript):
                raise ValueError("Incomplete disclosure agent transcript")
            saved = transcript[number]
            if saved.get("messages_hash") != _hash(messages) or saved.get("call") != call:
                raise ValueError("Disclosure exact-read context or call changed")
            action = deepcopy(saved.get("raw_output"))
        else:
            try:
                action = tracer.chat_json(STEP, messages, **call["params"])
            except Exception as exc:
                entry.update(error_type=type(exc).__name__, error=str(exc))
                raise
        entry["raw_output"] = deepcopy(action)
        if isinstance(action, dict) and "__error__" in action:
            raise RuntimeError("Original author execution failed: " + str(action["__error__"]))
        try:
            raw, finished = _author_action(action, window, raw, ws)
            progress.observe((window.progress_marker(), _hash(raw)))
        except (ValueError, KeyError, TypeError) as exc:
            action_error, failures = str(exc), failures + 1
            if failures >= 4:
                raise ValueError("Four consecutive invalid disclosure actions: " + action_error) from exc
            continue
        action_error, failures = None, 0
        if not finished:
            continue
        if transcript is not None and number + 1 != len(transcript):
            raise ValueError("Extra disclosure agent transcript entries")
        plan = _compile(ws, raw)
        plan["binding"].update(wp_hash=_hash(wp), task_hash=_hash(task_input), feedback_hash=_hash(feedback),
            messages_hash=_hash(records), call_hash=_hash(call), protocol_hash=_hash(system))
        plan.update(strategy="agentic", author_input=payload, author_output=deepcopy(raw), call=call,
            transcript=records, source_inputs={"whitepaper": deepcopy(wp), "task": deepcopy(task_input), "feedback": deepcopy(feedback)})
        plan["plan_hash"] = _hash(plan)
        return plan
    raise ValueError("Disclosure author exhausted bounded agent steps")


def _author_agentic(wp, ws, tracer, task_input, feedback):
    report = {"version": VERSION, "strategy": "agentic", "status": "error", "logical_calls": 0, "caller_entered": False, "attempts": []}
    try:
        if not enabled(wp):
            raise ValueError("Public disclosure author was not enabled")
        import config
        plan = _scoped_author_session(wp, ws, task_input, feedback, config.STRUCTURE_MODEL, tracer=tracer, records=report["attempts"])
        ws.disclosure = plan
        report.update(status="ready", caller_entered=True, logical_calls=len(plan["transcript"]),
                      attempts=deepcopy(plan["transcript"]), raw_output=deepcopy(plan["raw_output"]),
                      plan_hash=plan["plan_hash"], binding=deepcopy(plan["binding"]))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), logical_calls=len(report["attempts"]),
                      caller_entered=bool(report["attempts"]))
    return report


def _validate_agentic_plan(ws):
    try:
        plan = ws.disclosure
        source = plan["source_inputs"]
        rebuilt = _scoped_author_session(source["whitepaper"], ws, source["task"], source["feedback"],
            plan["call"]["params"]["model"], transcript=plan["transcript"])
        if plan != rebuilt:
            raise ValueError("Disclosure agent receipt differs from exact replay")
        return []
    except Exception as exc:
        return [{"code": "missing_or_stale_disclosure", "message": f"{type(exc).__name__}: {exc}"}]


def _fact(record, item, ws):
    return {"entity": item["entity"], "entity_type": item.get("entity_type"),
            "field": item["field"], "value": None if item["stopped"] else deepcopy(item["value"]),
            "stopped": item["stopped"], "fact_session": item["fact_session"],
            "fact_date": item["fact_date"], "operation": item["operation"],
            "disclosure_id": record["id"], "disclosure_session": record["session"],
            "disclosure_date": ws.date_of_session(record["session"]),
            "target_ref": item["ref"], "channel": record["channel"],
            "acquisition_context": record["acquisition_context"]}


def session_facts(ws, s, *, validated=False):
    if not getattr(ws, "disclosure", None):
        from pipeline.render import _session_facts
        return _session_facts(ws, s)
    return [_fact(record, item, ws) for record, item in _occurrences(ws, s, validated=validated)
            if item["kind"] == "fact"]


def session_events(ws, s, *, validated=False):
    if not getattr(ws, "disclosure", None):
        return [deepcopy(item) for item in ws.events if item.get("session") == s]
    return [{**deepcopy(item["event"]), "target_ref": item["ref"], "disclosure_id": record["id"],
             "disclosure_session": record["session"], "disclosure_date": ws.date_of_session(record["session"]),
             "channel": record["channel"], "acquisition_context": record["acquisition_context"]}
            for record, item in _occurrences(ws, s, validated=validated) if item["kind"] == "event"]


def source_assertions(ws, s, *, validated=False):
    # Existing L5 declarations retain their original public publication times.
    if getattr(ws, "disclosure", None) and not validated:
        errors = validate_plan(ws)
        if errors:
            raise ValueError(errors[0]["message"])
    from pipeline.corpus_contract import fingerprint
    rows = []
    for item in ws.conflicts:
        if item.get("session") == s and item.get("authoritative_value") not in (None, ""):
            declaration = {"entity": item["entity"], "field": item["field"], "session": s,
                "date": item["date"], "value": deepcopy(item["authoritative_value"]),
                "source": item["authoritative_source"]}
            rows.append({"assertion_id": "source_assertion_" + fingerprint(declaration)[:20], **declaration})
    return sorted({row["assertion_id"]: row for row in rows}.values(), key=lambda item: item["assertion_id"])


def context(ws, s, entities=None, *, validated=False):
    if not getattr(ws, "disclosure", None):
        from pipeline.corpus_contract import canonical_context
        return canonical_context(ws, s, entities)
    from pipeline.corpus_contract import VERSION as CORPUS_VERSION, public_stage_rules
    occurrences = list(_occurrences(ws, s, history=True, validated=validated))
    selected = None if entities is None else set(entities)
    facts = []
    events, claims, public_records = [], [], {}
    for record, item in occurrences:
        if selected is not None:
            names = ({item.get("entity")} if item["kind"] != "event"
                     else set(item["event"].get("participants", {}).values()))
            if not names.intersection(selected):
                continue
        public_records.setdefault(record["id"], {k: deepcopy(record[k]) for k in
                                                 ("id", "session", "channel", "acquisition_context")})
        if item["kind"] == "fact":
            fact = _fact(record, item, ws)
            fact.update(known=True, history=[{"session": item["fact_session"], "date": item["fact_date"],
                "operation": item["operation"], "value": deepcopy(item["value"]),
                "disclosure_session": record["session"]}])
            facts.append(fact)
        elif item["kind"] == "event":
            events.append({**deepcopy(item["event"]), "target_ref": item["ref"],
                           "disclosure_id": record["id"], "disclosure_session": record["session"]})
        else:
            claims.append({**{k: deepcopy(v) for k, v in item.items() if k not in ("side",)},
                           "disclosure_id": record["id"], "disclosure_session": record["session"]})
    return {"version": CORPUS_VERSION, "disclosure_version": VERSION, "session": s,
        "document_date": ws.date_of_session(s), "period_label": f"第{s+1}{ws.period_unit()}",
        "truth_hash": ws.disclosure["binding"]["truth_hash"], "disclosure_hash": ws.disclosure["plan_hash"],
        "facts": facts, "events": events,
        "source_assertions": source_assertions(ws, s, validated=validated),
        "public_claims": claims, "disclosures": list(public_records.values()),
        "field_schemas": [{"entity_type": item.get("id"), "fields": deepcopy(item.get("fields", []))}
                          for item in (ws.world_blueprint or {}).get("entity_types", [])],
        "public_stage_rules": public_stage_rules(ws),
        "allowed_document_scaffolding": ["日志记录", "档案登记", "通报提及"],
        "policy": "仅上述已公开信息可用于本期正文；事实有效时点与披露时点分开，后出记录可准确回顾过去。"
                  "未提供的信息不能补写为已知；来源主张不自动当真值，不能把材料可见等同答案可复述。"}
