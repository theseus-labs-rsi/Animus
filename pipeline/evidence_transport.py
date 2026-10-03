"""Lossless shared JSON evidence; logical inputs and provider messages stay distinct.

This is a transport format, not an evidence selector or a semantic validator.
Shared values are complete original JSON values. References never acquire new
facts, and the caller's original review scope and response contract remain intact.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from string import Template

INLINE = "inline-json/v1"
SHARED = "shared-evidence/v1"
MIN_SHARED_CHARS = 80
SHARED_INSTRUCTIONS = """
用户输入采用 shared-evidence/v1 无损 JSON 表示。values 保存完整原值，payload 保留原任务结构；其中恰为 {"$evidence":"编号"} 的对象引用 values 中该完整值，表内值原样使用，不再次解释为引用。恰为 {"$literal":{...}} 的对象是原 JSON 对象的转义，保留其中原键，递归还原键的值。按引用还原完整原任务后，执行上文的原审阅要求；所有事实版本、时间、公开安排、正文与负面反馈均属于原输入。引用不是摘要、证据缺失、质量结论或额外事实。输出仍按原回复格式，引用原任务的记录编号和正文，不输出传输编号替代审阅意见。
"""
TEMPLATE_INSTRUCTIONS = """
用户输入采用 shared-evidence/v1 无损 JSON 表示。values 保存完整原值，payload 中恰为 {"$evidence":"编号"} 的对象引用该表内完整值；表内值原样使用，不再次解释为引用。恰为 {"$literal":{...}} 的对象是原对象的转义，保留原键并递归还原其值。
还原后的 payload 包含原任务 template、variables 和 json_variables。template 的 $变量名使用 variables 中对应的完整值替换；json_variables 列出的变量按完整 JSON 写入，其余变量保留原文本。$$ 表示原字面 $。完整还原原任务后执行上文要求，保留全部事实版本、时间、公开安排、上下文和负面反馈。引用不表示摘要、缺失或额外事实，原回复格式保持，不用传输编号代替正文或原记录编号。
"""


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def pack(payload):
    """Intern repeated strings/objects/arrays without pruning any input value."""
    counts = Counter()
    def count(value):
        if isinstance(value, (str, dict, list)):
            encoded = _json(value)
            if len(encoded) >= MIN_SHARED_CHARS:
                counts[encoded] += 1
        if isinstance(value, dict):
            for item in value.values():
                count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)
    count(payload)
    references, values = {}, {}
    def encode(value):
        if isinstance(value, (str, dict, list)):
            encoded = _json(value)
            if counts[encoded] > 1:
                ref = references.get(encoded)
                if ref is None:
                    ref = "e" + str(len(values))
                    references[encoded] = ref
                    values[ref] = deepcopy(value)
                return {"$evidence": ref}
        if isinstance(value, dict):
            result = {key: encode(item) for key, item in value.items()}
            return {"$literal": result} if set(value) in ({"$evidence"}, {"$literal"}) else result
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value
    encoded = encode(payload)
    return {"encoding": SHARED, "values": values, "payload": encoded}


def unpack(envelope):
    """Resolve the versioned representation; values in the table are literal."""
    if (not isinstance(envelope, dict) or set(envelope) != {"encoding", "values", "payload"}
            or envelope.get("encoding") != SHARED or not isinstance(envelope["values"], dict)):
        raise ValueError("Invalid shared evidence envelope")
    _json(envelope)  # Reject non-JSON values, including NaN, before decoding.
    values, used = envelope["values"], set()
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"$evidence"}:
                ref = value["$evidence"]
                if not isinstance(ref, str) or ref not in values:
                    raise ValueError("Unknown shared evidence reference")
                used.add(ref)
                return deepcopy(values[ref])
            if set(value) == {"$literal"}:
                if not isinstance(value["$literal"], dict):
                    raise ValueError("Invalid literal object escape")
                value = value["$literal"]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    payload = decode(envelope["payload"])
    if used != set(values):
        raise ValueError("Unreferenced shared evidence value")
    return payload


def contract(protocol):
    if protocol == INLINE:
        return None
    if protocol != SHARED:
        raise ValueError("Unknown evidence transport protocol")
    return {"protocol": protocol, "instructions": SHARED_INSTRUCTIONS,
            "minimum_shared_characters": MIN_SHARED_CHARS}


def request(system, payload, *, step, parameters, protocol=INLINE):
    """Build the actual provider messages and bind them to the complete input."""
    contract(protocol)
    if protocol == INLINE:
        content = json.dumps(payload, ensure_ascii=False)
        actual_system = system
    else:
        envelope = pack(payload)
        if unpack(envelope) != payload:
            raise ValueError("Shared evidence differs from the original logical input")
        content, actual_system = _json(envelope), system + SHARED_INSTRUCTIONS
    messages = [{"role": "system", "content": actual_system}, {"role": "user", "content": content}]
    return messages, _binding(system, payload, messages, step, parameters, protocol)


def template_contract(protocol):
    result = contract(protocol)
    if result is not None:
        result = {**result, "instructions": TEMPLATE_INSTRUCTIONS}
    return result


def template_text(payload):
    """Restore the complete mixed text/JSON task, including its original spacing."""
    template, variables, json_variables = (payload[k] for k in ("template", "variables", "json_variables"))
    if (not isinstance(template, str) or not isinstance(variables, dict)
            or not isinstance(json_variables, list) or any(not isinstance(k, str) for k in json_variables)
            or len(set(json_variables)) != len(json_variables) or not set(json_variables) <= variables.keys()):
        raise ValueError("Invalid complete template input")
    encoded = {k: json.dumps(v, ensure_ascii=False) if k in json_variables else v
               for k, v in variables.items()}
    return Template(template).substitute(encoded)


def _binding(system, payload, messages, step, parameters, protocol):
    return {"protocol": protocol, "logical_input_hash": fingerprint(payload),
        "logical_system_hash": fingerprint(system), "wire_messages_hash": fingerprint(messages),
        "step": step, "parameters_hash": fingerprint(parameters)}


def verify_template_receipt(request, binding):
    """Verify the saved original wire input without applying today's transport."""
    protocol = binding["protocol"]
    if protocol != SHARED:
        raise ValueError("Unexpected template receipt protocol")
    messages = request["messages"]
    payload = unpack(json.loads(messages[-1]["content"]))
    system = messages[0]["content"]
    if not system.endswith(TEMPLATE_INSTRUCTIONS):
        raise ValueError("Template transport/input receipt changed")
    actual = _binding(system.removesuffix(TEMPLATE_INSTRUCTIONS), payload, messages,
                      binding["step"], request["parameters"], protocol)
    if actual != binding:
        raise ValueError("Template transport/input receipt changed")


def template_request(system, template, variables, *, json_variables, step, parameters, protocol=INLINE):
    """Use the same lossless representation for an author's full original task."""
    template_contract(protocol)
    payload = {"template": template, "variables": deepcopy(variables), "json_variables": list(json_variables)}
    original_text = template_text(payload)
    if protocol == INLINE:
        content, actual_system = original_text, system
    else:
        envelope = pack(payload)
        restored = unpack(envelope)
        if restored != payload or template_text(restored) != original_text:
            raise ValueError("Shared evidence differs from the original complete task")
        content, actual_system = _json(envelope), system + TEMPLATE_INSTRUCTIONS
    messages = [{"role": "system", "content": actual_system}, {"role": "user", "content": content}]
    return messages, _binding(system, payload, messages, step, parameters, protocol)
