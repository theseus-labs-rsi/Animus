"""Business feasibility review inside the original whitepaper/world loop."""
from copy import deepcopy
import hashlib
import json

import config
from pipeline.seed_pack import validate_seed_blueprint
from pipeline.seed_world import seed_business_context
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint

VERSION = "blueprint-feasibility/v1"
SYSTEM = """你复核原 pipeline 的候选白皮书能否表达原任务，不生成世界、题目或评分。
逐项读 business 中的必需机制，构想一个具体而简短的完整业务过程，检查 blueprint 的字段、关系、事件、状态与因果约束是否允许该过程成立。
特别核对会反复确认、重开、恢复或修订的过程。fields.states 在当前编译器中表示不可逆的单向生命周期，不能用于可循环运行状态；后者用 kind=status 并省略 states，由后续世界语义审阅判断实际变化。
不要因为存在字段、事件名称或数量就判机制可实现。核对真实前态、依赖变更、触发、后续响应能否由同一对象上的事件表达；精确因果延迟、字段所有权和同一期同字段唯一写入也须相容。
business/任务是依据；普通行业习惯、可选示例和能力陷阱不能升级为额外必需规则。允许合法未知、不同等价流程和未触发实例，只要求每项必需机制至少有一种合法完整见证。
feedback 是下游可错的失败证据，需独立判断。不要为了迁就现有程序错误删掉业务要求，也不要把合成过程说成已经生成或已经通过世界验收。
输出严格 JSON：{"decision":"accept|repair|unresolved","reason":"整体依据",
"mechanism_checks":[{"mechanism_id":"逐一复制全部机制id","walkthrough":"按具体对象简述可行过程或阻塞所在","status":"feasible|blocked|unresolved"}],
"issues":[{"finding":"具体冲突字段/约束及原任务依据","suggestion":"最小修订建议或所缺信息"}]}。
accept 需要全部机制 feasible 且 issues=[]；repair 需要至少一个具体 issue；无法判断或原任务本身冲突用 unresolved。"""
REPAIR_SYSTEM = """你是原白皮书架构师，处理世界作者报告的约束矛盾。只提出有业务依据的最小字段约束修订。
可编辑字段的可选 states/range/monotonic。states 代表单向不可逆生命周期；可循环状态可删除 states，但必须保留字段及业务复核义务。不得改 seed、字段名/类型/单位、实体数量、时间安排、事件/关系定义或因果要求。
不能仅因生成器多次失败就放宽约束。结合 business、blueprint、feedback 判断是原声明过强/错误，还是生成器应修自己的事件。若当前范围没有有据修法，edits=[]并说明未决原因。
严格 JSON：{"reason":"整体依据","edits":[{"entity_type":"原类型id","field":"原字段名","remove":["states"],"set":{},"reason":"原任务证据与修改含义"}]}。
remove/set 仅包含确实需改变的 states/range/monotonic 键，不能同一键同时 remove 和 set。"""


class BlueprintReviewRequested(WorldBlueprintError):
    def __init__(self, evidence):
        self.evidence = deepcopy(evidence)
        super().__init__("world author requested upstream blueprint review: " + str(evidence.get("reason")))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _call(tracer, step, system, payload):
    result = tracer.chat_json(step, [{"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=config.STRUCTURE_MODEL, temperature=0.2, max_tokens=8192,
        retries=3, strict_json=True, response_format={"type": "json_object"})
    if not isinstance(result, dict) or "__error__" in result:
        raise WorldBlueprintError("blueprint review execution failed: " + str(result))
    return result


def assess(wp, tracer, feedback=None):
    payload = {"business": seed_business_context(wp), "blueprint": normalize_world_blueprint(wp),
               "feedback": feedback}
    raw = _call(tracer, "council.blueprint_feasibility", SYSTEM, payload)
    expected = [m["id"] for m in payload["business"].get("mechanisms", [])]
    checks, issues = raw.get("mechanism_checks"), raw.get("issues")
    valid = (raw.get("decision") in {"accept", "repair", "unresolved"}
             and isinstance(raw.get("reason"), str) and raw["reason"].strip()
             and isinstance(checks, list) and all(isinstance(c, dict) for c in checks)
             and sorted(str(c.get("mechanism_id")) for c in checks) == sorted(expected)
             and all(c.get("status") in {"feasible", "blocked", "unresolved"}
                     and isinstance(c.get("walkthrough"), str) and c["walkthrough"].strip() for c in checks)
             and isinstance(issues, list) and all(isinstance(i, dict) and isinstance(i.get("finding"), str)
                     and i["finding"].strip() and isinstance(i.get("suggestion"), str) for i in issues))
    if not valid or (raw["decision"] == "accept" and (issues or any(c["status"] != "feasible" for c in checks))) \
            or (raw["decision"] == "repair" and not issues):
        raise WorldBlueprintError("blueprint feasibility review has an invalid/incomplete decision")
    return {"version": VERSION, "input_hash": digest(payload), "input": payload,
            "decision": raw["decision"], "review": raw,
            "scope": "LLM assessment of expressibility; actual world still needs all original gates"}


def revise_constraints(wp, tracer, feedback, audit):
    """LLM authors edits; exact seed constraints and execution scale stay fixed."""
    audit.update(version=VERSION, before=deepcopy(wp), feedback=deepcopy(feedback), status="proposed")
    payload = {"business": seed_business_context(wp), "blueprint": normalize_world_blueprint(wp),
               "feedback": feedback}
    raw = _call(tracer, "council.blueprint_constraint_repair", REPAIR_SYSTEM, payload)
    audit["author"] = deepcopy(raw)
    edits = raw.get("edits")
    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip() or not isinstance(edits, list) or not edits:
        raise WorldBlueprintError("Upstream architect has no supported constraint repair")
    result = deepcopy(wp)
    fields = {(t["id"], f["name"]): f for t in result["world_blueprint"]["entity_types"] for f in t["fields"]}
    allowed, touched = {"states", "range", "monotonic"}, set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise WorldBlueprintError("Constraint edit must be an object")
        key = (edit.get("entity_type"), edit.get("field"))
        remove, values = edit.get("remove", []), edit.get("set", {})
        if (key not in fields or key in touched or not isinstance(remove, list)
                or any(not isinstance(k, str) for k in remove) or not isinstance(values, dict)
                or not (set(remove) | set(values))
                or (set(remove) | set(values)) - allowed or set(remove) & set(values)
                or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()):
            raise WorldBlueprintError("Unsupported, duplicate or empty constraint edit")
        touched.add(key)
        for name in remove:
            if name not in fields[key]:
                raise WorldBlueprintError("Cannot remove an absent field constraint")
            del fields[key][name]
        fields[key].update(deepcopy(values))
    normalize_world_blueprint(result)
    contract = result.get("seed_contract")
    if contract:
        seed_audit = validate_seed_blueprint(result, contract)
        if not seed_audit["passed"]:
            raise WorldBlueprintError("Constraint revision violates seed: " + "; ".join(seed_audit["issues"]))
        result["seed_audit"] = seed_audit
    if result["world_blueprint"] == wp["world_blueprint"]:
        raise WorldBlueprintError("Constraint revision made no change")
    # Refresh the original compatibility projection from its one authoritative blueprint.
    projected = {}
    for kind in result["world_blueprint"]["entity_types"]:
        for field in kind["fields"]:
            projected.setdefault(field["name"], deepcopy(field))
    profile = result.setdefault("domain_profile", {})
    profile["field_schema"] = list(projected.values())
    profile["state_machines"] = [{"field": f["name"], "states": f["states"]}
                                 for f in projected.values() if f.get("states")]
    audit["candidate"] = deepcopy(result)
    review = assess(result, tracer, feedback={"request": feedback, "author_changes": raw})
    audit["review"] = review
    if review["decision"] != "accept":
        raise WorldBlueprintError("Revised blueprint remains unresolved; original contract retained")
    result.setdefault("blueprint_revisions", []).append({"before_hash": digest(wp["world_blueprint"]),
        "after_hash": digest(result["world_blueprint"]), "author": deepcopy(raw), "review": review})
    audit.update(status="accepted", after=deepcopy(result))
    return result
