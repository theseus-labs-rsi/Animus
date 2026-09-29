"""Independent reference-package critique with no solver or blind answer input.

The critic chooses important claims in natural language. Original JSON nodes or
quoted spans bind claims to the proposed answer/rationale, not another answer.
This is an attributed opinion and does not itself select, score or publish items.
"""
from copy import deepcopy
from pathlib import Path

from pipeline import agent_editing
from pipeline import reference_locations
from pipeline.semantic_review import visible_documents, resolve_citations, fingerprint

VERSION = "independent-reference-audit/v5"
SYSTEM = agent_editing.COMMON + """
你独立对一份原参考答案及其理由做反例审查。不会收到盲答、solver答案或其他审查意见。
original_reference 是唯一被审查的答案提案；documents 与 public_protocol 是事实依据。
先理解实际任务，再检验决定答案的关键推理：尝试一个最强、具体、会改变答案或关键理由成立性的
替代解释。它必须保持题面对象、时点、公开协议和来源规则不变，与全部相关公开证据相容。
说明哪条公开依据排除了这一解释；若无法排除，指出缺失的具体前提，不能用“通常如此”或
“没看到反证”代替所需依据。这是检验而非新增事实，不可改动明确事实、假设所有记录都失实、
把未记载事件的假设当事实，或否定题面已给条件。问记录内状态时遵守公开的延续规则；问现实
是否发生时，未提及不自动等于未发生，登记完整性也须有公开依据。公开事实或合理语义已排除实质相反解释时应接受，
不强制凑反例，也不要求答案复述审阅过程。把检验写进现有reason、explanation或limitations。
分别检查原answer与rationale；不要替原提案补出正确答案再通过。区分被证据反驳、缺少支持和
自己尚未审查完成。只找到原答案某个片段为真或没想到反例，都不足以证明其主要任务已完成。
不要凭空增添题目没有要求的任务。短答案或同义表达本身不是缺陷。合理的“无法确定”可以是
正确且完整的答案；材料保留未知不等于题目有歧义，也不等于你自己尚未完成审查。
每项主张用 reference_part='answer|rationale' 指定原提案字段。优先用 location_scope='node'
和 json_pointer 定位原字段内的完整节点，此时不得提供 source_quote。路径相对该字段：空字符串
表示整个 answer 或 rationale，/结论/0 表示键“结论”下数组第0项；键名内~写作~0、/写作~1。
路径只能指向实际已有节点；程序从原提案取完整节点值，不能提供自己改写或补全的节点替代它。
也可保留旧 source_quote 方式引用逐字连续文字（省略location_scope或填quote，不能再带json_pointer）。
assessment 与 explanation 自由说明含义、成立范围及缺陷，不能引用你自己想象的参考内容。
只须选择会影响成立性的重要主张，不以主张数量或字面匹配评价质量。位置可核对不证明语义正确。
answer 可为任意 JSON 值，必须理解原有完整结构，不能改写后再审。node可以定位对象、数组、
字符串、数字、boolean或null本身；路径定位对象键所对应的值，不把键名当值。source_quote可引用某个
字符串值或对象键名内的逐字连续文字；数字、boolean、null、空对象和空数组只引用完整 JSON
字面量，例如 true、null、{}、[]。普通字符串即使看似 JSON 也不再解析；不能拼接多个节点作引文。
程序会记录所有匹配节点及位置；键名不是键值，重复引文不能猜成唯一位置。你须结合完整结构
解释实际主张，不能把定位到一个键或片段当成已经核查整个关系，也不能用定位数量证明覆盖。
节点定位同样只证明归属，不证明成立性或覆盖充分。若原提案漏答任务，在task_coverage和
substantive_defects说明缺失，不捏造原文主张、路径或引文来定位不存在的内容。
这些 JSON 表达都仍是已提供的参考，包括 null 和空容器；它们是否完成任务由实际题意决定。
输出 {decision:'accept|revise|unresolved',reason:'对这一原提案的明确判断与局限',
task_requirements:['题目实际主要要求'],
claims:[{reference_part:'answer|rationale',location_scope:'node',json_pointer:'相对该字段的原节点路径；根为双引号空串',
assessment:'原主张成立与否及影响',explanation:'依据和可能的不同解释',evidence:[]}],
task_coverage:'原提案是否完成题目要求，含可接受省略',
substantive_defects:['影响答案或理由成立的问题；说明范围'],
acceptable_brevity:['可接受简略'], editorial_suggestions:['非实质建议'],
suggested_revision:'若需修改，描述有据的最小修改；否则可为空',limitations:[],evidence:[]}。
上述两处 evidence 使用同一格式。优先直接定位公开文档的完整字段，例如
{"doc_id":"输入中的原doc_id","field":"content","location_scope":"field",
 "role":"support","explanation":"结合该字段全文说明支持范围、反证或限制"}。
这会由程序保存原字段，不需要你抄写正文。若确需摘句，则同时给
"location_scope":"quote" 和 "quote":"该原字段内逐字连续的原文"；只写 quote 模式却省略
quote 内容无法定位。不要把你概述的文字放进 quote，也不要捏造 doc_id 或 field。
两种定位仅记录实际出处，不能代替你对证据含义与反证的审查。
decision=accept 仅表示你认为原提案已可用，绝不是客观正确性证明。无法完成判断用unresolved。
"""


def reference_location_policy():
    return {"version": reference_locations.VERSION,
            "implementation_hash": fingerprint(Path(reference_locations.__file__).read_text(encoding="utf-8")),
            "pointer_base": "reference_part", "offset_unit": "unicode_code_points"}


def validate_reference_proposal(reference):
    """Require the producer's explicit JSON answer/text rationale contract."""
    if (type(reference) is not dict or "answer" not in reference
            or type(reference.get("rationale")) is not str):
        raise ValueError("Explicit JSON answer and string rationale required for this audit")
    reference_locations.validate_json_value(reference)


def audit_reference(question, corpus, public_protocol, *, model, chat_json,
                    max_calls=1, max_input_chars=500000, max_tokens=16384, record=None,
                    prepared_documents=None):
    # Validate before _questions' JSON copy can coerce non-string object keys.
    validate_reference_proposal(question.get("reference_proposal") if isinstance(question, dict) else None)
    row = agent_editing._questions([question])[0]
    reference = row.get("reference_proposal")
    if not isinstance(public_protocol, str) or not public_protocol.strip():
        raise ValueError("Public protocol required")
    if prepared_documents is None:
        documents, _ = visible_documents(corpus, include_titles=False)
    else:
        documents = deepcopy(prepared_documents)
        if (not isinstance(documents, list) or not documents
                or any(not isinstance(doc, dict) or not isinstance(doc.get("doc_id"), str)
                       for doc in documents)
                or len({doc["doc_id"] for doc in documents}) != len(documents)):
            raise ValueError("prepared_documents must be a nonempty unique visible-document list")
    original = {key: reference[key] for key in ("answer", "rationale")}
    location_policy = reference_location_policy()
    location_observations = {}

    def validate(value):
        from pipeline.paged_read import validate as validate_paged_read
        validate_paged_read(value, documents)
        errors, claim_locations = [], []
        if value.get("decision") not in {"accept", "revise", "unresolved"}:
            errors.append("invalid_decision")
        for key in ("reason", "task_coverage"):
            if not agent_editing._text(value.get(key)): errors.append("invalid_" + key)
        if not isinstance(value.get("suggested_revision"), str): errors.append("invalid_suggested_revision")
        for key in ("task_requirements", "substantive_defects", "acceptable_brevity", "editorial_suggestions", "limitations"):
            if not isinstance(value.get(key), list) or not all(isinstance(x, str) for x in value[key]):
                errors.append("invalid_" + key)
        if not isinstance(value.get("claims"), list):
            errors.append("claims_must_be_list")
        else:
            for index, claim in enumerate(value["claims"]):
                if not isinstance(claim, dict):
                    errors.append("claim_must_be_object"); continue
                part = claim.get("reference_part")
                if not isinstance(part, str) or part not in original:
                    errors.append("claim_not_in_original_reference:" + str(index)); continue
                try:
                    location = reference_locations.locate_reference_target(original[part], claim)
                except ValueError:
                    errors.append("claim_not_in_original_reference:" + str(index)); continue
                if location["location_status"] == "failed":
                    claim_locations.append({"claim_index": index, "reference_part": part,
                                            **location, "resolved_evidence": []})
                    errors.append("claim_not_in_original_reference:" + str(index)); continue
                if any(not agent_editing._text(claim.get(key)) for key in ("assessment", "explanation")):
                    errors.append("invalid_claim_analysis:" + str(index))
                located_claim = {"claim_index": index, "reference_part": part,
                                 **location, "resolved_evidence": []}
                claim_locations.append(located_claim)
                try:
                    located_claim["resolved_evidence"] = resolve_citations(claim.get("evidence"), documents)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append("claim_evidence:" + str(exc))
        try:
            evidence = resolve_citations(value.get("evidence"), documents)
        except (ValueError, TypeError, KeyError) as exc:
            evidence = []; errors.append("evidence_location:" + str(exc))
        location_observations["claim_locations"] = deepcopy(claim_locations)
        return errors, {"proposal": deepcopy(value), "claim_locations": claim_locations,
                        "resolved_evidence": evidence}

    result = agent_editing._run("independent_reference_audit", SYSTEM,
        {"documents": documents, "public_protocol": public_protocol, "question": row["question"],
         "original_reference": original, "reference_location_policy": location_policy}, model=model, chat_json=chat_json,
        max_calls=max_calls, max_input_chars=max_input_chars, max_tokens=max_tokens,
        record=record, original=row, validator=validate)
    result["reference_audit_version"] = VERSION
    result["reference_audit_implementation_hash"] = fingerprint(Path(__file__).read_text(encoding="utf-8"))
    result["reference_location_policy"] = location_policy
    # The existing input_hash/messages_hash bind the policy in the payload.
    # Do not mutate _run's binding after its records/output_hash are produced.
    # Retain failed/partial attribution checks without promoting a proposal.
    result.setdefault("claim_locations", deepcopy(location_observations.get("claim_locations", [])))
    return result
