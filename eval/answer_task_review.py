"""One independent reading of an actual answer under a public task policy.

The injected caller owns provider/transport budgets. This role sees no reference,
previous answer, grade or critic. Its attributed opinion never determines a score.
"""
from copy import deepcopy
from pathlib import Path

from pipeline import agent_editing, reference_locations
from pipeline.semantic_review import COMMON, fingerprint, locate_citations, visible_documents


VERSION = "independent-answer-task-review/v1"
POLICY_VERSION = "task-with-supporting-reasons/v1"
POLICY_TEXT = """评分范围包括题目要求的主要任务，以及回答实际用来直接支撑该任务的关键理由。
审查的是回答自己给出的论证，不能替它补出另一条正确理由后认定它已完成任务。
直接理由属于哪个任务，应按题意和完整回答的自然含义判断；不能因某理由在逻辑上可以删去，
就自动把它认作无关附言。关键理由存在实质错误、误导或所问的审核未完成，会影响任务得分。
允许不同但有效的证据路径、同义表达和自然简略；不要求长答案、固定术语或所有旁枝都正确。
确实与主要任务无关的附言瑕疵单独记录，不自动使主要任务错误。
若任务或评分范围仍不清楚，应暂缓确定评分并说明分歧，不以多数意见代替证据。"""
_START = "[benchmark-public-scoring-policy:"
_END = "[/benchmark-public-scoring-policy]"


def public_scoring_policy(policy_version=POLICY_VERSION):
    if policy_version != POLICY_VERSION:
        raise ValueError("Unknown explicit public scoring policy")
    return {"version": POLICY_VERSION, "text": POLICY_TEXT, "text_hash": fingerprint(POLICY_TEXT)}


def append_scoring_policy(protocol, policy_version=POLICY_VERSION):
    """Give solver and reviewers the same versioned text, exactly once.

Existing correctly appended text is returned byte-for-byte. Conflicting,
unversioned or repeated policy blocks are refused rather than silently rewritten.
"""
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("A nonempty public protocol is required")
    policy = public_scoring_policy(policy_version)
    block = "\n\n" + _START + policy["version"] + "]\n" + policy["text"] + "\n" + _END
    if _START in protocol or _END in protocol:
        if protocol.endswith(block) and protocol.count(_START) == protocol.count(_END) == 1:
            return protocol
        raise ValueError("Conflicting or repeated public scoring policy block")
    if POLICY_TEXT in protocol:
        raise ValueError("Unversioned policy text present; use a versioned policy block")
    return protocol + block


SYSTEM = COMMON + """
你是独立的实际回答读者，不是最终裁判。你只收到公开问题、材料、协议及solver原回答；
没有原参考、既往答案、旧判分或其他读者意见。公开协议包含本次独立审阅与后续裁决共用的评分政策；不据此假定历史回答产生时已看到该政策，原作答条件以运行记录为准。
先理解题目实际要求和完整回答，再说明回答自己的结论、直接依据、限定与可能撤回之间的关系。
不要代写一个正确答案再把它归给solver，也不要凭参考措辞或想象的评分要求制造遗漏。
一条理由即使不是证明结论的唯一办法，也可能承担题目所问的解释或审核任务；
但并非回答中的每个事实都属于主任务，不能因有一处附言瑕疵就认定整答失败。
用自然语言说明这两种解释中哪种更有依据，也应给出最强的相反解释及可接受简略。
observations只选择有意义的原回答片段，source_quote必须是solver_answer内逐字连续的文字。
不得引用你补出的答案、其他人的理由或改写后的文字充当solver原话。重复引文会记录全部位置，
不能猜一个唯一位置。若问题是原回答没有说出的内容，可在task_completion自由说明；
不为了填充observations伪造引文。observations可以为空，数量不证明完整性或正确性。
task_relation自由解释该片段与实际任务的关系，不使用预设业务槽位；assessment/explanation
说明公开支持、反证、不确定性及影响。evidence按上面的公开字段或逐字引文协议定位。
你的意见可被后续裁决有据推翻；不输出correct、incorrect、分数或最终answer_verdict。
输出单个JSON对象，必须有以下字段：
{task_interpretation:'实际任务',answer_meaning:'完整原回答实际意思',assessment:'可质疑的总体意见',
observations:[{source_quote:'solver原话',task_relation:'与所问任务的自然关系',
assessment:'支持/反驳/缺证据及影响的自然说明',explanation:'理由与相反解释',evidence:[]}],
task_completion:'实际回答是否完成所问内容，区分实质缺口与合理省略',
strongest_alternative:'对自己判断最有力的另一种解释或明确无法找到的原因',
limitations:['限制或未决'],evidence:[],coverage:{status:'complete|partial|unknown',
scope_conflict:false,limitations:[],inspected_doc_ids:[]}}。
coverage是模型自述，不是程序认证。任务范围不清或审阅不足须如实报告，不猜测确定结论。
"""

_TEXT_FIELDS = ("task_interpretation", "answer_meaning", "assessment", "task_completion", "strongest_alternative")
_FIELDS = {*_TEXT_FIELDS, "observations", "limitations", "evidence", "coverage"}
_OBSERVATION_FIELDS = {"source_quote", "task_relation", "assessment", "explanation", "evidence"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_answer_opinion(output, answer, documents):
    """Return shape/location findings, never a task score or a semantic vote."""
    if not isinstance(answer, str):
        raise ValueError("Actual solver answer must be text")
    shapes, locations, observations = [], [], []
    uninterpreted = {"opinion": [], "observations": []}
    if not isinstance(output, dict):
        return {"shape_errors": ["opinion_must_be_object"], "location_errors": [],
                "shape_valid": False, "observation_locations": [], "evidence_location": None,
                "uninterpreted_fields": uninterpreted}
    if not _FIELDS <= set(output):
        shapes.append("missing_opinion_fields")
    # Extra authored fields are retained as untrusted opinion, never merged
    # into this report's execution, readiness, identities or a final score.
    uninterpreted["opinion"] = sorted(set(output) - _FIELDS)
    for key in _TEXT_FIELDS:
        if not _text(output.get(key)):
            shapes.append("invalid_" + key)
    if not isinstance(output.get("limitations"), list) or not all(isinstance(v, str) for v in output["limitations"]):
        shapes.append("invalid_limitations")
    coverage = output.get("coverage")
    if not isinstance(coverage, dict):
        shapes.append("invalid_coverage")
    else:
        ids = coverage.get("inspected_doc_ids", [])
        known = {doc["doc_id"] for doc in documents}
        if (not isinstance(coverage.get("status"), str)
                or coverage.get("status") not in {"complete", "partial", "unknown"}
                or type(coverage.get("scope_conflict")) is not bool
                or not isinstance(coverage.get("limitations"), list)
                or not all(isinstance(v, str) for v in coverage.get("limitations", []))
                or not isinstance(ids, list) or any(not isinstance(v, str) or v not in known for v in ids)
                or (isinstance(ids, list) and len(set(str(v) for v in ids)) != len(ids))
                or (coverage.get("status") == "partial" and "inspected_doc_ids" not in coverage)):
            shapes.append("invalid_coverage")

    def evidence(value, label):
        if not isinstance(value, list):
            shapes.append(label + ":evidence_must_be_list")
            return None
        located = locate_citations(value, documents)
        if located["status"] == "failed":
            locations.append(label + ":evidence_location_failed")
        return located

    top_evidence = evidence(output.get("evidence"), "opinion")
    values = output.get("observations")
    if not isinstance(values, list):
        shapes.append("observations_must_be_list")
    else:
        for index, value in enumerate(values):
            label = "observation:" + str(index)
            if not isinstance(value, dict):
                shapes.append(label + ":must_be_object")
                continue
            if not _OBSERVATION_FIELDS <= set(value) or any(not _text(value.get(k)) for k in _OBSERVATION_FIELDS - {"evidence"}):
                shapes.append(label + ":invalid_fields")
            if set(value) - _OBSERVATION_FIELDS:
                uninterpreted["observations"].append({"observation_index": index,
                    "fields": sorted(set(value) - _OBSERVATION_FIELDS)})
            quote_location = None
            if _text(value.get("source_quote")):
                quote_location = reference_locations.locate_reference_quote(answer, value["source_quote"])
                if quote_location["location_status"] == "failed":
                    locations.append(label + ":quote_not_in_actual_answer")
            observations.append({"observation_index": index, "answer_quote_location": quote_location,
                                 "evidence_location": evidence(value.get("evidence"), label)})
    return {"shape_errors": shapes, "location_errors": locations, "shape_valid": not shapes,
            "observation_locations": observations, "evidence_location": top_evidence,
            "uninterpreted_fields": uninterpreted}


def review_answer_task(question, answer, corpus, public_protocol, *, model, chat_json,
                       policy_version=POLICY_VERSION, max_calls=1, max_input_chars=500000,
                       max_tokens=16384, record=None):
    """At most one injected attempt. An opinion is not approval or a score."""
    if (not isinstance(question, dict) or not _text(question.get("qid"))
            or not _text(question.get("question")) or not isinstance(answer, str)):
        raise ValueError("Explicit qid, question text and actual answer text required")
    public_question = {key: question[key] for key in ("qid", "question")}
    documents, _ = visible_documents(corpus, include_titles=False)
    if not documents:
        raise ValueError("At least one public document is required")
    policy = public_scoring_policy(policy_version)
    protocol = append_scoring_policy(public_protocol, policy_version)
    identity = {"version": VERSION, "implementation_hash": fingerprint(Path(__file__).read_text(encoding="utf-8")),
                "prompt_hash": fingerprint(SYSTEM), "policy_version": policy["version"],
                "policy_text_hash": policy["text_hash"], "question_hash": fingerprint(public_question),
                "answer_hash": fingerprint(answer), "protocol_hash": fingerprint(protocol),
                "documents_hash": fingerprint(documents), "model": model,
                "answer_locator": {"version": reference_locations.VERSION,
                    "implementation_hash": fingerprint(Path(reference_locations.__file__).read_text(encoding="utf-8")),
                    "target": "actual_solver_answer", "offset_unit": "unicode_code_points"}}
    payload = {"question": public_question["question"], "source_qid": public_question["qid"],
               "solver_answer": answer, "documents": documents, "public_protocol": protocol,
               "review_identity": identity}
    validation = {"shape_errors": [], "location_errors": [], "shape_valid": False,
                  "observation_locations": [], "evidence_location": None,
                  "uninterpreted_fields": {"opinion": [], "observations": []}}

    def validate(output):
        validation.update(validate_answer_opinion(output, answer, documents))
        return validation["shape_errors"] + validation["location_errors"], {"opinion": deepcopy(output)}

    def decorate(result):
        result.update(answer_task_review_version=VERSION, answer_task_review_binding=deepcopy(identity),
                      public_scoring_policy=deepcopy(policy), opinion_validation=deepcopy(validation),
                      opinion_ready=result["proposal_ready"] is True and result["execution"]["status"] == "ok",
                      opinion_ready_scope="Structure and locations only; not task completion, coverage certification, or a score",
                      coverage_claim_source="model_self_report")
        result["opinion"] = deepcopy(result["raw_output"]) if validation["shape_valid"] else None
        return result

    try:
        result = agent_editing._run("independent_answer_task_review", SYSTEM, payload, model=model,
            chat_json=chat_json, max_calls=max_calls, max_input_chars=max_input_chars, max_tokens=max_tokens,
            record=record, original={**public_question, "answer": answer}, validator=validate)
    except agent_editing.AuditRecordError as exc:
        exc.report = decorate(exc.report)
        raise
    return decorate(result)
