"""Shared fixed semantic-review opinions for offline tests."""


def audit_output(reference):
    return {
        "decision": "accept",
        "reason": "OFFLINE_AUDIT_OPINION: fixture only.",
        "task_requirements": ["依据材料回答题目"],
        "claims": [{
            "reference_part": "answer",
            "source_quote": reference["answer"],
            "assessment": "测试预设支持",
            "explanation": "测试固定意见",
            "evidence": [],
        }],
        "task_coverage": "测试预设已回答",
        "substantive_defects": [],
        "acceptable_brevity": [],
        "editorial_suggestions": [],
        "suggested_revision": "",
        "limitations": [],
        "evidence": [],
    }


def attach_targets(result, reference):
    for row in result["original_answer_review"]:
        row.update(target_scope="quoted_text", reference_targets=[{
            "reference_part": "answer", "source_quote": reference["answer"]}])
    for row in result["original_rationale_review"]["claims"]:
        row.update(target_scope="quoted_text", reference_targets=[{
            "reference_part": "rationale", "source_quote": reference["rationale"]}])
    result["reference_audit_response"] = (
        "测试预设：已对照参考审计的全部意见，不用模型一致代替材料。")
    return result
