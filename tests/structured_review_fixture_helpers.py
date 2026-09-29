"""Fixed three-role semantic-review transport used by offline tests."""
from copy import deepcopy
import json

from semantic_judge_selftest import grade_output, review_output


def anchor(value):
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        if not value:
            return "{}"
        return next((found for child in value.values() if (found := anchor(child))), None)
    if isinstance(value, list):
        if not value:
            return "[]"
        return next((found for child in value if (found := anchor(child))), None)
    return json.dumps(value, allow_nan=False)


class StructuredScript:
    def __init__(self, *, bad_quote=False):
        self.calls = []
        self.bad_quote = bad_quote

    def __call__(self, step, messages, **params):
        payload = json.loads(messages[1]["content"])
        self.calls.append({"step": step, "payload": payload, "params": deepcopy(params)})
        if step == "semantic_judge.answer":
            return grade_output(evidence=[], reasoning="OFFLINE_FIXTURE only.")
        if "original_reference" in payload:
            reference = payload["original_reference"]
            quote = "ABSENT_OFFLINE_QUOTE" if self.bad_quote else anchor(reference["answer"])
            return {
                "decision": "accept", "reason": "OFFLINE_FIXTURE; not real semantic approval.",
                "task_requirements": ["OFFLINE_FIXTURE"],
                "claims": [{"reference_part": "answer", "source_quote": quote,
                    "assessment": "OFFLINE_FIXTURE", "explanation": "OFFLINE_FIXTURE",
                    "evidence": []}] if quote else [],
                "task_coverage": "OFFLINE_FIXTURE", "substantive_defects": [],
                "acceptable_brevity": [], "editorial_suggestions": [], "suggested_revision": "",
                "limitations": [], "evidence": [],
            }
        if step.endswith("blind_read"):
            return review_output("blind_read", answer="BLIND_ONLY_SENTINEL", evidence=[],
                                 reasoning="OFFLINE_FIXTURE", interpretation="OFFLINE_FIXTURE")
        reference = payload["reference_proposal"]
        quote = anchor(reference["answer"])
        result = review_output(evidence=[], reasoning="OFFLINE_FIXTURE", interpretation="OFFLINE_FIXTURE")
        result["original_answer_review"] = ([{
            "requirement": "OFFLINE_FIXTURE", "assessment": "OFFLINE_FIXTURE",
            "explanation": "OFFLINE_FIXTURE", "target_scope": "quoted_text",
            "reference_targets": [{"reference_part": "answer", "source_quote": quote}],
        }] if quote else [])
        rationale = reference["rationale"]
        result["original_rationale_review"] = {
            "status": "reviewed" if rationale else "not_provided",
            "claims": [{"claim": "OFFLINE_FIXTURE", "assessment": "OFFLINE_FIXTURE",
                "explanation": "OFFLINE_FIXTURE", "target_scope": "quoted_text",
                "reference_targets": [{"reference_part": "rationale", "source_quote": rationale}]}]
                if rationale else [],
            "limitations": [],
        }
        result["reference_audit_response"] = "OFFLINE_FIXTURE response, not a semantic endorsement."
        return result
