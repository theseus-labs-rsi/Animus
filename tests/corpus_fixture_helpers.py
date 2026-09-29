"""Explicit scripted corpus opinions for plumbing tests, not semantic gold."""
import json


def fixed_positive_review(messages):
    payload = json.loads(messages[-1]["content"])
    return {"document_reviews": fixed_document_reviews(payload["documents"]), "verdict": "pass", "unsupported_claims": [], "coverage": [
        {"requirement_id": item["requirement_id"], "status": "supported",
         "reason": "Fixed offline opinion; this fixture does not assess model accuracy.",
         "evidence": [{"doc_index": 0, "quote": payload["documents"][0]["content"]}]}
        for item in payload["requirements"]]}


def fixed_document_reviews(documents, *, unsupported_indices=()):
    """Explicit fixture declarations only; never infer semantic quality from text."""
    return [{"doc_index": index,
             "status": "unsupported" if index in unsupported_indices else "supported",
             "reason": "Fixed offline per-document opinion; not a model quality judgement."}
            for index, _ in enumerate(documents)]
