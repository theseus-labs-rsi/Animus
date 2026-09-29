"""Lossless decoding of two explicit locations for one grading control pair.

This does not infer a verdict, a dispute, or a reason. The original response
must remain in the caller's audit log; normal grading validation still applies.
"""
from copy import deepcopy

from pipeline.semantic_review import fingerprint


VERSION = "grading-reassessment-layout/v1"
FIELDS = ("requires_item_reassessment", "reassessment_reason")


def normalize(response):
    if not isinstance(response, dict):
        raise ValueError("Grading response must be an object")
    nested = response.get("answer_review")
    nested = nested if isinstance(nested, dict) else {}
    present = {name: {key for key in FIELDS if key in container}
               for name, container in (("top", response), ("nested", nested))}
    # A split pair could express two different decisions. Never assemble one
    # from different locations, default a missing boolean, or parse prose.
    for name, keys in present.items():
        if keys and keys != set(FIELDS):
            raise ValueError("Incomplete reassessment pair at " + name)
    if not any(present.values()):
        raise ValueError("Missing explicit reassessment pair")
    for name, container in (("top", response), ("nested", nested)):
        if not present[name]:
            continue
        if type(container[FIELDS[0]]) is not bool or not isinstance(container[FIELDS[1]], str):
            raise ValueError("Invalid reassessment pair types at " + name)
        if container[FIELDS[0]] and not container[FIELDS[1]].strip():
            raise ValueError("Requested reassessment requires a nonempty reason")
    if present["top"] and present["nested"] and any(response[key] != nested[key] for key in FIELDS):
        raise ValueError("Conflicting reassessment pairs")
    result = deepcopy(response)
    actions = []
    if not present["top"]:
        for key in FIELDS:
            result[key] = deepcopy(nested[key])
            actions.append({"operation": "copy", "from": "/answer_review/" + key, "to": "/" + key})
    # Keep even identical nested fields. Nothing in the model's response is
    # deleted, rewritten, type-coerced, or treated as a semantic approval.
    return result, {"version": VERSION, "actions": actions,
        "raw_output_hash": fingerprint(response), "normalized_output_hash": fingerprint(result),
        "scope": "Explicit control-field location only; semantic and citation validation still required."}
