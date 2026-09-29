"""Identity of the public evaluation inputs, without importing model config.

Hashes record what was supplied; they do not certify factual correctness. Legacy
results without these identities remain readable but cannot establish difficulty.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = 1
VISIBLE_CORPUS_VERSION = "session-date-body-v1"
REFERENCE_FIELDS = ("line", "capability", "gt", "gold", "reference_proposal", "aux", "strict_scoring",
                    "question_contract", "question_version", "reference", "reference_version", "rubric",
                    "entity", "field", "entity_type", "answer_entity_type", "answer_field",
                    "evidence_sessions", "evidence_doc_ids", "reference_basis")
CONTEXT_FIELDS = ("version", "corpus_view", "corpus_hash", "protocol_hash", "context_id")


class ResultRows(list):
    """List-compatible rows retaining declared file scope even for an empty run."""

    def __init__(self, rows=(), *, result_scopes=()):
        super().__init__(rows)
        self.result_scopes = list(result_scopes)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def public_question(item) -> dict:
    """Only the question actually sent to the solver, not labels/gold/old scores."""
    return {"question": item.get("question")} if isinstance(item, dict) else {"question": item}


def reference_payload(item: dict) -> dict:
    return {name: item.get(name) for name in REFERENCE_FIELDS}


def question_hash(item) -> str:
    return digest({**public_question(item),
                   "question_version": item.get("question_version") if isinstance(item, dict) else None})


def reference_hash(item: dict) -> str:
    return digest(reference_payload(item))


def load_visible_corpus(path: Path) -> list:
    """The harness view: stable session sort, original doc order, date and body.

    Titles, fact refs and filler/conflict labels are not sent by the current
    harness and therefore are not secretly included in this public-input hash.
    Both the existing wrapped corpus and legacy bare sessions are readable.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    value = data.get("corpus", data)
    docs = []
    for session in sorted(value["sessions"], key=lambda s: int(s["session_id"])):
        for document in session["docs"]:
            content = document["content"]
            if not isinstance(content, str):
                raise ValueError("Corpus document content must be text")
            docs.append((int(session["session_id"]), session.get("date", ""), content))
    return docs


def protocol_scoring_policy(protocol):
    """Read an exact published policy block without adding or repairing one."""
    from eval.answer_task_review import _START, _END, POLICY_TEXT, POLICY_VERSION, append_scoring_policy
    if not isinstance(protocol, str):
        raise ValueError("Public protocol must be text")
    if _START not in protocol and _END not in protocol and POLICY_TEXT not in protocol:
        return None
    if append_scoring_policy(protocol) != protocol:
        raise ValueError("Public scoring policy must use its exact versioned block")
    return POLICY_VERSION


def public_protocol(about: dict) -> str:
    """Render exactly the public rules that the harness injects."""
    if "public_protocol" in about:
        # Research corpora can provide their exact public scope without
        # inheriting domain-specific legacy conventions or sentinel rules.
        protocol = about["public_protocol"]
        if not isinstance(protocol, str):
            raise ValueError("public_protocol must be explicit text")
        if "answer_protocol" in about:
            raise ValueError("Use one public protocol representation, not both")
        return protocol
    ap = about.get("answer_protocol", {})
    if not ap:
        return ""
    out = ["【答题约定(随题库交付,务必遵守)】"]
    for rule in ap.get("rules", []):
        out.append(f"- {rule}")
    sentinels = ap.get("gold_sentinel_map", {})
    if sentinels:
        refusal = (f"- 拒答措辞:从未涉及→『{sentinels.get('INSUFFICIENT', '无此项')}』;"
                   f"已停统→『{sentinels.get('forgotten=true', '已停止统计')}』")
        if "out_of_scope" in sentinels:
            refusal += f";超出记录时间范围→『{sentinels['out_of_scope']}』"
        out.append(refusal + "。")
    out.append("- 个人/角色不具备案件级属性;问及某实体它本身没有的属性 → 答『无此项/查无』,"
               "不得经关系链折算到关联实体的值。")
    protocol = "\n".join(out)
    if "scoring_policy" in ap:
        from eval.answer_task_review import append_scoring_policy
        protocol = append_scoring_policy(protocol, ap["scoring_policy"])
    return protocol


def load_public_protocol(path: Path) -> str:
    if not Path(path).is_file():
        return ""
    # Corrupt files are input errors, not an implicit no-protocol experiment.
    return public_protocol(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def make_evaluation_context(docs: list, protocol: str) -> dict:
    if not isinstance(protocol, str):
        raise ValueError("The actual public protocol must be text")
    base = {"version": VERSION, "corpus_view": VISIBLE_CORPUS_VERSION,
            "corpus_hash": digest(docs), "protocol_hash": digest(protocol)}
    return {**base, "context_id": digest(base)}


def valid_context(context) -> bool:
    if (not isinstance(context, dict) or type(context.get("version")) is not int
            or context["version"] != VERSION):
        return False
    if context.get("corpus_view") != VISIBLE_CORPUS_VERSION:
        return False
    if any(not isinstance(context.get(k), str) or len(context[k]) != 64
           or any(c not in "0123456789abcdef" for c in context[k])
           for k in ("corpus_hash", "protocol_hash", "context_id")):
        return False
    base = {k: context[k] for k in CONTEXT_FIELDS if k != "context_id"}
    return context["context_id"] == digest(base)


def record_provenance(item: dict, context: dict) -> dict:
    if not valid_context(context):
        raise ValueError("A record needs an identified public evaluation context")
    return {**{k: context[k] for k in CONTEXT_FIELDS},
            "question_hash": question_hash(item), "reference_hash": reference_hash(item)}


def provenance_issue(record: dict, item: dict, expected_context: dict | None) -> str | None:
    """Return a reason instead of upgrading unidentified historical results."""
    if not valid_context(expected_context):
        return "missing_expected_evaluation_context"
    actual = record.get("evaluation_provenance")
    if not valid_context(actual):
        return "missing_or_invalid_evaluation_provenance"
    for field in ("corpus_hash", "protocol_hash", "corpus_view", "context_id"):
        if actual[field] != expected_context[field]:
            return "evaluation_" + field + "_mismatch"
    if actual.get("question_hash") != question_hash(item):
        return "evaluation_question_mismatch"
    if actual.get("reference_hash") != reference_hash(item):
        return "evaluation_reference_mismatch"
    # Preserve and check declared aggregate/system contexts. Do not infer hashes
    # from a path that may now point to different material.
    envelopes = record.get("_result_contexts", [])
    if not isinstance(envelopes, list):
        return "invalid_result_envelope_context"
    for envelope in envelopes:
        if not valid_context(envelope):
            return "invalid_result_envelope_context"
        if envelope["context_id"] != actual["context_id"]:
            return "result_envelope_context_mismatch"
    return None


def preserve_result_contexts(rows: list[dict], *envelopes) -> list[dict]:
    """Carry supplied file-level identity through flattening, without inventing it."""
    contexts = [deepcopy(e["evaluation_context"]) for e in envelopes
                if isinstance(e, dict) and "evaluation_context" in e]
    legacy = [{k: deepcopy(e[k]) for k in ("bench", "corpus", "protocol_injected", "model") if k in e}
              for e in envelopes if isinstance(e, dict)]
    scopes = [*getattr(rows, "result_scopes", []),
              *[e["result_scope"] for e in envelopes if isinstance(e, dict) and "result_scope" in e]]
    result = ResultRows(result_scopes=deepcopy(scopes))
    for source in rows:
        row = deepcopy(source)
        if contexts:
            previous = row.get("_result_contexts", [])
            row["_result_contexts"] = [*(previous if isinstance(previous, list) else [previous]), *contexts]
        if scopes:
            previous_scopes = row.get("_result_scopes", [])
            row["_result_scopes"] = [*(previous_scopes if isinstance(previous_scopes, list) else [previous_scopes]), *scopes]
        if any(legacy):
            row["_result_legacy_context"] = legacy
        result.append(row)
    return result
