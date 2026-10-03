"""Explicit inheritance of spent corpus-author slots across renderer versions.

The old checkpoint remains immutable.  A separately prepared manifest binds
each old slot to the *same complete author input* under the new renderer.
Changing the task, model, transport, or review contract refuses inheritance;
it never turns a historical reply into a new semantic opinion.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pipeline.semantic_review import fingerprint
from pipeline import corpus_contract
from pipeline import evidence_transport as wire

VERSION = "signal-slot-inheritance/v1"


def _digest_key(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _verified_manifest(run_dir):
    path = run_dir / "05_signal_inheritance.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or not isinstance(manifest.get("groups"), list):
        raise ValueError("Unsupported signal inheritance manifest")
    rows = manifest["groups"]
    if (not all(isinstance(item, dict) and _digest_key(item.get("key"))
                and _digest_key(item.get("legacy_key")) for item in rows)
            or len({item["key"] for item in rows}) != len(rows)
            or len({item["legacy_key"] for item in rows}) != len(rows)):
        raise ValueError("Signal inheritance maps a historical slot more than once")
    if manifest.get("hash") != fingerprint({k: v for k, v in manifest.items() if k != "hash"}):
        raise ValueError("Signal inheritance manifest changed")
    return manifest


def _verify_review(report, task, blind_answers):
    if report.get("status") not in ("passed", "failed"):
        raise ValueError("Incomplete historical review needs separate trace reconciliation")
    history = report.get("review_validation")
    if (report.get("review_validation_hash") != fingerprint(history)
            or report.get("review_contract_hash") != corpus_contract._review_contract_hash()):
        raise ValueError("Historical review contract or validation changed")
    result, payload = corpus_contract._replay_review_validation(history)
    if (payload["CANON"] != task["context"]
            or report.get("context_hash") != fingerprint(task["context"])
            or report.get("fidelity_requirements") != task["requirements"]
            or payload["requirements"][:len(task["requirements"])] != task["requirements"]
            or payload["blind_reads"] != report.get("blind_reads")
            or any(blind_answers.get(item["key"]) != item["answer"]
                   for item in payload["blind_reads"])):
        raise ValueError("Historical review belongs to a different fact, scope, or blind read")
    for key, value in result.items():
        if report.get(key) != value:
            raise ValueError("Historical review outcome changed: " + key)


def inherit_progress(run_dir, *, key, group, task, material_needs, system,
                     quality, session_label, time_unit, date, model,
                     reviewer_model, discriminator_model, transport):
    """Return a verified copy of old progress, or None when no manifest exists.

    The caller must write the returned progress under its new task key before
    dispatch.  A manifest covers a closed set of groups: a missing entry fails
    instead of silently buying a second author attempt for an old task.
    """
    manifest = _verified_manifest(run_dir)
    if manifest is None:
        return None
    if (transport != wire.INLINE or manifest.get("model") != model
            or manifest.get("reviewer_model") != reviewer_model
            or manifest.get("discriminator_model") != discriminator_model):
        raise ValueError("Historical author, reviewer or blind-read model/transport changed")
    rows = [item for item in manifest["groups"] if item["key"] == key]
    if len(rows) != 1:
        raise ValueError("No uniquely inherited slot for this corpus task")
    item = rows[0]
    if (item.get("group_hash") != fingerprint(group)
            or item.get("task_hash") != fingerprint(task)
            or item.get("material_needs_hash") != fingerprint(material_needs)):
        raise ValueError("Historical author task or capacity need changed")
    old_path = run_dir / "05_signal_checkpoints" / (item["legacy_key"] + ".json")
    old_bytes = old_path.read_bytes()
    if _sha(old_bytes) != item["legacy_sha256"]:
        raise ValueError("Historical signal checkpoint bytes changed")
    old = json.loads(old_bytes)
    progress = old.get("progress")
    if (old.get("key") != item["legacy_key"] or not isinstance(progress, dict)
            or old.get("progress_hash") != fingerprint(progress)
            or ("documents" in old and old.get("hash") != fingerprint(old["documents"]))):
        raise ValueError("Historical signal checkpoint or progress changed")
    if old.get("acceptance_binding") is not None:
        raise ValueError("Expected an original signal checkpoint, not relabeled acceptance")
    drafts = progress.get("drafts", [])
    if (not isinstance(drafts, list) or len(drafts) > 4
            or len(drafts) != len(item["drafts"])):
        raise ValueError("Historical author slot count changed")
    if "documents" in old and (not drafts or
            (drafts[-1].get("quality_review") or {}).get("status") != "passed"):
        raise ValueError("Historical completed documents lack their final accepted review")
    plan = progress.get("planning", {})
    if plan.get("status") not in ("planned", "fallback"):
        raise ValueError("Historical optional planning call is not settled")
    plan_text = ("\n【作者的辅助文档编排；按原事实写作，不能改变事实或时点】\n"
                 + json.dumps(plan["plan"], ensure_ascii=False)
                 + "\nsources 中 fact:N、event:N、source:N 分别对应本组 facts、events、source_assertions 的零起始索引。"
                 if plan["status"] == "planned" else "")
    # Import locally to keep the production renderer's one request builder as
    # the sole authority for complete prompt spelling and call parameters.
    from pipeline.render import _signal_request
    inherited = deepcopy(progress)
    for index, (draft, receipt) in enumerate(zip(drafts, item["drafts"])):
        if (receipt.get("position") != index or not draft.get("admitted")
                or not isinstance(draft.get("hint"), str)):
            raise ValueError("Historical author slot identity changed")
        messages, _binding, parameters = _signal_request(system, quality=quality,
            session_label=session_label, time_unit=time_unit, date=date,
            facts=task["facts"], events=task["events"],
            story_context=task["story_context"], hint=draft["hint"],
            plan_text=plan_text, context=task["context"], attempt=index,
            evidence_transport=transport)
        if (receipt.get("messages_hash") != fingerprint(messages)
                or receipt.get("parameters") != parameters):
            raise ValueError("Historical author request differs in complete messages or parameters")
        copied = inherited["drafts"][index]
        copied["author_request"] = {"messages": messages, "parameters": parameters}
        if "author_output" not in draft:
            if any(k in draft for k in ("blind_answers", "quality_review")):
                raise ValueError("Historical admitted call lacks its author reply")
            continue  # Keep the consumed position; never redispatch it.
        if (not receipt.get("saved_author_reply") or not receipt.get("matching_original_physical_requests")
                or receipt.get("author_output_hash") != fingerprint(draft["author_output"])):
            raise ValueError("Historical author reply lacks its exact physical request")
        if "blind_answers" not in draft or "quality_review" not in draft:
            raise ValueError("Historical downstream subcall needs separate trace reconciliation")
        _verify_review(draft["quality_review"], task, draft["blind_answers"])
    inherited["legacy_inheritance"] = {"version": VERSION,
        "legacy_key": item["legacy_key"], "legacy_sha256": item["legacy_sha256"],
        "manifest_hash": manifest["hash"], "original_progress_hash": old["progress_hash"],
        "semantic_revalidation": False, "new_author_positions": 0}
    return inherited
