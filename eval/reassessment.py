"""Current-review, public-input-bound quarantine of semantic grading disputes.

This is routing, not another judge. A dispute survives cache resume until a new
independent review changes its review hash. Predictions and legacy grades are not
deleted. The original triggering judgement and every replaced judgement remain
available for audit.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading

from eval.grading import (SEMANTIC_JUDGE_VERSION, ANSWER_TASK_JUDGE_VERSION,
                          is_semantic_grade_version, valid_answer_task_grade)
from eval.provenance import digest, question_hash, reference_hash, valid_context

VERSION = "semantic-reassessment-closure/v1"


def closure_path(cache_dir, evaluation_context, review_hash):
    scope = {"evaluation_context": evaluation_context, "review_hash": review_hash}
    return Path(cache_dir) / "reassessment" / (digest(scope) + ".json")


class ReassessmentTracker:
    """Thread-safe, optional durable closure for one context and review version.

    Public item identity also includes qid, question/reference hashes and actual
    reference presence. No question-text matching or vote decides a quarantine.
    ``observe`` saves new triggers before returning. Storage failures propagate;
    losing a dispute must not silently allow cached scored rows to re-enter.
    """
    def __init__(self, evaluation_context, review_hash, *, path=None):
        if not valid_context(evaluation_context):
            raise ValueError("Reassessment requires an identified evaluation context")
        if not isinstance(review_hash, str) or len(review_hash) != 64:
            raise ValueError("Reassessment requires the current review hash")
        self.context = deepcopy(evaluation_context)
        self.review_hash = review_hash
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._entries = {}
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if (saved.get("version") != VERSION or saved.get("evaluation_context") != self.context
                    or saved.get("review_hash") != self.review_hash):
                raise ValueError("Stored reassessment closure has a different input/review scope")
            for entry in saved.get("items", []):
                identity = entry.get("identity", {})
                if (entry.get("key") != digest(identity)
                        or identity.get("context_id") != self.context["context_id"]
                        or identity.get("review_hash") != self.review_hash
                        or not isinstance(entry.get("triggers"), list) or not entry["triggers"]):
                    raise ValueError("Invalid stored reassessment identity or trigger")
                self._entries[entry["key"]] = deepcopy(entry)

    def validate_context(self, evaluation_context, review_hash):
        if evaluation_context != self.context or review_hash != self.review_hash:
            raise ValueError("Reassessment tracker differs from the current semantic grader context")

    @staticmethod
    def _semantic(record):
        grade = record.get("judgement") or {}
        return record.get("mode") == "semantic" or is_semantic_grade_version(grade.get("version"))

    def _identity(self, record):
        if not self._semantic(record):
            return None
        provenance = record.get("evaluation_provenance")
        if not valid_context(provenance) or provenance["context_id"] != self.context["context_id"]:
            return None
        qhash, rhash = question_hash(record), reference_hash(record)
        if provenance.get("question_hash") != qhash or provenance.get("reference_hash") != rhash:
            return None
        grade = record.get("judgement") or {}
        if is_semantic_grade_version(grade.get("version")) and grade.get("review_hash") != self.review_hash:
            return None
        if grade.get("version") == ANSWER_TASK_JUDGE_VERSION and not valid_answer_task_grade(
                grade, record, require_opinion=False):
            return None
        return {"context_id": self.context["context_id"], "question_hash": qhash,
                "source_qid": deepcopy(record.get("qid")), "reference_hash": rhash,
                "reference_provided": any(k in record for k in ("reference_proposal", "gold", "gt")),
                "review_hash": self.review_hash}

    def observe(self, records, *, system):
        """Persist each well-bound explicit semantic dispute, keeping its source."""
        changed = False
        with self._lock:
            for record in records:
                grade = record.get("judgement") or {}
                if not is_semantic_grade_version(grade.get("version")) or grade.get("requires_item_reassessment") is not True:
                    continue
                provenance = record.get("evaluation_provenance")
                if (grade.get("review_hash") != self.review_hash or
                        (valid_context(provenance) and provenance["context_id"] != self.context["context_id"])):
                    continue  # A different review/context is a different closure.
                if ((record.get("reassessment") or {}).get("version") == VERSION
                        and grade.get("execution_status") == "quarantined"):
                    continue  # Do not turn propagated quarantine into another vote.
                identity = self._identity(record)
                if identity is None or any(grade.get(field) != identity[field] for field in
                        ("question_hash", "reference_hash", "reference_provided", "review_hash")):
                    raise ValueError("Semantic dispute is not bound to the current public item and review")
                if "source_qid" not in grade or digest(grade["source_qid"]) != digest(identity["source_qid"]):
                    raise ValueError("Semantic dispute source qid differs from its public item")
                key = digest(identity)
                trigger = {"system": system, "judgement": deepcopy(grade),
                           "prediction": deepcopy(record.get("pred")),
                           "evaluation_provenance": deepcopy(record["evaluation_provenance"])}
                trigger["trigger_id"] = digest(trigger)
                entry = self._entries.setdefault(key, {"key": key, "identity": identity, "triggers": []})
                if trigger["trigger_id"] not in {t["trigger_id"] for t in entry["triggers"]}:
                    entry["triggers"].append(trigger)
                    changed = True
            if changed:
                self._save()
        return changed

    def is_disputed(self, record):
        with self._lock:
            identity = self._identity(record)
            return identity is not None and digest(identity) in self._entries

    def quarantine_record(self, record):
        """Mutate only grade disposition; retain prediction, provenance and scope."""
        with self._lock:
            identity = self._identity(record)
            if identity is None or digest(identity) not in self._entries:
                return False
            key = digest(identity)
            entry = self._entries[key]
            old_marker = record.get("reassessment") or {}
            if old_marker.get("version") != VERSION or old_marker.get("key") != key:
                record["original_judgement"] = deepcopy(record.get("judgement"))
                record["original_correct"] = record.get("correct")
            grade = deepcopy(record.get("judgement") or {
                "version": SEMANTIC_JUDGE_VERSION, "path": "llm_semantic",
                **{k: deepcopy(v) for k, v in identity.items() if k != "context_id"}})
            grade.update(verdict="uncertain", correct=None, execution_status="quarantined",
                requires_item_reassessment=True,
                reason="A same-item dispute under the current review requires fresh independent reassessment for all systems.")
            grade["item_reassessment"] = {**deepcopy(grade.get("item_reassessment") or {}),
                "required": True, "reasons": ["same_item_dispute"],
                "scope": "same_item_all_systems", **{k: deepcopy(v) for k, v in identity.items() if k != "context_id"}}
            record.update(judgement=grade, correct=None, quarantined=True,
                reassessment={"version": VERSION, "key": key, "identity": identity,
                    "scope": "same_item_all_systems", "trigger_systems": sorted({t["system"] for t in entry["triggers"]})})
            return True

    def close(self, records_by_system):
        """Observe all systems before propagating disputes to earlier cached rows."""
        for system, rows in records_by_system.items():
            self.observe(rows, system=system)
        counts = {system: sum(self.quarantine_record(row) for row in rows)
                  for system, rows in records_by_system.items()}
        return {**self.snapshot(), "quarantined_by_system": counts}

    def snapshot(self):
        with self._lock:
            return {"version": VERSION, "evaluation_context": deepcopy(self.context),
                    "review_hash": self.review_hash, "scope": "same_item_all_systems",
                    "items": deepcopy(list(self._entries.values())), "n_disputed_items": len(self._entries)}

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=self.path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.snapshot(), stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
