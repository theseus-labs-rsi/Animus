"""Pure planning and accounting for a generation-owned delivery target.

Legacy ``TargetSpec`` keeps its existing grounding-stage meaning. Version 1
targets count primary questions after four athlete scores and easy removal;
version 2 targets count candidates at generation quality, before athlete runs.
This module performs no I/O, generation, grading, selection, or mutation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math


LINE_IDS = (
    "L1_timeline", "L2_relational", "L3_process", "L4_preference", "L5_conflict",
    "L6_refusal", "L7_consolidation", "L8_transition", "L9_induction", "L10_admission",
)
QUALITY_STATUSES = ("released", "rejected", "pending_review", "scoped_excluded")
_TARGET_KEYS = {
    "version", "final_questions", "count_stage", "requested_lines", "per_line_min",
    "per_line_max", "corpus_tokens", "tokenizer", "max_supply_rounds",
}
_GENERATION_TARGET_KEYS = {
    "version", "count_stage", "candidate_questions", "requested_lines",
    "corpus_tokens", "core_tokens", "filler_ratio", "tokenizer", "max_supply_rounds",
}


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _line(value):
    if value not in LINE_IDS:
        raise ValueError(f"Unknown canonical line: {value!r}")
    return value


def _ordered(lines):
    return [line for line in LINE_IDS if line in lines]


@dataclass(frozen=True)
class AcceptanceScope:
    version: int
    count_stage: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict) or set(value) != {"version","count_stage"} or type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("AcceptanceScope requires version=1 and count_stage")
        if value["count_stage"] not in ("generation_quality","selection_complete"):
            raise ValueError("Unsupported acceptance count stage")
        return cls(1,value["count_stage"])

    def to_dict(self):
        return {"version":self.version,"count_stage":self.count_stage}


@dataclass(frozen=True)
class DeliveryTarget:
    version: int
    final_questions: int
    count_stage: str
    requested_lines: tuple[str, ...]
    per_line_min: dict
    per_line_max: dict
    corpus_tokens: int
    tokenizer: str
    max_supply_rounds: int
    core_tokens: int | None = None
    filler_ratio: int | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "DeliveryTarget":
        """Parse a selection target or a generation-only candidate target."""
        if (isinstance(value, dict) and type(value.get("version")) is int
                and value["version"] == 2):
            if set(value) != _GENERATION_TARGET_KEYS:
                raise ValueError("Generation target requires exactly: " +
                                 ", ".join(sorted(_GENERATION_TARGET_KEYS)))
            if value["count_stage"] != "generation_quality":
                raise ValueError("Generation target count_stage must be generation_quality")
            requested = value["requested_lines"]
            if (not isinstance(requested, list) or not requested
                    or any(not isinstance(line, str) for line in requested)
                    or len(set(requested)) != len(requested)):
                raise ValueError("requested_lines must contain unique canonical line IDs")
            for line in requested:
                _line(line)
            candidates = _integer(value["candidate_questions"], "candidate_questions", 1)
            if candidates < len(requested):
                raise ValueError("candidate_questions must cover every requested line")
            core = _integer(value["core_tokens"], "core_tokens", 1)
            total = _integer(value["corpus_tokens"], "corpus_tokens", 1)
            ratio = _integer(value["filler_ratio"], "filler_ratio")
            if total != core * (ratio + 1):
                raise ValueError("corpus_tokens must equal core_tokens * (filler_ratio + 1)")
            return cls(2, candidates, "generation_quality", tuple(_ordered(requested)),
                       {}, {}, total, _string(value["tokenizer"], "tokenizer"),
                       _integer(value["max_supply_rounds"], "max_supply_rounds", 1),
                       core, ratio)
        if not isinstance(value, dict) or set(value) != _TARGET_KEYS:
            raise ValueError("DeliveryTarget requires exactly: " + ", ".join(sorted(_TARGET_KEYS)))
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("Unsupported DeliveryTarget version")
        if value["count_stage"] != "selection_complete":
            raise ValueError("count_stage must be selection_complete")
        requested = value["requested_lines"]
        if not isinstance(requested, list) or any(not isinstance(line, str) for line in requested):
            raise ValueError("requested_lines must be a list of canonical line IDs")
        if len(set(requested)) != len(requested):
            raise ValueError("requested_lines contains duplicates")
        for line in requested:
            _line(line)
        bounds = {}
        for field in ("per_line_min", "per_line_max"):
            raw = value[field]
            if not isinstance(raw, dict):
                raise ValueError(f"{field} must be an object")
            bounds[field] = {_line(line): _integer(amount, f"{field}.{line}") for line, amount in raw.items()}
            if requested and set(raw) - set(requested):
                raise ValueError(f"{field} refers to a line absent from requested_lines")
        total = _integer(value["final_questions"], "final_questions", 1)
        if sum(bounds["per_line_min"].values()) > total:
            raise ValueError("per_line_min exceeds final_questions")
        for line in bounds["per_line_min"]:
            if bounds["per_line_min"][line] > bounds["per_line_max"].get(line, total):
                raise ValueError(f"Inverted bounds for {line}")
        return cls(1, total, "selection_complete", tuple(_ordered(requested)),
                   bounds["per_line_min"], bounds["per_line_max"],
                   _integer(value["corpus_tokens"], "corpus_tokens", 1),
                   _string(value["tokenizer"], "tokenizer"),
                   _integer(value["max_supply_rounds"], "max_supply_rounds", 1))

    def to_dict(self) -> dict:
        if self.version == 2:
            return {"version": 2, "count_stage": "generation_quality",
                    "candidate_questions": self.final_questions,
                    "requested_lines": list(self.requested_lines),
                    "corpus_tokens": self.corpus_tokens, "core_tokens": self.core_tokens,
                    "filler_ratio": self.filler_ratio, "tokenizer": self.tokenizer,
                    "max_supply_rounds": self.max_supply_rounds}
        return {"version": self.version, "final_questions": self.final_questions,
                "count_stage": self.count_stage, "requested_lines": list(self.requested_lines),
                "per_line_min": deepcopy(self.per_line_min), "per_line_max": deepcopy(self.per_line_max),
                "corpus_tokens": self.corpus_tokens, "tokenizer": self.tokenizer,
                "max_supply_rounds": self.max_supply_rounds}


def resolve_lines(target: DeliveryTarget, applicability: list[dict]) -> dict:
    """Keep requested lines visible, including unsupported and unreported ones.

    Rows: {line, applicable: bool, implemented: bool, reason: nonempty str}.
    An empty requested_lines selects reported applicable/implemented lines plus
    every explicit bound. Explicit bounds therefore remain visible on failure.
    """
    target = DeliveryTarget.from_dict(target.to_dict())
    if not isinstance(applicability, list):
        raise ValueError("applicability must be a list")
    indexed = {}
    for row in applicability:
        if not isinstance(row, dict) or set(row) != {"line", "applicable", "implemented", "reason"}:
            raise ValueError("Invalid applicability row")
        line = _line(row["line"])
        if line in indexed:
            raise ValueError(f"Duplicate applicability row: {line}")
        if type(row["applicable"]) is not bool or type(row["implemented"]) is not bool:
            raise ValueError("applicable and implemented must be booleans")
        _string(row["reason"], "applicability.reason")
        indexed[line] = deepcopy(row)
    desired = set(target.requested_lines)
    if not desired:
        desired = {line for line, row in indexed.items() if row["applicable"] and row["implemented"]}
        desired.update(target.per_line_min)
        desired.update(target.per_line_max)
    rows, issues = [], []
    for line in _ordered(desired):
        row = indexed.get(line)
        if row is None:
            row = {"line": line, "applicable": None, "implemented": None,
                   "reason": "No applicability declaration supplied"}
            status = "support_unreported"
        elif not row["implemented"]:
            status = "implementation_missing"
        elif not row["applicable"]:
            status = "seed_not_applicable"
        else:
            status = "supported"
        rows.append({**row, "status": status})
        if status != "supported":
            issues.append({"line": line, "code": status, "reason": row["reason"]})
    if not rows:
        issues.append({"code": "no_applicable_lines", "reason": "No supported lines or explicit requirements"})
    return {"status": "blocked" if issues else "ready", "lines": rows, "issues": issues}


def allocate_line_targets(target: DeliveryTarget, lines: list[str]) -> dict[str, int]:
    """Bounded equal allocation; canonical line order breaks remainder ties.

    Uses integer water filling, so large targets do not require one iteration
    per question. Explicit min/max constraints take precedence over equality.
    """
    target = DeliveryTarget.from_dict(target.to_dict())
    if not isinstance(lines, list) or len(set(lines)) != len(lines):
        raise ValueError("lines must be a unique list")
    ordered = _ordered({_line(line) for line in lines})
    required = set(target.requested_lines) | set(target.per_line_min) | set(target.per_line_max)
    if required - set(ordered):
        raise ValueError("Cannot omit an explicitly requested or bounded line")
    if not ordered:
        raise ValueError("Cannot allocate without lines")
    floors = {line: target.per_line_min.get(line, 0) for line in ordered}
    ceilings = {line: target.per_line_max.get(line, target.final_questions) for line in ordered}
    if sum(ceilings.values()) < target.final_questions:
        raise ValueError("per_line_max cannot hold final_questions")
    def at(level):
        return {line: min(ceilings[line], max(floors[line], level)) for line in ordered}
    low, high = 0, target.final_questions
    while low < high:
        middle = (low + high + 1) // 2
        if sum(at(middle).values()) <= target.final_questions:
            low = middle
        else:
            high = middle - 1
    allocation = at(low)
    remaining = target.final_questions - sum(allocation.values())
    for line in sorted(ordered, key=lambda lid: (allocation[lid], LINE_IDS.index(lid))):
        if remaining and allocation[line] < ceilings[line]:
            allocation[line] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Inconsistent allocation bounds")
    return allocation


def plan_supply(target: DeliveryTarget, applicability: list[dict], survival_rates: dict,
                *, reserve_factor: float = 1.25) -> dict:
    """Estimate formed candidate reserves from sourced candidate-to-final rates.

    Each rate row contains rate [0,1], source and sample_size. Missing/zero
    rates retain a null candidate estimate and a diagnostic. No prior is invented.
    These estimates do not change answers, quality labels or resource limits.
    """
    resolution = resolve_lines(target, applicability)
    if type(reserve_factor) not in (int, float) or not math.isfinite(reserve_factor) or reserve_factor < 1:
        raise ValueError("reserve_factor must be finite and >= 1")
    if not isinstance(survival_rates, dict):
        raise ValueError("survival_rates must be an object")
    rates = {}
    for line, row in survival_rates.items():
        _line(line)
        if not isinstance(row, dict) or set(row) != {"rate", "source", "sample_size"}:
            raise ValueError("A survival rate requires rate, source and sample_size")
        rate = row["rate"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Survival rate must be finite and between 0 and 1")
        _string(row["source"], "survival.source")
        _integer(row["sample_size"], "survival.sample_size", 1)
        rates[line] = deepcopy(row)
    allocation = allocate_line_targets(target, [row["line"] for row in resolution["lines"]]) if resolution["lines"] else {}
    rows, diagnostics = [], deepcopy(resolution["issues"])
    for support in resolution["lines"]:
        line, amount = support["line"], allocation[support["line"]]
        evidence, reserve, reason = rates.get(line), None, None
        if amount == 0:
            reserve = 0
        elif support["status"] != "supported":
            reason = support["status"]
        elif evidence is None:
            reason = "survival_rate_missing"
        elif evidence["rate"] == 0:
            reason = "zero_observed_retention"
        else:
            reserve = math.ceil(Fraction(amount) * Fraction(str(reserve_factor)) / Fraction(str(evidence["rate"])))
        if reason:
            diagnostics.append({"line": line, "code": reason})
        rows.append({**support, "final_target": amount, "candidate_reserve": reserve,
                     "retention_evidence": evidence, "estimate_issue": reason})
    total = sum(row["candidate_reserve"] for row in rows) if rows and all(row["candidate_reserve"] is not None for row in rows) else None
    return {"version": 1, "target": target.to_dict(), "status": "blocked" if resolution["issues"] else "needs_rate_evidence" if diagnostics else "ready",
            "rate_basis": "formed_candidate_to_complete_selection", "reserve_factor": reserve_factor,
            "allocation": allocation, "lines": rows, "candidate_reserve": total, "issues": diagnostics}


def _index_rows(rows, label):
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list or None")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid {label} row")
        qid = _string(row.get("qid"), f"{label}.qid")
        if qid in indexed:
            raise ValueError(f"Duplicate {label} qid: {qid}")
        _line(row.get("line"))
        _string(row.get("capability"), f"{label}.capability")
        indexed[qid] = row
    return indexed


def is_auxiliary_witness(question: dict) -> bool:
    """The current native helper capability is excluded from primary quotas."""
    return question.get("line") == "L10_admission" and question.get("capability") == "L10_witness"


def quality_rows_from_partition(candidates: list[dict], partition: dict) -> list[dict]:
    """Adapt quality_snapshot()['checks']['partition']['qids'] without I/O."""
    indexed = _index_rows(candidates, "candidates")
    if not isinstance(partition, dict) or set(partition) != set(QUALITY_STATUSES):
        raise ValueError("Expected the four native quality partition lists")
    seen, rows = set(), []
    for status in QUALITY_STATUSES:
        qids = partition[status]
        if not isinstance(qids, list):
            raise ValueError("Quality partition values must be lists")
        for qid in qids:
            if not isinstance(qid, str) or qid not in indexed or qid in seen:
                raise ValueError("Unknown or overlapping quality qid")
            seen.add(qid)
            rows.append({"qid": qid, "line": indexed[qid]["line"], "capability": indexed[qid]["capability"],
                         "quality_status": status})
    if seen != set(indexed):
        raise ValueError("Quality partition does not cover all candidates")
    return rows


def summarize_delivery(target: DeliveryTarget, applicability: list[dict], *,
                       raw_orders=None, effective_orders=None, candidates=None,
                       quality_rows=None, selection_report=None, selection_candidates=None) -> dict:
    """Account for source qids and native filter v3 items, retaining all losses.

    None means that a stage has not produced its artifact. Quality rows use
    {qid,line,capability,quality_status}; the partition adapter supplies these.
    The caller must verify the quality and filter artifacts against the run's
    current world/corpus/model bindings before supplying them here.
    """
    resolution = resolve_lines(target, applicability)
    allocation = allocate_line_targets(target, [r["line"] for r in resolution["lines"]]) if resolution["lines"] else {}
    raw = _index_rows(raw_orders, "raw_orders")
    effective = _index_rows(effective_orders, "effective_orders")
    formed = _index_rows(candidates, "candidates")
    quality = _index_rows(quality_rows, "quality_rows")
    evaluated = _index_rows(selection_candidates, "selection_candidates") if selection_candidates is not None else formed
    for parent, child, label in ((raw, effective, "effective_orders"), (effective, formed, "candidates"), (formed, quality, "quality_rows")):
        if child is not None and parent is None:
            raise ValueError(f"{label} supplied before its source stage")
        if child is not None:
            if set(child) - set(parent):
                raise ValueError(f"{label} contains unknown source qids")
            for qid, row in child.items():
                if any(row[field] != parent[qid][field] for field in ("line", "capability")):
                    raise ValueError(f"{label} changes source identity: {qid}")
    if quality is not None:
        if set(quality) != set(formed):
            raise ValueError("Quality rows must partition every formed candidate")
        if any(row.get("quality_status") not in QUALITY_STATUSES for row in quality.values()):
            raise ValueError("Unknown quality_status")
    if evaluated is not None and formed is not None:
        if set(evaluated) - set(formed):
            raise ValueError("Evaluated questions contain unknown candidate qids")
        for qid, row in evaluated.items():
            if any(row[field] != formed[qid][field] for field in ("line", "capability")):
                raise ValueError(f"Evaluated question changes source line or capability: {qid}")
    selections = None
    delivery_qids = None
    if selection_report is not None:
        from eval.question_filter import question_key
        if quality is None:
            raise ValueError("Selection supplied before quality partition")
        if not isinstance(selection_report, dict):
            raise ValueError("selection_report must be an object")
        # stage_selection's native empty branch has no filter/athlete metadata.
        # Accept that exact zero-result shape only when the quality set is empty.
        counts = selection_report.get("counts")
        native_empty = (selection_report.get("version") == 1
                        and "schema_version" not in selection_report
                        and selection_report.get("status") == "empty"
                        and selection_report.get("rule") == "all_four_athletes_correct"
                        and selection_report.get("release_ready") is False
                        and selection_report.get("items") == []
                        and isinstance(counts, dict)
                        and all(type(counts.get(k)) is int and counts[k] == 0 for k in ("input", "kept"))
                        and not any(q["quality_status"] == "released" for q in quality.values()))
        systems = selection_report.get("systems")
        if not native_empty:
            if selection_report.get("schema_version") != 3:
                raise ValueError("Expected native filter schema_version=3")
            if (not isinstance(systems, list) or len(systems) != 4 or
                    any(not isinstance(s, str) or not s.strip() for s in systems) or len(set(systems)) != 4):
                raise ValueError("Delivery requires four distinct athlete systems")
            if selection_report.get("rule") not in {"all_selected_systems_correct", "all_four_athletes_correct"}:
                raise ValueError("Unexpected selection rule")
            ratio = selection_report.get("keep_easy_ratio")
            if type(ratio) not in (int, float) or ratio != 0 or selection_report.get("result_scope") != "validated_input_identity":
                raise ValueError("Delivery requires zero easy retention and validated input identity")
        selections = _index_rows(selection_report.get("items"), "selection.items")
        if selections is None:
            raise ValueError("Selection items are required")
        for qid, item in selections.items():
            if qid not in quality or quality[qid]["quality_status"] != "released":
                raise ValueError(f"Selection includes an unreleased or unknown qid: {qid}")
            source = evaluated.get(qid) if evaluated is not None else None
            if source is None:
                raise ValueError(f"Selection question has no evaluated source: {qid}")
            if any(item[field] != source[field] for field in ("line", "capability")) or item.get("key") != question_key(source) or item.get("question") != source.get("question"):
                raise ValueError(f"Selection reference identity changed: {qid}")
            grades, issues = item.get("correct"), item.get("issues")
            if not isinstance(grades, dict) or set(grades) != set(systems) or not isinstance(issues, dict):
                raise ValueError("Invalid selection grades/issues")
            if any(value is not None and type(value) is not bool for value in grades.values()):
                raise ValueError("Athlete grades must be boolean or null")
            expected = "kept_incomplete" if issues or any(v is None for v in grades.values()) else "removed_easy" if all(grades.values()) else "kept_not_all_correct"
            if item.get("disposition") != expected:
                raise ValueError(f"Selection disposition contradicts actual grades: {qid}")
        subset = selection_report.get("delivery_subset")
        if subset is not None:
            if (not isinstance(subset, dict) or subset.get("version") != 1
                    or subset.get("target") != target.to_dict()
                    or subset.get("ideal_allocation") != allocation
                    or not isinstance(subset.get("qids"), list)
                    or len(set(subset["qids"])) != len(subset["qids"])):
                raise ValueError("Invalid delivery subset binding")
            delivery_qids = set(subset["qids"])
            if any(qid not in selections or selections[qid]["disposition"] != "kept_not_all_correct"
                   for qid in delivery_qids):
                raise ValueError("Delivery subset contains an unscored or easy item")
            by_line = {line: sum(qid in delivery_qids and not is_auxiliary_witness(formed[qid])
                                 and formed[qid]["line"] == line for qid in delivery_qids)
                       for line in allocation}
            if subset.get("selected_by_line") != by_line:
                raise ValueError("Delivery subset line counts differ from qids")
            surplus = {qid for qid, item in selections.items()
                       if item["disposition"] == "kept_not_all_correct" and qid not in delivery_qids}
            if (not isinstance(subset.get("surplus_qids"), list)
                    or set(subset["surplus_qids"]) != surplus):
                raise ValueError("Delivery surplus does not partition filtered items")

    observed_lines = set(allocation)
    for indexed in (raw, effective, formed):
        observed_lines.update(row["line"] for row in (indexed or {}).values())
    def count(indexed, line, auxiliary=False):
        if indexed is None:
            return None
        return sum(row["line"] == line and is_auxiliary_witness(row) == auxiliary for row in indexed.values())
    rows = []
    for line in _ordered(observed_lines):
        record = {"line": line, "final_target": allocation.get(line, 0)}
        for role, auxiliary in (("primary", False), ("auxiliary_witness", True)):
            qrows = [row for row in (quality or {}).values() if row["line"] == line and is_auxiliary_witness(row) == auxiliary]
            statuses = Counter(row["quality_status"] for row in qrows)
            released = {row["qid"] for row in qrows if row["quality_status"] == "released"}
            dispositions = Counter(selections[qid]["disposition"] if qid in selections else "kept_incomplete" for qid in released) if selections is not None else None
            delivered = (None if dispositions is None else
                         sum(qid in delivery_qids for qid in released) if delivery_qids is not None else
                         dispositions["kept_not_all_correct"])
            rcount, ecount, ccount = count(raw, line, auxiliary), count(effective, line, auxiliary), count(formed, line, auxiliary)
            record[role] = {"raw_orders": rcount, "effective_orders": ecount, "formed": ccount,
                            "order_dropped": None if rcount is None or ecount is None else rcount - ecount,
                            "unformed": None if ecount is None or ccount is None else ecount - ccount,
                            **{s: statuses[s] if quality is not None else None for s in QUALITY_STATUSES},
                            "athlete_complete": None if dispositions is None else dispositions["removed_easy"] + dispositions["kept_not_all_correct"],
                            "athlete_incomplete": None if dispositions is None else dispositions["kept_incomplete"],
                            "removed_easy": None if dispositions is None else dispositions["removed_easy"],
                            "available_after_filter": None if dispositions is None else dispositions["kept_not_all_correct"],
                            "final_selected": delivered}
        selected = record["primary"]["final_selected"]
        record["shortfall"] = None if selected is None else max(0, record["final_target"] - selected)
        record["excess_available"] = None if selected is None else max(0, selected - record["final_target"])
        ceiling = target.per_line_max.get(line)
        record["above_explicit_max"] = None if selected is None else max(0, selected - ceiling) if ceiling is not None else 0
        rows.append(record)
    metrics = list(rows[0]["primary"]) if rows else ["raw_orders", "effective_orders", "formed", "order_dropped", "unformed", *QUALITY_STATUSES, "athlete_complete", "athlete_incomplete", "removed_easy", "available_after_filter", "final_selected"]
    totals = {}
    for role in ("primary", "auxiliary_witness"):
        totals[role] = {name: None if not rows or any(row[role][name] is None for row in rows) else sum(row[role][name] for row in rows) for name in metrics}
    final = totals["primary"]["final_selected"]
    complete = selections is not None and all((row[role]["athlete_incomplete"] or 0) == 0 for row in rows for role in ("primary", "auxiliary_witness"))
    over_max = any((row["above_explicit_max"] or 0) > 0 for row in rows)
    line_floor_met = (all(row["primary"]["final_selected"] >= target.per_line_min.get(row["line"], 0)
                          for row in rows if row["line"] in allocation) if delivery_qids is not None else
                      all(row["shortfall"] == 0 for row in rows if row["final_target"]))
    met = bool(resolution["status"] == "ready" and complete and not over_max and final is not None
               and final >= target.final_questions and line_floor_met)
    return {"version": 1, "target": target.to_dict(), "line_resolution": resolution,
            "allocation": allocation, "by_line": rows, "totals": totals,
            "selection_complete": complete, "question_target_met": met,
            "distribution_within_explicit_max": None if selections is None else not over_max,
            "total_shortfall": None if final is None else max(0, target.final_questions - final),
            "status": "blocked" if resolution["status"] != "ready" else "not_run" if selections is None else "incomplete" if not complete else "exceeds_line_max" if over_max else "met" if met else "shortfall",
            "corpus_target": {"tokens": target.corpus_tokens, "tokenizer": target.tokenizer, "measurement": "not_supplied"},
            "note": "Question accounting; caller separately verifies source bindings, corpus token size and witness scoring dependencies"}
