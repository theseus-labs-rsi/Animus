"""Offline target allocation and actual native-filter accounting regressions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.delivery_target import (
    DeliveryTarget, LINE_IDS, allocate_line_targets, resolve_lines, plan_supply,
    quality_rows_from_partition, summarize_delivery,
)
from eval.question_filter import filter_questions
from eval.grading import JUDGE_VERSION
from eval.provenance import make_evaluation_context, record_provenance
from pipeline.supply import balanced_selection_subset


def target(total=500, lines=LINE_IDS, **updates):
    data = {"version": 1, "final_questions": total, "count_stage": "selection_complete",
            "requested_lines": list(lines), "per_line_min": {}, "per_line_max": {},
            "corpus_tokens": 1000000, "tokenizer": "test-encoding@1", "max_supply_rounds": 2}
    data.update(updates)
    return DeliveryTarget.from_dict(data)


def support(lines=LINE_IDS):
    return [{"line": line, "applicable": True, "implemented": True, "reason": "Declared fixture structure"} for line in lines]


def question(qid, line=LINE_IDS[0], capability="IE"):
    return {"qid": qid, "line": line, "capability": capability, "question": "问题" + qid,
            "gt": {"value": "答案" + qid}, "aux": {"at_week": 1}, "evidence_sessions": [1]}


def filter_report(questions, *, all_correct=(), missing=()):
    context = make_evaluation_context([], "fixture protocol")
    systems = {}
    for athlete in ("A", "B", "C", "D"):
        rows = []
        for q in questions:
            if (athlete, q["qid"]) in missing:
                continue
            correct = athlete != "A" or q["qid"] in all_correct
            rows.append({**deepcopy(q), "pred": "fixture answer", "judgeable": True, "correct": correct,
                         "judgement": {"version": JUDGE_VERSION, "verdict": "correct" if correct else "incorrect",
                                       "correct": correct, "path": "offline_fixture", "reason": "fixture"},
                         "evaluation_provenance": record_provenance(q, context)})
        systems[athlete] = rows
    return filter_questions(questions, systems, expected_context=context)[1]


def partition(candidates, **statuses):
    data = {"released": [q["qid"] for q in candidates], "rejected": [], "pending_review": [], "scoped_excluded": []}
    data.update(statuses)
    return quality_rows_from_partition(candidates, data)


class DeliveryTargetTests(unittest.TestCase):
    def test_balanced_export_keeps_scored_pool_and_reports_surplus(self):
        lines = LINE_IDS[:2]
        spec = target(4, lines)
        questions = [question(f"q{i}", lines[0]) for i in range(5)] + [question("q5", lines[1])]
        native = filter_report(questions)
        chosen, subset = balanced_selection_subset(questions, native, spec.to_dict())
        self.assertEqual(subset["selected_by_line"], {lines[0]: 3, lines[1]: 1})
        self.assertEqual((len(chosen), len(subset["surplus_qids"])), (4, 2))
        native["delivery_subset"] = subset
        result = summarize_delivery(spec, support(lines), raw_orders=questions,
            effective_orders=questions, candidates=questions, quality_rows=partition(questions),
            selection_report=native)
        self.assertEqual(result["totals"]["primary"]["available_after_filter"], 6)
        self.assertEqual(result["totals"]["primary"]["final_selected"], 4)
        self.assertTrue(result["question_target_met"])
        damaged = deepcopy(native)
        damaged["delivery_subset"]["qids"][0] = "absent"
        with self.assertRaises(ValueError):
            summarize_delivery(spec, support(lines), raw_orders=questions,
                effective_orders=questions, candidates=questions, quality_rows=partition(questions),
                selection_report=damaged)

    def test_strict_schema_and_roundtrip(self):
        spec = target()
        self.assertEqual(DeliveryTarget.from_dict(spec.to_dict()), spec)
        for update in ({"version": 2}, {"version": True}, {"final_questions": True},
                       {"count_stage": "grounding"}, {"corpus_tokens": 0},
                       {"tokenizer": ""}, {"max_supply_rounds": 0},
                       {"requested_lines": ["L1", "L8_transition"]},
                       {"requested_lines": [LINE_IDS[0], LINE_IDS[0]]},
                       {"per_line_max": {LINE_IDS[0]: -1}}, {"unknown": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                target(**update)
        data = spec.to_dict()
        del data["tokenizer"]
        with self.assertRaises(ValueError):
            DeliveryTarget.from_dict(data)

    def test_equal_ten_lines_and_eight_line_remainders(self):
        self.assertEqual(set(allocate_line_targets(target(), list(reversed(LINE_IDS))).values()), {50})
        result = allocate_line_targets(target(lines=LINE_IDS[:8]), list(reversed(LINE_IDS[:8])))
        self.assertEqual(list(result.values()), [63, 63, 63, 63, 62, 62, 62, 62])
        self.assertEqual(sum(result.values()), 500)

    def test_bounds_and_large_target_are_balanced(self):
        spec = target(500, LINE_IDS[:3], per_line_min={LINE_IDS[0]: 300}, per_line_max={LINE_IDS[2]: 50})
        self.assertEqual(list(allocate_line_targets(spec, list(LINE_IDS[:3])).values()), [300, 150, 50])
        self.assertEqual(sum(allocate_line_targets(target(10**9), list(LINE_IDS)).values()), 10**9)
        with self.assertRaises(ValueError):
            allocate_line_targets(target(500, LINE_IDS[:2], per_line_max={LINE_IDS[0]: 100, LINE_IDS[1]: 100}), list(LINE_IDS[:2]))
        with self.assertRaises(ValueError):
            allocate_line_targets(target(), list(LINE_IDS[:8]))

    def test_explicit_missing_and_inapplicable_lines_stay_visible(self):
        rows = support(LINE_IDS[:2])
        rows[0]["applicable"] = False
        rows[1]["implemented"] = False
        result = resolve_lines(target(lines=LINE_IDS[:3]), rows)
        self.assertEqual([r["status"] for r in result["lines"]], ["seed_not_applicable", "implementation_missing", "support_unreported"])
        self.assertEqual(len(result["lines"]), 3)
        automatic = resolve_lines(target(lines=(), per_line_min={LINE_IDS[2]: 20}), rows)
        self.assertEqual([r["line"] for r in automatic["lines"]], [LINE_IDS[2]])
        self.assertEqual(automatic["status"], "blocked")

    def test_evidence_based_reserve_and_missing_rates(self):
        rate = {LINE_IDS[0]: {"rate": 0.4, "source": "run-1/report.json", "sample_size": 100}}
        original = deepcopy(rate)
        plan = plan_supply(target(10, LINE_IDS[:1]), support(LINE_IDS[:1]), rate)
        self.assertEqual(plan["candidate_reserve"], 32)
        self.assertEqual(rate, original)
        self.assertEqual(plan["lines"][0]["retention_evidence"], rate[LINE_IDS[0]])
        plan = plan_supply(target(10, LINE_IDS[:2]), support(LINE_IDS[:2]), rate)
        self.assertIsNone(plan["candidate_reserve"])
        self.assertEqual(plan["status"], "needs_rate_evidence")
        rate[LINE_IDS[0]]["rate"] = 0
        self.assertEqual(plan_supply(target(10, LINE_IDS[:1]), support(LINE_IDS[:1]), rate)["lines"][0]["estimate_issue"], "zero_observed_retention")
        for invalid in (float("nan"), True, -1, 1.1):
            rate[LINE_IDS[0]]["rate"] = invalid
            with self.assertRaises(ValueError):
                plan_supply(target(10, LINE_IDS[:1]), support(LINE_IDS[:1]), rate)

    def test_partition_adapter_rejects_overlap_and_missing(self):
        qs = [question("q1"), question("q2")]
        self.assertEqual([q["quality_status"] for q in partition(qs)], ["released", "released"])
        with self.assertRaises(ValueError):
            partition(qs, rejected=["q1"])
        with self.assertRaises(ValueError):
            partition(qs, released=["q1"])

    def test_real_filter_report_complete_incomplete_and_witness(self):
        formed = [question("keep"), question("easy"), question("missing"), question("reject"), question("pending"),
                  question("scope"), question("witness", LINE_IDS[-1], "L10_witness")]
        raw = formed + [question("unformed"), question("order-dropped")]
        effective = raw[:-1]
        quality = partition(formed, released=["keep", "easy", "missing", "witness"], rejected=["reject"], pending_review=["pending"], scoped_excluded=["scope"])
        released = [q for q in formed if q["qid"] in {"keep", "easy", "missing", "witness"}]
        report = filter_report(released, all_correct={"easy"}, missing={("D", "missing")})
        inputs = deepcopy((raw, effective, formed, quality, report))
        result = summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), raw_orders=raw,
                                    effective_orders=effective, candidates=formed, quality_rows=quality, selection_report=report)
        counts = result["totals"]["primary"]
        self.assertEqual((counts["raw_orders"], counts["effective_orders"], counts["formed"]), (8, 7, 6))
        self.assertEqual((counts["unformed"], counts["order_dropped"]), (1, 1))
        self.assertEqual((counts["released"], counts["rejected"], counts["pending_review"], counts["scoped_excluded"]), (3, 1, 1, 1))
        self.assertEqual((counts["final_selected"], counts["removed_easy"], counts["athlete_incomplete"]), (1, 1, 1))
        self.assertEqual(result["totals"]["auxiliary_witness"]["final_selected"], 1)
        self.assertFalse(result["question_target_met"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual((raw, effective, formed, quality, report), inputs)

    def test_met_and_line_shortfall_use_complete_primary_counts(self):
        qs = [question("q1"), question("q2")]
        kwargs = {"raw_orders": qs, "effective_orders": qs, "candidates": qs,
                  "quality_rows": partition(qs), "selection_report": filter_report(qs)}
        met = summarize_delivery(target(2, LINE_IDS[:1]), support(LINE_IDS[:1]), **kwargs)
        self.assertTrue(met["question_target_met"])
        uneven = summarize_delivery(target(2, LINE_IDS[:2]), support(LINE_IDS[:2]), **kwargs)
        self.assertEqual(uneven["total_shortfall"], 0)
        self.assertFalse(uneven["question_target_met"])
        self.assertEqual(uneven["by_line"][1]["shortfall"], 1)

    def test_explicit_line_max_is_reported_before_balanced_subset_export(self):
        qs = [question("q1"), question("q2")]
        result = summarize_delivery(target(1, LINE_IDS[:1], per_line_max={LINE_IDS[0]: 1}), support(LINE_IDS[:1]),
                                    raw_orders=qs, effective_orders=qs, candidates=qs,
                                    quality_rows=partition(qs), selection_report=filter_report(qs))
        self.assertEqual(result["totals"]["primary"]["final_selected"], 2)
        self.assertEqual(result["by_line"][0]["above_explicit_max"], 1)
        self.assertFalse(result["question_target_met"])
        self.assertEqual(result["status"], "exceeds_line_max")

    def test_missing_item_is_incomplete_not_final(self):
        qs = [question("q1"), question("q2")]
        result = summarize_delivery(target(2, LINE_IDS[:1]), support(LINE_IDS[:1]), raw_orders=qs,
                                    effective_orders=qs, candidates=qs, quality_rows=partition(qs), selection_report=filter_report(qs[:1]))
        self.assertEqual(result["totals"]["primary"]["final_selected"], 1)
        self.assertEqual(result["totals"]["primary"]["athlete_incomplete"], 1)

    def test_empty_completed_stage_reports_zero(self):
        report = filter_report([])
        result = summarize_delivery(target(2, LINE_IDS[:1]), support(LINE_IDS[:1]), raw_orders=[],
                                    effective_orders=[], candidates=[], quality_rows=[], selection_report=report)
        self.assertEqual(result["totals"]["primary"]["final_selected"], 0)
        self.assertEqual(result["total_shortfall"], 2)
        self.assertEqual(result["status"], "shortfall")

    def test_native_stage_empty_branch_is_counted_without_inventing_athletes(self):
        empty = {"version": 1, "status": "empty", "rule": "all_four_athletes_correct", "release_ready": False,
                 "counts": {"input": 0, "kept": 0, "all_correct": 0, "removed_easy": 0, "kept_incomplete": 0}, "items": []}
        result = summarize_delivery(target(2, LINE_IDS[:1]), support(LINE_IDS[:1]), raw_orders=[],
                                    effective_orders=[], candidates=[], quality_rows=[], selection_report=empty)
        self.assertEqual(result["total_shortfall"], 2)
        self.assertEqual(result["status"], "shortfall")
        qs = [question("q1")]
        with self.assertRaises(ValueError):
            summarize_delivery(target(2, LINE_IDS[:1]), support(LINE_IDS[:1]), raw_orders=qs,
                               effective_orders=qs, candidates=qs, quality_rows=partition(qs), selection_report=empty)

    def test_unknown_support_and_sourced_rates_do_not_invent_supply(self):
        rows = support(LINE_IDS[:1])
        rates = {line: {"rate": 0.5, "source": "measurement.json", "sample_size": 10} for line in LINE_IDS[:2]}
        result = plan_supply(target(10, LINE_IDS[:2]), rows, rates)
        self.assertEqual(result["allocation"], {LINE_IDS[0]: 5, LINE_IDS[1]: 5})
        self.assertIsNone(result["lines"][1]["candidate_reserve"])
        self.assertEqual(result["status"], "blocked")
        bad = deepcopy(rates)
        bad[LINE_IDS[0]]["source"] = ""
        with self.assertRaises(ValueError):
            plan_supply(target(10, LINE_IDS[:2]), rows, bad)

    def test_filter_configuration_and_athlete_values_are_checked(self):
        qs = [question("q1")]
        report = filter_report(qs)
        base = {"raw_orders": qs, "effective_orders": qs, "candidates": qs, "quality_rows": partition(qs)}
        for key, value in (("systems", ["A", "B", "C"]), ("keep_easy_ratio", 0.2),
                           ("keep_easy_ratio", False), ("rule", "different_rule"),
                           ("result_scope", "research_only")):
            changed = deepcopy(report)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), selection_report=changed, **base)
        changed = deepcopy(report)
        changed["items"][0]["correct"]["A"] = 0
        with self.assertRaises(ValueError):
            summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), selection_report=changed, **base)

    def test_not_run_preserves_null_and_stages_require_sources(self):
        result = summarize_delivery(target(), support())
        self.assertIsNone(result["totals"]["primary"]["formed"])
        self.assertIsNone(result["total_shortfall"])
        self.assertEqual(result["status"], "not_run")
        with self.assertRaises(ValueError):
            summarize_delivery(target(), support(), candidates=[question("q1")])

    def test_stale_reference_duplicate_and_inconsistent_disposition_rejected(self):
        qs = [question("q1")]
        report = filter_report(qs)
        base = {"raw_orders": qs, "effective_orders": qs, "candidates": qs, "quality_rows": partition(qs)}
        bad = deepcopy(report)
        bad["items"][0]["disposition"] = "removed_easy"
        with self.assertRaises(ValueError):
            summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), selection_report=bad, **base)
        bad = deepcopy(report)
        bad["items"].append(deepcopy(bad["items"][0]))
        with self.assertRaises(ValueError):
            summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), selection_report=bad, **base)
        changed = deepcopy(qs)
        changed[0]["gt"] = {"value": "new answer"}
        base["candidates"] = changed
        with self.assertRaises(ValueError):
            summarize_delivery(target(1, LINE_IDS[:1]), support(LINE_IDS[:1]), selection_report=report, **base)

    def test_legacy_targetspec_stays_grounding_oriented(self):
        from pipeline.targetspec import TargetSpec
        self.assertEqual(TargetSpec.from_dict({"min_questions": 7}).min_questions, 7)
        self.assertFalse(hasattr(TargetSpec(), "count_stage"))


if __name__ == "__main__":
    unittest.main()
