"""Exercise the real closed-loop driver and grounding stage without network."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from pipeline import factory, run as run_module
from pipeline.closed_loop import build_to_target
from pipeline.targetspec import TargetSpec
from pipeline.world_blueprint import WorldBlueprintError
from pipeline.world_state import Op, SET, Timeline, WorldState


class ClosedLoopSemanticStopTests(unittest.TestCase):
    def exercise(self, *, semantic=True, outcome="pending", enough=False, execution_error=False,
                 target_spec=None, extra_line=False, world_failure_on_call=None):
        with tempfile.TemporaryDirectory(prefix="bc_closed_loop_") as directory, ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network prohibited")))
            stack.enter_context(patch.object(config, "chat", side_effect=AssertionError("API prohibited")))
            stack.enter_context(patch.object(config, "chat_json", side_effect=AssertionError("API prohibited")))
            stack.enter_context(patch.object(run_module, "RUNS_DIR", Path(directory)))
            run = run_module.Run("office", "test")
            run.log = lambda *_: None
            wp = {"active_lines": [{"line": "L1_timeline", "weight": 1}],
                  "shared_world_spec": {"entities": {"count": 1}, "timeline": {"n_sessions": 3}},
                  "domain_profile": {"field_schema": []}}
            if semantic:
                wp["quality_contract"] = {"public_semantic_review": True}
            if extra_line:
                wp["active_lines"].append({"line": "L2_relational", "weight": 1})
            run.write(factory.ART["whitepaper"], wp)
            run.write("00_about.json", {"answer_protocol": {"rules": ["Answer the question from the public record."]}})
            calls = []
            reports = []
            raw_reviews = []
            qs = [{"qid": f"q{i}", "line": "L1_timeline", "capability": "IE",
                   "question": f"Question {i}", "gt": "value"} for i in range(3)]

            def world(r):
                calls.append("world")
                if world_failure_on_call == calls.count("world"):
                    raise RuntimeError("injected later-round provider stop")
                ws = WorldState(entities={"entity": {"field": Timeline([Op(0, "2025-01-06", SET, "value")])}}, n_sessions=3)
                r.write(factory.ART["world"], ws.to_dict())

            def orders(r):
                calls.append("orders")
                r.write(factory.ART["orders"], qs)

            def well_posed(r):
                r.write("03_well_posed_report.json", {"passed": 3})

            def questions(r):
                candidates = qs[:1] if outcome == "none" and calls.count("world") == 1 else qs
                r.write(factory.ART["questions"], candidates)

            def corpus(r):
                calls.append("corpus")
                r.write(factory.ART["corpus"], {"sessions": []})

            def quality(r):
                calls.append("quality")
                r.write(factory.ART["quality"], {"test_fixture": True})

            def grounding_result(questions, corpus, *args, **kwargs):
                calls.append("grounding")
                iteration = calls.count("grounding")
                keep = 2 if enough else (1 if iteration == 1 else 3)
                total = len(questions)
                pending = (total-keep) if semantic and outcome == "pending" else 0
                rejected = (total-keep) if outcome == "rejected" or not semantic else 0
                # Free-text deliberately suggests the opposite action. Only
                # the explicit machine state may control the driver's stop.
                detail = "More entities might help; this opinion remains fallible."
                report = {"overall": {"n": total, "grounded": keep, "survival": keep/total},
                          "by_line": {"L1_timeline": {"n": total, "grounded": keep, "survival": keep/total}},
                          "by_capability": {"IE": {"n": total, "grounded": keep, "survival": keep/total}},
                          "n_pending": pending, "n_dropped": rejected,
                          "pending": [{"qid": q["qid"], "reason": detail} for q in questions[keep:keep+pending]],
                          "drops": [{"qid": q["qid"], "reason": detail} for q in questions[keep:keep+rejected]]}
                reports.append(deepcopy(report))
                if not semantic:
                    return deepcopy(questions[:keep]), report
                report.update(execution_complete=not execution_error, delivery_safe=not execution_error,
                              decision_basis="fallible_public_semantic_review")
                reports[-1] = deepcopy(report)
                raw = {"raw_opinions": [{"qid": q["qid"], "reason": detail} for q in questions],
                       "execution_complete": not execution_error}
                raw_reviews.append(deepcopy(raw))
                return deepcopy(questions[:keep]), report, raw

            for name, fn in (("stage_world", world), ("stage_orders", orders),
                             ("stage_well_posed", well_posed), ("stage_questions", questions),
                             ("stage_corpus", corpus), ("stage_quality", quality)):
                stack.enter_context(patch.object(factory, name, fn))
            if semantic:
                stack.enter_context(patch("pipeline.grounding_review.review_grounding", side_effect=grounding_result))
            else:
                stack.enter_context(patch("pipeline.grounding.run_grounding", side_effect=grounding_result))
            exception = None
            result = None
            try:
                result = build_to_target(run, target_spec or TargetSpec(min_questions=2, per_line_min={"L1_timeline": 2}),
                                         max_rounds=3, order_subrounds=1)
            except Exception as exc:
                exception = exc
            artifacts = {p.name: run.read(p.name) for p in run.dir.glob("*.json")}
            return calls, deepcopy(run.manifest), artifacts, reports, raw_reviews, exception, result

    def assert_quality_warning_completion(self, outcome):
        calls, manifest, artifacts, reports, reviews, exc, result = self.exercise(outcome=outcome)
        self.assertIsNone(exc)
        self.assertEqual(result[1], "MET")
        self.assertEqual(calls.count("world"), 2)
        self.assertEqual(calls.count("corpus"), 2)
        self.assertEqual(calls.count("quality"), 1)
        self.assertEqual(manifest["status"], "done")
        self.assertEqual(manifest["algo"]["met_status"], "MET")
        self.assertEqual(artifacts["06_grounding_report.json"], reports[-1])
        self.assertEqual(artifacts["06_semantic_review.json"], reviews[-1])
        self.assertEqual(len(artifacts[factory.ART["questions"]]), 3)
        self.assertEqual(len(artifacts[factory.ART["grounding"]]), 3)
        self.assertFalse({"augment", "render_only", "render_only_pairs"} & manifest["config"].keys())
        detail = manifest["algo"]["quality_review_shortfall"]
        self.assertEqual(detail["pending_qids"] if outcome == "pending" else detail["rejected_qids"], ["q1", "q2"])

    def test_unresolved_candidates_continue_through_quality(self):
        self.assert_quality_warning_completion("pending")

    def test_rejected_candidates_continue_without_changing_review(self):
        self.assert_quality_warning_completion("rejected")

    def test_enough_eligible_candidates_finish_despite_other_pending(self):
        calls, manifest, artifacts, reports, reviews, exc, result = self.exercise(enough=True)
        self.assertIsNone(exc)
        self.assertEqual(result[1], "MET")
        self.assertEqual(calls.count("world"), 1)
        self.assertEqual(calls.count("quality"), 1)
        self.assertEqual(manifest["status"], "done")
        self.assertEqual(artifacts["06_grounding_report.json"]["n_pending"], 1)

    def test_total_only_meets_total_with_a_naturally_absent_line(self):
        calls, manifest, artifacts, _, _, exc, result = self.exercise(enough=True, extra_line=True,
            target_spec=TargetSpec(min_questions=2, total_only=True, max_world_entities=16))
        self.assertIsNone(exc)
        self.assertEqual("MET", result[1])
        self.assertIn("quality", calls)
        self.assertEqual(2, manifest["algo"]["per_line_final"]["L1_timeline"])
        self.assertIn("L2_relational", manifest["config"]["quotas"])

    def test_total_only_preserves_explicit_line_floors(self):
        calls, manifest, _, _, _, exc, result = self.exercise(enough=True, extra_line=True,
            target_spec=TargetSpec(min_questions=2, total_only=True, per_line_min={"L2_relational": 1}))
        self.assertIsNone(exc)
        self.assertEqual(result[1], "COMPLETED_UNMET")
        self.assertIn("corpus", calls)
        self.assertIn("quality", calls)
        self.assertTrue(manifest["algo"]["met_status"].startswith("UNMET:"))

    def test_legacy_drops_still_use_original_second_round_growth(self):
        calls, manifest, artifacts, reports, reviews, exc, result = self.exercise(semantic=False)
        self.assertIsNone(exc)
        self.assertEqual(result[1], "MET")
        self.assertEqual(calls.count("world"), 2)
        self.assertNotIn("quality_review_shortfall", manifest["algo"])

    def test_semantic_supply_gap_without_review_pending_or_rejection_retains_growth(self):
        calls, manifest, artifacts, reports, reviews, exc, result = self.exercise(outcome="none")
        self.assertIsNone(exc)
        self.assertEqual(result[1], "MET")
        self.assertEqual(calls.count("world"), 2)

    def test_review_execution_failure_keeps_completed_subset(self):
        calls, manifest, artifacts, reports, reviews, exc, result = self.exercise(execution_error=True)
        self.assertIsNone(exc)
        self.assertEqual(result[1], "MET")
        self.assertEqual(calls.count("world"), 2)
        self.assertEqual(manifest["stages"]["grounding"]["status"], "succeeded")
        self.assertIn(factory.ART["grounding"], artifacts)
        self.assertEqual(artifacts["06_semantic_review.json"], reviews[-1])
        self.assertIn("quality", calls)

    def test_later_round_failure_stops_new_work_but_still_runs_final_quality(self):
        calls, manifest, artifacts, _, _, exc, result = self.exercise(world_failure_on_call=2)
        self.assertIsNone(exc)
        self.assertEqual(result[1], "COMPLETED_UNMET")
        self.assertEqual(calls.count("world"), 2)
        self.assertEqual(calls.count("quality"), 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["algo"]["met_status"], "UNMET_EXECUTION")
        self.assertEqual(artifacts["06_production_warning.json"]["release_eligible"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
