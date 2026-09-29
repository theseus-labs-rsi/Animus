"""The author's time contract is the original compiler's value projection."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from pipeline import world_state as state


class IntrinsicFactTimeTests(unittest.TestCase):
    def compile_field(self, spec, *, base="2025-01-06", step_days=7):
        table = {"entities": [{"name": "observer", "fields": {"knowledge": deepcopy(spec)}}]}
        world, issues = state.assemble_world(table, base=base, step_days=step_days,
                                            include_shape_diagnostics=False)
        self.assertFalse(issues)
        return world.timeline("observer", "knowledge")

    def test_stable_means_true_from_window_start_even_if_text_mentions_the_future(self):
        # This intentionally remains a semantic error for the original reviewers
        # to assess. No date parser or extra truth validator is introduced.
        value = "Witnessed an event on 2025-01-20; learned its result afterward."
        spec = {"type": "stable", "value": value}
        before = deepcopy(spec)
        timeline = self.compile_field(spec)
        contract = state.intrinsic_fact_time_contract(state._date_of)
        self.assertEqual({k: asdict(timeline.ops[0])[k] for k in ("op", "session", "date")},
                         {k: contract["stable"][k] for k in ("op", "session", "date")})
        self.assertEqual(timeline.value_at_session(0), value)
        self.assertEqual(spec, before)

    def test_later_truth_is_not_backfilled_and_versions_follow_original_fold(self):
        spec = {"type": "evolving", "trajectory": [
            {"session": 2, "value": "heard a report"},
            {"session": 3, "value": "heard a report"},
            {"session": 4, "value": "personally verified"},
            {"session": 5, "value": None},
            {"session": 6, "value": "new account"}]}
        timeline = self.compile_field(spec)
        self.assertEqual([(op.session, op.op, op.prev) for op in timeline.ops],
                         [(2, state.SET, None), (4, state.UPDATE, "heard a report"),
                          (5, state.EXPIRE, "personally verified"), (6, state.SET, None)])
        self.assertEqual([timeline.value_at_session(s) for s in range(7)],
                         [state.INSUFFICIENT, state.INSUFFICIENT, "heard a report", "heard a report",
                          "personally verified", state.INVALID, "new account"])
        self.assertEqual([asdict(op) for op in state.intrinsic_value_ops(spec, state._date_of)],
                         [asdict(op) for op in timeline.ops])

    def test_contract_uses_the_actual_calendar_and_preserves_legacy_value_coercion(self):
        date_of = lambda session: state._date_of(session, "2031-03-10", 2)
        contract = state.intrinsic_fact_time_contract(date_of)
        self.assertEqual(contract["stable"]["date"], "2031-03-10")
        for value in (None, "", 0, 5.25, "a future date 2031-03-18"):
            with self.subTest(value=value):
                timeline = self.compile_field({"type": "stable", "value": value},
                                              base="2031-03-10", step_days=2)
                self.assertEqual(asdict(timeline.ops[0]),
                                 {"session": 0, "date": "2031-03-10", "op": state.SET,
                                  "value": str(value), "prev": None})


if __name__ == "__main__":
    unittest.main(verbosity=2)
