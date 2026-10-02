"""Real wrapper admission waits for settlement without weakening any hard cap."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke


class GatedSDK(fixture.FakeSDK):
    def __init__(self, usage, failure=False):
        super().__init__()
        self.usage = usage
        self.failure = failure
        self.first_entered = threading.Event()
        self.release_first = threading.Event()

    def with_options(self, **options):
        def create(**kwargs):
            with self.lock:
                self.calls.append({'options': deepcopy(options), 'kwargs': deepcopy(kwargs)})
                first = len(self.calls) == 1
            if first:
                self.first_entered.set()
                if not self.release_first.wait(5):
                    raise AssertionError('Test failed to release provider')
            if self.failure:
                raise RuntimeError('Offline permanent provider failure')
            return fixture.SimpleNamespace(id='offline', model=kwargs['model'], usage=self.usage,
                choices=[fixture.SimpleNamespace(index=0, finish_reason='stop',
                    message=fixture.SimpleNamespace(content='{"ok":true}', refusal=None))])
        return fixture.SimpleNamespace(chat=fixture.SimpleNamespace(completions=fixture.SimpleNamespace(create=create)))


class AdmissionTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def exercise(self, *, usage, failure=False, settle=True, max_calls=12, expect_wait=True):
        sdk = GatedSDK(usage, failure)
        waiting = threading.Event()
        real_condition = threading.Condition
        answers = []

        class ObservedCondition(real_condition):
            def wait(self, timeout=None):
                if smoke._TRACER_STEP.get() == 'waiter':
                    waiting.set()
                return super().wait(timeout)

        def calls(directory, cfg):
            tracer = fixture.Tracer(fixture.SimpleNamespace(dir=directory))
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(tracer.chat_json, 'first', [], max_tokens=4096, retries=1)
                self.assertTrue(sdk.first_entered.wait(5))
                second = pool.submit(tracer.chat_json, 'waiter', [], max_tokens=4096, retries=1)
                try:
                    if expect_wait:
                        self.assertTrue(waiting.wait(5), 'Reservation pressure must wait for settlement')
                        self.assertEqual(len(sdk.calls), 1)
                        self.assertFalse(second.done())
                    else:
                        answers.append(second.result(timeout=5))
                        self.assertFalse(waiting.is_set())
                finally:
                    sdk.release_first.set()
                answers.append(first.result(timeout=5))
                if expect_wait:
                    answers.append(second.result(timeout=5))

        extra = ['--max-cny', '0.14', '--max-calls', str(max_calls)]
        if settle:
            extra.append('--settle-reported-usage')
        with patch.object(fixture, 'FakeSDK', return_value=sdk), \
                patch.object(smoke.threading, 'Condition', ObservedCondition):
            result = self.run_tool(calls, extra)
        self.assertIsNone(result.error)
        self.assertLessEqual(result.profile['admitted_calls'], max_calls)
        self.assertLessEqual(result.profile['budget_consumed_cny'], 0.14)
        return result, answers

    def test_parallel_reservation_pressure_releases_after_real_usage(self):
        usage = fixture.SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        result, answers = self.exercise(usage=usage)
        self.assertEqual(answers, [{'ok': True}, {'ok': True}])
        self.assertEqual(result.profile['admitted_calls'], 2)
        self.assertEqual(result.profile['settled_usage_calls'], 2)
        self.assertEqual(result.profile['unsettled_reservation_calls'], 0)
        self.assertFalse(result.profile['stopped'])

    def test_unknown_usage_retains_reservation_and_stops_after_drain(self):
        result, answers = self.exercise(usage=None)
        self.assertEqual(result.profile['admitted_calls'], 1)
        self.assertEqual(result.profile['unsettled_reservation_calls'], 1)
        self.assertEqual(result.profile['stop_reason']['reason'], 'estimated_budget_limit')
        self.assertEqual(sum('__error__' in answer for answer in answers), 1)

    def test_real_settled_cost_still_enforces_budget(self):
        usage = fixture.SimpleNamespace(prompt_tokens=20000, completion_tokens=2000, total_tokens=22000)
        result, answers = self.exercise(usage=usage)
        self.assertEqual(result.profile['admitted_calls'], 1)
        self.assertAlmostEqual(result.profile['reported_usage_estimate_cny'], 0.12)
        self.assertEqual(result.profile['stop_reason']['reason'], 'estimated_budget_limit')
        self.assertEqual(sum('__error__' in answer for answer in answers), 1)

    def test_provider_failure_wakes_waiter_and_preserves_first_cause(self):
        result, answers = self.exercise(usage=None, failure=True)
        self.assertEqual(result.profile['admitted_calls'], 1)
        self.assertEqual(result.profile['stop_reason']['reason'], 'provider_failure')
        self.assertTrue(all('__error__' in answer for answer in answers))

    def test_without_usage_settlement_reservations_cannot_release(self):
        result, answers = self.exercise(usage=None, settle=False, expect_wait=False)
        self.assertEqual(result.profile['admitted_calls'], 1)
        self.assertEqual(result.profile['stop_reason']['reason'], 'estimated_budget_limit')

    def test_call_limit_remains_immediate_even_with_active_provider(self):
        usage = fixture.SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        result, answers = self.exercise(usage=usage, max_calls=1, expect_wait=False)
        self.assertEqual(result.profile['admitted_calls'], 1)
        self.assertEqual(result.profile['stop_reason']['reason'], 'call_limit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
