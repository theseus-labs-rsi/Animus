"""Exercise actual wrapper/config/transport against fake SDK; network forbidden."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke
fixture.smoke = smoke
Tracer = fixture.Tracer
SimpleNamespace = fixture.SimpleNamespace


class TokenOverrideTests(fixture.CorpusReasoningOverrideTests):
    # The ten inherited tests also exercise the candidate's unchanged defaults,
    # reasoning override, strict JSON/SDK0, model guard and stop-on-failure.
    def test_tokens_alone_affect_only_exact_step_and_real_wire(self):
        steps = ['corpus.review', 'world.review', 'render.signal', 'corpus.review.extra', 'CORPUS.REVIEW']
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            for step in steps:
                tracer.chat_json(step, [{'role': 'user', 'content': 'corpus.review'}], max_tokens=4096)
            cfg.chat_json([{'role': 'user', 'content': 'corpus.review'}], max_tokens=4096)
        result = self.run_tool(calls, ['--corpus-review-max-tokens', '8192'])
        self.assertIsNone(result.error)
        wire = [call['kwargs'] for call in result.api.calls]
        self.assertEqual([call['max_completion_tokens'] for call in wire], [8192, 4096, 4096, 4096, 4096, 4096])
        self.assertTrue(all(call['reasoning_effort'] == 'low' for call in wire))
        self.assertTrue(all(call['model'] == 'gpt-5.4-mini' for call in wire))
        self.assertTrue(all(call['options'] == {'timeout': 150, 'max_retries': 0} for call in result.api.calls))
        self.assertTrue(all('temperature' not in call and 'top_p' not in call for call in wire))
        self.assertEqual(result.profile['admitted_calls'], 6)
        self.assertEqual(result.profile['step_transport_overrides']['corpus.review']['max_tokens'], 8192)
        self.assertEqual(result.profile['step_transport_overrides']['corpus.review']['reasoning_effort'], 'unchanged global setting')
        self.assertEqual(result.requests[0]['parameters']['max_completion_tokens'], 8192)
        self.assertTrue(all(row['max_attempts'] == 3 for row in result.rows if row.get('event') == 'json_attempt'))

    def test_reasoning_and_tokens_can_both_override_without_changing_other_steps(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            tracer.chat_json('corpus.review', [], max_tokens=4096)
            tracer.chat_json('world.review', [], max_tokens=4096)
        result = self.run_tool(calls, ['--corpus-review-max-tokens', '8192', '--corpus-review-reasoning-effort', 'medium'])
        self.assertIsNone(result.error)
        self.assertEqual([(c['kwargs']['reasoning_effort'], c['kwargs']['max_completion_tokens']) for c in result.api.calls],
                         [('medium', 8192), ('low', 4096)])

    def test_4096_override_and_default_caller_limit_are_distinct(self):
        def calls(directory, cfg):
            tracer = Tracer(SimpleNamespace(dir=directory))
            tracer.chat_json('corpus.review', [], max_tokens=8192)
            tracer.chat_json('render.signal', [], max_tokens=8192)
        controlled = self.run_tool(calls, ['--corpus-review-max-tokens', '4096'])
        default = self.run_tool(calls)
        self.assertIsNone(controlled.error)
        self.assertIsNone(default.error)
        self.assertEqual([c['kwargs']['max_completion_tokens'] for c in controlled.api.calls], [4096, 8192])
        self.assertEqual([c['kwargs']['max_completion_tokens'] for c in default.api.calls], [8192, 8192])
        self.assertEqual(default.profile['step_transport_overrides'], {})

    def test_budget_reserves_effective_8192_cap_before_dispatch(self):
        messages = [{'role': 'user', 'content': '预算边界'}]
        def calls(directory, cfg):
            Tracer(SimpleNamespace(dir=directory)).chat_json('corpus.review', messages, max_tokens=4096)
        low = self.run_tool(calls)
        high = self.run_tool(calls, ['--corpus-review-max-tokens', '8192'])
        self.assertEqual(len(low.api.calls), 1)
        self.assertEqual(len(high.api.calls), 1)
        self.assertAlmostEqual(high.profile['reserved_upper_estimate_cny'] - low.profile['reserved_upper_estimate_cny'],
                               4096 * high.profile['output_cny_per_m'] / 1e6, places=12)
        middle = (low.profile['reserved_upper_estimate_cny'] + high.profile['reserved_upper_estimate_cny']) / 2
        stopped = self.run_tool(calls, ['--corpus-review-max-tokens', '8192', '--max-cny', str(middle)])
        self.assertEqual(stopped.api.calls, [])
        self.assertEqual(stopped.profile['admitted_calls'], 0)
        self.assertEqual(stopped.profile['reserved_upper_estimate_cny'], 0)
        self.assertTrue(stopped.profile['stopped'])

    def test_token_override_restores_context_after_exception_without_reasoning_flag(self):
        def raising(self, step, messages, **kw):
            if smoke._TRACER_STEP.get() != step:
                raise AssertionError('Step scope was not entered')
            raise RuntimeError('Offline boundary exception')
        result = self.run_tool(lambda directory, cfg: Tracer(SimpleNamespace(dir=directory)).chat_json('corpus.review', []),
            ['--corpus-review-max-tokens', '8192'], tracer_method=raising)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(result.api.calls, [])
        self.assertEqual(result.profile['execution_status'], 'failed')
        self.assertIsNone(smoke._TRACER_STEP.get())

    def test_glm_and_unlisted_limits_reject_before_factory_or_directory(self):
        for flags in [['--model', 'glm-4.7-nothinking', '--corpus-review-max-tokens', '8192'],
                      ['--corpus-review-max-tokens', '2048']]:
            with self.subTest(flags=flags):
                result = self.run_tool(lambda *args: self.fail('Factory must not run'), flags)
                self.assertIsInstance(result.error, SystemExit)
                self.assertEqual(result.error.code, 2)
                self.assertFalse(result.directory.exists())
                self.assertEqual(result.api.calls, [])

    def test_parallel_step_context_does_not_share_token_override(self):
        def calls(directory, cfg):
            barrier = threading.Barrier(2)
            tracer = Tracer(SimpleNamespace(dir=directory))
            def invoke(step):
                barrier.wait(timeout=3)
                return tracer.chat_json(step, [{'role': 'user', 'content': step}], max_tokens=4096)
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(list(pool.map(invoke, ['corpus.review', 'render.signal'])), [{'ok': True}] * 2)
        result = self.run_tool(calls, ['--corpus-review-max-tokens', '8192'])
        self.assertIsNone(result.error)
        self.assertEqual({c['kwargs']['messages'][0]['content']: c['kwargs']['max_completion_tokens'] for c in result.api.calls},
                         {'corpus.review': 8192, 'render.signal': 4096})

    def test_existing_process_flag_remains_forwarded(self):
        observed = []
        def calls(directory, cfg):
            observed.extend(sys.argv)
        result = self.run_tool(calls, ['--process-questions', '--corpus-review-max-tokens', '8192'])
        self.assertIsNone(result.error)
        self.assertEqual(observed.count('--process-questions'), 1)


if __name__ == '__main__':
    unittest.main()
