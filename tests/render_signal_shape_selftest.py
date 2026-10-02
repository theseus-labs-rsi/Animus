"""Original four signal draft slots, including malformed model document shapes."""
import os
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['OPENAI_API_KEY'] = 'offline-dummy'
os.environ['OPENAI_BASE_URL'] = 'http://offline.invalid'
os.environ['MODEL'] = 'offline-dummy'
import socket
import sys
import types

def forbidden(*args, **kwargs):
    raise AssertionError('Network/provider forbidden')

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda *a, **k: False
dotenv.find_dotenv = lambda *a, **k: ''
sys.modules['dotenv'] = dotenv
import json
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get('SIGNAL_SHAPE_REPO', str(Path(__file__).resolve().parents[1])))
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import openai
openai.OpenAI = openai.AsyncOpenAI = forbidden
import config
config.chat = config.chat_json = forbidden
from execution_control import ExecutionStopped
from pipeline import render as renderer, evidence_transport as wire, corpus_contract as cc, run as run_module
from corpus_semantic_fidelity_selftest import Trace, world
from signal_evidence_transport_selftest import TransportTrace
from pipeline.task_execution import JournaledTracer, RecordedCallFailure
from pipeline.world_state import Op, UPDATE, _date_of

ORIGINAL_PMAP = config.pmap
GOOD = {'docs': [{'title': '记录', 'content': '甲项目目前已进入收尾。'}]}


class ShapeTrace(TransportTrace):
    def __init__(self, outputs, **kwargs):
        super().__init__(**kwargs)
        self.signal_outputs = list(outputs)

    def chat_json(self, step, messages, **parameters):
        if step == 'render.signal':
            self.author_inputs.append(deepcopy(wire.unpack(json.loads(messages[-1]['content']))))
            self.calls.append((step, deepcopy(messages), deepcopy(parameters)))
            if not self.signal_outputs:
                raise AssertionError('Unexpected extra author request')
            out = self.signal_outputs.pop(0)
            if isinstance(out, BaseException):
                raise out
            return deepcopy(out)
        return super().chat_json(step, messages, **parameters)


class SignalShapeTests(unittest.TestCase):
    def setUp(self):
        serial = patch.object(config, 'pmap', side_effect=lambda fn, xs, **kw: [fn(x) for x in xs])
        serial.start(); self.addCleanup(serial.stop)

    def render(self, trace, *, ws=None, corpus=None, done=None):
        wp = {'quality_contract': {'corpus_review': True}, 'generation_contract': {'material_first': True}}
        ws = ws or world()
        corpus = {'sessions': []} if corpus is None else corpus
        done = set() if done is None else done
        with patch.object(renderer, '_plan_signal', return_value=None):
            renderer.render_corpus(wp, ws, 0, trace, corpus, done, lambda: None,
                log=lambda *_: None, evidence_transport=wire.SHARED)
        return corpus

    def saved(self, td):
        path, = (Path(td) / '05_signal_checkpoints').glob('*.json')
        return json.loads(path.read_text(encoding='utf-8'))

    def test_invalid_document_shapes_reach_same_author_then_original_review(self):
        bad_outputs = [[], {}, {'docs': None}, {'docs': []}, {'docs': [None]},
                       {'docs': [{}]}, {'docs': [{'content': []}]}, {'docs': [{'content': '  '}]}]
        for bad in bad_outputs:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as td:
                ws = world(); before = deepcopy(ws.to_dict())
                trace = ShapeTrace([bad, GOOD], pfile=Path(td) / 'prompts.jsonl')
                result = self.render(trace, ws=ws)
                self.assertEqual([row[0] for row in trace.calls],
                    ['render.signal', 'render.signal', 'render.discriminate', 'corpus.review'])
                first, second = [row['variables'] for row in trace.author_inputs]
                for key in ('facts', 'events', 'story_context', 'context', 's', 'date', 'plan_text'):
                    self.assertEqual(first[key], second[key], key)
                self.assertIn('previous_response', second['hint'])
                saved = self.saved(td)['progress']['drafts']
                self.assertEqual(len(saved), 2)
                self.assertEqual(saved[0]['author_output'], bad)
                self.assertTrue(saved[0]['format_feedback']['issues'])
                self.assertNotIn('quality_review', saved[0])
                self.assertEqual(saved[1]['quality_review']['status'], 'passed')
                self.assertEqual(cc.validate_corpus(ws, result)['status'], 'passed')
                self.assertEqual(ws.to_dict(), before)

    def test_four_bad_shapes_stop_without_reader_and_resume_cannot_reset_slots(self):
        with tempfile.TemporaryDirectory() as td:
            trace = ShapeTrace([{}] * 4, pfile=Path(td) / 'prompts.jsonl')
            with self.assertRaises(renderer.MaterialRejected): self.render(trace)
            saved = self.saved(td)
            self.assertEqual(len(saved['progress']['drafts']), 4)
            self.assertEqual([row[0] for row in trace.calls], ['render.signal'] * 4)
            self.assertTrue(all(d['author_output'] == {} for d in saved['progress']['drafts']))
            resumed = ShapeTrace([], pfile=trace.pfile)
            with self.assertRaises(renderer.MaterialRejected): self.render(resumed)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(self.saved(td), saved)

    def test_mixed_bad_shape_and_real_negative_share_four_drafts(self):
        with tempfile.TemporaryDirectory() as td:
            trace = ShapeTrace([{}, GOOD, GOOD, GOOD], status='ambiguous', pfile=Path(td) / 'prompts.jsonl')
            with self.assertRaises(renderer.MaterialRejected): self.render(trace)
            self.assertEqual(len(trace.author_inputs), 4)
            self.assertEqual(len(trace.review_inputs), 3)
            drafts = self.saved(td)['progress']['drafts']
            self.assertEqual(len(drafts), 4)
            self.assertTrue(all(d['quality_review']['status'] == 'failed' for d in drafts[1:]))
            self.assertTrue(all('format_feedback' not in d for d in drafts[1:]))

    def test_valid_semantic_negative_is_not_format_retry_to_accept(self):
        trace = ShapeTrace([GOOD] * 4, status='ambiguous')
        with self.assertRaises(renderer.MaterialRejected): self.render(trace)
        self.assertEqual(len(trace.author_inputs), 4)
        self.assertEqual(len(trace.review_inputs), 4)
        self.assertNotIn('previous_response', trace.author_inputs[1]['variables']['hint'])
        self.assertIn('语义审阅', trace.author_inputs[1]['variables']['hint'])

    def test_provider_and_global_failures_do_not_retry_author(self):
        failures = [RuntimeError('provider offline failure'), {'__error__': 'provider offline failure'},
            ExecutionStopped('budget stopped', kind='estimated_budget_limit'),
            {'__error__': 'budget stopped', '__error_metadata__': {'kind': 'estimated_budget_limit', 'global_stop': True}}]
        for bad in failures:
            with self.subTest(bad=bad):
                trace = ShapeTrace([bad, GOOD])
                with self.assertRaises(RuntimeError): self.render(trace)
                self.assertEqual(len(trace.author_inputs), 1)
                self.assertEqual(trace.review_inputs, [])

    def test_checkpoint_write_failure_is_not_model_format_feedback(self):
        with tempfile.TemporaryDirectory() as td:
            trace = ShapeTrace([{}, GOOD], pfile=Path(td) / 'prompts.jsonl')
            real_write = run_module._atomic_write_json
            def write(path, value):
                drafts = value.get('progress', {}).get('drafts', [])
                if drafts and 'author_output' in drafts[-1]:
                    raise OSError('fixed checkpoint write failure')
                return real_write(path, value)
            with patch.object(run_module, '_atomic_write_json', side_effect=write):
                with self.assertRaisesRegex(OSError, 'fixed checkpoint'): self.render(trace)
            self.assertEqual(len(trace.author_inputs), 1)
            self.assertEqual(trace.review_inputs, [])

    def test_lost_third_reply_after_bad_shapes_keeps_original_four_total(self):
        with tempfile.TemporaryDirectory() as td:
            first = ShapeTrace([{}, {'docs': []}, RuntimeError('fixed interruption')], pfile=Path(td) / 'prompts.jsonl')
            with self.assertRaisesRegex(RuntimeError, 'fixed interruption'): self.render(first)
            old = self.saved(td)['progress']['drafts']
            self.assertEqual(len(old), 3)
            second = ShapeTrace([GOOD], pfile=first.pfile)
            self.render(second)
            self.assertEqual(len(first.author_inputs) + len(second.author_inputs), 4)
            now = self.saved(td)['progress']['drafts']
            self.assertEqual(len(now), 4)
            self.assertEqual(now[:3], old)

    def test_bad_period_does_not_erase_completed_parallel_sibling(self):
        with tempfile.TemporaryDirectory() as td:
            ready = threading.Event()
            class ParallelTrace(TransportTrace):
                def chat_json(inner, step, messages, **parameters):
                    if step == 'render.signal':
                        payload = wire.unpack(json.loads(messages[-1]['content']))
                        if payload['variables']['s'] == 1:
                            inner.author_inputs.append(deepcopy(payload))
                            inner.calls.append((step, deepcopy(messages), deepcopy(parameters)))
                            if not ready.wait(5): raise AssertionError('Sibling not admitted')
                            return {'docs': []}
                        ready.set()
                    return super().chat_json(step, messages, **parameters)
            trace = ParallelTrace(pfile=Path(td) / 'prompts.jsonl')
            ws = world(); ws.n_sessions = 2
            ws.entities['甲项目']['阶段'].ops.append(Op(1, _date_of(1), UPDATE, '收尾'))
            corpus = {'sessions': []}; done = set()
            with patch.object(config, 'pmap', ORIGINAL_PMAP):
                with self.assertRaises(renderer.MaterialRejected): self.render(trace, ws=ws, corpus=corpus, done=done)
            self.assertEqual(done, {1})
            self.assertEqual([s['session_id'] for s in corpus['sessions']], [1])
            checkpoints = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(td) / '05_signal_checkpoints').glob('*.json')]
            self.assertEqual(sum('documents' in item for item in checkpoints), 1)
            self.assertEqual(sorted(len(item['progress']['drafts']) for item in checkpoints), [1, 4])


class BlindProtocolTests(unittest.TestCase):
    docs = ['固定公开材料；不含隐藏答案。']
    queries = [{'key': 'q0', 'entity': '甲项目', 'field': '阶段'}]

    class Reader:
        def __init__(self, outputs): self.outputs, self.calls = list(outputs), []
        def chat_json(self, step, messages, **parameters):
            self.calls.append((step, deepcopy(messages), deepcopy(parameters)))
            if not self.outputs: raise AssertionError('Unexpected extra reader request')
            result = self.outputs.pop(0)
            if isinstance(result, BaseException): raise result
            return deepcopy(result)

    def read(self, reader):
        return renderer._discriminate_many(self.docs, self.queries, reader, semantic=True)

    def test_invalid_shapes_feed_same_reader_then_return_first_complete_answer(self):
        invalid = [[], {}, {'answers': []}, {'answers': [None]}, {'answers': [{'key': 'q0'}]},
            {'answers': [{'key': 'q0', 'answer': {}}]}, {'answers': [{'key': 'wrong', 'answer': '收尾'}]},
            {'answers': [{'key': 'q0', 'answer': ''}, {'key': 'q0', 'answer': ''}]}]
        for bad in invalid:
            with self.subTest(bad=bad):
                valid = {'answers': [{'key': 'q0', 'answer': '不确定'}]}
                reader = self.Reader([bad, valid])
                self.assertEqual(self.read(reader), {'q0': '不确定'})
                self.assertEqual(len(reader.calls), 2)
                first, second = reader.calls
                self.assertEqual(first[0], second[0]); self.assertEqual(first[2], second[2])
                self.assertEqual(first[2]['retries'], 3)
                self.assertEqual(second[1][:2], first[1])
                self.assertEqual(json.loads(second[1][2]['content']), bad)
                self.assertIn('protocol_feedback', json.loads(second[1][3]['content']))
                self.assertNotIn('true_value', json.dumps(second[1], ensure_ascii=False))

    def test_valid_wrong_uncertain_and_empty_answers_are_not_retried(self):
        for answer in ('明显错误答案', '不确定', '', 0, False):
            reader = self.Reader([{'answers': [{'key': 'q0', 'answer': answer}]}])
            self.assertEqual(self.read(reader), {'q0': str(answer)})
            self.assertEqual(len(reader.calls), 1)

    def test_three_schema_rounds_exhaust_without_changing_documents_or_queries(self):
        reader = self.Reader([{}] * 3)
        with self.assertRaisesRegex(RuntimeError, '3 schema attempts exhausted'): self.read(reader)
        self.assertEqual(len(reader.calls), 3)
        self.assertTrue(all(call[2]['retries'] == 3 for call in reader.calls))
        self.assertTrue(all(call[1][:2] == reader.calls[0][1] for call in reader.calls))

    def test_provider_and_global_stop_never_enter_schema_retry(self):
        for bad in [RuntimeError('reader transport failed'), {'__error__': 'reader provider failed'},
                ExecutionStopped('reader budget stop', kind='estimated_budget_limit'),
                {'__error__': 'reader budget stop', '__error_metadata__': {'global_stop': True, 'kind': 'estimated_budget_limit'}}]:
            reader = self.Reader([bad, {}])
            with self.subTest(bad=bad), self.assertRaises(RuntimeError): self.read(reader)
            self.assertEqual(len(reader.calls), 1)

    def test_same_version_journal_replays_invalid_and_valid_without_new_calls(self):
        state = {}; saved = []
        first = self.Reader([{}, {'answers': [{'key': 'q0', 'answer': '错误但格式完整'}]}])
        result = self.read(JournaledTracer(first, state, lambda: saved.append(deepcopy(state))))
        self.assertEqual(result, {'q0': '错误但格式完整'})
        before = deepcopy(state); second = self.Reader([])
        self.assertEqual(self.read(JournaledTracer(second, state, lambda: None)), result)
        self.assertEqual(second.calls, []); self.assertEqual(state, before)
        calls, = state['bindings'].values()
        self.assertEqual([call['output'] for call in calls], [{}, {'answers': [{'key': 'q0', 'answer': '错误但格式完整'}]}])

    def test_journal_interrupted_correction_is_not_redispatched(self):
        state = {}; reader = self.Reader([{}, OSError('fixed reader interruption')])
        with self.assertRaisesRegex(OSError, 'fixed reader interruption'):
            self.read(JournaledTracer(reader, state, lambda: None))
        before = deepcopy(state); resumed = self.Reader([])
        with self.assertRaises(RecordedCallFailure): self.read(JournaledTracer(resumed, state, lambda: None))
        self.assertEqual(resumed.calls, []); self.assertEqual(state, before)


if __name__ == '__main__': unittest.main(verbosity=2)
