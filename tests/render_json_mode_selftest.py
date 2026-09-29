"""Real renderer/Tracer/config path with a local fake SDK, never a provider."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
os.environ.setdefault('OPENAI_API_KEY', 'offline-render-json')
from pipeline import render as renderer
from pipeline.run import Tracer
from public_stage_material_selftest import world, FakeTracer


class RenderJsonTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)

    def render(self, quality=True, *, failures=(), filler=False):
        ws, oracle, wire = world(), FakeTracer(failures=failures), []
        filler_system = renderer._filler_system({}, ws.world_blueprint)
        def create(**kwargs):
            wire.append(deepcopy(kwargs))
            messages = kwargs['messages']
            user = messages[-1]['content']
            if messages[0]['content'] == filler_system:
                text = '外围仓库完成无关档案整理。' * 20
            else:
                if user.lstrip().startswith('{'):
                    step = 'corpus.review'
                elif '[{"key"' in user:
                    step = 'render.discriminate'
                else:
                    step = 'render.signal'
                text = json.dumps(oracle.chat_json(step, messages), ensure_ascii=False)
            return SimpleNamespace(id='offline-response', model='offline-model', usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')])
        api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        spec = importlib.util.spec_from_file_location('render_json_test_config', ROOT / 'config.py')
        offline_config = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'dotenv': SimpleNamespace(load_dotenv=lambda *a: None),
                'openai': SimpleNamespace(OpenAI=lambda **k: api)}), patch.dict(os.environ, {
                'OPENAI_API_KEY': 'offline-render-json', 'OPENAI_BASE_URL': 'https://offline.invalid',
                'MODEL': 'offline-model', 'LLM_MIN_COMPLETION_TOKENS': '0'}):
            spec.loader.exec_module(offline_config)
        # Current production dispatch uses the cancellable async transport
        # beneath config.chat. Keep this wire test local by replacing that
        # exact boundary with the same fake completion used above.
        offline_config._perform_request = lambda **kw: create(
            model=kw['model'], messages=kw['messages'], **kw['parameters'])
        offline_config.pmap = lambda fn, items, **kw: [fn(item) for item in items]
        corpus = {'sessions': []}
        with tempfile.TemporaryDirectory(prefix='render-json-mode-offline-') as tmp:
            with patch.dict(sys.modules, {'config': offline_config}), patch.object(renderer, 'config', offline_config):
                renderer.render_corpus({'quality_contract': {'corpus_review': quality}}, ws,
                    100 if filler else 0, Tracer(SimpleNamespace(dir=Path(tmp))), corpus,
                    set(), lambda: None, log=lambda *a: None)
            events = [json.loads(s) for s in (Path(tmp) / 'llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        return wire, events, corpus

    def assert_wire_matches_trace(self, wire, events):
        requests = [row for row in events if row['event'] == 'request']
        self.assertEqual(len(wire), len(requests))
        for actual, logged in zip(wire, requests):
            self.assertEqual(actual['messages'], logged['messages'])
            self.assertEqual(actual.get('response_format'), logged['logical_parameters']['response_format'])
            self.assertEqual(actual.get('response_format'), logged['parameters'].get('response_format'))
        return requests

    def test_quality_signal_rules_and_blind_reader_request_json_object(self):
        wire, events, corpus = self.render()
        requests = self.assert_wire_matches_trace(wire, events)
        authors = [row for row in requests if row['step'] == 'render.signal']
        self.assertEqual(3, len(authors))  # two ordinary periods + one public-rule author
        self.assertEqual(1, sum('【冻结的公共阶段定义】' in row['messages'][-1]['content'] for row in authors))
        selected = [row for row in requests if row['step'] in ('render.signal', 'render.discriminate')]
        self.assertEqual(5, len(selected))
        self.assertTrue(all(row['parameters']['response_format'] == {'type': 'json_object'} for row in selected))
        self.assertTrue(all(row['parameters']['max_tokens'] == 16384 for row in selected))

    def test_public_rule_semantic_repair_keeps_same_json_mode_and_call_count(self):
        wire, events, _ = self.render(failures=[True, False])
        requests = self.assert_wire_matches_trace(wire, events)
        authors = [row for row in requests if row['step'] == 'render.signal'
                   and '【冻结的公共阶段定义】' in row['messages'][-1]['content']]
        self.assertEqual(2, len(authors))
        self.assertTrue(all(row['parameters']['response_format'] == {'type': 'json_object'} for row in authors))
        self.assertEqual([0.6, 0.2], [row['parameters']['temperature'] for row in authors])
        self.assertTrue(all(row['parameters']['max_tokens'] == 16384 for row in authors))

    def test_legacy_signal_and_blind_reader_wire_stays_without_json_mode(self):
        wire, events, _ = self.render(quality=False)
        requests = self.assert_wire_matches_trace(wire, events)
        selected = [(row, kwargs) for row, kwargs in zip(requests, wire)
                    if row['step'] in ('render.signal', 'render.discriminate')]
        self.assertEqual(5, len(selected))  # legacy public-rule helper also runs for this typed fixture
        self.assertTrue(all('response_format' not in kwargs for row, kwargs in selected))

    def test_quality_filler_still_requests_plain_text(self):
        wire, events, _ = self.render(filler=True)
        requests = self.assert_wire_matches_trace(wire, events)
        filler = [row for row in requests if row['step'] == 'render.filler']
        self.assertEqual(2, len(filler))
        self.assertTrue(all('response_format' not in row['parameters'] for row in filler))
        self.assertTrue(all(row['parameters']['max_tokens'] == renderer.FILLER_TEXT_MAX_TOKENS for row in filler))


if __name__ == '__main__':
    unittest.main(verbosity=2)
