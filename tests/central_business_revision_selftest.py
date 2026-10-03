"""Offline architect routing and preservation after executable-business feedback."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import config
from pipeline import central_office as office
from pipeline.seed_pack import seed_input
from seed_contract_selftest import fixture, blueprint, CouncilTracer


class BusinessRevisionTests(unittest.TestCase):
    def setUp(self):
        for target, name in ((socket.socket, 'connect'), (socket, 'getaddrinfo'),
                             (config, 'chat'), (config, 'chat_json')):
            self.enterContext(patch.object(target, name, side_effect=AssertionError('offline only')))

    def test_business_issue_returns_to_architect_with_full_candidate_and_contract(self):
        pack = fixture()
        desc, few = seed_input(pack)
        class Capture(CouncilTracer):
            reviews = 0
            authors = 0
            def chat_json(inner, step, messages, **kwargs):
                if step in ('council.world', 'council.world_repair'):
                    inner.authors += 1
                    if inner.authors == 2:
                        self.assertIn('【业务可执行性修订】', messages[0]['content'])
                        self.assertIn('【业务复核意见】', messages[1]['content'])
                        self.assertIn('缺少支持程度写入事件', messages[1]['content'])
                        self.assertIn('blueprint_requirements', messages[1]['content'])
                if step == 'council.blueprint_feasibility':
                    inner.reviews += 1
                    if inner.reviews == 1:
                        result = super().chat_json(step, messages, **kwargs)
                        result['decision'] = 'repair'
                        result['reason'] = '缺少支持程度写入事件'
                        result['issues'] = [{'finding': result['reason'], 'suggestion': '补充可写效果'}]
                        return result
                return super().chat_json(step, messages, **kwargs)
        tracer = Capture(blueprint(pack))
        result = office.central_office(desc, few, tracer, seed_pack=pack, log=lambda *_: None)
        self.assertEqual(tracer.authors, 2)
        self.assertEqual(tracer.reviews, 2)
        self.assertTrue(result['seed_audit']['passed'])
        self.assertEqual(result['world_blueprint']['event_types'], blueprint(pack)['event_types'])

    def test_initial_schema_repair_bytes_unchanged(self):
        c = {'world_blueprint': {'version': 1}}
        s, u = office._world_revision_messages('schema error', c, '{}', '\nseed')
        self.assertEqual(s, office.WORLD_REPAIR_SYS)
        self.assertEqual(u, office.render('council.world_repair_user', errors='schema error',
            candidate=json.dumps(c, ensure_ascii=False)) + '\n【每轮都必须完整保留的 observe 冻结清单】\n{}\nseed')

    def test_returned_execution_failure_stops_before_structure_repair(self):
        pack = fixture(); desc, few = seed_input(pack)
        class Failure(CouncilTracer):
            def chat_json(inner, step, messages, **kwargs):
                if step == 'council.world':
                    inner.calls.append((step, messages))
                    return {'__error__':'closed JSON syntax failure', 'kind':'json_syntax'}
                return super().chat_json(step, messages, **kwargs)
        tracer = Failure(blueprint(pack))
        with self.assertRaises(office.CouncilExecutionError) as caught:
            office.central_office(desc, few, tracer, seed_pack=pack, log=lambda *_:None)
        self.assertEqual(caught.exception.result['kind'], 'json_syntax')
        self.assertFalse(any(step == 'council.world_repair' for step, _ in tracer.calls))

    def test_followup_label_repair_preserves_added_writer_instruction_and_candidate(self):
        c = {'world_blueprint': {'event_types': [{'id': 'hero_choice', 'label': 'seed-label',
            'roles': {'faction': 'faction'}, 'effect_fields': [{'role': 'faction', 'field': '支持程度'}]}]}}
        before = deepcopy(c)
        s, u = office._world_revision_messages('label mismatch', c, '{}', 'seed',
            preserve_business_additions=True)
        self.assertIn('label 错误仅恢复 label', s)
        self.assertIn('支持程度', u)
        self.assertEqual(c, before)

    def test_repeated_business_failure_keeps_six_attempt_bound_and_blocks_mapping(self):
        pack = fixture(); desc, few = seed_input(pack)
        class AlwaysRepair(CouncilTracer):
            def chat_json(inner, step, messages, **kwargs):
                result = super().chat_json(step, messages, **kwargs)
                if step == 'council.blueprint_feasibility':
                    result['decision'] = 'repair'
                    result['issues'] = [{'finding': 'missing writer', 'suggestion': 'add writer'}]
                return result
        tracer = AlwaysRepair(blueprint(pack))
        with self.assertRaises(office.WorldBlueprintError):
            office.central_office(desc, few, tracer, seed_pack=pack, log=lambda *_: None)
        self.assertEqual(sum(s == 'council.world_repair' for s, _ in tracer.calls), 5)
        self.assertFalse(any(s == 'council.map' for s, _ in tracer.calls))


if __name__ == '__main__': unittest.main()
