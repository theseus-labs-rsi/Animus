"""Offline record coverage and receipt tests; no model-semantic claim."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from pipeline import disclosure, world_semantics as review
from world_semantics_selftest import fixture, positive, FakeTracer


def planned():
    wp, ws, task = fixture(False)
    wp['quality_contract']['public_disclosure'] = True
    inventory = disclosure.catalogue(ws)
    fixed = {ref for row in disclosure._fixed(ws, inventory) for ref in row['refs']}
    free = [row for row in inventory if row['ref'] not in fixed]
    raw = {'records': [{'session': row['fact_session'], 'refs': [row['ref']],
                        'channel': '离线记录', 'acquisition_context': '离线获取说明，仅检查接口。'} for row in free],
           'undisclosed': [], 'reason': '离线安排不证明语义质量。'}
    ws.disclosure = disclosure.compile_plan(wp, ws, raw, task)
    assert not disclosure.validate_plan(ws)
    return wp, ws, task


def opinion(wp, ws, task):
    payload, _ = review._project(wp, ws, task)
    original_node = next(row['ref_id'] for row in payload['reference_index']
                         if row['source'] == 'world' and row['json_pointer'] == '/entities')
    raw = positive(False)
    raw['disclosure_reviews'] = [{'record_id': record['id'],
        'understanding': '此处仅保存离线桩对原记录主张的理解，不是实模型阅读结果。',
        'refs': [original_node], 'reason': '离线固定判断，用于校验覆盖与归属。',
        'status': 'compatible'} for record in ws.disclosure['records']]
    return raw


class RecordReviewTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        config = patch.dict(sys.modules, {'config': SimpleNamespace(REVIEWER_MODEL='offline-mini')})
        config.start(); self.addCleanup(config.stop)
        self.wp, self.ws, self.task = planned()
        self.raw = opinion(self.wp, self.ws, self.task)

    def run_review(self, raw=None, **kwargs):
        tracer = FakeTracer(deepcopy(self.raw if raw is None else raw))
        report = review.review_world(self.wp, self.ws, tracer, self.task, **kwargs)
        self.assertEqual(len(tracer.calls), 1)
        return report, tracer

    def test_one_call_records_include_fixed_source_claims_and_exact_locations(self):
        report, tracer = self.run_review()
        self.assertEqual(report['status'], 'passed', report)
        self.assertFalse(review.validate_review(report, self.wp, self.ws, self.task))
        self.assertTrue(any(row['id'].startswith('fixed_') for row in self.ws.disclosure['records']))
        locations = report['locations']['disclosures']
        self.assertEqual(len(locations), len(self.ws.disclosure['records']))
        for actual, record in zip(locations, self.ws.disclosure['records']):
            self.assertEqual(actual['record']['source_value'], record)
        self.assertEqual(tracer.calls[0]['params'], {'model': 'offline-mini', 'temperature': 0,
            'max_tokens': 16384, 'retries': 3, 'strict_json': True, 'response_format': {'type': 'json_object'}})

    def test_resolved_record_contexts_preserve_every_exact_original_target_and_location(self):
        snapshot = deepcopy(self.ws.to_dict())
        payload, _ = review._project(self.wp, self.ws, self.task)
        nodes = {item['ref_id']: item for item in payload['reference_index']}
        inventory = {item['ref']: (index, item) for index, item in
                     enumerate(payload['world']['disclosure']['catalogue'])}
        bundles = payload['disclosure_record_contexts']
        records = payload['world']['disclosure']['records']
        self.assertEqual(len(bundles), len(records))
        for index, (bundle, record) in enumerate(zip(bundles, records)):
            self.assertEqual(bundle['record_id'], record['id'])
            self.assertEqual(bundle['record'], record)
            self.assertEqual(bundle['publication_date'], self.ws.date_of_session(record['session']))
            self.assertEqual(nodes[bundle['record_ref_id']]['json_pointer'], f'/disclosure/records/{index}')
            self.assertEqual([item['target_ref'] for item in bundle['targets']], record['refs'])
            for target in bundle['targets']:
                ordinal, original = inventory[target['target_ref']]
                self.assertEqual(target['catalogue_entry'], original)
                self.assertEqual(nodes[target['catalogue_ref_id']]['json_pointer'], f'/disclosure/catalogue/{ordinal}')
                self.assertEqual(nodes[target['catalogue_ref_id']]['source'], 'world')
        # This convenience projection has no authority to mutate the source or
        # provide a second set of citable facts.
        bundles[0]['record']['acquisition_context'] = 'changed derived copy'
        bundles[0]['targets'][0]['catalogue_entry']['value'] = 'changed derived value'
        self.assertEqual(self.ws.to_dict(), snapshot)
        self.assertNotEqual(bundles[0]['record'], records[0])
        self.assertFalse(any('disclosure_record_contexts' in row['json_pointer']
                             for row in payload['reference_index']))

    def test_future_target_context_keeps_dates_and_defers_decision_to_original_reviewer(self):
        raw_plan = deepcopy(self.ws.disclosure['raw_output'])
        inventory = disclosure.catalogue(self.ws)
        future = next(item for item in inventory if item.get('fact_session') == 1)
        chosen = next(row for row in raw_plan['records'] if future['ref'] in row['refs'])
        chosen['session'] = 0
        chosen['acquisition_context'] = '本期预先取得明确注明未来生效日期的文件，记录其标注值，未声称已经生效。'
        self.ws.disclosure = disclosure.compile_plan(self.wp, self.ws, raw_plan, self.task)
        payload, _ = review._project(self.wp, self.ws, self.task)
        bundle = next(row for row in payload['disclosure_record_contexts']
                      if future['ref'] in row['record']['refs'])
        target = next(row for row in bundle['targets'] if row['target_ref'] == future['ref'])
        self.assertEqual(bundle['publication_date'], self.ws.date_of_session(0))
        self.assertEqual(target['original_calendar_date'], self.ws.date_of_session(1))
        self.assertEqual(target['catalogue_entry']['fact_date'], future['fact_date'])
        self.assertEqual(target['catalogue_entry']['value'], future['value'])
        # Identical ordering can receive either model judgment. The code checks
        # saved opinion consistency, never supplies a chronology rejection.
        raw = opinion(self.wp, self.ws, self.task)
        report, _ = self.run_review(raw)
        self.assertEqual(report['status'], 'passed')
        row = next(item for item in raw['disclosure_reviews'] if item['record_id'] == bundle['record_id'])
        row.update(status='repair', reason='离线反向意见：获取说明仍需澄清；此桩不宣称语义判断正确。')
        raw['decision'] = 'repair'; raw['repair_targets']['disclosure'] = True
        report, _ = self.run_review(raw)
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['repair_targets']['disclosure'])

    def test_old_review_version_and_changed_record_context_cannot_replay(self):
        report, _ = self.run_review()
        prior = deepcopy(report); prior['version'] = 'original-world-semantics/v12'
        self.assertTrue(review.validate_review(prior, self.wp, self.ws, self.task))
        changed = deepcopy(report)
        changed['input_snapshot']['disclosure_record_contexts'][0]['targets'][0]['catalogue_entry']['value'] = 'changed'
        self.assertTrue(review.validate_review(changed, self.wp, self.ws, self.task))

    def test_missing_array_and_missing_duplicate_unknown_rows_fail_without_retry(self):
        for mutation in ('array', 'missing', 'duplicate', 'unknown'):
            raw = deepcopy(self.raw)
            if mutation == 'array': del raw['disclosure_reviews']
            elif mutation == 'missing': raw['disclosure_reviews'].pop()
            elif mutation == 'duplicate': raw['disclosure_reviews'].append(deepcopy(raw['disclosure_reviews'][0]))
            else: raw['disclosure_reviews'][0]['record_id'] = 'invented'
            report, _ = self.run_review(raw)
            self.assertEqual(report['status'], 'error', mutation)
            self.assertEqual(report['raw_output'], raw)

    def test_negative_or_unresolved_record_cannot_be_accepted(self):
        for state in ('repair', 'unresolved'):
            raw = deepcopy(self.raw); raw['disclosure_reviews'][0]['status'] = state
            report, _ = self.run_review(raw)
            self.assertEqual(report['status'], 'error')
            self.assertEqual(report['raw_output']['disclosure_reviews'][0]['status'], state)

    def test_record_finding_can_use_original_disclosure_repair_without_duplicate_issue(self):
        raw = deepcopy(self.raw)
        raw['disclosure_reviews'][0].update(status='repair', reason='离线意见：本记录与所引原节点的时间说明需要修订。')
        raw['decision'] = 'repair'; raw['repair_targets']['disclosure'] = True
        self.assertEqual(raw['issues'], [])
        report, _ = self.run_review(raw)
        self.assertEqual(report['status'], 'failed', report)
        self.assertEqual(review.validate_review(report, self.wp, self.ws, self.task),
                         [{'code': 'world_review_not_passed', 'status': 'failed'}])

    def test_unresolved_without_permitted_target_is_retained(self):
        raw = deepcopy(self.raw); raw['disclosure_reviews'][0]['status'] = 'unresolved'
        raw['decision'] = 'unresolved'
        report, _ = self.run_review(raw)
        self.assertEqual(report['status'], 'unresolved')
        self.assertEqual(report['disclosure_reviews'], raw['disclosure_reviews'])

    def test_invalid_reference_and_empty_understanding_are_execution_errors(self):
        for field, value in (('refs', ['r_missing']), ('refs', []), ('understanding', ' '), ('reason', '')):
            raw = deepcopy(self.raw); raw['disclosure_reviews'][0][field] = value
            self.assertEqual(self.run_review(raw)[0]['status'], 'error')

    def test_opinion_or_located_record_mutation_cannot_replay(self):
        report, _ = self.run_review()
        for mode in ('summary', 'raw', 'location'):
            changed = deepcopy(report)
            if mode == 'summary': changed['disclosure_reviews'][0]['understanding'] = 'changed'
            elif mode == 'raw': changed['raw_output']['disclosure_reviews'][0]['reason'] = 'changed'
            else: changed['locations']['disclosures'][0]['record']['source_value']['channel'] = 'changed'
            self.assertTrue(review.validate_review(changed, self.wp, self.ws, self.task), mode)

    def test_same_truth_revised_plan_requires_fresh_bound_original_review(self):
        raw = deepcopy(self.raw); raw['disclosure_reviews'][0]['status'] = 'repair'
        raw['decision'] = 'repair'; raw['repair_targets']['disclosure'] = True
        previous, _ = self.run_review(raw)
        truth = disclosure._world(self.ws)
        plan_raw = deepcopy(self.ws.disclosure['raw_output'])
        plan_raw['records'][0]['acquisition_context'] = '原作者重新明确获取说明。'
        self.ws.disclosure = disclosure.compile_plan(self.wp, self.ws, plan_raw, self.task, feedback=previous)
        self.assertEqual(disclosure._world(self.ws), truth)
        self.assertTrue(review.validate_review(previous, self.wp, self.ws, self.task))
        next_raw = opinion(self.wp, self.ws, self.task)
        report, _ = self.run_review(next_raw, previous=previous)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['previous'], previous)
        self.assertFalse(review.validate_review(report, self.wp, self.ws, self.task))

    def test_future_plan_and_delayed_history_are_not_mechanically_rejected(self):
        original = deepcopy(self.ws.disclosure['raw_output'])
        for wording in ('本期回顾此前时点的事实，公开时间晚于实际发生时间。',
                        '本期预先取得文件，其中标注了尚未到来的计划日期；未声称计划已经执行。'):
            raw = deepcopy(original); raw['records'][0]['session'] = 1
            raw['records'][0]['acquisition_context'] = wording
            self.ws.disclosure = disclosure.compile_plan(self.wp, self.ws, raw, self.task)
            judgment = opinion(self.wp, self.ws, self.task)
            judgment['disclosure_reviews'][0]['understanding'] = wording
            report, _ = self.run_review(judgment)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['disclosure_reviews'][0]['understanding'], wording)

    def test_compatible_is_preserved_as_model_opinion_not_proven_by_code(self):
        raw = deepcopy(self.raw)
        raw['disclosure_reviews'][0]['understanding'] = '一个可能错误的自然理解；程序不靠词语推翻模型。'
        report, _ = self.run_review(raw)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['disclosure_reviews'], raw['disclosure_reviews'])

    def test_single_enabled_template_and_unchanged_legacy_schema(self):
        system = review._system(True)
        marker = '{\n  "mechanism_coverage":'
        self.assertEqual(system.count(marker), 1)
        template = json.loads(system[system.index(marker):])
        self.assertEqual(set(template), set(self.raw))
        self.assertLess(list(template).index('disclosure_reviews'), list(template).index('decision'))
        self.assertEqual(review._system(False), review.SYSTEM)
        wp, ws, task = fixture(False)
        legacy = review.review_world(wp, ws, FakeTracer(positive(False)), task)
        self.assertEqual(legacy['status'], 'passed')
        self.assertNotIn('disclosure_reviews', legacy)
        self.assertNotIn('disclosures', legacy['locations'])

    def test_empty_plan_has_empty_array_without_inventing_records(self):
        self.ws.conflicts = []
        inventory = disclosure.catalogue(self.ws)
        raw = {'records': [], 'undisclosed': [x['ref'] for x in inventory], 'reason': '保持全部未公开的离线边界。'}
        self.ws.disclosure = disclosure.compile_plan(self.wp, self.ws, raw, self.task)
        judgment = opinion(self.wp, self.ws, self.task)
        self.assertEqual(judgment['disclosure_reviews'], [])
        report, _ = self.run_review(judgment)
        self.assertEqual(report['status'], 'passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
