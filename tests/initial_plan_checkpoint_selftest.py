"""Offline physical-checkpoint continuation and epoch admission fault tests."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
# This fixture installs network, provider and dotenv guards before project imports.
import initial_plan_recovery_selftest as fixtures
from pipeline import initial_plan_recovery as recovery
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from tools import run_original_bc_smoke as smoke
from copy import deepcopy
from unittest.mock import patch
import json
import unittest


class CheckpointTests(unittest.TestCase):
    write = fixtures.RecoveryTests.write
    write_rows = fixtures.RecoveryTests.write_rows
    make_run = fixtures.RecoveryTests.make_run

    def setUp(self):
        fixtures.RecoveryTests.setUp(self)
        request = recovery.build_receipt(self.directory, self.current)
        self.old_receipt_sha = '7' * 64
        self.original_audit_bytes = (self.directory / '01_instance_plan_audit.json').read_bytes()
        profile_bytes = (self.directory / 'experiment_profile.json').read_bytes()
        (self.directory / 'experiment_profile_before_resume_fixture.json').write_bytes(profile_bytes)
        self.write('00_initial_plan_execution_claim.json', {'run_id': recovery.RUN_ID,
            'request_sha256': self.old_receipt_sha, 'original_profile_sha256': recovery._sha(profile_bytes)})
        self.write(recovery.CLAIM, {'run_id': recovery.RUN_ID, 'request_sha256': self.old_receipt_sha})
        self.write('experiment_source_hashes_initial_plan_recovery_start.json', self.current)
        self.write('experiment_source_hashes_at_end.json', self.current)
        self.proposal = {'decision': 'repair', 'reason': 'preserved original response', 'changes': [], 'cohort_changes': []}
        self.record = {'version': recovery.VERSION, 'status': 'repair_failed',
            'error_type': 'PlanConflict', 'request_sha256': self.old_receipt_sha,
            'source_migration': request['source_migration'],
            'original_audit_sha256': recovery._sha(self.original_audit_bytes),
            'business_review': None, 'logical_plan_attempt': 2, 'max_logical_plan_attempts': 3,
            'business_revision_remaining': 1,
            'transaction': {'drafts': [{'iteration': 1, 'candidate': deepcopy(self.audit['transactions'][0]['drafts'][0]['candidate']),
                                       'unit_proposals': {'report-case': self.proposal}, 'findings': self.audit['findings']}],
                            'edit_attempts': {'report-case': [{'attempt': 1, 'status': 'locally_compiled', 'proposal': self.proposal}]},
                            'unit_proposals': {'report-case': self.proposal}}}
        self.audit['recovery_history'] = [self.record]
        self.profile.update(admitted_calls=12, settled_usage_calls=12,
            resume_history=[{'transport_recovery': {'receipt_sha256': self.old_receipt_sha,
                                                  'source_migration': request['source_migration']}}])
        messages = [{'role': 'user', 'content': json.dumps({'unit': {'unit_id': 'report-case'}})}]
        self.trace += [{'event': 'request', 'step': 'council.instance_plan_unit_repair', 'call_id': '11', 'operation_id': '11',
                        'messages': messages, 'model': 'glm-5.3-flash'},
                       {'event': 'response', 'step': 'council.instance_plan_unit_repair', 'call_id': '11', 'operation_id': '11',
                        'response': {'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
                                     'choices': [{'content': json.dumps(self.proposal)}]}},
                       {'event': 'json_result', 'step': 'council.instance_plan_unit_repair', 'operation_id': '11',
                        'parsed': self.proposal}]
        self.prompts.append({'step': 'council.instance_plan_unit_repair', 'ok': True,
                             'messages': messages, 'output': self.proposal})
        self.current = dict(self.current, **{'pipeline/initial_plan_recovery.py': '6' * 64})
        self.persist()

    def persist(self):
        self.write('01_initial_plan_recovery.json', self.record)
        self.write('01_instance_plan_audit.json', self.audit)
        self.write('experiment_profile.json', self.profile)
        self.write_rows('llm_attempts.jsonl', self.trace)
        self.write_rows('prompts.jsonl', self.prompts)

    def receipt(self):
        return recovery.build_checkpoint_receipt(self.directory, self.current)

    def permit(self):
        self.write('checkpoint_request.json', self.receipt())
        return recovery.validate_checkpoint_receipt(self.directory / 'checkpoint_request.json', self.directory, self.current)

    def test_complete_physical_history_bound_and_read_only(self):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        receipt = self.receipt()
        self.assertEqual(receipt['epoch'], 2)
        self.assertEqual(receipt['prior_physical_calls'], 12)
        self.assertEqual(receipt['used_unit_author_attempts'], {'report-case': 1})
        self.assertEqual(receipt['author_bindings'][0]['call_id'], '11')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_old_one_shot_entry_stays_closed(self):
        with self.assertRaises(ValueError): recovery.build_receipt(self.directory, self.current)

    def test_unknown_physical_request_rejected(self):
        self.trace.append(dict(self.trace[-3], call_id='pending', operation_id='pending'))
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_profile_unsettled_rejected(self):
        self.profile['unsettled_reservation_calls'] = 1
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_forged_checkpoint_reply_rejected(self):
        self.record['transaction']['edit_attempts']['report-case'][0]['proposal'] = {'decision': 'repair', 'changes': []}
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_forged_raw_physical_reply_rejected(self):
        self.trace[-2]['response']['choices'][0]['content'] = '{}'
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_forged_prompt_reply_rejected(self):
        self.prompts[-1]['output'] = {}
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_counters_cannot_be_reset(self):
        self.profile['admitted_calls'] = 11
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_original_audit_immutable(self):
        self.audit['attempts'][0]['extra'] = 'tampered'
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_source_epoch_mismatch_rejected(self):
        self.write('experiment_source_hashes_initial_plan_recovery_start.json', self.current)
        with self.assertRaises(ValueError): self.receipt()

    def test_blueprint_review_protocol_migration_is_explicitly_allowed(self):
        self.current['pipeline/blueprint_feasibility.py'] = '8' * 64
        self.current['pipeline/render.py'] = 'a' * 64
        receipt = self.receipt()
        self.assertIn('pipeline/blueprint_feasibility.py', receipt['source_migration']['changed_paths'])
        self.assertIn('pipeline/render.py', receipt['source_migration']['changed_paths'])
        self.assertEqual(receipt['source_migration']['target_source_hashes'], self.current)
        permit = self.permit()
        self.assertEqual(permit['source_migration'], receipt['source_migration'])

    def test_blueprint_allowance_does_not_allow_adjacent_source(self):
        self.current['pipeline/blueprint_feasibility.py'] = '8' * 64
        self.current['pipeline/blueprint_feasibility_extra.py'] = '9' * 64
        with self.assertRaises(ValueError): self.receipt()

    def test_unapproved_source_path_rejected(self):
        self.current['config.py'] = '9' * 64
        with self.assertRaises(ValueError): self.receipt()

    def test_business_or_logical_allowance_cannot_reset(self):
        self.record['logical_plan_attempt'] = 1
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_prior_pending_author_rejected(self):
        self.record['transaction']['edit_attempts']['report-case'][0]['status'] = 'requested'
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_budget_cap_cannot_increase(self):
        self.profile['max_cny'] = 101
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_settled_total_cannot_hide_changed_original_cost_prefix(self):
        self.trace[31]['response']['usage']['prompt_tokens'] -= 1
        self.trace[-2]['response']['usage']['prompt_tokens'] += 1
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_three_joint_batches_retain_history_with_original_author_capacity(self):
        draft = self.record['transaction']['drafts'][0]
        self.record['transaction']['drafts'] = [dict(deepcopy(draft), iteration=n) for n in (1,2,3)]
        self.persist()
        receipt = self.receipt()
        self.assertEqual(receipt['joint_drafts_consumed'], 3)
        self.assertEqual(receipt['joint_batch_policy'], 'consume_remaining_original_author_slots')
        self.assertEqual(receipt['remaining_unit_author_slots']['report-case'], 2)
        self.assertEqual(receipt['maximum_additional_joint_batches'], sum(receipt['remaining_unit_author_slots'].values()))
        self.assertEqual(receipt['max_unit_author_attempts'], 3)

    def test_joint_drafts_cannot_be_renumbered_or_skipped(self):
        self.record['transaction']['drafts'][0]['iteration'] = 4
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_binary_float_tail_uses_exact_trace_price_without_mutating_ledger(self):
        self.profile['budget_consumed_cny'] = .13562878000000002
        self.profile['reported_usage_estimate_cny'] = .13562877999999998
        self.persist()
        before = (self.directory / 'experiment_profile.json').read_bytes()
        receipt = self.receipt()
        self.assertEqual(receipt['prior_estimated_cny'], '0.13562878')
        self.assertEqual(receipt['ledger_representation_tolerance_cny'], '0.000000000001')
        self.assertEqual((self.directory / 'experiment_profile.json').read_bytes(), before)

    def test_actual_small_cost_discrepancy_is_not_hidden_as_float_error(self):
        self.profile['budget_consumed_cny'] = .135628781
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_nonfinite_ledger_value_is_rejected(self):
        self.profile['budget_consumed_cny'] = float('nan')
        self.persist()
        with self.assertRaises(ValueError): self.receipt()

    def test_new_epoch_single_entry_claim_preserves_old_claims_and_profile(self):
        permit = self.permit()
        keep = ['00_initial_plan_execution_claim.json', recovery.CLAIM, 'experiment_profile.json']
        before = {name: (self.directory / name).read_bytes() for name in keep}
        smoke._claim_initial_plan_execution(self.directory, permit)
        with self.assertRaises(FileExistsError): smoke._claim_initial_plan_execution(self.directory, permit)
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in keep})
        with self.assertRaises(ValueError): self.receipt()

    def test_module_passes_cumulative_transaction_and_appends_epoch(self):
        permit = self.permit()
        before_tx = deepcopy(self.record['transaction'])
        before_history = deepcopy(self.audit['recovery_history'])
        keep = ['00_initial_plan_execution_claim.json', recovery.CLAIM, 'experiment_profile.json', 'llm_attempts.jsonl', 'prompts.jsonl']
        before = {name: (self.directory / name).read_bytes() for name in keep}
        class Boundary(Exception): pass
        def repair(wp, candidate, feedback, tracer, audit, *, checkpoint, resume_checkpoint):
            self.assertTrue(resume_checkpoint)
            self.assertEqual(audit, before_tx)
            tracer.chat_json('council.instance_plan_unit_repair', [])
            raise Boundary('zero provider boundary')
        run = self.make_run()
        bounded = enter_bounded_run(recovery.RUN_ID)
        token = recovery.enter_recovery(permit)
        try:
            with patch('pipeline.plan_transactions.repair', side_effect=repair):
                with self.assertRaises(Boundary): recovery.recover_if_permitted(run)
        finally:
            recovery.leave_recovery(token)
            leave_bounded_run(bounded)
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in keep})
        after = run.read('01_instance_plan_audit.json')
        self.assertEqual(after['recovery_history'][:-1], before_history)
        self.assertEqual(after['recovery_history'][-1]['transaction'], before_tx)
        self.assertEqual(after['recovery_history'][-1]['epoch'], 2)
        self.assertEqual(after['recovery_history'][-1]['logical_plan_attempt'], 2)
        self.assertEqual(after['recovery_history'][-1]['business_revision_remaining'], 1)
        self.assertTrue((self.directory / '01_initial_plan_recovery_snapshot_epoch_0001.json').exists())

    def test_compiler_and_independent_accept_required_before_install(self):
        permit = self.permit()
        before_history = deepcopy(self.audit['recovery_history'])
        def repair(wp, candidate, feedback, tracer, audit, *, checkpoint, resume_checkpoint):
            self.assertTrue(resume_checkpoint)
            tracer.chat_json('council.instance_plan_unit_repair', [])
            candidate['units'][0]['events'].append({'slot': 'report-case-extra-review', 'type': 'reviewed',
                'participants': {'report': 'report-slot'}, 'session': 1, 'caused_by': 'revision-zero'})
            return candidate
        run = self.make_run()
        bounded = enter_bounded_run(recovery.RUN_ID)
        token = recovery.enter_recovery(permit)
        try:
            with patch('pipeline.plan_transactions.repair', side_effect=repair), patch(
                    'pipeline.blueprint_feasibility.assess', return_value=fixtures.ACCEPT):
                self.assertTrue(recovery.recover_if_permitted(run))
        finally:
            recovery.leave_recovery(token)
            leave_bounded_run(bounded)
        after = run.read('01_instance_plan_audit.json')
        self.assertEqual(after['recovery_history'][:-1], before_history)
        self.assertEqual(after['recovery_history'][-1]['status'], 'accepted')
        self.assertEqual(after['recovery_history'][-1]['logical_plan_attempt'], 2)
        self.assertTrue(run.read('01_whitepaper.json')['business_instance_plan'])
        self.assertEqual(run.read('experiment_profile.json')['admitted_calls'], 12)


if __name__ == '__main__': unittest.main(verbosity=2)
