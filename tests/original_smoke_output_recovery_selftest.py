"""A billed length stop resumes with a larger ceiling and the same liabilities."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from original_smoke_json_retry_selftest import ReplaySDK
import original_smoke_corpus_review_override_selftest as fixture
from tools import run_original_bc_smoke as smoke


class OutputRecoveryTests(unittest.TestCase):
    setUp = fixture.CorpusReasoningOverrideTests.setUp
    run_tool = fixture.CorpusReasoningOverrideTests.run_tool

    def evidence(self):
        profile = {"model":"glm-5.3-flash", "stopped":True, "max_calls":1200,
            "admitted_calls":2, "max_cny":12, "budget_consumed_cny":3.5,
            "max_output_tokens":16384, "unsettled_reservation_calls":1, "source_drift":[],
            "stop_reason":{"reason":"json_execution_failure", "kind":"output_truncated",
                           "step":"world.instance.values", "call_id":"cut"}}
        rows = [{"event":"request", "model":profile["model"], "call_id":i} for i in ("old", "cut")]
        rows += [{"event":"response", "call_id":"cut", "response":{
            "usage":{"prompt_tokens":200, "completion_tokens":16384},
            "choices":[{"content":"", "finish_reason":"length"}]}},
            {"event":"call_error", "call_id":"cut", "step":"world.instance.values",
             "kind":"output_truncated", "usage_status":"reported", "local_cleanup_confirmed":True,
             "actual_output_cap":16384}]
        return profile, rows

    def check(self, profile, rows, ceiling=32768):
        path = self.directory / "trace.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return smoke.output_truncation_resume_evidence(profile, path, ceiling)

    def test_closed_billed_stop_preserves_old_ledger_and_unknown(self):
        p, rows = self.evidence(); before = deepcopy(p)
        evidence = self.check(p, rows)
        self.assertEqual(p, before)
        self.assertEqual(evidence["prior_admitted_calls"], 2)
        self.assertEqual(evidence["prior_budget_consumed_cny"], 3.5)
        self.assertEqual(evidence["unknown_reservations_preserved"], 1)
        self.assertEqual(evidence["new_output_ceiling"], 32768)

    def test_auth_drift_budget_and_same_ceiling_stay_closed(self):
        for key, value in (("source_drift", ["config.py"]), ("max_calls", 2),
            ("max_cny", 3.5), ("model", "different-model"),
            ("stop_reason", {"reason":"provider_failure", "kind":"http_status_401"})):
            with self.subTest(key=key):
                p, rows = self.evidence(); p[key] = value
                with self.assertRaises(ValueError): self.check(p, rows)
        for ceiling in (16384, 8192, 128001):
            p, rows = self.evidence()
            with self.assertRaises(ValueError): self.check(p, rows, ceiling)

    def test_incomplete_or_unbilled_evidence_stays_closed(self):
        p, rows = self.evidence()
        for missing in range(len(rows)):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.check(p, rows[:missing] + rows[missing + 1:])
        for change in ({"usage_status":"unknown"},
                       {"kind":"json_syntax"}, {"actual_output_cap":32768}):
            bad = deepcopy(rows); bad[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.check(p, bad)
        for change in ({"usage":None}, {"choices":[{"finish_reason":"stop"}]}):
            bad = deepcopy(rows); bad[-2]["response"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.check(p, bad)

    def test_terminal_same_ceiling_requires_no_reopening(self):
        p, rows = self.evidence(); p.update(stopped=False, max_output_tokens=32768)
        self.assertIsNone(self.check(p, rows))
        with self.assertRaises(ValueError): self.check(p, rows, 65536)

    def partition_evidence(self):
        p,rows=self.evidence();p['stop_reason']['step']='council.instance_plan_repair'
        rows[1].update(step='council.instance_plan_repair',messages=[{'role':'user','content':'immutable original task'}])
        rows[-1]['step']='council.instance_plan_repair'
        proposal={'decision':'plan','units':[{'unit_id':'actual-returned-proposal'}]}
        rows.insert(2,{'event':'response','call_id':'old','step':'council.instance_plan',
            'response':{'usage':{'prompt_tokens':10,'completion_tokens':10},'choices':[{'content':json.dumps(proposal),'finish_reason':'stop'}]}})
        digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        policy={'policy':'resource-allocation-then-unit-edits/v1','parent_call_id':'cut',
            'parent_step':'council.instance_plan_repair','parent_messages_sha256':digest(rows[1]['messages']),
            'original_proposal_sha256':digest(proposal),'first_new_steps':['council.instance_resources'],
            'max_semantic_attempts':3,'used_semantic_attempts':1,'remaining_semantic_attempts':2}
        path=self.directory/'partition.json';path.write_text(json.dumps(policy),encoding='utf8')
        trace=self.directory/'partition_trace.jsonl';trace.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
        return p,trace,path,policy

    def test_same_cap_partition_preserves_actual_proposal_and_ledger(self):
        p,trace,path,policy=self.partition_evidence();before=deepcopy(p)
        evidence=smoke.output_truncation_resume_evidence(p,trace,16384,path)
        self.assertEqual(p,before);self.assertEqual(evidence['reason'],'explicit_task_partition_resume')
        self.assertEqual(evidence['prior_budget_consumed_cny'],3.5)
        self.assertEqual(evidence['unknown_reservations_preserved'],1)

    def test_partition_cannot_replace_proposal_or_add_attempts(self):
        p,trace,path,policy=self.partition_evidence()
        for key,value in [('original_proposal_sha256','changed'),('remaining_semantic_attempts',3),
                          ('parent_messages_sha256','changed'),('first_new_steps',['blind_retry'])]:
            bad=dict(policy);bad[key]=value;path.write_text(json.dumps(bad),encoding='utf8')
            with self.subTest(key=key),self.assertRaises(ValueError):
                smoke.output_truncation_resume_evidence(p,trace,16384,path)

    def test_actual_wrapper_accounts_truncation_and_proves_output_recovery(self):
        empty = fixture.SimpleNamespace(id="cut", model="gpt-5.4-mini",
            usage=fixture.SimpleNamespace(prompt_tokens=20, completion_tokens=16384, total_tokens=16404),
            choices=[fixture.SimpleNamespace(index=0, finish_reason="length",
                message=fixture.SimpleNamespace(content="", refusal=None))])
        api = ReplaySDK([empty])
        def call(directory, cfg):
            fixture.Tracer(fixture.SimpleNamespace(dir=directory)).chat_json(
                "world.instance.values", [], max_tokens=32768, retries=3)
        with patch.object(fixture, "FakeSDK", return_value=api):
            result = self.run_tool(call, ["--max-output-tokens", "16384", "--min-output-tokens", "0", "--settle-reported-usage"])
        self.assertTrue(result.profile["stopped"])
        self.assertEqual(result.profile["settled_usage_calls"], 1)
        self.assertGreater(result.profile["reported_usage_estimate_cny"], 0)
        proof = smoke.output_truncation_resume_evidence(result.profile, result.directory / "llm_attempts.jsonl", 32768)
        self.assertEqual(proof["prior_admitted_calls"], 1)
        self.assertEqual(api.calls[0]["kwargs"]["max_completion_tokens"], 16384)


if __name__ == "__main__": unittest.main(verbosity=2)
