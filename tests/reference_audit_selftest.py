from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reference_audit import audit_reference
from pipeline.reference_locations import reference_value_hash

Q = {"qid": "q", "question": "材料能确定什么？", "reference_proposal": {
    "answer": "无法确定是否完成。", "rationale": "只见拟办通知。"},
    "blind_read": {"answer": "HIDDEN_BLIND"}, "solver_answer": "HIDDEN_SOLVER"}


def opinion():
    return {"decision": "accept", "reason": "明确证据边界。", "task_requirements": ["区分已知与未知"],
        "claims": [{"reference_part": "answer", "source_quote": "无法确定是否完成。",
            "assessment": "有据", "explanation": "材料仅有拟办。", "evidence": []}],
        "task_coverage": "已完成主要任务。", "substantive_defects": [], "acceptable_brevity": [],
        "editorial_suggestions": [], "suggested_revision": "", "limitations": [], "evidence": []}


class ReferenceAuditTests(unittest.TestCase):
    def run_audit(self, out=None, max_calls=1):
        self.calls = []
        def fake(step, messages, **params):
            self.calls.append((step, messages, params)); return deepcopy(opinion() if out is None else out)
        return audit_reference(Q, [{"content": "拟办理，请等待通知。", "title": "HIDDEN_TITLE"}],
            "仅依据给定材料", model="model", chat_json=fake, max_calls=max_calls)

    def test_target_isolation_and_exact_original(self):
        result = self.run_audit()
        payload = json.loads(self.calls[0][1][1]["content"])
        self.assertNotIn("HIDDEN", json.dumps(payload))
        self.assertEqual(payload["original_reference"], Q["reference_proposal"])
        self.assertTrue(result["proposal_ready"])
        self.assertEqual(result["claim_locations"][0]["source_quote"], "无法确定是否完成。")

    def test_structured_nodes_are_taken_only_from_the_unchanged_reference(self):
        question = deepcopy(Q)
        question["reference_proposal"]["answer"] = {"a/b~c": [{"状态": None, "计划": ["甲", "乙"]}]}
        original = deepcopy(question)
        output = opinion()
        claim = output["claims"][0]
        del claim["source_quote"]
        claim.update(location_scope="node", json_pointer="/a~1b~0c/0")
        calls = []
        def fake(step, messages, **params):
            calls.append(json.loads(messages[1]["content"]))
            return deepcopy(output)
        result = audit_reference(question, [{"content": "计划待定。"}], "只依据材料",
                                 model="offline", chat_json=fake)
        self.assertTrue(result["proposal_ready"])
        self.assertEqual(calls[0]["original_reference"], original["reference_proposal"])
        location = result["claim_locations"][0]
        self.assertEqual(location["source_value"], original["reference_proposal"]["answer"]["a/b~c"][0])
        self.assertEqual(location["source_value_hash"], reference_value_hash(location["source_value"]))
        self.assertEqual(question, original)
        self.assertEqual(result["raw_output"], output)

    def test_node_location_does_not_upgrade_a_semantic_unresolved_opinion(self):
        output = opinion(); output["decision"] = "unresolved"
        claim = output["claims"][0]; del claim["source_quote"]
        claim.update(location_scope="node", json_pointer="")
        result = self.run_audit(output)
        self.assertTrue(result["proposal_ready"])
        self.assertEqual(result["proposal"]["decision"], "unresolved")
        self.assertEqual(result["claim_locations"][0]["source_value"], Q["reference_proposal"]["answer"])

    def test_bad_or_mixed_nodes_keep_raw_and_cannot_be_ready(self):
        for target in [{"location_scope": "node", "json_pointer": "/missing"},
                       {"location_scope": "node", "json_pointer": "", "source_quote": ""},
                       {"location_scope": "node", "json_pointer": "/~2"}]:
            with self.subTest(target=target):
                output = opinion(); claim = output["claims"][0]; del claim["source_quote"]
                claim.update(target)
                result = self.run_audit(output)
                self.assertFalse(result["proposal_ready"])
                self.assertEqual(result["raw_output"], output)
                self.assertIn("claim_not_in_original_reference:0", result["format_issues"])

    def test_omitted_task_can_be_described_without_inventing_reference_claims(self):
        output = opinion(); output.update(decision="revise", claims=[],
            task_coverage="未解释要求的条件。", substantive_defects=["遗漏条件解释。"],
            suggested_revision="补充公开证据支持的条件解释。")
        result = self.run_audit(output)
        self.assertTrue(result["proposal_ready"])
        self.assertEqual(result["claim_locations"], [])
        self.assertEqual(result["proposal"]["decision"], "revise")

    def test_quote_from_other_answer_is_rejected(self):
        out = opinion(); out["claims"][0]["source_quote"] = "确定完成。"
        result = self.run_audit(out)
        self.assertFalse(result["proposal_ready"])
        self.assertEqual(result["raw_output"], out)

    def test_reference_part_cannot_be_swapped(self):
        out = opinion(); out["claims"][0]["reference_part"] = "rationale"
        self.assertFalse(self.run_audit(out)["proposal_ready"])

    def test_valid_unknown_not_programmatically_rejected(self):
        result = self.run_audit()
        self.assertEqual(result["proposal"]["decision"], "accept")
        self.assertNotIn("answerability", result)

    def test_bad_evidence_never_promoted(self):
        out = opinion(); out["claims"][0]["evidence"] = [{"doc_id": "d000001", "field": "content",
            "quote": "已经办理", "role": "support", "explanation": "wrong"}]
        self.assertFalse(self.run_audit(out)["proposal_ready"])

    def test_zero_budget_does_not_call(self):
        result = self.run_audit(max_calls=0)
        self.assertFalse(self.calls); self.assertFalse(result["proposal_ready"])

    def test_repeated_quote_is_not_falsely_located_to_first_claim(self):
        question = deepcopy(Q)
        question["reference_proposal"]["answer"] = "甲未确认，乙已确认，丙未确认。"
        output = opinion(); output["claims"][0]["source_quote"] = "未确认"
        result = audit_reference(question, [{"content": "拟办理，请等待通知。"}],
            "只依据材料", model="offline", chat_json=lambda *a, **k: output)
        location = result["claim_locations"][0]
        self.assertTrue(result["proposal_ready"])
        self.assertNotIn("start", location)
        self.assertEqual(location["location_status"], "ambiguous")
        self.assertEqual([p["start"] for p in location["matching_spans"]], [1, 11])
        self.assertTrue(all(p["json_pointer"] == "" and p["source_kind"] == "string_value"
                            for p in location["matching_spans"]))


if __name__ == "__main__": unittest.main()
