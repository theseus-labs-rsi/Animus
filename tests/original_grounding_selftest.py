"""Original merge/release integration; scripted model opinions, zero API calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from semantic_review_selftest import Scripted, blind, adjudication
from pipeline.grounding_review import review_grounding, validate_current_review, candidates_with_evidence, execution_complete
from pipeline.question_contract import attach_question_contract, bind_question_world
from pipeline.world_state import WorldState, Timeline, Op, SET
from pipeline.corpus_contract import review_documents, attach_receipts, fidelity_requirements
from corpus_fixture_helpers import fixed_positive_review
from pipeline.factory import ANSWER_PROTOCOL, LEGACY_ANSWER_PROTOCOL, _answer_protocol, STAGES
from pipeline.quality import evaluate_release, quality_snapshot
from eval.answer_task_review import POLICY_VERSION
from eval.provenance import public_protocol


def fixture():
    wp = {"quality_contract": {"version": 2, "scoring_policy": POLICY_VERSION,
                              "public_semantic_review": True},
          "active_lines": [{"line": "L1_timeline", "weight": 1}]}
    ws = WorldState({"测试报告": {"客户": Timeline([Op(0, "2025-01-06", SET, "北溟保险")])}}, n_sessions=1)
    q = attach_question_contract(bind_question_world({"line": "L1_timeline", "capability": "IE",
        "entity": "测试报告", "field": "客户", "gt": {"value": "北溟保险", "at_week": 0}, "aux": {"at_week": 0}, "evidence_sessions": [0]}, ws), wp)
    q["question"] = q["question_contract"]["canonical_question"]
    from pipeline.question_wording import review_wording
    q["question_validation"] = {"semantic_review": review_wording(q["question"], q["question_contract"],
        chat_json=lambda *a, **k: {"verdict": "equivalent", "reason": "Scripted fixture opinion", "issues": []},
        model="test")}
    docs = [{"doc_id": "doc0", "content": "测试报告客户是北溟保险，所属市场为欧洲。", "is_filler": False}]
    class CorpusReviewer:
        def chat_json(self, *a, **kw):
            return fixed_positive_review(a[1])
    attach_receipts(docs, review_documents(CorpusReviewer(), ws, 0, docs, requirements=fidelity_requirements(ws, 0)), 0)
    corpus = {"corpus": {"sessions": [{"session_id": 0, "date": "2025-01-06", "docs": docs}]}}
    return wp, ws, q, corpus, public_protocol({"answer_protocol": ANSWER_PROTOCOL})


def opinions(status="supported"):
    b, a = blind(), adjudication(reference_status=status, concerns=[],
        review_findings={"substantive_defects": [], "acceptable_brevity": [], "editorial_suggestions": []})
    for row in (b, a):
        row["coverage"]["inspected_doc_ids"] = ["d000001"]
    return Scripted(b, a)


class OriginalGroundingTests(unittest.TestCase):
    def test_original_value_answer_projection_preserves_raw_metadata(self):
        _, _, question, corpus, _ = fixture()
        for capability in ("IE", "MR"):
            q = deepcopy(question)
            q["capability"] = capability
            original = deepcopy(q)
            candidate = candidates_with_evidence([q], corpus, isolated_reference=True)[0]
            self.assertEqual(q, original)
            self.assertEqual(candidate["gt"], original["gt"])
            self.assertEqual(candidate["reference_proposal"], {"answer": "北溟保险", "rationale": ""})
            self.assertEqual(candidate["reference_projection"]["encoding"]["answer_pointer"], "/value")
        # Arbitrary structured references and explicit author proposals are
        # not interpreted as this legacy wire format merely for having a key.
        q = deepcopy(question); q["capability"] = "other"
        self.assertEqual(candidates_with_evidence([q], corpus, isolated_reference=True)[0]["reference_proposal"]["answer"], q["gt"])
        q["reference_proposal"] = {"answer": "保留明确提案", "rationale": "原理由"}
        self.assertEqual(candidates_with_evidence([q], corpus, isolated_reference=True)[0]["reference_proposal"], q["reference_proposal"])

    def test_isolated_execution_requires_its_independent_audit_stage(self):
        item = {"binding": {"reference_auditor_model": "test"}, "stage_execution": {
            "blind_read": {"status": "ok"}, "adjudicate": {"status": "ok"}}}
        self.assertFalse(execution_complete({"items": [item]}))
        item["stage_execution"]["reference_audit"] = {"status": "ok"}
        self.assertTrue(execution_complete({"items": [item]}))

    def test_policy_and_legacy_are_separate_and_phrasing_needs_well_posed(self):
        wp, _, q, _, protocol = fixture()
        self.assertIn(POLICY_VERSION, protocol)
        self.assertNotIn("只评分问题明确要求的主答案", protocol)
        self.assertEqual(q["question_contract"]["scoring_scope"], "task_with_supporting_reasons")
        self.assertEqual(_answer_protocol({}), LEGACY_ANSWER_PROTOCOL)
        self.assertIn("well_posed", next(s.needs for s in STAGES if s.name == "questions"))

    def test_current_review_replayed_without_network_and_stale_reference_refused(self):
        _, _, q, c, p = fixture()
        kept, r, review = review_grounding([q], c, p, chat_json=opinions(), model="test")
        self.assertEqual(len(kept), 1)
        validate_current_review(kept, c, p, review)
        changed = deepcopy(kept); changed[0]["gt"] = "别的客户"
        with self.assertRaises(ValueError):
            validate_current_review(changed, c, p, review)
        self.assertIn("lexical_diagnostic", r)

    def test_call_error_isolated_to_each_candidate_and_remains_pending(self):
        _, _, q, c, p = fixture()
        script = Scripted(RuntimeError("provider unavailable"))
        other = deepcopy(q); other["qid"] += "_second"
        kept, r, review = review_grounding([q, other], c, p, chat_json=script, model="test")
        self.assertEqual(len(script.calls), 2)
        self.assertEqual(r["n_dropped"], 0)
        self.assertEqual(r["n_pending"], 2)
        self.assertFalse(r["execution_stopped"])
        self.assertEqual(kept, [])

    def test_bad_reference_is_rejected_without_overwriting_it(self):
        _, _, q, c, p = fixture()
        before = deepcopy(q)
        kept, r, review = review_grounding([q], c, p, chat_json=opinions("contradicted"), model="test")
        self.assertEqual(kept, [])
        self.assertEqual(r["n_dropped"], 1)
        self.assertEqual(q, before)



if __name__ == "__main__":
    unittest.main()
