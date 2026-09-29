"""Offline checks of the public-only, explicitly identified answer boundary."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge_contract_selftest import load_runner, question, judge
from eval import qa_cache
from eval.provenance import make_evaluation_context


IDENTITY = {"model": "explicit-solver", "transport_fingerprint": "profile-one",
            "answer_prompt_hash": "prompt-one", "max_tokens": 8192,
            "implementation_version": "public-answer/v1"}


class PublicSystem:
    def evaluation_config(self):
        return {"configuration_status": "declared", "budget": 12345}

    def retrieve(self, question, top_k=None):
        return "公开正文"

    def get_diagnostics(self):
        return {}


class InjectedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket", side_effect=AssertionError("network disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.runner = load_runner()
        self.grader = Mock(return_value=judge.judgement("correct", "fixture", "public meaning checked"))
        self.grader.cache_context = {"version": "offline-grader"}
        self.q = {"qid": "free", "question": "给出结论和依据", "reference_proposal":
                  {"answer": {"private": "PRIVATE_REFERENCE"}, "rationale": "PRIVATE_RATIONALE"},
                  "seed": "PRIVATE_SEED", "reviewer": "PRIVATE_REVIEW"}
        self.context = make_evaluation_context([(0, "2026-01-01", "公开正文")], "公开答题约定")

    def run_answer(self, answer, identity=None, **kwargs):
        return self.runner.run_system("public", [self.q], PublicSystem(), verbose=False,
            protocol="公开答题约定", judge_fn=self.grader,
            cache_context={"evaluation_context": self.context, "solver": {"model": "wrong-config"}},
            answer_fn=answer, solver_identity=deepcopy(IDENTITY) if identity is None else identity,
            **kwargs)

    def test_public_only_arguments_preserve_full_text_and_identity(self):
        text = " 结论。\n理由一。\n理由二。 "
        answer = Mock(return_value=text)
        rows = self.run_answer(answer)
        answer.assert_called_once_with("给出结论和依据", "公开正文", protocol="公开答题约定", max_tokens=8192)
        self.assertEqual(rows[0]["pred"], text)
        self.assertEqual(rows[0]["solver_identity"], IDENTITY)
        self.assertEqual(rows[0]["retrieved_context_hash"], qa_cache.digest("公开正文"))
        self.assertTrue(rows[0]["correct"])
        self.assertNotIn("PRIVATE", repr(answer.call_args))

    def test_bad_identity_fails_before_retrieval_or_dispatch(self):
        for identity in [{}, {**IDENTITY, "model": ""}, {**IDENTITY, "max_tokens": True},
                         {**IDENTITY, "max_tokens": 0}, {**IDENTITY, "extra": float("nan")},
                         {**IDENTITY, "extra": object()}]:
            with self.subTest(identity=identity), patch.object(PublicSystem, "retrieve") as retrieve:
                answer = Mock()
                with self.assertRaises(ValueError):
                    self.run_answer(answer, identity)
                retrieve.assert_not_called()
                answer.assert_not_called()

    def test_identity_and_callable_must_be_supplied_together(self):
        for extra in [{"answer_fn": Mock()}, {"solver_identity": IDENTITY},
                      {"answer_fn": "not callable", "solver_identity": IDENTITY}]:
            with self.assertRaises(ValueError):
                self.runner.run_system("s", [], PublicSystem(), verbose=False, **extra)

    def test_bad_answer_is_execution_failure_not_incorrect_prediction(self):
        for output in [None, "", " \n", {"answer": "text"}, 4]:
            self.grader.reset_mock()
            rows = self.run_answer(Mock(return_value=output))
            self.assertIsNone(rows[0]["correct"])
            self.assertEqual(rows[0]["execution"]["stage"], "answer")
            self.grader.assert_not_called()

    def test_failed_answer_has_no_hidden_retry(self):
        answer = Mock(side_effect=TimeoutError("offline"))
        rows = self.run_answer(answer)
        answer.assert_called_once()
        self.grader.assert_not_called()
        self.assertIsNone(rows[0]["correct"])

    def test_none_bench_id_does_not_read_or_write_cache(self):
        with patch.object(qa_cache, "load") as load, patch.object(qa_cache, "load_predictions") as predictions, \
             patch.object(qa_cache, "append") as grade, patch.object(qa_cache, "append_prediction") as pred:
            self.run_answer(Mock(return_value="完整回答"), bench_id=None, resume=False)
        for operation in [load, predictions, grade, pred]:
            operation.assert_not_called()

    def test_explicit_model_profile_and_prompt_invalidate_prediction_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)):
            answer = Mock(return_value="完整回答")
            first = self.run_answer(answer, bench_id="explicit")
            repeated = self.run_answer(answer, bench_id="explicit")
            self.assertEqual(answer.call_count, 1)
            self.assertTrue(repeated[0]["_prediction_resumed"])
            self.assertEqual(first[0]["retrieved_context_hash"], repeated[0]["retrieved_context_hash"])
            for key in ["model", "transport_fingerprint", "answer_prompt_hash", "implementation_version", "max_tokens"]:
                value = 16000 if key == "max_tokens" else "changed-" + key
                self.run_answer(answer, {**IDENTITY, key: value}, bench_id="explicit")
            self.assertEqual(answer.call_count, 6)

    def test_legacy_callable_arguments_remain_unchanged(self):
        with patch.object(self.runner, "unified_answer", return_value="孔雪") as legacy:
            result = self.runner.run_system("legacy", [question()], PublicSystem(), verbose=False, protocol="p")
        legacy.assert_called_once_with(question()["question"], "公开正文", protocol="p",
                                       max_tokens=self.runner.QA_MAX_TOKENS)
        self.assertNotIn("solver_identity", result[0])
        self.assertNotIn("retrieved_context_hash", result[0])


if __name__ == "__main__":
    unittest.main()
