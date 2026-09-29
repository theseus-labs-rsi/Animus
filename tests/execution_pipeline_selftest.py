"""Real adapter -> runner -> grade -> filtering, with offline transports."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge_contract_selftest import load_runner, config, question, contract, judge, row
from semantic_judge_selftest import Q, CORPUS, report as reference_report, grade_output, review_output
from eval import qa_cache
from eval.provenance import make_evaluation_context, record_provenance
from eval.question_filter import filter_questions
from eval.semantic_judge import SemanticJudge
from pipeline.semantic_review import review_questions


class ExecutionPipelineTest(unittest.TestCase):
    def setUp(self):
        network = patch("socket.socket", side_effect=AssertionError("network disabled"))
        network.start()
        self.addCleanup(network.stop)
        self.runner = load_runner()
        self.modules = patch.dict(sys.modules, {"config": config, "eval.multi_system": self.runner,
            "eval.memory_interface": SimpleNamespace(EmbedMemory=object, _chunk=Mock(), build_embed_memory=Mock()),
            "eval.embed_cache": SimpleNamespace(cached_embed=Mock(), cache_size=Mock()),
            "requests": SimpleNamespace(Session=lambda: SimpleNamespace(headers={}))})
        self.modules.start()
        self.base = importlib.import_module("eval.memory_systems.base")
        self.memos = importlib.import_module("eval.memory_systems.memos_adapter")
        self.context = make_evaluation_context([(0, "2026-01-01", "负责人孔雪")], "")

    def tearDown(self):
        self.modules.stop()

    def adapter(self, payload=None, error=None):
        instance = self.memos.MemOSAdapter()
        instance._session = Mock()
        if error:
            instance._session.post.side_effect = error
        else:
            instance._session.post.return_value = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
        return instance

    def test_transport_error_cannot_become_wrong_or_correct_refusal(self):
        for q in [question(), question(capability="L6_refusal", gt=None,
                                      question_contract=contract("abstention", abstention_kind="never_known"))]:
            with patch.object(self.runner, "unified_answer", return_value="无此项") as answer:
                records = self.runner.run_system("memos", [q], self.adapter(error=TimeoutError()),
                    cache_context={"evaluation_context": self.context}, verbose=False)
            answer.assert_not_called()
            self.assertIsNone(records[0]["correct"])
            self.assertEqual(records[0]["execution"]["stage"], "retrieve")
            self.assertEqual(self.runner.aggregate(records)["overall"]["n"], 0)
            peer = row(q, judge.judgement("correct", "fixture", "peer successful"), "无此项")
            peer["evaluation_provenance"] = record_provenance(q, self.context)
            _, filtered = filter_questions([q], {"memos": records, "peer": [peer]}, expected_context=self.context)
            self.assertEqual(filtered["items"][0]["disposition"], "kept_incomplete")

    def test_healthy_empty_search_still_reaches_solver(self):
        adapter = self.adapter({"code": 0, "data": {"memory_detail_list": [], "preference_detail_list": []}})
        with patch.object(self.runner, "unified_answer", return_value="孔雪") as answer:
            records = self.runner.run_system("memos", [question()], adapter,
                cache_context={"evaluation_context": self.context}, verbose=False)
        answer.assert_called_once()
        self.assertEqual(records[0]["execution_status"], "ok")
        self.assertTrue(records[0]["correct"])

    def test_answer_transport_failure_never_reaches_semantic_judge(self):
        spec = importlib.util.spec_from_file_location("offline_real_answer",
            Path(__file__).resolve().parents[1] / "eval/baseline_r1.py")
        real_answer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_answer)
        adapter = self.adapter({"code": 0, "data": {"memory_detail_list": [], "preference_detail_list": []}})
        for failure in (TimeoutError("offline"), "", None):
            kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
            grader = Mock()
            grader.cache_context = {}
            with patch.object(config, "chat", create=True, **kwargs) as call, patch("time.sleep"), \
                 patch.object(self.runner, "unified_answer", real_answer.unified_answer):
                records = self.runner.run_system("memos", [question()], adapter,
                    cache_context={"evaluation_context": self.context}, verbose=False, judge_fn=grader)
            self.assertEqual(call.call_count, 3)
            grader.assert_not_called()
            self.assertEqual(records[0]["execution_status"], "error")
            self.assertEqual(records[0]["execution"]["stage"], "answer")
            self.assertIsNone(records[0]["correct"])
            self.assertEqual(self.runner.aggregate(records)["overall"]["n"], 0)

    def test_finalization_evidence_is_preserved(self):
        adapter = Mock()
        adapter.ingest_session.return_value = {"status": "ok", "n_docs": 1, "completion": "accepted"}
        adapter.finalize_ingest.return_value = {"scope": "message_embeddings", "verified_messages": 2}
        _, executions = self.runner.prepare_systems(["zep"], [{"session_id": 0, "docs": ["text"]}],
                                                    lambda name: adapter)
        self.assertEqual(executions["zep"]["finalization"], adapter.finalize_ingest.return_value)

    def test_actual_adapter_configuration_invalidates_prediction_cache(self):
        class ConfiguredSystem:
            budget = 100
            def evaluation_config(self):
                return {"budget": self.budget}
            def retrieve(self, question, top_k=None):
                return "负责人孔雪"
            def get_diagnostics(self):
                return {}
        adapter = ConfiguredSystem()
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)), \
             patch.object(self.runner, "unified_answer", return_value="孔雪") as answer:
            kwargs = dict(verbose=False, bench_id="actual-config", cache_context={"evaluation_context": self.context})
            self.runner.run_system("s", [question()], adapter, **kwargs)
            self.runner.run_system("s", [question()], adapter, **kwargs)
            self.assertEqual(answer.call_count, 1)
            adapter.budget = 200
            self.runner.run_system("s", [question()], adapter, **kwargs)
            self.assertEqual(answer.call_count, 2)

    def test_failed_preparation_never_uses_cached_answers(self):
        q = question()
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)), \
             patch.object(self.runner, "unified_answer", return_value="孔雪") as answer:
            healthy = Mock()
            healthy.retrieve.return_value = "负责人孔雪"
            healthy.get_diagnostics.return_value = {}
            kwargs = dict(verbose=False, bench_id="test", cache_context={"evaluation_context": self.context})
            self.runner.run_system("memos", [q], healthy, **kwargs)
            failed = self.runner.run_system("memos", [q], healthy, execution={"status": "error", "stage": "ingest"}, **kwargs)
            self.assertEqual(answer.call_count, 1)
            self.assertEqual(failed[0]["judgement"]["verdict"], "error")

    def test_unverified_completion_is_incomplete_not_service_failure(self):
        exc = self.base.MemoryExecutionError("finalize", "completion_unverified")
        execution = self.runner.execution_failure("finalize", exc)
        records = self.runner.run_system("zep", [question()], None, execution=execution, verbose=False)
        agg = self.runner.aggregate(records)
        self.assertEqual(agg["n_execution_errors"], 0)
        self.assertEqual(agg["n_execution_incomplete"], 1)
        self.assertEqual(agg["overall"]["n"], 0)

    def test_bad_protocol_identity_rejected_before_any_answer(self):
        with patch.object(self.runner, "unified_answer") as answer, self.assertRaises(ValueError):
            self.runner.run_system("a", [question()], Mock(), protocol="different",
                                   cache_context={"evaluation_context": self.context}, verbose=False)
        answer.assert_not_called()

    def test_wrapper_failure_does_not_poison_shared_embedding(self):
        healthy = Mock()
        healthy._mem = object()
        healthy.ingest_session.return_value = {"status": "ok", "n_docs": 1, "completion": "completed"}
        def factory(name, **kw):
            if name == "C":
                raise RuntimeError("wrapper failure")
            return healthy
        systems, executions = self.runner.prepare_systems(["C", "A"], [{"session_id": 0, "docs": ["text"]}], factory)
        self.assertEqual(executions["C"]["status"], "error")
        self.assertEqual(executions["A"]["status"], "ok")
        self.assertIsNotNone(systems["A"])

    def test_free_question_can_be_graded_cached_and_filtered(self):
        context = make_evaluation_context([(0, "", CORPUS[0]["content"])], "回答日期")
        call = Mock(return_value=grade_output())
        grader = SemanticJudge(reference_report(), [Q], CORPUS, "回答日期", model="semantic-test", chat_json=call)
        system = Mock()
        system.retrieve.return_value = CORPUS[0]["content"]
        system.get_diagnostics.return_value = {}
        with tempfile.TemporaryDirectory() as directory, patch.object(qa_cache, "CACHE_DIR", Path(directory)), \
             patch.object(self.runner, "unified_answer", return_value="发布日期是2024年3月20日。") as answer:
            kwargs = dict(verbose=False, bench_id="free", cache_context={"evaluation_context": context},
                          protocol="回答日期", judge_fn=grader)
            first = self.runner.run_system("s", [Q], system, **kwargs)
            second = self.runner.run_system("s", [Q], system, **kwargs)
            self.assertTrue(first[0]["correct"])
            self.assertTrue(second[0]["_resumed"])
            self.assertEqual(answer.call_count, 1)
            call.assert_called_once()
            self.assertEqual(self.runner.aggregate(first)["overall"]["n"], 1)
            kept, report = filter_questions([Q], {"s": first, "peer": first}, expected_context=context)
            self.assertEqual(kept, [])
            self.assertEqual(report["counts"]["all_correct"], 1)

    def test_question_without_proposed_reference_stays_without_reference(self):
        q = {"question": Q["question"]}
        reference = review_questions([q], CORPUS, "回答日期", reviewer_model="offline",
            chat_json=lambda step, messages, **kw: (review_output("blind_read")
                if step.endswith("blind_read") else review_output(reference_status="not_provided")))
        grader = SemanticJudge(reference, [q], CORPUS, "回答日期", model="offline",
            chat_json=Mock(return_value=grade_output(reference_status="not_provided")))
        system = Mock()
        system.retrieve.return_value = CORPUS[0]["content"]
        system.get_diagnostics.return_value = {}
        with patch.object(self.runner, "unified_answer", return_value="2024-03-20"):
            records = self.runner.run_system("s", [q], system, verbose=False, protocol="回答日期", judge_fn=grader)
        self.assertNotIn("gt", records[0])
        self.assertTrue(records[0]["correct"])
        self.assertEqual(self.runner.aggregate(records)["overall"]["n"], 1)

    def report_results(self, **verdicts):
        results = {}
        for name, statuses in verdicts.items():
            records = [row(question(), judge.judgement(status, "offline", "fixture"))
                       for status in statuses]
            results[name] = {"records": records, "agg": self.runner.aggregate(records)}
        return results

    def test_single_system_has_no_cross_system_verdict_in_json_terminal_or_report(self):
        results = self.report_results(B=["correct", "correct"])
        summary = self.runner.discrimination_summary(results, ["B"])
        self.assertEqual(summary["status"], "insufficient_systems")
        for field in ["ov_spread", "max_line_spread", "discriminates", "headroom", "best"]:
            self.assertIsNone(summary[field])
        self.assertEqual(summary["ranking"], [])
        self.assertEqual(summary["overall"], {"B": 1.0})
        self.assertEqual(self.runner.discrimination_summary(results, ["B", "B"])["status"],
                         "insufficient_systems")
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            self.runner.print_table(results, ["B"])
            path = Path(directory)/"report.md"
            meta = {"bench": "offline", "corpus": "offline", "n_sessions": 1,
                    "n_docs": 1, "corpus_chars": 10,
                    **self.runner.selection_metadata(2, 2, 2, results)}
            self.runner.write_report(results, ["B"], meta, path)
            report_text = path.read_text(encoding="utf-8")
        for text in [output.getvalue(), report_text, "\n".join(self.runner.L_interpret(results, ["B"]))]:
            self.assertIn("至少需要两个不同系统", text)
            self.assertNotIn("区分度弱", text)
            self.assertNotIn("bench 偏易", text)
        self.assertIn("| B | 2 | 0 |", report_text)

    def test_comparison_keeps_multiple_system_differences_but_blocks_unscored(self):
        results = self.report_results(A=["correct", "correct"], B=["incorrect", "incorrect"])
        summary = self.runner.discrimination_summary(results, ["A", "B"])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["ov_spread"], 1.0)
        self.assertTrue(summary["discriminates"])
        pending = self.report_results(A=["correct"], B=["error"])
        summary = self.runner.discrimination_summary(pending, ["A", "B"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["discriminates"])
        self.assertEqual(self.runner.discrimination_summary({}, [])["status"], "insufficient_systems")

    def test_selection_counts_are_distinct_from_each_systems_scored_count(self):
        results = self.report_results(A=["correct", "incorrect", "error", "uncertain", "unjudgeable"],
                                     B=["correct"] * 5)
        a = results["A"]["agg"]
        self.assertEqual(a["overall"], {"n": 2, "correct": 1, "acc": 0.5})
        self.assertEqual((a["n_scored"], a["n_unscored"], a["n_unjudgeable"]), (2, 3, 1))
        self.assertEqual(a["n_incomplete"], 3)
        counts = self.runner.selection_metadata(8, 7, 5, results)
        self.assertEqual((counts["n_eligible"], counts["n_selected"], counts["n_excluded"],
                          counts["n_not_selected"]), (7, 5, 1, 2))
        self.assertEqual(counts["n_scored_by_system"], {"A": 2, "B": 5})
        self.assertEqual(counts["n_unscored_by_system"], {"A": 3, "B": 0})
        self.assertEqual((counts["n_judgeable"], counts["n_unjudgeable"]), (7, 1))
        self.assertEqual(counts, self.runner.selection_metadata(8, 7, 5, dict(reversed(list(results.items())))))

    def test_progress_overwrites_existing_file_on_windows_and_persists_done(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.runner, "ROOT", Path(directory)), \
             patch.object(Path, "rename", side_effect=FileExistsError("Windows destination exists")):
            probe = self.runner._EvalProbe()
            probe.start(systems=["B"], n_selected=1)
            probe.sys_start("B", 1)
            probe.answer_tick("B", 1)
            probe.done({"status": "insufficient_systems"})
            state = json.loads(probe._progress.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "done")
            self.assertEqual(state["sys"]["B"]["done"], 1)
            self.assertEqual(state["disc"]["status"], "insufficient_systems")
            self.assertFalse(list(probe.run_dir.glob("*.tmp")))
            self.assertIsNone(probe.write_error)

    def test_progress_concurrent_updates_preserve_counts_and_atomic_json(self):
        from concurrent.futures import ThreadPoolExecutor
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(self.runner, "ROOT", Path(directory)), \
             redirect_stderr(errors):
            probe = self.runner._EvalProbe()
            probe.start(systems=["A", "B"], n_selected=20)
            probe.embed_start(40)
            for name in ["A", "B"]:
                probe.sys_start(name, 20)
            rec = row(question(), judge.judgement("correct", "offline", "fixture"))
            def update(item):
                name, i = item
                probe.answer_tick(name, i)
                probe.judge_tick(name, i, rec)
                probe.embed_tick(i + (20 if name == "B" else 0))
                # A reader opening between writes must always get complete JSON.
                json.loads(probe._progress.read_text(encoding="utf-8"))
            jobs = [(name, i) for name in ["A", "B"] for i in reversed(range(1, 21))]
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(update, jobs))
            probe.done()
            state = json.loads(probe._progress.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "done")
            self.assertEqual(state["embed"]["done"], 40)
            for name in ["A", "B"]:
                self.assertEqual(state["sys"][name]["done"], 20)
                self.assertEqual(state["sys"][name]["judged"], 20)
                self.assertEqual(state["sys"][name]["correct"], 20)
            self.assertEqual(len(state["feed"]), 40)
            self.assertFalse(list(probe.run_dir.glob("*.tmp")))
            self.assertEqual(errors.getvalue(), "")

    def test_progress_retries_transient_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.runner, "ROOT", Path(directory)):
            probe = self.runner._EvalProbe()
            probe.start(systems=[])
            replace = self.runner.os.replace
            attempts = []
            def locked_then_available(source, destination):
                attempts.append(True)
                if len(attempts) < 3:
                    raise PermissionError()
                return replace(source, destination)
            with patch.object(self.runner.os, "replace", side_effect=locked_then_available), \
                 patch.object(self.runner.time, "sleep"):
                probe.done()
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(probe._progress.read_text(encoding="utf-8"))["status"], "done")
            self.assertIsNone(probe.write_error)

    def test_progress_write_failure_is_visible_and_recovery_is_possible(self):
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(self.runner, "ROOT", Path(directory)):
            probe = self.runner._EvalProbe()
            probe.start(systems=[])
            with patch.object(self.runner.os, "replace", side_effect=PermissionError()), redirect_stderr(errors):
                probe.done()
            self.assertEqual(probe.write_error, "PermissionError")
            self.assertIn("PermissionError", errors.getvalue())
            self.assertEqual(json.loads(probe._progress.read_text(encoding="utf-8"))["status"], "running")
            self.assertFalse(list(probe.run_dir.glob("*.tmp")))
            probe.done()
            self.assertEqual(json.loads(probe._progress.read_text(encoding="utf-8"))["status"], "done")
            self.assertIsNone(probe.write_error)

    def test_progress_run_directories_do_not_collide_in_the_same_second(self):
        with patch.object(self.runner.time, "strftime", return_value="20260917-000000"):
            first, second = self.runner._EvalProbe(), self.runner._EvalProbe()
        self.assertNotEqual(first.run_dir, second.run_dir)


if __name__ == "__main__":
    unittest.main()
