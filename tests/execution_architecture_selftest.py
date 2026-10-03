"""Fault injection through the shared runtime and real renderer, without providers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-disabled")
os.environ.setdefault("MODEL", "offline-disabled")
from execution_control import (ExecutionStopped, bounded_map, global_failure,
                               global_stop_exception, observe_failure, check_stop)
from pipeline.task_execution import JournaledTracer, UnfinishedCall, RecordedCallFailure
from pipeline import run as run_module, render as renderer, corpus_contract
from corpus_review_execution_selftest import world, opinion


class ExecutionArchitectureTests(unittest.TestCase):
    def test_failed_batch_stops_admission_and_drains_success(self):
        first, started, persisted, consumed = threading.Event(), [], [], []
        def source():
            for item in range(100):
                consumed.append(item)
                yield item
        def worker(item):
            started.append(item)
            if item == 0:
                first.set()
                raise ValueError("local failure")
            self.assertTrue(first.wait(2))
            persisted.append(item)
            return item
        with self.assertRaisesRegex(ValueError, "local failure"):
            bounded_map(worker, source(), 2)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(persisted, [1])

    def test_late_global_failure_keeps_local_cause(self):
        first = threading.Event()
        def worker(item):
            if item == 0:
                first.set()
                raise ValueError("original local failure")
            self.assertTrue(first.wait(2))
            raise ExecutionStopped("budget exhausted", kind="estimated_budget_limit")
        with self.assertRaises(ExecutionStopped) as caught:
            bounded_map(worker, [0, 1], 2)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_result_commit_precedes_sibling_stop(self):
        done = threading.Event()
        saved = []
        def worker(item):
            if item == 0:
                done.set()
                return "complete original result"
            self.assertTrue(done.wait(2))
            raise ExecutionStopped("budget exhausted")
        with self.assertRaises(ExecutionStopped):
            bounded_map(worker, [0, 1], 2, on_result=saved.append)
        self.assertEqual(saved, ["complete original result"])

    def test_character_topup_commits_and_measures_success(self):
        corpus = {"sessions": [{"session_id": 0, "date": "2026-01-01",
                              "docs": [{"doc_id": "original", "content": "原始材料"}]}]}
        saved = []
        renderer._top_up_haystack(corpus, 700, 0, None, "system", (), "周",
            lambda: saved.append(deepcopy(corpus)), lambda *_: None,
            filler_writer=lambda *_args: [{"content": "外围记录" * 200}])
        self.assertTrue(renderer.corpus_scale(corpus, 700, 0)["target_met"])
        self.assertEqual(saved[-1], corpus)

    def test_nested_stop_context_blocks_followups(self):
        def worker(_item):
            observe_failure(ExecutionStopped("authentication failed"))
            check_stop()
        with self.assertRaises(ExecutionStopped):
            bounded_map(lambda item: bounded_map(worker, [item], 1), [0], 1)
        self.assertEqual(bounded_map(lambda item: item * 2, [1, 2], 1), [2, 4])

    def test_business_json_cannot_be_a_global_stop(self):
        self.assertFalse(global_failure({"__error_metadata__": {"global_stop": True}}))
        self.assertFalse(global_failure({"__error__": "malformed", "__error_metadata__": "text"}))
        self.assertTrue(global_failure({"__error__": "stop", "__error_metadata__": {"kind": "http_status_401"}}))
        self.assertIsInstance(global_stop_exception(
            "Original experiment model/call/estimated budget limit; no provider dispatch"), ExecutionStopped)

    def test_journal_preserves_format_reply_across_interruption(self):
        class Provider:
            calls = 0
            def chat_json(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise SystemExit("interrupted second format call")
                return {"verdict": "original malformed opinion"}
        provider, state, saved = Provider(), {}, []
        save = lambda: saved.append(deepcopy(state))
        first = JournaledTracer(provider, state, save)
        old = first.chat_json("review", [{"content": "complete original input"}], max_tokens=100)
        with self.assertRaises(SystemExit):
            first.chat_json("review", [{"content": "original format correction"}], max_tokens=100)
        restored = deepcopy(saved[-1])
        resumed = JournaledTracer(provider, restored, lambda: None)
        self.assertEqual(resumed.chat_json("review", [{"content": "complete original input"}], max_tokens=100), old)
        with self.assertRaises(UnfinishedCall):
            resumed.chat_json("review", [{"content": "original format correction"}], max_tokens=100)
        self.assertEqual(provider.calls, 2)

    def test_journal_retains_raised_global_failure_without_redispatch(self):
        class Provider:
            calls = 0
            def chat_text(self, *_args, **_kwargs):
                self.calls += 1
                raise ExecutionStopped("original budget stop", kind="estimated_budget_limit")
        provider, state = Provider(), {}
        with self.assertRaises(ExecutionStopped):
            JournaledTracer(provider, state, lambda: None).chat_text("body", [])
        with self.assertRaises(RecordedCallFailure) as caught:
            JournaledTracer(provider, state, lambda: None).chat_text("body", [])
        self.assertTrue(global_failure(caught.exception))
        self.assertEqual(provider.calls, 1)

    def test_stage_cannot_succeed_without_artifact_or_finish_unstarted_stages(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)):
            run = run_module.Run("office", "lifecycle")
            run.log = lambda *_: None
            with self.assertRaisesRegex(RuntimeError, "without publishing"):
                run_module._run_stage(run, "a", lambda _run: None, "a.json")
            self.assertFalse(run.is_done("a"))
            run_module._run_stage(run, "a", lambda run: run.write("a.json", {}), "a.json")
            self.assertEqual(run_module._finalize_run(run, ["a", "b"]), "incomplete")
            run_module._run_stage(run, "b", lambda run: run.write("b.json", {}), "b.json")
            self.assertEqual(run_module._finalize_run(run, ["a", "b"]), "done")

    def test_stage_cannot_publish_success_after_swallowing_global_provider_stop(self):
        import config
        stopped = {"__error__": "budget admission stopped",
                   "__error_metadata__": {"kind": "estimated_budget_limit", "global_stop": True}}
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(config, "chat_json", return_value=stopped) as provider:
            run = run_module.Run("office", "global-stop")
            run.log = lambda *_: None
            def swallowed(stage_run):
                stage_run.tracer.chat_json("world.semantic_review", [{"role": "user", "content": "offline"}])
                stage_run.write("review.json", {"status": "passed"})
            with self.assertRaises(ExecutionStopped) as caught:
                run_module._run_stage(run, "review", swallowed, "review.json")
            self.assertEqual(caught.exception.kind, "estimated_budget_limit")
            self.assertEqual(run.manifest["stages"]["review"]["status"], "failed")
            self.assertEqual(run.tracer.n, 1)
            self.assertEqual(provider.call_count, 1)
            with self.assertRaises(ExecutionStopped):
                run_module._run_stage(run, "next", lambda r: r.write("next.json", {}), "next.json")
            self.assertNotIn("next", run.manifest["stages"])

    def test_local_provider_diagnostic_does_not_close_global_stage_gate(self):
        import config
        local = {"__error__": "connection unavailable",
                 "__error_metadata__": {"kind": "http_connect_error"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(config, "chat_json", return_value=local):
            run = run_module.Run("office", "local-diagnostic")
            run.log = lambda *_: None
            def diagnostic(stage_run):
                result = stage_run.tracer.chat_json("review", [{"role": "user", "content": "offline"}])
                stage_run.write("diagnostic.json", result)
            run_module._run_stage(run, "diagnostic", diagnostic, "diagnostic.json")
            self.assertEqual(run.manifest["stages"]["diagnostic"]["status"], "succeeded")
            self.assertEqual(run.tracer.n, 1)

    def test_stage_local_error_cannot_mask_prior_global_stop(self):
        import config
        stopped = {"__error__": "authentication stopped",
                   "__error_metadata__": {"kind": "http_status_401"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(run_module, "RUNS_DIR", Path(directory)), \
                patch.object(config, "chat_json", return_value=stopped):
            run = run_module.Run("office", "mixed-failure")
            run.log = lambda *_: None
            def mixed(stage_run):
                stage_run.tracer.chat_json("review", [{"role": "user", "content": "offline"}])
                raise ValueError("later local cleanup failed")
            with self.assertRaises(ExecutionStopped) as caught:
                run_module._run_stage(run, "review", mixed, "review.json")
            self.assertEqual(caught.exception.kind, "http_status_401")
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertEqual(run.manifest["stages"]["review"]["status"], "failed")


class RendererLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.calls = []
        self.negative = False
        parent = self
        class Tracer:
            pfile = parent.directory / "prompts.jsonl"
            def chat_json(self, step, messages, **_parameters):
                parent.calls.append(step)
                if step == "render.plan":
                    return {"documents": [{"purpose": "original fact", "sources": ["fact:0"], "time_context": "本期"}]}
                if step == "render.signal":
                    return {"docs": [{"title": "当期记录", "content": "测试报告的状态为待接收。"}]}
                if step == "render.discriminate":
                    return {"answers": [{"key": "q0", "answer": "待接收"}]}
                if step == "corpus.review":
                    payload = json.loads(messages[-1]["content"])
                    claims = [{"doc_index": 0, "quote": payload["documents"][0]["content"],
                               "reason": "Explicit offline negative opinion."}] if parent.negative else []
                    return opinion(payload, claims)
                raise AssertionError("unexpected provider task: " + step)
        self.tracer = Tracer()
        self.wp = {"generation_contract": {"material_first": True},
                   "quality_contract": {"corpus_review": True}, "domain_profile": {"doc_genres": ["纪要"]}}
        self.ws = world()
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def render(self):
        corpus, done = {"sessions": []}, set()
        renderer.render_corpus(self.wp, self.ws, 0, self.tracer, corpus, done,
                               lambda: None, log=lambda *_: None, haystack_ratio=0)
        return corpus

    def changed_implementation(self):
        original = Path.read_text
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            return text + "\n# explicit offline implementation change\n" if path.name == "render.py" else text
        return patch.object(Path, "read_text", changed)

    def test_implementation_change_revalidates_saved_calls_without_new_author_slots(self):
        original = self.render()
        calls = list(self.calls)
        with self.changed_implementation():
            restored = self.render()
        self.assertEqual(restored, original)
        self.assertEqual(self.calls, calls)

    def test_exhausted_negative_task_stays_exhausted_after_implementation_change(self):
        self.negative = True
        with self.assertRaises(renderer.MaterialRejected) as caught:
            self.render()
        self.assertEqual(len(caught.exception.report["drafts"]), 4)
        self.assertEqual(self.calls.count("render.signal"), 4)
        calls = list(self.calls)
        with self.changed_implementation(), self.assertRaises(renderer.MaterialRejected):
            self.render()
        self.assertEqual(self.calls, calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
