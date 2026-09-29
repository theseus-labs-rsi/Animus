"""Offline original-CLI A-policy wiring; fake outputs are not semantic evidence."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "config" not in sys.modules:
    cfg = ModuleType("config")
    cfg.MODEL = cfg.JUDGE_MODEL = cfg.STRUCTURE_MODEL = cfg.DISCRIMINATOR_MODEL = "offline-model"
    cfg.chat = cfg.chat_json = Mock(side_effect=AssertionError("No provider in offline tests"))
    cfg.pmap = lambda fn, items, **kw: [fn(item) for item in items]
    cfg._trace_secrets = lambda: []
    sys.modules["config"] = cfg


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture = load("_multi_policy_fixture", ROOT / "tests/answer_task_judge_selftest.py")
from eval import answer_task_review as policy, qa_cache
from eval.grading import ANSWER_TASK_JUDGE_VERSION, JUDGE_VERSION, SEMANTIC_JUDGE_VERSION, is_scored
from eval.provenance import public_protocol, protocol_scoring_policy
from eval.semantic_judge import resolve_scoring_config
from pipeline.semantic_review import review_questions


class FakeMemory:
    def ingest_session(self, session):
        return {"status": "ok", "n_docs": len(session["docs"])}

    def finalize_ingest(self, **kwargs):
        pass

    def retrieve(self, question, **kwargs):
        return "报告于2024年3月20日发布。"

    def get_diagnostics(self):
        return {}


class OriginalPolicyCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        optional = {"eval.memory_interface": SimpleNamespace(EmbedMemory=object, _chunk=Mock(), build_embed_memory=Mock()),
                    "eval.embed_cache": SimpleNamespace(cached_embed=Mock(), cache_size=Mock())}
        with patch.dict(sys.modules, optional):
            self.module = load("_original_policy_cli", ROOT / "eval/multi_system.py")
        self.questions = [{**deepcopy(fixture.Q), "line": "L1_timeline", "capability": "IE",
                           "gt": {"value": "2024-03-20"}}]
        self.corpus = {"corpus": {"sessions": [{"session_id": 0, "date": "2024-03-20",
            "docs": [{"doc_id": "original-doc", "content": "报告于2024年3月20日发布。",
                      "title": "PRIVATE_TITLE", "writer_notes": "PRIVATE_NOTES"}]}]}}
        self.bench = self.base / "06_grounded_questions.json"
        self.material = self.base / "05_corpus.json"
        self.about = self.base / "00_about.json"
        self.review = self.base / "review.json"
        self.bench.write_text(json.dumps(self.questions, ensure_ascii=False), encoding="utf-8")
        self.material.write_text(json.dumps(self.corpus, ensure_ascii=False), encoding="utf-8")
        self.set_protocol()
        self.probe = self.module._EvalProbe()
        self.probe.run_dir = self.base / "execution"
        self.probe._progress = self.probe.run_dir / "_progress.json"
        self.package = ModuleType("eval.memory_systems")
        self.package.make_system = Mock(side_effect=lambda *_a, **_k: FakeMemory())
        self.calls = []
        self.solver_messages = []
        self.final_outputs = []
        self.reader_output = fixture.opinion()
        self.reader_output["observations"][0]["source_quote"] = "2024-03-20"
        self.guard = patch.object(socket, "socket", side_effect=AssertionError("Network disabled"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def set_protocol(self, *, a=True, text=None):
        protocol = {"rules": ["回答日期。"]}
        if a:
            protocol["scoring_policy"] = policy.POLICY_VERSION
        about = {"answer_protocol": protocol} if text is None else {"public_protocol": text}
        self.protocol = public_protocol(about)
        self.about.write_text(json.dumps(about, ensure_ascii=False), encoding="utf-8")
        report = review_questions(self.questions, self.corpus, self.protocol, reviewer_model="offline-reference",
            chat_json=lambda step, messages, **kw: fixture.fixture.review_output(
                "blind_read" if step.endswith("blind_read") else "adjudicate"))
        self.review.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    def fake_solver(self, messages, **params):
        self.solver_messages.append(deepcopy(messages))
        self.assertIn(self.protocol, messages[0]["content"])
        self.assertNotIn("PRIVATE_", json.dumps(messages))
        return "2024-03-20"

    def fake_grade(self, messages, **params):
        self.calls.append({"messages": deepcopy(messages), "params": deepcopy(params)})
        payload = json.loads(messages[-1]["content"])
        self.assertNotIn("PRIVATE_", json.dumps(payload))
        self.assertEqual(params["model"], "offline-judge")
        self.assertEqual(params["retries"], 1)
        if "review_identity" in payload:
            self.assertNotIn("reference_proposal", payload)
            return deepcopy(self.reader_output)
        self.assertEqual(payload["solver_answer"], "2024-03-20")
        return deepcopy(self.final_outputs.pop(0) if self.final_outputs else fixture.final_output())

    def execute(self, extra=(), *, review=True, filter_enabled=True):
        argv = ["multi_system", "--bench", str(self.bench), "--corpus", str(self.material),
                "--systems", "fake1,fake2", "--workers", "1", "--allow-unverified"]
        if review:
            argv += ["--semantic-review", str(self.review), "--judge-model", "offline-judge"]
        if filter_enabled:
            argv += ["--keep-easy-ratio", "1"]
        argv += list(extra)
        from pipeline import quality
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"eval.memory_systems": self.package}))
            stack.enter_context(patch.object(self.module, "_EvalProbe", return_value=self.probe))
            stack.enter_context(patch.object(quality, "require_release", return_value={
                "status": "failed", "eligible": False, "override": True}))
            stack.enter_context(patch.object(self.module.config, "chat", side_effect=self.fake_solver))
            stack.enter_context(patch.object(self.module.config, "chat_json", side_effect=self.fake_grade))
            stack.enter_context(patch.object(self.module.config, "_trace_secrets", return_value=[], create=True))
            stack.enter_context(patch.object(qa_cache, "CACHE_DIR", self.base / "cache"))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            if protocol_scoring_policy(self.protocol):
                stack.enter_context(patch.object(self.module, "judge_record", side_effect=AssertionError("No legacy fast path")))
                stack.enter_context(patch.object(self.module, "is_judgeable", side_effect=AssertionError("No legacy eligibility gate")))
            self.module.main()
        return json.loads((self.probe.run_dir / "results.json").read_text(encoding="utf-8"))

    def test_A_auto_routes_real_main_and_exact_answer_through_direct_judge(self):
        result = self.execute()
        self.assertEqual((len(self.calls), len(self.solver_messages)), (2, 2))
        self.assertEqual(result["judge_mode"], "semantic")
        self.assertEqual(result["judge_version"], ANSWER_TASK_JUDGE_VERSION)
        self.assertEqual(result["scoring_configuration"], resolve_scoring_config(policy.POLICY_VERSION))
        self.assertEqual(result["result_scope"], "research_only")
        for system in result["results"].values():
            self.assertTrue(is_scored(system["records"][0]))
        filtered = json.loads((self.probe.run_dir / "filtered/filter_report.json").read_text(encoding="utf-8"))
        self.assertEqual(filtered["counts"]["all_correct"], 1)
        self.assertEqual(filtered["counts"]["kept_easy_sample"], 1)

    def test_explicit_reader_method_defaults_to_four_physical_grading_calls(self):
        result = self.execute(["--scoring-policy", policy.POLICY_VERSION,
                               "--grading-method", policy.VERSION])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(result["scoring_configuration"]["calls_per_answer"], 2)
        self.assertTrue(all(is_scored(s["records"][0]) for s in result["results"].values()))

    def test_A_auto_discovers_complete_source_review_and_export_keeps_its_bytes(self):
        saved = self.base / "06_semantic_review.json"
        saved.write_bytes(self.review.read_bytes())
        result = self.execute(["--judge-model", "offline-judge"], review=False)
        self.assertEqual(result["judge_version"], ANSWER_TASK_JUDGE_VERSION)
        copied = self.probe.run_dir / "filtered/06_semantic_review.json"
        self.assertEqual(copied.read_bytes(), saved.read_bytes())
        manifest = json.loads((copied.parent / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["derived_from"]["semantic_review"]["scope"], "unchanged_complete_source_review")
        self.assertTrue(manifest["derived_from"]["research_only"])
        self.assertFalse(json.loads((copied.parent / "07_release.json").read_text(encoding="utf-8"))["eligible"])

    def test_bad_auto_discovered_review_stops_before_ingest(self):
        (self.base / "06_semantic_review.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.execute(["--judge-model", "offline-judge"], review=False)
        self.package.make_system.assert_not_called()

    def test_reader_pair_budget_does_not_spend_an_orphan_call(self):
        result = self.execute(["--grading-method", policy.VERSION, "--max-judge-calls", "1"])
        self.assertEqual(len(self.calls), 0)
        self.assertTrue(all(s["agg"]["n_scored"] == 0 for s in result["results"].values()))
        filtered = json.loads((self.probe.run_dir / "filtered/filter_report.json").read_text(encoding="utf-8"))
        self.assertEqual(filtered["counts"]["kept_incomplete"], 1)

    def test_same_item_hold_closes_before_aggregate_and_filter(self):
        self.final_outputs = [fixture.final_output(), fixture.final_output(
            requires_item_reassessment=True, reassessment_reason="离线构造：评分范围未决。")]
        result = self.execute()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result["reassessment_closure"]["n_disputed_items"], 1)
        for system in result["results"].values():
            self.assertEqual(system["agg"]["n_scored"], 0)
            self.assertIsNone(system["records"][0]["correct"])
        self.assertTrue(result["results"]["fake1"]["records"][0]["original_correct"])
        filtered = json.loads((self.probe.run_dir / "filtered/filter_report.json").read_text(encoding="utf-8"))
        self.assertEqual(filtered["counts"]["kept_incomplete"], 1)
        self.assertEqual(filtered["counts"]["all_correct"], 0)

    def test_A_invalid_entry_options_stop_before_ingest_or_solver(self):
        for args, review in [(["--judge-mode", "legacy"], True), (["--no-protocol"], True),
                             (["--scoring-policy", "other"], True), ([], False),
                             (["--grading-method", "unknown"], True)]:
            with self.subTest(args=args, review=review), self.assertRaises(SystemExit):
                self.execute(args, review=review)
        self.package.make_system.assert_not_called()
        self.assertEqual((self.calls, self.solver_messages), ([], []))

    def test_unpublished_explicit_A_rejected_before_ingest(self):
        self.set_protocol(a=False)
        with self.assertRaises(SystemExit):
            self.execute(["--judge-mode", "semantic", "--scoring-policy", policy.POLICY_VERSION])
        self.package.make_system.assert_not_called()

    def test_A_cannot_be_downgraded_with_an_about_override(self):
        other = self.base / "old-about.json"
        other.write_text(json.dumps({"public_protocol": "回答日期"}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.execute(["--about", str(other), "--judge-mode", "legacy"], review=False)
        self.package.make_system.assert_not_called()

    def test_malformed_policy_and_stale_review_fail_before_ingest(self):
        self.about.write_text(json.dumps({"public_protocol": self.protocol + "extra"}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.execute()
        self.set_protocol()
        report = json.loads(self.review.read_text(encoding="utf-8"))
        report["public_protocol"] = "stale"
        self.review.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.execute()
        self.package.make_system.assert_not_called()
        self.assertEqual((self.calls, self.solver_messages), ([], []))

    def test_old_no_policy_protocol_defaults_to_legacy(self):
        self.set_protocol(a=False)
        result = self.execute(review=False)
        self.assertEqual(result["judge_mode"], "legacy")
        self.assertEqual(result["judge_version"], JUDGE_VERSION)
        self.assertEqual(len(self.calls), 0)
        self.assertTrue(all(is_scored(s["records"][0]) for s in result["results"].values()))

    def test_old_explicit_semantic_remains_legacy_semantic_policy(self):
        self.set_protocol(a=False)
        result = self.execute(["--judge-mode", "semantic"])
        self.assertEqual(result["judge_version"], SEMANTIC_JUDGE_VERSION)
        self.assertNotIn("scoring_configuration", result)
        self.assertEqual(len(self.calls), 2)

    def test_run_system_cannot_bypass_A_routing(self):
        memory = Mock()
        for judge in (None, SimpleNamespace(scoring_configuration=resolve_scoring_config())):
            with self.subTest(judge=judge), self.assertRaises(ValueError):
                self.module.run_system("fake", self.questions, memory, protocol=self.protocol, judge_fn=judge)
        memory.retrieve.assert_not_called()

    def test_renderer_preserves_legacy_and_validates_exact_new_block(self):
        legacy = {"answer_protocol": {"rules": ["业务规则"], "gold_sentinel_map": {"INSUFFICIENT": "查无"}}}
        before = public_protocol(legacy)
        self.assertIsNone(protocol_scoring_policy(before))
        current = deepcopy(legacy)
        current["answer_protocol"]["scoring_policy"] = policy.POLICY_VERSION
        after = public_protocol(current)
        self.assertEqual(after, policy.append_scoring_policy(before))
        self.assertEqual(protocol_scoring_policy(after), policy.POLICY_VERSION)
        current["answer_protocol"]["scoring_policy"] = "unknown"
        with self.assertRaises(ValueError):
            public_protocol(current)
        for malformed in (after + "extra", after + after, after.replace(policy.POLICY_VERSION, "unknown"), policy.POLICY_TEXT):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                protocol_scoring_policy(malformed)


class GroundingExecutionBoundary(unittest.TestCase):
    def setUp(self):
        self.fx = load("_original_grounding_fixture", ROOT / "tests/original_grounding_selftest.py")
        self.wp, self.world, q, self.corpus, self.protocol = self.fx.fixture()
        self.questions = [q, {**deepcopy(q), "qid": q["qid"] + "_second"}]
        self.guard = patch.object(socket, "socket", side_effect=AssertionError("Network disabled"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def release(self, kept, review):
        with tempfile.TemporaryDirectory() as directory:
            from pipeline.grounding_review import candidates_with_evidence, selection
            candidates = candidates_with_evidence(self.questions, self.corpus)
            _, routing = selection(candidates, review)
            artifacts = {"01_whitepaper.json": self.wp, "02_world.json": self.world.to_dict(),
                "04_questions.json": self.questions, "05_corpus.json": self.corpus,
                "06_grounded_questions.json": kept,
                "06_grounding_report.json": routing,
                "00_about.json": {"answer_protocol": self.fx.ANSWER_PROTOCOL},
                "06_semantic_review.json": review}
            for name, value in artifacts.items():
                (Path(directory) / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            return self.fx.evaluate_release(directory)

    def test_partial_call_budget_releases_only_completed_question(self):
        from pipeline.grounding_review import execution_complete
        kept, routing, review = self.fx.review_grounding(self.questions, self.corpus, self.protocol,
            chat_json=self.fx.opinions(), model="test", max_calls=2)
        self.assertEqual((len(kept), routing["n_pending"], routing["n_dropped"]), (1, 1, 0))
        self.assertFalse(execution_complete(review))
        self.assertFalse(routing["execution_stopped"])
        self.assertEqual(routing["caller_invocations"], 2)
        result = self.release(kept, review)
        self.assertTrue(result["eligible"], result)
        self.assertEqual(result["checks"]["partition"]["counts"]["pending_review"], 1)

    def test_provider_failure_keeps_matching_completed_subset(self):
        old_final, _, _ = self.fx.review_grounding(self.questions[:1], self.corpus, self.protocol,
            chat_json=self.fx.opinions(), model="test")
        script = self.fx.opinions()
        script.responses.append(TimeoutError("offline provider failure"))
        kept, routing, review = self.fx.review_grounding(self.questions, self.corpus, self.protocol,
            chat_json=script, model="test")
        self.assertEqual(kept, old_final)  # The stale file can exactly match the successful subset.
        self.assertTrue(routing["execution_stopped"])
        self.assertEqual(routing["caller_invocations"], 3)
        result = self.release(old_final, review)
        self.assertTrue(result["eligible"], result)
        self.assertEqual(result["checks"]["partition"]["counts"]["pending_review"], 1)

    def test_complete_semantic_unknown_is_not_an_execution_failure(self):
        from pipeline.grounding_review import execution_complete
        script = self.fx.opinions("unresolved")
        kept, routing, review = self.fx.review_grounding(self.questions[:1], self.corpus, self.protocol,
            chat_json=script, model="test")
        self.assertTrue(execution_complete(review))
        self.assertTrue(routing["execution_complete"])
        self.assertEqual((kept, routing["n_pending"], routing["n_dropped"]), ([], 1, 0))

    def test_latched_logical_calls_are_not_reported_as_provider_calls(self):
        from pipeline.grounding_review import execution_complete
        script = self.fx.Scripted(TimeoutError("offline provider failure"))
        questions = [*self.questions, {**self.questions[0], "qid": "third"}]
        kept, routing, review = self.fx.review_grounding(questions, self.corpus, self.protocol,
            chat_json=script, model="test")
        self.assertEqual(len(script.calls), 1)
        self.assertEqual((routing["caller_invocations"], routing["suppressed_after_failure"],
                          routing["logical_calls_used"]), (1, 2, 3))
        self.assertIsNone(routing["paid_provider_calls"])
        self.assertEqual(review["execution_accounting"]["caller_invocations"], 1)
        self.assertFalse(execution_complete(review))
        self.assertEqual((kept, routing["n_pending"], routing["n_dropped"]), ([], 3, 0))

    def test_filtered_A_subset_keeps_full_review_and_research_scope(self):
        from eval.semantic_judge import SemanticJudge
        from eval.provenance import record_provenance
        from eval.question_filter import export_filtered_benchmark
        from pipeline.quality import ReleaseError
        self.world = self.fx.WorldState({"测试报告": {
            "客户": self.fx.Timeline([self.fx.Op(0, "2025-01-06", self.fx.SET, "北溟保险")]),
            "所属市场": self.fx.Timeline([self.fx.Op(0, "2025-01-06", self.fx.SET, "欧洲")])}}, n_sessions=1)
        docs = [{"doc_id": "doc0", "content": "测试报告客户是北溟保险，所属市场为欧洲。", "is_filler": False}]
        from pipeline.corpus_contract import fidelity_requirements
        requirements = fidelity_requirements(self.world, 0)
        reviewer = SimpleNamespace(chat_json=lambda *a, **k: {
            "document_reviews": [{"doc_index": 0, "status": "supported",
                                  "reason": "Fixed offline fixture opinion."}],
            "verdict": "pass", "unsupported_claims": [], "coverage": [
                {"requirement_id": item['requirement_id'], "status": "supported",
                 "reason": "Fixed fixture: both fields occur in the supplied sentence.",
                 "evidence": [{"doc_index": 0, "quote": docs[0]['content']}]}
                for item in requirements]})
        self.fx.attach_receipts(docs, self.fx.review_documents(
            reviewer, self.world, 0, docs, requirements=requirements), 0)
        self.corpus["corpus"]["sessions"][0]["docs"] = docs
        q2 = self.fx.attach_question_contract(self.fx.bind_question_world({
            "line": "L1_timeline", "capability": "IE", "entity": "测试报告", "field": "所属市场",
            "gt": {"value": "欧洲", "at_week": 0}, "aux": {"at_week": 0}, "evidence_sessions": [0]}, self.world), self.wp)
        q2["question"] = q2["question_contract"]["canonical_question"]
        from pipeline.question_wording import review_wording
        q2["question_validation"] = {"semantic_review": review_wording(q2["question"], q2["question_contract"],
            chat_json=lambda *a, **k: {"verdict": "equivalent", "reason": "Offline fixture opinion", "issues": []},
            model="test")}
        self.questions[1] = q2
        script = self.fx.opinions()
        script.responses.extend(self.fx.opinions().responses)
        final, _, review = self.fx.review_grounding(self.questions, self.corpus, self.protocol,
            chat_json=script, model="test")
        self.assertEqual(len(final), 2)
        response = self.fx.opinions().responses[-1]
        response.update(answer_verdict="correct", format_compliance="compliant",
            additional_facts={"status": "not_assessed", "reason": "无附言"},
            answer_review={"primary_task": "报告客户身份", "answer_meaning": "北溟保险",
                "claims": [{"claim": "客户为北溟保险", "task_role": "primary", "assessment": "supported",
                    "evidence_indices": [0], "explanation": "正文给出"}], "reference_comparison": "符合原文"},
            requires_item_reassessment=False, reassessment_reason="无题目争议",
            answer_task_review_response="按公开材料核对主任务，未发现其他关键理由。")
        judge = SemanticJudge(review, final, self.corpus, self.protocol, model="test",
            chat_json=lambda *a, **k: deepcopy(response), max_calls=4,
            scoring_policy=policy.POLICY_VERSION)
        rows = {}
        for system in ("one", "two"):
            rows[system] = []
            for i, q in enumerate(final):
                incorrect = system == "one" and i == 1
                response["answer_verdict"] = "incorrect" if incorrect else "correct"
                pred = "另一客户" if incorrect else "北溟保险"
                grade = judge(q, pred)
                row = {**q, "pred": pred, "correct": grade["correct"], "judgement": grade,
                    "mode": "semantic", "execution_status": "ok", "evaluation_scope": "research_only",
                    "evaluation_provenance": record_provenance(q, judge.public_context)}
                self.assertTrue(is_scored(row), grade)
                rows[system].append(row)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            artifacts = {"01_whitepaper.json": self.wp, "02_world.json": self.world.to_dict(),
                "04_questions.json": self.questions, "05_corpus.json": self.corpus,
                "06_grounded_questions.json": final, "06_semantic_review.json": review,
                "06_grounding_report.json": {"drops": [], "pending": []},
                "00_about.json": {"answer_protocol": self.fx.ANSWER_PROTOCOL}}
            for name, value in artifacts.items():
                (source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            bench = source / "06_grounded_questions.json"
            with self.assertRaises(ReleaseError):
                export_filtered_benchmark(bench, rows, source / "blocked")
            self.assertFalse((source / "blocked").exists())
            receipt = self.fx.evaluate_release(source)
            self.assertTrue(receipt["eligible"], receipt["issues"])
            (source / "07_release.json").write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
            exported = source / "filtered"
            report = export_filtered_benchmark(bench, rows, exported)
            self.assertEqual(report["counts"]["kept"], 1)
            for name in ("04_questions.json", "06_semantic_review.json"):
                self.assertEqual((source / name).read_bytes(), (exported / name).read_bytes())
            self.assertEqual(len(json.loads((exported / "06_semantic_review.json").read_text(encoding="utf-8"))["items"]), 2)
            issues = report["release"]["issues"]
            self.assertTrue(any(i["code"] == "unverified_source_derivation" for i in issues))
            self.assertFalse(any(i["code"] in {"semantic_selection_mismatch", "semantic_filter_derivation_mismatch"}
                                 for i in issues), issues)
            self.assertEqual(report["result_scope"], "research_only")
            manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
            manifest["derived_from"]["semantic_review"]["sha256"] = "0" * 64
            (exported / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            altered = self.fx.evaluate_release(exported)
            self.assertTrue(altered["eligible"], altered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
