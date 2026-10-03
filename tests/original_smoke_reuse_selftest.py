"""Offline identity checks for comparisons through the original factory stages."""
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-selftest")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("MODEL", "offline-model")
from pipeline.seed_pack import load_seed_pack, seed_contract, seed_digest, seed_input
from tools.run_original_bc_smoke import reuse_original_stages
from seed_contract_selftest import fixture as synthetic_seed


class ReuseTests(unittest.TestCase):
    def setUp(self):
        for method in ("connect", "connect_ex"):
            guard = patch.object(socket.socket, method, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.target = Path(self.temp.name) / "target"
        self.source.mkdir()
        pack = synthetic_seed()
        description, examples = seed_input(pack)
        digest = seed_digest(pack)
        self.write("00_seed_pack.json", pack)
        self.write("00_input.json", {"description": description, "few_shot": examples,
            "seed": {"seed_id": pack["seed_id"], "family": pack["family"], "digest": digest,
                     "artifact": "00_seed_pack.json", "schema_version": pack["schema_version"]}})
        self.write("01_whitepaper.json", {"seed_contract": seed_contract(pack)})
        for name in ("00_about.json", "01_seed_audit.json", "02_seed_audit.json", "02_world.json"):
            self.write(name, {})
        self.manifest = {"scenario": "seed_" + pack["seed_id"], "status": "done",
            "config": {"seed_id": pack["seed_id"], "seed_pack_digest": digest,
                       "seed_pack_path": "missing mutable source.json", "augment": True},
            "algo": {"seed": {"digest": digest}, "entities": 8, "quality": "stale"},
            "stages": {name: {"done": True, "artifact": artifact} for name, artifact in
                       [("input", "00_input.json"), ("whitepaper", "01_whitepaper.json"), ("world", "02_world.json")]}}
        self.write("manifest.json", self.manifest)

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_whitepaper_reuse_keeps_frozen_seed_and_excludes_old_world(self):
        result = reuse_original_stages(self.source, self.target, "whitepaper")
        for name in ("00_input.json", "00_seed_pack.json", "01_whitepaper.json", "01_seed_audit.json"):
            self.assertEqual((self.source / name).read_bytes(), (self.target / name).read_bytes())
        self.assertFalse((self.target / "02_world.json").exists())
        self.assertNotIn("entities", result["algo"])
        self.assertNotIn("quality", result["algo"])
        self.assertNotIn("augment", result["config"])
        self.assertEqual(set(result["stages"]), {"input", "whitepaper"})

    def test_world_reuse_keeps_seed_world_audit(self):
        result = reuse_original_stages(self.source, self.target, "world")
        self.assertTrue((self.target / "02_seed_audit.json").is_file())
        self.assertEqual(result["algo"]["entities"], 8)

    def test_changed_seed_identity_rejected_before_destination_created(self):
        pack = json.loads((self.source / "00_seed_pack.json").read_text(encoding="utf-8"))
        pack["title"] += " changed"
        self.write("00_seed_pack.json", pack)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "whitepaper")
        self.assertFalse(self.target.exists())

    def test_incomplete_or_active_source_rejected(self):
        self.manifest["status"] = "running"
        self.write("manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "world")
        self.assertFalse(self.target.exists())

    def test_office_reuse_remains_unseeded(self):
        self.manifest.update(scenario="bc_small_office", config={})
        self.write("manifest.json", self.manifest)
        self.write("01_whitepaper.json", {})
        (self.source / "00_seed_pack.json").unlink()
        result = reuse_original_stages(self.source, self.target, "world")
        self.assertEqual(result["scenario"], "bc_small_office")
        self.assertFalse((self.target / "00_seed_pack.json").exists())

    def question_source(self):
        for name, artifact in [("orders", "03_orders.json"),
                               ("well_posed", "03_well_posed_report.json"),
                               ("questions", "04_questions.json")]:
            self.write(artifact, {"frozen": name})
            self.manifest["stages"][name] = {"done": True, "artifact": artifact}
        self.write("04_wording_report.json", {"status": "passed"})
        self.manifest["stages"]["corpus"] = {"done": False, "status": "failed"}
        self.manifest["status"] = "failed"
        self.manifest["algo"].update(questions=4, docs=3, render_strategy="stale")
        self.write("manifest.json", self.manifest)

    def test_question_resume_preserves_checkpoint_bytes_and_provenance(self):
        self.question_source()
        self.write("05_corpus.ckpt.json", {"identity": "factory-validates-this",
            "done_weeks": [2, 3], "corpus": {"sessions": []}})
        result = reuse_original_stages(self.source, self.target, "questions")
        for name in ("04_questions.json", "04_wording_report.json", "05_corpus.ckpt.json"):
            self.assertEqual((self.source / name).read_bytes(), (self.target / name).read_bytes())
            self.assertIn(name, result["derived_from"]["input_files"])
        self.assertNotIn("corpus", result["stages"])
        self.assertEqual(result["algo"]["questions"], 4)
        self.assertNotIn("docs", result["algo"])
        self.assertNotIn("render_strategy", result["algo"])

    def test_question_resume_also_supports_no_checkpoint(self):
        self.question_source()
        reuse_original_stages(self.source, self.target, "questions")
        self.assertFalse((self.target / "05_corpus.ckpt.json").exists())

    def test_exact_corpus_resume_copies_review_checkpoints_with_receipt(self):
        self.question_source()
        self.write("05_corpus.json", {"corpus": {"sessions": []}})
        self.manifest["stages"]["corpus"] = {
            "done": True, "status": "succeeded", "artifact": "05_corpus.json"}
        self.manifest["status"] = "done"
        self.write("manifest.json", self.manifest)
        checkpoints = self.source / "06_review_checkpoints"
        checkpoints.mkdir()
        (checkpoints / "a.json").write_bytes(b'{"saved":1}')
        (checkpoints / "b.json").write_bytes(b'{"saved":2}')
        result = reuse_original_stages(
            self.source, self.target, "corpus", reuse_review_checkpoints=True)
        for name in ("a.json", "b.json"):
            self.assertEqual((checkpoints / name).read_bytes(),
                             (self.target / "06_review_checkpoints" / name).read_bytes())
        receipt = result["derived_from"]["semantic_review_checkpoints"]
        self.assertEqual(receipt["count"], 2)
        self.assertEqual(receipt["total_bytes"], 22)
        self.assertEqual(len(receipt["set_sha256"]), 64)

    def test_review_checkpoints_require_exact_corpus_and_existing_files(self):
        self.question_source()
        with self.assertRaisesRegex(ValueError, "exact corpus reuse"):
            reuse_original_stages(
                self.source, self.target, "questions", reuse_review_checkpoints=True)
        self.assertFalse(self.target.exists())

    def test_disclosure_recovery_accepts_stale_downstream_running_manifest(self):
        self.question_source()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        self.manifest.update(status="running", current_stage="grounding")
        self.write("manifest.json", self.manifest)
        result = reuse_original_stages(
            self.source, self.target, "questions", repair_disclosure=True)
        provenance = result["derived_from"]
        self.assertTrue(provenance["stale_downstream_run_accepted"])
        self.assertEqual(provenance["source_manifest_status"], "running")
        self.assertEqual(provenance["source_manifest_current_stage"], "grounding")
        self.assertEqual((self.source / "04_questions.json").read_bytes(),
                         (self.target / "04_questions.json").read_bytes())

    def test_disclosure_recovery_accepts_interrupted_disclosure_manifest(self):
        self.question_source()
        self.manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        self.manifest.update(status="running", current_stage="disclosure")
        self.write("manifest.json", self.manifest)
        result = reuse_original_stages(
            self.source, self.target, "questions", repair_disclosure=True)
        provenance = result["derived_from"]
        self.assertTrue(provenance["stale_downstream_run_accepted"])
        self.assertEqual(provenance["source_manifest_current_stage"], "disclosure")
        self.assertEqual((self.source / "04_questions.json").read_bytes(),
                         (self.target / "04_questions.json").read_bytes())

    def test_disclosure_recovery_refreshes_stale_review_instead_of_reusing_it(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "passed", "historical": True})
        with patch("pipeline.world_semantics.enabled", return_value=True), \
             patch("pipeline.world_semantics.validate_review",
                   return_value=[{"code": "missing_or_stale_world_review"}]):
            result = reuse_original_stages(
                self.source, self.target, "questions", repair_disclosure=True)
        self.assertEqual((self.target / "02_world_review.json").read_text(encoding="utf-8"),
                         (self.source / "02_world_review.json").read_text(encoding="utf-8"))
        self.assertEqual(result["derived_from"]["recovery"],
                         "complete_or_validate_disclosure_then_resume_downstream")

    def test_disclosure_recovery_drops_errored_review_checkpoint(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"bad": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertFalse((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_nonerror_review_checkpoint(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "unresolved"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_checkpoint_after_temporary_windows_lock(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error",
                   "error": "Original reviewer execution failed: [WinError 5] access denied"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_disclosure_recovery_keeps_checkpoint_after_repeated_review_revision(self):
        self.question_source()
        self.write("02_world_review.json", {"status": "error",
                   "error": "Four consecutive invalid world review actions: "
                            "Submitted review revision did not change the opinion"})
        self.write("02_world_review_deadbeef.ckpt.json", {"transcript": [{"saved": True}]})
        with patch("pipeline.world_semantics.enabled", return_value=False):
            reuse_original_stages(self.source, self.target, "questions", repair_disclosure=True)
        self.assertTrue((self.target / "02_world_review_deadbeef.ckpt.json").exists())

    def test_incomplete_questions_cannot_be_reused(self):
        self.question_source()
        self.manifest["stages"]["questions"]["done"] = False
        self.write("manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            reuse_original_stages(self.source, self.target, "questions")
        self.assertFalse(self.target.exists())

    def test_historical_world_checkpoint_recompiles_without_generation_calls(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        from pipeline.world_agent import _binding
        wp = json.loads((source / "01_whitepaper.json").read_text(encoding="utf-8"))
        checkpoint_state = next(json.loads(path.read_text(encoding="utf-8"))
                                for path in source.glob("02_world_agent_*.json")
                                if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed")
        if checkpoint_state["binding"]["version"] != _binding(wp, None)["version"]:
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                reuse_original_stages(source, self.target, "whitepaper",
                                      reuse_world_checkpoint=True, model="glm-5.3-flash")
            self.assertFalse(self.target.exists())
            return
        result = reuse_original_stages(source, self.target, "whitepaper",
                                      reuse_world_checkpoint=True, model="glm-5.3-flash")
        checkpoint = result["derived_from"]["world_construction_checkpoint"]
        self.assertEqual((source / checkpoint).read_bytes(), (self.target / checkpoint).read_bytes())
        self.assertTrue(result["derived_from"]["world_review_required"])
        self.assertFalse((self.target / "02_world.json").exists())
        from pipeline import world_agent, disclosure, closed_loop, run as run_module
        from pipeline.world_state import WorldState
        from pipeline.targetspec import TargetSpec
        captured = []
        class DisclosureReached(Exception):
            pass
        def denied(*args, **kwargs):
            self.fail("A completed construction checkpoint must not call any author")
        def at_disclosure(wp, ws, *args, **kwargs):
            captured.append(disclosure._world(ws))
            raise DisclosureReached()
        with patch.object(world_agent.config, "STRUCTURE_MODEL", "glm-5.3-flash"), \
             patch.object(run_module, "RUNS_DIR", self.target.parent), \
             patch.object(disclosure, "author_plan", side_effect=at_disclosure):
            run = run_module.Run(result["scenario"], self.target.name)
            run.log = lambda *_: None
            run.tracer.chat_json = denied
            with self.assertRaises(DisclosureReached):
                closed_loop.build_to_target(run, TargetSpec(min_questions=200, total_only=True,
                                            max_world_entities=80, time_span_weeks=24))
        old = WorldState.from_dict(json.loads((source / "02_world_candidate.json").read_text(encoding="utf-8")))
        self.assertEqual(captured, [disclosure._world(old)])

    def test_wrong_world_checkpoint_model_rejected_before_creating_destination(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            reuse_original_stages(source, self.target, "whitepaper",
                                  reuse_world_checkpoint=True, model="wrong-model")
        self.assertFalse(self.target.exists())

    def test_corrupt_world_checkpoint_rejected_and_never_promoted_to_reviewed_world(self):
        source = ROOT / "output/experiment_snapshots/insurance_large_20260919_v7/output/runs/insurance_large_20260919_v7_01_insurance_r001"
        if not source.exists():
            self.skipTest("Local historical run is optional")
        from pipeline.world_agent import _binding
        wp = json.loads((source / "01_whitepaper.json").read_text(encoding="utf-8"))
        checkpoint_state = next(json.loads(path.read_text(encoding="utf-8"))
                                for path in source.glob("02_world_agent_*.json")
                                if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed")
        if checkpoint_state["binding"]["version"] != _binding(wp, None)["version"]:
            self.skipTest("Historical checkpoint predates the current world-agent binding")
        result = reuse_original_stages(source, self.target, "whitepaper",
                                      reuse_world_checkpoint=True, model="glm-5.3-flash")
        checkpoint = self.target / result["derived_from"]["world_construction_checkpoint"]
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["steps"] += 1
        checkpoint.write_text(json.dumps(state), encoding="utf-8")
        other = Path(self.temp.name) / "rejected"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            reuse_original_stages(self.target, other, "whitepaper",
                                  reuse_world_checkpoint=True, model="glm-5.3-flash")
        self.assertFalse(other.exists())


class ProductionReuseTests(unittest.TestCase):
    """The primary V2 source is a completed real CLI run, not hand-made receipts."""

    @classmethod
    def setUpClass(cls):
        from generation_cli_e2e_selftest import ScriptedServices, offline_factory, create_production_run
        cls.production_temp = tempfile.TemporaryDirectory(prefix="reuse-v2-")
        cls.addClassCleanup(cls.production_temp.cleanup)
        root = Path(cls.production_temp.name)
        cls.services = ScriptedServices()
        with offline_factory(root, cls.services):
            run, _ = create_production_run(root, cls.services)
            cls.production_source = run.dir
            if len(run.read("06_grounded_questions.json")) != 4:
                raise AssertionError("The real factory fixture must produce four reviewed questions")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reuse-copy-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.target = self.root / "runs" / "derived"
        shutil.copytree(self.production_source, self.source)
        for method in ("connect", "connect_ex"):
            guard = patch.object(socket.socket, method, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def read(self, name):
        return json.loads((self.source / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def assert_rejected_without_target(self, through="questions"):
        with self.assertRaisesRegex(ValueError, "binding.*missing or stale"):
            reuse_original_stages(self.source, self.target, through)
        self.assertFalse(self.target.exists())

    def test_cli_artifacts_copy_and_grounding_resume_without_any_model_call(self):
        from generation_cli_e2e_selftest import ScriptedServices, offline_factory
        from pipeline import factory, run as run_module
        from contextlib import redirect_stdout
        import io
        before = {p.relative_to(self.source).as_posix(): p.read_bytes()
                  for p in self.source.rglob("*") if p.is_file()}
        manifest = reuse_original_stages(self.source, self.target, "questions", reuse_review_checkpoints=True)
        self.assertEqual(manifest["derived_from"]["resume_from"], "grounding")
        for name in ("05_material_plan.json", "05_material_binding.json", "04_authoring_binding.json"):
            self.assertFalse((self.source / name).exists(), name)
            self.assertFalse((self.target / name).exists(), name)
        services = ScriptedServices()
        with offline_factory(self.root, services), redirect_stdout(io.StringIO()), patch.object(sys, "argv", [
                "factory", "--run", "derived", "--from", "grounding", "--to", "quality"]):
            factory.main()
            run = run_module.Run(manifest["scenario"], "derived")
            self.assertEqual(len(run.read("06_grounded_questions.json")), 4)
        self.assertEqual(services.calls, [])
        self.assertEqual(before, {p.relative_to(self.source).as_posix(): p.read_bytes()
                                  for p in self.source.rglob("*") if p.is_file()})

    def test_corpus_prefix_does_not_copy_questions_and_resumes_at_questions(self):
        self.write("05_corpus_token_scale.json", {"marker": "frozen-token-receipt"})
        result = reuse_original_stages(self.source, self.target, "corpus")
        self.assertEqual(result["derived_from"]["resume_from"], "questions")
        self.assertTrue((self.target / "05_corpus.json").is_file())
        self.assertEqual((self.target / "05_corpus_token_scale.json").read_bytes(),
                         (self.source / "05_corpus_token_scale.json").read_bytes())
        self.assertFalse((self.target / "04_questions.json").exists())
        self.assertNotIn("questions", result["stages"])

    def test_corpus_prefix_cannot_reuse_question_review_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "exact corpus reuse"):
            reuse_original_stages(self.source, self.target, "corpus", reuse_review_checkpoints=True)
        self.assertFalse(self.target.exists())

    def test_real_corpus_receipt_rejects_changed_public_text_before_copy(self):
        corpus = self.read("05_corpus.json")
        body = corpus.get("corpus", corpus)
        doc = next(doc for session in body["sessions"] for doc in session["docs"])
        doc["content"] += " Changed without the original review."
        self.write("05_corpus.json", corpus)
        self.assert_rejected_without_target("corpus")

    def test_real_question_authoring_missing_rejected_before_copy(self):
        questions = self.read("04_questions.json")
        self.assertIn("authoring", questions[0]["question_validation"])
        questions[0]["question_validation"].pop("authoring")
        self.write("04_questions.json", questions)
        self.assert_rejected_without_target()

    def test_real_question_contract_changed_rejected_before_copy(self):
        questions = self.read("04_questions.json")
        questions[0]["answer"] = "tampered answer"
        self.write("04_questions.json", questions)
        self.assert_rejected_without_target()

    def test_real_wording_report_binding_missing_rejected_before_copy(self):
        report = self.read("04_wording_report.json")
        self.assertIn("candidate_binding", report)
        report.pop("candidate_binding")
        self.write("04_wording_report.json", report)
        self.assert_rejected_without_target()


class BoundedContinuationTests(unittest.TestCase):
    """The actual wrapper and factory retain one ledger across checkpoint stops."""

    def test_source_snapshot_covers_shared_execution_and_diversity_modules(self):
        from tools import run_original_bc_smoke as smoke
        snapshot = smoke.implementation_hashes()
        self.assertIn("execution_control.py", snapshot)
        self.assertIn("tools/diversity_metrics.py", snapshot)

    def test_resume_requires_complete_matching_prior_source_snapshot(self):
        from tools import run_original_bc_smoke as smoke
        with tempfile.TemporaryDirectory(prefix="source-epoch-") as temp:
            root = Path(temp)
            current = {"pipeline/a.py": "a" * 64}
            with self.assertRaises(ValueError):
                smoke.verify_resume_source(root, current)
            (root / "experiment_source_hashes.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke.verify_resume_source(root, current)
            end = root / "experiment_source_hashes_at_end.json"
            end.write_text('{"pipeline/a.py":"bad"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke.verify_resume_source(root, current)
            end.write_text(json.dumps(current), encoding="utf-8")
            self.assertEqual(smoke.verify_resume_source(root, current), current)
            with self.assertRaises(ValueError):
                smoke.verify_resume_source(root, {"pipeline/a.py": "b" * 64})

    def test_corpus_stop_resume_quality_and_terminal_resume_keep_original_budget(self):
        self.run_canary()

    def test_optional_planner_json_failure_continues_body_without_replanning(self):
        self.run_canary(fault="json")

    def test_optional_planner_local_deadline_continues_body_without_replanning(self):
        self.run_canary(fault="local_deadline")

    def test_optional_planner_401_keeps_global_stop(self):
        self.run_canary(fault="auth")

    def test_changed_source_cannot_silently_resume_cumulative_run(self):
        self.run_canary(fault="source_drift")

    def test_filler_transport_retry_completes_real_pipeline_and_zero_call_resume(self):
        self.run_canary(fault="text_connection")

    def run_canary(self, fault=None):
        from contextlib import ExitStack, redirect_stdout
        from functools import wraps
        import io
        from types import SimpleNamespace
        from generation_cli_e2e_selftest import ScriptedServices
        from pipeline import factory, run as runtime
        from tools import run_original_bc_smoke as smoke
        import config
        services = ScriptedServices()
        snapshot = smoke.implementation_hashes()
        provider_counter = 0
        planner_requests = []
        failed_input = None
        filler_failed = False

        class Unauthorized(RuntimeError):
            status_code = 401

        class LocalDeadline(TimeoutError):
            kind = "total_deadline"
            local_cleanup_confirmed = True

        @wraps(config.chat)
        def scripted_provider(messages, *args, **kwargs):
            nonlocal provider_counter, failed_input, filler_failed
            provider_counter += 1
            this_call = provider_counter
            step = smoke._TRACER_STEP.get()
            if step == "render.plan":
                signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                planner_requests.append(signature)
                if fault in {"json", "local_deadline", "auth"} and failed_input is None:
                    failed_input = signature
                    if fault == "auth":
                        raise Unauthorized("Offline injected planner 401")
                    if fault == "local_deadline":
                        raise LocalDeadline("Offline injected locally-cleaned planner deadline")
                    config.emit("response", response={"usage": {"prompt_tokens": 100, "completion_tokens": 20}})
                    return '{"documents":'
            text_call = not smoke._JSON_REQUEST_ACTIVE.get()
            if text_call and fault == "text_connection" and not filler_failed:
                filler_failed = True
                failure = RuntimeError("Offline injected filler connection error")
                failure.kind = "http_connect_error"
                failure.retryable = True
                raise failure
            output = (services.text(SimpleNamespace(n=0), step, messages, **kwargs) if text_call else
                      services.json(SimpleNamespace(n=0), step, messages, **kwargs))
            # Preserve one real unknown-usage liability across both resumes.
            usage = None if this_call == 1 else {"prompt_tokens": 100, "completion_tokens": 20}
            config.emit("response", response={"usage": usage})
            return output if text_call else json.dumps(output, ensure_ascii=False)

        with tempfile.TemporaryDirectory(prefix="bounded-v2-") as temp, ExitStack() as stack:
            root = Path(temp)
            seed = root / "seed.json"
            seed.write_text(json.dumps(services.pack, ensure_ascii=False), encoding="utf-8")
            for method in ("connect", "connect_ex"):
                stack.enter_context(patch.object(socket.socket, method, side_effect=AssertionError("Network forbidden")))
            for role in ("MODEL", "STRUCTURE_MODEL", "REVIEWER_MODEL", "DISCRIMINATOR_MODEL", "JUDGE_MODEL",
                         "LLM_CONCURRENCY", "_LLM_SEM"):
                stack.enter_context(patch.object(config, role, getattr(config, role)))
            stack.enter_context(patch.object(smoke, "ROOT", root))
            stack.enter_context(patch.object(smoke, "implementation_hashes", return_value=snapshot))
            for target in (runtime, factory):
                stack.enter_context(patch.object(target, "RUNS_DIR", root / "output/runs"))
            stack.enter_context(patch.object(config, "chat", scripted_provider))
            initial = ["smoke", "--run", "canary", "--seed-pack", str(seed), "--to", "corpus",
                "--model", "glm-5.3-flash", "--question-budget", "4", "--time-span-weeks", "6",
                "--max-world-entities", "8", "--target-mtokens", "0", "--max-calls", "500",
                "--max-cny", "4", "--settle-reported-usage"]
            profile_path = root / "output/runs/canary/experiment_profile.json"
            if fault == "text_connection":
                initial[initial.index("--target-mtokens") + 1] = "0.000001"
            initial_error = None
            with patch.object(sys, "argv", initial), redirect_stdout(io.StringIO()):
                try:
                    smoke.main()
                except (Exception, SystemExit) as exc:
                    initial_error = exc
            first = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertGreater(first["admitted_calls"], 0)
            self.assertEqual(first["admitted_calls"], provider_counter)
            if fault in {"json", "local_deadline", "auth"}:
                self.assertIsNotNone(failed_input)
                self.assertEqual(planner_requests.count(failed_input), 1)
            if fault == "auth":
                self.assertTrue(first["stopped"])
                calls_before_resume = provider_counter
                profile_before_resume = profile_path.read_bytes()
                with patch.object(sys, "argv", ["smoke", "--run", "canary", "--resume-existing", "--to", "quality"]), \
                     redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                    try:
                        smoke.main()
                    except (Exception, SystemExit):
                        pass
                self.assertEqual(provider_counter, calls_before_resume)
                self.assertEqual(profile_path.read_bytes(), profile_before_resume)
                self.assertTrue(json.loads(profile_path.read_text(encoding="utf-8"))["stopped"])
                return
            self.assertIsNone(initial_error, repr(initial_error))
            if fault == "source_drift":
                source_path = profile_path.parent / "experiment_source_hashes.json"
                manifest_path = profile_path.parent / "manifest.json"
                before = {path: path.read_bytes() for path in
                          (profile_path, source_path, manifest_path)}
                calls_before_resume = provider_counter
                drifted_profile = json.loads(profile_path.read_text(encoding="utf-8"))
                drifted_profile["source_drift"] = ["config.py"]
                profile_path.write_text(json.dumps(drifted_profile), encoding="utf-8")
                drifted_bytes = profile_path.read_bytes()
                with patch.object(sys, "argv", ["smoke", "--run", "canary",
                                                "--resume-existing", "--to", "quality"]), \
                     redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        smoke.main()
                self.assertEqual(profile_path.read_bytes(), drifted_bytes)
                profile_path.write_bytes(before[profile_path])
                snapshot["config.py"] = "0" * 64
                with patch.object(sys, "argv", ["smoke", "--run", "canary",
                                                "--resume-existing", "--to", "quality"]), \
                     redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        smoke.main()
                self.assertEqual(provider_counter, calls_before_resume)
                self.assertEqual({path: path.read_bytes() for path in before}, before)
                self.assertFalse(list(profile_path.parent.glob("experiment_profile_before_resume_*.json")))
                return
            unknown_calls = 2 if fault in {"local_deadline", "text_connection"} else 1
            if fault == "text_connection":
                self.assertTrue(filler_failed, (json.loads((profile_path.parent / "manifest.json").read_text(encoding="utf-8"))["config"],
                                               [step for step, _ in services.calls]))
                self.assertTrue(any(step == "render.filler" for step, _ in services.calls))
            self.assertEqual(first["unsettled_reservation_calls"], unknown_calls)
            self.assertFalse(first["stopped"])
            self.assertTrue(any(step == "render.signal" for step, _ in services.calls))
            self.assertIn("--time-span-weeks", first["quantity_arguments"])
            world_calls = sum(step == "world.agent.write" for step, _ in services.calls)
            self.assertEqual(world_calls, 1)
            resume = ["smoke", "--run", "canary", "--resume-existing", "--to", "quality"]
            with patch.object(sys, "argv", resume), redirect_stdout(io.StringIO()):
                smoke.main()
            second = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertGreater(second["admitted_calls"], first["admitted_calls"])
            self.assertEqual(second["admitted_calls"], provider_counter)
            self.assertGreaterEqual(second["budget_consumed_cny"], first["budget_consumed_cny"])
            self.assertEqual(second["unsettled_reservation_calls"], unknown_calls)
            for name in ("created_utc", "model", "max_calls", "max_cny", "transport", "settle_reported_usage"):
                self.assertEqual(first[name], second[name], name)
            with patch.object(sys, "argv", resume), redirect_stdout(io.StringIO()):
                smoke.main()
            third = json.loads(profile_path.read_text(encoding="utf-8"))
            for name in ("admitted_calls", "budget_consumed_cny", "reported_usage_estimate_cny",
                         "reserved_upper_estimate_cny", "settled_usage_calls", "unsettled_reservation_calls"):
                self.assertEqual(third[name], second[name], name)
            self.assertEqual(third["admitted_calls"], provider_counter)
            if failed_input is not None:
                self.assertEqual(planner_requests.count(failed_input), 1)
            self.assertEqual(sum(step == "world.agent.write" for step, _ in services.calls), world_calls)
            run_log = (profile_path.parent / "run.log").read_text(encoding="utf-8")
            for stage in ("world", "corpus", "questions", "grounding"):
                self.assertEqual(run_log.count("→ stage: " + stage + "\n"), 1, stage)
            before = profile_path.read_bytes()
            with patch.object(sys, "argv", resume + ["--max-cny", "99"]), \
                 redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                smoke.main()
            self.assertEqual(profile_path.read_bytes(), before)
            # A saved hard stop cannot be cleared by the resume entry.
            stopped = {**third, "stopped": True}
            profile_path.write_text(json.dumps(stopped), encoding="utf-8")
            stopped_bytes = profile_path.read_bytes()
            with patch.object(sys, "argv", resume), redirect_stdout(io.StringIO()), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                smoke.main()
            self.assertEqual(profile_path.read_bytes(), stopped_bytes)
            self.assertEqual(provider_counter, third["admitted_calls"])


if __name__ == "__main__":
    unittest.main()
