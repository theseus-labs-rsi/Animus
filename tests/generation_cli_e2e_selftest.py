"""Offline fresh-seed CLI integration, with scripted service-boundary opinions.

This exercises the actual factory stages and four-athlete selection/export. It
does not measure model quality, provider latency, or native CLI installation.
Run from a clean checkout: python -B -X utf8 tests/generation_cli_e2e_selftest.py
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
# Importing config must never require a developer's private .env or credentials.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-selftest")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("MODEL", "offline-model")

import config
from pipeline import factory, run as run_module, calibration, native_evaluation
from pipeline import disclosure, world_semantics
from agent_harnesses.artifacts import (inspect_benchmark, load_benchmark_questions,
    load_benchmark_references, load_benchmark_protocol)
from agent_harnesses.config import BenchmarkRef
from seed_world_selftest import fixture as world_fixture
from seed_contract_selftest import CouncilTracer
from world_agent_selftest import ScriptedTracer, PLAN, WRITE, FINISH, write, table_part
from world_scoped_agents_selftest import ScopedAgentFixture
from disclosure_batches_selftest import DirectFixture
from corpus_fixture_helpers import fixed_positive_review
from semantic_review_selftest import blind, adjudication
from semantic_review_fixture_helpers import audit_output


class ScriptedServices:
    """Fixed synthetic author/reviewer opinions, never semantic acceptance proof."""
    def __init__(self, *, construction=False):
        wp, self.world, table = world_fixture()
        self.pack = deepcopy(wp["seed_contract"])
        self.pack.pop("digest", None)
        self.pack.pop("contract_digest", None)
        self.pack["seed_id"] = "cli"
        self.council = CouncilTracer(wp["world_blueprint"])
        self.writer = ScriptedTracer([(PLAN, write("whole_fixture")),
            (WRITE, table_part(table, [0, 1], True)), (PLAN, FINISH)])
        self.disclosure, self.reviewer = DirectFixture(), ScopedAgentFixture()
        self.calls = []
        self.athlete_calls = []
        self.judge_calls = 0
        self.settings = calibration.load_config(ROOT / "examples/release_four.json")
        self.instance_plan = None
        if construction:
            self._construction_fixture(wp)

    def _construction_fixture(self, wp):
        """Four distinct synthetic report carriers for the current v5 entry."""
        from instance_plan_selftest import setup_plan
        from pipeline.world_state import assemble_world
        _, original, _, full, _ = setup_plan()
        blueprint = deepcopy(wp["world_blueprint"])
        blueprint["entity_types"][1]["count"] = 4
        self.pack["blueprint_requirements"]["entity_types"][1]["count"] = 4
        self.pack["seed_id"] = "cli-construction"
        self.council = CouncilTracer(blueprint)
        source = original["units"][0]
        unit = deepcopy(source)
        unit.update(objects=[deepcopy(source["objects"][0])], relations=[], events=[],
                    observations=[], obligations=[], publications=[])
        expanded = {"entities": [deepcopy(full["entities"][0])], "relations": [],
                    "events": [], "initial_states": [], "slot_mapping": {"company-slot": "Aster"}}
        def rename(value, mapping):
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, list):
                return [rename(item, mapping) for item in value]
            if isinstance(value, dict):
                return {key: rename(item, mapping) for key, item in value.items()}
            return value
        for index in range(4):
            mapping = {slot: f"{slot}-{index}" for slot in full["slot_mapping"] if slot != "company-slot"}
            actual = {value: f"{value}-{index}" for slot, value in full["slot_mapping"].items()
                      if slot != "company-slot"}
            unit["objects"].append(rename(source["objects"][1], mapping))
            for key in ("relations", "events", "observations", "publications"):
                unit[key].extend(rename(source[key], mapping))
            unit["obligations"].append({"id": f"timeline-{index}", "line": "L1_timeline",
                "carrier": {"entities": [mapping["report-slot"]], "field": "profit"}})
            expanded["entities"].append(rename(full["entities"][1], actual))
            for key in ("relations", "events", "initial_states"):
                expanded[key].extend(rename(full[key], actual))
            expanded["slot_mapping"].update({mapping[slot]: actual[value]
                for slot, value in full["slot_mapping"].items() if slot != "company-slot"})
        self.instance_plan = {"decision": "plan", "reason": "Four independent synthetic report histories",
                              "units": [unit], "cohorts": []}
        self.instance_values = expanded
        self.world, issues = assemble_world(expanded, blueprint=blueprint, include_shape_diagnostics=False)
        if issues:
            raise AssertionError(issues)

    def _author_instance_values(self, assigned):
        full = self.instance_values
        actual = full["slot_mapping"]
        entities = {entity["name"]: entity for entity in full["entities"]}
        events = {event["id"]: event for event in full["events"]}
        return {"objects": {obj["slot"]: {"name": actual[obj["slot"]],
            "fields": deepcopy(entities[actual[obj["slot"]]]["fields"]),
            "initial_values": {row["field"]: row["value"] for row in full["initial_states"]
                               if row["entity"] == actual[obj["slot"]]}} for obj in assigned["objects"]},
            "event_values": {event["slot"]: {role: {field["name"]: next(effect["set"]
                for effect in events[actual[event["slot"]]]["effects"]
                if effect["entity"] == actual[event["participants"][role]]
                and effect["field"] == field["name"]) for field in fields}
                for role, fields in event["value_fields"].items()} for event in assigned["events"]}}

    def json(self, tracer, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages)))
        tracer.n += 1
        if step == "council.instance_plan" and self.instance_plan is not None:
            return deepcopy(self.instance_plan)
        if step == "world.instance.values" and self.instance_plan is not None:
            return self._author_instance_values(json.loads(messages[-1]["content"])["task"])
        if step.startswith("council."):
            return self.council.chat_json(step, messages, **kwargs)
        if step in (PLAN, WRITE):
            return self.writer.chat_json(step, messages, **kwargs)
        if step == disclosure.STEP:
            return self.disclosure.chat_json(step, messages, **kwargs)
        if step == world_semantics.STEP:
            return self.reviewer.chat_json(step, messages, **kwargs)
        if step == "render.plan":
            return {"documents": [{"purpose": "Record the supplied public facts.",
                "sources": ["fact:0"], "time_context": "Preserve original and public dates."}]}
        if step == "render.signal":
            # The fixture author renders the provided facts, never stage outputs.
            text = messages[-1]["content"]
            decoder = json.JSONDecoder()
            arrays = []
            for i, char in enumerate(text):
                if char == "[":
                    try:
                        value, _ = decoder.raw_decode(text[i:])
                    except ValueError:
                        continue
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        arrays.append(value)
            facts = next((xs for xs in arrays if "entity" in xs[0] and "field" in xs[0]), [])
            body = "\n".join(f"{f['entity']} {f['field']} at session {f.get('fact_session', 0)}: {f.get('value')}." for f in facts)
            return {"docs": [{"title": "Synthetic source record", "content": body or "Synthetic public operational record."}]}
        if step == "render.discriminate":
            text = messages[-1]["content"]
            start = text.index('[{"key"')
            queries, _ = json.JSONDecoder().raw_decode(text[start:])
            return {"answers": [{"key": q["key"], "answer": str(self.world.entities[
                q["entity"]][q["field"]].value_at_session(q.get("fact_session", 5)))} for q in queries]}
        if step == "corpus.review":
            return fixed_positive_review(messages)
        payload = json.loads(messages[-1]["content"])
        if step == "phrase":
            docs = payload["material_context"]["documents"]
            return {"question": payload["question_contract"]["canonical_question"],
                "material_support": {"status": "sufficient", "gap_kind": "none", "gaps": [],
                    "reason": "Fixed offline author opinion.", "evidence": ([{"doc_id": docs[0]["doc_id"],
                        "quote": docs[0]["content"][:80]}] if docs else [])}}
        if step == "phrase.review":
            return {"verdict": "equivalent", "reason": "Fixed offline wording opinion.", "issues": []}
        docs = payload.get("documents", [])
        evidence = ([{"doc_id": docs[0]["doc_id"], "field": "content", "location_scope": "field",
                      "role": "support", "explanation": "Fixed fixture source opinion."}] if docs else [])
        coverage = {"status": "complete", "scope_conflict": False,
                    "inspected_doc_ids": [d["doc_id"] for d in docs], "limitations": []}
        if step == "semantic_review.blind_read":
            return blind(answer="150", coverage=coverage, evidence=evidence)
        if step == "agent_editing.independent_reference_audit":
            out = audit_output(payload["original_reference"])
            out["claims"] = [{"reference_part": "answer", "location_scope": "node", "json_pointer": "",
                              "assessment": "Fixture support", "explanation": "Fixed offline opinion.", "evidence": evidence}]
            out["evidence"] = evidence
            return out
        if step == "semantic_review.adjudicate":
            reference = payload["reference_proposal"]
            out = adjudication(reference_status="supported", reviewed_answer=reference["answer"],
                concerns=[], coverage=coverage, evidence=evidence,
                review_findings={"substantive_defects": [], "acceptable_brevity": [], "editorial_suggestions": []})
            out["original_answer_review"] = [{"requirement": "Original fixture task", "assessment": "Supported",
                "explanation": "Fixed offline opinion.", "target_scope": "quoted_text", "reference_targets": [
                    {"reference_part": "answer", "location_scope": "node", "json_pointer": ""}]}]
            out["reference_audit_response"] = "Fixed fixture response to the independent audit."
            return out
        raise AssertionError("Unscripted model boundary: " + step)

    def text(self, tracer, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages)))
        tracer.n += 1
        if step != "render.filler":
            raise AssertionError("Unscripted text boundary: " + step)
        return "外围资料保管部门完成普通物品盘点，登记书架维护和会议室使用事项。"

    @staticmethod
    def preflight(system, model):
        return {"ok": True, "errors": [], "binary": "offline-cli", "binary_version": "fixture",
                "adapter": system.implementation["adapter"], "model": model["model_id"],
                "protocol_style": model["protocol_style"], "runtime_options": {}}

    def native_cli(self, command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="offline-fixture\n")
        plan = json.loads(Path(command[command.index("--plan") + 1]).read_text(encoding="utf-8"))
        benchmark = Path(plan["benchmark"]["path"])
        references = load_benchmark_references(benchmark)
        path = Path(plan["output_dir"]) / "results.jsonl"
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()] if path.exists() else []
        done = {row["question_id"] for row in existing if row["status"] == "completed"}
        with path.open("a", encoding="utf-8") as stream:
            for index, q in enumerate(load_benchmark_questions(benchmark)):
                if q["qid"] in done:
                    continue
                self.athlete_calls.append((plan["target_id"], q["qid"]))
                answer = references[q["qid"]]["answer"]
                if index == 0 and plan["target_id"] == self.settings["systems"][-1]:
                    answer = "INTENTIONALLY_INCORRECT_OFFLINE_FIXTURE"
                if not isinstance(answer, str):
                    answer = json.dumps(answer, ensure_ascii=False)
                stream.write(json.dumps({"question_index": index, "question_id": q["qid"], "answer": answer,
                    "status": "completed", "error_type": None, "returncode": 0}, ensure_ascii=False) + "\n")
        return SimpleNamespace(returncode=0)

    def judge(self, messages, **kwargs):
        self.judge_calls += 1
        wrong = "INTENTIONALLY_INCORRECT_OFFLINE_FIXTURE" in json.dumps(messages)
        return json.dumps({"correct": not wrong, "reason": "Fixed offline scoring opinion."})


@contextmanager
def offline_factory(root, services):
    """Real factory with only model/CLI services replaced; reusable by recovery tests."""
    with ExitStack() as stack:
        runs = Path(root) / "runs"
        for target in (run_module, factory):
            stack.enter_context(patch.object(target, "RUNS_DIR", runs))
        for method in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, method,
                side_effect=AssertionError("Network forbidden")))
        stack.enter_context(patch.object(run_module.Tracer, "chat_json", autospec=True,
                                        side_effect=services.json))
        stack.enter_context(patch.object(run_module.Tracer, "chat_text", autospec=True,
                                        side_effect=services.text))
        stack.enter_context(patch("agent_harnesses.tracks.native.preflight_system", side_effect=services.preflight))
        stack.enter_context(patch("agent_harnesses.runners.native_cli.preflight_system", side_effect=services.preflight))
        stack.enter_context(patch("agent_harnesses.execution.subprocess.run", side_effect=services.native_cli))
        stack.enter_context(patch("config.chat", side_effect=services.judge))
        # A temporary storage root isolates native fixtures from user caches;
        # production still decides target paths, identities and resume behavior.
        if hasattr(native_evaluation, "_short_native_root"):
            stack.enter_context(patch.object(native_evaluation, "_short_native_root",
                                            return_value=Path(root) / "n"))
        yield runs


def create_production_run(root, services, *, release=False, name="cli"):
    """Produce every stage artifact via the public CLI, never a receipt fixture."""
    seed = Path(root) / "seed.json"
    seed.write_text(json.dumps(services.pack, ensure_ascii=False), encoding="utf-8")
    argv = ["factory", "--seed-pack", str(seed), "--run", name,
            "--question-budget", "4", "--target-mchars", "0", "--haystack-ratio", "0",
            "--semantic-workers", "1", *( ["--release"] if release else ["--to", "quality"])]
    capture = io.StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(capture):
        try:
            factory.main()
        except SystemExit as exc:
            raise AssertionError(f"Real factory CLI aborted: {exc}\n{capture.getvalue()}") from exc
    return run_module.Run("seed_" + services.pack["seed_id"], name), argv


class FreshSeedCliTest(unittest.TestCase):
    def test_fresh_seed_generation_four_athletes_selection_and_standard_reload(self):
        self.run_fresh_cli(long_names=False)

    def test_long_seed_default_run_name_completes_the_same_public_cli(self):
        self.run_fresh_cli(long_names=True)

    def run_fresh_cli(self, *, long_names):
        services = ScriptedServices()
        if long_names:
            services.pack["seed_id"] = "legal_" + "revision_" * 5
        with tempfile.TemporaryDirectory(prefix="gcli-") as temp, ExitStack() as stack:
            root = Path(temp)
            # The owned temporary root can contain >260-character Windows
            # checkpoint paths; clean it through the same filesystem adapter.
            resolved = root.resolve()
            if resolved.parent != Path(tempfile.gettempdir()).resolve() or not resolved.name.startswith("gcli-"):
                raise AssertionError("Unexpected temporary test directory")
            stack.callback(shutil.rmtree, run_module._io_path(resolved))
            seed = root / "seed.json"
            seed.write_text(json.dumps(services.pack, ensure_ascii=False), encoding="utf-8")
            runs = root / "checkout" / "output" / "runs" if long_names else root / "runs"
            for target in (run_module, factory):
                stack.enter_context(patch.object(target, "RUNS_DIR", runs))
            for method in ("connect", "connect_ex"):
                stack.enter_context(patch.object(socket.socket, method, side_effect=AssertionError("Network forbidden")))
            stack.enter_context(patch.object(run_module.Tracer, "chat_json", autospec=True, side_effect=services.json))
            stack.enter_context(patch.object(run_module.Tracer, "chat_text", autospec=True, side_effect=services.text))
            stack.enter_context(patch("agent_harnesses.tracks.native.preflight_system", side_effect=services.preflight))
            stack.enter_context(patch("agent_harnesses.runners.native_cli.preflight_system", side_effect=services.preflight))
            if hasattr(native_evaluation, "_short_native_root"):
                stack.enter_context(patch.object(native_evaluation, "_short_native_root", return_value=root / "n"))
            stack.enter_context(patch("agent_harnesses.execution.subprocess.run", side_effect=services.native_cli))
            stack.enter_context(patch("config.chat", side_effect=services.judge))
            argv = ["factory", "--seed-pack", str(seed), "--question-budget", "4", "--target-mchars", "0",
                    "--haystack-ratio", "0", "--semantic-workers", "1", "--release"]
            if not long_names:
                argv += ["--run", "cli"]
            stack.enter_context(patch.object(sys, "argv", argv))
            capture = io.StringIO()
            with redirect_stdout(capture):
                try:
                    factory.main()
                except BaseException:
                    print(capture.getvalue(), file=sys.stderr)
                    raise
            directories = list(runs.iterdir())
            self.assertEqual(len(directories), 1)
            run = run_module.Run("seed_" + services.pack["seed_id"], directories[0].name)
            report = run.read(calibration.SELECTION_ARTIFACT)
            self.assertTrue(report["release_ready"], (report, [{key: item.get(key) for key in
                ("execution", "stage_execution", "item_certification")} for item in run.read("06_semantic_review.json")["items"]],
                                                      [step for step, _ in services.calls], capture.getvalue()))
            self.assertGreater(report["counts"]["removed_easy"], 0)
            self.assertEqual(report["counts"]["kept"], 1)
            package = run.dir / report["benchmark"]
            inspection = inspect_benchmark(BenchmarkRef(package, False, "as_provided"))
            self.assertEqual(inspection.n_questions, 1)
            self.assertGreater(inspection.n_docs, 0)
            self.assertTrue(load_benchmark_protocol(package).strip())
            self.assertEqual(set(load_benchmark_references(package)), {q["qid"] for q in load_benchmark_questions(package)})
            stage_names = [s.name for s in factory.generation_stages(run)]
            self.assertLess(stage_names.index("corpus"), stage_names.index("questions"))
            self.assertTrue(all(run.manifest["stages"][s]["done"] for s in stage_names))
            steps = [step for step, _ in services.calls]
            self.assertLess(steps.index("render.signal"), steps.index("phrase"))
            for step in (PLAN, WRITE, "corpus.review", "phrase.review",
                         "semantic_review.blind_read", "agent_editing.independent_reference_audit", "semantic_review.adjudicate"):
                self.assertIn(step, steps)
            self.assertEqual(len({target for target, _ in services.athlete_calls}), 4)
            self.assertEqual(len(services.athlete_calls), 4 * len(run.read("06_grounded_questions.json")))
            self.assertTrue(run.has("00_seed_pack.json"))
            self.assertFalse(run.has("05_material_plan.json"))
            self.assertEqual(report["counts"]["input"], 4)
            self.assertEqual(report["counts"]["removed_easy"], 3)
            author_payloads = [json.loads(messages[-1]["content"]) for step, messages in services.calls
                               if step == "phrase"]
            exported_protocol = load_benchmark_protocol(package)
            for payload in author_payloads:
                self.assertEqual(payload["public_protocol"], exported_protocol)
                self.assertTrue(payload["question_contract"])
                self.assertTrue(payload["material_context"]["documents"])
            before = (len(services.calls), len(services.athlete_calls), services.judge_calls)
            # The same public CLI resumes this completed run without regenerating
            # material, rescoring athletes, or losing the selected package.
            with redirect_stdout(capture):
                with patch.object(sys, "argv", [*argv, "--run", run.run_id] if long_names else argv):
                    factory.main()
            self.assertEqual((len(services.calls), len(services.athlete_calls), services.judge_calls), before)
            self.assertEqual(run.read(calibration.SELECTION_ARTIFACT)["benchmark"], report["benchmark"])


class PublicInputInvalidationTests(unittest.TestCase):
    def run_mutation(self, *, public_change):
        from pipeline import corpus_contract
        from pipeline.world_state import WorldState
        from pipeline.benchmark_export import public_view
        services = ScriptedServices()
        with tempfile.TemporaryDirectory(prefix="input-change-") as temp, offline_factory(temp, services):
            run, argv = create_production_run(temp, services, release=True)
            before_public = public_view(run)
            before_questions = run.read("04_questions.json")
            before_calls = [step for step, _ in services.calls]
            before_athletes = len(services.athlete_calls)
            before_judges = services.judge_calls
            old_review = run.read("06_semantic_review.json")
            corpus = run.read("05_corpus.json")
            body = corpus.get("corpus", corpus)
            if public_change:
                session = next(session for session in reversed(body["sessions"]) if session["docs"])
                docs = [{"doc_id": "counterevidence-offline", "title": "A later disagreement",
                    "content": "A public notice disputes numeric figures in earlier reports; an alternative reported value is 999.",
                    "date": session["date"], "type": "record", "role": "signal"}]
                # The real corpus-review entry writes receipts from scripted
                # service opinions. This case tests invalidation, not semantics.
                ws = WorldState.from_dict(run.read("02_world.json"))
                report = corpus_contract.review_documents(run.tracer, ws, session["session_id"], docs)
                self.assertEqual(report["status"], "passed", report)
                corpus_contract.attach_receipts(docs, report, session["session_id"])
                session["docs"].extend(docs)
            else:
                body["internal_note"] = "Fixture bookkeeping; absent from public projection."
                for session in body["sessions"]:
                    for doc in session["docs"]:
                        doc["internal_note"] = "Auditor queue metadata only."
            run.write("05_corpus.json", corpus)
            if public_change:
                self.assertNotEqual(before_public, public_view(run))
                self.assertFalse(factory._grounding_is_current(run))
                self.assertFalse(calibration.calibration_is_current(run))
            else:
                self.assertEqual(before_public, public_view(run))
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                factory.main()
            after_calls = [step for step, _ in services.calls]
            self.assertEqual(after_calls.count("phrase"), before_calls.count("phrase"))
            self.assertEqual(run.read("04_questions.json"), before_questions)
            self.assertEqual(after_calls.count("render.signal"), before_calls.count("render.signal"))
            if public_change:
                for step in ("semantic_review.blind_read", "agent_editing.independent_reference_audit",
                             "semantic_review.adjudicate"):
                    self.assertEqual(after_calls.count(step) - before_calls.count(step), 4, step)
                self.assertEqual(len(services.athlete_calls) - before_athletes, 16)
                self.assertNotEqual(run.read("06_semantic_review.json"), old_review)
                self.assertTrue(any("counterevidence-offline" not in
                    item.get("document_scope", {}).get("source_doc_ids", [])
                    for item in run.read("06_semantic_review.json")["items"]))
            else:
                self.assertEqual(after_calls, before_calls)
                self.assertEqual(len(services.athlete_calls), before_athletes)
                self.assertEqual(services.judge_calls, before_judges)
            self.assertTrue(run.read(calibration.SELECTION_ARTIFACT)["release_ready"])

    def test_internal_metadata_keeps_all_completed_calls_and_evaluations(self):
        self.run_mutation(public_change=False)

    def test_public_counterevidence_rechecks_semantics_and_athletes_without_reauthoring(self):
        self.run_mutation(public_change=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
