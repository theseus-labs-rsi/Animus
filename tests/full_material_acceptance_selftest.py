"""Offline fixtures: exact reader transport/acceptance, not real model quality."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-dummy-credential-only",
                  OPENAI_BASE_URL="http://127.0.0.1:9/v1", MODEL="offline-dummy")


def forbidden(*args, **kwargs):
    raise AssertionError("Offline full-material tests prohibit provider/network access")


socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules["dotenv"] = dotenv
import openai
openai.OpenAI = forbidden
openai.AsyncOpenAI = forbidden

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from pipeline import full_material_acceptance as acceptance
from pipeline.cumulative_entry_guard import enter_bounded_run, leave_bounded_run
from pipeline.quality import evaluate_release
from semantic_review_selftest import (CORPUS, ISOLATED_PROTOCOL, blind, evidence,
                                      reference_audit_response, isolated_adjudication)


def opinions(*, negative=False):
    reference = reference_audit_response(decision="accept", reason="原参考准确回答了客户身份。",
        claims=[{"reference_part": "answer", "source_quote": "北溟保险", "assessment": "成立",
                 "explanation": "原文明确记录。", "evidence": evidence()}],
        task_coverage="已回答客户身份。", substantive_defects=[], suggested_revision="")
    final = isolated_adjudication(reference_status="contradicted" if negative else "supported",
        original_answer_review=[{"requirement": "给出报告客户身份", "assessment": "完成",
            "explanation": "与原文一致。", "target_scope": "quoted_text",
            "reference_targets": [{"reference_part": "answer", "source_quote": "北溟保险"}]}],
        original_rationale_review={"status": "not_provided", "claims": [], "limitations": []},
        review_findings={"substantive_defects": [], "acceptable_brevity": [], "editorial_suggestions": []},
        concerns=[], reasoning="原参考回答了客户身份。", reference_audit_response="已复核独立审计及全部原文。")
    return {"blind_read": blind(), "reference_audit": reference, "adjudicate": final}


class FixtureTracer:
    """Synthetic replies and accounting only; no provider ever runs."""
    def __init__(self, run, *, negative=False, fail=False):
        self.run, self.n, self.calls = run, 7, []
        self.opinions, self.fail = opinions(negative=negative), fail

    def chat_json(self, step, messages, **params):
        self.n += 1
        payload = json.loads(messages[-1]["content"])
        role = ("reference_audit" if step.endswith("independent_reference_audit")
                else step.rsplit(".", 1)[-1])
        output = deepcopy(self.opinions[role])
        if self.fail:
            output = {"__error__": "offline cumulative cap", "__error_metadata__": {"kind": "budget_limit"}}
        elif "exact_reads" in payload:
            # unread_ids describes the window before mark_sent; continuing once
            # after its last page is legal and retains the original raw pages.
            output = ({"action": "continue", "notes": "保留原文支持和反证。"}
                      if payload["unread_ids"] else {"action": "finish", "opinion": output})
        row = {"i": self.n, "step": step, "messages": deepcopy(messages),
               "params": params, "output": deepcopy(output), "ok": not self.fail}
        self.calls.append(row)
        with (self.run.dir / "prompts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger = self.run.read("experiment_profile.json")
        ledger["admitted_calls"] += 1
        ledger["settled_usage_calls"] += 1
        ledger["budget_consumed_cny"] += 0.01
        self.run.write("experiment_profile.json", ledger)
        return output


class FixtureRun:
    def __init__(self, folder, **kwargs):
        self.dir, self.run_id = folder, "offline-full-material"
        self.manifest = {"env": {"reviewer_model": "offline-dummy"}}
        self.tracer = FixtureTracer(self, **kwargs)

    def has(self, name):
        return (self.dir / name).is_file()

    def read(self, name):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.dir / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class FullMaterialAcceptanceTests(unittest.TestCase):
    def make_run(self, *, long=False, **kwargs):
        temporary = tempfile.TemporaryDirectory(dir=str(acceptance._io_path(Path(tempfile.gettempdir()))))
        self.addCleanup(temporary.cleanup)
        run = FixtureRun(Path(temporary.name), **kwargs)
        corpus = deepcopy(CORPUS)
        if long:
            corpus["sessions"][0]["docs"][0]["content"] += "\n逐字材料甲乙。" * 14000
        question = {"qid": "q1", "question": "报告的客户是谁？", "gt": {"value": "北溟保险"},
                    "line": "L1_timeline", "capability": "IE", "entity": "不存在的实体",
                    "field": "客户", "semantic_scope_doc_ids": ["PRIVATE_SIGNAL_ID"],
                    "reference_proposal": {"answer": "不能替换原参考", "rationale": ""}}
        values = {"00_about.json": {"public_protocol": ISOLATED_PROTOCOL},
            "02_world.json": {"entities": {}}, "04_questions.json": [question],
            "05_corpus.json": corpus, "06_grounded_questions.json": [question],
            "06_grounding_report.json": {"drops": [], "pending": []},
            "manifest.json": {},
            "11_production.json": {"status": "generation_ready_awaiting_selection"},
            "experiment_profile.json": {"max_calls": 5000, "max_cny": 100,
                "admitted_calls": 7, "settled_usage_calls": 7, "budget_consumed_cny": 0.5,
                "unsettled_reservation_calls": 0, "stopped": False}}
        for name, value in values.items():
            run.write(name, value)
        run.write("07_release.json", evaluate_release(run.dir))
        return run

    def execute(self, run, **kwargs):
        token = enter_bounded_run(run.run_id)
        try:
            return acceptance.run_full_material_acceptance(run, **kwargs)
        finally:
            leave_bounded_run(token)

    def test_not_run_is_never_success(self):
        run = self.make_run()
        self.assertEqual(acceptance.snapshot(run)["status"], "not_run")
        self.assertFalse(acceptance.snapshot(run)["passed"])
        self.assertEqual(run.tracer.calls, [])

    def test_plain_or_early_entry_never_calls(self):
        run = self.make_run()
        with self.assertRaisesRegex(ValueError, "bounded"):
            acceptance.run_full_material_acceptance(run)
        run.write("11_production.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "generation_ready"):
            self.execute(run)
        self.assertEqual(run.tracer.calls, [])

    def test_actual_tracer_preserves_reader_binding_without_provider(self):
        import config
        from pipeline.run import Tracer
        run = self.make_run()
        script = iter(opinions().values())
        def bounded_fixture(messages, **params):
            ledger = run.read("experiment_profile.json")
            ledger["admitted_calls"] += 1
            ledger["settled_usage_calls"] += 1
            ledger["budget_consumed_cny"] += 0.01
            run.write("experiment_profile.json", ledger)
            return deepcopy(next(script))
        run.tracer = Tracer(run)
        with patch.object(config, "chat_json", bounded_fixture):
            result = self.execute(run)
        self.assertTrue(result["passed"], result)
        self.assertEqual(run.tracer.n, 3)
        self.assertEqual(result["ledger_after"]["admitted_calls"], 10)

    def test_unresolved_independent_reader_cannot_be_overridden_by_confident_final(self):
        run = self.make_run()
        run.tracer.opinions["reference_audit"]["decision"] = "unresolved"
        result = self.execute(run)
        self.assertFalse(result["passed"])
        self.assertEqual(len(run.tracer.calls), 3)
        self.assertTrue(acceptance.quality_snapshot(run.dir)["eligible"])

    def test_all_roles_get_full_public_material_and_original_reference(self):
        run = self.make_run()
        original = {name: (run.dir / name).read_bytes() for name in acceptance.INPUTS if run.has(name)}
        result = self.execute(run)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(run.tracer.calls), 3)
        self.assertEqual(len(result["coverage"]), 3)
        for row in run.tracer.calls:
            payload = json.loads(row["messages"][-1]["content"])
            self.assertEqual(len(payload["documents"]), 2)
            self.assertIn("之前提到的登记", payload["documents"][1]["content"])
            self.assertNotIn("PRIVATE_", json.dumps(payload))
        audit = json.loads(run.tracer.calls[1]["messages"][-1]["content"])
        self.assertEqual(audit["original_reference"], {"answer": "北溟保险", "rationale": ""})
        receipt = run.read(acceptance.ARTIFACT)
        docs = receipt["binding"]["documents"]
        self.assertGreater(docs[0]["content_utf8_bytes"], len(CORPUS["sessions"][0]["docs"][0]["content"]))
        for name, data in original.items():
            self.assertEqual((run.dir / name).read_bytes(), data, name)
        self.assertEqual(result["ledger_before"]["admitted_calls"], 7)
        self.assertEqual(result["ledger_after"]["admitted_calls"], 10)
        self.assertEqual(run.read("experiment_profile.json")["max_cny"], 100)
        self.assertTrue(self.execute(run)["passed"])
        self.assertEqual(len(run.tracer.calls), 3, "completed acceptance is replay-only")

    def test_long_document_all_parts_replay_and_tamper_fails(self):
        run = self.make_run(long=True)
        result = self.execute(run)
        self.assertTrue(result["passed"], result)
        self.assertGreater(len(run.tracer.calls), 3)
        self.assertTrue(all(row["transport"] == "exact_public_pages" for row in result["coverage"]))
        for role in result["coverage"]:
            self.assertGreater(len(role["page_requests"]), 2)
            self.assertEqual(role["document_count"], 2)
        receipt = run.read(acceptance.ARTIFACT)
        path = run.dir / acceptance.DIRECTORY / receipt["identity"] / "review.json"
        review = acceptance._read_sealed(path)
        review["items"][0]["blind_read_raw_output"]["_paged_read"]["transcript"].pop(0)
        acceptance._atomic_write_json(path, acceptance._sealed(review))
        receipt["review_hash"] = acceptance.fingerprint(review)
        receipt.pop("hash")
        run.write(acceptance.ARTIFACT, acceptance._sealed(receipt))
        self.assertFalse(acceptance.snapshot(run)["passed"])

    def test_tracer_tamper_cannot_be_resealed_as_success(self):
        run = self.make_run()
        self.assertTrue(self.execute(run)["passed"])
        path = run.dir / "prompts.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["messages"][1]["content"] = "different material"
        lines[0] = json.dumps(first)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(acceptance.snapshot(run)["passed"])

    def test_changed_public_material_invalidates_receipt(self):
        run = self.make_run()
        self.assertTrue(self.execute(run)["passed"])
        corpus = run.read("05_corpus.json")
        corpus["sessions"][0]["docs"][1]["content"] += "新的反证。"
        run.write("05_corpus.json", corpus)
        run.write("07_release.json", evaluate_release(run.dir))
        result = acceptance.snapshot(run)
        self.assertEqual(result["status"], "stale")
        self.assertFalse(result["passed"])

    def test_negative_full_context_keeps_original_release_and_no_retry(self):
        run = self.make_run(negative=True)
        before = (run.dir / "07_release.json").read_bytes()
        result = self.execute(run)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["current"], result)
        self.assertFalse(result["passed"])
        self.assertEqual(result["routing"]["n_dropped"], 1)
        self.assertEqual((run.dir / "07_release.json").read_bytes(), before)
        self.execute(run)
        self.assertEqual(len(run.tracer.calls), 3)

    def test_failed_physical_call_stops_later_roles_and_reentry(self):
        run = self.make_run(fail=True)
        result = self.execute(run)
        self.assertFalse(result["passed"])
        self.assertEqual(len(run.tracer.calls), 1)
        self.execute(run)
        self.assertEqual(len(run.tracer.calls), 1)

    def test_unsettled_writeahead_never_resends(self):
        run = self.make_run()
        original = run.tracer.chat_json
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt("simulated death after charged response")
        with patch.object(run.tracer, "chat_json", interrupted), self.assertRaises(KeyboardInterrupt):
            self.execute(run)
        self.assertFalse(acceptance.snapshot(run)["passed"])
        result = self.execute(run)
        self.assertFalse(result["passed"])
        self.assertEqual(len(run.tracer.calls), 1)


class SevenLineTargetTests(unittest.TestCase):
    """Real denominator/token checks; existing world enumerator is a fixture seam."""
    def setUp(self):
        self.base = FullMaterialAcceptanceTests()
        self.run = self.base.make_run()
        self.addCleanup(self.base.doCleanups)
        from pipeline.seed_pack import seed_contract
        from pipeline.seed_run import prepare_seed_input
        from pipeline.capability_contract import family_key
        from pipeline import factory
        pack = json.loads((ROOT / "output/seven_line_new_experiment_review_20260928/seed.json").read_text(encoding="utf-8"))
        target = {"requested_lines": list(acceptance.SEVEN_LINES),
                  "per_line_min": {line: 2 for line in acceptance.SEVEN_LINES},
                  "corpus_tokens": 4000, "tokenizer": "cl100k_base@0.12.0", "max_supply_rounds": 3}
        self.run.manifest["config"] = {"delivery_target": target, "haystack_ratio": 0,
            "seed_pack_digest": acceptance.SHOWCASE_SEED_DIGEST, "seed_id": pack["seed_id"]}
        self.run.write("00_seed_pack.json", pack)
        self.run.write("00_input.json", prepare_seed_input(self.run))
        self.run.write("01_whitepaper.json", {"seed_contract": seed_contract(pack),
            "supply_plan": {"instance_policy": "business-instance-plan/v3"},
            "business_instance_plan": {"plan_hash": "synthetic-enumerator-fixture"}})
        self.run.write("02_world.json", {"n_sessions": 6, "entities": {f"entity-{i}": {} for i in range(24)}})
        self.run.write("02_instance_fulfillment.json", {"verification_policy": "actual-independent-supply/v1"})
        self.run.write("05_corpus.json", {"sessions": [{"session_id": i,
            "docs": [{"doc_id": f"d{i}", "content": "hello world " * 500}]} for i in range(6)]})
        self.run.write("11_production.json", {"status": "generation_ready_awaiting_selection",
                                             "round": 1, "supply_round": 1})
        self.orders = [{"qid": f"{line}-{i}", "line": line, "capability": "IE", "entity": f"entity-{i}",
                        "field": line, "question": "合成测试题", "gt": {"value": i},
                        "aux": ({"sub": "S1_trend", "supply_origin": "native_numeric_trajectory/v1"}
                                if line == "L7_consolidation" and i == 0 else {})}
                       for line in acceptance.SEVEN_LINES for i in range(6)]
        for name in ("03_raw_orders.json", "03_orders.json", "04_questions.json"):
            self.run.write(name, self.orders)
        self.released = [row for row in self.orders if int(row["qid"].rsplit("-", 1)[1]) < 2]
        self.refresh_quality()
        certificates = [{"line": row["line"], "family_key": family_key(row), "passed": True} for row in self.orders]
        mock_capacity = patch("pipeline.capability_contract.actual_capacity", return_value={"certificates": certificates})
        mock_current = patch.object(factory, "_corpus_is_current", return_value=True)
        cache = patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(ROOT / "output/tokenizer_validation/encoding_cache")})
        self.capacity_mock = mock_capacity.start()
        self.current_mock = mock_current.start()
        cache.start()
        for item in (mock_capacity, mock_current, cache):
            self.addCleanup(item.stop)

    def refresh_quality(self):
        self.run.write("06_grounded_questions.json", self.released)
        kept = {row["qid"] for row in self.released}
        self.run.write("06_grounding_report.json", {"pending": [],
            "drops": [{"qid": row["qid"]} for row in self.orders if row["qid"] not in kept]})
        self.run.write("07_release.json", evaluate_release(self.run.dir))

    def assert_failed(self, name):
        result = acceptance.target_checks(self.run)
        self.assertFalse(result["passed"], result)
        self.assertIn(name, result["failed_checks"], result)
        return result

    def test_complete_denominators_and_actual_tokens_are_read_only(self):
        before = {p.name: p.read_bytes() for p in self.run.dir.iterdir() if p.is_file()}
        result = acceptance.target_checks(self.run)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(result["by_line"]), 7)
        for row in result["by_line"]:
            self.assertEqual(row["raw"]["independent_count"], 6)
            self.assertEqual(row["effective"]["certified_count"], 6)
            self.assertEqual(row["released"]["certified_count"], 2)
            self.assertEqual(len(row["quality_partition"]["rejected"]), 4)
        self.assertGreaterEqual(result["checks"]["body_tokens"]["core"]["tokens"], 4000)
        self.assertEqual(result["four_athletes_and_easy_filter"]["status"], "not_run")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.dir.iterdir() if p.is_file()})
        self.assertEqual(self.run.tracer.calls, [])

    def test_duplicate_candidate_ids_do_not_make_independent_supply(self):
        raw = self.run.read("03_raw_orders.json")
        for i in range(1, 6):
            qid = raw[i]["qid"]
            raw[i] = {**deepcopy(raw[0]), "qid": qid}
        self.run.write("03_raw_orders.json", raw)
        self.assert_failed("seven_line_denominators")

    def test_full_release_counts_cannot_hide_effective_shortfall(self):
        self.run.write("03_orders.json", self.orders[1:])
        result = self.assert_failed("seven_line_denominators")
        self.assertEqual(result["by_line"][0]["effective"]["certified_count"], 5)

    def test_two_released_comparisons_cannot_replace_native_trend(self):
        self.released = [q for q in self.released if q["qid"] != "L7_consolidation-0"]
        self.released.append(next(q for q in self.orders if q["qid"] == "L7_consolidation-2"))
        self.refresh_quality()
        result = self.assert_failed("released_native_l7")
        self.assertTrue(result["checks"]["seven_line_denominators"]["passed"])

    def test_world_shape_and_missing_period_are_not_hidden_by_release(self):
        world = self.run.read("02_world.json")
        world["entities"].pop("entity-23")
        world["n_sessions"] = 5
        self.run.write("02_world.json", world)
        corpus = self.run.read("05_corpus.json")
        corpus["sessions"].pop()
        self.run.write("05_corpus.json", corpus)
        self.refresh_quality()
        result = self.assert_failed("world_shape")
        self.assertIn("all_six_periods_published", result["failed_checks"])

    def test_token_receipt_claim_and_zero_ratio_cannot_hide_filler(self):
        corpus = self.run.read("05_corpus.json")
        for session in corpus["sessions"]:
            session["docs"][0]["is_filler"] = True
        self.run.write("05_corpus.json", corpus)
        self.refresh_quality()
        result = self.assert_failed("haystack_zero")
        self.assertIn("body_tokens", result["failed_checks"])
        self.assertGreater(result["checks"]["body_tokens"]["total"]["tokens"], 4000)
        self.assertEqual(result["checks"]["body_tokens"]["core"]["tokens"], 0)

    def test_supply_round_limit_is_separate_from_layout_attempts(self):
        state = self.run.read("11_production.json")
        state.update(round=9, layout_revisions=3, supply_round=3)
        self.run.write("11_production.json", state)
        self.assertTrue(acceptance.target_checks(self.run)["passed"])
        state["supply_round"] = 4
        self.run.write("11_production.json", state)
        self.assert_failed("supply_rounds")


if __name__ == "__main__":
    unittest.main()
