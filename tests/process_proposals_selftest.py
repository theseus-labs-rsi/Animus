"""Network-blocked integration tests for original typed process proposals."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import process_proposals as helper
from pipeline import lines
from pipeline.lines import L3_process as l3
from pipeline.world_state import WorldState, Timeline, Op, SET, build_demo_world


def fixture():
    bp = {"version": 1, "temporal_model": {"n_sessions": 3, "unit": "week"},
          "entity_types": [{"id": "source", "fields": [{"name": "state", "kind": "text"}]},
                           {"id": "analysis", "fields": [{"name": "state", "kind": "text"}]}],
          "event_types": [{"id": "received"}, {"id": "adopted"}, {"id": "reviewed"}]}
    ws = WorldState(entities={"Doc": {"state": Timeline([Op(0, "2025-01-06", SET, "received")])},
                              "Analysis": {"state": Timeline([Op(1, "2025-01-13", SET, "pending")])}},
        n_sessions=3, entity_types={"Doc": "source", "Analysis": "analysis"}, world_blueprint=bp,
        relations=[{"id": "r1", "type": "uses", "from": "Analysis", "to": "Doc", "session": 1}],
        events=[{"id": "e1", "type": "received", "session": 0, "participants": {"source": "Doc"},
                 "effects": [{"entity": "Doc", "field": "state", "set": "received"}]},
                {"id": "e2", "type": "adopted", "session": 1,
                 "participants": {"source": "Doc", "analysis": "Analysis"}, "caused_by": "e1",
                 "effects": [{"entity": "Analysis", "field": "state", "set": "pending"}]}])
    wp = {"world_blueprint": deepcopy(bp), "quality_contract": {
        "scoring_policy": "task-with-supporting-reasons/v1",
        "public_semantic_review": True, "isolated_reference_audit": True},
        "active_lines": [{"line": "L3_process", "weight": 1}],
        "capability_targets": {"total_q": 4}}
    proposal = {"id": "p1", "entity": "Analysis", "at_session": 2,
                "intent": "说明 Analysis 采用 Doc 后的实际状态及该过程的记录依据。",
                "witness_refs": {"event_ids": ["e2", "e1"], "relation_ids": ["r1"],
                    "facts": [{"entity": "Analysis", "field": "state", "session": 2}]},
                "reference_proposal": {"answer": {"state": "pending"}, "rationale": "待公开正文核验"},
                "unresolved_preconditions": "正文是否表达接收与采用关联尚未检查"}
    return wp, ws, proposal


class Fake:
    def __init__(self, raw=None, error=None):
        self.raw, self.error, self.calls = raw, error, []

    def __call__(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if self.error:
            raise self.error
        return deepcopy(self.raw)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        cls.net.start()

    @classmethod
    def tearDownClass(cls):
        cls.net.stop()

    def setUp(self):
        self.wp, self.ws, self.proposal = fixture()

    def propose(self, proposals=None, **kwargs):
        fake = Fake({"proposals": proposals if proposals is not None else [self.proposal], "reason": "proposal"})
        report = helper.propose_process_orders(self.wp, self.ws, target=kwargs.pop("target", 2),
            chat_json=fake, model="offline-cheap", **kwargs)
        return report, fake

    def test_actual_messages_single_call_no_mutation(self):
        before = deepcopy(self.ws.to_dict()), deepcopy(self.wp)
        report, fake = self.propose()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(fake.calls), 1)
        step, messages, params = fake.calls[0]
        self.assertEqual(step, "orders.process_propose")
        self.assertEqual(params["retries"], 1)
        self.assertTrue(params["strict_json"])
        self.assertEqual(params["response_format"], {"type": "json_object"})
        self.assertEqual(json.loads(messages[1]["content"])["world"], before[0])
        self.assertEqual((self.ws.to_dict(), self.wp), before)
        order = helper.validate_process_report(report, self.wp, self.ws)[0]
        self.assertNotEqual(order["gt"], order["reference_proposal"]["answer"])
        self.assertEqual([x["id"] for x in order["gt"]["events"]], ["e1", "e2"])
        self.assertEqual(order["gt"]["causal_edges"], [{"parent_id": "e1", "child_id": "e2"}])

    def test_typed_set_events_do_not_lower_old_sorting(self):
        old = l3.ProcessLine()
        self.assertFalse(old.feasible(self.ws, {})[0])
        self.assertEqual(old.enumerate(self.ws), [])
        report, _ = self.propose()
        stats = {}
        orders = lines.run_lines(self.wp, self.ws, lambda *_: None, question_budget=2,
                                 stats=stats, process_proposals=report)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["capability"], helper.CAPABILITY)
        self.assertEqual(stats["shortfall"], 1)
        self.assertEqual(old.well_posed(orders[0], self.ws)[0], "well_posed")
        self.assertEqual(old.gt(self.ws, orders[0]), orders[0]["gt"])
        self.assertEqual(old.intent(orders[0]), (self.proposal["intent"], []))
        self.assertEqual(old.ground(orders[0], [])[0], "drop")

    def test_no_proposal_report_keeps_default_path_without_new_author(self):
        ws = build_demo_world()
        with patch.object(helper, "validate_process_report", side_effect=AssertionError("Unexpected typed branch")):
            stats = {}
            orders = lines.run_lines(self.wp, ws, lambda *_: None, question_budget=5, stats=stats)
        self.assertTrue(orders)
        self.assertEqual({order["capability"] for order in orders}, {"L3_order"})
        self.assertNotIn("process_proposals", stats)
        self.assertTrue(all("process_proposals" not in row for row in stats["lines"]))
        # Old sorting behavior itself is exercised by the existing L3/registry selftests.

    def test_total_quota_and_legacy_selection_remain_bounded(self):
        p2 = deepcopy(self.proposal); p2.update(id="p2", intent="另一自然问题")
        report, _ = self.propose([self.proposal, p2])
        for kwargs in ({"question_budget": 1}, {"quotas": {"L3_process": 1}}):
            got = lines.run_lines(self.wp, self.ws, lambda *_: None, process_proposals=report, **kwargs)
            self.assertEqual(len(got), 1)
        self.assertEqual(len(lines.run_lines(self.wp, self.ws, lambda *_: None, process_proposals=report)), 2)

    def test_bad_witness_rejected_whole_candidate_denominator_kept(self):
        bad = deepcopy(self.proposal); bad["id"] = "bad"; bad["witness_refs"]["event_ids"] = ["fake"]
        report, fake = self.propose([bad, self.proposal])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual([x["status"] for x in report["items"]], ["rejected", "structurally_bound"])
        self.assertEqual(len(helper.validate_process_report(report, self.wp, self.ws)), 1)

    def test_future_duplicates_and_unknown_field_are_not_bound(self):
        mutations = [lambda p: p.update(at_session=0),
                     lambda p: p["witness_refs"].update(event_ids=["e1", "e1"]),
                     lambda p: p["witness_refs"]["facts"][0].update(field="missing")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                p = deepcopy(self.proposal); mutate(p)
                report, _ = self.propose([p])
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["orders"], [])
                self.assertEqual(report["items"][0]["status"], "rejected")

    def test_report_raw_reference_and_witness_cannot_diverge(self):
        report, _ = self.propose()
        for key in ("reference", "witness", "raw", "messages", "items"):
            with self.subTest(key=key):
                bad = deepcopy(report)
                if key == "reference": bad["orders"][0]["reference_proposal"]["answer"] = "changed"
                elif key == "witness": bad["orders"][0]["gt"]["events"][0]["participants"]["source"] = "Analysis"
                elif key == "raw": bad["raw_output"]["proposals"][0]["intent"] = "changed"
                elif key == "messages": bad["messages"][0]["content"] += "changed"
                else: bad["items"] = []
                with self.assertRaises(ValueError): helper.validate_process_report(bad, self.wp, self.ws)

    def test_world_whitepaper_or_policy_change_refuses_before_dispatch(self):
        report, _ = self.propose()
        changed = deepcopy(self.ws); changed.events[0]["effects"][0]["set"] = "different"
        with self.assertRaises(ValueError): helper.validate_process_report(report, self.wp, changed)
        for key, value in (("scoring_policy", "arbitrary"), ("public_semantic_review", False),
                           ("isolated_reference_audit", False)):
            wp = deepcopy(self.wp); wp["quality_contract"][key] = value
            fake = Fake({})
            with self.assertRaises(ValueError):
                helper.propose_process_orders(wp, self.ws, target=1, chat_json=fake, model="offline")
            self.assertEqual(fake.calls, [])

    def test_provider_and_callback_failure_preserve_raw_stop_no_retry(self):
        for error in (RuntimeError("quota"), None):
            fake = Fake({"proposals": [self.proposal], "reason": "ok"}, error)
            def recorder(event):
                if event["event"] == "finished": raise OSError("audit write failure")
            report = helper.propose_process_orders(self.wp, self.ws, target=1, chat_json=fake,
                model="offline", record=recorder)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["orders"], [])
            if error is None: self.assertEqual(report["raw_output"], fake.raw)
            with self.assertRaises(ValueError):
                lines.run_lines(self.wp, self.ws, process_proposals=report)

    def test_budget_and_oversupply_no_silent_truncation(self):
        for kwargs in ({"max_calls": 0}, {"max_input_chars": 1}):
            report, fake = self.propose(**kwargs)
            self.assertEqual(report["status"], "error"); self.assertEqual(fake.calls, [])
        report, fake = self.propose([], target=0, max_calls=0)
        self.assertEqual(report["status"], "completed"); self.assertEqual(fake.calls, [])
        self.assertEqual(helper.validate_process_report(report, self.wp, self.ws), [])
        p2=deepcopy(self.proposal);p2["id"]="p2"
        report, fake=self.propose([self.proposal,p2],target=1)
        self.assertEqual(report["status"],"error");self.assertEqual(len(fake.calls),1)
        self.assertEqual(len(report["raw_output"]["proposals"]),2)

    def test_natural_wrong_or_uncertain_reference_is_not_program_scored(self):
        proposal = deepcopy(self.proposal)
        proposal["reference_proposal"] = {"answer": "错误的自然业务主张", "rationale": "仍须审阅"}
        proposal["unresolved_preconditions"] = "公开正文可能没有依据"
        report,_=self.propose([proposal]);order=report["orders"][0]
        self.assertEqual(order["reference_proposal"],proposal["reference_proposal"])
        self.assertEqual(report["items"][0]["status"],"structurally_bound")
        self.assertNotIn("correct",order)
        self.assertEqual(order["aux"]["gt_scope"],"canonical_witness_not_semantic_proof")

    def test_finished_callback_mutation_cannot_change_bound_world(self):
        def record(event):
            if event["event"] == "finished": self.ws.events[0]["type"] = "mutated"
        report,fake=self.propose(record=record)
        self.assertEqual(report["status"],"error");self.assertEqual(len(fake.calls),1)
        self.assertIn("changed",report["error"])

    def test_empty_null_structured_answers_are_preserved(self):
        for answer in (None, {}, [], False):
            proposal=deepcopy(self.proposal);proposal["reference_proposal"]["answer"]=answer
            report,_=self.propose([proposal]);self.assertEqual(report["status"],"completed")
            self.assertEqual(report["orders"][0]["reference_proposal"]["answer"],answer)

    def test_world_period_unit_is_not_assumed_week(self):
        self.ws.world_blueprint["temporal_model"]["unit"] = "day"
        report, _ = self.propose()
        self.assertEqual(report["orders"][0]["aux"]["time_unit"], "天")

    def test_provider_error_marker_and_invalid_json_value_are_not_completed(self):
        for raw in ({"proposals": [], "reason": "unused", "__error__": "provider failed"},
                    {"proposals": [], "reason": "unused", "extra": float("nan")}, [1]):
            fake = Fake(raw)
            report = helper.propose_process_orders(self.wp, self.ws, target=1,
                chat_json=fake, model="offline")
            self.assertEqual(report["status"], "error")
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(report["orders"], [])

    def test_root_contract_witness_reference_and_wording_binding(self):
        from pipeline import question_contract as contract
        from pipeline.question_wording import review_wording, validate_wording
        from pipeline.well_posed import run_well_posed
        report, _ = self.propose()
        order = report["orders"][0]
        question = contract.attach_question_contract(contract.bind_question_world(order, self.ws), self.wp)
        question["question"] = self.proposal["intent"]
        frozen = question["question_contract"]
        self.assertEqual(frozen["gold"], order["reference_proposal"]["answer"])
        self.assertEqual(frozen["canonical_witness"], order["gt"])
        self.assertEqual(frozen["parameters"]["process"], order["aux"]["process"])
        fake = Fake({"verdict": "equivalent", "reason": "Offline interface test only", "issues": []})
        receipt = review_wording(question["question"], frozen, chat_json=fake, model="offline")
        question["question_validation"] = {"semantic_review": receipt}
        self.assertEqual(contract.validate_question(question), [])
        self.assertEqual(validate_wording(question), [])
        kept, audit = run_well_posed([question], self.ws)
        self.assertEqual(kept, [question]); self.assertEqual(audit["n_dropped"], 0)
        for kind in ("reference", "intent", "witness"):
            bad = deepcopy(question)
            if kind == "reference": bad["reference_proposal"]["answer"] = "different"
            elif kind == "intent": bad["aux"]["process"]["intent"] = "different task"
            else: bad["gt"]["events"][0]["type"] = "different"
            self.assertTrue(contract.validate_question(bad))
            self.assertEqual(l3.ProcessLine().well_posed(bad, self.ws)[0], "drop")

    def test_duplicate_content_different_id_does_not_inflate_supply(self):
        duplicate = deepcopy(self.proposal); duplicate["id"] = "another-id"
        report, _ = self.propose([self.proposal, duplicate])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["items"]), 2)
        self.assertEqual(len(report["orders"]), 1)
        self.assertIn("duplicate", report["items"][1]["reason"])
        self.assertEqual(helper.validate_process_report(report, self.wp, self.ws), report["orders"])

    def test_explicit_quota_mixes_old_sorting_and_new_process(self):
        demo = build_demo_world()
        self.ws.entities.update(deepcopy(demo.entities))
        self.ws.n_sessions = demo.n_sessions
        self.assertTrue(l3.ProcessLine().enumerate(self.ws))
        report, _ = self.propose()
        for kwargs in ({"quotas": {"L3_process": 2}}, {"question_budget": 2}):
            orders = lines.run_lines(self.wp, self.ws, lambda *_: None,
                                     process_proposals=report, **kwargs)
            self.assertEqual(len(orders), 2)
            self.assertEqual({o["capability"] for o in orders}, {"L3_order", "L3_process_trace"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
