"""Integration in original stages, with every network call blocked."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import process_proposals_selftest as core
from pipeline import question_contract as contract, factory
from pipeline.render import phrase_questions
from pipeline.question_wording import validate_wording
from pipeline.well_posed import run_well_posed


class Run:
    def __init__(self, wp, ws, proposal, budget=2, enabled=True):
        self._temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self._temporary.name)
        self.manifest = {"config": {"question_budget": budget, "process_proposals": enabled}}
        self.data = {"01_whitepaper.json": deepcopy(wp), "02_world.json": ws.to_dict()}
        self.calls = []
        self.tracer = self
        self.proposal = proposal
        self.log = lambda *a: None
        self.algo = {}

    def read(self, key):
        return deepcopy(self.data[key])

    def write(self, key, value):
        self.data[key] = deepcopy(value)

    def set_algo(self, **kwargs):
        self.algo.update(kwargs)

    def chat_json(self, step, messages, **kwargs):
        self.calls.append((step, deepcopy(messages), deepcopy(kwargs)))
        if step == "orders.process_propose":
            return {"proposals": [deepcopy(self.proposal)], "reason": "offline author"}
        if step == "phrase":
            return {"question": self.proposal["intent"]}
        if step == "phrase.review":
            return {"verdict": "equivalent", "reason": "Offline opinion for execution test", "issues": []}
        raise AssertionError(step)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.wp, self.ws, self.proposal = core.fixture()
        self.block = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.block.start(); self.addCleanup(self.block.stop)
        self.patches = [patch.object(factory, name, return_value=None) for name in (
            "validate_seed_identity", "validate_seed_world", "_require_current_world_review")]
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)

    def order(self):
        return core.helper._proposal_order(self.proposal, self.ws)

    def test_contract_distinguishes_natural_reference_from_world_witness(self):
        order = contract.attach_question_contract(contract.bind_question_world(self.order(), self.ws), self.wp)
        c = order["question_contract"]
        self.assertEqual(c["gold"], self.proposal["reference_proposal"]["answer"])
        self.assertEqual(c["canonical_witness"], order["gt"])
        self.assertNotEqual(c["gold"], c["canonical_witness"])
        self.assertEqual(c["reference_authority"], "proposed_not_mechanically_proven")
        self.assertEqual(c["render_policy"], "semantic_review")
        self.assertEqual(c["query_time"]["ordinal"], 3)
        self.assertEqual(contract.validate_question({**order, "question": self.proposal["intent"]}), [])

    def test_reference_and_intent_tampering_invalidates_contract(self):
        original = contract.attach_question_contract(self.order(), self.wp)
        original["question"] = self.proposal["intent"]
        for mutate in (lambda q: q["reference_proposal"].update(rationale="invented reason"),
                       lambda q: q["aux"]["process"].update(intent="different task"),
                       lambda q: q["gt"]["events"][0].update(type="invented event")):
            changed = deepcopy(original); mutate(changed)
            self.assertIn("contract_mismatch", {r["code"] for r in contract.validate_question(changed)})

    def test_missing_A_policy_cannot_fall_back_to_fixed_template(self):
        with self.assertRaises(ValueError):
            contract.build_question_contract(self.order(), {})

    def test_original_orders_well_posed_and_wording_execute_same_frozen_order(self):
        run = Run(self.wp, self.ws, self.proposal)
        before = self.ws.to_dict()
        factory.stage_orders(run)
        self.assertEqual([row[0] for row in run.calls], ["orders.process_propose"])
        orders = run.data["03_orders.json"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(run.data["03_supply_report.json"]["shortfall"], 1)
        kept, report = run_well_posed(orders, self.ws)
        self.assertEqual(kept, orders)
        questions = phrase_questions(orders, self.wp, run.tracer, lambda *a: None)
        self.assertEqual(len(questions), 1)
        self.assertEqual(validate_wording(questions[0]), [])
        self.assertEqual(core.helper.validate_process_order(questions[0], self.ws), orders[0]["gt"])
        self.assertEqual(questions[0]["reference_proposal"], self.proposal["reference_proposal"])
        self.assertEqual(run.data["02_world.json"], before)
        self.assertEqual(self.ws.to_dict(), before)

    def test_zero_allocation_or_historical_disabled_has_no_call(self):
        for budget, enabled in ((0, True), (2, False)):
            run = Run(self.wp, self.ws, self.proposal, budget=budget, enabled=enabled)
            factory.stage_orders(run)
            self.assertEqual(run.calls, [])
            self.assertEqual(run.data["03_orders.json"], [])

    def test_missing_public_review_fails_before_paid_call(self):
        self.wp["quality_contract"]["isolated_reference_audit"] = False
        run = Run(self.wp, self.ws, self.proposal)
        with self.assertRaises(ValueError):
            factory.stage_orders(run)
        self.assertEqual(run.calls, [])
        self.assertNotIn("03_orders.json", run.data)

    def test_bad_model_response_preserved_and_other_orders_continue(self):
        run = Run(self.wp, self.ws, self.proposal)
        run.chat_json = lambda *a, **kw: {"__error__": "offline failure"}
        factory.stage_orders(run)
        self.assertEqual(run.data["03_process_proposals.json"]["status"], "error")
        self.assertIn("03_orders.json", run.data)
        self.assertEqual(run.data[factory.ORDER_WARNING]["status"], "warning")


if __name__ == "__main__":
    unittest.main(verbosity=2)
