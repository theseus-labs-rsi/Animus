"""Offline world truth / task-shape separation and saved smoke-response replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from pipeline.world_gen import _blocking_world_defects, build_world
from pipeline.world_state import Op, SET, UPDATE, Timeline, WorldState, validate
from pipeline.lines.L1_timeline import _candidates, TimelineLine


def world(values, kind="numeric", **declaration):
    field = {"name": "arbitrary_field", "kind": kind, **declaration}
    ws = WorldState(entities={"entity": {"arbitrary_field": Timeline([
        Op(i, f"2025-01-{i+1:02d}", SET if i == 0 else UPDATE, value,
           None if i == 0 else values[i-1]) for i, value in enumerate(values)])}},
        n_sessions=len(values), entity_types={"entity": "thing"},
        world_blueprint={"entity_types": [{"id": "thing", "fields": [field]}]})
    return ws, {"field_schema": [field]}


class ShapeTests(unittest.TestCase):
    def test_text_digits_are_not_partial_numeric_values(self):
        for values in (["瀚历4月初二", "瀚历4月初九", "瀚历4月十六"],
                       ["document 2 rev A", "document 3 rev B", "document 4 rev C"],
                       ["2", "3", "4"]):
            with self.subTest(values=values):
                ws, profile = world(values, "text")
                self.assertEqual(validate(ws, profile=profile), [])
                self.assertEqual(_candidates(ws, "MR"), [])

    def test_numeric_endpoint_is_diagnostic_not_world_error(self):
        ws, profile = world(["10", "20", "30", "40"])
        before = deepcopy(ws.to_dict())
        findings = validate(ws, profile=profile)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["extrema_sessions"], {"max": 3, "min": 0})
        self.assertEqual(findings[0]["severity"], "diagnostic")
        self.assertEqual(len(findings[0]["samples"]), 4)
        self.assertEqual(_blocking_world_defects(findings, False), [])
        self.assertEqual(ws.to_dict(), before)
        mr = _candidates(ws, "MR")
        self.assertEqual(len(mr), 2)
        for order in mr:
            passed, reason = TimelineLine().well_posed(asdict(order), ws)
            self.assertEqual(passed, "well_posed", reason)

    def test_complete_dates_share_comparison_without_field_name_exceptions(self):
        ws, profile = world(["2025-01-30", "2025-02-01", "2025-03-01"], "date")
        finding, = validate(ws, profile=profile)
        self.assertEqual(finding["value_kind"], "date")
        self.assertEqual(finding["extrema_sessions"], {"max": 2, "min": 0})
        self.assertEqual(_blocking_world_defects([finding], False), [])

    def test_no_subset_or_partial_parse_diagnostic(self):
        ws, profile = world(["10", "not a value 20", "30", "40"])
        self.assertEqual(validate(ws, profile=profile), [])

    def test_declared_direction_and_range_still_block(self):
        ws, profile = world(["1", "3", "2", "5"], monotonic="up", range=[0, 4])
        findings = validate(ws, profile=profile)
        self.assertEqual({x["type"] for x in findings}, {"monotonic_violation", "out_of_range"})
        self.assertEqual(_blocking_world_defects(findings, False), findings)

    def test_declared_state_transition_still_blocks(self):
        ws, profile = world(["draft", "final", "draft"], "status")
        profile["state_machines"] = [{"field": "arbitrary_field", "states": ["draft", "final"]}]
        findings = validate(ws, profile=profile)
        self.assertEqual([x["type"] for x in findings], ["illegal_transition"])
        self.assertEqual(_blocking_world_defects(findings, False), findings)

    def test_declared_evolving_contract_still_blocks_non_narrative(self):
        ws, profile = world(["unchanged"], "text")
        table = {"entities": [{"name": "entity", "fields": {"arbitrary_field": {"type": "evolving"}}}]}
        findings = validate(ws, table, profile)
        self.assertEqual([x["type"] for x in findings], ["fake_evolving"])
        self.assertEqual(_blocking_world_defects(findings, False), findings)

    def test_diagnostics_roundtrip_and_legacy_empty_shape(self):
        ws, profile = world(["1", "2", "3"])
        self.assertNotIn("generation_diagnostics", ws.to_dict())
        ws.generation_diagnostics = validate(ws, profile=profile)
        saved = ws.to_dict()
        self.assertEqual(WorldState.from_dict(saved).to_dict(), saved)
        saved["generation_diagnostics"][0]["detail"] = "changed copy"
        self.assertNotEqual(ws.generation_diagnostics[0]["detail"], "changed copy")

    @unittest.skipUnless((ROOT / "output/runs/bc_original_small_20260918/prompts.jsonl").exists(),
                         "Local real-run fixture absent; synthetic tests still run")
    def test_original_failed_smoke_raw_world_now_needs_no_repair(self):
        directory = ROOT / "output/runs/bc_original_small_20260918"
        paths = [directory / "01_whitepaper.json", directory / "prompts.jsonl"]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        wp = json.loads(paths[0].read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
        originals = [r for r in rows if r["step"] in ("world.batch", "world.structure")]
        self.assertEqual([r["step"] for r in originals], ["world.batch"]*4+["world.structure"])
        class Replay:
            def __init__(self): self.steps = []
            def chat_json(self, step, messages, **kwargs):
                index = len(self.steps)
                self.steps.append(step)
                if index >= len(originals) or originals[index]["step"] != step:
                    raise AssertionError(f"Unexpected extra/repaired request: {step}")
                return deepcopy(originals[index]["output"])
        tracer = Replay()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network prohibited")), \
             patch.object(config, "chat", side_effect=AssertionError("API prohibited")), \
             patch.object(config, "chat_json", side_effect=AssertionError("API prohibited")):
            ws = build_world(wp, tracer, log=lambda *_: None)
        self.assertEqual(tracer.steps, ["world.batch"]*4+["world.structure"])
        self.assertEqual(len(ws.entities), 10)
        for row in originals[:4]:
            for entity in row["output"]["entities"]:
                for name, spec in entity["fields"].items():
                    if name != "发布日期":
                        continue
                    expected = [(p["session"], p["value"]) for p in spec["trajectory"] if p["value"] is not None]
                    self.assertEqual([(s, v) for s, _, v in ws.timeline(entity["name"], name).set_values()], expected)
        self.assertFalse(any(d["field"] == "发布日期" for d in ws.generation_diagnostics))
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths], hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
