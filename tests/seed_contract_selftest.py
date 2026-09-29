"""Offline seed schema, role isolation, identity and whitepaper repair gates.

Run: venv/Scripts/python.exe tests/seed_contract_selftest.py
No model calls or attachment reads are performed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.seed_pack import (
    SeedPackError, attach_seed_contract, load_seed_pack, seed_context,
    seed_contract, seed_digest, seed_input, validate_seed_blueprint, validate_seed_pack,
)
from pipeline.world_blueprint import WorldBlueprintError


def fixture():
    requirements = {
        "entity_types": [
            {"id": "report", "noun": "报告", "count": 2, "primary": True,
             "fields": [{"name": "version", "kind": "status"}]},
            {"id": "metric", "noun": "指标", "count": 2,
             "fields": [{"name": "adopted_report", "kind": "reference"},
                        {"name": "amount", "kind": "numeric", "unit": "万元", "range": [0, 1000]}]},
        ],
        "relation_types": [{"id": "uses", "from_type": "metric", "to_type": "report",
                            "field": "adopted_report", "temporal": True, "min_count": 1}],
        "event_types": [
            {"id": "revise", "label": "修订报告", "roles": {"report": "report"},
             "effect_fields": [{"role": "report", "field": "version"}], "min_count": 1},
            {"id": "recompute", "label": "重算指标", "roles": {"metric": "metric"},
             "effect_fields": [{"role": "metric", "field": "amount"}], "min_count": 1},
        ],
        "causal_rules": [{"id": "refresh", "trigger_event": "revise", "effect_event": "recompute",
                          "delay_sessions": 1}],
        "evidence_channels": ["报告", "工作记录"],
        "temporal_model": {"min_sessions": 4},
    }
    return {
        "schema_version": 1, "seed_id": "test_report", "family": "synthetic_reporting",
        "title": "报告修订", "description": "在报告修订后更新采用值和测算。",
        "sources": [
            {"id": "design", "path": "missing/design.md", "sha256": "a" * 64, "role": "builder_only"},
            {"id": "public", "path": "missing/report.pdf", "sha256": "b" * 64, "role": "corpus"},
            {"id": "judge", "path": "DO_NOT_EXPOSE_EVALUATOR_PATH.xlsx", "sha256": "c" * 64,
             "role": "evaluator_only", "title": "DO_NOT_EXPOSE_EVALUATOR_TITLE"},
        ],
        "task": {"objective": "更新工作底稿", "instructions": "依据正式修订重算指标，并保留来源。"},
        "mechanisms": [{"id": "version_chain", "description": "修订导致重算。",
                        "source_refs": [{"source_id": "design", "locator": "版本依赖节"}],
                        "required_entity_types": ["report", "metric"],
                        "required_relation_types": ["uses"],
                        "required_event_types": ["revise", "recompute"],
                        "required_causal_rules": ["refresh"]}],
        "exemplars": [{"title": "合成修订说明", "content": "正式修订到达，重新核算指标。",
                       "doc_type": "工作记录", "date": "2026-01-01", "origin": "synthetic",
                       "source_refs": [], "mechanism_refs": ["version_chain"]}],
        "blueprint_requirements": requirements,
    }


def blueprint(pack):
    value = deepcopy(pack["blueprint_requirements"])
    value["temporal_model"] = {"unit": "轮", "cadence": "daily", "n_sessions": 6, "step_days": 1}
    return value


def binding_fixture():
    pack = fixture()
    requirements = pack["blueprint_requirements"]
    for event in requirements["event_types"]:
        event["roles"] = {"report": "report", "metric": "metric"}
        event["relation_bindings"] = [{"relation": "uses", "from_role": "metric", "to_role": "report"}]
    requirements["causal_rules"][0]["shared_roles"] = ["report", "metric"]
    return pack


class CouncilTracer:
    def __init__(self, good_blueprint, *, first_bad=False, always_bad=False):
        self.blueprint = good_blueprint
        self.first_bad, self.always_bad = first_bad, always_bad
        self.calls = []

    def chat_json(self, tag, messages, **_kwargs):
        self.calls.append((tag, json.dumps(messages, ensure_ascii=False)))
        if tag == "council.observe":
            return {"observed_media": [], "observed_entities": [], "observed_fields": []}
        if tag in ("council.skeptic", "council.medium", "council.style", "council.traps"):
            return {}
        if tag in ("council.world", "council.world_repair"):
            candidate = deepcopy(self.blueprint)
            if self.always_bad or (self.first_bad and tag == "council.world"):
                candidate["causal_rules"] = []
            return {"world_blueprint": candidate}
        if tag == "council.blueprint_feasibility":
            business = json.loads(messages[1]["content"])["business"]
            return {"decision": "accept", "reason": "Fixture process is expressible", "issues": [],
                    "mechanism_checks": [{"mechanism_id": m["id"], "status": "feasible",
                        "walkthrough": "A revision is followed by review of the same report"}
                        for m in business["mechanisms"]]}
        if tag == "council.map":
            return {"per_line": [
                {"line": identity, "applicable": identity == "L1_timeline",
                 "gt_feasible": identity == "L1_timeline", "instantiation": "读取修订时序",
                 "weight_hint": 0.5 if identity == "L1_timeline" else 0}
                for identity in ("L1_timeline", "L2_relational", "L3_process", "L4_preference",
                                 "L5_conflict", "L6_refusal", "L7_consolidation")]}
        raise AssertionError(tag)


def main():
    from pipeline.central_office import central_office

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    def rejected(name, change, base=fixture):
        value = base()
        change(value)
        try:
            validate_seed_pack(value)
        except SeedPackError:
            check(name, True)
        else:
            check(name, False)

    pack = fixture()
    valid = validate_seed_pack(pack)
    check("validator returns an independent copy", valid == pack and valid is not pack)
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "seed.json"
        target.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
        check("loader needs no original attachment access", load_seed_pack(target) == pack)
        target.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        try:
            load_seed_pack(target)
        except SeedPackError:
            check("duplicate JSON keys rejected", True)
        else:
            check("duplicate JSON keys rejected", False)

    rejected("boolean version rejected", lambda p: p.update(schema_version=True))
    rejected("path traversal seed id rejected", lambda p: p.update(seed_id="../../escape"))
    rejected("unknown raw source content rejected", lambda p: p["sources"][2].update(content="answer leak"))
    rejected("unknown top-level evaluator content rejected", lambda p: p.update(evaluator_answers="answer leak"))
    rejected("duplicate source IDs rejected", lambda p: p["sources"].append(deepcopy(p["sources"][0])))
    rejected("bad source hash rejected", lambda p: p["sources"][0].update(sha256="bad"))
    rejected("dangling provenance rejected", lambda p: p["mechanisms"][0]["source_refs"][0].update(source_id="missing"))
    rejected("evaluator-derived mechanisms rejected", lambda p: p["mechanisms"][0]["source_refs"][0].update(source_id="judge"))
    rejected("dangling mechanism structure rejected", lambda p: p["mechanisms"][0].update(required_causal_rules=["missing"]))
    rejected("dangling relation endpoint rejected", lambda p: p["blueprint_requirements"]["relation_types"][0].update(to_type="missing"))
    rejected("duplicate field rejected", lambda p: p["blueprint_requirements"]["entity_types"][0]["fields"].append({"name": "version", "kind": "text"}))
    rejected("unknown blueprint rules rejected", lambda p: p["blueprint_requirements"].update(instructions="leak"))
    rejected("non-finite JSON rejected", lambda p: p["blueprint_requirements"]["entity_types"][1]["fields"][1].update(range=[0, float("nan")]))
    rejected("invalid example date rejected", lambda p: p["exemplars"][0].update(date="2026-02-30"))
    rejected("synthetic examples cannot pose as source records", lambda p: p["exemplars"][0].update(source_refs=[{"source_id": "public", "locator": "p1"}]))
    rejected("source examples cannot quote builder material", lambda p: p["exemplars"][0].update(origin="source", source_refs=[{"source_id": "design", "locator": "p1"}]))
    rejected("source examples cannot quote evaluator material", lambda p: p["exemplars"][0].update(origin="source", source_refs=[{"source_id": "judge", "locator": "R1"}]))
    bound = binding_fixture()
    check("explicit business-object bindings accepted", bool(validate_seed_pack(bound)))
    rejected("empty shared roles rejected", lambda p: p["blueprint_requirements"]["causal_rules"][0].update(shared_roles=[]), binding_fixture)
    rejected("duplicate shared roles rejected", lambda p: p["blueprint_requirements"]["causal_rules"][0].update(shared_roles=["report", "report"]), binding_fixture)
    rejected("missing shared role rejected", lambda p: p["blueprint_requirements"]["causal_rules"][0].update(shared_roles=["unknown"]), binding_fixture)
    rejected("empty relation bindings rejected", lambda p: p["blueprint_requirements"]["event_types"][0].update(relation_bindings=[]), binding_fixture)
    rejected("duplicate relation bindings rejected", lambda p: p["blueprint_requirements"]["event_types"][0]["relation_bindings"].append(deepcopy(p["blueprint_requirements"]["event_types"][0]["relation_bindings"][0])), binding_fixture)
    rejected("missing bound relation rejected", lambda p: p["blueprint_requirements"]["event_types"][0]["relation_bindings"][0].update(relation="unknown"), binding_fixture)
    rejected("missing bound role rejected", lambda p: p["blueprint_requirements"]["event_types"][0]["relation_bindings"][0].update(from_role="unknown"), binding_fixture)
    rejected("bound endpoint type mismatch rejected", lambda p: p["blueprint_requirements"]["event_types"][0]["relation_bindings"][0].update(from_role="report", to_role="metric"), binding_fixture)
    def mismatched_shared_role(p):
        event = p["blueprint_requirements"]["event_types"][1]
        event.pop("relation_bindings")
        event["roles"]["report"] = "metric"
    rejected("shared role types must match", mismatched_shared_role, binding_fixture)
    bound_bp = blueprint(bound)
    check("binding contract retained by executable blueprint", validate_seed_blueprint(bound_bp, bound)["passed"])
    removed_binding = deepcopy(bound_bp)
    removed_binding["event_types"][0].pop("relation_bindings")
    check("blueprint cannot drop event FK binding", not validate_seed_blueprint(removed_binding, bound)["passed"])
    removed_shared = deepcopy(bound_bp)
    removed_shared["causal_rules"][0]["shared_roles"] = ["report"]
    check("blueprint cannot drop required shared role", not validate_seed_blueprint(removed_shared, bound)["passed"])
    reordered_shared = deepcopy(bound_bp)
    reordered_shared["causal_rules"][0]["shared_roles"].reverse()
    check("shared role declaration order is irrelevant", validate_seed_blueprint(reordered_shared, bound)["passed"])
    invalid_extra_binding = deepcopy(bound_bp)
    invalid_extra_binding["event_types"][0]["relation_bindings"].append(
        {"relation": "missing_extension", "from_role": "metric", "to_role": "report"})
    check("additional generated bindings still need valid references", not validate_seed_blueprint(invalid_extra_binding, bound)["passed"])
    original_example = fixture()
    original_example["exemplars"][0].update(origin="source", source_refs=[{"source_id": "public", "locator": "p1"}])
    check("public source example accepted", bool(validate_seed_pack(original_example)))
    reordered = {key: pack[key] for key in reversed(list(pack))}
    check("identity independent of JSON key order", seed_digest(pack) == seed_digest(reordered))
    altered = deepcopy(pack)
    altered["task"]["objective"] += "并记录历史"
    check("identity changes with task contents", seed_digest(pack) != seed_digest(altered))
    context = seed_context(pack)
    check("evaluator source metadata absent from context", "DO_NOT_EXPOSE" not in context and '"judge"' not in context)
    check("synthetic origin visible in context", '"origin": "synthetic"' in context)
    description, examples = seed_input(pack)
    check("seed input carries task and origin", pack["task"]["objective"] in description and examples[0]["seed_origin"] == "synthetic")

    bp = blueprint(pack)
    report = validate_seed_blueprint(bp, pack)
    check("required structure accepted", report["passed"] and all(item["passed"] for item in report["mechanism_coverage"]))
    enlarged = deepcopy(bp)
    enlarged["entity_types"][0]["count"] += 2
    enlarged["entity_types"][0]["fields"].append({"name": "note", "kind": "text"})
    enlarged["evidence_channels"].append("邮件")
    check("additional structure and capacity accepted", validate_seed_blueprint(enlarged, pack)["passed"])
    for name, change in (
        ("missing causal dependency rejected", lambda b: b.update(causal_rules=[])),
        ("changed field unit rejected", lambda b: b["entity_types"][1]["fields"][1].update(unit="元")),
        ("changed numeric range rejected", lambda b: b["entity_types"][1]["fields"][1].update(range=[0, 2000])),
        ("insufficient entity count rejected", lambda b: b["entity_types"][0].update(count=1)),
        ("too few sessions rejected", lambda b: b["temporal_model"].update(n_sessions=2)),
        ("missing evidence channel rejected", lambda b: b.update(evidence_channels=["报告"])),
    ):
        altered_bp = deepcopy(bp)
        change(altered_bp)
        check(name, not validate_seed_blueprint(altered_bp, pack)["passed"])
    whitepaper = {"world_blueprint": bp}
    attached = attach_seed_contract(whitepaper, pack)
    contract = attached["seed_contract"]
    check("attachment keeps original whitepaper untouched", "seed_contract" not in whitepaper)
    check("contract keeps original identity", seed_digest(contract) == seed_digest(pack))
    check("contract validates without private evaluator metadata", validate_seed_blueprint(bp, contract)["passed"])
    check("public snapshot helper deterministic", contract == seed_contract(pack))
    edited_contract = deepcopy(contract)
    edited_contract["task"]["objective"] = "mutated after freeze"
    try:
        validate_seed_pack(edited_contract)
    except SeedPackError:
        check("stale snapshot digest rejected", True)
    else:
        check("stale snapshot digest rejected", False)

    tracer = CouncilTracer(bp, first_bad=True)
    result = central_office(description, examples, tracer, log=lambda *_: None, seed_pack=pack)
    tags = [tag for tag, _ in tracer.calls]
    check("seed gate repairs missing structure before map", tags.index("council.world_repair") < tags.index("council.map"))
    check("repaired whitepaper includes passing audit", result["seed_audit"]["passed"])
    check("every repair receives seed requirements", all('required_causal_rules' in prompt for tag, prompt in tracer.calls if tag == "council.world_repair"))
    check("no council prompt sees evaluator metadata", all("DO_NOT_EXPOSE" not in prompt for _, prompt in tracer.calls))
    bad_tracer = CouncilTracer(bp, always_bad=True)
    try:
        central_office(description, examples, bad_tracer, log=lambda *_: None, seed_pack=pack)
    except WorldBlueprintError:
        bad_tags = [tag for tag, _ in bad_tracer.calls]
        check("failed seed coverage cannot freeze/map", bad_tags.count("council.world_repair") == 5 and "council.map" not in bad_tags)
    else:
        check("failed seed coverage cannot freeze/map", False)
    plain_tracer = CouncilTracer(bp)
    plain = central_office("普通场景", [], plain_tracer, log=lambda *_: None)
    check("unseeded path remains unchanged", "seed_contract" not in plain and len(plain_tracer.calls) == 7)

    observed_pack = fixture()
    observed_pack["blueprint_requirements"]["entity_types"][0]["fields"].append({"name": "issued_on", "kind": "date"})
    class MisclassifiedObserveTracer(CouncilTracer):
        def chat_json(self, tag, messages, **kwargs):
            result = super().chat_json(tag, messages, **kwargs)
            if tag == "council.observe":
                result["observed_fields"] = [
                    {"name": "adopted_report", "kind": "text"},
                    {"name": "issued_on", "kind": "text"},
                    {"name": "amount", "kind": "text", "unit": "元"},
                ]
            return result
    observation_tracer = MisclassifiedObserveTracer(blueprint(observed_pack))
    corrected = central_office("来源包含日期和引用", [], observation_tracer,
                               log=lambda *_: None, seed_pack=observed_pack)
    check("reviewed seed overrides model-inferred reference/date in first pass",
          corrected["seed_audit"]["passed"] and all(tag != "council.world_repair" for tag, _ in observation_tracer.calls))
    raw_fields = corrected["_council_views"]["observe_raw"]["observed_fields"]
    effective_fields = corrected["_council_views"]["observe"]["observed_fields"]
    check("raw observe retained unchanged", all(item["kind"] == "text" for item in raw_fields))
    check("observe overrides record actual schema changes",
          {item["field"] for item in corrected["_council_views"]["seed_observation_overrides"]}
          == {"adopted_report", "issued_on", "amount"}
          and [item["kind"] for item in effective_fields] == ["reference", "date", "numeric"])
    check("seed canonical FK not split into a display field",
          corrected["world_blueprint"]["relation_types"][0]["field"] == "adopted_report"
          and all(field["name"] != "adopted_report引用"
                  for entity in corrected["world_blueprint"]["entity_types"] for field in entity["fields"]))

    for name, passed in checks:
        if not passed:
            print("FAIL:", name)
    print(f"[seed contract self-test] {sum(passed for _, passed in checks)}/{len(checks)} PASS")
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
