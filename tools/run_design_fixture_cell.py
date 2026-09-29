"""Run one bounded E1 design cell from an immutable historical candidate.

This entry uses the normal physical request ledger in run_original_bc_smoke.
Both arms start from the same whitepaper and fifth complete plan draft; only
the design controller differs. The cell stops at the accepted whitepaper.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_fixture(path: Path):
    whitepaper_path = path / "01_whitepaper.json"
    audit_path = path / "01_instance_plan_audit.json"
    raw_wp = whitepaper_path.read_bytes()
    raw_audit = audit_path.read_bytes()
    wp = json.loads(raw_wp)
    history = json.loads(raw_audit)["recovery_history"]
    drafts = history[-1]["transaction"]["drafts"]
    if len(drafts) != 5 or drafts[-1].get("iteration") != 5:
        raise ValueError("E1 requires the historical fifth complete draft")
    proposal = drafts[-1]["candidate"]
    from pipeline.capability_contract import digest
    receipt = {"version": "fixed-design-fixture/v1", "source_directory": str(path.resolve()),
               "whitepaper_sha256": hashlib.sha256(raw_wp).hexdigest(),
               "instance_plan_audit_sha256": hashlib.sha256(raw_audit).hexdigest(),
               "fifth_candidate_hash": digest(proposal),
               "historical_draft_count": len(drafts)}
    return wp, proposal, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--arm", choices=("legacy", "joint"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed-pack", type=Path, required=True)
    parser.add_argument("--delivery-target", type=Path, required=True)
    args = parser.parse_args()
    wp, proposal, fixture = load_fixture(args.fixture)

    from pipeline import factory
    from pipeline.run import Stage
    from pipeline.seed_run import validate_seed_input, validate_seed_identity
    from pipeline.capability_contract import digest
    from pipeline.supply import author_brief

    def design_from_fixture(run):
        sc = run.read("00_input.json")
        pack = validate_seed_input(run, sc)
        if pack is None:
            raise ValueError("E1 must use the frozen seed pack")
        cfg = run.manifest["config"]
        if (cfg.get("production_control") or {}).get("version") != "supply-driven/v6":
            raise ValueError("E1 requires the bounded delivery entry")
        draft = deepcopy(wp)
        validate_seed_identity(run, draft)
        allocation = draft["supply_plan"]["candidate_allocation"]
        if allocation != {line: 6 for line in cfg["delivery_target"]["requested_lines"]}:
            raise ValueError("Fixture candidate allocation differs from the frozen E1 target")
        if draft["supply_plan"]["world_limits"] != cfg["single_pass_world_limits"]:
            raise ValueError("Fixture world limits differ from the frozen E1 target")
        if draft.get("business_instance_plan"):
            raise ValueError("E1 fixture is already committed")
        identity = digest({"fixture": fixture, "arm": args.arm,
                           "input": sc, "config": cfg})
        run.write("01_design_fixture_receipt.json", {**fixture, "arm": args.arm,
                  "run_identity": identity, "initial_proposal_hash": digest(proposal)})
        if args.arm == "joint":
            from pipeline.joint_design import VERSION, MAX_REVISIONS, accept
            audit = {"version": VERSION, "identity": identity,
                     "status": "initial_draft_ready", "draft": deepcopy(draft),
                     "fixture": fixture, "max_revisions": MAX_REVISIONS}
            name = "01_joint_design_audit.json"
            run.write(name, audit)
            brief = author_brief(cfg["delivery_target"], cfg.get("question_budget"),
                                 cfg.get("single_pass_world_limits"),
                                 cfg.get("delivery_survival_rates"))
            revised, receipt = accept(run, draft, pack, brief, audit,
                                      lambda current: run.write(name, current),
                                      initial_proposal=proposal)
            run.write("01_joint_design_receipt.json", receipt)
        else:
            from pipeline.instance_plan import create
            audit = {"version": "legacy-fixed-design-cell/v2", "fixture": fixture,
                     "max_joint_batches": 8}
            name = "01_instance_plan_audit.json"
            try:
                revised = create(draft, run.tracer,
                    {"phase": "fixed_fifth_draft_repair", "fixture": fixture}, audit,
                    max_attempts=8, initial_proposal=proposal,
                    mechanical_attempts=3, max_joint_batches=8,
                    checkpoint=lambda current: run.write(name, current))
            finally:
                run.write(name, audit)
        validate_seed_identity(run, revised)
        run.write("01_supply_plan.json", revised["supply_plan"])
        run.write("01_instance_plan.json", revised["business_instance_plan"])
        if revised.get("seed_audit"):
            run.write("01_seed_audit.json", revised["seed_audit"])
        run.write("01_whitepaper.json", revised)
        run.set_algo(active_lines=[line.get("line") for line in revised.get("active_lines", [])],
                     medium=revised.get("output_medium") or
                            revised.get("domain_profile", {}).get("medium"))

    factory.STAGES[1] = Stage("whitepaper", ["input"], design_from_fixture,
                              "01_whitepaper.json")
    sys.argv = ["tools.run_original_bc_smoke", "--run", args.run,
        "--to", "whitepaper", "--seed-pack", str(args.seed_pack.resolve()),
        "--delivery-target", str(args.delivery_target.resolve()),
        "--question-budget", "42", "--max-world-entities", "32",
        "--time-span-weeks", "6", "--model", "glm-5.3-flash",
        "--max-calls", "20", "--max-cny", "1.5", "--json-attempts", "5",
        "--max-output-tokens", "32768", "--call-timeout-seconds", "600",
        "--llm-concurrency", "1", "--semantic-workers", "1",
        "--disclosure-format-attempts", "4", "--reasoning-effort", "low",
        "--settle-reported-usage", "--haystack-ratio", "0.0",
        "--world-semantic-review", "--process-questions"]
    from tools.run_original_bc_smoke import main as run_bounded
    run_bounded()


if __name__ == "__main__":
    main()
