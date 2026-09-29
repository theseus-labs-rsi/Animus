"""Seed input snapshots and run identity, shared by all factory entry points.

Seed packs are curated inputs. A run freezes one pack; resuming never rereads
mutable source files or silently switches the world to another seed version.
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import hashlib
import json

from pipeline.seed_pack import (SeedPackError, load_seed_pack, seed_contract, seed_digest,
                                seed_input, validate_seed_pack, generation_context)

SEED_ARTIFACT = "00_seed_pack.json"
AUDIT_ARTIFACT = "00_seed_audit.json"
GENERATION_ARTIFACT = "00_seed_generation_context.json"


def require_raw_seed(pack):
    """Run inputs must preserve v2 extraction evidence, before any files are written."""
    if pack.get("schema_version") == 2 and (
            {"digest", "contract_digest"} & set(pack)
            or not {"document_map", "review"}.issubset(pack)):
        raise SeedPackError("Seed v2 run input requires the full raw seed with document_map/review; "
                            "the internal sanitized seed_contract cannot replace it")


def _v2_snapshots(pack):
    """Bind the full extraction record and the actual generator view separately."""
    require_raw_seed(pack)
    context = generation_context(pack)
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    identity = {"seed_id": pack["seed_id"], "seed_digest": seed_digest(pack)}
    return {
        AUDIT_ARTIFACT: {"version": "seed-extraction-snapshot/v1", **identity,
                        **{key: deepcopy(pack[key]) for key in ("sources", "document_map", "review")},
                        "scope": "Frozen declared extraction evidence; no automatic semantic certification."},
        GENERATION_ARTIFACT: {"version": "seed-generation-context/v1", **identity,
                              "context": context,
                              "context_sha256": hashlib.sha256(encoded).hexdigest()},
    }


def _check_v2_snapshots(run, pack, *, create=False):
    if pack["schema_version"] != 2:
        return
    expected = _v2_snapshots(pack)
    # Validate all existing files before adding any missing snapshot.
    for name, value in expected.items():
        if run.has(name) and run.read(name) != value:
            raise SeedPackError(f"Frozen seed extraction/context snapshot changed: {name}")
        if not create and not run.has(name):
            raise SeedPackError(f"Frozen seed extraction/context snapshot missing: {name}")
    if create:
        for name, value in expected.items():
            if not run.has(name):
                run.write(name, value)


def seed_config(path: str | Path, existing_config: dict | None = None) -> dict:
    """Resolve an explicit CLI seed before creating/mutating a Run manifest."""
    resolved = Path(path).resolve()
    pack = load_seed_pack(resolved)
    require_raw_seed(pack)
    if pack["schema_version"] == 2:
        generation_context(pack)  # Refuse explicit blockers before creating a run.
    digest = seed_digest(pack)
    if existing_config is not None:
        previous = existing_config.get("seed_pack_digest")
        if previous != digest:
            raise SeedPackError(
                "不能更换已有 run 的种子；请新建 run。"
                "原 run 的输入、白皮书和世界必须使用同一份种子。")
    return {"seed_pack_path": str(resolved), "seed_pack_digest": digest,
            "seed_id": pack["seed_id"]}


def frozen_seed(run) -> dict | None:
    """Read and verify a run's frozen pack; absence is allowed only for legacy runs."""
    cfg = run.manifest.get("config") or {}
    expected = cfg.get("seed_pack_digest")
    if not expected:
        if run.has(SEED_ARTIFACT):
            raise SeedPackError("发现种子快照但 run 未绑定种子身份，拒绝降级为普通运行")
        return None
    if not run.has(SEED_ARTIFACT):
        raise SeedPackError("种子快照缺失；请先运行 input 阶段")
    pack = validate_seed_pack(run.read(SEED_ARTIFACT))
    require_raw_seed(pack)
    if seed_digest(pack) != expected or pack["seed_id"] != cfg.get("seed_id"):
        raise SeedPackError("种子快照与 run 身份不一致；拒绝复用旧产物")
    return pack


def prepare_seed_input(run) -> dict | None:
    """Freeze a new pack once, then derive the existing description/few_shot input."""
    cfg = run.manifest.get("config") or {}
    if not cfg.get("seed_pack_digest"):
        if cfg.get("seed_pack_path") or run.has(SEED_ARTIFACT):
            raise SeedPackError("seed 输入必须同时声明 seed_pack_digest 和 seed_id")
        return None
    if not run.has(SEED_ARTIFACT):
        if not cfg.get("seed_pack_path"):
            raise SeedPackError("首次 seed 输入缺少 seed_pack_path")
        pack = load_seed_pack(cfg["seed_pack_path"])
        require_raw_seed(pack)
        if seed_digest(pack) != cfg["seed_pack_digest"] or pack["seed_id"] != cfg.get("seed_id"):
            raise SeedPackError("种子文件在运行配置后发生变化，请使用新 run")
        run.write(SEED_ARTIFACT, pack)
    pack = frozen_seed(run)
    description, few_shot = seed_input(pack)
    _check_v2_snapshots(run, pack, create=True)
    return {"description": description, "few_shot": few_shot,
            "seed": {"seed_id": pack["seed_id"], "family": pack["family"],
                     "digest": seed_digest(pack), "artifact": SEED_ARTIFACT,
                     "schema_version": pack["schema_version"]}}


def validate_seed_input(run, scenario: dict) -> dict | None:
    """Prevent a seeded manifest from consuming an old/unseeded 00_input."""
    pack = frozen_seed(run)
    ref = scenario.get("seed")
    if pack is None:
        if ref:
            raise SeedPackError("input 含 seed，但 run 缺少冻结的种子快照")
        return None
    expected = {"seed_id": pack["seed_id"], "family": pack["family"],
                "digest": seed_digest(pack), "artifact": SEED_ARTIFACT,
                "schema_version": pack["schema_version"]}
    if ref != expected:
        raise SeedPackError("input 与冻结种子身份不一致，请重跑 input")
    description, few_shot = seed_input(pack)
    _check_v2_snapshots(run, pack)
    if scenario.get("description") != description or scenario.get("few_shot") != few_shot:
        raise SeedPackError("input 的场景/样例与冻结种子不一致，请重跑 input")
    return pack


def validate_seed_identity(run, whitepaper: dict) -> dict | None:
    """Fail before expensive downstream work if input and whitepaper diverge."""
    cfg = run.manifest.get("config") or {}
    contract = whitepaper.get("seed_contract")
    if not cfg.get("seed_pack_digest") and not contract:
        if run.has(SEED_ARTIFACT):
            raise SeedPackError("种子快照仍存在，但 run/白皮书丢失种子身份，拒绝降级")
        return None
    if not run.has("00_input.json"):
        raise SeedPackError("种子运行缺少 00_input.json")
    pack = validate_seed_input(run, run.read("00_input.json"))
    if not pack or not isinstance(contract, dict):
        raise SeedPackError("白皮书缺少与 input 配套的 seed_contract")
    validate_seed_pack(contract)
    if contract != seed_contract(pack):
        raise SeedPackError("白皮书与 input 使用了不同种子版本")
    return pack
