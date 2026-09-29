"""Curated real-task seeds for the existing input -> whitepaper stage.

A seed is a reviewed, declarative JSON contract, not an instruction to execute
an attachment. Source paths are provenance metadata (repository-relative by
convention); this module never opens them. ``load_seed_pack`` reads only the
specified JSON. Raw evaluator material never enters a generation prompt.

Blueprint requirements are structural subsets: named fields/objects must be
retained, entity counts and event/relation minima are lower bounds. This checks
structural coverage, not the truth of a model's interpretation of prose.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint


class SeedPackError(ValueError):
    """A seed, its identity, or the required blueprint structure is invalid."""


_GROUPS = ("entity_types", "relation_types", "event_types", "causal_rules")
_TOP = {"schema_version", "seed_id", "family", "title", "description", "sources",
        "task", "exemplars", "mechanisms", "blueprint_requirements"}
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_SEED_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _canonical(value: dict) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SeedPackError(f"Seed must contain finite JSON values: {error}") from error


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(value, allowed: set[str], required: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise SeedPackError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SeedPackError(f"{where} must use string keys")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise SeedPackError(f"{where}: unknown fields={sorted(unknown)}, missing={sorted(missing)}")
    return value


def _string(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SeedPackError(f"{where} must be a nonempty, trimmed string")
    return value


def _array(value, where: str, *, nonempty=False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        raise SeedPackError(f"{where} must be {'a nonempty' if nonempty else 'an'} array")
    return value


def _integer(value, minimum: int, where: str) -> int:
    if type(value) is not int or value < minimum:
        raise SeedPackError(f"{where} must be an integer >= {minimum}")
    return value


def _strings(value, where: str, *, nonempty=False) -> list[str]:
    result = [_string(item, f"{where}[{index}]")
              for index, item in enumerate(_array(value, where, nonempty=nonempty))]
    if len(set(result)) != len(result):
        raise SeedPackError(f"{where} contains duplicates")
    return result


def _indexed(value, where: str, allowed: set[str], required: set[str], *, nonempty=False) -> dict:
    result = {}
    for index, item in enumerate(_array(value, where, nonempty=nonempty)):
        _object(item, allowed, required, f"{where}[{index}]")
        identity = _string(item.get("id"), f"{where}[{index}].id")
        if identity in result:
            raise SeedPackError(f"{where} contains duplicate id: {identity}")
        result[identity] = item
    return result


def _source_refs(value, sources: dict, where: str, *, roles: set[str], nonempty=False) -> None:
    seen = set()
    for index, ref in enumerate(_array(value, where, nonempty=nonempty)):
        _object(ref, {"source_id", "locator"}, {"source_id", "locator"}, f"{where}[{index}]")
        source_id = _string(ref["source_id"], f"{where}[{index}].source_id")
        locator = _string(ref["locator"], f"{where}[{index}].locator")
        if source_id not in sources or sources[source_id]["role"] not in roles:
            raise SeedPackError(f"{where}: missing or disallowed source: {source_id}")
        if (source_id, locator) in seen:
            raise SeedPackError(f"{where}: duplicate source reference: {source_id}/{locator}")
        seen.add((source_id, locator))


def _validate_bindings(indexed: dict[str, dict]) -> None:
    """Bind event participants to FK endpoints and causal business identities."""
    for identity, event in indexed["event_types"].items():
        if "relation_bindings" not in event:
            continue
        where = f"event_types.{identity}.relation_bindings"
        seen = set()
        roles = event.get("roles") or {}
        if not isinstance(roles, dict):
            raise SeedPackError(f"{where}: event roles must be an object")
        for binding in _array(event["relation_bindings"], where, nonempty=True):
            keys = {"relation", "from_role", "to_role"}
            _object(binding, keys, keys, where + "[]")
            for key in keys:
                _string(binding[key], where + "[]." + key)
            key = (binding["relation"], binding["from_role"], binding["to_role"])
            if key in seen:
                raise SeedPackError(f"{where} contains duplicate binding: {key}")
            seen.add(key)
            relation = indexed["relation_types"].get(binding["relation"])
            if relation is None:
                raise SeedPackError(f"{where} references missing relation: {binding['relation']}")
            for side in ("from", "to"):
                role = binding[side + "_role"]
                if role not in roles or roles[role] != relation.get(side + "_type"):
                    raise SeedPackError(f"{where}: {side}_role {role!r} must match relation {side}_type")
    for identity, rule in indexed["causal_rules"].items():
        if "shared_roles" not in rule:
            continue
        where = f"causal_rules.{identity}.shared_roles"
        roles = _strings(rule["shared_roles"], where, nonempty=True)
        _string(rule.get("trigger_event"), f"causal_rules.{identity}.trigger_event")
        _string(rule.get("effect_event"), f"causal_rules.{identity}.effect_event")
        trigger = indexed["event_types"].get(rule.get("trigger_event"))
        effect = indexed["event_types"].get(rule.get("effect_event"))
        if trigger is None or effect is None:
            raise SeedPackError(f"{where} references missing trigger/effect event")
        trigger_roles, effect_roles = trigger.get("roles") or {}, effect.get("roles") or {}
        if not isinstance(trigger_roles, dict) or not isinstance(effect_roles, dict):
            raise SeedPackError(f"{where}: both events must have object roles")
        for role in roles:
            if role not in trigger_roles or role not in effect_roles or trigger_roles[role] != effect_roles[role]:
                raise SeedPackError(f"{where}: {role!r} must exist with the same type in both events")


def _validate_requirements(value: dict) -> dict[str, dict]:
    _object(value, set(_GROUPS) | {"evidence_channels", "temporal_model"},
            set(_GROUPS) | {"evidence_channels", "temporal_model"}, "blueprint_requirements")
    specifications = {
        "entity_types": ({"id", "noun", "count", "primary", "fields", "cardinality_policy"}, {"id", "fields"}),
        "relation_types": ({"id", "from_type", "to_type", "field", "temporal", "min_count"},
                           {"id", "from_type", "to_type", "field", "temporal", "min_count"}),
        "event_types": ({"id", "label", "roles", "effect_fields", "min_count", "relation_bindings"},
                        {"id", "label", "roles", "effect_fields", "min_count"}),
        "causal_rules": ({"id", "trigger_event", "effect_event", "delay_sessions", "shared_roles"},
                         {"id", "trigger_event", "effect_event", "delay_sessions"}),
    }
    indexed = {key: _indexed(value[key], f"blueprint_requirements.{key}", *specifications[key],
                             nonempty=key == "entity_types") for key in _GROUPS}
    for identity, entity in indexed["entity_types"].items():
        where = f"entity_types.{identity}"
        if "noun" in entity:
            _string(entity["noun"], where + ".noun")
        if "count" in entity:
            _integer(entity["count"], 1, where + ".count")
        if "primary" in entity and type(entity["primary"]) is not bool:
            raise SeedPackError(f"{where}.primary must be a boolean")
        if "cardinality_policy" in entity and entity["cardinality_policy"] != "exact":
            raise SeedPackError(f"{where}.cardinality_policy only supports exact")
        if entity.get("cardinality_policy") == "exact" and "count" not in entity:
            raise SeedPackError(f"{where}.count is required for exact cardinality")
        names = set()
        for field in _array(entity["fields"], where + ".fields"):
            _object(field, {"name", "kind", "unit", "monotonic", "range", "states"},
                    {"name", "kind"}, where + ".fields[]")
            name = _string(field["name"], where + ".fields[].name")
            _string(field["kind"], where + ".fields[].kind")
            if name in names:
                raise SeedPackError(f"{where} contains duplicate field: {name}")
            names.add(name)
            for attr in ("unit", "monotonic"):
                if attr in field:
                    _string(field[attr], f"{where}.{name}.{attr}")
            if "states" in field:
                _strings(field["states"], f"{where}.{name}.states", nonempty=True)
            if "range" in field:
                bounds = _array(field["range"], f"{where}.{name}.range")
                if len(bounds) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds):
                    raise SeedPackError(f"{where}.{name}.range must contain two finite numbers")
    for identity, relation in indexed["relation_types"].items():
        for key in ("from_type", "to_type", "field"):
            _string(relation[key], f"relation_types.{identity}.{key}")
        if type(relation["temporal"]) is not bool:
            raise SeedPackError(f"relation_types.{identity}.temporal must be a boolean")
        _integer(relation["min_count"], 1, f"relation_types.{identity}.min_count")
    for identity, event in indexed["event_types"].items():
        _string(event["label"], f"event_types.{identity}.label")
        _integer(event["min_count"], 1, f"event_types.{identity}.min_count")
        if not isinstance(event["roles"], dict) or not event["roles"]:
            raise SeedPackError(f"event_types.{identity}.roles must be a nonempty object")
        for role, target in event["roles"].items():
            _string(role, f"event_types.{identity}.role")
            _string(target, f"event_types.{identity}.roles.{role}")
        for effect in _array(event["effect_fields"], f"event_types.{identity}.effect_fields", nonempty=True):
            _object(effect, {"role", "field"}, {"role", "field"}, f"event_types.{identity}.effect")
            _string(effect["role"], f"event_types.{identity}.effect.role")
            _string(effect["field"], f"event_types.{identity}.effect.field")
    for identity, rule in indexed["causal_rules"].items():
        _string(rule["trigger_event"], f"causal_rules.{identity}.trigger_event")
        _string(rule["effect_event"], f"causal_rules.{identity}.effect_event")
        _integer(rule["delay_sessions"], 0, f"causal_rules.{identity}.delay_sessions")
    _validate_bindings(indexed)
    _strings(value["evidence_channels"], "blueprint_requirements.evidence_channels", nonempty=True)
    temporal = _object(value["temporal_model"], {"min_sessions"}, {"min_sessions"},
                       "blueprint_requirements.temporal_model")
    _integer(temporal["min_sessions"], 2, "blueprint_requirements.temporal_model.min_sessions")

    # Validate reference closure and executable field semantics using the same
    # validator as whitepapers. Unspecified counts may grow in the final world;
    # use sufficient temporary capacity instead of inventing a hard seed limit.
    candidate = deepcopy(value)
    capacity = max([100] + [item.get("count", 1) for item in value["entity_types"]]
                   + [item["min_count"] for key in ("relation_types", "event_types") for item in value[key]])
    primaries = [entity for entity in candidate["entity_types"] if entity.get("primary") is True]
    if not primaries:
        eligible = [entity for entity in candidate["entity_types"] if "primary" not in entity]
        if not eligible:
            raise SeedPackError("Seed requirements cannot forbid every primary entity type")
        eligible[0]["primary"] = True
    for entity in candidate["entity_types"]:
        entity.setdefault("noun", entity["id"])
        if entity.get("cardinality_policy") != "exact":
            entity["count"] = capacity
    sessions = temporal["min_sessions"]
    if any(rule["delay_sessions"] >= sessions for rule in value["causal_rules"]):
        raise SeedPackError("causal rule delay_sessions must be below temporal_model.min_sessions")
    candidate["temporal_model"] = {"unit": "session", "cadence": "daily", "step_days": 1,
                                   "n_sessions": sessions}
    try:
        normalize_world_blueprint(candidate)
    except WorldBlueprintError as error:
        raise SeedPackError(f"Invalid seed blueprint requirements: {error}") from error
    return indexed


def validate_seed_pack(pack: dict) -> dict:
    """Validate and copy a v1 pack or an attached, sanitized seed contract.

    ``digest`` identifies the original pack; ``contract_digest`` checks the
    exact sanitized snapshot, including its original identity. Neither hash is
    a signature or a claim of independent verification of source contents.
    """
    if isinstance(pack, dict) and type(pack.get("schema_version")) is int and pack["schema_version"] == 2:
        from pipeline.seed_v2 import validate
        return validate(pack)
    _object(pack, _TOP | {"digest", "contract_digest"}, _TOP, "seed_pack")
    _canonical(pack)
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 1:
        raise SeedPackError("seed_pack.schema_version must be integer 1")
    for key in ("seed_id", "family", "title", "description"):
        _string(pack[key], "seed_pack." + key)
    if not _SEED_ID.fullmatch(pack["seed_id"]):
        raise SeedPackError("seed_pack.seed_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
    is_contract = "digest" in pack or "contract_digest" in pack
    if is_contract:
        if not all(isinstance(pack.get(key), str) and _HASH.fullmatch(pack[key])
                   for key in ("digest", "contract_digest")):
            raise SeedPackError("Seed contract needs both digest and contract_digest SHA256 fields")
        actual = _digest({key: value for key, value in pack.items() if key != "contract_digest"})
        if actual != pack["contract_digest"]:
            raise SeedPackError("Seed contract_digest does not match its contents")
    sources = _indexed(pack["sources"], "sources", {"id", "path", "sha256", "role", "title"},
                       {"id", "path", "sha256", "role"}, nonempty=True)
    for identity, source in sources.items():
        for key in ("path", "role"):
            _string(source[key], f"sources.{identity}.{key}")
        if "title" in source:
            _string(source["title"], f"sources.{identity}.title")
        if not isinstance(source["sha256"], str) or not _HASH.fullmatch(source["sha256"]):
            raise SeedPackError(f"sources.{identity}.sha256 must be a SHA256 hex digest")
        if source["role"] not in {"corpus", "builder_only", "evaluator_only"}:
            raise SeedPackError(f"sources.{identity}.role is invalid")
        if is_contract and source["role"] == "evaluator_only":
            raise SeedPackError("Sanitized seed contracts cannot contain evaluator_only sources")
    task = _object(pack["task"], {"objective", "instructions"}, {"objective", "instructions"}, "task")
    for key, value in task.items():
        _string(value, "task." + key)
    requirements = _validate_requirements(pack["blueprint_requirements"])
    required_keys = {"required_" + key for key in _GROUPS}
    mechanisms = _indexed(pack["mechanisms"], "mechanisms",
                          {"id", "description", "source_refs"} | required_keys,
                          {"id", "description", "source_refs"} | required_keys, nonempty=True)
    for identity, mechanism in mechanisms.items():
        _string(mechanism["description"], f"mechanisms.{identity}.description")
        _source_refs(mechanism["source_refs"], sources, f"mechanisms.{identity}.source_refs",
                     roles={"corpus", "builder_only"}, nonempty=True)
        total = 0
        for group in _GROUPS:
            refs = _strings(mechanism["required_" + group], f"mechanisms.{identity}.required_{group}")
            total += len(refs)
            if set(refs) - set(requirements[group]):
                raise SeedPackError(f"mechanisms.{identity}.required_{group} has dangling references")
        if not total:
            raise SeedPackError(f"mechanisms.{identity} must require at least one structural element")
    seen_titles = set()
    for index, example in enumerate(_array(pack["exemplars"], "exemplars")):
        where = f"exemplars[{index}]"
        keys = {"title", "content", "doc_type", "date", "origin", "source_refs", "mechanism_refs"}
        _object(example, keys, keys, where)
        for key in ("title", "content", "doc_type", "date", "origin"):
            _string(example[key], where + "." + key)
        if example["title"] in seen_titles:
            raise SeedPackError(f"{where}: duplicate title")
        seen_titles.add(example["title"])
        try:
            if date.fromisoformat(example["date"]).isoformat() != example["date"]:
                raise ValueError()
        except ValueError as error:
            raise SeedPackError(f"{where}.date must be YYYY-MM-DD") from error
        refs = _strings(example["mechanism_refs"], where + ".mechanism_refs")
        if set(refs) - set(mechanisms):
            raise SeedPackError(f"{where}.mechanism_refs has dangling references")
        if example["origin"] == "source":
            _source_refs(example["source_refs"], sources, where + ".source_refs", roles={"corpus"}, nonempty=True)
        elif example["origin"] == "synthetic":
            if example["source_refs"] != [] or not refs:
                raise SeedPackError(f"{where}: synthetic exemplars require empty source_refs and nonempty mechanism_refs")
        else:
            raise SeedPackError(f"{where}.origin must be source or synthetic")
    return deepcopy(pack)


def load_seed_pack(path: str | Path) -> dict:
    """Load only the seed JSON, rejecting ambiguous duplicate JSON keys."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SeedPackError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        pack = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SeedPackError(f"Cannot read seed pack {path}: {error}") from error
    return validate_seed_pack(pack)


def seed_digest(pack: dict) -> str:
    pack = validate_seed_pack(pack)
    return pack.get("digest") or _digest(pack)


def seed_contract(pack: dict) -> dict:
    """Build the exact sanitized snapshot for provenance/resume comparisons."""
    pack = validate_seed_pack(pack)
    if pack["schema_version"] == 2:
        from pipeline.seed_v2 import sanitize_contract
        return sanitize_contract(pack)
    if "contract_digest" in pack:
        return pack
    identity = _digest(pack)
    contract = {key: deepcopy(pack[key]) for key in _TOP}
    contract["sources"] = [source for source in contract["sources"] if source["role"] != "evaluator_only"]
    contract["digest"] = identity
    contract["contract_digest"] = _digest(contract)
    return contract


def seed_context(pack: dict) -> str:
    """Allowlisted generation context; source files are never read or included."""
    return json.dumps(generation_context(pack), ensure_ascii=False, indent=2, allow_nan=False)


def generation_context(pack: dict) -> dict:
    """Validated generator projection; excludes v2 audit and evaluator inputs."""
    from pipeline.seed_v2 import generation_context as project
    return project(pack)


def core_requirements(pack: dict) -> dict:
    """Explicit executable requirements, excluding semantic-only v2 metadata."""
    from pipeline.seed_v2 import core_requirements as project
    return project(validate_seed_pack(pack))


def seed_input(pack: dict) -> tuple[str, list[dict]]:
    """Return a CLI-ready scene description and existing-format few-shot docs."""
    pack = validate_seed_pack(pack)
    if pack["schema_version"] == 2:
        from pipeline.seed_v2 import require_generation_ready
        require_generation_ready(pack)
    task = pack["task"]
    instructions = task["instructions"]
    if pack["schema_version"] == 2:
        instructions = "\n".join(instructions)
    description = f"{pack['description']}\n\n任务目标：{task['objective']}\n任务说明：{instructions}"
    examples = [{"title": item["title"], "content": item["content"],
                 "doc_type": item["doc_type"], "date": item["date"],
                 "seed_origin": item["origin"], "source_refs": deepcopy(item["source_refs"]),
                 "mechanism_refs": deepcopy(item["mechanism_refs"])} for item in pack["exemplars"]]
    return description, examples


def _subset_issues(expected, actual, path: str) -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: required object missing"]
        issues = []
        for key, value in expected.items():
            child = f"{path}.{key}"
            if key in {"count", "min_count"}:
                found = actual.get(key)
                exact = key == "count" and expected.get("cardinality_policy") == "exact"
                if type(found) is not int or (found != value if exact else found < value):
                    issues.append(f"{child}: required {'=' if exact else '>='}{value}, got {found!r}")
            else:
                issues.extend(_subset_issues(value, actual.get(key), child))
        return issues
    if isinstance(expected, list) and path.endswith(".shared_roles"):
        if not isinstance(actual, list) or any(role not in actual for role in expected):
            return [f"{path}: required shared roles {expected!r}, got {actual!r}"]
        return []
    if isinstance(expected, list) and path.endswith((".fields", ".effect_fields", ".relation_bindings")):
        if not isinstance(actual, list):
            return [f"{path}: required array missing"]
        issues = []
        for item in expected:
            identity = (("name",) if "name" in item else
                        ("relation", "from_role", "to_role") if "relation" in item else ("role", "field"))
            match = next((candidate for candidate in actual if isinstance(candidate, dict)
                          and all(candidate.get(key) == item[key] for key in identity)), None)
            label = "/".join(str(item[key]) for key in identity)
            issues.extend(_subset_issues(item, match, f"{path}[{label}]"))
        return issues
    if type(actual) is not type(expected) or actual != expected:
        # JSON int and float equality is valid for numeric range endpoints.
        if type(expected) in (int, float) and type(actual) in (int, float) and actual == expected:
            return []
        return [f"{path}: required {expected!r}, got {actual!r}"]
    return []


def validate_seed_blueprint(wp_or_bp: dict, pack: dict) -> dict:
    """Report deterministic structural retention; never claim prose semantics."""
    pack = validate_seed_pack(pack)
    issues = []
    try:
        blueprint = normalize_world_blueprint(wp_or_bp)
    except (WorldBlueprintError, TypeError, AttributeError) as error:
        blueprint = wp_or_bp.get("world_blueprint", wp_or_bp) if isinstance(wp_or_bp, dict) else {}
        issues.append(f"Invalid executable world blueprint: {error}")
    if not isinstance(blueprint, dict):
        blueprint = {}
    # Existing world normalization preserves extension keys. Validate all such
    # bindings, including legitimate additions beyond the seed's required subset.
    try:
        _validate_bindings({group: {item["id"]: item for item in
                                   (blueprint.get(group) if isinstance(blueprint.get(group), list) else [])
                                   if isinstance(item, dict) and isinstance(item.get("id"), str)}
                            for group in _GROUPS})
    except SeedPackError as error:
        issues.append(f"Invalid business-object bindings: {error}")
    requirements = core_requirements(pack)
    coverage = {}
    for group in _GROUPS:
        values = blueprint.get(group)
        actual = {item.get("id"): item for item in values if isinstance(item, dict)
                  and isinstance(item.get("id"), str)} if isinstance(values, list) else {}
        coverage[group] = {}
        for requirement in requirements[group]:
            local = _subset_issues(requirement, actual.get(requirement["id"]), f"{group}[{requirement['id']}]")
            coverage[group][requirement["id"]] = not local
            issues.extend(local)
    actual_channels = blueprint.get("evidence_channels")
    actual_channels = actual_channels if isinstance(actual_channels, list) else []
    for channel in requirements["evidence_channels"]:
        if channel not in actual_channels:
            issues.append(f"evidence_channels: missing {channel!r}")
    minimum = requirements["temporal_model"]["min_sessions"]
    temporal = blueprint.get("temporal_model")
    sessions = temporal.get("n_sessions") if isinstance(temporal, dict) else None
    if type(sessions) is not int or sessions < minimum:
        issues.append(f"temporal_model.n_sessions: required >={minimum}, got {sessions!r}")
    mechanisms = [{"id": mechanism["id"], "passed": all(
        coverage[group].get(identity, False) for group in _GROUPS
        for identity in mechanism["required_" + group])} for mechanism in pack["mechanisms"]]
    return {"schema_version": 1, "seed_id": pack["seed_id"], "seed_digest": seed_digest(pack),
            "scope": "structural_blueprint_retention", "passed": not issues,
            "issues": issues, "requirement_coverage": coverage, "mechanism_coverage": mechanisms}


def attach_seed_contract(wp: dict, pack: dict) -> dict:
    """Return a validated whitepaper with a sanitized, self-checking snapshot."""
    report = validate_seed_blueprint(wp, pack)
    if not report["passed"]:
        raise SeedPackError("Seed blueprint requirements were not retained:\n- " + "\n- ".join(report["issues"]))
    result = deepcopy(wp)
    result["seed_contract"] = seed_contract(pack)
    result["seed_audit"] = report
    return result
