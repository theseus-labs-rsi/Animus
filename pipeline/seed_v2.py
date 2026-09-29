"""Version 2 extraction metadata and its explicit generation projection.

Source text is never opened. Semantic declarations remain LLM context; only
the existing executable blueprint vocabulary is used by structural checks.
"""
from __future__ import annotations

from copy import deepcopy

CORE_KEYS = {
    "entity_types": {"id", "noun", "count", "primary", "fields", "cardinality_policy"},
    "relation_types": {"id", "from_type", "to_type", "field", "temporal", "min_count"},
    "event_types": {"id", "label", "roles", "effect_fields", "min_count", "relation_bindings"},
    "causal_rules": {"id", "trigger_event", "effect_event", "delay_sessions", "shared_roles"},
}
FIELD_KEYS = {"name", "kind", "unit", "monotonic", "range", "states"}
METADATA = {"source_refs", "description", "conditions", "basis", "status"}
CLAIM_LISTS = {"must_include", "must_distinguish", "must_not_infer", "sensitive_fields",
               "active_lines", "traps", "conditional_requirements", "unresolved", "synthetic_designs"}
STATUSES = {"accepted", "conditional", "unresolved", "synthetic_design"}


def core_requirements(pack: dict, *, for_validation: bool = False) -> dict:
    """Project explicit executable requirements, with no invented minima/states."""
    source = pack["blueprint_requirements"]
    if pack.get("schema_version") == 1:
        return deepcopy(source)
    if not for_validation:
        require_generation_ready(pack)
    result = {group: [{key: deepcopy(value) for key, value in item.items() if key in keys}
                      for item in source[group]] for group, keys in CORE_KEYS.items()}
    for entity in result["entity_types"]:
        entity["fields"] = [{key: value for key, value in field.items() if key in FIELD_KEYS}
                            for field in entity["fields"]]
    result["evidence_channels"] = deepcopy(source["evidence_channels"])
    result["temporal_model"] = {"min_sessions": source["temporal_model"]["min_sessions"]}
    return result


def validate(pack: dict) -> dict:
    """Validate v2 using the unchanged v1 structural validator as a substrate."""
    from pipeline import seed_pack as sp

    extra = {"document_map", "generation_contract", "review"}
    sp._object(pack, sp._TOP | extra | {"digest", "contract_digest"}, sp._TOP | {"generation_contract"}, "seed_pack")
    sp._canonical(pack)
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 2:
        raise sp.SeedPackError("seed_pack.schema_version must be integer 2")
    is_contract = "digest" in pack or "contract_digest" in pack
    if not is_contract and not {"document_map", "review"}.issubset(pack):
        raise sp.SeedPackError("Raw v2 seeds require document_map and review; only a bound sanitized contract may omit them")
    if is_contract:
        if not all(isinstance(pack.get(key), str) and sp._HASH.fullmatch(pack[key])
                   for key in ("digest", "contract_digest")):
            raise sp.SeedPackError("Seed contract needs both digest and contract_digest SHA256 fields")
        if sp._digest({k: v for k, v in pack.items() if k != "contract_digest"}) != pack["contract_digest"]:
            raise sp.SeedPackError("Seed contract_digest does not match its contents")

    source_keys = {"id", "path", "sha256", "role", "title", "source_kind", "date_scope", "allowed_use", "sensitivity"}
    sources = sp._indexed(pack["sources"], "sources", source_keys,
                          {"id", "path", "sha256", "role", "source_kind"}, nonempty=True)
    for identity, source in sources.items():
        sp._string(source["role"], f"sources.{identity}.role")
        for key in ("source_kind", "sensitivity"):
            if key in source:
                sp._string(source[key], f"sources.{identity}.{key}")
        if "allowed_use" in source:
            sp._strings(source["allowed_use"], f"sources.{identity}.allowed_use")
        if "date_scope" in source:
            if not isinstance(source["date_scope"], dict):
                raise sp.SeedPackError(f"sources.{identity}.date_scope must be an object")
            for key, value in source["date_scope"].items():
                sp._string(key, "date_scope key")
                sp._string(value, f"sources.{identity}.date_scope.{key}")
        if is_contract and source.get("role") == "evaluator_only":
            raise sp.SeedPackError("Sanitized seed contracts cannot contain evaluator_only sources")

    def refs(value, where):
        sp._source_refs(value, sources, where, roles={"corpus", "builder_only"})

    def metadata(item, where):
        if "source_refs" in item:
            refs(item["source_refs"], where + ".source_refs")
        for key in ("description", "basis"):
            if key in item:
                sp._string(item[key], where + "." + key)
        if "conditions" in item:
            sp._strings(item["conditions"], where + ".conditions")
        if "status" in item:
            if not isinstance(item["status"], str) or item["status"] not in STATUSES:
                raise sp.SeedPackError(where + ".status is invalid")
            if item["status"] == "conditional" and not item.get("conditions"):
                raise sp.SeedPackError(where + ": conditional statements require conditions")

    task = sp._object(pack["task"], {"objective", "instructions", "forbidden_inferences", "source_refs"},
                      {"objective", "instructions"}, "task")
    sp._string(task["objective"], "task.objective")
    sp._strings(task["instructions"], "task.instructions", nonempty=True)
    if "forbidden_inferences" in task:
        sp._strings(task["forbidden_inferences"], "task.forbidden_inferences")
    if "source_refs" in task:
        refs(task["source_refs"], "task.source_refs")

    bp = sp._object(pack["blueprint_requirements"], set(CORE_KEYS) | {"evidence_channels", "temporal_model", "state_machines", "metric_definitions"},
                    set(CORE_KEYS) | {"evidence_channels", "temporal_model"}, "blueprint_requirements")
    for group, keys in CORE_KEYS.items():
        for index, item in enumerate(sp._array(bp[group], group)):
            where = f"blueprint_requirements.{group}[{index}]"
            sp._object(item, keys | METADATA, {"id"}, where)
            sp._string(item["id"], where + ".id")
            metadata(item, where)
            if group == "entity_types":
                for n, field in enumerate(sp._array(item.get("fields"), where + ".fields")):
                    fw = f"{where}.fields[{n}]"
                    sp._object(field, FIELD_KEYS | METADATA | {"allowed_values", "evolving"}, {"name", "kind"}, fw)
                    metadata(field, fw)
                    if "evolving" in field and type(field["evolving"]) is not bool:
                        raise sp.SeedPackError(fw + ".evolving must be a boolean")
                    if "allowed_values" in field:
                        values = sp._array(field["allowed_values"], fw + ".allowed_values", nonempty=True)
                        if any(isinstance(value, (dict, list)) for value in values):
                            raise sp.SeedPackError(fw + ".allowed_values must contain JSON scalar values")
                        if len({sp._canonical(value) for value in values}) != len(values):
                            raise sp.SeedPackError(fw + ".allowed_values contains duplicates")
    temporal = bp["temporal_model"]
    sp._object(temporal, {"min_sessions", "time_granularity", "reporting_period", "event_spacing", "allow_retrospective", "description", "source_refs"},
               {"min_sessions"}, "blueprint_requirements.temporal_model")
    for key in ("time_granularity", "reporting_period", "event_spacing", "description"):
        if key in temporal:
            sp._string(temporal[key], "temporal_model." + key)
    if "allow_retrospective" in temporal and type(temporal["allow_retrospective"]) is not bool:
        raise sp.SeedPackError("temporal_model.allow_retrospective must be boolean")
    if "source_refs" in temporal:
        refs(temporal["source_refs"], "temporal_model.source_refs")

    for group in ("state_machines", "metric_definitions"):
        seen = set()
        for index, item in enumerate(sp._array(bp.get(group, []), group)):
            where = f"blueprint_requirements.{group}[{index}]"
            if group == "state_machines":
                keys = {"id", "field", "states", "allowed_transitions", "entity_type"} | METADATA
                required = {"id", "field", "states", "allowed_transitions", "source_refs"}
                identity_key = "id"
            else:
                keys = {"name", "kind", "unit", "definition", "period_basis", "comparators", "entity_type", "formula"} | METADATA
                required = {"name", "kind", "unit", "definition", "period_basis", "comparators", "source_refs"}
                identity_key = "name"
            sp._object(item, keys, required, where)
            metadata(item, where)
            identity = sp._string(item[identity_key], where + "." + identity_key)
            if identity in seen:
                raise sp.SeedPackError(where + ": duplicate identity")
            seen.add(identity)
            if "entity_type" in item:
                sp._string(item["entity_type"], where + ".entity_type")
                if item["entity_type"] not in {entity["id"] for entity in bp["entity_types"]}:
                    raise sp.SeedPackError(where + ".entity_type is unknown")
            if group == "state_machines":
                sp._string(item["field"], where + ".field")
                owners = [entity for entity in bp["entity_types"]
                          if ("entity_type" not in item or entity["id"] == item["entity_type"])
                          and any(field.get("name") == item["field"] for field in entity["fields"])]
                if len(owners) != 1:
                    raise sp.SeedPackError(where + ": state machine field must resolve to one entity type; supply entity_type for ambiguous fields")
                states = sp._strings(item["states"], where + ".states", nonempty=True)
                for transition in sp._array(item["allowed_transitions"], where + ".allowed_transitions"):
                    if not isinstance(transition, list) or len(transition) != 2 or any(x not in states for x in transition):
                        raise sp.SeedPackError(where + ": transitions must be pairs of declared states")
            else:
                for key in ("kind", "unit", "definition", "period_basis", "formula"):
                    if key in item:
                        sp._string(item[key], where + "." + key)
                sp._strings(item["comparators"], where + ".comparators")

    required_keys = {"required_" + key for key in CORE_KEYS}
    for index, mechanism in enumerate(sp._array(pack["mechanisms"], "mechanisms", nonempty=True)):
        where = f"mechanisms[{index}]"
        sp._object(mechanism, {"id", "failure_modes", "capability_hooks"} | METADATA | required_keys,
                   {"id", "description", "source_refs"} | required_keys, where)
        metadata(mechanism, where)
        for key in ("failure_modes", "capability_hooks"):
            if key in mechanism:
                sp._strings(mechanism[key], where + "." + key)

    for index, entry in enumerate(sp._array(pack.get("document_map", []), "document_map")):
        where = f"document_map[{index}]"
        sp._object(entry, {"source_id", "segments"}, {"source_id", "segments"}, where)
        sp._string(entry["source_id"], where + ".source_id")
        if entry["source_id"] not in sources:
            raise sp.SeedPackError(where + ": unknown source_id")
        for segment in sp._array(entry["segments"], where + ".segments"):
            if not isinstance(segment, dict):
                raise sp.SeedPackError(where + ": segments must contain objects")
            sp._string(segment.get("locator"), where + ".segments[].locator")
    review = pack.get("review", {})
    if not isinstance(review, dict):
        raise sp.SeedPackError("review must be an object")
    sp._array(review.get("blocking_issues", []), "review.blocking_issues")

    contract = sp._object(pack.get("generation_contract", {}), CLAIM_LISTS | {"syntheticization"}, set(), "generation_contract")
    for key in CLAIM_LISTS:
        for index, claim in enumerate(sp._array(contract.get(key, []), "generation_contract." + key)):
            where = f"generation_contract.{key}[{index}]"
            if isinstance(claim, str):
                sp._string(claim, where)
            else:
                sp._object(claim, {"id", "text"} | METADATA, {"text", "status", "source_refs"}, where)
                sp._string(claim["text"], where + ".text")
                if "id" in claim:
                    sp._string(claim["id"], where + ".id")
                metadata(claim, where)
    synth = contract.get("syntheticization", {})
    if not isinstance(synth, dict):
        raise sp.SeedPackError("generation_contract.syntheticization must be an object")
    for key, value in synth.items():
        sp._string(key, "syntheticization key")
        sp._string(value, "syntheticization." + key)

    # Reuse the v1 validator for executable referential integrity and exemplars.
    projected = {key: deepcopy(pack[key]) for key in sp._TOP}
    projected["schema_version"] = 1
    projected["sources"] = [{key: value for key, value in item.items()
                              if key in {"id", "path", "sha256", "role", "title"}} for item in pack["sources"]]
    projected["task"] = {"objective": task["objective"], "instructions": "\n".join(task["instructions"])}
    projected["blueprint_requirements"] = core_requirements(pack, for_validation=True)
    projected["mechanisms"] = [{key: value for key, value in item.items()
                                 if key in {"id", "description", "source_refs"} | required_keys}
                                for item in pack["mechanisms"]]
    sp.validate_seed_pack(projected)
    return deepcopy(pack)


def require_generation_ready(pack: dict) -> None:
    from pipeline.seed_pack import SeedPackError
    if pack.get("schema_version") == 2 and (pack.get("review") or {}).get("blocking_issues"):
        raise SeedPackError("Seed review.blocking_issues is nonempty; resolve blockers before generation")
    if pack.get("schema_version") != 2:
        return
    pending = []
    for group in CORE_KEYS:
        for item in pack["blueprint_requirements"][group]:
            if item.get("status") in {"conditional", "unresolved"}:
                pending.append(f"{group}.{item['id']}")
            for field in item.get("fields", []):
                if field.get("status") in {"conditional", "unresolved"}:
                    pending.append(f"{group}.{item['id']}.{field['name']}")
    for item in pack["mechanisms"]:
        if item.get("status") in {"conditional", "unresolved"}:
            pending.append("mechanisms." + item["id"])
    if pending:
        raise SeedPackError("Executable seed requirements need accepted or synthetic_design status; "
                            "resolve these declarations or move their uncertain semantic claims to "
                            "generation_contract before generation: " + ", ".join(pending))


def sanitize_contract(pack: dict) -> dict:
    """Keep generator-safe contract; original audit remains in frozen raw seed."""
    from pipeline import seed_pack as sp
    require_generation_ready(pack)
    result = {key: deepcopy(value) for key, value in pack.items()
              if key not in {"digest", "contract_digest", "document_map", "review"}}
    result["sources"] = [source for source in result["sources"] if source["role"] != "evaluator_only"]
    result["digest"] = pack.get("digest") or sp._digest(pack)
    result["contract_digest"] = sp._digest(result)
    return result


def generation_context(pack: dict) -> dict:
    """Generator-visible material, distinct from audit and grading information."""
    from pipeline import seed_pack as sp
    pack = sp.validate_seed_pack(pack)
    if pack["schema_version"] == 1:
        return sp.seed_contract(pack)
    result = sanitize_contract(pack)
    # Filesystem paths, hashes, sensitivity labels and review prose are audit
    # metadata. Retain IDs for all source_refs and semantic source distinctions.
    result["sources"] = [{key: deepcopy(source[key]) for key in ("id", "role", "source_kind", "date_scope", "allowed_use") if key in source}
                         for source in result["sources"]]
    result.pop("contract_digest", None)
    result["context_policy"] = {
        "accepted": "Apply within its stated scope; provenance does not establish truth.",
        "conditional": "Apply only when all stated conditions hold.",
        "unresolved": "Do not promote to facts or mandatory rules; preserve uncertainty.",
        "synthetic_design": "An explicit synthetic arrangement, not a sourced business obligation.",
        "semantic_metadata": "Interpret using the existing LLM authors and reviewers; not an executable state machine or formula.",
        "public_answerability": "Rules needed to answer a question must appear in public material or protocol.",
    }
    return result
