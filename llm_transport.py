"""Pure, explicit Chat Completions transport configuration.

This module performs no I/O and imports no credentials, SDK or model config.
An absent transport, or a selected ``legacy`` profile, preserves the existing
request-parameter construction. Explicit profiles use the caller's token limit
exactly; they never raise it to an environment-derived minimum.

Profiles do not select models, alter messages, dispatch requests, or retry.
The caller can use resolve_profile() for timeout options and audit identity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

VERSION = "chat-completions-transport/v1"
LEGACY = "legacy"
_OUTER_KEYS = {"version", "profiles", "model_profiles", "default_profile"}
_PROFILE_KEYS = {"token_limit_parameter", "reasoning_effort", "omit_parameters",
                 "response_format", "http_timeout_seconds", "deadline_seconds"}
_OMITTABLE = {"temperature", "top_p", "logprobs", "top_logprobs"}

# Exact, reviewed routes for the bounded original-pipeline runners. Unsupported
# values fail before dispatch; a profile never selects a different model.
RUN_MODEL_CAPABILITIES = {
    "gpt-5.4-mini": {"token_limit_parameter": "max_completion_tokens",
                     "reasoning_efforts": ("low", "medium", "high"),
                     "omit_parameters": ("temperature", "top_p")},
    "glm-5.3-flash": {"token_limit_parameter": "max_tokens",
                      "reasoning_efforts": ("low", "high", "max"), "omit_parameters": ()},
    "glm-4.7-nothinking": {"token_limit_parameter": "max_tokens",
                          "reasoning_efforts": (), "omit_parameters": ()},
}


def original_run_transport(model, reasoning_effort, timeout_seconds):
    """Freeze the real wire options of an explicitly selected economical model."""
    capability = RUN_MODEL_CAPABILITIES.get(model)
    if capability is None:
        raise ValueError("Unsupported original-run model: " + str(model))
    allowed = capability["reasoning_efforts"]
    if allowed and reasoning_effort not in allowed:
        raise ValueError(f"{model} supports reasoning_effort in {allowed}; got {reasoning_effort!r}")
    if not allowed and reasoning_effort != "low":
        raise ValueError("The selected nothinking route has no reasoning override; use the default low option")
    profile = {"token_limit_parameter": capability["token_limit_parameter"],
               "http_timeout_seconds": timeout_seconds, "deadline_seconds": timeout_seconds}
    if allowed:
        profile["reasoning_effort"] = reasoning_effort
    if capability["omit_parameters"]:
        profile["omit_parameters"] = list(capability["omit_parameters"])
    return validate_transport({"version": VERSION, "default_profile": "low",
        "model_profiles": {model: "low"}, "profiles": {"low": profile}})


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    # Exact identifiers are preserved, including case; there is no model heuristic.
    return value


def _positive_seconds(value, label):
    if type(value) not in (int, float):
        raise ValueError(label + " must be a finite positive number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(label + " must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(label + " must be a finite positive number")
    # Canonicalize 300 and 300.0 alike without rounding a large integer timeout.
    return value if type(value) is int else int(result) if result.is_integer() else result


def validate_transport(transport):
    """Return an independent canonical config, or None for the legacy default.

    A config has exactly version/profiles/model_profiles/default_profile.
    Each named profile requires token_limit_parameter; all other profile keys
    are optional. ``legacy`` is a reserved built-in name, never redefinable.
    Omission-list order is irrelevant and canonicalized; duplicates are errors.
    """
    if transport is None:
        return None
    if not isinstance(transport, dict) or set(transport) != _OUTER_KEYS:
        raise ValueError("Transport must contain exactly version, profiles, model_profiles and default_profile")
    if transport["version"] != VERSION:
        raise ValueError("Unsupported transport version")
    profiles, mappings = transport["profiles"], transport["model_profiles"]
    if not isinstance(profiles, dict) or not isinstance(mappings, dict):
        raise ValueError("profiles and model_profiles must be objects")
    normalized = {"version": VERSION, "profiles": {}, "model_profiles": {},
                  "default_profile": _name(transport["default_profile"], "default_profile")}
    for name, profile in profiles.items():
        _name(name, "Profile name")
        if name == LEGACY:
            raise ValueError("The legacy profile cannot be redefined")
        if not isinstance(profile, dict) or not set(profile) <= _PROFILE_KEYS:
            raise ValueError("Unknown profile field or non-object profile: " + name)
        token_parameter = profile.get("token_limit_parameter")
        if not isinstance(token_parameter, str) or token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Profile requires an explicit supported token_limit_parameter")
        result = deepcopy(profile)
        if "reasoning_effort" in profile:
            _name(profile["reasoning_effort"], "reasoning_effort")
        if "omit_parameters" in profile:
            omissions = profile["omit_parameters"]
            if (not isinstance(omissions, list)
                    or any(not isinstance(name, str) or name not in _OMITTABLE for name in omissions)):
                raise ValueError("omit_parameters must list only temperature/top_p/logprobs/top_logprobs")
            if len(set(omissions)) != len(omissions):
                raise ValueError("omit_parameters cannot contain duplicate entries")
            result["omit_parameters"] = sorted(omissions)
        if "response_format" in profile and profile["response_format"] != {"type": "json_object"}:
            raise ValueError("Profile response_format must be exactly {'type': 'json_object'}")
        for field in ("http_timeout_seconds", "deadline_seconds"):
            if field in profile:
                result[field] = _positive_seconds(profile[field], field)
        normalized["profiles"][name] = result
    valid_names = set(profiles) | {LEGACY}
    if normalized["default_profile"] not in valid_names:
        raise ValueError("default_profile does not name a declared or legacy profile")
    for model, name in mappings.items():
        _name(model, "Model identifier")
        _name(name, "Mapped profile name")
        if name not in valid_names:
            raise ValueError("Model mapping refers to an unknown profile: " + name)
        normalized["model_profiles"][model] = name
    return normalized


def transport_fingerprint(transport):
    """Hash the validated whole config, not a model response or semantic result."""
    normalized = validate_transport(transport)
    encoded = json.dumps({"version": VERSION, "transport": normalized}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_profile(transport, model):
    """Resolve an exact model key; unmatched keys use the declared default.

    Returns name, an independent profile (None for legacy), and the fingerprint
    of the entire transport configuration. The actual model remains unchanged.
    """
    _name(model, "Model identifier")
    normalized = validate_transport(transport)
    name = LEGACY if normalized is None else normalized["model_profiles"].get(model, normalized["default_profile"])
    return {"name": name,
            "profile": None if name == LEGACY else deepcopy(normalized["profiles"][name]),
            "fingerprint": transport_fingerprint(normalized)}


def build_request_parameters(*, model, max_tokens=4096, temperature=0.7, top_p=1.0,
                             response_format=None, min_completion_tokens=0, transport=None):
    """Build only request parameters; never include model/messages/SDK options.

    Legacy retains temperature (even None), omits top_p only when None, includes
    a provided response_format, and applies max(max_tokens, minimum). Explicit
    profiles ignore that minimum, optionally omit sampling fields, add only the
    declared reasoning/format options, and reject conflicting response formats.
    Timeout fields remain in resolve_profile()['profile'], outside the request.
    """
    profile = resolve_profile(transport, model)["profile"]
    if profile is None:
        parameters = {"temperature": temperature, "max_tokens": max(max_tokens, min_completion_tokens)}
    else:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("An explicit profile requires a positive integer caller token limit")
        parameters = {"temperature": temperature, profile["token_limit_parameter"]: max_tokens}
    if top_p is not None:
        parameters["top_p"] = top_p
    if response_format is not None:
        parameters["response_format"] = deepcopy(response_format)
    if profile is not None:
        for field in profile.get("omit_parameters", []):
            parameters.pop(field, None)
        if "reasoning_effort" in profile:
            parameters["reasoning_effort"] = profile["reasoning_effort"]
        if "response_format" in profile:
            if response_format is not None and response_format != profile["response_format"]:
                raise ValueError("Caller response_format conflicts with the explicit transport profile")
            parameters["response_format"] = deepcopy(profile["response_format"])
    return parameters
