"""Locate quotations or nodes in unchanged JSON, without judging their meaning.

Pointers are relative to the supplied value (the root pointer is ``""``).
Offsets count Unicode code points in decoded strings or complete JSON literals,
not positions in serialized provider output. A key and its value can share a
pointer; ``source_kind`` distinguishes them. Every possible occurrence is kept.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re


VERSION = "reference-json-locations/v2"


def validate_json_value(value):
    """Reject non-JSON Python values without casting, copying or changing input."""
    active = set()

    def visit(node):
        kind = type(node)
        if node is None or kind in (str, bool, int):
            return
        if kind is float:
            if not math.isfinite(node):
                raise ValueError("Reference JSON numbers must be finite")
            return
        if kind not in (list, dict):
            raise ValueError("Reference must contain only native JSON value types")
        identity = id(node)
        if identity in active:
            raise ValueError("Reference JSON cannot contain a cycle")
        active.add(identity)
        try:
            if kind is dict:
                if any(type(key) is not str for key in node):
                    raise ValueError("Reference JSON object keys must be strings")
                children = node.values()
            else:
                children = node
            for child in children:
                visit(child)
        finally:
            active.remove(identity)

    visit(value)


def _hash(value):
    # This encoding is only for value identity, never a substitute model input.
    # ASCII escapes preserve all original string code points without Unicode
    # normalization. JSON distinguishes numbers, booleans, null and strings.
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def reference_value_hash(value):
    """Hash the actual JSON value; a JSON-looking string is still a string."""
    validate_json_value(value)
    return _hash(value)


def _pointer_child(parent, key):
    return parent + "/" + key.replace("~", "~0").replace("/", "~1")


def locate_reference_target(value, target):
    """Resolve one original-reference target, never a model's replacement text.

    Legacy targets omit ``location_scope`` and supply ``source_quote``. A node
    target supplies only ``location_scope='node'`` and an RFC 6901 pointer,
    relative to the actual answer/rationale value passed by the caller. Node
    receipts contain a copy of the whole original value, not a claimed quote.
    Location establishes attribution only, never semantic correctness.
    """
    validate_json_value(value)
    if type(target) is not dict:
        raise ValueError("Reference target must be an object")
    scope = target.get("location_scope", "quote")
    if scope == "quote":
        if "json_pointer" in target:
            raise ValueError("Quote targets cannot also supply a node pointer")
        return locate_reference_quote(value, target.get("source_quote"))
    if scope != "node":
        raise ValueError("Reference location_scope must be quote or node")
    if "source_quote" in target:
        raise ValueError("Node targets cannot also supply source_quote")
    pointer = target.get("json_pointer")
    if type(pointer) is not str or (pointer and not pointer.startswith("/")):
        raise ValueError("Node target requires an RFC 6901 JSON pointer")
    node = value
    for token in pointer.split("/")[1:] if pointer else []:
        if re.search(r"~(?:[^01]|$)", token):
            raise ValueError("Invalid JSON pointer escape")
        token = re.sub(r"~[01]", lambda match: "~" if match[0] == "~0" else "/", token)
        if type(node) is dict:
            if token not in node:
                raise ValueError("Reference JSON pointer does not exist")
            node = node[token]
        elif type(node) is list:
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                raise ValueError("Reference array pointer needs a canonical nonnegative index")
            index = int(token)
            if index >= len(node):
                raise ValueError("Reference JSON pointer does not exist")
            node = node[index]
        else:
            raise ValueError("Reference JSON pointer cannot traverse a scalar")
    return {"locator_version": VERSION, "reference_value_hash": _hash(value),
            "location_scope": "node", "json_pointer": pointer,
            "source_value": deepcopy(node), "source_value_hash": _hash(node),
            "location_status": "unique"}


def locate_reference_quote(value, quote):
    """Return all literal source locations for a nonempty exact quotation.

    Strings and authored object keys allow substrings, including overlapping
    occurrences. Other terminals match only their complete JSON literal.
    Nonempty objects/arrays are never flattened into synthetic source text.
    Successful location says nothing about a claim's relevance or correctness.
    """
    validate_json_value(value)
    if type(quote) is not str or not quote:
        raise ValueError("Reference quotation must be a nonempty string")
    matches = []

    def locate(node, text, pointer, source_kind, *, whole=False):
        if whole:
            starts = [0] if quote == text else []
        else:
            starts, position = [], 0
            while (position := text.find(quote, position)) >= 0:
                starts.append(position)
                position += 1
        if not starts:
            return
        source_hash = _hash(node)
        for start in starts:
            matches.append({"json_pointer": pointer, "source_kind": source_kind,
                "source_value_hash": source_hash, "start": start, "end": start + len(quote)})

    def visit(node, pointer):
        kind = type(node)
        if kind is str:
            locate(node, node, pointer, "string_value")
        elif kind is dict:
            if not node:
                locate(node, "{}", pointer, "empty_object", whole=True)
            # Sorting affects only receipt order, not the original object or
            # its presentation to a model. Pointers keep the authored keys.
            for key in sorted(node):
                child = _pointer_child(pointer, key)
                locate(key, key, child, "object_key")
                visit(node[key], child)
        elif kind is list:
            if not node:
                locate(node, "[]", pointer, "empty_array", whole=True)
            for index, child in enumerate(node):
                visit(child, _pointer_child(pointer, str(index)))
        else:
            source_kind = "null" if node is None else "boolean" if kind is bool else "number"
            literal = json.dumps(node, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            locate(node, literal, pointer, source_kind, whole=True)

    visit(value, "")
    return {"locator_version": VERSION, "reference_value_hash": _hash(value),
        "source_quote": quote,
        "location_status": "failed" if not matches else "unique" if len(matches) == 1 else "ambiguous",
        "matching_spans": matches}
