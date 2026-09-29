"""Strict, declared-type comparisons for benchmark gold values.

Parsing consumes the entire value. Legacy fields are inferred only when every
value is a complete ISO date or a number with a consistent unit.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import re
import unicodedata


class ValueComparisonError(ValueError):
    pass


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_LIKE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_NUMBER = re.compile(
    r"(?P<number>[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>[^\d\s+\-.,/]+(?:\s+[^\d\s+\-.,/]+)*)?"
)
_SCALES = {"千": Decimal(1000), "万": Decimal(10000), "亿": Decimal(100000000)}
_LEGACY_UNITS = {"", "%", "次", "个", "人", "件", "份", "级", "分", "分数", "小时", "分钟", "秒", "天", "周", "月", "年", "元", "美元", "人民币", "米", "公里", "千克", "公斤", "吨", "点", "倍"}


def _text(value) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def looks_like_date(value) -> bool:
    """Includes malformed dates, so they never silently become a year."""
    return bool(_DATE_LIKE.match(_text(value)))


def parse_date(value) -> date:
    text = _text(value)
    if not _DATE.fullmatch(text):
        raise ValueComparisonError(f"Expected complete YYYY-MM-DD date: {value!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueComparisonError(f"Invalid calendar date: {value!r}") from exc


def _unit(unit: str) -> tuple[Decimal, str]:
    if unit == "%":
        return Decimal("0.01"), "ratio"
    if unit == "千克":
        return Decimal(1), unit
    for prefix, multiplier in _SCALES.items():
        if unit.startswith(prefix):
            return multiplier, unit[len(prefix):]
    return Decimal(1), unit


def parse_number(value, declared_unit: str | None = None) -> tuple[Decimal, str]:
    if isinstance(value, bool) or value is None or looks_like_date(value):
        raise ValueComparisonError(f"Not a numeric value: {value!r}")
    match = _NUMBER.fullmatch(_text(value))
    if not match:
        raise ValueComparisonError(f"Expected one complete number and unit: {value!r}")
    unit = match.group("unit") or ""
    expected = _text(declared_unit) if declared_unit else ""
    if not unit:
        unit = expected
    scale, dimension = _unit(unit)
    if expected:
        _, expected_dimension = _unit(expected)
        if dimension != expected_dimension:
            raise ValueComparisonError(f"Incompatible unit {unit!r}; expected {expected!r}")
    elif dimension not in _LEGACY_UNITS and dimension != "ratio":
        raise ValueComparisonError(f"Undeclared numeric unit: {unit!r}")
    try:
        number = Decimal(match.group("number").replace(",", "")) * scale
    except InvalidOperation as exc:
        raise ValueComparisonError(f"Invalid numeric value: {value!r}") from exc
    if not number.is_finite():
        raise ValueComparisonError(f"Non-finite numeric value: {value!r}")
    return number, dimension


def field_schema(ws, entity: str, field: str, wp: dict | None = None) -> dict:
    """The world's typed declaration wins over optional legacy profile hints."""
    blueprint = getattr(ws, "world_blueprint", {}) or (wp or {}).get("world_blueprint") or {}
    entity_type = (getattr(ws, "entity_types", {}) or {}).get(entity)
    declarations = [f for t in blueprint.get("entity_types", [])
                    if entity_type is None or t.get("id") == entity_type
                    for f in t.get("fields", []) if f.get("name") == field]
    if declarations:
        first = declarations[0]
        if any(d != first for d in declarations):
            raise ValueComparisonError(f"Ambiguous field declaration: {field}")
        return dict(first)
    for declaration in (wp or {}).get("domain_profile", {}).get("field_schema", []):
        if declaration.get("name") == field:
            return dict(declaration)
    return {}


def comparison_keys(values: list, schema: dict | None = None) -> tuple[str, list]:
    if not values:
        raise ValueComparisonError("No comparable values")
    declaration = schema or {}
    kind = declaration.get("kind")
    if kind in ("date", "calendar_date"):
        return "date", [parse_date(value) for value in values]
    if kind and kind not in ("numeric", "number"):
        raise ValueComparisonError(f"Extrema are undefined for declared kind {kind!r}")
    if not kind and any(looks_like_date(value) for value in values):
        return "date", [parse_date(value) for value in values]
    numbers = [parse_number(value, declaration.get("unit")) for value in values]
    if len({dimension for _, dimension in numbers}) != 1:
        raise ValueComparisonError("Cannot compare values with different units")
    return "numeric", [number for number, _ in numbers]


def values_equal(expected, actual, schema: dict | None = None) -> bool:
    """Compare complete typed scalars, retaining units and percent semantics."""
    declaration = schema or {}
    kind = declaration.get("kind")
    try:
        if kind in ("date", "calendar_date") or (not kind and (looks_like_date(expected) or looks_like_date(actual))):
            return parse_date(expected) == parse_date(actual)
        if kind in ("numeric", "number"):
            return parse_number(expected, declaration.get("unit")) == parse_number(actual, declaration.get("unit"))
        if not kind:
            try:
                return parse_number(expected) == parse_number(actual)
            except ValueComparisonError:
                pass
        return _text(expected) == _text(actual)
    except ValueComparisonError:
        return False
