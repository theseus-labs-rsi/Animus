"""Recognize a provider's explicit rejection of one background document."""


def is_content_rejection(error):
    raw = getattr(error, "raw", error)
    if isinstance(raw, dict):
        kind = (raw.get("__error_metadata__") or {}).get("kind") or raw.get("kind")
        detail = str(raw.get("__error__", raw.get("error", "")))
    else:
        kind = getattr(error, "kind", None)
        detail = str(error)
    return kind == "http_status_400" and any(marker in detail for marker in (
        "SensitiveContentDetected", "input may contain sensitive information"))
