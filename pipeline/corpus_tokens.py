"""Reproducible corpus-body token measurement and bounded filler planning.

Each document body is encoded separately, including its whitespace, then the
counts are summed. Titles, protocol text and JSON serialization are outside this
metric. The caller owns cache persistence; this module never writes corpus files.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from importlib import metadata
import json
import math


DEFAULT_ENCODING = "cl100k_base"
PINNED_TIKTOKEN_VERSION = "0.12.0"
COUNTING_MODE = "sum_document_bodies/ordinary_text/v1"
CACHE_VERSION = "corpus-token-cache/v1"
MEASUREMENT_VERSION = "corpus-token-measurement/v1"


class TokenizerDependencyError(RuntimeError):
    """The explicitly selected tokenizer cannot be loaded reproducibly."""


def _digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _nonnegative_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _encoding_fingerprint(encoding) -> str:
    """Bind the actual encoding tables and pretokenizer, not just its label."""
    digest = hashlib.sha256()
    digest.update(encoding._pat_str.encode("utf-8"))
    digest.update(b"\0mergeable-ranks\0")
    for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda p: p[1]):
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)
        digest.update(rank.to_bytes(8, "big"))
    digest.update(b"\0special-tokens\0")
    digest.update(_digest(encoding._special_tokens).encode("ascii"))
    return digest.hexdigest()


class CorpusTokenCounter:
    """Count with a named, version-pinned tokenizer; never use character fallback.

    Loading an uncached tiktoken vocabulary may download its official data file.
    Deployment preflight should load the encoding before an offline acceptance.
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING,
                 expected_version: str = PINNED_TIKTOKEN_VERSION):
        if not isinstance(encoding_name, str) or not encoding_name.strip():
            raise ValueError("encoding_name must be an explicit encoding name")
        if not isinstance(expected_version, str) or not expected_version.strip():
            raise ValueError("expected_version must explicitly pin tiktoken")
        try:
            installed = metadata.version("tiktoken")
            if installed != expected_version:
                raise TokenizerDependencyError(
                    f"Corpus counting requires tiktoken=={expected_version}; "
                    f"installed={installed}. Install the pinned dependency.")
            module = importlib.import_module("tiktoken")
            self._encoding = module.get_encoding(encoding_name)
        except TokenizerDependencyError:
            raise
        except Exception as error:
            raise TokenizerDependencyError(
                f"Cannot load tiktoken=={expected_version}/{encoding_name}: "
                f"{type(error).__name__}: {error}. Install tiktoken=={expected_version} "
                "and prepare its encoding data before counting.") from error
        self.tokenizer = {
            "library": "tiktoken", "version": installed,
            "encoding": self._encoding.name,
            "encoding_sha256": _encoding_fingerprint(self._encoding),
            "counting_mode": COUNTING_MODE,
        }
        self.tokenizer_id = _digest(self.tokenizer)

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise ValueError("Document content must be a string")
        text.encode("utf-8")  # Reject invalid Unicode rather than silently replacing it.
        return len(self._encoding.encode_ordinary(text))

    def _cache_entry(self, content_hash: str, characters: int, tokens: int) -> dict:
        entry = {"content_sha256": content_hash, "characters": characters,
                 "tokens": tokens, "tokenizer_id": self.tokenizer_id}
        return {**entry, "entry_sha256": _digest(entry)}

    def _valid_entry(self, value, content_hash: str, characters: int) -> bool:
        if not isinstance(value, dict):
            return False
        if set(value) != {"content_sha256", "characters", "tokens",
                          "tokenizer_id", "entry_sha256"}:
            return False
        if (value["content_sha256"] != content_hash
                or value["characters"] != characters
                or type(value["characters"]) is not int
                or type(value["tokens"]) is not int or value["tokens"] < 0
                or value["tokenizer_id"] != self.tokenizer_id):
            return False
        expected = self._cache_entry(content_hash, characters, value["tokens"])
        return value == expected

    def measure(self, corpus: dict, cache: dict | None = None) -> dict:
        """Return actual counts, per-body hashes and a replacement cache.

        Accepts a corpus with ``sessions[].docs[]`` or its saved ``corpus`` wrapper.
        Cache checks cover version, encoding, body hash and receipt integrity.
        Receipts detect stale or damaged local caches; callers must keep caches
        in trusted local storage. A caller can omit cache for a fresh recount.
        """
        if not isinstance(corpus, dict):
            raise ValueError("corpus must be an object")
        if "corpus" in corpus:
            corpus = corpus["corpus"]
        if not isinstance(corpus, dict) or not isinstance(corpus.get("sessions"), list):
            raise ValueError("corpus.sessions must be an array")
        cached = {}
        if (isinstance(cache, dict) and cache.get("version") == CACHE_VERSION
                and cache.get("tokenizer") == self.tokenizer
                and cache.get("tokenizer_id") == self.tokenizer_id
                and isinstance(cache.get("entries"), dict)):
            cached = cache["entries"]
        entries, documents = {}, []
        stats = {name: {"documents": 0, "tokens": 0, "characters": 0}
                 for name in ("total", "core", "filler")}
        reused = encoded = 0
        for session_index, session in enumerate(corpus["sessions"]):
            if not isinstance(session, dict) or not isinstance(session.get("docs"), list):
                raise ValueError(f"sessions[{session_index}].docs must be an array")
            session_id = session.get("session_id", session_index)
            if not isinstance(session_id, (int, str)) or isinstance(session_id, bool):
                raise ValueError("session_id must be a string or integer")
            for document_index, document in enumerate(session["docs"]):
                if not isinstance(document, dict) or not isinstance(document.get("content"), str):
                    raise ValueError(f"sessions[{session_index}].docs[{document_index}].content must be text")
                if type(document.get("is_filler", False)) is not bool:
                    raise ValueError("is_filler must be boolean when present")
                doc_id = document.get("doc_id")
                if doc_id is not None and not isinstance(doc_id, str):
                    raise ValueError("doc_id must be text when present")
                text = document["content"]
                characters = len(text)
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                entry = entries.get(content_hash, cached.get(content_hash))
                if self._valid_entry(entry, content_hash, characters):
                    tokens = entry["tokens"]
                    reused += 1
                else:
                    tokens = self.count_text(text)
                    encoded += 1
                entries[content_hash] = self._cache_entry(content_hash, characters, tokens)
                is_filler = document.get("is_filler", False)
                row = {"session_index": session_index, "session_id": session_id,
                       "document_index": document_index, "doc_id": doc_id,
                       "is_filler": is_filler, "characters": characters,
                       "tokens": tokens, "content_sha256": content_hash}
                documents.append(row)
                for group in ("total", "filler" if is_filler else "core"):
                    stats[group]["documents"] += 1
                    stats[group]["tokens"] += tokens
                    stats[group]["characters"] += characters
        return {
            "version": MEASUREMENT_VERSION, "tokenizer": deepcopy(self.tokenizer),
            "tokenizer_id": self.tokenizer_id, "counting_mode": COUNTING_MODE,
            "corpus_sha256": _digest(documents), "documents": documents,
            **stats, "cache_reused_documents": reused, "encoded_documents": encoded,
            "cache": {"version": CACHE_VERSION, "tokenizer": deepcopy(self.tokenizer),
                      "tokenizer_id": self.tokenizer_id, "entries": entries},
        }


def plan_filler_batch(measurement: dict, target_tokens: int,
                      haystack_ratio: float | None = None, *,
                      max_documents: int = 32,
                      default_document_tokens: int = 800) -> dict:
    """Plan a bounded batch from measured token shortfall and observed filler size.

    ``haystack_ratio`` is filler tokens / core tokens. Both constraints are lower
    bounds. A ratio may require a total larger than ``target_tokens``. Re-measure
    every completed batch; document lengths and this estimate are not guarantees.
    """
    _nonnegative_int(target_tokens, "target_tokens")
    if (_nonnegative_int(max_documents, "max_documents") == 0
            or _nonnegative_int(default_document_tokens, "default_document_tokens") == 0):
        raise ValueError("Batch and document token limits must be positive")
    if (not isinstance(measurement, dict)
            or measurement.get("version") != MEASUREMENT_VERSION
            or measurement.get("counting_mode") != COUNTING_MODE
            or not isinstance(measurement.get("tokenizer"), dict)
            or measurement.get("tokenizer_id") != _digest(measurement["tokenizer"])):
        raise ValueError("A current tokenizer-bound corpus measurement is required")
    groups = {}
    for name in ("core", "filler", "total"):
        values = measurement.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"measurement.{name} is missing")
        groups[name] = {key: _nonnegative_int(values.get(key), f"{name}.{key}")
                        for key in ("tokens", "characters", "documents")}
    if any(groups["total"][key] != groups["core"][key] + groups["filler"][key]
           for key in ("tokens", "characters", "documents")):
        raise ValueError("Measurement group totals are inconsistent")
    if haystack_ratio is not None:
        if (type(haystack_ratio) not in (int, float)
                or not math.isfinite(haystack_ratio) or haystack_ratio < 0):
            raise ValueError("haystack_ratio must be finite and nonnegative")
    ratio_tokens = (math.ceil(haystack_ratio * groups["core"]["tokens"])
                    if haystack_ratio is not None else 0)
    required_filler = max(0, target_tokens - groups["core"]["tokens"], ratio_tokens)
    remaining = max(0, required_filler - groups["filler"]["tokens"])
    sample_count, sample_tokens = groups["filler"]["documents"], groups["filler"]["tokens"]
    observed = sample_tokens / sample_count if sample_count and sample_tokens else None
    expected = observed or default_document_tokens
    batch_documents = min(max_documents, math.ceil(remaining / expected)) if remaining else 0
    per_document = min(math.ceil(expected), math.ceil(remaining / batch_documents)) if batch_documents else 0
    return {
        "tokenizer_id": measurement["tokenizer_id"], "target_tokens": target_tokens,
        "haystack_ratio": haystack_ratio, "required_filler_tokens": required_filler,
        "required_total_tokens": groups["core"]["tokens"] + required_filler,
        "remaining_filler_tokens": remaining, "target_met": remaining == 0,
        "observed_filler_mean_tokens": observed,
        "estimated_tokens_per_document": expected,
        "next_batch_documents": batch_documents,
        "next_document_token_hint": per_document,
        "next_batch_estimated_tokens": batch_documents * expected,
    }
