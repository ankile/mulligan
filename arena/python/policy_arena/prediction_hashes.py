"""Cross-language hashes for ``arena-prediction-content/v1``.

This is a typed binary encoding of JSON values, not JSON text serialization.
See docs/stage-prediction-versions.md for the complete wire contract.
"""

import hashlib
import math
import re
import struct

CONTENT_PROTOCOL = "arena-prediction-content/v1"
MAX_SAFE_INTEGER = 2**53 - 1
MAX_EPISODE_INDEX = 2**63 - 1
MAX_ROW_BYTES = 128 * 1024
MAX_BATCH_BYTES = 1024 * 1024
PREDICTION_REQUIRED_FIELDS = frozenset({"episode_index", "label", "episode_duration_s", "evidence"})
PREDICTION_OPTIONAL_FIELDS = frozenset(
    {
        "canonical_response",
        "review_reason",
        "violation_codes",
        "confidence",
        "vote_summary",
        "source_revision",
    }
)
PREDICTION_FIELDS = PREDICTION_REQUIRED_FIELDS | PREDICTION_OPTIONAL_FIELDS


def canonical_bytes(value, _depth: int = 0) -> bytes:
    """Encode a JSON value identically to the Convex implementation.

    Integers in JSON payloads must fit the JavaScript safe-integer range.
    Episode indexes are separately normalized to decimal strings before hashing.
    """
    if _depth > 64:
        raise ValueError("Prediction JSON exceeds maximum nesting depth")
    if value is None:
        return b"z"
    if type(value) is bool:
        return b"t" if value else b"f"
    if type(value) in (int, float):
        if type(value) is int and abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("JSON integer exceeds JavaScript safe-integer range")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Prediction content cannot contain nonfinite numbers")
        if number == 0:
            number = 0.0
        return b"n" + struct.pack(">d", number).hex().encode("ascii")
    if type(value) is str:
        encoded = value.encode("utf-8", errors="strict")
        return b"s" + str(len(encoded)).encode("ascii") + b":" + encoded
    if type(value) is list:
        return (
            b"a"
            + str(len(value)).encode("ascii")
            + b":"
            + b"".join(canonical_bytes(item, _depth + 1) for item in value)
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Prediction object keys must be strings")
        keys = sorted(value, key=lambda key: key.encode("utf-8", errors="strict"))
        return (
            b"o"
            + str(len(keys)).encode("ascii")
            + b":"
            + b"".join(
                canonical_bytes(key, _depth + 1) + canonical_bytes(value[key], _depth + 1)
                for key in keys
            )
        )
    raise ValueError(f"Prediction content is not JSON: {type(value).__name__}")


def canonical_digest(value) -> str:
    """SHA-256 of the canonical typed encoding, as lowercase hexadecimal."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def episode_index_value(value) -> int:
    """Require a nonnegative int64 episode index without lossy coercion."""
    # ConvexInt64 is accepted for verified reads; importing here keeps hashing
    # itself independent of the SDK.
    from convex import ConvexInt64

    if isinstance(value, ConvexInt64):
        value = value.value
    if type(value) is not int or not 0 <= value <= MAX_EPISODE_INDEX:
        raise ValueError("episode_index must be a nonnegative int64 integer")
    return value


def prediction_payload(row: dict, *, stored: bool = False) -> dict:
    """Validate and extract the fields included in a prediction's content hash.

    ``stored=True`` permits server metadata on a read-back row. Input rows reject
    unknown fields so an exporter cannot silently drop a misspelled field.
    """
    if type(row) is not dict:
        raise ValueError("Prediction row must be an object")
    missing = PREDICTION_REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Prediction row is missing fields: {sorted(missing)}")
    unknown = row.keys() - PREDICTION_FIELDS
    if unknown and not stored:
        raise ValueError(f"Unknown prediction fields: {sorted(unknown)}")
    result = {key: value for key, value in row.items() if key in PREDICTION_FIELDS}
    result["episode_index"] = episode_index_value(result["episode_index"])
    if type(result["label"]) is not dict:
        raise ValueError("Prediction label must be an object")
    duration = result["episode_duration_s"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("episode_duration_s must be finite and positive")
    for field in ("review_reason", "confidence", "source_revision"):
        if field in result and type(result[field]) is not str:
            raise ValueError(f"{field} must be a string when present")
    if "violation_codes" in result and (
        type(result["violation_codes"]) is not list
        or any(type(code) is not str for code in result["violation_codes"])
    ):
        raise ValueError("violation_codes must be a list of strings")
    if "source_revision" in result and not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", result["source_revision"]
    ):
        raise ValueError("source_revision must be a pinned content revision")
    encoded = canonical_bytes({**result, "episode_index": str(result["episode_index"])})
    if len(encoded) > MAX_ROW_BYTES:
        raise ValueError(f"Prediction content exceeds {MAX_ROW_BYTES} bytes")
    return result


def prediction_digest(row: dict, *, stored: bool = False) -> str:
    payload = prediction_payload(row, stored=stored)
    payload["episode_index"] = str(payload["episode_index"])
    return canonical_digest(payload)


def manifest_digest(rows: list[dict], *, stored: bool = False) -> str:
    """Hash numeric-episode-ordered pairs of episode string and content hash."""
    pairs = [
        (
            episode_index_value(row["episode_index"]),
            prediction_digest(row, stored=stored),
        )
        for row in rows
    ]
    if len({episode for episode, _ in pairs}) != len(pairs):
        raise ValueError("Prediction manifest contains duplicate episode_index values")
    return canonical_digest([[str(episode), digest] for episode, digest in sorted(pairs)])
