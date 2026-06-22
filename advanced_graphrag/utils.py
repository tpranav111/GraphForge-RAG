"""Shared utilities."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def keywords(text: str, *, limit: int = 12) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
    }
    counts = Counter(t for t in tokenize(text) if t not in stop and len(t) > 2)
    return [word for word, _ in counts.most_common(limit)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def batched(items: Sequence[object], batch_size: int) -> Iterable[Sequence[object]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]

