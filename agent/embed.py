"""Stable hashed n-gram embeddings. Swap for a Hugging Face encoder for semantic RAG."""

from __future__ import annotations

import hashlib
import math
import re

TOKEN = re.compile(r"[a-z0-9]+")
DIM = 96


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    toks = tokens(text)
    if not toks:
        return vec
    for tok in toks:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % DIM
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
