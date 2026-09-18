"""Hashed n-gram embeddings for Qdrant cosine search.

Default runtime avoids a GPU encoder. Swap :func:`embed` for a Hugging Face
model when you want semantic RAG.
"""

from __future__ import annotations

import hashlib
import math
import re

TOKEN = re.compile(r"[a-z0-9]+")
DIM = 96


def tokens(text: str) -> list[str]:
    """Split ``text`` into lowercase alphanumeric tokens.

    Args:
        text: Raw document or query text.

    Returns:
        Tokens longer than one character, in order of appearance.
    """
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


def embed(text: str) -> list[float]:
    """Return a unit-length hashed bag-of-words vector.

    Args:
        text: Text to encode.

    Returns:
        A list of ``DIM`` floats with L2 norm 1, or zeros if ``text`` has no
        tokens.
    """
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
    """Compute cosine similarity of two equal-length vectors.

    Args:
        a: Left embedding.
        b: Right embedding.

    Returns:
        Dot product of ``a`` and ``b``. Callers must pass unit vectors for a
        true cosine.
    """
    return sum(x * y for x, y in zip(a, b))
