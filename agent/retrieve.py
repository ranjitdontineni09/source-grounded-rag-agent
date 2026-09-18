"""In-process retrieval used when Qdrant is not configured."""

from __future__ import annotations

from agent.chunk import Chunk
from agent.embed import cosine, embed, tokens

STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "it"}


def lexical(question: str, kb: list[Chunk], k: int = 3, min_score: float = 0.12) -> list[Chunk]:
    """Rank chunks by token overlap with ``question``.

    Args:
        question: User query.
        kb: Indexed chunks.
        k: Maximum hits to return.
        min_score: Minimum overlap ratio (matched query tokens / query tokens).

    Returns:
        Up to ``k`` chunks at or above ``min_score``, highest first. Empty when
        nothing clears the threshold (caller must refuse).
    """
    q = {t for t in tokens(question) if t not in STOP}
    if not q:
        return []
    scored: list[Chunk] = []
    for chunk in kb:
        overlap = q & {t for t in tokens(chunk.text) if t not in STOP}
        score = len(overlap) / len(q)
        if score >= min_score:
            scored.append(Chunk(doc=chunk.doc, text=chunk.text, score=score))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]


def vector(question: str, kb: list[Chunk], k: int = 3, min_score: float = 0.22) -> list[Chunk]:
    """Rank chunks by hashed-embedding cosine similarity.

    Args:
        question: User query.
        kb: Indexed chunks.
        k: Maximum hits to return.
        min_score: Minimum cosine score.

    Returns:
        Up to ``k`` chunks at or above ``min_score``, highest first.
    """
    qv = embed(question)
    scored: list[Chunk] = []
    for chunk in kb:
        score = cosine(qv, embed(chunk.text))
        if score >= min_score:
            scored.append(Chunk(doc=chunk.doc, text=chunk.text, score=score))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]
