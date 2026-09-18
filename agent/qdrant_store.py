"""Qdrant cosine search.

The harness falls back to in-process retrieval when ``QDRANT_URL`` is unset or
the broker is unreachable.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from agent.chunk import Chunk
from agent.embed import DIM, embed

COLLECTION = "kb_chunks"


def client() -> Any | None:
    """Connect to Qdrant and ensure the knowledge-base collection exists.

    Returns:
        A ``QdrantClient`` when ``QDRANT_URL`` is set and reachable, otherwise
        ``None``.
    """
    url = os.environ.get("QDRANT_URL")
    if not url:
        return None
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        return None
    qc = QdrantClient(url=url, timeout=5)
    try:
        qc.get_collections()
    except Exception:
        return None
    existing = {c.name for c in qc.get_collections().collections}
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        )
    return qc


def upsert(qc: Any, chunks: list[Chunk]) -> int:
    """Write chunk embeddings into Qdrant.

    Args:
        qc: Connected Qdrant client.
        chunks: Documents to index.

    Returns:
        Number of points upserted.
    """
    from qdrant_client.http import models

    points = []
    for chunk in chunks:
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.doc + chunk.text[:80])),
                vector=embed(chunk.text),
                payload={"doc": chunk.doc, "text": chunk.text},
            )
        )
    qc.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(qc: Any, question: str, k: int = 3, min_score: float = 0.22) -> list[Chunk]:
    """Search the collection for chunks similar to ``question``.

    Args:
        qc: Connected Qdrant client.
        question: User query.
        k: Maximum hits requested from Qdrant.
        min_score: Minimum cosine score kept after search.

    Returns:
        Matching chunks, possibly empty.
    """
    hits = qc.search(collection_name=COLLECTION, query_vector=embed(question), limit=k)
    out: list[Chunk] = []
    for h in hits:
        if h.score < min_score:
            continue
        payload = h.payload or {}
        out.append(Chunk(doc=payload.get("doc", "unknown"), text=payload.get("text", ""), score=float(h.score)))
    return out


def available(qc: Any | None) -> Optional[str]:
    """Return the backend name when a client is live.

    Args:
        qc: Client from :func:`client`, or ``None``.

    Returns:
        ``"qdrant"`` or ``None``.
    """
    return "qdrant" if qc is not None else None
