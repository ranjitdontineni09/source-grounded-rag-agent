"""Qdrant cosine search. Optional — harness falls back to in-process retrieval."""

from __future__ import annotations

import os
import uuid
from typing import Optional

from agent.chunk import Chunk
from agent.embed import DIM, embed

COLLECTION = "kb_chunks"


def client():
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


def upsert(qc, chunks: list[Chunk]) -> int:
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


def search(qc, question: str, k: int = 3, min_score: float = 0.22) -> list[Chunk]:
    hits = qc.search(collection_name=COLLECTION, query_vector=embed(question), limit=k)
    out: list[Chunk] = []
    for h in hits:
        if h.score < min_score:
            continue
        payload = h.payload or {}
        out.append(Chunk(doc=payload.get("doc", "unknown"), text=payload.get("text", ""), score=float(h.score)))
    return out


def available(qc) -> Optional[str]:
    return "qdrant" if qc is not None else None
