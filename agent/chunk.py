"""Load markdown knowledge-base files into scored chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """One retrieval unit from a markdown document.

    Attributes:
        doc: Source filename stem used as the citation title.
        text: Paragraph body.
        score: Similarity assigned at retrieve time; default 0.
    """

    doc: str
    text: str
    score: float = 0.0


def split_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file on blank lines.

    Args:
        path: File to read.

    Returns:
        One :class:`Chunk` per non-empty paragraph.
    """
    body = path.read_text(encoding="utf-8").strip()
    parts = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
    return [Chunk(doc=path.stem, text=part) for part in parts]


def load_kb(root: Path) -> list[Chunk]:
    """Load every ``*.md`` file under ``root``.

    Args:
        root: Knowledge-base directory.

    Returns:
        Chunks sorted by filename then paragraph order.
    """
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        chunks.extend(split_markdown(path))
    return chunks
