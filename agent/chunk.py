from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc: str
    text: str
    score: float = 0.0


def split_markdown(path: Path) -> list[Chunk]:
    body = path.read_text(encoding="utf-8").strip()
    parts = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
    return [Chunk(doc=path.stem, text=part) for part in parts]


def load_kb(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        chunks.extend(split_markdown(path))
    return chunks
