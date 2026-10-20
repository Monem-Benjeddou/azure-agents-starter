"""Brute-force cosine index for local development and tests, optionally persisted to a JSON file."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ...domain.models import Chunk, SearchHit


class InMemoryIndex:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}
        if path and path.exists():
            for row in json.loads(path.read_text()):
                self._rows[row["chunk"]["id"]] = (Chunk(**row["chunk"]), row["vector"])

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, list(vector))
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"chunk": asdict(c), "vector": v} for c, v in self._rows.values()]
            self._path.write_text(json.dumps(rows))

    async def search(self, query: str, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        scored = [SearchHit(chunk, _cosine(vector, v)) for chunk, v in self._rows.values()]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
