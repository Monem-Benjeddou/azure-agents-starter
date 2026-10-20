from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..domain.chunking import chunk_document
from ..domain.models import Document, SearchHit
from .ports import Embedder, VectorIndex
from .tools import Tool

SEARCH_TOOL_NAME = "search_knowledge_base"


class KnowledgeBase:
    """Use cases for the retrieval side: ingest documents, search them, expose search as a tool."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        batch_size: int = 64,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

    async def ingest(self, documents: Sequence[Document]) -> int:
        """Chunk, embed and upsert documents. Returns the number of chunks written."""
        await self._index.ensure_ready()
        chunks = [
            chunk for doc in documents for chunk in chunk_document(doc, self._chunk_size, self._chunk_overlap)
        ]
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = await self._embedder.embed([c.content for c in batch])
            await self._index.upsert(batch, vectors)
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        [vector] = await self._embedder.embed([query])
        return await self._index.search(query, vector, top_k)

    def search_tool(self, top_k: int = 5, max_chars: int = 4000) -> Tool:
        async def handler(args: dict[str, Any]) -> str:
            hits = await self.search(str(args["query"]), top_k)
            if not hits:
                return "No relevant documents found."
            return "\n\n".join(
                f"[source: {h.chunk.document_id}]\n{h.chunk.content[:max_chars]}" for h in hits
            )

        return Tool(
            name=SEARCH_TOOL_NAME,
            description="Search the internal knowledge base. Returns passages with their source ids.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for."}},
                "required": ["query"],
            },
            handler=handler,
        )
