"""VectorIndex over Azure AI Search: hybrid (keyword + vector) retrieval.

Works with an index this app owns (created on first ingest) or an existing index from another
system, given its field names. An existing index is only ever searched, never modified.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass

from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ...domain.errors import ReadOnlyIndexError
from ...domain.models import Chunk, SearchHit


@dataclass(frozen=True)
class IndexFields:
    source: str
    content: str
    vector: str


_OWNED = IndexFields(source="document_id", content="content", vector="embedding")


class AzureSearchIndex:
    def __init__(
        self,
        search_client: SearchClient,
        index_client: SearchIndexClient,
        *,
        index_name: str,
        dimensions: int,
        existing: IndexFields | None = None,
    ) -> None:
        self._search = search_client
        self._indexes = index_client
        self._index_name = index_name
        self._dimensions = dimensions
        self._fields = existing or _OWNED
        self._read_only = existing is not None
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._read_only:
            raise ReadOnlyIndexError(self._index_name)
        if self._ready:
            return
        await self._indexes.create_or_update_index(
            SearchIndex(
                name=self._index_name,
                fields=[
                    SimpleField(name="key", type="Edm.String", key=True),
                    SimpleField(name="chunk_id", type="Edm.String"),
                    SimpleField(name="document_id", type="Edm.String", filterable=True),
                    SearchableField(name="content"),
                    SimpleField(name="metadata", type="Edm.String"),
                    SearchField(
                        name="embedding",
                        type="Collection(Edm.Single)",
                        searchable=True,
                        vector_search_dimensions=self._dimensions,
                        vector_search_profile_name="default",
                    ),
                ],
                vector_search=VectorSearch(
                    algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                    profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
                ),
            )
        )
        self._ready = True

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if self._read_only:
            raise ReadOnlyIndexError(self._index_name)
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        results = await self._search.merge_or_upload_documents(
            [
                {
                    "key": _key(c.id),
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "metadata": json.dumps(c.metadata),
                    "embedding": list(v),
                }
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )
        if failed := [r.key for r in results if not r.succeeded]:
            raise RuntimeError(f"Azure AI Search rejected {len(failed)} chunks: {failed[:5]}")

    async def search(self, query: str, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        f = self._fields
        results = await self._search.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=list(vector), k_nearest_neighbors=top_k, fields=f.vector)],
            select=[f.source, f.content],
            top=top_k,
        )
        return [
            SearchHit(
                Chunk(id=str(r[f.source]), document_id=str(r[f.source]), content=r[f.content] or ""),
                float(r["@search.score"]),
            )
            async for r in results
        ]


def _key(chunk_id: str) -> str:
    return base64.urlsafe_b64encode(chunk_id.encode()).decode()
