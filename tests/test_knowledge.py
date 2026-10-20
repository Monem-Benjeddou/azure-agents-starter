from pathlib import Path

import pytest

from agents_starter.adapters.index.in_memory import InMemoryIndex
from agents_starter.application.knowledge import KnowledgeBase
from agents_starter.domain.chunking import chunk_document
from agents_starter.domain.models import Document

from .fakes import HashEmbedder

DOCS = [
    Document("expenses", "Hotel costs are reimbursed up to 180 EUR per night in capital cities."),
    Document("onboarding", "New engineers are paired with an onboarding buddy for 30 days."),
]


def test_short_document_is_one_chunk() -> None:
    [chunk] = chunk_document(Document("d", "one\n\ntwo"))
    assert chunk.content == "one\n\ntwo"
    assert chunk.id == "d-0"


def test_long_text_is_split_with_overlap_and_bounded_size() -> None:
    text = "\n\n".join(f"paragraph {i} " + "x" * 80 for i in range(50))
    chunks = chunk_document(Document("d", text), size=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.content) <= 300 + 50 + 2 for c in chunks)
    assert chunks[0].content[-50:] == chunks[1].content[:50]
    assert "paragraph 49" in chunks[-1].content


def test_oversized_paragraph_is_hard_split() -> None:
    chunks = chunk_document(Document("d", "y" * 1000), size=300, overlap=0)
    assert "".join(c.content for c in chunks) == "y" * 1000


def test_invalid_chunk_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_document(Document("d", "x"), size=100, overlap=100)


async def test_ingest_then_search_ranks_relevant_document_first() -> None:
    kb = KnowledgeBase(HashEmbedder(), InMemoryIndex(), batch_size=1)

    assert await kb.ingest(DOCS) == 2
    hits = await kb.search("hotel costs per night", top_k=2)

    assert [h.chunk.document_id for h in hits] == ["expenses", "onboarding"]


async def test_search_tool_formats_citations_and_handles_empty_index() -> None:
    empty = KnowledgeBase(HashEmbedder(), InMemoryIndex()).search_tool()
    assert await empty.handler({"query": "x"}) == "No relevant documents found."

    kb = KnowledgeBase(HashEmbedder(), InMemoryIndex())
    await kb.ingest(DOCS)
    output = await kb.search_tool(top_k=1).handler({"query": "onboarding buddy"})
    assert output.startswith("[source: onboarding]\n")


async def test_memory_index_persists_between_instances(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    await KnowledgeBase(HashEmbedder(), InMemoryIndex(path)).ingest(DOCS)

    reloaded = KnowledgeBase(HashEmbedder(), InMemoryIndex(path))
    [hit] = await reloaded.search("onboarding buddy", top_k=1)
    assert hit.chunk.document_id == "onboarding"
