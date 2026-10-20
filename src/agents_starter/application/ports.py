"""Outbound ports. The application depends on these; adapters implement them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from ..domain.models import AgentResult, Chunk, Message, SearchHit
from .tools import Tool


class ChatModel(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        on_token: Callable[[str], None] | None = None,
    ) -> Message:
        """Return the assistant's next message. With `on_token`, text is streamed as it arrives."""
        ...


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    async def ensure_ready(self) -> None:
        """Create the index if it does not exist. Idempotent."""
        ...

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def search(self, query: str, vector: Sequence[float], top_k: int) -> list[SearchHit]: ...


class AgentRuntime(Protocol):
    async def run(self, agent: str, conversation: Sequence[Message]) -> AgentResult: ...
