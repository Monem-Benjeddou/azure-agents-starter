"""Test doubles for the outbound ports. Deterministic, no network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence

from agents_starter.application.tools import Tool
from agents_starter.domain.models import Message, ToolCall


class ScriptedChatModel:
    """Replies per agent from a script, keyed by the first line of the system prompt."""

    def __init__(self, script: dict[str, list[Message]]) -> None:
        self._script = {key: list(replies) for key, replies in script.items()}
        self.calls: list[tuple[str, list[Message], list[str]]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        on_token: Callable[[str], None] | None = None,
    ) -> Message:
        key = next(k for k in self._script if (messages[0].content or "").startswith(k))
        self.calls.append((key, list(messages), [t.name for t in tools]))
        reply = self._script[key].pop(0)
        if on_token is not None and reply.content:
            for word in re.findall(r"\S+\s*", reply.content):
                on_token(word)
        return reply


def say(text: str) -> Message:
    return Message("assistant", text)


def call(name: str, call_id: str = "c1", **arguments: object) -> Message:
    return Message("assistant", None, tool_calls=(ToolCall(call_id, name, dict(arguments)),))


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector: similar wording -> similar vectors."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dims = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
