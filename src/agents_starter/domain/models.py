"""Core value objects. Pure Python: no SDKs, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
TraceKind = Literal["tool_call", "delegation", "handoff", "plan", "turn", "answer"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)

    @classmethod
    def tool_result(cls, call_id: str, content: str) -> Message:
        return cls("tool", content, tool_call_id=call_id)


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class AgentSpec:
    """Declarative agent definition shared by every runtime.

    `delegates` are agents this one calls as tools and gets an answer back from.
    `handoffs` are agents this one may transfer the whole conversation to.
    """

    name: str
    description: str
    instructions: str
    tools: tuple[str, ...] = ()
    delegates: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceEvent:
    agent: str
    kind: TraceKind
    detail: str


@dataclass(frozen=True)
class TokenDelta:
    agent: str
    text: str


StreamEvent = TraceEvent | TokenDelta


@dataclass(frozen=True)
class AgentResult:
    agent: str
    answer: str
    trace: tuple[TraceEvent, ...] = ()
