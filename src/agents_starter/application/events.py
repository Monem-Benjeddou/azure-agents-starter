"""Live event delivery for streaming, scoped to the current task with a ContextVar.

Child tasks created by asyncio (for example in `asyncio.gather`) inherit the listener, so every
runtime and pattern can emit events without threading a callback through each call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ..domain.models import StreamEvent, TokenDelta, TraceEvent

Listener = Callable[[StreamEvent], None]

_listener: ContextVar[Listener | None] = ContextVar("agents_starter_listener", default=None)


@contextmanager
def listening(listener: Listener) -> Iterator[None]:
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def record(trace: list[TraceEvent], event: TraceEvent) -> None:
    trace.append(event)
    listener = _listener.get()
    if listener is not None:
        listener(event)


def token_sink(agent: str) -> Callable[[str], None] | None:
    listener = _listener.get()
    if listener is None:
        return None
    return lambda text: listener(TokenDelta(agent, text))
