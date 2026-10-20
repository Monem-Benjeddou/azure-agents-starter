from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence

from ..domain.errors import UnknownWorkflowError
from ..domain.models import AgentResult, Message, StreamEvent
from .events import listening
from .patterns import Workflow


class ChatService:
    """Inbound use case: answer a user message with a named workflow, whole or streamed."""

    def __init__(self, workflows: Mapping[str, Workflow], default_workflow: str) -> None:
        if default_workflow not in workflows:
            raise ValueError(f"Default workflow {default_workflow!r} is not registered")
        self._workflows = dict(workflows)
        self._default = default_workflow

    @property
    def workflows(self) -> dict[str, str]:
        return {name: w.pattern for name, w in self._workflows.items()}

    async def ask(
        self, message: str, *, workflow: str | None = None, history: Sequence[Message] = ()
    ) -> AgentResult:
        return await self._select(workflow).run([*history, Message.user(message)])

    def stream(
        self, message: str, *, workflow: str | None = None, history: Sequence[Message] = ()
    ) -> AsyncIterator[StreamEvent | AgentResult]:
        """Yield trace events and tokens as they happen, then the final AgentResult.

        The workflow is resolved eagerly so an unknown name fails before streaming starts.
        """
        return _stream(self._select(workflow), [*history, Message.user(message)])

    def _select(self, workflow: str | None) -> Workflow:
        name = workflow or self._default
        selected = self._workflows.get(name)
        if selected is None:
            raise UnknownWorkflowError(name)
        return selected


async def _stream(
    workflow: Workflow, conversation: list[Message]
) -> AsyncIterator[StreamEvent | AgentResult]:
    queue: asyncio.Queue[StreamEvent | AgentResult | Exception] = asyncio.Queue()

    async def produce() -> None:
        with listening(queue.put_nowait):
            try:
                queue.put_nowait(await workflow.run(conversation))
            except Exception as exc:
                queue.put_nowait(exc)

    task = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if isinstance(item, Exception):
                raise item
            yield item
            if isinstance(item, AgentResult):
                return
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
