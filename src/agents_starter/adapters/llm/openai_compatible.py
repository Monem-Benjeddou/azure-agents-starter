"""ChatModel and Embedder over the OpenAI SDK.

One adapter covers Azure OpenAI / Foundry deployments (pass an AsyncAzureOpenAI client) and any
OpenAI-compatible server: Ollama, vLLM, LM Studio, llama.cpp (pass AsyncOpenAI with base_url).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ...application.tools import Tool, parse_arguments
from ...domain.models import Message, ToolCall


class OpenAIChatModel:
    def __init__(self, client: AsyncOpenAI, model: str, *, temperature: float | None = None) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        on_token: Callable[[str], None] | None = None,
    ) -> Message:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [_tool_schema(t) for t in tools]
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        wire = cast("list[ChatCompletionMessageParam]", [_to_wire(m) for m in messages])
        if on_token is not None:
            return await self._stream(wire, kwargs, on_token)

        response = await self._client.chat.completions.create(model=self._model, messages=wire, **kwargs)
        reply = response.choices[0].message
        calls = tuple(
            ToolCall(c.id, c.function.name, parse_arguments(c.function.arguments))
            for c in reply.tool_calls or ()
            if c.type == "function"
        )
        return Message("assistant", reply.content, tool_calls=calls)

    async def _stream(
        self,
        wire: list[ChatCompletionMessageParam],
        kwargs: dict[str, Any],
        on_token: Callable[[str], None],
    ) -> Message:
        stream = await self._client.chat.completions.create(
            model=self._model, messages=wire, stream=True, **kwargs
        )
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                on_token(delta.content)
            for fragment in delta.tool_calls or ():
                call = calls.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = fragment.id or call["id"]
                if fragment.function is not None:
                    call["name"] += fragment.function.name or ""
                    call["arguments"] += fragment.function.arguments or ""
        tool_calls = tuple(
            ToolCall(c["id"], c["name"], parse_arguments(c["arguments"])) for _, c in sorted(calls.items())
        )
        return Message("assistant", "".join(text) or None, tool_calls=tool_calls)


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, *, dimensions: int | None = None) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(model=self._model, input=list(texts), **kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def _tool_schema(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in message.tool_calls
        ]
    if message.tool_call_id:
        wire["tool_call_id"] = message.tool_call_id
    return wire
