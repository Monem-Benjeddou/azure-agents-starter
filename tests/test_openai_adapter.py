"""Exercises the real OpenAI SDK against a mocked HTTP transport to pin the wire format."""

import json
from typing import Any

import httpx
from openai import AsyncOpenAI

from agents_starter.adapters.llm.openai_compatible import OpenAIChatModel, OpenAIEmbedder
from agents_starter.application.tools import Tool
from agents_starter.domain.models import Message, ToolCall


def client_returning(payload: dict[str, Any], seen: list[dict[str, Any]]) -> AsyncOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    return AsyncOpenAI(
        api_key="test",
        base_url="http://llm.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
    }


async def _noop(_: dict[str, Any]) -> str:
    return ""


SEARCH = Tool("search", "Search docs", {"type": "object", "properties": {}}, _noop)


async def test_sends_history_and_tools_in_openai_format() -> None:
    seen: list[dict[str, Any]] = []
    model = OpenAIChatModel(client_returning(completion({"content": "done"}), seen), "gpt", temperature=0)
    history = [
        Message.system("be brief"),
        Message.user("hi"),
        Message("assistant", None, tool_calls=(ToolCall("c1", "search", {"q": "x"}),)),
        Message.tool_result("c1", "result"),
    ]

    reply = await model.complete(history, [SEARCH])

    assert reply == Message("assistant", "done")
    [body] = seen
    assert body["model"] == "gpt"
    assert body["temperature"] == 0
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert body["messages"][2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
    ]
    assert body["messages"][3] == {"role": "tool", "content": "result", "tool_call_id": "c1"}


async def test_omits_tools_when_agent_has_none() -> None:
    seen: list[dict[str, Any]] = []
    await OpenAIChatModel(client_returning(completion({"content": "ok"}), seen), "m").complete(
        [Message.user("hi")]
    )
    assert "tools" not in seen[0]
    assert "temperature" not in seen[0]


async def test_parses_tool_calls_and_tolerates_malformed_arguments() -> None:
    tool_calls = [
        {"id": "a", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'}},
        {"id": "b", "type": "function", "function": {"name": "search", "arguments": "{not json"}},
    ]
    model = OpenAIChatModel(
        client_returning(completion({"content": None, "tool_calls": tool_calls}), []), "m"
    )

    reply = await model.complete([Message.user("hi")], [SEARCH])

    assert reply.tool_calls == (ToolCall("a", "search", {"query": "x"}), ToolCall("b", "search", {}))


async def test_embeddings_are_returned_in_input_order() -> None:
    payload = {
        "object": "list",
        "model": "e",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
        ],
    }
    seen: list[dict[str, Any]] = []
    embedder = OpenAIEmbedder(client_returning(payload, seen), "e", dimensions=2)

    assert await embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert seen[0]["dimensions"] == 2
