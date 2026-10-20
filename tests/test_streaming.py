import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from agents_starter.adapters.inbound.http import create_app
from agents_starter.adapters.index.in_memory import InMemoryIndex
from agents_starter.adapters.llm.openai_compatible import OpenAIChatModel
from agents_starter.application.chat import ChatService
from agents_starter.application.knowledge import KnowledgeBase
from agents_starter.application.orchestrator import LocalAgentRuntime
from agents_starter.application.patterns import Concurrent, SingleAgent
from agents_starter.config import Settings
from agents_starter.container import Container
from agents_starter.domain.models import AgentResult, AgentSpec, Message, TokenDelta, ToolCall, TraceEvent

from .fakes import HashEmbedder, ScriptedChatModel, call, say

TRIAGE = AgentSpec("triage", "", "triage", delegates=("writer",))
WRITER = AgentSpec("writer", "", "writer")


def service() -> ChatService:
    model = ScriptedChatModel(
        {"triage": [call("ask_writer", request="draft"), say("Final text.")], "writer": [say("Draft text.")]}
    )
    runtime = LocalAgentRuntime(model, [TRIAGE, WRITER], [])
    return ChatService({"support": SingleAgent(runtime, "triage")}, "support")


async def test_stream_yields_steps_and_tokens_in_order_then_the_result() -> None:
    events = [e async for e in service().stream("hi")]

    assert events[0] == TraceEvent("triage", "delegation", "ask_writer")
    assert [e.text for e in events if isinstance(e, TokenDelta) and e.agent == "writer"] == [
        "Draft ",
        "text.",
    ]
    assert [e.text for e in events if isinstance(e, TokenDelta) and e.agent == "triage"] == [
        "Final ",
        "text.",
    ]
    result = events[-1]
    assert isinstance(result, AgentResult)
    assert result.answer == "Final text."
    assert sum(isinstance(e, AgentResult) for e in events) == 1


async def test_stream_reaches_agents_running_concurrently() -> None:
    model = ScriptedChatModel({"a": [say("one")], "b": [say("two")]})
    runtime = LocalAgentRuntime(model, [AgentSpec("a", "", "a"), AgentSpec("b", "", "b")], [])
    chat = ChatService({"both": Concurrent(runtime, ("a", "b"))}, "both")

    tokens = {e.agent: e.text async for e in chat.stream("hi") if isinstance(e, TokenDelta)}

    assert tokens == {"a": "one", "b": "two"}


def test_sse_endpoint_streams_events_and_final_result() -> None:
    @asynccontextmanager
    async def factory(_: Settings) -> AsyncIterator[Container]:
        yield Container(chat=service(), knowledge=KnowledgeBase(HashEmbedder(), InMemoryIndex()), tools=())

    client = TestClient(create_app(Settings(llm_provider="openai_compatible"), factory))
    with client, client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    frames = [f.split("\n", 1) for f in body.strip().split("\n\n")]
    kinds = [head.removeprefix("event: ") for head, _ in frames]
    payloads = [json.loads(data.removeprefix("data: ")) for _, data in frames]
    assert kinds[0] == "trace" and payloads[0]["kind"] == "delegation"
    assert "token" in kinds
    assert kinds[-1] == "result"
    assert payloads[-1]["answer"] == "Final text."


def test_sse_endpoint_rejects_unknown_workflow_before_streaming() -> None:
    @asynccontextmanager
    async def factory(_: Settings) -> AsyncIterator[Container]:
        yield Container(chat=service(), knowledge=KnowledgeBase(HashEmbedder(), InMemoryIndex()), tools=())

    with TestClient(create_app(Settings(llm_provider="openai_compatible"), factory)) as client:
        assert client.post("/chat/stream", json={"message": "hi", "workflow": "nope"}).status_code == 404


def sse_chunks(deltas: list[dict[str, Any]]) -> str:
    frames = [
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": d, "finish_reason": None}],
        }
        for d in deltas
    ]
    frames.insert(
        0, {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": []}
    )
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


async def test_openai_adapter_streams_text_and_reassembles_tool_call_fragments() -> None:
    deltas = [
        {"role": "assistant", "content": "Let me "},
        {"content": "check."},
        {
            "tool_calls": [
                {"index": 0, "id": "c1", "type": "function", "function": {"name": "search", "arguments": ""}}
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"query": '}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"hotel"}'}}]},
    ]
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=sse_chunks(deltas), headers={"content-type": "text/event-stream"})

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://llm.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tokens: list[str] = []

    reply = await OpenAIChatModel(client, "m").complete([Message.user("hi")], on_token=tokens.append)

    assert sent[0]["stream"] is True
    assert tokens == ["Let me ", "check."]
    assert reply == Message(
        "assistant", "Let me check.", tool_calls=(ToolCall("c1", "search", {"query": "hotel"}),)
    )
