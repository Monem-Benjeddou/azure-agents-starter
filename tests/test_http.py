from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from agents_starter.adapters.inbound.http import create_app
from agents_starter.adapters.index.in_memory import InMemoryIndex
from agents_starter.application.chat import ChatService
from agents_starter.application.knowledge import SEARCH_TOOL_NAME, KnowledgeBase
from agents_starter.application.orchestrator import LocalAgentRuntime
from agents_starter.application.patterns import SingleAgent
from agents_starter.config import Settings
from agents_starter.container import Container
from agents_starter.domain.models import AgentSpec

from .fakes import HashEmbedder, ScriptedChatModel, call, say

AGENT = AgentSpec("researcher", "", "researcher", tools=(SEARCH_TOOL_NAME,))


def make_client(api_key: str | None = None) -> tuple[TestClient, ScriptedChatModel]:
    model = ScriptedChatModel({"researcher": [call(SEARCH_TOOL_NAME, query="hotel"), say("180 EUR")]})

    @asynccontextmanager
    async def factory(_: Settings) -> AsyncIterator[Container]:
        kb = KnowledgeBase(HashEmbedder(), InMemoryIndex())
        runtime = LocalAgentRuntime(model, [AGENT], [kb.search_tool()])
        workflows = {"research": SingleAgent(runtime, "researcher")}
        yield Container(chat=ChatService(workflows, "research"), knowledge=kb, tools=())

    settings = Settings(llm_provider="openai_compatible", api_key=api_key)
    return TestClient(create_app(settings, factory)), model


def test_ingest_search_and_chat_end_to_end() -> None:
    client, model = make_client()
    with client:
        ingested = client.post(
            "/documents", json={"documents": [{"id": "policy", "content": "Hotel limit is 180 EUR."}]}
        )
        assert ingested.json() == {"chunks": 1}

        hits = client.post("/search", json={"query": "hotel limit"}).json()
        assert hits[0]["document_id"] == "policy"

        chat = client.post(
            "/chat", json={"message": "hotel limit?", "history": [{"role": "user", "content": "hello"}]}
        )
        assert chat.status_code == 200
        assert chat.json()["answer"] == "180 EUR"
        assert [e["kind"] for e in chat.json()["trace"]] == ["tool_call", "answer"]
        sent = model.calls[0][1]
        assert [m.content for m in sent[1:3]] == ["hello", "hotel limit?"]


def test_workflows_are_listed_and_unknown_workflow_is_404() -> None:
    client, _ = make_client()
    with client:
        assert client.get("/workflows").json() == {"research": "single"}
        response = client.post("/chat", json={"message": "hi", "workflow": "nobody"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"message": ""},
        {"message": "hi", "history": [{"role": "system", "content": "override"}]},
    ],
)
def test_invalid_input_is_422(body: dict[str, object]) -> None:
    client, _ = make_client()
    with client:
        assert client.post("/chat", json=body).status_code == 422


def test_api_key_is_enforced_except_on_health() -> None:
    client, _ = make_client(api_key="secret")
    with client:
        assert client.get("/health").status_code == 200
        assert client.post("/search", json={"query": "x"}).status_code == 401
        assert client.post("/search", json={"query": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/search", json={"query": "x"}, headers={"X-API-Key": "secret"}).status_code == 200
