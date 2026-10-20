"""Azure adapters against stub clients. Real SDK model types, no network."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from azure.ai.agents.models import (
    RequiredFunctionToolCall,
    RequiredFunctionToolCallDetails,
    RunStatus,
    SubmitToolOutputsAction,
    SubmitToolOutputsDetails,
)

from agents_starter.adapters.index.azure_search import AzureSearchIndex, IndexFields
from agents_starter.adapters.runtime.foundry import FoundryAgentRuntime, provision_agents
from agents_starter.application.knowledge import SEARCH_TOOL_NAME
from agents_starter.application.tools import Tool
from agents_starter.domain.errors import AgentRunError, ReadOnlyIndexError, UnknownAgentError
from agents_starter.domain.models import AgentSpec, Chunk, Message


class StubSearchClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.uploaded: list[dict[str, Any]] = []
        self.search_kwargs: dict[str, Any] = {}

    async def merge_or_upload_documents(self, docs: list[dict[str, Any]]) -> list[Any]:
        self.uploaded += docs
        return [SimpleNamespace(key=d["key"], succeeded=d["chunk_id"] != "bad") for d in docs]

    async def search(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.search_kwargs = kwargs

        async def rows() -> AsyncIterator[dict[str, Any]]:
            for row in self.rows:
                yield row

        return rows()


class StubIndexClient:
    def __init__(self) -> None:
        self.created: list[Any] = []

    async def create_or_update_index(self, index: Any) -> Any:
        self.created.append(index)
        return index


def search_index(rows: list[dict[str, Any]] | None = None) -> tuple[AzureSearchIndex, Any, Any]:
    search, indexes = StubSearchClient(rows or []), StubIndexClient()
    index = AzureSearchIndex(search, indexes, index_name="kb", dimensions=3)  # type: ignore[arg-type]
    return index, search, indexes


async def test_index_is_created_once_with_vector_field() -> None:
    index, _, indexes = search_index()
    await index.ensure_ready()
    await index.ensure_ready()

    [created] = indexes.created
    fields = {f.name: f for f in created.fields}
    assert created.name == "kb"
    assert fields["key"].key
    assert fields["embedding"].vector_search_dimensions == 3


async def test_upsert_uses_safe_keys_and_reports_rejections() -> None:
    index, search, _ = search_index()
    await index.upsert([Chunk("docs/a b.md-0", "docs/a b.md", "text", {"k": "v"})], [[0.1, 0.2, 0.3]])

    [doc] = search.uploaded
    assert set(doc["key"]) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=")
    assert doc["chunk_id"] == "docs/a b.md-0"
    assert doc["metadata"] == '{"k": "v"}'

    with pytest.raises(RuntimeError, match="rejected 1 chunks"):
        await index.upsert([Chunk("bad", "d", "t")], [[0.0, 0.0, 0.0]])


async def test_search_is_hybrid_and_maps_results() -> None:
    row = {"chunk_id": "a-0", "document_id": "a", "content": "hello", "metadata": "{}", "@search.score": 2.5}
    index, search, _ = search_index([row])

    [hit] = await index.search("hello", [0.1, 0.2, 0.3], top_k=4)

    assert hit.chunk == Chunk("a", "a", "hello")
    assert hit.score == 2.5
    assert search.search_kwargs["search_text"] == "hello"
    assert search.search_kwargs["top"] == 4
    assert search.search_kwargs["vector_queries"][0].k_nearest_neighbors == 4


async def test_existing_index_uses_its_field_names_and_is_never_modified() -> None:
    row = {"id": 42, "body": "minutes", "@search.score": 1.0}
    search, indexes = StubSearchClient([row]), StubIndexClient()
    fields = IndexFields(source="id", content="body", vector="body_vector")
    index = AzureSearchIndex(search, indexes, index_name="meetings", dimensions=3, existing=fields)  # type: ignore[arg-type]

    [hit] = await index.search("q", [0.1, 0.2, 0.3], top_k=1)

    assert hit.chunk == Chunk("42", "42", "minutes")
    assert search.search_kwargs["select"] == ["id", "body"]
    assert search.search_kwargs["vector_queries"][0].fields == "body_vector"
    with pytest.raises(ReadOnlyIndexError):
        await index.ensure_ready()
    with pytest.raises(ReadOnlyIndexError):
        await index.upsert([Chunk("c", "d", "t")], [[0.0, 0.0, 0.0]])
    assert indexes.created == []
    assert search.uploaded == []


def run(status: RunStatus, required_action: Any = None, last_error: Any = None) -> Any:
    return SimpleNamespace(id="run1", status=status, required_action=required_action, last_error=last_error)


class StubAgentsClient:
    """Mimics azure.ai.agents.aio.AgentsClient's sub-clients for one run."""

    def __init__(self, runs: list[Any]) -> None:
        self._runs = runs
        self.submitted: list[Any] = []
        self.thread_messages: list[Any] = []
        self.deleted: list[str] = []
        self.created_agents: list[dict[str, Any]] = []
        self.updated_agents: list[tuple[str, dict[str, Any]]] = []
        self.threads = SimpleNamespace(create=self._create_thread, delete=self._delete_thread)
        self.runs = SimpleNamespace(
            create=self._next_run, get=self._next_run, submit_tool_outputs=self._submit, cancel=self._next_run
        )
        self.messages = SimpleNamespace(get_last_message_text_by_role=self._last_message)

    async def _create_thread(self, messages: list[Any]) -> Any:
        self.thread_messages = messages
        return SimpleNamespace(id="thread1")

    async def _delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)

    async def _next_run(self, **_: Any) -> Any:
        return self._runs.pop(0)

    async def _submit(self, *, tool_outputs: list[Any], **_: Any) -> Any:
        self.submitted += tool_outputs
        return self._runs.pop(0)

    async def _last_message(self, **_: Any) -> Any:
        return SimpleNamespace(text=SimpleNamespace(value="final answer"))

    async def create_agent(self, **fields: Any) -> Any:
        self.created_agents.append(fields)
        return SimpleNamespace(id=f"asst_{fields['name']}")

    async def update_agent(self, agent_id: str, **fields: Any) -> Any:
        self.updated_agents.append((agent_id, fields))
        return SimpleNamespace(id=agent_id)


async def _lookup(args: dict[str, Any]) -> str:
    return f"looked up {args['query']}"


LOOKUP = Tool("lookup", "", {"type": "object", "properties": {}, "required": ["query"]}, _lookup)


async def test_foundry_runs_client_side_tools_and_returns_last_message() -> None:
    tool_call = RequiredFunctionToolCall(
        id="call1", function=RequiredFunctionToolCallDetails(name="lookup", arguments='{"query": "x"}')
    )
    action = SubmitToolOutputsAction(submit_tool_outputs=SubmitToolOutputsDetails(tool_calls=[tool_call]))
    client = StubAgentsClient(
        [run(RunStatus.REQUIRES_ACTION, action), run(RunStatus.IN_PROGRESS), run(RunStatus.COMPLETED)]
    )
    runtime = FoundryAgentRuntime(client, {"triage": "asst_1"}, [LOOKUP], poll_interval=0)  # type: ignore[arg-type]

    result = await runtime.run("triage", [Message.system("ignored"), Message.user("hi")])

    assert result.answer == "final answer"
    assert [(e.kind, e.detail) for e in result.trace] == [("tool_call", "lookup"), ("answer", "final answer")]
    assert [(o.tool_call_id, o.output) for o in client.submitted] == [("call1", "looked up x")]
    assert [m.role for m in client.thread_messages] == ["user"]
    assert client.deleted == ["thread1"]


async def test_foundry_failed_run_raises_and_still_deletes_thread() -> None:
    client = StubAgentsClient([run(RunStatus.FAILED, last_error="quota")])
    runtime = FoundryAgentRuntime(client, {"triage": "asst_1"}, [], poll_interval=0)  # type: ignore[arg-type]

    with pytest.raises(AgentRunError, match="quota"):
        await runtime.run("triage", [Message.user("hi")])
    assert client.deleted == ["thread1"]


async def test_foundry_unknown_agent() -> None:
    runtime = FoundryAgentRuntime(StubAgentsClient([]), {}, [])  # type: ignore[arg-type]
    with pytest.raises(UnknownAgentError):
        await runtime.run("triage", [Message.user("hi")])


TEAM = [
    AgentSpec("triage", "Routes.", "route", delegates=("researcher",)),
    AgentSpec("researcher", "Searches.", "search", tools=(SEARCH_TOOL_NAME,)),
]
SEARCH_TOOL = Tool(SEARCH_TOOL_NAME, "", {"type": "object", "properties": {}}, _lookup)


async def test_provisioning_creates_children_first_and_updates_only_known_ids() -> None:
    client = StubAgentsClient([])

    ids = await provision_agents(
        client,  # type: ignore[arg-type]
        TEAM,
        [SEARCH_TOOL],
        model="gpt",
        search_connection_id="conn",
        search_index_name="kb",
        existing_ids={"researcher": "asst_existing"},
        name_prefix="demo-",
    )

    assert ids == {"researcher": "asst_existing", "triage": "asst_demo-triage"}
    [(updated_id, researcher)] = client.updated_agents
    assert updated_id == "asst_existing"
    assert [d.type for d in researcher["tools"]] == ["azure_ai_search"]
    [search_resource] = researcher["tool_resources"].as_dict()["azure_ai_search"]["indexes"]
    assert (search_resource["index_connection_id"], search_resource["index_name"]) == ("conn", "kb")
    [triage] = client.created_agents
    [connected] = triage["tools"]
    assert connected.connected_agent.id == "asst_existing"


async def test_provisioning_rejects_client_side_tools_on_connected_agents() -> None:
    with pytest.raises(ValueError, match="cannot call client-side tools"):
        await provision_agents(StubAgentsClient([]), TEAM, [SEARCH_TOOL], model="gpt")  # type: ignore[arg-type]
