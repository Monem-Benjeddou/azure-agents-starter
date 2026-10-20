import pytest

from agents_starter.adapters.index.in_memory import InMemoryIndex
from agents_starter.application.knowledge import SEARCH_TOOL_NAME, KnowledgeBase
from agents_starter.application.orchestrator import LocalAgentRuntime
from agents_starter.application.tools import Tool
from agents_starter.domain.errors import MaxTurnsExceededError, UnknownAgentError
from agents_starter.domain.models import AgentSpec, Document, Message

from .fakes import HashEmbedder, ScriptedChatModel, call, say

TRIAGE = AgentSpec("triage", "Routes requests.", "triage", delegates=("researcher",))
RESEARCHER = AgentSpec("researcher", "Searches docs.", "researcher", tools=(SEARCH_TOOL_NAME,))


async def knowledge() -> KnowledgeBase:
    kb = KnowledgeBase(HashEmbedder(), InMemoryIndex())
    await kb.ingest([Document("policy", "Hotel costs are reimbursed up to 180 EUR per night.")])
    return kb


async def test_triage_delegates_to_researcher_which_uses_the_index() -> None:
    model = ScriptedChatModel(
        {
            "triage": [call("ask_researcher", request="hotel limit?"), say("Up to 180 EUR [source: policy]")],
            "researcher": [call(SEARCH_TOOL_NAME, query="hotel limit"), say("180 EUR [source: policy]")],
        }
    )
    kb = await knowledge()
    runtime = LocalAgentRuntime(model, [TRIAGE, RESEARCHER], [kb.search_tool()])

    result = await runtime.run("triage", [Message.user("What is the hotel limit?")])

    assert result.answer == "Up to 180 EUR [source: policy]"
    assert [(e.agent, e.kind, e.detail) for e in result.trace] == [
        ("triage", "delegation", "ask_researcher"),
        ("researcher", "tool_call", SEARCH_TOOL_NAME),
        ("researcher", "answer", "180 EUR [source: policy]"),
        ("triage", "answer", "Up to 180 EUR [source: policy]"),
    ]
    researcher_last_call = [c for c in model.calls if c[0] == "researcher"][-1]
    assert "[source: policy]" in (researcher_last_call[1][-1].content or "")
    triage_last_call = [c for c in model.calls if c[0] == "triage"][-1]
    assert triage_last_call[1][-1] == Message.tool_result("c1", "180 EUR [source: policy]")
    assert model.calls[0][2] == ["ask_researcher"]
    assert model.calls[1][2] == [SEARCH_TOOL_NAME]


@pytest.mark.parametrize(
    ("bad_call", "expected_error"),
    [
        (call("does_not_exist"), "unknown tool 'does_not_exist'"),
        (call(SEARCH_TOOL_NAME), "missing required arguments ['query']"),
    ],
)
async def test_bad_tool_calls_are_reported_back_to_the_model(bad_call: Message, expected_error: str) -> None:
    model = ScriptedChatModel({"researcher": [bad_call, say("recovered")]})
    runtime = LocalAgentRuntime(model, [RESEARCHER], [(await knowledge()).search_tool()])

    result = await runtime.run("researcher", [Message.user("hi")])

    assert result.answer == "recovered"
    assert expected_error in (model.calls[1][1][-1].content or "")


async def test_tool_exceptions_are_reported_back_to_the_model() -> None:
    async def boom(_: dict[str, object]) -> str:
        raise RuntimeError("index offline")

    tool = Tool(SEARCH_TOOL_NAME, "", {"type": "object", "properties": {}}, boom)
    model = ScriptedChatModel({"researcher": [call(SEARCH_TOOL_NAME), say("sorry")]})
    runtime = LocalAgentRuntime(model, [RESEARCHER], [tool])

    result = await runtime.run("researcher", [Message.user("hi")])

    assert result.answer == "sorry"
    assert "index offline" in (model.calls[1][1][-1].content or "")


async def test_runaway_tool_loop_stops_at_max_turns() -> None:
    model = ScriptedChatModel({"researcher": [call(SEARCH_TOOL_NAME, query="x")] * 3})
    runtime = LocalAgentRuntime(model, [RESEARCHER], [(await knowledge()).search_tool()], max_turns=3)

    with pytest.raises(MaxTurnsExceededError):
        await runtime.run("researcher", [Message.user("hi")])


async def test_unknown_agent() -> None:
    runtime = LocalAgentRuntime(ScriptedChatModel({}), [], [])
    with pytest.raises(UnknownAgentError):
        await runtime.run("nobody", [Message.user("hi")])


@pytest.mark.parametrize(
    ("agents", "error"),
    [
        ([AgentSpec("a", "", "", tools=("missing",))], "unknown tools"),
        ([AgentSpec("a", "", "", delegates=("ghost",))], "unknown agents"),
        (
            [AgentSpec("a", "", "", delegates=("b",)), AgentSpec("b", "", "", delegates=("a",))],
            "Delegation cycle: a -> b -> a",
        ),
    ],
)
def test_wiring_errors_fail_at_startup(agents: list[AgentSpec], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        LocalAgentRuntime(ScriptedChatModel({}), agents, [])


CONCIERGE = AgentSpec("concierge", "Routes.", "concierge", handoffs=("researcher",))


async def test_handoff_transfers_the_conversation_to_the_target_agent() -> None:
    model = ScriptedChatModel(
        {
            "concierge": [call("transfer_to_researcher")],
            "researcher": [call(SEARCH_TOOL_NAME, query="hotel"), say("180 EUR [source: policy]")],
        }
    )
    runtime = LocalAgentRuntime(model, [CONCIERGE, RESEARCHER], [(await knowledge()).search_tool()])

    result = await runtime.run("concierge", [Message.user("hotel limit?")])

    assert result.agent == "researcher"
    assert result.answer == "180 EUR [source: policy]"
    assert [(e.agent, e.kind) for e in result.trace] == [
        ("concierge", "handoff"),
        ("researcher", "tool_call"),
        ("researcher", "answer"),
    ]
    researcher_first_call = model.calls[1]
    assert researcher_first_call[1][0].content == "researcher"
    assert researcher_first_call[1][1:] == [Message.user("hotel limit?")]
    assert model.calls[0][2] == ["transfer_to_researcher"]
