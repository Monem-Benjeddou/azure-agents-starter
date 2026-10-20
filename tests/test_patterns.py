import json
from collections.abc import Sequence

import pytest

from agents_starter.application.chat import ChatService
from agents_starter.application.patterns import Concurrent, GroupChat, Magentic, Sequential, SingleAgent
from agents_starter.domain.errors import UnknownWorkflowError
from agents_starter.domain.models import AgentResult, AgentSpec, Message, TraceEvent

from .fakes import ScriptedChatModel, say


class RecordingRuntime:
    """Answers `<agent> #<n>` and records the conversation each agent received."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Message]]] = []

    async def run(self, agent: str, conversation: Sequence[Message]) -> AgentResult:
        self.calls.append((agent, list(conversation)))
        answer = f"{agent} #{sum(1 for a, _ in self.calls if a == agent)}"
        return AgentResult(agent, answer, (TraceEvent(agent, "answer", answer),))


def spec(name: str) -> AgentSpec:
    return AgentSpec(name, f"{name} role", name)


QUESTION = [Message.user("question")]


async def test_sequential_passes_each_output_to_the_next_agent() -> None:
    runtime = RecordingRuntime()

    result = await Sequential(runtime, ("a", "b", "c")).run(QUESTION)

    assert result.answer == "c #1"
    assert [agent for agent, _ in runtime.calls] == ["a", "b", "c"]
    assert runtime.calls[2][1] == [
        Message.user("question"),
        Message.user("Output from a:\na #1"),
        Message.user("Output from b:\nb #1"),
    ]
    assert [e.agent for e in result.trace] == ["a", "b", "c"]


async def test_concurrent_fans_out_and_aggregates() -> None:
    runtime = RecordingRuntime()

    result = await Concurrent(runtime, ("a", "b"), aggregator="writer").run(QUESTION)

    assert result.answer == "writer #1"
    assert {agent for agent, _ in runtime.calls[:2]} == {"a", "b"}
    aggregator_input = runtime.calls[2][1][-1].content or ""
    assert "## a\na #1" in aggregator_input and "## b\nb #1" in aggregator_input


async def test_concurrent_without_aggregator_returns_all_answers() -> None:
    result = await Concurrent(RecordingRuntime(), ("a", "b")).run(QUESTION)
    assert result.answer == "## a\na #1\n\n## b\nb #1"


async def test_group_chat_round_robin_shares_one_transcript() -> None:
    runtime = RecordingRuntime()

    result = await GroupChat(runtime, (spec("writer"), spec("reviewer")), max_rounds=3).run(QUESTION)

    assert [agent for agent, _ in runtime.calls] == ["writer", "reviewer", "writer"]
    assert runtime.calls[2][1][-2:] == [
        Message.user("writer: writer #1"),
        Message.user("reviewer: reviewer #1"),
    ]
    assert result.answer == "writer #2"


async def test_group_chat_manager_picks_speakers_and_ends_the_chat() -> None:
    runtime = RecordingRuntime()
    manager = ScriptedChatModel({"You manage": [say("reviewer"), say("Writer."), say("DONE")]})

    result = await GroupChat(runtime, (spec("writer"), spec("reviewer")), manager=manager).run(QUESTION)

    assert [agent for agent, _ in runtime.calls] == ["reviewer", "writer"]
    assert [e.detail for e in result.trace if e.kind == "turn"] == ["reviewer", "writer"]
    assert result.answer == "writer #1"


def ledger(**fields: object) -> Message:
    return say(json.dumps(fields))


async def test_magentic_plans_delegates_tracks_progress_and_synthesizes() -> None:
    runtime = RecordingRuntime()
    manager = ScriptedChatModel(
        {
            "You lead": [
                say("1. researcher finds facts 2. writer drafts"),
                ledger(
                    is_complete=False, is_progressing=True, next_agent="researcher", instruction="Find facts"
                ),
                say(
                    "```json\n"
                    + json.dumps({"is_complete": False, "next_agent": "writer", "instruction": "Draft"})
                    + "\n```"
                ),
                ledger(is_complete=True),
            ],
            "Task:": [say("final answer")],
        }
    )

    result = await Magentic(runtime, manager, (spec("researcher"), spec("writer"))).run(QUESTION)

    assert result.answer == "final answer"
    assert [agent for agent, _ in runtime.calls] == ["researcher", "writer"]
    assert (runtime.calls[0][1][0].content or "").startswith("Find facts")
    assert "researcher: researcher #1" in (runtime.calls[1][1][0].content or "")
    assert [e.kind for e in result.trace if e.agent == "manager"] == ["plan", "turn", "turn", "answer"]


async def test_magentic_replans_after_repeated_stalls() -> None:
    runtime = RecordingRuntime()
    stalled = ledger(is_complete=False, is_progressing=False, next_agent="nobody")
    manager = ScriptedChatModel(
        {
            "You lead": [say("plan A"), stalled, stalled, say("plan B"), ledger(is_complete=True)],
            "Task:": [say("done")],
        }
    )

    result = await Magentic(runtime, manager, (spec("writer"),), max_stalls=1).run(QUESTION)

    assert runtime.calls == []
    assert [e.detail for e in result.trace if e.kind == "plan"] == ["plan A", "plan B"]


async def test_chat_service_routes_to_named_workflows() -> None:
    runtime = RecordingRuntime()
    service = ChatService({"one": SingleAgent(runtime, "a"), "two": SingleAgent(runtime, "b")}, "one")

    assert (await service.ask("hi")).answer == "a #1"
    assert (await service.ask("hi", workflow="two")).answer == "b #1"
    assert service.workflows == {"one": "single", "two": "single"}
    with pytest.raises(UnknownWorkflowError):
        await service.ask("hi", workflow="three")
    with pytest.raises(ValueError):
        ChatService({}, "missing")
