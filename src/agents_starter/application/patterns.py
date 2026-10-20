"""Multi-agent orchestration patterns, composed over the AgentRuntime port.

Each pattern works with any runtime (in-process or Azure AI Foundry), because it only needs
"run agent X on this conversation". Agents-as-tools and handoff live inside the runtime, since
they need control of the tool-calling loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..domain.models import AgentResult, AgentSpec, Message, TraceEvent
from .events import record, token_sink
from .ports import AgentRuntime, ChatModel


class Workflow(Protocol):
    @property
    def pattern(self) -> str: ...

    async def run(self, conversation: Sequence[Message]) -> AgentResult: ...


@dataclass(frozen=True)
class SingleAgent:
    """One agent, which may still delegate to sub-agents or hand off within the runtime."""

    runtime: AgentRuntime
    agent: str
    pattern: str = "single"

    async def run(self, conversation: Sequence[Message]) -> AgentResult:
        return await self.runtime.run(self.agent, conversation)


@dataclass(frozen=True)
class Sequential:
    """Agents run in a fixed order; each one sees the conversation plus all earlier outputs."""

    runtime: AgentRuntime
    agents: Sequence[str]
    pattern: str = "sequential"

    async def run(self, conversation: Sequence[Message]) -> AgentResult:
        history = list(conversation)
        trace: list[TraceEvent] = []
        result = AgentResult(agent="", answer="")
        for agent in self.agents:
            result = await self.runtime.run(agent, history)
            trace += result.trace
            history.append(Message.user(f"Output from {agent}:\n{result.answer}"))
        return AgentResult(agent=result.agent, answer=result.answer, trace=tuple(trace))


@dataclass(frozen=True)
class Concurrent:
    """All agents work on the same input in parallel; an optional aggregator merges the results."""

    runtime: AgentRuntime
    agents: Sequence[str]
    aggregator: str | None = None
    pattern: str = "concurrent"

    async def run(self, conversation: Sequence[Message]) -> AgentResult:
        results = await asyncio.gather(*(self.runtime.run(a, conversation) for a in self.agents))
        trace = [event for r in results for event in r.trace]
        combined = "\n\n".join(f"## {a}\n{r.answer}" for a, r in zip(self.agents, results, strict=True))
        if self.aggregator is None:
            return AgentResult(agent="concurrent", answer=combined, trace=tuple(trace))

        merged = await self.runtime.run(
            self.aggregator,
            [*conversation, Message.user(f"Combine these independent answers into one:\n\n{combined}")],
        )
        return AgentResult(agent=merged.agent, answer=merged.answer, trace=(*trace, *merged.trace))


_SELECT_PROMPT = """You manage a group conversation between these participants:
{team}

Read the conversation and reply with only the name of the participant who should speak next,
or DONE if the request has been fully answered."""


@dataclass(frozen=True)
class GroupChat:
    """Participants share one transcript; a manager picks the next speaker each round.

    Without a manager model, speakers take turns in order (round robin) for `max_rounds`.
    """

    runtime: AgentRuntime
    participants: Sequence[AgentSpec]
    manager: ChatModel | None = None
    max_rounds: int = 6
    pattern: str = "group_chat"

    async def run(self, conversation: Sequence[Message]) -> AgentResult:
        transcript = list(conversation)
        trace: list[TraceEvent] = []
        last = AgentResult(agent="", answer="")
        for round_number in range(self.max_rounds):
            speaker = await self._next_speaker(transcript, round_number)
            if speaker is None:
                break
            record(trace, TraceEvent("manager", "turn", speaker))
            last = await self.runtime.run(speaker, transcript)
            trace += last.trace
            transcript.append(Message.user(f"{speaker}: {last.answer}"))
        return AgentResult(agent=last.agent, answer=last.answer, trace=tuple(trace))

    async def _next_speaker(self, transcript: list[Message], round_number: int) -> str | None:
        names = [p.name for p in self.participants]
        if self.manager is None:
            return names[round_number % len(names)]
        team = "\n".join(f"- {p.name}: {p.description}" for p in self.participants)
        reply = await self.manager.complete([Message.system(_SELECT_PROMPT.format(team=team)), *transcript])
        choice = (reply.content or "").strip().strip(".").strip()
        if choice.upper() == "DONE":
            return None
        return next((n for n in names if n.lower() == choice.lower()), names[round_number % len(names)])


_PLAN_PROMPT = """You lead a team of agents:
{team}

Task: {task}

Write a short numbered plan that assigns each step to one team member. Facts must come from
team members who can search; others may only reshape what was found."""

_LEDGER_PROMPT = """You lead a team of agents:
{team}

Task: {task}

Plan:
{plan}

Work so far:
{work}

Reply with a JSON object only, with these keys:
- "is_complete": true when the task is fully done
- "is_progressing": false when the last rounds added nothing new
- "next_agent": the team member who should act next
- "instruction": what that member should do next"""

_FINAL_PROMPT = """Task: {task}

Work done by the team:
{work}

Write the final answer to the task for the user using only facts that appear in the team's
work, keeping their [source: ...] citations. If the work does not answer the task, say so."""


@dataclass(frozen=True)
class Magentic:
    """A manager plans, assigns each step, tracks progress in a ledger, and replans when stalled.

    Based on the Magentic-One orchestrator design.
    """

    runtime: AgentRuntime
    manager: ChatModel
    participants: Sequence[AgentSpec]
    max_rounds: int = 10
    max_stalls: int = 2
    pattern: str = "magentic"

    async def run(self, conversation: Sequence[Message]) -> AgentResult:
        task = next((m.content for m in reversed(conversation) if m.role == "user" and m.content), "")
        team = "\n".join(f"- {p.name}: {p.description}" for p in self.participants)
        names = {p.name for p in self.participants}
        trace: list[TraceEvent] = []
        work: list[str] = []

        plan = await self._ask(_PLAN_PROMPT.format(team=team, task=task))
        record(trace, TraceEvent("manager", "plan", plan))
        stalls = 0
        for _ in range(self.max_rounds):
            ledger = _parse_ledger(
                await self._ask(_LEDGER_PROMPT.format(team=team, task=task, plan=plan, work=_join(work)))
            )
            if ledger.get("is_complete") is True:
                break
            agent = ledger.get("next_agent")
            stalled = ledger.get("is_progressing") is False or agent not in names
            stalls = stalls + 1 if stalled else 0
            if stalls > self.max_stalls:
                plan = await self._ask(_PLAN_PROMPT.format(team=team, task=task) + _replan(work))
                record(trace, TraceEvent("manager", "plan", plan))
                stalls = 0
            if agent not in names:
                continue
            record(trace, TraceEvent("manager", "turn", str(agent)))
            result = await self.runtime.run(
                str(agent),
                [Message.user(f"{ledger.get('instruction') or task}\n\nTeam work so far:\n{_join(work)}")],
            )
            trace += result.trace
            work.append(f"{agent}: {result.answer}")

        answer = await self._ask(_FINAL_PROMPT.format(task=task, work=_join(work)), stream=True)
        record(trace, TraceEvent("manager", "answer", answer))
        return AgentResult(agent="manager", answer=answer, trace=tuple(trace))

    async def _ask(self, prompt: str, *, stream: bool = False) -> str:
        on_token = token_sink("manager") if stream else None
        reply = await self.manager.complete([Message.user(prompt)], on_token=on_token)
        return reply.content or ""


def _join(work: list[str]) -> str:
    return "\n\n".join(work) or "(nothing yet)"


def _replan(work: list[str]) -> str:
    return f"\n\nThe previous plan stalled. Work so far:\n{_join(work)}\n\nWrite a new plan."


def _parse_ledger(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
