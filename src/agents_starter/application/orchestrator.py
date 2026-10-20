"""In-process agent runtime: a tool-calling loop with agents-as-tools and handoffs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..domain.errors import MaxTurnsExceededError, UnknownAgentError
from ..domain.models import AgentResult, AgentSpec, Message, ToolCall, TraceEvent
from .events import record, token_sink
from .ports import ChatModel
from .tools import Tool, invoke_tool

DELEGATE_PREFIX = "ask_"
HANDOFF_PREFIX = "transfer_to_"


class LocalAgentRuntime:
    def __init__(
        self,
        model: ChatModel,
        agents: Sequence[AgentSpec],
        tools: Sequence[Tool],
        *,
        max_turns: int = 8,
    ) -> None:
        self._model = model
        self._agents = {a.name: a for a in agents}
        self._tools = {t.name: t for t in tools}
        self._max_turns = max_turns
        _validate(self._agents, self._tools)

    async def run(self, agent: str, conversation: Sequence[Message]) -> AgentResult:
        spec = self._agents.get(agent)
        if spec is None:
            raise UnknownAgentError(agent)
        trace: list[TraceEvent] = []
        final_agent, answer = await self._run_agent(spec, list(conversation), trace)
        return AgentResult(agent=final_agent, answer=answer, trace=tuple(trace))

    async def _run_agent(
        self, spec: AgentSpec, conversation: list[Message], trace: list[TraceEvent]
    ) -> tuple[str, str]:
        history = list(conversation)
        for _ in range(self._max_turns):
            tools = self._tools_for(spec, trace)
            reply = await self._model.complete(
                [Message.system(spec.instructions), *history],
                list(tools.values()),
                on_token=token_sink(spec.name),
            )
            if not reply.tool_calls:
                answer = reply.content or ""
                record(trace, TraceEvent(spec.name, "answer", answer))
                return spec.name, answer

            handoff = next((c for c in reply.tool_calls if c.name.startswith(HANDOFF_PREFIX)), None)
            if handoff is not None and handoff.name in tools:
                target = handoff.name.removeprefix(HANDOFF_PREFIX)
                record(trace, TraceEvent(spec.name, "handoff", target))
                if reply.content:
                    history.append(Message.assistant(reply.content))
                spec = self._agents[target]
                continue

            history.append(reply)
            for call in reply.tool_calls:
                result = await self._invoke(tools, call, spec.name, trace)
                history.append(Message.tool_result(call.id, result))
        raise MaxTurnsExceededError(spec.name, self._max_turns)

    def _tools_for(self, spec: AgentSpec, trace: list[TraceEvent]) -> dict[str, Tool]:
        tools = {name: self._tools[name] for name in spec.tools}
        for name in spec.delegates:
            tool = self._delegate_tool(self._agents[name], trace)
            tools[tool.name] = tool
        for name in spec.handoffs:
            tool = _handoff_tool(self._agents[name])
            tools[tool.name] = tool
        return tools

    def _delegate_tool(self, sub: AgentSpec, trace: list[TraceEvent]) -> Tool:
        async def handler(args: dict[str, Any]) -> str:
            _, answer = await self._run_agent(sub, [Message.user(str(args["request"]))], trace)
            return answer

        return Tool(
            name=f"{DELEGATE_PREFIX}{sub.name}",
            description=f"Ask the {sub.name} agent and get its answer back. {sub.description}",
            parameters={
                "type": "object",
                "properties": {"request": {"type": "string", "description": "Self-contained task."}},
                "required": ["request"],
            },
            handler=handler,
        )

    async def _invoke(
        self, tools: Mapping[str, Tool], call: ToolCall, agent: str, trace: list[TraceEvent]
    ) -> str:
        is_delegation = call.name.startswith(DELEGATE_PREFIX) and call.name not in self._tools
        record(trace, TraceEvent(agent, "delegation" if is_delegation else "tool_call", call.name))
        return await invoke_tool(tools, call)


def _handoff_tool(target: AgentSpec) -> Tool:
    async def handler(_: dict[str, Any]) -> str:
        return f"Transferred to {target.name}."

    return Tool(
        name=f"{HANDOFF_PREFIX}{target.name}",
        description=f"Transfer the conversation to the {target.name} agent. {target.description}",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def _validate(agents: Mapping[str, AgentSpec], tools: Mapping[str, Tool]) -> None:
    for spec in agents.values():
        if unknown := [t for t in spec.tools if t not in tools]:
            raise ValueError(f"Agent {spec.name!r} references unknown tools {unknown}")
        if unknown := [d for d in (*spec.delegates, *spec.handoffs) if d not in agents]:
            raise ValueError(f"Agent {spec.name!r} references unknown agents {unknown}")

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in path:
            raise ValueError(f"Delegation cycle: {' -> '.join((*path, name))}")
        for sub in agents[name].delegates:
            visit(sub, (*path, name))

    for name in agents:
        visit(name, ())
