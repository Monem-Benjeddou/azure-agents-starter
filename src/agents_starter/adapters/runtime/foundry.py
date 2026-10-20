"""AgentRuntime over Azure AI Foundry Agent Service (hosted agents, threads and runs).

Multi-agent in Foundry uses connected agents: a parent agent calls children as tools on the
server side. `provision_agents` turns the same AgentSpec catalog the local runtime uses into
Foundry agents, so switching AGENT_RUNTIME does not change behaviour definitions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AzureAISearchTool,
    ConnectedAgentTool,
    FunctionDefinition,
    FunctionToolDefinition,
    MessageRole,
    RequiredFunctionToolCall,
    RunStatus,
    SubmitToolOutputsAction,
    ThreadMessageOptions,
    ToolOutput,
)

from ...application.events import record
from ...application.knowledge import SEARCH_TOOL_NAME
from ...application.tools import Tool, invoke_tool, parse_arguments
from ...domain.errors import AgentRunError, UnknownAgentError
from ...domain.models import AgentResult, AgentSpec, Message, ToolCall, TraceEvent

log = logging.getLogger(__name__)

_ACTIVE = {RunStatus.QUEUED, RunStatus.IN_PROGRESS, RunStatus.REQUIRES_ACTION}


class FoundryAgentRuntime:
    def __init__(
        self,
        client: AgentsClient,
        agent_ids: Mapping[str, str],
        tools: Sequence[Tool],
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._agent_ids = dict(agent_ids)
        self._tools = {t.name: t for t in tools}
        self._poll_interval = poll_interval
        self._timeout = timeout

    async def run(self, agent: str, conversation: Sequence[Message]) -> AgentResult:
        agent_id = self._agent_ids.get(agent)
        if agent_id is None:
            raise UnknownAgentError(agent)

        thread = await self._client.threads.create(
            messages=[
                ThreadMessageOptions(role=m.role, content=m.content)
                for m in conversation
                if m.role in ("user", "assistant") and m.content
            ]
        )
        try:
            trace: list[TraceEvent] = []
            await self._drive_run(thread.id, agent_id, agent, trace)
            last = await self._client.messages.get_last_message_text_by_role(
                thread_id=thread.id, role=MessageRole.AGENT
            )
            answer = last.text.value if last else ""
            record(trace, TraceEvent(agent, "answer", answer))
            return AgentResult(agent=agent, answer=answer, trace=tuple(trace))
        finally:
            await self._client.threads.delete(thread.id)

    async def _drive_run(self, thread_id: str, agent_id: str, agent: str, trace: list[TraceEvent]) -> None:
        run = await self._client.runs.create(thread_id=thread_id, agent_id=agent_id)
        deadline = time.monotonic() + self._timeout
        while run.status in _ACTIVE:
            if run.status == RunStatus.REQUIRES_ACTION and isinstance(
                run.required_action, SubmitToolOutputsAction
            ):
                outputs = [
                    ToolOutput(tool_call_id=call.id, output=await self._call(call, agent, trace))
                    for call in run.required_action.submit_tool_outputs.tool_calls
                ]
                run = await self._client.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=outputs
                )
                continue
            if time.monotonic() > deadline:
                await self._client.runs.cancel(thread_id=thread_id, run_id=run.id)
                raise AgentRunError(f"Foundry run {run.id} timed out after {self._timeout}s")
            await asyncio.sleep(self._poll_interval)
            run = await self._client.runs.get(thread_id=thread_id, run_id=run.id)

        if run.status != RunStatus.COMPLETED:
            raise AgentRunError(f"Foundry run {run.id} ended {run.status}: {run.last_error}")

    async def _call(self, call: Any, agent: str, trace: list[TraceEvent]) -> str:
        if not isinstance(call, RequiredFunctionToolCall):
            return f"Error: unsupported tool call type {call.type!r}"
        record(trace, TraceEvent(agent, "tool_call", call.function.name))
        arguments = parse_arguments(call.function.arguments)
        return await invoke_tool(self._tools, ToolCall(call.id, call.function.name, arguments))


async def provision_agents(
    client: AgentsClient,
    agents: Sequence[AgentSpec],
    tools: Sequence[Tool],
    *,
    model: str,
    search_connection_id: str | None = None,
    search_index_name: str | None = None,
    existing_ids: Mapping[str, str] | None = None,
    name_prefix: str = "",
) -> dict[str, str]:
    """Create or update a Foundry agent per spec (children first). Returns spec name -> agent id.

    Only agents listed in `existing_ids` (the output of a previous run) are updated; everything
    else is created. Matching by name would be slow in large projects and could overwrite
    another team's agent that happens to share a name.
    """
    by_name = {a.name: a for a in agents}
    tool_map = {t.name: t for t in tools}
    existing = existing_ids or {}
    _check_foundry_constraints(agents, search_connection_id)

    ids: dict[str, str] = {}
    for spec in _children_first(by_name):
        if spec.handoffs:
            log.warning("skipping %s: Foundry agents do not support handoffs", spec.name)
            continue
        definitions: list[Any] = []
        resources = None
        for name in spec.tools:
            if name == SEARCH_TOOL_NAME and search_connection_id:
                search = AzureAISearchTool(
                    index_connection_id=search_connection_id, index_name=search_index_name or ""
                )
                definitions += search.definitions
                resources = search.resources
            else:
                t = tool_map[name]
                definitions.append(
                    FunctionToolDefinition(
                        function=FunctionDefinition(
                            name=t.name, description=t.description, parameters=t.parameters
                        )
                    )
                )
        for child in spec.delegates:
            definitions += ConnectedAgentTool(
                id=ids[child], name=child, description=by_name[child].description
            ).definitions

        fields: dict[str, Any] = dict(
            model=model,
            name=f"{name_prefix}{spec.name}",
            description=spec.description,
            instructions=spec.instructions,
            tools=definitions,
            tool_resources=resources,
        )
        if spec.name in existing:
            agent = await client.update_agent(existing[spec.name], **fields)
        else:
            agent = await client.create_agent(**fields)
        ids[spec.name] = agent.id
        log.info("provisioned %s -> %s", spec.name, agent.id)
    return ids


def _check_foundry_constraints(agents: Sequence[AgentSpec], search_connection_id: str | None) -> None:
    children = {d for a in agents for d in a.delegates}
    for spec in agents:
        if spec.name not in children:
            continue
        client_side = [t for t in spec.tools if not (t == SEARCH_TOOL_NAME and search_connection_id)]
        if client_side:
            raise ValueError(
                f"Agent {spec.name!r} is a connected agent in Foundry and cannot call client-side "
                f"tools {client_side}. Set FOUNDRY_SEARCH_CONNECTION_ID to use the built-in "
                "Azure AI Search tool, or make the agent a top-level one."
            )


def _children_first(agents: Mapping[str, AgentSpec]) -> list[AgentSpec]:
    ordered: list[AgentSpec] = []

    def visit(name: str) -> None:
        if any(a.name == name for a in ordered):
            return
        for child in agents[name].delegates:
            visit(child)
        ordered.append(agents[name])

    for name in agents:
        visit(name)
    return ordered
