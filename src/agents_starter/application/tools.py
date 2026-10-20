from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.models import ToolCall

log = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    """A function the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Decode model-produced JSON arguments.

    Smaller models emit malformed JSON; an empty dict makes `invoke_tool` report the missing
    arguments back to the model instead of crashing the run.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        log.warning("invalid JSON tool arguments: %r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def invoke_tool(tools: Mapping[str, Tool], call: ToolCall) -> str:
    """Run one tool call. Errors go back to the model as text so it can correct itself."""
    tool = tools.get(call.name)
    if tool is None:
        return f"Error: unknown tool {call.name!r}. Available: {', '.join(tools) or 'none'}."
    missing = [k for k in tool.parameters.get("required", []) if k not in call.arguments]
    if missing:
        return f"Error: missing required arguments {missing} for {call.name!r}."
    try:
        return await tool.handler(call.arguments)
    except Exception as exc:
        log.exception("tool %s failed", call.name)
        return f"Error: {call.name} failed: {exc}"
