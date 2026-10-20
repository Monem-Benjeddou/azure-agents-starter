"""Command-line inbound adapter: ingest files, chat, serve the API, provision Foundry agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from azure.identity.aio import DefaultAzureCredential

from ...adapters.runtime.foundry import provision_agents
from ...application.catalog import AGENTS
from ...config import Settings
from ...container import build_container, foundry_client
from ...domain.errors import DomainError
from ...domain.models import Document, TokenDelta, TraceEvent

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agents-starter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Index .md/.txt/.rst files or folders")
    p.add_argument("paths", nargs="+", type=Path)

    p = sub.add_parser("chat", help="Ask a question through a workflow")
    p.add_argument("message")
    p.add_argument("--workflow", help="Workflow or agent name (default: support)")
    p.add_argument("--stream", action="store_true", help="Print steps and tokens as they happen")

    sub.add_parser("workflows", help="List workflows and their orchestration patterns")

    p = sub.add_parser("serve", help="Run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    sub.add_parser("provision-foundry", help="Create/update Foundry agents from the catalog")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings()

    if args.command == "serve":
        import uvicorn

        from .http import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    commands = {
        "ingest": lambda: _ingest(settings, args.paths),
        "chat": lambda: _chat(settings, args.message, args.workflow, stream=args.stream),
        "workflows": lambda: _workflows(settings),
        "provision-foundry": lambda: _provision(settings),
    }
    try:
        asyncio.run(commands[args.command]())
    except DomainError as exc:
        raise SystemExit(f"Error: {exc}") from None


async def _ingest(settings: Settings, paths: list[Path]) -> None:
    docs = [
        Document(id=str(f), content=f.read_text(encoding="utf-8"), metadata={"path": str(f)})
        for f in _text_files(paths)
    ]
    async with build_container(settings) as c:
        chunks = await c.knowledge.ingest(docs)
    print(f"Indexed {len(docs)} documents as {chunks} chunks.")


async def _workflows(settings: Settings) -> None:
    async with build_container(settings) as c:
        for name, pattern in c.chat.workflows.items():
            print(f"{name:<18} {pattern}")


async def _chat(settings: Settings, message: str, workflow: str | None, *, stream: bool) -> None:
    if stream:
        await _chat_stream(settings, message, workflow)
        return
    async with build_container(settings) as c:
        result = await c.chat.ask(message, workflow=workflow)
    for event in result.trace[:-1]:
        print(f"  [{event.agent}] {event.kind}: {event.detail}")
    print(f"\n{result.answer}")


async def _chat_stream(settings: Settings, message: str, workflow: str | None) -> None:
    speaking: str | None = None
    async with build_container(settings) as c:
        async for event in c.chat.stream(message, workflow=workflow):
            if isinstance(event, TokenDelta):
                if event.agent != speaking:
                    print(f"\n[{event.agent}] ", end="", flush=True)
                    speaking = event.agent
                print(event.text, end="", flush=True)
            elif isinstance(event, TraceEvent) and event.kind != "answer":
                print(f"\n  [{event.agent}] {event.kind}: {event.detail}", flush=True)
                speaking = None
    print()


async def _provision(settings: Settings) -> None:
    if not settings.foundry_project_endpoint:
        raise SystemExit("Set FOUNDRY_PROJECT_ENDPOINT to provision Foundry agents.")
    async with (
        DefaultAzureCredential() as credential,
        build_container(settings) as c,
        foundry_client(settings, credential) as client,
    ):
        ids = await provision_agents(
            client,
            AGENTS,
            c.tools,
            model=settings.foundry_model or settings.chat_model,
            search_connection_id=settings.foundry_search_connection_id,
            search_index_name=settings.azure_search_index,
            existing_ids=settings.foundry_agent_ids,
            name_prefix=settings.foundry_agent_prefix,
        )
    print("Add this to your .env:\n")
    print(f"FOUNDRY_AGENT_IDS='{json.dumps(ids)}'")


def _text_files(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(f for f in path.rglob("*") if f.suffix in TEXT_SUFFIXES)
        elif path.is_file():
            yield path
        else:
            raise SystemExit(f"Not found: {path}")
