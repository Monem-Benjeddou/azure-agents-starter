"""Composition root: the only module that picks concrete adapters for the ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from azure.ai.agents.aio import AgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .adapters.index.azure_search import AzureSearchIndex, IndexFields
from .adapters.index.in_memory import InMemoryIndex
from .adapters.llm.openai_compatible import OpenAIChatModel, OpenAIEmbedder
from .adapters.runtime.foundry import FoundryAgentRuntime
from .application.catalog import AGENTS, DEFAULT_WORKFLOW, build_workflows
from .application.chat import ChatService
from .application.knowledge import KnowledgeBase
from .application.orchestrator import LocalAgentRuntime
from .application.ports import AgentRuntime, VectorIndex
from .application.tools import Tool
from .config import Settings

_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass(frozen=True)
class Container:
    chat: ChatService
    knowledge: KnowledgeBase
    tools: tuple[Tool, ...]


@asynccontextmanager
async def build_container(settings: Settings) -> AsyncIterator[Container]:
    async with AsyncExitStack() as stack:
        credential = _credential(settings, stack)
        llm_client = await stack.enter_async_context(_llm_client(settings, credential))
        embedder = OpenAIEmbedder(
            llm_client, settings.embedding_model, dimensions=settings.embedding_dimensions
        )
        knowledge = KnowledgeBase(embedder, await _index(settings, credential, stack))
        tools = (knowledge.search_tool(),)

        model = OpenAIChatModel(llm_client, settings.chat_model, temperature=settings.temperature)
        runtime: AgentRuntime
        if settings.agent_runtime == "foundry":
            agents_client = await stack.enter_async_context(foundry_client(settings, credential))
            runtime = FoundryAgentRuntime(
                agents_client,
                settings.foundry_agent_ids,
                tools,
                timeout=settings.foundry_run_timeout_seconds,
            )
        else:
            runtime = LocalAgentRuntime(model, AGENTS, tools, max_turns=settings.max_turns)

        chat = ChatService(build_workflows(runtime, model), DEFAULT_WORKFLOW)
        yield Container(chat=chat, knowledge=knowledge, tools=tools)


def foundry_client(settings: Settings, credential: AsyncTokenCredential | None) -> AgentsClient:
    endpoint = _required(settings.foundry_project_endpoint, "FOUNDRY_PROJECT_ENDPOINT")
    return AgentsClient(endpoint=endpoint, credential=_entra(credential))


def _entra(credential: AsyncTokenCredential | None) -> AsyncTokenCredential:
    if credential is None:
        raise ValueError("This configuration requires Microsoft Entra ID credentials")
    return credential


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"Missing required setting {name}")
    return value


def _credential(settings: Settings, stack: AsyncExitStack) -> DefaultAzureCredential | None:
    needs_entra = (
        (settings.llm_provider == "azure_openai" and settings.azure_openai_api_key is None)
        or (settings.index_provider == "azure_search" and settings.azure_search_api_key is None)
        or settings.agent_runtime == "foundry"
    )
    if not needs_entra:
        return None
    credential = DefaultAzureCredential()
    stack.push_async_callback(credential.close)
    return credential


def _llm_client(settings: Settings, credential: AsyncTokenCredential | None) -> AsyncOpenAI:
    if settings.llm_provider == "openai_compatible":
        return AsyncOpenAI(
            base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value()
        )
    endpoint = _required(settings.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT")
    if settings.azure_openai_api_key is not None:
        return AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key.get_secret_value(),
        )
    return AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=settings.azure_openai_api_version,
        azure_ad_token_provider=get_bearer_token_provider(_entra(credential), _COGNITIVE_SCOPE),
    )


async def _index(
    settings: Settings, credential: AsyncTokenCredential | None, stack: AsyncExitStack
) -> VectorIndex:
    if settings.index_provider == "memory":
        return InMemoryIndex(settings.memory_index_path)

    endpoint = _required(settings.azure_search_endpoint, "AZURE_SEARCH_ENDPOINT")
    search_credential: AzureKeyCredential | AsyncTokenCredential
    if settings.azure_search_api_key is not None:
        search_credential = AzureKeyCredential(settings.azure_search_api_key.get_secret_value())
    else:
        search_credential = _entra(credential)
    search = await stack.enter_async_context(
        SearchClient(endpoint, settings.azure_search_index, search_credential)
    )
    indexes = await stack.enter_async_context(SearchIndexClient(endpoint, search_credential))
    existing = None
    if settings.azure_search_existing_index:
        existing = IndexFields(
            source=settings.azure_search_source_field,
            content=settings.azure_search_content_field,
            vector=settings.azure_search_vector_field,
        )
    return AzureSearchIndex(
        search,
        indexes,
        index_name=settings.azure_search_index,
        dimensions=settings.embedding_dimensions,
        existing=existing,
    )
