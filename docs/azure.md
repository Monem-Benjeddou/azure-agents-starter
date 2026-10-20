# Azure setup and reference

Detailed setup for each Azure service, every setting, and the design behind the starter.
For a quick introduction, see the [README](../README.md).

## Contents

- [Azure OpenAI](#azure-openai)
- [Azure AI Search](#azure-ai-search)
- [Azure AI Foundry Agent Service](#azure-ai-foundry-agent-service)
- [Streaming](#streaming)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [References](#references)

## Azure OpenAI

```dotenv
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=            # leave empty to use Microsoft Entra ID
CHAT_MODEL=<chat deployment name>
EMBEDDING_MODEL=<embedding deployment name>
EMBEDDING_DIMENSIONS=1536        # must match the embedding model
```

When the key is empty, the app signs in with
[`DefaultAzureCredential`](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential):
`az login` on your machine, managed identity in Azure, or `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`
and `AZURE_TENANT_ID` for a service principal. See
[Entra ID for Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity).

Tools are sent with
[function calling](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/function-calling).
If a model returns malformed tool arguments, the error goes back to the model so it can correct itself.

The same adapter works with any OpenAI-compatible server (Ollama, vLLM, LM Studio, llama.cpp):
set `LLM_PROVIDER=openai_compatible` and `OPENAI_BASE_URL`.

## Azure AI Search

```dotenv
INDEX_PROVIDER=azure_search
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
AZURE_SEARCH_API_KEY=            # leave empty to use Microsoft Entra ID
AZURE_SEARCH_INDEX=knowledge
```

There are two modes:

- **Owned index (default).** The app creates the index on first ingest, splits documents into
  overlapping chunks, embeds them and uploads them.
- **Existing index (read-only).** Search an index that another system owns. The app never creates,
  changes or uploads to it; ingest is refused with HTTP 409.

  ```dotenv
  AZURE_SEARCH_EXISTING_INDEX=true
  AZURE_SEARCH_INDEX=<existing index>
  AZURE_SEARCH_SOURCE_FIELD=id
  AZURE_SEARCH_CONTENT_FIELD=content
  AZURE_SEARCH_VECTOR_FIELD=content_vector
  ```

Queries are [hybrid](https://learn.microsoft.com/azure/search/hybrid-search-overview): keyword and
[vector](https://learn.microsoft.com/azure/search/vector-search-overview) search combined. With
Entra ID, the identity needs the *Search Index Data Reader* role, plus *Search Index Data Contributor*
and *Search Service Contributor* for an owned index
([roles](https://learn.microsoft.com/azure/search/search-security-rbac)). Each pricing tier limits the
number of indexes ([limits](https://learn.microsoft.com/azure/search/search-limits-quotas-capacity)).

## Azure AI Foundry Agent Service

The same agents can run as hosted
[Foundry agents](https://learn.microsoft.com/azure/ai-foundry/agents/overview).

```dotenv
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL=<model deployment in the Foundry resource>
FOUNDRY_SEARCH_CONNECTION_ID=<project connection id of your Azure AI Search resource>
```

```bash
agents-starter provision-foundry      # prints FOUNDRY_AGENT_IDS; add it to .env
AGENT_RUNTIME=foundry agents-starter chat "What is the hotel limit?"
```

| In the agent catalog | In Foundry |
|----------------------|------------|
| `delegates` (agents as tools) | [Connected agents](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/connected-agents) |
| `search_knowledge_base` tool | The built-in [Azure AI Search tool](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/azure-ai-search) |
| Other tools | [Function tools](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/function-calling), run by this app |
| `handoffs` | Not supported by Foundry; these agents are skipped |

Provisioning is safe in shared projects:

- It updates only the agents it created before (listed in `FOUNDRY_AGENT_IDS`) and creates the rest.
  It never matches by name, so it can't overwrite another team's agent.
- Agent names get a prefix (`FOUNDRY_AGENT_PREFIX`, default `agents-starter-`).
- It refuses setups Foundry can't run. For example, connected agents can't call tools that run in
  this app.

Each request uses a new thread, which is deleted afterwards. Runs that take longer than
`FOUNDRY_RUN_TIMEOUT_SECONDS` are cancelled.

## Streaming

`POST /chat/stream` and `agents-starter chat --stream` send events as they happen:

| Event | Contains | Meaning |
|-------|----------|---------|
| `trace` | `agent`, `kind`, `detail` | A step: `delegation`, `handoff`, `tool_call`, `plan`, `turn`, `answer` |
| `token` | `agent`, `text` | Part of a model's reply |
| `result` | same body as `POST /chat` | The final answer and full trace (always last) |
| `error` | `detail` | The run failed after streaming started |

An unknown workflow is rejected with HTTP 404 before the stream starts. Events are delivered through a
[`ContextVar`](https://docs.python.org/3/library/contextvars.html), so agents running in parallel
stream too. With the Foundry runtime, steps stream live but the answer arrives in one piece.
Format: [Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html).

## HTTP API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | none |
| GET | `/workflows` | none |
| POST | `/chat` | `{"message": "...", "workflow": "support", "history": []}` |
| POST | `/chat/stream` | same as `/chat` |
| POST | `/documents` | `{"documents": [{"id": "...", "content": "..."}]}` |
| POST | `/search` | `{"query": "...", "top_k": 5}` |

Set `API_KEY` to require an `X-API-Key` header on every route except `/health`. `history` accepts
only `user` and `assistant` roles, so callers can't inject system prompts.

| Status | When |
|--------|------|
| 401 | Missing or wrong API key |
| 404 | Unknown workflow or agent |
| 409 | Writing to a read-only index |
| 422 | Invalid request body |
| 502 | An agent run failed or hit its turn limit |

## Configuration

All settings come from environment variables or `.env`; empty values count as unset. Missing
required settings are reported together at startup. Every setting is listed with a description in
[`config.py`](../src/agents_starter/config.py) and [`.env.example`](../.env.example).

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_PROVIDER` | `azure_openai` | `azure_openai` or `openai_compatible` |
| `INDEX_PROVIDER` | `memory` | `memory` or `azure_search` |
| `AGENT_RUNTIME` | `local` | `local` or `foundry` |
| `MAX_TURNS` | `8` | Tool-call limit per agent |
| `API_KEY` | unset | Protect the HTTP API |

## Architecture

The project uses [hexagonal architecture](https://alistair.cockburn.us/hexagonal-architecture/)
(ports and adapters). The core defines interfaces (ports); adapters implement them for each vendor.

```
src/agents_starter/
├── domain/          Data types, chunking, errors. No I/O.
├── application/     Use cases and ports. No vendor imports.
│   ├── ports.py         ChatModel, Embedder, VectorIndex, AgentRuntime
│   ├── orchestrator.py  Local runtime: tool loop, agents as tools, handoff
│   ├── patterns.py      Sequential, concurrent, group chat, magentic
│   ├── catalog.py       The agents and workflows
│   ├── chat.py          Run or stream a workflow
│   ├── knowledge.py     Ingest and search documents
│   └── events.py        Live events for streaming
├── adapters/        Azure OpenAI, Azure AI Search, Foundry, FastAPI, CLI
├── config.py        Settings
└── container.py     Picks the adapters (the only place that does)
```

Patterns are defined as in Microsoft's
[AI agent orchestration patterns](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
and [Agent Framework orchestrations](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/overview).
The magentic manager is based on [Magentic-One](https://arxiv.org/abs/2411.04468).

## Design decisions

- **Tool errors go back to the model**, so it can fix its own mistakes.
- **Wiring errors fail at startup**: unknown tools or agents, delegation loops, missing settings.
- **Requests are stateless**: the client sends the history.
- **Agents are declared once** and run on either runtime.
- **Data owned by other systems is read-only.**

## Limitations

- Agent instructions are tuned for capable models (gpt-4o class). Small local models need more
  explicit instructions.
- Group chat and magentic make many model calls. Over many rounds the manager can still get a detail
  wrong; prefer sequential or agents as tools when the steps are known.
- With the Foundry runtime, handoff isn't available and answers aren't streamed token by token.
- Not included: saved conversations, OpenTelemetry tracing, rate limiting.

## References

- [AI agent orchestration patterns](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns), Azure Architecture Center
- [Agent Framework orchestrations](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/overview): [sequential](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/sequential), [concurrent](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/concurrent), [handoff](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/handoff), [group chat](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/group-chat), [magentic](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/magentic)
- [Magentic-One](https://arxiv.org/abs/2411.04468), Microsoft Research
- [Azure AI Foundry Agent Service](https://learn.microsoft.com/azure/ai-foundry/agents/overview)
- [Azure AI Search: vector search](https://learn.microsoft.com/azure/search/vector-search-overview) and [hybrid search](https://learn.microsoft.com/azure/search/hybrid-search-overview)
- [Azure OpenAI API version lifecycle](https://learn.microsoft.com/azure/ai-services/openai/api-version-deprecation)
- [Authentication in the Azure SDK for Python](https://learn.microsoft.com/azure/developer/python/sdk/authentication/overview)
- [OpenAI Python SDK](https://github.com/openai/openai-python), [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [FastAPI custom responses](https://fastapi.tiangolo.com/advanced/custom-response/), [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
