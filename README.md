# Azure Agents Starter

A production-shaped starting point for LLM applications on Azure, built with
**hexagonal architecture** (ports and adapters). It includes:

- **All six multi-agent orchestration patterns**: agents as tools, sequential, concurrent,
  handoff, group chat and magentic.
- **Two interchangeable agent runtimes** behind one port: an in-process tool-calling loop, or
  hosted agents on **Azure AI Foundry Agent Service**.
- **Retrieval (RAG)** with hybrid keyword + vector search on **Azure AI Search**. It works on
  indexes it creates, or read-only on indexes other systems already own.
- **Streaming**: Server-Sent Events with live agent steps and token-by-token output.
- **Swappable models**: Azure OpenAI or Foundry deployments, or any OpenAI-compatible server
  (Ollama, vLLM, LM Studio, llama.cpp).
- **Entry points**: an HTTP API (FastAPI) and a CLI.

The core has no Azure imports. Every SDK sits behind a port, so the full test suite runs
offline and a vendor can be swapped without touching business logic.

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Orchestration patterns](#orchestration-patterns)
- [Streaming](#streaming)
- [HTTP API](#http-api)
- [Azure OpenAI](#azure-openai)
- [Azure AI Search](#azure-ai-search)
- [Azure AI Foundry Agent Service](#azure-ai-foundry-agent-service)
- [Configuration reference](#configuration-reference)
- [Extending](#extending)
- [Development](#development)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [References](#references)

## Architecture

```
           inbound adapters                     application (core)                   outbound adapters
 ┌──────────────────────────────┐   ┌─────────────────────────────────────┐   ┌──────────────────────────────┐
 │ HTTP  (FastAPI, SSE)         │──▶│ ChatService        KnowledgeBase    │   │ OpenAIChatModel / Embedder   │
 │ CLI   (ingest, chat,         │   │ patterns: Sequential, Concurrent,   │──▶│  ├ Azure OpenAI / Foundry    │
 │        workflows, serve,     │   │   GroupChat, Magentic, SingleAgent  │   │  └ Ollama / vLLM / LM Studio │
 │        provision-foundry)    │   │ LocalAgentRuntime (tools, handoff)  │   │ AzureSearchIndex / InMemory  │
 └──────────────────────────────┘   │ ports: ChatModel, Embedder,         │   │ FoundryAgentRuntime          │
                                    │        VectorIndex, AgentRuntime    │   └──────────────────────────────┘
                                    └─────────────────────────────────────┘
                                         domain: models, chunking, errors (pure Python)
```

```
src/agents_starter/
├── domain/              Value objects, chunking, errors. No I/O.
├── application/         Use cases and ports. Depends only on domain.
│   ├── ports.py         ChatModel, Embedder, VectorIndex, AgentRuntime
│   ├── orchestrator.py  LocalAgentRuntime: tool loop, agents as tools, handoff
│   ├── patterns.py      Sequential, Concurrent, GroupChat, Magentic, SingleAgent
│   ├── events.py        Live event delivery for streaming
│   ├── knowledge.py     Ingest, search, search exposed as a tool
│   ├── chat.py          ChatService: run or stream a named workflow
│   ├── catalog.py       The agent team and one workflow per pattern
│   └── tools.py         Tool type and safe invocation
├── adapters/
│   ├── llm/             OpenAI SDK (Azure OpenAI and OpenAI-compatible servers)
│   ├── index/           Azure AI Search, in-memory
│   ├── runtime/         Foundry Agent Service runtime and provisioning
│   └── inbound/         FastAPI app, CLI
├── config.py            Typed settings from environment / .env, validated at startup
└── container.py         Composition root: the only place adapters are chosen
```

Dependencies point inward only: `adapters → application → domain`. The pattern is described
in Alistair Cockburn's [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/).

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

**Option A: Azure OpenAI.** Set `AZURE_OPENAI_ENDPOINT`, then either `AZURE_OPENAI_API_KEY`
or sign in with `az login` (Microsoft Entra ID). Set `CHAT_MODEL` and `EMBEDDING_MODEL` to your
deployment names.

**Option B: fully local with [Ollama](https://docs.ollama.com/api/openai-compatibility).**

```bash
ollama pull qwen3:4b && ollama pull nomic-embed-text
```

```dotenv
LLM_PROVIDER=openai_compatible
CHAT_MODEL=qwen3:4b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

Then:

```bash
agents-starter ingest examples/knowledge
agents-starter workflows
agents-starter chat "How much can I spend on a hotel in a capital city?"
#   [triage] delegation: ask_researcher
#   [researcher] tool_call: search_knowledge_base
#   [researcher] answer: ...
#
# Hotel costs are reimbursed up to 180 EUR per night in capital cities. [source: examples/knowledge/expenses-policy.md]

agents-starter chat "Draft a note about the travel policy" --workflow research-report --stream
agents-starter serve            # http://127.0.0.1:8000/docs
```

## Orchestration patterns

Each pattern is a **workflow** with a name. Every agent is also available as a single-agent
workflow under its own name. List them with `agents-starter workflows` or `GET /workflows`.

| Pattern | Workflow | How it works | Use it when |
|---------|----------|--------------|-------------|
| Agents as tools (supervisor) | `support` | A supervisor calls specialists as tools and composes their answers. | One agent should stay in charge of the reply. |
| Sequential | `research-report` | researcher → writer → reviewer. Each agent sees the conversation plus all earlier outputs. | The work is a fixed pipeline of steps. |
| Concurrent | `perspectives` | researcher and critic work on the same input in parallel; the writer merges the results. | You want independent views, or lower latency. |
| Handoff | `handoff` | The concierge transfers the whole conversation to one specialist, who answers directly. | Routing to the right expert, as in a support desk. |
| Group chat | `editorial` | researcher, writer and reviewer share one transcript; an LLM manager picks who speaks next or ends the chat. A round-robin mode is available. | Iterative refinement through discussion. |
| Magentic | `magentic` | A manager writes a plan, assigns each step, tracks progress in a ledger, replans when stalled, then writes the final answer. | Open-ended tasks where the steps aren't known in advance. |

Where each pattern lives:

- **Agents as tools** and **handoff** are part of `LocalAgentRuntime`, because they need
  control of the tool-calling loop. Sub-agents appear to the model as `ask_<agent>` tools, and
  handoffs as `transfer_to_<agent>` tools.
- **Sequential, concurrent, group chat and magentic** are composed in `application/patterns.py`
  over the `AgentRuntime` port, so they work unchanged with hosted Foundry agents.

Safeguards: unknown tools or agents and delegation cycles are rejected at startup. Every
agent has a turn limit (`MAX_TURNS`). Group chat and magentic have round limits.

These patterns follow Microsoft's definitions:
[AI agent orchestration patterns](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
(Azure Architecture Center) and
[Agent Framework orchestrations](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/overview):
[sequential](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/sequential),
[concurrent](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/concurrent),
[handoff](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/handoff),
[group chat](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/group-chat),
[magentic](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/magentic).
The magentic manager is based on
[Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks](https://arxiv.org/abs/2411.04468).

## Streaming

Every workflow can stream. You get each step as it happens (delegation, handoff, tool call,
plan, turn), every model completion token by token tagged with the agent producing it, and
then the final result.

```bash
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message": "What is the hotel limit?", "workflow": "support"}'
```

```
event: trace
data: {"agent": "triage", "kind": "delegation", "detail": "ask_researcher"}

event: token
data: {"agent": "researcher", "text": "Hotel costs are reimbursed"}

event: result
data: {"agent": "triage", "answer": "...", "trace": [...]}
```

| Event | Payload | Meaning |
|-------|---------|---------|
| `trace` | `agent`, `kind`, `detail` | A step: `delegation`, `handoff`, `tool_call`, `plan`, `turn` or `answer` |
| `token` | `agent`, `text` | Part of a model completion, in order |
| `result` | same body as `POST /chat` | Final answer and full trace; always the last event |
| `error` | `detail` | The run failed after streaming started |

An unknown workflow is rejected with HTTP 404 before the stream starts. On the CLI, add
`--stream` to `chat`.

How it works: runtimes and patterns publish events to a listener stored in a
[`ContextVar`](https://docs.python.org/3/library/contextvars.html). Tasks created with
`asyncio.gather` inherit it, so concurrent agents stream too, without a callback passed through
every call. Model tokens come from the OpenAI SDK's
[streaming responses](https://github.com/openai/openai-python#streaming-responses), and tool-call
fragments are reassembled into complete calls. With the Foundry runtime, step events stream live
and the answer arrives as one piece.

Further reading: [Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
(HTML standard), [Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
(MDN), and [StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/) (FastAPI).

## HTTP API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | none |
| GET | `/workflows` | none. Returns `{name: pattern}` |
| POST | `/chat` | `{"message": "...", "workflow": "support", "history": [{"role": "user", "content": "..."}]}` |
| POST | `/chat/stream` | same as `/chat`. Responds with `text/event-stream` |
| POST | `/documents` | `{"documents": [{"id": "...", "content": "...", "metadata": {}}]}` |
| POST | `/search` | `{"query": "...", "top_k": 5}` |

The interactive schema is served at `/docs`. Set `API_KEY` to require an `X-API-Key` header on
every route except `/health`. Requests are validated with size limits, and `history` only
accepts `user` and `assistant` roles, so clients cannot inject system prompts.

| Status | When |
|--------|------|
| 401 | Missing or wrong API key |
| 404 | Unknown workflow or agent |
| 409 | Writing to a read-only (existing) index |
| 422 | Invalid request body |
| 502 | An agent run failed or exceeded its turn limit |

## Azure OpenAI

One adapter serves chat and embeddings through the official `openai` SDK: `AsyncAzureOpenAI`
for Azure, `AsyncOpenAI` with a `base_url` for other servers. Tools are sent with
[function calling](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/function-calling).
Malformed tool arguments, which smaller models often produce, are reported back to the model
instead of failing the request.

Authentication uses an API key, or **Microsoft Entra ID** when the key is unset, through
[`DefaultAzureCredential`](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential):
`az login` locally, managed identity in Azure, or `AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET`/`AZURE_TENANT_ID`
for a service principal. See
[Entra ID for Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity)
and [authentication in the Azure SDK for Python](https://learn.microsoft.com/azure/developer/python/sdk/authentication/overview).
`AZURE_OPENAI_API_VERSION` defaults to a GA version; see
[API version lifecycle](https://learn.microsoft.com/azure/ai-services/openai/api-version-deprecation).

## Azure AI Search

```dotenv
INDEX_PROVIDER=azure_search
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX=knowledge
```

**Owned index (default).** Created on first ingest with a key, content, metadata and an HNSW
vector field. Documents are split into overlapping chunks and embedded in batches.
`EMBEDDING_DIMENSIONS` must match your embedding model.

**Existing index (read-only).** Search an index another system owns. The app never creates,
changes or uploads to it, and ingest is refused with HTTP 409.

```dotenv
AZURE_SEARCH_INDEX=meetings
AZURE_SEARCH_EXISTING_INDEX=true
AZURE_SEARCH_SOURCE_FIELD=id
AZURE_SEARCH_CONTENT_FIELD=content
AZURE_SEARCH_VECTOR_FIELD=content_vector
```

Queries are hybrid (keyword + vector). Passages are capped in length before they reach the
model, because whole-record indexes can hold very large documents. With Entra ID, the identity
needs the *Search Index Data Reader* role (plus *Search Index Data Contributor* and *Search
Service Contributor* for an owned index).

Further reading: [vector search](https://learn.microsoft.com/azure/search/vector-search-overview),
[hybrid search](https://learn.microsoft.com/azure/search/hybrid-search-overview),
[role-based access](https://learn.microsoft.com/azure/search/search-security-rbac), and
[service limits](https://learn.microsoft.com/azure/search/search-limits-quotas-capacity) (each
tier caps the number of indexes).

## Azure AI Foundry Agent Service

The same agent catalog can run as hosted
[Foundry agents](https://learn.microsoft.com/azure/ai-foundry/agents/overview):

```dotenv
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_SEARCH_CONNECTION_ID=<project connection id of your Azure AI Search resource>
FOUNDRY_MODEL=<model deployment in the Foundry resource>
```

```bash
agents-starter provision-foundry      # prints FOUNDRY_AGENT_IDS; add it to .env
AGENT_RUNTIME=foundry agents-starter chat "What is the hotel limit?"
```

How the catalog maps to Foundry:

| Catalog | Foundry |
|---------|---------|
| `delegates=(...)` (agents as tools) | [Connected agents](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/connected-agents), run server-side |
| `search_knowledge_base` tool | The built-in [Azure AI Search tool](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/azure-ai-search) on the same index |
| Other tools | [Function tools](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/function-calling), executed by this app when a run reports `requires_action` |
| `handoffs=(...)` | Not supported by Foundry; these agents are skipped with a warning |

Provisioning is safe for shared projects:

- It **updates only the agent IDs it created before** (from `FOUNDRY_AGENT_IDS`) and creates
  the rest. It never matches by name, so it can't overwrite another team's agent that happens
  to share a name.
- Agent names get a prefix (`FOUNDRY_AGENT_PREFIX`, default `agents-starter-`).
- It refuses configurations Foundry cannot run. For example, connected agents can't call
  client-side function tools.

Each request runs in a new thread, which is deleted when the run ends. Runs that take longer
than `FOUNDRY_RUN_TIMEOUT_SECONDS` are cancelled. Sequential, concurrent, group chat and
magentic work on top of Foundry agents unchanged.

## Configuration reference

All settings are environment variables (or `.env`). Empty values count as unset. Required
settings are checked at startup and reported together. The full list with descriptions is in
[`config.py`](src/agents_starter/config.py) and [`.env.example`](.env.example). Settings are
loaded with [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_PROVIDER` | `azure_openai` | `azure_openai` or `openai_compatible` |
| `CHAT_MODEL`, `EMBEDDING_MODEL` | `gpt-4o-mini`, `text-embedding-3-small` | Model or deployment names |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the embedding model |
| `INDEX_PROVIDER` | `memory` | `memory` or `azure_search` |
| `AZURE_SEARCH_EXISTING_INDEX` | `false` | Read-only mode for an index another system owns |
| `AGENT_RUNTIME` | `local` | `local` or `foundry` |
| `MAX_TURNS` | `8` | Tool-loop limit per agent |
| `API_KEY` | unset | Require `X-API-Key` on the HTTP API |

## Extending

- **New tool:** create a `Tool` (name, JSON schema, async handler), add it to `tools` in
  `container.py`, and list it on an agent in `catalog.py`.
- **New agent:** add an `AgentSpec` to `catalog.py`.
- **New workflow:** add an entry to `build_workflows` in `catalog.py`, for example
  `Sequential(runtime, ("researcher", "writer"))`.
- **New pattern:** implement the `Workflow` protocol (`pattern` and `async run(conversation)`)
  over the `AgentRuntime` port. Record steps with `events.record` so they stream.
- **New vendor** (another vector database or LLM API): implement the port in `adapters/` and
  select it in `container.py`.

## Development

```bash
pytest                                   # offline: fakes for models, embeddings and Azure clients
ruff check . && ruff format --check .
mypy                                     # strict
docker build -t agents-starter .
docker run --env-file .env -p 8000:8000 agents-starter
```

GitHub Actions runs lint, type checks and tests on Python 3.10 and 3.12.

## Design decisions

- **Errors from tools go back to the model.** Unknown tools, missing arguments, malformed
  JSON and handler exceptions become tool messages, so the model can correct itself.
- **Fail fast on wiring.** Unknown tools or agents, delegation cycles and missing settings
  raise at startup, not mid-conversation.
- **Stateless requests.** The client sends history; the server stores none.
- **One catalog, two runtimes.** Agents are declared once (`AgentSpec`) and run in-process or
  in Foundry.
- **Existing data is read-only.** An index owned by another system is never modified.
- **Docstrings over comments.** Public types document intent; the code carries no inline
  comments ([PEP 257](https://peps.python.org/pep-0257/)).

## Limitations

- Instructions in `catalog.py` are tuned for capable models (gpt-4o class). Small local models
  need more explicit instructions to delegate reliably.
- Magentic and group chat run many model calls and cost more; their round limits keep that
  bounded. Over many rounds the manager can still mislabel a detail, so prefer sequential or
  agents-as-tools when the steps are known.
- With the Foundry runtime, handoff is unavailable and answers aren't streamed token by token.
- Not included: persistent conversation storage, OpenTelemetry tracing, rate limiting.

## References

**Patterns and architecture**
- [AI agent orchestration patterns](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns), Azure Architecture Center
- [Agent Framework orchestrations](https://learn.microsoft.com/agent-framework/user-guide/workflows/orchestrations/overview), Microsoft Learn
- [Magentic-One](https://arxiv.org/abs/2411.04468), Microsoft Research
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/), Alistair Cockburn

**Azure**
- [Azure AI Foundry Agent Service](https://learn.microsoft.com/azure/ai-foundry/agents/overview)
- [Connected agents](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/connected-agents)
- [Azure AI Search tool for agents](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/azure-ai-search)
- [Function calling with agents](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/function-calling)
- [Function calling with Azure OpenAI](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/function-calling)
- [Entra ID for Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity)
- [Azure OpenAI API version lifecycle](https://learn.microsoft.com/azure/ai-services/openai/api-version-deprecation)
- [Vector search](https://learn.microsoft.com/azure/search/vector-search-overview) and [hybrid search](https://learn.microsoft.com/azure/search/hybrid-search-overview) in Azure AI Search
- [Azure AI Search role-based access](https://learn.microsoft.com/azure/search/search-security-rbac)
- [Azure AI Search service limits](https://learn.microsoft.com/azure/search/search-limits-quotas-capacity)
- [DefaultAzureCredential](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential)
- [Authentication in the Azure SDK for Python](https://learn.microsoft.com/azure/developer/python/sdk/authentication/overview)

**SDKs and libraries**
- [azure-ai-agents](https://pypi.org/project/azure-ai-agents/), [azure-search-documents](https://pypi.org/project/azure-search-documents/), [openai](https://github.com/openai/openai-python)
- [FastAPI custom responses](https://fastapi.tiangolo.com/advanced/custom-response/)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [contextvars](https://docs.python.org/3/library/contextvars.html)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)

**Streaming**
- [Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html), HTML standard
- [Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events), MDN

## License

[MIT](LICENSE)
