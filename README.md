# Azure Agents Starter

[![CI](https://github.com/Monem-Benjeddou/azure-agents-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/Monem-Benjeddou/azure-agents-starter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python starter for building AI apps where several agents work together, on Azure or on your own
machine.

- **Six ways for agents to work together**, ready to use
- **Answers from your own documents** with Azure AI Search
- **Agents run inside your app or on Azure AI Foundry**, with no code changes
- **Live streaming** of each step and each word as it's generated
- **Works with Azure OpenAI or local models** like Ollama

## Try it locally

No Azure account needed. You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/Monem-Benjeddou/azure-agents-starter.git
cd azure-agents-starter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ollama pull qwen3:4b && ollama pull nomic-embed-text
cp .env.example .env
```

Set these in `.env`:

```dotenv
LLM_PROVIDER=openai_compatible
CHAT_MODEL=qwen3:4b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

Then load the sample documents and ask a question:

```bash
agents-starter ingest examples/knowledge
agents-starter chat "How much can I spend on a hotel in a capital city?"
```

```
  [triage] delegation: ask_researcher
  [researcher] tool_call: search_knowledge_base
  [researcher] answer: ...

Hotel costs are reimbursed up to 180 EUR per night in capital cities. [source: examples/knowledge/expenses-policy.md]
```

Add `--stream` to watch the answer appear word by word.

## Use Azure

Put your Azure OpenAI details in `.env`. Leave the key empty to sign in with `az login` instead.

```dotenv
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=
CHAT_MODEL=<chat deployment name>
EMBEDDING_MODEL=<embedding deployment name>
```

To add Azure AI Search or run the agents on Azure AI Foundry, see the
[Azure setup guide](docs/azure.md).

## How agents work together

Pick a workflow with `--workflow <name>`. List them all with `agents-starter workflows`.

| Workflow | Pattern | What happens |
|----------|---------|--------------|
| `support` (default) | Agents as tools | A lead agent asks specialists for help, then writes the answer |
| `research-report` | Sequential | Agents work one after another: research, write, review |
| `perspectives` | Concurrent | Agents work at the same time; their answers are combined |
| `handoff` | Handoff | A front-desk agent passes the conversation to the right specialist |
| `editorial` | Group chat | A manager decides which agent speaks next in a shared conversation |
| `magentic` | Magentic | A manager makes a plan, hands out steps, and adjusts the plan as it goes |

These follow
[Microsoft's orchestration patterns](https://learn.microsoft.com/azure/architecture/ai-ml/guide/ai-agent-design-patterns).

## Run it as an API

```bash
agents-starter serve        # interactive docs at http://127.0.0.1:8000/docs
```

```bash
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message": "What is the hotel limit?", "workflow": "support"}'
```

`POST /chat` returns the full answer at once; `POST /chat/stream` sends it live. Set `API_KEY` in
`.env` to protect the API. Full endpoint list: [Azure setup guide](docs/azure.md#http-api).

## Make it yours

- **Add an agent:** add an `AgentSpec` in [`catalog.py`](src/agents_starter/application/catalog.py).
- **Add a workflow:** add one line to `build_workflows` in the same file.
- **Add a tool:** create a `Tool` with a name, a JSON schema and an async function, then register it
  in [`container.py`](src/agents_starter/container.py).
- **Swap a service** (another database or model provider): write a new adapter in
  [`adapters/`](src/agents_starter/adapters) and select it in `container.py`.

The core of the app never imports Azure or OpenAI code directly. That's why services can be swapped
and why all tests run without the cloud. More in [Architecture](docs/azure.md#architecture).

## Development

```bash
pytest          # 50 tests, no cloud needed
ruff check .    # lint
mypy            # type checks
```

## License

[MIT](LICENSE)
