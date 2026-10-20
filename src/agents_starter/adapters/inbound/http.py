"""FastAPI inbound adapter: translates HTTP to application use cases, with no business logic."""

import hmac
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ...config import Settings
from ...container import Container, build_container
from ...domain.errors import DomainError, ReadOnlyIndexError, UnknownAgentError, UnknownWorkflowError
from ...domain.models import AgentResult, Document, Message, TokenDelta, TraceEvent

ContainerFactory = Callable[[Settings], AbstractAsyncContextManager[Container]]

log = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=32_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    workflow: str | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)


class TraceEventOut(BaseModel):
    agent: str
    kind: str
    detail: str


class ChatResponse(BaseModel):
    agent: str
    answer: str
    trace: list[TraceEventOut]


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHitOut(BaseModel):
    document_id: str
    content: str
    score: float


def create_app(
    settings: Settings | None = None, container_factory: ContainerFactory = build_container
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with container_factory(settings) as container:
            app.state.container = container
            yield

    app = FastAPI(title="Azure Agents Starter", version="0.1.0", lifespan=lifespan)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: Annotated[str | None, Depends(api_key_header)]) -> None:
        expected = settings.api_key
        if expected and not hmac.compare_digest(key or "", expected.get_secret_value()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def container(request: Request) -> Container:
        container: Container = request.app.state.container
        return container

    Secured = [Depends(require_api_key)]
    AppContainer = Annotated[Container, Depends(container)]

    @app.exception_handler(UnknownAgentError)
    @app.exception_handler(UnknownWorkflowError)
    async def not_found(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ReadOnlyIndexError)
    async def read_only_index(_: Request, exc: ReadOnlyIndexError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DomainError)
    async def domain_error(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/workflows", dependencies=Secured)
    async def workflows(c: AppContainer) -> dict[str, str]:
        return c.chat.workflows

    @app.post("/chat", dependencies=Secured)
    async def chat(body: ChatRequest, c: AppContainer) -> ChatResponse:
        history = [Message(t.role, t.content) for t in body.history]
        return _response(await c.chat.ask(body.message, workflow=body.workflow, history=history))

    @app.post("/chat/stream", dependencies=Secured, response_class=StreamingResponse)
    async def chat_stream(body: ChatRequest, c: AppContainer) -> StreamingResponse:
        history = [Message(t.role, t.content) for t in body.history]
        events = c.chat.stream(body.message, workflow=body.workflow, history=history)
        return StreamingResponse(
            _server_sent_events(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/documents", dependencies=Secured)
    async def ingest(body: IngestRequest, c: AppContainer) -> IngestResponse:
        docs = [Document(d.id, d.content, d.metadata) for d in body.documents]
        return IngestResponse(chunks=await c.knowledge.ingest(docs))

    @app.post("/search", dependencies=Secured)
    async def search(body: SearchRequest, c: AppContainer) -> list[SearchHitOut]:
        hits = await c.knowledge.search(body.query, body.top_k)
        return [
            SearchHitOut(document_id=h.chunk.document_id, content=h.chunk.content, score=h.score)
            for h in hits
        ]

    return app


def _response(result: AgentResult) -> ChatResponse:
    return ChatResponse(
        agent=result.agent,
        answer=result.answer,
        trace=[TraceEventOut(agent=e.agent, kind=e.kind, detail=e.detail) for e in result.trace],
    )


async def _server_sent_events(
    events: AsyncIterator[TraceEvent | TokenDelta | AgentResult],
) -> AsyncIterator[str]:
    try:
        async for event in events:
            if isinstance(event, TokenDelta):
                yield _sse("token", {"agent": event.agent, "text": event.text})
            elif isinstance(event, TraceEvent):
                yield _sse("trace", {"agent": event.agent, "kind": event.kind, "detail": event.detail})
            else:
                yield _sse("result", _response(event).model_dump())
    except DomainError as exc:
        yield _sse("error", {"detail": str(exc)})
    except Exception:
        log.exception("streaming chat failed")
        yield _sse("error", {"detail": "Internal error"})


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
