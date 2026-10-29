import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import psycopg
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from assistant.api.threads import (
    ChatResponse,
    answer_turn,
    exclusive_approval,
    thread_config,
)
from assistant.config import get_settings
from assistant.graph.build import build_graph, checkpoint_serializer, open_assistant_context
from assistant.graph.state import new_turn_state
from assistant.knowledge.ingest import ingest_documents
from assistant.mcp_server.auth import (
    CALLER_HEADER,
    Role,
    require_ingest_permission,
    require_ticket_write_permission,
    role_for_user,
)
from assistant.observability.tracing import flush_traces

UserIdHeader = Annotated[str, Header(alias=CALLER_HEADER)]


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool


class IngestResponse(BaseModel):
    chunk_count: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncPostgresSaver.from_conn_string(
        get_settings().database_url, serde=checkpoint_serializer()
    ) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer=checkpointer)
        app.state.open_assistant_context = open_assistant_context
        yield
        # Turns still writing memory after their reply must finish before the checkpointer closes.
        await asyncio.gather(*app.state.finishing_turns.values(), return_exceptions=True)
    flush_traces()


app = FastAPI(title="Enterprise Assistant", lifespan=lifespan)
app.state.finishing_turns = {}


@app.exception_handler(PermissionError)
async def refuse_forbidden_request(request: Request, error: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(error)})


def role_in_database(user_id: str) -> Role:
    with psycopg.connect(get_settings().database_url) as connection:
        return role_for_user(connection, user_id)


def require_in_database(check: Callable[[psycopg.Connection, str], None], user_id: str) -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        check(connection, user_id)


@app.post("/chat")
async def chat(request: ChatRequest, user_id: UserIdHeader) -> ChatResponse:
    graph: CompiledStateGraph = app.state.graph
    thread_id = request.thread_id or str(uuid4())
    snapshot = await graph.aget_state(thread_config(thread_id))
    if snapshot.values and snapshot.values["user_id"] != user_id:
        raise PermissionError(f"Thread {thread_id} belongs to another user")
    if snapshot.interrupts:
        raise HTTPException(409, f"Thread {thread_id} is waiting for approval; call /approve")
    if thread_id in app.state.finishing_turns:
        raise HTTPException(409, f"Thread {thread_id} is still finishing its previous turn")
    role = role_in_database(user_id)
    return await answer_turn(
        graph,
        new_turn_state(user_id, request.message),
        thread_id,
        user_id,
        role,
        app.state.open_assistant_context,
        app.state.finishing_turns,
    )


@app.post("/approve/{thread_id}")
async def approve(
    thread_id: str, request: ApprovalRequest, approver_id: UserIdHeader
) -> ChatResponse:
    graph: CompiledStateGraph = app.state.graph
    require_in_database(require_ticket_write_permission, approver_id)
    async with exclusive_approval(thread_id):
        # Read the thread only after taking the lock, so a decision that just finished is seen.
        snapshot = await graph.aget_state(thread_config(thread_id))
        if not snapshot.interrupts:
            raise HTTPException(409, f"Thread {thread_id} has no action waiting for approval")
        # The approved tools run as the user who asked for them, not as the approver.
        owner_id = snapshot.values["user_id"]
        return await answer_turn(
            graph,
            Command(resume=request.approved),
            thread_id,
            owner_id,
            role_in_database(owner_id),
            app.state.open_assistant_context,
            app.state.finishing_turns,
        )


@app.post("/ingest")
def ingest(user_id: UserIdHeader) -> IngestResponse:
    require_in_database(require_ingest_permission, user_id)
    settings = get_settings()
    chunk_count = ingest_documents(settings.sample_docs_dir, settings.database_url)
    return IngestResponse(chunk_count=chunk_count)
