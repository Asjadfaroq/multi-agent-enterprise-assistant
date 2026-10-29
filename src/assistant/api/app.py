from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4

import psycopg
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel

from assistant.config import get_settings
from assistant.graph.build import build_graph, checkpoint_serializer, open_assistant_context
from assistant.graph.nodes.executor import ProposedToolCall
from assistant.graph.state import new_turn_state
from assistant.knowledge.ingest import ingest_documents
from assistant.mcp_server.auth import (
    CALLER_HEADER,
    Role,
    require_ingest_permission,
    require_ticket_write_permission,
    role_for_user,
)
from assistant.observability.tracing import flush_traces, trace_attributes, tracing_config

UserIdHeader = Annotated[str, Header(alias=CALLER_HEADER)]


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool


class ChatResponse(BaseModel):
    status: Literal["answered", "awaiting_approval"]
    thread_id: str
    answer: str | None = None
    proposed_action: list[ProposedToolCall] = []


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
    flush_traces()


app = FastAPI(title="Enterprise Assistant", lifespan=lifespan)


@app.exception_handler(PermissionError)
async def refuse_forbidden_request(request: Request, error: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(error)})


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def traced_thread_config(thread_id: str, user_id: str, role: Role) -> RunnableConfig:
    return {**thread_config(thread_id), **tracing_config(user_id, role, thread_id)}


def role_in_database(user_id: str) -> Role:
    with psycopg.connect(get_settings().database_url) as connection:
        return role_for_user(connection, user_id)


def require_in_database(check: Callable[[psycopg.Connection, str], None], user_id: str) -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        check(connection, user_id)


@asynccontextmanager
async def exclusive_approval(thread_id: str) -> AsyncIterator[None]:
    # The advisory lock lives in Postgres, so it also holds across several API workers.
    async with await psycopg.AsyncConnection.connect(
        get_settings().database_url, autocommit=True
    ) as connection:
        lock_key = "hashtextextended(%s, 0)"
        cursor = await connection.execute(f"SELECT pg_try_advisory_lock({lock_key})", (thread_id,))
        (acquired,) = await cursor.fetchone()
        if not acquired:
            raise HTTPException(409, f"Thread {thread_id} is already being approved or rejected")
        try:
            yield
        finally:
            await connection.execute(f"SELECT pg_advisory_unlock({lock_key})", (thread_id,))


def response_for_thread(thread_id: str, snapshot: StateSnapshot) -> ChatResponse:
    if snapshot.interrupts:
        return ChatResponse(
            status="awaiting_approval",
            thread_id=thread_id,
            proposed_action=snapshot.interrupts[0].value["proposed_action"],
        )
    answer = snapshot.values["messages"][-1].text
    return ChatResponse(status="answered", thread_id=thread_id, answer=answer)


@app.post("/chat")
async def chat(request: ChatRequest, user_id: UserIdHeader) -> ChatResponse:
    graph: CompiledStateGraph = app.state.graph
    thread_id = request.thread_id or str(uuid4())
    snapshot = await graph.aget_state(thread_config(thread_id))
    if snapshot.values and snapshot.values["user_id"] != user_id:
        raise PermissionError(f"Thread {thread_id} belongs to another user")
    if snapshot.interrupts:
        raise HTTPException(409, f"Thread {thread_id} is waiting for approval; call /approve")
    role = role_in_database(user_id)
    async with app.state.open_assistant_context(user_id) as context:
        with trace_attributes(user_id, role, thread_id):
            await graph.ainvoke(
                new_turn_state(user_id, request.message),
                traced_thread_config(thread_id, user_id, role),
                context=context,
            )
    return response_for_thread(thread_id, await graph.aget_state(thread_config(thread_id)))


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
        owner_role = role_in_database(owner_id)
        config = traced_thread_config(thread_id, owner_id, owner_role)
        async with app.state.open_assistant_context(owner_id) as context:
            with trace_attributes(owner_id, owner_role, thread_id):
                await graph.ainvoke(Command(resume=request.approved), config, context=context)
        snapshot = await graph.aget_state(thread_config(thread_id))
    return response_for_thread(thread_id, snapshot)


@app.post("/ingest")
def ingest(user_id: UserIdHeader) -> IngestResponse:
    require_in_database(require_ingest_permission, user_id)
    settings = get_settings()
    chunk_count = ingest_documents(settings.sample_docs_dir, settings.database_url)
    return IngestResponse(chunk_count=chunk_count)
