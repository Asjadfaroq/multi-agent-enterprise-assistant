from collections.abc import AsyncIterator
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
from assistant.mcp_server.auth import require_known_user, require_ticket_write_permission

UserIdHeader = Annotated[str, Header(alias="X-User")]


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
        yield


app = FastAPI(title="Enterprise Assistant", lifespan=lifespan)


@app.exception_handler(PermissionError)
async def refuse_forbidden_request(request: Request, error: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(error)})


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def response_for_thread(thread_id: str, snapshot: StateSnapshot) -> ChatResponse:
    if snapshot.interrupts:
        return ChatResponse(
            status="awaiting_approval",
            thread_id=thread_id,
            proposed_action=snapshot.interrupts[0].value["proposed_action"],
        )
    answer = snapshot.values["messages"][-1].text
    return ChatResponse(status="answered", thread_id=thread_id, answer=answer)


def require_known_user_in_database(user_id: str) -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        require_known_user(connection, user_id)


def require_ticket_writer_in_database(user_id: str) -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        require_ticket_write_permission(connection, user_id)


@app.post("/chat")
async def chat(request: ChatRequest, user_id: UserIdHeader) -> ChatResponse:
    graph: CompiledStateGraph = app.state.graph
    thread_id = request.thread_id or str(uuid4())
    config = thread_config(thread_id)
    snapshot = await graph.aget_state(config)
    if snapshot.values and snapshot.values["user_id"] != user_id:
        raise PermissionError(f"Thread {thread_id} belongs to another user")
    if snapshot.interrupts:
        raise HTTPException(409, f"Thread {thread_id} is waiting for approval; call /approve")
    require_known_user_in_database(user_id)
    async with open_assistant_context(user_id) as context:
        await graph.ainvoke(new_turn_state(user_id, request.message), config, context=context)
    return response_for_thread(thread_id, await graph.aget_state(config))


@app.post("/approve/{thread_id}")
async def approve(
    thread_id: str, request: ApprovalRequest, approver_id: UserIdHeader
) -> ChatResponse:
    graph: CompiledStateGraph = app.state.graph
    config = thread_config(thread_id)
    snapshot = await graph.aget_state(config)
    if not snapshot.interrupts:
        raise HTTPException(409, f"Thread {thread_id} has no action waiting for approval")
    require_ticket_writer_in_database(approver_id)
    # The approved tools run as the user who asked for them, not as the approver.
    async with open_assistant_context(snapshot.values["user_id"]) as context:
        await graph.ainvoke(Command(resume=request.approved), config, context=context)
    return response_for_thread(thread_id, await graph.aget_state(config))


@app.post("/ingest")
def ingest() -> IngestResponse:
    settings = get_settings()
    return IngestResponse(
        chunk_count=ingest_documents(settings.sample_docs_dir, settings.database_url)
    )
