from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import psycopg
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from pydantic import BaseModel

from assistant.config import get_settings
from assistant.graph.nodes.executor import ProposedToolCall
from assistant.mcp_server.auth import Role
from assistant.observability.tracing import tracing_config


class ChatResponse(BaseModel):
    status: Literal["answered", "awaiting_approval"]
    thread_id: str
    answer: str | None = None
    citations: list[str] = []
    confidence: float | None = None
    proposed_action: list[ProposedToolCall] = []


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def traced_thread_config(thread_id: str, user_id: str, role: Role) -> RunnableConfig:
    return {**thread_config(thread_id), **tracing_config(user_id, role, thread_id)}


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
    final_message = snapshot.values["messages"][-1].text
    # Citations and confidence only describe an answer the reviewer accepted; refusals, fallbacks
    # and cancellations have neither.
    accepted = snapshot.values["answer"] if snapshot.values["verdict"] == "accepted" else None
    return ChatResponse(
        status="answered",
        thread_id=thread_id,
        answer=final_message,
        citations=accepted.citations if accepted else [],
        confidence=accepted.confidence if accepted else None,
    )
