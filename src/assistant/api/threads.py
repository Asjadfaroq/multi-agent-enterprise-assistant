import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Literal

import psycopg
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from pydantic import BaseModel

from assistant.config import get_settings
from assistant.graph.nodes.approval import ProposedToolCall
from assistant.graph.state import AssistantContext
from assistant.mcp_server.auth import Role
from assistant.observability.tracing import trace_attributes, tracing_config

logger = logging.getLogger(__name__)

ContextOpener = Callable[[str], AbstractAsyncContextManager[AssistantContext]]
# Once the reviewer settles on one of these, the reply is final; only the memory step remains.
ANSWERED_VERDICTS = frozenset({"accepted", "give_up"})


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
    return answered_response(thread_id, snapshot.values)


def answered_response(thread_id: str, values: dict[str, Any]) -> ChatResponse:
    final_message = values["messages"][-1].text
    # Citations and confidence only describe an answer the reviewer accepted; refusals, fallbacks
    # and cancellations have neither.
    accepted = values["answer"] if values["verdict"] == "accepted" else None
    return ChatResponse(
        status="answered",
        thread_id=thread_id,
        answer=final_message,
        citations=accepted.citations if accepted else [],
        confidence=accepted.confidence if accepted else None,
    )


async def answer_turn(
    graph: CompiledStateGraph,
    graph_input: object,
    thread_id: str,
    user_id: str,
    role: Role,
    open_context: ContextOpener,
    finishing_turns: dict[str, asyncio.Task[None]],
) -> ChatResponse:
    """Runs a turn and replies as soon as its answer is final; the memory step finishes after."""
    answered: asyncio.Future[ChatResponse] = asyncio.get_running_loop().create_future()

    async def run_turn() -> None:
        config = traced_thread_config(thread_id, user_id, role)
        async with open_context(user_id) as context:
            with trace_attributes(user_id, role, thread_id):
                async for values in graph.astream(
                    graph_input, config, context=context, stream_mode="values"
                ):
                    if not answered.done() and values.get("verdict") in ANSWERED_VERDICTS:
                        answered.set_result(answered_response(thread_id, values))

    def forget_finished_turn(turn: asyncio.Task[None]) -> None:
        finishing_turns.pop(thread_id, None)
        # Errors before the reply reach the caller; only later ones would otherwise go unseen.
        if answered.done() and not turn.cancelled() and turn.exception() is not None:
            logger.error(
                "Turn on thread %s failed after its reply was sent",
                thread_id,
                exc_info=turn.exception(),
            )

    turn = asyncio.create_task(run_turn())
    finishing_turns[thread_id] = turn
    turn.add_done_callback(forget_finished_turn)
    await asyncio.wait({turn, answered}, return_when=asyncio.FIRST_COMPLETED)
    if answered.done():
        return answered.result()
    turn.result()
    return response_for_thread(thread_id, await graph.aget_state(thread_config(thread_id)))
