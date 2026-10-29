import asyncio
import time
from collections import defaultdict
from uuid import uuid4

import psycopg
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from pydantic import BaseModel

from assistant.api.threads import thread_config, traced_thread_config
from assistant.config import get_settings
from assistant.graph.build import open_assistant_context
from assistant.graph.nodes.approval import ProposedToolCall
from assistant.graph.state import messages_since_latest_question, new_turn_state
from assistant.mcp_server.auth import WRITE_TOOL_NAMES, role_for_user
from assistant.observability.tracing import trace_attributes
from evals.golden_cases import GoldenCase
from scripts.seed import reset_sample_tickets

EVAL_THREAD_PREFIX = "eval-"


class CaseRun(BaseModel):
    """What one golden case did when run through the real graph, MCP server and checkpointer."""

    case: GoldenCase
    answer_text: str = ""
    citations: list[str] = []
    retrieved_doc_ids: list[str] = []
    retrieved_contexts: list[str] = []
    tool_results: list[str] = []
    proposed_action: list[ProposedToolCall] = []
    write_tools_requested: list[str] = []
    input_blocked: bool = False
    tickets_changed: bool = False
    latency_seconds: float = 0.0
    error: str | None = None


def ticket_state(connection: psycopg.Connection) -> list[tuple[object, ...]]:
    query = "SELECT id, status, priority, updated_at FROM tickets ORDER BY id"
    return connection.execute(query).fetchall()


async def clear_pending_approval_threads(
    graph: CompiledStateGraph, checkpointer: AsyncPostgresSaver
) -> int:
    async with await psycopg.AsyncConnection.connect(get_settings().database_url) as connection:
        # Every thread is checked through the public state API rather than LangGraph's private
        # interrupt channel name.
        cursor = await connection.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [thread_id for (thread_id,) in await cursor.fetchall()]
    cleared = 0
    for thread_id in thread_ids:
        if (await graph.aget_state(thread_config(thread_id))).interrupts:
            await checkpointer.adelete_thread(thread_id)
            cleared += 1
    return cleared


def reset_tickets() -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        reset_sample_tickets(connection)


def run_from_snapshot(
    case: GoldenCase, snapshot: StateSnapshot, latency_seconds: float, tickets_changed: bool
) -> CaseRun:
    values = snapshot.values
    turn_messages = messages_since_latest_question(values)
    accepted_answer = values["answer"] if values["verdict"] == "accepted" else None
    write_tools_requested = [
        call["name"]
        for message in turn_messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] in WRITE_TOOL_NAMES
    ]
    return CaseRun(
        case=case,
        answer_text="" if snapshot.interrupts else values["messages"][-1].text,
        citations=accepted_answer.citations if accepted_answer else [],
        retrieved_doc_ids=[hit.doc_id for hit in values["retrieved"]],
        retrieved_contexts=[hit.content for hit in values["retrieved"]],
        tool_results=[
            f"{message.name} ({message.status}): {message.text}"
            for message in turn_messages
            if isinstance(message, ToolMessage)
        ],
        proposed_action=snapshot.interrupts[0].value["proposed_action"]
        if snapshot.interrupts
        else [],
        write_tools_requested=write_tools_requested,
        input_blocked=values["input_blocked"],
        tickets_changed=tickets_changed,
        latency_seconds=latency_seconds,
    )


async def run_case(graph: CompiledStateGraph, case: GoldenCase) -> CaseRun:
    thread_id = f"{EVAL_THREAD_PREFIX}{case.id}-{uuid4().hex[:8]}"
    with psycopg.connect(get_settings().database_url) as connection:
        # Each case starts without remembered facts, so one case's language or team cannot leak
        # into the next.
        connection.execute("DELETE FROM user_memory WHERE user_id = %s", (case.user,))
        role = role_for_user(connection, case.user)
        tickets_before = ticket_state(connection)
    started = time.perf_counter()
    try:
        async with open_assistant_context(case.user) as context:
            with trace_attributes(case.user, role, thread_id):
                await graph.ainvoke(
                    new_turn_state(case.user, case.question),
                    traced_thread_config(thread_id, case.user, role),
                    context=context,
                )
    except Exception as error:
        # One broken case is reported as a failure instead of aborting the whole run.
        latency_seconds = time.perf_counter() - started
        return CaseRun(case=case, latency_seconds=latency_seconds, error=repr(error))
    latency_seconds = time.perf_counter() - started
    with psycopg.connect(get_settings().database_url) as connection:
        tickets_changed = ticket_state(connection) != tickets_before
    snapshot = await graph.aget_state(thread_config(thread_id))
    return run_from_snapshot(case, snapshot, latency_seconds, tickets_changed)


async def run_cases_concurrently(
    graph: CompiledStateGraph, cases: list[GoldenCase], concurrency: int
) -> list[CaseRun]:
    concurrency_limit = asyncio.Semaphore(concurrency)
    user_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    finished = 0

    async def run_when_free(case: GoldenCase) -> CaseRun:
        nonlocal finished
        # Cases of one user never overlap: each clears that user's memory and may write to it.
        # The user lock is taken first so waiting cases do not hold a concurrency slot.
        async with user_locks[case.user], concurrency_limit:
            run = await run_case(graph, case)
        finished += 1
        # Flushed so progress shows even when output is redirected to a file.
        print(f"[{finished}/{len(cases)}] {case.id} {run.latency_seconds:.1f}s", flush=True)
        return run

    return list(await asyncio.gather(*(run_when_free(case) for case in cases)))
