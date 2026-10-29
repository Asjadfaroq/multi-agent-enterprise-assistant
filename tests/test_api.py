import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from conftest import NO_PROFILE_FACTS, FakeToolbox, scripted_model_after_safe_input
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from assistant.api import app as api_module
from assistant.api.app import app
from assistant.graph.build import build_graph, checkpoint_serializer
from assistant.graph.state import AssistantContext, new_turn_state

OWNER_ID, APPROVER_ID, VIEWER_ID = "test-api-agent", "test-api-manager", "test-api-viewer"
THREAD_ID = "test-api-thread"
# Opening the tools sits between reading the paused thread and resuming it, the window in which a
# second approval could also see the pause; the delay makes both requests land inside it.
CONTEXT_OPEN_DELAY_SECONDS = 0.3
SCRIPTED_REPLIES = [
    json.dumps({"steps": ["create the ticket"], "needs_tools": ["create_ticket"]}),
    AIMessage(
        "",
        tool_calls=[
            {
                "name": "create_ticket",
                "args": {"title": "Printer jam", "category": "it", "priority": "P3"},
                "id": "call-create",
            }
        ],
    ),
    json.dumps(
        {"text": "Created ticket 1 [ticket-1].", "citations": ["ticket-1"], "confidence": 0.9}
    ),
    json.dumps({"supported": True, "reason": "Matches the tool result."}),
    NO_PROFILE_FACTS,
]


@pytest.fixture
def api_users(committed_users: Callable[[str, str], None]) -> None:
    committed_users(OWNER_ID, "agent")
    committed_users(APPROVER_ID, "manager")
    committed_users(VIEWER_ID, "viewer")


@pytest.fixture
def paused_thread_toolbox(api_users: None) -> FakeToolbox:
    """Serves a graph whose thread is paused on a create_ticket call waiting for approval."""
    toolbox = FakeToolbox()
    scripted_model = scripted_model_after_safe_input(SCRIPTED_REPLIES)
    graph = build_graph(
        chat_model_factory=lambda role: scripted_model,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
    )
    asyncio.run(
        graph.ainvoke(
            new_turn_state(OWNER_ID, "Open a P3 IT ticket: printer jam."),
            {"configurable": {"thread_id": THREAD_ID}},
            context=toolbox.context(can_write_tickets=True),
        )
    )

    @asynccontextmanager
    async def open_slow_fake_context(user_id: str) -> AsyncIterator[AssistantContext]:
        await asyncio.sleep(CONTEXT_OPEN_DELAY_SECONDS)
        yield toolbox.context(can_write_tickets=True)

    app.state.graph = graph
    app.state.open_assistant_context = open_slow_fake_context
    return toolbox


async def post_decisions(decisions: list[bool]) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await asyncio.gather(
            *(
                client.post(
                    f"/approve/{THREAD_ID}",
                    headers={"X-User": APPROVER_ID},
                    json={"approved": approved},
                )
                for approved in decisions
            )
        )


def test_two_simultaneous_approvals_resolve_the_thread_exactly_once(
    paused_thread_toolbox: FakeToolbox,
) -> None:
    responses = asyncio.run(post_decisions([True, True]))

    assert sorted(response.status_code for response in responses) == [200, 409]
    assert len(paused_thread_toolbox.created_tickets) == 1


def test_deciding_an_already_resolved_thread_returns_409_and_writes_nothing(
    paused_thread_toolbox: FakeToolbox,
) -> None:
    [first_approval] = asyncio.run(post_decisions([True]))
    [late_approval] = asyncio.run(post_decisions([True]))
    [late_rejection] = asyncio.run(post_decisions([False]))

    assert first_approval.status_code == 200
    assert (late_approval.status_code, late_rejection.status_code) == (409, 409)
    assert len(paused_thread_toolbox.created_tickets) == 1


def post_ingest(user_id: str) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            return await client.post("/ingest", headers={"X-User": user_id})

    return asyncio.run(send())


def test_ingest_is_refused_for_non_managers_and_allowed_for_managers(
    api_users: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_calls: list[str] = []
    monkeypatch.setattr(
        api_module, "ingest_documents", lambda docs_dir, url: ingest_calls.append(url) or 13
    )

    refused = post_ingest(VIEWER_ID)
    allowed = post_ingest(APPROVER_ID)

    assert refused.status_code == 403
    assert allowed.status_code == 200 and allowed.json() == {"chunk_count": 13}
    assert len(ingest_calls) == 1
