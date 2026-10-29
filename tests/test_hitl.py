import asyncio
import json
from typing import Any

from conftest import FakeToolbox, GraphBuilder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from assistant.graph.build import checkpoint_serializer
from assistant.graph.nodes.executor import CANCELLED_ANSWER, REJECTED_ACTION_MESSAGE
from assistant.graph.state import AssistantContext, new_turn_state
from assistant.mcp_server.server import build_server

THREAD_CONFIG = {"configurable": {"thread_id": "ticket-thread"}}
REQUEST = "Open a P2 IT ticket: my laptop will not boot."
TICKET_ARGS = {"title": "Laptop will not boot", "category": "it", "priority": "P2"}

PLAN_TICKET_WRITE = json.dumps({"steps": ["create the ticket"], "needs_tools": ["create_ticket"]})
PLAN_TICKET_READ = json.dumps({"steps": ["list tickets"], "needs_tools": ["list_tickets"]})
CREATE_TICKET_CALL = AIMessage(
    "", tool_calls=[{"name": "create_ticket", "args": TICKET_ARGS, "id": "call-create"}]
)
LIST_TICKETS_CALL = AIMessage(
    "", tool_calls=[{"name": "list_tickets", "args": {}, "id": "call-list"}]
)
FINAL_DRAFT = json.dumps({"text": "Done.", "citations": []})
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "Matches the tool result."})
UNSUPPORTED_REVIEW = json.dumps({"supported": False, "reason": "Claim not in tool results."})


def start_turn(
    graph: CompiledStateGraph, context: AssistantContext, user_id: str = "khalid"
) -> dict[str, Any]:
    return asyncio.run(
        graph.ainvoke(new_turn_state(user_id, REQUEST), THREAD_CONFIG, context=context)
    )


def resume_turn(graph: CompiledStateGraph, context: AssistantContext, approved: bool) -> None:
    asyncio.run(graph.ainvoke(Command(resume=approved), THREAD_CONFIG, context=context))


def tool_messages(graph: CompiledStateGraph) -> list[ToolMessage]:
    messages = asyncio.run(graph.aget_state(THREAD_CONFIG)).values["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


def test_write_tool_call_pauses_for_approval_before_running(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    graph = build_scripted_graph(
        [PLAN_TICKET_WRITE, CREATE_TICKET_CALL], InMemorySaver(serde=checkpoint_serializer())
    )

    start_turn(graph, toolbox.context(can_write_tickets=True))

    snapshot = asyncio.run(graph.aget_state(THREAD_CONFIG))
    assert snapshot.next == ("request_approval",)
    assert snapshot.interrupts[0].value == {
        "proposed_action": [{"tool": "create_ticket", "args": TICKET_ARGS}]
    }
    assert toolbox.created_tickets == []


def test_approved_write_resumes_and_creates_the_ticket(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, FINAL_DRAFT, SUPPORTED_REVIEW]
    graph = build_scripted_graph(replies, InMemorySaver(serde=checkpoint_serializer()))
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=True)

    assert [ticket["title"] for ticket in toolbox.created_tickets] == ["Laptop will not boot"]
    assert asyncio.run(graph.aget_state(THREAD_CONFIG)).next == ()


def test_unverified_answer_still_reports_the_approved_write_that_ran(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [
        PLAN_TICKET_WRITE,
        CREATE_TICKET_CALL,
        FINAL_DRAFT,
        UNSUPPORTED_REVIEW,
        FINAL_DRAFT,
        UNSUPPORTED_REVIEW,
    ]
    graph = build_scripted_graph(replies, InMemorySaver(serde=checkpoint_serializer()))
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=True)

    final_answer = asyncio.run(graph.aget_state(THREAD_CONFIG)).values["messages"][-1].text
    assert final_answer.endswith("These approved actions did run: create_ticket.")


def test_rejected_write_is_cancelled_and_nothing_is_written(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    graph = build_scripted_graph(
        [PLAN_TICKET_WRITE, CREATE_TICKET_CALL], InMemorySaver(serde=checkpoint_serializer())
    )
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=False)

    snapshot = asyncio.run(graph.aget_state(THREAD_CONFIG))
    assert toolbox.created_tickets == []
    assert [message.text for message in tool_messages(graph)] == [REJECTED_ACTION_MESSAGE]
    assert snapshot.next == ()
    assert snapshot.values["messages"][-1].text == CANCELLED_ANSWER.format(tools="create_ticket")


def test_read_tool_call_runs_without_pausing(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_READ, LIST_TICKETS_CALL, FINAL_DRAFT, SUPPORTED_REVIEW]
    graph = build_scripted_graph(replies, InMemorySaver(serde=checkpoint_serializer()))

    start_turn(graph, toolbox.context(can_write_tickets=False), user_id="nora")

    snapshot = asyncio.run(graph.aget_state(THREAD_CONFIG))
    assert snapshot.interrupts == ()
    assert [message.name for message in tool_messages(graph)] == ["list_tickets"]


def test_viewer_cannot_reach_a_write_tool_even_if_the_model_calls_one(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, FINAL_DRAFT, SUPPORTED_REVIEW]
    graph = build_scripted_graph(replies, InMemorySaver(serde=checkpoint_serializer()))

    start_turn(graph, toolbox.context(can_write_tickets=False), user_id="nora")

    snapshot = asyncio.run(graph.aget_state(THREAD_CONFIG))
    [refusal] = tool_messages(graph)
    assert snapshot.interrupts == ()
    assert refusal.status == "error" and "not available" in refusal.text
    assert toolbox.created_tickets == []


def test_mcp_server_lists_write_tools_only_for_ticket_writers() -> None:
    def listed_tool_names(role: str) -> set[str]:
        return {tool.name for tool in asyncio.run(build_server(role).list_tools())}

    assert {"create_ticket", "update_ticket"}.isdisjoint(listed_tool_names("viewer"))
    assert {"create_ticket", "update_ticket"} <= listed_tool_names("agent")
