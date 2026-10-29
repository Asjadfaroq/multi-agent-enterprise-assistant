import asyncio
import json

from conftest import NO_PROFILE_FACTS, FakeToolbox, GraphBuilder, scripted_model_after_safe_input
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from assistant.graph.build import build_graph, checkpoint_serializer
from assistant.graph.nodes.approval import (
    CANCELLED_ANSWER,
    REJECTED_ACTION_MESSAGE,
    UNREQUESTED_WRITE_MESSAGE,
    WRITE_ALREADY_DONE_MESSAGE,
    permission_refusal,
)
from assistant.graph.state import AssistantContext, new_turn_state

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
FINAL_ANSWER = json.dumps(
    {"text": "Done [ticket-1].", "citations": ["ticket-1"], "confidence": 0.9, "answer_found": True}
)
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "Matches the tool result."})
UNSUPPORTED_REVIEW = json.dumps({"supported": False, "reason": "Claim not in tool results."})
ACCEPTED_ANSWER_REPLIES = [FINAL_ANSWER, SUPPORTED_REVIEW, NO_PROFILE_FACTS]


def checkpointed_graph(build: GraphBuilder, replies: list[str | AIMessage]) -> CompiledStateGraph:
    return build(replies, InMemorySaver(serde=checkpoint_serializer()))


def start_turn(
    graph: CompiledStateGraph, context: AssistantContext, user_id: str = "khalid"
) -> None:
    asyncio.run(graph.ainvoke(new_turn_state(user_id, REQUEST), THREAD_CONFIG, context=context))


def resume_turn(graph: CompiledStateGraph, context: AssistantContext, approved: bool) -> None:
    asyncio.run(graph.ainvoke(Command(resume=approved), THREAD_CONFIG, context=context))


def thread_snapshot(graph: CompiledStateGraph) -> StateSnapshot:
    return asyncio.run(graph.aget_state(THREAD_CONFIG))


def tool_messages(graph: CompiledStateGraph) -> list[ToolMessage]:
    messages = thread_snapshot(graph).values["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


def test_write_tool_call_pauses_for_approval_before_running(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    graph = checkpointed_graph(build_scripted_graph, [PLAN_TICKET_WRITE, CREATE_TICKET_CALL])

    start_turn(graph, toolbox.context(can_write_tickets=True))

    snapshot = thread_snapshot(graph)
    assert snapshot.next == ("request_approval",)
    assert snapshot.interrupts[0].value == {
        "proposed_action": [{"tool": "create_ticket", "args": TICKET_ARGS}]
    }
    assert toolbox.created_tickets == []


def test_approved_write_resumes_and_creates_the_ticket(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, *ACCEPTED_ANSWER_REPLIES]
    graph = checkpointed_graph(build_scripted_graph, replies)
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=True)

    assert [ticket["title"] for ticket in toolbox.created_tickets] == ["Laptop will not boot"]
    assert thread_snapshot(graph).next == ()


def test_unverified_answer_still_reports_the_approved_write_that_ran(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    unverified_twice = [FINAL_ANSWER, UNSUPPORTED_REVIEW, FINAL_ANSWER, UNSUPPORTED_REVIEW]
    replies = [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, *unverified_twice, NO_PROFILE_FACTS]
    graph = checkpointed_graph(build_scripted_graph, replies)
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=True)

    final_answer = thread_snapshot(graph).values["messages"][-1].text
    assert final_answer.endswith("These approved actions did run: create_ticket.")


def test_write_proposed_again_after_it_ran_is_neither_paused_nor_repeated(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    repeated_call = AIMessage(
        "", tool_calls=[{"name": "create_ticket", "args": TICKET_ARGS, "id": "call-again"}]
    )
    replies = [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, repeated_call, *ACCEPTED_ANSWER_REPLIES]
    graph = checkpointed_graph(build_scripted_graph, replies)
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=True)

    assert thread_snapshot(graph).next == ()
    assert len(toolbox.created_tickets) == 1
    assert tool_messages(graph)[-1].text == WRITE_ALREADY_DONE_MESSAGE


def test_rejected_write_is_cancelled_and_nothing_is_written(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    graph = checkpointed_graph(build_scripted_graph, [PLAN_TICKET_WRITE, CREATE_TICKET_CALL])
    context = toolbox.context(can_write_tickets=True)

    start_turn(graph, context)
    resume_turn(graph, context, approved=False)

    snapshot = thread_snapshot(graph)
    assert toolbox.created_tickets == []
    assert [message.text for message in tool_messages(graph)] == [REJECTED_ACTION_MESSAGE]
    assert snapshot.next == ()
    assert snapshot.values["messages"][-1].text == CANCELLED_ANSWER.format(tools="create_ticket")


def test_read_tool_call_runs_without_pausing(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_READ, LIST_TICKETS_CALL, *ACCEPTED_ANSWER_REPLIES]
    graph = checkpointed_graph(build_scripted_graph, replies)

    start_turn(graph, toolbox.context(can_write_tickets=False), user_id="nora")

    assert thread_snapshot(graph).interrupts == ()
    assert [message.name for message in tool_messages(graph)] == ["list_tickets"]


def test_viewer_asking_for_a_write_gets_the_fixed_permission_refusal(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    graph = checkpointed_graph(build_scripted_graph, [PLAN_TICKET_WRITE, NO_PROFILE_FACTS])

    start_turn(graph, toolbox.context(can_write_tickets=False), user_id="nora")

    snapshot = thread_snapshot(graph)
    assert snapshot.interrupts == () and snapshot.values["verdict"] == "accepted"
    assert snapshot.values["messages"][-1].text == permission_refusal(["create_ticket"]).text
    assert tool_messages(graph) == [] and toolbox.created_tickets == []


def test_write_the_user_did_not_ask_for_is_refused_without_a_pause(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_TICKET_READ, CREATE_TICKET_CALL, *ACCEPTED_ANSWER_REPLIES]
    graph = checkpointed_graph(build_scripted_graph, replies)

    start_turn(graph, toolbox.context(can_write_tickets=True))

    assert thread_snapshot(graph).interrupts == ()
    assert [message.text for message in tool_messages(graph)] == [UNREQUESTED_WRITE_MESSAGE]
    assert toolbox.created_tickets == []


def test_write_tools_are_offered_only_when_the_plan_asks_for_a_change(
    toolbox: FakeToolbox,
) -> None:
    read_only_model = scripted_model_after_safe_input([PLAN_TICKET_READ, *ACCEPTED_ANSWER_REPLIES])
    write_model = scripted_model_after_safe_input(
        [PLAN_TICKET_WRITE, CREATE_TICKET_CALL, *ACCEPTED_ANSWER_REPLIES]
    )
    for model in (read_only_model, write_model):
        graph = build_graph(lambda role, model=model: model, InMemorySaver())
        start_turn(graph, toolbox.context(can_write_tickets=True))

    assert "create_ticket" not in read_only_model.bound_tool_names[0]
    assert "create_ticket" in write_model.bound_tool_names[0]
