from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from assistant.graph.state import (
    AgentState,
    AssistantContext,
    completed_write_actions,
    pending_tool_calls,
    planned_write_tools,
)
from assistant.mcp_server.auth import WRITE_TOOL_NAMES

REJECTED_ACTION_MESSAGE = "The approver rejected this action, so nothing was changed."
UNREQUESTED_WRITE_MESSAGE = (
    "Not run: the user asked for information, not for a ticket change. Answer the question."
)
WRITE_ALREADY_DONE_MESSAGE = (
    "Not run: an approved change already ran in this turn. Report its result to the user."
)
CANCELLED_ANSWER = "The approver rejected {tools}, so nothing was changed."


class ProposedToolCall(BaseModel):
    tool: str
    args: dict[str, Any]


def refuse_tool_calls(calls: list[ToolCall], reason: str) -> list[ToolMessage]:
    return [
        ToolMessage(reason, tool_call_id=call["id"], name=call["name"], status="error")
        for call in calls
    ]


async def request_approval(
    state: AgentState, runtime: Runtime[AssistantContext]
) -> Command[Literal["run_tools", "executor", "__end__"]]:
    calls = pending_tool_calls(state)
    # One approved write per turn: in a live run the model re-proposed the write it had just made,
    # which would have created a duplicate ticket.
    if completed_write_actions(state):
        return Command(
            goto="executor",
            update={"messages": refuse_tool_calls(calls, WRITE_ALREADY_DONE_MESSAGE)},
        )
    # A write the plan did not ask for is refused without bothering an approver: in a live run the
    # model offered to open a ticket when the user only asked what to tell a vendor.
    if any(
        call["name"] in WRITE_TOOL_NAMES and call["name"] not in planned_write_tools(state)
        for call in calls
    ):
        return Command(
            goto="executor",
            update={"messages": refuse_tool_calls(calls, UNREQUESTED_WRITE_MESSAGE)},
        )
    # A write the user's session does not expose fails in run_tools without bothering an
    # approver, so roles without write access never reach a pause or a write tool.
    proposed_action = [
        ProposedToolCall(tool=call["name"], args=call["args"])
        for call in calls
        if call["name"] in WRITE_TOOL_NAMES and call["name"] in runtime.context.tools_by_name
    ]
    if not proposed_action:
        return Command(goto="run_tools")
    approved = interrupt({"proposed_action": [call.model_dump() for call in proposed_action]})
    if approved is True:
        return Command(goto="run_tools")
    # A rejection ends the turn without asking the model again, so it cannot re-propose the write.
    rejections = refuse_tool_calls(calls, REJECTED_ACTION_MESSAGE)
    rejected_tools = ", ".join(call.tool for call in proposed_action)
    cancellation = AIMessage(CANCELLED_ANSWER.format(tools=rejected_tools))
    return Command(goto=END, update={"messages": [*rejections, cancellation]})
