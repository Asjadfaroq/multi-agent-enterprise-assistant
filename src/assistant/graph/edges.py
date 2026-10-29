from typing import Literal

from langgraph.graph import END

from assistant.graph.state import AgentState, pending_tool_calls
from assistant.mcp_server.auth import WRITE_TOOL_NAMES


def route_after_input_guard(state: AgentState) -> Literal["retriever", "executor", "__end__"]:
    return END if state["input_blocked"] else route_by_plan(state)


def route_by_plan(state: AgentState) -> Literal["retriever", "executor"]:
    plan = state["plan"]
    if plan is not None and "search_documents" in plan.needs_tools:
        return "retriever"
    return "executor"


def route_after_executor(
    state: AgentState,
) -> Literal["request_approval", "run_tools", "reviewer", "__end__"]:
    calls = pending_tool_calls(state)
    if not calls:
        # No answer and no tool calls means the executor failed closed and already replied.
        return "reviewer" if state["answer"] is not None else END
    if any(call["name"] in WRITE_TOOL_NAMES for call in calls):
        return "request_approval"
    return "run_tools"


def route_after_review(state: AgentState) -> Literal["executor", "remember"]:
    return "executor" if state["verdict"] == "revise" else "remember"
